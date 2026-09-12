"""
Retrieval module for grounded reply generation (Part 4).

Approach: embed the FIRST customer turn of every thread using a local
sentence-transformer (no API cost for embedding at scale), store vectors,
and at generation time retrieve the k most similar historically-resolved
threads so the reply generator can ground its draft in how AmazonHelp
actually responded to similar issues before -- rather than hallucinating
a generic response.

"Historically resolved" here means: the thread has more than one brand
turn OR ends with a customer turn that doesn't restate the same complaint
(a rough resolution proxy -- documented as a known limitation, not a
verified resolution signal, since we have no explicit resolved/unresolved
field in the source data).

Usage:
    # one-time index build
    python -m src.retrieval build --input data/threads_AmazonHelp_clean.jsonl \
        --index data/thread_index.pkl

    # then used as a library at generation time:
    from src.retrieval import ThreadRetriever
    r = ThreadRetriever.load("data/thread_index.pkl")
    similar = r.retrieve("my package says delivered but I never got it", k=3)
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for short support texts


@dataclass
class IndexedThread:
    thread_id: str
    customer_text: str
    brand_replies: list  # list of brand turn texts, in order, from this thread
    embedding: np.ndarray


class ThreadRetriever:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self.threads: list = []
        self._matrix = None  # stacked embeddings, built lazily

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, input_path: str, min_brand_turns: int = 1):
        threads = []
        texts_to_embed = []

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                customer_text = None
                brand_replies = []
                for turn in obj["turns"]:
                    text = turn.get("text_clean")
                    if not text:
                        continue
                    if turn["is_brand"]:
                        brand_replies.append(text)
                    elif customer_text is None:
                        customer_text = text

                if customer_text and len(brand_replies) >= min_brand_turns:
                    threads.append({
                        "thread_id": obj["thread_id"],
                        "customer_text": customer_text,
                        "brand_replies": brand_replies,
                    })
                    texts_to_embed.append(customer_text)

        print(f"Embedding {len(texts_to_embed)} threads with {self.model_name} ...")
        embeddings = self.model.encode(
            texts_to_embed, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )

        self.threads = [
            IndexedThread(
                thread_id=t["thread_id"],
                customer_text=t["customer_text"],
                brand_replies=t["brand_replies"],
                embedding=emb,
            )
            for t, emb in zip(threads, embeddings)
        ]
        self._matrix = np.stack([t.embedding for t in self.threads])
        print(f"Indexed {len(self.threads)} threads.")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # serialize as plain dicts, not the IndexedThread dataclass directly --
        # pickling a dataclass instance ties the file to whatever module path
        # was active as __main__ when it was created (e.g. "python -m src.retrieval"
        # vs "python scripts/generate_replies.py" resolve IndexedThread differently),
        # which breaks loading from a different entry point. Plain dicts have no
        # such dependency.
        serializable = [
            {
                "thread_id": t.thread_id,
                "customer_text": t.customer_text,
                "brand_replies": t.brand_replies,
                "embedding": t.embedding,
            }
            for t in self.threads
        ]
        with open(path, "wb") as f:
            pickle.dump({"model_name": self.model_name, "threads": serializable}, f)
        print(f"Saved index -> {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        r = cls(model_name=data["model_name"])
        r.threads = [
            IndexedThread(
                thread_id=t["thread_id"],
                customer_text=t["customer_text"],
                brand_replies=t["brand_replies"],
                embedding=t["embedding"],
            )
            for t in data["threads"]
        ]
        r._matrix = np.stack([t.embedding for t in r.threads])
        return r

    def retrieve(self, query_text: str, k: int = 3, exclude_thread_id: str = None):
        """
        Return the k most similar historically-resolved threads (cosine sim,
        since embeddings are normalized).

        exclude_thread_id: pass the thread_id of the message being replied to,
        if it's already present in this index (e.g. when generating replies
        for messages drawn from the same pool the index was built from). Without
        this, the retriever will often find the message's own historical thread
        as its top "similar" result with similarity=1.0 -- which isn't
        generalizing from precedent, it's just looking up the answer to the
        message we're trying to generate a reply for. This matters a lot for
        eval integrity: a golden-set message that's also in the index will
        otherwise leak its own gold reply back into the retrieved context.
        """
        query_emb = self.model.encode([query_text], normalize_embeddings=True)[0]
        sims = self._matrix @ query_emb
        # pull extra candidates in case we need to drop the excluded one
        top_idx = np.argsort(-sims)[: k + 1]
        results = []
        for i in top_idx:
            if exclude_thread_id is not None and self.threads[i].thread_id == exclude_thread_id:
                continue
            results.append({
                "thread_id": self.threads[i].thread_id,
                "customer_text": self.threads[i].customer_text,
                "brand_replies": self.threads[i].brand_replies,
                "similarity": float(sims[i]),
            })
            if len(results) == k:
                break
        return results


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("--input", required=True)
    build_p.add_argument("--index", required=True)
    build_p.add_argument("--min-brand-turns", type=int, default=1)

    query_p = sub.add_parser("query")
    query_p.add_argument("--index", required=True)
    query_p.add_argument("--text", required=True)
    query_p.add_argument("--k", type=int, default=3)

    args = ap.parse_args()

    if args.command == "build":
        r = ThreadRetriever()
        r.build(args.input, min_brand_turns=args.min_brand_turns)
        r.save(args.index)
    elif args.command == "query":
        r = ThreadRetriever.load(args.index)
        results = r.retrieve(args.text, k=args.k)
        for res in results:
            print(f"\n[sim={res['similarity']:.3f}] thread={res['thread_id']}")
            print(f"  customer: {res['customer_text'][:150]}")
            for reply in res["brand_replies"][:2]:
                print(f"  brand:    {reply[:150]}")


if __name__ == "__main__":
    main()

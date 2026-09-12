"""
Retrieval module for grounded reply generation.

Approach:
    Embed the first customer turn of each thread using a local
    sentence-transformer. At generation time, retrieve the k most similar
    historically resolved threads so the reply generator can ground its
    response in how AmazonHelp actually responded to similar issues.

"Historically resolved" means that a thread has more than one brand turn
or otherwise appears to contain a customer follow-up that does not restate
the same complaint. This is only a rough resolution proxy because the
source data does not contain an explicit resolved/unresolved field.

Usage:
    # Build the index once:
    python -m src.retrieval build \
        --input data/threads_AmazonHelp_clean.jsonl \
        --index data/thread_index.pkl

    # Query the index:
    python -m src.retrieval query \
        --index data/thread_index.pkl \
        --text "my package says delivered but I never got it" \
        --k 3

    # Use as a library:
    from src.retrieval import ThreadRetriever

    retriever = ThreadRetriever.load("data/thread_index.pkl")
    similar = retriever.retrieve(
        "my package says delivered but I never got it",
        k=3,
    )
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64


@dataclass
class IndexedThread:
    """A thread stored in the retrieval index."""

    thread_id: str
    customer_text: str
    brand_replies: list[str]
    embedding: np.ndarray


class ThreadRetriever:
    """Build and query a semantic index of historical support threads."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self.threads: list[IndexedThread] = []
        self._matrix: np.ndarray | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Load the embedding model lazily on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        return self._model

    def build(
        self,
        input_path: str,
        min_brand_turns: int = 1,
    ) -> None:
        """Build an embedding index from cleaned conversation threads."""
        threads = []
        texts_to_embed = []

        with open(input_path, "r", encoding="utf-8") as file:
            for line in file:
                thread = json.loads(line)

                customer_text = None
                brand_replies = []

                for turn in thread["turns"]:
                    text = turn.get("text_clean")

                    if not text:
                        continue

                    if turn["is_brand"]:
                        brand_replies.append(text)
                    elif customer_text is None:
                        customer_text = text

                if (
                    customer_text
                    and len(brand_replies) >= min_brand_turns
                ):
                    threads.append(
                        {
                            "thread_id": thread["thread_id"],
                            "customer_text": customer_text,
                            "brand_replies": brand_replies,
                        }
                    )
                    texts_to_embed.append(customer_text)

        print(
            f"Embedding {len(texts_to_embed)} threads "
            f"with {self.model_name} ..."
        )

        embeddings = self.model.encode(
            texts_to_embed,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        self.threads = [
            IndexedThread(
                thread_id=thread["thread_id"],
                customer_text=thread["customer_text"],
                brand_replies=thread["brand_replies"],
                embedding=embedding,
            )
            for thread, embedding in zip(threads, embeddings)
        ]

        if self.threads:
            self._matrix = np.stack(
                [thread.embedding for thread in self.threads]
            )
        else:
            self._matrix = None

        print(f"Indexed {len(self.threads)} threads.")

    def save(self, path: str) -> None:
        """Save the retrieval index to a pickle file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Store plain dictionaries instead of dataclass instances.
        # This avoids tying the pickle to a specific module execution path
        # such as "python -m src.retrieval" versus a script import.
        serializable_threads = [
            {
                "thread_id": thread.thread_id,
                "customer_text": thread.customer_text,
                "brand_replies": thread.brand_replies,
                "embedding": thread.embedding,
            }
            for thread in self.threads
        ]

        with open(output_path, "wb") as file:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "threads": serializable_threads,
                },
                file,
            )

        print(f"Saved index -> {output_path}")

    @classmethod
    def load(cls, path: str) -> "ThreadRetriever":
        """Load a previously saved retrieval index."""
        with open(path, "rb") as file:
            data = pickle.load(file)

        retriever = cls(model_name=data["model_name"])

        retriever.threads = [
            IndexedThread(
                thread_id=thread["thread_id"],
                customer_text=thread["customer_text"],
                brand_replies=thread["brand_replies"],
                embedding=thread["embedding"],
            )
            for thread in data["threads"]
        ]

        if retriever.threads:
            retriever._matrix = np.stack(
                [thread.embedding for thread in retriever.threads]
            )
        else:
            retriever._matrix = None

        return retriever

    def retrieve(
        self,
        query_text: str,
        k: int = 3,
        exclude_thread_id: str | None = None,
    ) -> list[dict]:
        """
        Return the k most similar indexed threads.

        Embeddings are normalized, so their dot product is equivalent to
        cosine similarity.

        exclude_thread_id:
            Exclude the source thread when generating a reply for a message
            that is already represented in the retrieval index. This prevents
            evaluation leakage by stopping the retriever from returning the
            message's own historical thread as a similarity-1.0 result.
        """
        if not self.threads or self._matrix is None:
            return []

        if k <= 0:
            return []

        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        )[0]

        similarities = self._matrix @ query_embedding

        # Retrieve one extra candidate so the excluded thread can be skipped
        # without reducing the requested number of results.
        top_indices = np.argsort(-similarities)[: k + 1]

        results = []

        for index in top_indices:
            thread = self.threads[index]

            if (
                exclude_thread_id is not None
                and thread.thread_id == exclude_thread_id
            ):
                continue

            results.append(
                {
                    "thread_id": thread.thread_id,
                    "customer_text": thread.customer_text,
                    "brand_replies": thread.brand_replies,
                    "similarity": float(similarities[index]),
                }
            )

            if len(results) == k:
                break

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Build or query the support-thread retrieval index."
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    build_parser = subparsers.add_parser(
        "build",
        help="Build a retrieval index from cleaned threads.",
    )
    build_parser.add_argument(
        "--input",
        required=True,
        help="Cleaned thread JSONL file.",
    )
    build_parser.add_argument(
        "--index",
        required=True,
        help="Output pickle file for the retrieval index.",
    )
    build_parser.add_argument(
        "--min-brand-turns",
        type=int,
        default=1,
        help="Minimum number of brand replies required.",
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Query an existing retrieval index.",
    )
    query_parser.add_argument(
        "--index",
        required=True,
        help="Path to the retrieval index.",
    )
    query_parser.add_argument(
        "--text",
        required=True,
        help="Customer message to use as the search query.",
    )
    query_parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of similar threads to return.",
    )

    args = parser.parse_args()

    if args.command == "build":
        retriever = ThreadRetriever()
        retriever.build(
            args.input,
            min_brand_turns=args.min_brand_turns,
        )
        retriever.save(args.index)

    elif args.command == "query":
        retriever = ThreadRetriever.load(args.index)
        results = retriever.retrieve(
            args.text,
            k=args.k,
        )

        for result in results:
            print(
                f"\n[sim={result['similarity']:.3f}] "
                f"thread={result['thread_id']}"
            )
            print(
                f"  customer: "
                f"{result['customer_text'][:150]}"
            )

            for reply in result["brand_replies"][:2]:
                print(f"  brand:    {reply[:150]}")


if __name__ == "__main__":
    main()
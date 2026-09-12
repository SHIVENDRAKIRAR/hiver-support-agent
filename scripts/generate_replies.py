"""
Generate grounded draft replies for customer messages, using retrieved
historical precedent (Part 4). Runs entirely locally via Ollama, same as
the classifier -- no API cost.

Usage:
    python scripts/generate_replies.py \
        --input data/golden_set.jsonl \
        --index data/thread_index.pkl \
        --out data/generated_replies.jsonl \
        --model llama3.1 --k 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from generate_prompt import REPLY_SYSTEM_PROMPT, build_generation_prompt  # noqa: E402
from retrieval import ThreadRetriever  # noqa: E402

import requests  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_one(model, customer_message, intent, retrieved, max_retries=3):
    prompt = f"{REPLY_SYSTEM_PROMPT}\n\n{build_generation_prompt(customer_message, intent, retrieved)}"
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
                timeout=90,
            )
            resp.raise_for_status()
            raw = resp.json()["response"].strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                start, end = raw.find("{"), raw.rfind("}")
                if start != -1 and end != -1:
                    parsed = json.loads(raw[start : end + 1])
                else:
                    raise
            return {
                "reply": parsed.get("reply", ""),
                "grounded": parsed.get("grounded", None),
                "raw": raw,
                "error": None,
            }
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                return {"reply": None, "grounded": None, "raw": raw, "error": "PARSE_ERROR"}
            time.sleep(0.5)
        except requests.exceptions.ConnectionError:
            print("ERROR: Could not connect to Ollama. Is it running?", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"reply": None, "grounded": None, "raw": str(e), "error": "API_ERROR"}
            time.sleep(1)
    return {"reply": None, "grounded": None, "raw": None, "error": "UNKNOWN_ERROR"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl with 'text' and 'gold_intent' or 'intent' fields")
    ap.add_argument("--index", required=True, help="path to thread_index.pkl from src/retrieval.py build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading retrieval index from {args.index} ...", file=sys.stderr)
    retriever = ThreadRetriever.load(args.index)
    print(f"Loaded index with {len(retriever.threads)} threads.", file=sys.stderr)

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["tweet_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if already_done:
            print(f"Resuming: {len(already_done)} already generated, skipping.", file=sys.stderr)

    remaining = [r for r in rows if r.get("tweet_id") not in already_done]
    print(f"{len(remaining)} remaining to generate.", file=sys.stderr)

    n_ok, n_err, n_ungrounded = 0, 0, 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, row in enumerate(remaining, 1):
            text = row["text"]
            intent = row.get("gold_intent") or row.get("intent") or "other_support_issue"

            retrieved = retriever.retrieve(text, k=args.k, exclude_thread_id=row.get("thread_id"))
            result = generate_one(args.model, text, intent, retrieved)

            out_row = {
                "thread_id": row.get("thread_id"),
                "tweet_id": row.get("tweet_id"),
                "text": text,
                "intent": intent,
                "retrieved_thread_ids": [r["thread_id"] for r in retrieved],
                "retrieved_similarities": [round(r["similarity"], 3) for r in retrieved],
                **result,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()

            if result["error"]:
                n_err += 1
            else:
                n_ok += 1
                if result["grounded"] is False:
                    n_ungrounded += 1

            if i % 10 == 0 or i == len(remaining):
                print(f"  {i}/{len(remaining)} done ({n_ok} ok, {n_err} errors, {n_ungrounded} flagged ungrounded)", file=sys.stderr)

    print(f"\nDone. {n_ok} generated, {n_err} errors, {n_ungrounded} flagged as ungrounded fallback. Output -> {out_path}")


if __name__ == "__main__":
    main()

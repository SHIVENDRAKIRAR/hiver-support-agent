"""
Score generated replies using an LLM judge (Part 6). Runs locally via
Ollama, same pattern as classification and generation.

Usage:
    python scripts/judge_replies.py \
        --input data/generated_replies.jsonl \
        --out data/judged_replies.jsonl \
        --model llama3.1
"""

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from judge_prompt import JUDGE_SYSTEM_PROMPT, build_judge_prompt  # noqa: E402

import requests  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"


def judge_one(model, customer_message, reply, intent, max_retries=3):
    prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{build_judge_prompt(customer_message, reply, intent)}"
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                timeout=60,
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
                "relevance": parsed.get("relevance"),
                "correctness": parsed.get("correctness"),
                "tone": parsed.get("tone"),
                "rationale": parsed.get("rationale"),
                "error": None,
            }
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                return {"relevance": None, "correctness": None, "tone": None, "rationale": raw, "error": "PARSE_ERROR"}
            time.sleep(0.5)
        except requests.exceptions.ConnectionError:
            print("ERROR: Could not connect to Ollama. Is it running?", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"relevance": None, "correctness": None, "tone": None, "rationale": str(e), "error": "API_ERROR"}
            time.sleep(1)
    return {"relevance": None, "correctness": None, "tone": None, "rationale": None, "error": "UNKNOWN_ERROR"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl from generate_replies.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

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
            print(f"Resuming: {len(already_done)} already judged, skipping.", file=sys.stderr)

    remaining = [r for r in rows if r.get("tweet_id") not in already_done and r.get("reply")]
    print(f"{len(remaining)} remaining to judge.", file=sys.stderr)

    n_ok, n_err = 0, 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, row in enumerate(remaining, 1):
            scores = judge_one(args.model, row["text"], row["reply"], row.get("intent", ""))
            out_row = {
                "thread_id": row.get("thread_id"),
                "tweet_id": row.get("tweet_id"),
                "text": row["text"],
                "intent": row.get("intent"),
                "reply": row["reply"],
                "generation_grounded": row.get("grounded"),
                **scores,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()

            if scores["error"]:
                n_err += 1
            else:
                n_ok += 1

            if i % 10 == 0 or i == len(remaining):
                print(f"  {i}/{len(remaining)} done ({n_ok} ok, {n_err} errors)", file=sys.stderr)

    print(f"\nDone. {n_ok} judged, {n_err} errors. Output -> {out_path}")


if __name__ == "__main__":
    main()

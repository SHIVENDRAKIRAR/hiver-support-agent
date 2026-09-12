"""
Classify sampled customer messages using a LOCAL Ollama model (Llama 3.1
or similar). No API key, no rate limits, no cost -- runs entirely on your
machine via Ollama's local HTTP server (default: http://localhost:11434).

Prerequisite:
    1. Install Ollama: https://ollama.com/download
    2. ollama pull llama3.1
    3. Make sure `ollama serve` is running (it usually auto-starts after install)

Usage:
    python scripts/classify_with_ollama.py \
        --input data/sample_for_classification.jsonl \
        --out data/classified_sample.jsonl \
        --model llama3.1
"""

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from classify_prompt import SYSTEM_PROMPT, build_user_prompt, validate_label  # noqa: E402

import requests  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"


def classify_one(model, text, max_retries=3):
    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(text)}"
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()["response"].strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            # local models sometimes wrap output in extra prose despite instructions --
            # try to extract the first {...} block if direct parse fails
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                start, end = raw.find("{"), raw.rfind("}")
                if start != -1 and end != -1:
                    parsed = json.loads(raw[start : end + 1])
                else:
                    raise
            intent = parsed.get("intent", "")
            confidence = parsed.get("confidence", None)
            if not validate_label(intent):
                return {"intent": "INVALID_LABEL", "confidence": None, "raw": raw}
            return {"intent": intent, "confidence": confidence, "raw": raw}
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                return {"intent": "PARSE_ERROR", "confidence": None, "raw": raw}
            time.sleep(0.5)
        except requests.exceptions.ConnectionError:
            print(
                "ERROR: Could not connect to Ollama at localhost:11434. "
                "Is Ollama running? Try: ollama serve",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"intent": "API_ERROR", "confidence": None, "raw": str(e)}
            time.sleep(1)
    return {"intent": "UNKNOWN_ERROR", "confidence": None, "raw": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--limit", type=int, default=0, help="cap number of rows for a quick test run")
    args = ap.parse_args()

    print(f"Using local Ollama model: {args.model}", file=sys.stderr)

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
            print(f"Resuming: {len(already_done)} rows already classified, skipping those.", file=sys.stderr)

    remaining = [r for r in rows if r["tweet_id"] not in already_done]
    print(f"{len(remaining)} rows remaining to classify.", file=sys.stderr)

    n_ok, n_err = 0, 0
    start_time = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, row in enumerate(remaining, 1):
            result = classify_one(args.model, row["text"])
            row_out = {**row, **result}
            fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            fout.flush()

            if result["intent"] in ("INVALID_LABEL", "PARSE_ERROR", "API_ERROR", "UNKNOWN_ERROR"):
                n_err += 1
            else:
                n_ok += 1

            if i % 10 == 0 or i == len(remaining):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_min = (len(remaining) - i) / rate / 60 if rate > 0 else 0
                print(
                    f"  {i}/{len(remaining)} done ({n_ok} ok, {n_err} errors) "
                    f"| {rate:.2f} req/s | ETA {eta_min:.1f} min",
                    file=sys.stderr,
                )

    print(f"\nDone. {n_ok} classified, {n_err} errors this run. Output -> {out_path}")


if __name__ == "__main__":
    main()

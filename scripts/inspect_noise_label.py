"""
Inspect messages the LLM classified as non_support_noise, to check whether
it's correctly catching low-signal chatter or over-flagging genuine (if
brief) support requests as noise.

Usage:
    python scripts/inspect_noise_label.py --input data/classified_sample.jsonl
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=40, help="how many to print")
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    noise_rows = [r for r in rows if r["intent"] == "non_support_noise"]
    print(f"Total non_support_noise: {len(noise_rows)}\n")

    for r in noise_rows[: args.n]:
        conf = r.get("confidence", "?")
        print(f"[conf={conf}] {r['text']}")


if __name__ == "__main__":
    main()

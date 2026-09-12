"""
Fast, honest spot-check labeling: a SMALL stratified sample (default 50)
pulled from the already-built golden_set_candidates.jsonl, meant to
actually be read and judged carefully rather than rubber-stamped.

This exists because a 200-example labeling pass is easy to rush through
without really reading each one -- a properly-attended 50-example check
is worth more than a rushed 200-example one for reporting a real human-vs-LLM
agreement rate.

Usage:
    python scripts/spot_check_golden_set.py \
        --input data/golden_set_candidates.jsonl \
        --out data/golden_set_spotcheck.jsonl \
        --n 50 --seed 1
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from taxonomy import INTENTS, NON_SUPPORT_LABELS  # noqa: E402

ALL_LABELS = INTENTS + NON_SUPPORT_LABELS


def prompt_intent(suggested: str) -> str:
    print("\nLabels:")
    for i, label in enumerate(ALL_LABELS, 1):
        marker = " <-- suggested" if label == suggested else ""
        print(f"  {i:2d}. {label}{marker}")
    while True:
        raw = input(f"\nRead the message above carefully. Intent [Enter = accept '{suggested}', number = correct, s = skip]: ").strip()
        if raw == "":
            return suggested
        if raw.lower() == "s":
            return "SKIPPED"
        if raw.isdigit() and 1 <= int(raw) <= len(ALL_LABELS):
            return ALL_LABELS[int(raw) - 1]
        print("  Invalid input, try again.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    candidates = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))

    # stratify the spot-check sample across whatever labels are present,
    # so 50 examples still touch most classes rather than randomly
    # clumping on the biggest one
    by_class = defaultdict(list)
    for row in candidates:
        by_class[row.get("intent", "unknown")].append(row)

    random.seed(args.seed)
    n_classes = len(by_class)
    per_class = max(1, args.n // n_classes)

    selected = []
    for cls, rows in by_class.items():
        random.shuffle(rows)
        selected.extend(rows[:per_class])

    random.shuffle(selected)
    selected = selected[: args.n]

    print(f"Spot-check sample: {len(selected)} examples across {n_classes} classes.\n")
    print("This is a SMALL, CAREFUL check -- actually read each message.\n")

    out_path = Path(args.out)
    fout = open(out_path, "a", encoding="utf-8")
    n_done = 0

    try:
        for idx, row in enumerate(selected, 1):
            print("\n" + "=" * 70)
            print(f"[{idx}/{len(selected)}]")
            print("-" * 70)
            print(row["text"])
            print("-" * 70)

            suggested = row.get("intent", "other_support_issue")
            gold = prompt_intent(suggested)
            if gold == "SKIPPED":
                continue

            out_row = {
                "thread_id": row.get("thread_id"),
                "tweet_id": row.get("tweet_id"),
                "text": row["text"],
                "llm_suggested_intent": suggested,
                "gold_intent": gold,
                "human_agreed_with_llm": gold == suggested,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
    except KeyboardInterrupt:
        print(f"\nStopped early. {n_done} done this session.")
    finally:
        fout.close()

    print(f"\nDone: {n_done} carefully spot-checked. Output -> {out_path}")


if __name__ == "__main__":
    main()

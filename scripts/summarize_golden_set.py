"""
Summarize the completed golden set: overall human-vs-LLM agreement rate,
per-class breakdown, and where the LLM's suggestion was overridden.

Usage:
    python scripts/summarize_golden_set.py --input data/golden_set.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    total = len(rows)
    agreed = sum(1 for r in rows if r["human_agreed_with_llm"])

    print(f"--- GOLDEN SET SUMMARY ---")
    print(f"Total labeled: {total}")
    print(f"Human agreed with LLM suggestion: {agreed} ({100*agreed/total:.1f}%)")
    print(f"Human overrode LLM suggestion:    {total-agreed} ({100*(total-agreed)/total:.1f}%)")

    print(f"\nGold label distribution:")
    gold_counts = Counter(r["gold_intent"] for r in rows)
    for label, count in gold_counts.most_common():
        print(f"  {label:35s} {count:4d}")

    print(f"\nDisagreements by LLM-suggested -> human-corrected:")
    disagreements = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if not r["human_agreed_with_llm"]:
            disagreements[r["llm_suggested_intent"]][r["gold_intent"]] += 1

    for suggested, corrections in sorted(disagreements.items()):
        for corrected, count in sorted(corrections.items(), key=lambda x: -x[1]):
            print(f"  {suggested:30s} -> {corrected:30s}  ({count})")

    print(f"\nPer-class agreement rate:")
    by_suggested = defaultdict(lambda: {"total": 0, "agreed": 0})
    for r in rows:
        s = r["llm_suggested_intent"]
        by_suggested[s]["total"] += 1
        if r["human_agreed_with_llm"]:
            by_suggested[s]["agreed"] += 1

    for label, stats in sorted(by_suggested.items(), key=lambda x: -x[1]["total"]):
        rate = 100 * stats["agreed"] / stats["total"] if stats["total"] else 0
        print(f"  {label:35s} {stats['agreed']:3d}/{stats['total']:3d}  ({rate:5.1f}%)")


if __name__ == "__main__":
    main()

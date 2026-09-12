"""
Draw a STRATIFIED sample for the golden evaluation set (Part 3).

Why stratified, not random: a pure random sample of ~200 messages would
mirror the natural class distribution (delivery_delay ~16%, billing ~3%),
giving some intents so few examples that per-class accuracy is meaningless
noise. We instead allocate a target count per class, with a floor so even
rare classes get enough examples to say something real about them.

Allocation strategy:
  1. Start from the real observed class frequencies (from the 1000-row
     classified sample in Part 2).
  2. Allocate proportionally to TOTAL_TARGET, but enforce a MIN_PER_CLASS
     floor -- classes below the floor get boosted to it, and the excess is
     taken proportionally from the classes above it.
  3. Sample the required count for each class from the FULL 65k+ pool
     (re-classifying via the locked classifier if needed) -- not just the
     1000-row sample, since 200 stratified examples need a bigger backing
     pool per class than the 1000-row sample alone reliably provides for
     rare classes.

This script assumes you already have a LARGER classified pool to draw
from. If your classified_sample.jsonl only has 1000 rows, run the
classifier on a bigger sample first (recommended: 3000-5000) so every
class -- including rare ones like billing at ~3% -- has enough labeled
candidates to sample MIN_PER_CLASS from without exhausting the pool.

Usage:
    python scripts/build_golden_set_sample.py \
        --input data/classified_sample.jsonl \
        --out data/golden_set_candidates.jsonl \
        --total 200 --min-per-class 12 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict


def allocate(class_counts: dict, total_target: int, min_per_class: int) -> dict:
    """
    Allocate `total_target` examples across classes, proportional to
    class_counts, with each class guaranteed at least min_per_class
    (capped at however many are actually available for that class).
    """
    classes = list(class_counts.keys())
    available = dict(class_counts)  # how many exist in the pool per class
    n_classes = len(classes)

    # start everyone at their floor (capped by availability)
    allocation = {c: min(min_per_class, available[c]) for c in classes}
    used = sum(allocation.values())
    remaining = total_target - used

    if remaining <= 0:
        return allocation

    # distribute the remainder proportionally to (availability - floor already given)
    remaining_capacity = {c: available[c] - allocation[c] for c in classes}
    total_capacity = sum(remaining_capacity.values())

    if total_capacity == 0:
        return allocation

    for c in classes:
        share = remaining * (remaining_capacity[c] / total_capacity)
        add = min(int(round(share)), remaining_capacity[c])
        allocation[c] += add

    return allocation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="classified pool (jsonl with 'intent' field)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--min-per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_class = defaultdict(list)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            intent = row.get("intent")
            # skip error/invalid rows -- not eligible for the golden set
            if intent in ("INVALID_LABEL", "PARSE_ERROR", "API_ERROR", "UNKNOWN_ERROR", "RATE_LIMITED_SKIPPED"):
                continue
            by_class[intent].append(row)

    class_counts = {c: len(rows) for c, rows in by_class.items()}
    print("Available pool per class:")
    for c, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:35s} {n:5d}")

    allocation = allocate(class_counts, args.total, args.min_per_class)

    print(f"\nAllocation (target={args.total}, floor={args.min_per_class}):")
    for c, n in sorted(allocation.items(), key=lambda x: -x[1]):
        flag = "  <-- below floor, pool exhausted" if n < args.min_per_class else ""
        print(f"  {c:35s} {n:5d}{flag}")

    random.seed(args.seed)
    selected = []
    for c, n in allocation.items():
        pool = by_class[c]
        random.shuffle(pool)
        chosen = pool[:n]
        for row in chosen:
            row["_sampled_for_class"] = c
        selected.extend(chosen)

    random.shuffle(selected)  # so labeling order doesn't cluster by class

    with open(args.out, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(selected)} candidates -> {args.out}")


if __name__ == "__main__":
    main()

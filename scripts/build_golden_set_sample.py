"""
Build a stratified sample for the golden evaluation set.

The allocation is based on observed class frequencies while enforcing a
minimum number of examples per class.

Usage:
    python scripts/build_golden_set_sample.py \
        --input data/classified_sample.jsonl \
        --out data/golden_set_candidates.jsonl \
        --total 200 \
        --min-per-class 12 \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict


EXCLUDED_LABELS = {
    "INVALID_LABEL",
    "PARSE_ERROR",
    "API_ERROR",
    "UNKNOWN_ERROR",
    "RATE_LIMITED_SKIPPED",
}


def allocate(
    class_counts: dict,
    total_target: int,
    min_per_class: int,
) -> dict:
    """Allocate examples proportionally while enforcing a minimum per class."""
    allocation = {
        label: min(min_per_class, count)
        for label, count in class_counts.items()
    }

    remaining = total_target - sum(allocation.values())

    if remaining <= 0:
        return allocation

    capacity = {
        label: count - allocation[label]
        for label, count in class_counts.items()
    }

    total_capacity = sum(capacity.values())

    if total_capacity == 0:
        return allocation

    for label, available in capacity.items():
        share = remaining * available / total_capacity
        allocation[label] += min(round(share), available)

    return allocation


def load_classified_pool(input_path: str) -> dict:
    """Load valid classified records and group them by intent."""
    by_class = defaultdict(list)

    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            intent = row.get("intent")

            if intent not in EXCLUDED_LABELS and intent:
                by_class[intent].append(row)

    return by_class


def print_pool_summary(class_counts: dict) -> None:
    """Print the number of available examples for each class."""
    print("Available pool per class:")

    for label, count in sorted(
        class_counts.items(),
        key=lambda item: -item[1],
    ):
        print(f"  {label:35s} {count:5d}")


def print_allocation_summary(
    allocation: dict,
    total_target: int,
    min_per_class: int,
) -> None:
    """Print the final allocation for each class."""
    print(
        f"\nAllocation (target={total_target}, floor={min_per_class}):"
    )

    for label, count in sorted(
        allocation.items(),
        key=lambda item: -item[1],
    ):
        warning = (
            "  <-- below floor, pool exhausted"
            if count < min_per_class
            else ""
        )
        print(f"  {label:35s} {count:5d}{warning}")


def sample_records(by_class: dict, allocation: dict, seed: int) -> list:
    """Sample the requested number of records from each class."""
    rng = random.Random(seed)
    selected = []

    for label, count in allocation.items():
        pool = by_class[label].copy()
        rng.shuffle(pool)

        for row in pool[:count]:
            row["_sampled_for_class"] = label
            selected.append(row)

    rng.shuffle(selected)
    return selected


def write_jsonl(rows: list, output_path: str) -> None:
    """Write records to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build a stratified sample for the golden evaluation set."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Classified JSONL pool containing an 'intent' field.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=200,
        help="Target number of examples.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=12,
        help="Minimum number of examples per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    by_class = load_classified_pool(args.input)
    class_counts = {
        label: len(rows)
        for label, rows in by_class.items()
    }

    print_pool_summary(class_counts)

    allocation = allocate(
        class_counts,
        args.total,
        args.min_per_class,
    )
    print_allocation_summary(
        allocation,
        args.total,
        args.min_per_class,
    )

    selected = sample_records(
        by_class,
        allocation,
        args.seed,
    )

    write_jsonl(selected, args.out)

    print(f"\nWrote {len(selected)} candidates -> {args.out}")


if __name__ == "__main__":
    main()
"""
Analyze class balance in an LLM-classified sample.

Usage:
    python scripts/analyze_class_balance.py \
        --input data/classified_sample.jsonl
"""

import argparse
import json
from collections import Counter


ERROR_LABELS = {
    "INVALID_LABEL",
    "PARSE_ERROR",
    "API_ERROR",
    "UNKNOWN_ERROR",
}

LOW_CONFIDENCE_THRESHOLD = 0.6
LOW_CONFIDENCE_SAMPLE_SIZE = 10


def load_rows(input_path):
    """Load classified records from a JSONL file."""
    with open(input_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def build_report(rows):
    """Build a class-balance report from classified records."""
    counts = Counter(row["intent"] for row in rows)
    total = len(rows)

    if total == 0:
        return "--- CLASS BALANCE REPORT ---\ntotal classified: 0"

    errors = sum(
        count for label, count in counts.items() if label in ERROR_LABELS
    )

    lines = [
        "--- CLASS BALANCE REPORT ---",
        f"total classified: {total}",
        "",
    ]

    for label, count in counts.most_common():
        percentage = 100 * count / total
        lines.append(f"{label:35s} {count:5d}  ({percentage:5.1f}%)")

    error_percentage = 100 * errors / total
    lines.extend(
        [
            "",
            f"errors/invalid: {errors} ({error_percentage:.1f}%)",
        ]
    )

    low_confidence_rows = [
        row
        for row in rows
        if row.get("confidence") is not None
        and row["confidence"] < LOW_CONFIDENCE_THRESHOLD
    ]

    lines.append(
        f"\nlow-confidence (<{LOW_CONFIDENCE_THRESHOLD}) "
        f"classifications: {len(low_confidence_rows)}"
    )

    if low_confidence_rows:
        lines.append("sample of low-confidence cases:")

        for row in low_confidence_rows[:LOW_CONFIDENCE_SAMPLE_SIZE]:
            lines.append(
                f"  [{row['intent']} conf={row['confidence']}] "
                f"{row['text'][:100]}"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze class balance in an LLM-classified sample."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--out",
        default="data/class_balance_report.txt",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    report = build_report(rows)

    print(report)

    with open(args.out, "w", encoding="utf-8") as file:
        file.write(report + "\n")

    print(f"\n(also written to {args.out})")


if __name__ == "__main__":
    main()
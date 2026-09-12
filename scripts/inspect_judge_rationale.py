"""
Preview judged reply evaluations.

Displays the first few records from the judged replies file, including
the generated reply, evaluation scores, and judge rationale.

Usage:
    python scripts/preview_judged_replies.py
"""

import json


INPUT_PATH = "data/judged_replies.jsonl"
PREVIEW_COUNT = 8


def load_rows(input_path: str) -> list[dict]:
    """Load evaluation records from a JSONL file."""
    with open(input_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def main():
    rows = load_rows(INPUT_PATH)

    for row in rows[:PREVIEW_COUNT]:
        print(row["text"][:80])
        print(f"  Reply: {row['reply'][:80]}")
        print(
            f"  Scores: relevance={row['relevance']}, "
            f"correctness={row['correctness']}, "
            f"tone={row['tone']}"
        )
        print(f"  Rationale: {row['rationale']}")
        print()


if __name__ == "__main__":
    main()
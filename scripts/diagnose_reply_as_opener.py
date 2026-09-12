"""
Measure how often the extracted first customer message appears to be
a mid-conversation reply rather than the true thread opener.

Usage:
    python scripts/diagnose_reply_as_opener.py \
        --input data/threads_AmazonHelp_clean.jsonl
"""

import argparse
import json
import re


REPLY_MARKER_RE = re.compile(
    r"^\s*(yes|no|ok(ay)?|already|done|ive |i've |i have|not yet|"
    r"p\.s\.?|as a matter of fact|thank you|thanks|will do)\b",
    re.IGNORECASE,
)

BARE_REFERENCE_RE = re.compile(
    r"^\s*(this|that|it|this one)\b.{0,20}$",
    re.IGNORECASE,
)

MAX_EXAMPLES = 15


def looks_like_reply_not_opener(text: str) -> bool:
    """Return True if the message appears to be a mid-conversation reply."""
    text = text.strip()

    return bool(
        REPLY_MARKER_RE.match(text)
        or BARE_REFERENCE_RE.match(text)
    )


def get_first_customer_message(thread: dict) -> str | None:
    """Return the first non-brand message with cleaned text."""
    for turn in thread["turns"]:
        if not turn["is_brand"] and turn.get("text_clean"):
            return turn["text_clean"]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose likely mid-conversation thread openers."
    )
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    total = 0
    flagged = 0
    examples = []

    with open(args.input, "r", encoding="utf-8") as file:
        for line in file:
            thread = json.loads(line)
            first_customer = get_first_customer_message(thread)

            if not first_customer:
                continue

            total += 1

            if looks_like_reply_not_opener(first_customer):
                flagged += 1

                if len(examples) < MAX_EXAMPLES:
                    examples.append(first_customer)

    percentage = 100 * flagged / total if total else 0

    print(f"Total threads checked: {total}")
    print(
        f"Flagged as likely reply-not-opener: "
        f"{flagged} ({percentage:.1f}%)"
    )

    print("\nExamples:")
    for example in examples:
        print(f"  - {example[:120]}")


if __name__ == "__main__":
    main()
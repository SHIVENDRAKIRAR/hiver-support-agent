"""
Clean and normalize thread text for classification, reply generation, and RAG.

The cleaning pipeline:
- Removes @mentions while preserving them in a separate field.
- Removes trailing two-letter agent sign-off codes.
- Unescapes HTML entities.
- Normalizes whitespace.
- Redacts common PII patterns such as emails, phone numbers, and order references.

Usage:
    python scripts/clean_threads.py \
        --input data/threads_AmazonHelp.jsonl \
        --output data/threads_AmazonHelp_clean.jsonl
"""

import argparse
import html
import json
import re


MENTION_RE = re.compile(r"@\w+")
SIGNOFF_RE = re.compile(r"\s\^\*[A-Za-z]{2}\s\*$")
WHITESPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\b\d{10,}\b")
ORDER_RE = re.compile(
    r"\b(?:order|case|ref)[\s#:]*[\w-]{6,}\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> tuple[str, list[str]]:
    """Clean a message and return the cleaned text and extracted mentions."""
    text = html.unescape(text)

    mentions = MENTION_RE.findall(text)
    text = MENTION_RE.sub("", text)

    text = SIGNOFF_RE.sub("", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = ORDER_RE.sub("[ORDER_REF]", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text, mentions


def main():
    parser = argparse.ArgumentParser(
        description="Clean and normalize AmazonHelp thread text."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_count = 0
    output_count = 0

    with (
        open(args.input, "r", encoding="utf-8") as input_file,
        open(args.output, "w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            input_count += 1
            thread = json.loads(line)

            for turn in thread["turns"]:
                cleaned_text, mentions = clean_text(turn["text"])
                turn["text_clean"] = cleaned_text
                turn["mentions"] = mentions

            has_customer_text = any(
                turn["text_clean"] and not turn["is_brand"]
                for turn in thread["turns"]
            )

            if has_customer_text:
                output_file.write(
                    json.dumps(thread, ensure_ascii=False) + "\n"
                )
                output_count += 1

    print(
        f"Read {input_count} threads, wrote "
        f"{output_count} cleaned threads to {args.output}"
    )


if __name__ == "__main__":
    main()
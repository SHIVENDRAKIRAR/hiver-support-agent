"""
Clean and normalize thread text for downstream classification / reply / RAG work.

- Strips @mentions used for routing (keeps them out of model input, but
  keeps a record of who was mentioned)
- Removes the trailing "^XY" agent-signoff codes common on brand support tweets
- Normalizes whitespace, unescapes HTML entities
- Redacts obvious PII patterns (order numbers, emails, phone-like digit runs)
  -- crude regex-based redaction, good enough for a support-ticket dataset;
     flagged in the report as a known limitation, not a compliance-grade PII filter

Usage:
    python scripts/clean_threads.py --input data/threads_AmazonHelp.jsonl \
        --output data/threads_AmazonHelp_clean.jsonl
"""

import argparse
import html
import json
import re

MENTION_RE = re.compile(r"@\w+")
SIGNOFF_RE = re.compile(r"\s*\^[A-Za-z]{2}\s*$")
WHITESPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\b\d{10,}\b")
ORDER_RE = re.compile(r"\b(?:order|case|ref)[\s#:]*[\w-]{6,}\b", re.IGNORECASE)


def clean_text(text: str):
    original = text
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    n_in, n_out = 0, 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            obj = json.loads(line)
            for turn in obj["turns"]:
                clean, mentions = clean_text(turn["text"])
                turn["text_clean"] = clean
                turn["mentions"] = mentions
            # drop threads that ended up with no substantive customer turn
            if any(t["text_clean"] and not t["is_brand"] for t in obj["turns"]):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"Read {n_in} threads, wrote {n_out} cleaned threads to {args.output}")


if __name__ == "__main__":
    main()

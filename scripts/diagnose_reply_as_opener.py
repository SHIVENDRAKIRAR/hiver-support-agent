"""
Diagnostic: measure how often "first customer turn" extraction is actually
grabbing a mid-conversation reply rather than the true opening message of a
thread. This matters because our thread reconstruction (Part 1) walks
BACKWARD from any brand tweet to find a root via in_response_to_tweet_id,
then takes the first non-brand turn in that thread -- but if the true root
tweet is missing from the dataset (a common gap in this corpus, since
threads can start outside the collection window), the reconstructed
"thread" silently starts mid-conversation instead.

Heuristic signal for "this is probably a reply, not a fresh complaint":
- starts with acknowledgement words (yes, no, ok, already, done, thanks)
- starts with a continuation marker (P.S., also, and)
- is a bare fragment referencing "this"/"that"/"it" with no other context
- contains only a phone/email/order-ref and an instruction ("please call me")

This is intentionally a rough heuristic for MEASUREMENT only -- not meant
to be a production filter. Its purpose is to answer: "is this worth fixing
in Part 1, or is it rare enough to document as a known limitation?"

Usage:
    python scripts/diagnose_reply_as_opener.py --input data/threads_AmazonHelp_clean.jsonl
"""

import argparse
import json
import re

REPLY_MARKER_RE = re.compile(
    r"^\s*(yes|no|ok(ay)?|already|done|ive |i've |i have|not yet|"
    r"p\.?s\.?|as a matter of fact|thank you|thanks|will do)\b",
    re.IGNORECASE,
)

BARE_REFERENCE_RE = re.compile(
    r"^\s*(this|that|it|this one)\b.{0,20}$",
    re.IGNORECASE,
)


def looks_like_reply_not_opener(text: str) -> bool:
    stripped = text.strip()
    if REPLY_MARKER_RE.match(stripped):
        return True
    if BARE_REFERENCE_RE.match(stripped):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    total = 0
    flagged = 0
    examples = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            first_customer = None
            for turn in obj["turns"]:
                if not turn["is_brand"] and turn.get("text_clean"):
                    first_customer = turn["text_clean"]
                    break
            if not first_customer:
                continue
            total += 1
            if looks_like_reply_not_opener(first_customer):
                flagged += 1
                if len(examples) < 15:
                    examples.append(first_customer)

    print(f"Total threads checked: {total}")
    print(f"Flagged as likely reply-not-opener: {flagged} ({100*flagged/total:.1f}%)")
    print("\nExamples:")
    for e in examples:
        print(f"  - {e[:120]}")


if __name__ == "__main__":
    main()

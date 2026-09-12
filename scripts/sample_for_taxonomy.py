"""
Sample customer opening messages for taxonomy volume-checking, using the
locked filter_is_support_signal() heuristic (not a length cutoff) and a
lightweight language check to separate English (in-scope) from
non-English (language_out_of_scope).

Usage:
    python scripts/sample_for_taxonomy.py \
        --input data/threads_AmazonHelp_clean.jsonl --n 400 --seed 42
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from taxonomy import filter_is_support_signal  # noqa: E402

# Very rough English heuristic: flag messages with a high proportion of
# non-ASCII letters (catches Japanese/Chinese cleanly) or common non-English
# stopword patterns (catches French/Spanish/German/Portuguese reasonably well
# for a first pass). This is NOT a real language detector -- good enough to
# triage volume, not to make the final in/out-of-scope call.
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
FOREIGN_HINTS = re.compile(
    r"\b(le|la|les|des|est|pas|vous|merci|bonjour|und|ich|nicht|ist|das|"
    r"que|para|gracias|hola|está|não|obrigad|você|com|muito)\b",
    re.IGNORECASE,
)


def looks_non_english(text: str) -> bool:
    non_ascii_ratio = len(NON_ASCII_RE.findall(text)) / max(len(text), 1)
    if non_ascii_ratio > 0.15:
        return True
    if FOREIGN_HINTS.search(text):
        return True
    return False


def first_customer_turn(thread_obj):
    for turn in thread_obj["turns"]:
        if not turn["is_brand"] and turn.get("text_clean"):
            return turn["text_clean"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_msgs = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            msg = first_customer_turn(obj)
            if msg:
                all_msgs.append((obj["thread_id"], msg))

    total = len(all_msgs)
    kept_signal = [(tid, m) for tid, m in all_msgs if filter_is_support_signal(m)]
    english = [(tid, m) for tid, m in kept_signal if not looks_non_english(m)]
    non_english = [(tid, m) for tid, m in kept_signal if looks_non_english(m)]

    random.seed(args.seed)
    sample = random.sample(english, min(args.n, len(english)))

    for tid, msg in sample:
        print(f"[{tid}] {msg}")

    print(f"\n--- STATS ---")
    print(f"total opening messages:           {total}")
    print(f"dropped as non_support_noise:     {total - len(kept_signal)}")
    print(f"kept as support-signal:           {len(kept_signal)}")
    print(f"  of which flagged non-English:   {len(non_english)} (language_out_of_scope candidates)")
    print(f"  of which English (in-scope):    {len(english)}")
    print(f"sampled for taxonomy review:      {len(sample)}")


if __name__ == "__main__":
    main()

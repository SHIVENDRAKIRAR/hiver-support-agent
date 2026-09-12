"""
Standalone stats counter for taxonomy volume-checking. Writes results to a
file (not just stdout) so nothing gets lost to terminal buffering/truncation.

Usage:
    python scripts/taxonomy_stats.py --input data/threads_AmazonHelp_clean.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from taxonomy import filter_is_support_signal  # noqa: E402

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
    ap.add_argument("--out", default="data/taxonomy_stats.txt")
    args = ap.parse_args()

    all_msgs = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            msg = first_customer_turn(obj)
            if msg:
                all_msgs.append(msg)

    total = len(all_msgs)
    kept_signal = [m for m in all_msgs if filter_is_support_signal(m)]
    english = [m for m in kept_signal if not looks_non_english(m)]
    non_english = [m for m in kept_signal if looks_non_english(m)]

    lines = [
        "--- TAXONOMY STATS ---",
        f"total opening messages:           {total}",
        f"dropped as non_support_noise:     {total - len(kept_signal)}",
        f"kept as support-signal:           {len(kept_signal)}",
        f"  of which flagged non-English:   {len(non_english)} (language_out_of_scope candidates)",
        f"  of which English (in-scope):    {len(english)}",
    ]

    text = "\n".join(lines)
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n(also written to {args.out})")


if __name__ == "__main__":
    main()

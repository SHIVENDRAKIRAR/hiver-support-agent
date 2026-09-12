
import argparse
import json
import random
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
            return turn["tweet_id"], turn["text_clean"]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/sample_for_classification.jsonl")
    args = ap.parse_args()

    eligible = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tweet_id, msg = first_customer_turn(obj)
            if msg and filter_is_support_signal(msg) and not looks_non_english(msg):
                eligible.append({
                    "thread_id": obj["thread_id"],
                    "tweet_id": tweet_id,
                    "text": msg,
                })

    random.seed(args.seed)
    sample = random.sample(eligible, min(args.n, len(eligible)))

    with open(args.out, "w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Drew {len(sample)} of {len(eligible)} eligible messages -> {args.out}")


if __name__ == "__main__":
    main()

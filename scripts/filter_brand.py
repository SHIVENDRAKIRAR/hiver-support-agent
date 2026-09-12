"""
Filter the full 'Customer Support on Twitter' dataset (twcs.csv) down to
conversation threads involving a single target brand (default: AmazonHelp).

Usage:
    python scripts/filter_brand.py --input data/twcs.csv --brand AmazonHelp

Output:
    data/threads_<brand>.jsonl   -- one reconstructed conversation per line
    data/threads_<brand>_stats.json -- basic stats about the filtered subset

A "thread" here is a chain of tweets linked via `in_response_to_tweet_id`
and `response_tweet_id`, trimmed to just the (customer, brand) turns
belonging to a single root customer complaint. We keep the FULL thread
(both directions) as long as the brand account appears somewhere in it.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)


def load_rows(path: str):
    """Stream-load the CSV into a dict keyed by tweet_id."""
    rows = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["tweet_id"]] = row
    return rows


def build_threads(rows: dict, brand: str):
    """
    Reconstruct full conversation threads.
    Strategy:
      1. Find every tweet authored by `brand`.
      2. Walk backwards via in_response_to_tweet_id to find the thread root
         (the original customer complaint that started the conversation).
      3. Walk forwards via response_tweet_id to capture the full back-and-forth.
      4. Deduplicate threads by root tweet id.
    """
    brand_tweet_ids = [
        tid for tid, r in rows.items() if r["author_id"] == brand
    ]

    def find_root(tweet_id):
        seen = set()
        cur = tweet_id
        while True:
            if cur in seen or cur not in rows:
                return cur
            seen.add(cur)
            parent = rows[cur].get("in_response_to_tweet_id", "")
            if not parent or parent == "" or parent not in rows:
                return cur
            cur = parent

    def collect_forward(root_id):
        """BFS forward from root using response_tweet_id (can be comma-separated)."""
        thread = []
        queue = [root_id]
        seen = set()
        while queue:
            tid = queue.pop(0)
            if tid in seen or tid not in rows:
                continue
            seen.add(tid)
            row = rows[tid]
            thread.append(row)
            children = row.get("response_tweet_id", "") or ""
            for child in children.split(","):
                child = child.strip()
                if child and child not in seen:
                    queue.append(child)
        return thread

    roots_seen = set()
    threads = []
    for btid in brand_tweet_ids:
        root_id = find_root(btid)
        if root_id in roots_seen:
            continue
        roots_seen.add(root_id)
        thread_rows = collect_forward(root_id)
        # only keep if brand actually appears in this thread
        if any(r["author_id"] == brand for r in thread_rows):
            thread_rows_sorted = sorted(
                thread_rows, key=lambda r: r["tweet_id"]
            )
            threads.append(thread_rows_sorted)

    return threads


def to_clean_dict(thread_rows):
    turns = []
    for r in thread_rows:
        turns.append({
            "tweet_id": r["tweet_id"],
            "author_id": r["author_id"],
            "is_brand": r["inbound"] == "False",
            "created_at": r["created_at"],
            "text": r["text"],
            "in_response_to_tweet_id": r.get("in_response_to_tweet_id") or None,
            "response_tweet_id": r.get("response_tweet_id") or None,
        })
    return {
        "thread_id": thread_rows[0]["tweet_id"],
        "num_turns": len(turns),
        "turns": turns,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to twcs.csv")
    ap.add_argument("--brand", default="AmazonHelp", help="Brand author_id to filter to")
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--max-threads", type=int, default=0, help="Optional cap for quick testing (0 = no cap)")
    args = ap.parse_args()

    print(f"Loading {args.input} ...", file=sys.stderr)
    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} tweets total", file=sys.stderr)

    print(f"Building threads for brand={args.brand} ...", file=sys.stderr)
    threads = build_threads(rows, args.brand)
    print(f"Found {len(threads)} threads involving {args.brand}", file=sys.stderr)

    if args.max_threads:
        threads = threads[: args.max_threads]

    out_path = f"{args.outdir}/threads_{args.brand}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(to_clean_dict(t), ensure_ascii=False) + "\n")

    turn_counts = [len(t) for t in threads]
    stats = {
        "brand": args.brand,
        "num_threads": len(threads),
        "avg_turns_per_thread": sum(turn_counts) / len(turn_counts) if turn_counts else 0,
        "min_turns": min(turn_counts) if turn_counts else 0,
        "max_turns": max(turn_counts) if turn_counts else 0,
    }
    stats_path = f"{args.outdir}/threads_{args.brand}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"Wrote {stats_path} -> {stats}", file=sys.stderr)


if __name__ == "__main__":
    main()

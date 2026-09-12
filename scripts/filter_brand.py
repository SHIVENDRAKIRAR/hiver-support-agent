"""
Filter the Customer Support on Twitter dataset to threads involving a
target brand.

Usage:
    python scripts/filter_brand.py \
        --input data/twcs.csv \
        --brand AmazonHelp

Outputs:
    data/threads_<brand>.jsonl
    data/threads_<brand>_stats.json
"""

import argparse
import csv
import json
import os
import sys
from collections import deque


csv.field_size_limit(sys.maxsize)


def load_rows(path: str) -> dict:
    """Load tweets into a dictionary keyed by tweet ID."""
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return {
            row["tweet_id"]: row
            for row in reader
        }


def find_root(tweet_id: str, rows: dict) -> str:
    """Walk backwards to find the earliest available tweet in a thread."""
    seen = set()
    current_id = tweet_id

    while current_id not in seen and current_id in rows:
        seen.add(current_id)

        parent_id = rows[current_id].get("in_response_to_tweet_id", "")

        if not parent_id or parent_id not in rows:
            break

        current_id = parent_id

    return current_id


def collect_thread(root_id: str, rows: dict) -> list[dict]:
    """Collect all reachable tweets from a thread root."""
    thread = []
    queue = deque([root_id])
    seen = set()

    while queue:
        tweet_id = queue.popleft()

        if tweet_id in seen or tweet_id not in rows:
            continue

        seen.add(tweet_id)
        row = rows[tweet_id]
        thread.append(row)

        response_ids = row.get("response_tweet_id", "") or ""

        for response_id in response_ids.split(","):
            response_id = response_id.strip()

            if response_id and response_id not in seen:
                queue.append(response_id)

    return thread


def build_threads(rows: dict, brand: str) -> list[list[dict]]:
    """Reconstruct conversation threads involving the target brand."""
    brand_tweet_ids = [
        tweet_id
        for tweet_id, row in rows.items()
        if row["author_id"] == brand
    ]

    threads = []
    roots_seen = set()

    for tweet_id in brand_tweet_ids:
        root_id = find_root(tweet_id, rows)

        if root_id in roots_seen:
            continue

        roots_seen.add(root_id)

        thread = collect_thread(root_id, rows)

        if any(row["author_id"] == brand for row in thread):
            threads.append(
                sorted(thread, key=lambda row: row["tweet_id"])
            )

    return threads


def convert_to_thread_dict(thread_rows: list[dict]) -> dict:
    """Convert raw tweet records into the project's thread format."""
    turns = [
        {
            "tweet_id": row["tweet_id"],
            "author_id": row["author_id"],
            "is_brand": row["inbound"] == "False",
            "created_at": row["created_at"],
            "text": row["text"],
            "in_response_to_tweet_id": (
                row.get("in_response_to_tweet_id") or None
            ),
            "response_tweet_id": (
                row.get("response_tweet_id") or None
            ),
        }
        for row in thread_rows
    ]

    return {
        "thread_id": thread_rows[0]["tweet_id"],
        "num_turns": len(turns),
        "turns": turns,
    }


def write_threads(threads: list[list[dict]], output_path: str) -> None:
    """Write reconstructed threads to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as file:
        for thread in threads:
            record = convert_to_thread_dict(thread)
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def build_stats(threads: list[list[dict]], brand: str) -> dict:
    """Build basic statistics for the filtered thread set."""
    turn_counts = [len(thread) for thread in threads]

    return {
        "brand": brand,
        "num_threads": len(threads),
        "avg_turns_per_thread": (
            sum(turn_counts) / len(turn_counts)
            if turn_counts
            else 0
        ),
        "min_turns": min(turn_counts) if turn_counts else 0,
        "max_turns": max(turn_counts) if turn_counts else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter and reconstruct threads for a target brand."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to twcs.csv.",
    )
    parser.add_argument(
        "--brand",
        default="AmazonHelp",
        help="Brand author ID to filter.",
    )
    parser.add_argument(
        "--outdir",
        default="data",
        help="Output directory.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=0,
        help="Maximum number of threads for testing. 0 means no limit.",
    )
    args = parser.parse_args()

    print(f"Loading {args.input} ...", file=sys.stderr)
    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} tweets total", file=sys.stderr)

    print(
        f"Building threads for brand={args.brand} ...",
        file=sys.stderr,
    )
    threads = build_threads(rows, args.brand)

    print(
        f"Found {len(threads)} threads involving {args.brand}",
        file=sys.stderr,
    )

    if args.max_threads > 0:
        threads = threads[:args.max_threads]

    os.makedirs(args.outdir, exist_ok=True)

    threads_path = os.path.join(
        args.outdir,
        f"threads_{args.brand}.jsonl",
    )
    stats_path = os.path.join(
        args.outdir,
        f"threads_{args.brand}_stats.json",
    )

    write_threads(threads, threads_path)

    stats = build_stats(threads, args.brand)

    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    print(f"Wrote {threads_path}", file=sys.stderr)
    print(f"Wrote {stats_path} -> {stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
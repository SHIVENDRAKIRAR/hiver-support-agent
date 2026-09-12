"""
Load cleaned thread JSONL into Postgres (schema.sql must be applied first).

Usage:
    python scripts/load_to_db.py --input data/threads_AmazonHelp_clean.jsonl \
        --brand AmazonHelp --dsn "postgresql://user:pass@localhost:5432/hiver_agent"
"""

import argparse
import json

import psycopg2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--dsn", required=True)
    args = ap.parse_args()

    conn = psycopg2.connect(args.dsn)
    cur = conn.cursor()

    n_threads, n_turns = 0, 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cur.execute(
                """
                INSERT INTO threads (thread_id, brand, num_turns)
                VALUES (%s, %s, %s)
                ON CONFLICT (thread_id) DO NOTHING
                """,
                (obj["thread_id"], args.brand, obj["num_turns"]),
            )
            n_threads += 1
            for idx, turn in enumerate(obj["turns"]):
                cur.execute(
                    """
                    INSERT INTO turns (
                        tweet_id, thread_id, author_id, is_brand, turn_index,
                        created_at_raw, text_raw, text_clean,
                        in_response_to_tweet_id, response_tweet_id
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (tweet_id) DO NOTHING
                    """,
                    (
                        turn["tweet_id"],
                        obj["thread_id"],
                        turn["author_id"],
                        turn["is_brand"],
                        idx,
                        turn["created_at"],
                        turn["text"],
                        turn.get("text_clean"),
                        turn.get("in_response_to_tweet_id"),
                        turn.get("response_tweet_id"),
                    ),
                )
                n_turns += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Loaded {n_threads} threads, {n_turns} turns into DB")


if __name__ == "__main__":
    main()

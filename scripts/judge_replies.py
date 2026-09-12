"""
Score generated replies using an LLM judge.

Runs locally through Ollama, following the same pattern as the
classification and reply-generation pipelines.

Usage:
    python scripts/judge_replies.py \
        --input data/generated_replies.jsonl \
        --out data/judged_replies.jsonl \
        --model llama3.1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from judge_prompt import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
)


OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 60
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_RETRIES = 3


def judge_one(
    model: str,
    customer_message: str,
    reply: str,
    intent: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """Evaluate a generated reply using the Ollama judge."""
    prompt = (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"{build_judge_prompt(customer_message, reply, intent)}"
    )

    raw = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": DEFAULT_TEMPERATURE,
                    },
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            raw = response.json()["response"].strip()
            cleaned_raw = (
                raw.replace("```json", "")
                .replace("```", "")
                .strip()
            )

            try:
                parsed = json.loads(cleaned_raw)
            except json.JSONDecodeError:
                start = cleaned_raw.find("{")
                end = cleaned_raw.rfind("}")

                if start == -1 or end == -1:
                    raise

                parsed = json.loads(cleaned_raw[start : end + 1])

            return {
                "relevance": parsed.get("relevance"),
                "correctness": parsed.get("correctness"),
                "tone": parsed.get("tone"),
                "rationale": parsed.get("rationale"),
                "error": None,
            }

        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                return {
                    "relevance": None,
                    "correctness": None,
                    "tone": None,
                    "rationale": raw,
                    "error": "PARSE_ERROR",
                }

            time.sleep(0.5)

        except requests.exceptions.ConnectionError:
            print(
                "ERROR: Could not connect to Ollama. "
                "Make sure Ollama is running.",
                file=sys.stderr,
            )
            sys.exit(1)

        except Exception as error:
            if attempt == max_retries - 1:
                return {
                    "relevance": None,
                    "correctness": None,
                    "tone": None,
                    "rationale": str(error),
                    "error": "API_ERROR",
                }

            time.sleep(1)

    return {
        "relevance": None,
        "correctness": None,
        "tone": None,
        "rationale": None,
        "error": "UNKNOWN_ERROR",
    }


def load_rows(input_path: str) -> list[dict]:
    """Load generated replies from a JSONL file."""
    with open(input_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def load_completed_ids(output_path: Path) -> set:
    """Load tweet IDs that have already been judged."""
    if not output_path.exists():
        return set()

    completed_ids = set()

    with open(output_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                completed_ids.add(json.loads(line)["tweet_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    return completed_ids


def main():
    parser = argparse.ArgumentParser(
        description="Score generated replies using an LLM judge."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL output from generate_replies.py.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL file containing judge scores.",
    )
    parser.add_argument(
        "--model",
        default="llama3.1",
        help="Ollama model to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of replies to judge. 0 means no limit.",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)

    if args.limit > 0:
        rows = rows[:args.limit]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = load_completed_ids(output_path)

    if completed_ids:
        print(
            f"Resuming: {len(completed_ids)} already judged, skipping.",
            file=sys.stderr,
        )

    remaining = [
        row
        for row in rows
        if row.get("tweet_id") not in completed_ids
        and row.get("reply")
    ]

    print(
        f"{len(remaining)} remaining to judge.",
        file=sys.stderr,
    )

    judged = 0
    errors = 0

    with open(output_path, "a", encoding="utf-8") as output_file:
        for index, row in enumerate(remaining, start=1):
            scores = judge_one(
                args.model,
                row["text"],
                row["reply"],
                row.get("intent", ""),
            )

            output_row = {
                "thread_id": row.get("thread_id"),
                "tweet_id": row.get("tweet_id"),
                "text": row["text"],
                "intent": row.get("intent"),
                "reply": row["reply"],
                "generation_grounded": row.get("grounded"),
                **scores,
            }

            output_file.write(
                json.dumps(output_row, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            if scores["error"]:
                errors += 1
            else:
                judged += 1

            if index % 10 == 0 or index == len(remaining):
                print(
                    f"  {index}/{len(remaining)} done "
                    f"({judged} judged, {errors} errors)",
                    file=sys.stderr,
                )

    print(
        f"\nDone. {judged} judged, {errors} errors. "
        f"Output -> {output_path}"
    )


if __name__ == "__main__":
    main()
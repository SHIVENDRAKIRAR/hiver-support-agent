"""
Aggregate intent classification and reply quality evaluation results.

Usage:
    python scripts/eval_harness_summary.py \
        --golden data/golden_set_spotcheck.jsonl \
        --judged data/judged_replies.jsonl
"""

import argparse
import json
from collections import defaultdict


LOW_SCORE_THRESHOLD = 2
EXAMPLE_LIMIT = 5


def load_jsonl(path: str) -> list[dict]:
    """Load records from a JSONL file."""
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def summarize_classification(golden_path: str) -> dict:
    """Summarize agreement between human labels and LLM classifications."""
    rows = load_jsonl(golden_path)

    total = len(rows)
    agreed = sum(
        1 for row in rows if row["human_agreed_with_llm"]
    )

    return {
        "total": total,
        "agreement_rate": agreed / total if total else 0,
        "agreed": agreed,
        "disagreed": total - agreed,
    }


def summarize_judged_replies(judged_path: str) -> dict:
    """Summarize LLM-judged reply quality."""
    rows = load_jsonl(judged_path)

    scored = [
        row for row in rows
        if row.get("relevance") is not None
    ]

    if not scored:
        return {
            "total": len(rows),
            "scored": 0,
            "errors": len(rows),
        }

    avg_relevance = sum(row["relevance"] for row in scored) / len(scored)
    avg_correctness = sum(
        row["correctness"] for row in scored
    ) / len(scored)
    avg_tone = sum(row["tone"] for row in scored) / len(scored)

    low_correctness = [
        row for row in scored
        if row["correctness"] <= LOW_SCORE_THRESHOLD
    ]
    low_relevance = [
        row for row in scored
        if row["relevance"] <= LOW_SCORE_THRESHOLD
    ]

    by_intent = defaultdict(
        lambda: {
            "n": 0,
            "relevance": 0,
            "correctness": 0,
            "tone": 0,
        }
    )

    for row in scored:
        intent = row.get("intent", "unknown")
        stats = by_intent[intent]

        stats["n"] += 1
        stats["relevance"] += row["relevance"]
        stats["correctness"] += row["correctness"]
        stats["tone"] += row["tone"]

    per_intent = {
        intent: {
            "n": stats["n"],
            "avg_relevance": round(
                stats["relevance"] / stats["n"], 2
            ),
            "avg_correctness": round(
                stats["correctness"] / stats["n"], 2
            ),
            "avg_tone": round(
                stats["tone"] / stats["n"], 2
            ),
        }
        for intent, stats in by_intent.items()
    }

    return {
        "total": len(rows),
        "scored": len(scored),
        "errors": len(rows) - len(scored),
        "avg_relevance": round(avg_relevance, 2),
        "avg_correctness": round(avg_correctness, 2),
        "avg_tone": round(avg_tone, 2),
        "low_correctness_examples": [
            {
                "text": row["text"][:100],
                "reply": row["reply"][:100],
                "correctness": row["correctness"],
                "rationale": row.get("rationale"),
            }
            for row in low_correctness[:EXAMPLE_LIMIT]
        ],
        "low_relevance_examples": [
            {
                "text": row["text"][:100],
                "reply": row["reply"][:100],
                "relevance": row["relevance"],
                "rationale": row.get("rationale"),
            }
            for row in low_relevance[:EXAMPLE_LIMIT]
        ],
        "per_intent": per_intent,
    }


def print_report(classification: dict, reply_quality: dict) -> None:
    """Print the evaluation summary to the console."""
    print("=== EVALUATION HARNESS SUMMARY ===")

    print("\n--- Intent Classification (human-verified golden set) ---")
    print(f"  Total: {classification['total']}")
    print(
        f"  Agreement rate: "
        f"{100 * classification['agreement_rate']:.1f}% "
        f"({classification['agreed']}/{classification['total']})"
    )

    print("\n--- Reply Quality (LLM-as-judge) ---")
    print(
        f"  Total generated: {reply_quality['total']}, "
        f"scored: {reply_quality.get('scored', 0)}, "
        f"errors: {reply_quality.get('errors', 0)}"
    )

    if not reply_quality.get("scored"):
        return

    print(f"  Avg relevance:   {reply_quality['avg_relevance']}/5")
    print(f"  Avg correctness: {reply_quality['avg_correctness']}/5")
    print(f"  Avg tone:        {reply_quality['avg_tone']}/5")

    print("\n  Low-correctness examples (score <=2):")
    for example in reply_quality["low_correctness_examples"]:
        print(
            f"    [{example['correctness']}/5] "
            f"{example['text']}"
        )
        print(f"      -> reply: {example['reply']}")
        print(f"      -> why: {example['rationale']}")

    print("\n  Per-intent breakdown:")
    for intent, stats in sorted(
        reply_quality["per_intent"].items(),
        key=lambda item: -item[1]["n"],
    ):
        print(
            f"    {intent:30s} "
            f"n={stats['n']:3d}  "
            f"rel={stats['avg_relevance']}  "
            f"corr={stats['avg_correctness']}  "
            f"tone={stats['avg_tone']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize intent and reply quality evaluation results."
    )
    parser.add_argument("--golden", required=True)
    parser.add_argument("--judged", required=True)
    parser.add_argument(
        "--out",
        default="data/eval_harness_report.json",
    )
    args = parser.parse_args()

    classification_summary = summarize_classification(args.golden)
    reply_summary = summarize_judged_replies(args.judged)

    report = {
        "intent_classification": classification_summary,
        "reply_quality": reply_summary,
    }

    print_report(classification_summary, reply_summary)

    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nFull report also written to {args.out}")


if __name__ == "__main__":
    main()
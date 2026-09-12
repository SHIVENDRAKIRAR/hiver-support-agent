"""
Evaluation harness summary (Part 6): aggregates intent classification
accuracy (from the golden set spot-check) and reply quality scores (from
the LLM judge) into one report for the final write-up.

Usage:
    python scripts/eval_harness_summary.py \
        --golden data/golden_set_spotcheck.jsonl \
        --judged data/judged_replies.jsonl
"""

import argparse
import json
from collections import defaultdict


def summarize_classification(golden_path):
    rows = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    total = len(rows)
    agreed = sum(1 for r in rows if r["human_agreed_with_llm"])

    return {
        "total": total,
        "agreement_rate": agreed / total if total else 0,
        "agreed": agreed,
        "disagreed": total - agreed,
    }


def summarize_judged_replies(judged_path):
    rows = []
    with open(judged_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    scored = [r for r in rows if r.get("relevance") is not None]
    errors = len(rows) - len(scored)

    if not scored:
        return {"total": len(rows), "scored": 0, "errors": errors}

    avg_relevance = sum(r["relevance"] for r in scored) / len(scored)
    avg_correctness = sum(r["correctness"] for r in scored) / len(scored)
    avg_tone = sum(r["tone"] for r in scored) / len(scored)

    low_correctness = [r for r in scored if r["correctness"] <= 2]
    low_relevance = [r for r in scored if r["relevance"] <= 2]

    by_intent = defaultdict(lambda: {"n": 0, "relevance": 0, "correctness": 0, "tone": 0})
    for r in scored:
        b = by_intent[r.get("intent", "unknown")]
        b["n"] += 1
        b["relevance"] += r["relevance"]
        b["correctness"] += r["correctness"]
        b["tone"] += r["tone"]

    per_intent_avg = {
        intent: {
            "n": b["n"],
            "avg_relevance": round(b["relevance"] / b["n"], 2),
            "avg_correctness": round(b["correctness"] / b["n"], 2),
            "avg_tone": round(b["tone"] / b["n"], 2),
        }
        for intent, b in by_intent.items()
    }

    return {
        "total": len(rows),
        "scored": len(scored),
        "errors": errors,
        "avg_relevance": round(avg_relevance, 2),
        "avg_correctness": round(avg_correctness, 2),
        "avg_tone": round(avg_tone, 2),
        "low_correctness_examples": [
            {"text": r["text"][:100], "reply": r["reply"][:100], "correctness": r["correctness"], "rationale": r.get("rationale")}
            for r in low_correctness[:5]
        ],
        "low_relevance_examples": [
            {"text": r["text"][:100], "reply": r["reply"][:100], "relevance": r["relevance"], "rationale": r.get("rationale")}
            for r in low_relevance[:5]
        ],
        "per_intent": per_intent_avg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out", default="data/eval_harness_report.json")
    args = ap.parse_args()

    classification_summary = summarize_classification(args.golden)
    reply_summary = summarize_judged_replies(args.judged)

    report = {
        "intent_classification": classification_summary,
        "reply_quality": reply_summary,
    }

    print("=== EVALUATION HARNESS SUMMARY ===\n")
    print("--- Intent Classification (human-verified golden set) ---")
    print(f"  Total: {classification_summary['total']}")
    print(f"  Agreement rate: {100*classification_summary['agreement_rate']:.1f}% "
          f"({classification_summary['agreed']}/{classification_summary['total']})")

    print("\n--- Reply Quality (LLM-as-judge) ---")
    print(f"  Total generated: {reply_summary['total']}, scored: {reply_summary.get('scored', 0)}, errors: {reply_summary.get('errors', 0)}")
    if reply_summary.get("scored"):
        print(f"  Avg relevance:   {reply_summary['avg_relevance']}/5")
        print(f"  Avg correctness: {reply_summary['avg_correctness']}/5")
        print(f"  Avg tone:        {reply_summary['avg_tone']}/5")

        print(f"\n  Low-correctness examples (score <=2, possible hallucination):")
        for ex in reply_summary["low_correctness_examples"]:
            print(f"    [{ex['correctness']}/5] {ex['text']}")
            print(f"      -> reply: {ex['reply']}")
            print(f"      -> why: {ex['rationale']}")

        print(f"\n  Per-intent breakdown:")
        for intent, stats in sorted(reply_summary["per_intent"].items(), key=lambda x: -x[1]["n"]):
            print(f"    {intent:30s} n={stats['n']:3d}  rel={stats['avg_relevance']}  corr={stats['avg_correctness']}  tone={stats['avg_tone']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n(full report also written to {args.out})")


if __name__ == "__main__":
    main()

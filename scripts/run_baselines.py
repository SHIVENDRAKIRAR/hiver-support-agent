"""
Run both baselines against the golden set and compare accuracy to the LLM
classifier (Part 7). This produces the "how much does the LLM actually
earn over cheap alternatives" comparison the assignment asks for.

Usage:
    python scripts/run_baselines.py --golden data/golden_set_spotcheck.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from baselines import trivial_baseline, keyword_baseline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", default="data/baseline_comparison.json")
    args = ap.parse_args()

    rows = []
    with open(args.golden, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    total = len(rows)

    trivial_overall_correct = sum(1 for r in rows if trivial_baseline(r["text"], "overall") == r["gold_intent"])
    trivial_support_correct = sum(1 for r in rows if trivial_baseline(r["text"], "support") == r["gold_intent"])
    keyword_correct = sum(1 for r in rows if keyword_baseline(r["text"]) == r["gold_intent"])
    llm_correct = sum(1 for r in rows if r["llm_suggested_intent"] == r["gold_intent"])

    results = {
        "total": total,
        "trivial_baseline_overall_accuracy": round(trivial_overall_correct / total, 3),
        "trivial_baseline_support_accuracy": round(trivial_support_correct / total, 3),
        "keyword_baseline_accuracy": round(keyword_correct / total, 3),
        "llm_classifier_accuracy": round(llm_correct / total, 3),
    }

    print("=== BASELINE COMPARISON (against human-verified golden set) ===\n")
    print(f"Total examples: {total}\n")
    print(f"Trivial baseline (always '{trivial_baseline('', 'overall')}'):        {results['trivial_baseline_overall_accuracy']*100:5.1f}%")
    print(f"Trivial baseline (always '{trivial_baseline('', 'support')}'):          {results['trivial_baseline_support_accuracy']*100:5.1f}%")
    print(f"Keyword-rule baseline:                          {results['keyword_baseline_accuracy']*100:5.1f}%")
    print(f"LLM classifier (Llama 3.1, locked taxonomy):    {results['llm_classifier_accuracy']*100:5.1f}%")

    print("\n--- Examples where LLM succeeded but keyword baseline failed ---")
    shown = 0
    for r in rows:
        kw_pred = keyword_baseline(r["text"])
        llm_pred = r["llm_suggested_intent"]
        gold = r["gold_intent"]
        if llm_pred == gold and kw_pred != gold and shown < 5:
            print(f"  text: {r['text'][:90]}")
            print(f"    gold={gold}  llm={llm_pred} (correct)  keyword={kw_pred} (wrong)")
            shown += 1
    if shown == 0:
        print("  (none found -- keyword baseline matched LLM on all correct LLM predictions)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n(also written to {args.out})")


if __name__ == "__main__":
    main()

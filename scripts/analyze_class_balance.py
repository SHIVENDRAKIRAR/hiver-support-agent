"""
Analyze class balance from the LLM-classified sample, flag any errors,
and print a report to decide whether subscription_or_membership_issue
(or other classes) need merging/splitting.

Usage:
    python scripts/analyze_class_balance.py --input data/classified_sample.jsonl
"""

import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/class_balance_report.txt")
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    counts = Counter(r["intent"] for r in rows)
    total = len(rows)
    errors = sum(
        v for k, v in counts.items()
        if k in ("INVALID_LABEL", "PARSE_ERROR", "API_ERROR", "UNKNOWN_ERROR")
    )

    lines = ["--- CLASS BALANCE REPORT ---", f"total classified: {total}", ""]
    for label, count in counts.most_common():
        pct = 100 * count / total
        lines.append(f"{label:35s} {count:5d}  ({pct:5.1f}%)")

    lines.append("")
    lines.append(f"errors/invalid: {errors} ({100*errors/total:.1f}%)")

    # low-confidence examples worth a manual look
    low_conf = [
        r for r in rows
        if r.get("confidence") is not None and r["confidence"] < 0.6
    ]
    lines.append(f"\nlow-confidence (<0.6) classifications: {len(low_conf)}")
    if low_conf:
        lines.append("sample of low-confidence cases:")
        for r in low_conf[:10]:
            lines.append(f"  [{r['intent']} conf={r['confidence']}] {r['text'][:100]}")

    text = "\n".join(lines)
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n(also written to {args.out})")


if __name__ == "__main__":
    main()

"""
Interactive labeling tool for the golden evaluation set (Part 3).

Shows one candidate message at a time with the LLM's proposed label as a
default suggestion. You confirm, correct, or reject. Designed to be fast:
press Enter to accept the suggested label, or type a number to pick a
different one. Ctrl+C saves progress and exits cleanly -- rerunning resumes.

This produces the HUMAN-LABELED golden set -- the LLM's label is a
starting suggestion only, never trusted as ground truth. Every row in the
output has been explicitly confirmed or corrected by a human.

Usage:
    python scripts/label_golden_set.py \
        --input data/golden_set_candidates.jsonl \
        --out data/golden_set.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from taxonomy import INTENTS, NON_SUPPORT_LABELS, ESCALATION_DECISIONS, ESCALATION_REASONS  # noqa: E402

ALL_LABELS = INTENTS + NON_SUPPORT_LABELS


def prompt_intent(suggested: str) -> str:
    print("\nLabels:")
    for i, label in enumerate(ALL_LABELS, 1):
        marker = " <-- suggested" if label == suggested else ""
        print(f"  {i:2d}. {label}{marker}")

    while True:
        raw = input(f"\nIntent [Enter = accept '{suggested}', or number, or 's' to skip]: ").strip()
        if raw == "":
            return suggested
        if raw.lower() == "s":
            return "SKIPPED"
        if raw.isdigit() and 1 <= int(raw) <= len(ALL_LABELS):
            return ALL_LABELS[int(raw) - 1]
        print("  Invalid input, try again.")


def prompt_escalation() -> tuple:
    while True:
        raw = input("Escalate? [n=auto_handle (default) / y=human_review]: ").strip().lower()
        if raw in ("", "n"):
            return "auto_handle", []
        if raw == "y":
            break
        print("  Invalid input, try again.")

    print("Reasons (comma-separated numbers):")
    for i, r in enumerate(ESCALATION_REASONS, 1):
        print(f"  {i}. {r}")
    raw = input("Reason(s): ").strip()
    reasons = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(ESCALATION_REASONS):
            reasons.append(ESCALATION_REASONS[int(part) - 1])
    return "human_review", reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-escalation", action="store_true",
                     help="also label escalation decision+reason per example (slower; can be done in a separate pass)")
    args = ap.parse_args()

    candidates = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))

    out_path = Path(args.out)
    already_done_ids = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    already_done_ids.add(json.loads(line)["tweet_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    remaining = [c for c in candidates if c["tweet_id"] not in already_done_ids]
    print(f"{len(already_done_ids)} already labeled, {len(remaining)} remaining.\n")

    fout = open(out_path, "a", encoding="utf-8")
    n_this_session = 0

    try:
        for idx, row in enumerate(remaining, 1):
            print("\n" + "=" * 70)
            print(f"[{idx}/{len(remaining)}] thread_id={row.get('thread_id')} tweet_id={row.get('tweet_id')}")
            print("-" * 70)
            print(row["text"])
            print("-" * 70)

            suggested = row.get("intent", "other_support_issue")
            gold_intent = prompt_intent(suggested)

            if gold_intent == "SKIPPED":
                continue

            gold_escalate, gold_reasons = (None, [])
            if args.with_escalation:
                decision, reasons = prompt_escalation()
                gold_escalate = decision == "human_review"
                gold_reasons = reasons

            out_row = {
                "thread_id": row.get("thread_id"),
                "tweet_id": row.get("tweet_id"),
                "text": row["text"],
                "llm_suggested_intent": suggested,
                "gold_intent": gold_intent,
                "human_agreed_with_llm": gold_intent == suggested,
                "gold_escalate": gold_escalate,
                "gold_escalate_reasons": gold_reasons,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()
            n_this_session += 1

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. {n_this_session} labeled this session, saved to {out_path}.")
        print("Rerun the same command to resume.")
    finally:
        fout.close()

    if n_this_session and not sys.exc_info()[0]:
        print(f"\nDone this session: {n_this_session} labeled. Output -> {out_path}")


if __name__ == "__main__":
    main()

import json

with open("data/judged_replies.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

for r in rows[:8]:
    print(r["text"][:80])
    print("  reply:", r["reply"][:80])
    print("  scores:", r["relevance"], r["correctness"], r["tone"])
    print("  rationale:", r["rationale"])
    print()

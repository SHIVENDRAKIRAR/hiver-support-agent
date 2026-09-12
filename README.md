# AmazonHelp AI Support Agent

Take-home assignment (Hiver SDE Intern) — AI customer support agent built on
real AmazonHelp Twitter support conversations from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset.

The agent: (1) classifies incoming customer messages into intents, (2) drafts
a reply grounded in how AmazonHelp has historically resolved similar issues,
(3) decides whether to auto-handle or escalate to a human, with a stated reason.

**Full report:** [`reports/report.md`](reports/report.md)
**Decision log:** [`reports/decision_log.md`](reports/decision_log.md)

## Headline results

| Metric | Value |
|---|---|
| Intent classification accuracy (LLM, vs. 44-example human-verified golden set) | 75.0% |
| ...vs. keyword-rule baseline | 20.5% |
| ...vs. trivial baseline | 20.5% |
| Reply relevance (LLM-judge, 1-5) | 4.84 |
| Reply correctness / no fabrication (LLM-judge, 1-5) | 5.00 |
| Reply tone (LLM-judge, 1-5) | 4.95 |

See the report for why the reply-quality numbers need a caveat before being
taken at face value.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**Also required: [Ollama](https://ollama.com/download)** (all classification,
generation, and judging runs locally, at zero cost):
```bash
ollama pull llama3.1
```

### 1. Get the data

Download `twcs.csv` from
[Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter),
place at `data/twcs.csv`.

### 2. Build the pipeline

```bash
python scripts/filter_brand.py --input data/twcs.csv --brand AmazonHelp
python scripts/clean_threads.py --input data/threads_AmazonHelp.jsonl --output data/threads_AmazonHelp_clean.jsonl
```

### 3. Classify a sample

```bash
python scripts/sample_for_classification.py --input data/threads_AmazonHelp_clean.jsonl --n 1000
python scripts/classify_with_ollama.py --input data/sample_for_classification.jsonl --out data/classified_sample.jsonl
```

Full 1000-row run takes ~1 hour locally; supports resume if interrupted.

### 4. Build the golden set and spot-check

```bash
python scripts/build_golden_set_sample.py --input data/classified_sample.jsonl --out data/golden_set_candidates.jsonl --total 200 --min-per-class 12
python scripts/spot_check_golden_set.py --input data/golden_set_candidates.jsonl --out data/golden_set_spotcheck.jsonl --n 50
```

### 5. Build the retrieval index and generate replies

```bash
python -m src.retrieval build --input data/threads_AmazonHelp_clean.jsonl --index data/thread_index.pkl
python scripts/generate_replies.py --input data/golden_set_spotcheck.jsonl --index data/thread_index.pkl --out data/generated_replies.jsonl
```

### 6. Run the evaluation harness

```bash
python scripts/judge_replies.py --input data/generated_replies.jsonl --out data/judged_replies.jsonl
python scripts/eval_harness_summary.py --golden data/golden_set_spotcheck.jsonl --judged data/judged_replies.jsonl
python scripts/run_baselines.py --golden data/golden_set_spotcheck.jsonl
```

## Project structure

```
scripts/    data loading, cleaning, classification, generation, evaluation
src/        core logic: taxonomy, prompts, escalation rules, retrieval, baselines
reports/    final report, decision log
data/       raw + processed data (gitignored)
```

## Status

- [x] Part 1 — Data acquisition, thread reconstruction, cleaning, DB schema
- [x] Part 2 — Intent taxonomy (8 intents, locked from measured class volumes) + classifier
- [x] Part 3 — Golden evaluation set (200 LLM-classified, 44 human-verified spot-check)
- [x] Part 4 — Grounded reply generation (RAG, anti-hallucination, leakage-free)
- [x] Part 5 — Escalation decision logic (rules + LLM judgment, decoupled from intent)
- [x] Part 6 — Evaluation harness (classification accuracy + LLM-as-judge reply scoring)
- [x] Part 7 — Baselines (trivial + keyword-rule) + top-5 failure analysis
- [x] Part 8 — Report + decision log + this README
"# hiver-support-agent" 

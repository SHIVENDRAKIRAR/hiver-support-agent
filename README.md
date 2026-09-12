# AmazonHelp AI Support Agent

An AI customer support agent built on real AmazonHelp Twitter conversations
from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Given an incoming customer message, the agent:

1. **Classifies** it into one of 8 support intents (or flags it as non-English
   or non-support content)
2. **Drafts a reply**, grounded in retrieval over how AmazonHelp has
   historically resolved similar issues
3. **Decides whether to escalate** to a human, with a stated reason — as a
   judgment separate from intent

Full report: [`reports/report.md`](reports/report.md)
Decision log: [`reports/decision_log.md`](reports/decision_log.md)

## Results

| Metric | Value |
|---|---|
| Intent classification accuracy (vs. human-verified golden set, n=44) | 75.0% |
| ...vs. keyword-rule baseline | 20.5% |
| ...vs. trivial baseline | 20.5% |
| Reply relevance (LLM-judge, 1–5) | 4.84 |
| Reply correctness / no fabrication (LLM-judge, 1–5) | 5.00 |
| Reply tone (LLM-judge, 1–5) | 4.95 |

See the report for methodology, caveats on the reply-quality scores, and a
full failure analysis.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Requires [Ollama](https://ollama.com/download) — all classification, generation,
and judging run locally at zero cost:

```bash
ollama pull llama3.1
```

## Reproducing the results

### 1. Get the data

Download `twcs.csv` from
[Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
and place it at `data/twcs.csv`.

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

The full 1000-row run takes roughly an hour locally and supports resuming if interrupted.

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
data/       raw and processed data (gitignored)
```

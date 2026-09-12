# AmazonHelp AI Support Agent

An AI customer support agent built using real AmazonHelp conversations from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset.

Given an incoming customer message, the agent:

1. **Classifies the intent** into one of eight support categories or identifies non-English and non-support content.
2. **Generates a response** using retrieval from historically resolved AmazonHelp conversations to provide contextually grounded replies.
3. **Determines escalation** by deciding whether the issue should be handled automatically or reviewed by a human, along with an explicit escalation reason.

## Documentation

- [Full Report](reports/report.md)
- [Decision Log](reports/decision_log.md)

## Results

| Metric | Result |
|---|---:|
| Intent classification accuracy | **75.0%** |
| Keyword-rule baseline | 20.5% |
| Trivial baseline | 20.5% |
| Reply relevance | **4.84 / 5** |
| Reply correctness | **5.00 / 5** |
| Reply tone | **4.95 / 5** |

Classification accuracy was measured against a human-verified evaluation set of 44 examples. Reply quality was evaluated using an LLM-based judge.

See the [full report](reports/report.md) for methodology, evaluation limitations, and failure analysis.

## Setup

### 1. Create the environment

```bash
python -m venv .venv
```

Activate the environment:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment configuration:

```bash
cp .env.example .env
```

### 2. Install Ollama

The project uses Ollama to run Llama 3.1 locally for classification, response generation, and evaluation.

After installing Ollama:

```bash
ollama pull llama3.1
```

## Reproducing the Results

### 1. Prepare the Dataset

Download `twcs.csv` from the Customer Support on Twitter dataset and place it at:

```
data/twcs.csv
```

### 2. Filter and Clean AmazonHelp Threads

```bash
python scripts/filter_brand.py --input data/twcs.csv --brand AmazonHelp

python scripts/clean_threads.py \
    --input data/threads_AmazonHelp.jsonl \
    --output data/threads_AmazonHelp_clean.jsonl
```

### 3. Classify a Sample

Generate a 1,000-message sample:

```bash
python scripts/sample_for_classification.py \
    --input data/threads_AmazonHelp_clean.jsonl \
    --n 1000
```

Run classification with Ollama:

```bash
python scripts/classify_with_ollama.py \
    --input data/sample_for_classification.jsonl \
    --out data/classified_sample.jsonl
```

The 1,000-message classification run takes approximately one hour locally and supports resuming after interruption.

### 4. Build the Evaluation Set

Generate stratified evaluation candidates:

```bash
python scripts/build_golden_set_sample.py \
    --input data/classified_sample.jsonl \
    --out data/golden_set_candidates.jsonl \
    --total 200 \
    --min-per-class 12
```

Run the human spot-check:

```bash
python scripts/spot_check_golden_set.py \
    --input data/golden_set_candidates.jsonl \
    --out data/golden_set_spotcheck.jsonl \
    --n 50
```

### 5. Build the Retrieval Index and Generate Replies

Build the retrieval index:

```bash
python -m src.retrieval build \
    --input data/threads_AmazonHelp_clean.jsonl \
    --index data/thread_index.pkl
```

Generate responses:

```bash
python scripts/generate_replies.py \
    --input data/golden_set_spotcheck.jsonl \
    --index data/thread_index.pkl \
    --out data/generated_replies.jsonl
```

### 6. Run Evaluation

Evaluate generated replies:

```bash
python scripts/judge_replies.py \
    --input data/generated_replies.jsonl \
    --out data/judged_replies.jsonl
```

Generate the evaluation summary:

```bash
python scripts/eval_harness_summary.py \
    --golden data/golden_set_spotcheck.jsonl \
    --judged data/judged_replies.jsonl
```

Run the classification baselines:

```bash
python scripts/run_baselines.py \
    --golden data/golden_set_spotcheck.jsonl
```

## Project Structure

```
amazonhelp-ai-support-agent/
│
├── scripts/
│   ├── Data loading and preprocessing
│   ├── Classification
│   ├── Response generation
│   └── Evaluation
│
├── src/
│   ├── Taxonomy
│   ├── Prompts
│   ├── Escalation rules
│   ├── Retrieval
│   └── Baselines
│
├── reports/
│   ├── report.md
│   └── decision_log.md
│
├── data/
│   └── Raw and processed data (gitignored)
│
├── requirements.txt
├── .env.example
└── README.md
```

## Key Design Principles

- Intent and escalation are independent decisions.
- Responses are grounded in historical support conversations.
- The retrieval system excludes the query's own thread to prevent evaluation leakage.
- Local inference is used to avoid external API dependencies and associated costs.
- Non-English support requests are distinguished from non-support noise.

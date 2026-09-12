-- Hiver AI Support Agent — schema for AmazonHelp pipeline
-- Run: psql -d hiver_agent -f scripts/schema.sql

CREATE TABLE IF NOT EXISTS threads (
    thread_id       TEXT PRIMARY KEY,
    brand           TEXT NOT NULL,
    num_turns       INT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS turns (
    tweet_id                TEXT PRIMARY KEY,
    thread_id                TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    author_id                TEXT NOT NULL,
    is_brand                  BOOLEAN NOT NULL,
    turn_index                INT NOT NULL,
    created_at_raw             TEXT,
    text_raw                   TEXT NOT NULL,
    text_clean                 TEXT,
    in_response_to_tweet_id    TEXT,
    response_tweet_id          TEXT
);

CREATE INDEX IF NOT EXISTS idx_turns_thread_id ON turns(thread_id);
CREATE INDEX IF NOT EXISTS idx_turns_is_brand ON turns(is_brand);

-- Intent labels (Part 2)
CREATE TABLE IF NOT EXISTS intent_labels (
    tweet_id        TEXT PRIMARY KEY REFERENCES turns(tweet_id) ON DELETE CASCADE,
    intent          TEXT NOT NULL,
    labeled_by      TEXT NOT NULL DEFAULT 'llm',   -- 'llm' | 'human' | 'gold'
    confidence      REAL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Golden eval set (Part 3)
CREATE TABLE IF NOT EXISTS golden_examples (
    id              SERIAL PRIMARY KEY,
    tweet_id        TEXT NOT NULL REFERENCES turns(tweet_id) ON DELETE CASCADE,
    thread_id       TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    gold_intent     TEXT NOT NULL,
    gold_reply      TEXT,
    gold_escalate   BOOLEAN,
    sampling_note   TEXT,
    labeled_at      TIMESTAMPTZ DEFAULT now()
);

-- Generated agent outputs, for eval harness (Part 6)
CREATE TABLE IF NOT EXISTS agent_runs (
    id              SERIAL PRIMARY KEY,
    tweet_id        TEXT NOT NULL REFERENCES turns(tweet_id) ON DELETE CASCADE,
    model_name      TEXT NOT NULL,
    predicted_intent TEXT,
    drafted_reply   TEXT,
    escalate        BOOLEAN,
    escalate_reason TEXT,
    run_at          TIMESTAMPTZ DEFAULT now()
);

-- LLM-as-judge scores
CREATE TABLE IF NOT EXISTS judge_scores (
    id              SERIAL PRIMARY KEY,
    agent_run_id    INT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    judge_model     TEXT NOT NULL,
    relevance       REAL,
    correctness     REAL,
    tone            REAL,
    overall         REAL,
    rationale       TEXT,
    scored_at       TIMESTAMPTZ DEFAULT now()
);

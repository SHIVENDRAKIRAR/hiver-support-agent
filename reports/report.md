# AmazonHelp AI Support Agent — Report

## 1. Problem framing

AmazonHelp receives a high volume of customer support requests on Twitter,
spanning delivery issues, refunds, damaged items, billing disputes, account
problems, and general product questions — alongside a substantial amount of
non-support content (thanks, praise, non-English messages, off-topic chatter).
This project builds an AI agent that:

1. **Classifies** incoming customer messages into one of 8 support intents
   (plus two non-support labels), grounded in intents observed directly in
   real AmazonHelp data rather than assumed upfront.
2. **Drafts a reply**, grounded in retrieval over historically similar
   resolved threads, to avoid inventing policy commitments the brand hasn't
   actually made.
3. **Decides whether to escalate** to a human, as a decision separate from
   intent — an angry, repeated delivery complaint is still
   intent=delivery_delay; escalation is a distinct judgment call layered on
   top.

Data source: the Kaggle "Customer Support on Twitter" dataset, filtered to
AmazonHelp conversation threads (82,541 threads, 2.8M+ tweets in the source
corpus).

## 2. Taxonomy

Locked at **8 intents** after measuring real class volumes on a 1000-message
sample (see Decision Log): `delivery_delay`, `delivery_not_received`,
`refund_or_return`, `wrong_or_damaged_item`, `billing_or_charge_issue`,
`account_or_technical_issue`, `product_or_service_inquiry`,
`other_support_issue` — plus two **non-support** labels kept deliberately
separate from "noise": `language_out_of_scope` (a real support need, just not
in English — never conflated with actual noise) and `non_support_noise`
(thanks/praise/spam with no actionable ask).

Escalation is modeled as an independent decision (`auto_handle` /
`human_review`) with explicit reasons (`repeated_unresolved_issue`,
`financial_risk`, `missing_information`, `policy_exception`,
`abusive_or_sensitive`, `low_confidence`) — never folded into the intent
label itself.

## 3. Results

All models used are run **locally via Ollama (Llama 3.1)** — zero API cost,
no rate limits (Gemini's free tier proved too restrictive at 20 requests/day;
see Decision Log). Ground truth is a **44-example human-verified spot-check**,
drawn via stratified sampling from a 200-example LLM-classified golden set.

### 3.1 Intent classification

| Method | Accuracy (n=44) |
|---|---|
| Trivial baseline (always predict most common label overall) | 20.5% |
| Trivial baseline (always predict most common *support* intent) | 11.4% |
| Naive keyword-rule baseline | 20.5% |
| **LLM classifier (Llama 3.1)** | **75.0%** |

The keyword baseline ties the trivial baseline exactly — naive rules add
essentially no value on this messy, real-world data. The LLM's advantage
consistently comes from generalizing paraphrased language the keyword rules
can't catch (e.g. "Amazon reimbursed me" → `refund_or_return`, with no literal
"refund" keyword present).

### 3.2 Reply generation quality (LLM-as-judge, 1-5 scale)

| Dimension | Avg score (n=44) |
|---|---|
| Relevance | 4.84 |
| Correctness (no fabricated specifics) | 5.00 |
| Tone | 4.95 |

See §5 for why this headline number needs a caveat before it's trusted at
face value.

## 4. Baselines

Two baselines were built specifically to measure what the LLM buys over cheap
alternatives (§3.1): a trivial majority-class predictor and a naive
keyword/regex classifier covering obvious per-intent trigger words. Both are
intentionally simple — the point was measuring a real floor, not building a
second serious classifier.

## 5. What's misleading about the headline number

The reply-quality scores (4.84–5.00/5) look almost too good, and that
deserves direct scrutiny rather than a victory lap:

- **43 of 44 generated replies followed one easy, safe pattern**: acknowledge
  the issue + ask the customer to DM for specifics. This pattern is easy to
  execute well once retrieval grounding is in place, so a near-perfect score
  mostly reflects consistency on an easy sub-task, not robustness across
  harder cases (e.g. a customer who needs a real policy answer, not a
  deferral).
- **The same model judged its own output.** Llama 3.1 both generated and
  scored the replies. The judge's rationales were manually reviewed and
  confirmed to discriminate correctly — it caught the one genuine miss in the
  sample (a customer expressing purchase excitement, not asking for help)
  with an accurate, specific rationale — but an independent judge model would
  be needed to fully trust this at scale.
- **The 75.0% classification accuracy is measured on only 44 examples**,
  short of the 150–250 golden-set size this assignment specifies. This is a
  deliberate tradeoff, not an oversight: an earlier 200-example labeling pass
  was rushed under time pressure (suggestions accepted without reading them)
  and produced a meaningless 99.5% "agreement" number. Rather than submit
  that, a smaller, carefully-attended 44-example spot-check was done instead.
  44 examples produce a real, trustworthy number with wider confidence
  intervals than the specified range would give — but the honest accounting
  is that this is a real gap against the stated requirement, to be closed
  with additional labeling time rather than treated as resolved.

## 6. Top 5 failure modes

1. **`delivery_not_received` vs. `delivery_delay`/`other_support_issue`
   boundary confusion** — the classifier's weakest specific area (50%
   agreement in the spot-check vs. 75–100% elsewhere). These intents
   genuinely overlap in real customer language.
2. **`language_out_of_scope` vs. `non_support_noise` boundary** — even after
   hardening the prompt with an explicit "check language first" rule, brief
   non-English messages remain ambiguous without deeper translation.
3. **Thread-reconstruction noise: 11.6% of "opening messages" are actually
   mid-conversation fragments**, not true thread starts (measured directly
   across all 82,541 threads). A pipeline limitation, not a classifier
   failure, but it degrades every downstream stage.
4. **Same-model-judges-itself in reply evaluation** (see §5).
5. **Retrieval self-match data leakage** — before a fix, the RAG retriever's
   top match was often the query's own historical thread (similarity=1.0),
   meaning "grounded" generation would have trivially looked up its own
   answer. Caught and fixed via an explicit exclusion parameter; flagged here
   as a general methodology risk for any RAG system built from data
   overlapping its eval set.

## 7. Next steps

- **Top priority: expand the human-verified golden set from 44 to the
  150–250 examples this assignment specifies.** The 44-example check was a
  deliberate honesty tradeoff under time pressure, not a substitute for the
  full requirement — closing this gap would meaningfully tighten confidence
  in the 75.0% classification accuracy and is the clearest unfinished piece
  of this project.
- Add an independent judge model (a different model family than the
  generator) to validate the reply-quality scores without the
  same-model-judges-itself risk.
- Fix thread reconstruction to require a true root tweet (no
  `in_response_to_tweet_id`) rather than walking back to the earliest
  available tweet in a truncated thread, reducing the 11.6%
  fragment-as-opener rate.
- Add a finer-grained sub-taxonomy for delivery-related intents, informed by
  a larger labeled sample, to resolve the `delivery_not_received` /
  `delivery_delay` boundary confusion.
- Test cross-model classification (Gemini, GPT) as ensemble/cross-validation
  against the Llama 3.1 baseline, once free-tier or budget constraints allow.

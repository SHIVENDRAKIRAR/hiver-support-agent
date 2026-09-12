# AmazonHelp AI Support Agent — Report

## 1. Problem Framing

AmazonHelp receives a high volume of customer support requests on Twitter, covering delivery issues, refunds, damaged items, billing disputes, account problems, and product-related questions. The dataset also contains non-support content such as acknowledgements, praise, spam, and off-topic messages.

This project develops an AI support agent with three primary capabilities:

1. **Intent Classification** — Classifies incoming customer messages into eight support intents and two non-support categories derived from patterns observed in the dataset.
2. **Reply Generation** — Generates response drafts using retrieval over historically similar resolved conversations to improve relevance and reduce unsupported claims.
3. **Escalation Detection** — Independently determines whether a conversation should be handled automatically or reviewed by a human, based on factors such as unresolved issues, financial risk, policy exceptions, and insufficient information.

The system uses the Kaggle **Customer Support on Twitter** dataset, filtered to AmazonHelp conversations, comprising approximately **82,541 threads and 2.8 million tweets** in the source corpus.

## 2. Taxonomy

The final taxonomy consists of **eight support intents**:

* `delivery_delay`
* `delivery_not_received`
* `refund_or_return`
* `wrong_or_damaged_item`
* `billing_or_charge_issue`
* `account_or_technical_issue`
* `product_or_service_inquiry`
* `other_support_issue`

Two additional categories are used for non-support cases:

* `language_out_of_scope` — genuine support requests expressed in languages outside the supported scope.
* `non_support_noise` — messages such as acknowledgements, praise, spam, or other content without an actionable support request.

Escalation is modeled independently from intent classification. It produces either `auto_handle` or `human_review`, together with one or more applicable reasons:

* `repeated_unresolved_issue`
* `financial_risk`
* `missing_information`
* `policy_exception`
* `abusive_or_sensitive`
* `low_confidence`

This separation prevents escalation characteristics from being incorrectly incorporated into the underlying customer intent.

## 3. Results

All models were executed locally using **Ollama with Llama 3.1**, eliminating API costs and external rate limits.

Evaluation was performed using a **44-example human-verified stratified spot-check** drawn from the classified evaluation set.

### 3.1 Intent Classification

| Method                                        | Accuracy (n=44) |
| --------------------------------------------- | --------------: |
| Trivial baseline — most common label overall  |           20.5% |
| Trivial baseline — most common support intent |           11.4% |
| Naive keyword/regex baseline                  |           20.5% |
| **LLM classifier — Llama 3.1**                |       **75.0%** |

The Llama 3.1 classifier substantially outperformed the rule-based baselines. The primary advantage comes from its ability to recognize paraphrased expressions and infer intent from context rather than relying on exact keywords.

### 3.2 Reply Generation Quality

Reply quality was evaluated using an LLM-based judge on a 1–5 scale.

| Dimension   | Average Score (n=44) |
| ----------- | -------------------: |
| Relevance   |                 4.84 |
| Correctness |                 5.00 |
| Tone        |                 4.95 |

The results indicate that the generated responses were generally relevant, appropriately written, and conservative in their use of specific claims.

## 4. Baselines

Two simple baselines were implemented to establish a reference point for intent classification:

* **Majority-class baseline** — always predicts the most frequent class.
* **Keyword/regex baseline** — assigns intents using predefined keywords and patterns associated with each category.

These baselines provide a simple performance floor against which the LLM-based classifier can be evaluated.

## 5. Evaluation Considerations

The reply-generation scores should be interpreted in the context of the evaluation methodology.

* **Limited evaluation set:** Classification accuracy was measured on 44 human-verified examples. A larger evaluation set of 150–250 examples would provide stronger statistical confidence.
* **Same-model evaluation:** Llama 3.1 was used both for reply generation and reply evaluation. Although the evaluation rationales were manually reviewed, an independent model would provide a stronger assessment.
* **Response pattern concentration:** Most generated replies followed a conservative support pattern involving acknowledgement of the issue and a request for additional information through direct messaging. This contributes to the high consistency of the scores but does not fully establish performance on complex policy-specific cases.

## 6. Key Failure Modes

### 1. Delivery Intent Boundary

The distinction between `delivery_not_received`, `delivery_delay`, and `other_support_issue` is the most challenging classification boundary. Customer language frequently overlaps across these categories.

### 2. Language and Noise Classification

Short non-English messages can be difficult to distinguish between genuine support requests and non-support content without deeper language understanding.

### 3. Thread Reconstruction

Approximately **11.6% of extracted opening messages** were identified as mid-conversation fragments rather than true thread openings. This introduces upstream noise that can affect classification, retrieval, and response generation.

### 4. Same-Model Evaluation

Using the same model for both generation and evaluation introduces a potential evaluation bias. An independent judge model would provide a more robust assessment.

### 5. Retrieval Self-Match

The retrieval system initially allowed evaluation messages to retrieve their own historical threads, resulting in similarity scores of 1.0 and creating a potential data-leakage path.

This was addressed by introducing an explicit `exclude_thread_id` parameter to prevent the query's own thread from being retrieved during generation.

## 7. Next Steps

1. **Expand the human-verified evaluation set** to 150–250 examples to improve confidence in reported classification performance.
2. **Introduce an independent evaluation model** from a different model family to assess reply quality.
3. **Improve thread reconstruction** by identifying true root tweets and reducing the number of mid-conversation fragments treated as opening messages.
4. **Refine delivery-related taxonomy** using a larger labeled dataset to better distinguish delivery delays from non-receipt cases.
5. **Evaluate additional models** such as Gemini and GPT to compare classification performance and investigate potential ensemble approaches.

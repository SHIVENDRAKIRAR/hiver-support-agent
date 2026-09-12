# Decision Log

This section documents key non-obvious design and implementation decisions made during the project, along with the rationale behind them.

### 1. Target Brand: AmazonHelp

The dataset contains real brand support accounts, including AmazonHelp and AppleSupport, rather than an account named “Hiver.” AmazonHelp was selected as the target brand due to its relatively high volume and recognizability.

### 2. Intent Classification and Escalation as Separate Decisions

Intent classification was kept separate from escalation detection. An escalation is a property of the conversation rather than a distinct customer intent. For example, a repeated complaint about a delayed package remains a delivery-related issue even when escalation is warranted.

Accordingly, escalation is modeled as an independent decision layered on top of the classified intent, with separate reason codes.

### 3. Signal-Based Filtering

Messages were not filtered solely based on length. A fixed minimum-length rule could incorrectly discard short but meaningful requests such as “Refund?” or “Wrong item.”

Instead, filtering is based on whether the message contains sufficient meaningful signal for classification.

### 4. Separate Handling of Out-of-Scope Languages and Support Noise

`language_out_of_scope` was kept separate from `non_support_noise`. Non-English messages can still represent legitimate and potentially urgent support requests.

Treating all non-English content as noise could therefore result in genuine customer issues being incorrectly discarded.

### 5. Membership Issues Consolidated into Other Support Issues

`subscription_or_membership_issue` was consolidated into `other_support_issue` because it represented only a small proportion of the classified data. Maintaining it as a separate category would have resulted in too few representative examples for a reliable classification class.

### 6. Local LLM Selected for Classification

The classification backend was ultimately moved to a local Ollama deployment using Llama 3.1. This provided a practical combination of zero API cost, no external rate limits, and reproducible execution without requiring an API key.

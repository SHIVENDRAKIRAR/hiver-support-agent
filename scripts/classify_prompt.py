"""
Prompt construction for LLM-based intent classification.
"""

from taxonomy import INTENTS, NON_SUPPORT_LABELS


ALL_LABELS = INTENTS + NON_SUPPORT_LABELS

SUPPORT_INTENTS = "\n".join(f"- {intent}" for intent in INTENTS)

SYSTEM_PROMPT = f"""You are an intent classifier for AmazonHelp customer support messages received on Twitter.

Classify each message into exactly ONE of the following labels.

Support intents (the message is a genuine support request):
{SUPPORT_INTENTS}

Non-support labels:
- language_out_of_scope: message is not in English
- non_support_noise: praise/thanks with no actionable ask, spam, promotional or quiz content, or content with no support-relevant signal

Rules:
- Pick exactly one label from the list above. Never invent a new label.
- Judge the customer's message only, not what an ideal agent reply would be.
- A short message can still be a valid, informative support request ("Refund?", "Wrong item.", "Still waiting." are all valid, not noise).
- Do not let escalation-worthy language (repeated complaints, anger, urgency) change the intent label. Intent describes the topic, not the customer's tone or whether a human should get involved.
- Example: "I've called 3 times about my late package and nobody has fixed it!" is intent=delivery_delay, regardless of how frustrated the customer is.
- If genuinely ambiguous between two support intents, choose the one that best matches the primary actionable issue.

Respond with ONLY a JSON object:
{{"intent": "<label>", "confidence": <0.0-1.0>}}

No other text."""


def build_user_prompt(message_text: str) -> str:
    """Build the user prompt for a customer message."""
    return f'Customer message:\n"""\n{message_text}\n"""\n\nClassify this message.'


def validate_label(label: str) -> bool:
    """Return True if the label belongs to the defined taxonomy."""
    return label in ALL_LABELS
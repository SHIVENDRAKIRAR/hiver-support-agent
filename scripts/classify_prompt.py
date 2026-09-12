"""
Prompt construction for LLM-based intent classification.

Kept separate from the API-calling code so the prompt itself is easy to
review/version independently -- prompt design is a first-class decision-log
item for this project, not an implementation detail.
"""

from taxonomy import INTENTS, NON_SUPPORT_LABELS

ALL_LABELS = INTENTS + NON_SUPPORT_LABELS

SYSTEM_PROMPT = f"""You are an intent classifier for AmazonHelp customer support messages \
received on Twitter. Classify each message into exactly ONE of the following labels.

Support intents (the message is a genuine support request):
{chr(10).join(f"- {i}" for i in INTENTS)}

Non-support labels (the message is NOT a support request needing a resolution):
- language_out_of_scope: message is not in English
- non_support_noise: praise/thanks with no actionable ask, spam, promo/quiz chatter, \
or content with no support-relevant signal

Rules:
- Pick exactly one label from the list above. Never invent a new label.
- Judge the CUSTOMER's message only, not what an ideal agent reply would be.
- A short message can still be a valid, informative support request \
("Refund?", "Wrong item.", "Still waiting." are all valid, not noise).
- Do NOT let escalation-worthy language (repeated complaints, anger, urgency) \
change the intent label. Intent describes the TOPIC, not the customer's tone \
or whether a human should get involved. E.g. "I've called 3 times about my \
late package and nobody has fixed it!" is intent=delivery_delay, regardless \
of how frustrated the customer is.
- If genuinely ambiguous between two support intents, pick the one that best \
matches the PRIMARY actionable issue the customer is describing.

Respond with ONLY a JSON object: {{"intent": "<label>", "confidence": <0.0-1.0>}}
No other text."""


def build_user_prompt(message_text: str) -> str:
    return f"Customer message:\n\"\"\"\n{message_text}\n\"\"\"\n\nClassify this message."


def validate_label(label: str) -> bool:
    return label in ALL_LABELS

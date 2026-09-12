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
- FIRST, check the language. If the message is not written in English, the \
label MUST be language_out_of_scope -- regardless of whether it's a real \
complaint, a genuine support request, praise, or anything else. Do NOT use \
non_support_noise for non-English text, even if it looks like noise to you. \
Example: "Merci beaucoup!" (French, just thanks) is language_out_of_scope, \
NOT non_support_noise -- the language check always comes first.
- Only consider non_support_noise for ENGLISH messages that lack a real \
support-relevant ask (pure thanks, praise, spam, quiz chatter, off-topic).
- Pick exactly one label from the list above. Never invent a new label.
- Judge the CUSTOMER's message only, not what an ideal agent reply would be.
- A short message can still be a valid, informative support request \
("Refund?", "Wrong item.", "Still waiting." are all valid, not noise).
- This message may be an isolated fragment from a longer conversation and \
may lack full context (e.g. "I have not.", "Already done", "As a matter of \
fact, you did"). Do NOT default to non_support_noise just because a message \
is short or references unstated prior context -- if it reads as part of an \
ongoing support conversation (a complaint, an answer to a question, a \
frustration), classify it by its most likely topic instead. Reserve \
non_support_noise for messages that would carry no support signal even \
with full thread context (pure thanks, spam, praise, off-topic chatter).
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

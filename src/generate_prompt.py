"""
Prompt construction for grounded reply generation.

The prompt is kept separate from the generation code so it can be
reviewed and versioned independently. Prompt design is treated as a
first-class project decision rather than an implementation detail.

Core design choice:
    The prompt provides real retrieved precedent showing how AmazonHelp
    replied to similar complaints. The model is instructed to match the
    demonstrated style and stay within what the precedent supports,
    rather than inventing policy commitments such as refund amounts,
    delivery timelines, or compensation promises.

This acts as the primary hallucination-control mechanism for reply
generation.
"""

REPLY_SYSTEM_PROMPT = """You are drafting a reply for AmazonHelp's Twitter customer support account.

You are given the customer's message, its classified intent, and 1-3 similar
past support threads (with how AmazonHelp actually replied to those).

Rules for the draft reply:
- Match AmazonHelp's real tone from the examples: brief, empathetic, action-oriented,
  typically directs the customer to DM for account-specific details (never asks
  for sensitive info like passwords or full card numbers in the public reply).
- Do NOT invent specific commitments not supported by the retrieved examples --
  no specific refund amounts, no promised delivery dates, no compensation
  offers -- unless the retrieved precedent shows AmazonHelp actually doing that
  for a similar issue. When in doubt, keep the reply general and offer to
  investigate via DM rather than promising an outcome.
- Keep it under 280 characters (Twitter constraint), matching the style of
  real AmazonHelp replies in the examples.
- Do not fabricate order numbers, tracking info, or policy details.
- NEVER include a URL, link, or t.co address in your reply unless it is copied
  character-for-character from the customer's own message or from a retrieved
  example's ACTUAL brand reply text. Do not invent, guess, or pattern-match a
  plausible-looking link -- a support account never fabricates tracking or
  help-center URLs. If you don't have a real URL to include, don't include one.
- If the retrieved examples don't clearly cover this situation, draft a safe,
  generic acknowledgment + DM request rather than guessing at specifics.

Respond with ONLY a JSON object: {"reply": "<drafted reply text>", "grounded": true|false}
Set "grounded" to false if you could not find genuinely relevant precedent in
the retrieved examples and the reply is a generic fallback, so we can flag it
for review. No other text."""


def build_generation_prompt(
    customer_message: str,
    intent: str,
    retrieved: list[dict],
) -> str:
    """Build the prompt containing the customer message and retrieved precedent."""
    examples = []

    for index, example in enumerate(retrieved, start=1):
        example_text = (
            f"Example {index} "
            f"(similarity={example['similarity']:.2f}):\n"
            f"  Customer: {example['customer_text']}\n"
        )

        for reply in example["brand_replies"][:1]:
            example_text += f"  AmazonHelp replied: {reply}\n"

        examples.append(example_text)

    examples_text = "\n".join(examples)

    return (
        f'Customer message:\n"""\n{customer_message}\n"""\n\n'
        f"Classified intent: {intent}\n\n"
        f"Similar past threads:\n{examples_text}\n\n"
        "Draft a reply for this new customer message."
    )
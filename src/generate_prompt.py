"""
Prompt construction for grounded reply generation (Part 4).

Kept separate from the generation code, same as classify_prompt.py --
prompt design is a first-class decision here, not an implementation detail.

Core design choice: the prompt shows the LLM real retrieved precedent
(how AmazonHelp actually replied to similar past complaints) and instructs
it to draft in a similar STYLE and to stay within what precedent supports,
rather than inventing specific policy commitments (refund amounts, exact
timelines, compensation promises) that aren't grounded in the retrieved
examples. This is the key hallucination-control lever for this task --
without it, a support-reply generator will happily promise things Amazon's
real policy may not support.
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


def build_generation_prompt(customer_message: str, intent: str, retrieved: list) -> str:
    """
    retrieved: list of dicts from ThreadRetriever.retrieve(), each with
    'customer_text', 'brand_replies', 'similarity'.
    """
    examples_text = ""
    for i, ex in enumerate(retrieved, 1):
        examples_text += f"\nExample {i} (similarity={ex['similarity']:.2f}):\n"
        examples_text += f"  Customer: {ex['customer_text']}\n"
        for reply in ex["brand_replies"][:1]:  # just the first brand reply per example
            examples_text += f"  AmazonHelp replied: {reply}\n"

    return (
        f"Customer message:\n\"\"\"\n{customer_message}\n\"\"\"\n\n"
        f"Classified intent: {intent}\n\n"
        f"Similar past threads:{examples_text}\n\n"
        f"Draft a reply for this new customer message."
    )

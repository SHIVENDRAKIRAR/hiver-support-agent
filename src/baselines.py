"""
Baseline classifiers (Part 7) -- measure how much the LLM classifier
actually earns over cheap, obvious alternatives. Required by the
assignment (2+ baselines) and genuinely useful: if a keyword baseline
gets close to the LLM's accuracy, that's an important, humbling finding
worth reporting rather than hiding.

Baseline 1: TRIVIAL -- always predict the single most common class.
Baseline 2: KEYWORD RULES -- simple substring/regex matching per intent,
no ML or LLM at all. Rules are intentionally naive (a handful of obvious
keywords per class) -- the point is to measure the floor, not to build a
second real classifier.
"""

import re

TRIVIAL_MOST_COMMON_OVERALL = "non_support_noise"
TRIVIAL_MOST_COMMON_SUPPORT_INTENT = "delivery_delay"


def trivial_baseline(text: str, variant: str = "overall") -> str:
    """Always predicts the same label, regardless of input."""
    if variant == "overall":
        return TRIVIAL_MOST_COMMON_OVERALL
    return TRIVIAL_MOST_COMMON_SUPPORT_INTENT


KEYWORD_RULES = [
    ("billing_or_charge_issue", re.compile(r"\b(charged twice|double charg|overcharg|unauthoriz(ed)? charge|wrong (amount|price) charged)\b", re.I)),
    ("refund_or_return", re.compile(r"\b(refund|return (it|the item|my order)|money back)\b", re.I)),
    ("wrong_or_damaged_item", re.compile(r"\b(wrong item|damaged|broken|defective|torn|cracked|not what i ordered)\b", re.I)),
    ("delivery_not_received", re.compile(r"\b(marked delivered|says delivered but|never (arrived|received)|left (it |the package )?(outside|at the door)|wrong address)\b", re.I)),
    ("delivery_delay", re.compile(r"\b(still (hasn'?t|has not) (arrived|shipped)|late|delayed?|running late|hasn'?t (shipped|arrived)|where is my (order|package)|shipping soon)\b", re.I)),
    ("account_or_technical_issue", re.compile(r"\b(password|login|log in|account locked|app (crash|freeze|bug)|won'?t (open|load|work)|error message)\b", re.I)),
    ("product_or_service_inquiry", re.compile(r"\b(when will|does .* support|how do i|is there a|can i (buy|order)|available in)\b", re.I)),
]

NOISE_RULES = re.compile(
    r"^\s*(thanks?( you)?|thank you( so much)?|thx|ty|great|awesome|nice|cool|lol|ok(ay)?)[!.\s]*$",
    re.I,
)


def keyword_baseline(text: str) -> str:
    """
    Naive rule-based classifier: check obvious keyword patterns in order,
    fall back to other_support_issue if nothing matches. Does NOT attempt
    language detection or sophisticated noise filtering -- deliberately
    simple, to measure a real floor.
    """
    if NOISE_RULES.match(text.strip()):
        return "non_support_noise"

    for label, pattern in KEYWORD_RULES:
        if pattern.search(text):
            return label

    return "other_support_issue"

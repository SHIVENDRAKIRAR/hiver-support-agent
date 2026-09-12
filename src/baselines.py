"""
Baseline classifiers for measuring the value of the LLM classifier.

These baselines provide cheap, simple alternatives for comparison.

Baseline 1: TRIVIAL
    Always predicts the most common class.

Baseline 2: KEYWORD RULES
    Uses simple substring/regex rules for obvious intent patterns.
    No ML or LLM is used.

The rules are intentionally naive. The goal is to establish a simple
performance floor rather than build a competing classifier.
"""

import re


TRIVIAL_MOST_COMMON_OVERALL = "non_support_noise"
TRIVIAL_MOST_COMMON_SUPPORT_INTENT = "delivery_delay"


def trivial_baseline(
    text: str,
    variant: str = "overall",
) -> str:
    """Always predict the same label regardless of the input."""
    if variant == "overall":
        return TRIVIAL_MOST_COMMON_OVERALL

    return TRIVIAL_MOST_COMMON_SUPPORT_INTENT


KEYWORD_RULES = [
    (
        "billing_or_charge_issue",
        re.compile(
            r"\b("
            r"charged twice|double charg|overcharg|"
            r"unauthoriz(ed)? charge|wrong (amount|price) charged"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "refund_or_return",
        re.compile(
            r"\b("
            r"refund|return (it|the item|my order)|money back"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "wrong_or_damaged_item",
        re.compile(
            r"\b("
            r"wrong item|damaged|broken|defective|torn|"
            r"cracked|not what i ordered"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delivery_not_received",
        re.compile(
            r"\b("
            r"marked delivered|says delivered but|"
            r"never (arrived|received)|"
            r"left (it |the package )?(outside|at the door)|"
            r"wrong address"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delivery_delay",
        re.compile(
            r"\b("
            r"still (hasn'?t|has not) (arrived|shipped)|"
            r"late|delayed?|running late|"
            r"hasn'?t (shipped|arrived)|"
            r"where is my (order|package)|shipping soon"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "account_or_technical_issue",
        re.compile(
            r"\b("
            r"password|login|log in|account locked|"
            r"app (crash|freeze|bug)|"
            r"won'?t (open|load|work)|error message"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "product_or_service_inquiry",
        re.compile(
            r"\b("
            r"when will|does .* support|how do i|"
            r"is there a|can i (buy|order)|available in"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]


NOISE_RULES = re.compile(
    r"^\s*("
    r"thanks?( you)?|"
    r"thank you( so much)?|"
    r"thx|"
    r"ty|"
    r"great|"
    r"awesome|"
    r"nice|"
    r"cool|"
    r"lol|"
    r"ok(ay)?"
    r")[!.\s]*$",
    re.IGNORECASE,
)


def keyword_baseline(text: str) -> str:
    """
    Classify a message using simple keyword rules.

    Rules are checked in order. If no rule matches, the classifier
    returns other_support_issue.

    This baseline deliberately does not perform language detection,
    semantic classification, or sophisticated noise filtering.
    """
    if NOISE_RULES.match(text.strip()):
        return "non_support_noise"

    for label, pattern in KEYWORD_RULES:
        if pattern.search(text):
            return label

    return "other_support_issue"
"""
Locked taxonomy for the AmazonHelp support agent.

Two SEPARATE label spaces, deliberately decoupled:
  1. INTENT        -- what the customer's message is about
  2. ESCALATION     -- whether a human should handle it, and why

Keeping these separate avoids leaking the escalation answer into the
intent label (e.g. a repeated, unresolved delivery_delay complaint is
still intent=delivery_delay; escalation is a *derived* decision, not
a relabeling of intent).

Filtering rule (pre-classification): we do NOT filter on message length.
We filter on whether the message carries a support-relevant signal.
Length is irrelevant -- "Wrong item." and "Refund?" are short but valid.
"""

INTENTS = [
    "delivery_delay",              # order hasn't arrived yet / running late / ETA questions
    "delivery_not_received",       # marked delivered / left somewhere / never received / wrong address
    "refund_or_return",            # refund status, return process, money back requests
    "wrong_or_damaged_item",       # received wrong item, item damaged/defective, packaging failure
    "billing_or_charge_issue",     # double charge, unexpected charge, price discrepancy
    "account_or_technical_issue",  # login/password, app bugs, streaming/download issues, site errors
    "product_or_service_inquiry",  # pre-purchase questions, feature availability, how-to
    "other_support_issue",         # genuine support need that doesn't fit above, INCLUDING subscription/
                                    # membership issues -- measured at only 1.0% of a 1000-message classified
                                    # sample (10/1000), too thin to justify a standalone class in a ~200-example
                                    # golden set; folded in here rather than force-split a near-empty category
]

# Messages that are not support requests at all. Excluded from intent
# classification entirely (never force-fit into other_support_issue).
NON_SUPPORT_LABELS = [
    "language_out_of_scope",  # non-English message; may well be a valid support request,
                               # just outside this project's language scope — NOT noise
    "non_support_noise",      # thanks/praise with no ask, pure URLs, quiz/promo chatter,
                               # one-word non-informative replies ("lol", "ok", emoji-only)
]

ESCALATION_DECISIONS = ["auto_handle", "human_review"]

ESCALATION_REASONS = [
    "repeated_unresolved_issue",  # customer states this has happened before / multiple contacts, unresolved
    "financial_risk",             # refund/chargeback amounts, payment disputes, fraud claims
    "missing_information",        # agent lacks order ID, account info, etc. needed to resolve confidently
    "policy_exception",           # request falls outside standard policy (e.g. return window exceeded)
    "abusive_or_sensitive",       # hostile language, legal threats, vulnerable-person context
    "low_confidence",             # classifier/agent confidence below threshold on intent or reply
]


def filter_is_support_signal(text: str) -> bool:
    """
    Heuristic pre-filter: does this message carry enough signal to be a
    support request worth classifying? Deliberately NOT length-based.

    This is a first-pass heuristic only -- final filtering decisions get
    reviewed during golden-set labeling (Part 3), not trusted blindly here.
    """
    import re

    t = text.strip()
    if not t:
        return False

    # pure URL or URL + minimal text
    stripped = re.sub(r"https?://\S+", "", t).strip()
    if not stripped:
        return False

    # thanks/praise-only, no ask (very short, no question mark, no support keywords)
    low = stripped.lower()
    thanks_only = re.fullmatch(
        r"(thanks?( you)?|thank you( so much)?|thx|ty|great|awesome|nice|cool|lol|ok(ay)?|👍|❤️?|😊)[!.\s]*",
        low,
    )
    if thanks_only:
        return False

    return True

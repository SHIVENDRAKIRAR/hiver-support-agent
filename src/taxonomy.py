"""
Locked taxonomy for the AmazonHelp support agent.

The project uses two separate label spaces:

1. INTENT
   What the customer's message is about.

2. ESCALATION
   Whether a human should handle the case and why.

Keeping these spaces separate prevents escalation-related information from
leaking into the intent label. For example, a repeated unresolved
delivery_delay complaint remains delivery_delay; escalation is a derived
decision, not a different intent.

Pre-classification filtering:
    Messages are not filtered based on length. A message is retained when
    it carries a support-relevant signal. Short messages such as
    "Wrong item." and "Refund?" are therefore valid support messages.
"""

import re


INTENTS = [
    "delivery_delay",
    # Order has not arrived yet, is running late, or has an ETA question.

    "delivery_not_received",
    # Marked delivered, left somewhere, never received, or wrong address.

    "refund_or_return",
    # Refund status, return process, or money-back requests.

    "wrong_or_damaged_item",
    # Wrong item received, damaged/defective item, or packaging failure.

    "billing_or_charge_issue",
    # Double charge, unexpected charge, or price discrepancy.

    "account_or_technical_issue",
    # Login/password, app bugs, streaming/download issues, or site errors.

    "product_or_service_inquiry",
    # Pre-purchase questions, feature availability, or how-to requests.

    "other_support_issue",
    # Genuine support need that does not fit the categories above.
    #
    # Subscription/membership issues are included here. They represented
    # only 1.0% of a 1,000-message classified sample (10/1,000), which was
    # too small to justify a standalone class for a ~200-example golden set.
]


# Messages that are not support requests.
#
# These labels are handled separately and should never be force-fit into
# other_support_issue.
NON_SUPPORT_LABELS = [
    "language_out_of_scope",
    # Non-English message. It may be a valid support request, but it is
    # outside this project's language scope and is NOT considered noise.

    "non_support_noise",
    # Thanks/praise without an ask, pure URLs, quiz/promotional chatter,
    # and non-informative replies such as "lol", "ok", or emoji-only text.
]


ESCALATION_DECISIONS = [
    "auto_handle",
    "human_review",
]


ESCALATION_REASONS = [
    "repeated_unresolved_issue",
    # Customer says the issue happened before or has involved multiple
    # unsuccessful contacts.

    "financial_risk",
    # Refund/chargeback amounts, payment disputes, or fraud claims.

    "missing_information",
    # Information such as an order ID or account details is needed to
    # resolve the issue confidently.

    "policy_exception",
    # Request falls outside standard policy, such as an expired return
    # window.

    "abusive_or_sensitive",
    # Hostile language, legal threats, or vulnerable-person context.

    "low_confidence",
    # Classifier or reply-generation confidence is below the threshold.
]


def filter_is_support_signal(text: str) -> bool:
    """
    Determine whether a message carries a support-relevant signal.

    This is a first-pass heuristic and intentionally does not use message
    length as a filtering criterion. Final filtering decisions are reviewed
    during golden-set labeling rather than being trusted blindly.
    """
    text = text.strip()

    if not text:
        return False

    # Remove URLs. If nothing remains, the message is effectively a
    # URL-only message.
    text_without_urls = re.sub(
        r"https?://\S+",
        "",
        text,
    ).strip()

    if not text_without_urls:
        return False

    # Filter simple thanks/praise-only messages that contain no question
    # or support-relevant signal.
    normalized_text = text_without_urls.lower()

    thanks_only = re.fullmatch(
        r"(thanks?( you)?|"
        r"thank you( so much)?|"
        r"thx|"
        r"ty|"
        r"great|"
        r"awesome|"
        r"nice|"
        r"cool|"
        r"lol|"
        r"ok(ay)?|"
        r"👍|"
        r"❤️?|"
        r"😊)"
        r"[!.\s]*",
        normalized_text,
    )

    if thanks_only:
        return False

    return True
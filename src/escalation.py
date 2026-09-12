"""
Escalation decision logic for the AmazonHelp support agent.

Deliberately a TWO-STAGE design:
  1. Deterministic rule checks (cheap, auditable, catch obvious cases)
  2. LLM judgment call for anything the rules don't confidently resolve

This mirrors how real support tooling works -- you don't want an LLM
making the call on "did the customer already contact us 3 times", that's
a fact you can check directly from thread history. The LLM's job is
judgment on ambiguous/subjective cases (tone, policy exceptions, missing
context), not re-deriving facts already available in structured form.

Escalation is a SEPARATE decision from intent (see src/taxonomy.py) --
an angry, repeated delivery_delay complaint is still intent=delivery_delay;
this module decides escalate=True and why, independently.
"""

import re
from dataclasses import dataclass

from taxonomy import ESCALATION_DECISIONS, ESCALATION_REASONS

# Rule 1: repeated unresolved issue -- customer explicitly states they've
# contacted before, multiple times, still unresolved
REPEATED_CONTACT_RE = re.compile(
    r"\b(again|already (told|said|contacted|called|emailed|reported)|"
    r"\d+\s*(times|times now)|for the (second|third|\d+(st|nd|rd|th)) time|"
    r"still (not|no) (resolved|fixed|sorted)|"
    r"(called|contacted|emailed) (you |amazon )?\d+)\b",
    re.IGNORECASE,
)

# Rule 2: financial risk -- specific amounts, chargebacks, fraud claims
FINANCIAL_RISK_RE = re.compile(
    r"\b(chargeback|dispute|fraud(ulent)?|unauthoriz(ed|ation)|"
    r"\$\s?\d+|€\s?\d+|£\s?\d+|₹\s?\d+|refund of \d+|"
    r"legal action|small claims|lawyer|sue|report(ed)? to (my )?bank)\b",
    re.IGNORECASE,
)

# Rule 3: abusive or sensitive language
ABUSIVE_RE = re.compile(
    r"\b(scam(ming)?|thief|steal(ing)?|criminal|illegal|fuck|shit|bullshit|"
    r"disgusting|pathetic|useless|incompetent|threat(en(ing)?)?)\b",
    re.IGNORECASE,
)

# Rule 4: missing information the agent would need -- no order ref, no
# specific detail, purely vague ("my order is late" with zero identifiers)
# handled at the confidence-check stage, not via regex (see escalate())


@dataclass
class EscalationResult:
    decision: str          # one of ESCALATION_DECISIONS
    reasons: list[str]     # subset of ESCALATION_REASONS, can be multiple
    rule_hits: list[str]   # which regex rules fired, for auditability
    notes: str = ""


def rule_based_check(message_text: str, classifier_confidence: float | None = None) -> EscalationResult:
    """
    Fast, deterministic pass. Returns escalate=True if ANY hard rule fires.
    classifier_confidence: pass the intent classifier's confidence score if
    available -- low confidence is itself an escalation reason (rule 5).
    """
    reasons = []
    rule_hits = []

    if REPEATED_CONTACT_RE.search(message_text):
        reasons.append("repeated_unresolved_issue")
        rule_hits.append("REPEATED_CONTACT_RE")

    if FINANCIAL_RISK_RE.search(message_text):
        reasons.append("financial_risk")
        rule_hits.append("FINANCIAL_RISK_RE")

    if ABUSIVE_RE.search(message_text):
        reasons.append("abusive_or_sensitive")
        rule_hits.append("ABUSIVE_RE")

    if classifier_confidence is not None and classifier_confidence < 0.6:
        reasons.append("low_confidence")
        rule_hits.append("LOW_CLASSIFIER_CONFIDENCE")

    decision = "human_review" if reasons else "auto_handle"

    return EscalationResult(decision=decision, reasons=reasons, rule_hits=rule_hits)


LLM_ESCALATION_SYSTEM_PROMPT = f"""You are deciding whether an AmazonHelp customer support \
message should be auto-handled by an AI agent or escalated to a human.

Escalate to human_review if ANY of these apply:
- policy_exception: the request falls outside standard, clearly-defined policy \
(e.g. return window has passed, non-standard compensation requested)
- missing_information: you would need account/order specifics not present in \
the message to give a confident, correct reply
- abusive_or_sensitive: hostile language, legal threats, or a vulnerable-person context
- Anything else where an automated reply carries real risk of being wrong or \
making the situation worse

Otherwise, auto_handle.

Valid reasons: {", ".join(ESCALATION_REASONS)}

Respond with ONLY JSON: {{"decision": "auto_handle" | "human_review", \
"reasons": ["<reason>", ...], "rationale": "<one sentence>"}}"""


def build_llm_escalation_prompt(message_text: str, intent: str, rule_result: EscalationResult) -> str:
    """
    Give the LLM the rule-based pre-check as context so it isn't re-deriving
    facts the rules already caught -- it should focus on the judgment calls
    the rules can't make (policy exceptions, ambiguous missing-info cases).
    """
    return (
        f"Message:\n\"\"\"\n{message_text}\n\"\"\"\n\n"
        f"Classified intent: {intent}\n"
        f"Rule-based pre-check already flagged: {rule_result.reasons or 'none'}\n\n"
        f"Confirm or revise the escalation decision, adding any judgment-based "
        f"reasons (policy_exception, missing_information) the rules can't detect."
    )

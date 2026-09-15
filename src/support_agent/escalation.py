"""Runtime escalation policy (Phase 10).

Approximates the ground-truth policy in docs/escalation_policy.md using
signals actually available at inference time - it cannot see "this needs
account-specific investigation" directly (that's a judgment about the
world, not a measurable feature), so it infers escalation from proxies:

  1. Intent-based prior risk: some intents escalate by policy regardless of
     confidence (REFUND_REQUEST always - see docs/escalation_policy.md).
  2. Classifier confidence below a per-intent threshold.
  3. Retrieval evidence quality: if the best-matching historical resolutions
     are only weakly similar to the incoming message, there's nothing solid
     to ground a reply in.
  4. Keyword risk flags: safety-net overrides for high-stakes language
     (hacked, fraud, lawyer, etc.) that force escalation even if the
     classifier and retrieval both look confident - misclassifying one of
     these is worse than an unnecessary escalation.

Every decision returns which signal fired, so the policy is inspectable
(Phase 10 explicitly asks for this) rather than a black box.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Intents that escalate unconditionally - see docs/escalation_policy.md
# "Escalate when... SENSITIVE_FINANCIAL... applies to essentially all
# REFUND_REQUEST cases."
ALWAYS_ESCALATE_INTENTS = {
    "REFUND_REQUEST": "SENSITIVE_FINANCIAL",
}

# Per-intent confidence floor below which we escalate as LOW_CONFIDENCE.
# Structurally riskier intents (money, account access) get a higher bar.
# Defaults chosen conservatively pending calibration against the golden set
# (see scripts/13_calibrate_thresholds.py) - documented as a placeholder,
# not a claimed-optimal value, until real classifier confidence scores exist.
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
INTENT_CONFIDENCE_THRESHOLDS = {
    "PAYMENT_BILLING_ISSUE": 0.65,
    "ACCOUNT_LOGIN_ACCESS": 0.65,
    "SUBSCRIPTION_PLAN_MANAGEMENT": 0.6,
}

# Below this top-1 retrieval similarity, treat historical evidence as too
# weak to ground a reply in (see docs/escalation_policy.md "Retrieval
# evidence quality"). TF-IDF cosine similarities on short tweets are
# naturally low (see scripts/09's sample outputs, ~0.25-0.4 for good
# matches), so this threshold is calibrated to that scale, not to a dense
# embedding scale.
MIN_RETRIEVAL_SIMILARITY = 0.15

# Safety-net keyword overrides. Each maps to the escalation reason it forces
# regardless of predicted intent or confidence - deliberately broader/blunter
# than the intent taxonomy so a misclassification on a high-stakes message
# doesn't slip through.
RISK_KEYWORD_PATTERNS: dict[str, re.Pattern] = {
    "SECURITY_CONCERN": re.compile(r"\b(hack(ed|ing)?|compromised|unauthorized access|phishing|fraudulent(ly)?)\b", re.I),
    "SENSITIVE_FINANCIAL": re.compile(r"\b(chargeback|lawyer|legal action|sue|fraud(ulent)? charge|dispute the charge)\b", re.I),
}


@dataclass
class EscalationDecision:
    action: str  # "AUTO_HANDLE" | "ESCALATE"
    reason: str | None
    signals: dict = field(default_factory=dict)


def decide(
    intent: str,
    confidence: float,
    customer_message: str,
    retrieval_results: list[dict] | None = None,
) -> EscalationDecision:
    signals: dict = {"intent": intent, "confidence": confidence}

    for reason, pattern in RISK_KEYWORD_PATTERNS.items():
        match = pattern.search(customer_message)
        if match:
            signals["matched_risk_keyword"] = match.group(0)
            return EscalationDecision("ESCALATE", reason, signals)

    if intent in ALWAYS_ESCALATE_INTENTS:
        return EscalationDecision("ESCALATE", ALWAYS_ESCALATE_INTENTS[intent], signals)

    threshold = INTENT_CONFIDENCE_THRESHOLDS.get(intent, DEFAULT_CONFIDENCE_THRESHOLD)
    signals["confidence_threshold"] = threshold
    if confidence < threshold:
        return EscalationDecision("ESCALATE", "LOW_CONFIDENCE", signals)

    if retrieval_results:
        top_similarity = retrieval_results[0]["similarity"]
        signals["top_retrieval_similarity"] = top_similarity
        if top_similarity < MIN_RETRIEVAL_SIMILARITY:
            return EscalationDecision("ESCALATE", "WEAK_OR_CONFLICTING_EVIDENCE", signals)

    return EscalationDecision("AUTO_HANDLE", None, signals)

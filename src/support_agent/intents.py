"""Intent taxonomy for the SpotifyCares support agent.

The authoritative human-readable definitions (inclusion/exclusion criteria,
examples, ambiguous cases) live in docs/intent_taxonomy.md - this module is
the machine-readable mirror used by prompts, the weak labeler, and tests.
Keep the two in sync; INTENT_NAMES order here doesn't matter semantically but
should match the taxonomy doc's numbering for readability.
"""
from __future__ import annotations

INTENT_NAMES = [
    "ACCOUNT_LOGIN_ACCESS",
    "PAYMENT_BILLING_ISSUE",
    "REFUND_REQUEST",
    "SUBSCRIPTION_PLAN_MANAGEMENT",
    "TECHNICAL_PLAYBACK_ISSUE",
    "CONTENT_CATALOG_AVAILABILITY",
    "FEATURE_REQUEST_OR_INFO",
    "POSITIVE_FEEDBACK_OR_RESOLVED",
    "OTHER_AMBIGUOUS",
]

INTENT_DESCRIPTIONS = {
    "ACCOUNT_LOGIN_ACCESS": (
        "Customer cannot log in, reset a password, recover, or verify "
        "ownership of their account, including broken third-party "
        "(Facebook) login links."
    ),
    "PAYMENT_BILLING_ISSUE": (
        "Something about how the customer is being charged is wrong or "
        "confusing; they are NOT explicitly asking for money back."
    ),
    "REFUND_REQUEST": (
        "Customer explicitly asks to cancel and get money back, or disputes "
        "a charge as unauthorized and wants it reversed."
    ),
    "SUBSCRIPTION_PLAN_MANAGEMENT": (
        "Questions or problems about which plan/tier the customer is on, "
        "moving between plans, family/student plan membership, or plain "
        "cancellation (no refund ask)."
    ),
    "TECHNICAL_PLAYBACK_ISSUE": (
        "The app or a device integration is mechanically broken: playback "
        "errors, crashes, offline download failures, sync/device issues, "
        "app bugs."
    ),
    "CONTENT_CATALOG_AVAILABILITY": (
        "A specific song, album, artist, or podcast is missing, "
        "region-locked, mislabeled, or the customer wants something added "
        "to the catalog."
    ),
    "FEATURE_REQUEST_OR_INFO": (
        "Not reporting something broken - asking how something works, "
        "requesting a feature that doesn't exist, or a general "
        "informational question."
    ),
    "POSITIVE_FEEDBACK_OR_RESOLVED": (
        "Thanking the brand, confirming an issue is already resolved, or "
        "unsolicited praise. No action needed."
    ),
    "OTHER_AMBIGUOUS": (
        "Too vague to act on, off-topic, multi-issue rambles with no "
        "dominant intent, or outside what a generic support agent handles."
    ),
}

assert set(INTENT_DESCRIPTIONS) == set(INTENT_NAMES)

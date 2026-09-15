"""Heuristic (keyword/regex) intent labeler.

This is NOT a source of ground truth. It has two legitimate uses in this
project, both documented in docs/decision_log.md:

  1. Producing "silver" training labels for the TF-IDF baseline (Phase 6),
     since hand-labeling the full ~10.5k-example train split is out of scope
     for a take-home. The baseline's ceiling is therefore bounded by how
     good these heuristics are, not by the modeling approach - this
     limitation is called out explicitly in REPORT.md.
  2. Stratifying which conversations get sampled as golden-set *candidates*
     (Phase 4), so the ~200 hand-labelled examples aren't all the single
     most common intent. The heuristic label is discarded after sampling -
     every golden example gets a real intent label from the annotation pass
     in scripts/06_label_golden_set.py, not from this module.

Rules are ordered by precedence: more specific/high-precision patterns are
checked first so e.g. a message mentioning both "refund" and "premium" is
labeled REFUND_REQUEST rather than SUBSCRIPTION_PLAN_MANAGEMENT.
"""
from __future__ import annotations

import re

RULES: list[tuple[str, re.Pattern]] = [
    (
        "REFUND_REQUEST",
        re.compile(
            r"\b(refund|reimburse|money back|charged.*(cancel|stop)|"
            r"give (me )?my money|chargeback|dispute.*charge|keeps? "
            r"charging|still (being |getting )?charged|cancel.*(twice|"
            r"multiple times).*charg)\b", re.I
        ),
    ),
    (
        "ACCOUNT_LOGIN_ACCESS",
        re.compile(
            r"\b(log[\s-]?in|login|sign[\s-]?in|password|locked out|"
            r"hacked|csrf|can'?t access my account|forgot.*(password|login))\b",
            re.I,
        ),
    ),
    (
        "PAYMENT_BILLING_ISSUE",
        re.compile(
            r"\b(charged|billing|payment(s)?|credit card|debit card|"
            r"invoice|overcharged|double charged|declined|won'?t accept "
            r"(my|the|any) card|cards? (won'?t|wouldn'?t) (work|accept))\b",
            re.I,
        ),
    ),
    (
        "SUBSCRIPTION_PLAN_MANAGEMENT",
        re.compile(
            r"\b(premium|family plan|student (discount|plan|verification|"
            r"eligib)|upgrade|downgrade|cancel my (subscription|account|"
            r"premium)|subscription|which plan)\b", re.I
        ),
    ),
    (
        "TECHNICAL_PLAYBACK_ISSUE",
        re.compile(
            r"\b(crash(es|ed|ing)?|bug|won'?t play|can'?t play|can'?t "
            r"listen|not playing|not working|isn'?t working|doesn'?t "
            r"work|stopped working|stopped? \w+ing|freeze(s|d)?|frozen|"
            r"offline (download|mode)|sync(ing)?|glitch|buffering|keeps? "
            r"(stopping|skipping|pausing|coming up|un[- ]?download)|app "
            r"(hangs|hung|froze)|doesn'?t connect|won'?t (load|connect|"
            r"open)|never loads?|broken)\b", re.I
        ),
    ),
    (
        "CONTENT_CATALOG_AVAILABILITY",
        re.compile(
            r"\b(missing from|not available in|taken down|removed from "
            r"spotify|isn'?t on spotify|add .*(album|artist|song|podcast) "
            r"to spotify|region( ?-?locked)?|catalog|only (exists|"
            r"available) in|only lists?|wrong (artist|album|song)|"
            r"mislabel)\b", re.I
        ),
    ),
    (
        "FEATURE_REQUEST_OR_INFO",
        re.compile(
            r"\b(how do i|is there (a|any) way|can you add|feature "
            r"request|would be (great|nice|cool) if|any plans to|will "
            r"there be|gift card|when will you|why don'?t you have|please "
            r"(add|keep|bring back)|wish (you|there was)|shortcut)\b",
            re.I,
        ),
    ),
    (
        "POSITIVE_FEEDBACK_OR_RESOLVED",
        re.compile(
            r"\b(thank you|thanks(?! for the (charge|bug))|much appreciated|"
            r"great service|fixed now|problem fixed|resolved now|works "
            r"now)\b", re.I
        ),
    ),
]

DEFAULT_INTENT = "OTHER_AMBIGUOUS"


def weak_label(text: str) -> tuple[str, str | None]:
    """Return (intent, matched_pattern_name_or_None).

    Falls back to OTHER_AMBIGUOUS with no match if nothing fires, or if the
    message is too short to carry signal (<3 words is already filtered
    upstream in scripts/03, but keep the guard here for reuse elsewhere).
    """
    if len(text.split()) < 3:
        return DEFAULT_INTENT, None
    for intent, pattern in RULES:
        if pattern.search(text):
            return intent, pattern.pattern
    return DEFAULT_INTENT, None

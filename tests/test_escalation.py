from support_agent.escalation import decide


def test_refund_always_escalates_even_with_high_confidence():
    decision = decide(
        intent="REFUND_REQUEST",
        confidence=0.99,
        customer_message="Can I get a refund for last month?",
    )
    assert decision.action == "ESCALATE"
    assert decision.reason == "SENSITIVE_FINANCIAL"


def test_low_confidence_triggers_escalation():
    decision = decide(
        intent="FEATURE_REQUEST_OR_INFO",
        confidence=0.2,
        customer_message="How do I sort my playlist?",
    )
    assert decision.action == "ESCALATE"
    assert decision.reason == "LOW_CONFIDENCE"


def test_high_confidence_low_risk_intent_auto_handles():
    decision = decide(
        intent="FEATURE_REQUEST_OR_INFO",
        confidence=0.9,
        customer_message="How do I sort my playlist alphabetically?",
        retrieval_results=[{"similarity": 0.4}, {"similarity": 0.3}],
    )
    assert decision.action == "AUTO_HANDLE"
    assert decision.reason is None


def test_security_keyword_overrides_high_confidence_and_intent():
    decision = decide(
        intent="ACCOUNT_LOGIN_ACCESS",
        confidence=0.95,
        customer_message="I think my account was hacked, please help",
        retrieval_results=[{"similarity": 0.5}],
    )
    assert decision.action == "ESCALATE"
    assert decision.reason == "SECURITY_CONCERN"


def test_weak_retrieval_evidence_triggers_escalation():
    decision = decide(
        intent="TECHNICAL_PLAYBACK_ISSUE",
        confidence=0.9,
        customer_message="The app is doing something weird today.",
        retrieval_results=[{"similarity": 0.05}, {"similarity": 0.02}],
    )
    assert decision.action == "ESCALATE"
    assert decision.reason == "WEAK_OR_CONFLICTING_EVIDENCE"


def test_no_retrieval_results_does_not_crash():
    decision = decide(
        intent="FEATURE_REQUEST_OR_INFO",
        confidence=0.9,
        customer_message="How do I sort my playlist?",
        retrieval_results=None,
    )
    assert decision.action == "AUTO_HANDLE"

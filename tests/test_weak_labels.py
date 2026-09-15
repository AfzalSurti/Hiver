from support_agent.weak_labels import weak_label, DEFAULT_INTENT


def test_refund_keyword_detected():
    intent, rule = weak_label("Can I get a refund for this month please")
    assert intent == "REFUND_REQUEST"
    assert rule is not None


def test_login_keyword_detected():
    intent, _ = weak_label("I forgot my password and can't log in")
    assert intent == "ACCOUNT_LOGIN_ACCESS"


def test_short_text_falls_back_to_default():
    intent, rule = weak_label("hi there")
    assert intent == DEFAULT_INTENT
    assert rule is None


def test_no_keyword_match_falls_back_to_default():
    intent, rule = weak_label("This is a message with absolutely no matching keywords at all today")
    assert intent == DEFAULT_INTENT
    assert rule is None


def test_refund_takes_precedence_over_subscription_keywords():
    # Message contains both "premium" (subscription) and "refund" - refund
    # rule is checked first and should win (see RULES ordering).
    intent, _ = weak_label("I cancelled my premium subscription, can I get a refund?")
    assert intent == "REFUND_REQUEST"

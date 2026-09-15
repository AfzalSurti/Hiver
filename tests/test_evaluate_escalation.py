from support_agent.evaluate_escalation import compute_escalation_metrics


def test_perfect_predictions():
    y_true = ["AUTO_HANDLE", "ESCALATE", "ESCALATE", "AUTO_HANDLE"]
    y_pred = ["AUTO_HANDLE", "ESCALATE", "ESCALATE", "AUTO_HANDLE"]
    metrics = compute_escalation_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["per_class"]["ESCALATE"]["f1"] == 1.0
    assert metrics["per_class"]["AUTO_HANDLE"]["f1"] == 1.0


def test_all_wrong_predictions():
    y_true = ["AUTO_HANDLE", "AUTO_HANDLE"]
    y_pred = ["ESCALATE", "ESCALATE"]
    metrics = compute_escalation_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.0
    assert metrics["per_class"]["AUTO_HANDLE"]["recall"] == 0.0


def test_confusion_matrix_totals_match_support():
    y_true = ["AUTO_HANDLE", "ESCALATE", "ESCALATE"]
    y_pred = ["ESCALATE", "ESCALATE", "AUTO_HANDLE"]
    metrics = compute_escalation_metrics(y_true, y_pred)
    total = sum(sum(row) for row in metrics["confusion_matrix"]["matrix"])
    assert total == 3
    assert metrics["n_examples"] == 3

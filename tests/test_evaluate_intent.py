from support_agent.evaluate_intent import compute_intent_metrics


def test_perfect_predictions_score_1():
    labels = ["A", "B"]
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "A", "B", "B"]
    metrics = compute_intent_metrics(y_true, y_pred, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["per_class"]["A"]["support"] == 2


def test_all_wrong_predictions_score_0():
    labels = ["A", "B"]
    y_true = ["A", "A", "B", "B"]
    y_pred = ["B", "B", "A", "A"]
    metrics = compute_intent_metrics(y_true, y_pred, labels)
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_confusion_matrix_shape_matches_labels():
    labels = ["A", "B", "C"]
    y_true = ["A", "B", "C", "A"]
    y_pred = ["A", "C", "C", "A"]
    metrics = compute_intent_metrics(y_true, y_pred, labels)
    cm = metrics["confusion_matrix"]
    assert cm["labels"] == labels
    assert len(cm["matrix"]) == 3
    assert all(len(row) == 3 for row in cm["matrix"])


def test_n_examples_matches_input_length():
    labels = ["A", "B"]
    y_true = ["A", "B", "A"]
    y_pred = ["A", "B", "B"]
    metrics = compute_intent_metrics(y_true, y_pred, labels)
    assert metrics["n_examples"] == 3

"""Shared ACTION/ESCALATION metrics (Phase 13), used both for the
TF-IDF-driven sanity check (scripts/15, no API key needed) and, once an
OpenRouter key is available, the real LLM-driven pipeline evaluation.
"""
from __future__ import annotations

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ACTIONS = ["AUTO_HANDLE", "ESCALATE"]


def compute_escalation_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ACTIONS, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=ACTIONS).tolist()

    per_class = {
        label: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(ACTIONS)
    }

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": ACTIONS, "matrix": cm},
        "n_examples": len(y_true),
    }

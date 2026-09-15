"""Phase 13 (partial, no API key needed): evaluate the escalation policy
end-to-end using the TF-IDF baseline as the intent/confidence source and
the TF-IDF retrieval index as the evidence source.

This is NOT the final AI system's escalation performance - it's a sanity
check that the escalation policy mechanics (src/support_agent/escalation.py)
behave sensibly end-to-end, runnable today without an LLM. Once
OPENROUTER_API_KEY is available, scripts/11_run_full_pipeline.py produces
the real LLM-driven decisions and the equivalent metrics should be computed
from that output instead (see docs/decision_log.md for why the two numbers
are not interchangeable - the LLM classifier's confidence calibration and
the TF-IDF's predict_proba are not the same kind of signal).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import PROCESSED_DIR, GOLDEN_DIR, REPORTS_DIR, SEED, ensure_dirs  # noqa: E402
from support_agent.escalation import decide  # noqa: E402
from support_agent.evaluate_escalation import compute_escalation_metrics  # noqa: E402
from support_agent.retrieval import RetrievalIndex  # noqa: E402


def main() -> None:
    ensure_dirs()

    train_path = PROCESSED_DIR / "train_weak_labels.jsonl"
    train_records = [json.loads(line) for line in train_path.open(encoding="utf-8")]
    X_train = [r["customer_message"] for r in train_records]
    y_train = [r["weak_intent"] for r in train_records]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]
    )
    pipeline.fit(X_train, y_train)
    class_order = list(pipeline.named_steps["clf"].classes_)

    index = RetrievalIndex()

    golden_path = GOLDEN_DIR / "golden_set.jsonl"
    golden = [json.loads(line) for line in golden_path.open(encoding="utf-8")]

    y_true_action, y_pred_action = [], []
    detail_records = []

    for rec in golden:
        message = rec["customer_message"]
        proba = pipeline.predict_proba([message])[0]
        pred_intent = class_order[proba.argmax()]
        confidence = float(proba.max())

        retrieval_results = index.retrieve(message, k=5)
        decision = decide(
            intent=pred_intent,
            confidence=confidence,
            customer_message=message,
            retrieval_results=retrieval_results,
        )

        y_true_action.append(rec["expected_action"])
        y_pred_action.append(decision.action)
        detail_records.append(
            {
                "conversation_id": rec["conversation_id"],
                "customer_message": message,
                "true_intent": rec["intent"],
                "predicted_intent": pred_intent,
                "confidence": round(confidence, 4),
                "true_action": rec["expected_action"],
                "true_reason": rec["escalation_reason"],
                "predicted_action": decision.action,
                "predicted_reason": decision.reason,
                "signals": decision.signals,
            }
        )

    metrics = compute_escalation_metrics(y_true_action, y_pred_action)
    metrics["source"] = "tfidf_baseline_intent_and_confidence"

    out_dir = REPORTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "escalation_tfidf.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "escalation_tfidf_details.jsonl").open("w", encoding="utf-8") as f:
        for rec in detail_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    for label, m in metrics["per_class"].items():
        print(f"  {label:12s} precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} support={m['support']}")
    print("\nConfusion matrix (rows=true, cols=predicted):", metrics["confusion_matrix"]["labels"])
    for row in metrics["confusion_matrix"]["matrix"]:
        print(" ", row)
    print(f"\nWrote {out_dir / 'escalation_tfidf.json'}")
    print(f"Wrote {out_dir / 'escalation_tfidf_details.jsonl'}")


if __name__ == "__main__":
    main()

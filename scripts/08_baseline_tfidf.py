"""Phase 6: simple ML baseline - TF-IDF + Logistic Regression.

Trained on the TRAIN split's weak/heuristic labels only (scripts/04) -
hand-labelling the full ~10.5k train examples was out of scope for this
project, so this baseline's ceiling is bounded by heuristic label quality,
not by TF-IDF/LogReg as a method (see docs/decision_log.md #8). Evaluated
on the golden set, which is fully hand-labelled and was never used to train
or tune this model (it lives in the eval split, disjoint from train - see
scripts/03).

class_weight="balanced" is used because the weak labels are heavily skewed
toward OTHER_AMBIGUOUS (53.4% of train, see scripts/04's output) - without
balancing, the classifier would trivially approach the trivial baseline's
behavior on the majority class. This is a legitimate, standard mitigation,
not a way of hiding the noisy-label problem; the model still never sees a
single clean label during training.
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
from support_agent.evaluate_intent import compute_intent_metrics  # noqa: E402
from support_agent.intents import INTENT_NAMES  # noqa: E402


def main() -> None:
    ensure_dirs()

    train_path = PROCESSED_DIR / "train_weak_labels.jsonl"
    train_records = [json.loads(line) for line in train_path.open(encoding="utf-8")]
    X_train = [r["customer_message"] for r in train_records]
    y_train = [r["weak_intent"] for r in train_records]
    print(f"Training on {len(X_train):,} weakly-labeled examples.")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    golden_path = GOLDEN_DIR / "golden_set.jsonl"
    golden = [json.loads(line) for line in golden_path.open(encoding="utf-8")]
    X_eval = [r["customer_message"] for r in golden]
    y_true = [r["intent"] for r in golden]
    y_pred = list(pipeline.predict(X_eval))

    metrics = compute_intent_metrics(y_true, y_pred, INTENT_NAMES)
    metrics["baseline_name"] = "tfidf_logreg_weak_labels"
    metrics["train_size"] = len(X_train)
    metrics["train_label_source"] = "heuristic_weak_labels (scripts/04)"

    out_dir = REPORTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_tfidf.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_path = out_dir / "baseline_tfidf_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for rec, pred in zip(golden, y_pred):
            f.write(
                json.dumps(
                    {
                        "conversation_id": rec["conversation_id"],
                        "customer_message": rec["customer_message"],
                        "true_intent": rec["intent"],
                        "predicted_intent": pred,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"\nAccuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nPer-class F1:")
    for label, m in metrics["per_class"].items():
        print(f"  {label:32s} f1={m['f1']:.3f}  support={m['support']}")
    print(f"\nWrote {out_path}")
    print(f"Wrote {predictions_path}")


if __name__ == "__main__":
    main()

"""Phase 5: trivial baseline - always predict the single most common intent.

"Most common" is determined from the TRAIN split's weak/heuristic labels
(scripts/04), NOT from the golden set's own label distribution - using the
golden set to pick the trivial baseline's constant answer would be
circular (the baseline would be tuned on the data it's evaluated on). Using
the noisy train-side signal instead mirrors what would actually be available
before ever looking at held-out evaluation data, even though that signal is
itself imperfect (see docs/decision_log.md #8).

Note the train weak-label distribution is dominated by OTHER_AMBIGUOUS
(53.4% - see scripts/04's output), which is itself an artifact of the
heuristic's limited recall, not a claim that most real SpotifyCares traffic
is unclassifiable. This baseline is expected to score poorly - that's the
point of a trivial baseline.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import PROCESSED_DIR, GOLDEN_DIR, REPORTS_DIR, ensure_dirs  # noqa: E402
from support_agent.evaluate_intent import compute_intent_metrics  # noqa: E402
from support_agent.intents import INTENT_NAMES  # noqa: E402


def main() -> None:
    ensure_dirs()

    train_weak_path = PROCESSED_DIR / "train_weak_labels.jsonl"
    weak_intents = [json.loads(line)["weak_intent"] for line in train_weak_path.open(encoding="utf-8")]
    majority_intent = Counter(weak_intents).most_common(1)[0][0]
    print(f"Majority intent from train weak labels: {majority_intent}")

    golden_path = GOLDEN_DIR / "golden_set.jsonl"
    golden = [json.loads(line) for line in golden_path.open(encoding="utf-8")]

    y_true = [rec["intent"] for rec in golden]
    y_pred = [majority_intent] * len(golden)

    metrics = compute_intent_metrics(y_true, y_pred, INTENT_NAMES)
    metrics["baseline_name"] = "trivial_majority_class"
    metrics["majority_intent"] = majority_intent

    out_dir = REPORTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_trivial.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nAccuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

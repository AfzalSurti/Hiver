"""Phase 7: run the LLM intent classifier over the golden set and compare
against both baselines (Phase 5/6).

Requires OPENROUTER_API_KEY (see .env.example). Caches every raw
classification to data/llm_cache/classifier_cache.jsonl keyed by
conversation_id, so re-running after a partial failure or to add more
examples doesn't re-spend API credits on examples already classified.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import GOLDEN_DIR, CACHE_DIR, REPORTS_DIR, ensure_dirs, load_env  # noqa: E402
from support_agent.classifier import classify  # noqa: E402
from support_agent.evaluate_intent import compute_intent_metrics  # noqa: E402
from support_agent.intents import INTENT_NAMES  # noqa: E402
from support_agent.llm_client import LLMConfigError  # noqa: E402

CACHE_PATH = CACHE_DIR / "classifier_cache.jsonl"


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    cache = {}
    for line in CACHE_PATH.open(encoding="utf-8"):
        rec = json.loads(line)
        cache[rec["conversation_id"]] = rec
    return cache


def append_cache(rec: dict) -> None:
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ensure_dirs()
    load_env()

    golden_path = GOLDEN_DIR / "golden_set.jsonl"
    golden = [json.loads(line) for line in golden_path.open(encoding="utf-8")]

    cache = load_cache()
    y_true, y_pred, confidences = [], [], []

    for i, rec in enumerate(golden):
        cid = rec["conversation_id"]
        if cid in cache:
            result = cache[cid]["result"]
        else:
            try:
                result = classify(rec["customer_message"])
            except LLMConfigError as e:
                print(f"\nERROR: {e}")
                print("Cannot run the LLM classifier without an API key. Exiting.")
                sys.exit(1)
            append_cache({"conversation_id": cid, "result": result})
            time.sleep(0.1)  # light rate-limit courtesy

        y_true.append(rec["intent"])
        y_pred.append(result["intent"])
        confidences.append(result["confidence"])
        if (i + 1) % 25 == 0:
            print(f"  classified {i + 1}/{len(golden)} ...")

    metrics = compute_intent_metrics(y_true, y_pred, INTENT_NAMES)
    metrics["classifier_name"] = "llm_classifier"
    metrics["avg_confidence"] = round(sum(confidences) / len(confidences), 4)

    out_dir = REPORTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ai_classifier.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nAccuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Avg confidence: {metrics['avg_confidence']:.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

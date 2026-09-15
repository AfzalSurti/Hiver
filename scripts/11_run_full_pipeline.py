"""Phase 9/10: run the full pipeline (classify -> retrieve -> escalate ->
generate) over the golden set and write structured outputs matching the
schema in Phase 9 of the assignment:

    {"intent": ..., "reply": ..., "decision": "AUTO_HANDLE"|"ESCALATE",
     "reason": ..., "evidence_ids": [...]}

For ESCALATE decisions, "reply" is a short templated hand-off message, not
an LLM-generated one - there is no point asking a generator to draft a
grounded resolution for a case we've already decided needs a human, and
templating avoids any hallucination risk on exactly the cases flagged as
highest-risk.

Requires OPENROUTER_API_KEY. Caches per-conversation_id to
data/llm_cache/pipeline_cache.jsonl so re-runs don't re-spend API credits.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import GOLDEN_DIR, CACHE_DIR, REPORTS_DIR, ensure_dirs, load_env  # noqa: E402
from support_agent.classifier import classify  # noqa: E402
from support_agent.escalation import decide  # noqa: E402
from support_agent.generation import generate_reply  # noqa: E402
from support_agent.llm_client import LLMConfigError  # noqa: E402
from support_agent.retrieval import RetrievalIndex  # noqa: E402

CACHE_PATH = CACHE_DIR / "pipeline_cache.jsonl"
RETRIEVAL_K = 5

ESCALATE_TEMPLATE = (
    "Thanks for reaching out - this one needs a closer look from a member of our team "
    "who can access account-specific details, so we're passing it along to them now."
)


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


def run_one(customer_message: str, index: RetrievalIndex) -> dict:
    classification = classify(customer_message)
    intent = classification["intent"]
    confidence = classification["confidence"]

    retrieval_results = index.retrieve(customer_message, k=RETRIEVAL_K)

    decision = decide(
        intent=intent,
        confidence=confidence,
        customer_message=customer_message,
        retrieval_results=retrieval_results,
    )

    if decision.action == "AUTO_HANDLE":
        gen = generate_reply(customer_message, intent, retrieval_results)
        reply = gen["reply"]
        evidence_ids = gen["evidence_ids"]
        grounding_note = gen["grounding_note"]
    else:
        reply = ESCALATE_TEMPLATE
        evidence_ids = []
        grounding_note = "Escalated before generation - no reply drafted to avoid unnecessary risk."

    return {
        "intent": intent,
        "confidence": confidence,
        "classifier_rationale": classification["rationale"],
        "reply": reply,
        "decision": decision.action,
        "reason": decision.reason,
        "escalation_signals": decision.signals,
        "evidence_ids": evidence_ids,
        "grounding_note": grounding_note,
        "retrieved_evidence": retrieval_results,
    }


def main() -> None:
    ensure_dirs()
    load_env()

    golden_path = GOLDEN_DIR / "golden_set.jsonl"
    golden = [json.loads(line) for line in golden_path.open(encoding="utf-8")]

    try:
        index = RetrievalIndex()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    cache = load_cache()
    out_records = []

    for i, rec in enumerate(golden):
        cid = rec["conversation_id"]
        if cid in cache:
            result = cache[cid]["result"]
        else:
            try:
                result = run_one(rec["customer_message"], index)
            except LLMConfigError as e:
                print(f"\nERROR: {e}")
                print("Cannot run the pipeline without an API key. Exiting.")
                sys.exit(1)
            append_cache({"conversation_id": cid, "result": result})
            time.sleep(0.1)

        out_records.append({"conversation_id": cid, "customer_message": rec["customer_message"], **result})
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(golden)} ...")

    out_dir = REPORTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pipeline_outputs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_auto = sum(1 for r in out_records if r["decision"] == "AUTO_HANDLE")
    n_escalate = len(out_records) - n_auto
    print(f"\nAUTO_HANDLE: {n_auto} ({n_auto / len(out_records):.1%})")
    print(f"ESCALATE:    {n_escalate} ({n_escalate / len(out_records):.1%})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

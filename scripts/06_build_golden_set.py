"""Phase 4 (final step): merge sampled candidates + manual annotations into
the golden evaluation set.

Reads data/golden/golden_candidates.jsonl (sampled in scripts/05) and
data/golden/annotations.jsonl (manual labels, one JSON object per line: see
data/golden/ANNOTATION_METHODOLOGY.md for how these were produced), joins on
conversation_id, and writes data/golden/golden_set.jsonl - the frozen
evaluation set used everywhere downstream.

Fails loudly if any candidate is missing an annotation or vice versa, so the
golden set can never silently drop or duplicate an example.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import GOLDEN_DIR  # noqa: E402
from support_agent.intents import INTENT_NAMES  # noqa: E402

VALID_ACTIONS = {"AUTO_HANDLE", "ESCALATE"}
VALID_REASONS = {
    "SENSITIVE_FINANCIAL",
    "ACCOUNT_SPECIFIC_INVESTIGATION",
    "SECURITY_CONCERN",
    "AMBIGUOUS_OR_INSUFFICIENT_INFO",
    "REPEATED_UNRESOLVED_ISSUE",
    "OUT_OF_SCOPE",
}


def main() -> None:
    candidates_path = GOLDEN_DIR / "golden_candidates.jsonl"
    annotations_path = GOLDEN_DIR / "annotations.jsonl"
    out_path = GOLDEN_DIR / "golden_set.jsonl"

    candidates = {}
    with candidates_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            candidates[rec["conversation_id"]] = rec

    annotations = {}
    with annotations_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            rec = json.loads(line)
            cid = rec["conversation_id"]
            if cid in annotations:
                raise ValueError(f"Duplicate annotation for {cid} at line {line_no}")
            if rec["intent"] not in INTENT_NAMES:
                raise ValueError(f"Unknown intent {rec['intent']!r} for {cid}")
            if rec["expected_action"] not in VALID_ACTIONS:
                raise ValueError(f"Invalid action {rec['expected_action']!r} for {cid}")
            if rec["expected_action"] == "ESCALATE" and rec["escalation_reason"] not in VALID_REASONS:
                raise ValueError(f"ESCALATE requires a valid escalation_reason for {cid}")
            if rec["expected_action"] == "AUTO_HANDLE" and rec["escalation_reason"] is not None:
                raise ValueError(f"AUTO_HANDLE must have escalation_reason=null for {cid}")
            annotations[cid] = rec

    missing_annotations = set(candidates) - set(annotations)
    missing_candidates = set(annotations) - set(candidates)
    if missing_annotations:
        raise ValueError(f"{len(missing_annotations)} candidates have no annotation: {sorted(missing_annotations)[:5]}...")
    if missing_candidates:
        raise ValueError(f"{len(missing_candidates)} annotations have no matching candidate: {sorted(missing_candidates)[:5]}...")

    golden = []
    for cid, cand in candidates.items():
        ann = annotations[cid]
        golden.append(
            {
                "conversation_id": cid,
                "customer_message": cand["customer_message"],
                "conversation_context": cand["conversation_context"],
                "brand_response": cand["brand_response"],
                "timestamp": cand["timestamp"],
                "sample_bucket": cand["sample_bucket"],
                "intent": ann["intent"],
                "expected_action": ann["expected_action"],
                "escalation_reason": ann["escalation_reason"],
                "annotation_notes": ann["notes"],
            }
        )

    golden.sort(key=lambda r: r["conversation_id"])

    with out_path.open("w", encoding="utf-8") as f:
        for rec in golden:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(golden)} golden examples -> {out_path}")
    print("\nIntent distribution:")
    for intent, c in Counter(r["intent"] for r in golden).most_common():
        print(f"  {intent:32s} {c:4d}  ({c / len(golden):.1%})")
    print("\nAction distribution:")
    for action, c in Counter(r["expected_action"] for r in golden).most_common():
        print(f"  {action:32s} {c:4d}  ({c / len(golden):.1%})")
    print("\nEscalation reason distribution (ESCALATE only):")
    for reason, c in Counter(r["escalation_reason"] for r in golden if r["escalation_reason"]).most_common():
        print(f"  {reason:32s} {c:4d}")


if __name__ == "__main__":
    main()

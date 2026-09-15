"""Phase 12 (setup step): sample generated replies for human review against
the same rubric the LLM judge used, to measure judge-human agreement.

Samples from reports/eval/judge_scores.jsonl (so only replies that were
actually judged are eligible) with a target of ~50 examples (within the
brief's 40-60 range), stratified across the judge's own overall-score
buckets (1-2, 3, 4-5) so the human reviewer sees a mix of clearly-bad,
middling, and clearly-good replies rather than whatever the natural score
distribution happens to concentrate on - agreement is more informative when
both ends of the quality range are represented.

Writes data/eval/human_review_template.jsonl with the rubric score fields
left as null for a human to fill in by hand (see
data/eval/HUMAN_REVIEW_METHODOLOGY.md for the disclosed single-annotator
process, same rationale as the golden set's).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import REPORTS_DIR, DATA_DIR, SEED, ensure_dirs  # noqa: E402
from support_agent.judge import RUBRIC_DIMENSIONS  # noqa: E402

TARGET_SAMPLE_SIZE = 50
EVAL_DIR = DATA_DIR / "eval"


def bucket_for(overall: int) -> str:
    if overall <= 2:
        return "low"
    if overall == 3:
        return "mid"
    return "high"


def main() -> None:
    ensure_dirs()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    judge_path = REPORTS_DIR / "eval" / "judge_scores.jsonl"
    if not judge_path.exists():
        print(f"ERROR: {judge_path} not found. Run scripts/12_run_judge.py first.")
        sys.exit(1)

    judged = [json.loads(line) for line in judge_path.open(encoding="utf-8")]

    by_bucket: dict[str, list] = {"low": [], "mid": [], "high": []}
    for rec in judged:
        by_bucket[bucket_for(rec["scores"]["overall"])].append(rec)

    per_bucket_target = TARGET_SAMPLE_SIZE // 3
    selected = []
    for bucket, pool in by_bucket.items():
        rng.shuffle(pool)
        selected.extend(pool[:per_bucket_target])

    remaining_target = TARGET_SAMPLE_SIZE - len(selected)
    if remaining_target > 0:
        already_ids = {r["conversation_id"] for r in selected}
        leftover = [r for r in judged if r["conversation_id"] not in already_ids]
        rng.shuffle(leftover)
        selected.extend(leftover[:remaining_target])

    rng.shuffle(selected)

    # Deliberately do NOT include the LLM judge's scores in this file - a human
    # rater seeing them while scoring would anchor on them, which would inflate
    # the agreement measurement instead of testing it honestly. scripts/14
    # re-joins by conversation_id against judge_scores.jsonl separately.
    out_path = EVAL_DIR / "human_review_template.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in selected:
            template = {
                "conversation_id": rec["conversation_id"],
                "customer_message": rec["customer_message"],
                "reply": rec["reply"],
            }
            for dim in RUBRIC_DIMENSIONS + ["overall"]:
                template[f"human_{dim}"] = None
            template["human_rationale"] = None
            f.write(json.dumps(template, ensure_ascii=False) + "\n")

    print(f"Sampled {len(selected)} examples for human review -> {out_path}")
    print("The LLM judge's own scores are deliberately withheld from this file to avoid")
    print("anchoring the human rater - see data/eval/HUMAN_REVIEW_METHODOLOGY.md.")
    print("Next: fill in the human_* fields by hand, then run scripts/14_compute_judge_agreement.py")


if __name__ == "__main__":
    main()

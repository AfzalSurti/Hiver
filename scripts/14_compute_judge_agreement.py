"""Phase 12: compute LLM-judge vs. human agreement.

Joins the human-filled data/eval/human_review_template.jsonl (see
scripts/13) against reports/eval/judge_scores.jsonl by conversation_id, and
reports Spearman correlation (appropriate for ordinal 1-5 rubric scores)
plus a simple within-1-point agreement rate, per rubric dimension and
overall. Never invents this number - fails loudly if the human template
still has unfilled (null) fields, since that would silently score partial
labels as agreement/disagreement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import REPORTS_DIR, DATA_DIR, ensure_dirs  # noqa: E402
from support_agent.judge import RUBRIC_DIMENSIONS  # noqa: E402

EVAL_DIR = DATA_DIR / "eval"


def main() -> None:
    ensure_dirs()

    human_path = EVAL_DIR / "human_review_template.jsonl"
    judge_path = REPORTS_DIR / "eval" / "judge_scores.jsonl"
    if not human_path.exists():
        print(f"ERROR: {human_path} not found. Run scripts/13_sample_human_review.py first.")
        sys.exit(1)
    if not judge_path.exists():
        print(f"ERROR: {judge_path} not found. Run scripts/12_run_judge.py first.")
        sys.exit(1)

    human = [json.loads(line) for line in human_path.open(encoding="utf-8")]
    judge_by_id = {json.loads(line)["conversation_id"]: json.loads(line) for line in judge_path.open(encoding="utf-8")}

    unfilled = [r["conversation_id"] for r in human if r["human_overall"] is None]
    if unfilled:
        print(
            f"ERROR: {len(unfilled)} examples in {human_path} still have human_overall=null. "
            "Fill in every human_* field before computing agreement (see "
            "data/eval/HUMAN_REVIEW_METHODOLOGY.md). First few: "
            f"{unfilled[:5]}"
        )
        sys.exit(1)

    dims = RUBRIC_DIMENSIONS + ["overall"]
    results = {}
    for dim in dims:
        human_scores, judge_scores = [], []
        for rec in human:
            cid = rec["conversation_id"]
            if cid not in judge_by_id:
                continue
            human_scores.append(rec[f"human_{dim}"])
            judge_scores.append(judge_by_id[cid]["scores"][dim])

        if len(human_scores) < 2:
            results[dim] = {"error": "not enough matched examples to compute correlation"}
            continue

        corr, p_value = spearmanr(human_scores, judge_scores)
        within_1 = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= 1) / len(human_scores)
        exact = sum(1 for h, j in zip(human_scores, judge_scores) if h == j) / len(human_scores)

        results[dim] = {
            "n": len(human_scores),
            "spearman_r": round(float(corr), 4),
            "p_value": round(float(p_value), 4),
            "exact_agreement_rate": round(exact, 4),
            "within_1_point_agreement_rate": round(within_1, 4),
        }

    out_path = REPORTS_DIR / "eval" / "judge_human_agreement.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

"""Phase 11: LLM-as-judge scoring of generated replies.

Only scores AUTO_HANDLE replies (pipeline_outputs.jsonl from scripts/11) -
ESCALATE outputs are a fixed template with no grounding claims to judge, so
scoring them would just be re-measuring the template's constant quality.

Requires OPENROUTER_API_KEY. Caches to data/llm_cache/judge_cache.jsonl.
Per-example failures are recorded but don't abort the whole run, since a
judge call failing shouldn't lose already-collected scores.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import CACHE_DIR, REPORTS_DIR, ensure_dirs, load_env  # noqa: E402
from support_agent.judge import RUBRIC_DIMENSIONS, judge_reply  # noqa: E402
from support_agent.llm_client import LLMConfigError  # noqa: E402

CACHE_PATH = CACHE_DIR / "judge_cache.jsonl"


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    return {json.loads(line)["conversation_id"]: json.loads(line) for line in CACHE_PATH.open(encoding="utf-8")}


def append_cache(rec: dict) -> None:
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ensure_dirs()
    load_env()

    pipeline_path = REPORTS_DIR / "eval" / "pipeline_outputs.jsonl"
    if not pipeline_path.exists():
        print(f"ERROR: {pipeline_path} not found. Run scripts/11_run_full_pipeline.py first.")
        sys.exit(1)

    pipeline_outputs = [json.loads(line) for line in pipeline_path.open(encoding="utf-8")]
    auto_handled = [r for r in pipeline_outputs if r["decision"] == "AUTO_HANDLE"]
    print(f"Judging {len(auto_handled)} AUTO_HANDLE replies (of {len(pipeline_outputs)} total)...")

    cache = load_cache()
    scored = []
    n_failed = 0

    for i, rec in enumerate(auto_handled):
        cid = rec["conversation_id"]
        if cid in cache:
            scores = cache[cid]["scores"]
        else:
            try:
                scores = judge_reply(rec["customer_message"], rec["reply"], rec["retrieved_evidence"])
            except LLMConfigError as e:
                print(f"\nERROR: {e}")
                sys.exit(1)
            except Exception as e:  # noqa: BLE001
                print(f"  WARNING: judge failed for {cid}: {e}")
                n_failed += 1
                continue
            append_cache({"conversation_id": cid, "scores": scores})
            time.sleep(0.1)

        scored.append({"conversation_id": cid, "customer_message": rec["customer_message"], "reply": rec["reply"], "scores": scores})
        if (i + 1) % 25 == 0:
            print(f"  judged {i + 1}/{len(auto_handled)} ...")

    out_dir = REPORTS_DIR / "eval"
    out_path = out_dir / "judge_scores.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in scored:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {"n_scored": len(scored), "n_failed": n_failed}
    for dim in RUBRIC_DIMENSIONS + ["overall"]:
        vals = [r["scores"][dim] for r in scored]
        summary[f"avg_{dim}"] = round(sum(vals) / len(vals), 3) if vals else None

    summary_path = out_dir / "judge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"\nWrote {out_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

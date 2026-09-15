"""Phase 4 (sampling step): sample ~210 golden-set CANDIDATES from the eval
split only (never train - see scripts/03's leakage-prevention split).

This script does NOT assign ground-truth labels. It picks which
conversations get hand-labelled, using deliberate stratification so the
golden set isn't dominated by the single most common intent:

  - up to 20 candidates per intent, selected via the weak/heuristic labeler
    (used here purely as a sampling filter to get topical diversity - the
    real label comes later from manual annotation, see scripts/06)
  - ~15 "multi-signal" candidates where 2+ intent rules fire at once, as a
    cheap proxy for multi-issue/ambiguous messages
  - ~15 short messages (<=6 words after cleaning) to stress-test the
    classifier on low-signal input
  - a pure-random top-up from the remaining eval pool so the set isn't
    entirely an artifact of the heuristics' blind spots

All buckets are deduplicated by conversation_id. Output:
data/golden/golden_candidates.jsonl (unlabelled - annotation happens in
scripts/06_label_golden_set.py).

Seed is fixed (see support_agent.config.SEED) for reproducibility.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import PROCESSED_DIR, GOLDEN_DIR, SEED, ensure_dirs  # noqa: E402
from support_agent.intents import INTENT_NAMES  # noqa: E402
from support_agent.weak_labels import weak_label  # noqa: E402
from support_agent.weak_labels_debug import matching_intents  # noqa: E402

PER_INTENT_TARGET = 20
MULTI_SIGNAL_TARGET = 15
SHORT_MSG_TARGET = 15
RANDOM_TOPUP_TARGET = 20


def main() -> None:
    ensure_dirs()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    conv_path = PROCESSED_DIR / "conversations.jsonl"
    eval_pool = []
    with conv_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] == "eval":
                eval_pool.append(rec)
    print(f"Eval pool size: {len(eval_pool):,}")

    by_intent = defaultdict(list)
    multi_signal = []
    short_msgs = []
    for rec in eval_pool:
        text = rec["customer_message"]
        intent, rule = weak_label(text)
        rec["_weak_intent"] = intent
        rec["_matched_rule"] = rule
        by_intent[intent].append(rec)
        if len(matching_intents(text)) >= 2:
            multi_signal.append(rec)
        if len(text.split()) <= 6:
            short_msgs.append(rec)

    selected: dict[str, dict] = {}
    bucket_of: dict[str, str] = {}

    for intent in INTENT_NAMES:
        pool = by_intent.get(intent, [])
        rng.shuffle(pool)
        for rec in pool[:PER_INTENT_TARGET]:
            cid = rec["conversation_id"]
            if cid not in selected:
                selected[cid] = rec
                bucket_of[cid] = f"stratified_{intent}"

    rng.shuffle(multi_signal)
    n_added = 0
    for rec in multi_signal:
        if n_added >= MULTI_SIGNAL_TARGET:
            break
        cid = rec["conversation_id"]
        if cid not in selected:
            selected[cid] = rec
            bucket_of[cid] = "multi_signal"
            n_added += 1

    rng.shuffle(short_msgs)
    n_added = 0
    for rec in short_msgs:
        if n_added >= SHORT_MSG_TARGET:
            break
        cid = rec["conversation_id"]
        if cid not in selected:
            selected[cid] = rec
            bucket_of[cid] = "short_message"
            n_added += 1

    remaining = [r for r in eval_pool if r["conversation_id"] not in selected]
    rng.shuffle(remaining)
    n_added = 0
    for rec in remaining:
        if n_added >= RANDOM_TOPUP_TARGET:
            break
        cid = rec["conversation_id"]
        selected[cid] = rec
        bucket_of[cid] = "random_topup"
        n_added += 1

    print(f"Total candidates selected: {len(selected)}")
    from collections import Counter
    bucket_counts = Counter(bucket_of.values())
    for b, c in bucket_counts.most_common():
        print(f"  {b}: {c}")

    out_path = GOLDEN_DIR / "golden_candidates.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        records = list(selected.values())
        rng.shuffle(records)  # shuffle so annotation order doesn't cluster by bucket
        for rec in records:
            cid = rec["conversation_id"]
            out_rec = {
                "conversation_id": cid,
                "customer_message": rec["customer_message"],
                "conversation_context": rec["conversation_context"],
                "brand_response": rec["brand_response"],
                "timestamp": rec["timestamp"],
                "sample_bucket": bucket_of[cid],
                "weak_intent": rec["_weak_intent"],
                "matched_rule": rec["_matched_rule"],
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

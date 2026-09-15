"""Phase 3/6 support: apply heuristic weak labels to the TRAIN split only.

Produces data/processed/train_weak_labels.jsonl, used exclusively to train
the TF-IDF baseline (Phase 6). See src/support_agent/weak_labels.py for why
these are heuristic "silver" labels, not ground truth, and why that's an
acceptable/disclosed limitation for a baseline model.

Never applied to the eval split for training purposes - that split is
reserved for golden-set sampling (Phase 4), where every label instead comes
from the manual annotation pass.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import PROCESSED_DIR  # noqa: E402
from support_agent.weak_labels import weak_label  # noqa: E402


def main() -> None:
    conv_path = PROCESSED_DIR / "conversations.jsonl"
    out_path = PROCESSED_DIR / "train_weak_labels.jsonl"

    counts = Counter()
    n_written = 0
    with conv_path.open(encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            if rec["split"] != "train":
                continue
            intent, rule = weak_label(rec["customer_message"])
            counts[intent] += 1
            f_out.write(
                json.dumps(
                    {
                        "conversation_id": rec["conversation_id"],
                        "customer_message": rec["customer_message"],
                        "weak_intent": intent,
                        "matched_rule": rule,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(f"Weak-labeled {n_written:,} train examples -> {out_path}")
    print("\nDistribution:")
    for intent, c in counts.most_common():
        print(f"  {intent:32s} {c:6,}  ({c / n_written:.1%})")


if __name__ == "__main__":
    main()

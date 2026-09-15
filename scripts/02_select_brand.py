"""Phase 1 (cont.): Formalize brand selection based on scripts/01 EDA evidence.

Writes data/processed/selected_brand.txt so every downstream script agrees,
and prints the evidence used to justify the choice. See docs/decision_log.md
entry #1 for the full write-up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import REPORTS_DIR, PROCESSED_DIR, SELECTED_BRAND_PATH, ensure_dirs  # noqa: E402

SELECTED_BRAND = "SpotifyCares"


def main() -> None:
    ensure_dirs()
    eda_path = REPORTS_DIR / "eda" / "eda_summary.json"
    summary = json.loads(eda_path.read_text(encoding="utf-8"))

    brand_msg_count = summary["top_brands_by_agent_messages"].get(SELECTED_BRAND)
    conv_stats = summary["candidate_conversation_stats"].get(SELECTED_BRAND)

    print(f"Selected brand: {SELECTED_BRAND}")
    print(f"  Agent-side (outbound) message volume: {brand_msg_count:,}")
    print(f"  Conversation structure (sampled): {conv_stats}")
    print(
        "\nRationale: single, well-bounded product domain (one app/service) vs. "
        "AmazonHelp (spans all retail categories) or AppleSupport (spans many "
        "hardware product lines) which fragment into far more than ~8-10 clean "
        "intents. High resolution rate in-thread (pct_with_brand_reply="
        f"{conv_stats['pct_with_brand_reply']}) and sufficient volume "
        f"({brand_msg_count:,} agent messages, {conv_stats['n_root_conversations_sampled']:,}+ "
        "root conversation threads sampled) to support train/golden splits without "
        "the full 2.8M-row dataset. See docs/decision_log.md #1 for full reasoning."
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_BRAND_PATH.write_text(SELECTED_BRAND, encoding="utf-8")
    print(f"\nWrote {SELECTED_BRAND_PATH}")


if __name__ == "__main__":
    main()

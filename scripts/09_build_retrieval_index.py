"""Phase 8: build the embedding index over historically-resolved
conversations (TRAIN split only). Run once; cached under data/embeddings/.

Runtime: ~1-3 minutes on CPU for ~10.5k short messages with all-MiniLM-L6-v2
(plus a one-time ~90MB model download on first run).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.retrieval import build_index  # noqa: E402

if __name__ == "__main__":
    build_index()

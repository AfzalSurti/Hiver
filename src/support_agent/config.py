"""Central paths, seeds, and environment configuration for the project.

Kept dependency-free (no dotenv/openai import at module load) so that
importing this module never requires an API key to be set.
"""
from __future__ import annotations

import os
from pathlib import Path

SEED = 42

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"
CACHE_DIR = DATA_DIR / "llm_cache"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_TWCS_PATH = RAW_DIR / "twcs.csv"

# Populated once a brand is selected in Phase 1 (kept here so every
# downstream script agrees on the same value without re-deriving it).
SELECTED_BRAND_PATH = PROCESSED_DIR / "selected_brand.txt"


def get_selected_brand() -> str:
    if not SELECTED_BRAND_PATH.exists():
        raise FileNotFoundError(
            f"{SELECTED_BRAND_PATH} not found. Run scripts/01_explore_data.py "
            "and scripts/02_select_brand.py first."
        )
    return SELECTED_BRAND_PATH.read_text(encoding="utf-8").strip()


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, GOLDEN_DIR, CACHE_DIR, EMBEDDINGS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    """Load .env if present. Only called by scripts that actually need API keys."""
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")

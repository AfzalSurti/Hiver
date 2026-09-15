"""Diagnostic helper used only for golden-set sampling (Phase 4) - finds
messages that match MULTIPLE intent rule patterns at once, as a cheap proxy
for "multi-issue" messages worth deliberately including in the golden set.
Not used anywhere in the trained baseline or production path.
"""
from __future__ import annotations

from support_agent.weak_labels import RULES


def matching_intents(text: str) -> list[str]:
    return [intent for intent, pattern in RULES if pattern.search(text)]

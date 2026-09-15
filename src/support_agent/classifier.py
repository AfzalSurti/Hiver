"""Phase 7: LLM-based intent classifier with structured, validated output.

Deliberately narrow scope: the LLM's only job here is to pick one of the 9
taxonomy intents and self-report a confidence - it is NOT asked to draft a
reply or make the escalation decision in this call (see
docs/decision_log.md for why responsibilities are split across separate,
smaller LLM calls rather than one do-everything prompt).
"""
from __future__ import annotations

import os

from support_agent.intents import INTENT_DESCRIPTIONS, INTENT_NAMES
from support_agent.llm_client import LLMConfigError, call_json

FALLBACK_INTENT = "OTHER_AMBIGUOUS"
FALLBACK_CONFIDENCE = 0.0

SYSTEM_PROMPT = """You are an intent classifier for SpotifyCares customer support messages.
Classify the customer's message into exactly one of these intents:

{intent_list}

Respond with ONLY a JSON object of the form:
{{"intent": "<ONE_OF_THE_INTENT_NAMES_ABOVE>", "confidence": <float 0.0-1.0>, "rationale": "<one short sentence>"}}

confidence should reflect how certain you are given only the message text - use lower
values for vague, multi-issue, or ambiguous messages, higher values for clear-cut cases."""


def _build_system_prompt() -> str:
    lines = [f"- {name}: {INTENT_DESCRIPTIONS[name]}" for name in INTENT_NAMES]
    return SYSTEM_PROMPT.format(intent_list="\n".join(lines))


def _validate(parsed: dict) -> None:
    if parsed.get("intent") not in INTENT_NAMES:
        raise ValueError(f"intent must be one of {INTENT_NAMES}, got {parsed.get('intent')!r}")
    conf = parsed.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        raise ValueError(f"confidence must be a float in [0,1], got {conf!r}")


def classify(customer_message: str, model: str | None = None) -> dict:
    """Returns {"intent": str, "confidence": float, "rationale": str}.

    Falls back to OTHER_AMBIGUOUS/confidence 0.0 if the model produces
    invalid output after retries, rather than raising and crashing a batch
    evaluation run - callers can check confidence==0.0 to detect this case.

    Does NOT catch LLMConfigError (e.g. missing API key) - that's a setup
    problem, not a per-example model-output robustness issue, and must
    propagate so callers fail loudly instead of silently writing fallback
    results for an entire batch (see docs/decision_log.md for why this
    distinction matters).
    """
    model = model or os.environ.get("OPENROUTER_CLASSIFIER_MODEL", "openai/gpt-4o-mini")
    try:
        return call_json(
            model=model,
            system_prompt=_build_system_prompt(),
            user_prompt=f"Customer message:\n{customer_message}",
            validate=_validate,
        )
    except LLMConfigError:
        raise
    except Exception as e:  # noqa: BLE001
        return {
            "intent": FALLBACK_INTENT,
            "confidence": FALLBACK_CONFIDENCE,
            "rationale": f"Classifier failed after retries, defaulting to safe fallback: {e}",
        }

"""Phase 9: retrieval-grounded reply generation.

Given a customer message, its predicted intent, and the top-k retrieved
historical resolutions (from src/support_agent/retrieval.py), generates a
customer-facing reply that is grounded in what evidence actually shows the
brand doing for similar issues - and is explicitly instructed to prefer
admitting it doesn't know over inventing policies, refunds, URLs, or
account actions with no supporting evidence (Phase 8's core requirement).

This module only drafts a reply candidate; the AUTO_HANDLE/ESCALATE call
itself is made separately by src/support_agent/escalation.py so a weak or
missing evidence signal there can override generation even if the drafted
reply reads fine.
"""
from __future__ import annotations

import os

from support_agent.llm_client import LLMConfigError, call_json

SYSTEM_PROMPT = """You are drafting a reply for SpotifyCares' Twitter customer support account.

You will be given the customer's message and a set of historical examples showing how
SpotifyCares actually resolved similar past issues (customer message -> brand's real reply).

Rules:
- Ground your reply in what the historical evidence actually shows. Do not invent specific
  policies, refund amounts, compensation, account actions, URLs, or guarantees that are not
  supported by the evidence provided.
- If the evidence is weak, thin, or the historical replies mostly just say "please DM us" with
  no real resolution content, write a short, honest, generic reply (e.g., basic troubleshooting
  or acknowledging you'll look into it) rather than fabricating specifics - do NOT pretend the
  evidence supports something it doesn't.
- Keep it brief, friendly, and in SpotifyCares' typical tone (casual, uses the customer's first
  name if given, signs off naturally) - but do not invent a sign-off initials pattern if none is
  evident.
- Cite which of the provided evidence_ids actually informed your reply (empty list if none did).

Respond with ONLY a JSON object of the form:
{"reply": "<the drafted reply text>", "evidence_ids": ["<id1>", "<id2>", ...], "grounding_note": "<one sentence on what evidence you used, or 'insufficient evidence' if none>"}"""


def _validate(parsed: dict) -> None:
    if not isinstance(parsed.get("reply"), str) or not parsed["reply"].strip():
        raise ValueError("reply must be a non-empty string")
    if not isinstance(parsed.get("evidence_ids"), list):
        raise ValueError("evidence_ids must be a list")


def _format_evidence(retrieval_results: list[dict]) -> str:
    if not retrieval_results:
        return "(no historical evidence retrieved)"
    lines = []
    for r in retrieval_results:
        lines.append(
            f"- id={r['conversation_id']} (similarity={r['similarity']:.3f})\n"
            f"  customer: {r['customer_message']}\n"
            f"  brand reply: {r['brand_response']}"
        )
    return "\n".join(lines)


def generate_reply(
    customer_message: str,
    intent: str,
    retrieval_results: list[dict],
    model: str | None = None,
) -> dict:
    """Returns {"reply": str, "evidence_ids": list[str], "grounding_note": str}."""
    model = model or os.environ.get("OPENROUTER_GENERATION_MODEL", "openai/gpt-4o-mini")
    user_prompt = (
        f"Predicted intent: {intent}\n\n"
        f"Customer message:\n{customer_message}\n\n"
        f"Historical evidence (most similar past resolutions):\n{_format_evidence(retrieval_results)}"
    )
    try:
        return call_json(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            validate=_validate,
        )
    except LLMConfigError:
        raise
    except Exception as e:  # noqa: BLE001
        return {
            "reply": (
                "Thanks for reaching out - we want to make sure this gets looked at properly, "
                "so we're passing it to a member of our team to help you directly."
            ),
            "evidence_ids": [],
            "grounding_note": f"Generation failed after retries, used safe fallback reply: {e}",
        }

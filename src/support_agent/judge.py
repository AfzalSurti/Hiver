"""Phase 11: LLM-as-judge for generated reply quality.

Scores a generated reply against the customer message and the retrieved
evidence it was supposed to be grounded in, on a fixed 1-5 rubric per
dimension (never a vague "good/bad"). Uses a DIFFERENT default model than
generation (see .env.example: OPENROUTER_JUDGE_MODEL defaults to an
Anthropic model while generation defaults to an OpenAI model) specifically
to reduce self-preference bias - a model judging its own outputs tends to
rate them more favorably.
"""
from __future__ import annotations

import os

from support_agent.llm_client import call_json

RUBRIC_DIMENSIONS = [
    "correctness",
    "relevance",
    "groundedness",
    "helpfulness",
    "brand_tone",
    "safety",
]

SYSTEM_PROMPT = """You are an expert quality reviewer for customer-support replies from SpotifyCares.

You will be given: the customer's message, the historical evidence the reply was supposed to be
grounded in, and the drafted reply. Score the reply from 1-5 on each dimension below. Be strict -
most rubrics fail by being too generous. Use the full 1-5 range across a batch of ratings.

1. correctness (1-5): Is everything stated in the reply actually true / consistent with the
   evidence? 1 = confidently states something false or unsupported, 5 = everything checks out.
2. relevance (1-5): Does the reply actually address what the customer asked? 1 = off-topic or
   generic non-answer, 5 = directly addresses the specific issue.
3. groundedness (1-5): Is the reply's content traceable to the provided historical evidence
   (not invented)? 1 = fabricates specifics (policies, refunds, URLs, account actions) with no
   evidence support, 5 = every specific claim is backed by the evidence, and the reply is
   appropriately generic/honest when evidence is weak.
4. helpfulness (1-5): Would this reply actually move the customer's problem forward? 1 = useless
   or deflecting with no substance, 5 = clear, actionable next step or resolution.
5. brand_tone (1-5): Is the tone appropriate for a friendly Twitter support account? 1 = robotic,
   cold, or inappropriately casual/rude, 5 = warm and on-brand without being saccharine.
6. safety (1-5): Does the reply avoid making commitments/promises (refunds, compensation, account
   changes, guarantees) it has no authority or evidence to make? 1 = makes an unsupported binding
   promise, 5 = appropriately cautious given what's actually known.

Respond with ONLY a JSON object of the form:
{"correctness": <int 1-5>, "relevance": <int 1-5>, "groundedness": <int 1-5>,
 "helpfulness": <int 1-5>, "brand_tone": <int 1-5>, "safety": <int 1-5>,
 "overall": <int 1-5>, "rationale": "<2-3 sentences justifying the scores, mentioning any
 specific fabrication or inaccuracy if found>"}"""


def _validate(parsed: dict) -> None:
    for dim in RUBRIC_DIMENSIONS + ["overall"]:
        val = parsed.get(dim)
        if not isinstance(val, int) or not (1 <= val <= 5):
            raise ValueError(f"{dim} must be an int in [1,5], got {val!r}")
    if not isinstance(parsed.get("rationale"), str) or not parsed["rationale"].strip():
        raise ValueError("rationale must be a non-empty string")


def judge_reply(
    customer_message: str,
    reply: str,
    retrieval_results: list[dict],
    model: str | None = None,
) -> dict:
    """Returns the rubric dict: 6 dimension scores (1-5) + overall + rationale."""
    model = model or os.environ.get("OPENROUTER_JUDGE_MODEL", "anthropic/claude-3.5-sonnet")

    if retrieval_results:
        evidence_text = "\n".join(
            f"- customer: {r['customer_message']}\n  brand reply: {r['brand_response']}"
            for r in retrieval_results
        )
    else:
        evidence_text = "(no historical evidence was retrieved)"

    user_prompt = (
        f"Customer message:\n{customer_message}\n\n"
        f"Historical evidence provided to the reply-writer:\n{evidence_text}\n\n"
        f"Drafted reply to score:\n{reply}"
    )
    return call_json(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        validate=_validate,
    )

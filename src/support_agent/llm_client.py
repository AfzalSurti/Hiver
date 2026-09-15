"""Thin OpenRouter client wrapper (OpenAI-compatible API, see
docs/decision_log.md #3) with a JSON-structured-output helper that validates
and retries on malformed output rather than trusting the model blindly.

Every LLM-dependent script in this project (scripts/10+) calls through this
module so retry/validation behavior is consistent everywhere.
"""
from __future__ import annotations

import json
import os
import re


class LLMConfigError(RuntimeError):
    """Raised when required LLM configuration (API key, etc.) is missing."""


def _client():
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in "
            "(see README's 'Getting an OpenRouter API key' section)."
        )
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of model output that may have stray prose,
    markdown code fences, etc. wrapped around it."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))
    raise ValueError(f"Could not extract JSON from model output: {text[:300]!r}")


def call_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    validate: callable,
    max_retries: int = 2,
    temperature: float = 0.2,
) -> dict:
    """Call the model, parse JSON, and validate it with `validate(dict) ->
    None` (raise ValueError/KeyError on invalid content). Retries with a
    corrective message on parse or validation failure. Raises the last
    error if all retries are exhausted - callers decide the fallback
    behavior (e.g. classifier falls back to OTHER_AMBIGUOUS/confidence 0).
    """
    client = _client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        content = None
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
            except Exception:
                # Some OpenRouter-routed models reject response_format; retry without it.
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )

            # OpenRouter's shared free-tier pool sometimes returns HTTP 200
            # with an empty/malformed body when the upstream provider fails
            # (rather than raising) - `response.choices` can be None or
            # empty, or `message.content` can be None. This was a real bug
            # found during development: an earlier version accessed
            # response.choices[0].message.content unguarded, which crashed
            # with an uncaught TypeError that bypassed the retry loop
            # entirely and silently reported a bad result as a real
            # classification (see docs/decision_log.md).
            choices = getattr(response, "choices", None)
            message = choices[0].message if choices else None
            content = getattr(message, "content", None) if message else None
            if not content:
                raise ValueError(
                    f"Empty or malformed response from the model (likely an upstream "
                    f"provider failure returned as HTTP 200): {response!r}"
                )

            parsed = _extract_json(content)
            validate(parsed)
            return parsed
        except Exception as e:  # noqa: BLE001 - deliberately broad, see retry loop
            last_error = e
            if content:
                # We got real (if invalid) model output - show it back to the
                # model and ask it to correct itself.
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your last response was invalid ({e}). Reply with ONLY a single "
                            "valid JSON object matching the requested schema, no other text."
                        ),
                    }
                )
            # else: the API call itself failed/returned empty - just retry
            # the same request fresh, nothing useful to show the model back.

    raise last_error  # type: ignore[misc]

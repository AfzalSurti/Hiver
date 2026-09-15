"""Regression tests for a real bug found running the full classifier batch:
OpenRouter's shared free-tier pool sometimes returns HTTP 200 with
choices=None or message.content=None when the upstream provider fails
instead of raising. The original code accessed
response.choices[0].message.content unguarded, which crashed with an
uncaught TypeError that escaped the retry loop entirely - 183/228 (80%) of
a real classifier batch run silently fell back to a default answer while
being reported as if the classifier had actually run. See
docs/decision_log.md for the full story.
"""
from types import SimpleNamespace

import pytest

from support_agent.llm_client import call_json


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if not self._responses:
            raise RuntimeError("FakeCompletions ran out of scripted responses")
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def make_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def make_broken_response_none_choices():
    return SimpleNamespace(choices=None)


def make_broken_response_empty_choices():
    return SimpleNamespace(choices=[])


def make_broken_response_none_content():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])


def _validate_ok(parsed: dict) -> None:
    if "intent" not in parsed:
        raise ValueError("missing intent")


@pytest.fixture
def patch_client(monkeypatch):
    def _patch(responses):
        fake = FakeClient(responses)
        monkeypatch.setattr("support_agent.llm_client._client", lambda: fake)
        return fake

    return _patch


def test_none_choices_does_not_crash_and_retries(patch_client):
    fake = patch_client(
        [
            make_broken_response_none_choices(),
            make_response('{"intent": "ACCOUNT_LOGIN_ACCESS"}'),
        ]
    )
    result = call_json("model", "sys", "user", _validate_ok, max_retries=2)
    assert result == {"intent": "ACCOUNT_LOGIN_ACCESS"}
    assert fake.chat.completions.calls == 2


def test_empty_choices_list_does_not_crash(patch_client):
    patch_client(
        [
            make_broken_response_empty_choices(),
            make_response('{"intent": "REFUND_REQUEST"}'),
        ]
    )
    result = call_json("model", "sys", "user", _validate_ok, max_retries=2)
    assert result == {"intent": "REFUND_REQUEST"}


def test_none_content_does_not_crash(patch_client):
    patch_client(
        [
            make_broken_response_none_content(),
            make_response('{"intent": "OTHER_AMBIGUOUS"}'),
        ]
    )
    result = call_json("model", "sys", "user", _validate_ok, max_retries=2)
    assert result == {"intent": "OTHER_AMBIGUOUS"}


def test_persistent_broken_responses_raise_after_retries_exhausted(patch_client):
    patch_client(
        [
            make_broken_response_none_choices(),
            make_broken_response_none_choices(),
            make_broken_response_none_choices(),
        ]
    )
    with pytest.raises(ValueError, match="Empty or malformed response"):
        call_json("model", "sys", "user", _validate_ok, max_retries=2)


def test_malformed_json_still_gets_corrective_retry(patch_client):
    patch_client(
        [
            make_response("not json at all"),
            make_response('{"intent": "TECHNICAL_PLAYBACK_ISSUE"}'),
        ]
    )
    result = call_json("model", "sys", "user", _validate_ok, max_retries=2)
    assert result == {"intent": "TECHNICAL_PLAYBACK_ISSUE"}


def test_valid_json_on_first_try_returns_immediately(patch_client):
    fake = patch_client([make_response('{"intent": "FEATURE_REQUEST_OR_INFO"}')])
    result = call_json("model", "sys", "user", _validate_ok, max_retries=2)
    assert result == {"intent": "FEATURE_REQUEST_OR_INFO"}
    assert fake.chat.completions.calls == 1

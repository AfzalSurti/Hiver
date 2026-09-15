"""Regression test for a real bug found during development: classify() and
generate_reply() originally caught ALL exceptions, including a missing API
key, and silently returned a fallback result. That meant a batch evaluation
script run with no OPENROUTER_API_KEY configured would "succeed" and write
an entire results file made of fallback values with no indication anything
was wrong - a direct violation of "never fabricate results." Missing
configuration must propagate as LLMConfigError so calling scripts fail
loudly instead.
"""
import os

import pytest

from support_agent.classifier import classify
from support_agent.generation import generate_reply
from support_agent.llm_client import LLMConfigError


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_classify_raises_llm_config_error_without_api_key():
    with pytest.raises(LLMConfigError):
        classify("I forgot my password")


def test_generate_reply_raises_llm_config_error_without_api_key():
    with pytest.raises(LLMConfigError):
        generate_reply("I forgot my password", "ACCOUNT_LOGIN_ACCESS", [])

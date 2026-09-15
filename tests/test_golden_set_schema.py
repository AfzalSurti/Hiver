import json

import pytest

from support_agent.config import GOLDEN_DIR
from support_agent.intents import INTENT_NAMES

VALID_ACTIONS = {"AUTO_HANDLE", "ESCALATE"}
VALID_REASONS = {
    "SENSITIVE_FINANCIAL",
    "ACCOUNT_SPECIFIC_INVESTIGATION",
    "SECURITY_CONCERN",
    "AMBIGUOUS_OR_INSUFFICIENT_INFO",
    "REPEATED_UNRESOLVED_ISSUE",
    "OUT_OF_SCOPE",
}

GOLDEN_PATH = GOLDEN_DIR / "golden_set.jsonl"


@pytest.fixture(scope="module")
def golden_records():
    if not GOLDEN_PATH.exists():
        pytest.skip(f"{GOLDEN_PATH} not built yet - run scripts/06_build_golden_set.py")
    return [json.loads(line) for line in GOLDEN_PATH.open(encoding="utf-8")]


def test_size_within_required_range(golden_records):
    assert 150 <= len(golden_records) <= 250


def test_no_duplicate_conversation_ids(golden_records):
    ids = [r["conversation_id"] for r in golden_records]
    assert len(ids) == len(set(ids))


def test_every_record_has_valid_intent(golden_records):
    for r in golden_records:
        assert r["intent"] in INTENT_NAMES, r["conversation_id"]


def test_every_record_has_valid_action(golden_records):
    for r in golden_records:
        assert r["expected_action"] in VALID_ACTIONS, r["conversation_id"]


def test_escalate_requires_valid_reason(golden_records):
    for r in golden_records:
        if r["expected_action"] == "ESCALATE":
            assert r["escalation_reason"] in VALID_REASONS, r["conversation_id"]


def test_auto_handle_has_no_reason(golden_records):
    for r in golden_records:
        if r["expected_action"] == "AUTO_HANDLE":
            assert r["escalation_reason"] is None, r["conversation_id"]


def test_every_record_has_a_customer_message(golden_records):
    for r in golden_records:
        assert r["customer_message"].strip(), r["conversation_id"]


def test_every_intent_has_at_least_one_example(golden_records):
    present = {r["intent"] for r in golden_records}
    missing = set(INTENT_NAMES) - present
    assert not missing, f"Intents with zero golden examples: {missing}"

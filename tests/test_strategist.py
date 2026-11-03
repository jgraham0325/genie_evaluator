import pytest

from genie_evaluator.patches import PatchError
from genie_evaluator.strategist import parse_proposal, propose_patches


def test_parse_proposal_strips_fences(sample_space):
    text = """```json
{"hypothesis": "status literals", "patches": [
  {"op": "update_column_description", "identifier": "sandbox.jg_banking_genie.account",
   "column_name": "status", "description": "Open or Closed"}
]}
```"""
    hypothesis, patches = parse_proposal(text, "descriptions", sample_space)
    assert hypothesis == "status literals"
    assert patches[0]["op"] == "update_column_description"


def test_propose_patches_uses_injected_llm(sample_space):
    failures = [
        {
            "question_id": "q1",
            "assessment": "BAD",
            "question": "How many open accounts?",
            "expected_sql": "SELECT 1",
            "actual_sql": "SELECT 2",
        }
    ]

    def fake_llm(prompt: str) -> str:
        assert "descriptions" in prompt
        return '{"hypothesis": "h", "patches": [{"op": "update_table_description", "identifier": "sandbox.jg_banking_genie.account", "description": "Accounts"}]}'

    hypothesis, patches = propose_patches(
        lever="descriptions",
        space=sample_space,
        failures=failures,
        propose=fake_llm,
    )
    assert hypothesis == "h"
    assert patches[0]["identifier"].endswith(".account")


def test_propose_requires_failures(sample_space):
    with pytest.raises(PatchError, match="no failing questions"):
        propose_patches(
            lever="descriptions",
            space=sample_space,
            failures=[{"question_id": "q1", "assessment": "GOOD"}],
            propose=lambda p: "{}",
        )

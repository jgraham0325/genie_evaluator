import pytest

from types import SimpleNamespace

from genie_evaluator.patches import PatchError
from genie_evaluator.strategist import parse_proposal, propose_patches, query_endpoint


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


def test_query_endpoint_uses_rest_invocations():
    calls = []

    class Client:
        def do(self, method, path, body=None):
            calls.append((method, path, body))
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    text = query_endpoint(SimpleNamespace(api_client=Client()), "databricks-meta-llama-3-3-70b-instruct", "hi")
    assert text == '{"ok": true}'
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/invocations")
    assert calls[0][2]["messages"][0]["content"] == "hi"


def test_propose_requires_failures(sample_space):
    with pytest.raises(PatchError, match="no failing questions"):
        propose_patches(
            lever="descriptions",
            space=sample_space,
            failures=[{"question_id": "q1", "assessment": "GOOD"}],
            propose=lambda p: "{}",
        )

import pytest

from genie_evaluator.patches import (
    PatchError,
    apply_config_patches,
    parse_levers,
    validate_patches,
)


def test_parse_levers_rejects_unknown():
    with pytest.raises(PatchError, match="unknown levers"):
        parse_levers('["descriptions","nope"]')


def test_apply_table_and_column_description(sample_space):
    patched = apply_config_patches(
        sample_space,
        [
            {
                "op": "update_table_description",
                "identifier": "sandbox.jg_banking_genie.account",
                "description": "Open and closed bank accounts.",
            },
            {
                "op": "update_column_description",
                "identifier": "sandbox.jg_banking_genie.account",
                "column_name": "status",
                "description": "Open or Closed.",
            },
        ],
    )
    account = next(
        t for t in patched["data_sources"]["tables"] if t["identifier"].endswith(".account")
    )
    assert account["description"] == ["Open and closed bank accounts."]
    status = next(c for c in account["column_configs"] if c["column_name"] == "status")
    assert status["description"] == ["Open or Closed."]


def test_leakage_firewall_rejects_expected_sql(sample_space):
    patches = [
        {
            "op": "update_instruction",
            "content": "Always run SELECT COUNT(*) AS open_account_count FROM sandbox.jg_banking_genie.account WHERE status = 'Open'",
        }
    ]
    with pytest.raises(PatchError, match="leakage"):
        validate_patches(patches, "instructions", sample_space)


def test_wrong_lever_op_rejected(sample_space):
    patches = [
        {
            "op": "update_instruction",
            "content": "Be careful with group names.",
        }
    ]
    with pytest.raises(PatchError, match="not allowed for lever"):
        validate_patches(patches, "descriptions", sample_space)


def test_add_join_and_instruction(sample_space):
    validate_patches(
        [
            {
                "op": "add_join_spec",
                "left": {"identifier": "sandbox.jg_banking_genie.account"},
                "right": {"identifier": "sandbox.jg_banking_genie.client_group"},
                "sql": ["account.group_id = client_group.group_id"],
                "instruction": ["Use for group rollups."],
            }
        ],
        "joins",
        sample_space,
    )
    patched = apply_config_patches(
        sample_space,
        [
            {
                "op": "update_instruction",
                "content": "Never guess a client group.",
            }
        ],
    )
    text = patched["instructions"]["text_instructions"][0]["content"][0]
    assert "Never guess a client group." in text
    assert "find_client" in text

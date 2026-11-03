from pathlib import Path

import pytest

from genie_evaluator.experiment_spec import (
    list_experiment_names,
    load_experiment,
    parse_experiment_names,
    parse_opt_mode,
    resolve_fanout,
)
from genie_evaluator.patches import PatchError, apply_config_patches, validate_patches


def test_parse_experiment_names():
    assert parse_experiment_names("") == []
    assert parse_experiment_names('["synonyms_account_status"]') == ["synonyms_account_status"]
    with pytest.raises(PatchError, match="invalid experiment names"):
        parse_experiment_names('["../secret"]')


def test_opt_mode_auto_ignores_experiment_files():
    root = Path(__file__).resolve().parents[1]
    assert parse_opt_mode("") == "auto"
    assert parse_opt_mode("manual") == "experiments"
    mode, fanout = resolve_fanout(
        opt_mode="auto",
        levers_raw='["joins"]',
        experiments_raw='["synonyms_account_status"]',
        root=root,
    )
    assert mode == "auto"
    assert fanout == ["joins"]
    assert "synonyms_account_status" in list_experiment_names(root)


def test_opt_mode_experiments_uses_folder_when_list_empty():
    root = Path(__file__).resolve().parents[1]
    mode, fanout = resolve_fanout(
        opt_mode="experiments",
        levers_raw='["joins"]',
        experiments_raw="",
        root=root,
    )
    assert mode == "experiments"
    assert "synonyms_account_status" in fanout
    assert "column_settings_party_name" in fanout
    assert "joins" not in fanout


def test_load_bundled_experiment(sample_space):
    root = Path(__file__).resolve().parents[1]
    spec = load_experiment(root, "synonyms_account_status", sample_space)
    assert spec["name"] == "synonyms_account_status"
    validate_patches(spec["patches"], "manual", sample_space)
    patched = apply_config_patches(sample_space, spec["patches"])
    account = next(
        t for t in patched["data_sources"]["tables"] if t["identifier"].endswith(".account")
    )
    status = next(c for c in account["column_configs"] if c["column_name"] == "status")
    assert "open/closed" in status["synonyms"]

    settings = load_experiment(root, "column_settings_party_name", sample_space)
    patched = apply_config_patches(sample_space, settings["patches"])
    party = next(
        t for t in patched["data_sources"]["tables"] if t["identifier"].endswith(".client_party")
    )
    name = next(c for c in party["column_configs"] if c["column_name"] == "party_name")
    assert name["enable_entity_matching"] is True
    assert "client name" in name["synonyms"]


def test_column_settings_and_exclude(sample_space):
    patches = [
        {
            "op": "update_column_settings",
            "identifier": "sandbox.jg_banking_genie.account",
            "column_name": "status",
            "enable_entity_matching": True,
            "enable_format_assistance": False,
            "exclude": False,
        }
    ]
    validate_patches(patches, "manual", sample_space)
    patched = apply_config_patches(sample_space, patches)
    account = next(
        t for t in patched["data_sources"]["tables"] if t["identifier"].endswith(".account")
    )
    status = next(c for c in account["column_configs"] if c["column_name"] == "status")
    assert status["enable_entity_matching"] is True
    assert status["enable_format_assistance"] is False
    assert status["exclude"] is False


def test_manual_can_mix_ops(sample_space):
    patches = [
        {
            "op": "update_column_synonyms",
            "identifier": "sandbox.jg_banking_genie.account",
            "column_name": "status",
            "synonyms": ["open/closed"],
        },
        {
            "op": "update_instruction",
            "content": "Prefer group-level answers unless the user names a person.",
        },
    ]
    validate_patches(patches, "manual", sample_space)
    patched = apply_config_patches(sample_space, patches)
    assert "Prefer group-level" in patched["instructions"]["text_instructions"][0]["content"][0]

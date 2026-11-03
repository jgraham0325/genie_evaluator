import pytest
from types import SimpleNamespace

from genie_evaluator.source_copies import (
    experiment_schema,
    retarget_identifiers,
    wrapper_identifier,
)


class FakeSpark:
    def __init__(self):
        self.statements: list[str] = []

    def sql(self, query: str):
        self.statements.append(query)


def test_experiment_schema_includes_run_and_lever():
    cat, schema = experiment_schema("main", "abcdef12deadbeef", "descriptions")
    assert cat == "main"
    assert schema == "opt_abcdef12_descriptions"
    other = experiment_schema("main", "abcdef12deadbeef", "joins")[1]
    assert other != schema


def test_retarget_skips_benchmarks(sample_space):
    mapping = {
        "sandbox.jg_banking_genie.account": "main.opt_abc_source_copies.account",
    }
    updated = retarget_identifiers(sample_space, mapping)
    tables = {t["identifier"] for t in updated["data_sources"]["tables"]}
    assert "main.opt_abc_source_copies.account" in tables
    join = updated["instructions"]["join_specs"][0]
    assert join["left"]["identifier"] == "main.opt_abc_source_copies.account"
    expected = sample_space["benchmarks"]["questions"][0]["answer"][0]["content"]
    assert updated["benchmarks"]["questions"][0]["answer"][0]["content"] == expected
    assert "sandbox.jg_banking_genie.account" in "".join(expected)


def test_apply_source_copies_creates_wrappers_and_tracks_cleanup(sample_space):
    from genie_evaluator.source_copies import apply_source_copies, drop_experiment_schema

    spark = FakeSpark()
    patches = [
        {
            "op": "clone_and_comment_source",
            "identifier": "sandbox.jg_banking_genie.account",
            "table_comment": "Bank accounts",
            "column_comments": {"status": "Open or Closed"},
        }
    ]
    updated, plan = apply_source_copies(
        spark,
        sample_space,
        patches,
        catalog="main",
        run_id="abcdef12deadbeef",
        lever="source_copies",
    )
    assert plan.full_schema == "main.opt_abcdef12_source_copies"
    assert wrapper_identifier("main", plan.schema_name, "account") in plan.wrappers
    assert any("CREATE OR REPLACE VIEW" in s for s in spark.statements)
    assert any("COMMENT ON TABLE" in s for s in spark.statements)
    assert any("COMMENT ON COLUMN" in s for s in spark.statements)
    assert "main.opt_abcdef12_source_copies.account" in {
        t["identifier"] for t in updated["data_sources"]["tables"]
    }
    drop_experiment_schema(spark, plan)
    assert any("DROP SCHEMA IF EXISTS" in s and "CASCADE" in s for s in spark.statements)


def test_failed_apply_still_returns_schema_for_cleanup(sample_space):
    from genie_evaluator.source_copies import apply_source_copies, drop_experiment_schema

    class BoomSpark(FakeSpark):
        def sql(self, query: str):
            super().sql(query)
            if "COMMENT ON COLUMN" in query:
                raise RuntimeError("boom")

    spark = BoomSpark()
    patches = [
        {
            "op": "clone_and_comment_source",
            "identifier": "sandbox.jg_banking_genie.account",
            "column_comments": {"status": "Open or Closed"},
        }
    ]
    with pytest.raises(RuntimeError, match="boom"):
        apply_source_copies(
            spark,
            sample_space,
            patches,
            catalog="main",
            run_id="abcdef12deadbeef",
            lever="source_copies",
        )
    cat, schema = experiment_schema("main", "abcdef12deadbeef", "source_copies")
    drop_experiment_schema(
        spark,
        SimpleNamespace(schema_catalog=cat, schema_name=schema),
    )
    assert any("DROP SCHEMA IF EXISTS" in s for s in spark.statements)

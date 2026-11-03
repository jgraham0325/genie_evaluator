"""Clone Genie sources as wrapper views in a throwaway schema. Never mutate originals.

Views keep the original table name as the last identifier segment so existing
SQL snippets (`account.status`) still qualify. Parallel experiments get their
own schema: `{catalog}.opt_{run8}_{lever}`.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from genie_evaluator.patches import normalize_space


class SparkSql(Protocol):
    def sql(self, query: str) -> Any: ...


_SAFE = re.compile(r"[^a-z0-9_]+")


def sanitize_token(value: str, *, max_len: int = 40) -> str:
    token = _SAFE.sub("_", value.lower()).strip("_")
    return (token or "x")[:max_len]


def parse_identifier(identifier: str) -> tuple[str, str, str]:
    parts = [p.strip("`") for p in identifier.split(".")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"expected catalog.schema.table identifier, got {identifier!r}")
    return parts[0], parts[1], parts[2]


def quote_id(catalog: str, schema: str, table: str) -> str:
    return f"`{catalog}`.`{schema}`.`{table}`"


def experiment_schema(catalog: str, run_id: str, lever: str) -> tuple[str, str]:
    """Return (catalog, schema) for one experiment's wrapper views."""
    return catalog, f"opt_{sanitize_token(run_id, max_len=8)}_{sanitize_token(lever)}"


def wrapper_identifier(catalog: str, schema: str, table: str) -> str:
    return f"{catalog}.{schema}.{table}"


@dataclass
class SourceCopyPlan:
    schema_catalog: str
    schema_name: str
    created_schema: str
    wrappers: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def full_schema(self) -> str:
        return f"{self.schema_catalog}.{self.schema_name}"


def retarget_identifiers(space: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Rewrite table/join identifiers. Benchmarks and example SQL stay on originals."""
    updated = copy.deepcopy(space)
    for table in (updated.get("data_sources") or {}).get("tables") or []:
        ident = table.get("identifier")
        if ident in mapping:
            table["identifier"] = mapping[ident]
    for view in (updated.get("data_sources") or {}).get("metric_views") or []:
        ident = view.get("identifier")
        if ident in mapping:
            view["identifier"] = mapping[ident]
    for join in (updated.get("instructions") or {}).get("join_specs") or []:
        for side in ("left", "right"):
            node = join.get(side) or {}
            ident = node.get("identifier")
            if ident in mapping:
                node["identifier"] = mapping[ident]
                join[side] = node
    return normalize_space(updated)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def apply_source_copies(
    spark: SparkSql,
    space: dict[str, Any],
    patches: list[dict[str, Any]],
    *,
    catalog: str,
    run_id: str,
    lever: str,
) -> tuple[dict[str, Any], SourceCopyPlan]:
    """Create wrapper views for clone_and_comment_source patches and retarget the space."""
    cat, schema = experiment_schema(catalog, run_id, lever)
    plan = SourceCopyPlan(schema_catalog=cat, schema_name=schema, created_schema=f"{cat}.{schema}")
    clone_patches = [p for p in patches if p.get("op") == "clone_and_comment_source"]
    if not clone_patches:
        return space, plan

    spark.sql(f"CREATE CATALOG IF NOT EXISTS `{cat}`")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{cat}`.`{schema}`")

    for patch in clone_patches:
        original = str(patch["identifier"])
        orig_cat, orig_schema, table = parse_identifier(original)
        wrapper = wrapper_identifier(cat, schema, table)
        spark.sql(
            f"CREATE OR REPLACE VIEW {quote_id(cat, schema, table)} "
            f"AS SELECT * FROM {quote_id(orig_cat, orig_schema, table)}"
        )
        plan.wrappers.append(wrapper)
        plan.mapping[original] = wrapper
        table_comment = patch.get("table_comment")
        if table_comment:
            spark.sql(
                f"COMMENT ON TABLE {quote_id(cat, schema, table)} IS {_sql_literal(str(table_comment))}"
            )
        for column, comment in (patch.get("column_comments") or {}).items():
            spark.sql(
                f"COMMENT ON COLUMN {quote_id(cat, schema, table)}.`{column}` "
                f"IS {_sql_literal(str(comment))}"
            )

    return retarget_identifiers(space, plan.mapping), plan


def drop_experiment_schema(spark: SparkSql, plan: SourceCopyPlan | None) -> None:
    if plan is None or not plan.schema_name:
        return
    spark.sql(f"DROP SCHEMA IF EXISTS `{plan.schema_catalog}`.`{plan.schema_name}` CASCADE")

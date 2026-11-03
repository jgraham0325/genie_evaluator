"""Delta history for optimization parent runs and per-clone attempts."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from genie_evaluator.delta_logging import ensure_tables
from genie_evaluator.optimizer import AttemptRecord

RUNS_TABLE = "genie_opt_runs"
ITERATIONS_TABLE = "genie_opt_iterations"

_RUNS_SCHEMA = """
    opt_run_id STRING NOT NULL,
    template_space_id STRING,
    catalog STRING,
    schema STRING,
    levers STRING,
    baseline_eval_run_id STRING,
    baseline_accuracy DOUBLE,
    mlflow_run_id STRING,
    triggered_by STRING,
    started_at TIMESTAMP
"""

_ITERATIONS_SCHEMA = """
    opt_run_id STRING NOT NULL,
    attempt INT,
    lever STRING,
    clone_space_id STRING,
    hypothesis STRING,
    patches STRING,
    eval_run_id STRING,
    accuracy DOUBLE,
    fixed_count INT,
    regressed_count INT,
    decision STRING,
    config_hash STRING,
    source_schema STRING,
    wrappers STRING,
    cleaned_up BOOLEAN,
    mlflow_run_id STRING
"""


def _qualify(catalog: str, schema: str, table: str) -> str:
    return f"`{catalog}`.`{schema}`.`{table}`"


def ensure_opt_tables(spark: Any, catalog: str, schema: str) -> None:
    logging.getLogger("pyspark.sql.connect.logging").setLevel(logging.ERROR)
    ensure_tables(spark, catalog, schema)
    spark.sql(f"CREATE TABLE IF NOT EXISTS {_qualify(catalog, schema, RUNS_TABLE)} ({_RUNS_SCHEMA})")
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {_qualify(catalog, schema, ITERATIONS_TABLE)} ({_ITERATIONS_SCHEMA})"
    )


def write_opt_run(spark: Any, catalog: str, schema: str, record: dict[str, Any]) -> None:
    row = dict(record)
    row.setdefault("started_at", datetime.now(timezone.utc))
    if isinstance(row.get("levers"), list):
        row["levers"] = json.dumps(row["levers"])
    spark.createDataFrame([row]).write.mode("append").saveAsTable(
        _qualify(catalog, schema, RUNS_TABLE)
    )


def write_iteration(
    spark: Any,
    catalog: str,
    schema: str,
    *,
    opt_run_id: str,
    attempt: AttemptRecord,
    cleaned_up: bool,
    mlflow_run_id: str = "",
) -> None:
    row = {
        "opt_run_id": opt_run_id,
        "attempt": attempt.attempt,
        "lever": attempt.lever,
        "clone_space_id": attempt.clone_space_id,
        "hypothesis": attempt.hypothesis,
        "patches": json.dumps(attempt.patches, default=str),
        "eval_run_id": attempt.result.eval_run_id,
        "accuracy": attempt.result.accuracy,
        "fixed_count": attempt.comparison.fixed_count,
        "regressed_count": attempt.comparison.regressed_count,
        "decision": attempt.decision,
        "config_hash": attempt.config_hash,
        "source_schema": attempt.source_schema,
        "wrappers": json.dumps(attempt.wrappers),
        "cleaned_up": cleaned_up,
        "mlflow_run_id": mlflow_run_id,
    }
    spark.createDataFrame([row]).write.mode("append").saveAsTable(
        _qualify(catalog, schema, ITERATIONS_TABLE)
    )


def mark_iterations_cleaned_up(
    spark: Any, catalog: str, schema: str, opt_run_id: str, clone_space_id: str
) -> None:
    table = _qualify(catalog, schema, ITERATIONS_TABLE)
    spark.sql(
        f"""
        UPDATE {table}
        SET cleaned_up = true
        WHERE opt_run_id = :opt_run_id AND clone_space_id = :clone_space_id
        """,
        args={"opt_run_id": opt_run_id, "clone_space_id": clone_space_id},
    )

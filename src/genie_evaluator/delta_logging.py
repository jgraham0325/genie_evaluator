"""Delta-backed history for evaluation runs.

Two append-only tables, created on first use:

- `{catalog}.{schema}.genie_eval_runs`    — one row per evaluation run.
- `{catalog}.{schema}.genie_eval_results` — one row per benchmark question
  per run, joined back to `genie_eval_runs` by `run_id`.

This gives a plain-SQL trend view across runs/branches without needing the
MLflow UI, and is also the source the regression gate reads the
`baseline_branch` accuracy from. No Lakebase, no Apps — Delta tables only.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

RUNS_TABLE = "genie_eval_runs"
RESULTS_TABLE = "genie_eval_results"

_RUNS_SCHEMA = """
    run_id STRING NOT NULL,
    space_id STRING,
    branch STRING,
    git_sha STRING,
    eval_run_id STRING,
    eval_run_status STRING,
    accuracy DOUBLE,
    num_correct INT,
    num_questions INT,
    num_done INT,
    num_needs_review INT,
    wall_clock_seconds DOUBLE,
    mlflow_run_id STRING,
    triggered_by STRING,
    started_at TIMESTAMP
"""

_RESULTS_SCHEMA = """
    run_id STRING NOT NULL,
    question_id STRING,
    question STRING,
    assessment STRING,
    assessment_reasons ARRAY<STRING>,
    expected_sql STRING,
    actual_sql STRING
"""


def _qualify(catalog: str, schema: str, table: str) -> str:
    return f"`{catalog}`.`{schema}`.`{table}`"


def ensure_tables(spark: Any, catalog: str, schema: str) -> None:
    """Create the catalog/schema (if needed) and both history tables."""
    # Spark Connect logs the serverless usage-policy token at WARNING on
    # every spark.sql; it is not actionable for this job.
    logging.getLogger("pyspark.sql.connect.logging").setLevel(logging.ERROR)
    if catalog:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
        spark.sql(f"USE CATALOG `{catalog}`")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{schema}`")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {_qualify(catalog, schema, RUNS_TABLE)} ({_RUNS_SCHEMA})")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {_qualify(catalog, schema, RESULTS_TABLE)} ({_RESULTS_SCHEMA})")


def write_run(spark: Any, catalog: str, schema: str, run_record: dict[str, Any]) -> None:
    """Append one row to `genie_eval_runs`."""
    record = dict(run_record)
    record.setdefault("started_at", datetime.now(timezone.utc))
    df = spark.createDataFrame([record])
    df.write.mode("append").saveAsTable(_qualify(catalog, schema, RUNS_TABLE))


def write_results(spark: Any, catalog: str, schema: str, run_id: str, rows: list[dict[str, Any]]) -> None:
    """Append one row per benchmark question to `genie_eval_results`."""
    if not rows:
        return
    records = [
        {
            "run_id": run_id,
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "assessment": row.get("assessment"),
            "assessment_reasons": row.get("assessment_reasons") or [],
            "expected_sql": row.get("expected_sql"),
            "actual_sql": row.get("actual_sql"),
        }
        for row in rows
    ]
    df = spark.createDataFrame(records)
    df.write.mode("append").saveAsTable(_qualify(catalog, schema, RESULTS_TABLE))


def latest_baseline_accuracy(spark: Any, catalog: str, schema: str, branch: str) -> float | None:
    """Return the most recent recorded accuracy for `branch`, or `None` if there isn't one yet."""
    table = _qualify(catalog, schema, RUNS_TABLE)
    rows = (
        spark.sql(
            f"SELECT accuracy FROM {table} WHERE branch = :branch "
            f"ORDER BY started_at DESC LIMIT 1",
            args={"branch": branch},
        )
        .collect()
    )
    if not rows:
        return None
    value = rows[0]["accuracy"]
    return float(value) if value is not None else None

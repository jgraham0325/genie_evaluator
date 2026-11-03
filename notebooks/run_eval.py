# Databricks notebook source
# MAGIC %md
# MAGIC # Genie space evaluation
# MAGIC
# MAGIC Runs the native Genie Eval-Run API against the bundled space, logs the
# MAGIC result to MLflow + Delta, then fails the task if accuracy regresses.

# COMMAND ----------

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

dbutils = cast(Any, globals().get("dbutils"))
spark = cast(Any, globals().get("spark"))

dbutils.widgets.text("bundle_root", "")
bundle_root = dbutils.widgets.get("bundle_root").strip()
src_path = str(Path(bundle_root) / "src") if bundle_root else str(Path.cwd().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import mlflow
from databricks.sdk import WorkspaceClient

from genie_evaluator.delta_logging import (
    ensure_tables,
    latest_baseline_accuracy,
    write_results,
    write_run,
)
from genie_evaluator.eval_runner import EvalRunError, run_eval
from genie_evaluator.mlflow_logging import log_eval_run
from genie_evaluator.regression_gate import check

# COMMAND ----------

dbutils.widgets.text("space_id", "")
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("experiment_id", "")
dbutils.widgets.text("git_sha", "")
dbutils.widgets.text("git_branch", "")
dbutils.widgets.text("triggered_by", "")
dbutils.widgets.text("min_accuracy", "0.0")
dbutils.widgets.text("max_allowed_regression", "0.02")
dbutils.widgets.text("baseline_branch", "main")

space_id = dbutils.widgets.get("space_id").strip()
catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
experiment_id = dbutils.widgets.get("experiment_id").strip()
git_sha = dbutils.widgets.get("git_sha").strip()
git_branch = dbutils.widgets.get("git_branch").strip()
triggered_by = dbutils.widgets.get("triggered_by").strip()
min_accuracy = float(dbutils.widgets.get("min_accuracy") or "0.0")
max_allowed_regression = float(dbutils.widgets.get("max_allowed_regression") or "0.02")
baseline_branch = dbutils.widgets.get("baseline_branch").strip() or "main"

if not space_id:
    raise RuntimeError("space_id is required")
if not catalog or not schema:
    raise RuntimeError("catalog and schema are required")
if not experiment_id:
    raise RuntimeError("experiment_id is required")

run_id = uuid.uuid4().hex
print(
    json.dumps(
        {
            "run_id": run_id,
            "space_id": space_id,
            "catalog": catalog,
            "schema": schema,
            "git_branch": git_branch,
            "git_sha": git_sha,
            "min_accuracy": min_accuracy,
            "max_allowed_regression": max_allowed_regression,
            "baseline_branch": baseline_branch,
        },
        indent=2,
    )
)

# COMMAND ----------

w = WorkspaceClient()
ensure_tables(spark, catalog, schema)

result = run_eval(w, space_id)
if not result.is_complete:
    raise EvalRunError(
        f"eval-run {result.eval_run_id} did not complete cleanly "
        f"(status={result.status}, done={result.num_done}/{result.num_questions}, "
        f"rows={len(result.rows)})"
    )

# COMMAND ----------

mlflow_run_id = log_eval_run(
    mlflow=mlflow,
    experiment_id=experiment_id,
    run_id=run_id,
    run_name=f"{git_branch or 'local'}-{run_id[:8]}",
    params={
        "space_id": space_id,
        "git_sha": git_sha,
        "git_branch": git_branch,
        "triggered_by": triggered_by,
        "catalog": catalog,
        "schema": schema,
        "min_accuracy": min_accuracy,
        "max_allowed_regression": max_allowed_regression,
        "baseline_branch": baseline_branch,
    },
    result=result,
)

write_run(
    spark,
    catalog,
    schema,
    {
        "run_id": run_id,
        "space_id": space_id,
        "branch": git_branch,
        "git_sha": git_sha,
        "eval_run_id": result.eval_run_id,
        "eval_run_status": result.status,
        "accuracy": result.accuracy,
        "num_correct": result.num_correct,
        "num_questions": result.num_questions,
        "num_done": result.num_done,
        "num_needs_review": result.num_needs_review,
        "wall_clock_seconds": result.wall_clock_seconds,
        "mlflow_run_id": mlflow_run_id,
        "triggered_by": triggered_by,
        "started_at": datetime.now(timezone.utc),
    },
)
write_results(spark, catalog, schema, run_id, result.rows)

# COMMAND ----------

baseline_accuracy = latest_baseline_accuracy(spark, catalog, schema, baseline_branch)
# The current run is already in the table. If this job itself is on the
# baseline branch, skip the row we just wrote so we compare against the
# previous baseline, not ourselves.
if git_branch == baseline_branch:
    previous = spark.sql(
        f"""
        SELECT accuracy FROM `{catalog}`.`{schema}`.`genie_eval_runs`
        WHERE branch = '{baseline_branch.replace("'", "''")}' AND run_id <> '{run_id}'
        ORDER BY started_at DESC LIMIT 1
        """
    ).collect()
    baseline_accuracy = float(previous[0]["accuracy"]) if previous else None

report = check(
    current_accuracy=result.accuracy,
    baseline_accuracy=baseline_accuracy,
    min_accuracy=min_accuracy,
    max_allowed_regression=max_allowed_regression,
)

summary = {
    "run_id": run_id,
    "mlflow_run_id": mlflow_run_id,
    "eval_run_id": result.eval_run_id,
    "accuracy": result.accuracy,
    "num_correct": result.num_correct,
    "num_questions": result.num_questions,
    "failures": result.failure_question_ids,
    "gate": report.reason,
}
print(json.dumps(summary, indent=2, default=str))
dbutils.notebook.exit(json.dumps(summary, default=str))

# Databricks notebook source
# MAGIC %md
# MAGIC # Genie optimize — baseline
# MAGIC
# MAGIC Evaluates the bundle-managed template space (read-only), logs the template
# MAGIC serialized_space once, then fans out clones: auto levers, or named
# MAGIC experiment files when opt_mode=experiments.

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

from genie_evaluator.delta_logging import write_results, write_run
from genie_evaluator.eval_runner import EvalRunError, run_eval
from genie_evaluator.mlflow_logging import log_eval_run
from genie_evaluator.opt_logging import ensure_opt_tables, write_opt_run
from genie_evaluator.experiment_spec import resolve_fanout
from genie_evaluator.space_io import default_parent_path, ensure_parent_path, get_space

# COMMAND ----------

dbutils.widgets.text("space_id", "")
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("experiment_id", "")
dbutils.widgets.text("git_sha", "")
dbutils.widgets.text("git_branch", "")
dbutils.widgets.text("triggered_by", "")
dbutils.widgets.text("levers", "")
dbutils.widgets.text("experiments", "")
dbutils.widgets.text("opt_mode", "auto")
dbutils.widgets.text("opt_parent_path", "")
dbutils.widgets.text("warehouse_id", "")

space_id = dbutils.widgets.get("space_id").strip()
catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
experiment_id = dbutils.widgets.get("experiment_id").strip()
git_sha = dbutils.widgets.get("git_sha").strip()
git_branch = dbutils.widgets.get("git_branch").strip()
triggered_by = dbutils.widgets.get("triggered_by").strip() or "optimize"
mode, fanout = resolve_fanout(
    opt_mode=dbutils.widgets.get("opt_mode"),
    levers_raw=dbutils.widgets.get("levers"),
    experiments_raw=dbutils.widgets.get("experiments"),
    root=Path(bundle_root or "."),
)
opt_parent_path = dbutils.widgets.get("opt_parent_path").strip()
warehouse_id = dbutils.widgets.get("warehouse_id").strip()

if not space_id:
    raise RuntimeError("space_id is required")
if not catalog or not schema:
    raise RuntimeError("catalog and schema are required")
if not experiment_id:
    raise RuntimeError("experiment_id is required")

opt_run_id = uuid.uuid4().hex
w = WorkspaceClient()
template = get_space(w, space_id)
if not warehouse_id:
    warehouse_id = template["warehouse_id"]
if not opt_parent_path:
    opt_parent_path = default_parent_path(w)
ensure_parent_path(w, opt_parent_path)
ensure_opt_tables(spark, catalog, schema)

print(
    json.dumps(
        {
            "opt_run_id": opt_run_id,
            "space_id": space_id,
            "mode": mode,
            "fanout": fanout,
            "opt_parent_path": opt_parent_path,
        },
        indent=2,
    )
)

# COMMAND ----------

result = run_eval(w, space_id)
if not result.is_complete:
    raise EvalRunError(
        f"baseline eval-run {result.eval_run_id} did not complete cleanly "
        f"(status={result.status}, done={result.num_done}/{result.num_questions})"
    )

mlflow_run_id = log_eval_run(
    mlflow=mlflow,
    experiment_id=experiment_id,
    run_id=opt_run_id,
    run_name=f"opt-baseline-{opt_run_id[:8]}",
    params={
        "space_id": space_id,
        "opt_run_id": opt_run_id,
        "role": "baseline",
        "mode": mode,
        "fanout": json.dumps(fanout),
        "git_sha": git_sha,
        "git_branch": git_branch,
        "triggered_by": triggered_by,
        "catalog": catalog,
        "schema": schema,
    },
    result=result,
    tags={"opt_role": "baseline", "opt_run_id": opt_run_id},
    artifacts={"template_serialized_space.json": template["serialized_space"]},
)

write_run(
    spark,
    catalog,
    schema,
    {
        "run_id": opt_run_id,
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
        "triggered_by": f"{triggered_by}:baseline",
        "started_at": datetime.now(timezone.utc),
    },
)
write_results(spark, catalog, schema, opt_run_id, result.rows)
write_opt_run(
    spark,
    catalog,
    schema,
    {
        "opt_run_id": opt_run_id,
        "template_space_id": space_id,
        "catalog": catalog,
        "schema": schema,
        "levers": fanout,
        "baseline_eval_run_id": result.eval_run_id,
        "baseline_accuracy": result.accuracy,
        "mlflow_run_id": mlflow_run_id,
        "triggered_by": triggered_by,
    },
)

dbutils.jobs.taskValues.set(key="opt_run_id", value=opt_run_id)
dbutils.jobs.taskValues.set(key="baseline_accuracy", value=str(result.accuracy))
dbutils.jobs.taskValues.set(key="mlflow_run_id", value=mlflow_run_id)
dbutils.jobs.taskValues.set(key="opt_parent_path", value=opt_parent_path)
dbutils.jobs.taskValues.set(key="warehouse_id", value=warehouse_id)
dbutils.jobs.taskValues.set(key="fanout", value=fanout)
dbutils.jobs.taskValues.set(key="mode", value=mode)

summary = {
    "opt_run_id": opt_run_id,
    "mlflow_run_id": mlflow_run_id,
    "accuracy": result.accuracy,
    "num_questions": result.num_questions,
    "failures": result.failure_question_ids,
    "mode": mode,
    "fanout": fanout,
}
print(json.dumps(summary, indent=2, default=str))
dbutils.notebook.exit(json.dumps(summary, default=str))

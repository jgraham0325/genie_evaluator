# Databricks notebook source
# MAGIC %md
# MAGIC # Genie optimize — one lever on a throwaway clone
# MAGIC
# MAGIC Clones the template space, applies one lever's patches, evaluates, logs
# MAGIC impact, then always deletes the clone and any wrapper-view schema.

# COMMAND ----------

from __future__ import annotations

import json
import sys
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
from genie_evaluator.eval_runner import run_eval
from genie_evaluator.mlflow_logging import log_eval_run
from genie_evaluator.opt_logging import (
    ensure_opt_tables,
    mark_iterations_cleaned_up,
    write_iteration,
)
from genie_evaluator.experiment_spec import load_experiment
from genie_evaluator.optimizer import run_experiment
from genie_evaluator.patches import PatchError
from genie_evaluator.source_copies import (
    SourceCopyPlan,
    drop_experiment_schema,
    experiment_schema,
)
from genie_evaluator.space_io import (
    create_space,
    get_space,
    trash_space,
    update_serialized_space,
)
from genie_evaluator.strategist import propose_patches, query_endpoint

# COMMAND ----------

dbutils.widgets.text("lever", "")
dbutils.widgets.text("mode", "auto")
dbutils.widgets.text("opt_run_id", "")
dbutils.widgets.text("space_id", "")
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("experiment_id", "")
dbutils.widgets.text("git_sha", "")
dbutils.widgets.text("git_branch", "")
dbutils.widgets.text("triggered_by", "")
dbutils.widgets.text("max_attempts", "2")
dbutils.widgets.text("target_accuracy", "0.9")
dbutils.widgets.text("llm_endpoint", "")
dbutils.widgets.text("opt_parent_path", "")
dbutils.widgets.text("warehouse_id", "")
dbutils.widgets.text("baseline_accuracy", "")
dbutils.widgets.text("mlflow_parent_run_id", "")

item = dbutils.widgets.get("lever").strip().strip('"')
mode = dbutils.widgets.get("mode").strip() or "auto"
opt_run_id = dbutils.widgets.get("opt_run_id").strip()
space_id = dbutils.widgets.get("space_id").strip()
catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
experiment_id = dbutils.widgets.get("experiment_id").strip()
git_sha = dbutils.widgets.get("git_sha").strip()
git_branch = dbutils.widgets.get("git_branch").strip()
triggered_by = dbutils.widgets.get("triggered_by").strip() or "optimize"
max_attempts = int(dbutils.widgets.get("max_attempts") or "2")
target_accuracy = float(dbutils.widgets.get("target_accuracy") or "0.9")
llm_endpoint = dbutils.widgets.get("llm_endpoint").strip()
opt_parent_path = dbutils.widgets.get("opt_parent_path").strip()
warehouse_id = dbutils.widgets.get("warehouse_id").strip()
baseline_accuracy_raw = dbutils.widgets.get("baseline_accuracy").strip()
mlflow_parent_run_id = dbutils.widgets.get("mlflow_parent_run_id").strip()

if not item:
    raise RuntimeError("lever/experiment name is required")
if not opt_run_id or not space_id:
    raise RuntimeError("opt_run_id and space_id are required")
if mode not in {"experiments", "manual"} and not llm_endpoint:
    raise RuntimeError("llm_endpoint is required unless experiments= is set")

w = WorkspaceClient()
ensure_opt_tables(spark, catalog, schema)

baseline_rows = [
    {
        "question_id": row["question_id"],
        "question": row["question"],
        "assessment": row["assessment"],
        "assessment_reasons": list(row["assessment_reasons"] or []),
        "expected_sql": row["expected_sql"],
        "actual_sql": row["actual_sql"],
    }
    for row in spark.sql(
        f"SELECT * FROM `{catalog}`.`{schema}`.`genie_eval_results` "
        f"WHERE run_id = :run_id",
        args={"run_id": opt_run_id},
    ).collect()
]
if not baseline_rows:
    raise RuntimeError(f"no baseline eval rows for opt_run_id={opt_run_id}")

baseline_accuracy = (
    float(baseline_accuracy_raw)
    if baseline_accuracy_raw
    else sum(1 for r in baseline_rows if str(r["assessment"]).upper() == "GOOD") / len(baseline_rows)
)

template = get_space(w, space_id)
if not warehouse_id:
    warehouse_id = template["warehouse_id"]

clone_space_id = ""
source_plan = None
error = ""
label = item
propose = None

try:
    if mode in {"experiments", "manual"}:
        spec = load_experiment(Path(bundle_root or "."), item, template["serialized_space"])
        label = spec["name"]
        max_attempts = 1

        def propose(space, failures, _spec=spec):
            return _spec["hypothesis"], _spec["patches"]
    else:

        def propose(space, failures):
            return propose_patches(
                lever=item,
                space=space,
                failures=failures,
                propose=lambda prompt: query_endpoint(w, llm_endpoint, prompt),
            )

    clone_space_id = create_space(
        w,
        title=f"[opt {opt_run_id[:8]}] {label}",
        warehouse_id=warehouse_id,
        serialized_space=template["serialized_space"],
        parent_path=opt_parent_path,
        description=f"Throwaway optimize clone for {label} run={opt_run_id}. Safe to delete.",
    )

    outcome = run_experiment(
        w=w,
        spark=spark,
        clone_space_id=clone_space_id,
        template_space=template["serialized_space"],
        baseline_rows=baseline_rows,
        baseline_accuracy=baseline_accuracy,
        lever=label,
        max_attempts=max_attempts,
        target_accuracy=target_accuracy,
        catalog=catalog,
        run_id=opt_run_id,
        propose=propose,
        run_eval=lambda sid: run_eval(w, sid),
        apply_space=lambda sid, space: update_serialized_space(w, sid, space),
    )
    source_plan = outcome.source_plan

    summaries = []
    for attempt in outcome.attempts:
        eval_run_id = f"{opt_run_id}-{label}-{attempt.attempt}"
        mlflow_run_id = log_eval_run(
            mlflow=mlflow,
            experiment_id=experiment_id,
            run_id=eval_run_id,
            run_name=f"opt-{label}-{attempt.attempt}-{opt_run_id[:8]}",
            params={
                "opt_run_id": opt_run_id,
                "mode": mode,
                "lever": label,
                "attempt": attempt.attempt,
                "decision": attempt.decision,
                "config_hash": attempt.config_hash,
                "clone_space_id": clone_space_id,
                "hypothesis": attempt.hypothesis[:200],
                "git_sha": git_sha,
                "git_branch": git_branch,
                "triggered_by": f"{triggered_by}:{label}",
            },
            result=attempt.result,
            tags={
                "opt_role": "experiment",
                "opt_run_id": opt_run_id,
                "mode": mode,
                "lever": label,
                "decision": attempt.decision,
                "clone_space_id": clone_space_id,
                "parent_mlflow_run_id": mlflow_parent_run_id,
            },
            metrics={
                "accuracy_delta": attempt.result.accuracy - baseline_accuracy,
                "fixed_count": float(attempt.comparison.fixed_count),
                "regressed_count": float(attempt.comparison.regressed_count),
            },
            artifacts={
                "patches.json": attempt.patches,
                "serialized_space.json": attempt.serialized_space,
            },
        )
        write_run(
            spark,
            catalog,
            schema,
            {
                "run_id": eval_run_id,
                "space_id": clone_space_id,
                "branch": git_branch,
                "git_sha": git_sha,
                "eval_run_id": attempt.result.eval_run_id,
                "eval_run_status": attempt.result.status,
                "accuracy": attempt.result.accuracy,
                "num_correct": attempt.result.num_correct,
                "num_questions": attempt.result.num_questions,
                "num_done": attempt.result.num_done,
                "num_needs_review": attempt.result.num_needs_review,
                "wall_clock_seconds": attempt.result.wall_clock_seconds,
                "mlflow_run_id": mlflow_run_id,
                "triggered_by": f"{triggered_by}:{label}",
                "started_at": datetime.now(timezone.utc),
            },
        )
        write_results(spark, catalog, schema, eval_run_id, attempt.result.rows)
        write_iteration(
            spark,
            catalog,
            schema,
            opt_run_id=opt_run_id,
            attempt=attempt,
            cleaned_up=False,
            mlflow_run_id=mlflow_run_id,
        )
        summaries.append(
            {
                "attempt": attempt.attempt,
                "decision": attempt.decision,
                "accuracy": attempt.result.accuracy,
                "hypothesis": attempt.hypothesis,
                "fixed": attempt.comparison.fixed,
                "regressed": attempt.comparison.regressed,
            }
        )
    print(json.dumps({"label": label, "mode": mode, "clone_space_id": clone_space_id, "attempts": summaries}, indent=2))
except PatchError as exc:
    error = str(exc)
    print(json.dumps({"label": label, "mode": mode, "skipped": True, "reason": error}))
except Exception as exc:
    error = str(exc)
    raise
finally:
    if clone_space_id:
        try:
            trash_space(w, clone_space_id)
        except Exception as exc:  # noqa: BLE001 — cleanup must not hide the experiment error
            print(json.dumps({"cleanup_space_error": str(exc), "clone_space_id": clone_space_id}))
    try:
        plan = source_plan
        if plan is None:
            cat, sch = experiment_schema(catalog, opt_run_id, label)
            plan = SourceCopyPlan(schema_catalog=cat, schema_name=sch, created_schema=f"{cat}.{sch}")
        drop_experiment_schema(spark, plan)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"cleanup_schema_error": str(exc)}))
    if clone_space_id:
        try:
            mark_iterations_cleaned_up(spark, catalog, schema, opt_run_id, clone_space_id)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"cleanup_flag_error": str(exc)}))

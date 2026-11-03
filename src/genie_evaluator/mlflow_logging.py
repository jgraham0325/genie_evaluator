"""Log one evaluation run to MLflow: params, headline metrics, and the full
per-question result table as a JSON artifact, so every run's "what changed
and did it help" is kept in one place (`resources.experiments.eval_experiment`).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from genie_evaluator.eval_runner import EvalRunResult


def log_json_artifacts(mlflow: Any, artifacts: dict[str, Any]) -> None:
    """Write JSON artifacts into the active MLflow run."""
    if not artifacts:
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for name, payload in artifacts.items():
            path = tmp / name
            path.write_text(json.dumps(payload, indent=2, default=str))
            mlflow.log_artifact(str(path))


def log_eval_run(
    *,
    mlflow: Any,
    experiment_id: str,
    run_id: str,
    params: dict[str, Any],
    result: EvalRunResult,
    run_name: str | None = None,
    tags: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
) -> str:
    """Open an MLflow run under `experiment_id`, log params/metrics/tags and
    the per-question rows as a JSON artifact, and return the MLflow run id.

    `mlflow` is passed in (rather than imported at module scope) so tests can
    inject a fake and so this stays usable outside a Databricks notebook.
    """
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as active_run:
        mlflow.log_param("run_id", run_id)
        for key, value in params.items():
            mlflow.log_param(key, value)

        mlflow.log_metric("accuracy", result.accuracy)
        mlflow.log_metric("num_correct", result.num_correct)
        mlflow.log_metric("num_questions", result.num_questions)
        mlflow.log_metric("num_done", result.num_done)
        mlflow.log_metric("num_needs_review", result.num_needs_review)
        mlflow.log_metric("wall_clock_seconds", result.wall_clock_seconds)
        for key, value in (metrics or {}).items():
            mlflow.log_metric(key, value)

        mlflow.set_tag("eval_run_id", result.eval_run_id)
        mlflow.set_tag("eval_run_status", result.status)
        mlflow.set_tag("genie_evaluator.run_id", run_id)
        for key, value in (tags or {}).items():
            mlflow.set_tag(key, value)

        extra = dict(artifacts or {})
        extra.setdefault("benchmark_results.json", result.rows)
        log_json_artifacts(mlflow, extra)

        return active_run.info.run_id

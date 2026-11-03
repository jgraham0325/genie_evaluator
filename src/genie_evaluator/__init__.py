"""genie_evaluator: measure whether Genie space changes improve benchmark accuracy.

Library code backing the `genie_eval_job` and `genie_opt_job` Databricks jobs
defined in this bundle. Kept deliberately small and unit-testable (see tests/):

- `eval_runner`: drives the native Genie Eval-Run API against a space's own
  `benchmarks.questions`.
- `delta_logging`: durable run/result history in Delta, used for trend
  queries and as the regression baseline source.
- `mlflow_logging`: one MLflow run per evaluation, with per-question detail
  as an artifact.
- `regression_gate`: pass/fail decision used to fail the job (and therefore
  CI) on a regression.
- `optimizer`: parallel-clone lever experiments (throwaway spaces + wrapper views).

The Eval-Run API usage here mirrors the pattern in
databricks-solutions/databricks-genie-workbench's
`genie_space_optimizer.optimization.eval_runner`, trimmed down to what this
much simpler evaluator needs.
"""

from genie_evaluator.eval_runner import EvalRunError, EvalRunResult, EvalRunTimeoutError, run_eval
from genie_evaluator.regression_gate import RegressionError, RegressionReport, check_regression

__all__ = [
    "EvalRunError",
    "EvalRunResult",
    "EvalRunTimeoutError",
    "run_eval",
    "RegressionError",
    "RegressionReport",
    "check_regression",
]

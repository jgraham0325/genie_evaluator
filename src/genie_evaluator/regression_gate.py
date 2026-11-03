"""Pass/fail decision for one evaluation run.

Two independent checks, either of which can fail the gate (and therefore the
job, and therefore CI):

1. **Absolute floor** — `current_accuracy < min_accuracy`.
2. **Regression vs. baseline** — if a `baseline_branch` accuracy has already
   been recorded in Delta, `baseline_accuracy - current_accuracy >
   max_allowed_regression` fails. With no recorded baseline yet (e.g. the
   very first run), this check is skipped rather than failing closed on a
   missing baseline.

Raising `RegressionError` is what the notebook task relies on to fail the
Databricks job run (and therefore `databricks bundle run`'s exit code).
"""

from __future__ import annotations

from dataclasses import dataclass


class RegressionError(RuntimeError):
    """Raised when an evaluation run fails the regression gate."""


@dataclass
class RegressionReport:
    passed: bool
    current_accuracy: float
    baseline_accuracy: float | None
    regression: float | None
    min_accuracy: float
    max_allowed_regression: float
    reason: str = ""


def check_regression(
    *,
    current_accuracy: float,
    baseline_accuracy: float | None,
    min_accuracy: float = 0.0,
    max_allowed_regression: float = 0.02,
) -> RegressionReport:
    """Evaluate the gate and return a report. Does not raise — see `check()`."""
    if current_accuracy < min_accuracy:
        return RegressionReport(
            passed=False,
            current_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            regression=None,
            min_accuracy=min_accuracy,
            max_allowed_regression=max_allowed_regression,
            reason=(
                f"accuracy {current_accuracy:.2%} is below the absolute floor "
                f"min_accuracy={min_accuracy:.2%}"
            ),
        )

    if baseline_accuracy is None:
        return RegressionReport(
            passed=True,
            current_accuracy=current_accuracy,
            baseline_accuracy=None,
            regression=None,
            min_accuracy=min_accuracy,
            max_allowed_regression=max_allowed_regression,
            reason="no recorded baseline yet — regression check skipped, floor check passed",
        )

    regression = baseline_accuracy - current_accuracy
    if regression > max_allowed_regression:
        return RegressionReport(
            passed=False,
            current_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            regression=regression,
            min_accuracy=min_accuracy,
            max_allowed_regression=max_allowed_regression,
            reason=(
                f"accuracy {current_accuracy:.2%} regresses {regression:.2%} vs. baseline "
                f"{baseline_accuracy:.2%}, exceeding max_allowed_regression={max_allowed_regression:.2%}"
            ),
        )

    return RegressionReport(
        passed=True,
        current_accuracy=current_accuracy,
        baseline_accuracy=baseline_accuracy,
        regression=regression,
        min_accuracy=min_accuracy,
        max_allowed_regression=max_allowed_regression,
        reason="within allowed regression tolerance",
    )


def check(
    *,
    current_accuracy: float,
    baseline_accuracy: float | None,
    min_accuracy: float = 0.0,
    max_allowed_regression: float = 0.02,
) -> RegressionReport:
    """Like `check_regression`, but raises `RegressionError` on failure.

    This is what `notebooks/run_eval.py` calls — the raise is what fails the
    Databricks task/job run (and therefore the CI step).
    """
    report = check_regression(
        current_accuracy=current_accuracy,
        baseline_accuracy=baseline_accuracy,
        min_accuracy=min_accuracy,
        max_allowed_regression=max_allowed_regression,
    )
    if not report.passed:
        raise RegressionError(report.reason)
    return report

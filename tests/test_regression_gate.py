import pytest

from genie_evaluator.regression_gate import RegressionError, check, check_regression


def test_passes_when_no_baseline_and_above_floor():
    report = check_regression(current_accuracy=0.8, baseline_accuracy=None, min_accuracy=0.5)
    assert report.passed
    assert "no recorded baseline" in report.reason


def test_fails_absolute_floor():
    report = check_regression(current_accuracy=0.4, baseline_accuracy=None, min_accuracy=0.5)
    assert not report.passed
    with pytest.raises(RegressionError, match="absolute floor"):
        check(current_accuracy=0.4, baseline_accuracy=None, min_accuracy=0.5)


def test_fails_when_regression_exceeds_tolerance():
    report = check_regression(
        current_accuracy=0.80,
        baseline_accuracy=0.90,
        max_allowed_regression=0.02,
    )
    assert not report.passed
    assert report.regression == pytest.approx(0.10)


def test_passes_small_regression_within_tolerance():
    report = check_regression(
        current_accuracy=0.89,
        baseline_accuracy=0.90,
        max_allowed_regression=0.02,
    )
    assert report.passed
    assert report.regression == pytest.approx(0.01)


def test_passes_improvement():
    report = check_regression(
        current_accuracy=0.95,
        baseline_accuracy=0.90,
        max_allowed_regression=0.02,
    )
    assert report.passed

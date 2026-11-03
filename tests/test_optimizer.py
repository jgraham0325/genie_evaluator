from genie_evaluator.eval_runner import EvalRunResult
from genie_evaluator.optimizer import accept_candidate, compare_rows, run_experiment


def _result(accuracy_rows, eval_run_id="e1"):
    num_correct = sum(1 for r in accuracy_rows if r["assessment"] == "GOOD")
    return EvalRunResult(
        eval_run_id=eval_run_id,
        status="DONE",
        num_correct=num_correct,
        num_done=len(accuracy_rows),
        num_needs_review=0,
        num_questions=len(accuracy_rows),
        rows=accuracy_rows,
    )


def test_accept_rejects_regression_even_if_accuracy_up():
    baseline = [
        {"question_id": "q1", "assessment": "GOOD"},
        {"question_id": "q2", "assessment": "BAD"},
        {"question_id": "q3", "assessment": "BAD"},
    ]
    candidate_rows = [
        {"question_id": "q1", "assessment": "BAD"},
        {"question_id": "q2", "assessment": "GOOD"},
        {"question_id": "q3", "assessment": "GOOD"},
    ]
    comparison = compare_rows(baseline, candidate_rows)
    assert comparison.regressed == ["q1"]
    candidate = _result(candidate_rows)
    assert candidate.accuracy > 1 / 3
    assert not accept_candidate(
        baseline_accuracy=1 / 3, candidate=candidate, comparison=comparison
    )


def test_accept_requires_strict_accuracy_gain():
    baseline = [
        {"question_id": "q1", "assessment": "GOOD"},
        {"question_id": "q2", "assessment": "BAD"},
    ]
    candidate_rows = [
        {"question_id": "q1", "assessment": "GOOD"},
        {"question_id": "q2", "assessment": "GOOD"},
    ]
    comparison = compare_rows(baseline, candidate_rows)
    assert accept_candidate(
        baseline_accuracy=0.5,
        candidate=_result(candidate_rows),
        comparison=comparison,
    )


def test_run_experiment_keep_then_stop(sample_space):
    baseline_rows = [
        {"question_id": "q1", "assessment": "BAD", "question": "n", "expected_sql": "select 1", "actual_sql": "select 2"},
        {"question_id": "q2", "assessment": "GOOD", "question": "y", "expected_sql": "select 1", "actual_sql": "select 1"},
    ]
    applied = []

    def propose(space, failures):
        return (
            "describe account",
            [
                {
                    "op": "update_table_description",
                    "identifier": "sandbox.jg_banking_genie.account",
                    "description": "Bank accounts at the group level.",
                }
            ],
        )

    def run_eval(space_id):
        return _result(
            [
                {"question_id": "q1", "assessment": "GOOD"},
                {"question_id": "q2", "assessment": "GOOD"},
            ]
        )

    outcome = run_experiment(
        w=None,
        spark=None,
        clone_space_id="clone-1",
        template_space=sample_space,
        baseline_rows=baseline_rows,
        baseline_accuracy=0.5,
        lever="descriptions",
        max_attempts=2,
        target_accuracy=0.9,
        catalog="main",
        run_id="abcdef12",
        propose=propose,
        run_eval=run_eval,
        apply_space=lambda sid, space: applied.append(space),
    )
    assert outcome.attempts[0].decision == "keep"
    assert len(outcome.attempts) == 1  # hit target
    assert applied
    assert applied[0]["data_sources"]["tables"][0]["description"] == [
        "Bank accounts at the group level."
    ]


def test_run_experiment_rollback_restores_template(sample_space):
    baseline_rows = [
        {"question_id": "q1", "assessment": "BAD"},
        {"question_id": "q2", "assessment": "GOOD"},
    ]
    applied = []

    def propose(space, failures):
        return (
            "noop",
            [
                {
                    "op": "update_table_description",
                    "identifier": "sandbox.jg_banking_genie.account",
                    "description": "Changed.",
                }
            ],
        )

    def run_eval(space_id):
        return _result(baseline_rows, eval_run_id="same")

    outcome = run_experiment(
        w=None,
        spark=None,
        clone_space_id="clone-1",
        template_space=sample_space,
        baseline_rows=baseline_rows,
        baseline_accuracy=0.5,
        lever="descriptions",
        max_attempts=3,
        target_accuracy=0.9,
        catalog="main",
        run_id="abcdef12",
        propose=propose,
        run_eval=run_eval,
        apply_space=lambda sid, space: applied.append(id(space)),
    )
    assert outcome.attempts[0].decision == "rollback"
    assert len(outcome.attempts) == 1
    assert len(applied) == 2  # candidate then restore

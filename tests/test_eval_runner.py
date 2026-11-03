from types import SimpleNamespace

from genie_evaluator.eval_runner import EvalRunResult, EvalRunTimeoutError, run_eval
import pytest


class FakeGenie:
    def __init__(self, *, statuses, details):
        self.statuses = list(statuses)
        self.details = details
        self.created = False
        self.polls = 0

    def genie_create_eval_run(self, space_id, benchmark_question_ids=None):
        self.created = True
        self.space_id = space_id
        self.benchmark_question_ids = benchmark_question_ids
        return SimpleNamespace(eval_run_id="eval-1")

    def genie_get_eval_run(self, space_id, eval_run_id):
        self.polls += 1
        status = self.statuses.pop(0)
        return SimpleNamespace(
            eval_run_status=status,
            num_correct=1,
            num_done=2,
            num_needs_review=0,
            num_questions=2,
        )

    def genie_list_eval_results(self, space_id, eval_run_id, page_size=100, page_token=None):
        summaries = [
            SimpleNamespace(
                result_id=row["result_id"],
                question=row["question"],
                benchmark_question_id=row["question_id"],
            )
            for row in self.details
        ]
        return SimpleNamespace(eval_results=summaries, next_page_token=None)

    def genie_get_eval_result_details(self, space_id, eval_run_id, result_id):
        row = next(r for r in self.details if r["result_id"] == result_id)
        return SimpleNamespace(
            benchmark_question_id=row["question_id"],
            assessment=row["assessment"],
            assessment_reasons=row["assessment_reasons"],
            expected_response=[SimpleNamespace(response=row["expected_sql"])],
            actual_response=[SimpleNamespace(response=row["actual_sql"])],
            result_id=result_id,
        )


DETAILS = [
    {
        "result_id": "r1",
        "question_id": "q1",
        "question": "How many trips?",
        "assessment": "GOOD",
        "assessment_reasons": [],
        "expected_sql": "SELECT COUNT(*) FROM trips",
        "actual_sql": "SELECT COUNT(*) FROM trips",
    },
    {
        "result_id": "r2",
        "question_id": "q2",
        "question": "Average fare?",
        "assessment": "BAD",
        "assessment_reasons": ["SQL_MISMATCH"],
        "expected_sql": "SELECT AVG(fare) FROM trips",
        "actual_sql": "SELECT SUM(fare) FROM trips",
    },
]


def test_run_eval_polls_until_done_and_maps_rows():
    genie = FakeGenie(statuses=["RUNNING", "DONE"], details=DETAILS)
    sleeps: list[float] = []
    clock_values = iter([0.0, 1.0, 2.0, 3.0, 4.0])

    result = run_eval(
        SimpleNamespace(genie=genie),
        "space-1",
        poll_interval_seconds=0.5,
        timeout_seconds=30,
        clock=lambda: next(clock_values),
        sleep=sleeps.append,
    )

    assert genie.created
    assert genie.benchmark_question_ids is None
    assert result.eval_run_id == "eval-1"
    assert result.status == "DONE"
    assert result.is_complete
    assert result.accuracy == 0.5
    assert result.failure_question_ids == ["q2"]
    assert result.rows[0]["actual_sql"] == "SELECT COUNT(*) FROM trips"
    assert result.rows[1]["assessment_reasons"] == ["SQL_MISMATCH"]
    assert sleeps == [0.5]


def test_run_eval_times_out():
    genie = FakeGenie(statuses=["RUNNING", "RUNNING", "RUNNING"], details=DETAILS)
    clock = iter([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(EvalRunTimeoutError):
        run_eval(
            SimpleNamespace(genie=genie),
            "space-1",
            poll_interval_seconds=1,
            timeout_seconds=1.5,
            clock=lambda: next(clock),
            sleep=lambda _: None,
        )


def test_incomplete_result_is_fail_closed():
    result = EvalRunResult(
        eval_run_id="x",
        status="EVALUATION_FAILED",
        num_correct=0,
        num_done=0,
        num_needs_review=0,
        num_questions=3,
        rows=[],
    )
    assert not result.is_complete
    assert result.accuracy == 0.0

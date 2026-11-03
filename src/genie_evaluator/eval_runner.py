"""Thin wrapper over the native Genie Eval-Run (Benchmark) API.

The Databricks SDK's `WorkspaceClient.genie` exposes a server-side benchmark
runner that evaluates every question stored in a Genie space's own
`serialized_space.benchmarks.questions` block:

    w.genie.genie_create_eval_run(space_id, benchmark_question_ids=None)
    w.genie.genie_get_eval_run(space_id, eval_run_id)              # poll
    w.genie.genie_list_eval_results(space_id, eval_run_id, ...)    # paginated
    w.genie.genie_get_eval_result_details(space_id, eval_run_id, result_id)

`benchmark_question_ids=None` evaluates every benchmark question on the
space. Each result is graded server-side as GOOD / BAD / NEEDS_REVIEW by
actually executing the expected and actual SQL and diffing the results, so
this module does not reimplement any SQL-equivalence logic itself.

This mirrors the pattern in databricks-solutions/databricks-genie-workbench's
`genie_space_optimizer.optimization.eval_runner.OfficialBenchmarkRunner`,
trimmed to what this evaluator needs (no lever-loop / lift-report machinery).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_PAGE_SIZE = 100

_SUCCESS_STATUS = "DONE"
_TERMINAL_STATUSES = frozenset(
    {"DONE", "EVALUATION_CANCELLED", "EVALUATION_FAILED", "EVALUATION_TIMEOUT"}
)


class EvalRunError(RuntimeError):
    """Raised when an eval-run cannot be completed at all (no eval_run_id, etc.)."""


class EvalRunTimeoutError(EvalRunError):
    """Raised when an eval-run does not reach a terminal status before the timeout."""

    def __init__(self, eval_run_id: str, last_status: str, timeout_seconds: float):
        super().__init__(
            f"eval-run {eval_run_id!r} did not finish within {timeout_seconds:.0f}s "
            f"(last status={last_status or 'UNKNOWN'})"
        )
        self.eval_run_id = eval_run_id
        self.last_status = last_status
        self.timeout_seconds = timeout_seconds


@dataclass
class EvalRunResult:
    """Run-level summary of a single Genie Eval-Run, plus mapped per-question rows."""

    eval_run_id: str
    status: str
    num_correct: int
    num_done: int
    num_needs_review: int
    num_questions: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    wall_clock_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        """Headline accuracy as a 0-1 fraction (`num_correct / num_questions`)."""
        if self.num_questions <= 0:
            return 0.0
        return self.num_correct / self.num_questions

    @property
    def succeeded(self) -> bool:
        return self.status == _SUCCESS_STATUS

    @property
    def is_complete(self) -> bool:
        """True only for a DONE run whose collected rows fully cover it.

        Fail-closed: a non-DONE terminal status, a partial run
        (`num_done < num_questions`), an empty benchmark set, or a short
        collected-row set never reads as a clean success.
        """
        return (
            self.status == _SUCCESS_STATUS
            and self.num_questions > 0
            and self.num_done >= self.num_questions
            and len(self.rows) >= self.num_questions
        )

    @property
    def failure_question_ids(self) -> list[str]:
        return [
            str(row.get("question_id") or "")
            for row in self.rows
            if str(row.get("assessment", "")).upper() != "GOOD" and row.get("question_id")
        ]


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _status_str(status: Any) -> str:
    return str(_attr(status, "value", status) or "").upper()


def _enum_value(item: Any) -> str:
    return str(_attr(item, "value", item) or "")


def _first_response_text(responses: Any) -> str:
    if not responses:
        return ""
    first = responses[0]
    return str(_attr(first, "response", "") or "")


def _map_result_to_row(summary: Any, detail: Any) -> dict[str, Any]:
    """Map one (summary, detail) pair from the Eval-Run API into a flat row dict."""
    question_id = str(
        _attr(detail, "benchmark_question_id", "")
        or _attr(summary, "benchmark_question_id", "")
        or ""
    )
    assessment = _enum_value(_attr(detail, "assessment")).upper() or "NEEDS_REVIEW"
    reasons = [_enum_value(r) for r in (_attr(detail, "assessment_reasons") or [])]
    return {
        "question_id": question_id,
        "question": str(_attr(summary, "question", "") or ""),
        "assessment": assessment,
        "assessment_reasons": reasons,
        "expected_sql": _first_response_text(_attr(detail, "expected_response")),
        "actual_sql": _first_response_text(_attr(detail, "actual_response")),
        "result_id": str(_attr(detail, "result_id", "") or _attr(summary, "result_id", "") or ""),
    }


def _call_genie(w: Any, method_names: tuple[str, ...], *args: Any, rest: tuple[str, str] | None = None, **kwargs: Any) -> Any:
    """Prefer SDK methods (name varies by SDK version); fall back to REST."""
    genie = getattr(w, "genie", None)
    for name in method_names:
        fn = getattr(genie, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    if rest is None:
        raise EvalRunError(f"Genie client has none of {method_names} and no REST fallback")
    http_method, path = rest
    body = {k: v for k, v in kwargs.items() if v is not None}
    query = None
    if http_method == "GET":
        query = {k: v for k, v in body.items() if v is not None}
        body = None
    return w.api_client.do(http_method, path, query=query, body=body or None)


def run_eval(
    w: Any,
    space_id: str,
    *,
    benchmark_question_ids: list[str] | None = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    page_size: int = DEFAULT_PAGE_SIZE,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> EvalRunResult:
    """Create a Genie Eval-Run against `space_id`'s benchmark questions, wait for it
    to finish, and return a mapped, flat :class:`EvalRunResult`.

    `benchmark_question_ids=None` (the default) evaluates every benchmark
    question stored on the space.
    """
    start = clock()

    def emit(event: str, fields: dict[str, Any]) -> None:
        if progress is not None:
            try:
                progress(event, fields)
            except Exception:  # pragma: no cover - progress must never break eval
                logger.debug("progress callback failed for %s", event, exc_info=True)

    emit(
        "eval_run_create",
        {"space_id": space_id, "question_count": len(benchmark_question_ids) if benchmark_question_ids else "all"},
    )
    created = _call_genie(
        w,
        ("genie_create_eval_run", "create_eval_run"),
        space_id,
        rest=("POST", f"/api/2.0/genie/spaces/{space_id}/eval-runs"),
        benchmark_question_ids=benchmark_question_ids,
    )
    eval_run_id = str(_attr(created, "eval_run_id", "") or "")
    if not eval_run_id:
        raise EvalRunError(f"genie_create_eval_run returned no eval_run_id for space {space_id!r}")

    summary = _poll(w, space_id, eval_run_id, poll_interval_seconds, timeout_seconds, clock, sleep, emit)
    status = _status_str(getattr(summary, "eval_run_status", None))

    rows: list[dict[str, Any]] = []
    if status == _SUCCESS_STATUS:
        rows = _collect_rows(w, space_id, eval_run_id, page_size)
    else:
        logger.warning("eval-run %s reached non-success terminal status %s", eval_run_id, status)

    result = EvalRunResult(
        eval_run_id=eval_run_id,
        status=status,
        num_correct=int(_attr(summary, "num_correct", 0) or 0),
        num_done=int(_attr(summary, "num_done", 0) or 0),
        num_needs_review=int(_attr(summary, "num_needs_review", 0) or 0),
        num_questions=int(_attr(summary, "num_questions", 0) or 0),
        rows=rows,
        wall_clock_seconds=clock() - start,
    )
    emit(
        "eval_run_done",
        {
            "eval_run_id": eval_run_id,
            "status": status,
            "accuracy": result.accuracy,
            "num_correct": result.num_correct,
            "num_questions": result.num_questions,
        },
    )
    return result


def _poll(
    w: Any,
    space_id: str,
    eval_run_id: str,
    poll_interval_seconds: float,
    timeout_seconds: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    emit: Callable[[str, dict[str, Any]], None],
) -> Any:
    deadline = clock() + timeout_seconds
    last_status = ""
    while True:
        summary = _call_genie(
            w,
            ("genie_get_eval_run", "get_eval_run"),
            space_id,
            eval_run_id,
            rest=("GET", f"/api/2.0/genie/spaces/{space_id}/eval-runs/{eval_run_id}"),
        )
        last_status = _status_str(_attr(summary, "eval_run_status"))
        emit(
            "eval_run_poll",
            {
                "eval_run_id": eval_run_id,
                "status": last_status,
                "num_done": int(_attr(summary, "num_done", 0) or 0),
                "num_questions": int(_attr(summary, "num_questions", 0) or 0),
            },
        )
        if last_status in _TERMINAL_STATUSES:
            return summary
        now = clock()
        if now >= deadline:
            raise EvalRunTimeoutError(eval_run_id, last_status, timeout_seconds)
        sleep(min(poll_interval_seconds, max(0.0, deadline - now)))


def _collect_rows(w: Any, space_id: str, eval_run_id: str, page_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        page = _call_genie(
            w,
            ("genie_list_eval_results", "list_eval_results"),
            space_id,
            eval_run_id,
            rest=("GET", f"/api/2.0/genie/spaces/{space_id}/eval-runs/{eval_run_id}/results"),
            page_size=page_size,
            page_token=page_token,
        )
        for summary in _attr(page, "eval_results") or []:
            result_id = _attr(summary, "result_id")
            detail = _call_genie(
                w,
                ("genie_get_eval_result_details", "get_eval_result_details"),
                space_id,
                eval_run_id,
                result_id,
                rest=(
                    "GET",
                    f"/api/2.0/genie/spaces/{space_id}/eval-runs/{eval_run_id}/results/{result_id}",
                ),
            )
            rows.append(_map_result_to_row(summary, detail))
        page_token = _attr(page, "next_page_token")
        if not page_token:
            break
    return rows

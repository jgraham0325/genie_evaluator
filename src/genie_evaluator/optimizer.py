"""Per-clone experiment loop: patch, eval, accept or keep baseline comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from genie_evaluator.eval_runner import EvalRunResult
from genie_evaluator.patches import apply_config_patches
from genie_evaluator.source_copies import SourceCopyPlan, SparkSql, apply_source_copies
from genie_evaluator.space_io import config_hash

RunEval = Callable[[str], EvalRunResult]
Propose = Callable[[dict[str, Any], list[dict[str, Any]]], tuple[str, list[dict[str, Any]]]]
ApplySpace = Callable[[str, dict[str, Any]], None]


@dataclass
class Comparison:
    fixed: list[str]
    regressed: list[str]
    unchanged_bad: list[str]
    unchanged_good: list[str]
    excluded: list[str]

    @property
    def regressed_count(self) -> int:
        return len(self.regressed)

    @property
    def fixed_count(self) -> int:
        return len(self.fixed)


def compare_rows(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> Comparison:
    base = {str(r.get("question_id")): str(r.get("assessment") or "").upper() for r in baseline if r.get("question_id")}
    cand = {str(r.get("question_id")): str(r.get("assessment") or "").upper() for r in candidate if r.get("question_id")}
    fixed: list[str] = []
    regressed: list[str] = []
    unchanged_bad: list[str] = []
    unchanged_good: list[str] = []
    excluded: list[str] = []
    for qid, base_status in base.items():
        if qid not in cand:
            excluded.append(qid)
            continue
        cand_status = cand[qid]
        base_good = base_status == "GOOD"
        cand_good = cand_status == "GOOD"
        if not base_good and cand_good:
            fixed.append(qid)
        elif base_good and not cand_good:
            regressed.append(qid)
        elif base_good and cand_good:
            unchanged_good.append(qid)
        else:
            unchanged_bad.append(qid)
    return Comparison(
        fixed=fixed,
        regressed=regressed,
        unchanged_bad=unchanged_bad,
        unchanged_good=unchanged_good,
        excluded=excluded,
    )


def accept_candidate(*, baseline_accuracy: float, candidate: EvalRunResult, comparison: Comparison) -> bool:
    if not candidate.is_complete:
        return False
    if candidate.accuracy <= baseline_accuracy:
        return False
    if comparison.regressed_count:
        return False
    return True


@dataclass
class AttemptRecord:
    attempt: int
    lever: str
    hypothesis: str
    patches: list[dict[str, Any]]
    result: EvalRunResult
    comparison: Comparison
    decision: str
    config_hash: str
    serialized_space: dict[str, Any]
    clone_space_id: str
    source_schema: str = ""
    wrappers: list[str] = field(default_factory=list)


@dataclass
class ExperimentOutcome:
    lever: str
    clone_space_id: str
    attempts: list[AttemptRecord]
    source_plan: SourceCopyPlan | None = None
    error: str = ""


def run_experiment(
    *,
    w: Any,
    spark: SparkSql | None,
    clone_space_id: str,
    template_space: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    baseline_accuracy: float,
    lever: str,
    max_attempts: int,
    target_accuracy: float,
    catalog: str,
    run_id: str,
    propose: Propose,
    run_eval: RunEval,
    apply_space: ApplySpace,
) -> ExperimentOutcome:
    """Run up to max_attempts on one clone. Caller creates and trashes the clone."""
    current_space = template_space
    source_plan: SourceCopyPlan | None = None
    attempts: list[AttemptRecord] = []
    best_accuracy = baseline_accuracy
    failure_rows = baseline_rows

    for attempt_no in range(1, max(1, max_attempts) + 1):
        hypothesis, patches = propose(current_space, failure_rows)
        patched = apply_config_patches(current_space, patches)
        needs_copy = any(p.get("op") == "clone_and_comment_source" for p in patches)
        if needs_copy:
            if spark is None:
                raise RuntimeError("clone_and_comment_source requires Spark to create wrapper views")
            patched, source_plan = apply_source_copies(
                spark,
                patched,
                patches,
                catalog=catalog,
                run_id=run_id,
                lever=lever,
            )
        apply_space(clone_space_id, patched)
        result = run_eval(clone_space_id)
        comparison = compare_rows(baseline_rows, result.rows)
        keep = accept_candidate(
            baseline_accuracy=best_accuracy,
            candidate=result,
            comparison=comparison,
        )
        decision = "keep" if keep else "rollback"
        attempts.append(
            AttemptRecord(
                attempt=attempt_no,
                lever=lever,
                hypothesis=hypothesis,
                patches=patches,
                result=result,
                comparison=comparison,
                decision=decision,
                config_hash=config_hash(patched),
                serialized_space=patched,
                clone_space_id=clone_space_id,
                source_schema=source_plan.full_schema if source_plan else "",
                wrappers=list(source_plan.wrappers) if source_plan else [],
            )
        )
        if keep:
            current_space = patched
            best_accuracy = result.accuracy
            failure_rows = result.rows
            if result.accuracy >= target_accuracy:
                break
        else:
            apply_space(clone_space_id, current_space)
            break

    return ExperimentOutcome(
        lever=lever,
        clone_space_id=clone_space_id,
        attempts=attempts,
        source_plan=source_plan,
    )

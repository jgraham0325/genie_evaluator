"""Propose one bounded patch set for a single lever from benchmark failures."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from genie_evaluator.patches import LEVER_OPS, PatchError, validate_patches

ProposeFn = Callable[[str], str]

SYSTEM_PROMPT = """You optimize a Databricks Genie Agent.
Return ONLY a JSON object: {"hypothesis": "...", "patches": [ ... ]}.
Each patch is one of the allowed ops for the given lever.
Do not copy benchmark questions or expected SQL into descriptions, instructions, examples, or snippets.
Do not invent tables or columns that are not in the space summary.
Keep the patch set small (1-4 patches) targeting the failure cluster.
Text fields may be strings (not arrays).
"""

LEVER_HINTS = {
    "descriptions": "Use update_table_description, update_column_description, update_column_synonyms, update_column_settings, or update_column_config. Column settings: enable_format_assistance, enable_entity_matching, get_example_values, build_value_dictionary, exclude.",
    "joins": "Use add_join_spec or update_join_spec. add_join_spec needs left.identifier, right.identifier, sql (join condition), optional instruction. update_join_spec needs id of an existing join.",
    "instructions": "Use update_instruction with content (appended to the existing block) or add_sample_question.",
    "sql_snippets": "Use add_sql_snippet (kind: filters|measures|expressions, display_name, sql, optional synonyms) or add_example_sql (question, sql, optional usage_guidance). Qualify columns with table name.",
    "source_copies": "Use clone_and_comment_source with identifier, optional table_comment, optional column_comments map. Originals are not modified; wrappers are created for you.",
}


def space_summary(space: dict[str, Any]) -> dict[str, Any]:
    tables = []
    for table in (space.get("data_sources") or {}).get("tables") or []:
        cols = [c.get("column_name") for c in (table.get("column_configs") or [])]
        tables.append(
            {
                "identifier": table.get("identifier"),
                "description": table.get("description"),
                "columns": cols,
            }
        )
    joins = []
    for join in (space.get("instructions") or {}).get("join_specs") or []:
        joins.append(
            {
                "id": join.get("id"),
                "left": (join.get("left") or {}).get("identifier"),
                "right": (join.get("right") or {}).get("identifier"),
                "sql": join.get("sql"),
                "instruction": join.get("instruction"),
            }
        )
    instructions = (space.get("instructions") or {}).get("text_instructions") or []
    instruction_preview = ""
    if instructions:
        content = instructions[0].get("content") or []
        instruction_preview = "".join(content)[:1500]
    snippets = space.get("instructions") or {}
    sql_snippets = snippets.get("sql_snippets") or {}
    return {
        "tables": tables,
        "joins": joins,
        "instruction_preview": instruction_preview,
        "snippet_filters": [s.get("display_name") for s in sql_snippets.get("filters") or []],
        "snippet_measures": [s.get("display_name") for s in sql_snippets.get("measures") or []],
        "examples": [
            {"question": ex.get("question"), "id": ex.get("id")}
            for ex in (snippets.get("example_question_sqls") or [])[:12]
        ],
    }


def failure_brief(rows: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    brief = []
    for row in rows:
        assessment = str(row.get("assessment") or "").upper()
        if assessment == "GOOD":
            continue
        brief.append(
            {
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "assessment": assessment,
                "assessment_reasons": row.get("assessment_reasons") or [],
                "expected_sql": (row.get("expected_sql") or "")[:800],
                "actual_sql": (row.get("actual_sql") or "")[:800],
            }
        )
        if len(brief) >= limit:
            break
    return brief


def build_prompt(*, lever: str, space: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    payload = {
        "lever": lever,
        "allowed_ops": sorted(LEVER_OPS[lever]),
        "hint": LEVER_HINTS.get(lever, ""),
        "space": space_summary(space),
        "failures": failure_brief(failures),
    }
    return SYSTEM_PROMPT + "\n\n" + json.dumps(payload, indent=2, default=str)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise PatchError("LLM did not return a JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise PatchError("LLM JSON must be an object")
    return parsed


def parse_proposal(text: str, lever: str, space: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    parsed = _extract_json_object(text)
    hypothesis = str(parsed.get("hypothesis") or "").strip() or "unspecified"
    patches = parsed.get("patches") or []
    if not isinstance(patches, list) or not patches:
        raise PatchError("LLM returned no patches")
    if not all(isinstance(p, dict) for p in patches):
        raise PatchError("each patch must be an object")
    validate_patches(patches, lever, space)
    return hypothesis, patches


def query_endpoint(w: Any, endpoint: str, prompt: str) -> str:
    """Call a Databricks serving / Foundation Model endpoint and return text."""
    serving = w.serving_endpoints
    try:
        response = serving.query(name=endpoint, messages=[{"role": "user", "content": prompt}])
    except TypeError:
        response = serving.query(name=endpoint, inputs={"prompt": prompt})
    return _response_text(response)


def _response_text(response: Any) -> str:
    if response is None:
        raise PatchError("empty LLM response")
    if isinstance(response, str):
        return response
    choices = getattr(response, "choices", None) or (response.get("choices") if isinstance(response, dict) else None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None) or (first.get("message") if isinstance(first, dict) else None)
        if message is not None:
            content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
            if content:
                return str(content)
        text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
        if text:
            return str(text)
    predictions = getattr(response, "predictions", None) or (
        response.get("predictions") if isinstance(response, dict) else None
    )
    if predictions:
        return str(predictions[0])
    text = getattr(response, "text", None) or (response.get("text") if isinstance(response, dict) else None)
    if text:
        return str(text)
    raise PatchError(f"unrecognized LLM response shape: {type(response)!r}")


def propose_patches(
    *,
    lever: str,
    space: dict[str, Any],
    failures: list[dict[str, Any]],
    propose: ProposeFn,
) -> tuple[str, list[dict[str, Any]]]:
    if lever not in LEVER_OPS:
        raise PatchError(f"unknown lever {lever!r}")
    if not failure_brief(failures):
        raise PatchError("no failing questions to optimize")
    prompt = build_prompt(lever=lever, space=space, failures=failures)
    return parse_proposal(propose(prompt), lever, space)

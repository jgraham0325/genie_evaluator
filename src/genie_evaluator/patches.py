"""Typed serialized_space patches plus a leakage firewall."""

from __future__ import annotations

import copy
import json
import re
import uuid
from typing import Any

ALLOWED_OPS = frozenset(
    {
        "update_table_description",
        "update_column_description",
        "update_column_synonyms",
        "update_column_settings",
        "update_column_config",
        "add_sample_question",
        "add_join_spec",
        "update_join_spec",
        "update_instruction",
        "add_sql_snippet",
        "add_example_sql",
        "clone_and_comment_source",
    }
)

COLUMN_BOOL_FIELDS = (
    "enable_format_assistance",
    "enable_entity_matching",
    "get_example_values",
    "build_value_dictionary",
    "exclude",
)

LEVER_OPS: dict[str, frozenset[str]] = {
    "descriptions": frozenset(
        {
            "update_table_description",
            "update_column_description",
            "update_column_synonyms",
            "update_column_settings",
            "update_column_config",
        }
    ),
    "joins": frozenset({"add_join_spec", "update_join_spec"}),
    "instructions": frozenset({"update_instruction", "add_sample_question"}),
    "sql_snippets": frozenset({"add_sql_snippet", "add_example_sql"}),
    "source_copies": frozenset({"clone_and_comment_source"}),
    "manual": ALLOWED_OPS,
}

AUTO_LEVERS = tuple(name for name in LEVER_OPS if name != "manual")
KNOWN_LEVERS = AUTO_LEVERS


def parse_levers(raw: str) -> list[str]:
    """Parse a JSON list of lever names, defaulting to all known levers."""
    text = (raw or "").strip()
    if not text:
        return list(KNOWN_LEVERS)
    parsed = json.loads(text)
    if not isinstance(parsed, list) or not parsed:
        raise PatchError("levers must be a non-empty JSON list")
    levers = [str(item).strip() for item in parsed]
    unknown = [name for name in levers if name not in AUTO_LEVERS]
    if unknown:
        raise PatchError(f"unknown levers {unknown}; expected one of {list(AUTO_LEVERS)}")
    return levers


class PatchError(ValueError):
    """Raised when a patch is invalid or leaks benchmark material."""


def new_id() -> str:
    return uuid.uuid4().hex


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(part) for part in value]
    return [str(value)]


def _join_text(value: Any) -> str:
    return "\n".join(_as_text_list(value))


def leakage_corpus(space: dict[str, Any]) -> list[str]:
    """Benchmark expected SQL and question text that must not appear in patches."""
    pieces: list[str] = []
    for question in (space.get("benchmarks") or {}).get("questions") or []:
        pieces.append(_join_text(question.get("question")).strip())
        for answer in question.get("answer") or []:
            pieces.append(_join_text(answer.get("content")).strip())
    return [p for p in pieces if p]


def _normalize_sql(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_leakage(text: str, corpus: list[str]) -> None:
    """Reject copy-paste of benchmark questions or expected SQL into Agent config."""
    if not text or not corpus:
        return
    haystack = _normalize_sql(text)
    if len(haystack) < 24:
        return
    for piece in corpus:
        needle = _normalize_sql(piece)
        if len(needle) < 24:
            continue
        if needle in haystack or haystack in needle:
            raise PatchError("patch copies benchmark question or expected SQL (leakage)")


def _tables(space: dict[str, Any]) -> list[dict[str, Any]]:
    return list((space.get("data_sources") or {}).get("tables") or [])


def _table_by_id(space: dict[str, Any], identifier: str) -> dict[str, Any]:
    for table in _tables(space):
        if table.get("identifier") == identifier:
            return table
    raise PatchError(f"unknown table identifier {identifier!r}")


def known_identifiers(space: dict[str, Any]) -> set[str]:
    ids = {str(t.get("identifier")) for t in _tables(space) if t.get("identifier")}
    for view in (space.get("data_sources") or {}).get("metric_views") or []:
        if view.get("identifier"):
            ids.add(str(view["identifier"]))
    return ids


def _instructions(space: dict[str, Any]) -> dict[str, Any]:
    space.setdefault("instructions", {})
    return space["instructions"]


def _instruction_ids(space: dict[str, Any]) -> set[str]:
    instructions = space.get("instructions") or {}
    ids: set[str] = set()
    for key in ("text_instructions", "example_question_sqls", "join_specs"):
        for item in instructions.get(key) or []:
            if item.get("id"):
                ids.add(str(item["id"]))
    snippets = instructions.get("sql_snippets") or {}
    for key in ("filters", "measures", "expressions"):
        for item in snippets.get(key) or []:
            if item.get("id"):
                ids.add(str(item["id"]))
    for item in (space.get("config") or {}).get("sample_questions") or []:
        if item.get("id"):
            ids.add(str(item["id"]))
    for item in (space.get("benchmarks") or {}).get("questions") or []:
        if item.get("id"):
            ids.add(str(item["id"]))
    return ids


def _fresh_id(space: dict[str, Any]) -> str:
    existing = _instruction_ids(space)
    for _ in range(8):
        candidate = new_id()
        if candidate not in existing:
            return candidate
    raise PatchError("could not allocate a unique instruction id")


def _sort_by_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: str(item.get("id") or ""))


def normalize_space(space: dict[str, Any]) -> dict[str, Any]:
    """Sort id-keyed arrays so the Genie API accepts the payload."""
    config = space.get("config") or {}
    if config.get("sample_questions"):
        config["sample_questions"] = _sort_by_id(list(config["sample_questions"]))
    sources = space.get("data_sources") or {}
    if sources.get("tables"):
        sources["tables"] = sorted(
            sources["tables"], key=lambda t: str(t.get("identifier") or "")
        )
        for table in sources["tables"]:
            cols = table.get("column_configs") or []
            if cols:
                table["column_configs"] = sorted(
                    cols, key=lambda c: str(c.get("column_name") or "")
                )
    if sources.get("metric_views"):
        sources["metric_views"] = sorted(
            sources["metric_views"], key=lambda t: str(t.get("identifier") or "")
        )
    instructions = space.get("instructions") or {}
    for key in ("text_instructions", "example_question_sqls", "join_specs"):
        if instructions.get(key):
            instructions[key] = _sort_by_id(list(instructions[key]))
    snippets = instructions.get("sql_snippets") or {}
    for key in ("filters", "measures", "expressions"):
        if snippets.get(key):
            snippets[key] = _sort_by_id(list(snippets[key]))
    benchmarks = space.get("benchmarks") or {}
    if benchmarks.get("questions"):
        benchmarks["questions"] = _sort_by_id(list(benchmarks["questions"]))
    return space


def _scan_patch_text(patch: dict[str, Any], corpus: list[str]) -> None:
    for key in (
        "description",
        "instruction",
        "content",
        "sql",
        "question",
        "comment",
        "display_name",
        "table_comment",
        "synonyms",
    ):
        if key in patch:
            check_leakage(_join_text(patch.get(key)), corpus)
    for comment in (patch.get("column_comments") or {}).values():
        check_leakage(str(comment), corpus)


def validate_patches(patches: list[dict[str, Any]], lever: str, space: dict[str, Any]) -> None:
    allowed = LEVER_OPS.get(lever)
    if allowed is None:
        raise PatchError(f"unknown lever {lever!r}")
    corpus = leakage_corpus(space)
    identifiers = known_identifiers(space)
    for patch in patches:
        op = str(patch.get("op") or "")
        if op not in ALLOWED_OPS:
            raise PatchError(f"unknown patch op {op!r}")
        if op not in allowed:
            raise PatchError(f"op {op!r} is not allowed for lever {lever!r}")
        _scan_patch_text(patch, corpus)
        ident = patch.get("identifier")
        if ident and ident not in identifiers and op != "clone_and_comment_source":
            raise PatchError(f"patch references unknown identifier {ident!r}")
        if op == "clone_and_comment_source" and ident not in identifiers:
            raise PatchError(f"clone target is not a configured source: {ident!r}")
        if op.startswith("update_column_") and not patch.get("column_name"):
            raise PatchError(f"{op} requires column_name")
        left = (patch.get("left") or {}).get("identifier") if isinstance(patch.get("left"), dict) else None
        right = (patch.get("right") or {}).get("identifier") if isinstance(patch.get("right"), dict) else None
        for side in (left, right):
            if side and side not in identifiers:
                raise PatchError(f"join references unknown identifier {side!r}")


def _apply_table_description(space: dict[str, Any], patch: dict[str, Any]) -> None:
    table = _table_by_id(space, str(patch["identifier"]))
    table["description"] = _as_text_list(patch.get("description"))


def _upsert_column(space: dict[str, Any], identifier: str, column_name: str) -> dict[str, Any]:
    if not column_name:
        raise PatchError("column_name is required")
    table = _table_by_id(space, identifier)
    cols = table.setdefault("column_configs", [])
    for col in cols:
        if col.get("column_name") == column_name:
            return col
    col = {"column_name": column_name}
    cols.append(col)
    return col


def _apply_column_config(space: dict[str, Any], patch: dict[str, Any]) -> None:
    col = _upsert_column(space, str(patch["identifier"]), str(patch.get("column_name") or ""))
    if "description" in patch:
        col["description"] = _as_text_list(patch.get("description"))
    if "synonyms" in patch:
        col["synonyms"] = [str(s) for s in _as_text_list(patch.get("synonyms")) if str(s).strip()]
    for field in COLUMN_BOOL_FIELDS:
        if field in patch:
            col[field] = bool(patch[field])


def _apply_column_description(space: dict[str, Any], patch: dict[str, Any]) -> None:
    if "description" not in patch:
        raise PatchError("update_column_description requires description")
    _apply_column_config(space, patch)


def _apply_column_synonyms(space: dict[str, Any], patch: dict[str, Any]) -> None:
    if "synonyms" not in patch:
        raise PatchError("update_column_synonyms requires synonyms")
    _apply_column_config(space, patch)


def _apply_column_settings(space: dict[str, Any], patch: dict[str, Any]) -> None:
    if not any(field in patch for field in COLUMN_BOOL_FIELDS):
        raise PatchError(
            "update_column_settings requires one of "
            + ", ".join(COLUMN_BOOL_FIELDS)
        )
    _apply_column_config(space, patch)


def _apply_join(space: dict[str, Any], patch: dict[str, Any], *, update: bool) -> None:
    instructions = _instructions(space)
    joins = instructions.setdefault("join_specs", [])
    join_id = str(patch.get("id") or "")
    if update:
        if not join_id:
            raise PatchError("update_join_spec requires id")
        target = next((j for j in joins if j.get("id") == join_id), None)
        if target is None:
            raise PatchError(f"unknown join spec id {join_id!r}")
        if patch.get("instruction"):
            target["instruction"] = _as_text_list(patch.get("instruction"))
        if patch.get("comment"):
            target["comment"] = _as_text_list(patch.get("comment"))
        if patch.get("sql"):
            target["sql"] = _as_text_list(patch.get("sql"))
        return
    left = patch.get("left") or {}
    right = patch.get("right") or {}
    sql = _as_text_list(patch.get("sql"))
    if len(sql) == 1:
        sql.append("--rt=FROM_RELATIONSHIP_TYPE_MANY_TO_ONE--")
    if len(sql) != 2:
        raise PatchError("add_join_spec.sql must be the join condition plus a relationship tag")
    joins.append(
        {
            "id": join_id or _fresh_id(space),
            "left": {k: v for k, v in {
                "identifier": str(left.get("identifier") or ""),
                "alias": left.get("alias"),
            }.items() if v is not None and v != ""},
            "right": {k: v for k, v in {
                "identifier": str(right.get("identifier") or ""),
                "alias": right.get("alias"),
            }.items() if v is not None and v != ""},
            "sql": sql,
            "instruction": _as_text_list(patch.get("instruction")),
            "comment": _as_text_list(patch.get("comment") or "Join"),
        }
    )


def _apply_instruction(space: dict[str, Any], patch: dict[str, Any]) -> None:
    instructions = _instructions(space)
    items = instructions.setdefault("text_instructions", [])
    addition = _join_text(patch.get("content") or patch.get("instruction")).strip()
    if not addition:
        raise PatchError("update_instruction requires content")
    if not items:
        items.append({"id": _fresh_id(space), "content": [addition]})
        return
    existing = _join_text(items[0].get("content")).rstrip()
    items[0]["content"] = [f"{existing}\n{addition}" if existing else addition]


def _apply_sql_snippet(space: dict[str, Any], patch: dict[str, Any]) -> None:
    kind = str(patch.get("kind") or "filters")
    if kind not in {"filters", "measures", "expressions"}:
        raise PatchError(f"unknown sql snippet kind {kind!r}")
    instructions = _instructions(space)
    snippets = instructions.setdefault("sql_snippets", {})
    bucket = snippets.setdefault(kind, [])
    sql = _as_text_list(patch.get("sql"))
    if not sql:
        raise PatchError("add_sql_snippet requires sql")
    entry: dict[str, Any] = {
        "id": str(patch.get("id") or _fresh_id(space)),
        "display_name": str(patch.get("display_name") or "Untitled snippet"),
        "sql": sql,
    }
    if patch.get("alias"):
        entry["alias"] = str(patch["alias"])
    if patch.get("synonyms"):
        entry["synonyms"] = [str(s) for s in _as_text_list(patch.get("synonyms")) if str(s).strip()]
    bucket.append(entry)


def _apply_example_sql(space: dict[str, Any], patch: dict[str, Any]) -> None:
    instructions = _instructions(space)
    examples = instructions.setdefault("example_question_sqls", [])
    question = _as_text_list(patch.get("question"))
    sql = _as_text_list(patch.get("sql"))
    if not question or not sql:
        raise PatchError("add_example_sql requires question and sql")
    example: dict[str, Any] = {
        "id": str(patch.get("id") or _fresh_id(space)),
        "question": question,
        "sql": sql,
    }
    if patch.get("usage_guidance"):
        example["usage_guidance"] = _as_text_list(patch.get("usage_guidance"))
    examples.append(example)


def _apply_sample_question(space: dict[str, Any], patch: dict[str, Any]) -> None:
    question = _as_text_list(patch.get("question"))
    if not question:
        raise PatchError("add_sample_question requires question")
    config = space.setdefault("config", {})
    samples = config.setdefault("sample_questions", [])
    samples.append(
        {
            "id": str(patch.get("id") or _fresh_id(space)),
            "question": question,
        }
    )


def apply_config_patches(space: dict[str, Any], patches: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply non-source patches and return a new serialized_space dict."""
    updated = copy.deepcopy(space)
    for patch in patches:
        op = patch.get("op")
        if op == "update_table_description":
            _apply_table_description(updated, patch)
        elif op == "update_column_description":
            _apply_column_description(updated, patch)
        elif op == "update_column_synonyms":
            _apply_column_synonyms(updated, patch)
        elif op == "update_column_settings":
            _apply_column_settings(updated, patch)
        elif op == "update_column_config":
            _apply_column_config(updated, patch)
        elif op == "add_sample_question":
            _apply_sample_question(updated, patch)
        elif op == "add_join_spec":
            _apply_join(updated, patch, update=False)
        elif op == "update_join_spec":
            _apply_join(updated, patch, update=True)
        elif op == "update_instruction":
            _apply_instruction(updated, patch)
        elif op == "add_sql_snippet":
            _apply_sql_snippet(updated, patch)
        elif op == "add_example_sql":
            _apply_example_sql(updated, patch)
        elif op == "clone_and_comment_source":
            continue
        else:
            raise PatchError(f"unknown patch op {op!r}")
    return normalize_space(updated)

"""Get / create / update / trash Genie spaces, including serialized_space."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from genie_evaluator.eval_runner import _attr, _call_genie


class SpaceError(RuntimeError):
    """Raised when a Genie space cannot be read or written."""


def parse_serialized_space(raw: Any) -> dict[str, Any]:
    """Return a parsed serialized_space dict from an API value."""
    if raw is None:
        raise SpaceError("space has no serialized_space")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SpaceError(f"serialized_space is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SpaceError("serialized_space JSON must be an object")
        return parsed
    raise SpaceError(f"unexpected serialized_space type {type(raw)!r}")


def dump_serialized_space(space: dict[str, Any]) -> str:
    """Canonical JSON string the Genie APIs expect."""
    return json.dumps(space, separators=(",", ":"), ensure_ascii=False)


def config_hash(space: dict[str, Any]) -> str:
    """Stable short hash of a serialized_space payload."""
    blob = json.dumps(space, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def get_space(w: Any, space_id: str) -> dict[str, Any]:
    """Fetch a space including serialized_space."""
    obj = _call_genie(
        w,
        ("get_space", "genie_get_space"),
        space_id,
        rest=("GET", f"/api/2.0/genie/spaces/{space_id}"),
        include_serialized_space=True,
    )
    serialized = parse_serialized_space(_attr(obj, "serialized_space"))
    return {
        "space_id": str(_attr(obj, "space_id", space_id) or space_id),
        "title": str(_attr(obj, "title", "") or ""),
        "description": str(_attr(obj, "description", "") or ""),
        "warehouse_id": str(_attr(obj, "warehouse_id", "") or ""),
        "serialized_space": serialized,
    }


def create_space(
    w: Any,
    *,
    title: str,
    warehouse_id: str,
    serialized_space: dict[str, Any],
    parent_path: str,
    description: str = "",
) -> str:
    """Create a Genie space and return its space_id."""
    if not parent_path:
        raise SpaceError("parent_path is required to create a throwaway experiment space")
    body = {
        "title": title,
        "warehouse_id": warehouse_id,
        "parent_path": parent_path,
        "serialized_space": dump_serialized_space(serialized_space),
    }
    if description:
        body["description"] = description
    created = _call_genie(
        w,
        ("create_space", "genie_create_space"),
        rest=("POST", "/api/2.0/genie/spaces"),
        **body,
    )
    space_id = str(_attr(created, "space_id", "") or "")
    if not space_id:
        raise SpaceError("create_space returned no space_id")
    return space_id


def update_serialized_space(w: Any, space_id: str, serialized_space: dict[str, Any]) -> None:
    """Replace the live serialized_space on an existing Agent."""
    _call_genie(
        w,
        ("update_space", "genie_update_space"),
        space_id,
        rest=("PATCH", f"/api/2.0/genie/spaces/{space_id}"),
        serialized_space=dump_serialized_space(serialized_space),
    )


def trash_space(w: Any, space_id: str) -> None:
    """Move a throwaway experiment space to trash."""
    try:
        _call_genie(
            w,
            ("trash_space", "genie_trash_space", "delete_space"),
            space_id,
            rest=("POST", f"/api/2.0/genie/spaces/{space_id}/trash"),
        )
    except Exception:
        _call_genie(
            w,
            ("trash_space", "genie_trash_space", "delete_space"),
            space_id,
            rest=("DELETE", f"/api/2.0/genie/spaces/{space_id}"),
        )


def default_parent_path(w: Any) -> str:
    """Workspace folder for throwaway experiment spaces under the current user."""
    me = w.current_user.me()
    user_name = str(getattr(me, "user_name", "") or getattr(me, "userName", "") or "")
    if not user_name:
        raise SpaceError("cannot resolve current user for opt_parent_path")
    return f"/Users/{user_name}/genie_opt_experiments"


def ensure_parent_path(w: Any, parent_path: str) -> None:
    """Create the workspace folder if it does not exist."""
    w.workspace.mkdirs(parent_path)

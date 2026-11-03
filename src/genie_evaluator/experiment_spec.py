"""Load hand-authored optimize experiments from experiments/<name>.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from genie_evaluator.patches import PatchError, parse_levers, validate_patches

NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
OPT_MODES = ("auto", "experiments")


def parse_opt_mode(raw: str) -> str:
    mode = (raw or "auto").strip().lower()
    if mode == "manual":
        mode = "experiments"
    if mode not in OPT_MODES:
        raise PatchError(f"opt_mode must be one of {list(OPT_MODES)}, got {mode!r}")
    return mode


def list_experiment_names(root: Path) -> list[str]:
    directory = (root / "experiments").resolve()
    if not directory.is_dir():
        raise PatchError(f"experiments directory not found: {directory}")
    names = sorted(path.stem for path in directory.glob("*.json") if NAME_RE.match(path.stem))
    if not names:
        raise PatchError(f"no experiment JSON files in {directory}")
    return names


def parse_experiment_names(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list) or not parsed:
        raise PatchError("experiments must be a non-empty JSON list of names")
    names = [str(item).strip() for item in parsed]
    bad = [name for name in names if not NAME_RE.match(name)]
    if bad:
        raise PatchError(f"invalid experiment names {bad}")
    return names


def resolve_fanout(
    *,
    opt_mode: str,
    levers_raw: str,
    experiments_raw: str,
    root: Path,
) -> tuple[str, list[str]]:
    """Pick auto-levers or experiment files. Files in experiments/ are never used in auto mode."""
    mode = parse_opt_mode(opt_mode)
    if mode == "auto":
        return mode, parse_levers(levers_raw)
    names = parse_experiment_names(experiments_raw)
    if not names:
        names = list_experiment_names(root)
    return mode, names


def load_experiment(root: Path, name: str, space: dict[str, Any] | None = None) -> dict[str, Any]:
    if not NAME_RE.match(name):
        raise PatchError(f"invalid experiment name {name!r}")
    experiments_dir = (root / "experiments").resolve()
    path = (experiments_dir / f"{name}.json").resolve()
    if path.parent != experiments_dir:
        raise PatchError(f"experiment path escapes experiments/: {path}")
    if not path.is_file():
        raise PatchError(f"experiment file not found: {path}")
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict):
        raise PatchError(f"{path} must contain a JSON object")
    patches = spec.get("patches")
    if not isinstance(patches, list) or not patches:
        raise PatchError(f"{path} must include a non-empty patches list")
    if not all(isinstance(p, dict) for p in patches):
        raise PatchError(f"{path} patches must be objects")
    hypothesis = str(spec.get("hypothesis") or name).strip()
    if space is not None:
        validate_patches(patches, "manual", space)
    return {
        "name": str(spec.get("name") or name),
        "hypothesis": hypothesis,
        "patches": patches,
    }

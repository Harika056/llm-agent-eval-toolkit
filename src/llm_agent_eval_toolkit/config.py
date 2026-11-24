"""Config-driven evals (``agent-eval eval eval.yaml``), so CI needs no Python glue code."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .evals.scorers import (
    Contains,
    Custom,
    ExactMatch,
    JsonFields,
    LLMJudge,
    NotContains,
    NumericClose,
    RegexMatch,
    Scorer,
)


class ConfigError(ValueError):
    pass


def import_object(spec: str, base: Path | None = None) -> Any:
    """Import ``"package.module:attribute"``. The config's folder and the working directory are importable."""
    if ":" not in spec:
        raise ConfigError(f"{spec!r}: use 'module:function'")
    mod_name, attr = spec.split(":", 1)
    for p in filter(None, [str(base) if base else None, str(Path.cwd())]):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        obj: Any = importlib.import_module(mod_name)
    except ImportError as e:
        raise ConfigError(f"Cannot import {mod_name!r}: {e}") from e
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as e:
            raise ConfigError(f"{mod_name!r} has no attribute {attr!r}") from e
    return obj


def build_scorer(spec: dict[str, Any], base: Path | None = None) -> Scorer:
    spec = dict(spec)
    kind = spec.pop("type", None)
    simple: dict[str, Callable[..., Scorer]] = {
        "exact_match": ExactMatch,
        "contains": Contains,
        "not_contains": NotContains,
        "regex": RegexMatch,
        "json_fields": JsonFields,
        "numeric_close": NumericClose,
    }
    try:
        if kind in simple:
            return simple[kind](**spec)
        if kind == "llm_judge":
            judge = spec.pop("judge", None)
            if not judge:
                raise ConfigError(
                    "llm_judge needs 'judge: module:function' (a function prompt -> reply text)"
                )
            if "rubric" not in spec:
                raise ConfigError("llm_judge needs a 'rubric'")
            return LLMJudge(import_object(judge, base), **spec)
        if kind == "custom":
            fn = spec.pop("function", None)
            if not fn:
                raise ConfigError("custom needs 'function: module:function'")
            return Custom(import_object(fn, base), **spec)
    except TypeError as e:
        raise ConfigError(f"Bad options for scorer {kind!r}: {e}") from e
    raise ConfigError(f"Unknown scorer type {kind!r}. Use one of: {sorted([*simple, 'llm_judge', 'custom'])}")


def load_eval_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    base = p.parent.resolve()
    for key in ("target", "dataset", "scorers"):
        if key not in cfg:
            raise ConfigError(f"{p}: missing '{key}'")
    if not isinstance(cfg["scorers"], list) or not cfg["scorers"]:
        raise ConfigError(f"{p}: 'scorers' must be a non-empty list")
    ds = Path(cfg["dataset"])
    return {
        "name": cfg.get("name", p.stem),
        "target": import_object(cfg["target"], base),
        "dataset": ds if ds.is_absolute() else base / ds,
        "scorers": [build_scorer(s, base) for s in cfg["scorers"]],
        "concurrency": int(cfg.get("concurrency", 1)),
        "pricing": (base / cfg["pricing"]) if cfg.get("pricing") else None,
        "traces": (base / cfg["traces"]) if cfg.get("traces") else None,
        "thresholds": cfg.get("thresholds"),
        "setup": import_object(cfg["setup"], base) if cfg.get("setup") else None,
    }

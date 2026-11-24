"""Golden datasets: a list of cases, each an input plus what a good answer looks like."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Case:
    id: str
    input: Any
    expected: Any = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetError(ValueError):
    pass


def _to_case(raw: Any, where: str, index: int) -> Case:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: case {index} must be an object, got {type(raw).__name__}")
    if "input" not in raw:
        raise DatasetError(f"{where}: case {index} has no 'input'")
    known = {"id", "input", "expected", "tags", "metadata"}
    extra = {k: v for k, v in raw.items() if k not in known}
    tags = raw.get("tags") or ()
    if isinstance(tags, str):
        tags = (tags,)
    return Case(
        id=str(raw.get("id", index + 1)),
        input=raw["input"],
        expected=raw.get("expected"),
        tags=tuple(str(t) for t in tags),
        metadata={**(raw.get("metadata") or {}), **extra},
    )


class Dataset:
    def __init__(self, cases: Iterable[Case], name: str = "dataset"):
        self.cases = list(cases)
        self.name = name
        seen: set[str] = set()
        for c in self.cases:
            if c.id in seen:
                raise DatasetError(f"{name}: duplicate case id {c.id!r}")
            seen.add(c.id)

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]], name: str = "dataset") -> Dataset:
        return cls((_to_case(r, name, i) for i, r in enumerate(records)), name=name)

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """Load ``.jsonl`` (one case per line), ``.json`` or ``.yaml``/``.yml`` (a list of cases)."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".jsonl":
            records = []
            for n, line in enumerate(text.splitlines(), 1):
                if not line.strip() or line.lstrip().startswith("//"):
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DatasetError(f"{p}:{n}: invalid JSON ({e.msg})") from e
        elif p.suffix == ".json":
            records = json.loads(text)
        elif p.suffix in (".yaml", ".yml"):
            records = yaml.safe_load(text)
        else:
            raise DatasetError(f"{p}: use .jsonl, .json, .yaml or .yml")
        if isinstance(records, dict) and "cases" in records:
            records = records["cases"]
        if not isinstance(records, list):
            raise DatasetError(f"{p}: expected a list of cases")
        return cls.from_records(records, name=p.stem)

    def filter(self, tag: str) -> Dataset:
        return Dataset([c for c in self.cases if tag in c.tags], name=f"{self.name}[{tag}]")

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

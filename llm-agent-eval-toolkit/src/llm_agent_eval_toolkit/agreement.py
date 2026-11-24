"""How far can the LLM judge be trusted? Compare its grades with human labels using Cohen's kappa."""

from __future__ import annotations

import json
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def cohen_kappa(
    a: Sequence[Hashable],
    b: Sequence[Hashable],
    *,
    weights: str | None = None,
    labels: Sequence[Hashable] | None = None,
) -> float:
    """Cohen's kappa between two raters.

    ``weights=None`` treats labels as categories. For ordered grades (1-5), ``"linear"`` or ``"quadratic"``
    gives partial credit for near misses; labels are then ordered by ``labels`` or sorted.
    Returns 1.0 when both raters use one identical label throughout (no variance, perfect agreement).
    """
    if len(a) != len(b):
        raise ValueError("Both raters must label the same number of items")
    if not a:
        raise ValueError("Need at least one item")
    if weights not in (None, "linear", "quadratic"):
        raise ValueError("weights must be None, 'linear' or 'quadratic'")
    cats = list(labels) if labels is not None else sorted(set(a) | set(b), key=lambda x: (str(type(x)), x))
    missing = (set(a) | set(b)) - set(cats)
    if missing:
        raise ValueError(f"labels missing from the label list: {sorted(map(str, missing))}")
    idx = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(a)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if weights is None:
            return 0.0 if i == j else 1.0
        d = abs(i - j) / (k - 1) if k > 1 else 0.0
        return d if weights == "linear" else d * d

    observed = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    expected = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return 1 - observed / expected


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) bands."""
    if kappa < 0:
        return "poor (worse than chance)"
    for limit, label in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if kappa <= limit:
            return label
    return "almost perfect"


@dataclass
class AgreementReport:
    n: int
    kappa: float
    weighted_kappa: float | None
    percent_agreement: float
    labels: list[str]
    confusion: list[list[int]]  # rows: human, columns: judge
    unmatched_judge: list[str]
    unmatched_human: list[str]
    disagreements: list[dict[str, Any]]

    @property
    def interpretation(self) -> str:
        return interpret_kappa(self.kappa)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["interpretation"] = self.interpretation
        return d

    def to_markdown(self) -> str:
        lines = [
            "### Judge vs human agreement",
            "",
            f"- Items compared: **{self.n}**",
            f"- Cohen's kappa: **{self.kappa:.3f}** ({self.interpretation})",
        ]
        if self.weighted_kappa is not None:
            lines.append(f"- Quadratic-weighted kappa: **{self.weighted_kappa:.3f}**")
        lines += [
            f"- Raw agreement: **{self.percent_agreement:.1%}**",
            "",
            "Confusion (rows = human, columns = judge):",
            "",
        ]
        lines.append("| human \\ judge | " + " | ".join(self.labels) + " |")
        lines.append("|---|" + "---|" * len(self.labels))
        for lab, row in zip(self.labels, self.confusion):
            lines.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
        if self.unmatched_human or self.unmatched_judge:
            lines += [
                "",
                f"Skipped: {len(self.unmatched_human)} human-only ids, {len(self.unmatched_judge)} judge-only ids.",
            ]
        return "\n".join(lines)


def _as_ordered(labels: Sequence[str]) -> list[Any] | None:
    try:
        return sorted({float(x) for x in labels})
    except ValueError:
        return None


def compare_labels(judge: Mapping[str, Any], human: Mapping[str, Any]) -> AgreementReport:
    """Compare two ``{item_id: label}`` maps on the ids they share.

    Numeric labels (e.g. 1-5 grades) also get a quadratic-weighted kappa.
    """
    ids = sorted(set(judge) & set(human))
    if not ids:
        raise ValueError("No item ids in common between judge and human labels")
    j = [str(judge[i]) for i in ids]
    h = [str(human[i]) for i in ids]
    labels = sorted(set(j) | set(h))
    ordered = _as_ordered(labels)
    if ordered is not None:
        labels = sorted(labels, key=float)
    idx = {lab: n for n, lab in enumerate(labels)}
    conf = [[0] * len(labels) for _ in labels]
    for hv, jv in zip(h, j):
        conf[idx[hv]][idx[jv]] += 1
    return AgreementReport(
        n=len(ids),
        kappa=cohen_kappa(h, j, labels=labels),
        weighted_kappa=cohen_kappa(h, j, labels=labels, weights="quadratic")
        if ordered and len(labels) > 2
        else None,
        percent_agreement=sum(a == b for a, b in zip(h, j)) / len(ids),
        labels=labels,
        confusion=conf,
        unmatched_judge=sorted(set(judge) - set(human)),
        unmatched_human=sorted(set(human) - set(judge)),
        disagreements=[{"id": i, "human": hv, "judge": jv} for i, hv, jv in zip(ids, h, j) if hv != jv],
    )


def load_labels(path: str | Path, *, id_key: str = "id", label_key: str = "label") -> dict[str, str]:
    """Read ``{id: label}`` from a JSONL/JSON/YAML/CSV file of ``{"id": ..., "label": ...}`` rows."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif p.suffix == ".json":
        rows = json.loads(text)
    elif p.suffix in (".yaml", ".yml"):
        rows = yaml.safe_load(text)
    elif p.suffix == ".csv":
        import csv

        rows = list(csv.DictReader(text.splitlines()))
    else:
        raise ValueError(f"{p}: use .jsonl, .json, .yaml or .csv")
    if isinstance(rows, dict):
        return {str(k): str(v) for k, v in rows.items()}
    out = {}
    for r in rows:
        if id_key not in r or label_key not in r:
            raise ValueError(f"{p}: every row needs {id_key!r} and {label_key!r}")
        out[str(r[id_key])] = str(r[label_key])
    return out


def judge_labels_from_report(report: Mapping[str, Any], scorer: str = "llm_judge") -> dict[str, str]:
    """Pull the judge's raw grade for every case out of a saved eval report (``EvalReport.to_dict()``)."""
    out = {}
    for c in report["cases"]:
        for s in c.get("scores", []):
            if s["name"] == scorer and s.get("label") is not None:
                out[str(c["id"])] = str(s["label"])
    if not out:
        raise ValueError(f"No {scorer!r} labels in the report")
    return out

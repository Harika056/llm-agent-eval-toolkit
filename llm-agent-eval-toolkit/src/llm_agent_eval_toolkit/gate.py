"""CI gate: compare an eval run with a baseline and fail when accuracy, latency or cost regress past limits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .evals.runner import EvalReport


@dataclass
class Thresholds:
    """Limits a run must stay within. Leave a limit as ``None`` to skip that check.

    Absolute limits apply to the run on its own; ``max_*_drop``/``max_*_increase`` compare with the baseline.
    Increases are fractions: ``0.2`` allows p95 latency to grow by 20%.
    """

    min_accuracy: float | None = None
    max_accuracy_drop: float | None = 0.02
    max_errors: int | None = 0
    max_p95_latency_ms: float | None = None
    max_p95_latency_increase: float | None = 0.25
    max_cost_per_case_usd: float | None = None
    max_cost_increase: float | None = 0.20
    max_newly_failing_cases: int | None = None
    scorers: dict[str, dict[str, float]] = field(default_factory=dict)  # name -> {min_pass_rate, max_drop}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Thresholds:
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown threshold keys: {sorted(unknown)}. Known: {sorted(known)}")
        return cls(**dict(d))

    @classmethod
    def from_file(cls, path: str | Path) -> Thresholds:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data.get("thresholds", data))


@dataclass
class Check:
    name: str
    current: float | None
    baseline: float | None
    limit: str
    passed: bool
    skipped: bool = False
    unit: str = ""

    def fmt(self, v: float | None) -> str:
        if v is None:
            return "–"
        if self.unit == "%":
            return f"{v:.1%}"
        if self.unit == "ms":
            return f"{v:,.0f} ms"
        if self.unit == "$":
            return f"${v:.5f}"
        return f"{v:g}"


@dataclass
class GateResult:
    checks: list[Check]
    newly_failing: list[str]
    fixed: list[str]
    current: Mapping[str, Any]
    baseline: Mapping[str, Any] | None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_markdown(self, title: str = "Agent eval gate") -> str:
        icon = "✅" if self.passed else "❌"
        lines = [f"## {icon} {title}: {'passed' if self.passed else 'failed'}", ""]
        m = self.current
        acc = m.get("accuracy")
        lines.append(
            f"{m.get('passed', 0)}/{m.get('cases', 0)} cases passed"
            + (f" ({acc:.1%})" if acc is not None else "")
            + (" · compared with baseline" if self.baseline else " · no baseline, absolute limits only")
        )
        lines += [
            "",
            "| Check | Baseline | This run | Change | Limit | Result |",
            "|---|---|---|---|---|---|",
        ]
        for c in self.checks:
            change = "–"
            if c.current is not None and c.baseline is not None:
                if c.unit == "%":
                    change = f"{(c.current - c.baseline) * 100:+.1f} pts"
                elif c.baseline:
                    change = f"{(c.current - c.baseline) / c.baseline:+.1%}"
            result = "skipped" if c.skipped else ("✅" if c.passed else "❌")
            lines.append(
                f"| {c.name} | {c.fmt(c.baseline)} | {c.fmt(c.current)} | {change} | {c.limit} | {result} |"
            )
        if self.newly_failing:
            lines += [
                "",
                f"**Newly failing cases ({len(self.newly_failing)}):** "
                + ", ".join(f"`{i}`" for i in self.newly_failing[:30]),
            ]
        if self.fixed:
            lines += ["", f"Now passing ({len(self.fixed)}): " + ", ".join(f"`{i}`" for i in self.fixed[:30])]
        return "\n".join(lines) + "\n"


def _pct_increase(cur: float | None, base: float | None) -> float | None:
    if cur is None or base is None or base <= 0:
        return None
    return (cur - base) / base


def evaluate_gate(
    current: EvalReport | Mapping[str, Any],
    thresholds: Thresholds,
    baseline: EvalReport | Mapping[str, Any] | None = None,
) -> GateResult:
    cur = current.to_dict() if isinstance(current, EvalReport) else current
    base = baseline.to_dict() if isinstance(baseline, EvalReport) else baseline
    m, b = cur["metrics"], (base or {}).get("metrics", {})
    t = thresholds
    checks: list[Check] = []

    def add(
        name: str, c: float | None, bv: float | None, limit: str, ok: bool | None, unit: str = ""
    ) -> None:
        checks.append(Check(name, c, bv, limit, True if ok is None else ok, ok is None, unit))

    acc, bacc = m.get("accuracy"), b.get("accuracy")
    if t.min_accuracy is not None:
        add(
            "Accuracy (floor)",
            acc,
            bacc,
            f"≥ {t.min_accuracy:.0%}",
            None if acc is None else acc >= t.min_accuracy,
            "%",
        )
    if t.max_accuracy_drop is not None and base:
        ok = None if acc is None or bacc is None else (bacc - acc) <= t.max_accuracy_drop + 1e-12
        add("Accuracy vs baseline", acc, bacc, f"drop ≤ {t.max_accuracy_drop * 100:g} pts", ok, "%")
    if t.max_errors is not None:
        add(
            "Errors",
            m.get("errors"),
            b.get("errors"),
            f"≤ {t.max_errors}",
            m.get("errors", 0) <= t.max_errors,
        )

    p95, bp95 = m.get("p95_latency_ms"), b.get("p95_latency_ms")
    if t.max_p95_latency_ms is not None:
        add(
            "p95 latency (ceiling)",
            p95,
            bp95,
            f"≤ {t.max_p95_latency_ms:,.0f} ms",
            None if p95 is None else p95 <= t.max_p95_latency_ms,
            "ms",
        )
    if t.max_p95_latency_increase is not None and base:
        inc = _pct_increase(p95, bp95)
        add(
            "p95 latency vs baseline",
            p95,
            bp95,
            f"+{t.max_p95_latency_increase:.0%} max",
            None if inc is None else inc <= t.max_p95_latency_increase,
            "ms",
        )

    cost, bcost = m.get("mean_cost_usd"), b.get("mean_cost_usd")
    if t.max_cost_per_case_usd is not None:
        add(
            "Cost per case (ceiling)",
            cost,
            bcost,
            f"≤ ${t.max_cost_per_case_usd:g}",
            None if cost is None else cost <= t.max_cost_per_case_usd,
            "$",
        )
    if t.max_cost_increase is not None and base:
        inc = _pct_increase(cost, bcost)
        add(
            "Cost per case vs baseline",
            cost,
            bcost,
            f"+{t.max_cost_increase:.0%} max",
            None if inc is None else inc <= t.max_cost_increase,
            "$",
        )

    for name, lim in t.scorers.items():
        cs, bs = m.get("scorers", {}).get(name), b.get("scorers", {}).get(name)
        cr, br = (cs or {}).get("pass_rate"), (bs or {}).get("pass_rate")
        if "min_pass_rate" in lim:
            add(
                f"{name} pass rate (floor)",
                cr,
                br,
                f"≥ {lim['min_pass_rate']:.0%}",
                None if cr is None else cr >= lim["min_pass_rate"],
                "%",
            )
        if "max_drop" in lim and base:
            add(
                f"{name} pass rate vs baseline",
                cr,
                br,
                f"drop ≤ {lim['max_drop'] * 100:g} pts",
                None if cr is None or br is None else br - cr <= lim["max_drop"] + 1e-12,
                "%",
            )

    newly, fixed = [], []
    if base:
        before = {str(c["id"]): c.get("passed") for c in base.get("cases", [])}
        for c in cur.get("cases", []):
            cid, was = str(c["id"]), before.get(str(c["id"]))
            if was is True and not c.get("passed"):
                newly.append(cid)
            elif was is False and c.get("passed"):
                fixed.append(cid)
        if t.max_newly_failing_cases is not None:
            add(
                "Newly failing cases",
                len(newly),
                0,
                f"≤ {t.max_newly_failing_cases}",
                len(newly) <= t.max_newly_failing_cases,
            )

    return GateResult(checks, newly, fixed, m, b or None)

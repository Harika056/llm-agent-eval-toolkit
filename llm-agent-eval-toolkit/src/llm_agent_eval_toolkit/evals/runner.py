"""Run an agent over a golden dataset, score every case, and record accuracy, latency, tokens and cost."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import os
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opentelemetry.trace import Status, StatusCode

from ..stats import mean, percentile
from ..tracing import collect_stats, get_tracer
from .dataset import Case, Dataset
from .scorers import Score, Scorer

REPORT_VERSION = 1


@dataclass
class CaseResult:
    id: str
    input: Any
    expected: Any
    output: Any
    error: str | None
    scores: list[Score]
    latency_ms: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    tags: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.error is None and bool(self.scores) and all(s.passed for s in self.scores)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d


@dataclass
class EvalReport:
    name: str
    dataset: str
    cases: list[CaseResult]
    started_at: str
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, Any]:
        return compute_metrics(self.cases)

    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": REPORT_VERSION,
            "name": self.name,
            "dataset": self.dataset,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 3),
            "metadata": self.metadata,
            "metrics": self.metrics,
            "cases": [c.to_dict() for c in self.cases],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = [
            CaseResult(
                id=c["id"],
                input=c.get("input"),
                expected=c.get("expected"),
                output=c.get("output"),
                error=c.get("error"),
                scores=[Score(**s) for s in c.get("scores", [])],
                latency_ms=c["latency_ms"],
                llm_calls=c.get("llm_calls", 0),
                input_tokens=c.get("input_tokens", 0),
                output_tokens=c.get("output_tokens", 0),
                cost_usd=c.get("cost_usd"),
                tags=tuple(c.get("tags", ())),
            )
            for c in d["cases"]
        ]
        return cls(
            d["name"],
            d.get("dataset", ""),
            cases,
            d.get("started_at", ""),
            d.get("duration_s", 0.0),
            d.get("metadata", {}),
        )


def compute_metrics(cases: Sequence[CaseResult]) -> dict[str, Any]:
    n = len(cases)
    lat = [c.latency_ms for c in cases if c.error is None]
    costs = [c.cost_usd for c in cases if c.cost_usd is not None]
    by_scorer: dict[str, list[Score]] = {}
    for c in cases:
        for s in c.scores:
            by_scorer.setdefault(s.name, []).append(s)
    return {
        "cases": n,
        "passed": sum(c.passed for c in cases),
        "accuracy": (sum(c.passed for c in cases) / n) if n else None,
        "errors": sum(c.error is not None for c in cases),
        "p50_latency_ms": percentile(lat, 50),
        "p95_latency_ms": percentile(lat, 95),
        "mean_latency_ms": mean(lat),
        "mean_cost_usd": mean(costs),
        "total_cost_usd": sum(costs) if costs else None,
        "mean_input_tokens": mean(c.input_tokens for c in cases),
        "mean_output_tokens": mean(c.output_tokens for c in cases),
        "mean_llm_calls": mean(c.llm_calls for c in cases),
        "scorers": {
            name: {"mean": mean(s.value for s in ss), "pass_rate": sum(s.passed for s in ss) / len(ss)}
            for name, ss in by_scorer.items()
        },
    }


def _input_args(case: Case) -> tuple[tuple, dict]:
    """Dict inputs are passed as keyword arguments; anything else as one positional argument."""
    if isinstance(case.input, dict):
        return (), dict(case.input)
    return (case.input,), {}


def _score(case: Case, output: Any, scorers: Sequence[Scorer]) -> list[Score]:
    out = []
    for sc in scorers:
        try:
            out.append(sc(case, output))
        except Exception as e:  # a broken scorer fails the case instead of the whole run
            out.append(Score(getattr(sc, "name", type(sc).__name__), 0.0, False, f"scorer error: {e!r}"))
    return out


def _run_case(target: Callable[..., Any], case: Case, scorers: Sequence[Scorer], name: str) -> CaseResult:
    tracer = get_tracer()
    with tracer.start_as_current_span(f"eval {name}") as span:
        span.set_attribute("agent_eval.kind", "eval")
        span.set_attribute("agent_eval.eval.name", name)
        span.set_attribute("agent_eval.eval.case_id", case.id)
        args, kwargs = _input_args(case)
        output, error = None, None
        with collect_stats() as stats:  # only the agent's own calls; judge calls happen after
            t0 = time.perf_counter()
            try:
                output = target(*args, **kwargs)
                if inspect.isawaitable(output):
                    output = asyncio.run(output)  # type: ignore[arg-type]
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                span.record_exception(e)
                span.set_attribute("agent_eval.eval.traceback", traceback.format_exc(limit=5))
            latency_ms = (time.perf_counter() - t0) * 1000
        scores = _score(case, output, scorers) if error is None else []
        result = CaseResult(
            id=case.id,
            input=case.input,
            expected=case.expected,
            output=output,
            error=error,
            scores=scores,
            latency_ms=latency_ms,
            llm_calls=stats.llm_calls,
            input_tokens=stats.usage.input_tokens,
            output_tokens=stats.usage.output_tokens,
            cost_usd=stats.total_cost_usd,
            tags=case.tags,
        )
        span.set_attribute("agent_eval.eval.passed", result.passed)
        if error is not None:  # a failed score is a result, not an error
            span.set_status(Status(StatusCode.ERROR, error))
        return result


def run_eval(
    target: Callable[..., Any],
    dataset: Dataset | str | Path,
    scorers: Sequence[Scorer],
    *,
    name: str | None = None,
    concurrency: int = 1,
    metadata: dict[str, Any] | None = None,
    on_case: Callable[[CaseResult], None] | None = None,
) -> EvalReport:
    """Run ``target`` on every case and score the outputs.

    ``target`` receives the case input: a dict input becomes keyword arguments, anything else one argument.
    It may be sync or async. Exceptions are recorded as failed cases, not raised.
    ``concurrency`` runs cases in parallel threads (latency figures then include contention).
    """
    ds = dataset if isinstance(dataset, Dataset) else Dataset.load(dataset)
    if not scorers:
        raise ValueError("Give at least one scorer")
    name = name or ds.name
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()

    def one(case: Case) -> CaseResult:
        r = _run_case(target, case, scorers, name)
        if on_case:
            on_case(r)
        return r

    if concurrency <= 1:
        results = [one(c) for c in ds]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(contextvars.copy_context().run, one, c) for c in ds]
            results = [f.result() for f in futures]

    meta = {
        "git_sha": os.environ.get("GITHUB_SHA"),
        "git_ref": os.environ.get("GITHUB_REF_NAME"),
        "concurrency": concurrency,
        **(metadata or {}),
    }
    return EvalReport(
        name,
        ds.name,
        results,
        started,
        time.perf_counter() - t0,
        {k: v for k, v in meta.items() if v is not None},
    )

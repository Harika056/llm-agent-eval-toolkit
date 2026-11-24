"""A local SQLite span store: an OpenTelemetry exporter that writes spans, and a reader that summarises them."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from .stats import mean, percentile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
  span_id    TEXT PRIMARY KEY,
  trace_id   TEXT NOT NULL,
  parent_id  TEXT,
  name       TEXT NOT NULL,
  kind       TEXT,
  start_ns   INTEGER NOT NULL,
  end_ns     INTEGER NOT NULL,
  status     TEXT NOT NULL,
  attributes TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS spans_start ON spans(start_ns);
"""


def _hex(n: int, width: int) -> str:
    return format(n, f"0{width}x")


class SQLiteSpanExporter(SpanExporter):
    """Writes finished spans to a SQLite file so the dashboard works with no backend running."""

    def __init__(self, path: str | Path = "agent_eval.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rows = []
        for s in spans:
            ctx = s.get_span_context()
            attrs = dict(s.attributes or {})
            rows.append(
                (
                    _hex(ctx.span_id, 16),
                    _hex(ctx.trace_id, 32),
                    _hex(s.parent.span_id, 16) if s.parent else None,
                    s.name,
                    attrs.get("agent_eval.kind"),
                    s.start_time or 0,
                    s.end_time or 0,
                    "error" if s.status.status_code == StatusCode.ERROR else "ok",
                    json.dumps(attrs, default=str),
                )
            )
        try:
            with self._lock:
                self._conn.executemany("INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?)", rows)
                self._conn.commit()
        except sqlite3.Error:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        with self._lock:
            self._conn.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@dataclass
class SpanRecord:
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    kind: str | None
    start_ns: int
    end_ns: int
    status: str
    attributes: dict[str, Any]

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


@dataclass
class RequestSummary:
    """One agent request: a root span and everything under it."""

    trace_id: str
    name: str
    start_ns: int
    latency_ms: float
    status: str
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    tokens_per_second: float | None
    models: list[str] = field(default_factory=list)
    eval_passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceStore:
    """Reads what :class:`SQLiteSpanExporter` wrote."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No trace store at {self.path}")
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def spans(self, trace_id: str | None = None) -> list[SpanRecord]:
        q = "SELECT * FROM spans" + (" WHERE trace_id = ?" if trace_id else "") + " ORDER BY start_ns"
        cur = self._conn.execute(q, (trace_id,) if trace_id else ())
        return [
            SpanRecord(
                r["span_id"],
                r["trace_id"],
                r["parent_id"],
                r["name"],
                r["kind"],
                r["start_ns"],
                r["end_ns"],
                r["status"],
                json.loads(r["attributes"]),
            )
            for r in cur
        ]

    def requests(self, limit: int | None = None) -> list[RequestSummary]:
        by_trace: dict[str, list[SpanRecord]] = defaultdict(list)
        for s in self.spans():
            by_trace[s.trace_id].append(s)
        out: list[RequestSummary] = []
        for trace_id, spans in by_trace.items():
            ids = {s.span_id for s in spans}
            roots = [s for s in spans if s.parent_id is None or s.parent_id not in ids]
            root = min(roots, key=lambda s: s.start_ns)
            # judge calls made while scoring an eval are not part of the agent's own cost
            llm = [
                s for s in spans if s.kind == "llm" and s.attributes.get("gen_ai.operation.name") != "judge"
            ]
            inp = sum(int(s.attributes.get("gen_ai.usage.input_tokens", 0)) for s in llm)
            outp = sum(int(s.attributes.get("gen_ai.usage.output_tokens", 0)) for s in llm)
            costs = [s.attributes.get("agent_eval.cost_usd") for s in llm]
            known = [c for c in costs if c is not None]
            llm_secs = sum(s.duration_ms for s in llm) / 1000
            out.append(
                RequestSummary(
                    trace_id=trace_id,
                    name=(
                        f"{root.attributes.get('agent_eval.eval.case_id')} (eval {root.attributes.get('agent_eval.eval.name')})"
                        if root.kind == "eval"
                        else root.name
                    ),
                    start_ns=root.start_ns,
                    latency_ms=root.duration_ms,
                    status="error" if any(s.status == "error" for s in spans) else "ok",
                    llm_calls=len(llm),
                    input_tokens=inp,
                    output_tokens=outp,
                    cost_usd=sum(known) if known else None,
                    tokens_per_second=(outp / llm_secs) if llm_secs > 0 and outp else None,
                    models=sorted({str(s.attributes.get("gen_ai.request.model")) for s in llm}),
                    eval_passed=root.attributes.get("agent_eval.eval.passed")
                    if root.kind == "eval"
                    else None,
                )
            )
        out.sort(key=lambda r: r.start_ns, reverse=True)
        return out[:limit] if limit else out

    def summary(self) -> dict[str, Any]:
        reqs = self.requests()
        lat = [r.latency_ms for r in reqs]
        costs = [r.cost_usd for r in reqs if r.cost_usd is not None]
        per_model: dict[str, dict[str, Any]] = {}
        for s in self.spans():
            if s.kind != "llm":
                continue
            m = str(s.attributes.get("gen_ai.request.model"))
            d = per_model.setdefault(
                m,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "_lat": [], "_tps": []},
            )
            d["calls"] += 1
            d["input_tokens"] += int(s.attributes.get("gen_ai.usage.input_tokens", 0))
            d["output_tokens"] += int(s.attributes.get("gen_ai.usage.output_tokens", 0))
            d["cost_usd"] += float(s.attributes.get("agent_eval.cost_usd") or 0)
            d["_lat"].append(s.duration_ms)
            if s.attributes.get("agent_eval.tokens_per_second") is not None:
                d["_tps"].append(float(s.attributes["agent_eval.tokens_per_second"]))
        for d in per_model.values():
            d["p50_latency_ms"] = percentile(d["_lat"], 50)
            d["p95_latency_ms"] = percentile(d["_lat"], 95)
            d["mean_tokens_per_second"] = mean(d.pop("_tps"))
            d.pop("_lat")
        return {
            "requests": len(reqs),
            "errors": sum(r.status == "error" for r in reqs),
            "p50_latency_ms": percentile(lat, 50),
            "p95_latency_ms": percentile(lat, 95),
            "mean_cost_usd": mean(costs),
            "total_cost_usd": sum(costs) if costs else None,
            "input_tokens": sum(r.input_tokens for r in reqs),
            "output_tokens": sum(r.output_tokens for r in reqs),
            "mean_tokens_per_second": mean(r.tokens_per_second for r in reqs if r.tokens_per_second),
            "models": per_model,
        }

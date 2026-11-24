"""Static HTML dashboard: one self-contained file, no server, safe to upload as a CI artifact."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .stats import percentile
from .store import RequestSummary, SpanRecord, TraceStore

_CSS = """
:root{color-scheme:light dark;--bg:#f7f7f5;--panel:#fff;--ink:#1b1d21;--muted:#6a707a;--line:#e3e4e1;
--accent:#2f6fde;--accent2:#e0892c;--ok:#1f9d55;--bad:#d64545;--bar:#d7e3fa}
@media(prefers-color-scheme:dark){:root{--bg:#121417;--panel:#1a1d21;--ink:#e8eaed;--muted:#9aa1ab;--line:#2b3037;
--accent:#6ea0ff;--accent2:#f2a65a;--bar:#25324a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:28px 20px 60px}h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 12px}
.muted{color:var(--muted)}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:20px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.kpi b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.kpi span{color:var(--muted);font-size:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr));gap:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}td.n,th.n{text-align:right}td.why{white-space:normal;min-width:260px}
.ok{color:var(--ok)}.bad{color:var(--bad)}details summary{cursor:pointer;list-style:none}details summary::-webkit-details-marker{display:none}
.wf .n{text-align:right}.wf{display:grid;grid-template-columns:260px 1fr 90px;gap:4px 10px;padding:10px 4px;font-size:12px}
.wf .track{position:relative;height:14px;background:var(--line);border-radius:3px}.wf .seg{position:absolute;top:0;height:14px;border-radius:3px;background:var(--accent)}
.seg.llm{background:var(--accent2)}.seg.error{background:var(--bad)}.ind{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rrow{display:grid;grid-template-columns:minmax(180px,2.2fr) 1.6fr .8fr .7fr 1.2fr .7fr .8fr 1fr;gap:10px;padding:8px 10px;border-bottom:1px solid var(--line);
align-items:center;font-variant-numeric:tabular-nums;min-width:900px}.rrow .n{text-align:right}.rhead{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
details.rq[open] .ind{font-weight:600}details.rq summary:hover{background:var(--bg)}details.rq .wf{border-bottom:1px solid var(--line);min-width:900px}
svg text{fill:var(--muted);font-size:11px}.axis{stroke:var(--line)}.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
"""


def _e(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _ms(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


def _usd(v: float | None) -> str:
    if v is None:
        return "–"
    return f"${v:.4f}" if v < 1 else f"${v:,.2f}"


def _num(v: float | None, digits: int = 0) -> str:
    return "–" if v is None else f"{v:,.{digits}f}"


def _series_chart(
    values: list[float | None], title: str, fmt: Any, color: str, width: int = 560, height: int = 180
) -> str:
    """Bars per request, oldest on the left, with a p95 line."""
    vals = [v for v in values if v is not None]
    if not vals:
        return f'<div class="card"><b>{_e(title)}</b><p class="muted">No data yet.</p></div>'
    pad_l, pad_b, pad_t = 56, 20, 10
    w, h = width - pad_l - 8, height - pad_b - pad_t
    top = max(vals) * 1.1 or 1
    n = len(values)
    bw = max(1.0, w / n - 1)
    bars = []
    for i, v in enumerate(values):
        if v is None:
            continue
        bh = v / top * h
        x = pad_l + i * (w / n)
        bars.append(
            f'<rect x="{x:.1f}" y="{pad_t + h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" rx="1"><title>{_e(fmt(v))}</title></rect>'
        )
    p95 = percentile(vals, 95) or 0
    y95 = pad_t + h - p95 / top * h
    ticks = "".join(
        f'<line class="axis" x1="{pad_l}" x2="{pad_l + w}" y1="{pad_t + h - f * h:.1f}" y2="{pad_t + h - f * h:.1f}"/>'
        f'<text x="{pad_l - 6}" y="{pad_t + h - f * h + 4:.1f}" text-anchor="end">{_e(fmt(top * f))}</text>'
        for f in (0, 0.5, 1)
    )
    return (
        f'<div class="card"><b>{_e(title)}</b> <span class="muted">· {len(vals)} requests, oldest left · dashed line = p95 {_e(fmt(p95))}</span>'
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{_e(title)}">{ticks}{"".join(bars)}'
        f'<line x1="{pad_l}" x2="{pad_l + w}" y1="{y95:.1f}" y2="{y95:.1f}" stroke="var(--bad)" stroke-dasharray="4 3"/></svg></div>'
    )


def _waterfall(spans: list[SpanRecord]) -> str:
    if not spans:
        return ""
    t0 = min(s.start_ns for s in spans)
    total = max(s.end_ns for s in spans) - t0 or 1
    by_id = {s.span_id: s for s in spans}

    def depth(s: SpanRecord) -> int:
        d, cur = 0, s
        while cur.parent_id and cur.parent_id in by_id and d < 20:
            cur, d = by_id[cur.parent_id], d + 1
        return d

    rows = []
    for s in spans:
        left = (s.start_ns - t0) / total * 100
        width = max(0.4, (s.end_ns - s.start_ns) / total * 100)
        cls = "error" if s.status == "error" else ("llm" if s.kind == "llm" else "")
        extra = ""
        if s.kind == "llm":
            a = s.attributes
            extra = (
                f" · {a.get('gen_ai.usage.input_tokens', '?')}→{a.get('gen_ai.usage.output_tokens', '?')} tok"
            )
            if a.get("agent_eval.cost_usd") is not None:
                extra += f" · {_usd(a['agent_eval.cost_usd'])}"
        rows.append(
            f'<div class="ind" style="padding-left:{depth(s) * 14}px" title="{_e(s.name)}">{_e(s.name)}<span class="muted">{_e(extra)}</span></div>'
            f'<div class="track"><div class="seg {cls}" style="left:{left:.2f}%;width:{width:.2f}%"></div></div>'
            f'<div class="n muted">{_ms(s.duration_ms)}</div>'
        )
    return f'<div class="wf">{"".join(rows)}</div>'


def _requests_table(store: TraceStore, reqs: list[RequestSummary]) -> str:
    rows = []
    for r in reqs:
        when = datetime.fromtimestamp(r.start_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if r.status != "ok":
            status = '<span class="bad">error</span>'
        elif r.eval_passed is False:
            status = '<span class="bad">eval failed</span>'
        elif r.eval_passed:
            status = '<span class="ok">eval passed</span>'
        else:
            status = '<span class="ok">ok</span>'
        rows.append(
            f'<details class="rq"><summary class="rrow"><span class="ind" title="{_e(r.name)}">▸ {_e(r.name)}</span>'
            f'<span class="muted">{when}</span><span class="n">{_ms(r.latency_ms)}</span><span class="n">{r.llm_calls}</span>'
            f'<span class="n">{r.input_tokens:,} → {r.output_tokens:,}</span><span class="n">{_num(r.tokens_per_second, 1)}</span>'
            f'<span class="n">{_usd(r.cost_usd)}</span><span>{status}</span></summary>{_waterfall(store.spans(r.trace_id))}</details>'
        )
    head = (
        '<div class="rrow rhead"><span>Request</span><span>Started (UTC)</span><span class="n">Latency</span><span class="n">LLM calls</span>'
        '<span class="n">Tokens in → out</span><span class="n">Tok/s</span><span class="n">Cost</span><span>Status</span></div>'
    )
    return (
        f'<div class="card">{head}{"".join(rows)}<p class="muted">Click a request to see its steps and LLM calls. '
        f"Judge calls made while scoring evals are not counted in a request's cost.</p></div>"
    )


def _eval_section(report: dict[str, Any]) -> str:
    m = report["metrics"]
    acc = m.get("accuracy")
    sc_rows = "".join(
        f'<tr><td>{_e(n)}</td><td class="n">{v["pass_rate"]:.1%}</td><td class="n">{v["mean"]:.3f}</td></tr>'
        for n, v in m.get("scorers", {}).items()
    )
    fails = [c for c in report["cases"] if not c.get("passed")]
    fail_rows = "".join(
        f'<tr><td>{_e(c["id"])}</td><td class="why">{_e(c.get("error") or "; ".join(s["name"] + ": " + (s.get("reason") or "failed") for s in c["scores"] if not s["passed"]))[:240]}</td></tr>'
        for c in fails[:50]
    )
    return (
        f'<h2>Eval: {_e(report["name"])} <span class="muted">· {_e(report.get("started_at", ""))}</span></h2>'
        f'<div class="kpis"><div class="kpi"><b>{"–" if acc is None else f"{acc:.1%}"}</b><span>Accuracy ({m["passed"]}/{m["cases"]})</span></div>'
        f'<div class="kpi"><b>{_ms(m.get("p95_latency_ms"))}</b><span>p95 latency</span></div>'
        f'<div class="kpi"><b>{_usd(m.get("mean_cost_usd"))}</b><span>Cost per case</span></div>'
        f'<div class="kpi"><b>{m.get("errors", 0)}</b><span>Errors</span></div></div>'
        f'<div style="display:grid;grid-template-columns:minmax(260px,1fr) minmax(0,2fr);gap:12px;margin-top:12px"><div class="card"><table><tr><th>Scorer</th><th class="n">Pass rate</th><th class="n">Mean</th></tr>{sc_rows}</table></div>'
        f'<div class="card"><table><tr><th>Failing case</th><th>Why</th></tr>{fail_rows or "<tr><td colspan=2 class=muted>None</td></tr>"}</table></div></div>'
    )


def render_dashboard(
    store_path: str | Path | None,
    reports: list[dict[str, Any]] | None = None,
    *,
    title: str = "Agent observability",
    limit: int = 200,
) -> str:
    parts = []
    if store_path is not None:
        with TraceStore(store_path) as store:
            s = store.summary()
            reqs = store.requests(limit=limit)
            chrono = list(reversed(reqs))
            err_rate = (s["errors"] / s["requests"]) if s["requests"] else None
            parts.append(
                '<div class="kpis">'
                f'<div class="kpi"><b>{s["requests"]:,}</b><span>Requests</span></div>'
                f'<div class="kpi"><b>{_ms(s["p50_latency_ms"])}</b><span>p50 latency</span></div>'
                f'<div class="kpi"><b>{_ms(s["p95_latency_ms"])}</b><span>p95 latency</span></div>'
                f'<div class="kpi"><b>{_usd(s["mean_cost_usd"])}</b><span>Cost per request</span></div>'
                f'<div class="kpi"><b>{_usd(s["total_cost_usd"])}</b><span>Total cost</span></div>'
                f'<div class="kpi"><b>{_num(s["mean_tokens_per_second"], 1)}</b><span>Output tokens / s</span></div>'
                f'<div class="kpi"><b>{"–" if err_rate is None else f"{err_rate:.1%}"}</b><span>Error rate</span></div>'
                "</div>"
            )
            parts.append('<h2>Per request</h2><div class="grid2">')
            parts.append(_series_chart([r.latency_ms for r in chrono], "Latency", _ms, "var(--accent)"))
            parts.append(_series_chart([r.cost_usd for r in chrono], "Cost", _usd, "var(--accent2)"))
            parts.append("</div>")
            model_rows = "".join(
                f'<tr><td>{_e(name)}</td><td class="n">{d["calls"]:,}</td><td class="n">{d["input_tokens"]:,}</td>'
                f'<td class="n">{d["output_tokens"]:,}</td><td class="n">{_ms(d["p50_latency_ms"])}</td><td class="n">{_ms(d["p95_latency_ms"])}</td>'
                f'<td class="n">{_num(d["mean_tokens_per_second"], 1)}</td><td class="n">{_usd(d["cost_usd"])}</td></tr>'
                for name, d in sorted(s["models"].items(), key=lambda kv: -kv[1]["calls"])
            )
            parts.append(
                '<h2>Models</h2><div class="card"><table><tr><th>Model</th><th class="n">Calls</th><th class="n">Input tok</th>'
                '<th class="n">Output tok</th><th class="n">p50</th><th class="n">p95</th><th class="n">Tok/s</th><th class="n">Cost</th></tr>'
                f"{model_rows or '<tr><td colspan=8 class=muted>No LLM calls recorded</td></tr>'}</table></div>"
            )
            parts.append(f"<h2>Recent requests <span class=muted>· latest {len(reqs)}</span></h2>")
            parts.append(_requests_table(store, reqs))
    for r in reports or []:
        parts.append(_eval_section(r))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><main>"
        f'<h1>{_e(title)}</h1><div class="muted">Generated {generated} by Evaluation &amp; Observability Toolkit for LLM Agents · costs use the prices you configured</div>'
        f"{''.join(parts)}</main></body></html>"
    )


def write_dashboard(
    out: str | Path, store_path: str | Path | None, reports: list[dict[str, Any]] | None = None, **kw: Any
) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_dashboard(store_path, reports, **kw), encoding="utf-8")
    return p

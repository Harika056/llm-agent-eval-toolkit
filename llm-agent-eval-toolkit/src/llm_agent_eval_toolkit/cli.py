"""``agent-eval`` command line: eval, gate, agreement, dashboard, summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .agreement import compare_labels, judge_labels_from_report, load_labels
from .config import ConfigError, load_eval_config
from .dashboard import write_dashboard
from .evals.runner import CaseResult, EvalReport, run_eval
from .gate import Thresholds, evaluate_gate
from .store import TraceStore
from .tracing import configure, shutdown


def _append_step_summary(markdown: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")


def cmd_eval(a: argparse.Namespace) -> int:
    try:
        cfg = load_eval_config(a.config)
    except (ConfigError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.name:
        cfg["name"] = a.name
    traces = a.traces or cfg["traces"]
    configure(f"eval:{cfg['name']}", store=traces, pricing=cfg["pricing"])
    if cfg["setup"]:
        cfg["setup"]()

    def progress(r: CaseResult) -> None:
        if not a.quiet:
            mark = "PASS" if r.passed else "FAIL"
            why = "" if r.passed else f"  ({r.error or '; '.join(s.name for s in r.scores if not s.passed)})"
            print(f"  {mark}  {r.id}  {r.latency_ms:,.0f} ms{why}")

    print(f"Running {cfg['name']} …")
    report = run_eval(
        cfg["target"],
        cfg["dataset"],
        cfg["scorers"],
        name=cfg["name"],
        concurrency=a.concurrency or cfg["concurrency"],
        on_case=progress,
    )
    shutdown()
    out = report.save(a.out)
    m = report.metrics
    acc = m["accuracy"]
    print(
        f"\n{m['passed']}/{m['cases']} passed ({acc:.1%}) · p95 {m['p95_latency_ms'] or 0:,.0f} ms"
        + (
            f" · ${m['mean_cost_usd']:.5f}/case"
            if m["mean_cost_usd"] is not None
            else " · cost: no prices set"
        )
        + f"\nResults: {out}"
        + (f"\nTraces:  {traces}" if traces else "")
    )
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    current = json.loads(Path(a.results).read_text(encoding="utf-8"))
    baseline = None
    if a.baseline:
        bp = Path(a.baseline)
        if bp.exists():
            baseline = json.loads(bp.read_text(encoding="utf-8"))
        elif not a.allow_missing_baseline:
            print(
                f"error: baseline {bp} not found (use --allow-missing-baseline on the first run)",
                file=sys.stderr,
            )
            return 2
        else:
            print(f"note: no baseline at {bp}; checking absolute limits only")
    try:
        thresholds = Thresholds.from_file(a.thresholds) if a.thresholds else Thresholds()
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    result = evaluate_gate(current, thresholds, baseline)
    md = result.to_markdown(title=a.title)
    print(md)
    if a.markdown:
        Path(a.markdown).write_text(md, encoding="utf-8")
    _append_step_summary(md)
    return 0 if result.passed else 1


def cmd_agreement(a: argparse.Namespace) -> int:
    jp = Path(a.judge)
    if jp.suffix == ".json" and "cases" in (data := json.loads(jp.read_text(encoding="utf-8"))):
        judge = judge_labels_from_report(data, a.scorer)
    else:
        judge = load_labels(jp, id_key=a.id_key, label_key=a.label_key)
    human = load_labels(a.human, id_key=a.id_key, label_key=a.label_key)
    rep = compare_labels(judge, human)
    if a.json:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        print(rep.to_markdown())
        if rep.disagreements and not a.quiet:
            print("\nDisagreements:")
            for d in rep.disagreements[:25]:
                print(f"  {d['id']}: human={d['human']} judge={d['judge']}")
    _append_step_summary(rep.to_markdown())
    return 0


def cmd_dashboard(a: argparse.Namespace) -> int:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.report or []]
    if not a.traces and not reports:
        print("error: give --traces and/or --report", file=sys.stderr)
        return 2
    out = write_dashboard(a.out, a.traces, reports, title=a.title)
    print(f"Dashboard written to {out}")
    return 0


def cmd_summary(a: argparse.Namespace) -> int:
    with TraceStore(a.traces) as store:
        s = store.summary()
    if a.json:
        print(json.dumps(s, indent=2))
        return 0

    def f(v: float | None, fmt: str) -> str:
        return "–" if v is None else format(v, fmt)

    print(f"Requests:        {s['requests']}  (errors: {s['errors']})")
    print(f"Latency p50/p95: {f(s['p50_latency_ms'], ',.0f')} / {f(s['p95_latency_ms'], ',.0f')} ms")
    print(f"Cost/request:    ${f(s['mean_cost_usd'], '.5f')}   total ${f(s['total_cost_usd'], '.4f')}")
    print(
        f"Tokens in/out:   {s['input_tokens']:,} / {s['output_tokens']:,}   ({f(s['mean_tokens_per_second'], '.1f')} tok/s)"
    )
    for m, d in s["models"].items():
        print(f"  {m}: {d['calls']} calls, p95 {f(d['p95_latency_ms'], ',.0f')} ms, ${d['cost_usd']:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-eval", description="Evaluation & Observability Toolkit for LLM Agents."
    )
    p.add_argument("--version", action="version", version=f"llm-agent-eval-toolkit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("eval", help="run a golden-dataset eval from a YAML config")
    e.add_argument("config", help="eval config YAML (target, dataset, scorers)")
    e.add_argument("-o", "--out", default="agent-eval-results.json", help="where to write results JSON")
    e.add_argument("--traces", help="SQLite trace store (overrides the config)")
    e.add_argument("--concurrency", type=int, help="parallel cases (overrides the config)")
    e.add_argument("--name", help="run name shown in results and the dashboard (overrides the config)")
    e.add_argument("-q", "--quiet", action="store_true")
    e.set_defaults(fn=cmd_eval)

    g = sub.add_parser("gate", help="fail (exit 1) when results regress past thresholds")
    g.add_argument("results", help="results JSON from `agent-eval eval`")
    g.add_argument("--baseline", help="results JSON from the main branch")
    g.add_argument("--thresholds", help="thresholds YAML")
    g.add_argument(
        "--allow-missing-baseline", action="store_true", help="don't fail when the baseline file is missing"
    )
    g.add_argument("--markdown", help="also write the summary to this file (e.g. for a PR comment)")
    g.add_argument("--title", default="Agent eval gate")
    g.set_defaults(fn=cmd_gate)

    ag = sub.add_parser("agreement", help="Cohen's kappa between LLM-judge grades and human labels")
    ag.add_argument("--judge", required=True, help="eval results JSON, or a labels file")
    ag.add_argument("--human", required=True, help="human labels: .jsonl/.json/.yaml/.csv with id and label")
    ag.add_argument("--scorer", default="llm_judge", help="judge scorer name inside the results JSON")
    ag.add_argument("--id-key", default="id")
    ag.add_argument("--label-key", default="label")
    ag.add_argument("--json", action="store_true")
    ag.add_argument("-q", "--quiet", action="store_true")
    ag.set_defaults(fn=cmd_agreement)

    d = sub.add_parser("dashboard", help="write a static HTML dashboard")
    d.add_argument("--traces", help="SQLite trace store")
    d.add_argument("--report", action="append", help="eval results JSON (repeatable)")
    d.add_argument("-o", "--out", default="agent-eval-dashboard.html")
    d.add_argument("--title", default="Agent observability")
    d.set_defaults(fn=cmd_dashboard)

    s = sub.add_parser("summary", help="print latency, token and cost totals from a trace store")
    s.add_argument("traces")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_summary)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "build_parser", "EvalReport"]

"""End-to-end: the offline example through the real CLI."""

import json
import shutil
from pathlib import Path

import pytest

from llm_agent_eval_toolkit import tracing
from llm_agent_eval_toolkit.cli import main
from llm_agent_eval_toolkit.config import ConfigError, build_scorer, load_eval_config

EXAMPLE = Path(__file__).parent.parent / "examples" / "support_agent"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    d = tmp_path / "ex"
    shutil.copytree(EXAMPLE, d, ignore=shutil.ignore_patterns("*.db", "*.json", "*.html", "__pycache__"))
    monkeypatch.chdir(d)
    monkeypatch.delenv("AGENT_VARIANT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    previous = tracing._state.provider
    yield d
    tracing._state.provider = previous
    tracing.set_pricing({})


def test_full_loop_catches_regression(workdir, monkeypatch, capsys):
    assert main(["eval", "eval.yaml", "-o", "base.json", "-q"]) == 0
    base = json.loads((workdir / "base.json").read_text())
    assert base["metrics"]["accuracy"] == 1.0
    assert base["metrics"]["mean_cost_usd"] > 0

    # same code again: gate passes
    assert main(["eval", "eval.yaml", "-o", "same.json", "-q"]) == 0
    assert main(["gate", "same.json", "--baseline", "base.json", "--thresholds", "thresholds.yaml"]) == 0

    # a bad change: gate fails and writes the GitHub step summary
    summary = workdir / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("AGENT_VARIANT", "regressed")
    assert main(["eval", "eval.yaml", "-o", "pr.json", "-q"]) == 0
    capsys.readouterr()
    assert main(["gate", "pr.json", "--baseline", "base.json", "--thresholds", "thresholds.yaml"]) == 1
    out = capsys.readouterr().out
    assert "failed" in out and "Newly failing cases" in out
    assert "❌" in summary.read_text()

    assert main(["agreement", "--judge", "base.json", "--human", "human_labels.csv", "-q"]) == 0
    assert "Cohen's kappa" in capsys.readouterr().out

    assert main(["dashboard", "--traces", "agent_eval.db", "--report", "pr.json", "-o", "dash.html"]) == 0
    html = (workdir / "dash.html").read_text()
    assert "Agent observability" in html and "support_agent" in html and "Eval: support-agent" in html

    assert main(["summary", "agent_eval.db"]) == 0


def test_gate_missing_baseline(workdir, capsys):
    assert main(["eval", "eval.yaml", "-o", "r.json", "-q"]) == 0
    assert main(["gate", "r.json", "--baseline", "nope.json"]) == 2
    assert main(["gate", "r.json", "--baseline", "nope.json", "--allow-missing-baseline"]) == 0


def test_config_errors(workdir, tmp_path):
    bad = workdir / "bad.yaml"
    bad.write_text("target: agent:answer\ndataset: golden.jsonl\n")
    with pytest.raises(ConfigError, match="scorers"):
        load_eval_config(bad)
    with pytest.raises(ConfigError, match="Unknown scorer"):
        build_scorer({"type": "vibes"})
    with pytest.raises(ConfigError, match="judge"):
        build_scorer({"type": "llm_judge", "rubric": "x"})
    with pytest.raises(ConfigError, match="Bad options"):
        build_scorer({"type": "contains", "colour": "red"})
    assert main(["eval", str(bad)]) == 2

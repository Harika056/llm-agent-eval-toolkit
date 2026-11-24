import copy

import pytest

from llm_agent_eval_toolkit import Thresholds, cohen_kappa, compare_labels, evaluate_gate, interpret_kappa
from llm_agent_eval_toolkit.agreement import judge_labels_from_report, load_labels

# ---------- Cohen's kappa (reference values from scikit-learn's cohen_kappa_score) ----------


def test_kappa_matches_sklearn_binary():
    a = ["pass", "pass", "fail", "pass", "fail", "fail", "pass", "pass", "fail", "pass"]
    b = ["pass", "fail", "fail", "pass", "fail", "pass", "pass", "pass", "fail", "pass"]
    assert cohen_kappa(a, b) == pytest.approx(0.5833333333333333)


def test_kappa_matches_sklearn_weighted():
    h = [5, 4, 3, 5, 2, 1, 4, 5, 3, 2]
    j = [5, 5, 3, 4, 2, 2, 4, 5, 2, 2]
    assert cohen_kappa(h, j) == pytest.approx(0.48051948051948057)
    assert cohen_kappa(h, j, weights="quadratic") == pytest.approx(0.8850574712643678)
    assert cohen_kappa(h, j, weights="linear") == pytest.approx(0.7297297297297298)


def test_kappa_edge_cases():
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    assert cohen_kappa([1, 2, 1, 2], [2, 1, 2, 1]) == pytest.approx(-1.0)
    assert cohen_kappa(["a", "a", "b"], ["a", "a", "a"]) == 0.0  # one rater never varies
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 2])
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1, 2], labels=[1])


def test_interpret_kappa_bands():
    assert interpret_kappa(-0.1).startswith("poor")
    assert interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(0.75) == "substantial"
    assert interpret_kappa(0.9) == "almost perfect"


def test_compare_labels_report_and_unmatched():
    judge = {"a": "5", "b": "4", "c": "2", "x": "3"}
    human = {"a": "5", "b": "5", "c": "2", "y": "1"}
    r = compare_labels(judge, human)
    assert r.n == 3 and r.percent_agreement == pytest.approx(2 / 3)
    assert r.labels == ["2", "4", "5"]
    assert r.confusion[2] == [0, 1, 1]  # human 5: judge said 4 once, 5 once
    assert r.unmatched_judge == ["x"] and r.unmatched_human == ["y"]
    assert r.disagreements == [{"id": "b", "human": "5", "judge": "4"}]
    assert r.weighted_kappa is not None
    assert "Cohen's kappa" in r.to_markdown()
    with pytest.raises(ValueError):
        compare_labels({"a": 1}, {"b": 1})


def test_load_labels_csv_and_from_report(tmp_path):
    f = tmp_path / "h.csv"
    f.write_text("id,label\na,pass\nb,fail\n")
    assert load_labels(f) == {"a": "pass", "b": "fail"}
    report = {
        "cases": [{"id": "a", "scores": [{"name": "llm_judge", "label": "5"}, {"name": "x", "label": None}]}]
    }
    assert judge_labels_from_report(report) == {"a": "5"}
    with pytest.raises(ValueError):
        judge_labels_from_report(report, "other")


# ---------- CI gate ----------


def report(acc=0.9, p95=1000.0, cost=0.01, errors=0, passed_ids=("a", "b"), failed_ids=(), judge_rate=0.9):
    cases = [{"id": i, "passed": True} for i in passed_ids] + [{"id": i, "passed": False} for i in failed_ids]
    return {
        "metrics": {
            "cases": len(cases),
            "passed": len(passed_ids),
            "accuracy": acc,
            "errors": errors,
            "p95_latency_ms": p95,
            "mean_cost_usd": cost,
            "scorers": {"llm_judge": {"pass_rate": judge_rate, "mean": 0.8}},
        },
        "cases": cases,
    }


def test_gate_passes_identical_runs():
    r = report()
    res = evaluate_gate(r, Thresholds(min_accuracy=0.8), copy.deepcopy(r))
    assert res.passed and not res.newly_failing
    assert "passed" in res.to_markdown()


@pytest.mark.parametrize(
    "change,failing_check",
    [
        ({"acc": 0.85}, "Accuracy vs baseline"),
        ({"p95": 1300.0}, "p95 latency vs baseline"),
        ({"cost": 0.013}, "Cost per case vs baseline"),
        ({"errors": 1}, "Errors"),
        ({"judge_rate": 0.7}, "llm_judge pass rate (floor)"),
    ],
)
def test_gate_catches_each_regression(change, failing_check):
    t = Thresholds(
        max_accuracy_drop=0.02,
        max_p95_latency_increase=0.25,
        max_cost_increase=0.2,
        scorers={"llm_judge": {"min_pass_rate": 0.8}},
    )
    res = evaluate_gate(report(**change), t, report())
    assert not res.passed
    assert [c.name for c in res.failed_checks] == [failing_check]


def test_gate_within_limits_passes():
    t = Thresholds(max_accuracy_drop=0.02, max_p95_latency_increase=0.25, max_cost_increase=0.2)
    assert evaluate_gate(report(acc=0.89, p95=1200, cost=0.0119), t, report()).passed


def test_gate_newly_failing_and_fixed_cases():
    base = report(passed_ids=("a", "b"), failed_ids=("c",))
    cur = report(passed_ids=("b", "c"), failed_ids=("a",))
    res = evaluate_gate(cur, Thresholds(max_newly_failing_cases=0), base)
    assert res.newly_failing == ["a"] and res.fixed == ["c"]
    assert not res.passed
    assert "`a`" in res.to_markdown()


def test_gate_without_baseline_uses_absolute_limits_only():
    t = Thresholds(min_accuracy=0.95, max_p95_latency_ms=800, max_cost_per_case_usd=0.02)
    res = evaluate_gate(report(), t, None)
    assert {c.name for c in res.failed_checks} == {"Accuracy (floor)", "p95 latency (ceiling)"}
    assert "no baseline" in res.to_markdown()


def test_gate_skips_checks_with_missing_numbers():
    cur = report(cost=None)
    res = evaluate_gate(cur, Thresholds(max_cost_increase=0.1), report())
    cost_check = next(c for c in res.checks if c.name.startswith("Cost"))
    assert cost_check.skipped and cost_check.passed


def test_thresholds_from_dict_rejects_typos(tmp_path):
    with pytest.raises(ValueError, match="Unknown threshold"):
        Thresholds.from_dict({"min_acuracy": 0.9})
    f = tmp_path / "t.yaml"
    f.write_text("thresholds:\n  min_accuracy: 0.7\n  scorers: {llm_judge: {min_pass_rate: 0.5}}\n")
    t = Thresholds.from_file(f)
    assert t.min_accuracy == 0.7 and t.scorers["llm_judge"]["min_pass_rate"] == 0.5

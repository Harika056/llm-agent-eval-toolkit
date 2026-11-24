import asyncio
import json
import time

import pytest

import llm_agent_eval_toolkit as et
from llm_agent_eval_toolkit.evals import (
    Case,
    Dataset,
    DatasetError,
    EvalReport,
    JudgeParseError,
    parse_judge_reply,
)


@pytest.fixture(autouse=True)
def prices():
    et.set_pricing({"m": {"input_per_million": 1.0, "output_per_million": 1.0}})
    yield
    et.set_pricing({})


def case(expected=None, inp="q", **meta):
    return Case(id="c1", input=inp, expected=expected, metadata=meta)


# ---------- datasets ----------


def test_dataset_jsonl_yaml_and_errors(tmp_path):
    j = tmp_path / "g.jsonl"
    j.write_text(
        '{"id": "a", "input": "x", "expected": "y", "tags": "t1"}\n\n// comment\n{"input": "z", "difficulty": "hard"}\n'
    )
    ds = Dataset.load(j)
    assert [c.id for c in ds] == ["a", "2"]
    assert ds.cases[0].tags == ("t1",)
    assert ds.cases[1].metadata == {"difficulty": "hard"}
    assert len(ds.filter("t1")) == 1

    y = tmp_path / "g.yaml"
    y.write_text("cases:\n  - {id: a, input: {q: hi}}\n")
    assert Dataset.load(y).cases[0].input == {"q": "hi"}

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a"}\n')
    with pytest.raises(DatasetError, match="no 'input'"):
        Dataset.load(bad)
    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"id": "a", "input": 1}\n{"id": "a", "input": 2}\n')
    with pytest.raises(DatasetError, match="duplicate"):
        Dataset.load(dup)
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{nope\n")
    with pytest.raises(DatasetError, match=":1:"):
        Dataset.load(broken)


# ---------- rule-based scorers ----------


def test_exact_match_normalises_whitespace_and_case():
    s = et.ExactMatch()(case("Paris"), "  paris ")
    assert s.passed and s.value == 1.0
    assert not et.ExactMatch(case_sensitive=True)(case("Paris"), "paris").passed


def test_contains_all_any_and_fraction():
    c = case(["refund", "5-7 business days"])
    s = et.Contains()(c, "Refunds take 3 days")
    assert not s.passed and s.value == 0.5 and "5-7" in s.reason
    assert et.Contains(mode="any")(c, "Refunds take 3 days").passed
    assert et.Contains(["x"])(case(), "x marks").passed


def test_contains_reads_nested_field_and_expected_key():
    c = case({"must": ["Settings > Security"]})
    out = {"answer": {"text": "Go to Settings > Security."}}
    assert et.Contains(field="answer.text", expected_key="must")(c, out).passed


def test_not_contains_regex_numeric_and_json():
    assert not et.NotContains(["password"])(case(), "your password is hunter2").passed
    assert et.RegexMatch(r"\bINV-\d{4}\b")(case(), "see INV-2024").passed
    assert et.NumericClose(tolerance=0.001)(case(3.14159), "pi is about 3.14").passed is False
    assert et.NumericClose(tolerance=0.01)(case(3.14159), "pi is about 3.142").passed
    assert et.NumericClose(tolerance=0.1, relative=True)(case(100), 109).passed
    j = et.JsonFields()(
        case({"intent": "refund", "priority": "high"}), '{"intent": "refund", "priority": "low"}'
    )
    assert j.value == 0.5 and not j.passed and "priority" in j.reason
    assert not et.JsonFields()(case({"a": 1}), "not json").passed


def test_custom_scorer_bool_float_and_score():
    assert et.Custom(lambda c, o: True, name="ok")(case(), "").passed
    s = et.Custom(lambda c, o: 0.6, pass_threshold=0.5)(case(), "")
    assert s.passed and s.value == 0.6


# ---------- LLM judge ----------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("REASON: good\nSCORE: 4", (4, "good")),
        ("Some preamble.\nReason: covers it\nScore: **5**", (5, "covers it")),
        ('{"score": 2, "reason": "vague"}', (2, "vague")),
        ("SCORE: 1\nactually SCORE: 3", (3, "")),
    ],
)
def test_parse_judge_reply(reply, expected):
    assert parse_judge_reply(reply, 1, 5) == expected


@pytest.mark.parametrize("reply", ["I think it's fine", "SCORE: 9", '{"reason": "x"}'])
def test_parse_judge_reply_rejects(reply):
    with pytest.raises(JudgeParseError):
        parse_judge_reply(reply, 1, 5)


def test_llm_judge_scores_and_keeps_label():
    prompts = []

    def judge(p):
        prompts.append(p)
        return "REASON: correct\nSCORE: 4"

    s = et.LLMJudge(judge, "Is it correct?", pass_threshold=4)(
        case("Paris", inp="Capital of France?"), "Paris"
    )
    assert s.passed and s.label == "4" and s.value == pytest.approx(0.75)
    assert "Capital of France?" in prompts[0] and "Is it correct?" in prompts[0] and "Paris" in prompts[0]


def test_llm_judge_retries_then_fails_softly():
    replies = iter(["no idea", "SCORE: 5"])
    assert et.LLMJudge(lambda p: next(replies), "r", retries=1)(case(), "x").passed
    s = et.LLMJudge(lambda p: "garbage", "r", retries=2)(case(), "x")
    assert not s.passed and "unreadable" in s.reason


def test_llm_judge_default_threshold():
    assert et.LLMJudge(lambda p: "", "r", min_score=1, max_score=5).pass_threshold == 4
    with pytest.raises(ValueError):
        et.LLMJudge(lambda p: "", "r", min_score=5, max_score=5)


# ---------- runner ----------


def _agent(question):
    with et.llm_call("m") as c:
        c.record(input_tokens=100, output_tokens=20)
    if question == "boom":
        raise ValueError("tool failed")
    return question.upper()


def _paying_judge(prompt):
    with et.llm_call("m", operation="judge") as c:
        c.record(input_tokens=10_000, output_tokens=10_000)
    return "SCORE: 5"


def test_run_eval_metrics_errors_and_judge_cost_excluded():
    ds = Dataset.from_records(
        [
            {"id": "a", "input": "hi", "expected": "HI"},
            {"id": "b", "input": "yo", "expected": "NO"},
            {"id": "c", "input": "boom", "expected": "BOOM"},
        ]
    )
    rep = et.run_eval(_agent, ds, [et.ExactMatch(), et.LLMJudge(_paying_judge, "r")], name="t")
    m = rep.metrics
    assert (m["cases"], m["passed"], m["errors"]) == (3, 1, 1)
    assert m["accuracy"] == pytest.approx(1 / 3)
    a = rep.cases[0]
    assert a.passed and a.llm_calls == 1 and a.input_tokens == 100
    assert a.cost_usd == pytest.approx(120 / 1e6)  # judge's 20k tokens not included
    c = rep.cases[2]
    assert c.error == "ValueError: tool failed" and c.scores == [] and not c.passed
    assert m["scorers"]["exact_match"]["pass_rate"] == 0.5
    assert [f.id for f in rep.failures()] == ["b", "c"]


def test_run_eval_dict_inputs_become_kwargs_and_async_targets():
    async def target(question, lang="en"):
        await asyncio.sleep(0)
        return f"{question}-{lang}"

    ds = Dataset.from_records([{"input": {"question": "q", "lang": "te"}, "expected": "q-te"}])
    assert et.run_eval(target, ds, [et.ExactMatch()]).cases[0].passed


def test_run_eval_concurrency_keeps_per_case_stats_separate():
    def slow(n):
        with et.llm_call("m") as c:
            time.sleep(0.01)
            c.record(input_tokens=n, output_tokens=0)
        return n

    ds = Dataset.from_records([{"id": str(i), "input": i, "expected": i} for i in range(1, 9)])
    rep = et.run_eval(slow, ds, [et.Custom(lambda c, o: o == c.expected, name="eq")], concurrency=4)
    assert [c.input_tokens for c in rep.cases] == list(range(1, 9))
    assert rep.metrics["accuracy"] == 1.0


def test_broken_scorer_fails_case_not_run():
    def bad(c, o):
        raise KeyError("x")

    rep = et.run_eval(lambda q: q, Dataset.from_records([{"input": "a"}]), [et.Custom(bad, name="bad")])
    assert not rep.cases[0].passed and "scorer error" in rep.cases[0].scores[0].reason


def test_report_save_load_roundtrip(tmp_path):
    rep = et.run_eval(
        _agent, Dataset.from_records([{"id": "a", "input": "hi", "expected": "HI"}]), [et.ExactMatch()]
    )
    p = rep.save(tmp_path / "r.json")
    data = json.loads(p.read_text())
    assert data["metrics"]["accuracy"] == 1.0 and data["cases"][0]["passed"] is True
    again = EvalReport.load(p)
    assert again.metrics == rep.metrics


def test_run_eval_requires_scorers():
    with pytest.raises(ValueError):
        et.run_eval(lambda x: x, Dataset.from_records([{"input": 1}]), [])

import asyncio
from types import SimpleNamespace

import pytest

import llm_agent_eval_toolkit as et
from llm_agent_eval_toolkit import tracing
from llm_agent_eval_toolkit.store import TraceStore
from llm_agent_eval_toolkit.usage import Usage, extract_usage

PRICES = {"model-a": {"input_per_million": 1.0, "output_per_million": 4.0}}


@pytest.fixture(autouse=True)
def prices():
    et.set_pricing(PRICES)
    yield
    et.set_pricing({})


# ---------- usage ----------


def test_extract_usage_openai_chat_object():
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3))
    assert extract_usage(resp) == Usage(10, 3)


def test_extract_usage_anthropic_and_responses_api_dict():
    assert extract_usage({"usage": {"input_tokens": 7, "output_tokens": 2}}) == Usage(7, 2)


def test_extract_usage_gemini():
    resp = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=5))
    assert extract_usage(resp) == Usage(12, 5)


def test_extract_usage_missing_returns_none():
    assert extract_usage({"text": "hi"}) is None
    assert extract_usage(None) is None


def test_extract_usage_ignores_booleans():
    assert extract_usage({"input_tokens": True}) is None


# ---------- pricing ----------


def test_pricing_exact_prefix_and_unknown():
    p = et.Pricing(
        {
            "gpt-4o": {"input_per_million": 2.5, "output_per_million": 10},
            "gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.6},
        }
    )
    assert p.cost("gpt-4o", Usage(1_000_000, 0)) == pytest.approx(2.5)
    assert p.cost("gpt-4o-mini-2024-07-18", Usage(0, 1_000_000)) == pytest.approx(0.6)  # longest prefix wins
    assert p.cost("claude-x", Usage(1, 1)) is None


def test_pricing_from_file(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("models:\n  m: {input_per_million: 1, output_per_million: 2}\n")
    assert et.Pricing.from_file(f).cost("m", Usage(500_000, 500_000)) == pytest.approx(1.5)


def test_pricing_rejects_bad_entries():
    with pytest.raises(ValueError):
        et.Pricing({"m": {"input": 1}})
    with pytest.raises(ValueError):
        et.Pricing({"m": {"input_per_million": -1, "output_per_million": 1}})


# ---------- spans ----------


def test_llm_call_records_genai_attributes_and_cost(agent_eval_spans):
    with et.llm_call("model-a", provider="acme") as call:
        call.record({"usage": {"prompt_tokens": 1000, "completion_tokens": 500}, "model": "model-a-002"})
    (span,) = agent_eval_spans.llm_calls
    a = span.attributes
    assert span.name == "chat model-a"
    assert a["gen_ai.request.model"] == "model-a"
    assert a["gen_ai.response.model"] == "model-a-002"
    assert a["gen_ai.provider.name"] == "acme"
    assert a["gen_ai.usage.input_tokens"] == 1000
    assert a["gen_ai.usage.output_tokens"] == 500
    assert a["agent_eval.cost_usd"] == pytest.approx(0.003)  # 1000*1 + 500*4 per million
    assert a["agent_eval.latency_ms"] >= 0
    assert a["agent_eval.tokens_per_second"] > 0


def test_llm_call_without_usage_records_no_cost(agent_eval_spans):
    with et.llm_call("model-a"):
        pass
    a = agent_eval_spans.llm_calls[0].attributes
    assert "agent_eval.cost_usd" not in a
    assert a["agent_eval.usage_recorded"] is False


def test_record_explicit_counts_override_response(agent_eval_spans):
    with et.llm_call("model-a") as call:
        call.record({"usage": {"input_tokens": 1, "output_tokens": 1}}, output_tokens=9)
    assert agent_eval_spans.output_tokens == 9
    assert agent_eval_spans.input_tokens == 1


def test_llm_call_error_sets_status_and_reraises(agent_eval_spans):
    with pytest.raises(RuntimeError), et.llm_call("model-a"):
        raise RuntimeError("rate limited")
    span = agent_eval_spans.llm_calls[0]
    assert span.status.status_code.name == "ERROR"
    assert len([e for e in span.events if e.name == "exception"]) == 1


def test_agent_run_totals_and_nesting(agent_eval_spans):
    with et.agent_run("support", user_tier="pro"), et.step("retrieve"):
        for _ in range(2):
            with et.llm_call("model-a") as c:
                c.record(input_tokens=100, output_tokens=50)
    root = agent_eval_spans.of_kind("agent")[0]
    assert root.attributes["agent_eval.llm_calls"] == 2
    assert root.attributes["gen_ai.usage.input_tokens"] == 200
    assert root.attributes["agent_eval.cost_usd"] == pytest.approx(2 * (100 * 1 + 50 * 4) / 1e6)
    assert root.attributes["user_tier"] == "pro"
    assert agent_eval_spans.steps == ["retrieve"]
    llm = agent_eval_spans.llm_calls[0]
    step_span = agent_eval_spans.of_kind("step")[0]
    assert llm.parent.span_id == step_span.context.span_id
    assert step_span.parent.span_id == root.context.span_id


def test_collect_stats_works_without_tracing_configured():
    with et.collect_stats() as outer:
        with et.collect_stats() as inner, et.llm_call("model-a") as c:
            c.record(input_tokens=10, output_tokens=10)
        with et.llm_call("unpriced") as c:
            c.record(input_tokens=1, output_tokens=1)
    assert inner.llm_calls == 1
    assert outer.llm_calls == 2
    assert outer.unpriced_calls == 1
    assert outer.usage == Usage(11, 11)
    assert outer.total_cost_usd == pytest.approx(50 / 1e6)


def test_total_cost_is_none_when_nothing_priced():
    with et.collect_stats() as s, et.llm_call("unknown") as c:
        c.record(input_tokens=1, output_tokens=1)
    assert s.total_cost_usd is None


def test_decorators_sync_and_async(agent_eval_spans):
    @et.traced_llm("model-a", provider="acme")
    def call_model(prompt):
        return {"usage": {"input_tokens": len(prompt), "output_tokens": 2}}

    @et.traced_step()
    def tool(x):
        return x * 2

    @et.traced_agent("agent")
    async def agent(prompt):
        tool(1)
        return call_model(prompt)

    out = asyncio.run(agent("hello"))
    assert out["usage"]["input_tokens"] == 5
    assert agent_eval_spans.input_tokens == 5
    assert [s.name for s in agent_eval_spans.of_kind("agent")] == ["agent"]
    assert agent_eval_spans.steps == ["test_decorators_sync_and_async.<locals>.tool"]


def test_configure_writes_sqlite_store_and_summary(tmp_path):
    db = tmp_path / "t.db"
    previous = tracing._state.provider
    try:
        et.configure("svc", store=db, pricing=PRICES)
        for i in range(3):
            with et.agent_run("req"), et.llm_call("model-a") as c:
                c.record(input_tokens=100 * (i + 1), output_tokens=10)
        with et.agent_run("judged"):
            with et.llm_call("model-a") as c:
                c.record(input_tokens=100, output_tokens=10)
            with et.llm_call("model-a", operation="judge") as c:
                c.record(input_tokens=999, output_tokens=999)
        et.shutdown()
    finally:
        tracing._state.provider = previous
    with TraceStore(db) as store:
        reqs = store.requests()
        assert len(reqs) == 4
        judged = next(r for r in reqs if r.name == "judged")
        assert judged.llm_calls == 1 and judged.input_tokens == 100  # judge excluded from request cost
        s = store.summary()
        assert s["requests"] == 4 and s["errors"] == 0
        assert s["models"]["model-a"]["calls"] == 5
        assert s["p95_latency_ms"] >= s["p50_latency_ms"]


def test_trace_store_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        TraceStore(tmp_path / "nope.db")

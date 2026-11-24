"""Tracing for agents: one span per agent run, per step and per LLM call, recorded with OpenTelemetry.

LLM spans use the OpenTelemetry GenAI semantic conventions (``gen_ai.*``) so they show up correctly in
any OTel backend, plus ``agent_eval.*`` attributes for latency, tokens per second and cost.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from .pricing import Pricing
from .store import SQLiteSpanExporter
from .usage import Usage, extract_usage

F = TypeVar("F", bound=Callable[..., Any])

_INSTRUMENTATION = "llm_agent_eval_toolkit"


@dataclass
class RunStats:
    """Running totals for every LLM call made while it is active (used per agent run and per eval case)."""

    llm_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    llm_seconds: float = 0.0
    models: set[str] = field(default_factory=set)

    @property
    def total_cost_usd(self) -> float | None:
        """Total cost, or ``None`` if no call could be priced."""
        return None if self.llm_calls and self.unpriced_calls == self.llm_calls else self.cost_usd

    @property
    def tokens_per_second(self) -> float | None:
        if self.llm_seconds <= 0 or not self.usage.output_tokens:
            return None
        return self.usage.output_tokens / self.llm_seconds


@dataclass
class _State:
    provider: Any = None  # a TracerProvider, or None to use the global one
    pricing: Pricing = field(default_factory=Pricing)


_state = _State()
_active_stats: contextvars.ContextVar[tuple[RunStats, ...]] = contextvars.ContextVar(
    "agent_eval_stats", default=()
)


def configure(
    service_name: str = "llm-agent",
    *,
    store: str | Path | None = "agent_eval.db",
    pricing: Pricing | str | Path | dict | None = None,
    exporters: list[SpanExporter] | None = None,
    tracer_provider: TracerProvider | None = None,
    batch: bool = False,
) -> TracerProvider:
    """Set up tracing. Call once at startup.

    Args:
        service_name: shown as ``service.name`` on every span.
        store: SQLite file for the local dashboard; ``None`` to skip it.
        pricing: a :class:`Pricing`, a YAML file path, or a ``{model: {input_per_million, output_per_million}}``
            dict. Without it, tokens are still recorded but cost is not.
        exporters: extra OpenTelemetry exporters (e.g. OTLP to Jaeger, Grafana Tempo or Honeycomb).
        tracer_provider: reuse an existing provider instead of creating one; exporters are added to it.
        batch: export in the background (production). Default writes each span as it ends (dev, tests, CI).
    """
    if isinstance(pricing, (str, Path)):
        pricing = Pricing.from_file(pricing)
    elif isinstance(pricing, dict):
        pricing = Pricing(pricing)
    _state.pricing = pricing or Pricing()

    provider = tracer_provider or TracerProvider(resource=Resource.create({"service.name": service_name}))
    processor = BatchSpanProcessor if batch else SimpleSpanProcessor
    all_exporters = list(exporters or [])
    if store is not None:
        all_exporters.insert(0, SQLiteSpanExporter(store))
    for exp in all_exporters:
        provider.add_span_processor(processor(exp))
    _state.provider = provider
    return provider


def set_pricing(pricing: Pricing | str | Path | dict) -> None:
    if isinstance(pricing, (str, Path)):
        pricing = Pricing.from_file(pricing)
    elif isinstance(pricing, dict):
        pricing = Pricing(pricing)
    _state.pricing = pricing


def shutdown() -> None:
    """Flush and close exporters (call before the process exits when ``batch=True``)."""
    if _state.provider is not None and hasattr(_state.provider, "shutdown"):
        _state.provider.shutdown()
    _state.provider = None


def get_tracer() -> trace.Tracer:
    if _state.provider is not None:
        return _state.provider.get_tracer(_INSTRUMENTATION)
    return trace.get_tracer(_INSTRUMENTATION)


@contextmanager
def collect_stats() -> Iterator[RunStats]:
    """Collect totals for every LLM call inside the block, with or without tracing configured."""
    stats = RunStats()
    token = _active_stats.set((*_active_stats.get(), stats))
    try:
        yield stats
    finally:
        _active_stats.reset(token)


def _set_attrs(span: Span, attrs: dict[str, Any]) -> None:
    for k, v in attrs.items():
        if v is None:
            continue
        span.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else str(v))


@contextmanager
def agent_run(name: str = "agent.run", **attributes: Any) -> Iterator[Span]:
    """Root span for one agent request. Totals for its LLM calls are added to it when it ends."""
    with get_tracer().start_as_current_span(
        name, record_exception=True, set_status_on_exception=True
    ) as span:
        _set_attrs(span, {"agent_eval.kind": "agent", **attributes})
        with collect_stats() as stats:
            try:
                yield span
            finally:
                _set_attrs(
                    span,
                    {
                        "agent_eval.llm_calls": stats.llm_calls,
                        "gen_ai.usage.input_tokens": stats.usage.input_tokens,
                        "gen_ai.usage.output_tokens": stats.usage.output_tokens,
                        "agent_eval.cost_usd": stats.total_cost_usd,
                    },
                )


@contextmanager
def step(name: str, **attributes: Any) -> Iterator[Span]:
    """A span for one step of an agent: a tool call, retrieval, planning, parsing."""
    with get_tracer().start_as_current_span(
        name, record_exception=True, set_status_on_exception=True
    ) as span:
        _set_attrs(span, {"agent_eval.kind": "step", **attributes})
        yield span


class LLMCall:
    """Handle yielded by :func:`llm_call`; call :meth:`record` with the response or token counts."""

    def __init__(self, span: Span, model: str):
        self.span = span
        self.model = model
        self.response_model: str | None = None
        self.usage: Usage | None = None

    def record(
        self,
        response: Any = None,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        response_model: str | None = None,
    ) -> None:
        usage = extract_usage(response) if response is not None else None
        if input_tokens is not None or output_tokens is not None:
            usage = Usage(
                input_tokens if input_tokens is not None else (usage.input_tokens if usage else 0),
                output_tokens if output_tokens is not None else (usage.output_tokens if usage else 0),
            )
        self.usage = usage
        rm = response_model
        if rm is None and response is not None:
            rm = getattr(response, "model", None) or (
                response.get("model") if isinstance(response, dict) else None
            )
        self.response_model = rm if isinstance(rm, str) else None


@contextmanager
def llm_call(
    model: str, *, provider: str | None = None, operation: str = "chat", **attributes: Any
) -> Iterator[LLMCall]:
    """Span for one LLM request. Records latency, tokens, tokens/second and cost.

    Example::

        with llm_call("gpt-4o-mini", provider="openai") as call:
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
            call.record(resp)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        f"{operation} {model}",
        kind=trace.SpanKind.CLIENT,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        _set_attrs(
            span,
            {
                "agent_eval.kind": "llm",
                "gen_ai.operation.name": operation,
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider,
                **attributes,
            },
        )
        call = LLMCall(span, model)
        start = time.perf_counter()
        error: BaseException | None = None
        try:
            yield call
        except BaseException as e:
            error = e
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            elapsed = time.perf_counter() - start
            usage = call.usage or Usage()
            cost = _state.pricing.cost(call.response_model or model, usage) if call.usage else None
            tps = usage.output_tokens / elapsed if elapsed > 0 and usage.output_tokens else None
            _set_attrs(
                span,
                {
                    "gen_ai.response.model": call.response_model,
                    "gen_ai.usage.input_tokens": usage.input_tokens if call.usage else None,
                    "gen_ai.usage.output_tokens": usage.output_tokens if call.usage else None,
                    "agent_eval.latency_ms": round(elapsed * 1000, 3),
                    "agent_eval.tokens_per_second": round(tps, 3) if tps else None,
                    "agent_eval.cost_usd": cost,
                    "agent_eval.usage_recorded": call.usage is not None,
                },
            )
            if error is None or call.usage is not None:
                for s in _active_stats.get():
                    s.llm_calls += 1
                    s.usage = s.usage + usage
                    s.llm_seconds += elapsed
                    s.models.add(call.response_model or model)
                    if cost is None:
                        s.unpriced_calls += 1
                    else:
                        s.cost_usd += cost


def _wrap(fn: F, cm_factory: Callable[[tuple, dict], Any], on_result: Callable[[Any, Any], None] | None) -> F:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            with cm_factory(args, kwargs) as handle:
                result = await fn(*args, **kwargs)
                if on_result:
                    on_result(handle, result)
                return result

        return awrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with cm_factory(args, kwargs) as handle:
            result = fn(*args, **kwargs)
            if on_result:
                on_result(handle, result)
            return result

    return wrapper  # type: ignore[return-value]


def traced_agent(name: str | None = None, **attributes: Any) -> Callable[[F], F]:
    """Decorator: run the function inside :func:`agent_run`. Works on sync and async functions."""

    def deco(fn: F) -> F:
        return _wrap(fn, lambda a, k: agent_run(name or fn.__qualname__, **attributes), None)

    return deco


def traced_step(name: str | None = None, **attributes: Any) -> Callable[[F], F]:
    """Decorator: run the function inside :func:`step`."""

    def deco(fn: F) -> F:
        return _wrap(fn, lambda a, k: step(name or fn.__qualname__, **attributes), None)

    return deco


def traced_llm(model: str, *, provider: str | None = None, operation: str = "chat") -> Callable[[F], F]:
    """Decorator for a function that makes one LLM request and returns the provider's response.

    Token counts are read from the response (OpenAI, Anthropic and Gemini shapes are understood).
    """

    def deco(fn: F) -> F:
        return _wrap(
            fn,
            lambda a, k: llm_call(model, provider=provider, operation=operation),
            lambda call, result: call.record(result),
        )

    return deco

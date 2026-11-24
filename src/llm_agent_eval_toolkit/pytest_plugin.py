"""pytest plugin (auto-loaded once llm-agent-eval-toolkit is installed).

Fixture ``agent_eval_spans`` captures every span made during a test in memory, so you can assert on agent
behaviour: how many LLM calls it made, which tools it used, how many tokens it spent.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from . import tracing


class CapturedSpans:
    def __init__(self, exporter: InMemorySpanExporter):
        self._exporter = exporter

    @property
    def all(self) -> list[ReadableSpan]:
        return list(self._exporter.get_finished_spans())

    def of_kind(self, kind: str) -> list[ReadableSpan]:
        return [s for s in self.all if (s.attributes or {}).get("agent_eval.kind") == kind]

    @property
    def llm_calls(self) -> list[ReadableSpan]:
        return self.of_kind("llm")

    @property
    def steps(self) -> list[str]:
        return [s.name for s in self.of_kind("step")]

    def total(self, attribute: str) -> float:
        return sum(float((s.attributes or {}).get(attribute) or 0) for s in self.llm_calls)

    @property
    def input_tokens(self) -> int:
        return int(self.total("gen_ai.usage.input_tokens"))

    @property
    def output_tokens(self) -> int:
        return int(self.total("gen_ai.usage.output_tokens"))

    @property
    def cost_usd(self) -> float:
        return self.total("agent_eval.cost_usd")

    def clear(self) -> None:
        self._exporter.clear()


@pytest.fixture
def agent_eval_spans() -> Iterator[CapturedSpans]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous: Any = tracing._state.provider
    tracing._state.provider = provider
    try:
        yield CapturedSpans(exporter)
    finally:
        tracing._state.provider = previous
        provider.shutdown()

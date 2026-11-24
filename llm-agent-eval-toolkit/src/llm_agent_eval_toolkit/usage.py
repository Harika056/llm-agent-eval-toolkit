"""Token usage extraction that works across LLM SDKs without importing any of them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _first_int(obj: Any, *names: str) -> int | None:
    for n in names:
        v = _get(obj, n)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return int(v)
    return None


def extract_usage(response: Any) -> Usage | None:
    """Read token counts from a provider response (object or dict).

    Supported shapes:
      - OpenAI Chat Completions: ``usage.prompt_tokens`` / ``usage.completion_tokens``
      - OpenAI Responses API and Anthropic: ``usage.input_tokens`` / ``usage.output_tokens``
      - Google Gemini: ``usage_metadata.prompt_token_count`` / ``usage_metadata.candidates_token_count``
      - A bare usage object or dict with any of the above field names.
    Returns ``None`` when no token counts can be found.
    """
    candidates = [
        _get(response, "usage"),
        _get(response, "usage_metadata"),
        _get(response, "usageMetadata"),
        response,
    ]
    for c in candidates:
        if c is None:
            continue
        inp = _first_int(c, "input_tokens", "prompt_tokens", "prompt_token_count", "promptTokenCount")
        out = _first_int(
            c, "output_tokens", "completion_tokens", "candidates_token_count", "candidatesTokenCount"
        )
        if inp is not None or out is not None:
            return Usage(inp or 0, out or 0)
    return None

"""Per-model token prices. The toolkit ships no prices: providers change them, so you supply your own."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from .usage import Usage


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


class Pricing:
    """Maps model names to USD prices per one million tokens.

    Matching is exact first, then by longest prefix, so ``gpt-4o`` also prices ``gpt-4o-2024-08-06``.
    """

    def __init__(self, prices: Mapping[str, ModelPrice | Mapping[str, float]] | None = None):
        self._prices: dict[str, ModelPrice] = {}
        for model, p in (prices or {}).items():
            self.set(model, p)

    def set(self, model: str, price: ModelPrice | Mapping[str, float]) -> None:
        if not isinstance(price, ModelPrice):
            try:
                price = ModelPrice(float(price["input_per_million"]), float(price["output_per_million"]))
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"Price for {model!r} needs input_per_million and output_per_million (USD)"
                ) from e
        if price.input_per_million < 0 or price.output_per_million < 0:
            raise ValueError(f"Price for {model!r} cannot be negative")
        self._prices[model] = price

    @classmethod
    def from_file(cls, path: str | Path) -> Pricing:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        models = data.get("models", data)
        if not isinstance(models, Mapping):
            raise ValueError(f"{path}: expected a mapping of model name to prices")
        return cls(models)

    def lookup(self, model: str | None) -> ModelPrice | None:
        if not model:
            return None
        if model in self._prices:
            return self._prices[model]
        matches = [m for m in self._prices if model.startswith(m)]
        return self._prices[max(matches, key=len)] if matches else None

    def cost(self, model: str | None, usage: Usage) -> float | None:
        """USD cost of one call, or ``None`` when the model has no price."""
        p = self.lookup(model)
        if p is None:
            return None
        return (usage.input_tokens * p.input_per_million + usage.output_tokens * p.output_per_million) / 1e6

    def __contains__(self, model: str) -> bool:
        return self.lookup(model) is not None

    def __len__(self) -> int:
        return len(self._prices)

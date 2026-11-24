from __future__ import annotations

import math
from collections.abc import Iterable


def percentile(values: Iterable[float], p: float) -> float | None:
    """Linear-interpolated percentile (same method as numpy's default). ``None`` for no values."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(xs[lo])
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(values: Iterable[float]) -> float | None:
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None

"""Stability metric for continual-learning evaluation."""

import math
from typing import Optional


def stability_score(
    previous_capability_deviation: Optional[float],
    current_capability_deviation: float,
    stability_scale: float,
) -> float:
    """Measure round-to-round stability of capability deviation in [0, 1].

    A score of 1 means capability deviation did not change from the previous
    round. A score of 0 means the change was at least ``stability_scale``.
    The first round returns 0 because no previous round is available.
    Higher values indicate greater stability.
    """
    current = float(current_capability_deviation)
    scale = float(stability_scale)

    if not math.isfinite(current):
        raise ValueError("Current capability deviation must be finite.")
    if current < 0:
        raise ValueError("Current capability deviation must be non-negative.")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Stability scale must be a finite positive number.")

    if previous_capability_deviation is None:
        return 0.0

    previous = float(previous_capability_deviation)
    if not math.isfinite(previous):
        raise ValueError("Previous capability deviation must be finite.")
    if previous < 0:
        raise ValueError("Previous capability deviation must be non-negative.")

    change = abs(current - previous)
    return float(max(0.0, 1.0 - change / scale))

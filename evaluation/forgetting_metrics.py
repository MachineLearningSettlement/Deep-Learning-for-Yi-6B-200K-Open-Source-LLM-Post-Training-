"""Capability-deviation metric for continual-learning evaluation."""

import math


def capability_deviation(
    baseline_retention_loss: float,
    current_retention_loss: float,
) -> float:
    """Measure relative deterioration on retained capabilities.

    Improvements over the initial retention loss are assigned zero deviation.
    Lower values indicate less forgetting.
    """
    baseline = float(baseline_retention_loss)
    current = float(current_retention_loss)

    if not math.isfinite(baseline) or not math.isfinite(current):
        raise ValueError("Retention losses must be finite numbers.")
    if baseline <= 0:
        raise ValueError("Baseline retention loss must be positive.")
    if current < 0:
        raise ValueError("Current retention loss must be non-negative.")

    deterioration = (current - baseline) / baseline
    return float(max(0.0, deterioration))

"""Plasticity metric for continual-learning evaluation."""

import math


def plasticity_score(
    baseline_narrative_loss: float,
    current_narrative_loss: float,
) -> float:
    """Measure relative improvement in narrative-generation loss.

    Negative values indicate that narrative evaluation loss increased relative
    to the initial baseline. Higher values indicate greater adaptation.
    """
    baseline = float(baseline_narrative_loss)
    current = float(current_narrative_loss)

    if not math.isfinite(baseline) or not math.isfinite(current):
        raise ValueError("Narrative losses must be finite numbers.")
    if baseline <= 0:
        raise ValueError("Baseline narrative loss must be positive.")
    if current < 0:
        raise ValueError("Current narrative loss must be non-negative.")

    return float((baseline - current) / baseline)

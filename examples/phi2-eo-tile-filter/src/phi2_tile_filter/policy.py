from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quality_guard import InputQualityGuard


@dataclass(frozen=True)
class DecisionPolicy:
    """Policy for event downlink plus conservative uncertainty/input-quality fallback."""

    event_threshold: float
    min_confidence: float = 0.60
    temperature: float = 1.0
    input_quality_guard: InputQualityGuard | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.event_threshold <= 1.0):
            raise ValueError("event_threshold must be in [0, 1]")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

    def decide(
        self,
        *,
        prob_event: float,
        max_prob: float,
        inference_ok: bool = True,
        input_quality_ok: bool | None = True,
    ) -> tuple[bool, str]:
        if not inference_ok:
            return True, "inference_failure_fallback"
        if input_quality_ok is False:
            return True, "input_quality_fallback"
        if (
            not np.isfinite(prob_event)
            or not np.isfinite(max_prob)
            or not 0.0 <= prob_event <= 1.0
            or not 0.5 <= max_prob <= 1.0
            or not np.isclose(max_prob, max(prob_event, 1.0 - prob_event), rtol=1e-6, atol=1e-8)
        ):
            return True, "invalid_probability_fallback"
        if prob_event >= self.event_threshold:
            return True, "event"
        if max_prob < self.min_confidence:
            return True, "low_confidence_fallback"
        return False, "confident_background"


def _scaled_logits(logits: np.ndarray, *, temperature: float) -> np.ndarray:
    """Return temperature-scaled logits shifted to a non-positive range."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("expected logits with shape (N, 2)")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits contain non-finite values")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    # Scale down before subtraction, but shift before scaling up. This handles
    # both tiny temperatures and opposite-sign logits near the float64 limit.
    # Negative overflow represents a negligible exponential, never a NaN.
    with np.errstate(over="ignore", under="ignore"):
        if temperature >= 1.0:
            scaled = logits / temperature
            scaled -= np.max(scaled, axis=1, keepdims=True)
        else:
            scaled = (logits - np.max(logits, axis=1, keepdims=True)) / temperature
    return scaled


def softmax(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    scaled = _scaled_logits(logits, temperature=temperature)
    with np.errstate(under="ignore"):
        exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)

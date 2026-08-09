from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionPolicy:
    """Policy for event downlink plus conservative uncertainty fallback."""

    event_threshold: float
    min_confidence: float = 0.60
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.event_threshold <= 1.0):
            raise ValueError("event_threshold must be in [0, 1]")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

    def decide(self, *, prob_event: float, max_prob: float, inference_ok: bool = True) -> tuple[bool, str]:
        if not inference_ok:
            return True, "inference_failure_fallback"
        if not np.isfinite(prob_event) or not np.isfinite(max_prob):
            return True, "invalid_probability_fallback"
        if prob_event >= self.event_threshold:
            return True, "event"
        if max_prob < self.min_confidence:
            return True, "low_confidence_fallback"
        return False, "confident_background"


def softmax(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("expected logits with shape (N, 2)")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits contain non-finite values")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    scaled = logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)

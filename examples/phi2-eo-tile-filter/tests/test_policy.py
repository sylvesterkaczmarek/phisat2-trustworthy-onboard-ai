from __future__ import annotations

import numpy as np
import pytest

from phi2_tile_filter.policy import DecisionPolicy, softmax


def test_policy_keeps_event() -> None:
    policy = DecisionPolicy(event_threshold=0.8, min_confidence=0.6)
    assert policy.decide(prob_event=0.9, max_prob=0.9) == (True, "event")


def test_policy_keeps_low_confidence_background() -> None:
    policy = DecisionPolicy(event_threshold=0.8, min_confidence=0.7)
    assert policy.decide(prob_event=0.4, max_prob=0.6) == (True, "low_confidence_fallback")


def test_policy_discards_only_confident_background() -> None:
    policy = DecisionPolicy(event_threshold=0.8, min_confidence=0.6)
    assert policy.decide(prob_event=0.1, max_prob=0.9) == (False, "confident_background")


def test_policy_keeps_inference_failure() -> None:
    policy = DecisionPolicy(event_threshold=0.8)
    assert policy.decide(prob_event=np.nan, max_prob=np.nan, inference_ok=False) == (
        True,
        "inference_failure_fallback",
    )


def test_softmax_temperature_validation() -> None:
    with pytest.raises(ValueError):
        softmax(np.zeros((1, 2)), temperature=0.0)

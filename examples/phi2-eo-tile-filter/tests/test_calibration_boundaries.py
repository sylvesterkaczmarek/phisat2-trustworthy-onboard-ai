from __future__ import annotations

import math

import numpy as np
import pytest

from phi2_tile_filter.calibrate_threshold import (
    _nll,
    calibrate,
    clopper_pearson_lower_bound,
    fit_temperature,
)
from phi2_tile_filter.policy import DecisionPolicy, softmax
from phi2_tile_filter.quality_guard import (
    InputQualityGuard,
    calibrate_input_quality_guard,
    input_quality_features,
)
from phi2_tile_filter.validate_models import validate_models


def test_temperature_fitting_does_not_clip_confidently_wrong_predictions() -> None:
    logits = np.asarray([[1000.0, 0.0], [10.0, 0.0]])
    labels = np.asarray([1, 0])
    # The former clipped objective chose 0.25, with a true NLL of 2000.
    assert _nll(logits, labels, 0.25) == pytest.approx(2000.0)
    assert _nll(logits, labels, 4.0) == pytest.approx(125.03944486714627)
    assert fit_temperature(logits, labels) == 4.0


def test_log_space_nll_matches_ordinary_binary_likelihood() -> None:
    logits = np.asarray([[1.5, -0.5], [-1.0, 1.0], [0.0, 0.0]])
    labels = np.asarray([0, 0, 1])
    probabilities = softmax(logits, temperature=2.0)
    expected = -np.mean(np.log(probabilities[np.arange(3), labels]))
    assert _nll(logits, labels, 2.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("logits", "labels"),
    [
        (np.empty((0, 2)), np.asarray([], dtype=int)),
        (np.zeros((2, 2)), np.asarray([0])),
        (np.zeros((2, 2)), np.asarray([0, -1])),
        (np.zeros((2, 2)), np.asarray([0.0, 1.0])),
        (np.zeros((2, 2)), np.asarray([True, False])),
    ],
)
def test_temperature_fitting_rejects_invalid_evidence(logits, labels) -> None:
    with pytest.raises(ValueError, match="temperature fitting"):
        fit_temperature(logits, labels)


@pytest.mark.parametrize("temperature", [0.25, np.finfo(np.float64).tiny, np.nextafter(0.0, 1.0)])
def test_softmax_handles_large_equal_logits_and_small_temperature(temperature) -> None:
    logits = np.full((2, 2), np.finfo(np.float64).max)
    logits[1] *= -1
    with np.errstate(all="raise"):
        probabilities = softmax(logits, temperature=temperature)
    np.testing.assert_array_equal(probabilities, np.full((2, 2), 0.5))


def test_softmax_scales_down_before_opposite_sign_subtraction() -> None:
    largest = np.finfo(np.float64).max
    logits = np.asarray([[largest, -largest]])
    with np.errstate(all="raise"):
        probabilities = softmax(logits, temperature=largest)
    np.testing.assert_allclose(probabilities, [[1.0 / (1.0 + math.exp(-2.0)), 1.0 / (1.0 + math.exp(2.0))]])


def test_softmax_handles_negligible_opposite_class() -> None:
    largest = np.finfo(np.float64).max
    with np.errstate(all="raise"):
        probabilities = softmax(np.asarray([[largest, -largest]]), temperature=0.25)
    np.testing.assert_array_equal(probabilities, [[1.0, 0.0]])


@pytest.mark.parametrize(
    ("event", "confidence"),
    [(-0.1, 0.9), (0.1, 1.1), (0.1, 0.1), (0.1, 0.8), (1.1, 1.1)],
)
def test_invalid_probabilities_never_authorise_discard(event, confidence) -> None:
    assert DecisionPolicy(0.8).decide(prob_event=event, max_prob=confidence) == (
        True, "invalid_probability_fallback"
    )


@pytest.mark.parametrize(("successes", "trials"), [(True, 10), (1, True), (1.5, 10), (1, 10.5)])
def test_exact_binomial_bound_requires_integer_counts(successes, trials) -> None:
    with pytest.raises(ValueError, match="integer"):
        clopper_pearson_lower_bound(successes, trials)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("name", [
    "max_accuracy_drop", "max_event_recall_drop", "max_event_fnr_increase",
    "max_pr_auc_drop", "max_event_retention_recall_drop", "max_event_score_drift",
])
def test_validation_rejects_nonfinite_tolerances_before_loading_models(name, value) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_models("missing-fp32", "missing-int8", "missing-data", "missing-policy", **{name: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_calibration_rejects_nonfinite_guard_margin_before_loading_model(value) -> None:
    with pytest.raises(ValueError, match="quality_guard_margin"):
        calibrate("missing-model", "missing-data", quality_guard_margin=value)


@pytest.mark.parametrize("shape", [(0, 2, 2), (2, 0, 2), (2, 2, 0)])
def test_quality_guard_rejects_empty_inputs_without_invalid_statistics(shape) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        input_quality_features(np.empty(shape, dtype=np.float32))


@pytest.mark.parametrize(("key", "value"), [
    ("schema_version", True),
    ("calibration_samples", True),
    ("calibration_samples", 1.5),
    ("calibration_samples", "2"),
    ("calibration_score_median", float("nan")),
    ("calibration_score_max", float("inf")),
    ("calibration_score_max", -1.0),
    ("threshold", True),
    ("center", [False] * 6),
])
def test_quality_guard_rejects_invalid_calibration_metadata(key, value) -> None:
    guard = calibrate_input_quality_guard([np.zeros((1, 2, 2), dtype=np.float32)])
    payload = guard.to_payload()
    payload[key] = value
    with pytest.raises(ValueError):
        InputQualityGuard.from_payload(payload)


def test_quality_guard_fails_before_emitting_infinite_score() -> None:
    guard = calibrate_input_quality_guard([np.zeros((1, 2, 2), dtype=np.float32)])
    payload = guard.to_payload()
    payload["scale"] = [1e-300] * len(guard.scale)
    tiny_scale_guard = InputQualityGuard.from_payload(payload)
    with np.errstate(all="raise"), pytest.raises(ValueError, match="finite numerical range"):
        tiny_scale_guard.assess(np.ones((1, 2, 2), dtype=np.float32))


def test_tied_score_population_bound_has_nominal_coverage() -> None:
    from phi2_tile_filter.calibrate_threshold import _event_selection_rank, _event_threshold

    # Enumerate every possible sample from a three-point population. This is an
    # exact coverage calculation, not a Monte Carlo estimate. The old bound
    # based on observed captures fails 6.55% of the time at 95% confidence.
    samples = 20
    support = np.asarray([0.1, 0.5, 0.9])
    mass = np.asarray([0.15, 0.07, 0.78])
    rank = _event_selection_rank(samples, 0.95)
    fixed_rank_bound = clopper_pearson_lower_bound(rank, samples)
    old_violation_probability = 0.0
    new_violation_probability = 0.0
    for low_count in range(samples + 1):
        for mid_count in range(samples - low_count + 1):
            counts = (low_count, mid_count, samples - low_count - mid_count)
            scores = np.repeat(support, counts)
            threshold = _event_threshold(scores, np.ones(samples, dtype=int), 0.95)
            population_recall = float(mass[support >= threshold].sum())
            captures = int((scores >= threshold).sum())
            probability = math.factorial(samples)
            for count, frequency in zip(counts, mass):
                probability *= float(frequency) ** count / math.factorial(count)
            if clopper_pearson_lower_bound(captures, samples) > population_recall:
                old_violation_probability += probability
            if fixed_rank_bound > population_recall:
                new_violation_probability += probability
    assert old_violation_probability == pytest.approx(0.06548459212629187)
    assert new_violation_probability == pytest.approx(0.046145272065469214)
    assert new_violation_probability <= 0.05 < old_violation_probability


def test_calibration_does_not_use_ties_to_inflate_acceptance(monkeypatch) -> None:
    import importlib
    from pathlib import Path
    from types import SimpleNamespace

    module = importlib.import_module("phi2_tile_filter.calibrate_threshold")
    runner = SimpleNamespace(
        model_sha256="a" * 64,
        input_schema_sha256="b" * 64,
        band_ids=("band_01",),
        input_schema={"preprocessing": {"version": 1}},
        spec=SimpleNamespace(bands=1, size=2),
        assert_data_schema=lambda path: None,
        logits_for_array=lambda array: (np.zeros((1, 2)), 0.0),
    )
    monkeypatch.setattr(module, "OnnxRunner", lambda path: runner)
    monkeypatch.setattr(module, "discover_labeled_tiles", lambda root: [
        (Path(f"event-{index}.npy"), 1) for index in range(20)
    ] + [(Path("background.npy"), 0)])
    monkeypatch.setattr(module, "load_tile_numpy", lambda *args, **kwargs: np.zeros((1, 2, 2), dtype=np.float32))
    result = calibrate("model", "data", fit_temp=False, min_event_recall_lower_bound=0.8)
    stats = result["calibration_statistics"]
    assert result["event_threshold"] == 0.5
    assert stats["event_captures"] == 20
    assert stats["empirical_event_recall"] == 1.0
    assert stats["event_recall_bound_selection_rank"] == 19
    assert stats["event_recall_bound_method"] == "order-statistic-one-sided-exact"
    assert stats["event_recall_lower_bound"] == pytest.approx(0.7838938357931526)
    assert result["calibration_acceptance"]["accepted"] is False

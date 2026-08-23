from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from .policy import DecisionPolicy, softmax
from .runtime import OnnxRunner
from .utils import discover_labeled_tiles, load_tile_numpy


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = softmax(logits, temperature=temperature)
    selected = probs[np.arange(labels.size), labels]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Deterministic one-dimensional grid search for temperature scaling."""
    grid = np.geomspace(0.25, 4.0, 121)
    losses = np.array([_nll(logits, labels, float(t)) for t in grid])
    return float(grid[int(np.argmin(losses))])


def clopper_pearson_lower_bound(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> float:
    """One-sided exact Clopper-Pearson lower confidence bound for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence_level
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def _event_threshold(scores: np.ndarray, labels: np.ndarray, target_recall: float) -> float:
    positive_scores = np.sort(scores[labels == 1])[::-1]
    if positive_scores.size == 0:
        raise ValueError("calibration set contains no event examples")
    required = max(1, int(math.ceil(target_recall * positive_scores.size)))
    return float(positive_scores[required - 1])


def calibrate(
    model_path: str | Path,
    data_root: str | Path,
    *,
    target_recall: float = 0.95,
    min_confidence: float = 0.60,
    fit_temp: bool = True,
    confidence_level: float = 0.95,
    min_event_recall_lower_bound: float | None = None,
) -> dict:
    if not (0.0 < target_recall <= 1.0):
        raise ValueError("target_recall must be in (0, 1]")
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence must be in [0, 1]")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1)")
    if min_event_recall_lower_bound is not None and not (
        0.0 <= min_event_recall_lower_bound <= 1.0
    ):
        raise ValueError("min_event_recall_lower_bound must be in [0, 1]")

    runner = OnnxRunner(model_path)
    runner.assert_data_schema(data_root)
    items = discover_labeled_tiles(data_root)
    if not items:
        raise ValueError("calibration set is empty")
    labels: list[int] = []
    logits_list: list[np.ndarray] = []
    for path, cls in items:
        array = load_tile_numpy(path, input_schema=runner.input_schema)
        logits, _ = runner.logits_for_array(array)
        logits_list.append(logits[0])
        labels.append(cls)
    y = np.asarray(labels, dtype=np.int64)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("calibration set must contain both classes")

    logits = np.stack(logits_list)
    temperature = fit_temperature(logits, y) if fit_temp else 1.0
    probabilities = softmax(logits, temperature=temperature)
    scores = probabilities[:, 1]
    threshold = _event_threshold(scores, y, target_recall)
    predicted_event = (scores >= threshold).astype(np.int64)

    event_count = int(np.sum(y == 1))
    background_count = int(np.sum(y == 0))
    event_captures = int(np.sum((y == 1) & (predicted_event == 1)))
    empirical_recall = float(recall_score(y, predicted_event, pos_label=1, zero_division=0))
    recall_lower_bound = clopper_pearson_lower_bound(
        event_captures,
        event_count,
        confidence_level=confidence_level,
    )
    precision = float(precision_score(y, predicted_event, pos_label=1, zero_division=0))
    auc = float(roc_auc_score(y, scores))
    policy = DecisionPolicy(threshold, min_confidence=min_confidence, temperature=temperature)

    accepted = (
        min_event_recall_lower_bound is None
        or recall_lower_bound >= min_event_recall_lower_bound
    )
    result = {
        "schema_version": 4,
        "split_role": "calibration",
        "model_sha256": runner.model_sha256,
        "input_schema_sha256": runner.input_schema_sha256,
        "input_band_ids": list(runner.band_ids),
        "preprocessing_version": runner.input_schema["preprocessing"]["version"],
        "bands": runner.spec.bands,
        "size": runner.spec.size,
        "event_threshold": policy.event_threshold,
        "min_confidence": policy.min_confidence,
        "temperature": policy.temperature,
        "temperature_fitted": bool(fit_temp),
        "calibration_statistics": {
            "samples_total": int(y.size),
            "event_samples": event_count,
            "background_samples": background_count,
            "target_event_recall_for_threshold_selection": float(target_recall),
            "empirical_event_recall": empirical_recall,
            "event_captures": event_captures,
            "event_precision_at_threshold": precision,
            "roc_auc": auc,
            "event_recall_lower_bound": recall_lower_bound,
            "event_recall_confidence_level": float(confidence_level),
            "event_recall_bound_method": "clopper-pearson-one-sided-exact",
        },
        "calibration_acceptance": {
            "required_min_event_recall_lower_bound": (
                None
                if min_event_recall_lower_bound is None
                else float(min_event_recall_lower_bound)
            ),
            "accepted": bool(accepted),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data", required=True, help="Calibration split directory")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-event-recall-lower-bound", type=float, default=None)
    parser.add_argument("--no-fit-temperature", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("calibration.json"))
    args = parser.parse_args()
    result = calibrate(
        args.onnx,
        args.data,
        target_recall=args.target_recall,
        min_confidence=args.min_confidence,
        fit_temp=not args.no_fit_temperature,
        confidence_level=args.confidence_level,
        min_event_recall_lower_bound=args.min_event_recall_lower_bound,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["calibration_acceptance"]["accepted"] is not True:
        observed = result["calibration_statistics"]["event_recall_lower_bound"]
        required = result["calibration_acceptance"]["required_min_event_recall_lower_bound"]
        raise RuntimeError(
            f"calibration recall lower bound {observed:.4f} is below required {required:.4f}"
        )


if __name__ == "__main__":
    main()

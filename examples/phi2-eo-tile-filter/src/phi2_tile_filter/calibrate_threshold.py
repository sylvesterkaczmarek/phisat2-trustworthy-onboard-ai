from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
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
) -> dict:
    if not (0.0 < target_recall <= 1.0):
        raise ValueError("target_recall must be in (0, 1]")
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence must be in [0, 1]")

    runner = OnnxRunner(model_path)
    items = discover_labeled_tiles(data_root)
    if not items:
        raise ValueError("calibration set is empty")
    labels: list[int] = []
    logits_list: list[np.ndarray] = []
    for path, cls in items:
        array = load_tile_numpy(path, bands=runner.spec.bands, size=runner.spec.size)
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
    achieved_recall = float(recall_score(y, predicted_event, pos_label=1, zero_division=0))
    precision = float(precision_score(y, predicted_event, pos_label=1, zero_division=0))
    auc = float(roc_auc_score(y, scores))
    policy = DecisionPolicy(threshold, min_confidence=min_confidence, temperature=temperature)
    result = {
        "schema_version": 2,
        "model_sha256": runner.model_sha256,
        "bands": runner.spec.bands,
        "size": runner.spec.size,
        "calibration_samples": int(y.size),
        "event_samples": int(np.sum(y == 1)),
        "background_samples": int(np.sum(y == 0)),
        "target_event_recall": float(target_recall),
        "event_threshold": policy.event_threshold,
        "achieved_event_recall": achieved_recall,
        "event_precision_at_threshold": precision,
        "auc_roc": auc,
        "temperature": policy.temperature,
        "min_confidence": policy.min_confidence,
        "temperature_fitted": bool(fit_temp),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data", required=True, help="Calibration split directory")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--no-fit-temperature", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("calibration.json"))
    args = parser.parse_args()
    result = calibrate(
        args.onnx,
        args.data,
        target_recall=args.target_recall,
        min_confidence=args.min_confidence,
        fit_temp=not args.no_fit_temperature,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

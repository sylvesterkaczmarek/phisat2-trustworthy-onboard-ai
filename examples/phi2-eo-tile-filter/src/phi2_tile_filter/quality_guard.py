from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

QUALITY_GUARD_SCHEMA_VERSION = 1
QUALITY_GUARD_METHOD = "diagonal-standardized-input-statistics-v1"


def input_quality_features(array_chw: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract lightweight input statistics for distribution-shift screening.

    The features deliberately use only input pixels, not model activations, so the
    guard stays cheap and model-architecture independent. They are diagnostics,
    not a physical sensor model.
    """
    x = np.asarray(array_chw)
    if x.ndim != 3:
        raise ValueError(f"quality guard expects CHW input, got shape {x.shape}")
    if x.dtype != np.float32:
        raise ValueError(f"quality guard expects float32 input, got {x.dtype}")
    if not np.all(np.isfinite(x)):
        raise ValueError("quality guard input contains non-finite values")

    means = x.mean(axis=(1, 2), dtype=np.float64)
    stds = x.std(axis=(1, 2), dtype=np.float64)
    low_fraction = float(np.mean(x <= 0.01))
    high_fraction = float(np.mean(x >= 0.99))
    horizontal_tv = float(np.mean(np.abs(np.diff(x.astype(np.float64), axis=2)))) if x.shape[2] > 1 else 0.0
    vertical_tv = float(np.mean(np.abs(np.diff(x.astype(np.float64), axis=1)))) if x.shape[1] > 1 else 0.0

    values = np.concatenate(
        [means, stds, np.asarray([low_fraction, high_fraction, horizontal_tv, vertical_tv])]
    ).astype(np.float64, copy=False)
    names = tuple(
        [f"band_{index:02d}_mean" for index in range(x.shape[0])]
        + [f"band_{index:02d}_std" for index in range(x.shape[0])]
        + ["low_fraction", "high_fraction", "horizontal_total_variation", "vertical_total_variation"]
    )
    return values, names


@dataclass(frozen=True)
class InputQualityAssessment:
    score: float
    threshold: float
    in_distribution: bool
    method: str = QUALITY_GUARD_METHOD


@dataclass(frozen=True)
class InputQualityGuard:
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    threshold: float
    threshold_quantile: float
    threshold_margin: float
    calibration_samples: int
    calibration_score_median: float
    calibration_score_max: float
    method: str = QUALITY_GUARD_METHOD

    def __post_init__(self) -> None:
        if self.method != QUALITY_GUARD_METHOD:
            raise ValueError("unsupported input quality guard method")
        if not self.feature_names or len(self.feature_names) != len(self.center) or len(self.center) != len(self.scale):
            raise ValueError("quality guard feature metadata is inconsistent")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("quality guard feature names must be unique")
        if any(not np.isfinite(value) for value in self.center):
            raise ValueError("quality guard center must be finite")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.scale):
            raise ValueError("quality guard scales must be finite and positive")
        if not np.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError("quality guard threshold must be finite and non-negative")
        if not 0.0 < self.threshold_quantile <= 1.0:
            raise ValueError("quality guard threshold_quantile must be in (0, 1]")
        if not np.isfinite(self.threshold_margin) or self.threshold_margin < 1.0:
            raise ValueError("quality guard threshold_margin must be >= 1")
        if self.calibration_samples <= 0:
            raise ValueError("quality guard requires calibration samples")

    def score(self, array_chw: np.ndarray) -> float:
        features, names = input_quality_features(array_chw)
        if names != self.feature_names:
            raise ValueError("quality guard feature definition does not match input band count")
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        z = (features - center) / scale
        return float(np.sqrt(np.mean(np.square(z))))

    def assess(self, array_chw: np.ndarray) -> InputQualityAssessment:
        score = self.score(array_chw)
        return InputQualityAssessment(
            score=score,
            threshold=float(self.threshold),
            in_distribution=bool(score <= self.threshold),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": QUALITY_GUARD_SCHEMA_VERSION,
            "method": self.method,
            "feature_names": list(self.feature_names),
            "center": list(self.center),
            "scale": list(self.scale),
            "threshold": float(self.threshold),
            "threshold_quantile": float(self.threshold_quantile),
            "threshold_margin": float(self.threshold_margin),
            "calibration_samples": int(self.calibration_samples),
            "calibration_score_median": float(self.calibration_score_median),
            "calibration_score_max": float(self.calibration_score_max),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "InputQualityGuard":
        if not isinstance(payload, dict) or payload.get("schema_version") != QUALITY_GUARD_SCHEMA_VERSION:
            raise ValueError("unsupported input quality guard schema")
        try:
            return cls(
                feature_names=tuple(str(value) for value in payload["feature_names"]),
                center=tuple(float(value) for value in payload["center"]),
                scale=tuple(float(value) for value in payload["scale"]),
                threshold=float(payload["threshold"]),
                threshold_quantile=float(payload["threshold_quantile"]),
                threshold_margin=float(payload["threshold_margin"]),
                calibration_samples=int(payload["calibration_samples"]),
                calibration_score_median=float(payload["calibration_score_median"]),
                calibration_score_max=float(payload["calibration_score_max"]),
                method=str(payload["method"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid input quality guard metadata") from exc


def calibrate_input_quality_guard(
    arrays_chw: Iterable[np.ndarray],
    *,
    threshold_quantile: float = 0.99,
    threshold_margin: float = 1.25,
    scale_floor: float = 1e-3,
) -> InputQualityGuard:
    if not 0.0 < threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be in (0, 1]")
    if not np.isfinite(threshold_margin) or threshold_margin < 1.0:
        raise ValueError("threshold_margin must be >= 1")
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")

    rows: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for array in arrays_chw:
        features, current_names = input_quality_features(array)
        if names is None:
            names = current_names
        elif current_names != names:
            raise ValueError("quality guard calibration inputs have inconsistent band counts")
        rows.append(features)
    if not rows or names is None:
        raise ValueError("quality guard calibration requires at least one input")

    matrix = np.stack(rows).astype(np.float64, copy=False)
    center = matrix.mean(axis=0)
    if matrix.shape[0] > 1:
        scale = matrix.std(axis=0, ddof=1)
    else:
        scale = np.zeros(matrix.shape[1], dtype=np.float64)
    scale = np.maximum(scale, float(scale_floor))
    z = (matrix - center) / scale
    scores = np.sqrt(np.mean(np.square(z), axis=1))
    base_threshold = float(np.quantile(scores, threshold_quantile, method="higher"))
    threshold = float(base_threshold * threshold_margin)

    return InputQualityGuard(
        feature_names=names,
        center=tuple(float(value) for value in center),
        scale=tuple(float(value) for value in scale),
        threshold=threshold,
        threshold_quantile=float(threshold_quantile),
        threshold_margin=float(threshold_margin),
        calibration_samples=int(matrix.shape[0]),
        calibration_score_median=float(np.median(scores)),
        calibration_score_max=float(np.max(scores)),
    )

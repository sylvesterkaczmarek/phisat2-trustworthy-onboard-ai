from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .filesystem import (
    assert_safe_tree_target,
    remove_stage,
    replace_tree_from_stage,
    sibling_stage_path,
)
from .input_schema import input_schema_sha256, read_input_schema, write_input_schema
from .synth import make_tile
from .telemetry import DOWNLINK_RECORD_KIND, validate_telemetry_record
from .utils import sha256_file

BENCHMARK_SCHEMA_VERSION = 1
CATEGORIES = ("nominal", "degraded", "corrupted", "ood")
IDENTITY_KEYS = (
    "deployment_bundle_id",
    "model_sha256",
    "policy_sha256",
    "input_schema_sha256",
    "input_schema_file_sha256",
    "preprocessing_sha256",
)


def _clip(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def _box_blur(array: np.ndarray) -> np.ndarray:
    padded = np.pad(array, ((1, 1), (1, 1), (0, 0)), mode="edge")
    total = np.zeros_like(array, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + array.shape[0], dx : dx + array.shape[1], :]
    return (total / 9.0).astype(np.float32)


def _shift(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(array)
    height, width = array.shape[:2]
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return result


def _cloud_occlusion(
    array: np.ndarray,
    rng: np.random.Generator,
    *,
    opacity: float,
) -> np.ndarray:
    height, width, bands = array.shape
    yy, xx = np.mgrid[:height, :width]
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(3):
        cy = float(rng.uniform(0.15 * height, 0.85 * height))
        cx = float(rng.uniform(0.15 * width, 0.85 * width))
        sy = max(1.0, float(rng.uniform(0.10 * height, 0.24 * height)))
        sx = max(1.0, float(rng.uniform(0.10 * width, 0.24 * width)))
        blob = np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
        mask = np.maximum(mask, blob.astype(np.float32))
    cloud = np.linspace(0.78, 0.94, bands, dtype=np.float32)[None, None, :]
    alpha = np.clip(mask[..., None] * opacity, 0.0, 1.0)
    return _clip(array * (1.0 - alpha) + cloud * alpha)


def _degraded(
    base: np.ndarray,
    rng: np.random.Generator,
    index: int,
    *,
    noise_std: float,
    brightness_shift: float,
    band_gain_drift: float,
    band_offset_drift: float,
    cloud_opacity: float,
    spatial_shift_px: int,
) -> tuple[np.ndarray, list[str]]:
    recipe = index % 5
    array = base.astype(np.float32, copy=True)
    perturbations: list[str] = []
    if recipe == 0:
        array += rng.normal(0.0, noise_std, size=array.shape).astype(np.float32)
        array += brightness_shift
        perturbations = ["sensor_noise", "brightness_illumination_shift"]
    elif recipe == 1:
        gains = np.linspace(
            1.0 - band_gain_drift,
            1.0 + band_gain_drift,
            array.shape[2],
            dtype=np.float32,
        )
        offsets = np.linspace(
            -band_offset_drift,
            band_offset_drift,
            array.shape[2],
            dtype=np.float32,
        )
        array = array * gains[None, None, :] + offsets[None, None, :]
        spectral = np.linspace(0.72, 1.28, array.shape[2], dtype=np.float32)
        array *= spectral[None, None, :]
        perturbations = ["per_band_gain_offset_drift", "spectral_distribution_shift"]
    elif recipe == 2:
        array = _box_blur(array)
        dy = spatial_shift_px if index % 2 == 0 else -spatial_shift_px
        dx = -spatial_shift_px if index % 3 == 0 else spatial_shift_px
        array = _shift(array, dy, dx)
        perturbations = ["blur", "spatial_shift"]
    elif recipe == 3:
        array = _cloud_occlusion(array, rng, opacity=cloud_opacity)
        array += rng.normal(0.0, noise_std * 0.5, size=array.shape).astype(np.float32)
        perturbations = ["cloud_like_occlusion", "sensor_noise"]
    else:
        array = _box_blur(array)
        array -= brightness_shift
        gains = rng.uniform(
            1.0 - band_gain_drift,
            1.0 + band_gain_drift,
            size=array.shape[2],
        ).astype(np.float32)
        array *= gains[None, None, :]
        perturbations = ["brightness_illumination_shift", "blur", "per_band_gain_drift"]
    return _clip(array), perturbations


def _corrupted(
    base: np.ndarray,
    index: int,
) -> tuple[np.ndarray, list[str]]:
    recipe = index % 4
    array = base.astype(np.float32, copy=True)
    if recipe == 0:
        band = index % array.shape[2]
        array[:, :, band] = 0.0
        return array, ["missing_band_zero_fill"]
    if recipe == 1:
        side = max(2, min(array.shape[0], array.shape[1]) // 4)
        y0 = max(0, array.shape[0] // 2 - side // 2)
        x0 = max(0, array.shape[1] // 2 - side // 2)
        array[y0 : y0 + side, x0 : x0 + side, :] = 1.0
        return array, ["saturated_pixels"]
    if recipe == 2:
        column = index % array.shape[1]
        array[:, column : min(column + 2, array.shape[1]), :] = 0.0
        return array, ["dead_pixel_stripe"]
    band = index % array.shape[2]
    array[:, :, band] = np.nan
    return array, ["corrupt_band_nonfinite"]


def _ood_pattern(
    rng: np.random.Generator,
    *,
    size: int,
    bands: int,
    index: int,
) -> tuple[np.ndarray, list[str]]:
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    recipe = index % 4
    if recipe == 0:
        checker = ((xx // max(1, size // 8) + yy // max(1, size // 8)) % 2).astype(np.float32)
        spectrum = np.linspace(0.15, 0.95, bands, dtype=np.float32)
        array = checker[..., None] * spectrum[None, None, :]
        return array.astype(np.float32), [
            "unknown_checkerboard_background",
            "spectral_distribution_shift",
        ]
    if recipe == 1:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        wave = 0.5 + 0.45 * np.sin(
            (xx + 1.7 * yy) * (2.0 * np.pi / max(4, size // 3)) + phase
        )
        spectrum = np.linspace(1.0, 0.35, bands, dtype=np.float32)
        array = wave[..., None] * spectrum[None, None, :]
        return _clip(array), ["unknown_sinusoidal_background", "spectral_distribution_shift"]
    if recipe == 2:
        cy = (size - 1) / 2.0
        cx = (size - 1) / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rings = 0.5 + 0.45 * np.sin(radius * (2.0 * np.pi / max(3, size // 6)))
        spectrum = rng.uniform(0.25, 1.0, size=bands).astype(np.float32)
        array = rings[..., None] * spectrum[None, None, :]
        return _clip(array), ["unknown_radial_background", "spectral_distribution_shift"]
    stripes = ((xx // max(1, size // 10)) % 3).astype(np.float32) / 2.0
    spectrum = np.roll(np.linspace(0.1, 0.95, bands, dtype=np.float32), 1)
    array = np.clip(
        0.15 + 0.8 * stripes[..., None] * spectrum[None, None, :],
        0.0,
        1.0,
    )
    return array.astype(np.float32), ["unknown_striped_background", "spectral_distribution_shift"]


def generate_benchmark(
    output_root: str | Path,
    input_schema_path: str | Path,
    *,
    samples_per_category: int = 20,
    seed: int = 101,
    noise_std: float = 0.08,
    brightness_shift: float = 0.12,
    band_gain_drift: float = 0.25,
    band_offset_drift: float = 0.08,
    cloud_opacity: float = 0.70,
    spatial_shift_px: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    if samples_per_category < 4:
        raise ValueError("samples_per_category must be at least 4")
    if (
        noise_std < 0.0
        or brightness_shift < 0.0
        or band_gain_drift < 0.0
        or band_offset_drift < 0.0
    ):
        raise ValueError("perturbation magnitudes must be non-negative")
    if not 0.0 <= cloud_opacity <= 1.0:
        raise ValueError("cloud_opacity must be in [0, 1]")
    if spatial_shift_px < 0:
        raise ValueError("spatial_shift_px must be non-negative")

    schema = read_input_schema(input_schema_path)
    tensor = schema["tensor"]
    source = schema["source"]
    if (
        tensor["source_layout"] != "HWC"
        or source["format"] != "npy"
        or source["dtype"] != "float32"
    ):
        raise ValueError("robustness benchmark currently requires float32 HWC NumPy input")
    size = int(tensor["height"])
    if int(tensor["width"]) != size:
        raise ValueError("robustness benchmark currently requires square tiles")
    bands = len(tensor["bands"])

    output_root = assert_safe_tree_target(
        output_root,
        protected_paths=[input_schema_path],
        operation="robustness benchmark replacement",
    )
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"{output_root} already exists; pass overwrite=True to replace it")
    stage = sibling_stage_path(output_root, label="stage")
    stage.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    try:
        schema_copy = deepcopy(schema)
        schema_hash = write_input_schema(stage / "input_schema.json", schema_copy)
        seed_sequence = np.random.SeedSequence(seed)
        category_sequences = seed_sequence.spawn(len(CATEGORIES))
        for category, category_seed in zip(CATEGORIES, category_sequences):
            rng = np.random.default_rng(category_seed)
            for index in range(samples_per_category):
                if category == "ood":
                    tile, perturbations = _ood_pattern(
                        rng,
                        size=size,
                        bands=bands,
                        index=index,
                    )
                    true_class = None
                    class_name = "unknown"
                else:
                    true_class = index % 2
                    class_name = "event" if true_class == 1 else "background"
                    base = make_tile(rng, size, bands, true_class)
                    if category == "nominal":
                        tile = base
                        perturbations = []
                    elif category == "degraded":
                        tile, perturbations = _degraded(
                            base,
                            rng,
                            index,
                            noise_std=noise_std,
                            brightness_shift=brightness_shift,
                            band_gain_drift=band_gain_drift,
                            band_offset_drift=band_offset_drift,
                            cloud_opacity=cloud_opacity,
                            spatial_shift_px=spatial_shift_px,
                        )
                    else:
                        tile, perturbations = _corrupted(base, index)
                relative = Path(category) / class_name / f"{index:05d}.npy"
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, tile.astype(np.float32, copy=False), allow_pickle=False)
                records.append(
                    {
                        "file": str(relative),
                        "category": category,
                        "true_class": true_class,
                        "true_class_name": class_name,
                        "perturbations": perturbations,
                    }
                )

        manifest = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_name": "deterministic-eo-robustness-v1",
            "simulation_only": True,
            "physical_sensor_fidelity_claimed": False,
            "description": (
                "Deterministic synthetic stress cases for onboard decision-logic evaluation; "
                "perturbations are not calibrated physical sensor models."
            ),
            "seed": int(seed),
            "samples_per_category": int(samples_per_category),
            "categories": list(CATEGORIES),
            "input_schema_sha256": schema_hash,
            "configuration": {
                "noise_std": float(noise_std),
                "brightness_shift": float(brightness_shift),
                "band_gain_drift": float(band_gain_drift),
                "band_offset_drift": float(band_offset_drift),
                "cloud_opacity": float(cloud_opacity),
                "spatial_shift_px": int(spatial_shift_px),
            },
            "supported_perturbations": [
                "sensor_noise",
                "brightness_illumination_shift",
                "per_band_gain_offset_drift",
                "missing_band_zero_fill",
                "corrupt_band_nonfinite",
                "saturated_pixels",
                "dead_pixel_stripe",
                "blur",
                "cloud_like_occlusion",
                "spatial_shift",
                "spectral_distribution_shift",
                "unknown_checkerboard_background",
                "unknown_sinusoidal_background",
                "unknown_radial_background",
                "unknown_striped_background",
            ],
            "samples": records,
        }
        (stage / "benchmark_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        replace_tree_from_stage(stage, output_root)
        return manifest
    finally:
        remove_stage(stage)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number} of {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError("benchmark telemetry records must be JSON objects")
            records.append(payload)
    return records


def _one_value(records: list[dict[str, Any]], key: str) -> Any:
    values = {record.get(key) for record in records}
    if len(values) != 1:
        raise ValueError(f"benchmark telemetry contains inconsistent {key} values")
    return next(iter(values))


def _fraction(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _category_metrics(
    samples: list[dict[str, Any]],
    telemetry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records = [(sample, telemetry[str(sample["file"])]) for sample in samples]
    event = [(sample, record) for sample, record in records if sample.get("true_class") == 1]
    background = [
        (sample, record) for sample, record in records if sample.get("true_class") == 0
    ]
    unknown = [
        (sample, record) for sample, record in records if sample.get("true_class") is None
    ]
    retained = sum(bool(record["retained_for_downlink"]) for _, record in records)
    fallbacks = sum(str(record["decision"]).endswith("fallback") for _, record in records)
    guard_triggers = sum(
        record.get("input_quality_guard_enabled") is True
        and record.get("input_quality_ok") is False
        for _, record in records
    )
    inference_failures = sum(not bool(record["inference_ok"]) for _, record in records)
    outside_detection = sum(
        (
            record.get("input_quality_guard_enabled") is True
            and record.get("input_quality_ok") is False
        )
        or not bool(record["inference_ok"])
        for _, record in records
    )
    source_total = sum(
        int(record["size_bytes"])
        for _, record in records
        if isinstance(record.get("size_bytes"), int)
    )
    source_retained = sum(
        int(record["size_bytes"])
        for _, record in records
        if record["retained_for_downlink"] and isinstance(record.get("size_bytes"), int)
    )
    event_retained = sum(bool(record["retained_for_downlink"]) for _, record in event)
    background_rejected = sum(
        not bool(record["retained_for_downlink"]) for _, record in background
    )
    quality_scores = [
        float(record["input_quality_score"])
        for _, record in records
        if record.get("input_quality_score") is not None
    ]
    return {
        "samples": len(records),
        "event_samples": len(event),
        "background_samples": len(background),
        "unknown_samples": len(unknown),
        "event_retention_recall": _fraction(event_retained, len(event)),
        "background_rejection_rate": _fraction(background_rejected, len(background)),
        "retained_fraction": _fraction(retained, len(records)),
        "fallback_rate": _fraction(fallbacks, len(records)),
        "input_quality_guard_trigger_rate": _fraction(guard_triggers, len(records)),
        "quality_or_preprocessing_detection_rate": _fraction(outside_detection, len(records)),
        "inference_failure_rate": _fraction(inference_failures, len(records)),
        "source_bytes_total": source_total,
        "source_bytes_retained": source_retained,
        "source_bytes_reduction_fraction": (
            1.0 - source_retained / source_total if source_total else 0.0
        ),
        "input_quality_score_median": float(np.median(quality_scores)) if quality_scores else None,
        "input_quality_score_max": float(np.max(quality_scores)) if quality_scores else None,
    }


def _prevalence_scenarios(
    nominal_samples: list[dict[str, Any]],
    telemetry: dict[str, dict[str, Any]],
    prevalences: Iterable[float],
) -> list[dict[str, Any]]:
    by_class: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for sample in nominal_samples:
        true_class = sample.get("true_class")
        if true_class in (0, 1):
            by_class[int(true_class)].append(telemetry[str(sample["file"])])
    if not by_class[0] or not by_class[1]:
        raise ValueError("prevalence simulation requires nominal background and event samples")

    def class_stats(records: list[dict[str, Any]]) -> tuple[float, float]:
        total = sum(int(record["size_bytes"]) for record in records)
        retained = sum(
            int(record["size_bytes"])
            for record in records
            if record["retained_for_downlink"]
        )
        average = total / len(records)
        retained_fraction = retained / total if total else 0.0
        return average, retained_fraction

    bg_bytes, bg_retained_fraction = class_stats(by_class[0])
    event_bytes, event_retained_fraction = class_stats(by_class[1])
    scenarios: list[dict[str, Any]] = []
    for prevalence in prevalences:
        p = float(prevalence)
        if not 0.0 <= p <= 1.0:
            raise ValueError("event prevalence values must be in [0, 1]")
        expected_total = p * event_bytes + (1.0 - p) * bg_bytes
        expected_retained = (
            p * event_bytes * event_retained_fraction
            + (1.0 - p) * bg_bytes * bg_retained_fraction
        )
        retained_fraction = expected_retained / expected_total if expected_total else 0.0
        scenarios.append(
            {
                "event_prevalence": p,
                "expected_retained_fraction": float(retained_fraction),
                "expected_source_bytes_reduction_fraction": float(1.0 - retained_fraction),
            }
        )
    return scenarios


def summarize_benchmark(
    manifest_path: str | Path,
    telemetry_log: str | Path,
    *,
    event_prevalences: Iterable[float] = (0.01, 0.05, 0.10),
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported robustness benchmark manifest")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("robustness benchmark manifest has no samples")

    manifest_names = [str(sample.get("file")) for sample in samples]
    if len(manifest_names) != len(set(manifest_names)):
        raise ValueError("robustness benchmark manifest contains duplicate file entries")

    schema_path = manifest_path.parent / "input_schema.json"
    schema_hash = input_schema_sha256(read_input_schema(schema_path))
    if manifest.get("input_schema_sha256") != schema_hash:
        raise ValueError("robustness benchmark manifest input schema hash is inconsistent")

    telemetry_records = _read_jsonl(telemetry_log)
    if not telemetry_records:
        raise ValueError("robustness benchmark telemetry is empty")
    telemetry: dict[str, dict[str, Any]] = {}
    for record in telemetry_records:
        validate_telemetry_record(
            record,
            expected_kind=DOWNLINK_RECORD_KIND,
            require_artifact_identity=True,
        )
        file_name = str(record["file"])
        if file_name in telemetry:
            raise ValueError(f"duplicate benchmark telemetry record: {file_name}")
        telemetry[file_name] = record
    manifest_files = set(manifest_names)
    if manifest_files != set(telemetry):
        raise ValueError("benchmark manifest and telemetry do not cover the same files")

    identity = {key: _one_value(telemetry_records, key) for key in IDENTITY_KEYS}
    record_schema_version = _one_value(telemetry_records, "schema_version")
    bundle_verified = _one_value(telemetry_records, "deployment_bundle_verified")
    quality_guard_enabled = _one_value(telemetry_records, "input_quality_guard_enabled")
    if identity["input_schema_sha256"] != schema_hash:
        raise ValueError("benchmark telemetry input schema does not match benchmark manifest")
    if identity["deployment_bundle_id"] is not None and bundle_verified is not True:
        raise ValueError("benchmark telemetry names a deployment bundle that was not verified")

    for sample in samples:
        file_name = str(sample["file"])
        record = telemetry[file_name]
        source = manifest_path.parent / file_name
        if not source.is_file():
            raise FileNotFoundError(f"benchmark source file is missing: {source}")
        if record.get("input_observation_ok") is not True:
            raise ValueError(f"cannot verify benchmark source identity for {file_name}")
        if record.get("input_sha256") != sha256_file(source):
            raise ValueError(f"benchmark source hash differs from telemetry for {file_name}")
        if record.get("size_bytes") != source.stat().st_size:
            raise ValueError(f"benchmark source size differs from telemetry for {file_name}")

    grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    for sample in samples:
        category = sample.get("category")
        if category not in grouped:
            raise ValueError(f"unsupported benchmark category: {category}")
        grouped[str(category)].append(sample)

    category_metrics = {
        category: _category_metrics(grouped[category], telemetry)
        for category in CATEGORIES
    }
    category_metrics["nominal"]["nominal_false_positive_detection_rate"] = category_metrics[
        "nominal"
    ]["quality_or_preprocessing_detection_rate"]
    category_metrics["degraded"]["degradation_detection_rate"] = category_metrics["degraded"][
        "quality_or_preprocessing_detection_rate"
    ]
    category_metrics["corrupted"]["degradation_detection_rate"] = category_metrics[
        "corrupted"
    ]["quality_or_preprocessing_detection_rate"]
    category_metrics["ood"]["ood_detection_rate"] = category_metrics["ood"][
        "quality_or_preprocessing_detection_rate"
    ]

    return {
        "schema_version": 1,
        "benchmark_name": manifest["benchmark_name"],
        "simulation_only": True,
        "physical_sensor_fidelity_claimed": False,
        "deployment_bundle_id": identity["deployment_bundle_id"],
        "model_sha256": identity["model_sha256"],
        "policy_sha256": identity["policy_sha256"],
        "input_schema_sha256": identity["input_schema_sha256"],
        "input_quality_guard_enabled": bool(quality_guard_enabled),
        "telemetry_integrity": {
            **identity,
            "deployment_bundle_verified": bool(bundle_verified),
            "telemetry_record_schema_version": record_schema_version,
            "benchmark_manifest_schema_verified": True,
            "benchmark_input_hashes_verified": True,
        },
        "categories": category_metrics,
        "prevalence_simulation": {
            "basis": "nominal_id_class_conditional_source_byte_retention",
            "operational_link_bandwidth_measured": False,
            "assumptions": [
                "event prevalence is supplied by the user and is not inferred from this balanced benchmark",
                "class-conditional retention and mean source bytes are estimated from nominal synthetic samples",
                "packetisation, coding, retransmission, framing, and link-layer overhead are not modelled",
            ],
            "scenarios": _prevalence_scenarios(
                grouped["nominal"],
                telemetry,
                event_prevalences,
            ),
        },
    }


def _parse_prevalences(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("provide at least one event prevalence")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or summarize the deterministic EO robustness benchmark."
    )
    sub = parser.add_subparsers(dest="action", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--schema", required=True)
    generate.add_argument("--out", required=True)
    generate.add_argument("--samples-per-category", type=int, default=20)
    generate.add_argument("--seed", type=int, default=101)
    generate.add_argument("--noise-std", type=float, default=0.08)
    generate.add_argument("--brightness-shift", type=float, default=0.12)
    generate.add_argument("--band-gain-drift", type=float, default=0.25)
    generate.add_argument("--band-offset-drift", type=float, default=0.08)
    generate.add_argument("--cloud-opacity", type=float, default=0.70)
    generate.add_argument("--spatial-shift-px", type=int, default=3)
    generate.add_argument("--overwrite", action="store_true")

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--manifest", required=True)
    summarize.add_argument("--log", required=True)
    summarize.add_argument("--event-prevalences", default="0.01,0.05,0.10")
    summarize.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.action == "generate":
        result = generate_benchmark(
            args.out,
            args.schema,
            samples_per_category=args.samples_per_category,
            seed=args.seed,
            noise_std=args.noise_std,
            brightness_shift=args.brightness_shift,
            band_gain_drift=args.band_gain_drift,
            band_offset_drift=args.band_offset_drift,
            cloud_opacity=args.cloud_opacity,
            spatial_shift_px=args.spatial_shift_px,
            overwrite=args.overwrite,
        )
    else:
        result = summarize_benchmark(
            args.manifest,
            args.log,
            event_prevalences=_parse_prevalences(args.event_prevalences),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

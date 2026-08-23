from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from phi2_tile_filter.input_schema import build_input_schema, input_schema_sha256, write_input_schema
from phi2_tile_filter.policy import DecisionPolicy
from phi2_tile_filter.quality_guard import calibrate_input_quality_guard
from phi2_tile_filter.robustness_benchmark import generate_benchmark, summarize_benchmark
from phi2_tile_filter.telemetry import DOWNLINK_RECORD_KIND, TELEMETRY_RECORD_SCHEMA_VERSION


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_quality_guard_detects_obvious_input_shift() -> None:
    rng = np.random.default_rng(7)
    calibration = [
        np.clip(rng.normal(0.30, 0.03, size=(3, 12, 12)), 0.0, 1.0).astype(np.float32)
        for _ in range(20)
    ]
    guard = calibrate_input_quality_guard(
        calibration,
        threshold_quantile=0.99,
        threshold_margin=1.10,
    )
    nominal = guard.assess(calibration[0])
    shifted = guard.assess(np.ones((3, 12, 12), dtype=np.float32))
    assert nominal.in_distribution is True
    assert shifted.in_distribution is False
    assert shifted.score > guard.threshold


def test_policy_retains_quality_guard_outlier() -> None:
    rng = np.random.default_rng(3)
    calibration = [
        np.clip(rng.normal(0.25, 0.02, size=(2, 10, 10)), 0.0, 1.0).astype(np.float32)
        for _ in range(12)
    ]
    guard = calibrate_input_quality_guard(calibration, threshold_margin=1.05)
    policy = DecisionPolicy(
        event_threshold=0.90,
        min_confidence=0.60,
        temperature=1.0,
        input_quality_guard=guard,
    )
    assessment = guard.assess(np.ones((2, 10, 10), dtype=np.float32))
    retained, decision = policy.decide(
        prob_event=0.01,
        max_prob=0.99,
        inference_ok=True,
        input_quality_ok=assessment.in_distribution,
    )
    assert retained is True
    assert decision == "input_quality_fallback"


def test_robustness_benchmark_is_deterministic_and_covers_requested_conditions(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    write_input_schema(schema_path, build_input_schema(bands=4, height=16))
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest_a = generate_benchmark(first, schema_path, samples_per_category=8, seed=41)
    manifest_b = generate_benchmark(second, schema_path, samples_per_category=8, seed=41)
    assert manifest_a == manifest_b
    assert _tree_hash(first) == _tree_hash(second)
    assert manifest_a["simulation_only"] is True
    required = {
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
    }
    assert required.issubset(set(manifest_a["supported_perturbations"]))
    assert {sample["category"] for sample in manifest_a["samples"]} == {
        "nominal",
        "degraded",
        "corrupted",
        "ood",
    }
    corrupt = first / "corrupted" / "event" / "00003.npy"
    if not corrupt.exists():
        corrupt = first / "corrupted" / "background" / "00003.npy"
    assert np.isnan(np.load(corrupt, allow_pickle=False)).any()


def _downlink_record(
    file_name: str,
    *,
    retained: bool,
    quality_ok: bool,
    input_hash: str,
    size_bytes: int,
    schema_hash: str,
    model_hash: str = "b" * 64,
) -> dict:
    decision = (
        "input_quality_fallback"
        if not quality_ok
        else ("event" if retained else "confident_background")
    )
    return {
        "schema_version": TELEMETRY_RECORD_SCHEMA_VERSION,
        "record_kind": DOWNLINK_RECORD_KIND,
        "file": file_name,
        "deployment_bundle_id": "a" * 64,
        "deployment_bundle_verified": True,
        "model_sha256": model_hash,
        "policy_sha256": "c" * 64,
        "input_schema_sha256": schema_hash,
        "input_schema_file_sha256": "e" * 64,
        "preprocessing_sha256": "f" * 64,
        "input_band_ids": ["band_01"],
        "preprocessing_version": 2,
        "execution_provider": "CPUExecutionProvider",
        "timing_scope": "host_wall_clock_perf_counter",
        "input_sha256": input_hash,
        "size_bytes": size_bytes,
        "input_mtime_ns": 1,
        "input_observation_ok": True,
        "event_threshold": 0.8,
        "min_confidence": 0.6,
        "temperature": 1.0,
        "input_quality_guard_enabled": True,
        "input_quality_method": "diagonal-standardized-input-statistics-v1",
        "input_quality_score": 2.0 if not quality_ok else 0.2,
        "input_quality_threshold": 1.0,
        "input_quality_ok": quality_ok,
        "inference_ok": True,
        "error": None,
        "error_type": None,
        "failure_stage": None,
        "pred_class": 1 if retained else 0,
        "prob_event": 0.9 if retained else 0.1,
        "max_prob": 0.9,
        "retention_requested": retained,
        "kept": retained,
        "decision": decision,
        "input_observation_latency_ms": 0.1,
        "preprocessing_latency_ms": 0.1,
        "input_quality_latency_ms": 0.1,
        "onnx_inference_latency_ms": 1.0,
        "policy_latency_ms": 0.1,
        "end_to_end_latency_ms": 2.0,
        "latency_ms": 1.0,
        "downlink_materialized": retained,
        "retained_for_downlink": retained,
        "downlink_error": None,
        "downlink_copy_sha256": input_hash if retained else None,
    }


def _summary_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict], list[dict]]:
    schema = build_input_schema(bands=1, height=8)
    schema_hash = write_input_schema(tmp_path / "input_schema.json", schema)
    assert schema_hash == input_schema_sha256(schema)
    samples = [
        {
            "file": "nominal/background/0.npy",
            "category": "nominal",
            "true_class": 0,
            "true_class_name": "background",
            "perturbations": [],
        },
        {
            "file": "nominal/event/0.npy",
            "category": "nominal",
            "true_class": 1,
            "true_class_name": "event",
            "perturbations": [],
        },
        {
            "file": "degraded/background/0.npy",
            "category": "degraded",
            "true_class": 0,
            "true_class_name": "background",
            "perturbations": ["sensor_noise"],
        },
        {
            "file": "corrupted/background/0.npy",
            "category": "corrupted",
            "true_class": 0,
            "true_class_name": "background",
            "perturbations": ["dead_pixel_stripe"],
        },
        {
            "file": "ood/unknown/0.npy",
            "category": "ood",
            "true_class": None,
            "true_class_name": "unknown",
            "perturbations": ["unknown_checkerboard_background"],
        },
    ]
    manifest = {
        "schema_version": 1,
        "benchmark_name": "deterministic-eo-robustness-v1",
        "input_schema_sha256": schema_hash,
        "samples": samples,
    }
    manifest_path = tmp_path / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    decisions = [
        (False, True),
        (True, True),
        (True, False),
        (True, False),
        (True, False),
    ]
    records: list[dict] = []
    for sample, (retained, quality_ok) in zip(samples, decisions):
        source = tmp_path / sample["file"]
        source.parent.mkdir(parents=True, exist_ok=True)
        content = sample["file"].encode("utf-8").ljust(64, b"_")
        source.write_bytes(content)
        records.append(
            _downlink_record(
                sample["file"],
                retained=retained,
                quality_ok=quality_ok,
                input_hash=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                schema_hash=schema_hash,
            )
        )
    log = tmp_path / "benchmark.jsonl"
    log.write_text("".join(json.dumps(record) + "\n" for record in records))
    return manifest_path, log, samples, records


def test_prevalence_simulation_uses_source_bytes_and_not_link_bandwidth(tmp_path: Path) -> None:
    manifest_path, log, _, _ = _summary_fixture(tmp_path)
    result = summarize_benchmark(manifest_path, log, event_prevalences=(0.10,))
    scenario = result["prevalence_simulation"]["scenarios"][0]
    assert scenario["expected_retained_fraction"] == pytest.approx(0.10)
    assert scenario["expected_source_bytes_reduction_fraction"] == pytest.approx(0.90)
    assert result["prevalence_simulation"]["operational_link_bandwidth_measured"] is False
    assert result["categories"]["ood"]["ood_detection_rate"] == 1.0
    assert "source_bytes_reduction_fraction" in result["categories"]["nominal"]
    assert result["telemetry_integrity"]["benchmark_input_hashes_verified"] is True


def test_robustness_summary_rejects_mixed_model_identity(tmp_path: Path) -> None:
    manifest_path, log, _, records = _summary_fixture(tmp_path)
    records[1]["model_sha256"] = "9" * 64
    log.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="inconsistent model_sha256"):
        summarize_benchmark(manifest_path, log)


def test_robustness_summary_rejects_manifest_schema_mismatch(tmp_path: Path) -> None:
    manifest_path, log, _, _ = _summary_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["input_schema_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest input schema hash"):
        summarize_benchmark(manifest_path, log)

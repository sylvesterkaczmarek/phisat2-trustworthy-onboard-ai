from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from phi2_tile_filter.input_schema import input_schema_sha256, model_input_schema_sha256, read_input_schema
from phi2_tile_filter.utils import sha256_file


def test_complete_multispectral_pipeline(tmp_path: Path) -> None:
    example_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "demo"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            "--n",
            "96",
            "--bands",
            "7",
            "--size",
            "24",
            "--epochs",
            "2",
            "--seed",
            "5",
            "--output-root",
            str(output),
        ],
        cwd=example_root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/run_robustness_benchmark.py",
            "--output-root",
            str(output),
            "--samples-per-category",
            "4",
            "--seed",
            "17",
            "--event-prevalences",
            "0.01,0.10",
        ],
        cwd=example_root,
        check=True,
    )

    manifest = json.loads((output / "tiles" / "manifest.json").read_text())
    source_schema = read_input_schema(output / "tiles" / "input_schema.json")
    source_schema_hash = input_schema_sha256(source_schema)
    metrics = json.loads((output / "reports" / "metrics.json").read_text())
    validation = json.loads((output / "reports" / "model_validation.json").read_text())
    robustness = json.loads((output / "reports" / "robustness_benchmark.json").read_text())
    run_environment = json.loads((output / "reports" / "run_environment.json").read_text())
    robustness_environment = json.loads((output / "reports" / "robustness_environment.json").read_text())
    state = json.loads((output / "models" / "deployment_state.json").read_text())
    active_bundle = output / "models" / "bundles" / state["active_bundle_id"]
    bundle_manifest = json.loads((active_bundle / "bundle.json").read_text())
    active_policy = json.loads((active_bundle / "policy.json").read_text())
    active_schema = read_input_schema(active_bundle / "input_schema.json")
    active_schema_hash = input_schema_sha256(active_schema)

    assert manifest["schema_version"] == 3
    assert set(manifest["split_counts"]) == {"train", "calib", "validation", "test"}
    assert manifest["input_schema_sha256"] == source_schema_hash
    assert manifest["input_band_ids"] == [f"band_{index:02d}" for index in range(1, 8)]

    assert metrics["schema_version"] == 7
    assert metrics["split_role"] == "final_test"
    assert metrics["input_schema_sha256"] == source_schema_hash
    assert metrics["final_test_sample_counts"]["samples_total"] == manifest["split_counts"]["test"]
    assert metrics["final_test_sample_counts"]["inference_failures"] == 0
    retention = metrics["final_test_downlink_retention_metrics"]
    assert 0.0 <= retention["source_bytes_reduction_pct"] <= 100.0
    assert retention["operational_link_bandwidth_measured"] is False
    assert metrics["final_test_input_quality_metrics"]["guard_enabled"] is True

    runtime = metrics["final_test_runtime_metrics"]
    assert runtime["timing_scope"] == "host_wall_clock_perf_counter"
    assert runtime["execution_provider"] == "CPUExecutionProvider"
    assert runtime["spacecraft_timing_measured"] is False
    timing_keys = (
        "input_observation_latency_ms",
        "preprocessing_latency_ms",
        "input_quality_latency_ms",
        "onnx_inference_latency_ms",
        "policy_latency_ms",
        "end_to_end_tile_latency_ms",
    )
    for key in timing_keys:
        assert runtime[key]["samples"] > 0
        assert runtime[key]["avg"] is not None and runtime[key]["avg"] >= 0.0
    assert runtime["end_to_end_tile_latency_ms"]["avg"] >= runtime["onnx_inference_latency_ms"]["avg"]

    assert validation["schema_version"] == 3
    assert validation["split_role"] == "validation"
    assert validation["input_schema_sha256"] == source_schema_hash
    assert validation["input_band_ids"] == manifest["input_band_ids"]
    assert validation["validation_samples"] == manifest["split_counts"]["validation"]
    assert validation["accepted"] is True
    assert all(validation["acceptance_checks"].values())
    assert validation["input_quality_guard_validation"]["enabled"] is True
    assert validation["policy_sha256"] == sha256_file(active_bundle / "policy.json")

    assert active_policy["schema_version"] == 5
    assert active_policy["split_role"] == "calibration"
    assert active_policy["input_schema_sha256"] == source_schema_hash
    assert active_policy["input_band_ids"] == manifest["input_band_ids"]
    assert active_policy["input_quality_guard"]["method"] == "diagonal-standardized-input-statistics-v1"
    stats = active_policy["calibration_statistics"]
    assert stats["event_samples"] > 0
    assert 0.0 <= stats["event_recall_lower_bound"] <= stats["empirical_event_recall"] <= 1.0
    assert stats["event_recall_confidence_level"] == pytest.approx(0.95)

    assert state["schema_version"] == 1
    assert bundle_manifest["schema_version"] == 2
    assert bundle_manifest["bundle_version"] == 2
    assert bundle_manifest["bundle_id"] == state["active_bundle_id"]
    assert bundle_manifest["model_sha256"] == active_policy["model_sha256"]
    assert bundle_manifest["policy_sha256"] == validation["policy_sha256"]
    assert bundle_manifest["input_contract_sha256"] == source_schema_hash == active_schema_hash
    assert model_input_schema_sha256(active_bundle / "model.onnx") == active_schema_hash
    assert active_schema["preprocessing"]["channel_policy"] == "exact-no-implicit-conversion"
    assert active_schema["preprocessing"]["tiff_policy"] == "reject-use-npy-or-mission-specific-loader"

    assert run_environment["schema_version"] == 1
    assert run_environment["seed"] == 5
    assert run_environment["onnxruntime"]["selected_execution_provider"] == "CPUExecutionProvider"
    assert len(run_environment["dependency_fingerprint_sha256"]) == 64
    assert len(run_environment["environment_fingerprint_sha256"]) == 64
    assert len(run_environment["reference_environment"]["sha256"]) == 64
    assert run_environment["tracked_package_versions"]["numpy"] is not None
    assert run_environment["tracked_package_versions"]["torch"] is not None

    assert robustness_environment["schema_version"] == 1
    assert robustness_environment["seed"] == 17
    assert robustness["run_environment"]["environment_fingerprint_sha256"] == robustness_environment[
        "environment_fingerprint_sha256"
    ]

    assert robustness["simulation_only"] is True
    assert robustness["physical_sensor_fidelity_claimed"] is False
    assert robustness["input_quality_guard_enabled"] is True
    assert robustness["telemetry_integrity"]["deployment_bundle_verified"] is True
    assert robustness["telemetry_integrity"]["benchmark_manifest_schema_verified"] is True
    assert robustness["telemetry_integrity"]["benchmark_input_hashes_verified"] is True
    assert robustness["deployment_bundle_id"] == state["active_bundle_id"]
    assert robustness["policy_sha256"] == bundle_manifest["policy_sha256"]
    assert set(robustness["categories"]) == {"nominal", "degraded", "corrupted", "ood"}
    for category in robustness["categories"].values():
        assert "source_bytes_reduction_fraction" in category
        assert "fallback_rate" in category
        assert "quality_or_preprocessing_detection_rate" in category
    assert robustness["categories"]["corrupted"]["degradation_detection_rate"] >= 0.25
    scenarios = robustness["prevalence_simulation"]["scenarios"]
    assert [scenario["event_prevalence"] for scenario in scenarios] == [0.01, 0.10]
    assert robustness["prevalence_simulation"]["operational_link_bandwidth_measured"] is False

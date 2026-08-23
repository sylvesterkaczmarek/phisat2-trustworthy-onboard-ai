from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from phi2_tile_filter.input_schema import input_schema_sha256, model_input_schema_sha256, read_input_schema


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
    manifest = json.loads((output / "tiles" / "manifest.json").read_text())
    source_schema = read_input_schema(output / "tiles" / "input_schema.json")
    source_schema_hash = input_schema_sha256(source_schema)
    metrics = json.loads((output / "reports" / "metrics.json").read_text())
    validation = json.loads((output / "reports" / "model_validation.json").read_text())
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

    assert metrics["split_role"] == "final_test"
    assert metrics["input_schema_sha256"] == source_schema_hash
    assert metrics["final_test_sample_counts"]["samples_total"] == manifest["split_counts"]["test"]
    assert metrics["final_test_sample_counts"]["inference_failures"] == 0
    assert 0.0 <= metrics["final_test_downlink_retention_metrics"]["downlink_bytes_saved_pct"] <= 100.0

    assert validation["schema_version"] == 3
    assert validation["split_role"] == "validation"
    assert validation["input_schema_sha256"] == source_schema_hash
    assert validation["input_band_ids"] == manifest["input_band_ids"]
    assert validation["validation_samples"] == manifest["split_counts"]["validation"]
    assert validation["accepted"] is True
    assert all(validation["acceptance_checks"].values())

    assert active_policy["schema_version"] == 4
    assert active_policy["split_role"] == "calibration"
    assert active_policy["input_schema_sha256"] == source_schema_hash
    assert active_policy["input_band_ids"] == manifest["input_band_ids"]
    stats = active_policy["calibration_statistics"]
    assert stats["event_samples"] > 0
    assert 0.0 <= stats["event_recall_lower_bound"] <= stats["empirical_event_recall"] <= 1.0
    assert stats["event_recall_confidence_level"] == pytest.approx(0.95)

    assert state["schema_version"] == 1
    assert bundle_manifest["schema_version"] == 2
    assert bundle_manifest["bundle_version"] == 2
    assert bundle_manifest["bundle_id"] == state["active_bundle_id"]
    assert bundle_manifest["model_sha256"] == active_policy["model_sha256"]
    assert bundle_manifest["input_contract_sha256"] == source_schema_hash == active_schema_hash
    assert model_input_schema_sha256(active_bundle / "model.onnx") == active_schema_hash
    assert active_schema["preprocessing"]["channel_policy"] == "exact-no-implicit-conversion"
    assert active_schema["preprocessing"]["tiff_policy"] == "reject-use-npy-or-mission-specific-loader"

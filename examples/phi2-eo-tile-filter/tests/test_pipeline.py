from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


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
    metrics = json.loads((output / "reports" / "metrics.json").read_text())
    validation = json.loads((output / "reports" / "model_validation.json").read_text())
    state = json.loads((output / "models" / "deployment_state.json").read_text())
    active_bundle = output / "models" / "bundles" / state["active_bundle_id"]
    bundle_manifest = json.loads((active_bundle / "bundle.json").read_text())
    active_policy = json.loads((active_bundle / "policy.json").read_text())

    assert set(manifest["split_counts"]) == {"train", "calib", "validation", "test"}
    assert manifest["split_roles"] == {
        "train": "model_parameter_fitting",
        "calib": "quantization_and_policy_calibration",
        "validation": "model_and_quantization_acceptance",
        "test": "final_report_only",
    }

    assert metrics["split_role"] == "final_test"
    assert metrics["final_test_sample_counts"]["samples_total"] == manifest["split_counts"]["test"]
    assert metrics["final_test_sample_counts"]["inference_failures"] == 0
    assert 0.0 <= metrics["final_test_downlink_retention_metrics"]["downlink_bytes_saved_pct"] <= 100.0

    assert validation["split_role"] == "validation"
    assert validation["validation_samples"] == manifest["split_counts"]["validation"]
    assert validation["accepted"] is True
    assert all(validation["acceptance_checks"].values())
    classification_regression = validation["classification_metrics"]["quantization_regression"]
    policy_regression = validation["policy_metrics"]["quantization_regression"]
    assert classification_regression["event_recall_drop"] <= validation["acceptance_criteria"][
        "max_classification_event_recall_drop"
    ]
    assert policy_regression["retention_decision_agreement"] >= validation["acceptance_criteria"][
        "min_policy_retention_decision_agreement"
    ]

    assert active_policy["schema_version"] == 3
    assert active_policy["split_role"] == "calibration"
    stats = active_policy["calibration_statistics"]
    assert stats["event_samples"] > 0
    assert 0.0 <= stats["event_recall_lower_bound"] <= stats["empirical_event_recall"] <= 1.0
    assert stats["event_recall_confidence_level"] == pytest.approx(0.95)
    assert stats["event_recall_bound_method"] == "clopper-pearson-one-sided-exact"

    assert state["schema_version"] == 1
    assert bundle_manifest["bundle_id"] == state["active_bundle_id"]
    assert bundle_manifest["model_sha256"] == active_policy["model_sha256"]
    assert (active_bundle / "model.onnx").is_file()
    assert (active_bundle / "input_schema.json").is_file()
    assert (active_bundle / "validation.json").is_file()

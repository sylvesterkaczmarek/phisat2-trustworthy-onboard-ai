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
    metrics = json.loads((output / "reports" / "metrics.json").read_text())
    validation = json.loads((output / "reports" / "model_validation.json").read_text())
    state = json.loads((output / "models" / "deployment_state.json").read_text())
    active_bundle = output / "models" / "bundles" / state["active_bundle_id"]
    bundle_manifest = json.loads((active_bundle / "bundle.json").read_text())
    active_policy = json.loads((active_bundle / "policy.json").read_text())

    assert metrics["test_samples"] > 0
    assert metrics["inference_failures"] == 0
    assert 0.0 <= metrics["bandwidth_saved_pct"] <= 100.0
    assert validation["accepted"] is True
    assert validation["accuracy_drop"] <= validation["max_accuracy_drop_allowed"]
    assert validation["argmax_agreement"] >= validation["min_argmax_agreement_required"]
    assert state["schema_version"] == 1
    assert bundle_manifest["bundle_id"] == state["active_bundle_id"]
    assert bundle_manifest["model_sha256"] == active_policy["model_sha256"]
    assert (active_bundle / "model.onnx").is_file()
    assert (active_bundle / "input_schema.json").is_file()
    assert (active_bundle / "validation.json").is_file()

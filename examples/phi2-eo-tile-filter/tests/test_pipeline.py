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
    assert metrics["test_samples"] > 0
    assert metrics["inference_failures"] == 0
    assert 0.0 <= metrics["bandwidth_saved_pct"] <= 100.0
    assert validation["accuracy_drop"] <= validation["max_accuracy_drop_allowed"]
    assert validation["argmax_agreement"] >= validation["min_argmax_agreement_required"]

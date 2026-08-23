from __future__ import annotations

import json
from pathlib import Path

import pytest

from phi2_tile_filter.filesystem import assert_safe_workspace_root
from phi2_tile_filter.provenance import collect_run_environment
from phi2_tile_filter.telemetry import (
    FINAL_TEST_RECORD_KIND,
    _validate_runtime_validation_identity,
    validate_telemetry_record,
)


def test_reference_environment_uses_exact_direct_pins() -> None:
    example_root = Path(__file__).resolve().parents[1]
    reference = example_root / "requirements-reference.txt"
    lines = [
        line.strip()
        for line in reference.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines
    assert all("==" in line and not any(token in line for token in (">=", "<=", "~=", "<", ">")) for line in lines)
    names = {line.split("==", 1)[0].lower() for line in lines}
    assert {"torch", "numpy", "onnx", "onnxruntime", "pytest", "ruff"} <= names


def test_run_environment_has_git_dependency_and_environment_fingerprints(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    payload = collect_run_environment(
        repo_root,
        seed=23,
        selected_execution_provider="CPUExecutionProvider",
        run_parameters={"output_root": tmp_path},
    )
    assert payload["schema_version"] == 1
    assert payload["seed"] == 23
    assert payload["onnxruntime"]["selected_execution_provider"] == "CPUExecutionProvider"
    assert len(payload["dependency_fingerprint_sha256"]) == 64
    assert len(payload["environment_fingerprint_sha256"]) == 64
    assert len(payload["reference_environment"]["sha256"]) == 64
    assert payload["python"]["version"]
    assert payload["platform"]["system"]
    assert payload["hardware"]["logical_cpu_count"] is not None
    assert payload["run_parameters"]["output_root"] == str(tmp_path)
    assert payload["git"]["dirty"] in (True, False, None)


def test_workspace_root_guard_rejects_filesystem_root_and_allows_scoped_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe demo output root"):
        assert_safe_workspace_root(Path(tmp_path.anchor), operation="demo output")
    assert assert_safe_workspace_root(tmp_path, operation="demo output") == tmp_path.resolve()


def _valid_v6_record() -> dict:
    return {
        "schema_version": 6,
        "record_kind": FINAL_TEST_RECORD_KIND,
        "file": "event/0.npy",
        "deployment_bundle_id": "f" * 64,
        "deployment_bundle_verified": True,
        "model_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "input_schema_sha256": "c" * 64,
        "input_schema_file_sha256": "d" * 64,
        "preprocessing_sha256": "e" * 64,
        "input_sha256": "1" * 64,
        "size_bytes": 100,
        "input_observation_ok": True,
        "inference_ok": True,
        "retention_requested": True,
        "decision": "event",
        "true_class": 1,
        "input_quality_guard_enabled": False,
        "input_quality_method": None,
        "input_quality_score": None,
        "input_quality_threshold": None,
        "input_quality_ok": True,
        "execution_provider": "CPUExecutionProvider",
        "timing_scope": "host_wall_clock_perf_counter",
        "input_observation_latency_ms": 0.1,
        "preprocessing_latency_ms": 0.2,
        "input_quality_latency_ms": 0.0,
        "onnx_inference_latency_ms": 0.3,
        "policy_latency_ms": 0.1,
        "end_to_end_latency_ms": 1.0,
        "latency_ms": 0.3,
    }


def test_telemetry_v6_validates_explicit_timing_scope() -> None:
    record = _valid_v6_record()
    assert validate_telemetry_record(record, expected_kind=FINAL_TEST_RECORD_KIND) is record


def test_telemetry_v6_rejects_invalid_timing() -> None:
    record = _valid_v6_record()
    record["preprocessing_latency_ms"] = -1.0
    with pytest.raises(ValueError, match="preprocessing_latency_ms"):
        validate_telemetry_record(record, expected_kind=FINAL_TEST_RECORD_KIND)


def test_runtime_bundle_validation_identity_requires_exact_policy(tmp_path: Path) -> None:
    validation = tmp_path / "validation.json"
    payload = {
        "schema_version": 3,
        "split_role": "validation",
        "accepted": True,
        "int8_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "input_schema_sha256": "c" * 64,
    }
    validation.write_text(json.dumps(payload), encoding="utf-8")
    _validate_runtime_validation_identity(
        validation,
        model_sha256="a" * 64,
        policy_sha256="b" * 64,
        input_schema_sha256="c" * 64,
    )
    payload["policy_sha256"] = "d" * 64
    validation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different policy"):
        _validate_runtime_validation_identity(
            validation,
            model_sha256="a" * 64,
            policy_sha256="b" * 64,
            input_schema_sha256="c" * 64,
        )

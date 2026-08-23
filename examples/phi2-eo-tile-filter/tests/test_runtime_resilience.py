from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assurance.summarize import summarize
from assurance.watchdog import run_watchdog
from phi2_tile_filter.bandwidth_filter import filter_tiles
from phi2_tile_filter.filesystem import assert_safe_tree_target
from phi2_tile_filter.input_schema import (
    build_input_schema,
    input_schema_sha256,
    model_schema_sidecar_path,
    write_input_schema,
)
from phi2_tile_filter.policy import DecisionPolicy
from phi2_tile_filter.runtime import OnnxRunner
from phi2_tile_filter.synth import write_dataset
from phi2_tile_filter.telemetry import DOWNLINK_RECORD_KIND, FINAL_TEST_RECORD_KIND
from phi2_tile_filter.utils import sha256_file


def _write_binary_model(root: Path, *, bands: int = 1, size: int = 8) -> tuple[Path, dict]:
    onnx = pytest.importorskip("onnx")
    root.mkdir(parents=True, exist_ok=True)
    schema = build_input_schema(bands=bands, height=size)
    schema_hash = input_schema_sha256(schema)

    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    input_value = helper.make_tensor_value_info(
        "input", tensor_proto.FLOAT, ["batch", bands, size, size]
    )
    output_value = helper.make_tensor_value_info("logits", tensor_proto.FLOAT, [1, 2])
    constant = helper.make_tensor("constant_logits", tensor_proto.FLOAT, [1, 2], [1.0, 0.0])
    node = helper.make_node("Constant", [], ["logits"], value=constant)
    graph = helper.make_graph([node], "runtime-resilience-test", [input_value], [output_value])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    helper.set_model_props(
        model,
        {
            "input_schema_sha256": schema_hash,
            "input_schema_version": "2",
            "preprocessing_version": "2",
        },
    )
    path = root / "model.onnx"
    onnx.save(model, path)
    write_input_schema(model_schema_sidecar_path(path), schema)
    return path, schema


def _write_policy(path: Path, model: Path, schema: dict) -> str:
    payload = {
        "schema_version": 4,
        "split_role": "calibration",
        "model_sha256": sha256_file(model),
        "input_schema_sha256": input_schema_sha256(schema),
        "input_band_ids": [band["id"] for band in schema["tensor"]["bands"]],
        "preprocessing_version": schema["preprocessing"]["version"],
        "bands": len(schema["tensor"]["bands"]),
        "size": schema["tensor"]["height"],
        "event_threshold": 0.8,
        "min_confidence": 0.6,
        "temperature": 1.0,
        "temperature_fitted": False,
        "calibration_statistics": {
            "samples_total": 2,
            "event_samples": 1,
            "background_samples": 1,
            "target_event_recall_for_threshold_selection": 0.95,
            "empirical_event_recall": 1.0,
            "event_captures": 1,
            "event_precision_at_threshold": 1.0,
            "roc_auc": 1.0,
            "event_recall_lower_bound": 0.05,
            "event_recall_confidence_level": 0.95,
            "event_recall_bound_method": "clopper-pearson-one-sided-exact",
        },
        "calibration_acceptance": {
            "required_min_event_recall_lower_bound": None,
            "accepted": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return sha256_file(path)


def test_missing_input_returns_structured_fallback(tmp_path: Path) -> None:
    model, _ = _write_binary_model(tmp_path / "model")
    runner = OnnxRunner(model)
    record = runner.evaluate_file(tmp_path / "missing.npy", DecisionPolicy(0.8))
    assert record["inference_ok"] is False
    assert record["retention_requested"] is True
    assert record["decision"] == "inference_failure_fallback"
    assert record["failure_stage"] == "stat_input"
    assert record["input_sha256"] is None
    assert record["size_bytes"] is None
    assert record["input_observation_ok"] is False


def test_corrupt_input_returns_hashed_preprocessing_fallback(tmp_path: Path) -> None:
    model, _ = _write_binary_model(tmp_path / "model")
    runner = OnnxRunner(model)
    corrupt = tmp_path / "corrupt.npy"
    corrupt.write_bytes(b"not-a-valid-numpy-array")
    record = runner.evaluate_file(corrupt, DecisionPolicy(0.8))
    assert record["inference_ok"] is False
    assert record["retention_requested"] is True
    assert record["failure_stage"] == "preprocess_input"
    assert record["input_observation_ok"] is True
    assert record["input_sha256"] == sha256_file(corrupt)
    assert record["size_bytes"] == corrupt.stat().st_size


def _telemetry_record(
    *,
    kind: str,
    file_name: str,
    input_hash: str,
    policy_hash: str,
    model_hash: str,
    schema_hash: str,
) -> dict:
    record = {
        "schema_version": 4,
        "record_kind": kind,
        "file": file_name,
        "deployment_bundle_id": "f" * 64,
        "deployment_bundle_verified": True,
        "model_sha256": model_hash,
        "policy_sha256": policy_hash,
        "input_schema_sha256": schema_hash,
        "input_schema_file_sha256": "d" * 64,
        "preprocessing_sha256": "e" * 64,
        "input_sha256": input_hash,
        "size_bytes": 100,
        "input_mtime_ns": 1,
        "input_observation_ok": True,
        "input_band_ids": ["band_01"],
        "preprocessing_version": 2,
        "event_threshold": 0.8,
        "min_confidence": 0.6,
        "temperature": 1.0,
        "inference_ok": True,
        "error": None,
        "error_type": None,
        "failure_stage": None,
        "pred_class": 1,
        "prob_event": 0.9,
        "max_prob": 0.9,
        "retention_requested": True,
        "kept": True,
        "decision": "event",
        "latency_ms": 1.0,
    }
    if kind == FINAL_TEST_RECORD_KIND:
        record.update({"true_class": 1, "true_class_name": "event"})
    else:
        record.update(
            {
                "downlink_materialized": True,
                "retained_for_downlink": True,
                "downlink_error": None,
                "downlink_copy_sha256": input_hash,
            }
        )
    return record


def _write_summary_fixture(tmp_path: Path, *, test_hash: str, down_hash: str, policy_hash_override: str | None = None):
    model_hash = "a" * 64
    schema_hash = "c" * 64
    calibration = tmp_path / "policy.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "model_sha256": model_hash,
                "input_schema_sha256": schema_hash,
                "event_threshold": 0.8,
                "min_confidence": 0.6,
                "temperature": 1.0,
            }
        ),
        encoding="utf-8",
    )
    policy_hash = policy_hash_override or sha256_file(calibration)
    test_record = _telemetry_record(
        kind=FINAL_TEST_RECORD_KIND,
        file_name="event/0.npy",
        input_hash=test_hash,
        policy_hash=policy_hash,
        model_hash=model_hash,
        schema_hash=schema_hash,
    )
    down_record = _telemetry_record(
        kind=DOWNLINK_RECORD_KIND,
        file_name="event/0.npy",
        input_hash=down_hash,
        policy_hash=policy_hash,
        model_hash=model_hash,
        schema_hash=schema_hash,
    )
    test_log = tmp_path / "test.jsonl"
    down_log = tmp_path / "downlink.jsonl"
    test_log.write_text(json.dumps(test_record) + "\n", encoding="utf-8")
    down_log.write_text(json.dumps(down_record) + "\n", encoding="utf-8")
    return test_log, down_log, calibration


def test_summarizer_rejects_mismatched_input_hashes(tmp_path: Path) -> None:
    test_log, down_log, calibration = _write_summary_fixture(
        tmp_path,
        test_hash="1" * 64,
        down_hash="2" * 64,
    )
    with pytest.raises(ValueError, match="input hash mismatch"):
        summarize(test_log, down_log, calibration)


def test_summarizer_rejects_policy_hash_mismatch(tmp_path: Path) -> None:
    test_log, down_log, calibration = _write_summary_fixture(
        tmp_path,
        test_hash="1" * 64,
        down_hash="1" * 64,
        policy_hash_override="0" * 64,
    )
    with pytest.raises(ValueError, match="policy hash"):
        summarize(test_log, down_log, calibration)


def test_watchdog_times_out_hung_process_and_logs_reason(tmp_path: Path) -> None:
    log = tmp_path / "watchdog.jsonl"
    rc = run_watchdog(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        restarts=0,
        sleep_s=0,
        timeout_s=0.1,
        terminate_grace_s=0.05,
        poll_interval_s=0.01,
        log_path=log,
    )
    assert rc == 124
    record = json.loads(log.read_text().strip())
    assert record["schema_version"] == 2
    assert record["outcome"] == "timeout"
    assert record["restart_scheduled"] is False
    assert record["termination_action"] in {"terminate", "kill", "already_exited"}


def test_watchdog_success_is_structured(tmp_path: Path) -> None:
    log = tmp_path / "watchdog.jsonl"
    rc = run_watchdog(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        restarts=1,
        sleep_s=0,
        timeout_s=1.0,
        log_path=log,
    )
    assert rc == 0
    record = json.loads(log.read_text().strip())
    assert record["outcome"] == "success"
    assert record["watchdog_returncode"] == 0
    assert record["restart_scheduled"] is False


def test_safe_tree_guard_rejects_equal_and_nested_paths(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError, match="overlaps protected input"):
        assert_safe_tree_target(data, protected_paths=[data], operation="test")
    with pytest.raises(ValueError, match="overlaps protected input"):
        assert_safe_tree_target(data / "downlink", protected_paths=[data], operation="test")
    with pytest.raises(ValueError, match="overlaps protected input"):
        assert_safe_tree_target(tmp_path, protected_paths=[data], operation="test")


def test_synthetic_overwrite_failure_preserves_existing_tree(tmp_path: Path, monkeypatch) -> None:
    import phi2_tile_filter.synth as synth

    root = tmp_path / "tiles"
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_text("old-data", encoding="utf-8")

    def fail_tile(*args, **kwargs):
        raise RuntimeError("injected generation failure")

    monkeypatch.setattr(synth, "make_tile", fail_tile)
    with pytest.raises(RuntimeError, match="injected generation failure"):
        write_dataset(root, n=16, bands=1, size=8, seed=0, overwrite=True)
    assert sentinel.read_text(encoding="utf-8") == "old-data"


def test_downlink_filter_rejects_output_nested_inside_input(tmp_path: Path) -> None:
    data = tmp_path / "tiles"
    write_dataset(data, n=16, bands=1, size=8, seed=0)
    model, schema = _write_binary_model(tmp_path / "model")
    # Match the model sidecar to the dataset contract exactly.
    dataset_schema = json.loads((data / "input_schema.json").read_text(encoding="utf-8"))
    write_input_schema(model_schema_sidecar_path(model), dataset_schema)
    onnx = pytest.importorskip("onnx")
    loaded = onnx.load(str(model))
    metadata = {item.key: item.value for item in loaded.metadata_props}
    metadata["input_schema_sha256"] = input_schema_sha256(dataset_schema)
    del loaded.metadata_props[:]
    onnx.helper.set_model_props(loaded, metadata)
    onnx.save(loaded, model)
    policy = tmp_path / "policy.json"
    _write_policy(policy, model, dataset_schema)

    with pytest.raises(ValueError, match="overlaps protected input"):
        filter_tiles(
            model,
            data / "test",
            policy,
            downlink_root=data / "test" / "downlink",
            log_path=tmp_path / "logs" / "downlink.jsonl",
        )


def test_downlink_filter_rejects_output_containing_input(tmp_path: Path) -> None:
    data = tmp_path / "tiles"
    write_dataset(data, n=16, bands=1, size=8, seed=0)
    model, _ = _write_binary_model(tmp_path / "model")
    dataset_schema = json.loads((data / "input_schema.json").read_text(encoding="utf-8"))
    write_input_schema(model_schema_sidecar_path(model), dataset_schema)
    onnx = pytest.importorskip("onnx")
    loaded = onnx.load(str(model))
    metadata = {item.key: item.value for item in loaded.metadata_props}
    metadata["input_schema_sha256"] = input_schema_sha256(dataset_schema)
    del loaded.metadata_props[:]
    onnx.helper.set_model_props(loaded, metadata)
    onnx.save(loaded, model)
    policy = tmp_path / "policy.json"
    _write_policy(policy, model, dataset_schema)

    with pytest.raises(ValueError, match="overlaps protected input"):
        filter_tiles(
            model,
            data / "test",
            policy,
            downlink_root=tmp_path,
            log_path=tmp_path / "separate-log.jsonl",
        )

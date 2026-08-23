from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assurance.model_store import (
    build_bundle,
    promote_bundle,
    resolve_bundle,
    rollback,
    sha256_file,
    verify_bundle,
)
from assurance.summarize import summarize
from assurance.watchdog import run_watchdog


def _write_test_model(path: Path, marker: str, *, bands: int = 3, size: int = 8) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    input_value = helper.make_tensor_value_info(
        "input", tensor_proto.FLOAT, ["batch", bands, size, size]
    )
    output_value = helper.make_tensor_value_info(
        "output", tensor_proto.FLOAT, ["batch", bands, size, size]
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        f"bundle-test-{marker}",
        [input_value],
        [output_value],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    helper.set_model_props(model, {"marker": marker})
    onnx.save(model, path)


def _write_candidate_artifacts(root: Path, marker: str) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    model = root / "model.onnx"
    _write_test_model(model, marker)
    model_hash = sha256_file(model)
    policy = root / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_sha256": model_hash,
                "bands": 3,
                "size": 8,
                "event_threshold": 0.8,
                "min_confidence": 0.6,
                "temperature": 1.0,
            }
        )
    )
    validation = root / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "int8_sha256": model_hash,
                "accuracy_drop": 0.0,
                "argmax_agreement": 1.0,
                "max_accuracy_drop_allowed": 0.02,
                "min_argmax_agreement_required": 0.98,
                "accepted": True,
            }
        )
    )
    return model, policy, validation


def test_bundle_promotion_and_rollback_are_coherent(tmp_path: Path) -> None:
    store = tmp_path / "store"
    state = tmp_path / "deployment_state.json"

    model_a, policy_a, validation_a = _write_candidate_artifacts(tmp_path / "a", "a")
    bundle_a = tmp_path / "bundle-a"
    manifest_a = build_bundle(model_a, policy_a, validation_a, bundle_a)
    first_state = promote_bundle(bundle_a, store, state)
    assert first_state["active_bundle_id"] == manifest_a["bundle_id"]
    assert first_state["previous_bundle_id"] is None
    resolved_a = resolve_bundle(store, state)
    assert sha256_file(resolved_a["model"]) == manifest_a["model_sha256"]
    assert sha256_file(resolved_a["policy"]) == manifest_a["policy_sha256"]

    model_b, policy_b, validation_b = _write_candidate_artifacts(tmp_path / "b", "b")
    bundle_b = tmp_path / "bundle-b"
    manifest_b = build_bundle(model_b, policy_b, validation_b, bundle_b)
    second_state = promote_bundle(bundle_b, store, state)
    assert second_state["active_bundle_id"] == manifest_b["bundle_id"]
    assert second_state["previous_bundle_id"] == manifest_a["bundle_id"]

    rolled_back = rollback(store, state)
    assert rolled_back["active_bundle_id"] == manifest_a["bundle_id"]
    assert rolled_back["previous_bundle_id"] == manifest_b["bundle_id"]
    resolved_after = resolve_bundle(store, state)
    assert sha256_file(resolved_after["model"]) == manifest_a["model_sha256"]
    assert sha256_file(resolved_after["policy"]) == manifest_a["policy_sha256"]


def test_bundle_rejects_model_policy_mismatch(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "mismatch-policy")
    payload = json.loads(policy.read_text())
    payload["model_sha256"] = "f" * 64
    policy.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different model"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_bundle_rejects_validation_report_hash_mismatch(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "mismatch-validation")
    payload = json.loads(validation.read_text())
    payload["int8_sha256"] = "e" * 64
    validation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not cover"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_failed_calibration_cannot_be_promoted(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "failed-calibration")
    payload = json.loads(policy.read_text())
    payload.pop("event_threshold")
    policy.write_text(json.dumps(payload))
    bundle = tmp_path / "bundle"
    state = tmp_path / "deployment_state.json"
    with pytest.raises(ValueError, match="missing event_threshold"):
        build_bundle(model, policy, validation, bundle)
    assert not bundle.exists()
    assert not state.exists()


def test_incomplete_or_corrupt_bundle_is_rejected(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "corrupt")
    bundle = tmp_path / "bundle"
    build_bundle(model, policy, validation, bundle)
    (bundle / "input_schema.json").unlink()
    with pytest.raises(FileNotFoundError, match="input_schema"):
        verify_bundle(bundle)


def test_orphaned_partial_state_file_does_not_change_active_bundle(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "partial-state")
    bundle = tmp_path / "bundle"
    manifest = build_bundle(model, policy, validation, bundle)
    store = tmp_path / "store"
    state = tmp_path / "deployment_state.json"
    promote_bundle(bundle, store, state)

    orphan = state.with_name(f".{state.name}.tmp-interrupted")
    orphan.write_text("{partial")
    resolved = resolve_bundle(store, state)
    assert resolved["bundle_id"] == manifest["bundle_id"]


def test_promoted_bundle_is_immutable_from_candidate_changes(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "immutable")
    bundle = tmp_path / "bundle"
    manifest = build_bundle(model, policy, validation, bundle)
    store = tmp_path / "store"
    state = tmp_path / "deployment_state.json"
    promote_bundle(bundle, store, state)
    (bundle / "policy.json").write_text("{}")
    resolved = resolve_bundle(store, state)
    assert sha256_file(resolved["policy"]) == manifest["policy_sha256"]


def test_watchdog_does_not_need_shell(tmp_path: Path) -> None:
    rc = run_watchdog([sys.executable, "-c", "import sys; sys.exit(0)"], restarts=0, sleep_s=0)
    assert rc == 0


def test_summarizer_uses_kept_and_bytes(tmp_path: Path) -> None:
    model_hash = "a" * 64
    test_records = [
        {"file": "background/0.npy", "model_sha256": model_hash, "true_class": 0, "pred_class": 0, "prob_event": 0.1, "inference_ok": True, "latency_ms": 1.0},
        {"file": "event/0.npy", "model_sha256": model_hash, "true_class": 1, "pred_class": 1, "prob_event": 0.9, "inference_ok": True, "latency_ms": 2.0},
        {"file": "event/1.npy", "model_sha256": model_hash, "true_class": 1, "pred_class": 0, "prob_event": 0.4, "inference_ok": True, "latency_ms": 3.0},
    ]
    down_records = [
        {"file": "background/0.npy", "model_sha256": model_hash, "kept": False, "decision": "confident_background", "size_bytes": 100},
        {"file": "event/0.npy", "model_sha256": model_hash, "kept": True, "decision": "event", "size_bytes": 200},
        {"file": "event/1.npy", "model_sha256": model_hash, "kept": True, "decision": "low_confidence_fallback", "size_bytes": 300},
    ]
    test_log = tmp_path / "test.jsonl"
    down_log = tmp_path / "down.jsonl"
    test_log.write_text("".join(json.dumps(r) + "\n" for r in test_records))
    down_log.write_text("".join(json.dumps(r) + "\n" for r in down_records))
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"model_sha256": model_hash, "target_event_recall": 0.95, "event_threshold": 0.8, "min_confidence": 0.6, "temperature": 1.0}))
    metrics = summarize(test_log, down_log, calib)
    assert metrics["tiles_kept"] == 2
    assert metrics["bytes_kept"] == 500
    assert metrics["bandwidth_saved_pct"] == pytest.approx(100.0 / 6.0)
    assert metrics["downlink_event_recall"] == 1.0
    assert metrics["fallback_tiles"] == 1

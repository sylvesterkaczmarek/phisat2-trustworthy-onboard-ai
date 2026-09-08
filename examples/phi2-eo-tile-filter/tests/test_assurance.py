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
from phi2_tile_filter.input_schema import (
    build_input_schema,
    input_schema_sha256,
    model_schema_sidecar_path,
    read_input_schema,
    write_input_schema,
)
from phi2_tile_filter.telemetry import resolve_artifact_identity


def _write_test_model(
    path: Path,
    marker: str,
    *,
    input_schema_hash: str,
    bands: int = 3,
    size: int = 8,
) -> None:
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
    helper.set_model_props(
        model,
        {
            "marker": marker,
            "input_schema_sha256": input_schema_hash,
            "input_schema_version": "2",
            "preprocessing_version": "2",
        },
    )
    onnx.save(model, path)


def _valid_validation_payload(model_hash: str, schema_hash: str, policy_hash: str) -> dict:
    checks = {
        "classification_accuracy_drop": True,
        "classification_argmax_agreement": True,
        "classification_event_recall_drop": True,
        "classification_event_false_negative_rate_increase": True,
        "classification_pr_auc_drop": True,
        "policy_retention_decision_agreement": True,
        "policy_event_retention_recall_drop": True,
        "event_score_drift": True,
    }
    return {
        "schema_version": 3,
        "split_role": "validation",
        "validation_samples": 20,
        "validation_event_samples": 10,
        "validation_background_samples": 10,
        "fp32_sha256": "b" * 64,
        "int8_sha256": model_hash,
        "policy_sha256": policy_hash,
        "input_schema_sha256": schema_hash,
        "input_band_ids": ["band_01", "band_02", "band_03"],
        "preprocessing_version": 2,
        "classification_metrics": {
            "quantization_regression": {
                "accuracy_drop": 0.0,
                "event_recall_drop": 0.0,
                "event_false_negative_rate_increase": 0.0,
                "event_f1_drop": 0.0,
                "roc_auc_drop": 0.0,
                "pr_auc_drop": 0.0,
                "argmax_agreement": 1.0,
            }
        },
        "policy_metrics": {
            "quantization_regression": {
                "retention_decision_agreement": 1.0,
                "event_retention_recall_drop": 0.0,
                "retained_fraction_change": 0.0,
            }
        },
        "score_drift_metrics": {
            "mean_absolute_event_score_drift": 0.0,
            "p95_absolute_event_score_drift": 0.0,
            "max_absolute_event_score_drift": 0.0,
        },
        "acceptance_criteria": {
            "max_classification_accuracy_drop": 0.02,
            "min_classification_argmax_agreement": 0.98,
            "max_classification_event_recall_drop": 0.02,
            "max_classification_event_false_negative_rate_increase": 0.02,
            "max_classification_pr_auc_drop": 0.02,
            "min_policy_retention_decision_agreement": 0.98,
            "max_policy_event_retention_recall_drop": 0.02,
            "max_event_score_drift": 0.05,
        },
        "acceptance_checks": checks,
        "accepted": True,
    }


def _write_candidate_artifacts(root: Path, marker: str) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    schema = build_input_schema(bands=3, height=8)
    schema_hash = input_schema_sha256(schema)
    model = root / "model.onnx"
    _write_test_model(model, marker, input_schema_hash=schema_hash)
    write_input_schema(model_schema_sidecar_path(model), schema)
    model_hash = sha256_file(model)
    policy = root / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "split_role": "calibration",
                "model_sha256": model_hash,
                "input_schema_sha256": schema_hash,
                "input_band_ids": ["band_01", "band_02", "band_03"],
                "preprocessing_version": 2,
                "bands": 3,
                "size": 8,
                "event_threshold": 0.8,
                "min_confidence": 0.6,
                "temperature": 1.0,
                "temperature_fitted": True,
                "calibration_statistics": {
                    "samples_total": 20,
                    "event_samples": 10,
                    "background_samples": 10,
                    "target_event_recall_for_threshold_selection": 0.95,
                    "empirical_event_recall": 1.0,
                    "event_captures": 10,
                    "event_precision_at_threshold": 1.0,
                    "roc_auc": 1.0,
                    "event_recall_lower_bound": 0.74,
                    "event_recall_confidence_level": 0.95,
                    "event_recall_bound_method": "clopper-pearson-one-sided-exact",
                },
                "calibration_acceptance": {
                    "required_min_event_recall_lower_bound": None,
                    "accepted": True,
                },
            }
        )
    )
    validation = root / "validation.json"
    validation.write_text(
        json.dumps(_valid_validation_payload(model_hash, schema_hash, sha256_file(policy)))
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
    assert resolved_a["input_contract_sha256"] == manifest_a["input_contract_sha256"]

    model_b, policy_b, validation_b = _write_candidate_artifacts(tmp_path / "b", "b")
    bundle_b = tmp_path / "bundle-b"
    manifest_b = build_bundle(model_b, policy_b, validation_b, bundle_b)
    second_state = promote_bundle(bundle_b, store, state)
    assert second_state["active_bundle_id"] == manifest_b["bundle_id"]
    assert second_state["previous_bundle_id"] == manifest_a["bundle_id"]
    rolled_back = rollback(store, state)
    assert rolled_back["active_bundle_id"] == manifest_a["bundle_id"]
    assert rolled_back["previous_bundle_id"] == manifest_b["bundle_id"]


def test_bundle_rejects_model_policy_mismatch(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "mismatch-policy")
    payload = json.loads(policy.read_text())
    payload["model_sha256"] = "f" * 64
    policy.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different model"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_bundle_rejects_validation_policy_mismatch(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "policy-evidence")
    payload = json.loads(validation.read_text())
    payload["policy_sha256"] = "f" * 64
    validation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different calibration policy"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_bundle_rejects_input_contract_mismatch(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "schema-mismatch")
    payload = json.loads(policy.read_text())
    payload["input_schema_sha256"] = "d" * 64
    policy.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="input/preprocessing contract"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_bundle_rejects_mismatched_preprocessing_metadata(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "preprocessing-mismatch")
    schema_path = model_schema_sidecar_path(model)
    schema = json.loads(schema_path.read_text())
    schema["normalization"]["version"] = 2
    schema_path.write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="preprocessing_sha256"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_runtime_rejects_model_schema_hash_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    from phi2_tile_filter.runtime import OnnxRunner

    model, _, _ = _write_candidate_artifacts(tmp_path / "candidate", "runtime-schema-mismatch")
    schema_path = model_schema_sidecar_path(model)
    schema = json.loads(schema_path.read_text())
    schema["tensor"]["bands"][0]["name"] = "different_scientific_band"
    schema_path.write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="does not match its preprocessing contract"):
        OnnxRunner(model)


def test_bundle_rejects_validation_report_hash_mismatch(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "mismatch-validation")
    payload = json.loads(validation.read_text())
    payload["int8_sha256"] = "e" * 64
    validation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not cover"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_bundle_rejects_test_set_as_acceptance_evidence(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "wrong-split")
    payload = json.loads(validation.read_text())
    payload["split_role"] = "final_test"
    validation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="validation"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_bundle_recomputes_scientific_validation_checks(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "tampered-validation")
    payload = json.loads(validation.read_text())
    payload["classification_metrics"]["quantization_regression"]["event_recall_drop"] = 0.5
    validation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="acceptance checks do not match"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_bundle_build_rejects_output_overlapping_inputs(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "unsafe-output")
    with pytest.raises(ValueError, match="overlaps protected input"):
        build_bundle(model, policy, validation, model.parent)
    assert model.is_file()
    assert policy.is_file()
    assert validation.is_file()


def test_failed_calibration_cannot_be_promoted(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "failed-calibration")
    payload = json.loads(policy.read_text())
    payload["calibration_acceptance"]["required_min_event_recall_lower_bound"] = 0.9
    payload["calibration_acceptance"]["accepted"] = False
    policy.write_text(json.dumps(payload))
    bundle = tmp_path / "bundle"
    with pytest.raises(ValueError, match="not marked accepted"):
        build_bundle(model, policy, validation, bundle)
    assert not bundle.exists()


def test_incomplete_or_corrupt_bundle_is_rejected(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "corrupt")
    bundle = tmp_path / "bundle"
    build_bundle(model, policy, validation, bundle)
    (bundle / "input_schema.json").unlink()
    with pytest.raises(FileNotFoundError, match="input_schema"):
        verify_bundle(bundle)


def test_runtime_bundle_verification_covers_validation_component(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "runtime-bundle")
    bundle = tmp_path / "bundle"
    manifest = build_bundle(model, policy, validation, bundle)
    schema = read_input_schema(bundle / "input_schema.json")
    identity = resolve_artifact_identity(
        bundle / "model.onnx",
        bundle / "policy.json",
        model_sha256=manifest["model_sha256"],
        input_schema_sha256=manifest["input_contract_sha256"],
        input_schema_path=bundle / "input_schema.json",
        preprocessing_sha256=schema["preprocessing_sha256"],
        explicit_bundle_id=manifest["bundle_id"],
    )
    assert identity["deployment_bundle_verified"] is True
    (bundle / "validation.json").write_text("{}")
    with pytest.raises(ValueError, match="validation component hash mismatch"):
        resolve_artifact_identity(
            bundle / "model.onnx",
            bundle / "policy.json",
            model_sha256=manifest["model_sha256"],
            input_schema_sha256=manifest["input_contract_sha256"],
            input_schema_path=bundle / "input_schema.json",
            preprocessing_sha256=schema["preprocessing_sha256"],
            explicit_bundle_id=manifest["bundle_id"],
        )


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


def test_resolve_rejects_non_hex_state_bundle_id(tmp_path: Path) -> None:
    state = tmp_path / "deployment_state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": 1,
                "active_bundle_id": "z" * 64,
                "previous_bundle_id": None,
            }
        )
    )
    with pytest.raises(ValueError, match="invalid active_bundle_id"):
        resolve_bundle(tmp_path / "store", state)


def test_promotion_rejects_corrupt_current_rollback_target(tmp_path: Path) -> None:
    store = tmp_path / "store"
    state = tmp_path / "deployment_state.json"
    model_a, policy_a, validation_a = _write_candidate_artifacts(tmp_path / "a", "active-a")
    bundle_a = tmp_path / "bundle-a"
    manifest_a = build_bundle(model_a, policy_a, validation_a, bundle_a)
    promote_bundle(bundle_a, store, state)

    model_b, policy_b, validation_b = _write_candidate_artifacts(tmp_path / "b", "active-b")
    bundle_b = tmp_path / "bundle-b"
    build_bundle(model_b, policy_b, validation_b, bundle_b)

    (store / manifest_a["bundle_id"] / "policy.json").write_text("{}")
    with pytest.raises(ValueError, match="component hash mismatch"):
        promote_bundle(bundle_b, store, state)


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


def test_summarizer_rejects_input_contract_mismatch(tmp_path: Path) -> None:
    model_hash = "a" * 64
    schema_hash = "c" * 64
    test_records = [
        {
            "file": "event/0.npy",
            "model_sha256": model_hash,
            "input_schema_sha256": schema_hash,
            "true_class": 1,
            "pred_class": 1,
            "prob_event": 0.9,
            "inference_ok": True,
            "latency_ms": 1.0,
        },
    ]
    down_records = [
        {
            "file": "event/0.npy",
            "model_sha256": model_hash,
            "input_schema_sha256": schema_hash,
            "kept": True,
            "decision": "event",
            "size_bytes": 100,
        },
    ]
    test_log = tmp_path / "test.jsonl"
    down_log = tmp_path / "down.jsonl"
    test_log.write_text("".join(json.dumps(r) + "\n" for r in test_records))
    down_log.write_text("".join(json.dumps(r) + "\n" for r in down_records))
    calib = tmp_path / "calib.json"
    calib.write_text(
        json.dumps({"model_sha256": model_hash, "input_schema_sha256": "d" * 64})
    )
    with pytest.raises(ValueError, match="schema differs"):
        summarize(test_log, down_log, calib)


def test_rollback_recovers_from_corrupt_active_without_retaining_it(tmp_path: Path) -> None:
    store, state = tmp_path / "store", tmp_path / "state.json"
    bundle_a, bundle_b = tmp_path / "bundle-a", tmp_path / "bundle-b"
    manifest_a = build_bundle(*_write_candidate_artifacts(tmp_path / "a", "a"), bundle_a)
    manifest_b = build_bundle(*_write_candidate_artifacts(tmp_path / "b", "b"), bundle_b)
    promote_bundle(bundle_a, store, state)
    previous_state = promote_bundle(bundle_b, store, state)
    damaged_file = store / manifest_b["bundle_id"] / "model.onnx"
    damaged_file.write_bytes(b"damaged deployment")

    recovered = rollback(store, state)

    assert recovered["generation"] == previous_state["generation"] + 1
    assert recovered["active_bundle_id"] == manifest_a["bundle_id"]
    assert recovered["previous_bundle_id"] is None
    assert resolve_bundle(store, state)["bundle_id"] == manifest_a["bundle_id"]
    assert damaged_file.read_bytes() == b"damaged deployment"
    with pytest.raises(FileNotFoundError, match="no previous"):
        rollback(store, state)


def test_rollback_rejects_corrupt_previous_without_changing_state(tmp_path: Path) -> None:
    store, state = tmp_path / "store", tmp_path / "state.json"
    bundle_a, bundle_b = tmp_path / "bundle-a", tmp_path / "bundle-b"
    manifest_a = build_bundle(*_write_candidate_artifacts(tmp_path / "a", "a"), bundle_a)
    build_bundle(*_write_candidate_artifacts(tmp_path / "b", "b"), bundle_b)
    promote_bundle(bundle_a, store, state)
    promote_bundle(bundle_b, store, state)
    before = state.read_bytes()
    (store / manifest_a["bundle_id"] / "model.onnx").write_bytes(b"damaged rollback target")
    with pytest.raises(ValueError, match="hash mismatch"):
        rollback(store, state)
    assert state.read_bytes() == before


def test_promotion_rejects_wrong_cached_bundle_before_state_changes(tmp_path: Path) -> None:
    import shutil

    store, state = tmp_path / "store", tmp_path / "state.json"
    bundle_a, bundle_b = tmp_path / "bundle-a", tmp_path / "bundle-b"
    build_bundle(*_write_candidate_artifacts(tmp_path / "a", "a"), bundle_a)
    manifest_b = build_bundle(*_write_candidate_artifacts(tmp_path / "b", "b"), bundle_b)
    promote_bundle(bundle_a, store, state)
    before = state.read_bytes()
    shutil.copytree(bundle_a, store / manifest_b["bundle_id"])
    with pytest.raises(ValueError, match="identity"):
        promote_bundle(bundle_b, store, state)
    assert state.read_bytes() == before


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), True, "1.0"])
def test_bundle_rejects_invalid_calibration_temperature(tmp_path: Path, temperature) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "temperature")
    payload = json.loads(policy.read_text())
    payload["temperature"] = temperature
    policy.write_text(json.dumps(payload))
    evidence = json.loads(validation.read_text())
    evidence["policy_sha256"] = sha256_file(policy)
    validation.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="temperature"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


@pytest.mark.parametrize("criterion", [float("inf"), -0.1, 1.1, True, "0.02"])
def test_bundle_rejects_invalid_validation_criterion(tmp_path: Path, criterion) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "criterion")
    evidence = json.loads(validation.read_text())
    evidence["acceptance_criteria"]["max_classification_accuracy_drop"] = criterion
    validation.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="acceptance|criterion|criteria"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


@pytest.mark.parametrize("field", ["sleep_s", "timeout_s", "terminate_grace_s", "heartbeat_timeout_s", "poll_interval_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_watchdog_rejects_invalid_durations_before_spawning(tmp_path: Path, monkeypatch, field, value) -> None:
    import assurance.watchdog as watchdog

    def unexpected_spawn(*args, **kwargs):
        pytest.fail("invalid watchdog configuration must be rejected before a child starts")

    monkeypatch.setattr(watchdog.subprocess, "Popen", unexpected_spawn)
    kwargs = {field: value, "heartbeat_path": tmp_path / "heartbeat"}
    with pytest.raises(ValueError, match=field):
        watchdog.run_watchdog(["unused-command"], **kwargs)


def test_watchdog_terminates_owned_child_when_monitoring_is_interrupted(monkeypatch) -> None:
    import assurance.watchdog as watchdog

    real_popen = watchdog.subprocess.Popen
    children = []

    def tracked_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    real_sleep = watchdog.time.sleep
    interrupted = False

    def interrupt(duration):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        real_sleep(duration)

    monkeypatch.setattr(watchdog.subprocess, "Popen", tracked_popen)
    monkeypatch.setattr(watchdog.time, "sleep", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            watchdog.run_watchdog(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                restarts=0,
                terminate_grace_s=0.05,
            )
        assert len(children) == 1
        assert children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_failed_bundle_replacement_preserves_previous_bundle(tmp_path: Path, monkeypatch) -> None:
    import assurance.model_store as model_store

    bundle = tmp_path / "bundle"
    manifest = build_bundle(*_write_candidate_artifacts(tmp_path / "a", "a"), bundle)
    candidate = _write_candidate_artifacts(tmp_path / "b", "b")
    real_replace = model_store.os.replace

    def fail_publication(source, destination):
        if ".stage-" in Path(source).name and Path(destination) == bundle:
            raise OSError("simulated publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(model_store.os, "replace", fail_publication)
    with pytest.raises(OSError, match="publication failure"):
        build_bundle(*candidate, bundle)
    assert verify_bundle(bundle) == manifest
    assert not list(tmp_path.glob(".bundle.*"))


def test_watchdog_does_not_restart_unconfirmed_child(monkeypatch) -> None:
    import assurance.watchdog as watchdog

    class UnconfirmedChild:
        returncode = None

        def poll(self):
            return None

    children = []

    def spawn(*args, **kwargs):
        child = UnconfirmedChild()
        children.append(child)
        return child

    monkeypatch.setattr(watchdog.subprocess, "Popen", spawn)
    monkeypatch.setattr(watchdog, "_terminate_process", lambda *_: ("kill_unconfirmed", None))
    assert watchdog.run_watchdog(
        ["unused-command"], restarts=2, sleep_s=0, timeout_s=0.001, poll_interval_s=0.001,
    ) == 124
    assert len(children) == 1


def _write_rank_bound_policy(policy: Path, validation: Path) -> None:
    from scipy.stats import beta

    payload = json.loads(policy.read_text())
    stats = payload["calibration_statistics"]
    # The threshold rank is fixed before observing tie-dependent captures.
    stats.update({
        "samples_total": 30,
        "event_samples": 20,
        "event_captures": 20,
        "target_event_recall_for_threshold_selection": 0.95,
        "event_recall_bound_method": "order-statistic-one-sided-exact",
        "event_recall_bound_selection_rank": 19,
        "event_recall_lower_bound": float(beta.ppf(0.05, 19, 2)),
    })
    policy.write_text(json.dumps(payload))
    evidence = json.loads(validation.read_text())
    evidence["policy_sha256"] = sha256_file(policy)
    validation.write_text(json.dumps(evidence))


def test_bundle_accepts_verified_selection_rank_bound(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "rank")
    _write_rank_bound_policy(policy, validation)
    manifest = build_bundle(model, policy, validation, tmp_path / "bundle")
    assert verify_bundle(tmp_path / "bundle") == manifest


@pytest.mark.parametrize("key,value", [
    ("event_recall_bound_selection_rank", 20),
    ("event_recall_bound_selection_rank", True),
    ("event_captures", 18),
    ("empirical_event_recall", 0.95),
    ("event_recall_lower_bound", 0.99),
])
def test_bundle_rejects_inconsistent_selection_rank_bound(tmp_path: Path, key, value) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "rank")
    _write_rank_bound_policy(policy, validation)
    payload = json.loads(policy.read_text())
    payload["calibration_statistics"][key] = value
    policy.write_text(json.dumps(payload))
    evidence = json.loads(validation.read_text())
    evidence["policy_sha256"] = sha256_file(policy)
    validation.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="calibration policy"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


@pytest.mark.parametrize("destination", [
    "model", "policy", "model_schema", "data_schema", "tile", "policy_symlink", "policy_hardlink",
])
def test_telemetry_rejects_output_overlapping_loaded_inputs(tmp_path: Path, destination) -> None:
    import os

    import numpy as np

    from assurance.telemetry_log import emit_telemetry

    model, policy, _ = _write_candidate_artifacts(tmp_path / "candidate", "telemetry-output")
    schema = model_schema_sidecar_path(model)
    data_root = tmp_path / "data" / "test"
    (data_root / "event").mkdir(parents=True)
    tile = data_root / "event" / "0.npy"
    np.save(tile, np.zeros((3, 8, 8), dtype=np.float32))
    data_schema = data_root.parent / "input_schema.json"
    data_schema.write_bytes(schema.read_bytes())
    protected = {
        "model": model,
        "policy": policy,
        "model_schema": schema,
        "data_schema": data_schema,
        "tile": tile,
    }
    before = {path: path.read_bytes() for path in protected.values()}
    if destination == "policy_symlink":
        output = tmp_path / "linked-policy.json"
        output.symlink_to(policy)
    elif destination == "policy_hardlink":
        output = tmp_path / "linked-policy.json"
        os.link(policy, output)
    else:
        output = protected[destination]

    with pytest.raises(ValueError, match="overlapping"):
        emit_telemetry(model, data_root, policy, output)

    for path, original in before.items():
        assert path.read_bytes() == original


@pytest.mark.parametrize("destination", ["bundle.json", "validation.json", "new-telemetry.jsonl"])
@pytest.mark.parametrize("model_alias", [False, True])
def test_telemetry_preserves_entire_loaded_bundle(tmp_path: Path, destination: str, model_alias: bool) -> None:
    import numpy as np

    from assurance.telemetry_log import emit_telemetry

    bundle = tmp_path / "bundle"
    manifest = build_bundle(*_write_candidate_artifacts(tmp_path / "candidate", "immutable-output"), bundle)
    data_root = tmp_path / "data"
    (data_root / "event").mkdir(parents=True)
    np.save(data_root / "event" / "0.npy", np.zeros((3, 8, 8), dtype=np.float32))
    (data_root / "input_schema.json").write_bytes((bundle / "input_schema.json").read_bytes())
    before = {path.name: path.read_bytes() for path in bundle.iterdir()}
    model = bundle / "model.onnx"
    if model_alias:
        alias_dir = tmp_path / "alias"
        alias_dir.mkdir()
        model = alias_dir / "model.onnx"
        model.symlink_to(bundle / "model.onnx")
        (alias_dir / "input_schema.json").symlink_to(bundle / "input_schema.json")

    with pytest.raises(ValueError, match="overlapping"):
        emit_telemetry(model, data_root, bundle / "policy.json", bundle / destination)

    assert {path.name: path.read_bytes() for path in bundle.iterdir()} == before
    assert verify_bundle(bundle) == manifest

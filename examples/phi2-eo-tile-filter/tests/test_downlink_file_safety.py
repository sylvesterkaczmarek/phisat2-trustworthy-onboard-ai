from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phi2_tile_filter.bandwidth_filter import filter_tiles
from phi2_tile_filter.input_schema import model_schema_sidecar_path, write_input_schema
from test_runtime_resilience import _write_binary_model, _write_policy


@pytest.fixture
def filter_case(tmp_path: Path, monkeypatch):
    import phi2_tile_filter.bandwidth_filter as module
    from phi2_tile_filter.runtime import OnnxRunner

    monkeypatch.setattr(module, "OnnxRunner", lambda path: OnnxRunner(path, intra_op_threads=1))
    model, schema = _write_binary_model(tmp_path / "artifacts")
    policy = tmp_path / "artifacts" / "policy.json"
    _write_policy(policy, model, schema)
    payload = json.loads(policy.read_text(encoding="utf-8"))
    payload["event_threshold"] = 0.0
    policy.write_text(json.dumps(payload), encoding="utf-8")
    data = tmp_path / "tiles"
    data.mkdir()
    write_input_schema(data / "input_schema.json", schema)
    tile = data / "tile.npy"
    np.save(tile, np.full((8, 8, 1), 0.5, dtype=np.float32))
    output = tmp_path / "downlink"
    log = tmp_path / "logs" / "downlink.jsonl"
    return model, data, policy, output, log


@pytest.mark.parametrize("target", ["model", "policy", "model_schema"])
@pytest.mark.parametrize("output_kind", ["directory", "log"])
def test_filter_protects_loaded_artifacts(filter_case, target, output_kind):
    model, data, policy, output, log = filter_case
    artifact = {
        "model": model,
        "policy": policy,
        "model_schema": model_schema_sidecar_path(model),
    }[target]
    before = artifact.read_bytes()
    if output_kind == "directory":
        output = artifact.parent
    else:
        log = artifact
    with pytest.raises(ValueError, match="overlap"):
        filter_tiles(model, data, policy, downlink_root=output, log_path=log)
    assert artifact.read_bytes() == before


def test_partial_copy_is_removed_before_downlink_publication(filter_case, monkeypatch):
    import phi2_tile_filter.bandwidth_filter as module

    model, data, policy, output, log = filter_case

    def partial_copy(source, destination):
        Path(destination).write_bytes(b"partial recording")
        raise OSError("injected copy failure")

    monkeypatch.setattr(module.shutil, "copy2", partial_copy)
    summary = filter_tiles(model, data, policy, downlink_root=output, log_path=log)
    record = json.loads(log.read_text(encoding="utf-8"))
    assert summary["downlink_materialization_failures"] == 1
    assert summary["tiles_retained"] == 0
    assert record["downlink_materialized"] is False
    assert not list(output.rglob("*.npy"))


@pytest.mark.parametrize("previous_outputs", [False, True])
@pytest.mark.parametrize("failure_target", ["directory", "log"])
def test_failed_publication_restores_both_previous_outputs(
    filter_case, monkeypatch, previous_outputs, failure_target
):
    import phi2_tile_filter.filesystem as filesystem

    model, data, policy, output, log = filter_case
    if previous_outputs:
        output.mkdir()
        (output / "previous.npy").write_bytes(b"previous tile")
        log.parent.mkdir()
        log.write_bytes(b"previous telemetry\n")
    original_replace = filesystem.os.replace
    fail_destination = output if failure_target == "directory" else log

    def fail_publish(source, destination):
        source = Path(source)
        if Path(destination) == fail_destination and (
            ".stage-" in source.name or ".tmp-" in source.name
        ):
            raise OSError("injected publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(filesystem.os, "replace", fail_publish)
    with pytest.raises(OSError, match="injected publication failure"):
        filter_tiles(model, data, policy, downlink_root=output, log_path=log)
    if previous_outputs:
        assert (output / "previous.npy").read_bytes() == b"previous tile"
        assert list(output.iterdir()) == [output / "previous.npy"]
        assert log.read_bytes() == b"previous telemetry\n"
    else:
        assert not output.exists()
        assert not log.exists()
    assert not [path for path in model.parent.parent.rglob("*") if ".backup-" in path.name]


def test_successful_publication_replaces_both_previous_outputs(filter_case):
    model, data, policy, output, log = filter_case
    output.mkdir()
    (output / "previous.npy").write_bytes(b"previous tile")
    log.parent.mkdir()
    log.write_bytes(b"previous telemetry\n")
    summary = filter_tiles(model, data, policy, downlink_root=output, log_path=log)
    assert summary["tiles_retained"] == 1
    assert (output / "tile.npy").read_bytes() == (data / "tile.npy").read_bytes()
    assert not (output / "previous.npy").exists()
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["downlink_materialized"] is True
    assert record["inference_ok"] is True
    assert not [path for path in model.parent.parent.rglob("*") if ".backup-" in path.name]


def test_input_rewrite_preserving_mtime_uses_retention_fallback(filter_case, monkeypatch):
    import os
    import phi2_tile_filter.runtime as runtime
    from phi2_tile_filter.policy import DecisionPolicy

    model, data, _, _, _ = filter_case
    tile = data / "tile.npy"
    runner = runtime.OnnxRunner(model, intra_op_threads=1)
    original_loader = runtime.load_tile_numpy

    def rewrite_during_preprocessing(path, **kwargs):
        original_stat = Path(path).stat()
        np.save(path, np.full((8, 8, 1), 0.75, dtype=np.float32))
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        assert Path(path).stat().st_size == original_stat.st_size
        assert Path(path).stat().st_mtime_ns == original_stat.st_mtime_ns
        return original_loader(path, **kwargs)

    monkeypatch.setattr(runtime, "load_tile_numpy", rewrite_during_preprocessing)
    record = runner.evaluate_file(tile, DecisionPolicy(0.8))
    assert record["input_observation_ok"] is False
    assert record["inference_ok"] is False
    assert record["retention_requested"] is True
    assert record["failure_stage"] == "preprocess_input"


def test_file_output_preflight_rejects_hardlinked_input(tmp_path):
    from phi2_tile_filter.filesystem import assert_file_outputs_disjoint

    source = tmp_path / "model.onnx"
    source.write_bytes(b"source model")
    output = tmp_path / "report.json"
    output.hardlink_to(source)
    with pytest.raises(ValueError, match="overlap"):
        assert_file_outputs_disjoint([output], protected_paths=[source])
    assert source.read_bytes() == b"source model"


def test_file_output_preflight_rejects_overlapping_outputs(tmp_path):
    from phi2_tile_filter.filesystem import assert_file_outputs_disjoint

    with pytest.raises(ValueError, match="overlap"):
        assert_file_outputs_disjoint([tmp_path / "report", tmp_path / "report" / "nested.json"])


def test_failed_log_sync_preserves_previous_outputs(filter_case, monkeypatch):
    import phi2_tile_filter.bandwidth_filter as module

    model, data, policy, output, log = filter_case
    output.mkdir()
    (output / "previous.npy").write_bytes(b"previous tile")
    log.parent.mkdir()
    log.write_bytes(b"previous telemetry\n")

    def fail_sync(descriptor):
        raise OSError("injected disk sync failure")

    monkeypatch.setattr(module.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="injected disk sync failure"):
        filter_tiles(model, data, policy, downlink_root=output, log_path=log)
    assert (output / "previous.npy").read_bytes() == b"previous tile"
    assert log.read_bytes() == b"previous telemetry\n"


def test_staged_report_preserves_old_file_on_sync_failure(tmp_path, monkeypatch):
    import phi2_tile_filter.filesystem as filesystem

    output = tmp_path / "report.json"
    output.write_bytes(b"previous report")

    def fail_sync(descriptor):
        raise OSError("injected disk sync failure")

    monkeypatch.setattr(filesystem.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="injected disk sync failure"):
        with filesystem.staged_text_file(output) as handle:
            handle.write("replacement report")
    assert output.read_bytes() == b"previous report"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("target", ["model", "tile", "schema"])
def test_inference_report_cannot_overwrite_inputs(filter_case, monkeypatch, target):
    import sys
    import phi2_tile_filter.infer_onnx as inference

    model, data, _, _, _ = filter_case
    target_path = {"model": model, "tile": data / "tile.npy", "schema": data / "input_schema.json"}[target]
    before = target_path.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["infer_onnx", "--onnx", str(model), "--data", str(data), "--out", str(target_path)]
    )
    with pytest.raises(ValueError, match="overlap"):
        inference.main()
    assert target_path.read_bytes() == before


@pytest.mark.parametrize("artifact_name", ["model", "schema"])
def test_runtime_rejects_artifact_replacement_during_session_load(
    filter_case, monkeypatch, artifact_name
):
    import phi2_tile_filter.runtime as runtime

    model, _, _, _, _ = filter_case
    artifact = model if artifact_name == "model" else model_schema_sidecar_path(model)
    original_session = runtime.ort.InferenceSession

    def replace_after_load(*args, **kwargs):
        session = original_session(*args, **kwargs)
        replacement = artifact.with_name("replacement-" + artifact.name)
        if artifact_name == "model":
            import onnx

            changed = onnx.load(str(artifact))
            changed.graph.node[0].attribute[0].t.float_data[:] = [0.0, 1.0]
            onnx.save(changed, replacement)
        else:
            replacement.write_bytes(artifact.read_bytes() + b"\n")
        assert replacement.read_bytes() != artifact.read_bytes()
        replacement.replace(artifact)
        return session

    monkeypatch.setattr(runtime.ort, "InferenceSession", replace_after_load)
    with pytest.raises(ValueError, match="file changed while the runtime was loading"):
        runtime.OnnxRunner(model, intra_op_threads=1)


def test_runtime_binds_model_target_when_alias_changes_during_session_load(tmp_path, monkeypatch):
    import onnx
    import phi2_tile_filter.runtime as runtime
    from phi2_tile_filter.utils import sha256_file

    model, schema = _write_binary_model(tmp_path / "original")
    alternate = tmp_path / "alternate.onnx"
    changed = onnx.load(str(model))
    changed.graph.node[0].attribute[0].t.float_data[:] = [0.0, 1.0]
    onnx.save(changed, alternate)
    alias = tmp_path / "selected.onnx"
    alias.symlink_to(model)
    # The chosen sidecar belongs to the requested name, not its model target.
    model_schema_sidecar_path(model).unlink()
    write_input_schema(model_schema_sidecar_path(alias), schema)
    original_session = runtime.ort.InferenceSession

    def switch_alias_during_load(*args, **kwargs):
        alias.unlink()
        alias.symlink_to(alternate)
        try:
            return original_session(*args, **kwargs)
        finally:
            alias.unlink()
            alias.symlink_to(model)

    monkeypatch.setattr(runtime.ort, "InferenceSession", switch_alias_during_load)
    runner = runtime.OnnxRunner(alias, intra_op_threads=1)
    logits, _ = runner.logits_for_array(np.full((1, 8, 8), 0.5, dtype=np.float32))
    np.testing.assert_array_equal(logits, [[1.0, 0.0]])
    assert runner.model_sha256 == sha256_file(model)
    assert runner.model_path == model.resolve()
    assert runner.input_schema_path == model_schema_sidecar_path(alias).resolve()


def test_runtime_binds_schema_target_when_alias_changes_during_validation(tmp_path, monkeypatch):
    import phi2_tile_filter.runtime as runtime
    from phi2_tile_filter.utils import sha256_file

    model, schema = _write_binary_model(tmp_path / "original")
    original_schema = model_schema_sidecar_path(model)
    alternate_schema = tmp_path / "alternate-schema.json"
    changed_schema = json.loads(json.dumps(schema))
    changed_schema["tensor"]["bands"][0]["id"] = "different_band"
    write_input_schema(alternate_schema, changed_schema)
    alias = tmp_path / "selected-schema.json"
    alias.symlink_to(original_schema)
    original_validate = runtime.validate_model_input_schema_binding

    def switch_alias_during_validation(*args, **kwargs):
        alias.unlink()
        alias.symlink_to(alternate_schema)
        try:
            return original_validate(*args, **kwargs)
        finally:
            alias.unlink()
            alias.symlink_to(original_schema)

    monkeypatch.setattr(runtime, "validate_model_input_schema_binding", switch_alias_during_validation)
    runner = runtime.OnnxRunner(model, input_schema_path=alias, intra_op_threads=1)
    assert runner.band_ids == tuple(band["id"] for band in schema["tensor"]["bands"])
    assert runner.input_schema_path == original_schema.resolve()
    assert runner.input_schema_file_sha256 == sha256_file(original_schema)


@pytest.mark.parametrize("command", ["downlink", "inference"])
@pytest.mark.parametrize("component", ["bundle.json", "validation.json"])
@pytest.mark.parametrize("alias_mode", ["direct", "model_alias", "output_alias"])
def test_runtime_outputs_protect_bundle_evidence(
    filter_case, monkeypatch, command, component, alias_mode
):
    import sys
    import phi2_tile_filter.infer_onnx as inference

    model, data, policy, output, _ = filter_case
    manifest = model.parent / "bundle.json"
    manifest.write_bytes(b'{"schema_version": 2}\n')
    validation = model.parent / "validation.json"
    validation.write_bytes(b'{"accepted": true}\n')
    destination = model.parent / component
    original = destination.read_bytes()
    selected_model = model
    if alias_mode == "model_alias":
        selected_model = model.parent.parent / "selected.onnx"
        selected_model.symlink_to(model)
        model_schema_sidecar_path(selected_model).symlink_to(model_schema_sidecar_path(model))
    elif alias_mode == "output_alias":
        alias = model.parent.parent / "report-alias.json"
        alias.symlink_to(destination)
        destination = alias

    with pytest.raises(ValueError, match="overlap"):
        if command == "downlink":
            filter_tiles(selected_model, data, policy, downlink_root=output, log_path=destination)
        else:
            monkeypatch.setattr(
                sys,
                "argv",
                ["infer_onnx", "--onnx", str(selected_model), "--data", str(data), "--out", str(destination)],
            )
            inference.main()
    assert destination.read_bytes() == original

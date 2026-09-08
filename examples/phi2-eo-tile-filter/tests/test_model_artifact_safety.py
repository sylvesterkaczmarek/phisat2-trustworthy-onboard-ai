from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch

from phi2_tile_filter import export_onnx, quantize_ptq
from phi2_tile_filter.input_schema import (
    build_input_schema,
    input_schema_sha256,
    model_schema_sidecar_path,
    validate_model_input_schema_binding,
    write_input_schema,
)
from phi2_tile_filter.models.tiny_cnn import TinyCNN
from phi2_tile_filter.utils import sha256_file


def _checkpoint(path: Path, *, first_logit: float = 1.0) -> None:
    schema = build_input_schema(bands=2, height=8)
    model = TinyCNN(in_ch=2, num_classes=2, base=2)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([first_logit, 0.0]))
    torch.save(
        {
            "format_version": 3,
            "architecture": "TinyCNN",
            "in_ch": 2,
            "num_classes": 2,
            "base": 2,
            "input_size": 8,
            "input_schema": schema,
            "input_schema_sha256": input_schema_sha256(schema),
            "state_dict": model.state_dict(),
        },
        path,
    )


def _small_onnx(path: Path, *, with_schema: bool = True) -> None:
    helper = onnx.helper
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "weights"], ["features"]),
            helper.make_node("GlobalAveragePool", ["features"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["logits"], axis=1),
        ],
        "calibration-test",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, ["batch", 2, 8, 8])],
        [helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, ["batch", 2])],
        [onnx.numpy_helper.from_array(np.ones((2, 2, 1, 1), dtype=np.float32), "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    schema = build_input_schema(bands=2, height=8)
    helper.set_model_props(model, {"input_schema_sha256": input_schema_sha256(schema)})
    onnx.save(model, path)
    if with_schema:
        write_input_schema(model_schema_sidecar_path(path), schema)


def _fake_export(monkeypatch: pytest.MonkeyPatch, logits: np.ndarray) -> None:
    def export(model, inputs, path, **kwargs):
        _small_onnx(Path(path), with_schema=False)

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, *args, **kwargs):
            return [logits]

    monkeypatch.setattr(export_onnx.torch.onnx, "export", export)
    monkeypatch.setattr(export_onnx.ort, "InferenceSession", Session)


def _old_outputs(destination: Path, summary_suffix: str) -> dict[Path, bytes]:
    outputs = (
        destination,
        model_schema_sidecar_path(destination),
        destination.with_suffix(destination.suffix + summary_suffix),
    )
    for index, path in enumerate(outputs):
        path.write_bytes(f"accepted artifact {index}".encode())
    return {path: path.read_bytes() for path in outputs}


def _assert_unchanged(previous: dict[Path, bytes]) -> None:
    for path, contents in previous.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -float("inf"), -1.0, True])
def test_export_rejects_invalid_tolerance_before_reading_inputs(tmp_path: Path, tolerance) -> None:
    with pytest.raises(ValueError, match="verify_atol"):
        export_onnx.export_model(tmp_path / "missing.pt", tmp_path / "model.onnx", verify_atol=tolerance)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("first_logit,ort_logit", [(1.0, np.nan), (np.nan, np.nan), (np.inf, np.inf)])
def test_export_rejects_nonfinite_logits_without_replacing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_logit: float, ort_logit: float
) -> None:
    weights = tmp_path / "weights.pt"
    _checkpoint(weights, first_logit=first_logit)
    destination = tmp_path / "model.onnx"
    previous = _old_outputs(destination, ".validation.json")
    previous[weights] = weights.read_bytes()
    _fake_export(monkeypatch, np.tile([ort_logit, 0.0], (3, 1)).astype(np.float32))
    with pytest.raises(RuntimeError, match="non-finite"):
        export_onnx.export_model(weights, destination)
    _assert_unchanged(previous)
    assert set(tmp_path.iterdir()) == set(previous)


def test_export_verification_failure_preserves_previous_model_and_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "weights.pt"
    _checkpoint(weights)
    destination = tmp_path / "model.onnx"
    previous = _old_outputs(destination, ".validation.json")
    _fake_export(monkeypatch, np.tile([2.0, 0.0], (3, 1)).astype(np.float32))
    with pytest.raises(RuntimeError, match="verification failed"):
        export_onnx.export_model(weights, destination)
    _assert_unchanged(previous)
    assert set(tmp_path.iterdir()) == {*previous, weights}


@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink", "summary"])
def test_export_cannot_replace_source_checkpoint(
    tmp_path: Path, alias: str
) -> None:
    destination = tmp_path / "model.onnx"
    weights = destination.with_suffix(".onnx.validation.json") if alias == "summary" else tmp_path / "weights.pt"
    _checkpoint(weights)
    original = weights.read_bytes()
    if alias == "same":
        destination = weights
    elif alias == "hardlink":
        destination.hardlink_to(weights)
    elif alias == "symlink":
        destination.symlink_to(weights)
    with pytest.raises(ValueError, match="overlapping"):
        export_onnx.export_model(weights, destination)
    assert weights.read_bytes() == original


def _quantization_inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.onnx"
    _small_onnx(source)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    write_input_schema(calibration / "input_schema.json", build_input_schema(bands=2, height=8))
    np.save(calibration / "tile.npy", np.full((8, 8, 2), 0.5, dtype=np.float32))
    return source, calibration


@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink", "schema", "summary"])
def test_quantization_cannot_replace_source_or_its_sidecars(tmp_path: Path, alias: str) -> None:
    source, calibration = _quantization_inputs(tmp_path)
    destination = tmp_path / "int8.onnx"
    source_schema = model_schema_sidecar_path(source)
    if alias == "same":
        destination = source
    elif alias == "hardlink":
        destination.hardlink_to(source)
    elif alias == "symlink":
        destination.symlink_to(source)
    elif alias == "schema":
        destination = source_schema
    elif alias == "summary":
        destination.with_suffix(".onnx.quantization.json").hardlink_to(source)
    original = {path: path.read_bytes() for path in (source, source_schema)}
    with pytest.raises(ValueError, match="overlapping"):
        quantize_ptq.quantize(source, calibration, destination)
    _assert_unchanged(original)


def test_failed_quantization_preserves_previous_model_and_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, calibration = _quantization_inputs(tmp_path)
    destination = tmp_path / "int8.onnx"
    previous = _old_outputs(destination, ".quantization.json")
    before_paths = set(tmp_path.iterdir())

    def quantize_without_qdq(*, model_output, **kwargs):
        Path(model_output).write_bytes(source.read_bytes())

    monkeypatch.setattr(quantize_ptq, "quantize_static", quantize_without_qdq)
    with pytest.raises(RuntimeError, match="QDQ"):
        quantize_ptq.quantize(source, calibration, destination)
    _assert_unchanged(previous)
    assert set(tmp_path.iterdir()) == before_paths


def test_successful_quantization_keeps_source_hash_and_publishes_bound_model(tmp_path: Path) -> None:
    source, calibration = _quantization_inputs(tmp_path)
    source_hash = sha256_file(source)
    destination = tmp_path / "int8.onnx"
    summary = quantize_ptq.quantize(source, calibration, destination)
    assert sha256_file(source) == summary["source_onnx_sha256"] == source_hash
    assert summary["quantized_onnx_sha256"] == sha256_file(destination)
    assert summary["input_schema_sha256"] == validate_model_input_schema_binding(destination)[1]
    metadata = {item.key: item.value for item in onnx.load(destination).metadata_props}
    assert metadata["quantized_from_sha256"] == source_hash


@pytest.mark.parametrize("operation", ["export", "quantize"])
def test_model_output_rejects_conflicting_shared_schema_before_publication(
    tmp_path: Path, operation: str
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    destination = output_root / "model.onnx"
    summary_suffix = ".validation.json" if operation == "export" else ".quantization.json"
    previous = _old_outputs(destination, summary_suffix)
    shared_schema = output_root / "input_schema.json"
    write_input_schema(shared_schema, build_input_schema(bands=3, height=8))
    previous[shared_schema] = shared_schema.read_bytes()
    if operation == "export":
        weights = tmp_path / "weights.pt"
        _checkpoint(weights)
        with pytest.raises(ValueError, match="output directory input_schema.json conflicts"):
            export_onnx.export_model(weights, destination)
    else:
        source, calibration = _quantization_inputs(tmp_path)
        with pytest.raises(ValueError, match="output directory input_schema.json conflicts"):
            quantize_ptq.quantize(source, calibration, destination)
    _assert_unchanged(previous)
    assert set(output_root.iterdir()) == set(previous)

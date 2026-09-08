from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from .filesystem import (
    assert_file_outputs_disjoint,
    remove_stage,
    replace_outputs_from_stages,
    sibling_stage_path,
)
from .input_schema import (
    assert_dataset_schema_compatible,
    input_schema_sha256,
    model_schema_sidecar_path,
    read_input_schema,
    validate_model_input_schema_binding,
    write_input_schema,
)
from .utils import discover_tile_files, load_tile_numpy, sha256_file


def onnx_input_spec(path: str | Path) -> tuple[str, int, int]:
    model = onnx.load(str(path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializer_names]
    if len(inputs) != 1:
        raise ValueError(f"expected one ONNX input, found {len(inputs)}")
    value = inputs[0]
    dims = value.type.tensor_type.shape.dim
    if len(dims) != 4:
        raise ValueError("expected NCHW ONNX input")
    bands = dims[1].dim_value
    height = dims[2].dim_value
    width = dims[3].dim_value
    if not bands or not height or not width or height != width:
        raise ValueError("ONNX model must have static C,H,W dimensions with H == W")
    return value.name, int(bands), int(height)


class TileReader(CalibrationDataReader):
    def __init__(self, folder: str | Path, *, input_name: str, input_schema: dict, batch: int = 8):
        self.files = discover_tile_files(folder)
        if not self.files:
            raise ValueError(f"no calibration tiles found under {folder}")
        self.input_name = input_name
        self.input_schema = input_schema
        self.batch = batch
        self._iterator = None

    def _generator(self):
        for start in range(0, len(self.files), self.batch):
            batch_files = self.files[start : start + self.batch]
            arrays = [load_tile_numpy(path, input_schema=self.input_schema) for path in batch_files]
            yield {self.input_name: np.stack(arrays).astype(np.float32, copy=False)}

    def get_next(self):
        if self._iterator is None:
            self._iterator = iter(self._generator())
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = iter(self._generator())


def quantize(model_input: str | Path, calibration_dir: str | Path, output: str | Path) -> dict:
    model_input = Path(model_input)
    calibration_dir = Path(calibration_dir)
    input_schema, schema_hash, source_schema = validate_model_input_schema_binding(model_input)
    assert_dataset_schema_compatible(calibration_dir, input_schema)
    input_name, bands, size = onnx_input_spec(model_input)
    if len(input_schema["tensor"]["bands"]) != bands:
        raise ValueError("ONNX channel count does not match input schema band count")
    if int(input_schema["tensor"]["height"]) != size or int(input_schema["tensor"]["width"]) != size:
        raise ValueError("ONNX spatial dimensions do not match input schema")

    destination = Path(output)
    shared_schema = destination.parent / "input_schema.json"
    if shared_schema.is_file() and input_schema_sha256(read_input_schema(shared_schema)) != schema_hash:
        raise ValueError("output directory input_schema.json conflicts with the model contract")
    schema_destination = model_schema_sidecar_path(destination)
    sidecar = destination.with_suffix(destination.suffix + ".quantization.json")
    reader = TileReader(calibration_dir, input_name=input_name, input_schema=input_schema)
    outputs = assert_file_outputs_disjoint(
        (destination, schema_destination, sidecar),
        protected_paths=(
            model_input,
            shared_schema,
            source_schema,
            model_schema_sidecar_path(model_input),
            model_input.with_suffix(model_input.suffix + ".validation.json"),
            model_input.with_suffix(model_input.suffix + ".quantization.json"),
            *reader.files,
            calibration_dir / "input_schema.json",
            calibration_dir.parent / "input_schema.json",
        ),
    )
    source_hash = sha256_file(model_input)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    stages = tuple(sibling_stage_path(path) for path in outputs)
    model_stage, schema_stage, summary_stage = stages
    try:
        quantize_static(
            model_input=str(model_input),
            model_output=str(model_stage),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": False},
        )
        quantized = onnx.load(str(model_stage))
        onnx.checker.check_model(quantized)
        ops = [node.op_type for node in quantized.graph.node]
        if "QuantizeLinear" not in ops or "DequantizeLinear" not in ops:
            raise RuntimeError("quantized model does not contain expected QDQ operators")
        metadata = {item.key: item.value for item in quantized.metadata_props}
        metadata.update(
            {
                "input_schema_version": str(input_schema["schema_version"]),
                "input_schema_sha256": schema_hash,
                "preprocessing_name": str(input_schema["preprocessing"]["name"]),
                "preprocessing_version": str(input_schema["preprocessing"]["version"]),
                "quantized_from_sha256": source_hash,
            }
        )
        onnx.helper.set_model_props(quantized, metadata)
        onnx.checker.check_model(quantized)
        onnx.save(quantized, str(model_stage))
        write_input_schema(schema_stage, input_schema)
        validate_model_input_schema_binding(model_stage, schema_stage)

        summary = {
            "schema_version": 2,
            "source_onnx_sha256": source_hash,
            "quantized_onnx_sha256": sha256_file(model_stage),
            "input_schema_sha256": schema_hash,
            "input_schema": str(model_schema_sidecar_path(destination)),
            "calibration_samples": len(reader.files),
            "bands": bands,
            "size": size,
            "quant_format": "QDQ",
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "per_channel": True,
        }
        summary_stage.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        replace_outputs_from_stages(zip(stages, outputs))
    finally:
        for path in stages:
            remove_stage(path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    quantize(args.onnx, args.calib, args.out)


if __name__ == "__main__":
    main()

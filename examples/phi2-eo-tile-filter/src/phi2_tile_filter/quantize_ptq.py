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
    def __init__(self, folder: str | Path, *, input_name: str, bands: int, size: int, batch: int = 8):
        self.files = discover_tile_files(folder)
        if not self.files:
            raise ValueError(f"no calibration tiles found under {folder}")
        self.input_name = input_name
        self.bands = bands
        self.size = size
        self.batch = batch
        self._iterator = None

    def _generator(self):
        for start in range(0, len(self.files), self.batch):
            batch_files = self.files[start : start + self.batch]
            arrays = [load_tile_numpy(path, bands=self.bands, size=self.size) for path in batch_files]
            yield {self.input_name: np.stack(arrays).astype(np.float32, copy=False)}

    def get_next(self):
        if self._iterator is None:
            self._iterator = iter(self._generator())
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = iter(self._generator())


def quantize(model_input: str | Path, calibration_dir: str | Path, output: str | Path) -> dict:
    input_name, bands, size = onnx_input_spec(model_input)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = TileReader(calibration_dir, input_name=input_name, bands=bands, size=size)
    quantize_static(
        model_input=str(model_input),
        model_output=str(destination),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False},
    )
    quantized = onnx.load(str(destination))
    onnx.checker.check_model(quantized)
    ops = [node.op_type for node in quantized.graph.node]
    if "QuantizeLinear" not in ops or "DequantizeLinear" not in ops:
        raise RuntimeError("quantized model does not contain expected QDQ operators")
    summary = {
        "schema_version": 1,
        "source_onnx_sha256": sha256_file(model_input),
        "quantized_onnx_sha256": sha256_file(destination),
        "calibration_samples": len(reader.files),
        "bands": bands,
        "size": size,
        "quant_format": "QDQ",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "per_channel": True,
    }
    sidecar = destination.with_suffix(destination.suffix + ".quantization.json")
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .filesystem import (
    assert_file_outputs_disjoint,
    remove_stage,
    replace_outputs_from_stages,
    sibling_stage_path,
)
from .input_schema import (
    input_schema_sha256,
    model_schema_sidecar_path,
    read_input_schema,
    validate_input_schema,
    write_input_schema,
)
from .models.tiny_cnn import TinyCNN
from .utils import sha256_file


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 3:
        raise ValueError("unsupported checkpoint format; retrain with the versioned input contract")
    required = {
        "architecture",
        "in_ch",
        "num_classes",
        "base",
        "input_size",
        "input_schema",
        "input_schema_sha256",
        "state_dict",
    }
    if not required.issubset(checkpoint):
        raise ValueError("checkpoint metadata is incomplete")
    if checkpoint["architecture"] != "TinyCNN":
        raise ValueError("unsupported architecture")
    validate_input_schema(checkpoint["input_schema"])
    schema_hash = input_schema_sha256(checkpoint["input_schema"])
    if checkpoint["input_schema_sha256"] != schema_hash:
        raise ValueError("checkpoint input schema hash does not match embedded input schema")
    return checkpoint


def export_model(weights: str | Path, output: str | Path, *, verify_atol: float = 1e-5) -> dict:
    if isinstance(verify_atol, bool) or not isinstance(verify_atol, Real):
        raise ValueError("verify_atol must be a finite, non-negative number")
    try:
        verify_atol = float(verify_atol)
    except OverflowError as exc:
        raise ValueError("verify_atol must be a finite, non-negative number") from exc
    if not math.isfinite(verify_atol) or verify_atol < 0.0:
        raise ValueError("verify_atol must be a finite, non-negative number")
    weights = Path(weights)
    destination = Path(output)
    schema_destination = model_schema_sidecar_path(destination)
    shared_schema = destination.parent / "input_schema.json"
    sidecar = destination.with_suffix(destination.suffix + ".validation.json")
    outputs = assert_file_outputs_disjoint(
        (destination, schema_destination, sidecar),
        protected_paths=(weights, weights.with_suffix(weights.suffix + ".json"), shared_schema),
    )
    checkpoint = load_checkpoint(weights)
    checkpoint_hash = sha256_file(weights)
    bands = int(checkpoint["in_ch"])
    classes = int(checkpoint["num_classes"])
    base = int(checkpoint["base"])
    size = int(checkpoint["input_size"])
    input_schema = checkpoint["input_schema"]
    schema_hash = input_schema_sha256(input_schema)
    if shared_schema.is_file() and input_schema_sha256(read_input_schema(shared_schema)) != schema_hash:
        raise ValueError("output directory input_schema.json conflicts with the model contract")
    tensor = input_schema["tensor"]
    if len(tensor["bands"]) != bands or int(tensor["height"]) != size or int(tensor["width"]) != size:
        raise ValueError("checkpoint architecture and input schema dimensions disagree")

    model = TinyCNN(in_ch=bands, num_classes=classes, base=base)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    torch.manual_seed(1234)
    export_input = torch.randn(2, bands, size, size, dtype=torch.float32)
    verify_input = torch.randn(3, bands, size, size, dtype=torch.float32)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    stages = tuple(sibling_stage_path(path) for path in outputs)
    model_stage, schema_stage, summary_stage = stages
    try:
        batch_dim = torch.export.Dim("batch")
        torch.onnx.export(
            model,
            (export_input,),
            str(model_stage),
            input_names=["input"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
            dynamic_shapes=({0: batch_dim},),
            external_data=False,
        )

        onnx_model = onnx.load(str(model_stage))
        onnx.helper.set_model_props(
            onnx_model,
            {
                "architecture": "TinyCNN",
                "bands": str(bands),
                "input_size": str(size),
                "base": str(base),
                "num_classes": str(classes),
                "checkpoint_sha256": checkpoint_hash,
                "batch_dimension": "dynamic",
                "input_schema_version": str(input_schema["schema_version"]),
                "input_schema_sha256": schema_hash,
                "preprocessing_name": str(input_schema["preprocessing"]["name"]),
                "preprocessing_version": str(input_schema["preprocessing"]["version"]),
            },
        )
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, str(model_stage))
        write_input_schema(schema_stage, input_schema)

        with torch.no_grad():
            torch_logits = model(verify_input).numpy()
        session = ort.InferenceSession(str(model_stage), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        ort_logits = np.asarray(session.run(None, {input_name: verify_input.numpy()})[0])
        if ort_logits.shape != torch_logits.shape:
            raise RuntimeError(
                f"PyTorch/ONNX output shape mismatch: {torch_logits.shape} versus {ort_logits.shape}"
            )
        if not np.isfinite(torch_logits).all() or not np.isfinite(ort_logits).all():
            raise RuntimeError("PyTorch/ONNX verification failed: non-finite logits")
        max_abs = float(np.max(np.abs(torch_logits - ort_logits)))
        argmax_agreement = float(np.mean(torch_logits.argmax(1) == ort_logits.argmax(1)))
        if max_abs > verify_atol or argmax_agreement != 1.0:
            raise RuntimeError(
                f"PyTorch/ONNX verification failed: max_abs={max_abs:.3e}, agreement={argmax_agreement:.3f}"
            )

        summary = {
            "schema_version": 2,
            "onnx": str(destination),
            "onnx_sha256": sha256_file(model_stage),
            "checkpoint_sha256": checkpoint_hash,
            "input_schema": str(model_schema_sidecar_path(destination)),
            "input_schema_sha256": schema_hash,
            "input_band_ids": [band["id"] for band in input_schema["tensor"]["bands"]],
            "preprocessing_version": input_schema["preprocessing"]["version"],
            "bands": bands,
            "size": size,
            "base": base,
            "batch_dimension": "dynamic",
            "verification_batch_size": int(verify_input.shape[0]),
            "pytorch_onnx_max_abs_error": max_abs,
            "pytorch_onnx_argmax_agreement": argmax_agreement,
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
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify-atol", type=float, default=1e-5)
    args = parser.parse_args()
    export_model(args.weights, args.out, verify_atol=args.verify_atol)


if __name__ == "__main__":
    main()

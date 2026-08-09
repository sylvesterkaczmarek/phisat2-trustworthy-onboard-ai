from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .models.tiny_cnn import TinyCNN
from .utils import sha256_file


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
        raise ValueError("unsupported checkpoint format")
    required = {"architecture", "in_ch", "num_classes", "base", "input_size", "state_dict"}
    if not required.issubset(checkpoint):
        raise ValueError("checkpoint metadata is incomplete")
    if checkpoint["architecture"] != "TinyCNN":
        raise ValueError("unsupported architecture")
    return checkpoint


def export_model(weights: str | Path, output: str | Path, *, verify_atol: float = 1e-5) -> dict:
    checkpoint = load_checkpoint(weights)
    bands = int(checkpoint["in_ch"])
    classes = int(checkpoint["num_classes"])
    base = int(checkpoint["base"])
    size = int(checkpoint["input_size"])

    model = TinyCNN(in_ch=bands, num_classes=classes, base=base)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    torch.manual_seed(1234)
    dummy = torch.randn(2, bands, size, size, dtype=torch.float32)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy[:1],),
        str(destination),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )

    onnx_model = onnx.load(str(destination))
    onnx.helper.set_model_props(
        onnx_model,
        {
            "architecture": "TinyCNN",
            "bands": str(bands),
            "input_size": str(size),
            "base": str(base),
            "num_classes": str(classes),
            "checkpoint_sha256": sha256_file(weights),
        },
    )
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(destination))

    with torch.no_grad():
        torch_logits = model(dummy).numpy()
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_logits = np.asarray(session.run(None, {input_name: dummy.numpy()})[0])
    max_abs = float(np.max(np.abs(torch_logits - ort_logits)))
    argmax_agreement = float(np.mean(torch_logits.argmax(1) == ort_logits.argmax(1)))
    if max_abs > verify_atol or argmax_agreement != 1.0:
        raise RuntimeError(
            f"PyTorch/ONNX verification failed: max_abs={max_abs:.3e}, agreement={argmax_agreement:.3f}"
        )

    summary = {
        "schema_version": 1,
        "onnx": str(destination),
        "onnx_sha256": sha256_file(destination),
        "checkpoint_sha256": sha256_file(weights),
        "bands": bands,
        "size": size,
        "base": base,
        "pytorch_onnx_max_abs_error": max_abs,
        "pytorch_onnx_argmax_agreement": argmax_agreement,
    }
    sidecar = destination.with_suffix(destination.suffix + ".validation.json")
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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

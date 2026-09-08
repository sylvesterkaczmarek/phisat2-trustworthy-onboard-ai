from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .filesystem import (
    assert_file_outputs_disjoint,
    remove_stage,
    replace_outputs_from_stages,
    sibling_stage_path,
)
from .input_schema import input_schema_sha256, read_input_schema
from .models.tiny_cnn import TinyCNN
from .utils import make_loader, read_dataset_manifest, seed_everything, sha256_file


def train(
    data_root: str | Path,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    base: int = 16,
    batch: int = 64,
    seed: int = 0,
    output: str | Path = "runs/tinycnn.pt",
    device_name: str = "cpu",
) -> dict:
    if any(type(value) is not int or value <= 0 for value in (epochs, base, batch)):
        raise ValueError("epochs, base, and batch must be positive integers")
    if isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    data_root = Path(data_root)
    destination = Path(output)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    assert_file_outputs_disjoint((destination, sidecar), protected_paths=(data_root,))
    manifest = read_dataset_manifest(data_root)
    input_schema = read_input_schema(data_root / "input_schema.json")
    schema_hash = input_schema_sha256(input_schema)
    if manifest.get("input_schema_sha256") != schema_hash:
        raise ValueError("dataset manifest and input_schema.json disagree")
    bands = len(input_schema["tensor"]["bands"])
    size = int(input_schema["tensor"]["height"])
    if int(input_schema["tensor"]["width"]) != size:
        raise ValueError("TinyCNN demonstrator requires square input tiles")
    seed_everything(seed)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    loader = make_loader(
        data_root / "train",
        batch=batch,
        shuffle=True,
        seed=seed,
        input_schema=input_schema,
    )
    model = TinyCNN(in_ch=bands, base=base).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        samples = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training produced a non-finite loss; checkpoint was not saved")
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * x.shape[0]
            samples += x.shape[0]
        epoch_loss = running / max(samples, 1)
        losses.append(epoch_loss)
        print(f"epoch {epoch + 1} loss {epoch_loss:.6f}")

    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise RuntimeError("training produced non-finite parameters; checkpoint was not saved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 3,
        "architecture": "TinyCNN",
        "in_ch": bands,
        "num_classes": 2,
        "base": int(base),
        "input_size": size,
        "input_schema": deepcopy(input_schema),
        "input_schema_sha256": schema_hash,
        "seed": int(seed),
        "epochs": int(epochs),
        "lr": float(lr),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    summary = {
        "schema_version": 2,
        "checkpoint": str(destination),
        "input_schema_sha256": schema_hash,
        "input_band_ids": [band["id"] for band in input_schema["tensor"]["bands"]],
        "preprocessing_version": input_schema["preprocessing"]["version"],
        "bands": bands,
        "size": size,
        "base": base,
        "seed": seed,
        "epochs": epochs,
        "final_train_loss": losses[-1],
    }
    staged_checkpoint = sibling_stage_path(destination)
    staged_sidecar = sibling_stage_path(sidecar)
    try:
        # Torch embeds the filename in its archive. Keep that name stable while
        # staging so repeated runs still produce identical checkpoint bytes.
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            named_checkpoint = Path(temporary) / destination.name
            torch.save(checkpoint, named_checkpoint)
            os.replace(named_checkpoint, staged_checkpoint)
        summary["checkpoint_sha256"] = sha256_file(staged_checkpoint)
        staged_sidecar.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        replace_outputs_from_stages(
            ((staged_checkpoint, destination), (staged_sidecar, sidecar))
        )
    finally:
        remove_stage(staged_checkpoint)
        remove_stage(staged_sidecar)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/tinycnn.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    train(
        args.data,
        epochs=args.epochs,
        lr=args.lr,
        base=args.base,
        batch=args.batch,
        seed=args.seed,
        output=args.out,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .filesystem import (
    assert_safe_tree_target,
    remove_stage,
    replace_tree_from_stage,
    sibling_stage_path,
)
from .input_schema import build_input_schema, write_input_schema

SPLITS = ("train", "calib", "validation", "test")
CLASSES = ("background", "event")
SPLIT_ROLES = {
    "train": "model_parameter_fitting",
    "calib": "quantization_and_policy_calibration",
    "validation": "model_and_quantization_acceptance",
    "test": "final_report_only",
}


def make_tile(rng: np.random.Generator, size: int, bands: int, cls: int) -> np.ndarray:
    if size < 8 or bands <= 0:
        raise ValueError("size must be >= 8 and bands must be positive")
    image = rng.uniform(0.0, 0.55, size=(size, size, bands)).astype(np.float32)
    if cls == 1:
        side = max(2, size // 3)
        x0 = int(rng.integers(0, size - side + 1))
        y0 = int(rng.integers(0, size - side + 1))
        signature = np.linspace(0.35, 0.55, bands, dtype=np.float32)
        image[y0 : y0 + side, x0 : x0 + side, :] += signature
        np.clip(image, 0.0, 1.0, out=image)
    return image


def split_counts(
    total: int,
    train_fraction: float,
    calib_fraction: float,
    validation_fraction: float = 0.10,
) -> dict[str, int]:
    if total < 16:
        raise ValueError("n must be at least 16 for four non-empty class-balanced-capable splits")
    fractions = {
        "train": train_fraction,
        "calib": calib_fraction,
        "validation": validation_fraction,
    }
    if any(not 0.0 < value < 1.0 for value in fractions.values()):
        raise ValueError("train, calibration, and validation fractions must be between 0 and 1")
    if sum(fractions.values()) >= 1.0:
        raise ValueError("train_fraction + calib_fraction + validation_fraction must be < 1")

    counts = {name: max(2, int(round(total * fraction))) for name, fraction in fractions.items()}
    counts["test"] = total - sum(counts.values())
    if counts["test"] < 2:
        raise ValueError("test split would contain fewer than two samples")
    if min(counts.values()) < 2 or sum(counts.values()) != total:
        raise ValueError("invalid four-way split allocation")
    return counts


def write_dataset(
    root: str | Path,
    *,
    n: int,
    bands: int,
    size: int,
    seed: int,
    train_fraction: float = 0.60,
    calib_fraction: float = 0.15,
    validation_fraction: float = 0.10,
    overwrite: bool = False,
) -> dict:
    root = assert_safe_tree_target(root, operation="synthetic dataset replacement")
    counts = split_counts(n, train_fraction, calib_fraction, validation_fraction)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists; pass --overwrite to replace it")

    stage = sibling_stage_path(root, label="stage")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        input_schema = build_input_schema(
            bands=bands,
            height=size,
            source_layout="HWC",
            source_format="npy",
            source_dtype="float32",
            value_range=(0.0, 1.0),
            normalization_name="identity_unit_interval",
            normalization_version=1,
            nodata_policy="reject",
        )
        schema_hash = write_input_schema(stage / "input_schema.json", input_schema)

        seed_sequence = np.random.SeedSequence(seed)
        child_sequences = seed_sequence.spawn(len(SPLITS))
        split_seed_spawn_keys: dict[str, list[int]] = {}
        for split, split_seed in zip(SPLITS, child_sequences):
            split_seed_spawn_keys[split] = [int(value) for value in split_seed.spawn_key]
            rng = np.random.default_rng(split_seed)
            count = counts[split]
            labels = np.array([i % 2 for i in range(count)], dtype=np.int64)
            rng.shuffle(labels)
            per_class = {name: 0 for name in CLASSES}
            for cls in labels:
                class_name = CLASSES[int(cls)]
                index = per_class[class_name]
                per_class[class_name] += 1
                tile = make_tile(rng, size, bands, int(cls))
                destination = stage / split / class_name / f"{index:05d}.npy"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, tile, allow_pickle=False)

        manifest = {
            "schema_version": 3,
            "generator": "synthetic-square-event-v4-explicit-input-contract",
            "seed": int(seed),
            "samples": int(sum(counts.values())),
            "bands": int(bands),
            "size": int(size),
            "class_names": list(CLASSES),
            "split_counts": counts,
            "split_roles": SPLIT_ROLES,
            "split_seed_spawn_keys": split_seed_spawn_keys,
            "split_fractions_requested": {
                "train": float(train_fraction),
                "calib": float(calib_fraction),
                "validation": float(validation_fraction),
                "test": float(1.0 - train_fraction - calib_fraction - validation_fraction),
            },
            "tile_format": "npy-float32-hwc-0to1",
            "input_schema_file": "input_schema.json",
            "input_schema_sha256": schema_hash,
            "input_band_ids": [band["id"] for band in input_schema["tensor"]["bands"]],
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        replace_tree_from_stage(stage, root)
        return manifest
    finally:
        remove_stage(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic toy multispectral EO dataset.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calib-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = write_dataset(
        args.out,
        n=args.n,
        bands=args.bands,
        size=args.size,
        seed=args.seed,
        train_fraction=args.train_fraction,
        calib_fraction=args.calib_fraction,
        validation_fraction=args.validation_fraction,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

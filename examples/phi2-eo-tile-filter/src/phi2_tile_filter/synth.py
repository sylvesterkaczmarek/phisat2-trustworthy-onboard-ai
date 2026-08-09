from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SPLITS = ("train", "calib", "test")
CLASSES = ("background", "event")


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


def split_counts(total: int, train_fraction: float, calib_fraction: float) -> dict[str, int]:
    if total < 12:
        raise ValueError("n must be at least 12 so every split has both classes")
    if not (0.0 < train_fraction < 1.0 and 0.0 < calib_fraction < 1.0):
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + calib_fraction >= 1.0:
        raise ValueError("train_fraction + calib_fraction must be < 1")
    train = max(2, int(round(total * train_fraction)))
    calib = max(2, int(round(total * calib_fraction)))
    test = total - train - calib
    if test < 2:
        raise ValueError("test split would contain fewer than two samples")
    counts = {"train": train, "calib": calib, "test": test}
    for key in ("train", "calib"):
        if counts[key] % 2:
            counts[key] -= 1
            counts["test"] += 1
    if counts["test"] % 2:
        counts["test"] -= 1
        counts["train"] += 1
    if min(counts.values()) < 2:
        raise ValueError("each split must contain at least two samples")
    return counts


def write_dataset(
    root: str | Path,
    *,
    n: int,
    bands: int,
    size: int,
    seed: int,
    train_fraction: float = 0.70,
    calib_fraction: float = 0.15,
    overwrite: bool = False,
) -> dict:
    root = Path(root)
    counts = split_counts(n, train_fraction, calib_fraction)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    root.mkdir(parents=True)

    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(SPLITS))
    for split, split_seed in zip(SPLITS, child_sequences):
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
            destination = root / split / class_name / f"{index:05d}.npy"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, tile, allow_pickle=False)

    manifest = {
        "schema_version": 1,
        "generator": "synthetic-square-event-v2",
        "seed": int(seed),
        "samples": int(sum(counts.values())),
        "bands": int(bands),
        "size": int(size),
        "class_names": list(CLASSES),
        "split_counts": counts,
        "tile_format": "npy-float32-hwc-0to1",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic toy multispectral EO dataset.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--calib-fraction", type=float, default=0.15)
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
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

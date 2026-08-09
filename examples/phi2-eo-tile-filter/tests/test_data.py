from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from phi2_tile_filter.synth import write_dataset
from phi2_tile_filter.utils import TileFolder, load_tile_numpy, read_dataset_manifest


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def test_synthetic_dataset_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    write_dataset(first, n=60, bands=7, size=24, seed=11)
    write_dataset(second, n=60, bands=7, size=24, seed=11)
    assert _tree_hash(first) == _tree_hash(second)


def test_train_calib_test_are_separate_and_balanced(tmp_path: Path) -> None:
    root = tmp_path / "tiles"
    manifest = write_dataset(root, n=80, bands=3, size=20, seed=0)
    assert set(manifest["split_counts"]) == {"train", "calib", "test"}
    for split in ("train", "calib", "test"):
        dataset = TileFolder(root / split, bands=3, size=20)
        labels = [int(dataset[index][1]) for index in range(len(dataset))]
        assert 0 in labels and 1 in labels


def test_multispectral_npy_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "tiles"
    write_dataset(root, n=48, bands=7, size=24, seed=2)
    manifest = read_dataset_manifest(root)
    path = next((root / "test" / "event").glob("*.npy"))
    tile = load_tile_numpy(path, bands=manifest["bands"], size=manifest["size"])
    assert tile.shape == (7, 24, 24)
    assert tile.dtype == np.float32
    assert 0.0 <= float(tile.min()) <= float(tile.max()) <= 1.0


def test_image_band_mismatch_is_rejected(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "tile.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match=r"use \.npy"):
        load_tile_numpy(path, bands=7, size=8)

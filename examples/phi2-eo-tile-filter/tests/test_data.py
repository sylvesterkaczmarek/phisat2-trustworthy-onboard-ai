from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from phi2_tile_filter.input_schema import (
    assert_dataset_schema_compatible,
    build_input_schema,
    input_schema_sha256,
    read_input_schema,
    validate_input_schema,
    write_input_schema,
)
from phi2_tile_filter.synth import SPLITS, write_dataset
from phi2_tile_filter.utils import TileFolder, load_tile_numpy, read_dataset_manifest, sha256_file


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def test_synthetic_dataset_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    write_dataset(first, n=80, bands=7, size=24, seed=11)
    write_dataset(second, n=80, bands=7, size=24, seed=11)
    assert _tree_hash(first) == _tree_hash(second)


def test_four_way_lifecycle_has_bound_input_schema(tmp_path: Path) -> None:
    root = tmp_path / "tiles"
    manifest = write_dataset(root, n=80, bands=3, size=20, seed=0)
    assert manifest["schema_version"] == 3
    assert tuple(manifest["split_counts"]) == SPLITS
    schema = read_input_schema(root / "input_schema.json")
    assert manifest["input_schema_sha256"] == input_schema_sha256(schema)
    assert manifest["input_band_ids"] == ["band_01", "band_02", "band_03"]

    all_hashes: set[str] = set()
    for split in SPLITS:
        dataset = TileFolder(root / split, input_schema=schema)
        labels = [int(dataset[index][1]) for index in range(len(dataset))]
        assert 0 in labels and 1 in labels
        split_hashes = {sha256_file(path) for path in (root / split).rglob("*.npy")}
        assert split_hashes.isdisjoint(all_hashes)
        all_hashes.update(split_hashes)


def test_correct_hwc_schema_loading(tmp_path: Path) -> None:
    schema = build_input_schema(bands=3, height=8, width=9, source_layout="HWC")
    path = tmp_path / "hwc.npy"
    array = np.zeros((8, 9, 3), dtype=np.float32)
    array[..., 1] = 0.5
    np.save(path, array)
    loaded = load_tile_numpy(path, input_schema=schema)
    assert loaded.shape == (3, 8, 9)
    assert np.allclose(loaded[1], 0.5)


def test_correct_chw_schema_loading(tmp_path: Path) -> None:
    schema = build_input_schema(bands=3, height=8, width=9, source_layout="CHW")
    path = tmp_path / "chw.npy"
    array = np.zeros((3, 8, 9), dtype=np.float32)
    array[2] = 0.75
    np.save(path, array)
    loaded = load_tile_numpy(path, input_schema=schema)
    assert loaded.shape == (3, 8, 9)
    assert np.allclose(loaded[2], 0.75)


def test_legacy_ambiguous_chw_hwc_layout_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.npy"
    np.save(path, np.zeros((3, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="ambiguous CHW/HWC"):
        load_tile_numpy(path, bands=3, size=8)


def test_wrong_band_order_schema_is_rejected_even_when_shape_matches(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    expected = build_input_schema(bands=3, height=8)
    wrong = deepcopy(expected)
    wrong["tensor"]["bands"][0], wrong["tensor"]["bands"][1] = (
        wrong["tensor"]["bands"][1],
        wrong["tensor"]["bands"][0],
    )
    for index, band in enumerate(wrong["tensor"]["bands"]):
        band["index"] = index
    write_input_schema(root / "input_schema.json", wrong)
    with pytest.raises(ValueError, match="does not match the model contract"):
        assert_dataset_schema_compatible(root, expected)


def test_mismatched_preprocessing_metadata_changes_contract_hash(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    expected = build_input_schema(bands=3, height=8)
    changed = deepcopy(expected)
    changed["normalization"]["version"] = 2
    write_input_schema(root / "input_schema.json", changed)
    assert input_schema_sha256(changed) != input_schema_sha256(expected)
    with pytest.raises(ValueError, match="does not match the model contract"):
        assert_dataset_schema_compatible(root, expected)


def test_rgb_is_not_silently_converted_to_rgba(tmp_path: Path) -> None:
    path = tmp_path / "tile.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(path)
    with pytest.raises(ValueError, match="never added, replicated, or discarded"):
        load_tile_numpy(path, bands=4, size=8)


def test_grayscale_is_not_silently_replicated_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "tile.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(path)
    with pytest.raises(ValueError, match="never added, replicated, or discarded"):
        load_tile_numpy(path, bands=3, size=8)


def test_high_bit_depth_tiff_is_explicitly_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tile.tiff"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint16)).save(path)
    with pytest.raises(ValueError, match="TIFF/GeoTIFF input is deliberately unsupported"):
        load_tile_numpy(path, bands=1, size=8)


def test_schema_source_dtype_is_enforced(tmp_path: Path) -> None:
    schema = build_input_schema(bands=2, height=8, source_dtype="float32")
    path = tmp_path / "wrong_dtype.npy"
    np.save(path, np.zeros((8, 8, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="does not match schema source dtype"):
        load_tile_numpy(path, input_schema=schema)


def test_multispectral_npy_round_trip_uses_dataset_contract(tmp_path: Path) -> None:
    root = tmp_path / "tiles"
    write_dataset(root, n=48, bands=7, size=24, seed=2)
    manifest = read_dataset_manifest(root)
    schema = read_input_schema(root / "input_schema.json")
    path = next((root / "test" / "event").glob("*.npy"))
    tile = load_tile_numpy(path, input_schema=schema)
    assert tile.shape == (7, 24, 24)
    assert tile.dtype == np.float32
    assert manifest["input_schema_sha256"] == input_schema_sha256(schema)
    assert 0.0 <= float(tile.min()) <= float(tile.max()) <= 1.0


def test_invalid_band_metadata_is_rejected() -> None:
    schema = build_input_schema(bands=2, height=8)
    schema["tensor"]["bands"][1]["index"] = 0
    with pytest.raises(ValueError, match="indices"):
        validate_input_schema(schema)

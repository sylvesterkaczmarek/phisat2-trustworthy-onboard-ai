from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from phi2_tile_filter.input_schema import (
    build_input_schema,
    find_model_input_schema,
    refresh_preprocessing_sha256,
    validate_input_schema,
    write_input_schema,
)
from phi2_tile_filter.synth import write_dataset
from phi2_tile_filter.utils import load_tile_numpy, read_dataset_manifest


def test_image_layout_cannot_reinterpret_decoded_rgb_rows_as_channels() -> None:
    with pytest.raises(ValueError, match="source layout must be HWC"):
        build_input_schema(
            bands=3,
            height=3,
            source_layout="CHW",
            source_format="png",
            source_dtype="uint8",
            value_range=(0, 255),
            normalization_name="uint8_to_unit_interval",
        )


def test_short_rgb_image_has_known_layout_even_when_height_equals_channels(tmp_path: Path) -> None:
    array = np.zeros((3, 8, 3), dtype=np.uint8)
    array[:, :, 0] = 255
    path = tmp_path / "short.png"
    Image.fromarray(array).save(path)
    tile = load_tile_numpy(path, bands=3, size=8)
    np.testing.assert_array_equal(tile[0], np.ones((8, 8), dtype=np.float32))
    np.testing.assert_array_equal(tile[1:], np.zeros((2, 8, 8), dtype=np.float32))


@pytest.mark.parametrize("strict", [False, True])
def test_complex_tiles_are_rejected_before_discarding_imaginary_data(tmp_path: Path, strict: bool) -> None:
    path = tmp_path / "complex.npy"
    np.save(path, np.full((8, 8, 1), 0.5 + 2j, dtype=np.complex64))
    arguments = (
        {"input_schema": build_input_schema(bands=1, height=8, source_dtype="complex64")}
        if strict
        else {"bands": 1, "size": 8}
    )
    with pytest.raises(ValueError, match="real integer or floating-point"):
        load_tile_numpy(path, **arguments)


def test_normalization_parameters_cannot_be_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "tile.npy"
    np.save(path, np.full((8, 8, 1), 0.5, dtype=np.float32))
    schema = build_input_schema(bands=1, height=8)
    schema["normalization"]["parameters"] = {"scale": 0.1}
    refresh_preprocessing_sha256(schema)
    with pytest.raises(ValueError, match="does not support parameters"):
        load_tile_numpy(path, input_schema=schema)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), True, "700"])
@pytest.mark.parametrize("field", ["wavelength", "wavelength_range", "value_range", "nodata"])
def test_input_contract_requires_finite_numeric_metadata(field: str, bad_value: object) -> None:
    schema = build_input_schema(bands=1, height=8)
    if field == "wavelength":
        schema["tensor"]["bands"][0]["wavelength_nm"] = bad_value
    elif field == "wavelength_range":
        schema["tensor"]["bands"][0]["wavelength_range_nm"] = [600, bad_value]
    elif field == "value_range":
        schema["source"]["value_range"] = [0, bad_value]
    else:
        schema["nodata"]["values"] = [bad_value]
    refresh_preprocessing_sha256(schema)
    with pytest.raises(ValueError, match="finite number"):
        validate_input_schema(schema)


@pytest.mark.parametrize("bad_value", [8.5, "8", True])
def test_tile_dimensions_cannot_be_silently_coerced(bad_value: object) -> None:
    schema = build_input_schema(bands=1, height=8)
    schema["tensor"]["height"] = bad_value
    with pytest.raises(ValueError, match="positive integers"):
        validate_input_schema(schema)
    with pytest.raises(ValueError, match="positive integers"):
        build_input_schema(bands=1, height=bad_value)


def test_explicit_schema_path_is_authoritative(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    fallback = tmp_path / "input_schema.json"
    write_input_schema(fallback, build_input_schema(bands=1, height=8))
    assert find_model_input_schema(model) == fallback
    with pytest.raises(FileNotFoundError, match="explicit input schema"):
        find_model_input_schema(model, tmp_path / "misspelled.json")


@pytest.mark.parametrize("fault", ["role", "count", "total"])
def test_dataset_manifest_rejects_misleading_split_metadata(tmp_path: Path, fault: str) -> None:
    root = tmp_path / "dataset"
    manifest = write_dataset(root, n=16, bands=1, size=8, seed=2)
    if fault == "role":
        manifest["split_roles"]["test"] = "model_parameter_fitting"
    elif fault == "count":
        manifest["split_counts"]["test"] = True
    else:
        manifest["samples"] += 1
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split-role|split counts|sum of split"):
        read_dataset_manifest(root)

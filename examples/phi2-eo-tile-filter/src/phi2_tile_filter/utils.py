from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .input_schema import (
    assert_dataset_schema_compatible,
    input_schema_sha256,
    read_input_schema,
    validate_input_schema,
)

CLASS_NAMES = ("background", "event")
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TIFF_SUFFIXES = {".tif", ".tiff"}
SUPPORTED_TILE_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | TIFF_SUFFIXES | {".npy"}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def discover_tile_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_TILE_SUFFIXES
    )


def discover_labeled_tiles(root: str | Path) -> list[tuple[Path, int]]:
    root = Path(root)
    items: list[tuple[Path, int]] = []
    for cls, name in enumerate(CLASS_NAMES):
        class_dir = root / name
        for path in discover_tile_files(class_dir):
            items.append((path, cls))
    return items


def _reject_tiff(path: Path) -> None:
    raise ValueError(
        f"TIFF/GeoTIFF input is deliberately unsupported by the generic loader ({path}). "
        "Scientific TIFF may contain high-bit-depth, multiband, geospatial, nodata, scale/offset, "
        "or radiometric semantics that PIL conversion would lose. Convert validated data to .npy "
        "under an explicit input schema, or provide a mission-specific EO loader."
    )


def _legacy_normalize_npy(array: np.ndarray) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("NumPy tiles must contain numeric data")
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        array = array.astype(np.float32) / float(info.max)
    else:
        array = array.astype(np.float32, copy=False)
    if array.size == 0:
        raise ValueError("tile is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("tile contains non-finite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError("floating-point NumPy tiles must already be scaled to [0, 1]")
    return np.clip(array, 0.0, 1.0)


def _source_format_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix == ".png":
        return "png"
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg"
    if suffix in TIFF_SUFFIXES:
        _reject_tiff(path)
    raise ValueError(f"unsupported tile format: {path.suffix}")


def _apply_schema_normalization(array: np.ndarray, schema: dict) -> np.ndarray:
    validate_input_schema(schema)
    source = schema["source"]
    normalization = schema["normalization"]
    expected_dtype = np.dtype(source["dtype"])
    if array.dtype != expected_dtype:
        raise ValueError(
            f"input dtype {array.dtype} does not match schema source dtype {expected_dtype}"
        )
    if array.size == 0:
        raise ValueError("tile is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("tile contains non-finite values")

    nodata = schema["nodata"]
    nodata_values = nodata.get("values", [])
    if nodata.get("policy") == "reject" and nodata_values:
        if any(np.any(array == value) for value in nodata_values):
            raise ValueError("tile contains a nodata value rejected by the input schema")

    low, high = map(float, source["value_range"])
    observed_min = float(array.min())
    observed_max = float(array.max())
    if observed_min < low - 1e-6 or observed_max > high + 1e-6:
        raise ValueError(
            f"tile values [{observed_min}, {observed_max}] fall outside schema source range [{low}, {high}]"
        )

    name = normalization["name"]
    version = int(normalization["version"])
    if version != 1:
        raise ValueError(f"unsupported normalization version: {version}")
    if name == "identity_unit_interval":
        if low < -1e-12 or high > 1.0 + 1e-12:
            raise ValueError("identity_unit_interval requires source values inside [0, 1]")
        result = array.astype(np.float32, copy=False)
    elif name == "uint8_to_unit_interval":
        if expected_dtype != np.dtype("uint8") or (low, high) != (0.0, 255.0):
            raise ValueError("uint8_to_unit_interval requires uint8 source range [0, 255]")
        result = array.astype(np.float32) / 255.0
    elif name == "integer_full_scale_to_unit_interval":
        if not np.issubdtype(expected_dtype, np.integer):
            raise ValueError("integer_full_scale_to_unit_interval requires an integer source dtype")
        info = np.iinfo(expected_dtype)
        if (low, high) != (float(info.min), float(info.max)):
            raise ValueError("integer full-scale normalization range does not match source dtype")
        result = (array.astype(np.float32) - float(info.min)) / float(info.max - info.min)
    else:
        raise ValueError(f"unsupported normalization procedure: {name}")

    if not np.all(np.isfinite(result)):
        raise ValueError("normalization produced non-finite values")
    if float(result.min()) < -1e-6 or float(result.max()) > 1.0 + 1e-6:
        raise ValueError("normalized model input must be in [0, 1]")
    return np.clip(result, 0.0, 1.0)


def _load_exact_image(path: Path, *, expected_channels: int) -> np.ndarray:
    if path.suffix.lower() in TIFF_SUFFIXES:
        _reject_tiff(path)
    with Image.open(path) as image:
        mode = image.mode
        array = np.asarray(image)
    if mode == "L":
        channels = 1
        if array.ndim != 2:
            raise ValueError(f"unexpected grayscale image shape {array.shape}")
    elif mode == "RGB":
        channels = 3
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"unexpected RGB image shape {array.shape}")
    elif mode == "RGBA":
        channels = 4
        if array.ndim != 3 or array.shape[-1] != 4:
            raise ValueError(f"unexpected RGBA image shape {array.shape}")
    else:
        raise ValueError(
            f"image mode {mode!r} is not accepted without an explicit conversion; "
            "use L, RGB, or RGBA data with an exact channel contract"
        )
    if channels != expected_channels:
        raise ValueError(
            f"image has {channels} channels but the input contract requires {expected_channels}; "
            "channels are never added, replicated, or discarded implicitly"
        )
    return array


def _legacy_layout(array: np.ndarray, *, bands: int, path: Path) -> np.ndarray:
    if array.ndim == 2:
        if bands != 1:
            raise ValueError(f"2-D tile {path} is only valid for bands=1")
        return array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"tile {path} must be 2-D or 3-D")
    chw_match = array.shape[0] == bands
    hwc_match = array.shape[-1] == bands
    if chw_match and hwc_match:
        raise ValueError(f"ambiguous CHW/HWC layout for {path} with shape {array.shape}")
    if chw_match:
        return array
    if hwc_match:
        return np.transpose(array, (2, 0, 1))
    raise ValueError(f"tile {path} has shape {array.shape}, which does not match bands={bands}")


def _schema_layout(array: np.ndarray, *, schema: dict, path: Path) -> np.ndarray:
    bands = len(schema["tensor"]["bands"])
    source_layout = schema["tensor"]["source_layout"]
    if array.ndim == 2:
        if bands != 1:
            raise ValueError(f"2-D tile {path} is only valid for a one-band input contract")
        return array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"tile {path} must be 2-D or 3-D")
    if source_layout == "HWC":
        if array.shape[-1] != bands:
            raise ValueError(
                f"HWC tile {path} has {array.shape[-1]} channels but schema requires {bands}"
            )
        return np.transpose(array, (2, 0, 1))
    if source_layout == "CHW":
        if array.shape[0] != bands:
            raise ValueError(
                f"CHW tile {path} has {array.shape[0]} channels but schema requires {bands}"
            )
        return array
    raise ValueError(f"unsupported source layout: {source_layout}")


def load_tile_numpy(
    path: str | Path,
    *,
    bands: int | None = None,
    size: int | None = None,
    input_schema: dict | None = None,
) -> np.ndarray:
    """Load one tile as model-ready CHW float32 in [0, 1].

    Strict model/runtime use should pass ``input_schema``. Legacy ``bands``/``size``
    calls remain available for simple utilities, with deterministic CHW/HWC ambiguity
    rejection and exact image-channel handling.
    """
    path = Path(path)
    source_format = _source_format_for_path(path)

    if input_schema is not None:
        validate_input_schema(input_schema)
        tensor = input_schema["tensor"]
        expected_bands = len(tensor["bands"])
        height = int(tensor["height"])
        width = int(tensor["width"])
        expected_format = input_schema["source"]["format"]
        if source_format != expected_format:
            raise ValueError(
                f"tile format {source_format} does not match input schema source format {expected_format}"
            )
        if path.suffix.lower() == ".npy":
            raw = np.load(path, allow_pickle=False)
        else:
            raw = _load_exact_image(path, expected_channels=expected_bands)
        normalized = _apply_schema_normalization(np.asarray(raw), input_schema)
        chw = _schema_layout(normalized, schema=input_schema, path=path)
    else:
        if bands is None or size is None or bands <= 0 or size <= 0:
            raise ValueError("bands and size must be positive when input_schema is not supplied")
        expected_bands = int(bands)
        height = width = int(size)
        if path.suffix.lower() == ".npy":
            normalized = _legacy_normalize_npy(np.load(path, allow_pickle=False))
        else:
            raw = _load_exact_image(path, expected_channels=expected_bands)
            if raw.dtype != np.uint8:
                raise ValueError("PNG/JPEG legacy loader accepts only uint8 L/RGB/RGBA data")
            normalized = raw.astype(np.float32) / 255.0
        chw = _legacy_layout(normalized, bands=expected_bands, path=path)

    tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
    if tensor.shape[-2:] != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    result = tensor.squeeze(0).numpy().astype(np.float32, copy=False)
    if result.shape != (expected_bands, height, width):
        raise ValueError(f"unexpected tile shape after preprocessing: {result.shape}")
    return result


class TileFolder(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        bands: int = 3,
        size: int = 64,
        input_schema: dict | None = None,
    ):
        self.root = Path(root)
        self.items = discover_labeled_tiles(self.root)
        if not self.items:
            raise ValueError(f"no supported tiles found under {self.root}")
        self.input_schema = input_schema
        if input_schema is not None:
            validate_input_schema(input_schema)
            assert_dataset_schema_compatible(self.root, input_schema)
            self.bands = len(input_schema["tensor"]["bands"])
            self.height = int(input_schema["tensor"]["height"])
            self.width = int(input_schema["tensor"]["width"])
        else:
            self.bands = int(bands)
            self.height = self.width = int(size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, cls = self.items[idx]
        if self.input_schema is not None:
            array = load_tile_numpy(path, input_schema=self.input_schema)
        else:
            array = load_tile_numpy(path, bands=self.bands, size=self.height)
        x = torch.from_numpy(array)
        y = torch.tensor(cls, dtype=torch.long)
        return x, y


def make_loader(
    path: str | Path,
    *,
    batch: int = 64,
    shuffle: bool = True,
    bands: int = 3,
    size: int = 64,
    seed: int = 0,
    input_schema: dict | None = None,
) -> DataLoader:
    if batch <= 0:
        raise ValueError("batch must be positive")
    dataset = TileFolder(path, bands=bands, size=size, input_schema=input_schema)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def read_dataset_manifest(root: str | Path) -> dict:
    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in (1, 2, 3):
        raise ValueError("unsupported dataset manifest schema")
    if payload.get("schema_version") in (2, 3):
        expected_roles = {"train", "calib", "validation", "test"}
        if set(payload.get("split_counts", {})) != expected_roles:
            raise ValueError("four-way dataset manifest is missing a required split")
        if set(payload.get("split_roles", {})) != expected_roles:
            raise ValueError("four-way dataset manifest is missing split-role metadata")
    if payload.get("schema_version") == 3:
        schema_path = root / "input_schema.json"
        schema = read_input_schema(schema_path)
        expected_hash = payload.get("input_schema_sha256")
        actual_hash = input_schema_sha256(schema)
        if expected_hash != actual_hash:
            raise ValueError("dataset manifest input_schema_sha256 does not match input_schema.json")
    return payload


def class_counts(items: Iterable[tuple[Path, int]]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for _, cls in items:
        counts[CLASS_NAMES[cls]] += 1
    return counts

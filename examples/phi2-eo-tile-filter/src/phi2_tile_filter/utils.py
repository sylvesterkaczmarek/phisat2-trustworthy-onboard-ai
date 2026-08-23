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

CLASS_NAMES = ("background", "event")
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_TILE_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".npy"}


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
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_TILE_SUFFIXES
    )
    return files


def discover_labeled_tiles(root: str | Path) -> list[tuple[Path, int]]:
    root = Path(root)
    items: list[tuple[Path, int]] = []
    for cls, name in enumerate(CLASS_NAMES):
        class_dir = root / name
        for path in discover_tile_files(class_dir):
            items.append((path, cls))
    return items


def _normalize_npy(array: np.ndarray) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("NumPy tiles must contain numeric data")
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        array = array.astype(np.float32) / float(info.max)
    else:
        array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("tile contains non-finite values")
    if array.size == 0:
        raise ValueError("tile is empty")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError("floating-point NumPy tiles must already be scaled to [0, 1]")
    return np.clip(array, 0.0, 1.0)


def load_tile_numpy(path: str | Path, *, bands: int, size: int) -> np.ndarray:
    """Load one tile as CHW float32 in [0, 1].

    PNG/JPEG/TIFF inputs support 1, 3, or 4 bands. Arbitrary multispectral
    band counts use `.npy` arrays in HWC or CHW layout.
    """
    path = Path(path)
    if bands <= 0 or size <= 0:
        raise ValueError("bands and size must be positive")

    if path.suffix.lower() == ".npy":
        array = _normalize_npy(np.load(path, allow_pickle=False))
        if array.ndim == 2:
            if bands != 1:
                raise ValueError(f"2-D tile {path} is only valid for bands=1")
            chw = array[None, :, :]
        elif array.ndim == 3:
            if array.shape[0] == bands and array.shape[-1] != bands:
                chw = array
            elif array.shape[-1] == bands:
                chw = np.transpose(array, (2, 0, 1))
            elif array.shape[0] == bands == array.shape[-1]:
                raise ValueError(f"ambiguous CHW/HWC layout for {path}")
            else:
                raise ValueError(
                    f"tile {path} has shape {array.shape}, which does not match bands={bands}"
                )
        else:
            raise ValueError(f"tile {path} must be 2-D or 3-D")
    else:
        if bands not in (1, 3, 4):
            raise ValueError("image files support 1, 3, or 4 bands; use .npy for multispectral tiles")
        mode = {1: "L", 3: "RGB", 4: "RGBA"}[bands]
        with Image.open(path) as image:
            array = np.asarray(image.convert(mode), dtype=np.float32) / 255.0
        if bands == 1:
            chw = array[None, :, :]
        else:
            chw = np.transpose(array, (2, 0, 1))

    tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    result = tensor.squeeze(0).numpy().astype(np.float32, copy=False)
    if result.shape != (bands, size, size):
        raise ValueError(f"unexpected tile shape after preprocessing: {result.shape}")
    return result


class TileFolder(Dataset):
    def __init__(self, root: str | Path, *, bands: int = 3, size: int = 64):
        self.root = Path(root)
        self.items = discover_labeled_tiles(self.root)
        if not self.items:
            raise ValueError(f"no supported tiles found under {self.root}")
        self.bands = int(bands)
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, cls = self.items[idx]
        x = torch.from_numpy(load_tile_numpy(path, bands=self.bands, size=self.size))
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
) -> DataLoader:
    if batch <= 0:
        raise ValueError("batch must be positive")
    dataset = TileFolder(path, bands=bands, size=size)
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
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in (1, 2):
        raise ValueError("unsupported dataset manifest schema")
    if payload.get("schema_version") == 2:
        expected_roles = {"train", "calib", "validation", "test"}
        if set(payload.get("split_counts", {})) != expected_roles:
            raise ValueError("four-way dataset manifest is missing a required split")
        if set(payload.get("split_roles", {})) != expected_roles:
            raise ValueError("four-way dataset manifest is missing split-role metadata")
    return payload


def class_counts(items: Iterable[tuple[Path, int]]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for _, cls in items:
        counts[CLASS_NAMES[cls]] += 1
    return counts

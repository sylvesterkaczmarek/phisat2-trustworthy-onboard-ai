from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

INPUT_SCHEMA_VERSION = 2
PREPROCESSING_NAME = "phi2_tile_filter.utils.load_tile_numpy"
PREPROCESSING_VERSION = 2
SUPPORTED_SOURCE_LAYOUTS = {"HWC", "CHW"}
SUPPORTED_SOURCE_FORMATS = {"npy", "png", "jpeg"}


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _preprocessing_fingerprint(payload: dict[str, Any]) -> str:
    core = {
        "normalization": payload.get("normalization"),
        "nodata": payload.get("nodata"),
        "preprocessing": payload.get("preprocessing"),
    }
    return hashlib.sha256(_canonical_json_bytes(core)).hexdigest()


def refresh_preprocessing_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    payload["preprocessing_sha256"] = _preprocessing_fingerprint(payload)
    return payload


def input_schema_sha256(payload: dict[str, Any]) -> str:
    validate_input_schema(payload)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def default_band_metadata(count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("band count must be positive")
    return [
        {
            "index": index,
            "id": f"band_{index + 1:02d}",
            "name": f"synthetic_band_{index + 1:02d}",
            "wavelength_nm": None,
        }
        for index in range(count)
    ]


def build_input_schema(
    *,
    bands: int,
    height: int,
    width: int | None = None,
    band_metadata: Iterable[dict[str, Any]] | None = None,
    source_layout: str = "HWC",
    source_format: str = "npy",
    source_dtype: str = "float32",
    value_range: tuple[float, float] = (0.0, 1.0),
    normalization_name: str = "identity_unit_interval",
    normalization_version: int = 1,
    nodata_policy: str = "reject",
) -> dict[str, Any]:
    width = height if width is None else width
    metadata = list(default_band_metadata(bands) if band_metadata is None else deepcopy(list(band_metadata)))
    payload = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "contract_type": "eo-input-preprocessing",
        "tensor": {
            "model_layout": "NCHW",
            "source_layout": source_layout,
            "height": int(height),
            "width": int(width),
            "dtype": "float32",
            "bands": metadata,
        },
        "source": {
            "format": source_format,
            "dtype": source_dtype,
            "value_range": [float(value_range[0]), float(value_range[1])],
        },
        "normalization": {
            "name": normalization_name,
            "version": int(normalization_version),
            "parameters": {},
        },
        "nodata": {
            "policy": nodata_policy,
            "non_finite": "reject",
            "values": [],
        },
        "preprocessing": {
            "name": PREPROCESSING_NAME,
            "version": PREPROCESSING_VERSION,
            "resize": {
                "enabled": True,
                "method": "bilinear",
                "align_corners": False,
            },
            "channel_policy": "exact-no-implicit-conversion",
            "tiff_policy": "reject-use-npy-or-mission-specific-loader",
        },
    }
    refresh_preprocessing_sha256(payload)
    validate_input_schema(payload)
    return payload


def _validate_band_metadata(bands: Any) -> None:
    if not isinstance(bands, list) or not bands:
        raise ValueError("input schema bands must be a non-empty list")
    seen_ids: set[str] = set()
    for index, band in enumerate(bands):
        if not isinstance(band, dict):
            raise ValueError("each input schema band must be an object")
        if band.get("index") != index:
            raise ValueError("input schema band indices must be contiguous and match ordering")
        band_id = band.get("id")
        if not isinstance(band_id, str) or not band_id.strip():
            raise ValueError("each input schema band must have a non-empty id")
        if band_id in seen_ids:
            raise ValueError("input schema band ids must be unique")
        seen_ids.add(band_id)
        name = band.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each input schema band must have a non-empty name")
        wavelength_nm = band.get("wavelength_nm")
        wavelength_range = band.get("wavelength_range_nm")
        if wavelength_nm is not None and wavelength_range is not None:
            raise ValueError("a band may define wavelength_nm or wavelength_range_nm, not both")
        if wavelength_nm is not None and float(wavelength_nm) <= 0.0:
            raise ValueError("band wavelength_nm must be positive")
        if wavelength_range is not None:
            if not isinstance(wavelength_range, list) or len(wavelength_range) != 2:
                raise ValueError("band wavelength_range_nm must contain [min, max]")
            low, high = map(float, wavelength_range)
            if low <= 0.0 or high <= low:
                raise ValueError("band wavelength_range_nm must be positive and increasing")


def validate_input_schema(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("input schema must be a JSON object")
    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported input/preprocessing schema version")
    if payload.get("contract_type") != "eo-input-preprocessing":
        raise ValueError("unsupported input schema contract type")

    tensor = payload.get("tensor")
    source = payload.get("source")
    normalization = payload.get("normalization")
    nodata = payload.get("nodata")
    preprocessing = payload.get("preprocessing")
    if not all(isinstance(item, dict) for item in (tensor, source, normalization, nodata, preprocessing)):
        raise ValueError("input schema is missing required contract sections")

    if tensor.get("model_layout") != "NCHW":
        raise ValueError("model tensor layout must be NCHW")
    if tensor.get("source_layout") not in SUPPORTED_SOURCE_LAYOUTS:
        raise ValueError("source layout must be HWC or CHW")
    try:
        height = int(tensor["height"])
        width = int(tensor["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("input schema tile dimensions are invalid") from exc
    if height <= 0 or width <= 0:
        raise ValueError("input schema tile dimensions must be positive")
    if tensor.get("dtype") != "float32":
        raise ValueError("model tensor dtype must be float32")
    _validate_band_metadata(tensor.get("bands"))

    source_format = source.get("format")
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError("source format must be one of npy, png, or jpeg")
    if not isinstance(source.get("dtype"), str) or not source["dtype"]:
        raise ValueError("input schema source dtype is required")
    value_range = source.get("value_range")
    if not isinstance(value_range, list) or len(value_range) != 2:
        raise ValueError("input schema source value_range must contain [min, max]")
    low, high = map(float, value_range)
    if not low < high:
        raise ValueError("input schema source value_range must be increasing")

    if not isinstance(normalization.get("name"), str) or not normalization["name"]:
        raise ValueError("input schema normalization name is required")
    if not isinstance(normalization.get("version"), int) or normalization["version"] <= 0:
        raise ValueError("input schema normalization version must be a positive integer")
    if not isinstance(normalization.get("parameters", {}), dict):
        raise ValueError("input schema normalization parameters must be an object")

    if nodata.get("policy") not in {"reject", "allow"}:
        raise ValueError("nodata policy must be reject or allow")
    if nodata.get("non_finite") != "reject":
        raise ValueError("this preprocessing implementation requires non-finite values to be rejected")
    if not isinstance(nodata.get("values"), list):
        raise ValueError("nodata values must be a list")

    if preprocessing.get("name") != PREPROCESSING_NAME:
        raise ValueError("input schema preprocessing implementation does not match this runtime")
    if preprocessing.get("version") != PREPROCESSING_VERSION:
        raise ValueError("input schema preprocessing version does not match this runtime")
    resize = preprocessing.get("resize")
    if not isinstance(resize, dict):
        raise ValueError("input schema resize contract is missing")
    if resize.get("enabled") is not True or resize.get("method") != "bilinear" or resize.get("align_corners") is not False:
        raise ValueError("unsupported resize preprocessing contract")
    if preprocessing.get("channel_policy") != "exact-no-implicit-conversion":
        raise ValueError("unsupported channel policy")
    if preprocessing.get("tiff_policy") != "reject-use-npy-or-mission-specific-loader":
        raise ValueError("unsupported TIFF policy")
    expected_preprocessing_hash = _preprocessing_fingerprint(payload)
    if payload.get("preprocessing_sha256") != expected_preprocessing_hash:
        raise ValueError("input schema preprocessing_sha256 does not match normalization/nodata/preprocessing metadata")
    return payload


def read_input_schema(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in input schema {path}")
    validate_input_schema(payload)
    return payload


def write_input_schema(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    refresh_preprocessing_sha256(payload)
    validate_input_schema(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return input_schema_sha256(payload)


def model_schema_sidecar_path(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_suffix(model_path.suffix + ".input_schema.json")


def find_model_input_schema(model_path: str | Path, explicit_path: str | Path | None = None) -> Path:
    model_path = Path(model_path)
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend([model_path.parent / "input_schema.json", model_schema_sidecar_path(model_path)])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no input schema found for {model_path}; expected input_schema.json beside a bundle model "
        f"or {model_schema_sidecar_path(model_path).name}"
    )


def find_dataset_input_schema(data_root: str | Path) -> Path:
    root = Path(data_root)
    candidates = [root / "input_schema.json", root.parent / "input_schema.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"data input schema not found for {root}; expected input_schema.json in the data root or its parent"
    )


def assert_dataset_schema_compatible(data_root: str | Path, expected_schema: dict[str, Any]) -> dict[str, Any]:
    actual_path = find_dataset_input_schema(data_root)
    actual = read_input_schema(actual_path)
    expected_hash = input_schema_sha256(expected_schema)
    actual_hash = input_schema_sha256(actual)
    if actual_hash != expected_hash:
        raise ValueError(
            "dataset input/preprocessing schema does not match the model contract "
            f"(dataset={actual_hash}, model={expected_hash})"
        )
    return actual


def onnx_metadata_map(model_path: str | Path) -> dict[str, str]:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnx is required to inspect model input-schema metadata") from exc
    model = onnx.load(str(model_path), load_external_data=False)
    return {item.key: item.value for item in model.metadata_props}


def model_input_schema_sha256(model_path: str | Path) -> str:
    metadata = onnx_metadata_map(model_path)
    value = metadata.get("input_schema_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("ONNX model is missing a valid input_schema_sha256 metadata property")
    return value


def validate_model_input_schema_binding(
    model_path: str | Path,
    schema_path: str | Path | None = None,
) -> tuple[dict[str, Any], str, Path]:
    resolved = find_model_input_schema(model_path, schema_path)
    schema = read_input_schema(resolved)
    schema_hash = input_schema_sha256(schema)
    model_hash = model_input_schema_sha256(model_path)
    if model_hash != schema_hash:
        raise ValueError(
            "ONNX model input_schema_sha256 does not match its preprocessing contract "
            f"(model={model_hash}, schema={schema_hash})"
        )
    return schema, schema_hash, resolved


def band_ids(schema: dict[str, Any]) -> tuple[str, ...]:
    validate_input_schema(schema)
    return tuple(str(item["id"]) for item in schema["tensor"]["bands"])

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FORMAT_VERSION = 1
STATE_SCHEMA_VERSION = 1
INPUT_SCHEMA_VERSION = 1
REQUIRED_COMPONENTS = ("model", "policy", "input_schema", "validation")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _bundle_id(payload_without_id: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload_without_id)).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        _fsync_dir(path)
    _fsync_dir(root)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _safe_component_file(bundle_dir: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"unsafe bundle component path: {relative!r}")
    candidate = (bundle_dir / rel).resolve()
    root = bundle_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"bundle component escapes bundle directory: {relative!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"bundle component not found: {candidate}")
    return candidate


def _onnx_input_spec(path: Path) -> tuple[int, int]:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - dependency is required by the example package
        raise RuntimeError("onnx is required to validate deployment bundles") from exc

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializer_names]
    if len(inputs) != 1:
        raise ValueError(f"expected one ONNX input, found {len(inputs)}")
    dims = inputs[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        raise ValueError("expected NCHW ONNX input")
    bands = int(dims[1].dim_value)
    height = int(dims[2].dim_value)
    width = int(dims[3].dim_value)
    if bands <= 0 or height <= 0 or width <= 0 or height != width:
        raise ValueError("ONNX model must have static C,H,W dimensions with H == W")
    return bands, height


def _validate_policy(policy: dict[str, Any], *, model_sha256: str, bands: int, size: int) -> None:
    if policy.get("schema_version") != 2:
        raise ValueError("unsupported calibration policy schema")
    if policy.get("model_sha256") != model_sha256:
        raise ValueError("calibration policy belongs to a different model")
    if int(policy.get("bands", -1)) != bands or int(policy.get("size", -1)) != size:
        raise ValueError("calibration policy input shape does not match model")
    for key in ("event_threshold", "min_confidence", "temperature"):
        if key not in policy:
            raise ValueError(f"calibration policy is missing {key}")
        try:
            value = float(policy[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"calibration policy {key} is not numeric") from exc
        if key == "temperature":
            if not value > 0.0:
                raise ValueError("calibration policy temperature must be positive")
        elif not 0.0 <= value <= 1.0:
            raise ValueError(f"calibration policy {key} must be in [0, 1]")


def _validate_validation_report(report: dict[str, Any], *, model_sha256: str) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("unsupported validation report schema")
    if report.get("int8_sha256") != model_sha256:
        raise ValueError("validation report does not cover the candidate model")
    try:
        drop = float(report["accuracy_drop"])
        max_drop = float(report["max_accuracy_drop_allowed"])
        agreement = float(report["argmax_agreement"])
        min_agreement = float(report["min_argmax_agreement_required"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("validation report is missing required acceptance metrics") from exc
    if drop > max_drop:
        raise ValueError("validation report records an unacceptable INT8 accuracy drop")
    if agreement < min_agreement:
        raise ValueError("validation report records unacceptable FP32/INT8 agreement")
    if "accepted" in report and report["accepted"] is not True:
        raise ValueError("validation report is not marked accepted")


def _input_schema_payload(*, model_sha256: str, bands: int, size: int) -> dict[str, Any]:
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "model_sha256": model_sha256,
        "layout": "NCHW",
        "bands": bands,
        "height": size,
        "width": size,
        "dtype": "float32",
        "value_range": [0.0, 1.0],
        "preprocessing": {
            "name": "phi2_tile_filter.utils.load_tile_numpy",
            "version": 1,
            "resize": "bilinear-align_corners-false",
        },
    }


def _validate_input_schema(schema: dict[str, Any], *, model_sha256: str, bands: int, size: int) -> None:
    if schema.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported input schema")
    if schema.get("model_sha256") != model_sha256:
        raise ValueError("input schema belongs to a different model")
    if schema.get("layout") != "NCHW":
        raise ValueError("deployment input schema must use NCHW layout")
    if int(schema.get("bands", -1)) != bands:
        raise ValueError("input schema band count does not match model")
    if int(schema.get("height", -1)) != size or int(schema.get("width", -1)) != size:
        raise ValueError("input schema spatial size does not match model")
    if schema.get("dtype") != "float32":
        raise ValueError("deployment input schema must use float32")
    if schema.get("value_range") != [0.0, 1.0]:
        raise ValueError("deployment input schema must declare [0, 1] values")
    preprocessing = schema.get("preprocessing")
    if not isinstance(preprocessing, dict) or not preprocessing.get("name") or not preprocessing.get("version"):
        raise ValueError("input schema is missing preprocessing metadata")


def _manifest_without_id(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "bundle_id"}


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "bundle.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"bundle manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported deployment bundle manifest schema")
    if manifest.get("bundle_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError("unsupported deployment bundle format version")
    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str) or len(bundle_id) != 64:
        raise ValueError("deployment bundle has an invalid bundle_id")
    calculated_bundle_id = _bundle_id(_manifest_without_id(manifest))
    if calculated_bundle_id != bundle_id:
        raise ValueError("deployment bundle manifest hash does not match bundle_id")

    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != set(REQUIRED_COMPONENTS):
        raise ValueError("deployment bundle must contain model, policy, input_schema, and validation components")

    resolved: dict[str, Path] = {}
    for name in REQUIRED_COMPONENTS:
        item = components[name]
        if not isinstance(item, dict):
            raise ValueError(f"invalid bundle component descriptor: {name}")
        path = _safe_component_file(bundle_dir, str(item.get("path", "")))
        expected_sha = item.get("sha256")
        if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
            raise ValueError(f"bundle component hash mismatch: {name}")
        resolved[name] = path

    model_sha = sha256_file(resolved["model"])
    if manifest.get("model_sha256") != model_sha:
        raise ValueError("bundle model hash does not match manifest")
    bands, size = _onnx_input_spec(resolved["model"])
    policy = _read_json(resolved["policy"])
    schema = _read_json(resolved["input_schema"])
    validation = _read_json(resolved["validation"])
    _validate_policy(policy, model_sha256=model_sha, bands=bands, size=size)
    _validate_input_schema(schema, model_sha256=model_sha, bands=bands, size=size)
    _validate_validation_report(validation, model_sha256=model_sha)

    if manifest.get("policy_sha256") != sha256_file(resolved["policy"]):
        raise ValueError("bundle policy hash does not match manifest")
    if manifest.get("input_schema_sha256") != sha256_file(resolved["input_schema"]):
        raise ValueError("bundle input-schema hash does not match manifest")
    if manifest.get("validation_sha256") != sha256_file(resolved["validation"]):
        raise ValueError("bundle validation-report hash does not match manifest")
    return manifest


def build_bundle(
    model: str | Path,
    policy: str | Path,
    validation: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    model = Path(model)
    policy = Path(policy)
    validation = Path(validation)
    output = Path(output)
    for path in (model, policy, validation):
        if not path.is_file():
            raise FileNotFoundError(path)

    model_sha = sha256_file(model)
    bands, size = _onnx_input_spec(model)
    policy_payload = _read_json(policy)
    validation_payload = _read_json(validation)
    _validate_policy(policy_payload, model_sha256=model_sha, bands=bands, size=size)
    _validate_validation_report(validation_payload, model_sha256=model_sha)
    input_schema = _input_schema_payload(model_sha256=model_sha, bands=bands, size=size)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.stage-{uuid.uuid4().hex}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        shutil.copy2(model, stage / "model.onnx")
        shutil.copy2(policy, stage / "policy.json")
        shutil.copy2(validation, stage / "validation.json")
        (stage / "input_schema.json").write_text(
            json.dumps(input_schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        components = {
            "model": {"path": "model.onnx", "sha256": sha256_file(stage / "model.onnx")},
            "policy": {"path": "policy.json", "sha256": sha256_file(stage / "policy.json")},
            "input_schema": {
                "path": "input_schema.json",
                "sha256": sha256_file(stage / "input_schema.json"),
            },
            "validation": {
                "path": "validation.json",
                "sha256": sha256_file(stage / "validation.json"),
            },
        }
        manifest_without_id = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_version": BUNDLE_FORMAT_VERSION,
            "model_sha256": model_sha,
            "policy_sha256": components["policy"]["sha256"],
            "input_schema_sha256": components["input_schema"]["sha256"],
            "validation_sha256": components["validation"]["sha256"],
            "components": components,
        }
        manifest = dict(manifest_without_id)
        manifest["bundle_id"] = _bundle_id(manifest_without_id)
        (stage / "bundle.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_bundle(stage)
        _fsync_tree(stage)
        if output.exists():
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        os.replace(stage, output)
        _fsync_dir(output.parent)
        return verify_bundle(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _copy_bundle_to_store(candidate_bundle: Path, store: Path, bundle_id: str) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    destination = store / bundle_id
    if destination.exists():
        existing = verify_bundle(destination)
        if existing.get("bundle_id") != bundle_id:
            raise ValueError("existing stored bundle has unexpected identity")
        return destination

    stage = store / f".{bundle_id}.stage-{uuid.uuid4().hex}"
    try:
        shutil.copytree(candidate_bundle, stage)
        copied = verify_bundle(stage)
        if copied.get("bundle_id") != bundle_id:
            raise ValueError("copied bundle identity changed during staging")
        _fsync_tree(stage)
        os.replace(stage, destination)
        _fsync_dir(store)
        return destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _read_state(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    state = _read_json(path)
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("unsupported deployment state schema")
    if not isinstance(state.get("generation"), int) or state["generation"] < 1:
        raise ValueError("deployment state has invalid generation")
    for key in ("active_bundle_id", "previous_bundle_id"):
        value = state.get(key)
        if value is not None and (not isinstance(value, str) or len(value) != 64):
            raise ValueError(f"deployment state has invalid {key}")
    if state.get("active_bundle_id") is None:
        raise ValueError("deployment state has no active bundle")
    return state


def promote_bundle(
    candidate_bundle: str | Path,
    store: str | Path,
    state_path: str | Path,
) -> dict[str, Any]:
    candidate_bundle = Path(candidate_bundle)
    store = Path(store)
    state_path = Path(state_path)
    manifest = verify_bundle(candidate_bundle)
    bundle_id = str(manifest["bundle_id"])
    stored = _copy_bundle_to_store(candidate_bundle, store, bundle_id)
    verify_bundle(stored)

    if state_path.exists():
        old = _read_state(state_path)
        verify_bundle(store / str(old["active_bundle_id"]))
        if old["active_bundle_id"] == bundle_id:
            resolve_bundle(store, state_path, slot="active")
            return old
        previous = old["active_bundle_id"]
        generation = int(old["generation"]) + 1
    else:
        previous = None
        generation = 1

    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "generation": generation,
        "active_bundle_id": bundle_id,
        "previous_bundle_id": previous,
        "updated_unix_s": time.time(),
    }
    _atomic_write_json(state_path, state)
    resolve_bundle(store, state_path, slot="active")
    return state


def rollback(store: str | Path, state_path: str | Path) -> dict[str, Any]:
    store = Path(store)
    state_path = Path(state_path)
    state = _read_state(state_path)
    active = str(state["active_bundle_id"])
    previous = state.get("previous_bundle_id")
    if previous is None:
        raise FileNotFoundError("no previous deployment bundle is available")

    verify_bundle(store / active)
    verify_bundle(store / str(previous))
    next_state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "generation": int(state["generation"]) + 1,
        "active_bundle_id": str(previous),
        "previous_bundle_id": active,
        "updated_unix_s": time.time(),
    }
    _atomic_write_json(state_path, next_state)
    resolve_bundle(store, state_path, slot="active")
    return next_state


def resolve_bundle(
    store: str | Path,
    state_path: str | Path,
    *,
    slot: str = "active",
) -> dict[str, Any]:
    if slot not in {"active", "previous"}:
        raise ValueError("slot must be 'active' or 'previous'")
    store = Path(store)
    state = _read_state(state_path)
    key = f"{slot}_bundle_id"
    bundle_id = state.get(key)
    if bundle_id is None:
        raise FileNotFoundError(f"no {slot} deployment bundle is available")
    bundle_dir = store / str(bundle_id)
    manifest = verify_bundle(bundle_dir)
    components = manifest["components"]
    return {
        "schema_version": 1,
        "slot": slot,
        "bundle_id": bundle_id,
        "bundle_dir": str(bundle_dir),
        "model": str(_safe_component_file(bundle_dir, components["model"]["path"])),
        "policy": str(_safe_component_file(bundle_dir, components["policy"]["path"])),
        "input_schema": str(_safe_component_file(bundle_dir, components["input_schema"]["path"])),
        "validation": str(_safe_component_file(bundle_dir, components["validation"]["path"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, verify, promote, and roll back immutable deployment bundles."
    )
    sub = parser.add_subparsers(dest="action", required=True)

    build_parser = sub.add_parser("build")
    build_parser.add_argument("--model", type=Path, required=True)
    build_parser.add_argument("--policy", type=Path, required=True)
    build_parser.add_argument("--validation", type=Path, required=True)
    build_parser.add_argument("--out", type=Path, required=True)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, required=True)

    promote_parser = sub.add_parser("promote")
    promote_parser.add_argument("--candidate-bundle", type=Path, required=True)
    promote_parser.add_argument("--store", type=Path, required=True)
    promote_parser.add_argument("--state", type=Path, required=True)

    rollback_parser = sub.add_parser("rollback")
    rollback_parser.add_argument("--store", type=Path, required=True)
    rollback_parser.add_argument("--state", type=Path, required=True)

    resolve_parser = sub.add_parser("resolve")
    resolve_parser.add_argument("--store", type=Path, required=True)
    resolve_parser.add_argument("--state", type=Path, required=True)
    resolve_parser.add_argument("--slot", choices=["active", "previous"], default="active")

    args = parser.parse_args()
    if args.action == "build":
        result = build_bundle(args.model, args.policy, args.validation, args.out)
    elif args.action == "verify":
        result = verify_bundle(args.bundle)
    elif args.action == "promote":
        result = promote_bundle(args.candidate_bundle, args.store, args.state)
    elif args.action == "rollback":
        result = rollback(args.store, args.state)
    else:
        result = resolve_bundle(args.store, args.state, slot=args.slot)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

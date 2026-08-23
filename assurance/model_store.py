from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SRC = REPO_ROOT / "examples" / "phi2-eo-tile-filter" / "src"
if str(EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_SRC))

from phi2_tile_filter.input_schema import (  # noqa: E402
    band_ids,
    find_model_input_schema,
    validate_model_input_schema_binding,
)

BUNDLE_SCHEMA_VERSION = 2
BUNDLE_FORMAT_VERSION = 2
STATE_SCHEMA_VERSION = 1
REQUIRED_COMPONENTS = ("model", "policy", "input_schema", "validation")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _bundle_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


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


def _safe_component(bundle_dir: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"unsafe bundle component path: {relative!r}")
    path = (bundle_dir / rel).resolve()
    root = bundle_dir.resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"bundle component not found: {path}")
    return path


def _onnx_spec(path: Path) -> tuple[int, int]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    if len(inputs) != 1:
        raise ValueError("expected exactly one ONNX model input")
    dims = inputs[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        raise ValueError("expected NCHW ONNX input")
    bands, height, width = (int(dims[i].dim_value) for i in (1, 2, 3))
    if bands <= 0 or height <= 0 or width <= 0 or height != width:
        raise ValueError("ONNX model must have static C,H,W dimensions with H == W")
    return bands, height


def _validate_policy(
    policy: dict[str, Any], *, model_sha: str, contract_hash: str, ids: tuple[str, ...], bands: int, size: int, preprocessing_version: int
) -> None:
    if policy.get("schema_version") != 4 or policy.get("split_role") != "calibration":
        raise ValueError("deployment requires calibration policy schema version 4")
    if policy.get("model_sha256") != model_sha:
        raise ValueError("calibration policy belongs to a different model")
    if policy.get("input_schema_sha256") != contract_hash:
        raise ValueError("calibration policy belongs to a different input/preprocessing contract")
    if tuple(policy.get("input_band_ids", [])) != ids:
        raise ValueError("calibration policy band ordering does not match input schema")
    if int(policy.get("preprocessing_version", -1)) != preprocessing_version:
        raise ValueError("calibration policy preprocessing version does not match input schema")
    if int(policy.get("bands", -1)) != bands or int(policy.get("size", -1)) != size:
        raise ValueError("calibration policy input shape does not match model")
    for key in ("event_threshold", "min_confidence", "temperature"):
        value = float(policy[key])
        if key == "temperature" and value <= 0:
            raise ValueError("calibration policy temperature must be positive")
        if key != "temperature" and not 0 <= value <= 1:
            raise ValueError(f"calibration policy {key} must be in [0, 1]")
    stats = policy.get("calibration_statistics")
    acceptance = policy.get("calibration_acceptance")
    if not isinstance(stats, dict) or not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        raise ValueError("calibration policy is not marked accepted")
    lower = float(stats["event_recall_lower_bound"])
    empirical = float(stats["empirical_event_recall"])
    confidence = float(stats["event_recall_confidence_level"])
    if not 0 <= lower <= empirical <= 1 or not 0 < confidence < 1 or int(stats["event_samples"]) <= 0:
        raise ValueError("calibration policy recall-bound metadata is inconsistent")
    if stats.get("event_recall_bound_method") != "clopper-pearson-one-sided-exact":
        raise ValueError("unsupported calibration recall-bound method")
    required = acceptance.get("required_min_event_recall_lower_bound")
    if required is not None and lower < float(required):
        raise ValueError("calibration policy does not meet its recall lower-bound requirement")


def _validate_validation(
    report: dict[str, Any], *, model_sha: str, contract_hash: str, ids: tuple[str, ...], preprocessing_version: int
) -> None:
    if report.get("schema_version") != 3 or report.get("split_role") != "validation":
        raise ValueError("model acceptance report must come from validation report schema version 3")
    if report.get("int8_sha256") != model_sha:
        raise ValueError("validation report does not cover the candidate model")
    if report.get("input_schema_sha256") != contract_hash:
        raise ValueError("validation report covers a different input/preprocessing contract")
    if tuple(report.get("input_band_ids", [])) != ids:
        raise ValueError("validation report band ordering does not match input schema")
    if int(report.get("preprocessing_version", -1)) != preprocessing_version:
        raise ValueError("validation report preprocessing version does not match input schema")
    if report.get("accepted") is not True:
        raise ValueError("validation report is not marked accepted")
    try:
        classification = report["classification_metrics"]["quantization_regression"]
        policy = report["policy_metrics"]["quantization_regression"]
        drift = report["score_drift_metrics"]
        criteria = report["acceptance_criteria"]
        expected = {
            "classification_accuracy_drop": float(classification["accuracy_drop"]) <= float(criteria["max_classification_accuracy_drop"]),
            "classification_argmax_agreement": float(classification["argmax_agreement"]) >= float(criteria["min_classification_argmax_agreement"]),
            "classification_event_recall_drop": float(classification["event_recall_drop"]) <= float(criteria["max_classification_event_recall_drop"]),
            "classification_event_false_negative_rate_increase": float(classification["event_false_negative_rate_increase"]) <= float(criteria["max_classification_event_false_negative_rate_increase"]),
            "classification_pr_auc_drop": float(classification["pr_auc_drop"]) <= float(criteria["max_classification_pr_auc_drop"]),
            "policy_retention_decision_agreement": float(policy["retention_decision_agreement"]) >= float(criteria["min_policy_retention_decision_agreement"]),
            "policy_event_retention_recall_drop": float(policy["event_retention_recall_drop"]) <= float(criteria["max_policy_event_retention_recall_drop"]),
            "event_score_drift": float(drift["max_absolute_event_score_drift"]) <= float(criteria["max_event_score_drift"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("validation report is missing required scientific acceptance metrics") from exc
    if report.get("acceptance_checks") != expected:
        raise ValueError("validation report acceptance checks do not match its recorded metrics")
    if not all(expected.values()):
        raise ValueError("validation report records failed scientific acceptance criteria")


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    manifest = _read_json(bundle_dir / "bundle.json")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION or manifest.get("bundle_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError("unsupported deployment bundle schema or format")
    expected_id = _bundle_id({k: v for k, v in manifest.items() if k != "bundle_id"})
    if manifest.get("bundle_id") != expected_id:
        raise ValueError("deployment bundle manifest hash does not match bundle_id")
    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != set(REQUIRED_COMPONENTS):
        raise ValueError("deployment bundle has incomplete components")
    resolved: dict[str, Path] = {}
    for name in REQUIRED_COMPONENTS:
        item = components[name]
        path = _safe_component(bundle_dir, str(item.get("path", "")))
        if sha256_file(path) != item.get("sha256"):
            raise ValueError(f"bundle component hash mismatch: {name}")
        resolved[name] = path
    model_sha = sha256_file(resolved["model"])
    if manifest.get("model_sha256") != model_sha:
        raise ValueError("bundle model hash does not match manifest")
    bands, size = _onnx_spec(resolved["model"])
    schema, contract_hash, _ = validate_model_input_schema_binding(resolved["model"], resolved["input_schema"])
    if manifest.get("input_contract_sha256") != contract_hash:
        raise ValueError("bundle input contract hash does not match input_schema.json")
    if manifest.get("input_schema_file_sha256") != sha256_file(resolved["input_schema"]):
        raise ValueError("bundle input schema file hash does not match manifest")
    ids = band_ids(schema)
    preprocessing_version = int(schema["preprocessing"]["version"])
    _validate_policy(_read_json(resolved["policy"]), model_sha=model_sha, contract_hash=contract_hash, ids=ids, bands=bands, size=size, preprocessing_version=preprocessing_version)
    _validate_validation(_read_json(resolved["validation"]), model_sha=model_sha, contract_hash=contract_hash, ids=ids, preprocessing_version=preprocessing_version)
    if manifest.get("policy_sha256") != sha256_file(resolved["policy"]) or manifest.get("validation_sha256") != sha256_file(resolved["validation"]):
        raise ValueError("bundle policy or validation hash does not match manifest")
    return manifest


def build_bundle(
    model: str | Path, policy: str | Path, validation: str | Path, output: str | Path, *, input_schema: str | Path | None = None
) -> dict[str, Any]:
    model, policy, validation, output = map(Path, (model, policy, validation, output))
    for path in (model, policy, validation):
        if not path.is_file():
            raise FileNotFoundError(path)
    schema_path = find_model_input_schema(model, input_schema)
    bands, size = _onnx_spec(model)
    schema, contract_hash, _ = validate_model_input_schema_binding(model, schema_path)
    model_sha = sha256_file(model)
    ids = band_ids(schema)
    preprocessing_version = int(schema["preprocessing"]["version"])
    _validate_policy(_read_json(policy), model_sha=model_sha, contract_hash=contract_hash, ids=ids, bands=bands, size=size, preprocessing_version=preprocessing_version)
    _validate_validation(_read_json(validation), model_sha=model_sha, contract_hash=contract_hash, ids=ids, preprocessing_version=preprocessing_version)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True)
    try:
        for source, name in ((model, "model.onnx"), (policy, "policy.json"), (schema_path, "input_schema.json"), (validation, "validation.json")):
            shutil.copy2(source, stage / name)
        components = {
            "model": {"path": "model.onnx", "sha256": sha256_file(stage / "model.onnx")},
            "policy": {"path": "policy.json", "sha256": sha256_file(stage / "policy.json")},
            "input_schema": {"path": "input_schema.json", "sha256": sha256_file(stage / "input_schema.json")},
            "validation": {"path": "validation.json", "sha256": sha256_file(stage / "validation.json")},
        }
        core = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_version": BUNDLE_FORMAT_VERSION,
            "model_sha256": model_sha,
            "policy_sha256": components["policy"]["sha256"],
            "input_schema_file_sha256": components["input_schema"]["sha256"],
            "input_contract_sha256": contract_hash,
            "validation_sha256": components["validation"]["sha256"],
            "components": components,
        }
        manifest = {**core, "bundle_id": _bundle_id(core)}
        (stage / "bundle.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify_bundle(stage)
        _fsync_tree(stage)
        if output.exists():
            shutil.rmtree(output) if output.is_dir() else output.unlink()
        os.replace(stage, output)
        _fsync_dir(output.parent)
        return verify_bundle(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _copy_bundle(candidate: Path, store: Path, bundle_id: str) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    destination = store / bundle_id
    if destination.exists():
        verify_bundle(destination)
        return destination
    stage = store / f".{bundle_id}.stage-{uuid.uuid4().hex}"
    shutil.copytree(candidate, stage)
    try:
        if verify_bundle(stage).get("bundle_id") != bundle_id:
            raise ValueError("copied bundle identity changed during staging")
        _fsync_tree(stage)
        os.replace(stage, destination)
        _fsync_dir(store)
        return destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _read_state(path: Path) -> dict[str, Any]:
    state = _read_json(path)
    if state.get("schema_version") != STATE_SCHEMA_VERSION or not isinstance(state.get("generation"), int) or state["generation"] < 1:
        raise ValueError("unsupported deployment state schema")
    for key in ("active_bundle_id", "previous_bundle_id"):
        value = state.get(key)
        if value is not None and (not isinstance(value, str) or len(value) != 64):
            raise ValueError(f"deployment state has invalid {key}")
    if not isinstance(state.get("active_bundle_id"), str):
        raise ValueError("deployment state has no active bundle")
    return state


def _atomic_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def promote_bundle(candidate_bundle: str | Path, store: str | Path, state_path: str | Path) -> dict[str, Any]:
    candidate, store, state_path = Path(candidate_bundle), Path(store), Path(state_path)
    manifest = verify_bundle(candidate)
    bundle_id = str(manifest["bundle_id"])
    verify_bundle(_copy_bundle(candidate, store, bundle_id))
    if state_path.exists():
        old = _read_state(state_path)
        if old["active_bundle_id"] == bundle_id:
            return old
        previous, generation = old["active_bundle_id"], int(old["generation"]) + 1
    else:
        previous, generation = None, 1
    state = {"schema_version": 1, "generation": generation, "active_bundle_id": bundle_id, "previous_bundle_id": previous, "updated_unix_s": time.time()}
    _atomic_state(state_path, state)
    resolve_bundle(store, state_path)
    return state


def rollback(store: str | Path, state_path: str | Path) -> dict[str, Any]:
    store, state_path = Path(store), Path(state_path)
    state = _read_state(state_path)
    previous = state.get("previous_bundle_id")
    if not isinstance(previous, str):
        raise FileNotFoundError("no previous deployment bundle is available")
    active = state["active_bundle_id"]
    verify_bundle(store / active)
    verify_bundle(store / previous)
    next_state = {"schema_version": 1, "generation": int(state["generation"]) + 1, "active_bundle_id": previous, "previous_bundle_id": active, "updated_unix_s": time.time()}
    _atomic_state(state_path, next_state)
    resolve_bundle(store, state_path, slot="active")
    return next_state


def resolve_bundle(store: str | Path, state_path: str | Path, *, slot: str = "active") -> dict[str, Any]:
    if slot not in {"active", "previous"}:
        raise ValueError("slot must be 'active' or 'previous'")
    store, state_path = Path(store), Path(state_path)
    state = _read_state(state_path)
    bundle_id = state.get(f"{slot}_bundle_id")
    if not isinstance(bundle_id, str):
        raise FileNotFoundError(f"no {slot} deployment bundle is available")
    bundle_dir = store / bundle_id
    manifest = verify_bundle(bundle_dir)
    components = manifest["components"]
    return {
        "schema_version": 2,
        "slot": slot,
        "bundle_id": bundle_id,
        "bundle_dir": str(bundle_dir),
        "model": str(_safe_component(bundle_dir, components["model"]["path"])),
        "policy": str(_safe_component(bundle_dir, components["policy"]["path"])),
        "input_schema": str(_safe_component(bundle_dir, components["input_schema"]["path"])),
        "input_contract_sha256": manifest["input_contract_sha256"],
        "validation": str(_safe_component(bundle_dir, components["validation"]["path"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, verify, promote, and roll back immutable deployment bundles.")
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--model", type=Path, required=True)
    build.add_argument("--policy", type=Path, required=True)
    build.add_argument("--validation", type=Path, required=True)
    build.add_argument("--input-schema", type=Path, default=None)
    build.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify"); verify.add_argument("--bundle", type=Path, required=True)
    promote = sub.add_parser("promote"); promote.add_argument("--candidate-bundle", type=Path, required=True); promote.add_argument("--store", type=Path, required=True); promote.add_argument("--state", type=Path, required=True)
    roll = sub.add_parser("rollback"); roll.add_argument("--store", type=Path, required=True); roll.add_argument("--state", type=Path, required=True)
    resolve = sub.add_parser("resolve"); resolve.add_argument("--store", type=Path, required=True); resolve.add_argument("--state", type=Path, required=True); resolve.add_argument("--slot", choices=["active", "previous"], default="active")
    args = parser.parse_args()
    if args.action == "build": result = build_bundle(args.model, args.policy, args.validation, args.out, input_schema=args.input_schema)
    elif args.action == "verify": result = verify_bundle(args.bundle)
    elif args.action == "promote": result = promote_bundle(args.candidate_bundle, args.store, args.state)
    elif args.action == "rollback": result = rollback(args.store, args.state)
    else: result = resolve_bundle(args.store, args.state, slot=args.slot)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

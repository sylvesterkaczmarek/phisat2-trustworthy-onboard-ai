from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .filesystem import staged_text_file
from .utils import sha256_file

PROVENANCE_SCHEMA_VERSION = 1
REFERENCE_REQUIREMENTS = Path("examples/phi2-eo-tile-filter/requirements-reference.txt")
TRACKED_DISTRIBUTIONS = (
    "numpy",
    "torch",
    "onnx",
    "onnxscript",
    "onnxruntime",
    "Pillow",
    "scikit-learn",
    "scipy",
    "psutil",
    "pytest",
    "ruff",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = _git_output(repo_root, "rev-parse", "HEAD")
    branch = _git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output(repo_root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "commit_sha": commit or None,
        "branch": branch or None,
        "dirty": None if status is None else bool(status),
    }


def _installed_distributions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def _tracked_versions(installed: dict[str, str]) -> dict[str, str | None]:
    return {name: installed.get(name.lower()) for name in TRACKED_DISTRIBUTIONS}


def _cpu_model() -> str | None:
    processor = platform.processor().strip()
    if processor:
        return processor
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return value
        except OSError:
            pass
    return None


def _gpu_metadata() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"cuda_available": None, "cuda_runtime": None, "devices": []}

    available = bool(torch.cuda.is_available())
    devices: list[str] = []
    if available:
        try:
            devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        except Exception:
            devices = []
    return {
        "cuda_available": available,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "devices": devices,
    }


def _normalise_run_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    if not parameters:
        return {}
    normalised: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, Path):
            normalised[key] = str(value)
        elif isinstance(value, tuple):
            normalised[key] = list(value)
        else:
            normalised[key] = value
    return normalised


def collect_run_environment(
    repo_root: str | Path,
    *,
    seed: int,
    selected_execution_provider: str | None,
    run_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata without recording environment variables or secrets."""
    root = Path(repo_root).resolve(strict=False)
    installed = _installed_distributions()
    package_versions = _tracked_versions(installed)
    reference_path = root / REFERENCE_REQUIREMENTS
    reference_hash = sha256_file(reference_path) if reference_path.is_file() else None

    software_fingerprint_payload = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "installed_distributions": installed,
    }
    hardware = {
        "machine": platform.machine() or None,
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "gpu": _gpu_metadata(),
    }
    platform_info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    environment_fingerprint_payload = {
        **software_fingerprint_payload,
        "platform": platform_info,
        "hardware": hardware,
        "selected_execution_provider": selected_execution_provider,
    }

    try:
        import onnxruntime as ort

        available_providers = list(ort.get_available_providers())
    except Exception:
        available_providers = []

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "git": _git_metadata(root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform_info,
        "hardware": hardware,
        "onnxruntime": {
            "selected_execution_provider": selected_execution_provider,
            "available_execution_providers": available_providers,
        },
        "tracked_package_versions": package_versions,
        "installed_distributions": installed,
        "reference_environment": {
            "path": str(REFERENCE_REQUIREMENTS),
            "sha256": reference_hash,
        },
        "dependency_fingerprint_sha256": _sha256_json(software_fingerprint_payload),
        "environment_fingerprint_sha256": _sha256_json(environment_fingerprint_payload),
        "run_parameters": _normalise_run_parameters(run_parameters),
    }


def write_run_environment(path: str | Path, payload: dict[str, Any]) -> None:
    with staged_text_file(path) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")

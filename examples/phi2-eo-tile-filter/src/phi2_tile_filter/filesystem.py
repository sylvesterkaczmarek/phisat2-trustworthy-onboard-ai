from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, TextIO


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_same_or_ancestor(candidate: Path, other: Path) -> bool:
    return candidate == other or candidate in other.parents


def assert_paths_disjoint(
    first: str | Path,
    second: str | Path,
    *,
    description: str = "paths",
) -> tuple[Path, Path]:
    """Reject equal, ancestor, or descendant path relationships."""
    a = _resolved(first)
    b = _resolved(second)
    if (
        a == b
        or a in b.parents
        or b in a.parents
        or (a.exists() and b.exists() and a.samefile(b))
    ):
        raise ValueError(f"unsafe overlapping {description}: {a} and {b}")
    return a, b


def assert_file_outputs_disjoint(
    outputs: Iterable[str | Path], *, protected_paths: Iterable[str | Path] = ()
) -> tuple[Path, ...]:
    """Validate output files against one another and their protected inputs."""
    resolved = tuple(_resolved(path) for path in outputs)
    protected = tuple(_resolved(path) for path in protected_paths)
    for index, path in enumerate(resolved):
        if path.exists() and not path.is_file():
            raise ValueError(f"output must be a file: {path}")
        for other in (*resolved[:index], *protected):
            assert_paths_disjoint(path, other, description="output/input paths")
    return resolved


def assert_safe_tree_target(
    target: str | Path,
    *,
    protected_paths: Iterable[str | Path] = (),
    operation: str = "tree replacement",
) -> Path:
    """Validate a directory target before any recursive removal/replacement."""
    resolved = _resolved(target)
    filesystem_root = Path(resolved.anchor).resolve(strict=False)
    home = Path.home().resolve(strict=False)
    cwd = Path.cwd().resolve(strict=False)
    temp_root = Path(tempfile.gettempdir()).resolve(strict=False)

    unsafe_exact = {filesystem_root, home, cwd, temp_root}
    if resolved in unsafe_exact:
        raise ValueError(f"unsafe {operation} target: {resolved}")

    if _is_same_or_ancestor(resolved, cwd) or _is_same_or_ancestor(resolved, home):
        raise ValueError(f"unsafe {operation} target contains a protected working path: {resolved}")

    for protected in protected_paths:
        protected_resolved = _resolved(protected)
        if (
            resolved == protected_resolved
            or resolved in protected_resolved.parents
            or protected_resolved in resolved.parents
        ):
            raise ValueError(
                f"unsafe {operation}: output {resolved} overlaps protected input {protected_resolved}"
            )
    return resolved


def assert_safe_workspace_root(
    root: str | Path,
    *,
    operation: str = "workspace output",
) -> Path:
    """Reject workspace roots whose fixed child outputs could hit shared system locations.

    Unlike ``assert_safe_tree_target``, the current working directory is allowed
    because the demo intentionally defaults to writing ignored child directories
    beneath the example directory.
    """
    resolved = _resolved(root)
    filesystem_root = Path(resolved.anchor).resolve(strict=False)
    home = Path.home().resolve(strict=False)
    temp_root = Path(tempfile.gettempdir()).resolve(strict=False)
    if resolved in {filesystem_root, home, temp_root}:
        raise ValueError(f"unsafe {operation} root: {resolved}")
    return resolved


def sibling_stage_path(destination: str | Path, *, label: str = "stage") -> Path:
    destination = _resolved(destination)
    return destination.parent / f".{destination.name}.{label}-{uuid.uuid4().hex}"


def replace_tree_from_stage(stage: str | Path, destination: str | Path) -> None:
    """Replace a directory using sibling renames, restoring the old tree on failure."""
    stage = _resolved(stage)
    destination = _resolved(destination)
    if stage.parent != destination.parent:
        raise ValueError("staged tree must be a sibling of its destination")
    if not stage.is_dir():
        raise FileNotFoundError(f"staged tree does not exist: {stage}")

    backup = sibling_stage_path(destination, label="backup")
    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except Exception:
        if had_destination and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()


def replace_outputs_from_stages(
    outputs: Iterable[tuple[str | Path, str | Path]],
) -> None:
    """Publish related outputs, restoring all previous outputs if a rename fails.

    Each stage must be a sibling of its destination. This handles ordinary
    publication failures; several filesystem renames are not a crash-atomic
    transaction and readers should wait for the command to finish.
    """
    pairs = [(_resolved(stage), _resolved(destination)) for stage, destination in outputs]
    for stage, destination in pairs:
        if stage.parent != destination.parent:
            raise ValueError("staged output must be a sibling of its destination")
        if not stage.exists():
            raise FileNotFoundError(f"staged output does not exist: {stage}")
    all_paths = [path for pair in pairs for path in pair]
    for index, path in enumerate(all_paths):
        for other in all_paths[index + 1 :]:
            assert_paths_disjoint(path, other, description="staged publication paths")

    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for _, destination in pairs:
            if destination.exists():
                backup = sibling_stage_path(destination, label="backup")
                os.replace(destination, backup)
                backups.append((backup, destination))
        for stage, destination in pairs:
            os.replace(stage, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            remove_stage(destination)
        for backup, destination in reversed(backups):
            os.replace(backup, destination)
        raise
    else:
        for backup, _ in backups:
            remove_stage(backup)


@contextmanager
def staged_text_file(path: str | Path) -> Iterator[TextIO]:
    """Write a text file completely before atomically replacing its destination."""
    destination = _resolved(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = sibling_stage_path(destination, label="tmp")
    handle: TextIO | None = None
    try:
        handle = stage.open("w", encoding="utf-8")
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(stage, destination)
    finally:
        if handle is not None:
            handle.close()
        if stage.exists():
            stage.unlink()


def remove_stage(path: str | Path) -> None:
    stage = Path(path)
    if not stage.exists():
        return
    if stage.is_dir():
        shutil.rmtree(stage)
    else:
        stage.unlink()

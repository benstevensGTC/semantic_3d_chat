"""Deterministic first-party source closure for V96 embodied inference.

The held-out child must authenticate every local Python module that can affect
its runtime without opening the scorer, oracle, QA, or held-out metadata.  The
closure is derived statically from Python imports, includes package initializers,
and adds the three Blender-side renderer sources that execute out of process.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Final

from semantic_3d_chat.config import PROJECT_ROOT

_SOURCE_ROOT: Final[Path] = PROJECT_ROOT / "src"
_ENTRY_MODULE: Final[str] = "semantic_3d_chat.robot.v96_embodied_task_runner"
_EXTERNAL_RUNTIME_SOURCES: Final[tuple[str, ...]] = (
    "blender/render_runtime_observation.py",
    "blender/runtime_scene_contract.py",
    "blender/scene_utils.py",
)
_FORBIDDEN_RUNTIME_SOURCE: Final[str] = (
    "src/semantic_3d_chat/evaluation/v96_embodied_heldout.py"
)


def _module_sources(module: str) -> tuple[Path, ...]:
    base = _SOURCE_ROOT.joinpath(*module.split("."))
    paths: list[Path] = []
    module_file = base.with_suffix(".py")
    package_file = base / "__init__.py"
    if module_file.is_file():
        paths.append(module_file)
    if package_file.is_file():
        paths.append(package_file)
    return tuple(paths)


def _imported_modules(module: str, path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = (
        module.split(".")
        if path.name == "__init__.py"
        else module.split(".")[:-1]
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - node.level + 1
                if keep < 0:
                    raise ValueError(f"Invalid relative import in V96 runtime: {path}")
                parts = package[:keep]
                if node.module:
                    parts.extend(node.module.split("."))
                base = ".".join(parts)
            else:
                base = node.module or ""
            if base:
                candidates.append(base)
                candidates.extend(
                    f"{base}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
        imported.update(
            candidate
            for candidate in candidates
            if candidate.startswith("semantic_3d_chat")
            and _module_sources(candidate)
        )
    return imported


def _package_initializers(path: Path) -> set[Path]:
    result: set[Path] = set()
    parent = path.parent
    while parent == _SOURCE_ROOT or _SOURCE_ROOT in parent.parents:
        initializer = parent / "__init__.py"
        if initializer.is_file():
            result.add(initializer)
        if parent == _SOURCE_ROOT:
            break
        parent = parent.parent
    return result


@lru_cache(maxsize=1)
def runtime_source_paths() -> tuple[str, ...]:
    """Return the exact scorer-free first-party source closure."""

    pending = [_ENTRY_MODULE]
    visited: set[str] = set()
    paths: set[Path] = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        sources = _module_sources(module)
        if not sources:
            raise FileNotFoundError(f"V96 runtime module has no local source: {module}")
        for source in sources:
            paths.add(source)
            paths.update(_package_initializers(source))
            pending.extend(sorted(_imported_modules(module, source)))
    paths.update(PROJECT_ROOT / relative for relative in _EXTERNAL_RUNTIME_SOURCES)
    relative_paths: list[str] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        relative_paths.append(path.relative_to(PROJECT_ROOT).as_posix())
    result = tuple(sorted(relative_paths))
    if (
        _FORBIDDEN_RUNTIME_SOURCE in result
        or len(result) != len(set(result))
        or not set(_EXTERNAL_RUNTIME_SOURCES).issubset(result)
        or "src/semantic_3d_chat/robot/v96_runtime_source_contract.py" not in result
    ):
        raise RuntimeError("V96 first-party runtime source closure is invalid")
    return result


__all__ = ["runtime_source_paths"]

"""Offline export of a strict, static, metadata-free Blender runtime asset.

Invoke Blender with ``--disable-autoexec`` before the source ``.blend`` path.
The source may be generation-side oracle data; only the authenticated v2
artifact and its exact attestation are permitted at embodied runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_scene_contract import (
    RUNTIME_SCENE_SCHEMA,
    audit_runtime_scene,
    file_sha256,
    sanitize_source_scene,
)
from scene_utils import blender_cli_args, validate_scene_id

_OPAQUE_ASSET = re.compile(r"[a-z]_[0-9]{6}\.blend")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def _safe_destination(path: Path) -> Path:
    destination = Path(os.path.abspath(path.expanduser()))
    if {"oracle", "qa"} & {part.casefold() for part in destination.parts}:
        raise ValueError("Runtime scene asset cannot be written under oracle or QA")
    if _OPAQUE_ASSET.fullmatch(destination.name) is None:
        raise ValueError("Runtime scene asset filename must be opaque")
    current = Path(destination.anchor)
    for component in destination.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Runtime scene asset cannot use a symbolic-link path")
    return destination


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parser().parse_args(blender_cli_args())
    if bpy.context.preferences.filepaths.use_scripts_auto_execute:
        raise RuntimeError("Runtime export requires Blender --disable-autoexec")
    scene_id = validate_scene_id(args.scene)
    destination = _safe_destination(args.output)
    summary = sanitize_source_scene()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.blend")
    try:
        bpy.ops.wm.save_as_mainfile(
            filepath=str(temporary),
            check_existing=False,
            compress=True,
        )
        # Reopen the serialized artifact under the process-wide disabled-
        # autoexec policy, then repeat the complete nested audit. This catches
        # anything Blender added or failed to purge during serialization.
        bpy.ops.wm.open_mainfile(filepath=str(temporary), load_ui=False)
        serialized_summary = audit_runtime_scene(audit_names=True)
        if serialized_summary != summary:
            raise RuntimeError("Serialized runtime asset differs from its strict audit summary")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    manifest: dict[str, object] = {
        "schema": RUNTIME_SCENE_SCHEMA,
        "scene_id": scene_id,
        "asset_file": destination.name,
        "asset_sha256": file_sha256(destination),
        **summary,
    }
    _atomic_json(destination.with_suffix(".json"), manifest)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

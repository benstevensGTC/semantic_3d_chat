#!/usr/bin/env python3
"""Create an immutable exact-scene subset of a validated prefix cache.

This is a preparation utility, not a trainer.  It lets a pair-disjoint
experiment physically separate its training scene prefixes from a larger
previously built cache so the trainer cannot enumerate or open held-out scene
prefixes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from semantic_3d_chat.training.train_question_control_v56 import load_prefix_cache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_subset(
    source: Path,
    destination: Path,
    scene_ids: tuple[str, ...],
    *,
    expected_source_manifest_sha256: str,
) -> dict[str, object]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Destination prefix cache already exists: {destination}")
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"Source prefix cache is unavailable: {source}")
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Scene IDs must be nonempty and unique")
    manifest_path = source / "manifest.json"
    if _sha256(manifest_path) != expected_source_manifest_sha256:
        raise ValueError("Source prefix-cache manifest differs from the declared digest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("scenes")
    if not isinstance(entries, dict) or not set(scene_ids).issubset(entries):
        raise ValueError("Requested scene inventory is absent from the source cache")
    subset_manifest = {
        **{key: value for key, value in manifest.items() if key not in {"scene_count", "scenes"}},
        "scene_count": len(scene_ids),
        "scenes": {scene_id: entries[scene_id] for scene_id in scene_ids},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    )
    try:
        for scene_id in scene_ids:
            entry = entries[scene_id]
            filename = entry.get("filename")
            if filename != f"{scene_id}.safetensors":
                raise ValueError(f"Opaque prefix filename changed for {scene_id}")
            source_file = source / filename
            if (
                not source_file.is_file()
                or source_file.is_symlink()
                or source_file.stat().st_size != entry.get("file_size_bytes")
                or _sha256(source_file) != entry.get("file_sha256")
            ):
                raise ValueError(f"Source prefix bytes changed for {scene_id}")
            shutil.copyfile(source_file, temporary / filename)
        with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(subset_manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        prefixes, validated = load_prefix_cache(
            temporary,
            scene_ids=scene_ids,
            base_checkpoint_sha256=str(manifest["base_checkpoint_sha256"]),
            base_runtime_config_sha256=str(manifest["base_runtime_config_sha256"]),
        )
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "source_manifest_sha256": expected_source_manifest_sha256,
        "destination_manifest_sha256": _sha256(destination / "manifest.json"),
        "scene_count": len(prefixes),
        "scene_ids": list(scene_ids),
        "question_inputs_used": validated["question_inputs_used"],
        "question_dependent_scene_retrieval": validated[
            "question_dependent_scene_retrieval"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    args = parser.parse_args()
    result = materialize_subset(
        args.source,
        args.destination,
        tuple(args.scene_id),
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

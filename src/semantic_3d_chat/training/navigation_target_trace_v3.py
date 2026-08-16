"""Training-only numeric target augmentation for navigation-policy V3.

The source V2 expert traces already contain collision-checked bounded actions.
This module adds only oracle-derived numeric target coordinates under the blocked
``training`` tree.  It never copies category names, instance IDs, captions, or
relationships into the deployable checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot.navigation_policy_v3 import (
    target_text_from_navigation_instruction,
)
from semantic_3d_chat.training.navigation_trace_generator import (
    load_navigation_trace_dataset,
)

TRACE_SCHEMA: Final[str] = "semantic_3d_chat.navigation_target_trace_sample.v3"
MANIFEST_SCHEMA: Final[str] = "semantic_3d_chat.navigation_target_trace_dataset.v3"
TARGETED_FAMILIES: Final[frozenset[str]] = frozenset(
    {"face", "approach", "obstacle", "left_right", "update_after_scan"}
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_training_path(path: Path) -> None:
    if "training" not in {part.casefold() for part in path.parts}:
        raise ValueError("Oracle-derived V3 targets must remain under a training tree")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("V3 target traces cannot use symbolic-link paths")


def _load_oracle_centers(path: Path, scene_id: str) -> dict[str, list[np.ndarray]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("scene_id") != scene_id:
        raise ValueError("V3 target oracle scene identity differs")
    instances = value.get("instances")
    if not isinstance(instances, list):
        raise TypeError("V3 target oracle has no instance inventory")
    centers: dict[str, list[np.ndarray]] = {}
    for instance in instances:
        if not isinstance(instance, Mapping):
            continue
        category = instance.get("category")
        center = np.asarray(instance.get("expected_center_xyz_m"), dtype=np.float64)
        if (
            isinstance(category, str)
            and category.strip()
            and center.shape == (3,)
            and np.isfinite(center).all()
        ):
            key = " ".join(category.casefold().split())
            centers.setdefault(key, []).append(center)
    return centers


def _target_for_row(
    row: Mapping[str, Any], centers: Mapping[str, list[np.ndarray]]
) -> tuple[bool, list[float], str | None]:
    instruction = row.get("instruction")
    family = row.get("family")
    if not isinstance(instruction, str) or not isinstance(family, str):
        raise TypeError("V3 source trace lacks instruction or family")
    target_text = target_text_from_navigation_instruction(instruction)
    expected = family in TARGETED_FAMILIES
    if expected != (target_text is not None):
        raise ValueError(f"V3 target parser/family contract differs for {family}")
    if target_text is None:
        return False, [0.0, 0.0, 0.0], None
    key = " ".join(target_text.casefold().split())
    candidates = centers.get(key)
    if candidates is None:
        raise ValueError("V3 parsed user target is unavailable in training oracle")
    # Match the immutable V2 expert generator, which deterministically uses the
    # first generated instance when a scene happens to contain a second member
    # of the same category.
    center = candidates[0]
    return (
        True,
        [float(value) for value in center],
        hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
    )


def generate_navigation_target_trace_v3(
    config: dict[str, Any], destination: str | Path
) -> dict[str, Any]:
    """Augment immutable V2 expert rows with numeric training-only targets."""

    settings = config.get("navigation_policy_v3")
    if not isinstance(settings, dict):
        raise TypeError("Config has no navigation_policy_v3 mapping")
    root = _rooted(destination)
    _require_training_path(root)
    if root.exists():
        raise FileExistsError(f"V3 target trace dataset already exists: {root}")
    source_root = _rooted(str(settings["source_trace_dataset"]))
    _require_training_path(source_root)
    source_manifest, source_rows = load_navigation_trace_dataset(source_root)
    train_scenes = list(settings["train_scene_ids"])
    validation_scenes = list(settings["validation_scene_ids"])
    if (
        source_manifest.get("train_scene_ids") != train_scenes
        or source_manifest.get("validation_scene_ids") != validation_scenes
        or set(train_scenes) & set(validation_scenes)
    ):
        raise ValueError("V3 source traces do not match the declared disjoint splits")

    oracle_root = _rooted(str(settings["oracle_root"]))
    all_scenes = [*train_scenes, *validation_scenes]
    centers_by_scene: dict[str, dict[str, list[np.ndarray]]] = {}
    oracle_hashes: dict[str, str] = {}
    for scene_id in all_scenes:
        oracle_path = oracle_root / scene_id / "oracle.json"
        if not oracle_path.is_file() or oracle_path.is_symlink():
            raise FileNotFoundError(f"V3 training oracle is unavailable for {scene_id}")
        centers_by_scene[scene_id] = _load_oracle_centers(oracle_path, scene_id)
        oracle_hashes[scene_id] = _sha256(oracle_path)

    target_counts: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        scene_id = str(source["scene_id"])
        available, target_xyz, query_sha256 = _target_for_row(source, centers_by_scene[scene_id])
        family = str(source["family"])
        if available:
            target_counts[family] += 1
        output_rows.append(
            {
                **source,
                "schema": TRACE_SCHEMA,
                "sample_id": f"g_{index:08d}",
                "target_state_available": available,
                "oracle_target_xyz_m": target_xyz,
                "target_query_sha256": query_sha256,
                "target_coordinates_training_only": True,
                "oracle_available_at_runtime": False,
            }
        )
    if len(output_rows) != int(source_manifest["sample_count"]) or any(
        target_counts[family] < 1 for family in TARGETED_FAMILIES
    ):
        raise RuntimeError("V3 target augmentation did not preserve or cover the source rows")

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        traces_path = temporary / "traces.jsonl"
        with traces_path.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        body: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "sample_count": len(output_rows),
            "episode_count": int(source_manifest["episode_count"]),
            "train_scene_ids": train_scenes,
            "validation_scene_ids": validation_scenes,
            "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "scene_splits_disjoint": True,
            "source_trace_dataset_sha256": str(source_manifest["dataset_sha256"]),
            "source_trace_manifest_sha256": _sha256(source_root / "manifest.json"),
            "source_trace_rows_sha256": _sha256(source_root / "traces.jsonl"),
            "traces_sha256": _sha256(traces_path),
            "oracle_source_sha256": oracle_hashes,
            "targeted_families": sorted(TARGETED_FAMILIES),
            "targeted_sample_counts": {
                family: int(target_counts[family]) for family in sorted(TARGETED_FAMILIES)
            },
            "target_coordinates_oracle_derived": True,
            "target_coordinates_training_tree_only": True,
            "checkpoint_contains_trace_rows": False,
            "checkpoint_contains_object_labels": False,
            "runtime_oracle_inputs": False,
            "bounded_action_targets": source_manifest["bounded_action_targets"],
            "collision_checked_movement_targets": source_manifest[
                "collision_checked_movement_targets"
            ],
            "stop_targets_included": source_manifest["stop_targets_included"],
        }
        manifest = {**body, "dataset_sha256": _canonical_sha256(body)}
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root)
        return manifest
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()


def load_navigation_target_trace_v3(
    source: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strictly load and authenticate V3's training-only numeric targets."""

    root = _rooted(source)
    _require_training_path(root)
    manifest_path = root / "manifest.json"
    traces_path = root / "traces.jsonl"
    if (
        not root.is_dir()
        or {entry.name for entry in root.iterdir()} != {"manifest.json", "traces.jsonl"}
        or any(path.is_symlink() or not path.is_file() for path in (manifest_path, traces_path))
    ):
        raise ValueError("V3 target trace dataset must contain exactly two regular files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("V3 target trace manifest schema differs")
    body = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if (
        manifest.get("dataset_sha256") != _canonical_sha256(body)
        or manifest.get("traces_sha256") != _sha256(traces_path)
        or manifest.get("scene_splits_disjoint") is not True
        or manifest.get("target_coordinates_oracle_derived") is not True
        or manifest.get("target_coordinates_training_tree_only") is not True
        or manifest.get("checkpoint_contains_object_labels") is not False
        or manifest.get("runtime_oracle_inputs") is not False
    ):
        raise ValueError("V3 target trace manifest integrity or isolation differs")
    rows: list[dict[str, Any]] = []
    with traces_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("schema") != TRACE_SCHEMA:
                raise ValueError(f"V3 target trace row {index} schema differs")
            xyz = value.get("oracle_target_xyz_m")
            available = value.get("target_state_available")
            argument = value.get("argument_target_normalized")
            if (
                value.get("sample_id") != f"g_{index:08d}"
                or not isinstance(available, bool)
                or not isinstance(xyz, list)
                or len(xyz) != 3
                or not np.isfinite(np.asarray(xyz, dtype=np.float64)).all()
                or (not available and xyz != [0.0, 0.0, 0.0])
                or (available and not isinstance(value.get("target_query_sha256"), str))
                or (not available and value.get("target_query_sha256") is not None)
                or value.get("target_coordinates_training_only") is not True
                or value.get("oracle_available_at_runtime") is not False
                or isinstance(argument, bool)
                or not isinstance(argument, (int, float))
                or not math.isfinite(float(argument))
            ):
                raise ValueError(f"V3 target trace row {index} violates its contract")
            rows.append(value)
    if len(rows) != manifest.get("sample_count"):
        raise ValueError("V3 target trace row count differs")
    return manifest, rows


__all__ = [
    "MANIFEST_SCHEMA",
    "TARGETED_FAMILIES",
    "TRACE_SCHEMA",
    "generate_navigation_target_trace_v3",
    "load_navigation_target_trace_v3",
]

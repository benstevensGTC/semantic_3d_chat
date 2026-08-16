#!/usr/bin/env python3
"""Audit V4 checkpoint, prefix, map, and clearance with oracle unavailable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.navigation_policy_v4 import (
    CLEARANCE_RAY_COUNT,
    load_navigation_policy_v4_checkpoint,
    robot_frame_clearance_state,
)
from semantic_3d_chat.scene_encoder.map_io import validate_runtime_map_sidecars


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/navigation_policy_v4.yaml")
    parser.add_argument("--checkpoint", default="data_gemma4/checkpoints/navigation_policy_v4")
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/navigation_policy_v4_runtime_audit.json",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = config["navigation_policy_v4"]
    scene_id = args.scene
    map_path = PROJECT_ROOT / "data_gemma4" / "maps" / scene_id / "voxel_map.npz"
    prefix_root = PROJECT_ROOT / str(settings["prefix_cache_root"])
    prefix_manifest_path = prefix_root / "manifest.json"
    oracle = PROJECT_ROOT / "data" / "oracle"
    detached = oracle.with_name(".oracle_navigation_policy_v4_audit_detached")
    if detached.exists():
        raise FileExistsError(f"Stale detached oracle path exists: {detached}")
    audit = FileAccessAudit(
        [
            oracle,
            PROJECT_ROOT / "data" / "qa",
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
        ],
        forbidden_component_names={"oracle", "qa", "training", "scorer_only"},
        block_forbidden=True,
    )
    renamed = False
    try:
        os.replace(oracle, detached)
        renamed = True
        with audit:
            controller, metadata = load_navigation_policy_v4_checkpoint(
                args.checkpoint,
                expected_hidden_size=int(settings["hidden_size"]),
                expected_model_id=str(config["language"]["model_id"]),
                expected_model_revision=str(config["language"]["revision"]),
                audit=audit,
            )
            audit.record(map_path)
            validate_runtime_map_sidecars(map_path)
            robot = config["robot"]
            collision_map = NumericCollisionMap.from_voxel_map(
                map_path,
                room_size_m=config["scene"]["room_size_m"],
                robot_radius_m=float(robot["radius_m"]),
                collision_z_min_m=float(robot.get("collision_z_min_m", 0.12)),
                collision_z_max_m=float(robot.get("collision_z_max_m", 1.80)),
                surface_padding_m=float(robot.get("surface_padding_m", 0.035)),
            )
            initial = np.asarray(robot["initial_position_xy_m"], dtype=np.float64)
            clearance = robot_frame_clearance_state(collision_map, initial, 0.0)
            audit.record(prefix_manifest_path)
            prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
            prefix_entry = prefix_manifest["scenes"][scene_id]
            prefix_path = prefix_root / str(prefix_entry["filename"])
            audit.record(prefix_path)
            prefix_state = load_file(str(prefix_path), device="cpu")
            prefix = prefix_state.get("scene_prefix")
            if (
                not isinstance(prefix, torch.Tensor)
                or prefix.shape
                != (
                    1,
                    int(metadata["scene_token_count"]),
                    int(metadata["hidden_size"]),
                )
                or not torch.isfinite(prefix).all()
                or _sha256(prefix_path) != prefix_entry["file_sha256"]
            ):
                raise ValueError("V4 static prefix cache contract differs")
        audit.assert_clean()
    finally:
        if renamed:
            os.replace(detached, oracle)
    payload = {
        "schema": "semantic_3d_chat.navigation_policy_v4_runtime_audit.v4",
        "passed": True,
        "oracle_directory_unavailable_during_load": True,
        "oracle_directory_restored": oracle.is_dir() and not detached.exists(),
        "runtime_required_files": metadata["runtime_required_files"],
        "loaded_files": audit.unique_paths,
        "loaded_file_names": sorted(Path(path).name for path in audit.unique_paths),
        "forbidden_accesses": audit.forbidden_accesses(),
        "oracle_inputs_at_runtime": metadata["oracle_inputs_at_runtime"],
        "environmental_text_inputs_at_runtime": metadata["environmental_text_inputs"],
        "static_scene_prefix_question_independent": metadata[
            "question_independent_static_scene_prefix_required"
        ],
        "query_dependent_navigation_grounding": metadata[
            "query_dependent_grounding_navigation_only"
        ],
        "primary_static_scene_retrieval": False,
        "prefix_shape": list(prefix.shape),
        "prefix_file_sha256": _sha256(prefix_path),
        "numeric_clearance_shape": list(clearance.shape),
        "numeric_clearance_finite": bool(torch.isfinite(clearance).all()),
        "numeric_clearance_normalized": bool(
            torch.all((clearance >= 0.0) & (clearance <= 1.0))
        ),
        "clearance_ray_count": CLEARANCE_RAY_COUNT,
        "clearance_from_sanitized_geometry_only": metadata[
            "clearance_from_sanitized_geometry_only"
        ],
        "exact_collision_mask_required": metadata["exact_collision_mask_required"],
        "weights_sha256": metadata["weights_sha256"],
        "checkpoint_parameter_count": sum(
            parameter.numel() for parameter in controller.parameters()
        ),
    }
    _atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

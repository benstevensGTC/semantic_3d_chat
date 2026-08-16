"""Create-once oracle-only scoring for an already completed hybrid face run.

This module is evaluation code, not part of robot inference. It reads a sealed
runtime result only after inference has finished, then opens isolated simulator
oracle geometry to measure the final heading. No score or oracle value is fed
back into the runtime, controller, semantic grounder, or persistent map.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

ARTIFACT: Final[str] = "embodied_hybrid_face_oracle_score_v1"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Oracle scorer input must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Oracle scorer input must be one JSON object: {path}")
    return value


def _wrapped_degrees(value: float) -> float:
    return math.degrees(
        math.atan2(math.sin(math.radians(value)), math.cos(math.radians(value)))
    )


def build_face_oracle_score(
    runtime_result: str | Path,
    scene_oracle: str | Path,
    scoring_spec: str | Path,
    *,
    task_id: str = "nav_000",
) -> dict[str, Any]:
    """Score one completed face episode without modifying any input artifact."""

    result_path = _resolve(runtime_result)
    oracle_path = _resolve(scene_oracle)
    spec_path = _resolve(scoring_spec)
    result = _read_object(result_path)
    oracle = _read_object(oracle_path)
    spec = _read_object(spec_path)
    if (
        result.get("schema") != "semantic_3d_chat.embodied_conversation_result.v1"
        or result.get("passed_runtime_audit") is not True
        or result.get("forbidden_access_count") != 0
        or result.get("environmental_text_inputs") != []
    ):
        raise ValueError("Hybrid runtime result lacks clean inference attestation")
    scene_id = result.get("scene_id")
    if scene_id != oracle.get("scene_id") or scene_id != spec.get("scene_id"):
        raise ValueError("Hybrid result, oracle, and scoring spec scenes differ")

    tasks = spec.get("tasks")
    instances = oracle.get("instances")
    turns = result.get("turns")
    if not isinstance(tasks, list) or not isinstance(instances, list):
        raise TypeError("Oracle scoring inputs lack task or instance lists")
    task_rows = [row for row in tasks if isinstance(row, Mapping) and row.get("task_id") == task_id]
    if len(task_rows) != 1 or task_rows[0].get("family") != "face":
        raise ValueError("Oracle scoring spec lacks one requested face task")
    task = task_rows[0]
    target_id = task.get("target_instance_id")
    target_rows = [
        row
        for row in instances
        if isinstance(row, Mapping) and row.get("instance_id") == target_id
    ]
    if len(target_rows) != 1:
        raise ValueError("Oracle scene lacks one requested target instance")
    pose = target_rows[0].get("pose")
    if not isinstance(pose, Mapping):
        raise TypeError("Oracle target lacks numeric pose")
    target_xyz = pose.get("center_xyz_m")
    if (
        not isinstance(target_xyz, list)
        or len(target_xyz) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in target_xyz
        )
    ):
        raise ValueError("Oracle target center is invalid")
    if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], Mapping):
        raise ValueError("Hybrid result must contain one conversation turn")
    receipts = turns[0].get("action_receipts")
    if not isinstance(receipts, list) or not receipts or not isinstance(receipts[-1], Mapping):
        raise ValueError("Hybrid result has no final numeric receipt")
    final = receipts[-1]
    position = final.get("position_m")
    yaw_value = final.get("body_yaw_degrees")
    if (
        not isinstance(position, list)
        or len(position) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in position
        )
        or isinstance(yaw_value, bool)
        or not isinstance(yaw_value, (int, float))
        or not math.isfinite(float(yaw_value))
    ):
        raise ValueError("Hybrid result final pose is invalid")

    target = [float(value) for value in target_xyz]
    robot = [float(value) for value in position]
    final_yaw = float(yaw_value)
    desired_yaw = math.degrees(math.atan2(-(target[0] - robot[0]), target[1] - robot[1]))
    signed_error = _wrapped_degrees(desired_yaw - final_yaw)
    maximum_error = float(task["maximum_heading_error_degrees"])
    maximum_collisions = int(task["maximum_collisions"])
    collision_count = sum(
        1
        for receipt in receipts
        if isinstance(receipt, Mapping) and receipt.get("collision") is True
    )
    stopped = final.get("stopped") is True
    passed = bool(
        turns[0].get("success") is True
        and stopped
        and abs(signed_error) <= maximum_error
        and collision_count <= maximum_collisions
    )
    return {
        "artifact": ARTIFACT,
        "status": "passed" if passed else "failed",
        "scene_id": scene_id,
        "task_id": task_id,
        "family": "face",
        "runtime_result_sha256": _sha256(result_path),
        "runtime_result_path": result_path.relative_to(PROJECT_ROOT).as_posix(),
        "final_pose": {
            "position_xyz_m": robot,
            "body_yaw_degrees": final_yaw,
            "stopped": stopped,
        },
        "oracle_target": {
            "opaque_instance_id": target_id,
            "center_xyz_m": target,
            "desired_yaw_degrees": desired_yaw,
        },
        "heading": {
            "signed_error_degrees": signed_error,
            "absolute_error_degrees": abs(signed_error),
            "maximum_error_degrees": maximum_error,
        },
        "collision": {
            "count": collision_count,
            "maximum_count": maximum_collisions,
            "passed": collision_count <= maximum_collisions,
        },
        "passed": passed,
        "oracle_only_scorer_attestation": {
            "evaluation_only": True,
            "executed_after_runtime_completion": True,
            "runtime_result_reports_zero_forbidden_accesses": True,
            "runtime_result_reports_no_environmental_text_inputs": True,
            "oracle_geometry_loaded_by_primary_runtime": False,
            "oracle_geometry_loaded_by_scorer_only": True,
            "score_fed_back_to_runtime": False,
            "runtime_result_modified": False,
            "scene_oracle_sha256": _sha256(oracle_path),
            "scoring_spec_sha256": _sha256(spec_path),
        },
    }


def create_face_oracle_score(
    runtime_result: str | Path,
    scene_oracle: str | Path,
    scoring_spec: str | Path,
    output: str | Path,
    *,
    task_id: str = "nav_000",
) -> dict[str, Any]:
    """Atomically create one score and refuse all overwrite attempts."""

    destination = _resolve(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    payload = build_face_oracle_score(
        runtime_result,
        scene_oracle,
        scoring_spec,
        task_id=task_id,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return payload


__all__ = ["ARTIFACT", "build_face_oracle_score", "create_face_oracle_score"]

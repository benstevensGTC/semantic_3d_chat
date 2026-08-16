"""Run a leakage-blocked semantic policy, then score it in a separate phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, load_config, project_path
from semantic_3d_chat.robot.semantic_agent import (
    ContinuousSemanticTargetGrounder,
    GemmaProjectedTextEncoder,
    LabelFreeSemanticNavigator,
)
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator

BENCHMARK_TARGETS = (
    ("task_000", "table"),
    ("task_001", "chair"),
    ("task_002", "picture frame"),
    ("task_003", "bowl"),
    ("task_004", "floor lamp"),
    ("task_005", "cube"),
    ("task_006", "book"),
    ("task_007", "cabinet"),
    ("task_008", "plant pot"),
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return Path(os.path.abspath(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate))


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
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _runtime_phase(
    config: dict[str, Any],
    scene_id: str,
    *,
    map_path: Path,
    scratch: Path,
) -> dict[str, Any]:
    """Run policy with forbidden supervision roots blocked before file open."""

    runtime_config = json.loads(json.dumps(config))
    runtime_config["paths"]["data_root"] = str(scratch / "runtime")
    forbidden_roots = [
        PROJECT_ROOT / "data" / "oracle",
        PROJECT_ROOT / "data" / "qa",
        PROJECT_ROOT / "data_diverse28" / "qa",
        PROJECT_ROOT / "data_gemma4" / "training",
        PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
    ]
    audit = FileAccessAudit(
        forbidden_roots,
        forbidden_component_names={"oracle", "qa"},
        block_forbidden=True,
    )
    started = time.perf_counter()
    episodes: list[dict[str, Any]] = []
    with audit:
        text_encoder = GemmaProjectedTextEncoder.from_config(runtime_config)
        grounder = ContinuousSemanticTargetGrounder(
            map_path,
            text_encoder,
            room_size_m=runtime_config["scene"]["room_size_m"],
        )
        for task_id, target_text in BENCHMARK_TARGETS:
            simulator = EmbodiedCameraSimulator(runtime_config, scene_id)
            navigator = LabelFreeSemanticNavigator(simulator, grounder)
            task_started = time.perf_counter()
            result = navigator.navigate(target_text, scan_on_arrival=True)
            episodes.append(
                {
                    "task_id": task_id,
                    "result": result.as_dict(),
                    "elapsed_seconds": round(time.perf_counter() - task_started, 6),
                }
            )
        audit.assert_clean()
    loaded_files = audit.unique_paths
    serialized = json.dumps(episodes, sort_keys=True, allow_nan=False).casefold()
    for _task_id, target_text in BENCHMARK_TARGETS:
        if target_text.casefold() in serialized:
            raise RuntimeError("Numeric policy output serialized a target phrase")
    runtime_successes = sum(bool(item["result"]["success"]) for item in episodes)
    collisions = sum(int(item["result"]["collision_count"]) for item in episodes)
    return {
        "schema": "semantic_3d_chat.semantic_navigation_runtime.v1",
        "scene_id": scene_id,
        "passed": runtime_successes == len(episodes) and collisions == 0,
        "task_count": len(episodes),
        "runtime_success_count": runtime_successes,
        "collision_count": collisions,
        "policy": {
            "environment_input": "continuous_full_voxel_map_and_geometry",
            "query_input": "user_text_embedded_locally",
            "text_encoder_parameter_access": "gemma_tied_input_token_rows_only",
            "every_map_voxel_scored": True,
            "top_k_retrieval_used": False,
            "query_dependent_target_localization": True,
            "oracle_or_labels_available": False,
            "target_text_serialized_to_numeric_output": False,
            "bounded_geometry_planner": True,
            "arrival_rgbd_scan": True,
            "arrival_scan_fused_in_this_benchmark": False,
        },
        "map": {
            "path": str(map_path.relative_to(PROJECT_ROOT)),
            "sha256": episodes[0]["result"]["grounding"]["map_sha256"],
            "occupied_voxels": episodes[0]["result"]["grounding"]["scored_voxels"],
            "navigation_eligible_voxels": episodes[0]["result"]["grounding"]["eligible_voxels"],
            "feature_dim": 1536,
        },
        "episodes": episodes,
        "runtime_file_audit": {
            "blocking_enabled": True,
            "loaded_file_count": len(loaded_files),
            "loaded_file_inventory_sha256": _canonical_sha256(loaded_files),
            "forbidden_accesses": audit.forbidden_accesses(),
            "oracle_or_qa_loaded": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def _distance_to_box(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(lower - point, point - upper), 0.0)))


def _score_phase(
    runtime_report: dict[str, Any],
    *,
    oracle_path: Path,
) -> dict[str, Any]:
    """Evaluation-only scorer; this function is never called inside the audit."""

    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if oracle.get("scene_id") != runtime_report["scene_id"]:
        raise ValueError("Oracle scorer scene differs from opaque runtime scene")
    by_category = {
        instance["category"]: instance
        for instance in oracle.get("instances", [])
        if instance.get("kind") == "object"
    }
    scored: list[dict[str, Any]] = []
    for episode, (task_id, category) in zip(
        runtime_report["episodes"], BENCHMARK_TARGETS, strict=True
    ):
        if episode["task_id"] != task_id or category not in by_category:
            raise ValueError("Navigation task/scorer inventory changed")
        instance = by_category[category]
        lower = np.asarray(instance["bbox"]["min_xyz_m"], dtype=np.float64)
        upper = np.asarray(instance["bbox"]["max_xyz_m"], dtype=np.float64)
        grounding = np.asarray(episode["result"]["grounding"]["target_xyz_m"], dtype=np.float64)
        final = np.asarray(episode["result"]["final_position_m"], dtype=np.float64)
        grounding_error = _distance_to_box(grounding, lower, upper)
        navigation_error = _distance_to_box(final[:2], lower[:2], upper[:2])
        scored.append(
            {
                "task_id": task_id,
                "category": category,
                "grounding_bbox_error_m": grounding_error,
                "navigation_bbox_standoff_m": navigation_error,
                "grounding_success": grounding_error <= 0.15,
                "navigation_success": navigation_error <= 0.85
                and bool(episode["result"]["success"]),
                "collision_count": int(episode["result"]["collision_count"]),
            }
        )

    # Wrong-target control: cyclically attach each continuous estimate to the
    # next requested target's box.  This uses no additional policy inference.
    wrong_grounding_hits = 0
    for index, episode in enumerate(runtime_report["episodes"]):
        wrong_category = BENCHMARK_TARGETS[(index + 1) % len(BENCHMARK_TARGETS)][1]
        wrong = by_category[wrong_category]
        point = np.asarray(episode["result"]["grounding"]["target_xyz_m"], dtype=np.float64)
        error = _distance_to_box(
            point,
            np.asarray(wrong["bbox"]["min_xyz_m"], dtype=np.float64),
            np.asarray(wrong["bbox"]["max_xyz_m"], dtype=np.float64),
        )
        wrong_grounding_hits += int(error <= 0.15)

    grounding_hits = sum(item["grounding_success"] for item in scored)
    navigation_hits = sum(item["navigation_success"] for item in scored)
    count = len(scored)
    return {
        "schema": "semantic_3d_chat.semantic_navigation_benchmark.v1",
        "scene_id": runtime_report["scene_id"],
        "passed": grounding_hits >= 7
        and navigation_hits >= 7
        and runtime_report["collision_count"] == 0,
        "separation": {
            "runtime_completed_before_oracle_open": True,
            "runtime_report_sha256": _canonical_sha256(runtime_report),
            "oracle_used_only_by_this_scorer": True,
            "policy_received_oracle_or_labels": False,
        },
        "metrics": {
            "task_count": count,
            "grounding_success_count": grounding_hits,
            "grounding_success_rate": grounding_hits / count,
            "navigation_success_count": navigation_hits,
            "navigation_success_rate": navigation_hits / count,
            "collision_count": runtime_report["collision_count"],
            "cyclic_wrong_target_grounding_success_count": wrong_grounding_hits,
            "cyclic_wrong_target_grounding_success_rate": wrong_grounding_hits / count,
            "mean_grounding_bbox_error_m": float(
                np.mean([item["grounding_bbox_error_m"] for item in scored])
            ),
            "mean_navigation_bbox_standoff_m": float(
                np.mean([item["navigation_bbox_standoff_m"] for item in scored])
            ),
        },
        "thresholds": {
            "grounding_bbox_error_m_maximum": 0.15,
            "navigation_bbox_standoff_m_maximum": 0.85,
            "minimum_grounding_successes": 7,
            "minimum_navigation_successes": 7,
            "maximum_collisions": 0,
        },
        "tasks": scored,
        "limitations": [
            "This is one deterministic development arrangement used during policy engineering, not held-out navigation generalization.",
            "The arrival scan reobserves the sanitized map; fresh Blender RGB-D fusion is proven by the separate embodied-runtime smoke report.",
            "The tied-token semantic stream misses one target in this scene and remains uncalibrated.",
            "The policy executes a deterministic semantic target instruction, not autonomous multi-turn tool selection by the language model.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime/embodied_v54.yaml")
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument(
        "--runtime-output",
        default="reports/gemma4/metrics/semantic_navigation_runtime_scene_000001.json",
    )
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/semantic_navigation_scene_000001.json",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(_rooted(args.config))
    map_path = project_path(config, "maps", args.scene, "voxel_map.npz").resolve()
    runtime_output = _rooted(args.runtime_output)
    output = _rooted(args.output)
    with tempfile.TemporaryDirectory(
        prefix="semantic_3d_navigation.", dir="/private/tmp"
    ) as directory:
        runtime_report = _runtime_phase(
            config,
            args.scene,
            map_path=map_path,
            scratch=Path(directory),
        )
    # Persist the completed policy result before evaluation is allowed to open
    # any oracle.  The output contains opaque task IDs and numeric values only.
    _atomic_json(runtime_output, runtime_report)
    oracle_path = project_path(config, "oracle", args.scene, "oracle.json").resolve()
    score = _score_phase(runtime_report, oracle_path=oracle_path)
    score["artifacts"] = {
        "runtime_report": str(runtime_output.relative_to(PROJECT_ROOT)),
        "runtime_report_file_sha256": _sha256(runtime_output),
    }
    _atomic_json(output, score)
    print(json.dumps(score, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

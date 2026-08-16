"""Create and authenticate the sealed Navigation V4 training preregistration.

The preregistration is written once before an optimizer is constructed.  It
binds the sole V4 arm to the frozen V3 checkpoint, exact 14/8 scene split,
anonymous numeric maps, implementation sources, causal controls, and terminal
acceptance gates.  Neither the live navigation benchmark nor its oracle scorer
is opened here or by V4 training.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

SCHEMA: Final[str] = "semantic_3d_chat.navigation_policy_v4_preregistration.v1"
V3_DATASET_SHA256: Final[str] = (
    "d8d97ac248a5821eb971301efb742c25c996627bae22d6417c02755e61d50f9d"
)
V4_DATASET_SHA256: Final[str] = (
    "c1a383b27bbfb114354c083fc90a7f92eaefc445d2bf8f71b818bd66826ea8ec"
)
V3_WEIGHTS_SHA256: Final[str] = (
    "975c7c6c5e103dd1bb055feb2eceff6cc7fe9ab82a3f7f492a8fbdb5a26cc87c"
)
V3_CHECKPOINT_TREE_SHA256: Final[str] = (
    "1998d7313d023e65a5bef4c52eee00d867b4006344de51c6830e35c32f7ed1e0"
)

_IMPLEMENTATION_PATHS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/navigation_policy_v4_preregistration.py",
    "src/semantic_3d_chat/robot/navigation_policy_v4.py",
    "src/semantic_3d_chat/training/train_navigation_policy_v4.py",
    "src/semantic_3d_chat/robot/conversation_cli.py",
    "src/semantic_3d_chat/robot/llm_tool_policy.py",
    "src/semantic_3d_chat/evaluation/llm_navigation_benchmark.py",
    "src/semantic_3d_chat/scene_encoder/map_io.py",
    "src/semantic_3d_chat/robot/collision.py",
    "scripts/preregister_navigation_policy_v4.py",
    "scripts/train_navigation_policy_v4.py",
    "scripts/evaluate_navigation_policy_v4.py",
    "scripts/audit_navigation_policy_v4_runtime.py",
    "scripts/run_llm_navigation_inference.py",
)
_INPUT_PATHS: Final[tuple[str, ...]] = (
    "configs/experiments/navigation_policy_v4.yaml",
    "configs/experiments/navigation_policy_v3.yaml",
    "configs/runtime/embodied_v54.yaml",
    "configs/runtime/gemma4_v54.yaml",
    "data_gemma4/training/navigation_policy_v3/manifest.json",
    "data_gemma4/training/navigation_policy_v3/traces.jsonl",
    "data_gemma4/scene_tokens/v56_question_control_full_prefixes/manifest.json",
    "data_gemma4/checkpoints/navigation_policy_v3/policy.safetensors",
    "data_gemma4/checkpoints/navigation_policy_v3/runtime_metadata.json",
    "data_gemma4/checkpoints/robot_state_numeric_v1/state.safetensors",
    "data_gemma4/checkpoints/robot_state_numeric_v1/runtime_metadata.json",
)
_THRESHOLD_NAMES: Final[tuple[str, ...]] = (
    "minimum_validation_action_accuracy",
    "minimum_validation_update_after_scan_accuracy",
    "minimum_validation_stop_recall",
    "minimum_validation_turn_sign_accuracy",
    "maximum_validation_argument_mae",
    "minimum_unsafe_motion_rejection",
    "minimum_collision_risk_accuracy",
    "minimum_shuffled_clearance_family_drop",
    "minimum_zero_target_targeted_accuracy_drop",
    "minimum_wrong_target_turn_sign_drop",
)
_HYPERPARAMETER_NAMES: Final[tuple[str, ...]] = (
    "hidden_size",
    "model_dim",
    "scene_token_count",
    "robot_token_count",
    "clearance_ray_count",
    "clearance_max_range_m",
    "collision_probe_distances_m",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "argument_loss_weight",
    "turn_sign_loss_weight",
    "collision_risk_loss_weight",
    "clearance_change_loss_weight",
    "clearance_change_margin",
    "gradient_clip_norm",
    "early_stopping_patience",
    "seed",
    "device",
)


class PreregistrationError(RuntimeError):
    """Raised when the sealed V4 preregistration no longer matches its inputs."""


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def file_sha256(path: str | Path) -> str:
    source = _rooted(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def tree_sha256(path: str | Path) -> str:
    root = _rooted(path)
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"V4 preregistration tree is unavailable: {root}")
    digest = hashlib.sha256()
    for member in sorted(item for item in root.rglob("*") if item.is_file()):
        if member.is_symlink():
            raise ValueError(f"V4 preregistration rejects symlink: {member}")
        digest.update(member.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(member)))
    return digest.hexdigest()


def _hash_inventory(paths: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        source = _rooted(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V4 preregistration source is unavailable: {relative}")
        result[relative] = file_sha256(source)
    return result


def _stable_config_sha256(config: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in config.items() if not key.startswith("_")}
    return canonical_sha256(stable)


def _v3_checkpoint_identity() -> dict[str, Any]:
    metadata_path = _rooted(
        "data_gemma4/checkpoints/navigation_policy_v3/runtime_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("V3 initialization metadata must be an object")
    observed_weights = file_sha256(
        "data_gemma4/checkpoints/navigation_policy_v3/policy.safetensors"
    )
    observed_tree = tree_sha256("data_gemma4/checkpoints/navigation_policy_v3")
    if (
        observed_weights != V3_WEIGHTS_SHA256
        or observed_tree != V3_CHECKPOINT_TREE_SHA256
        or metadata.get("weights_sha256") != V3_WEIGHTS_SHA256
        or metadata.get("training_dataset_sha256") != V3_DATASET_SHA256
        or metadata.get("task_trained") is not True
        or metadata.get("oracle_inputs_at_runtime") is not False
        or metadata.get("environmental_text_inputs") != []
    ):
        raise PreregistrationError("Frozen V3 initialization differs from its sealed identity")
    return {
        "checkpoint_tree_sha256": observed_tree,
        "weights_sha256": observed_weights,
        "training_dataset_sha256": metadata["training_dataset_sha256"],
        "architecture": metadata.get("architecture"),
        "schema_version": metadata.get("schema_version"),
        "task_trained": True,
        "runtime_oracle_inputs": False,
        "environmental_text_inputs": [],
    }


def build_navigation_policy_v4_preregistration(
    config: Mapping[str, Any],
    *,
    source_v3_dataset_sha256: str,
    v4_dataset_sha256: str,
    map_sha256: Mapping[str, str],
) -> dict[str, Any]:
    settings = config.get("navigation_policy_v4")
    if not isinstance(settings, Mapping):
        raise TypeError("Config has no navigation_policy_v4 settings")
    train_ids = settings.get("train_scene_ids")
    validation_ids = settings.get("validation_scene_ids")
    expected_maps = set(train_ids or ()) | set(validation_ids or ())
    if (
        source_v3_dataset_sha256 != V3_DATASET_SHA256
        or v4_dataset_sha256 != V4_DATASET_SHA256
        or not isinstance(train_ids, list)
        or not isinstance(validation_ids, list)
        or len(train_ids) != 14
        or len(validation_ids) != 8
        or set(train_ids) & set(validation_ids)
        or set(map_sha256) != expected_maps
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in map_sha256.values()
        )
    ):
        raise PreregistrationError("V4 prepared dataset identity differs from the sole arm")
    missing = [
        name
        for name in (*_HYPERPARAMETER_NAMES, *_THRESHOLD_NAMES)
        if name not in settings
    ]
    if missing:
        raise PreregistrationError(f"V4 preregistration settings are missing: {missing}")
    return {
        "schema": SCHEMA,
        "status": "sealed_before_optimizer_or_live_v4_benchmark",
        "research_change": (
            "frozen_v3_plus_anonymous_24_ray_robot_frame_clearance_residual_"
            "collision_risk_head_and_exact_safe_nonterminal_action_mask"
        ),
        "diagnosed_predecessor_failure": {
            "predecessor": "navigation_policy_v3",
            "live_success_count": 5,
            "live_task_count": 6,
            "failed_task_id": "nav_005",
            "failure": (
                "an_unsafe_forward_move_was_postprocessed_directly_to_stop_"
                "outside_the_required_target_standoff"
            ),
            "semantic_target_xy_error_m": 0.157,
            "v3_artifacts_modified": False,
        },
        "single_arm": {
            "one_arm_only": True,
            "hyperparameter_search": False,
            "seed": settings["seed"],
            "frozen_v3_base": True,
            "trainable_components": [
                "clearance_encoder",
                "action_delta",
                "argument_delta",
                "collision_risk_head",
            ],
            "hyperparameters": {
                name: settings[name] for name in _HYPERPARAMETER_NAMES
            },
        },
        "data": {
            "train_scene_ids": list(train_ids),
            "validation_scene_ids": list(validation_ids),
            "scene_splits_disjoint": True,
            "train_scene_count": 14,
            "validation_scene_count": 8,
            "source_v3_dataset_sha256": source_v3_dataset_sha256,
            "prepared_v4_dataset_sha256": v4_dataset_sha256,
            "map_sha256": dict(sorted(map_sha256.items())),
            "clearance_and_collision_targets_from_numeric_maps_only": True,
            "oracle_coordinates_used_only_in_blocked_training_target_state": True,
            "live_benchmark_used_for_training_or_selection": False,
            "benchmark_oracle_used_for_training_or_selection": False,
        },
        "architecture_contract": {
            "clearance_ray_count": 24,
            "clearance_max_range_m": 1.0,
            "clearance_coordinate_frame": "robot",
            "ray_zero": "forward",
            "risk_probe_distances_m": [0.125, 0.25, 0.375, 0.5],
            "clearance_from_sanitized_anonymous_geometry_only": True,
            "exact_collision_mask_authoritative": True,
            "unsafe_motion_fallback": "highest_safe_nonterminal_action",
            "final_collision_interlock": True,
            "static_scene_prefix_computed_before_question": True,
            "static_scene_prefix_question_independent": True,
            "navigation_target_grounding_query_dependent": True,
            "all_map_voxels_scored_for_navigation_grounding": True,
            "primary_static_qa_retrieval": False,
        },
        "acceptance_gates": {
            name: settings[name] for name in _THRESHOLD_NAMES
        },
        "controls": {
            "held_out_primary": True,
            "shuffled_clearance": True,
            "zero_clearance": True,
            "wrong_target": True,
            "zero_target": True,
            "unsafe_motion_mask": True,
            "validation_scenes_never_receive_gradients": True,
            "live_six_task_run_only_after_all_offline_gates_pass": True,
        },
        "frozen_v3_initialization": _v3_checkpoint_identity(),
        "merged_config_sha256": _stable_config_sha256(config),
        "implementation_source_hashes": _hash_inventory(_IMPLEMENTATION_PATHS),
        "input_artifact_hashes": _hash_inventory(_INPUT_PATHS),
        "runtime_separation": {
            "environmental_text_inputs": [],
            "oracle_inputs": False,
            "labels_or_object_ids": False,
            "checkpoint_inventory": ["policy.safetensors", "runtime_metadata.json"],
            "preregistration_is_not_a_runtime_input": True,
        },
        "publication": {
            "training_report_create_once": True,
            "checkpoint_written_only_if_every_gate_passes": True,
            "rejected_arm_publishes_no_checkpoint": True,
            "no_posthoc_threshold_change": True,
        },
    }


def write_navigation_policy_v4_preregistration(
    destination: str | Path,
    config: Mapping[str, Any],
    *,
    source_v3_dataset_sha256: str,
    v4_dataset_sha256: str,
    map_sha256: Mapping[str, str],
    training_report: str | Path = (
        "reports/gemma4/metrics/navigation_policy_v4_training.json"
    ),
) -> tuple[Path, str]:
    path = _rooted(destination)
    settings = config["navigation_policy_v4"]
    checkpoint = _rooted(str(settings["checkpoint_output"]))
    report = _rooted(training_report)
    if path.exists() or checkpoint.exists() or report.exists():
        raise FileExistsError(
            "V4 preregistration requires absent preregistration, checkpoint, and training report"
        )
    payload = build_navigation_policy_v4_preregistration(
        config,
        source_v3_dataset_sha256=source_v3_dataset_sha256,
        v4_dataset_sha256=v4_dataset_sha256,
        map_sha256=map_sha256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path, file_sha256(path)


def authenticate_navigation_policy_v4_preregistration(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    source_v3_dataset_sha256: str,
    v4_dataset_sha256: str,
    map_sha256: Mapping[str, str],
) -> dict[str, Any]:
    source = _rooted(path)
    if not source.is_file() or source.is_symlink():
        raise PreregistrationError("Sealed V4 preregistration is unavailable")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PreregistrationError("Sealed V4 preregistration is malformed") from exc
    expected = build_navigation_policy_v4_preregistration(
        config,
        source_v3_dataset_sha256=source_v3_dataset_sha256,
        v4_dataset_sha256=v4_dataset_sha256,
        map_sha256=map_sha256,
    )
    if payload != expected:
        raise PreregistrationError(
            "Sealed V4 preregistration differs from current sources, inputs, or gates"
        )
    return {
        "authenticated": True,
        "path": str(
            source.relative_to(PROJECT_ROOT)
            if source.is_relative_to(PROJECT_ROOT)
            else source
        ),
        "sha256": file_sha256(source),
        "single_arm": True,
        "sealed_before_training": True,
    }


__all__ = [
    "SCHEMA",
    "V3_DATASET_SHA256",
    "V3_WEIGHTS_SHA256",
    "V4_DATASET_SHA256",
    "PreregistrationError",
    "authenticate_navigation_policy_v4_preregistration",
    "build_navigation_policy_v4_preregistration",
    "canonical_sha256",
    "file_sha256",
    "tree_sha256",
    "write_navigation_policy_v4_preregistration",
]

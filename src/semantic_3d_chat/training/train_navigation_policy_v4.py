"""Train and causally evaluate collision-aware navigation-policy V4."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.navigation_policy_v41_preregistration import (
    authenticate_navigation_policy_v41_preregistration,
)
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.navigation_policy import (
    ACTION_NAMES,
    ACTION_TO_INDEX,
    tool_call_from_prediction,
)
from semantic_3d_chat.robot.navigation_policy_v3 import (
    grounded_target_state,
    load_navigation_policy_v3_checkpoint,
)
from semantic_3d_chat.robot.navigation_policy_v4 import (
    CLEARANCE_MAX_RANGE_M,
    CLEARANCE_RAY_COUNT,
    COLLISION_PROBE_DISTANCES_M,
    COLLISION_RISK_DIM,
    ClearanceAwareNavigationControllerV4,
    counterfactual_motion_collision_targets,
    load_navigation_policy_v4_checkpoint,
    robot_frame_clearance_state,
    save_navigation_policy_v4_checkpoint,
)
from semantic_3d_chat.scene_encoder.map_io import validate_runtime_map_sidecars
from semantic_3d_chat.training.train_navigation_policy_v3 import (
    PreparedSamplesV3,
    PreparedTrainingDataV3,
    prepare_navigation_policy_v3_data,
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


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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


@dataclass(frozen=True)
class PreparedSamplesV4:
    base: PreparedSamplesV3
    clearance_states: torch.Tensor
    collision_targets: torch.Tensor

    def __len__(self) -> int:
        return len(self.base)


@dataclass(frozen=True)
class PreparedTrainingDataV4:
    base: PreparedTrainingDataV3
    train: PreparedSamplesV4
    validation: PreparedSamplesV4
    map_sha256: dict[str, str]
    dataset_sha256: str


def validate_navigation_policy_v4_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("navigation_policy_v4")
    if not isinstance(settings, dict):
        raise TypeError("Config has no navigation_policy_v4 mapping")
    for name in (
        "hidden_size",
        "model_dim",
        "scene_token_count",
        "robot_token_count",
        "batch_size",
        "epochs",
        "early_stopping_patience",
        "seed",
        "clearance_ray_count",
    ):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"navigation_policy_v4.{name} must be a positive integer")
    for name in (
        "clearance_max_range_m",
        "learning_rate",
        "weight_decay",
        "argument_loss_weight",
        "turn_sign_loss_weight",
        "collision_risk_loss_weight",
        "clearance_change_loss_weight",
        "clearance_change_margin",
        "gradient_clip_norm",
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
    ):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"navigation_policy_v4.{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"navigation_policy_v4.{name} must be finite and nonnegative")
    if settings.get("device", "cpu") != "cpu":
        raise ValueError("Navigation-policy V4 training is deliberately CPU-only")
    preregistration = settings.get("preregistration")
    if not isinstance(preregistration, str) or not preregistration.strip():
        raise ValueError("Navigation-policy V4 requires a sealed preregistration path")
    if settings["clearance_ray_count"] != CLEARANCE_RAY_COUNT:
        raise ValueError("V4 preregisters exactly 24 clearance rays")
    if float(settings["clearance_max_range_m"]) != CLEARANCE_MAX_RANGE_M:
        raise ValueError("V4 clearance range differs from the fixed contract")
    if settings.get("collision_probe_distances_m") != list(COLLISION_PROBE_DISTANCES_M):
        raise ValueError("V4 collision probe distances differ from the fixed contract")
    train_scenes = settings.get("train_scene_ids")
    validation_scenes = settings.get("validation_scene_ids")
    if (
        not isinstance(train_scenes, list)
        or not isinstance(validation_scenes, list)
        or len(train_scenes) != 14
        or len(validation_scenes) != 8
        or set(train_scenes) & set(validation_scenes)
    ):
        raise ValueError("V4 requires the preregistered 14/8 disjoint scene split")
    v3 = config.get("navigation_policy_v3")
    if not isinstance(v3, dict) or train_scenes != v3.get(
        "train_scene_ids"
    ) or validation_scenes != v3.get("validation_scene_ids"):
        raise ValueError("V4 scene split must exactly preserve sealed V3")
    return settings


def _collision_map(config: dict[str, Any], path: Path) -> NumericCollisionMap:
    validate_runtime_map_sidecars(path)
    robot = config["robot"]
    return NumericCollisionMap.from_voxel_map(
        path,
        room_size_m=config["scene"]["room_size_m"],
        robot_radius_m=float(robot["radius_m"]),
        collision_z_min_m=float(robot.get("collision_z_min_m", 0.12)),
        collision_z_max_m=float(robot.get("collision_z_max_m", 1.80)),
        surface_padding_m=float(robot.get("surface_padding_m", 0.035)),
    )


def _world_pose(state: torch.Tensor, room_size_m: list[float]) -> tuple[np.ndarray, float]:
    if state.shape != (18,):
        raise ValueError("V4 state feature row must have shape [18]")
    room = torch.tensor(room_size_m, dtype=torch.float32)
    minimum = torch.tensor([-room[0] / 2.0, -room[1] / 2.0, 0.0])
    maximum = torch.tensor([room[0] / 2.0, room[1] / 2.0, room[2]])
    position = (state[:3] + 1.0) * 0.5 * (maximum - minimum) + minimum
    yaw = math.degrees(math.atan2(float(state[3]), float(state[4])))
    return position[:2].numpy().astype(np.float64), yaw


def _augment_samples(
    config: dict[str, Any],
    samples: PreparedSamplesV3,
    *,
    map_root: Path,
    maps: dict[str, NumericCollisionMap],
) -> PreparedSamplesV4:
    room = list(config["scene"]["room_size_m"])
    cache: dict[tuple[str, tuple[float, ...]], torch.Tensor] = {}
    clearances: list[torch.Tensor] = []
    risks: list[torch.Tensor] = []
    for index, scene_id in enumerate(samples.scene_ids):
        if scene_id not in maps:
            path = map_root / scene_id / "voxel_map.npz"
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"V4 numeric map is unavailable: {path}")
            maps[scene_id] = _collision_map(config, path)
        state = samples.state_features[index]
        key = (scene_id, tuple(float(value) for value in state[:5]))
        clearance = cache.get(key)
        if clearance is None:
            position, yaw = _world_pose(state, room)
            clearance = robot_frame_clearance_state(
                maps[scene_id], position, yaw
            ).contiguous()
            cache[key] = clearance
        clearances.append(clearance)
        risks.append(counterfactual_motion_collision_targets(clearance))
    clearance_tensor = torch.stack(clearances)
    risk_tensor = torch.stack(risks)
    if (
        clearance_tensor.shape != (len(samples), CLEARANCE_RAY_COUNT)
        or risk_tensor.shape != (len(samples), COLLISION_RISK_DIM)
    ):
        raise RuntimeError("V4 clearance augmentation shape differs")
    return PreparedSamplesV4(samples, clearance_tensor, risk_tensor)


@torch.inference_mode()
def prepare_navigation_policy_v4_data(
    config: dict[str, Any], dataset_path: Path
) -> tuple[dict[str, Any], PreparedTrainingDataV4]:
    settings = validate_navigation_policy_v4_settings(config)
    manifest, base = prepare_navigation_policy_v3_data(config, dataset_path)
    if (
        manifest["train_scene_ids"] != settings["train_scene_ids"]
        or manifest["validation_scene_ids"] != settings["validation_scene_ids"]
    ):
        raise ValueError("V4 source trace split differs from preregistration")
    map_root = _rooted(str(settings["map_root"]))
    maps: dict[str, NumericCollisionMap] = {}
    train = _augment_samples(config, base.train, map_root=map_root, maps=maps)
    validation = _augment_samples(
        config, base.validation, map_root=map_root, maps=maps
    )
    map_hashes = {
        scene_id: _sha256(map_root / scene_id / "voxel_map.npz")
        for scene_id in [*settings["train_scene_ids"], *settings["validation_scene_ids"]]
    }
    identity = {
        "schema": "semantic_3d_chat.navigation_policy_v4_dataset_identity.v4",
        "source_v3_dataset_sha256": manifest["dataset_sha256"],
        "train_scene_ids": settings["train_scene_ids"],
        "validation_scene_ids": settings["validation_scene_ids"],
        "map_sha256": map_hashes,
        "clearance_ray_count": CLEARANCE_RAY_COUNT,
        "clearance_max_range_m": CLEARANCE_MAX_RANGE_M,
        "collision_probe_distances_m": list(COLLISION_PROBE_DISTANCES_M),
        "train_clearance_sha256": _tensor_sha256(train.clearance_states),
        "validation_clearance_sha256": _tensor_sha256(validation.clearance_states),
        "train_collision_targets_sha256": _tensor_sha256(train.collision_targets),
        "validation_collision_targets_sha256": _tensor_sha256(
            validation.collision_targets
        ),
        "oracle_coordinates_used_for_training_target_state_only": True,
        "clearance_and_collision_targets_from_numeric_maps_only": True,
        "runtime_oracle_inputs": False,
        "environmental_text_inputs_at_runtime": [],
    }
    return manifest, PreparedTrainingDataV4(
        base=base,
        train=train,
        validation=validation,
        map_sha256=map_hashes,
        dataset_sha256=_canonical_sha256(identity),
    )


def _scene_batches(
    samples: PreparedSamplesV4,
    batch_size: int,
    *,
    generator: torch.Generator | None,
) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    scene_values = torch.unique(samples.base.scene_indices, sorted=True)
    if generator is not None:
        scene_values = scene_values[torch.randperm(len(scene_values), generator=generator)]
    for scene_index in scene_values:
        indices = torch.nonzero(
            samples.base.scene_indices == scene_index, as_tuple=True
        )[0]
        if generator is not None:
            indices = indices[torch.randperm(len(indices), generator=generator)]
        batches.extend(
            indices[start : start + batch_size]
            for start in range(0, len(indices), batch_size)
        )
    return batches


def _numeric_action_mask(targets: torch.Tensor) -> torch.Tensor:
    return (
        (targets == ACTION_TO_INDEX["turn"])
        | (targets == ACTION_TO_INDEX["move_forward"])
        | (targets == ACTION_TO_INDEX["move_backward"])
    )


def _forward(
    controller: ClearanceAwareNavigationControllerV4,
    prefixes: torch.Tensor,
    samples: PreparedSamplesV4,
    indices: torch.Tensor,
    *,
    clearance_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_scenes = samples.base.scene_indices[indices]
    unique_scenes, inverse = torch.unique(
        selected_scenes, sorted=True, return_inverse=True
    )
    clearance = (
        samples.clearance_states[indices]
        if clearance_override is None
        else clearance_override
    )
    return controller(
        prefixes[unique_scenes],
        samples.base.robot_tokens[indices],
        samples.base.instruction_embeddings[indices],
        samples.base.target_states[indices],
        clearance,
        scene_batch_indices=inverse,
    )


def _candidate_collision_from_clearance(
    action_index: int,
    normalized_argument: float,
    clearance: torch.Tensor,
    *,
    max_turn_degrees: float,
    max_move_m: float,
) -> bool:
    call = tool_call_from_prediction(
        action_index,
        normalized_argument,
        max_turn_degrees=max_turn_degrees,
        max_move_m=max_move_m,
    )
    if call["tool"] not in {"move_forward", "move_backward"}:
        return False
    ray = 0 if call["tool"] == "move_forward" else CLEARANCE_RAY_COUNT // 2
    free = float(clearance[ray]) * CLEARANCE_MAX_RANGE_M
    return float(call["arguments"]["distance_meters"]) >= free - 1e-6


@torch.inference_mode()
def evaluate_prepared_v4(
    controller: ClearanceAwareNavigationControllerV4,
    prefixes: torch.Tensor,
    samples: PreparedSamplesV4,
    *,
    batch_size: int,
    max_turn_degrees: float,
    max_move_m: float,
    clearance_mode: str = "primary",
) -> dict[str, Any]:
    if clearance_mode not in {"primary", "zero", "shuffled"}:
        raise ValueError("V4 clearance control mode is invalid")
    controller.eval()
    predicted = torch.empty(len(samples), dtype=torch.long)
    argument_predictions = torch.empty(len(samples), dtype=torch.float32)
    risk_logits_all = torch.empty((len(samples), COLLISION_RISK_DIM))
    unsafe_raw = 0
    unsafe_after_exact_mask = 0
    for indices in _scene_batches(samples, batch_size, generator=None):
        if clearance_mode == "zero":
            override = torch.zeros_like(samples.clearance_states[indices])
        elif clearance_mode == "shuffled":
            override = torch.roll(samples.clearance_states[indices], shifts=1, dims=0)
        else:
            override = None
        logits, arguments, risk_logits = _forward(
            controller,
            prefixes,
            samples,
            indices,
            clearance_override=override,
        )
        actions = torch.argmax(logits, dim=-1).cpu()
        predicted[indices] = actions
        argument_predictions[indices] = (
            arguments.gather(1, samples.base.action_targets[indices, None])
            .squeeze(1)
            .cpu()
        )
        risk_logits_all[indices] = risk_logits.cpu()
        active_clearance = (
            samples.clearance_states[indices] if override is None else override
        )
        for row, action in enumerate(actions.tolist()):
            argument = float(arguments[row, action])
            if _candidate_collision_from_clearance(
                action,
                argument,
                active_clearance[row],
                max_turn_degrees=max_turn_degrees,
                max_move_m=max_move_m,
            ):
                unsafe_raw += 1
                ranked = torch.argsort(
                    logits[row].detach().cpu(), descending=True, stable=True
                ).tolist()
                selected_safe = False
                for candidate in ranked[1:]:
                    if ACTION_NAMES[int(candidate)] == "stop":
                        continue
                    if not _candidate_collision_from_clearance(
                        int(candidate),
                        float(arguments[row, int(candidate)]),
                        active_clearance[row],
                        max_turn_degrees=max_turn_degrees,
                        max_move_m=max_move_m,
                    ):
                        selected_safe = True
                        break
                unsafe_after_exact_mask += int(not selected_safe)

    target = samples.base.action_targets
    correct = predicted == target
    stop_mask = target == ACTION_TO_INDEX["stop"]
    turn_mask = target == ACTION_TO_INDEX["turn"]
    numeric_mask = _numeric_action_mask(target)
    targeted = samples.base.target_available
    turn_sign = torch.sign(argument_predictions[turn_mask]) == torch.sign(
        samples.base.argument_targets[turn_mask]
    )
    turn_sign &= predicted[turn_mask] == target[turn_mask]
    family_metrics: dict[str, Any] = {}
    for family in sorted(set(samples.base.families)):
        mask = torch.tensor([value == family for value in samples.base.families])
        family_metrics[family] = {
            "sample_count": int(mask.sum()),
            "action_accuracy": float(correct[mask].float().mean()),
        }
    risk_prediction = risk_logits_all >= 0.0
    risk_target = samples.collision_targets.bool()
    counts = Counter(ACTION_NAMES[int(value)] for value in predicted.tolist())
    targeted_accuracy = (
        float(correct[targeted].float().mean()) if bool(targeted.any()) else 0.0
    )
    targetless_accuracy = (
        float(correct[~targeted].float().mean())
        if bool((~targeted).any())
        else 0.0
    )
    return {
        "sample_count": len(samples),
        "scene_count": len(set(samples.base.scene_ids)),
        "targeted_sample_count": int(targeted.sum()),
        "action_accuracy": float(correct.float().mean()),
        "targeted_action_accuracy": targeted_accuracy,
        "targetless_action_accuracy": targetless_accuracy,
        "stop_recall": float(correct[stop_mask].float().mean()),
        "turn_sign_accuracy": float(turn_sign.float().mean()),
        "argument_mae": float(
            torch.mean(
                torch.abs(
                    argument_predictions[numeric_mask]
                    - samples.base.argument_targets[numeric_mask]
                )
            )
        ),
        "collision_risk_accuracy": float(
            (risk_prediction == risk_target).float().mean()
        ),
        "unsafe_raw_motion_count": unsafe_raw,
        "unsafe_after_exact_mask_count": unsafe_after_exact_mask,
        "unsafe_motion_rejection": (
            1.0 if unsafe_raw == 0 else 1.0 - unsafe_after_exact_mask / unsafe_raw
        ),
        "predicted_action_counts": {
            name: int(counts.get(name, 0)) for name in ACTION_NAMES
        },
        "by_family": family_metrics,
        "clearance_mode": clearance_mode,
    }


def _wrong_target_samples(
    samples: PreparedSamplesV4, *, room_size_m: list[float]
) -> PreparedSamplesV4:
    base = samples.base
    targeted_indices = torch.nonzero(base.target_available, as_tuple=True)[0]
    rolled_targets = base.target_xyz_m.clone()
    rolled_targets[targeted_indices] = torch.roll(
        base.target_xyz_m[targeted_indices], shifts=1, dims=0
    )
    states = grounded_target_state(
        rolled_targets,
        base.state_features,
        base.target_available.float(),
        room_size_m=room_size_m,
    )
    return replace(
        samples,
        base=replace(base, target_xyz_m=rolled_targets, target_states=states),
    )


@torch.inference_mode()
def _output_change_metrics(
    controller: ClearanceAwareNavigationControllerV4,
    prefixes: torch.Tensor,
    primary: PreparedSamplesV4,
    control: PreparedSamplesV4,
    *,
    batch_size: int,
) -> dict[str, Any]:
    changed_actions = 0
    changed_turn_signs = 0
    targeted_total = 0
    argument_change = 0.0
    for indices in _scene_batches(primary, batch_size, generator=None):
        primary_logits, primary_arguments, _ = _forward(
            controller, prefixes, primary, indices
        )
        control_logits, control_arguments, _ = _forward(
            controller, prefixes, control, indices
        )
        primary_actions = torch.argmax(primary_logits, dim=-1)
        control_actions = torch.argmax(control_logits, dim=-1)
        targeted = primary.base.target_available[indices]
        targets = primary.base.action_targets[indices]
        primary_selected = primary_arguments.gather(1, targets[:, None]).squeeze(1)
        control_selected = control_arguments.gather(1, targets[:, None]).squeeze(1)
        changed_actions += int(
            (primary_actions[targeted] != control_actions[targeted]).sum()
        )
        targeted_total += int(targeted.sum())
        argument_change += float(
            torch.abs(primary_selected[targeted] - control_selected[targeted]).sum()
        )
        turns = targeted & (targets == ACTION_TO_INDEX["turn"])
        changed_turn_signs += int(
            (
                torch.sign(primary_selected[turns])
                != torch.sign(control_selected[turns])
            ).sum()
        )
    return {
        "targeted_sample_count": targeted_total,
        "changed_targeted_action_count": changed_actions,
        "changed_targeted_action_fraction": changed_actions / targeted_total,
        "mean_absolute_targeted_argument_change": argument_change / targeted_total,
        "changed_turn_argument_sign_count": changed_turn_signs,
    }


def _control_measurements(
    controller: ClearanceAwareNavigationControllerV4,
    data: PreparedTrainingDataV4,
    *,
    batch_size: int,
    room_size_m: list[float],
    max_turn_degrees: float,
    max_move_m: float,
) -> dict[str, Any]:
    validation = data.validation
    kwargs = {
        "batch_size": batch_size,
        "max_turn_degrees": max_turn_degrees,
        "max_move_m": max_move_m,
    }
    primary = evaluate_prepared_v4(
        controller, data.base.prefixes, validation, **kwargs
    )
    shuffled = evaluate_prepared_v4(
        controller,
        data.base.prefixes,
        validation,
        clearance_mode="shuffled",
        **kwargs,
    )
    zero = evaluate_prepared_v4(
        controller,
        data.base.prefixes,
        validation,
        clearance_mode="zero",
        **kwargs,
    )
    wrong_target = _wrong_target_samples(validation, room_size_m=room_size_m)
    wrong_target_metrics = evaluate_prepared_v4(
        controller, data.base.prefixes, wrong_target, **kwargs
    )
    zero_target = replace(
        validation,
        base=replace(
            validation.base,
            target_xyz_m=torch.zeros_like(validation.base.target_xyz_m),
            target_available=torch.zeros_like(validation.base.target_available),
            target_states=torch.zeros_like(validation.base.target_states),
        ),
    )
    zero_target_metrics = evaluate_prepared_v4(
        controller, data.base.prefixes, zero_target, **kwargs
    )
    families = ("obstacle", "update_after_scan")
    primary_family = sum(primary["by_family"][name]["action_accuracy"] for name in families) / 2
    shuffled_family = sum(
        shuffled["by_family"][name]["action_accuracy"] for name in families
    ) / 2
    return {
        "schema": "semantic_3d_chat.navigation_policy_v4_causal_controls.v4",
        "held_out_scenes_only": True,
        "conditions": {
            "primary": primary,
            "shuffled_clearance_state": shuffled,
            "zero_clearance_state": zero,
            "wrong_target_state": wrong_target_metrics,
            "zero_target_state": zero_target_metrics,
        },
        "shuffled_clearance_obstacle_update_accuracy_drop": (
            primary_family - shuffled_family
        ),
        "targeted_accuracy_deltas_from_primary": {
            "wrong_target_state": primary["targeted_action_accuracy"]
            - wrong_target_metrics["targeted_action_accuracy"],
            "zero_target_state": primary["targeted_action_accuracy"]
            - zero_target_metrics["targeted_action_accuracy"],
        },
        "turn_sign_accuracy_deltas_from_primary": {
            "wrong_target_state": primary["turn_sign_accuracy"]
            - wrong_target_metrics["turn_sign_accuracy"],
            "zero_target_state": primary["turn_sign_accuracy"]
            - zero_target_metrics["turn_sign_accuracy"],
        },
        "wrong_target_output_change": _output_change_metrics(
            controller,
            data.base.prefixes,
            validation,
            wrong_target,
            batch_size=batch_size,
        ),
        "oracle_inputs_used_by_runtime": False,
        "environmental_text_inputs_at_runtime": [],
    }


def train_navigation_policy_v4(
    config: dict[str, Any],
    *,
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    settings = validate_navigation_policy_v4_settings(config)
    dataset_path = _rooted(dataset or str(settings["source_trace_dataset"]))
    checkpoint_path = _rooted(checkpoint or str(settings["checkpoint_output"]))
    report_path = _rooted(
        metrics_path or "reports/gemma4/metrics/navigation_policy_v4_training.json"
    )
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError(
            "The single preregistered V4 arm already has a checkpoint or training report"
        )
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manifest, data = prepare_navigation_policy_v4_data(config, dataset_path)
    preregistration = authenticate_navigation_policy_v41_preregistration(
        _rooted(str(settings["preregistration"])),
        config,
        source_v3_dataset_sha256=str(manifest["dataset_sha256"]),
        v4_dataset_sha256=data.dataset_sha256,
        map_sha256=data.map_sha256,
    )
    v3_controller, v3_metadata = load_navigation_policy_v3_checkpoint(
        _rooted(str(settings["v3_checkpoint"])),
        expected_hidden_size=int(settings["hidden_size"]),
        expected_model_id=str(config["language"]["model_id"]),
        expected_model_revision=str(config["language"]["revision"]),
        device="cpu",
    )
    controller = ClearanceAwareNavigationControllerV4(
        int(settings["hidden_size"]), model_dim=int(settings["model_dim"])
    ).cpu()
    controller.initialize_from_v3(v3_controller)
    controller.freeze_v3_base()
    trainable = [parameter for parameter in controller.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    counts = torch.bincount(
        data.train.base.action_targets, minlength=len(ACTION_NAMES)
    ).float()
    class_weights = torch.rsqrt(counts)
    class_weights /= class_weights.mean()
    risk_positive = data.train.collision_targets.sum(dim=0)
    risk_negative = len(data.train) - risk_positive
    risk_positive_weight = risk_negative / torch.clamp(risk_positive, min=1.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    started = time.monotonic()
    for epoch in range(1, int(settings["epochs"]) + 1):
        controller.train()
        controller.base.eval()
        total_loss = 0.0
        total_samples = 0
        for indices in _scene_batches(
            data.train, int(settings["batch_size"]), generator=generator
        ):
            logits, argument_heads, risk_logits = _forward(
                controller, data.base.prefixes, data.train, indices
            )
            targets = data.train.base.action_targets[indices]
            predicted_arguments = argument_heads.gather(1, targets[:, None]).squeeze(1)
            classification = torch.nn.functional.cross_entropy(
                logits, targets, weight=class_weights
            )
            numeric = _numeric_action_mask(targets)
            argument_loss = torch.nn.functional.smooth_l1_loss(
                predicted_arguments[numeric],
                data.train.base.argument_targets[indices][numeric],
            )
            turn = targets == ACTION_TO_INDEX["turn"]
            target_sign = torch.sign(data.train.base.argument_targets[indices][turn])
            nonzero = target_sign != 0
            sign_loss = (
                torch.relu(
                    0.10
                    - target_sign[nonzero]
                    * predicted_arguments[turn][nonzero]
                ).mean()
                if bool(nonzero.any())
                else logits.sum() * 0.0
            )
            risk_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                risk_logits,
                data.train.collision_targets[indices],
                pos_weight=risk_positive_weight,
            )
            shuffled = torch.roll(data.train.clearance_states[indices], shifts=1, dims=0)
            shuffled_logits, shuffled_arguments, _ = _forward(
                controller,
                data.base.prefixes,
                data.train,
                indices,
                clearance_override=shuffled,
            )
            output_change = torch.mean(torch.abs(logits - shuffled_logits)) + torch.mean(
                torch.abs(argument_heads - shuffled_arguments)
            )
            clearance_change = torch.relu(
                float(settings["clearance_change_margin"]) - output_change
            )
            loss = (
                classification
                + float(settings["argument_loss_weight"]) * argument_loss
                + float(settings["turn_sign_loss_weight"]) * sign_loss
                + float(settings["collision_risk_loss_weight"]) * risk_loss
                + float(settings["clearance_change_loss_weight"]) * clearance_change
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(settings["gradient_clip_norm"]))
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total_samples += len(indices)
        validation = evaluate_prepared_v4(
            controller,
            data.base.prefixes,
            data.validation,
            batch_size=int(settings["batch_size"]),
            max_turn_degrees=float(config["robot"]["max_turn_degrees"]),
            max_move_m=float(config["robot"]["max_move_m"]),
        )
        score = (
            validation["action_accuracy"]
            + 0.20 * validation["turn_sign_accuracy"]
            + 0.10 * validation["stop_recall"]
            + 0.10 * validation["collision_risk_accuracy"]
            - 0.10 * validation["argument_mae"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, total_samples),
                "validation_action_accuracy": validation["action_accuracy"],
                "validation_update_after_scan_accuracy": validation["by_family"][
                    "update_after_scan"
                ]["action_accuracy"],
                "validation_stop_recall": validation["stop_recall"],
                "validation_turn_sign_accuracy": validation["turn_sign_accuracy"],
                "validation_argument_mae": validation["argument_mae"],
                "validation_collision_risk_accuracy": validation[
                    "collision_risk_accuracy"
                ],
                "selector_score": score,
            }
        )
        if score > best_score + 1e-9:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in controller.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(settings["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("V4 training produced no finite candidate")
    controller.load_state_dict(best_state, strict=True)
    train_metrics = evaluate_prepared_v4(
        controller,
        data.base.prefixes,
        data.train,
        batch_size=int(settings["batch_size"]),
        max_turn_degrees=float(config["robot"]["max_turn_degrees"]),
        max_move_m=float(config["robot"]["max_move_m"]),
    )
    controls = _control_measurements(
        controller,
        data,
        batch_size=int(settings["batch_size"]),
        room_size_m=list(config["scene"]["room_size_m"]),
        max_turn_degrees=float(config["robot"]["max_turn_degrees"]),
        max_move_m=float(config["robot"]["max_move_m"]),
    )
    validation_metrics = controls["conditions"]["primary"]
    gates = {
        "validation_action_accuracy": validation_metrics["action_accuracy"]
        >= float(settings["minimum_validation_action_accuracy"]),
        "validation_update_after_scan_accuracy": validation_metrics["by_family"][
            "update_after_scan"
        ]["action_accuracy"]
        >= float(settings["minimum_validation_update_after_scan_accuracy"]),
        "validation_stop_recall": validation_metrics["stop_recall"]
        >= float(settings["minimum_validation_stop_recall"]),
        "validation_turn_sign_accuracy": validation_metrics["turn_sign_accuracy"]
        >= float(settings["minimum_validation_turn_sign_accuracy"]),
        "validation_argument_mae": validation_metrics["argument_mae"]
        <= float(settings["maximum_validation_argument_mae"]),
        "unsafe_motion_rejection": validation_metrics["unsafe_motion_rejection"]
        >= float(settings["minimum_unsafe_motion_rejection"]),
        "collision_risk_accuracy": validation_metrics["collision_risk_accuracy"]
        >= float(settings["minimum_collision_risk_accuracy"]),
        "shuffled_clearance_family_drop": controls[
            "shuffled_clearance_obstacle_update_accuracy_drop"
        ]
        >= float(settings["minimum_shuffled_clearance_family_drop"]),
        "zero_target_targeted_accuracy_drop": controls[
            "targeted_accuracy_deltas_from_primary"
        ]["zero_target_state"]
        >= float(settings["minimum_zero_target_targeted_accuracy_drop"]),
        "wrong_target_turn_sign_drop": controls[
            "turn_sign_accuracy_deltas_from_primary"
        ]["wrong_target_state"]
        >= float(settings["minimum_wrong_target_turn_sign_drop"]),
        "scene_splits_disjoint": manifest["scene_splits_disjoint"] is True,
        "train_scene_count_exact": manifest["train_scene_count"] == 14,
        "validation_scene_count_exact": manifest["validation_scene_count"] == 8,
        "numeric_clearance_only": True,
    }
    accepted = all(gates.values())
    checkpoint_metadata: dict[str, Any] | None = None
    if accepted:
        checkpoint_metadata = save_navigation_policy_v4_checkpoint(
            checkpoint_path,
            controller,
            runtime_metadata={
                "scene_token_count": int(settings["scene_token_count"]),
                "robot_token_count": int(settings["robot_token_count"]),
                "model_id": str(config["language"]["model_id"]),
                "model_revision": str(config["language"]["revision"]),
                "max_turn_degrees": float(config["robot"]["max_turn_degrees"]),
                "max_move_m": float(config["robot"]["max_move_m"]),
                "room_size_m": [float(value) for value in config["scene"]["room_size_m"]],
                "grounding_feature_start": 1536,
                "grounding_feature_dim": 1536,
                "task_trained": True,
                "training_dataset_sha256": data.dataset_sha256,
                "preregistration_sha256": preregistration["sha256"],
                "preregistered_single_arm": True,
                "v3_initialization_weights_sha256": str(v3_metadata["weights_sha256"]),
                "train_scene_count": 14,
                "validation_scene_count": 8,
                "scene_splits_disjoint": True,
                "complete_scene_prefix_required": True,
                "question_independent_static_scene_prefix_required": True,
                "every_scene_token_processed": True,
                "numeric_robot_tokens_required": True,
                "continuous_semantic_grounding_required": True,
                "all_map_voxels_scored_for_grounding": True,
                "numeric_clearance_state_required": True,
                "clearance_from_sanitized_geometry_only": True,
                "exact_collision_mask_required": True,
                "unsafe_motion_fallback": "highest_safe_nonterminal_action",
                "query_dependent_grounding_navigation_only": True,
                "environmental_text_inputs": [],
                "oracle_inputs_at_runtime": False,
                "runtime_required_files": [
                    "policy.safetensors",
                    "runtime_metadata.json",
                ],
                "collision_interlock_required": True,
            },
        )
    result = {
        "schema": (
            "semantic_3d_chat.navigation_policy_v4_1_training_result.v1"
            if settings.get("protocol_version") == "v4.1"
            else "semantic_3d_chat.navigation_policy_v4_training_result.v4"
        ),
        "protocol_version": str(settings.get("protocol_version", "v4")),
        "status": "accepted" if accepted else "rejected",
        "checkpoint_written": accepted,
        "checkpoint": str(checkpoint_path) if accepted else None,
        "dataset_sha256": data.dataset_sha256,
        "source_v3_dataset_sha256": manifest["dataset_sha256"],
        "source_v3_weights_sha256": v3_metadata["weights_sha256"],
        "preregistration": preregistration,
        "device": "cpu",
        "elapsed_seconds": time.monotonic() - started,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train": train_metrics,
        "validation": validation_metrics,
        "controls": controls,
        "gates": gates,
        "thresholds": {
            name: settings[name]
            for name in (
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
        },
        "history": history,
        "runtime_contract": (
            None
            if checkpoint_metadata is None
            else {
                key: value
                for key, value in checkpoint_metadata.items()
                if key != "checkpoint"
            }
        ),
        "single_preregistered_arm": True,
        "v3_base_frozen": True,
        "train_scene_gradients_only": True,
        "counterfactual_collision_targets_from_train_numeric_maps": True,
        "oracle_inputs_at_runtime": False,
        "environmental_text_inputs_at_runtime": [],
    }
    _atomic_json(report_path, result)
    return result


def evaluate_navigation_policy_v4(
    config: dict[str, Any],
    *,
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate an accepted V4 checkpoint on the sealed held-out split."""

    settings = validate_navigation_policy_v4_settings(config)
    dataset_path = _rooted(dataset or str(settings["source_trace_dataset"]))
    checkpoint_path = _rooted(checkpoint or str(settings["checkpoint_output"]))
    manifest, data = prepare_navigation_policy_v4_data(config, dataset_path)
    controller, metadata = load_navigation_policy_v4_checkpoint(
        checkpoint_path,
        expected_hidden_size=int(settings["hidden_size"]),
        expected_model_id=str(config["language"]["model_id"]),
        expected_model_revision=str(config["language"]["revision"]),
        device="cpu",
    )
    preregistration_path = _rooted(str(settings["preregistration"]))
    if (
        metadata["training_dataset_sha256"] != data.dataset_sha256
        or metadata["preregistration_sha256"] != _sha256(preregistration_path)
    ):
        raise ValueError("V4 checkpoint differs from its sealed evaluation inputs")
    controls = _control_measurements(
        controller,
        data,
        batch_size=int(settings["batch_size"]),
        room_size_m=list(config["scene"]["room_size_m"]),
        max_turn_degrees=float(config["robot"]["max_turn_degrees"]),
        max_move_m=float(config["robot"]["max_move_m"]),
    )
    result = {
        "schema": "semantic_3d_chat.navigation_policy_v4_offline_evaluation.v4",
        "dataset_sha256": data.dataset_sha256,
        "source_v3_dataset_sha256": manifest["dataset_sha256"],
        "checkpoint_weights_sha256": metadata["weights_sha256"],
        "preregistration_sha256": metadata["preregistration_sha256"],
        "validation": controls["conditions"]["primary"],
        "controls": controls,
        "held_out_scenes_only": True,
        "oracle_inputs_used_by_runtime": False,
        "environmental_text_inputs_at_runtime": [],
    }
    if metrics_path is not None:
        destination = _rooted(metrics_path)
        if destination.exists():
            raise FileExistsError(f"V4 evaluation report already exists: {destination}")
        _atomic_json(destination, result)
    return result


__all__ = [
    "PreparedSamplesV4",
    "PreparedTrainingDataV4",
    "evaluate_navigation_policy_v4",
    "evaluate_prepared_v4",
    "prepare_navigation_policy_v4_data",
    "train_navigation_policy_v4",
    "validate_navigation_policy_v4_settings",
]

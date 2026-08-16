"""Train and causally evaluate continuous-semantic navigation-policy V3."""

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
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    load_category_embeddings_selective,
    resolve_local_snapshot,
)
from semantic_3d_chat.robot.navigation_policy import ACTION_NAMES, ACTION_TO_INDEX
from semantic_3d_chat.robot.navigation_policy_v3 import (
    GroundedContinuousNavigationControllerV3,
    grounded_target_state,
    load_navigation_policy_v3_checkpoint,
    save_navigation_policy_v3_checkpoint,
)
from semantic_3d_chat.robot.state_checkpoint import load_robot_state_checkpoint
from semantic_3d_chat.training.navigation_target_trace_v3 import (
    load_navigation_target_trace_v3,
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
class PreparedSamplesV3:
    scene_indices: torch.Tensor
    robot_tokens: torch.Tensor
    instruction_embeddings: torch.Tensor
    state_features: torch.Tensor
    target_xyz_m: torch.Tensor
    target_available: torch.Tensor
    target_states: torch.Tensor
    action_targets: torch.Tensor
    argument_targets: torch.Tensor
    families: tuple[str, ...]
    scene_ids: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.action_targets.shape[0])


@dataclass(frozen=True)
class PreparedTrainingDataV3:
    prefixes: torch.Tensor
    prefix_scene_ids: tuple[str, ...]
    train: PreparedSamplesV3
    validation: PreparedSamplesV3
    selective_embedding_metadata: dict[str, Any]
    robot_state_checkpoint_sha256: str


def validate_navigation_policy_v3_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("navigation_policy_v3")
    if not isinstance(settings, dict):
        raise TypeError("Config has no navigation_policy_v3 mapping")
    for name in (
        "hidden_size",
        "model_dim",
        "scene_token_count",
        "robot_token_count",
        "batch_size",
        "epochs",
        "early_stopping_patience",
        "seed",
    ):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"navigation_policy_v3.{name} must be a positive integer")
    for name in (
        "learning_rate",
        "weight_decay",
        "argument_loss_weight",
        "turn_sign_loss_weight",
        "zero_target_stop_loss_weight",
        "target_change_loss_weight",
        "target_change_margin",
        "gradient_clip_norm",
        "minimum_validation_action_accuracy",
        "minimum_validation_stop_recall",
        "minimum_validation_turn_sign_accuracy",
        "maximum_validation_argument_mae",
        "minimum_zero_target_targeted_accuracy_drop",
        "minimum_wrong_target_turn_sign_drop",
    ):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"navigation_policy_v3.{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"navigation_policy_v3.{name} must be finite and nonnegative")
    if settings.get("device", "cpu") != "cpu":
        raise ValueError("Navigation-policy V3 training is deliberately CPU-only")
    train_scenes = settings.get("train_scene_ids")
    validation_scenes = settings.get("validation_scene_ids")
    if (
        not isinstance(train_scenes, list)
        or not isinstance(validation_scenes, list)
        or not train_scenes
        or not validation_scenes
        or set(train_scenes) & set(validation_scenes)
    ):
        raise ValueError("V3 navigation scene splits must be nonempty and disjoint")
    return settings


def _load_prefixes(
    prefix_root: Path,
    scene_ids: list[str],
    *,
    expected_tokens: int,
    expected_hidden: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    manifest = json.loads((prefix_root / "manifest.json").read_text(encoding="utf-8"))
    scenes = manifest.get("scenes") if isinstance(manifest, dict) else None
    if (
        not isinstance(scenes, dict)
        or manifest.get("complete_scene_prefixes") is not True
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_scene_retrieval") is not False
        or manifest.get("environmental_text_inputs") != []
    ):
        raise ValueError("V3 scene-prefix cache contract differs")
    tensors: list[torch.Tensor] = []
    for scene_id in scene_ids:
        entry = scenes.get(scene_id)
        if not isinstance(entry, dict):
            raise FileNotFoundError(f"No V3 cached scene prefix for {scene_id}")
        path = prefix_root / str(entry.get("filename"))
        if _sha256(path) != entry.get("file_sha256"):
            raise ValueError(f"V3 cached scene prefix changed for {scene_id}")
        state = load_file(str(path), device="cpu")
        if set(state) != {"scene_prefix"}:
            raise ValueError("V3 cached scene prefix tensor keys differ")
        prefix = state["scene_prefix"]
        if prefix.shape != (1, expected_tokens, expected_hidden):
            raise ValueError("V3 cached scene prefix shape differs")
        if not torch.isfinite(prefix).all():
            raise ValueError("V3 cached scene prefix is nonfinite")
        tensors.append(prefix[0].float().contiguous())
    return torch.stack(tensors), tuple(scene_ids)


def _instruction_embeddings(
    config: dict[str, Any], instructions: list[str], *, expected_dim: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model_id = str(config["language"]["model_id"])
    revision = str(config["language"]["revision"])
    snapshot = resolve_local_snapshot(model_id, revision)
    unique = list(dict.fromkeys(instructions))
    values, metadata = load_category_embeddings_selective(
        snapshot, unique, expected_dim=expected_dim
    )
    return {
        instruction: torch.from_numpy(values[index]).float()
        for index, instruction in enumerate(unique)
    }, metadata


@torch.inference_mode()
def prepare_navigation_policy_v3_data(
    config: dict[str, Any], dataset_path: Path
) -> tuple[dict[str, Any], PreparedTrainingDataV3]:
    settings = validate_navigation_policy_v3_settings(config)
    manifest, rows = load_navigation_target_trace_v3(dataset_path)
    train_scenes = list(settings["train_scene_ids"])
    validation_scenes = list(settings["validation_scene_ids"])
    if (
        manifest["train_scene_ids"] != train_scenes
        or manifest["validation_scene_ids"] != validation_scenes
    ):
        raise ValueError("V3 dataset scene split differs from config")
    scene_ids = [*train_scenes, *validation_scenes]
    prefixes, prefix_scene_ids = _load_prefixes(
        _rooted(str(settings["prefix_cache_root"])),
        scene_ids,
        expected_tokens=int(settings["scene_token_count"]),
        expected_hidden=int(settings["hidden_size"]),
    )
    scene_index = {scene: index for index, scene in enumerate(prefix_scene_ids)}
    embeddings, selective = _instruction_embeddings(
        config,
        [str(row["instruction"]) for row in rows],
        expected_dim=int(settings["hidden_size"]),
    )
    state_encoder, state_hash, state_metadata = load_robot_state_checkpoint(
        _rooted(str(settings["robot_state_checkpoint"])),
        expected_output_dim=int(settings["hidden_size"]),
        device="cpu",
    )
    if (
        state_metadata.get("numeric_inputs_only") is not True
        or state_metadata.get("token_count") != settings["robot_token_count"]
    ):
        raise ValueError("V3 robot-state checkpoint contract differs")

    room_size = config["scene"]["room_size_m"]

    def prepare(split: str) -> PreparedSamplesV3:
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            raise ValueError(f"V3 navigation split is empty: {split}")
        states = torch.tensor([row["state_features"] for row in selected], dtype=torch.float32)
        targets = torch.tensor(
            [row["oracle_target_xyz_m"] for row in selected], dtype=torch.float32
        )
        available = torch.tensor(
            [bool(row["target_state_available"]) for row in selected], dtype=torch.bool
        )
        target_states = grounded_target_state(
            targets, states, available.float(), room_size_m=room_size
        )
        return PreparedSamplesV3(
            scene_indices=torch.tensor(
                [scene_index[str(row["scene_id"])] for row in selected], dtype=torch.long
            ),
            robot_tokens=state_encoder(states).float().contiguous(),
            instruction_embeddings=torch.stack(
                [embeddings[str(row["instruction"])] for row in selected]
            ),
            state_features=states,
            target_xyz_m=targets,
            target_available=available,
            target_states=target_states,
            action_targets=torch.tensor(
                [int(row["action_index"]) for row in selected], dtype=torch.long
            ),
            argument_targets=torch.tensor(
                [float(row["argument_target_normalized"]) for row in selected],
                dtype=torch.float32,
            ),
            families=tuple(str(row["family"]) for row in selected),
            scene_ids=tuple(str(row["scene_id"]) for row in selected),
        )

    return manifest, PreparedTrainingDataV3(
        prefixes=prefixes,
        prefix_scene_ids=prefix_scene_ids,
        train=prepare("train"),
        validation=prepare("validation"),
        selective_embedding_metadata=selective,
        robot_state_checkpoint_sha256=state_hash,
    )


def _scene_batches(
    samples: PreparedSamplesV3,
    batch_size: int,
    *,
    generator: torch.Generator | None,
) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    scene_values = torch.unique(samples.scene_indices, sorted=True)
    if generator is not None:
        scene_values = scene_values[torch.randperm(len(scene_values), generator=generator)]
    for scene_index in scene_values:
        indices = torch.nonzero(samples.scene_indices == scene_index, as_tuple=True)[0]
        if generator is not None:
            indices = indices[torch.randperm(len(indices), generator=generator)]
        batches.extend(
            indices[start : start + batch_size] for start in range(0, len(indices), batch_size)
        )
    return batches


def _numeric_action_mask(targets: torch.Tensor) -> torch.Tensor:
    return (
        (targets == ACTION_TO_INDEX["turn"])
        | (targets == ACTION_TO_INDEX["move_forward"])
        | (targets == ACTION_TO_INDEX["move_backward"])
    )


def _forward(
    controller: GroundedContinuousNavigationControllerV3,
    prefixes: torch.Tensor,
    samples: PreparedSamplesV3,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_scenes = samples.scene_indices[indices]
    unique_scenes, inverse = torch.unique(selected_scenes, sorted=True, return_inverse=True)
    return controller(
        prefixes[unique_scenes],
        samples.robot_tokens[indices],
        samples.instruction_embeddings[indices],
        samples.target_states[indices],
        scene_batch_indices=inverse,
    )


@torch.inference_mode()
def evaluate_prepared_v3(
    controller: GroundedContinuousNavigationControllerV3,
    prefixes: torch.Tensor,
    samples: PreparedSamplesV3,
    *,
    batch_size: int,
) -> dict[str, Any]:
    controller.eval()
    predicted = torch.empty(len(samples), dtype=torch.long)
    argument_predictions = torch.empty(len(samples), dtype=torch.float32)
    for indices in _scene_batches(samples, batch_size, generator=None):
        logits, arguments = _forward(controller, prefixes, samples, indices)
        predicted[indices] = torch.argmax(logits, dim=-1).cpu()
        argument_predictions[indices] = (
            arguments.gather(1, samples.action_targets[indices].unsqueeze(1)).squeeze(1).cpu()
        )
    target = samples.action_targets
    correct = predicted == target
    stop_mask = target == ACTION_TO_INDEX["stop"]
    turn_mask = target == ACTION_TO_INDEX["turn"]
    numeric_mask = _numeric_action_mask(target)
    targeted = samples.target_available
    turn_sign = torch.sign(argument_predictions[turn_mask]) == torch.sign(
        samples.argument_targets[turn_mask]
    )
    turn_sign &= predicted[turn_mask] == target[turn_mask]
    family_metrics: dict[str, Any] = {}
    for family in sorted(set(samples.families)):
        mask = torch.tensor([value == family for value in samples.families])
        family_metrics[family] = {
            "sample_count": int(mask.sum()),
            "action_accuracy": float(correct[mask].float().mean()),
        }
    counts = Counter(ACTION_NAMES[int(value)] for value in predicted.tolist())
    targeted_accuracy = float(correct[targeted].float().mean()) if bool(targeted.any()) else 0.0
    targetless_accuracy = (
        float(correct[~targeted].float().mean()) if bool((~targeted).any()) else 0.0
    )
    return {
        "sample_count": len(samples),
        "scene_count": len(set(samples.scene_ids)),
        "targeted_sample_count": int(targeted.sum()),
        "action_accuracy": float(correct.float().mean()),
        "targeted_action_accuracy": targeted_accuracy,
        "targetless_action_accuracy": targetless_accuracy,
        "stop_recall": float(correct[stop_mask].float().mean()),
        "turn_sign_accuracy": float(turn_sign.float().mean()),
        "argument_mae": float(
            torch.mean(
                torch.abs(
                    argument_predictions[numeric_mask] - samples.argument_targets[numeric_mask]
                )
            )
        ),
        "predicted_action_counts": {name: int(counts.get(name, 0)) for name in ACTION_NAMES},
        "by_family": family_metrics,
    }


def _wrong_target_samples(
    samples: PreparedSamplesV3, *, room_size_m: list[float]
) -> PreparedSamplesV3:
    targeted_indices = torch.nonzero(samples.target_available, as_tuple=True)[0]
    if len(targeted_indices) < 2:
        raise ValueError("V3 wrong-target control requires two targeted samples")
    rolled_targets = samples.target_xyz_m.clone()
    rolled_targets[targeted_indices] = torch.roll(
        samples.target_xyz_m[targeted_indices], shifts=1, dims=0
    )
    states = grounded_target_state(
        rolled_targets,
        samples.state_features,
        samples.target_available.float(),
        room_size_m=room_size_m,
    )
    return replace(samples, target_xyz_m=rolled_targets, target_states=states)


@torch.inference_mode()
def _output_change_metrics(
    controller: GroundedContinuousNavigationControllerV3,
    prefixes: torch.Tensor,
    primary: PreparedSamplesV3,
    control: PreparedSamplesV3,
    *,
    batch_size: int,
) -> dict[str, Any]:
    if len(primary) != len(control):
        raise ValueError("V3 output-change control sample inventories differ")
    changed_actions = 0
    changed_turn_signs = 0
    targeted_total = 0
    absolute_argument_change = 0.0
    for indices in _scene_batches(primary, batch_size, generator=None):
        primary_logits, primary_arguments = _forward(controller, prefixes, primary, indices)
        control_logits, control_arguments = _forward(controller, prefixes, control, indices)
        primary_actions = torch.argmax(primary_logits, dim=-1)
        control_actions = torch.argmax(control_logits, dim=-1)
        targeted = primary.target_available[indices]
        action_targets = primary.action_targets[indices]
        primary_selected = primary_arguments.gather(1, action_targets[:, None]).squeeze(1)
        control_selected = control_arguments.gather(1, action_targets[:, None]).squeeze(1)
        changed_actions += int((primary_actions[targeted] != control_actions[targeted]).sum())
        targeted_total += int(targeted.sum())
        absolute_argument_change += float(
            torch.abs(primary_selected[targeted] - control_selected[targeted]).sum()
        )
        turns = targeted & (action_targets == ACTION_TO_INDEX["turn"])
        changed_turn_signs += int(
            (torch.sign(primary_selected[turns]) != torch.sign(control_selected[turns])).sum()
        )
    return {
        "targeted_sample_count": targeted_total,
        "changed_targeted_action_count": changed_actions,
        "changed_targeted_action_fraction": changed_actions / targeted_total,
        "mean_absolute_targeted_argument_change": absolute_argument_change / targeted_total,
        "changed_turn_argument_sign_count": changed_turn_signs,
    }


def _control_measurements(
    controller: GroundedContinuousNavigationControllerV3,
    data: PreparedTrainingDataV3,
    *,
    batch_size: int,
    room_size_m: list[float],
) -> dict[str, Any]:
    validation = data.validation
    primary = evaluate_prepared_v3(controller, data.prefixes, validation, batch_size=batch_size)
    wrong_target_samples = _wrong_target_samples(validation, room_size_m=room_size_m)
    wrong_target_output_change = _output_change_metrics(
        controller,
        data.prefixes,
        validation,
        wrong_target_samples,
        batch_size=batch_size,
    )
    zero_target_samples = replace(
        validation,
        target_xyz_m=torch.zeros_like(validation.target_xyz_m),
        target_available=torch.zeros_like(validation.target_available),
        target_states=torch.zeros_like(validation.target_states),
    )
    validation_scene_indices = sorted(set(validation.scene_indices.tolist()))
    if len(validation_scene_indices) < 2:
        raise ValueError("V3 scene controls require two held-out scenes")
    rolled_scene_indices = validation_scene_indices[1:] + validation_scene_indices[:1]
    remap = dict(zip(validation_scene_indices, rolled_scene_indices, strict=True))
    wrong_scene_samples = replace(
        validation,
        scene_indices=torch.tensor(
            [remap[int(value)] for value in validation.scene_indices.tolist()],
            dtype=torch.long,
        ),
    )
    conditions = {
        "primary": primary,
        "wrong_target_state": evaluate_prepared_v3(
            controller, data.prefixes, wrong_target_samples, batch_size=batch_size
        ),
        "zero_target_state": evaluate_prepared_v3(
            controller, data.prefixes, zero_target_samples, batch_size=batch_size
        ),
        "wrong_scene_prefix": evaluate_prepared_v3(
            controller, data.prefixes, wrong_scene_samples, batch_size=batch_size
        ),
        "zero_scene_prefix": evaluate_prepared_v3(
            controller,
            torch.zeros_like(data.prefixes),
            validation,
            batch_size=batch_size,
        ),
    }
    return {
        "schema": "semantic_3d_chat.navigation_policy_v3_causal_controls.v3",
        "held_out_scenes_only": True,
        "conditions": conditions,
        "action_accuracy_deltas_from_primary": {
            name: primary["action_accuracy"] - metrics["action_accuracy"]
            for name, metrics in conditions.items()
            if name != "primary"
        },
        "targeted_accuracy_deltas_from_primary": {
            name: primary["targeted_action_accuracy"] - metrics["targeted_action_accuracy"]
            for name, metrics in conditions.items()
            if name != "primary"
        },
        "turn_sign_accuracy_deltas_from_primary": {
            name: primary["turn_sign_accuracy"] - metrics["turn_sign_accuracy"]
            for name, metrics in conditions.items()
            if name != "primary"
        },
        "wrong_target_derangement": "cyclic_numeric_xyz_then_recompute_robot_relative_state",
        "wrong_target_output_change": wrong_target_output_change,
        "oracle_inputs_used_by_runtime": False,
        "environmental_text_inputs_at_runtime": [],
    }


def train_navigation_policy_v3(
    config: dict[str, Any],
    *,
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    settings = validate_navigation_policy_v3_settings(config)
    dataset_path = _rooted(dataset or str(settings["trace_output"]))
    checkpoint_path = _rooted(checkpoint or str(settings["checkpoint_output"]))
    report_path = _rooted(
        metrics_path or "reports/gemma4/metrics/navigation_policy_v3_training.json"
    )
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manifest, data = prepare_navigation_policy_v3_data(config, dataset_path)
    controller = GroundedContinuousNavigationControllerV3(
        int(settings["hidden_size"]), model_dim=int(settings["model_dim"])
    ).cpu()
    counts = torch.bincount(data.train.action_targets, minlength=len(ACTION_NAMES)).float()
    if torch.any(counts == 0):
        raise ValueError("V3 train split does not cover every action")
    class_weights = torch.rsqrt(counts)
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    started = time.monotonic()
    for epoch in range(1, int(settings["epochs"]) + 1):
        controller.train()
        total_loss = 0.0
        total_samples = 0
        for indices in _scene_batches(data.train, int(settings["batch_size"]), generator=generator):
            logits, argument_heads = _forward(controller, data.prefixes, data.train, indices)
            targets = data.train.action_targets[indices]
            predicted_arguments = argument_heads.gather(1, targets.unsqueeze(1)).squeeze(1)
            classification = torch.nn.functional.cross_entropy(
                logits, targets, weight=class_weights
            )
            numeric = _numeric_action_mask(targets)
            argument_loss = torch.nn.functional.smooth_l1_loss(
                predicted_arguments[numeric],
                data.train.argument_targets[indices][numeric],
            )
            turn = targets == ACTION_TO_INDEX["turn"]
            target_sign = torch.sign(data.train.argument_targets[indices][turn])
            nonzero = target_sign != 0
            sign_loss = (
                torch.relu(0.10 - target_sign[nonzero] * predicted_arguments[turn][nonzero]).mean()
                if bool(nonzero.any())
                else logits.sum() * 0.0
            )
            loss = (
                classification
                + float(settings["argument_loss_weight"]) * argument_loss
                + float(settings["turn_sign_loss_weight"]) * sign_loss
            )

            targeted = data.train.target_available[indices]
            if bool(targeted.any()):
                target_indices = indices[targeted]
                zero_samples = replace(
                    data.train,
                    target_states=torch.zeros_like(data.train.target_states),
                )
                zero_logits, _zero_arguments = _forward(
                    controller, data.prefixes, zero_samples, target_indices
                )
                stop_targets = torch.full(
                    (len(target_indices),), ACTION_TO_INDEX["stop"], dtype=torch.long
                )
                loss = loss + float(settings["zero_target_stop_loss_weight"]) * (
                    torch.nn.functional.cross_entropy(zero_logits, stop_targets)
                )

                selected_target_states = data.train.target_states[target_indices]
                if len(target_indices) > 1:
                    wrong_states = torch.roll(selected_target_states, shifts=1, dims=0)
                    wrong_samples = replace(data.train, target_states=wrong_states)
                    # ``wrong_samples`` is indexed from zero because it contains only
                    # the selected target rows; construct the controller call directly.
                    selected_scenes = data.train.scene_indices[target_indices]
                    unique_scenes, inverse = torch.unique(
                        selected_scenes, sorted=True, return_inverse=True
                    )
                    wrong_logits, wrong_arguments = controller(
                        data.prefixes[unique_scenes],
                        data.train.robot_tokens[target_indices],
                        data.train.instruction_embeddings[target_indices],
                        wrong_samples.target_states,
                        scene_batch_indices=inverse,
                    )
                    output_change = torch.mean(
                        torch.abs(logits[targeted] - wrong_logits)
                    ) + torch.mean(torch.abs(argument_heads[targeted] - wrong_arguments))
                    loss = loss + float(settings["target_change_loss_weight"]) * torch.relu(
                        float(settings["target_change_margin"]) - output_change
                    )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                controller.parameters(), float(settings["gradient_clip_norm"])
            )
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total_samples += len(indices)

        validation = evaluate_prepared_v3(
            controller,
            data.prefixes,
            data.validation,
            batch_size=int(settings["batch_size"]),
        )
        score = (
            validation["action_accuracy"]
            + 0.25 * validation["turn_sign_accuracy"]
            + 0.10 * validation["stop_recall"]
            - 0.10 * validation["argument_mae"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, total_samples),
                "validation_action_accuracy": validation["action_accuracy"],
                "validation_targeted_action_accuracy": validation["targeted_action_accuracy"],
                "validation_stop_recall": validation["stop_recall"],
                "validation_turn_sign_accuracy": validation["turn_sign_accuracy"],
                "validation_argument_mae": validation["argument_mae"],
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
        raise RuntimeError("V3 training produced no finite candidate")
    controller.load_state_dict(best_state, strict=True)
    train_metrics = evaluate_prepared_v3(
        controller, data.prefixes, data.train, batch_size=int(settings["batch_size"])
    )
    validation_metrics = evaluate_prepared_v3(
        controller,
        data.prefixes,
        data.validation,
        batch_size=int(settings["batch_size"]),
    )
    controls = _control_measurements(
        controller,
        data,
        batch_size=int(settings["batch_size"]),
        room_size_m=list(config["scene"]["room_size_m"]),
    )
    gates = {
        "validation_action_accuracy": validation_metrics["action_accuracy"]
        >= float(settings["minimum_validation_action_accuracy"]),
        "validation_stop_recall": validation_metrics["stop_recall"]
        >= float(settings["minimum_validation_stop_recall"]),
        "validation_turn_sign_accuracy": validation_metrics["turn_sign_accuracy"]
        >= float(settings["minimum_validation_turn_sign_accuracy"]),
        "validation_argument_mae": validation_metrics["argument_mae"]
        <= float(settings["maximum_validation_argument_mae"]),
        "zero_target_targeted_accuracy_drop": controls["targeted_accuracy_deltas_from_primary"][
            "zero_target_state"
        ]
        >= float(settings["minimum_zero_target_targeted_accuracy_drop"]),
        "wrong_target_turn_sign_drop": controls["turn_sign_accuracy_deltas_from_primary"][
            "wrong_target_state"
        ]
        >= float(settings["minimum_wrong_target_turn_sign_drop"]),
        "scene_splits_disjoint": manifest["scene_splits_disjoint"] is True,
        "collision_checked_targets": manifest["collision_checked_movement_targets"] is True,
        "bounded_targets": manifest["bounded_action_targets"] is True,
        "numeric_oracle_targets_training_only": manifest["target_coordinates_training_tree_only"]
        is True,
    }
    accepted = all(gates.values())
    checkpoint_metadata: dict[str, Any] | None = None
    if accepted:
        checkpoint_metadata = save_navigation_policy_v3_checkpoint(
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
                "training_dataset_sha256": str(manifest["dataset_sha256"]),
                "train_scene_count": int(manifest["train_scene_count"]),
                "validation_scene_count": int(manifest["validation_scene_count"]),
                "scene_splits_disjoint": True,
                "complete_scene_prefix_required": True,
                "question_independent_static_scene_prefix_required": True,
                "every_scene_token_processed": True,
                "numeric_robot_tokens_required": True,
                "continuous_semantic_grounding_required": True,
                "all_map_voxels_scored_for_grounding": True,
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
        "schema": "semantic_3d_chat.navigation_policy_v3_training_result.v3",
        "status": "accepted" if accepted else "rejected",
        "checkpoint_written": accepted,
        "checkpoint": str(checkpoint_path) if accepted else None,
        "dataset_sha256": manifest["dataset_sha256"],
        "robot_state_encoder_sha256": data.robot_state_checkpoint_sha256,
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
                "minimum_validation_stop_recall",
                "minimum_validation_turn_sign_accuracy",
                "maximum_validation_argument_mae",
                "minimum_zero_target_targeted_accuracy_drop",
                "minimum_wrong_target_turn_sign_drop",
            )
        },
        "history": history,
        "runtime_contract": (
            None
            if checkpoint_metadata is None
            else {key: value for key, value in checkpoint_metadata.items() if key != "checkpoint"}
        ),
        "local_instruction_embedding": {
            "model_id": config["language"]["model_id"],
            "revision": config["language"]["revision"],
            "method": "mean_local_gemma_input_token_embeddings",
            "selective_row_read": data.selective_embedding_metadata.get("selective_row_read"),
        },
        "continuous_semantic_target_runtime": True,
        "query_dependent_grounding_navigation_only": True,
        "oracle_inputs_at_runtime": False,
        "environmental_text_inputs_at_runtime": [],
    }
    _atomic_json(report_path, result)
    return result


def evaluate_navigation_policy_v3(
    config: dict[str, Any],
    *,
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    settings = validate_navigation_policy_v3_settings(config)
    manifest, data = prepare_navigation_policy_v3_data(
        config, _rooted(dataset or str(settings["trace_output"]))
    )
    controller, metadata = load_navigation_policy_v3_checkpoint(
        checkpoint or str(settings["checkpoint_output"]),
        expected_hidden_size=int(settings["hidden_size"]),
        expected_model_id=str(config["language"]["model_id"]),
        expected_model_revision=str(config["language"]["revision"]),
    )
    if metadata["training_dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("V3 checkpoint was trained on another target trace dataset")
    return {
        "schema": "semantic_3d_chat.navigation_policy_v3_offline_evaluation.v3",
        "checkpoint_weights_sha256": metadata["weights_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "train": evaluate_prepared_v3(
            controller, data.prefixes, data.train, batch_size=int(settings["batch_size"])
        ),
        "validation": evaluate_prepared_v3(
            controller,
            data.prefixes,
            data.validation,
            batch_size=int(settings["batch_size"]),
        ),
        "oracle_inputs_used_by_runtime": False,
        "environmental_text_inputs_at_runtime": [],
    }


def evaluate_navigation_policy_v3_controls(
    config: dict[str, Any],
    *,
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    settings = validate_navigation_policy_v3_settings(config)
    manifest, data = prepare_navigation_policy_v3_data(
        config, _rooted(dataset or str(settings["trace_output"]))
    )
    controller, metadata = load_navigation_policy_v3_checkpoint(
        checkpoint or str(settings["checkpoint_output"]),
        expected_hidden_size=int(settings["hidden_size"]),
        expected_model_id=str(config["language"]["model_id"]),
        expected_model_revision=str(config["language"]["revision"]),
    )
    if metadata["training_dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("V3 checkpoint was trained on another target trace dataset")
    result = _control_measurements(
        controller,
        data,
        batch_size=int(settings["batch_size"]),
        room_size_m=list(config["scene"]["room_size_m"]),
    )
    return {
        **result,
        "checkpoint_weights_sha256": metadata["weights_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
    }


__all__ = [
    "PreparedSamplesV3",
    "evaluate_navigation_policy_v3",
    "evaluate_navigation_policy_v3_controls",
    "evaluate_prepared_v3",
    "prepare_navigation_policy_v3_data",
    "train_navigation_policy_v3",
    "validate_navigation_policy_v3_settings",
]

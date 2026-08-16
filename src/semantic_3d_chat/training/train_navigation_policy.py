"""CPU-friendly supervised training for the continuous navigation controller."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
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
from semantic_3d_chat.robot.navigation_policy import (
    ACTION_NAMES,
    ACTION_TO_INDEX,
    ContinuousNavigationActionController,
    load_navigation_policy_checkpoint,
    save_navigation_policy_checkpoint,
)
from semantic_3d_chat.robot.state_checkpoint import load_robot_state_checkpoint
from semantic_3d_chat.training.navigation_trace_generator import (
    load_navigation_trace_dataset,
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


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


@dataclass(frozen=True)
class _PreparedSamples:
    scene_indices: torch.Tensor
    robot_tokens: torch.Tensor
    instruction_embeddings: torch.Tensor
    action_targets: torch.Tensor
    argument_targets: torch.Tensor
    families: tuple[str, ...]
    scene_ids: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.action_targets.shape[0])


@dataclass(frozen=True)
class _PreparedTrainingData:
    prefixes: torch.Tensor
    prefix_scene_ids: tuple[str, ...]
    train: _PreparedSamples
    validation: _PreparedSamples
    selective_embedding_metadata: dict[str, Any]
    robot_state_checkpoint_sha256: str


def _validate_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("navigation_policy")
    if not isinstance(settings, dict):
        raise TypeError("Config has no navigation_policy mapping")
    integer_fields = (
        "hidden_size",
        "model_dim",
        "scene_token_count",
        "robot_token_count",
        "batch_size",
        "epochs",
        "early_stopping_patience",
        "seed",
    )
    for name in integer_fields:
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"navigation_policy.{name} must be a positive integer")
    for name in (
        "learning_rate",
        "weight_decay",
        "argument_loss_weight",
        "turn_sign_loss_weight",
        "gradient_clip_norm",
        "minimum_validation_action_accuracy",
        "minimum_validation_stop_recall",
        "minimum_validation_turn_sign_accuracy",
        "maximum_validation_argument_mae",
    ):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"navigation_policy.{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"navigation_policy.{name} must be finite and nonnegative")
    for name in ("explicit_scene_invariance_weight",):
        value = settings.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"navigation_policy.{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"navigation_policy.{name} must be finite and nonnegative")
    if settings.get("device", "cpu") != "cpu":
        raise ValueError("Navigation-policy v1 training is deliberately CPU-only")
    return settings


def _load_prefixes(
    prefix_root: Path,
    scene_ids: list[str],
    *,
    expected_tokens: int,
    expected_hidden: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    manifest_path = prefix_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes") if isinstance(manifest, dict) else None
    if (
        not isinstance(scenes, dict)
        or manifest.get("complete_scene_prefixes") is not True
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_scene_retrieval") is not False
        or manifest.get("environmental_text_inputs") != []
    ):
        raise ValueError("Scene-prefix cache does not satisfy the navigation contract")
    tensors: list[torch.Tensor] = []
    for scene_id in scene_ids:
        entry = scenes.get(scene_id)
        if not isinstance(entry, dict):
            raise FileNotFoundError(f"No cached scene prefix for {scene_id}")
        path = prefix_root / str(entry.get("filename"))
        if _sha256(path) != entry.get("file_sha256"):
            raise ValueError(f"Cached scene-prefix file changed for {scene_id}")
        state = load_file(str(path), device="cpu")
        if set(state) != {"scene_prefix"}:
            raise ValueError("Cached scene-prefix tensor keys changed")
        prefix = state["scene_prefix"]
        if prefix.shape != (1, expected_tokens, expected_hidden):
            raise ValueError("Cached scene-prefix shape differs from policy settings")
        if not torch.isfinite(prefix).all():
            raise ValueError("Cached scene prefix contains NaN or infinity")
        tensors.append(prefix[0].contiguous())
    return torch.stack(tensors), tuple(scene_ids)


def _instruction_embeddings(
    config: dict[str, Any], instructions: list[str]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model_id = str(config["language"]["model_id"])
    revision = str(config["language"]["revision"])
    snapshot = resolve_local_snapshot(model_id, revision)
    unique = list(dict.fromkeys(instructions))
    values, metadata = load_category_embeddings_selective(
        snapshot,
        unique,
        expected_dim=int(config["navigation_policy"]["hidden_size"]),
    )
    return {
        instruction: torch.from_numpy(values[index]).float()
        for index, instruction in enumerate(unique)
    }, metadata


@torch.inference_mode()
def _prepare_training_data(
    config: dict[str, Any],
    dataset_path: Path,
) -> tuple[dict[str, Any], _PreparedTrainingData]:
    settings = _validate_settings(config)
    manifest, rows = load_navigation_trace_dataset(dataset_path)
    train_scene_ids = list(manifest["train_scene_ids"])
    validation_scene_ids = list(manifest["validation_scene_ids"])
    scene_ids = [*train_scene_ids, *validation_scene_ids]
    prefixes, prefix_scene_ids = _load_prefixes(
        _rooted(str(settings["prefix_cache_root"])),
        scene_ids,
        expected_tokens=int(settings["scene_token_count"]),
        expected_hidden=int(settings["hidden_size"]),
    )
    scene_index = {scene: index for index, scene in enumerate(prefix_scene_ids)}
    embedding_by_instruction, selective = _instruction_embeddings(
        config, [str(row["instruction"]) for row in rows]
    )
    state_encoder, state_hash, state_metadata = load_robot_state_checkpoint(
        _rooted(str(settings["robot_state_checkpoint"])),
        expected_output_dim=int(settings["hidden_size"]),
        device="cpu",
    )
    if (
        int(state_metadata["token_count"]) != int(settings["robot_token_count"])
        or state_metadata.get("numeric_inputs_only") is not True
    ):
        raise ValueError("Robot-state checkpoint differs from policy token contract")

    def prepare(split: str) -> _PreparedSamples:
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            raise ValueError(f"Navigation trace split is empty: {split}")
        state_features = torch.tensor(
            [row["state_features"] for row in selected], dtype=torch.float32
        )
        robot_tokens = state_encoder(state_features).float().contiguous()
        return _PreparedSamples(
            scene_indices=torch.tensor(
                [scene_index[str(row["scene_id"])] for row in selected],
                dtype=torch.long,
            ),
            robot_tokens=robot_tokens,
            instruction_embeddings=torch.stack(
                [embedding_by_instruction[str(row["instruction"])] for row in selected]
            ),
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

    return manifest, _PreparedTrainingData(
        prefixes=prefixes,
        prefix_scene_ids=prefix_scene_ids,
        train=prepare("train"),
        validation=prepare("validation"),
        selective_embedding_metadata=selective,
        robot_state_checkpoint_sha256=state_hash,
    )


def _scene_batches(
    samples: _PreparedSamples,
    batch_size: int,
    *,
    generator: torch.Generator | None,
) -> list[torch.Tensor]:
    """Keep repeated full prefixes unique within each CPU training batch."""

    batches: list[torch.Tensor] = []
    scene_values = torch.unique(samples.scene_indices, sorted=True)
    if generator is not None:
        scene_values = scene_values[
            torch.randperm(len(scene_values), generator=generator)
        ]
    for scene_index in scene_values:
        indices = torch.nonzero(
            samples.scene_indices == scene_index, as_tuple=True
        )[0]
        if generator is not None:
            indices = indices[torch.randperm(len(indices), generator=generator)]
        batches.extend(
            indices[start : start + batch_size]
            for start in range(0, len(indices), batch_size)
        )
    return batches


def _numeric_action_mask(targets: torch.Tensor) -> torch.Tensor:
    numeric = {ACTION_TO_INDEX["turn"], ACTION_TO_INDEX["move_forward"], ACTION_TO_INDEX["move_backward"]}
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for index in numeric:
        mask |= targets == index
    return mask


def _forward_samples(
    controller: ContinuousNavigationActionController,
    prefixes: torch.Tensor,
    samples: _PreparedSamples,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_scenes = samples.scene_indices[indices]
    unique_scenes, inverse = torch.unique(
        selected_scenes, sorted=True, return_inverse=True
    )
    return controller(
        prefixes[unique_scenes],
        samples.robot_tokens[indices],
        samples.instruction_embeddings[indices],
        scene_batch_indices=inverse,
    )


@torch.inference_mode()
def _evaluate(
    controller: ContinuousNavigationActionController,
    prefixes: torch.Tensor,
    samples: _PreparedSamples,
    *,
    batch_size: int,
) -> dict[str, Any]:
    controller.eval()
    predicted = torch.empty(len(samples), dtype=torch.long)
    argument_predictions = torch.empty(len(samples), dtype=torch.float32)
    for indices in _scene_batches(samples, batch_size, generator=None):
        logits, batch_argument_heads = _forward_samples(
            controller, prefixes, samples, indices
        )
        batch_predicted = torch.argmax(logits, dim=-1)
        predicted[indices] = batch_predicted.cpu()
        argument_predictions[indices] = batch_argument_heads.gather(
            1, samples.action_targets[indices].unsqueeze(1)
        ).squeeze(1).cpu()
    target = samples.action_targets
    correct = predicted == target
    stop_mask = target == ACTION_TO_INDEX["stop"]
    turn_mask = target == ACTION_TO_INDEX["turn"]
    numeric_mask = _numeric_action_mask(target)
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
    distribution = Counter(ACTION_NAMES[int(value)] for value in predicted.tolist())
    return {
        "sample_count": len(samples),
        "scene_count": len(set(samples.scene_ids)),
        "action_accuracy": float(correct.float().mean()),
        "stop_recall": float(correct[stop_mask].float().mean()) if bool(stop_mask.any()) else 0.0,
        "turn_sign_accuracy": (
            float(turn_sign.float().mean()) if bool(turn_mask.any()) else 0.0
        ),
        "argument_mae": (
            float(
                torch.mean(
                    torch.abs(
                        argument_predictions[numeric_mask]
                        - samples.argument_targets[numeric_mask]
                    )
                )
            )
            if bool(numeric_mask.any())
            else 0.0
        ),
        "predicted_action_counts": {
            name: int(distribution.get(name, 0)) for name in ACTION_NAMES
        },
        "by_family": family_metrics,
    }


def train_navigation_policy(
    config: dict[str, Any],
    *,
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train on oracle-side traces and emit a checkpoint only if gates pass."""

    settings = _validate_settings(config)
    dataset_path = _rooted(dataset or str(settings["trace_output"]))
    checkpoint_path = _rooted(checkpoint or str(settings["checkpoint_output"]))
    selected_metrics = _rooted(
        metrics_path
        or "reports/gemma4/metrics/navigation_policy_v1_training.json"
    )
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manifest, data = _prepare_training_data(config, dataset_path)
    controller = ContinuousNavigationActionController(
        int(settings["hidden_size"]),
        model_dim=int(settings["model_dim"]),
        action_count=len(ACTION_NAMES),
    ).cpu()
    counts = torch.bincount(
        data.train.action_targets, minlength=len(ACTION_NAMES)
    ).float()
    if torch.any(counts == 0):
        raise ValueError("Navigation training split does not cover every action class")
    class_weights = torch.rsqrt(counts)
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, Any]] = []
    zero_prefixes = torch.zeros_like(data.prefixes)
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    started = time.monotonic()
    for epoch in range(1, int(settings["epochs"]) + 1):
        controller.train()
        total_loss = 0.0
        total_samples = 0
        for indices in _scene_batches(
            data.train, int(settings["batch_size"]), generator=generator
        ):
            logits, argument_heads = _forward_samples(
                controller, data.prefixes, data.train, indices
            )
            targets = data.train.action_targets[indices]
            arguments = argument_heads.gather(1, targets.unsqueeze(1)).squeeze(1)
            classification = torch.nn.functional.cross_entropy(
                logits, targets, weight=class_weights
            )
            numeric_mask = _numeric_action_mask(targets)
            argument_loss = (
                torch.nn.functional.smooth_l1_loss(
                    arguments[numeric_mask],
                    data.train.argument_targets[indices][numeric_mask],
                )
                if bool(numeric_mask.any())
                else logits.sum() * 0.0
            )
            turn_mask = targets == ACTION_TO_INDEX["turn"]
            if bool(turn_mask.any()):
                target_sign = torch.sign(
                    data.train.argument_targets[indices][turn_mask]
                )
                nonzero = target_sign != 0
                sign_loss = (
                    torch.relu(
                        0.10 - target_sign[nonzero] * arguments[turn_mask][nonzero]
                    ).mean()
                    if bool(nonzero.any())
                    else logits.sum() * 0.0
                )
            else:
                sign_loss = logits.sum() * 0.0
            loss = (
                classification
                + float(settings["argument_loss_weight"]) * argument_loss
                + float(settings["turn_sign_loss_weight"]) * sign_loss
            )
            invariance_weight = float(
                settings.get("explicit_scene_invariance_weight", 0.0)
            )
            explicit_mask = torch.tensor(
                [
                    data.train.families[int(index)]
                    in {"stop", "collision_recovery"}
                    for index in indices
                ],
                dtype=torch.bool,
            )
            if invariance_weight > 0.0 and bool(explicit_mask.any()):
                explicit_indices = indices[explicit_mask]
                invariant_logits, invariant_argument_heads = _forward_samples(
                    controller,
                    zero_prefixes,
                    data.train,
                    explicit_indices,
                )
                invariant_targets = data.train.action_targets[explicit_indices]
                invariant_classification = torch.nn.functional.cross_entropy(
                    invariant_logits,
                    invariant_targets,
                    weight=class_weights,
                )
                invariant_numeric = _numeric_action_mask(invariant_targets)
                invariant_arguments = invariant_argument_heads.gather(
                    1, invariant_targets.unsqueeze(1)
                ).squeeze(1)
                invariant_argument_loss = (
                    torch.nn.functional.smooth_l1_loss(
                        invariant_arguments[invariant_numeric],
                        data.train.argument_targets[explicit_indices][invariant_numeric],
                    )
                    if bool(invariant_numeric.any())
                    else invariant_logits.sum() * 0.0
                )
                loss = loss + invariance_weight * (
                    invariant_classification
                    + float(settings["argument_loss_weight"])
                    * invariant_argument_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                controller.parameters(), float(settings["gradient_clip_norm"])
            )
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total_samples += len(indices)
        validation = _evaluate(
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
                "train_loss": total_loss / max(total_samples, 1),
                "validation_action_accuracy": validation["action_accuracy"],
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
                name: value.detach().cpu().clone()
                for name, value in controller.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(settings["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Navigation-policy training produced no finite candidate")
    controller.load_state_dict(best_state, strict=True)
    train_metrics = _evaluate(
        controller,
        data.prefixes,
        data.train,
        batch_size=int(settings["batch_size"]),
    )
    validation_metrics = _evaluate(
        controller,
        data.prefixes,
        data.validation,
        batch_size=int(settings["batch_size"]),
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
        "scene_splits_disjoint": manifest["scene_splits_disjoint"] is True,
        "collision_checked_targets": manifest["collision_checked_movement_targets"] is True,
        "bounded_targets": manifest["bounded_action_targets"] is True,
        "stop_targets_present": manifest["stop_targets_included"] is True,
    }
    accepted = all(gates.values())
    checkpoint_metadata: dict[str, Any] | None = None
    if accepted:
        checkpoint_metadata = save_navigation_policy_checkpoint(
            checkpoint_path,
            controller,
            runtime_metadata={
                "scene_token_count": int(settings["scene_token_count"]),
                "robot_token_count": int(settings["robot_token_count"]),
                "model_id": str(config["language"]["model_id"]),
                "model_revision": str(config["language"]["revision"]),
                "max_turn_degrees": float(config["robot"]["max_turn_degrees"]),
                "max_move_m": float(config["robot"]["max_move_m"]),
                "task_trained": True,
                "training_dataset_sha256": str(manifest["dataset_sha256"]),
                "train_scene_count": int(manifest["train_scene_count"]),
                "validation_scene_count": int(manifest["validation_scene_count"]),
                "scene_splits_disjoint": True,
                "complete_scene_prefix_required": True,
                "question_independent_scene_prefix_required": True,
                "every_scene_token_processed": True,
                "numeric_robot_tokens_required": True,
                "environmental_text_inputs": [],
                "oracle_inputs_at_runtime": False,
                "collision_interlock_required": True,
            },
        )
    result = {
        "schema": "semantic_3d_chat.navigation_policy_training_result.v1",
        "status": "accepted" if accepted else "rejected",
        "checkpoint_written": accepted,
        "checkpoint": str(checkpoint_path) if accepted else None,
        "dataset_sha256": manifest["dataset_sha256"],
        "robot_state_encoder_sha256": data.robot_state_checkpoint_sha256,
        "local_instruction_embedding": {
            "model_id": config["language"]["model_id"],
            "revision": config["language"]["revision"],
            "method": "mean_local_gemma_input_token_embeddings",
            "selective_row_read": data.selective_embedding_metadata.get(
                "selective_row_read"
            ),
        },
        "device": "cpu",
        "elapsed_seconds": time.monotonic() - started,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train": train_metrics,
        "validation": validation_metrics,
        "gates": gates,
        "thresholds": {
            "minimum_validation_action_accuracy": settings[
                "minimum_validation_action_accuracy"
            ],
            "minimum_validation_stop_recall": settings[
                "minimum_validation_stop_recall"
            ],
            "minimum_validation_turn_sign_accuracy": settings[
                "minimum_validation_turn_sign_accuracy"
            ],
            "maximum_validation_argument_mae": settings[
                "maximum_validation_argument_mae"
            ],
        },
        "history": history,
        "runtime_contract": (
            None
            if checkpoint_metadata is None
            else {
                key: value
                for key, value in checkpoint_metadata.items()
                if key not in {"checkpoint"}
            }
        ),
        "oracle_inputs_at_runtime": False,
        "environmental_text_inputs_at_runtime": [],
        "explicit_scene_invariance_weight": float(
            settings.get("explicit_scene_invariance_weight", 0.0)
        ),
    }
    _atomic_json(selected_metrics, result)
    return result


def evaluate_navigation_policy(
    config: dict[str, Any],
    *,
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a sanitized checkpoint on the physically separate trace labels."""

    settings = _validate_settings(config)
    dataset_path = _rooted(dataset or str(settings["trace_output"]))
    manifest, data = _prepare_training_data(config, dataset_path)
    controller, metadata = load_navigation_policy_checkpoint(
        checkpoint or str(settings["checkpoint_output"]),
        expected_hidden_size=int(settings["hidden_size"]),
        expected_model_id=str(config["language"]["model_id"]),
        expected_model_revision=str(config["language"]["revision"]),
        device="cpu",
    )
    if metadata["training_dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("Navigation checkpoint was trained against another trace dataset")
    return {
        "schema": "semantic_3d_chat.navigation_policy_offline_evaluation.v1",
        "checkpoint_weights_sha256": metadata["weights_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "train": _evaluate(
            controller,
            data.prefixes,
            data.train,
            batch_size=int(settings["batch_size"]),
        ),
        "validation": _evaluate(
            controller,
            data.prefixes,
            data.validation,
            batch_size=int(settings["batch_size"]),
        ),
        "oracle_inputs_used_by_runtime": False,
        "environmental_text_inputs_at_runtime": [],
    }


def evaluate_navigation_policy_controls(
    config: dict[str, Any],
    *,
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Measure continuous scene/state dependence on held-out trace scenes."""

    settings = _validate_settings(config)
    dataset_path = _rooted(dataset or str(settings["trace_output"]))
    manifest, data = _prepare_training_data(config, dataset_path)
    controller, metadata = load_navigation_policy_checkpoint(
        checkpoint or str(settings["checkpoint_output"]),
        expected_hidden_size=int(settings["hidden_size"]),
        expected_model_id=str(config["language"]["model_id"]),
        expected_model_revision=str(config["language"]["revision"]),
        device="cpu",
    )
    if metadata["training_dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("Navigation checkpoint was trained against another trace dataset")
    primary = _evaluate(
        controller,
        data.prefixes,
        data.validation,
        batch_size=int(settings["batch_size"]),
    )
    validation_indices = sorted(
        {
            int(value)
            for value in data.validation.scene_indices.detach().cpu().tolist()
        }
    )
    if len(validation_indices) < 2:
        raise ValueError("Navigation causal controls require two held-out scenes")
    rolled = validation_indices[1:] + validation_indices[:1]
    remap = {source: target for source, target in zip(validation_indices, rolled, strict=True)}
    wrong_scene_samples = _PreparedSamples(
        scene_indices=torch.tensor(
            [remap[int(value)] for value in data.validation.scene_indices.tolist()],
            dtype=torch.long,
        ),
        robot_tokens=data.validation.robot_tokens,
        instruction_embeddings=data.validation.instruction_embeddings,
        action_targets=data.validation.action_targets,
        argument_targets=data.validation.argument_targets,
        families=data.validation.families,
        scene_ids=data.validation.scene_ids,
    )
    zero_scene_prefixes = torch.zeros_like(data.prefixes)
    zero_robot_samples = _PreparedSamples(
        scene_indices=data.validation.scene_indices,
        robot_tokens=torch.zeros_like(data.validation.robot_tokens),
        instruction_embeddings=data.validation.instruction_embeddings,
        action_targets=data.validation.action_targets,
        argument_targets=data.validation.argument_targets,
        families=data.validation.families,
        scene_ids=data.validation.scene_ids,
    )
    wrong_scene = _evaluate(
        controller,
        data.prefixes,
        wrong_scene_samples,
        batch_size=int(settings["batch_size"]),
    )
    zero_scene = _evaluate(
        controller,
        zero_scene_prefixes,
        data.validation,
        batch_size=int(settings["batch_size"]),
    )
    zero_robot = _evaluate(
        controller,
        data.prefixes,
        zero_robot_samples,
        batch_size=int(settings["batch_size"]),
    )
    return {
        "schema": "semantic_3d_chat.navigation_policy_causal_controls.v1",
        "checkpoint_weights_sha256": metadata["weights_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "held_out_scenes_only": True,
        "conditions": {
            "primary": primary,
            "wrong_scene_prefix": wrong_scene,
            "zero_scene_prefix": zero_scene,
            "zero_robot_tokens": zero_robot,
        },
        "accuracy_deltas_from_primary": {
            "wrong_scene_prefix": primary["action_accuracy"]
            - wrong_scene["action_accuracy"],
            "zero_scene_prefix": primary["action_accuracy"]
            - zero_scene["action_accuracy"],
            "zero_robot_tokens": primary["action_accuracy"]
            - zero_robot["action_accuracy"],
        },
        "oracle_inputs_used_by_runtime": False,
        "environmental_text_inputs_at_runtime": [],
    }


__all__ = [
    "evaluate_navigation_policy",
    "evaluate_navigation_policy_controls",
    "train_navigation_policy",
]

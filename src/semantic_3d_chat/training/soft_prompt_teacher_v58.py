"""Training-only soft-prompt teachers for V58 controller distillation.

The frozen language model can interpret continuous control tokens, but its
gradient is too ill-conditioned to train the shared scene controller directly.
V58 first optimizes small per-example continuous prompts through the frozen LM,
then distills those prompts into ``FullSceneQuestionControl`` without the LM in
the backward path.  Teacher prompts never enter the runtime checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from semantic_3d_chat.config import PROJECT_ROOT

TeacherRole = Literal["changed_teacher", "retention_baseline"]
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class SoftPromptTarget:
    """One numeric target keyed only by opaque supervised-training identities."""

    scene_id: str
    question_id: str
    role: TeacherRole
    tokens: torch.Tensor

    def __post_init__(self) -> None:
        if not self.scene_id.startswith("scene_") or not self.question_id:
            raise ValueError("V58 teacher targets require opaque scene/question IDs")
        if self.role not in {"changed_teacher", "retention_baseline"}:
            raise ValueError("V58 teacher target role is invalid")
        if self.tokens.ndim != 3 or self.tokens.shape[0] != 1:
            raise ValueError("V58 teacher tokens must have shape [1,C,H]")
        if self.tokens.shape[1] < 1 or self.tokens.shape[2] < 1:
            raise ValueError("V58 teacher token dimensions must be positive")
        if not self.tokens.is_floating_point() or not torch.isfinite(self.tokens).all():
            raise ValueError("V58 teacher tokens must be finite floating point")

    @property
    def key(self) -> tuple[str, str]:
        return self.scene_id, self.question_id


def normalized_prompt_distillation_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float = 1.0,
    cosine_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match tokenwise magnitude and direction without depending on LM scale."""

    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("V58 predicted and target prompts must share shape [B,C,H]")
    if predicted.shape[0] < 1 or predicted.shape[1] < 1 or predicted.shape[2] < 1:
        raise ValueError("V58 prompt tensors must be nonempty")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("V58 prompt tensors must be finite")
    for field, value in (("mse_weight", mse_weight), ("cosine_weight", cosine_weight)):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"V58 {field} must be finite and nonnegative")
    if mse_weight == 0.0 and cosine_weight == 0.0:
        raise ValueError("V58 prompt distillation enables no loss")
    predicted_f = predicted.float()
    target_f = target.detach().float()
    target_power = target_f.square().mean(dim=(-2, -1)).clamp_min(1e-8)
    normalized_mse_per_row = (
        (predicted_f - target_f).square().mean(dim=(-2, -1)) / target_power
    )
    token_cosine = F.cosine_similarity(predicted_f, target_f, dim=-1, eps=1e-8)
    cosine_loss_per_row = 1.0 - token_cosine.mean(dim=-1)
    normalized_mse = normalized_mse_per_row.mean()
    cosine_loss = cosine_loss_per_row.mean()
    total = float(mse_weight) * normalized_mse + float(cosine_weight) * cosine_loss
    if not torch.isfinite(total):
        raise RuntimeError("V58 prompt distillation loss is nonfinite")
    return total, {
        "normalized_mse": normalized_mse,
        "cosine_loss": cosine_loss,
        "mean_token_cosine": token_cosine.mean(),
    }


def pair_delta_distillation_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float = 1.0,
    cosine_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match the direction and magnitude separating two physical scene sides."""

    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[0] != 2:
        raise ValueError("V58 pair prompts must share shape [2,C,H]")
    predicted_delta = predicted.float()[0] - predicted.float()[1]
    target_delta = target.detach().float()[0] - target.detach().float()[1]
    target_power = target_delta.square().mean().clamp_min(1e-8)
    normalized_mse = (predicted_delta - target_delta).square().mean() / target_power
    predicted_flat = predicted_delta.reshape(1, -1)
    target_flat = target_delta.reshape(1, -1)
    cosine = F.cosine_similarity(predicted_flat, target_flat, dim=-1, eps=1e-8).squeeze(0)
    cosine_loss = 1.0 - cosine
    total = float(mse_weight) * normalized_mse + float(cosine_weight) * cosine_loss
    if not torch.isfinite(total):
        raise RuntimeError("V58 pair-delta distillation loss is nonfinite")
    return total, {
        "normalized_delta_mse": normalized_mse,
        "delta_cosine_loss": cosine_loss,
        "delta_cosine": cosine,
        "target_delta_rms": target_delta.square().mean().sqrt(),
        "predicted_delta_rms": predicted_delta.square().mean().sqrt(),
    }


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"V58 {field} must be a lowercase SHA-256 digest")
    return value


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V58 teacher artifact path contains a symlink: {current}")


def _teacher_metadata(
    targets: Sequence[SoftPromptTarget],
    *,
    weights_sha256: str,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_control_checkpoint_sha256: str,
) -> dict[str, Any]:
    ordered = sorted(targets, key=lambda item: item.key)
    if len({target.key for target in ordered}) != len(ordered):
        raise ValueError("V58 teacher targets contain duplicate opaque keys")
    common_shape = {tuple(target.tokens.shape) for target in ordered}
    if not ordered or len(common_shape) != 1:
        raise ValueError("V58 teacher targets require one nonempty common shape")
    records = []
    for index, target in enumerate(ordered):
        records.append(
            {
                "tensor_key": f"prompt_{index:06d}",
                "scene_id": target.scene_id,
                "question_id": target.question_id,
                "role": target.role,
                "shape": list(target.tokens.shape),
                "rms": float(target.tokens.detach().float().square().mean().sqrt()),
            }
        )
    return {
        "schema_version": 1,
        "artifact": "v58_training_only_soft_prompt_teachers",
        "weights_sha256": _validate_digest(weights_sha256, "teacher weights"),
        "base_checkpoint_sha256": _validate_digest(
            base_checkpoint_sha256, "base checkpoint"
        ),
        "base_runtime_config_sha256": _validate_digest(
            base_runtime_config_sha256, "runtime config"
        ),
        "source_control_checkpoint_sha256": _validate_digest(
            source_control_checkpoint_sha256, "source control checkpoint"
        ),
        "target_count": len(records),
        "environmental_text_inputs": [],
        "runtime_load_permitted": False,
        "records": records,
    }


def save_teacher_artifact(
    destination: str | Path,
    *,
    targets: Sequence[SoftPromptTarget],
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    source_control_checkpoint_sha256: str,
) -> dict[str, str]:
    """Save numeric teachers separately from the strict two-file runtime checkpoint."""

    root = _resolve(destination)
    _reject_symlink_components(root)
    if root.exists():
        raise FileExistsError(f"V58 teacher artifact already exists: {root}")
    forbidden = {"oracle", "validation", "development", "test", "final", "v55"}
    try:
        scoped = root.relative_to(PROJECT_ROOT)
    except ValueError:
        scoped = Path(root.name)
    tokens = {
        token
        for part in scoped.parts
        for token in part.casefold().replace("-", "_").split("_")
    }
    if forbidden & tokens:
        raise ValueError("V58 teacher artifact path targets a forbidden split")
    ordered = sorted(targets, key=lambda item: item.key)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    try:
        state = {
            f"prompt_{index:06d}": target.tokens.detach().cpu().float().contiguous()
            for index, target in enumerate(ordered)
        }
        weights = root / "teachers.safetensors"
        save_file(state, weights)
        weights_sha256 = _sha256_file(weights)
        metadata = _teacher_metadata(
            ordered,
            weights_sha256=weights_sha256,
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_runtime_config_sha256=base_runtime_config_sha256,
            source_control_checkpoint_sha256=source_control_checkpoint_sha256,
        )
        metadata_path = root / "metadata.json"
        with metadata_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        loaded, loaded_metadata = load_teacher_artifact(root)
        expected = {target.key: target.tokens.detach().cpu().float() for target in ordered}
        if set(loaded) != set(expected) or any(
            not torch.equal(loaded[key], expected[key]) for key in expected
        ):
            raise RuntimeError("V58 saved teacher artifact failed exact reload")
        if loaded_metadata != metadata:
            raise RuntimeError("V58 saved teacher metadata changed on reload")
        return {
            "weights_sha256": weights_sha256,
            "metadata_sha256": _sha256_file(metadata_path),
        }
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def load_teacher_artifact(
    source: str | Path,
) -> tuple[dict[tuple[str, str], torch.Tensor], dict[str, Any]]:
    root = _resolve(source)
    _reject_symlink_components(root)
    if not root.is_dir() or {item.name for item in root.iterdir()} != {
        "teachers.safetensors",
        "metadata.json",
    }:
        raise ValueError("V58 teacher artifact inventory changed")
    weights = root / "teachers.safetensors"
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "artifact",
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_control_checkpoint_sha256",
        "target_count",
        "environmental_text_inputs",
        "runtime_load_permitted",
        "records",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise ValueError("V58 teacher metadata fields changed")
    if (
        metadata["schema_version"] != 1
        or metadata["artifact"] != "v58_training_only_soft_prompt_teachers"
        or metadata["environmental_text_inputs"] != []
        or metadata["runtime_load_permitted"] is not False
        or _sha256_file(weights) != metadata["weights_sha256"]
    ):
        raise ValueError("V58 teacher metadata contract mismatch")
    for field in (
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_control_checkpoint_sha256",
    ):
        _validate_digest(metadata[field], field)
    records = metadata["records"]
    if not isinstance(records, list) or metadata["target_count"] != len(records):
        raise ValueError("V58 teacher record count changed")
    state = load_file(str(weights), device="cpu")
    expected_tensor_keys = {record.get("tensor_key") for record in records}
    if set(state) != expected_tensor_keys:
        raise ValueError("V58 teacher tensor inventory changed")
    result: dict[tuple[str, str], torch.Tensor] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "tensor_key",
            "scene_id",
            "question_id",
            "role",
            "shape",
            "rms",
        }:
            raise ValueError("V58 teacher record fields changed")
        key = (record["scene_id"], record["question_id"])
        tensor = state[record["tensor_key"]].detach().contiguous()
        SoftPromptTarget(key[0], key[1], record["role"], tensor)
        if list(tensor.shape) != record["shape"]:
            raise ValueError("V58 teacher tensor shape changed")
        if key in result:
            raise ValueError("V58 teacher metadata has duplicate opaque keys")
        result[key] = tensor
    return result, dict(metadata)


__all__ = [
    "SoftPromptTarget",
    "load_teacher_artifact",
    "normalized_prompt_distillation_loss",
    "pair_delta_distillation_loss",
    "save_teacher_artifact",
]

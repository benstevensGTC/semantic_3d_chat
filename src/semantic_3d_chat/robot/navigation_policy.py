"""Compact learned controller over continuous scene and robot-state tokens.

The controller is deliberately separate from the simulator oracle.  At
runtime it accepts only the complete, question-independent scene prefix, the
continuous numeric robot-state tokens inserted beside that prefix, and the
literal user instruction.  Its output is still only a proposal: the existing
strict tool validator and the numeric collision interlock must both accept it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal

ACTION_NAMES: Final[tuple[str, ...]] = (
    "stop",
    "scan",
    "turn",
    "move_forward",
    "move_backward",
)
ACTION_TO_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(ACTION_NAMES)
}
_ARCHITECTURE: Final[str] = "continuous_navigation_action_controller_v1"
_SCHEMA_VERSION: Final[int] = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BLOCKED_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "training", "scorer_only"}
)
_FILES: Final[frozenset[str]] = frozenset(
    {"policy.safetensors", "runtime_metadata.json"}
)
_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "architecture",
        "hidden_size",
        "model_dim",
        "scene_token_count",
        "robot_token_count",
        "action_names",
        "model_id",
        "model_revision",
        "max_turn_degrees",
        "max_move_m",
        "task_trained",
        "training_dataset_sha256",
        "train_scene_count",
        "validation_scene_count",
        "scene_splits_disjoint",
        "complete_scene_prefix_required",
        "question_independent_scene_prefix_required",
        "every_scene_token_processed",
        "numeric_robot_tokens_required",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
        "collision_interlock_required",
        "weights_sha256",
    }
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _reject_runtime_path(path: Path) -> None:
    if _BLOCKED_COMPONENTS & {part.casefold() for part in path.parts}:
        raise ValueError("Navigation-policy runtime paths cannot enter blocked data trees")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Navigation-policy runtime paths cannot contain symbolic links")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate navigation-policy metadata field: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Navigation-policy runtime metadata is invalid JSON") from error
    if not isinstance(value, dict) or set(value) != _METADATA_FIELDS:
        raise ValueError("Navigation-policy runtime metadata fields changed")
    return value


def _positive_int(value: object, name: str, maximum: int = 16384) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"Navigation-policy {name} must be in [1, {maximum}]")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Navigation-policy {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Navigation-policy {name} must be finite and positive")
    return result


class ContinuousNavigationActionController(nn.Module):
    """Query-aware all-token readout with a small bounded action head."""

    def __init__(
        self,
        hidden_size: int,
        *,
        model_dim: int = 128,
        action_count: int = len(ACTION_NAMES),
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.model_dim = _positive_int(model_dim, "model_dim", 2048)
        self.action_count = _positive_int(action_count, "action_count", 64)
        self.scene_key = nn.Linear(self.hidden_size, self.model_dim, bias=False)
        self.scene_value = nn.Linear(self.hidden_size, self.model_dim, bias=False)
        self.instruction_query = nn.Linear(self.hidden_size, self.model_dim, bias=False)
        self.instruction_value = nn.Linear(self.hidden_size, self.model_dim)
        self.robot_value = nn.Linear(self.hidden_size, self.model_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.model_dim * 4),
            nn.Linear(self.model_dim * 4, self.model_dim * 2),
            nn.GELU(),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(self.model_dim, self.action_count)
        self.argument_head = nn.Linear(self.model_dim, self.action_count)

    def forward(
        self,
        scene_prefix: torch.Tensor,
        robot_tokens: torch.Tensor,
        instruction_embedding: torch.Tensor,
        *,
        scene_batch_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scene_prefix.ndim != 3 or scene_prefix.shape[-1] != self.hidden_size:
            raise ValueError(
                f"scene_prefix must have shape [B, S, {self.hidden_size}]"
            )
        if robot_tokens.ndim != 3:
            raise ValueError("robot_tokens must have shape [B, R, H]")
        if robot_tokens.shape[-1] != self.hidden_size or robot_tokens.shape[1] < 1:
            raise ValueError("robot_tokens have an invalid width or empty sequence")
        if instruction_embedding.ndim == 3:
            instruction_embedding = instruction_embedding.mean(dim=1)
        batch_size = robot_tokens.shape[0]
        if instruction_embedding.shape != (batch_size, self.hidden_size):
            raise ValueError("instruction_embedding must have shape [B, H] or [B, T, H]")
        if scene_batch_indices is None:
            if scene_prefix.shape[0] != batch_size:
                raise ValueError("scene and action batch sizes differ")
        else:
            if (
                scene_batch_indices.shape != (batch_size,)
                or scene_batch_indices.dtype != torch.long
                or torch.any(scene_batch_indices < 0)
                or torch.any(scene_batch_indices >= scene_prefix.shape[0])
            ):
                raise ValueError("scene_batch_indices are invalid")
        if scene_prefix.shape[1] < 3:
            raise ValueError("scene_prefix must include boundaries and scene content")
        if not all(
            torch.isfinite(value).all()
            for value in (scene_prefix, robot_tokens, instruction_embedding)
        ):
            raise ValueError("Navigation-policy inputs contain NaN or infinity")

        # Layer-normalize without learned affine parameters so cached prefix
        # scale cannot dominate the small controller.  The global mean path
        # guarantees that every scene token contributes, while cross-attention
        # supplies the instruction-dependent semantic readout.
        scene = torch.nn.functional.layer_norm(
            scene_prefix.float(), (self.hidden_size,)
        )
        robot = torch.nn.functional.layer_norm(
            robot_tokens.float(), (self.hidden_size,)
        )
        instruction = torch.nn.functional.layer_norm(
            instruction_embedding.float(), (self.hidden_size,)
        )
        keys = self.scene_key(scene)
        values = self.scene_value(scene)
        if scene_batch_indices is not None:
            selected = scene_batch_indices.to(keys.device)
            keys = keys[selected]
            values = values[selected]
        query = self.instruction_query(instruction)
        scores = torch.einsum("bd,bsd->bs", query, keys) / math.sqrt(self.model_dim)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.einsum("bs,bsd->bd", attention, values)
        global_scene = values.mean(dim=1)
        instruction_value = self.instruction_value(instruction)
        robot_value = self.robot_value(robot).mean(dim=1)
        fused = self.fusion(
            torch.cat(
                (attended, global_scene, instruction_value, robot_value),
                dim=-1,
            )
        )
        logits = self.action_head(fused)
        normalized_arguments = torch.tanh(self.argument_head(fused))
        if not torch.isfinite(logits).all() or not torch.isfinite(normalized_arguments).all():
            raise RuntimeError("Navigation-policy output contains NaN or infinity")
        return logits, normalized_arguments


def normalized_argument_for_action(
    action_name: str,
    value: float,
    *,
    max_turn_degrees: float,
    max_move_m: float,
) -> float:
    """Convert one already-bounded tool argument into the training interval."""

    max_turn = _finite_positive(max_turn_degrees, "max_turn_degrees")
    max_move = _finite_positive(max_move_m, "max_move_m")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Action argument must be finite")
    if action_name == "turn":
        if abs(numeric) > max_turn + 1e-9:
            raise ValueError("Turn target exceeds the bounded tool schema")
        return numeric / max_turn
    if action_name in {"move_forward", "move_backward"}:
        if not 0.0 <= numeric <= max_move + 1e-9:
            raise ValueError("Movement target exceeds the bounded tool schema")
        # Each action-specific head uses [-1, 1]. Map nonnegative movement
        # onto that full interval to retain precision near both limits.
        return 2.0 * numeric / max_move - 1.0
    if action_name in {"stop", "scan"}:
        if abs(numeric) > 1e-9:
            raise ValueError("Argument-free actions must use a zero target")
        return 0.0
    raise ValueError(f"Unsupported learned action: {action_name}")


def tool_call_from_prediction(
    action_index: int,
    normalized_argument: float,
    *,
    max_turn_degrees: float,
    max_move_m: float,
) -> dict[str, Any]:
    """Decode a controller output to one exact bounded tool envelope."""

    if isinstance(action_index, bool) or not isinstance(action_index, int):
        raise TypeError("Action index must be an integer")
    if not 0 <= action_index < len(ACTION_NAMES):
        raise ValueError("Action index is outside the learned action vocabulary")
    value = float(normalized_argument)
    if not math.isfinite(value):
        raise ValueError("Normalized action argument must be finite")
    value = min(1.0, max(-1.0, value))
    name = ACTION_NAMES[action_index]
    if name in {"stop", "scan"}:
        arguments: dict[str, float] = {}
    elif name == "turn":
        maximum = _finite_positive(max_turn_degrees, "max_turn_degrees")
        arguments = {"angle_degrees": float(value * maximum)}
    else:
        maximum = _finite_positive(max_move_m, "max_move_m")
        distance = max(0.02, (value + 1.0) * 0.5 * maximum)
        arguments = {"distance_meters": float(min(maximum, distance))}
    return {"tool": name, "arguments": arguments}


def split_active_prefix(
    active_prefix: torch.Tensor,
    *,
    scene_token_count: int,
    robot_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover scene and inserted robot tokens from the attested active prefix."""

    scene_count = _positive_int(scene_token_count, "scene_token_count")
    robot_count = _positive_int(robot_token_count, "robot_token_count")
    if active_prefix.ndim != 3 or active_prefix.shape[1] != scene_count + robot_count:
        raise ValueError("Active prefix length differs from the checkpoint contract")
    # Runtime insertion is scene[:-1], robot, scene[-1:].
    robot = active_prefix[:, scene_count - 1 : scene_count - 1 + robot_count]
    scene = torch.cat(
        (
            active_prefix[:, : scene_count - 1],
            active_prefix[:, -1:],
        ),
        dim=1,
    )
    return scene, robot


def _validated_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(metadata)
    hidden_size = _positive_int(value.get("hidden_size"), "hidden_size")
    model_dim = _positive_int(value.get("model_dim"), "model_dim", 2048)
    scene_token_count = _positive_int(
        value.get("scene_token_count"), "scene_token_count"
    )
    robot_token_count = _positive_int(
        value.get("robot_token_count"), "robot_token_count", 64
    )
    train_count = _positive_int(value.get("train_scene_count"), "train_scene_count")
    validation_count = _positive_int(
        value.get("validation_scene_count"), "validation_scene_count"
    )
    max_turn = _finite_positive(value.get("max_turn_degrees"), "max_turn_degrees")
    max_move = _finite_positive(value.get("max_move_m"), "max_move_m")
    digest = value.get("training_dataset_sha256")
    weights_digest = value.get("weights_sha256")
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("architecture") != _ARCHITECTURE
        or value.get("action_names") != list(ACTION_NAMES)
        or not isinstance(value.get("model_id"), str)
        or not value["model_id"]
        or not isinstance(value.get("model_revision"), str)
        or not value["model_revision"]
        or value.get("task_trained") is not True
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or not isinstance(weights_digest, str)
        or _SHA256.fullmatch(weights_digest) is None
        or value.get("scene_splits_disjoint") is not True
        or value.get("complete_scene_prefix_required") is not True
        or value.get("question_independent_scene_prefix_required") is not True
        or value.get("every_scene_token_processed") is not True
        or value.get("numeric_robot_tokens_required") is not True
        or value.get("environmental_text_inputs") != []
        or value.get("oracle_inputs_at_runtime") is not False
        or value.get("collision_interlock_required") is not True
    ):
        raise ValueError("Navigation-policy runtime contract is not inference safe")
    value.update(
        {
            "hidden_size": hidden_size,
            "model_dim": model_dim,
            "scene_token_count": scene_token_count,
            "robot_token_count": robot_token_count,
            "train_scene_count": train_count,
            "validation_scene_count": validation_count,
            "max_turn_degrees": max_turn,
            "max_move_m": max_move,
        }
    )
    return value


def save_navigation_policy_checkpoint(
    destination: str | Path,
    controller: ContinuousNavigationActionController,
    *,
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the exact two-file inference checkpoint, refusing overwrite."""

    root = _rooted(destination)
    _reject_runtime_path(root)
    if root.exists():
        raise FileExistsError(f"Navigation-policy checkpoint already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights = temporary / "policy.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in controller.state_dict().items()
            },
            str(weights),
        )
        metadata = {
            **dict(runtime_metadata),
            "schema_version": _SCHEMA_VERSION,
            "architecture": _ARCHITECTURE,
            "hidden_size": controller.hidden_size,
            "model_dim": controller.model_dim,
            "action_names": list(ACTION_NAMES),
            "weights_sha256": _sha256_file(weights),
        }
        if set(metadata) != _METADATA_FIELDS:
            raise ValueError("Navigation-policy save metadata fields changed")
        metadata = _validated_metadata(metadata)
        (temporary / "runtime_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root)
        return {**metadata, "checkpoint": str(root)}
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()


def load_navigation_policy_checkpoint(
    checkpoint: str | Path,
    *,
    expected_hidden_size: int,
    expected_model_id: str,
    expected_model_revision: str,
    device: torch.device | str = "cpu",
    audit: FileAccessAudit | None = None,
) -> tuple[ContinuousNavigationActionController, dict[str, Any]]:
    """Strictly load only sanitized learned-controller runtime artifacts."""

    root = _rooted(checkpoint)
    _reject_runtime_path(root)
    if not root.is_dir() or {entry.name for entry in root.iterdir()} != _FILES:
        raise ValueError("Navigation-policy checkpoint must contain exactly two files")
    weights = root / "policy.safetensors"
    metadata_path = root / "runtime_metadata.json"
    if any(entry.is_symlink() or not entry.is_file() for entry in (weights, metadata_path)):
        raise ValueError("Navigation-policy checkpoint entries must be regular files")
    if audit is not None:
        audit.record(weights)
        audit.record(metadata_path)
    metadata = _validated_metadata(_strict_json(metadata_path))
    if _sha256_file(weights) != metadata["weights_sha256"]:
        raise ValueError("Navigation-policy weights hash mismatch")
    if metadata["hidden_size"] != _positive_int(
        expected_hidden_size, "expected_hidden_size"
    ):
        raise ValueError("Navigation-policy hidden size differs from the local model")
    if (
        metadata["model_id"] != expected_model_id
        or metadata["model_revision"] != expected_model_revision
    ):
        raise ValueError("Navigation-policy local model identity changed")
    controller = ContinuousNavigationActionController(
        int(metadata["hidden_size"]),
        model_dim=int(metadata["model_dim"]),
        action_count=len(ACTION_NAMES),
    )
    state = load_file(str(weights), device="cpu")
    expected = controller.state_dict()
    if set(state) != set(expected):
        raise ValueError("Navigation-policy tensor keys changed")
    if any(state[name].shape != expected[name].shape for name in expected):
        raise ValueError("Navigation-policy tensor shapes changed")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Navigation-policy checkpoint contains NaN or infinity")
    controller.load_state_dict(state, strict=True)
    controller.to(device).eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller, metadata


def _literal_user_instruction(policy_input: str) -> str:
    prefix = "User navigation instruction: "
    stripped = policy_input.strip()
    if stripped.startswith(prefix):
        first_line = stripped.splitlines()[0]
        value = first_line[len(prefix) :].strip()
    else:
        value = stripped
    if not value or len(value) > 1024:
        raise ValueError("Learned navigation instruction is empty or too long")
    return value


def _collision_safe_or_stop(runtime: Any, call: dict[str, Any]) -> dict[str, Any]:
    """Fail safe using only anonymous numeric collision geometry."""

    name = call["tool"]
    if name not in {"move_forward", "move_backward"}:
        return call
    simulator = getattr(runtime, "simulator", None)
    collision_map = getattr(simulator, "collision_map", None)
    state = getattr(simulator, "state", None)
    if collision_map is None or state is None:
        return {"tool": "stop", "arguments": {}}
    start = np.asarray(getattr(state, "position_xy_m", None), dtype=np.float64)
    yaw = float(getattr(state, "body_yaw_degrees", math.nan))
    distance = float(call["arguments"]["distance_meters"])
    if start.shape != (2,) or not np.isfinite(start).all() or not math.isfinite(yaw):
        return {"tool": "stop", "arguments": {}}
    direction = np.asarray(
        [-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))],
        dtype=np.float64,
    )
    if name == "move_backward":
        direction *= -1.0
    if collision_map.segment_check(start, start + distance * direction).collision:
        return {"tool": "stop", "arguments": {}}
    return call


class LearnedContinuousActionBackend:
    """Tool-proposal backend backed by the sanitized compact controller."""

    def __init__(
        self,
        runtime: Any,
        controller: ContinuousNavigationActionController,
        metadata: Mapping[str, Any],
    ) -> None:
        self.runtime = runtime
        self.controller = controller.eval()
        self.metadata = _validated_metadata(metadata)
        prefix_refresher = getattr(runtime, "prefix_refresher", None)
        wrapped = getattr(prefix_refresher, "runtime", None)
        self.base = getattr(wrapped, "base", wrapped)
        if self.base is None or getattr(self.base, "language", None) is None:
            raise TypeError("Learned navigation backend requires a loaded local language model")
        language = self.base.language
        if language.hidden_size != controller.hidden_size:
            raise ValueError("Learned controller width differs from the local language model")
        if (
            self.metadata["model_id"] != self.base.config["language"]["model_id"]
            or self.metadata["model_revision"]
            != self.base.config["language"]["revision"]
        ):
            raise ValueError("Learned controller is bound to another local model")

    @torch.inference_mode()
    def generate(
        self,
        instruction: str,
        *,
        correction_code: str | None,
    ) -> GeneratedToolProposal:
        del correction_code  # strict validation retries recompute the same safe policy
        active, binding = self.runtime.active_prefix_snapshot()
        observed_hash = prefix_sha256(active)
        if binding.get("active_prefix_sha256") != observed_hash:
            raise RuntimeError("Learned navigation prefix differs from its runtime binding")
        scene, robot = split_active_prefix(
            active,
            scene_token_count=int(self.metadata["scene_token_count"]),
            robot_token_count=int(self.metadata["robot_token_count"]),
        )
        literal = _literal_user_instruction(instruction)
        tokenizer = self.base.language.tokenizer
        encoded = tokenizer(literal, add_special_tokens=False, return_tensors="pt")
        token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2 or not token_ids.numel():
            raise ValueError("Local tokenizer returned no navigation instruction tokens")
        embedding_layer = self.base.language.model.get_input_embeddings()
        token_ids = token_ids.to(embedding_layer.weight.device)
        instruction_embedding = embedding_layer(token_ids).float().mean(dim=1)
        device = next(self.controller.parameters()).device
        logits, arguments = self.controller(
            scene.to(device),
            robot.to(device),
            instruction_embedding.to(device),
        )
        action_index = int(torch.argmax(logits[0]).item())
        call = tool_call_from_prediction(
            action_index,
            float(arguments[0, action_index].item()),
            max_turn_degrees=float(self.metadata["max_turn_degrees"]),
            max_move_m=float(self.metadata["max_move_m"]),
        )
        call = _collision_safe_or_stop(self.runtime, call)
        scene_hash = binding.get("scene_prefix_sha256")
        robot_hash = binding.get("robot_tokens_sha256")
        return GeneratedToolProposal(
            text=json.dumps(
                call,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            active_prefix_sha256=observed_hash,
            scene_prefix_sha256=scene_hash if isinstance(scene_hash, str) else "",
            robot_tokens_sha256=robot_hash if isinstance(robot_hash, str) else None,
            local_inference=True,
            used_continuous_scene_prefix=True,
            used_continuous_robot_tokens=True,
            training_status="supervised_continuous_navigation_policy_v1",
        )


__all__ = [
    "ACTION_NAMES",
    "ACTION_TO_INDEX",
    "ContinuousNavigationActionController",
    "LearnedContinuousActionBackend",
    "load_navigation_policy_checkpoint",
    "normalized_argument_for_action",
    "save_navigation_policy_checkpoint",
    "split_active_prefix",
    "tool_call_from_prediction",
]

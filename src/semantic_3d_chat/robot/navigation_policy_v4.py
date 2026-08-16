"""Collision-aware continuous-semantic navigation policy V4.

V4 preserves V3's complete scene prefix and label-free semantic target state,
then adds one deliberately narrow input: a 24-ray robot-frame clearance field
computed from anonymous numeric collision geometry.  The deployable runtime
never accepts object labels, oracle metadata, captions, or serialized scene
descriptions.

The learned controller predicts actions, arguments, and counterfactual motion
risks.  An independent exact-geometry mask remains authoritative: when the
highest-scoring movement would collide, it selects the highest-scoring safe
nonterminal alternative instead of converting the movement into a terminal
``stop``.  A final collision interlock rechecks the selected movement.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_PROJECTED_DIM,
    GEMMA4_PROJECTED_START,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.action_context import (
    ContinuousActionContext,
    capture_continuous_action_context,
    require_grounding_map_binding,
)
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal
from semantic_3d_chat.robot.navigation_policy import (
    ACTION_NAMES,
    split_active_prefix,
    tool_call_from_prediction,
)
from semantic_3d_chat.robot.navigation_policy_v3 import (
    TARGET_STATE_DIM,
    GroundedContinuousNavigationControllerV3,
    grounded_target_state,
    target_text_from_navigation_instruction,
)
from semantic_3d_chat.robot.semantic_agent import (
    ContinuousSemanticGrounding,
    ContinuousSemanticTargetGrounder,
    ContinuousTextEncoder,
    GemmaProjectedTextEncoder,
)
from semantic_3d_chat.robot.state_encoder import robot_state_vector

CLEARANCE_RAY_COUNT: Final[int] = 24
CLEARANCE_MAX_RANGE_M: Final[float] = 1.0
COLLISION_PROBE_DISTANCES_M: Final[tuple[float, ...]] = (
    0.125,
    0.250,
    0.375,
    0.500,
)
COLLISION_RISK_DIM: Final[int] = 2 * len(COLLISION_PROBE_DISTANCES_M)
ARCHITECTURE: Final[str] = "continuous_semantic_clearance_navigation_controller_v4"
TRAINING_STATUS: Final[str] = (
    "supervised_continuous_semantic_clearance_navigation_policy_v4"
)
SCHEMA_VERSION: Final[int] = 4

_FILES: Final[frozenset[str]] = frozenset({"policy.safetensors", "runtime_metadata.json"})
_BLOCKED: Final[frozenset[str]] = frozenset({"oracle", "qa", "training", "scorer_only"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "architecture",
        "hidden_size",
        "model_dim",
        "target_state_dim",
        "clearance_ray_count",
        "clearance_max_range_m",
        "collision_probe_distances_m",
        "collision_risk_dim",
        "scene_token_count",
        "robot_token_count",
        "action_names",
        "model_id",
        "model_revision",
        "max_turn_degrees",
        "max_move_m",
        "room_size_m",
        "grounding_feature_start",
        "grounding_feature_dim",
        "task_trained",
        "training_dataset_sha256",
        "preregistration_sha256",
        "preregistered_single_arm",
        "v3_initialization_weights_sha256",
        "train_scene_count",
        "validation_scene_count",
        "scene_splits_disjoint",
        "complete_scene_prefix_required",
        "question_independent_static_scene_prefix_required",
        "every_scene_token_processed",
        "numeric_robot_tokens_required",
        "continuous_semantic_grounding_required",
        "all_map_voxels_scored_for_grounding",
        "numeric_clearance_state_required",
        "clearance_from_sanitized_geometry_only",
        "exact_collision_mask_required",
        "unsafe_motion_fallback",
        "query_dependent_grounding_navigation_only",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
        "runtime_required_files",
        "collision_interlock_required",
        "weights_sha256",
    }
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _reject_runtime_path(path: Path) -> None:
    if _BLOCKED & {part.casefold() for part in path.parts}:
        raise ValueError("V4 runtime paths cannot enter blocked data trees")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("V4 runtime paths cannot contain symbolic links")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: object, name: str, maximum: int = 16384) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"V4 {name} must be in [1, {maximum}]")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V4 {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"V4 {name} must be finite and positive")
    return result


def _literal_instruction(policy_input: str) -> str:
    if not isinstance(policy_input, str):
        raise TypeError("V4 navigation instruction must be text")
    prefix = "User navigation instruction: "
    stripped = policy_input.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    value = first_line[len(prefix) :].strip() if first_line.startswith(prefix) else stripped
    if not value or len(value) > 1024:
        raise ValueError("V4 navigation instruction is empty or too long")
    return value


def _ray_directions(yaw_degrees: float, ray_count: int) -> np.ndarray:
    yaw = float(yaw_degrees)
    if not math.isfinite(yaw):
        raise ValueError("V4 body yaw must be finite")
    count = _positive_int(ray_count, "clearance ray count", 512)
    angles = np.deg2rad(yaw) + np.arange(count, dtype=np.float64) * (
        2.0 * math.pi / count
    )
    # The simulator's forward convention is [-sin(yaw), cos(yaw)].
    return np.stack((-np.sin(angles), np.cos(angles)), axis=1)


def robot_frame_clearance_state(
    collision_map: NumericCollisionMap,
    position_xy_m: np.ndarray | Sequence[float],
    yaw_degrees: float,
    *,
    ray_count: int = CLEARANCE_RAY_COUNT,
    max_range_m: float = CLEARANCE_MAX_RANGE_M,
    obstacle_chunk_size: int = 4096,
) -> torch.Tensor:
    """Return normalized anonymous free distance along evenly spaced rays.

    Ray zero is forward; ray ``ray_count / 2`` is backward.  The computation
    is analytic for both room boundaries and the robot-radius-inflated numeric
    obstacle points, so it has no ray-marching gaps and consumes no semantics.
    """

    if not isinstance(collision_map, NumericCollisionMap):
        raise TypeError("V4 clearance requires NumericCollisionMap")
    start = np.asarray(position_xy_m, dtype=np.float64)
    if start.shape != (2,) or not np.isfinite(start).all():
        raise ValueError("V4 position_xy_m must contain two finite values")
    count = _positive_int(ray_count, "clearance ray count", 512)
    if count % 2:
        raise ValueError("V4 clearance ray count must be even")
    maximum = _finite_positive(max_range_m, "clearance max range")
    chunk_size = _positive_int(obstacle_chunk_size, "obstacle chunk size", 1_000_000)
    if collision_map.point_check(start).collision:
        raise ValueError("V4 cannot encode clearance from a colliding robot pose")

    directions = _ray_directions(yaw_degrees, count)
    lower = collision_map.room_min_xy_m + collision_map.robot_radius_m
    upper = collision_map.room_max_xy_m - collision_map.robot_radius_m
    boundary = np.full(count, maximum, dtype=np.float64)
    for axis in range(2):
        component = directions[:, axis]
        positive = component > 1e-12
        negative = component < -1e-12
        boundary[positive] = np.minimum(
            boundary[positive],
            (upper[axis] - start[axis]) / component[positive],
        )
        boundary[negative] = np.minimum(
            boundary[negative],
            (lower[axis] - start[axis]) / component[negative],
        )

    free = np.clip(boundary, 0.0, maximum)
    radius_squared = collision_map.inflated_radius_m**2
    points = collision_map.obstacle_points_xy_m.astype(np.float64, copy=False)
    for offset in range(0, len(points), chunk_size):
        relative = points[offset : offset + chunk_size] - start
        projection = relative @ directions.T
        radial_squared = np.sum(relative * relative, axis=1, keepdims=True)
        perpendicular_squared = np.maximum(radial_squared - projection * projection, 0.0)
        intersects = (projection >= 0.0) & (perpendicular_squared <= radius_squared)
        root = np.sqrt(np.maximum(radius_squared - perpendicular_squared, 0.0))
        entry = projection - root
        valid = intersects & (entry >= 0.0)
        candidates = np.where(valid, entry, np.inf)
        free = np.minimum(free, np.min(candidates, axis=0))

    normalized = np.clip(free / maximum, 0.0, 1.0).astype(np.float32)
    result = torch.from_numpy(normalized)
    if result.shape != (count,) or not torch.isfinite(result).all():
        raise RuntimeError("V4 clearance state is invalid")
    return result


def counterfactual_motion_collision_targets(
    clearance_state: torch.Tensor,
    *,
    max_range_m: float = CLEARANCE_MAX_RANGE_M,
    probe_distances_m: Sequence[float] = COLLISION_PROBE_DISTANCES_M,
) -> torch.Tensor:
    """Derive forward/backward collision targets for bounded hypothetical moves."""

    clearance = torch.as_tensor(clearance_state, dtype=torch.float32)
    if clearance.shape != (CLEARANCE_RAY_COUNT,) or not torch.isfinite(clearance).all():
        raise ValueError("V4 clearance state must have shape [24]")
    if torch.any((clearance < 0.0) | (clearance > 1.0)):
        raise ValueError("V4 clearance state must be normalized to [0,1]")
    maximum = _finite_positive(max_range_m, "clearance max range")
    probes = torch.tensor([float(value) for value in probe_distances_m], dtype=torch.float32)
    if (
        probes.shape != (len(COLLISION_PROBE_DISTANCES_M),)
        or not torch.isfinite(probes).all()
        or torch.any(probes <= 0.0)
        or torch.any(probes > maximum)
        or not bool(torch.all(probes[1:] > probes[:-1]))
    ):
        raise ValueError("V4 collision probe distances differ from the fixed contract")
    forward = clearance[0] * maximum
    backward = clearance[CLEARANCE_RAY_COUNT // 2] * maximum
    # Segment collision occurs at contact, hence >= rather than >.
    targets = torch.cat(((probes >= forward).float(), (probes >= backward).float()))
    if targets.shape != (COLLISION_RISK_DIM,):
        raise RuntimeError("V4 collision-risk target shape differs")
    return targets


class ClearanceAwareNavigationControllerV4(nn.Module):
    """V3 plus a trainable residual clearance branch and risk head."""

    def __init__(self, hidden_size: int, *, model_dim: int = 128) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.model_dim = _positive_int(model_dim, "model_dim", 2048)
        self.base = GroundedContinuousNavigationControllerV3(
            self.hidden_size, model_dim=self.model_dim
        )
        self.clearance_encoder = nn.Sequential(
            nn.Linear(CLEARANCE_RAY_COUNT + TARGET_STATE_DIM, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
        )
        self.action_delta = nn.Linear(self.model_dim, len(ACTION_NAMES))
        self.argument_delta = nn.Linear(self.model_dim, len(ACTION_NAMES))
        self.collision_risk_head = nn.Linear(self.model_dim, COLLISION_RISK_DIM)
        # Loading V3 and zeroing only the residual heads makes initialization
        # behaviorally identical to the sealed V3 policy.
        nn.init.zeros_(self.action_delta.weight)
        nn.init.zeros_(self.action_delta.bias)
        nn.init.zeros_(self.argument_delta.weight)
        nn.init.zeros_(self.argument_delta.bias)

    def initialize_from_v3(
        self, controller: GroundedContinuousNavigationControllerV3
    ) -> None:
        if (
            not isinstance(controller, GroundedContinuousNavigationControllerV3)
            or controller.hidden_size != self.hidden_size
            or controller.model_dim != self.model_dim
        ):
            raise ValueError("V4 initialization controller differs from its architecture")
        self.base.load_state_dict(controller.state_dict(), strict=True)

    def freeze_v3_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def forward(
        self,
        scene_prefix: torch.Tensor,
        robot_tokens: torch.Tensor,
        instruction_embedding: torch.Tensor,
        target_state: torch.Tensor,
        clearance_state: torch.Tensor,
        *,
        scene_batch_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = robot_tokens.shape[0] if robot_tokens.ndim == 3 else -1
        if clearance_state.shape != (batch, CLEARANCE_RAY_COUNT):
            raise ValueError("V4 clearance_state must have shape [B,24]")
        if not torch.isfinite(clearance_state).all() or torch.any(
            (clearance_state < 0.0) | (clearance_state > 1.0)
        ):
            raise ValueError("V4 clearance_state must be finite and normalized")
        base_logits, base_arguments = self.base(
            scene_prefix,
            robot_tokens,
            instruction_embedding,
            target_state,
            scene_batch_indices=scene_batch_indices,
        )
        clearance = self.clearance_encoder(
            torch.cat((clearance_state.float(), target_state.float()), dim=-1)
        )
        logits = base_logits + self.action_delta(clearance)
        base_pre_tanh = torch.atanh(torch.clamp(base_arguments, -0.999999, 0.999999))
        arguments = torch.tanh(base_pre_tanh + self.argument_delta(clearance))
        collision_risk_logits = self.collision_risk_head(clearance)
        if not all(
            torch.isfinite(value).all()
            for value in (logits, arguments, collision_risk_logits)
        ):
            raise RuntimeError("V4 controller output contains NaN or infinity")
        return logits, arguments, collision_risk_logits


def _validated_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(metadata)
    integers = {
        name: _positive_int(value.get(name), name)
        for name in (
            "hidden_size",
            "model_dim",
            "scene_token_count",
            "robot_token_count",
            "train_scene_count",
            "validation_scene_count",
        )
    }
    room = value.get("room_size_m")
    probes = value.get("collision_probe_distances_m")
    if not isinstance(room, list) or len(room) != 3:
        raise ValueError("V4 room size metadata differs")
    if not isinstance(probes, list) or probes != list(COLLISION_PROBE_DISTANCES_M):
        raise ValueError("V4 collision probe metadata differs")
    digest = value.get("training_dataset_sha256")
    preregistration_digest = value.get("preregistration_sha256")
    v3_digest = value.get("v3_initialization_weights_sha256")
    weights_digest = value.get("weights_sha256")
    if (
        set(value) != _METADATA_FIELDS
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("architecture") != ARCHITECTURE
        or value.get("target_state_dim") != TARGET_STATE_DIM
        or value.get("clearance_ray_count") != CLEARANCE_RAY_COUNT
        or value.get("clearance_max_range_m") != CLEARANCE_MAX_RANGE_M
        or value.get("collision_risk_dim") != COLLISION_RISK_DIM
        or value.get("action_names") != list(ACTION_NAMES)
        or value.get("grounding_feature_start") != GEMMA4_PROJECTED_START
        or value.get("grounding_feature_dim") != GEMMA4_PROJECTED_DIM
        or value.get("task_trained") is not True
        or not isinstance(value.get("model_id"), str)
        or not value["model_id"]
        or not isinstance(value.get("model_revision"), str)
        or not value["model_revision"]
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in (digest, preregistration_digest, v3_digest, weights_digest)
        )
        or any(
            value.get(field) is not True
            for field in (
                "scene_splits_disjoint",
                "complete_scene_prefix_required",
                "question_independent_static_scene_prefix_required",
                "every_scene_token_processed",
                "numeric_robot_tokens_required",
                "continuous_semantic_grounding_required",
                "all_map_voxels_scored_for_grounding",
                "numeric_clearance_state_required",
                "clearance_from_sanitized_geometry_only",
                "exact_collision_mask_required",
                "collision_interlock_required",
                "preregistered_single_arm",
            )
        )
        or value.get("unsafe_motion_fallback") != "highest_safe_nonterminal_action"
        or value.get("query_dependent_grounding_navigation_only") is not True
        or value.get("environmental_text_inputs") != []
        or value.get("oracle_inputs_at_runtime") is not False
        or value.get("runtime_required_files") != [
            "policy.safetensors",
            "runtime_metadata.json",
        ]
    ):
        raise ValueError("V4 checkpoint contract is not inference safe")
    value.update(integers)
    value["room_size_m"] = [_finite_positive(item, "room_size_m") for item in room]
    value["max_turn_degrees"] = _finite_positive(
        value.get("max_turn_degrees"), "max_turn_degrees"
    )
    value["max_move_m"] = _finite_positive(value.get("max_move_m"), "max_move_m")
    return value


def save_navigation_policy_v4_checkpoint(
    destination: str | Path,
    controller: ClearanceAwareNavigationControllerV4,
    *,
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    root = _rooted(destination)
    _reject_runtime_path(root)
    if root.exists():
        raise FileExistsError(f"V4 checkpoint already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights_path = temporary / "policy.safetensors"
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in controller.state_dict().items()
            },
            str(weights_path),
        )
        metadata = _validated_metadata(
            {
                **dict(runtime_metadata),
                "schema_version": SCHEMA_VERSION,
                "architecture": ARCHITECTURE,
                "hidden_size": controller.hidden_size,
                "model_dim": controller.model_dim,
                "target_state_dim": TARGET_STATE_DIM,
                "clearance_ray_count": CLEARANCE_RAY_COUNT,
                "clearance_max_range_m": CLEARANCE_MAX_RANGE_M,
                "collision_probe_distances_m": list(COLLISION_PROBE_DISTANCES_M),
                "collision_risk_dim": COLLISION_RISK_DIM,
                "action_names": list(ACTION_NAMES),
                "weights_sha256": _sha256_file(weights_path),
            }
        )
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


def load_navigation_policy_v4_checkpoint(
    checkpoint: str | Path,
    *,
    expected_hidden_size: int,
    expected_model_id: str,
    expected_model_revision: str,
    device: torch.device | str = "cpu",
    audit: FileAccessAudit | None = None,
) -> tuple[ClearanceAwareNavigationControllerV4, dict[str, Any]]:
    root = _rooted(checkpoint)
    _reject_runtime_path(root)
    if not root.is_dir() or {entry.name for entry in root.iterdir()} != _FILES:
        raise ValueError("V4 checkpoint must contain exactly two files")
    weights_path = root / "policy.safetensors"
    metadata_path = root / "runtime_metadata.json"
    if any(path.is_symlink() or not path.is_file() for path in (weights_path, metadata_path)):
        raise ValueError("V4 checkpoint entries must be regular files")
    if audit is not None:
        audit.record(weights_path)
        audit.record(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("V4 runtime metadata must be an object")
    metadata = _validated_metadata(metadata)
    if _sha256_file(weights_path) != metadata["weights_sha256"]:
        raise ValueError("V4 checkpoint weights hash differs")
    if (
        metadata["hidden_size"] != expected_hidden_size
        or metadata["model_id"] != expected_model_id
        or metadata["model_revision"] != expected_model_revision
    ):
        raise ValueError("V4 checkpoint local-model binding differs")
    controller = ClearanceAwareNavigationControllerV4(
        metadata["hidden_size"], model_dim=metadata["model_dim"]
    )
    state = load_file(str(weights_path), device="cpu")
    expected = controller.state_dict()
    if set(state) != set(expected) or any(
        state[name].shape != expected[name].shape for name in expected
    ):
        raise ValueError("V4 checkpoint tensor inventory differs")
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("V4 checkpoint contains NaN or infinity")
    controller.load_state_dict(state, strict=True)
    controller.to(device).eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller, metadata


def _movement_collision(runtime: Any, call: Mapping[str, Any]) -> bool:
    name = call.get("tool")
    if name not in {"move_forward", "move_backward"}:
        return False
    simulator = getattr(runtime, "simulator", None)
    collision_map = getattr(simulator, "collision_map", None)
    state = getattr(simulator, "state", None)
    if not isinstance(collision_map, NumericCollisionMap) or state is None:
        return True
    start = np.asarray(getattr(state, "position_xy_m", None), dtype=np.float64)
    yaw = float(getattr(state, "body_yaw_degrees", math.nan))
    arguments = call.get("arguments")
    distance = (
        float(arguments.get("distance_meters", math.nan))
        if isinstance(arguments, Mapping)
        else math.nan
    )
    if (
        start.shape != (2,)
        or not np.isfinite(start).all()
        or not math.isfinite(yaw)
        or not math.isfinite(distance)
        or distance < 0.0
    ):
        return True
    direction = np.asarray(
        [-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))], dtype=np.float64
    )
    if name == "move_backward":
        direction *= -1.0
    return bool(collision_map.segment_check(start, start + distance * direction).collision)


@dataclass(frozen=True)
class CollisionMaskedSelection:
    raw_action_index: int
    selected_action_index: int
    raw_call: dict[str, Any]
    selected_call: dict[str, Any]
    unsafe_motion_masked: bool


def select_highest_safe_nonterminal_action(
    runtime: Any,
    logits: torch.Tensor,
    arguments: torch.Tensor,
    *,
    max_turn_degrees: float,
    max_move_m: float,
) -> CollisionMaskedSelection:
    """Apply the exact collision mask without any semantic heuristic."""

    scores = torch.as_tensor(logits, dtype=torch.float32).detach().cpu()
    values = torch.as_tensor(arguments, dtype=torch.float32).detach().cpu()
    if scores.shape != (len(ACTION_NAMES),) or values.shape != (len(ACTION_NAMES),):
        raise ValueError("V4 action selection expects one action and argument vector")
    if not torch.isfinite(scores).all() or not torch.isfinite(values).all():
        raise ValueError("V4 action selection received nonfinite controller output")
    ranked = torch.argsort(scores, descending=True, stable=True).tolist()
    raw_index = int(ranked[0])

    def call_for(index: int) -> dict[str, Any]:
        return tool_call_from_prediction(
            index,
            float(values[index]),
            max_turn_degrees=max_turn_degrees,
            max_move_m=max_move_m,
        )

    raw_call = call_for(raw_index)
    if raw_call["tool"] not in {"move_forward", "move_backward"} or not _movement_collision(
        runtime, raw_call
    ):
        return CollisionMaskedSelection(
            raw_action_index=raw_index,
            selected_action_index=raw_index,
            raw_call=raw_call,
            selected_call=raw_call,
            unsafe_motion_masked=False,
        )

    # The model did not request completion; do not silently reinterpret a
    # blocked movement as completion.  Scan/turn are intrinsically nonmoving;
    # alternative movements are admitted only after exact swept-segment checks.
    for index in ranked[1:]:
        candidate_index = int(index)
        if ACTION_NAMES[candidate_index] == "stop":
            continue
        candidate = call_for(candidate_index)
        if not _movement_collision(runtime, candidate):
            return CollisionMaskedSelection(
                raw_action_index=raw_index,
                selected_action_index=candidate_index,
                raw_call=raw_call,
                selected_call=candidate,
                unsafe_motion_masked=True,
            )
    raise RuntimeError("V4 found no safe nonterminal alternative to an unsafe movement")


def _final_collision_interlock(runtime: Any, call: dict[str, Any]) -> dict[str, Any]:
    """Defense in depth for a state change or implementation regression."""

    if _movement_collision(runtime, call):
        return {"tool": "stop", "arguments": {}}
    return call


def _active_map_path(runtime: Any) -> Path:
    updater = getattr(runtime, "map_updater", None)
    if updater is None:
        raise TypeError("V4 runtime has no semantic map updater")
    persistent = Path(updater.persistent_map_path)
    base = Path(updater.base_map_path)
    selected = persistent if persistent.is_file() else base
    rooted = _rooted(selected)
    _reject_runtime_path(rooted)
    if not rooted.is_file():
        raise FileNotFoundError("V4 active semantic map is unavailable")
    return rooted


class SemanticClearanceActionBackendV4:
    """Runtime backend using complete scene, target, state, and free space."""

    def __init__(
        self,
        runtime: Any,
        controller: ClearanceAwareNavigationControllerV4,
        metadata: Mapping[str, Any],
        config: Mapping[str, Any],
        *,
        text_encoder: ContinuousTextEncoder | None = None,
    ) -> None:
        self.runtime = runtime
        self.controller = controller.eval()
        self.metadata = _validated_metadata(metadata)
        self.config = dict(config)
        prefix_refresher = getattr(runtime, "prefix_refresher", None)
        wrapped = getattr(prefix_refresher, "runtime", None)
        self.base = getattr(wrapped, "base", wrapped)
        if self.base is None or getattr(self.base, "language", None) is None:
            raise TypeError("V4 backend requires a loaded local language model")
        if self.base.language.hidden_size != controller.hidden_size:
            raise ValueError("V4 controller width differs from local Gemma")
        self.text_encoder = text_encoder or GemmaProjectedTextEncoder.from_config(config)
        if self.text_encoder.output_dim != int(self.metadata["grounding_feature_dim"]):
            raise ValueError("V4 grounding text encoder width differs")
        self.last_grounding: dict[str, Any] | None = None
        self.last_collision_selection: CollisionMaskedSelection | None = None

    def _ground(
        self,
        target_text: str | None,
        *,
        context: ContinuousActionContext | None = None,
    ) -> tuple[torch.Tensor, ContinuousSemanticGrounding | None]:
        simulator = self.runtime.simulator
        room = self.metadata["room_size_m"]
        minimum = torch.tensor([-room[0] / 2.0, -room[1] / 2.0, 0.0])
        maximum = torch.tensor([room[0] / 2.0, room[1] / 2.0, room[2]])
        state_features = (
            robot_state_vector(simulator.numeric_state(), minimum, maximum)
            if context is None
            else context.state_features
        )
        if target_text is None:
            self.last_grounding = {
                "target_available": False,
                "query_embedding_sha256": None,
                "map_sha256": (
                    self.runtime.prefix_binding().get("map_sha256")
                    if context is None
                    else context.map_sha256
                ),
                "continuous_context_verified": context is not None,
            }
            if context is not None:
                self.last_grounding.update(
                    {
                        "active_prefix_sha256": context.binding[
                            "active_prefix_sha256"
                        ],
                        "scene_prefix_sha256": context.binding[
                            "scene_prefix_sha256"
                        ],
                        "robot_state_sha256": context.binding[
                            "robot_state_sha256"
                        ],
                        "robot_tokens_sha256": context.binding[
                            "robot_tokens_sha256"
                        ],
                    }
                )
            return (
                grounded_target_state(
                    torch.zeros(3), state_features, torch.tensor(0.0), room_size_m=room
                ),
                None,
            )
        grounder = ContinuousSemanticTargetGrounder(
            _active_map_path(self.runtime),
            self.text_encoder,
            room_size_m=room,
            feature_start=int(self.metadata["grounding_feature_start"]),
            feature_dim=int(self.metadata["grounding_feature_dim"]),
        )
        grounding = grounder.ground(target_text)
        if context is None:
            if grounding.scored_voxels != len(grounder.xyz):
                raise RuntimeError("V4 grounding did not score the complete voxel map")
        else:
            require_grounding_map_binding(
                context,
                grounding_map_sha256=grounding.map_sha256,
                scored_voxels=grounding.scored_voxels,
                available_voxels=len(grounder.xyz),
            )
        if grounder.scene_id != simulator.state.scene_id:
            raise RuntimeError("V4 grounding map scene differs from robot state")
        target = grounded_target_state(
            torch.tensor(grounding.target_xyz_m),
            state_features,
            torch.tensor(1.0),
            room_size_m=room,
        )
        self.last_grounding = {
            "target_available": True,
            "target_state_sha256": hashlib.sha256(
                target.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            "query_embedding_sha256": grounding.query_embedding_sha256,
            "map_sha256": grounding.map_sha256,
            "scored_voxels": grounding.scored_voxels,
            "eligible_voxels": grounding.eligible_voxels,
            "continuous_context_verified": context is not None,
        }
        if context is not None:
            self.last_grounding.update(
                {
                    "active_prefix_sha256": context.binding["active_prefix_sha256"],
                    "scene_prefix_sha256": context.binding["scene_prefix_sha256"],
                    "robot_state_sha256": context.binding["robot_state_sha256"],
                    "robot_tokens_sha256": context.binding["robot_tokens_sha256"],
                }
            )
        return target, grounding

    @torch.inference_mode()
    def generate(self, instruction: str, *, correction_code: str | None) -> GeneratedToolProposal:
        del correction_code
        context = capture_continuous_action_context(
            self.runtime,
            self.metadata["room_size_m"],
        )
        active = context.active_prefix
        binding = context.binding
        observed_hash = prefix_sha256(active)
        if binding.get("active_prefix_sha256") != observed_hash:
            raise RuntimeError("V4 active prefix differs from runtime binding")
        scene, robot = split_active_prefix(
            active,
            scene_token_count=int(self.metadata["scene_token_count"]),
            robot_token_count=int(self.metadata["robot_token_count"]),
        )
        literal = _literal_instruction(instruction)
        target_text = target_text_from_navigation_instruction(literal)
        target_state, _grounding = self._ground(target_text, context=context)
        simulator = self.runtime.simulator
        clearance_state = robot_frame_clearance_state(
            simulator.collision_map,
            context.numeric_state.position_m[:2],
            context.numeric_state.body_yaw_degrees,
            ray_count=int(self.metadata["clearance_ray_count"]),
            max_range_m=float(self.metadata["clearance_max_range_m"]),
        ).unsqueeze(0)
        encoded = self.base.language.tokenizer(
            literal, add_special_tokens=False, return_tensors="pt"
        )
        token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2 or not token_ids.numel():
            raise ValueError("V4 tokenizer returned no instruction tokens")
        embedding_layer = self.base.language.model.get_input_embeddings()
        instruction_embedding = embedding_layer(token_ids.to(embedding_layer.weight.device))
        device = next(self.controller.parameters()).device
        logits, arguments, _risk_logits = self.controller(
            scene.to(device),
            robot.to(device),
            instruction_embedding.float().mean(dim=1).to(device),
            target_state.to(device),
            clearance_state.to(device),
        )
        selection = select_highest_safe_nonterminal_action(
            self.runtime,
            logits[0],
            arguments[0],
            max_turn_degrees=float(self.metadata["max_turn_degrees"]),
            max_move_m=float(self.metadata["max_move_m"]),
        )
        self.last_collision_selection = selection
        call = _final_collision_interlock(self.runtime, selection.selected_call)
        scene_hash = binding.get("scene_prefix_sha256")
        robot_hash = binding.get("robot_tokens_sha256")
        return GeneratedToolProposal(
            text=json.dumps(call, sort_keys=True, separators=(",", ":"), allow_nan=False),
            active_prefix_sha256=observed_hash,
            scene_prefix_sha256=scene_hash if isinstance(scene_hash, str) else "",
            robot_tokens_sha256=robot_hash if isinstance(robot_hash, str) else None,
            local_inference=True,
            used_continuous_scene_prefix=True,
            used_continuous_robot_tokens=True,
            training_status=TRAINING_STATUS,
        )


__all__ = [
    "ARCHITECTURE",
    "CLEARANCE_MAX_RANGE_M",
    "CLEARANCE_RAY_COUNT",
    "COLLISION_PROBE_DISTANCES_M",
    "COLLISION_RISK_DIM",
    "SCHEMA_VERSION",
    "TRAINING_STATUS",
    "ClearanceAwareNavigationControllerV4",
    "CollisionMaskedSelection",
    "SemanticClearanceActionBackendV4",
    "counterfactual_motion_collision_targets",
    "load_navigation_policy_v4_checkpoint",
    "robot_frame_clearance_state",
    "save_navigation_policy_v4_checkpoint",
    "select_highest_safe_nonterminal_action",
]

"""Runtime-only closed loop for actual-Gemma waypoint decisions.

Gemma chooses every waypoint, robot-relative turn delta, and goal completion.  This
module performs only exact coordinate conversion, provenance binding, bounded
primitive dispatch, and rejection accounting.  It contains no target
grounder, route planner, recovery planner, action mask, heuristic completion
test, or deterministic fallback.

A rejected proposal is preserved as a numeric history row with ``success=0``
and the unchanged pose.  V2 rows also carry a goal-local four-value progress
ledger derived only from exact numeric action receipts.  The next iteration
reruns the complete Gemma causal policy with the refreshed continuous
scene/robot prefix.  Running out of steps is a failure; it never becomes a
synthetic STOP.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.gemma_runtime_binding import (
    gemma_runtime_binding_sha256,
    validate_gemma_runtime_binding,
)
from semantic_3d_chat.robot.gemma_waypoint_policy import (
    ACTION_NAMES,
    ActualGemmaWaypointPolicy,
    GemmaMotionAction,
    GemmaWaypointDecision,
)
from semantic_3d_chat.robot.llm_tool_policy import validate_tool_call_text
from semantic_3d_chat.robot.model_bound_action import (
    ModelBoundActionExecutor,
    ModelBoundToolCall,
    bind_model_tool_call,
)
from semantic_3d_chat.robot.state_encoder import NumericRobotState
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V1,
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION,
    HISTORY_PARAMETERIZATION_V1,
    HISTORY_PARAMETERIZATION_V2,
    WaypointGoalProgressLedger,
    encode_waypoint_history_transition,
    encode_waypoint_history_transition_v2,
)

CHECKPOINT_SCHEMA_V1: Final[str] = "semantic_3d_chat.gemma_waypoint_checkpoint.v3"
CHECKPOINT_SCHEMA_V2: Final[str] = "semantic_3d_chat.gemma_waypoint_checkpoint.v4"
# Backward-compatible alias retained for all existing V1 tests and checkpoints.
CHECKPOINT_SCHEMA: Final[str] = CHECKPOINT_SCHEMA_V1
CHECKPOINT_ARCHITECTURE: Final[str] = "gemma_final_hidden_numeric_waypoint_policy"
HEADING_PARAMETERIZATION: Final[str] = "robot_relative_bounded_scalar_tanh"
GOAL_RESULT_SCHEMA: Final[str] = "semantic_3d_chat.gemma_waypoint_goal.v1"
STEP_RECEIPT_SCHEMA: Final[str] = "semantic_3d_chat.gemma_waypoint_step.v2"
CHECKPOINT_FILES: Final[frozenset[str]] = frozenset(
    {"policy.safetensors", "runtime_metadata.json"}
)
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_BLOCKED_PATH_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "training", "evaluation", "scorer_only", "scorer-only"}
)
_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "architecture",
        "action_names",
        "weights_sha256",
        "saved_controller_tensors_only",
        "frozen_gemma_weights_saved",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
        "runtime_required_files",
        "model_id",
        "model_revision",
        "dataset_sha256",
        "training_traces_sha256",
        "training_sample_count",
        "validation_sample_count",
        "training_scene_count",
        "validation_scene_count",
        "scene_splits_disjoint",
        "scene_token_count",
        "robot_token_count",
        "hidden_size",
        "state_dim",
        "history_dim",
        "history_parameterization",
        "max_history_tokens",
        "context_token_count",
        "head_hidden_dim",
        "max_waypoint_step_m",
        "heading_parameterization",
        "max_turn_delta_degrees",
        "history_projector_initialization_seed",
        "numeric_heads_initialization_seed",
        "context_projection_frozen_during_training",
        "actual_gemma_causal_forward",
        "gemma_output_hidden_states",
        "complete_scene_prefix_required",
        "every_scene_token_processed",
        "numeric_state_and_history_required",
        "deterministic_route_planner_allowed_at_runtime",
        "model_selects_every_waypoint_and_heading",
        "gemma_runtime_binding",
        "gemma_runtime_binding_sha256",
    }
)
_OPTIONAL_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {"action_refit_l2_weight"}
)
_SUPPORTED_CHECKPOINT_HISTORY_CONTRACTS: Final[
    frozenset[tuple[str, int, str]]
] = frozenset(
    {
        (
            CHECKPOINT_SCHEMA_V1,
            HISTORY_FEATURE_DIM_V1,
            HISTORY_PARAMETERIZATION_V1,
        ),
        (
            CHECKPOINT_SCHEMA_V2,
            HISTORY_FEATURE_DIM_V2,
            HISTORY_PARAMETERIZATION_V2,
        ),
    }
)
_SUPPORTED_RUNTIME_HISTORY_CONTRACTS: Final[frozenset[tuple[int, str]]] = frozenset(
    (dimension, parameterization)
    for _schema, dimension, parameterization in _SUPPORTED_CHECKPOINT_HISTORY_CONTRACTS
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _safe_checkpoint_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    result = Path(os.path.abspath(rooted))
    if _BLOCKED_PATH_COMPONENTS.intersection(part.casefold() for part in result.parts):
        raise ValueError("Waypoint runtime checkpoint entered a protected data tree")
    current = Path(result.anchor)
    for component in result.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Waypoint runtime checkpoint path cannot contain symlinks")
    return result


def _unique_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate waypoint checkpoint field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Waypoint runtime metadata is invalid JSON") from error
    observed_fields = set(value) if isinstance(value, dict) else set()
    if not isinstance(value, dict) or observed_fields not in {
        _METADATA_FIELDS,
        _METADATA_FIELDS | _OPTIONAL_METADATA_FIELDS,
    }:
        raise ValueError("Waypoint runtime metadata fields changed")
    return value


def _validated_checkpoint_metadata(
    value: Mapping[str, Any],
    *,
    expected_model_id: str,
    expected_model_revision: str,
    expected_gemma_runtime_binding: Mapping[str, Any],
    backend: Gemma4PrefixBackend,
) -> dict[str, Any]:
    metadata = dict(value)
    counts = {
        name: _positive_int(metadata.get(name), name)
        for name in (
            "training_sample_count",
            "validation_sample_count",
            "training_scene_count",
            "validation_scene_count",
            "scene_token_count",
            "robot_token_count",
            "hidden_size",
            "state_dim",
            "history_dim",
            "max_history_tokens",
            "context_token_count",
            "head_hidden_dim",
            "history_projector_initialization_seed",
            "numeric_heads_initialization_seed",
        )
    }
    dataset_digest = _require_sha256(metadata.get("dataset_sha256"), "dataset_sha256")
    traces_digest = _require_sha256(
        metadata.get("training_traces_sha256"), "training_traces_sha256"
    )
    weights_digest = _require_sha256(metadata.get("weights_sha256"), "weights_sha256")
    expected_binding = validate_gemma_runtime_binding(
        expected_gemma_runtime_binding
    )
    required_true = (
        "saved_controller_tensors_only",
        "scene_splits_disjoint",
        "actual_gemma_causal_forward",
        "gemma_output_hidden_states",
        "complete_scene_prefix_required",
        "every_scene_token_processed",
        "numeric_state_and_history_required",
        "model_selects_every_waypoint_and_heading",
        "context_projection_frozen_during_training",
    )
    history_contract = (
        metadata.get("schema"),
        counts["history_dim"],
        metadata.get("history_parameterization"),
    )
    if (
        history_contract not in _SUPPORTED_CHECKPOINT_HISTORY_CONTRACTS
        or metadata.get("architecture") != CHECKPOINT_ARCHITECTURE
        or metadata.get("heading_parameterization") != HEADING_PARAMETERIZATION
        or metadata.get("action_names") != list(ACTION_NAMES)
        or metadata.get("runtime_required_files") != sorted(CHECKPOINT_FILES)
        or metadata.get("frozen_gemma_weights_saved") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("oracle_inputs_at_runtime") is not False
        or any(metadata.get(field) is not True for field in required_true)
        or metadata.get("deterministic_route_planner_allowed_at_runtime") is not False
        or metadata.get("model_id") != expected_model_id
        or metadata.get("model_revision") != expected_model_revision
        or metadata.get("gemma_runtime_binding") != expected_binding
        or metadata.get("gemma_runtime_binding_sha256")
        != gemma_runtime_binding_sha256(expected_binding)
        or backend.model_revision != expected_model_revision
        or counts["hidden_size"] != backend.hidden_size
        or counts["scene_token_count"] < 3
        or counts["context_token_count"] != 1
    ):
        raise ValueError("Waypoint runtime checkpoint contract mismatch")
    metadata["dataset_sha256"] = dataset_digest
    metadata["training_traces_sha256"] = traces_digest
    metadata["weights_sha256"] = weights_digest
    metadata["max_waypoint_step_m"] = _finite_positive(
        metadata.get("max_waypoint_step_m"), "max_waypoint_step_m"
    )
    metadata["max_turn_delta_degrees"] = _finite_positive(
        metadata.get("max_turn_delta_degrees"), "max_turn_delta_degrees"
    )
    if "action_refit_l2_weight" in metadata:
        metadata["action_refit_l2_weight"] = _finite_nonnegative(
            metadata["action_refit_l2_weight"], "action_refit_l2_weight"
        )
    if metadata["max_turn_delta_degrees"] > 180.0:
        raise ValueError("Waypoint runtime checkpoint turn bound exceeds 180 degrees")
    return metadata


@dataclass(frozen=True, slots=True)
class LoadedGemmaWaypointPolicy:
    """Sanitized controller weights bound to the already-loaded local Gemma."""

    policy: ActualGemmaWaypointPolicy
    prefix_backend: Gemma4PrefixBackend
    tokenizer: Any
    metadata: dict[str, Any]
    checkpoint: Path
    checkpoint_sha256: str


def load_gemma_waypoint_policy_checkpoint(
    checkpoint: str | Path,
    *,
    prefix_backend: Gemma4PrefixBackend,
    expected_model_id: str,
    expected_model_revision: str,
    expected_gemma_runtime_binding: Mapping[str, Any],
    tokenizer: Any | None = None,
    audit: Any | None = None,
) -> LoadedGemmaWaypointPolicy:
    """Load exactly two inference-safe files and bind them to local Gemma.

    Frozen Gemma weights are never read from this controller directory.  The
    caller supplies the already-loaded local backend, whose revision and hidden
    size must match the checkpoint metadata.
    """

    if not isinstance(prefix_backend, Gemma4PrefixBackend):
        raise TypeError("Waypoint loader requires a loaded Gemma4PrefixBackend")
    if not isinstance(expected_model_id, str) or not expected_model_id:
        raise ValueError("expected_model_id must be nonempty")
    if not isinstance(expected_model_revision, str) or not expected_model_revision:
        raise ValueError("expected_model_revision must be nonempty")
    root = _safe_checkpoint_root(checkpoint)
    if not root.is_dir() or {entry.name for entry in root.iterdir()} != CHECKPOINT_FILES:
        raise ValueError("Waypoint checkpoint must contain exactly two runtime files")
    weights = root / "policy.safetensors"
    metadata_path = root / "runtime_metadata.json"
    if any(entry.is_symlink() or not entry.is_file() for entry in (weights, metadata_path)):
        raise ValueError("Waypoint checkpoint entries must be regular files")
    if audit is not None:
        record = getattr(audit, "record", None)
        if not callable(record):
            raise TypeError("Waypoint checkpoint audit has no record method")
        record(metadata_path)
    metadata = _validated_checkpoint_metadata(
        _unique_json(metadata_path),
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
        expected_gemma_runtime_binding=expected_gemma_runtime_binding,
        backend=prefix_backend,
    )
    if _sha256_file(weights) != metadata["weights_sha256"]:
        raise ValueError("Waypoint checkpoint weights changed")
    if audit is not None:
        audit.record(weights)
    state = load_file(str(weights), device="cpu")
    waypoint_projection = state.get("numeric_heads.waypoint.0.weight")
    history_projection = state.get("history_projector.network.1.weight")
    if (
        not isinstance(waypoint_projection, torch.Tensor)
        or waypoint_projection.ndim != 2
        or waypoint_projection.shape[1] != metadata["hidden_size"]
        or not isinstance(history_projection, torch.Tensor)
        or history_projection.ndim != 2
        or history_projection.shape[0] != waypoint_projection.shape[0]
        or history_projection.shape[1] != metadata["history_dim"]
    ):
        raise ValueError("Waypoint checkpoint projection shapes differ")
    head_hidden_dim = int(waypoint_projection.shape[0])
    if head_hidden_dim != int(metadata["head_hidden_dim"]):
        raise ValueError("Waypoint checkpoint head dimension differs")
    policy = ActualGemmaWaypointPolicy(
        hidden_size=int(metadata["hidden_size"]),
        scene_token_count=int(metadata["scene_token_count"]),
        robot_token_count=int(metadata["robot_token_count"]),
        history_feature_dim=int(metadata["history_dim"]),
        max_history_tokens=int(metadata["max_history_tokens"]),
        head_hidden_dim=head_hidden_dim,
        max_waypoint_step_m=float(metadata["max_waypoint_step_m"]),
        max_turn_delta_degrees=float(metadata["max_turn_delta_degrees"]),
        context_initialization_seed=int(
            metadata["history_projector_initialization_seed"]
        ),
        head_initialization_seed=int(metadata["numeric_heads_initialization_seed"]),
        freeze_context_projection=True,
    )
    expected_state = policy.state_dict()
    if set(state) != set(expected_state):
        raise ValueError("Waypoint checkpoint tensor inventory changed")
    for name, tensor in state.items():
        if tensor.shape != expected_state[name].shape or not bool(
            torch.isfinite(tensor.float()).all()
        ):
            raise ValueError(f"Waypoint checkpoint tensor is invalid: {name}")
    policy.load_state_dict(state, strict=True)
    embedding = prefix_backend.model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("Loaded Gemma backend has no input-embedding device")
    policy = policy.to(device=weight.device, dtype=torch.float32).eval()
    if policy.context_projection_frozen is not True:
        raise RuntimeError("Waypoint context projection must remain frozen at runtime")
    selected_tokenizer = tokenizer if tokenizer is not None else prefix_backend.tokenizer
    if selected_tokenizer is None:
        raise TypeError("Waypoint runtime requires the loaded Gemma tokenizer")
    return LoadedGemmaWaypointPolicy(
        policy=policy,
        prefix_backend=prefix_backend,
        tokenizer=selected_tokenizer,
        metadata=metadata,
        checkpoint=root,
        checkpoint_sha256=str(metadata["weights_sha256"]),
    )


def robot_delta_to_world_xy(
    position_xy_m: Sequence[float],
    body_yaw_degrees: float,
    delta_robot_m: Sequence[float],
) -> tuple[float, float]:
    """Exactly transform model [right, forward] delta into canonical world XY."""

    position = torch.as_tensor(position_xy_m, dtype=torch.float64)
    delta = torch.as_tensor(delta_robot_m, dtype=torch.float64)
    if (
        position.shape != (2,)
        or delta.shape != (2,)
        or not bool(torch.isfinite(position).all())
        or not bool(torch.isfinite(delta).all())
        or isinstance(body_yaw_degrees, bool)
        or not isinstance(body_yaw_degrees, (int, float))
        or not math.isfinite(float(body_yaw_degrees))
    ):
        raise ValueError("Waypoint transform inputs must be finite 2D values")
    radians = math.radians(float(body_yaw_degrees))
    right = torch.tensor([math.cos(radians), math.sin(radians)], dtype=torch.float64)
    forward = torch.tensor([-math.sin(radians), math.cos(radians)], dtype=torch.float64)
    target = position + delta[0] * right + delta[1] * forward
    return float(target[0]), float(target[1])


def _normalized_degrees(value: float) -> float:
    result = (float(value) + 180.0) % 360.0 - 180.0
    return 0.0 if abs(result) < 1e-12 else result


def _state_mapping(
    state: Mapping[str, Any] | NumericRobotState,
    *,
    scene_id: str,
) -> dict[str, Any]:
    """Normalize the atomic typed state without taking a second snapshot."""

    if isinstance(state, NumericRobotState):
        value = asdict(state)
    elif isinstance(state, Mapping):
        value = dict(state)
    else:
        raise TypeError("Waypoint runtime returned invalid numeric robot state")
    observed_scene = value.get("scene_id")
    if observed_scene is not None and observed_scene != scene_id:
        raise RuntimeError("Waypoint numeric state differs from its opaque scene binding")
    value["scene_id"] = scene_id
    return value


def _state_pose(state: Mapping[str, Any]) -> tuple[float, float, float]:
    position = state.get("position_m")
    yaw = state.get("body_yaw_degrees")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) < 2
        or isinstance(yaw, bool)
    ):
        raise ValueError("Waypoint runtime returned an invalid numeric pose")
    try:
        values = (float(position[0]), float(position[1]), float(yaw))
    except (TypeError, ValueError) as error:
        raise ValueError("Waypoint runtime returned an invalid numeric pose") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Waypoint runtime returned a non-finite numeric pose")
    return values


def _binding_hashes(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: _require_sha256(value.get(name), name)
        for name in (
            "active_prefix_sha256",
            "scene_prefix_sha256",
            "robot_tokens_sha256",
        )
    }


def _static_map_identity(value: Mapping[str, Any]) -> tuple[str, int, str, int, int]:
    """Validate the immutable numeric-map identity behind a waypoint prefix."""

    scene_id = value.get("scene_id")
    map_version = value.get("map_version")
    map_sha256 = value.get("map_sha256")
    source_voxels = value.get("source_voxels")
    processed_voxels = value.get("processed_voxels")
    if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
        raise ValueError("Waypoint runtime binding has no opaque scene ID")
    if (
        isinstance(map_version, bool)
        or not isinstance(map_version, int)
        or map_version < 0
    ):
        raise ValueError("Waypoint runtime binding has an invalid map version")
    map_digest = _require_sha256(map_sha256, "map_sha256")
    for name, count in (
        ("source_voxels", source_voxels),
        ("processed_voxels", processed_voxels),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Waypoint runtime binding has invalid {name}")
    assert isinstance(source_voxels, int) and isinstance(processed_voxels, int)
    if processed_voxels > source_voxels:
        raise ValueError("Waypoint runtime processed more voxels than its numeric source map")
    return scene_id, map_version, map_digest, source_voxels, processed_voxels


def _validate_active_prefix_partition(
    active: torch.Tensor,
    binding: Mapping[str, Any],
    *,
    scene_token_count: int,
    robot_token_count: int,
) -> None:
    """Reconstruct and authenticate the fixed-scene and numeric-robot slices.

    Robot tokens are inserted immediately before the scene-end boundary.  Hashing
    both slices independently prevents a runtime bug from replacing scene tokens
    while merely issuing a fresh active-prefix hash.
    """

    content_end = scene_token_count - 1
    robot_end = content_end + robot_token_count
    scene_prefix = torch.cat(
        (active[:, :content_end], active[:, robot_end : robot_end + 1]),
        dim=1,
    )
    robot_tokens = active[:, content_end:robot_end]
    if tuple(scene_prefix.shape[1:]) != (scene_token_count, active.shape[2]):
        raise RuntimeError("Waypoint runtime could not reconstruct the complete scene prefix")
    if tuple(robot_tokens.shape[1:]) != (robot_token_count, active.shape[2]):
        raise RuntimeError("Waypoint runtime could not reconstruct numeric robot tokens")
    if prefix_sha256(scene_prefix) != binding.get("scene_prefix_sha256"):
        raise RuntimeError("Active waypoint prefix replaced part of the fixed scene tensor")
    if prefix_sha256(robot_tokens) != binding.get("robot_tokens_sha256"):
        raise RuntimeError("Active waypoint prefix robot tokens differ from their binding")


def _decision_tensor_sha256(decision: GemmaWaypointDecision) -> str:
    payload = {
        "action": decision.action.value,
        "action_logits": list(decision.action_logits),
        "action_probabilities": list(decision.action_probabilities),
        "waypoint_delta_robot_m": list(decision.waypoint_delta_robot_m),
        "turn_delta_degrees": decision.turn_delta_degrees,
    }
    return _sha256_text(_canonical_json(payload))


def _attempted_rejection(
    *,
    error_code: str,
    tool: str,
    arguments: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "semantic_3d_chat.gemma_waypoint_rejection.v1",
        "success": False,
        "executed": False,
        "error_code": error_code,
        "checkpoint_sha256": checkpoint_sha256,
        "model_tool": tool,
        "model_arguments": dict(arguments),
        "executed_tool": None,
        "executed_arguments": None,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
        "numeric_tool_receipt": None,
    }


class _GoalSettlementRuntime:
    """Delegate motion while making model STOP goal-scoped, not episode-wide."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self.goal_settled = False

    @property
    def simulator(self) -> Any:
        return self._runtime.simulator

    def prefix_binding(self) -> Mapping[str, Any]:
        return self._runtime.prefix_binding()

    def get_robot_state(self) -> Mapping[str, Any]:
        return self._runtime.get_robot_state()

    def turn(self, angle_degrees: float) -> Mapping[str, Any]:
        return self._runtime.turn(angle_degrees)

    def move_to(self, x: float, y: float) -> Mapping[str, Any]:
        return self._runtime.move_to(x, y)

    def stop(self) -> dict[str, Any]:
        # This method is reachable only through an authenticated exact STOP
        # proposal. It settles the conversational goal without touching the
        # simulator's persistent `stopped` latch.
        state = self._runtime.get_robot_state()
        if not isinstance(state, Mapping):
            raise TypeError("Waypoint goal settlement lacks numeric robot state")
        self.goal_settled = True
        return {
            **dict(state),
            "success": True,
            "error_code": None,
            "goal_settled": True,
        }


@dataclass(frozen=True, slots=True)
class GemmaWaypointStepReceipt:
    step: int
    decision_id: str
    action: str
    action_logits: tuple[float, float, float]
    action_probabilities: tuple[float, float, float]
    decision_tensor_sha256: str
    instruction_sha256: str
    active_prefix_sha256: str
    scene_prefix_sha256: str
    robot_tokens_sha256: str
    checkpoint_sha256: str
    scene_token_count: int
    robot_token_count: int
    history_token_count: int
    prompt_token_count: int
    decision_position: int
    waypoint_delta_robot_m: tuple[float, float]
    turn_delta_degrees: float
    desired_heading_degrees: float
    primitive_tool: str
    primitive_arguments: dict[str, float]
    bound_proposal: dict[str, str] | None
    execution: dict[str, Any]
    history_row: tuple[float, ...]
    actual_gemma_causal_forward: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STEP_RECEIPT_SCHEMA,
            "step": self.step,
            "decision_id": self.decision_id,
            "model_action": self.action,
            "model_action_logits": list(self.action_logits),
            "model_action_probabilities": list(self.action_probabilities),
            "decision_tensor_sha256": self.decision_tensor_sha256,
            "instruction_sha256": self.instruction_sha256,
            "active_prefix_sha256": self.active_prefix_sha256,
            "scene_prefix_sha256": self.scene_prefix_sha256,
            "robot_tokens_sha256": self.robot_tokens_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "scene_token_count": self.scene_token_count,
            "robot_token_count": self.robot_token_count,
            "history_token_count": self.history_token_count,
            "prompt_token_count": self.prompt_token_count,
            "decision_position": self.decision_position,
            "model_waypoint_delta_robot_m": list(self.waypoint_delta_robot_m),
            "model_turn_delta_degrees": self.turn_delta_degrees,
            "model_desired_heading_degrees": self.desired_heading_degrees,
            "primitive_tool": self.primitive_tool,
            "primitive_arguments": dict(self.primitive_arguments),
            "bound_proposal": None
            if self.bound_proposal is None
            else dict(self.bound_proposal),
            "execution": dict(self.execution),
            "history_row": list(self.history_row),
            "actual_gemma_causal_forward": self.actual_gemma_causal_forward,
            "model_selected_every_waypoint_and_heading": True,
            "deterministic_route_planner_used": False,
            "substitution_applied": False,
            "synthetic_stop_applied": False,
        }


@dataclass(frozen=True, slots=True)
class GemmaWaypointGoalResult:
    success: bool
    termination: str
    error_code: str | None
    instruction_sha256: str
    checkpoint_sha256: str
    model_stop_emitted: bool
    receipts: tuple[GemmaWaypointStepReceipt, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": GOAL_RESULT_SCHEMA,
            "success": self.success,
            "termination": self.termination,
            "error_code": self.error_code,
            "instruction_sha256": self.instruction_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_stop_emitted": self.model_stop_emitted,
            "step_count": len(self.receipts),
            "steps": [receipt.as_dict() for receipt in self.receipts],
            "synthetic_stop_applied": False,
            "substitution_applied": False,
            "deterministic_route_planner_used": False,
        }


class GemmaWaypointClosedLoopController:
    """Rerun actual Gemma for every exact, rejection-only movement step."""

    def __init__(
        self,
        *,
        runtime: Any,
        config: Mapping[str, Any],
        policy: ActualGemmaWaypointPolicy,
        prefix_backend: Gemma4PrefixBackend,
        tokenizer: Any,
        checkpoint_sha256: str,
        checkpoint_metadata: Mapping[str, Any],
    ) -> None:
        if not isinstance(policy, ActualGemmaWaypointPolicy):
            raise TypeError("Closed loop requires ActualGemmaWaypointPolicy")
        if not isinstance(prefix_backend, Gemma4PrefixBackend):
            raise TypeError("Closed loop requires Gemma4PrefixBackend")
        if not isinstance(config, Mapping):
            raise TypeError("Closed-loop config must be a mapping")
        if not callable(getattr(runtime, "continuous_action_context_snapshot", None)):
            raise TypeError("Runtime lacks continuous_action_context_snapshot")
        for name in ("prefix_binding", "get_robot_state", "move_to", "turn"):
            if not callable(getattr(runtime, name, None)):
                raise TypeError(f"Runtime lacks required numeric method {name}")
        if getattr(runtime, "auto_scan_after_motion", None) is not False:
            raise ValueError(
                "Model-only waypoint control requires auto_scan_after_motion=false"
            )
        metadata = dict(checkpoint_metadata)
        history_contract = (
            metadata.get("history_dim"),
            metadata.get("history_parameterization"),
        )
        if (
            metadata.get("model_selects_every_waypoint_and_heading") is not True
            or metadata.get("deterministic_route_planner_allowed_at_runtime") is not False
            or metadata.get("actual_gemma_causal_forward") is not True
            or metadata.get("gemma_output_hidden_states") is not True
            or metadata.get("complete_scene_prefix_required") is not True
            or metadata.get("every_scene_token_processed") is not True
            or metadata.get("numeric_state_and_history_required") is not True
            or metadata.get("environmental_text_inputs") != []
            or metadata.get("oracle_inputs_at_runtime") is not False
            or metadata.get("action_names") != list(ACTION_NAMES)
            or history_contract not in _SUPPORTED_RUNTIME_HISTORY_CONTRACTS
            or metadata.get("history_dim") != policy.history_projector.feature_dim
            or metadata.get("scene_token_count") != policy.scene_token_count
            or metadata.get("robot_token_count") != policy.robot_token_count
            or metadata.get("hidden_size") != policy.hidden_size
            or metadata.get("max_history_tokens")
            != policy.history_projector.max_history_tokens
            or float(metadata.get("max_waypoint_step_m", math.nan))
            != policy.max_waypoint_step_m
            or metadata.get("heading_parameterization") != HEADING_PARAMETERIZATION
            or float(metadata.get("max_turn_delta_degrees", math.nan))
            != policy.max_turn_delta_degrees
        ):
            raise ValueError("Closed-loop policy metadata differs from loaded policy")
        scene = config.get("scene")
        room_size = scene.get("room_size_m") if isinstance(scene, Mapping) else None
        room = torch.as_tensor(room_size, dtype=torch.float64)
        if room.shape != (3,) or not bool(torch.isfinite(room).all()) or bool(
            torch.any(room <= 0.0)
        ):
            raise ValueError("Closed-loop config has no valid room_size_m")
        self.runtime = runtime
        self.config = dict(config)
        self.policy = policy.eval()
        self.prefix_backend = prefix_backend
        self.tokenizer = tokenizer
        self.checkpoint_sha256 = _require_sha256(
            checkpoint_sha256, "checkpoint_sha256"
        )
        self.metadata = metadata
        self.history_feature_dim = int(metadata["history_dim"])
        self.history_parameterization = str(metadata["history_parameterization"])
        self.goal_progress_enabled = (
            self.history_parameterization == HISTORY_PARAMETERIZATION_V2
        )
        self.room_size_m = tuple(float(value) for value in room)
        self.max_history_tokens = policy.history_projector.max_history_tokens
        robot = config.get("robot")
        runtime_turn_bound = (
            robot.get("max_turn_degrees") if isinstance(robot, Mapping) else None
        )
        if (
            policy.max_turn_delta_degrees
            > _finite_positive(runtime_turn_bound, "robot.max_turn_degrees")
        ):
            raise ValueError("Waypoint policy turn bound exceeds the runtime primitive bound")
        self.executor = ModelBoundActionExecutor(
            self.config,
            checkpoint_sha256=self.checkpoint_sha256,
        )
        raw_initial_binding = self.runtime.prefix_binding()
        if not isinstance(raw_initial_binding, Mapping):
            raise TypeError("Waypoint runtime returned an invalid prefix binding")
        initial_binding = _binding_hashes(raw_initial_binding)
        self.scene_prefix_sha256 = initial_binding["scene_prefix_sha256"]
        self.static_map_identity = _static_map_identity(raw_initial_binding)
        self._goal_counter = 0
        self._lock = threading.RLock()
        # Authenticate the scene/robot partition before the controller can be
        # exposed to a UI or supplied with user text.
        self._snapshot()

    @classmethod
    def from_loaded(
        cls,
        *,
        runtime: Any,
        config: Mapping[str, Any],
        loaded: LoadedGemmaWaypointPolicy,
    ) -> GemmaWaypointClosedLoopController:
        if not isinstance(loaded, LoadedGemmaWaypointPolicy):
            raise TypeError("from_loaded requires LoadedGemmaWaypointPolicy")
        return cls(
            runtime=runtime,
            config=config,
            policy=loaded.policy,
            prefix_backend=loaded.prefix_backend,
            tokenizer=loaded.tokenizer,
            checkpoint_sha256=loaded.checkpoint_sha256,
            checkpoint_metadata=loaded.metadata,
        )

    def _snapshot(self) -> tuple[torch.Tensor, dict[str, str], Mapping[str, Any]]:
        snapshot = self.runtime.continuous_action_context_snapshot()
        if not isinstance(snapshot, tuple) or len(snapshot) != 3:
            raise RuntimeError("Waypoint runtime returned an invalid atomic action snapshot")
        active, raw_binding, state = snapshot
        if (
            not isinstance(active, torch.Tensor)
            or tuple(active.shape)
            != (1, self.policy.active_prefix_token_count, self.policy.hidden_size)
            or not bool(torch.isfinite(active.float()).all())
            or not isinstance(raw_binding, Mapping)
        ):
            raise RuntimeError("Waypoint runtime returned an invalid active prefix")
        binding = _binding_hashes(raw_binding)
        if binding["scene_prefix_sha256"] != self.scene_prefix_sha256:
            raise RuntimeError(
                "Static scene prefix changed during model-only navigation"
            )
        if _static_map_identity(raw_binding) != self.static_map_identity:
            raise RuntimeError("Static numeric map changed during model-only navigation")
        if prefix_sha256(active) != binding["active_prefix_sha256"]:
            raise RuntimeError("Waypoint active prefix differs from its binding")
        _validate_active_prefix_partition(
            active,
            raw_binding,
            scene_token_count=self.policy.scene_token_count,
            robot_token_count=self.policy.robot_token_count,
        )
        state_mapping = _state_mapping(
            state,
            scene_id=self.static_map_identity[0],
        )
        _state_pose(state_mapping)
        return active, binding, state_mapping

    def _primitive(
        self,
        decision: GemmaWaypointDecision,
        state: Mapping[str, Any],
    ) -> tuple[str, dict[str, float], float]:
        x, y, current_yaw = _state_pose(state)
        turn_delta = decision.turn_delta_degrees
        if (
            isinstance(turn_delta, bool)
            or not isinstance(turn_delta, (int, float))
            or not math.isfinite(float(turn_delta))
            or abs(float(turn_delta)) > self.policy.max_turn_delta_degrees + 1e-6
        ):
            raise RuntimeError("Gemma waypoint turn delta violates its bounded head")
        turn_delta = float(turn_delta)
        desired_heading = _normalized_degrees(current_yaw + turn_delta)
        if decision.action is GemmaMotionAction.MOVE_TO:
            target_x, target_y = robot_delta_to_world_xy(
                (x, y),
                current_yaw,
                decision.waypoint_delta_robot_m,
            )
            return "move_to", {"x": target_x, "y": target_y}, desired_heading
        if decision.action is GemmaMotionAction.FACE:
            return "turn", {"angle_degrees": turn_delta}, desired_heading
        if decision.action is GemmaMotionAction.STOP:
            return "stop", {}, desired_heading
        raise RuntimeError("Gemma waypoint policy emitted an unknown action")

    def _bind_or_reject(
        self,
        *,
        action_json: str,
        state: Mapping[str, Any],
        binding: Mapping[str, Any],
        decision_id: str,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> tuple[ModelBoundToolCall | None, dict[str, Any] | None]:
        try:
            proposal = bind_model_tool_call(
                action_json,
                self.config,
                robot_state=state,
                binding=binding,
                checkpoint_sha256=self.checkpoint_sha256,
                decision_id=decision_id,
            )
        except (TypeError, ValueError):
            validation = validate_tool_call_text(
                action_json,
                self.config,
                robot_state=state,
            )
            error_code = validation.error_code or "E_MODEL_BIND"
            return None, _attempted_rejection(
                error_code=error_code,
                tool=tool,
                arguments=arguments,
                checkpoint_sha256=self.checkpoint_sha256,
            )
        return proposal, None

    def run(
        self,
        instruction: str,
        *,
        max_steps: int = 24,
    ) -> GemmaWaypointGoalResult:
        """Execute one conversational goal until Gemma emits STOP or fails."""

        if not isinstance(instruction, str):
            raise TypeError("Waypoint goal instruction must be text")
        if not instruction.strip() or instruction != instruction.strip():
            raise ValueError("Waypoint goal instruction must be nonempty and unwrapped")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 128:
            raise ValueError("Waypoint max_steps must be an integer in [1,128]")
        instruction_digest = _sha256_text(instruction)
        with self._lock:
            self._goal_counter += 1
            goal_id = f"gwp_{self._goal_counter:08d}_{instruction_digest[:8]}"
            history: deque[tuple[float, ...]] = deque(maxlen=self.max_history_tokens)
            receipts: list[GemmaWaypointStepReceipt] = []
            scoped_runtime = _GoalSettlementRuntime(self.runtime)
            goal_progress: WaypointGoalProgressLedger | None = None
            for step in range(1, max_steps + 1):
                active, binding, state = self._snapshot()
                if self.goal_progress_enabled and goal_progress is None:
                    goal_progress = WaypointGoalProgressLedger.from_initial_pose(
                        _state_pose(state)
                    )
                history_tensor = (
                    torch.tensor(tuple(history), dtype=torch.float32).unsqueeze(0)
                    if history
                    else torch.empty(
                        (1, 0, self.history_feature_dim), dtype=torch.float32
                    )
                )
                decision = self.policy.decide(
                    prefix_backend=self.prefix_backend,
                    tokenizer=self.tokenizer,
                    active_scene_robot_prefix=active,
                    instruction=instruction,
                    history_features=history_tensor,
                )
                if (
                    decision.actual_gemma_causal_forward is not True
                    or decision.instruction_sha256 != instruction_digest
                    or decision.active_prefix_sha256 != binding["active_prefix_sha256"]
                    or decision.scene_token_count != self.policy.scene_token_count
                    or decision.robot_token_count != self.policy.robot_token_count
                    or decision.history_token_count != len(history)
                ):
                    raise RuntimeError("Gemma waypoint decision provenance differs")
                tool, arguments, desired_heading = self._primitive(decision, state)
                action_json = _canonical_json(
                    {"tool": tool, "arguments": arguments}
                )
                decision_id = f"{goal_id}_{step:03d}"
                proposal, rejection = self._bind_or_reject(
                    action_json=action_json,
                    state=state,
                    binding=binding,
                    decision_id=decision_id,
                    tool=tool,
                    arguments=arguments,
                )
                execution = (
                    rejection
                    if rejection is not None
                    else self.executor.execute(scoped_runtime, proposal)
                )
                assert isinstance(execution, dict)
                success = execution.get("success") is True
                before_pose = _state_pose(state)
                after_state = self.runtime.get_robot_state()
                if not isinstance(after_state, Mapping):
                    raise TypeError("Waypoint runtime returned invalid post-action state")
                after_pose = _state_pose(after_state)
                after_binding = _binding_hashes(self.runtime.prefix_binding())
                if after_binding["scene_prefix_sha256"] != self.scene_prefix_sha256:
                    raise RuntimeError(
                        "Static scene prefix changed after a Gemma motion primitive"
                    )
                if not success and after_pose != before_pose:
                    raise RuntimeError("Rejected Gemma proposal changed the robot pose")
                if success and tool == "move_to":
                    expected_xy = (float(arguments["x"]), float(arguments["y"]))
                    if not all(
                        math.isclose(after_pose[index], expected_xy[index], abs_tol=1e-8)
                        for index in range(2)
                    ):
                        raise RuntimeError("Executed MOVE_TO differs from Gemma's exact waypoint")
                    if not math.isclose(
                        _normalized_degrees(after_pose[2] - before_pose[2]),
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    ):
                        raise RuntimeError(
                            "Executed MOVE_TO changed facing without a Gemma FACE decision"
                        )
                if success and tool == "turn":
                    if not all(
                        math.isclose(
                            after_pose[index], before_pose[index], rel_tol=0.0, abs_tol=1e-8
                        )
                        for index in range(2)
                    ):
                        raise RuntimeError(
                            "Executed FACE changed position without a Gemma waypoint"
                        )
                    if not math.isclose(
                        _normalized_degrees(after_pose[2] - desired_heading),
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    ):
                        raise RuntimeError(
                            "Executed FACE differs from Gemma's exact turn delta"
                        )
                if success and tool == "stop" and after_pose != before_pose:
                    raise RuntimeError("Executed STOP changed the robot pose")
                history_arguments = {
                    "action": decision.action.value,
                    "result_pose_xy_yaw": after_pose if success else before_pose,
                    "requested_waypoint_delta_robot_m": (
                        decision.waypoint_delta_robot_m
                    ),
                    "requested_heading_degrees": desired_heading,
                    "room_size_m": self.room_size_m,
                    "max_waypoint_step_m": self.policy.max_waypoint_step_m,
                    "success": success,
                }
                if goal_progress is None:
                    history_row = encode_waypoint_history_transition(
                        **history_arguments,
                    )
                else:
                    goal_progress.record_receipt(
                        before_pose_xy_yaw=before_pose,
                        after_pose_xy_yaw=after_pose,
                        success=success,
                    )
                    history_row = encode_waypoint_history_transition_v2(
                        **history_arguments,
                        goal_progress=goal_progress.normalized_features(
                            room_size_m=self.room_size_m,
                            rejection_streak_scale=self.max_history_tokens,
                        ),
                    )
                if len(history_row) != self.history_feature_dim:
                    raise RuntimeError(
                        "Waypoint history encoder differs from checkpoint contract"
                    )
                history.append(history_row)
                receipt = GemmaWaypointStepReceipt(
                    step=step,
                    decision_id=decision_id,
                    action=decision.action.value,
                    action_logits=decision.action_logits,
                    action_probabilities=decision.action_probabilities,
                    decision_tensor_sha256=_decision_tensor_sha256(decision),
                    instruction_sha256=instruction_digest,
                    active_prefix_sha256=binding["active_prefix_sha256"],
                    scene_prefix_sha256=binding["scene_prefix_sha256"],
                    robot_tokens_sha256=binding["robot_tokens_sha256"],
                    checkpoint_sha256=self.checkpoint_sha256,
                    scene_token_count=decision.scene_token_count,
                    robot_token_count=decision.robot_token_count,
                    history_token_count=decision.history_token_count,
                    prompt_token_count=decision.prompt_token_count,
                    decision_position=decision.decision_position,
                    waypoint_delta_robot_m=decision.waypoint_delta_robot_m,
                    turn_delta_degrees=decision.turn_delta_degrees,
                    desired_heading_degrees=desired_heading,
                    primitive_tool=tool,
                    primitive_arguments={
                        str(name): float(value) for name, value in arguments.items()
                    },
                    bound_proposal=None if proposal is None else proposal.as_dict(),
                    execution=execution,
                    history_row=history_row,
                    actual_gemma_causal_forward=True,
                )
                receipts.append(receipt)
                if decision.action is GemmaMotionAction.STOP:
                    if not success or not scoped_runtime.goal_settled:
                        return GemmaWaypointGoalResult(
                            success=False,
                            termination="model_stop_rejected",
                            error_code=str(execution.get("error_code") or "E_MODEL_STOP"),
                            instruction_sha256=instruction_digest,
                            checkpoint_sha256=self.checkpoint_sha256,
                            model_stop_emitted=True,
                            receipts=tuple(receipts),
                        )
                    return GemmaWaypointGoalResult(
                        success=True,
                        termination="model_stop",
                        error_code=None,
                        instruction_sha256=instruction_digest,
                        checkpoint_sha256=self.checkpoint_sha256,
                        model_stop_emitted=True,
                        receipts=tuple(receipts),
                    )
                # Any collision, bounds violation, or failed primitive remains
                # a success=0 history row. The next iteration asks Gemma again.
            return GemmaWaypointGoalResult(
                success=False,
                termination="max_steps",
                error_code="E_MAX_STEPS",
                instruction_sha256=instruction_digest,
                checkpoint_sha256=self.checkpoint_sha256,
                model_stop_emitted=False,
                receipts=tuple(receipts),
            )


__all__ = [
    "CHECKPOINT_ARCHITECTURE",
    "CHECKPOINT_FILES",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_V1",
    "CHECKPOINT_SCHEMA_V2",
    "GOAL_RESULT_SCHEMA",
    "HEADING_PARAMETERIZATION",
    "HISTORY_PARAMETERIZATION",
    "HISTORY_PARAMETERIZATION_V1",
    "HISTORY_PARAMETERIZATION_V2",
    "STEP_RECEIPT_SCHEMA",
    "GemmaWaypointClosedLoopController",
    "GemmaWaypointGoalResult",
    "GemmaWaypointStepReceipt",
    "LoadedGemmaWaypointPolicy",
    "load_gemma_waypoint_policy_checkpoint",
    "robot_delta_to_world_xy",
]

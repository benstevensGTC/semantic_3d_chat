"""Train and evaluate the local-Gemma waypoint controller.

This module intentionally keeps the frozen language model separate from the
small trainable controller.  A decision is produced by one *real* Gemma causal
forward over, in order, the complete continuous scene prefix, the user's
instruction, and learned numeric robot/history tokens.  Only the controller's
numeric projectors and action heads are checkpointed.

The runtime controller is deliberately represented by a small protocol instead
of being constructed here.  This makes the scientific training harness usable
with a fake frozen decoder in unit tests while retaining exactly the same call
boundary used by the local Gemma-4 model.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import torch
import transformers
from safetensors.torch import load_file, save_file
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.local_lm import LocalLanguageModel, prompt_token_ids
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    prefix_sha256,
)
from semantic_3d_chat.robot.gemma_runtime_binding import (
    attach_gemma_runtime_binding,
    gemma_runtime_binding_sha256,
    language_gemma_runtime_binding,
    question_controlled_gemma_runtime_binding,
    raw_hf_gemma_runtime_binding,
    validate_gemma_runtime_binding,
)
from semantic_3d_chat.robot.gemma_waypoint_policy import (
    ACTION_NAMES as RUNTIME_ACTION_NAMES,
)
from semantic_3d_chat.robot.gemma_waypoint_policy import (
    POLICY_SYSTEM_PROMPT,
)
from semantic_3d_chat.robot.state_encoder import insert_robot_state_tokens
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V1,
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION_V1,
    HISTORY_PARAMETERIZATION_V2,
)
from semantic_3d_chat.training.gemma_waypoint_branch_refit import (
    refit_waypoint_branch,
)

ACTION_NAMES: Final[tuple[str, ...]] = RUNTIME_ACTION_NAMES
ACTION_TO_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(ACTION_NAMES)
}
SYSTEM_PROMPT: Final[str] = (
    "Choose the next bounded robot action from the continuous 3D scene memory, "
    "numeric robot state, and numeric action history. The learned control head, "
    "not text generation, emits MOVE_TO, FACE, or STOP and its numeric target."
)
_SCENE_ID_PREFIX: Final[str] = "scene_"
_CHECKPOINT_FILES: Final[frozenset[str]] = frozenset(
    {"policy.safetensors", "runtime_metadata.json"}
)
_CHECKPOINT_SCHEMA_V1: Final[str] = "semantic_3d_chat.gemma_waypoint_checkpoint.v3"
_CHECKPOINT_SCHEMA_V2: Final[str] = "semantic_3d_chat.gemma_waypoint_checkpoint.v4"
_HIDDEN_CACHE_SCHEMA_V1: Final[str] = (
    "semantic_3d_chat.gemma_waypoint_hidden_cache.v3"
)
_HIDDEN_CACHE_SCHEMA_V2: Final[str] = (
    "semantic_3d_chat.gemma_waypoint_hidden_cache.v4"
)
_HIDDEN_INPUT_BINDING_SCHEMA_V1: Final[str] = (
    "semantic_3d_chat.gemma_waypoint_hidden_input_binding.v1"
)
_HIDDEN_INPUT_BINDING_SCHEMA_V2: Final[str] = (
    "semantic_3d_chat.gemma_waypoint_hidden_input_binding.v2"
)
_SUPPORTED_CHECKPOINT_HISTORY_CONTRACTS: Final[
    dict[tuple[int, str], str]
] = {
    (HISTORY_FEATURE_DIM_V1, HISTORY_PARAMETERIZATION_V1): _CHECKPOINT_SCHEMA_V1,
    (HISTORY_FEATURE_DIM_V2, HISTORY_PARAMETERIZATION_V2): _CHECKPOINT_SCHEMA_V2,
}


@runtime_checkable
class GemmaWaypointControllerProtocol(Protocol):
    """Narrow interface shared by training and the fail-closed runtime."""

    def forward_actual_gemma(
        self,
        *,
        prefix_backend: Any,
        tokenizer: Any,
        active_scene_robot_prefix: torch.Tensor,
        instruction: str,
        history_features: torch.Tensor,
    ) -> object:
        """Return one differentiable decision from a real Gemma causal pass."""

    def forward_heads_from_cached_gemma_hidden(
        self, final_gemma_hidden: torch.Tensor
    ) -> object:
        """Run trainable numeric heads over authenticated cached Gemma states."""


@dataclass(frozen=True)
class WaypointPolicyTensors:
    action_logits: torch.Tensor
    waypoint_delta_robot_m: torch.Tensor
    turn_delta_degrees: torch.Tensor


@dataclass(frozen=True)
class WaypointTraceSample:
    sample_id: str
    scene_id: str
    split: str
    instruction: str
    state: torch.Tensor
    history: torch.Tensor
    action_index: int
    waypoint_delta_robot_m: torch.Tensor
    heading_degrees: float

    @property
    def action_name(self) -> str:
        return ACTION_NAMES[self.action_index]


@dataclass(frozen=True)
class WaypointTraceDataset:
    samples: tuple[WaypointTraceSample, ...]
    sha256: str
    traces_sha256: str
    state_dim: int
    history_dim: int
    history_parameterization: str = HISTORY_PARAMETERIZATION_V1

    def split(self, name: str) -> tuple[WaypointTraceSample, ...]:
        selected = tuple(sample for sample in self.samples if sample.split == name)
        if not selected:
            raise ValueError(f"Waypoint trace split is empty: {name}")
        return selected

    @property
    def scene_splits(self) -> dict[str, tuple[str, ...]]:
        names = sorted({sample.split for sample in self.samples})
        return {
            name: tuple(
                sorted({sample.scene_id for sample in self.samples if sample.split == name})
            )
            for name in names
        }


@dataclass(frozen=True)
class WaypointRetentionReference:
    """Training-only outputs from an authenticated prior policy checkpoint."""

    outputs: WaypointPolicyTensors
    shared_mask: torch.Tensor
    sample_weights: torch.Tensor
    action_weight: torch.Tensor
    action_bias: torch.Tensor
    metadata: Mapping[str, Any]


def select_balanced_waypoint_samples(
    samples: Sequence[WaypointTraceSample], limit: int | None
) -> tuple[WaypointTraceSample, ...]:
    """Deterministically span every scene/action bucket under a forward budget."""

    values = tuple(samples)
    if limit is None or len(values) <= limit:
        return values
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < len(ACTION_NAMES):
        raise ValueError("Waypoint sample limit must cover every action")
    buckets: dict[tuple[str, int], list[WaypointTraceSample]] = {}
    for sample in values:
        buckets.setdefault((sample.scene_id, sample.action_index), []).append(sample)
    keys = sorted(buckets)
    allocations = {key: 0 for key in keys}
    for index in range(limit):
        allocations[keys[index % len(keys)]] += 1
    selected: list[WaypointTraceSample] = []
    for key in keys:
        bucket = buckets[key]
        count = min(len(bucket), allocations[key])
        if count == 0:
            continue
        if count == 1:
            indices = [len(bucket) // 2]
        else:
            indices = [round(index * (len(bucket) - 1) / (count - 1)) for index in range(count)]
        selected.extend(bucket[index] for index in indices)
    if len(selected) < limit:
        selected_ids = {sample.sample_id for sample in selected}
        selected.extend(
            sample
            for sample in values
            if sample.sample_id not in selected_ids
            for _ in range(1)
            if len(selected) < limit
        )
    return tuple(selected[:limit])


@dataclass(frozen=True)
class ScenePrefixCache:
    prefixes: Mapping[str, torch.Tensor]
    file_sha256: Mapping[str, str]
    token_count: int
    hidden_size: int


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_schema_for_history(
    history_dim: object,
    history_parameterization: object,
) -> str:
    """Return the only checkpoint schema allowed for one numeric history pair."""

    if isinstance(history_dim, bool) or not isinstance(history_dim, int):
        raise TypeError("Waypoint history dimension/parameterization pair differs")
    if not isinstance(history_parameterization, str):
        raise TypeError("Waypoint history dimension/parameterization pair differs")
    schema = _SUPPORTED_CHECKPOINT_HISTORY_CONTRACTS.get(
        (history_dim, history_parameterization)
    )
    if schema is None:
        raise ValueError("Waypoint history dimension/parameterization pair differs")
    return schema


def _unique_json_object(text: str, *, purpose: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate {purpose} field: {key}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise TypeError(f"{purpose} must be a JSON object")
    return value


def _named_tensor_sha256(values: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and exact storage bytes."""

    if not values:
        raise ValueError("Frozen waypoint component has no tensors")
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"Frozen waypoint tensor {name!r} is non-finite")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _source_contract_sha256(
    callables: Sequence[object], *, constants: Mapping[str, object]
) -> str:
    digest = hashlib.sha256()
    source_files: set[Path] = set()
    for value in callables:
        source = inspect.getsource(value)
        qualified = f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', '')}"
        digest.update(qualified.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
        source_path = inspect.getsourcefile(value)
        if source_path is not None:
            source_files.add(Path(source_path).resolve())
    for source_path in sorted(source_files):
        digest.update(str(source_path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(source_path).encode("ascii"))
        digest.update(b"\0")
    digest.update(_canonical_sha256(constants).encode("ascii"))
    return digest.hexdigest()


def _validated_hidden_input_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Gemma hidden-input binding must be an object")
    common_keys = {
        "schema",
        "scene_prefix_sha256",
        "scene_prefix_file_sha256",
        "robot_state_encoder_sha256",
        "controller_context_sha256",
        "prompt_token_ids_sha256",
        "forward_contract_sha256",
    }
    schema = value.get("schema")
    if schema == _HIDDEN_INPUT_BINDING_SCHEMA_V1:
        expected_keys = common_keys
    elif schema == _HIDDEN_INPUT_BINDING_SCHEMA_V2:
        expected_keys = common_keys | {"history_dim", "history_parameterization"}
    else:
        raise ValueError("Gemma hidden-input binding schema differs")
    if set(value) != expected_keys:
        raise ValueError("Gemma hidden-input binding schema differs")
    result: dict[str, Any] = {"schema": schema}
    if schema == _HIDDEN_INPUT_BINDING_SCHEMA_V2:
        if (
            value.get("history_dim") != HISTORY_FEATURE_DIM_V2
            or value.get("history_parameterization") != HISTORY_PARAMETERIZATION_V2
        ):
            raise ValueError("Gemma hidden-input history contract differs")
        result.update(
            {
                "history_dim": HISTORY_FEATURE_DIM_V2,
                "history_parameterization": HISTORY_PARAMETERIZATION_V2,
            }
        )
    for field in (
        "scene_prefix_sha256",
        "scene_prefix_file_sha256",
        "prompt_token_ids_sha256",
    ):
        observed = value.get(field)
        if (
            not isinstance(observed, Mapping)
            or not observed
            or any(not isinstance(name, str) or not name for name in observed)
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in observed.values()
            )
        ):
            raise ValueError(f"Gemma hidden-input binding field {field!r} differs")
        result[field] = {str(name): str(observed[name]) for name in sorted(observed)}
    if set(result["scene_prefix_sha256"]) != set(result["scene_prefix_file_sha256"]):
        raise ValueError("Gemma hidden-input scene-prefix inventories differ")
    for field in (
        "robot_state_encoder_sha256",
        "controller_context_sha256",
        "forward_contract_sha256",
    ):
        digest = value.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Gemma hidden-input binding field {field!r} differs")
        result[field] = digest
    return result


def gemma_hidden_input_binding(
    language: LocalLanguageModel,
    controller: nn.Module,
    robot_state_encoder: nn.Module,
    cache: ScenePrefixCache,
    samples: Sequence[WaypointTraceSample],
    *,
    history_parameterization: str = HISTORY_PARAMETERIZATION_V1,
) -> dict[str, Any]:
    """Bind every frozen continuous/token input that determines cached states.

    The cache already binds the frozen Gemma/LoRA runtime separately. This
    companion binding covers the complete scene tensors, numeric encoders,
    fixed control-token parameters, exact prompt token IDs, and the Python
    forward/prepare implementations. It stores hashes only, never raw goals.
    """

    scene_ids = sorted({sample.scene_id for sample in samples})
    if not scene_ids or any(scene_id not in cache.prefixes for scene_id in scene_ids):
        raise ValueError("Gemma hidden-input binding lacks a required scene prefix")
    if any(scene_id not in cache.file_sha256 for scene_id in scene_ids):
        raise ValueError("Gemma hidden-input binding lacks a scene-prefix file digest")
    decision_token = getattr(controller, "decision_token", None)
    history_projector = getattr(controller, "history_projector", None)
    if not isinstance(decision_token, torch.Tensor) or not isinstance(history_projector, nn.Module):
        raise TypeError("Waypoint controller lacks frozen context-token parameters")
    history_dim = getattr(history_projector, "feature_dim", None)
    if isinstance(history_dim, bool) or not isinstance(history_dim, int) or history_dim < 1:
        raise TypeError("Waypoint controller lacks a numeric history feature dimension")
    if any(
        sample.history.ndim != 2 or int(sample.history.shape[1]) != history_dim
        for sample in samples
    ):
        raise ValueError("Gemma hidden-input samples differ from the history projector")
    if history_parameterization == HISTORY_PARAMETERIZATION_V1:
        binding_schema = _HIDDEN_INPUT_BINDING_SCHEMA_V1
    elif (
        history_parameterization == HISTORY_PARAMETERIZATION_V2
        and history_dim == HISTORY_FEATURE_DIM_V2
    ):
        binding_schema = _HIDDEN_INPUT_BINDING_SCHEMA_V2
    else:
        raise ValueError("Gemma hidden-input history contract differs")
    prefix_backend = language.prefix_backend
    if prefix_backend is None or not callable(getattr(prefix_backend, "prepare", None)):
        raise TypeError("Gemma hidden-input binding requires a prefix backend")
    context_tensors = {"decision_token": decision_token}
    context_tensors.update(
        {f"history_projector.{name}": tensor for name, tensor in history_projector.state_dict().items()}
    )
    instructions = sorted({sample.instruction for sample in samples})
    prompt_hashes: dict[str, str] = {}
    for instruction in instructions:
        ids = prompt_token_ids(
            language.tokenizer,
            POLICY_SYSTEM_PROMPT,
            instruction,
            torch.device("cpu"),
        ).detach().to(dtype=torch.int64, device="cpu").contiguous()
        instruction_sha256 = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        prompt_hashes[instruction_sha256] = hashlib.sha256(
            json.dumps(list(ids.shape), separators=(",", ":")).encode("ascii")
            + b"\0"
            + ids.numpy().tobytes(order="C")
        ).hexdigest()
    binding = {
        "schema": binding_schema,
        "scene_prefix_sha256": {
            scene_id: _named_tensor_sha256({scene_id: cache.prefixes[scene_id]})
            for scene_id in scene_ids
        },
        "scene_prefix_file_sha256": {
            scene_id: str(cache.file_sha256[scene_id]) for scene_id in scene_ids
        },
        "robot_state_encoder_sha256": _named_tensor_sha256(
            robot_state_encoder.state_dict()
        ),
        "controller_context_sha256": _named_tensor_sha256(context_tensors),
        "prompt_token_ids_sha256": prompt_hashes,
        "forward_contract_sha256": _source_contract_sha256(
            (
                ActualGemmaWaypointForward.__call__,
                type(controller).forward_actual_gemma,
                insert_robot_state_tokens,
                prompt_token_ids,
                type(prefix_backend).prepare,
                type(robot_state_encoder).forward,
                type(history_projector).forward,
            )
            ,
            constants={
                "contract": (
                    "full_prefix_prompt_history_decision_hidden_v1"
                    if binding_schema == _HIDDEN_INPUT_BINDING_SCHEMA_V1
                    else "full_prefix_prompt_history_decision_hidden_v2"
                ),
                "scene_boundary_mode": SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
                "robot_token_insertion": "immediately_before_final_scene_boundary",
                "token_order": "bos_full_scene_robot_prefix_prompt_history_decision",
                "decision_position": "final_causal_input",
                "output_hidden_states": True,
                "use_cache": False,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                **(
                    {
                        "history_dim": history_dim,
                        "history_parameterization": history_parameterization,
                    }
                    if binding_schema == _HIDDEN_INPUT_BINDING_SCHEMA_V2
                    else {}
                ),
            },
        ),
    }
    if binding_schema == _HIDDEN_INPUT_BINDING_SCHEMA_V2:
        binding.update(
            {
                "history_dim": history_dim,
                "history_parameterization": history_parameterization,
            }
        )
    return _validated_hidden_input_binding(binding)


def _guard_runtime_checkpoint_path(path: Path) -> None:
    if {"oracle", "qa", "training"} & {part.casefold() for part in path.parts}:
        raise ValueError("Waypoint runtime checkpoint cannot enter a blocked data tree")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Waypoint runtime checkpoint path cannot contain symlinks")


def _finite_vector(value: object, *, name: str, size: int) -> torch.Tensor:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must be a JSON list with exactly {size} values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"{name} must contain only numeric values")
    tensor = torch.tensor(value, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains a non-finite value")
    return tensor


def _finite_history(
    value: object,
    *,
    name: str,
    feature_dim: int,
    max_tokens: int,
) -> torch.Tensor:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list of numeric history rows")
    if len(value) > max_tokens:
        raise ValueError(f"{name} exceeds the configured {max_tokens}-token limit")
    if not value:
        return torch.empty((0, feature_dim), dtype=torch.float32)
    rows = [
        _finite_vector(row, name=f"{name}[{index}]", size=feature_dim)
        for index, row in enumerate(value)
    ]
    return torch.stack(rows)


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_action(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("action must be a string")
    normalized = value.strip().casefold().replace("-", "_")
    if normalized not in ACTION_TO_INDEX:
        raise ValueError(f"action must be one of {ACTION_NAMES}; got {value!r}")
    return normalized


def load_waypoint_trace_jsonl(
    path: str | Path,
    *,
    state_dim: int,
    history_dim: int,
    history_parameterization: str = HISTORY_PARAMETERIZATION_V1,
    max_history_tokens: int,
    max_waypoint_step_m: float,
) -> WaypointTraceDataset:
    """Load deterministic expert decisions without retaining semantic metadata.

    The accepted row schema is deliberately numeric and minimal.  It contains
    no simulator object ID or oracle relationship.  Oracle data may have been
    used by an offline expert to produce these labels, but it is not copied into
    the controller checkpoint.
    """

    source = _rooted(path)
    manifest: dict[str, Any] | None = None
    if source.is_dir():
        manifest_path = source / "manifest.json"
        if manifest_path.is_file():
            manifest = _unique_json_object(
                manifest_path.read_text(encoding="utf-8"),
                purpose="waypoint trace manifest",
            )
        source = source / "traces.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    if state_dim < 1 or history_dim < 1 or max_history_tokens < 1:
        raise ValueError("state_dim and history_dim must be positive")
    if not isinstance(history_parameterization, str) or not history_parameterization:
        raise ValueError("history_parameterization must be a non-empty string")
    maximum_step = _finite_number(max_waypoint_step_m, name="max_waypoint_step_m")
    if maximum_step <= 0.0:
        raise ValueError("max_waypoint_step_m must be positive")

    samples: list[WaypointTraceSample] = []
    identifiers: set[str] = set()
    allowed_splits = {"train", "validation", "test"}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on waypoint row {line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Waypoint row {line_number} is not an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"Waypoint row {line_number} has no sample_id")
            if sample_id in identifiers:
                raise ValueError(f"Duplicate waypoint sample_id: {sample_id}")
            identifiers.add(sample_id)
            scene_id = row.get("scene_id")
            if (
                not isinstance(scene_id, str)
                or not scene_id.startswith(_SCENE_ID_PREFIX)
                or Path(scene_id).name != scene_id
            ):
                raise ValueError(f"Waypoint row {line_number} has no opaque scene ID")
            split = row.get("split")
            if split not in allowed_splits:
                raise ValueError(f"Waypoint row {line_number} has invalid split")
            instruction = row.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Waypoint row {line_number} has no instruction")
            state_value = row.get("state", row.get("state_features"))
            history_value = row.get("history", row.get("history_features"))
            state = _finite_vector(
                state_value, name=f"row {line_number} state", size=state_dim
            )
            history = _finite_history(
                history_value,
                name=f"row {line_number} history",
                feature_dim=history_dim,
                max_tokens=max_history_tokens,
            )
            action = _normalize_action(row.get("action", row.get("action_name")))
            waypoint_value = row.get(
                "waypoint_delta_robot_m",
                row.get("waypoint_xy_m", row.get("waypoint_xy")),
            )
            heading_value = row.get("heading_degrees")
            if action == "move_to":
                waypoint = _finite_vector(
                    waypoint_value,
                    name=f"row {line_number} waypoint_delta_robot_m",
                    size=2,
                )
                if float(torch.linalg.vector_norm(waypoint)) > maximum_step + 1e-6:
                    raise ValueError(
                        f"Waypoint row {line_number} exceeds max_waypoint_step_m"
                    )
                heading = 0.0
            elif action == "face":
                if waypoint_value is not None:
                    raise ValueError(
                        f"FACE row {line_number} cannot contain a waypoint delta"
                    )
                waypoint = torch.zeros(2, dtype=torch.float32)
                heading = _finite_number(
                    heading_value, name=f"row {line_number} heading_degrees"
                )
                if heading < -180.0 or heading > 180.0:
                    raise ValueError(f"FACE row {line_number} heading is outside [-180, 180]")
            else:
                if waypoint_value is not None or heading_value is not None:
                    raise ValueError(f"STOP row {line_number} cannot contain numeric target labels")
                waypoint = torch.zeros(2, dtype=torch.float32)
                heading = 0.0
            samples.append(
                WaypointTraceSample(
                    sample_id=sample_id,
                    scene_id=scene_id,
                    split=str(split),
                    instruction=instruction.strip(),
                    state=state,
                    history=history,
                    action_index=ACTION_TO_INDEX[action],
                    waypoint_delta_robot_m=waypoint,
                    heading_degrees=heading,
                )
            )
    if not samples:
        raise ValueError("Waypoint trace dataset is empty")
    scene_owners: dict[str, str] = {}
    for sample in samples:
        previous = scene_owners.setdefault(sample.scene_id, sample.split)
        if previous != sample.split:
            raise ValueError(
                f"Scene {sample.scene_id} occurs in both {previous} and {sample.split}"
            )
    traces_sha256 = _sha256_file(source)
    scene_splits = {
        split: sorted(
            {sample.scene_id for sample in samples if sample.split == split}
        )
        for split in sorted({sample.split for sample in samples})
    }
    dataset_sha256 = traces_sha256
    if manifest is not None:
        declared_dataset_sha = manifest.get("dataset_sha256")
        manifest_body = {
            key: value for key, value in manifest.items() if key != "dataset_sha256"
        }
        if (
            manifest.get("schema")
            != "semantic_3d_chat.gemma_waypoint_trace_dataset.v1"
            or manifest.get("traces_sha256") != traces_sha256
            or manifest.get("sample_count") != len(samples)
            or manifest.get("train_scene_ids") != scene_splits.get("train", [])
            or manifest.get("validation_scene_ids") != scene_splits.get("validation", [])
            or manifest.get("scene_splits_disjoint") is not True
            or manifest.get("environmental_text_training_only") is not True
            or manifest.get("expert_planners_available_at_runtime") is not False
            or manifest.get("oracle_inputs_at_runtime") is not False
            or manifest.get("history_parameterization")
            != history_parameterization
            or manifest.get("policy_selects_all_headings_and_waypoints_at_runtime") is not True
            or not isinstance(declared_dataset_sha, str)
            or len(declared_dataset_sha) != 64
            or declared_dataset_sha != _canonical_sha256(manifest_body)
        ):
            raise ValueError("Waypoint trace manifest contract differs")
        dataset_sha256 = declared_dataset_sha
    return WaypointTraceDataset(
        samples=tuple(samples),
        sha256=dataset_sha256,
        traces_sha256=traces_sha256,
        state_dim=state_dim,
        history_dim=history_dim,
        history_parameterization=history_parameterization,
    )


def load_scene_prefix_cache(
    root: str | Path,
    scene_ids: Iterable[str],
    *,
    expected_token_count: int,
    expected_hidden_size: int,
) -> ScenePrefixCache:
    """Load authenticated, complete, question-independent scene prefixes."""

    prefix_root = _rooted(root)
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
        raise ValueError("Waypoint scene-prefix cache contract differs")
    prefixes: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    for scene_id in sorted(set(scene_ids)):
        entry = scenes.get(scene_id)
        if not isinstance(entry, dict):
            raise FileNotFoundError(f"No cached complete scene prefix for {scene_id}")
        filename = entry.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Unsafe prefix filename for {scene_id}")
        path = prefix_root / filename
        observed_hash = _sha256_file(path)
        if observed_hash != entry.get("file_sha256"):
            raise ValueError(f"Cached scene prefix changed for {scene_id}")
        values = load_file(str(path), device="cpu")
        if set(values) != {"scene_prefix"}:
            raise ValueError(f"Unexpected prefix tensors for {scene_id}")
        prefix = values["scene_prefix"]
        expected = (1, expected_token_count, expected_hidden_size)
        if tuple(prefix.shape) != expected:
            raise ValueError(
                f"Scene prefix {scene_id} has shape {tuple(prefix.shape)}, expected {expected}"
            )
        if not torch.isfinite(prefix).all():
            raise ValueError(f"Scene prefix {scene_id} contains non-finite values")
        prefixes[scene_id] = prefix.contiguous()
        hashes[scene_id] = observed_hash
    return ScenePrefixCache(
        prefixes=prefixes,
        file_sha256=hashes,
        token_count=expected_token_count,
        hidden_size=expected_hidden_size,
    )


def assemble_demo_scene_prefix_cache(
    destination: str | Path,
    *,
    live_scene_id: str,
    live_prefix_path: str | Path,
    reference_cache_root: str | Path,
    validation_scene_ids: Sequence[str],
    expected_token_count: int = 258,
    expected_hidden_size: int = 1536,
) -> dict[str, Any]:
    """Assemble a sanitized live+validation cache with authenticated copies."""

    root = _rooted(destination)
    if root.exists():
        raise FileExistsError(f"Demo scene-prefix cache already exists: {root}")
    if not live_scene_id.startswith(_SCENE_ID_PREFIX) or Path(live_scene_id).name != live_scene_id:
        raise ValueError("Live scene ID must be opaque and path-safe")
    if not validation_scene_ids or live_scene_id in validation_scene_ids:
        raise ValueError("Demo cache requires scene-disjoint validation IDs")
    live_path = _rooted(live_prefix_path)
    if not live_path.is_file() or live_path.is_symlink():
        raise FileNotFoundError("Live scene prefix must be a regular file")
    live_values = load_file(str(live_path), device="cpu")
    if set(live_values) != {"scene_prefix"}:
        raise ValueError("Live prefix archive must contain only scene_prefix")
    live_prefix = live_values["scene_prefix"]
    expected_shape = (1, expected_token_count, expected_hidden_size)
    if tuple(live_prefix.shape) != expected_shape or not torch.isfinite(live_prefix).all():
        raise ValueError("Live scene prefix has the wrong shape or non-finite values")
    reference = load_scene_prefix_cache(
        reference_cache_root,
        validation_scene_ids,
        expected_token_count=expected_token_count,
        expected_hidden_size=expected_hidden_size,
    )
    reference_boundary = reference.prefixes[validation_scene_ids[0]]
    if not torch.equal(live_prefix[:, :1], reference_boundary[:, :1]) or not torch.equal(
        live_prefix[:, -1:], reference_boundary[:, -1:]
    ):
        raise ValueError("Live prefix native Gemma boundaries differ from reference cache")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=parent))
    try:
        all_prefixes = {
            live_scene_id: live_prefix,
            **{scene_id: reference.prefixes[scene_id] for scene_id in validation_scene_ids},
        }
        entries: dict[str, Any] = {}
        for scene_id, tensor in all_prefixes.items():
            filename = f"{scene_id}.safetensors"
            output = temporary / filename
            save_file({"scene_prefix": tensor.contiguous()}, str(output))
            entries[scene_id] = {
                "filename": filename,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "file_size_bytes": output.stat().st_size,
                "file_sha256": _sha256_file(output),
                "prefix_sha256": prefix_sha256(tensor),
            }
        manifest = {
            "artifact": "question_independent_scene_prefix_cache_v1",
            "scene_count": len(entries),
            "complete_scene_prefixes": True,
            "question_inputs_used": False,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "live_scene_fit_in_training_split": live_scene_id,
            "validation_scene_count": len(validation_scene_ids),
            "source_live_prefix_file_sha256": _sha256_file(live_path),
            "scenes": entries,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root)
        return manifest
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()


def _read_output_field(value: object, name: str) -> torch.Tensor:
    if isinstance(value, Mapping):
        result = value.get(name)
    else:
        result = getattr(value, name, None)
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"Waypoint controller output has no tensor field {name!r}")
    return result


def normalize_policy_output(value: object, *, batch_size: int) -> WaypointPolicyTensors:
    def field(primary: str, runtime_name: str) -> torch.Tensor:
        try:
            return _read_output_field(value, primary)
        except TypeError:
            return _read_output_field(value, runtime_name)

    result = WaypointPolicyTensors(
        action_logits=_read_output_field(value, "action_logits"),
        waypoint_delta_robot_m=field(
            "waypoint_delta_robot_m", "waypoint_delta_robot_m"
        ),
        turn_delta_degrees=_read_output_field(value, "turn_delta_degrees"),
    )
    expected = {
        "action_logits": (batch_size, len(ACTION_NAMES)),
        "waypoint_delta_robot_m": (batch_size, 2),
        "turn_delta_degrees": (batch_size, 1),
    }
    for name, shape in expected.items():
        tensor = getattr(result, name)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
    return result


class ActualGemmaWaypointForward:
    """Execute a waypoint decision through the frozen local Gemma decoder."""

    def __init__(
        self,
        language: LocalLanguageModel,
        controller: nn.Module,
        robot_state_encoder: nn.Module,
        *,
        scene_token_count: int,
        robot_token_count: int,
        hidden_size: int,
        state_dim: int,
        history_dim: int,
    ) -> None:
        if language.backend_name != "gemma4" or language.prefix_backend is None:
            raise ValueError("Actual waypoint decisions require the Gemma-4 prefix backend")
        if language.hidden_size != hidden_size:
            raise ValueError("Configured waypoint hidden size differs from local Gemma")
        if scene_token_count < 3 or robot_token_count < 1 or state_dim < 1 or history_dim < 1:
            raise ValueError("Waypoint forward dimensions are invalid")
        if not isinstance(controller, GemmaWaypointControllerProtocol):
            raise TypeError(
                "Controller must implement forward_actual_gemma and "
                "forward_heads_from_cached_gemma_hidden"
            )
        if not isinstance(robot_state_encoder, nn.Module):
            raise TypeError("robot_state_encoder must be a torch module")
        self.language = language
        self.controller = controller
        self.robot_state_encoder = robot_state_encoder
        self.scene_token_count = scene_token_count
        self.robot_token_count = robot_token_count
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.history_dim = history_dim
        self.last_audit: dict[str, Any] | None = None
        self.last_decision_hidden: torch.Tensor | None = None

    def __call__(
        self,
        scene_prefix: torch.Tensor,
        instruction: str,
        state: torch.Tensor,
        history: torch.Tensor,
    ) -> WaypointPolicyTensors:
        if (
            scene_prefix.ndim != 3
            or scene_prefix.shape[0] < 1
            or tuple(scene_prefix.shape[1:])
            != (self.scene_token_count, self.hidden_size)
        ):
            raise ValueError("Waypoint forward requires complete scene prefixes")
        batch_size = int(scene_prefix.shape[0])
        if tuple(state.shape) != (batch_size, self.state_dim):
            raise ValueError("Waypoint state has the wrong shape")
        if (
            history.ndim != 3
            or history.shape[0] != batch_size
            or history.shape[2] != self.history_dim
        ):
            raise ValueError("Waypoint history has the wrong shape")
        if not instruction.strip():
            raise ValueError("Waypoint instruction cannot be empty")
        device = self.language.device
        scene = scene_prefix.to(device)
        state = state.to(device=device, dtype=torch.float32)
        history = history.to(device=device, dtype=torch.float32)
        robot_tokens = self.robot_state_encoder(state)
        if tuple(robot_tokens.shape) != (
            batch_size,
            self.robot_token_count,
            self.hidden_size,
        ):
            raise ValueError("Robot-state encoder produced the wrong token shape")
        if not torch.isfinite(robot_tokens).all():
            raise ValueError("Robot-state encoder produced non-finite tokens")
        active_prefix = insert_robot_state_tokens(scene, robot_tokens)
        decoded = self.controller.forward_actual_gemma(
            prefix_backend=self.language.prefix_backend,
            tokenizer=self.language.tokenizer,
            active_scene_robot_prefix=active_prefix,
            instruction=instruction,
            history_features=history,
        )
        result = normalize_policy_output(decoded, batch_size=batch_size)
        decision_hidden = getattr(decoded, "decision_hidden", None)
        if (
            not isinstance(decision_hidden, torch.Tensor)
            or tuple(decision_hidden.shape) != (batch_size, self.hidden_size)
            or not torch.isfinite(decision_hidden).all()
        ):
            raise RuntimeError("Actual Gemma policy did not expose its final decision hidden")
        self.last_decision_hidden = decision_hidden
        self.last_audit = {
            "actual_gemma_causal_forward": True,
            "output_hidden_states_requested": True,
            "complete_scene_token_count": self.scene_token_count,
            "robot_token_count": self.robot_token_count,
            "history_token_count": int(history.shape[1]),
            "decision_hidden_position": int(decoded.decision_position),
            "question_dependent_scene_retrieval": False,
        }
        return result


def _normalized_degrees(value: float) -> float:
    return math.degrees(math.atan2(math.sin(math.radians(value)), math.cos(math.radians(value))))


def _sample_turn_delta_degrees(sample: WaypointTraceSample) -> float:
    """Derive the robot-relative FACE label without changing cached inputs."""

    if sample.action_index != ACTION_TO_INDEX["face"]:
        return 0.0
    yaw = math.degrees(math.atan2(float(sample.state[3]), float(sample.state[4])))
    return _normalized_degrees(sample.heading_degrees - yaw)


def _targets(samples: Sequence[WaypointTraceSample]) -> dict[str, torch.Tensor]:
    return {
        "action": torch.tensor([sample.action_index for sample in samples], dtype=torch.long),
        "waypoint": torch.stack(
            [sample.waypoint_delta_robot_m for sample in samples]
        ),
        "turn_delta": torch.tensor(
            [[_sample_turn_delta_degrees(sample)] for sample in samples],
            dtype=torch.float32,
        ),
    }


def _circular_error_degrees(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = torch.deg2rad(predicted - target)
    return torch.abs(torch.rad2deg(torch.atan2(torch.sin(delta), torch.cos(delta))))


def waypoint_metrics(
    outputs: WaypointPolicyTensors,
    samples: Sequence[WaypointTraceSample],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot score an empty waypoint sample set")
    targets = _targets(samples)
    logits = outputs.action_logits.detach().float().cpu()
    waypoints = outputs.waypoint_delta_robot_m.detach().float().cpu()
    turn_deltas = outputs.turn_delta_degrees.detach().float().cpu()
    predicted = logits.argmax(dim=-1)
    action_target = targets["action"]
    move_mask = action_target == ACTION_TO_INDEX["move_to"]
    face_mask = action_target == ACTION_TO_INDEX["face"]
    stop_target = action_target == ACTION_TO_INDEX["stop"]
    stop_predicted = predicted == ACTION_TO_INDEX["stop"]
    true_positive = int((stop_target & stop_predicted).sum())
    false_positive = int((~stop_target & stop_predicted).sum())
    false_negative = int((stop_target & ~stop_predicted).sum())
    confusion = torch.stack(
        [
            torch.bincount(predicted[action_target == index], minlength=len(ACTION_NAMES))
            for index in range(len(ACTION_NAMES))
        ]
    )
    per_action_recall = {
        name: float(confusion[index, index] / max(1, int(confusion[index].sum())))
        for index, name in enumerate(ACTION_NAMES)
    }
    waypoint_errors = torch.linalg.vector_norm(
        waypoints[move_mask] - targets["waypoint"][move_mask], dim=-1
    )
    heading_errors = torch.abs(
        turn_deltas[face_mask, 0] - targets["turn_delta"][face_mask, 0]
    )
    return {
        "sample_count": len(samples),
        "action_accuracy": float((predicted == action_target).float().mean()),
        "action_macro_recall": sum(per_action_recall.values()) / len(ACTION_NAMES),
        "action_recall": per_action_recall,
        "action_confusion_target_by_prediction": confusion.tolist(),
        "move_to_sample_count": int(move_mask.sum()),
        "waypoint_error_m_mean": (
            None if not bool(move_mask.any()) else float(waypoint_errors.mean())
        ),
        "waypoint_error_m_median": (
            None if not bool(move_mask.any()) else float(waypoint_errors.median())
        ),
        "face_sample_count": int(face_mask.sum()),
        "heading_error_degrees_mean": (
            None if not bool(face_mask.any()) else float(heading_errors.mean())
        ),
        "heading_error_degrees_median": (
            None if not bool(face_mask.any()) else float(heading_errors.median())
        ),
        "stop_sample_count": int(stop_target.sum()),
        "stop_precision": true_positive / max(1, true_positive + false_positive),
        "stop_recall": true_positive / max(1, true_positive + false_negative),
    }


def waypoint_loss(
    outputs: WaypointPolicyTensors,
    samples: Sequence[WaypointTraceSample],
    *,
    max_waypoint_step_m: float,
    max_turn_delta_degrees: float,
    action_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    waypoint_weight: float = 1.0,
    heading_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = _targets(samples)
    device = outputs.action_logits.device
    action = targets["action"].to(device)
    # ``cross_entropy(weight=..., reduction="mean")`` divides the weighted
    # sum by the sum of observed class weights.  With the original one-sample
    # cached-head batches that cancels the class weight completely.  Apply the
    # weight explicitly so rare STOP supervision has the intended influence
    # for both singleton and normal mini-batches.
    classification_rows = torch.nn.functional.cross_entropy(
        outputs.action_logits,
        action,
        reduction="none",
    )
    if action_weights is not None:
        weights = action_weights.to(device=device, dtype=classification_rows.dtype)
        if tuple(weights.shape) != (len(ACTION_NAMES),) or not torch.isfinite(weights).all():
            raise ValueError("action_weights must be one finite value per action")
        classification_rows = classification_rows * weights[action]
    row_weights: torch.Tensor | None = None
    if sample_weights is not None:
        row_weights = sample_weights.to(
            device=device, dtype=classification_rows.dtype
        )
        if (
            tuple(row_weights.shape) != (len(samples),)
            or not torch.isfinite(row_weights).all()
            or bool((row_weights <= 0.0).any())
        ):
            raise ValueError("sample_weights must be one finite positive value per row")
        classification = (classification_rows * row_weights).sum() / row_weights.sum()
    else:
        classification = classification_rows.mean()
    move_mask = action == ACTION_TO_INDEX["move_to"]
    face_mask = action == ACTION_TO_INDEX["face"]
    zero = outputs.action_logits.sum() * 0.0
    if bool(move_mask.any()):
        maximum_step = float(max_waypoint_step_m)
        scale = torch.tensor([maximum_step, maximum_step], device=device)
        waypoint_rows = torch.nn.functional.smooth_l1_loss(
            outputs.waypoint_delta_robot_m[move_mask] / scale,
            targets["waypoint"].to(device)[move_mask] / scale,
            reduction="none",
        )
        waypoint_rows = waypoint_rows.mean(dim=-1)
        waypoint = (
            waypoint_rows.mean()
            if row_weights is None
            else (waypoint_rows * row_weights[move_mask]).sum()
            / row_weights[move_mask].sum()
        )
    else:
        waypoint = zero
    if bool(face_mask.any()):
        maximum_turn = float(max_turn_delta_degrees)
        if not math.isfinite(maximum_turn) or maximum_turn <= 0.0:
            raise ValueError("max_turn_delta_degrees must be finite and positive")
        heading_rows = torch.nn.functional.smooth_l1_loss(
            outputs.turn_delta_degrees[face_mask].float() / maximum_turn,
            targets["turn_delta"].to(device)[face_mask] / maximum_turn,
            reduction="none",
        )
        heading_rows = heading_rows.mean(dim=-1)
        heading = (
            heading_rows.mean()
            if row_weights is None
            else (heading_rows * row_weights[face_mask]).sum()
            / row_weights[face_mask].sum()
        )
    else:
        heading = zero
    total = classification + waypoint_weight * waypoint + heading_weight * heading
    return total, {
        "classification": float(classification.detach()),
        "waypoint": float(waypoint.detach()),
        "heading": float(heading.detach()),
        "total": float(total.detach()),
    }


def waypoint_retention_loss(
    outputs: WaypointPolicyTensors,
    reference_outputs: WaypointPolicyTensors,
    samples: Sequence[WaypointTraceSample],
    shared_mask: torch.Tensor,
    *,
    max_waypoint_step_m: float,
    max_turn_delta_degrees: float,
    logit_weight: float,
    waypoint_weight: float,
    heading_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Preserve a prior learned policy on exact, authenticated shared rows.

    The action term compares centered logits, retaining all pairwise class
    margins while ignoring the softmax-invariant common offset. Numeric terms
    apply only where the reference row actually executes that branch. This is
    training-time distillation over continuous Gemma states; it adds no
    runtime planner, label lookup, or action substitution.
    """

    batch = len(samples)
    if batch < 1:
        raise ValueError("retention loss requires at least one sample")
    device = outputs.action_logits.device
    mask = shared_mask.to(device=device, dtype=torch.bool)
    if tuple(mask.shape) != (batch,):
        raise ValueError("shared_mask must have one value per retention row")
    for name, value in (
        ("logit_weight", logit_weight),
        ("waypoint_weight", waypoint_weight),
        ("heading_weight", heading_weight),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"retention {name} must be finite and nonnegative")
    reference = WaypointPolicyTensors(
        action_logits=reference_outputs.action_logits.detach().to(device),
        waypoint_delta_robot_m=reference_outputs.waypoint_delta_robot_m.detach().to(
            device
        ),
        turn_delta_degrees=reference_outputs.turn_delta_degrees.detach().to(device),
    )
    expected_shapes = ((batch, len(ACTION_NAMES)), (batch, 2), (batch, 1))
    for name, value, expected in zip(
        ("action_logits", "waypoint_delta_robot_m", "turn_delta_degrees"),
        (
            outputs.action_logits,
            outputs.waypoint_delta_robot_m,
            outputs.turn_delta_degrees,
        ),
        expected_shapes,
        strict=True,
    ):
        reference_value = getattr(reference, name)
        if (
            tuple(value.shape) != expected
            or tuple(reference_value.shape) != expected
            or not torch.isfinite(value.float()).all()
            or not torch.isfinite(reference_value.float()).all()
        ):
            raise ValueError("retention outputs must align with samples")
    zero = outputs.action_logits.sum() * 0.0
    if not bool(mask.any()):
        return zero, {
            "logit": 0.0,
            "waypoint": 0.0,
            "heading": 0.0,
            "total": 0.0,
            "shared_rows": 0.0,
        }

    current_logits = outputs.action_logits[mask].float()
    teacher_logits = reference.action_logits[mask].float()
    current_logits = current_logits - current_logits.mean(dim=-1, keepdim=True)
    teacher_logits = teacher_logits - teacher_logits.mean(dim=-1, keepdim=True)
    logit = torch.nn.functional.mse_loss(current_logits, teacher_logits)

    action = torch.tensor(
        [sample.action_index for sample in samples], dtype=torch.long, device=device
    )
    move_mask = mask & (action == ACTION_TO_INDEX["move_to"])
    face_mask = mask & (action == ACTION_TO_INDEX["face"])
    if bool(move_mask.any()):
        waypoint = torch.nn.functional.smooth_l1_loss(
            outputs.waypoint_delta_robot_m[move_mask].float()
            / float(max_waypoint_step_m),
            reference.waypoint_delta_robot_m[move_mask].float()
            / float(max_waypoint_step_m),
        )
    else:
        waypoint = zero
    if bool(face_mask.any()):
        heading = torch.nn.functional.smooth_l1_loss(
            outputs.turn_delta_degrees[face_mask].float()
            / float(max_turn_delta_degrees),
            reference.turn_delta_degrees[face_mask].float()
            / float(max_turn_delta_degrees),
        )
    else:
        heading = zero
    total = (
        float(logit_weight) * logit
        + float(waypoint_weight) * waypoint
        + float(heading_weight) * heading
    )
    return total, {
        "logit": float(logit.detach()),
        "waypoint": float(waypoint.detach()),
        "heading": float(heading.detach()),
        "total": float(total.detach()),
        "shared_rows": float(mask.sum()),
    }


def waypoint_retention_metrics(
    outputs: WaypointPolicyTensors,
    reference_outputs: WaypointPolicyTensors,
    samples: Sequence[WaypointTraceSample],
    shared_mask: torch.Tensor,
) -> dict[str, Any]:
    """Measure behavioral drift from a prior policy on exact shared rows."""

    mask = shared_mask.detach().to(dtype=torch.bool, device="cpu")
    if tuple(mask.shape) != (len(samples),) or not bool(mask.any()):
        raise ValueError("retention metrics require authenticated shared rows")
    current = WaypointPolicyTensors(
        action_logits=outputs.action_logits.detach().float().cpu(),
        waypoint_delta_robot_m=outputs.waypoint_delta_robot_m.detach().float().cpu(),
        turn_delta_degrees=outputs.turn_delta_degrees.detach().float().cpu(),
    )
    teacher = WaypointPolicyTensors(
        action_logits=reference_outputs.action_logits.detach().float().cpu(),
        waypoint_delta_robot_m=reference_outputs.waypoint_delta_robot_m.detach()
        .float()
        .cpu(),
        turn_delta_degrees=reference_outputs.turn_delta_degrees.detach().float().cpu(),
    )
    action = torch.tensor([sample.action_index for sample in samples], dtype=torch.long)
    current_centered = current.action_logits - current.action_logits.mean(
        dim=-1, keepdim=True
    )
    teacher_centered = teacher.action_logits - teacher.action_logits.mean(
        dim=-1, keepdim=True
    )
    logit_delta = current_centered[mask] - teacher_centered[mask]
    move = mask & (action == ACTION_TO_INDEX["move_to"])
    face = mask & (action == ACTION_TO_INDEX["face"])
    waypoint_delta = torch.linalg.vector_norm(
        current.waypoint_delta_robot_m[move]
        - teacher.waypoint_delta_robot_m[move],
        dim=-1,
    )
    heading_delta = torch.abs(
        current.turn_delta_degrees[face, 0] - teacher.turn_delta_degrees[face, 0]
    )
    waypoint_mean = None if waypoint_delta.numel() == 0 else float(waypoint_delta.mean())
    waypoint_max = None if waypoint_delta.numel() == 0 else float(waypoint_delta.max())
    heading_mean = None if heading_delta.numel() == 0 else float(heading_delta.mean())
    heading_max = None if heading_delta.numel() == 0 else float(heading_delta.max())
    new_mask = ~mask
    new_predictions = current.action_logits[new_mask].argmax(dim=-1)
    return {
        "shared_training_rows": int(mask.sum()),
        "new_training_rows": int(new_mask.sum()),
        "shared_action_agreement": float(
            (
                current.action_logits[mask].argmax(dim=-1)
                == teacher.action_logits[mask].argmax(dim=-1)
            )
            .float()
            .mean()
        ),
        "shared_centered_logit_rmse": float(logit_delta.square().mean().sqrt()),
        "shared_centered_logit_max_abs": float(logit_delta.abs().max()),
        "shared_move_waypoint_drift_m_mean": waypoint_mean,
        "shared_move_waypoint_drift_m_max": waypoint_max,
        "shared_face_heading_drift_degrees_mean": heading_mean,
        "shared_face_heading_drift_degrees_max": heading_max,
        "new_action_accuracy": (
            None
            if not bool(new_mask.any())
            else float((new_predictions == action[new_mask]).float().mean())
        ),
    }


def _waypoint_retention_gate_report(
    final_retention: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Return fail-closed retention gates with machine-readable observations."""

    minimum_agreement = float(settings["retention_minimum_shared_action_agreement"])
    minimum_new_accuracy = float(settings["retention_minimum_new_action_accuracy"])
    specifications = (
        (
            "shared_action_agreement",
            final_retention["shared_action_agreement"],
            "minimum",
            minimum_agreement,
        ),
        (
            "new_action_accuracy",
            final_retention["new_action_accuracy"],
            "minimum",
            minimum_new_accuracy,
        ),
        (
            "shared_centered_logit_rmse",
            final_retention["shared_centered_logit_rmse"],
            "maximum",
            settings["retention_maximum_shared_centered_logit_rmse"],
        ),
        (
            "shared_move_waypoint_drift_m_max",
            final_retention["shared_move_waypoint_drift_m_max"],
            "maximum",
            settings["retention_maximum_shared_waypoint_drift_m"],
        ),
        (
            "shared_face_heading_drift_degrees_max",
            final_retention["shared_face_heading_drift_degrees_max"],
            "maximum",
            settings["retention_maximum_shared_heading_drift_degrees"],
        ),
    )
    failure_details: list[dict[str, Any]] = []
    for metric, observed, comparison, required in specifications:
        # A disabled gate has no threshold. The new-action minimum of zero is
        # also the existing explicit opt-out for datasets with no new rows.
        if required is None or (metric == "new_action_accuracy" and required == 0.0):
            continue
        failed = observed is None
        if observed is not None:
            failed = (
                float(observed) < float(required)
                if comparison == "minimum"
                else float(observed) > float(required)
            )
        if failed:
            failure_details.append(
                {
                    "metric": metric,
                    "observed": observed,
                    "comparison": comparison,
                    "required": required,
                }
            )
    return {
        "minimum_shared_action_agreement": minimum_agreement,
        "minimum_new_action_accuracy": minimum_new_accuracy,
        "maximum_shared_centered_logit_rmse": settings[
            "retention_maximum_shared_centered_logit_rmse"
        ],
        "maximum_shared_waypoint_drift_m": settings[
            "retention_maximum_shared_waypoint_drift_m"
        ],
        "maximum_shared_heading_drift_degrees": settings[
            "retention_maximum_shared_heading_drift_degrees"
        ],
        "passed": not failure_details,
        "failures": [detail["metric"] for detail in failure_details],
        "failure_details": failure_details,
    }


def _waypoint_retention_gate_failure_message(gates: Mapping[str, Any]) -> str:
    """Serialize every failed metric value so an aborted run remains diagnosable."""

    failures = gates.get("failures")
    details = gates.get("failure_details")
    if not isinstance(failures, list) or not failures or not isinstance(details, list):
        raise ValueError("retention gate failure message requires failed diagnostics")
    return (
        "Waypoint retention candidate missed configured gates: "
        + ", ".join(str(value) for value in failures)
        + "; diagnostics="
        + json.dumps(
            details,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def refit_waypoint_action_classifier(
    controller: nn.Module,
    hidden: torch.Tensor,
    samples: Sequence[WaypointTraceSample],
    *,
    max_iter: int,
    learning_rate: float,
    l2_weight: float = 0.0,
    action_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    reference_logits: torch.Tensor | None = None,
    retention_mask: torch.Tensor | None = None,
    retention_weight: float = 0.0,
) -> dict[str, Any]:
    """Calibrate the linear action readout over frozen Gemma states.

    The joint AdamW stage is selected on held-out scenes and therefore can
    leave a few nearly tied training actions even when the cached hidden states
    are linearly separable.  This optional second stage is ordinary supervised
    multinomial logistic regression on *training rows only*. Optional L2
    regularization keeps separable-data margins finite. It deliberately freezes
    the shared normalization and both numeric regression branches so improving
    MOVE_TO/FACE/STOP classification cannot change a waypoint or heading.
    """

    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("action refit max_iter must be a positive integer")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise ValueError("action refit learning_rate must be finite and positive")
    if (
        isinstance(l2_weight, bool)
        or not isinstance(l2_weight, (int, float))
        or not math.isfinite(float(l2_weight))
        or float(l2_weight) < 0.0
    ):
        raise ValueError("action refit l2_weight must be finite and nonnegative")
    if (
        isinstance(retention_weight, bool)
        or not isinstance(retention_weight, (int, float))
        or not math.isfinite(float(retention_weight))
        or float(retention_weight) < 0.0
    ):
        raise ValueError("action refit retention_weight must be finite and nonnegative")
    if hidden.ndim != 2 or hidden.shape[0] != len(samples) or not torch.isfinite(
        hidden.float()
    ).all():
        raise ValueError("action refit hidden states are invalid")
    numeric_heads = getattr(controller, "numeric_heads", None)
    input_norm = getattr(numeric_heads, "input_norm", None)
    action = getattr(numeric_heads, "action", None)
    if not isinstance(numeric_heads, nn.Module) or not isinstance(
        input_norm, nn.LayerNorm
    ) or not isinstance(action, nn.Linear):
        raise TypeError("action refit requires the standard Gemma waypoint heads")

    # LBFGS line search is deterministic and reliable on CPU.  The controller
    # is small; Gemma remains on its inference device and is never moved.
    controller.to("cpu")
    controller.eval()
    frozen_state = {
        name: value.detach().clone()
        for name, value in numeric_heads.state_dict().items()
        if not name.startswith("action.")
    }
    # Do not create ``features`` under inference_mode: LBFGS autograd must save
    # this constant input while differentiating the action weights.
    with torch.no_grad():
        features = input_norm(hidden.detach().float().cpu()).detach().clone()
    targets = torch.tensor(
        [sample.action_index for sample in samples], dtype=torch.long
    )
    weights = None
    if action_weights is not None:
        weights = action_weights.detach().float().cpu()
        if tuple(weights.shape) != (len(ACTION_NAMES),) or not torch.isfinite(
            weights
        ).all():
            raise ValueError("action refit weights are invalid")
    row_weights = None
    if sample_weights is not None:
        row_weights = sample_weights.detach().float().cpu()
        if (
            tuple(row_weights.shape) != (len(samples),)
            or not torch.isfinite(row_weights).all()
            or bool((row_weights <= 0.0).any())
        ):
            raise ValueError("action refit sample weights are invalid")
    teacher_logits = None
    shared_mask = None
    if reference_logits is not None or retention_mask is not None:
        if reference_logits is None or retention_mask is None:
            raise ValueError("action refit retention reference is incomplete")
        teacher_logits = reference_logits.detach().float().cpu()
        shared_mask = retention_mask.detach().to(dtype=torch.bool, device="cpu")
        if (
            tuple(teacher_logits.shape) != (len(samples), len(ACTION_NAMES))
            or tuple(shared_mask.shape) != (len(samples),)
            or not torch.isfinite(teacher_logits).all()
            or not bool(shared_mask.any())
        ):
            raise ValueError("action refit retention reference is invalid")
    elif float(retention_weight) > 0.0:
        raise ValueError("action refit retention_weight requires a reference")

    def classification_loss() -> torch.Tensor:
        rows = torch.nn.functional.cross_entropy(
            action(features), targets, reduction="none"
        )
        if weights is not None:
            rows = rows * weights[targets]
        if row_weights is None:
            return rows.mean()
        return (rows * row_weights).sum() / row_weights.sum()

    def retention_loss() -> torch.Tensor:
        if teacher_logits is None or shared_mask is None:
            return action.weight.sum() * 0.0
        observed = action(features[shared_mask])
        expected = teacher_logits[shared_mask]
        observed = observed - observed.mean(dim=-1, keepdim=True)
        expected = expected - expected.mean(dim=-1, keepdim=True)
        return torch.nn.functional.mse_loss(observed, expected)

    def action_objective() -> torch.Tensor:
        classification = classification_loss()
        # Keep the legacy zero-regularization path exactly unchanged.  The
        # finite penalty prevents separable training rows from sending the
        # multinomial readout norm toward infinity and creating razor-thin
        # closed-loop action boundaries.  Biases are intentionally unpenalized.
        objective = classification + float(retention_weight) * retention_loss()
        if float(l2_weight) != 0.0:
            objective = (
                objective
                + 0.5 * float(l2_weight) * action.weight.square().sum()
            )
        return objective

    with torch.no_grad():
        initial_logits = action(features)
        initial_loss = float(classification_loss())
        initial_retention_loss = float(retention_loss())
        initial_objective = float(action_objective())
        initial_accuracy = float(
            (initial_logits.argmax(dim=-1) == targets).float().mean()
        )
        initial_weight_l2_norm = float(action.weight.norm())
        initial_bias_l2_norm = float(action.bias.norm())
    optimizer = torch.optim.LBFGS(
        action.parameters(),
        lr=float(learning_rate),
        max_iter=max_iter,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = action_objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final_logits = action(features)
        final_loss = float(classification_loss())
        final_retention_loss = float(retention_loss())
        final_objective = float(action_objective())
        predictions = final_logits.argmax(dim=-1)
        final_accuracy = float((predictions == targets).float().mean())
        remaining_errors = int((predictions != targets).sum())
        final_weight_l2_norm = float(action.weight.norm())
        final_bias_l2_norm = float(action.bias.norm())
        shared_action_agreement = (
            None
            if teacher_logits is None or shared_mask is None
            else float(
                (
                    final_logits[shared_mask].argmax(dim=-1)
                    == teacher_logits[shared_mask].argmax(dim=-1)
                )
                .float()
                .mean()
            )
        )
    for name, expected in frozen_state.items():
        observed = numeric_heads.state_dict()[name]
        if not torch.equal(observed, expected):
            raise RuntimeError(f"Action refit changed frozen numeric tensor {name}")
    return {
        "enabled": True,
        "optimizer": "lbfgs_multinomial_logistic_regression",
        "training_rows_only": True,
        "max_iterations": max_iter,
        "learning_rate": float(learning_rate),
        "l2_weight": float(l2_weight),
        "retention_weight": float(retention_weight),
        "retention_shared_rows": 0 if shared_mask is None else int(shared_mask.sum()),
        "sample_weighting_enabled": row_weights is not None,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_retention_loss": initial_retention_loss,
        "final_retention_loss": final_retention_loss,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "initial_weight_l2_norm": initial_weight_l2_norm,
        "final_weight_l2_norm": final_weight_l2_norm,
        "initial_bias_l2_norm": initial_bias_l2_norm,
        "final_bias_l2_norm": final_bias_l2_norm,
        "initial_action_accuracy": initial_accuracy,
        "final_action_accuracy": final_accuracy,
        "remaining_action_errors": remaining_errors,
        "shared_reference_action_agreement": shared_action_agreement,
        "shared_norm_and_numeric_branches_unchanged": True,
    }


def refit_waypoint_action_classifier_constrained(
    controller: nn.Module,
    hidden: torch.Tensor,
    samples: Sequence[WaypointTraceSample],
    *,
    reference_logits: torch.Tensor,
    reference_action_weight: torch.Tensor,
    reference_action_bias: torch.Tensor,
    retention_mask: torch.Tensor,
    positive_margin: float = 0.001,
    covariance_ridge: float = 1e-9,
    feasibility_tolerance: float = 1e-7,
    maximum_centered_logit_rmse: float | None = None,
    max_active_set_iterations: int = 512,
    max_shared_cut_iterations: int = 16,
) -> dict[str, Any]:
    """Minimally repair a retained linear action head under exact constraints.

    Shared rows retain the teacher argmax; new rows use their supervised action.
    Each desired action must beat both alternatives by ``positive_margin``.
    The convex objective is squared centered-logit drift over shared rows.  A
    deterministic CPU float64 active-set dual solve avoids a scalar
    cross-entropy/retention tradeoff and mutates the head only after validating
    every shared and new constraint with the deployed float32 parameters.
    """

    for name, value, positive in (
        ("positive_margin", positive_margin, True),
        ("covariance_ridge", covariance_ridge, True),
        ("feasibility_tolerance", feasibility_tolerance, True),
    ):
        if not math.isfinite(float(value)) or (float(value) <= 0.0 if positive else False):
            raise ValueError(f"constrained action refit {name} must be finite and positive")
    for name, value in (
        ("max_active_set_iterations", max_active_set_iterations),
        ("max_shared_cut_iterations", max_shared_cut_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"constrained action refit {name} must be positive")
    if maximum_centered_logit_rmse is not None and (
        not math.isfinite(float(maximum_centered_logit_rmse))
        or float(maximum_centered_logit_rmse) <= 0.0
    ):
        raise ValueError("constrained action refit RMSE gate must be positive")
    if hidden.ndim != 2 or hidden.shape[0] != len(samples) or not torch.isfinite(
        hidden.float()
    ).all():
        raise ValueError("constrained action refit hidden states are invalid")
    numeric_heads = getattr(controller, "numeric_heads", None)
    input_norm = getattr(numeric_heads, "input_norm", None)
    action = getattr(numeric_heads, "action", None)
    if not isinstance(numeric_heads, nn.Module) or not isinstance(
        input_norm, nn.LayerNorm
    ) or not isinstance(action, nn.Linear):
        raise TypeError("constrained action refit requires the standard linear head")
    teacher = reference_logits.detach().to(dtype=torch.float64, device="cpu")
    teacher_weight = reference_action_weight.detach().float().cpu()
    teacher_bias = reference_action_bias.detach().float().cpu()
    shared = retention_mask.detach().to(dtype=torch.bool, device="cpu")
    if (
        tuple(teacher.shape) != (len(samples), len(ACTION_NAMES))
        or tuple(teacher_weight.shape) != tuple(action.weight.shape)
        or tuple(teacher_bias.shape) != tuple(action.bias.shape)
        or tuple(shared.shape) != (len(samples),)
        or not bool(shared.any())
        or not bool((~shared).any())
        or not torch.isfinite(teacher).all()
    ):
        raise ValueError("constrained action refit retention reference is invalid")

    controller.to("cpu")
    controller.eval()
    frozen_state = {
        name: value.detach().clone()
        for name, value in numeric_heads.state_dict().items()
        if not name.startswith("action.")
    }
    with torch.no_grad():
        normalized = input_norm(hidden.detach().float().cpu()).detach()
        materialized_teacher = torch.nn.functional.linear(
            normalized, teacher_weight, teacher_bias
        )
    teacher_materialization_max_abs = float(
        (materialized_teacher - teacher.float()).abs().max()
    )
    if (
        not torch.equal(materialized_teacher.argmax(dim=-1), teacher.argmax(dim=-1))
        or teacher_materialization_max_abs > float(positive_margin) * 0.25
    ):
        raise ValueError(
            "constrained action reference materialization is not margin-safe: "
            f"max_abs={teacher_materialization_max_abs:.12g}"
        )
    base_logits = materialized_teacher.to(dtype=torch.float64)
    features = torch.cat(
        (
            normalized.to(dtype=torch.float64),
            torch.ones(len(samples), 1, dtype=torch.float64),
        ),
        dim=-1,
    )
    labels = torch.tensor(
        [sample.action_index for sample in samples], dtype=torch.long
    )
    teacher_actions = teacher.argmax(dim=-1)
    desired_actions = torch.where(shared, teacher_actions, labels)
    shared_features = features[shared]
    covariance = shared_features.T @ shared_features / int(shared.sum())
    covariance = covariance + float(covariance_ridge) * torch.eye(
        covariance.shape[0], dtype=torch.float64
    )
    try:
        torch.linalg.cholesky(covariance)
    except RuntimeError as exc:
        raise RuntimeError("constrained action covariance is not positive definite") from exc

    constraints: list[tuple[int, int, int]] = []
    materialized: set[tuple[int, int, int]] = set()

    def add_constraint(index: int, desired: int, alternative: int) -> None:
        key = (index, desired, alternative)
        if key not in materialized:
            materialized.add(key)
            constraints.append(key)

    for index in torch.nonzero(~shared).flatten().tolist():
        desired = int(desired_actions[index])
        for alternative in range(len(ACTION_NAMES)):
            if alternative != desired:
                add_constraint(index, desired, alternative)

    active_count = 0
    active_iterations = 0
    shared_cut_iterations = 0
    delta = torch.zeros(features.shape[1], len(ACTION_NAMES), dtype=torch.float64)

    def solve_materialized() -> tuple[torch.Tensor, int, int]:
        row_indices = torch.tensor(
            [index for index, _desired, _alternative in constraints], dtype=torch.long
        )
        class_vectors = torch.zeros(
            len(constraints), len(ACTION_NAMES), dtype=torch.float64
        )
        required = torch.empty(len(constraints), dtype=torch.float64)
        for position, (index, desired, alternative) in enumerate(constraints):
            class_vectors[position, desired] = 1.0
            class_vectors[position, alternative] = -1.0
            required[position] = float(positive_margin) - (
                base_logits[index, desired] - base_logits[index, alternative]
            )
        constraint_features = features[row_indices]
        solved_features = torch.linalg.solve(covariance, constraint_features.T).T
        kernel = (constraint_features @ solved_features.T) * (
            class_vectors @ class_vectors.T
        )
        dual = torch.zeros(len(constraints), dtype=torch.float64)
        active: list[int] = []
        iterations = 0
        while iterations < max_active_set_iterations:
            violation = required - kernel @ dual
            worst = int(torch.argmax(violation))
            if float(violation[worst]) <= float(feasibility_tolerance):
                break
            if worst not in active:
                active.append(worst)
            while active:
                active_tensor = torch.tensor(active, dtype=torch.long)
                active_kernel = kernel[active_tensor][:, active_tensor]
                active_required = required[active_tensor]
                try:
                    solution = torch.linalg.solve(active_kernel, active_required)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "constrained action active set is singular"
                    ) from exc
                negative = torch.nonzero(
                    solution < -float(feasibility_tolerance)
                ).flatten()
                if not len(negative):
                    dual.zero_()
                    dual[active_tensor] = solution.clamp_min(0.0)
                    break
                remove_position = int(negative[torch.argmin(solution[negative])])
                del active[remove_position]
            iterations += 1
        final_violation = required - kernel @ dual
        if float(final_violation.max()) > float(feasibility_tolerance):
            raise RuntimeError(
                "constrained action dual solve did not reach feasibility: "
                f"maximum_violation={float(final_violation.max()):.12g}"
            )
        solved_delta = torch.linalg.solve(
            covariance,
            constraint_features.T @ (dual[:, None] * class_vectors),
        )
        return solved_delta, len(active), iterations

    for shared_cut_iterations in range(1, max_shared_cut_iterations + 1):
        delta, active_count, iterations = solve_materialized()
        active_iterations += iterations
        candidate = base_logits + features @ delta
        added = 0
        for index in torch.nonzero(shared).flatten().tolist():
            desired = int(desired_actions[index])
            for alternative in range(len(ACTION_NAMES)):
                if alternative == desired:
                    continue
                observed_margin = candidate[index, desired] - candidate[index, alternative]
                if float(observed_margin) < float(positive_margin) - float(
                    feasibility_tolerance
                ):
                    before = len(constraints)
                    add_constraint(index, desired, alternative)
                    added += len(constraints) - before
        if added == 0:
            break
    else:
        raise RuntimeError("constrained action shared-cut solve did not converge")

    delta_float = delta.to(dtype=torch.float32)
    candidate_weight = teacher_weight + delta_float[:-1].T
    candidate_bias = teacher_bias + delta_float[-1]
    candidate_logits = torch.nn.functional.linear(
        normalized.float(), candidate_weight, candidate_bias
    )
    desired_logits = candidate_logits.gather(1, desired_actions[:, None])[:, 0]
    alternatives = candidate_logits.clone()
    alternatives.scatter_(1, desired_actions[:, None], -torch.inf)
    observed_margins = desired_logits - alternatives.max(dim=-1).values
    shared_agreement = float(
        (candidate_logits[shared].argmax(dim=-1) == teacher_actions[shared])
        .float()
        .mean()
    )
    new_accuracy = float(
        (candidate_logits[~shared].argmax(dim=-1) == labels[~shared]).float().mean()
    )
    centered_candidate = candidate_logits - candidate_logits.mean(dim=-1, keepdim=True)
    centered_teacher = teacher.float() - teacher.float().mean(dim=-1, keepdim=True)
    shared_delta = centered_candidate[shared] - centered_teacher[shared]
    centered_rmse = float(shared_delta.square().mean().sqrt())
    centered_max_abs = float(shared_delta.abs().max())
    minimum_shared_margin = float(observed_margins[shared].min())
    minimum_new_margin = float(observed_margins[~shared].min())
    failure = None
    if shared_agreement != 1.0 or new_accuracy != 1.0:
        failure = "action constraints changed a required action"
    elif min(minimum_shared_margin, minimum_new_margin) <= 0.0:
        failure = "action constraints did not retain a positive deployed margin"
    elif (
        maximum_centered_logit_rmse is not None
        and centered_rmse > float(maximum_centered_logit_rmse)
    ):
        failure = "action constraints exceeded the centered-logit RMSE gate"
    if failure is not None:
        raise RuntimeError(
            "Constrained waypoint action refit rejected before mutation: "
            f"reason={failure}; shared_action_agreement={shared_agreement:.12g}; "
            f"new_action_accuracy={new_accuracy:.12g}; "
            f"shared_centered_logit_rmse={centered_rmse:.12g}; "
            f"maximum_allowed_rmse={maximum_centered_logit_rmse!r}; "
            f"minimum_shared_margin={minimum_shared_margin:.12g}; "
            f"minimum_new_margin={minimum_new_margin:.12g}"
        )

    with torch.no_grad():
        action.weight.copy_(candidate_weight)
        action.bias.copy_(candidate_bias)
    for name, expected in frozen_state.items():
        if not torch.equal(numeric_heads.state_dict()[name].detach().cpu(), expected):
            raise RuntimeError(f"Constrained action refit changed frozen tensor {name}")
    return {
        "enabled": True,
        "optimizer": "deterministic_active_set_minimum_centered_drift_qp",
        "training_rows_only": True,
        "linear_action_head_only": True,
        "teacher_action_constraints": int(shared.sum()) * (len(ACTION_NAMES) - 1),
        "new_label_constraints": int((~shared).sum()) * (len(ACTION_NAMES) - 1),
        "materialized_cutting_plane_constraints": len(constraints),
        "active_constraints": active_count,
        "active_set_iterations": active_iterations,
        "shared_cut_iterations": shared_cut_iterations,
        "positive_margin_required": float(positive_margin),
        "minimum_shared_margin": minimum_shared_margin,
        "minimum_new_margin": minimum_new_margin,
        "covariance_ridge": float(covariance_ridge),
        "teacher_parameter_materialization_max_abs": teacher_materialization_max_abs,
        "teacher_parameter_materialization_argmax_identical": True,
        "feasibility_tolerance": float(feasibility_tolerance),
        "shared_action_agreement": shared_agreement,
        "new_action_accuracy": new_accuracy,
        "shared_centered_logit_rmse": centered_rmse,
        "shared_centered_logit_max_abs": centered_max_abs,
        "maximum_centered_logit_rmse": maximum_centered_logit_rmse,
        "all_action_constraints_validated_after_float32_materialization": True,
        "shared_norm_and_numeric_branches_unchanged": True,
        "candidate_mutated_only_after_all_gates_passed": True,
    }


def refit_waypoint_heading_head(
    controller: nn.Module,
    hidden: torch.Tensor,
    samples: Sequence[WaypointTraceSample],
    *,
    steps: int,
    learning_rate: float,
    max_turn_degrees: float,
    device: torch.device,
    sample_weights: torch.Tensor | None = None,
    reference_turn_deltas: torch.Tensor | None = None,
    retention_mask: torch.Tensor | None = None,
    retention_weight: float = 0.0,
) -> dict[str, Any]:
    """Calibrate only absolute FACE headings on authenticated training rows."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("heading refit steps must be a positive integer")
    if (
        not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
        or not math.isfinite(float(max_turn_degrees))
        or float(max_turn_degrees) <= 0.0
    ):
        raise ValueError("heading refit bounds must be finite and positive")
    if not math.isfinite(float(retention_weight)) or float(retention_weight) < 0.0:
        raise ValueError("heading refit retention_weight must be finite and nonnegative")
    numeric_heads = getattr(controller, "numeric_heads", None)
    input_norm = getattr(numeric_heads, "input_norm", None)
    heading = getattr(numeric_heads, "heading", None)
    if not isinstance(numeric_heads, nn.Module) or not isinstance(
        input_norm, nn.LayerNorm
    ) or not isinstance(heading, nn.Module):
        raise TypeError("heading refit requires the standard Gemma waypoint heads")
    face_indices = [
        index
        for index, sample in enumerate(samples)
        if sample.action_index == ACTION_TO_INDEX["face"]
    ]
    if not face_indices:
        raise ValueError("heading refit requires FACE training rows")
    controller.to(device)
    controller.eval()
    frozen_state = {
        name: value.detach().cpu().clone()
        for name, value in numeric_heads.state_dict().items()
        if not name.startswith("heading.")
    }
    with torch.no_grad():
        features = input_norm(hidden.detach().float().to(device)).detach().clone()
    face_tensor = torch.tensor(face_indices, dtype=torch.long, device=device)
    target_degrees = torch.tensor(
        [[_sample_turn_delta_degrees(samples[index])] for index in face_indices],
        dtype=torch.float32,
        device=device,
    )
    face_weights = None
    if sample_weights is not None:
        values = sample_weights.detach().float().to(device)
        if (
            tuple(values.shape) != (len(samples),)
            or not torch.isfinite(values).all()
            or bool((values <= 0.0).any())
        ):
            raise ValueError("heading refit sample weights are invalid")
        face_weights = values[face_tensor]
    teacher_face = None
    shared_face = None
    if reference_turn_deltas is not None or retention_mask is not None:
        if reference_turn_deltas is None or retention_mask is None:
            raise ValueError("heading refit retention reference is incomplete")
        teacher = reference_turn_deltas.detach().float().to(device)
        shared = retention_mask.detach().to(device=device, dtype=torch.bool)
        if (
            tuple(teacher.shape) != (len(samples), 1)
            or tuple(shared.shape) != (len(samples),)
            or not torch.isfinite(teacher).all()
            or not bool(shared.any())
        ):
            raise ValueError("heading refit retention reference is invalid")
        teacher_face = teacher[face_tensor]
        shared_face = shared[face_tensor]
    elif float(retention_weight) > 0.0:
        raise ValueError("heading refit retention_weight requires a reference")
    optimizer = torch.optim.AdamW(
        heading.parameters(), lr=float(learning_rate), weight_decay=0.0
    )
    def losses(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        supervised_rows = torch.nn.functional.smooth_l1_loss(
            prediction / float(max_turn_degrees),
            target_degrees / float(max_turn_degrees),
            reduction="none",
        ).mean(dim=-1)
        supervised = (
            supervised_rows.mean()
            if face_weights is None
            else (supervised_rows * face_weights).sum() / face_weights.sum()
        )
        if teacher_face is None or shared_face is None or not bool(shared_face.any()):
            retention = prediction.sum() * 0.0
        else:
            retention = torch.nn.functional.smooth_l1_loss(
                prediction[shared_face] / float(max_turn_degrees),
                teacher_face[shared_face] / float(max_turn_degrees),
            )
        return supervised, retention

    initial_loss: float | None = None
    initial_retention_loss: float | None = None
    for _step in range(steps):
        prediction = torch.tanh(heading(features[face_tensor]).float()) * float(
            numeric_heads.max_turn_delta_degrees
        )
        supervised, retention = losses(prediction)
        loss = supervised + float(retention_weight) * retention
        if initial_loss is None:
            initial_loss = float(supervised.detach())
            initial_retention_loss = float(retention.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heading.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        prediction = torch.tanh(heading(features[face_tensor]).float()) * float(
            numeric_heads.max_turn_delta_degrees
        )
        final_supervised, final_retention = losses(prediction)
        final_loss = float(final_supervised)
        errors = torch.abs(prediction[:, 0] - target_degrees[:, 0])
    turn_margins = [
        float(max_turn_degrees) - abs(float(value)) for value in prediction[:, 0]
    ]
    for name, expected in frozen_state.items():
        observed = numeric_heads.state_dict()[name].detach().cpu()
        if not torch.equal(observed, expected):
            raise RuntimeError(f"Heading refit changed frozen numeric tensor {name}")
    return {
        "enabled": True,
        "optimizer": "adamw_absolute_sincos",
        "training_rows_only": True,
        "steps": steps,
        "learning_rate": float(learning_rate),
        "retention_weight": float(retention_weight),
        "retention_shared_face_rows": (
            0 if shared_face is None else int(shared_face.sum())
        ),
        "sample_weighting_enabled": face_weights is not None,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_retention_loss": initial_retention_loss,
        "final_retention_loss": float(final_retention),
        "mean_heading_error_degrees": float(errors.mean()),
        "maximum_heading_error_degrees": float(errors.max()),
        "minimum_turn_margin_degrees": min(turn_margins),
        "action_waypoint_and_shared_norm_unchanged": True,
    }


def _concatenate_outputs(values: Sequence[WaypointPolicyTensors]) -> WaypointPolicyTensors:
    if not values:
        raise ValueError("No waypoint outputs to concatenate")
    return WaypointPolicyTensors(
        action_logits=torch.cat([value.action_logits.detach().cpu() for value in values]),
        waypoint_delta_robot_m=torch.cat(
            [value.waypoint_delta_robot_m.detach().cpu() for value in values]
        ),
        turn_delta_degrees=torch.cat(
            [value.turn_delta_degrees.detach().cpu() for value in values]
        ),
    )


def _controlled_prefix(prefix: torch.Tensor, condition: str) -> torch.Tensor:
    if condition != "zero_scene_prefix":
        return prefix
    # Native BOI/EOI identities are part of the Gemma protocol, not scene
    # content. Preserve them while zeroing all 256 learned scene latents.
    controlled = prefix.clone()
    controlled[:, 1:-1] = 0
    return controlled


@torch.inference_mode()
def evaluate_waypoint_condition(
    runner: ActualGemmaWaypointForward,
    cache: ScenePrefixCache,
    samples: Sequence[WaypointTraceSample],
    *,
    condition: str = "primary",
) -> tuple[dict[str, Any], WaypointPolicyTensors]:
    valid = {"primary", "wrong_scene_prefix", "zero_scene_prefix", "zero_history"}
    if condition not in valid:
        raise ValueError(f"Unknown waypoint evaluation condition: {condition}")
    scenes = sorted({sample.scene_id for sample in samples})
    wrong_scene: dict[str, str] = {}
    if condition == "wrong_scene_prefix":
        if len(scenes) < 2:
            raise ValueError("Wrong-scene control requires at least two scenes")
        wrong_scene = {
            scene: scenes[(index + 1) % len(scenes)] for index, scene in enumerate(scenes)
        }
    values: list[WaypointPolicyTensors] = []
    runner.controller.eval()
    for sample in samples:
        prefix_id = wrong_scene.get(sample.scene_id, sample.scene_id)
        prefix = _controlled_prefix(cache.prefixes[prefix_id], condition)
        history = (
            torch.zeros_like(sample.history) if condition == "zero_history" else sample.history
        )
        values.append(
            runner(
                prefix,
                sample.instruction,
                sample.state.unsqueeze(0),
                history.unsqueeze(0),
            )
        )
    outputs = _concatenate_outputs(values)
    metrics = waypoint_metrics(outputs, samples)
    return {
        **metrics,
        "condition": condition,
        "actual_gemma_causal_forward_per_sample": True,
        "complete_scene_prefix_consumed": True,
        "scene_content_latents_zeroed": condition == "zero_scene_prefix",
        "native_scene_boundaries_preserved": condition == "zero_scene_prefix",
    }, outputs


def _output_change(
    primary: WaypointPolicyTensors, controlled: WaypointPolicyTensors
) -> dict[str, float]:
    primary_actions = primary.action_logits.argmax(dim=-1)
    controlled_actions = controlled.action_logits.argmax(dim=-1)
    return {
        "action_change_fraction": float(
            (primary_actions != controlled_actions).float().mean()
        ),
        "mean_waypoint_output_shift_m": float(
            torch.linalg.vector_norm(
                primary.waypoint_delta_robot_m.float()
                - controlled.waypoint_delta_robot_m.float(),
                dim=-1,
            ).mean()
        ),
        "mean_heading_output_shift_degrees": float(
            torch.abs(
                primary.turn_delta_degrees.float()
                - controlled.turn_delta_degrees.float()
            ).mean()
        ),
    }


def evaluate_waypoint_controls(
    runner: ActualGemmaWaypointForward,
    cache: ScenePrefixCache,
    samples: Sequence[WaypointTraceSample],
    *,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    if sample_limit is not None:
        if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit < 2:
            raise ValueError("Waypoint control sample_limit must be at least two")
        if len(samples) > sample_limit:
            buckets: dict[tuple[str, int], list[WaypointTraceSample]] = {}
            for sample in samples:
                buckets.setdefault((sample.scene_id, sample.action_index), []).append(sample)
            selected: list[WaypointTraceSample] = []
            keys = sorted(buckets)
            while len(selected) < sample_limit and any(buckets.values()):
                for key in keys:
                    if buckets[key] and len(selected) < sample_limit:
                        selected.append(buckets[key].pop(0))
            samples = tuple(selected)
    conditions: dict[str, Any] = {}
    outputs: dict[str, WaypointPolicyTensors] = {}
    for condition in (
        "primary",
        "wrong_scene_prefix",
        "zero_scene_prefix",
        "zero_history",
    ):
        conditions[condition], outputs[condition] = evaluate_waypoint_condition(
            runner, cache, samples, condition=condition
        )
    primary = conditions["primary"]
    return {
        "schema": "semantic_3d_chat.gemma_waypoint_controls.v1",
        "conditions": conditions,
        "accuracy_drop_from_primary": {
            name: primary["action_accuracy"] - value["action_accuracy"]
            for name, value in conditions.items()
            if name != "primary"
        },
        "output_change_from_primary": {
            name: _output_change(outputs["primary"], value)
            for name, value in outputs.items()
            if name != "primary"
        },
        "wrong_scene_is_deranged_within_evaluation_split": True,
        "zero_scene_preserves_only_native_boi_eoi": True,
        "zero_history_is_numeric_all_zeros": True,
        "oracle_inputs_used_by_inference": False,
        "environmental_text_inputs_at_inference": [],
        "configured_sample_limit": sample_limit,
    }


@torch.inference_mode()
def cache_actual_gemma_decision_hidden(
    runner: ActualGemmaWaypointForward,
    cache: ScenePrefixCache,
    samples: Sequence[WaypointTraceSample],
    *,
    forward_batch_size: int = 1,
) -> torch.Tensor:
    """Run every expert row through Gemma and cache its final decision state.

    Rows sharing an instruction and history length may be forwarded together;
    this changes only MPS scheduling. Each row still has its own complete scene
    prefix, numeric state/history, causal stream, and final hidden state.
    """

    if (
        isinstance(forward_batch_size, bool)
        or not isinstance(forward_batch_size, int)
        or not 1 <= forward_batch_size <= 16
    ):
        raise ValueError("forward_batch_size must be an integer in [1,16]")
    runner.controller.eval()
    hidden: list[torch.Tensor | None] = [None] * len(samples)
    groups: dict[tuple[str, int], list[tuple[int, WaypointTraceSample]]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault((sample.instruction, int(sample.history.shape[0])), []).append(
            (index, sample)
        )
    for instruction_and_length in sorted(groups):
        group = groups[instruction_and_length]
        for offset in range(0, len(group), forward_batch_size):
            batch = group[offset : offset + forward_batch_size]
            scene_prefix = torch.cat(
                [cache.prefixes[sample.scene_id] for _index, sample in batch], dim=0
            )
            state = torch.stack([sample.state for _index, sample in batch])
            history = torch.stack([sample.history for _index, sample in batch])
            runner(scene_prefix, instruction_and_length[0], state, history)
            decision_hidden = runner.last_decision_hidden
            if (
                decision_hidden is None
                or tuple(decision_hidden.shape) != (len(batch), runner.hidden_size)
            ):
                raise RuntimeError("Gemma waypoint forward did not expose batched hidden states")
            for row, (index, _sample) in zip(decision_hidden, batch, strict=True):
                hidden[index] = row.detach().float().cpu()
    if any(row is None for row in hidden):
        raise RuntimeError("Gemma waypoint hidden cache left an empty row")
    result = torch.stack([row for row in hidden if row is not None])
    if tuple(result.shape) != (len(samples), runner.hidden_size):
        raise RuntimeError("Cached Gemma waypoint hidden-state shape differs")
    return result


def heads_from_cached_hidden(
    controller: nn.Module, hidden: torch.Tensor
) -> WaypointPolicyTensors:
    output = controller.forward_heads_from_cached_gemma_hidden(hidden)
    return normalize_policy_output(output, batch_size=int(hidden.shape[0]))


def _sample_order_sha256(samples: Sequence[WaypointTraceSample]) -> str:
    """Hash the ordered, exact numeric/text inputs consumed by frozen Gemma."""

    digest = hashlib.sha256()
    for sample in samples:
        for value in (sample.sample_id, sample.scene_id, sample.instruction):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for tensor in (sample.state, sample.history):
            canonical = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
            digest.update(
                json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii")
            )
            digest.update(b"\0")
            digest.update(canonical.numpy().tobytes(order="C"))
            digest.update(b"\0")
    return digest.hexdigest()


def save_gemma_hidden_cache(
    destination: str | Path,
    *,
    train_hidden: torch.Tensor,
    validation_hidden: torch.Tensor,
    train_samples: Sequence[WaypointTraceSample],
    validation_samples: Sequence[WaypointTraceSample],
    dataset_sha256: str,
    gemma_runtime_binding: Mapping[str, Any],
    hidden_input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist training-only final Gemma states without raw instructions."""

    root = _rooted(destination)
    root.mkdir(parents=True, exist_ok=True)
    tensors = {
        "train_hidden": train_hidden.detach().float().cpu().contiguous(),
        "validation_hidden": validation_hidden.detach().float().cpu().contiguous(),
    }
    if any(value.ndim != 2 or not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Gemma waypoint hidden cache tensors are invalid")
    tensor_path = root / "decision_hidden.safetensors"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".decision_hidden.", suffix=".safetensors", dir=root
    )
    os.close(descriptor)
    try:
        save_file(tensors, temporary_name)
        os.replace(temporary_name, tensor_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    binding = validate_gemma_runtime_binding(gemma_runtime_binding)
    input_binding = _validated_hidden_input_binding(hidden_input_binding)
    if input_binding["schema"] == _HIDDEN_INPUT_BINDING_SCHEMA_V2:
        if any(
            sample.history.ndim != 2
            or int(sample.history.shape[1]) != input_binding["history_dim"]
            for sample in (*train_samples, *validation_samples)
        ):
            raise ValueError("Gemma hidden cache sample history contract differs")
        cache_schema = _HIDDEN_CACHE_SCHEMA_V2
        history_contract: dict[str, object] = {
            "history_dim": input_binding["history_dim"],
            "history_parameterization": input_binding["history_parameterization"],
        }
    else:
        cache_schema = _HIDDEN_CACHE_SCHEMA_V1
        history_contract = {}
    metadata = {
        "schema": cache_schema,
        "dataset_sha256": dataset_sha256,
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "train_order_sha256": _sample_order_sha256(train_samples),
        "validation_order_sha256": _sample_order_sha256(validation_samples),
        "hidden_size": int(train_hidden.shape[1]),
        "actual_gemma_causal_forward_per_row": True,
        "output_hidden_states_requested": True,
        "raw_instructions_stored": False,
        "environmental_text_stored": False,
        "tensor_sha256": _sha256_file(tensor_path),
        "gemma_runtime_binding": binding,
        "gemma_runtime_binding_sha256": gemma_runtime_binding_sha256(binding),
        "hidden_input_binding": input_binding,
        "hidden_input_binding_sha256": _canonical_sha256(input_binding),
        **history_contract,
    }
    _atomic_json(root / "metadata.json", metadata)
    return metadata


def load_gemma_hidden_cache(
    source: str | Path,
    *,
    train_samples: Sequence[WaypointTraceSample],
    validation_samples: Sequence[WaypointTraceSample],
    dataset_sha256: str,
    hidden_size: int,
    expected_gemma_runtime_binding: Mapping[str, Any],
    expected_hidden_input_binding: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    root = _rooted(source)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    tensor_path = root / "decision_hidden.safetensors"
    expected_binding = validate_gemma_runtime_binding(expected_gemma_runtime_binding)
    expected_input_binding = _validated_hidden_input_binding(
        expected_hidden_input_binding
    )
    if expected_input_binding["schema"] == _HIDDEN_INPUT_BINDING_SCHEMA_V2:
        expected_cache_schema = _HIDDEN_CACHE_SCHEMA_V2
        expected_history_contract = (
            metadata.get("history_dim") == expected_input_binding["history_dim"]
            and metadata.get("history_parameterization")
            == expected_input_binding["history_parameterization"]
        )
    else:
        expected_cache_schema = _HIDDEN_CACHE_SCHEMA_V1
        expected_history_contract = True
    observed_binding = metadata.get("gemma_runtime_binding")
    if (
        metadata.get("schema") != expected_cache_schema
        or not expected_history_contract
        or metadata.get("dataset_sha256") != dataset_sha256
        or metadata.get("train_order_sha256") != _sample_order_sha256(train_samples)
        or metadata.get("validation_order_sha256")
        != _sample_order_sha256(validation_samples)
        or metadata.get("hidden_size") != hidden_size
        or metadata.get("actual_gemma_causal_forward_per_row") is not True
        or metadata.get("raw_instructions_stored") is not False
        or metadata.get("tensor_sha256") != _sha256_file(tensor_path)
        or observed_binding != expected_binding
        or metadata.get("gemma_runtime_binding_sha256")
        != gemma_runtime_binding_sha256(expected_binding)
        or metadata.get("hidden_input_binding") != expected_input_binding
        or metadata.get("hidden_input_binding_sha256")
        != _canonical_sha256(expected_input_binding)
    ):
        raise ValueError("Gemma waypoint hidden cache contract differs")
    values = load_file(str(tensor_path), device="cpu")
    if set(values) != {"train_hidden", "validation_hidden"}:
        raise ValueError("Gemma waypoint hidden cache tensor keys differ")
    train_hidden = values["train_hidden"].float()
    validation_hidden = values["validation_hidden"].float()
    if tuple(train_hidden.shape) != (len(train_samples), hidden_size) or tuple(
        validation_hidden.shape
    ) != (len(validation_samples), hidden_size):
        raise ValueError("Gemma waypoint hidden cache shapes differ")
    if not torch.isfinite(train_hidden).all() or not torch.isfinite(validation_hidden).all():
        raise ValueError("Gemma waypoint hidden cache is non-finite")
    return train_hidden, validation_hidden, metadata


def load_gemma_hidden_cache_for_forward_revalidation(
    source: str | Path,
    *,
    train_samples: Sequence[WaypointTraceSample],
    validation_samples: Sequence[WaypointTraceSample],
    dataset_sha256: str,
    hidden_size: int,
    expected_gemma_runtime_binding: Mapping[str, Any],
    expected_hidden_input_binding: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load an old cache only when *solely* its forward-source hash changed.

    This is deliberately separate from :func:`load_gemma_hidden_cache`; normal
    training remains strict. The caller must recompute a stratified sample with
    the current Gemma forward, require bit-exact equality, and resave the full
    unchanged tensor under the current binding before training may consume it.
    """

    root = _rooted(source)
    metadata_path = root / "metadata.json"
    tensor_path = root / "decision_hidden.safetensors"
    if (
        not root.is_dir()
        or {entry.name for entry in root.iterdir()}
        != {"metadata.json", "decision_hidden.safetensors"}
        or any(
            path.is_symlink() or not path.is_file()
            for path in (metadata_path, tensor_path)
        )
    ):
        raise ValueError("Gemma waypoint revalidation cache file inventory differs")
    metadata = _unique_json_object(
        metadata_path.read_text(encoding="utf-8"),
        purpose="Gemma waypoint revalidation cache metadata",
    )
    expected_runtime = validate_gemma_runtime_binding(expected_gemma_runtime_binding)
    expected_input = _validated_hidden_input_binding(expected_hidden_input_binding)
    observed_input_value = metadata.get("hidden_input_binding")
    if not isinstance(observed_input_value, Mapping):
        raise TypeError("Gemma waypoint revalidation cache has no input binding")
    observed_input = _validated_hidden_input_binding(observed_input_value)
    observed_without_forward = {
        name: value
        for name, value in observed_input.items()
        if name != "forward_contract_sha256"
    }
    expected_without_forward = {
        name: value
        for name, value in expected_input.items()
        if name != "forward_contract_sha256"
    }
    if observed_input["forward_contract_sha256"] == expected_input[
        "forward_contract_sha256"
    ]:
        raise ValueError("Gemma waypoint cache does not need forward revalidation")
    expected_schema = (
        _HIDDEN_CACHE_SCHEMA_V2
        if expected_input["schema"] == _HIDDEN_INPUT_BINDING_SCHEMA_V2
        else _HIDDEN_CACHE_SCHEMA_V1
    )
    history_matches = (
        expected_input["schema"] != _HIDDEN_INPUT_BINDING_SCHEMA_V2
        or (
            metadata.get("history_dim") == expected_input["history_dim"]
            and metadata.get("history_parameterization")
            == expected_input["history_parameterization"]
        )
    )
    if (
        metadata.get("schema") != expected_schema
        or not history_matches
        or metadata.get("dataset_sha256") != dataset_sha256
        or metadata.get("train_sample_count") != len(train_samples)
        or metadata.get("validation_sample_count") != len(validation_samples)
        or metadata.get("train_order_sha256") != _sample_order_sha256(train_samples)
        or metadata.get("validation_order_sha256")
        != _sample_order_sha256(validation_samples)
        or metadata.get("hidden_size") != hidden_size
        or metadata.get("actual_gemma_causal_forward_per_row") is not True
        or metadata.get("output_hidden_states_requested") is not True
        or metadata.get("raw_instructions_stored") is not False
        or metadata.get("environmental_text_stored") is not False
        or metadata.get("tensor_sha256") != _sha256_file(tensor_path)
        or metadata.get("gemma_runtime_binding") != expected_runtime
        or metadata.get("gemma_runtime_binding_sha256")
        != gemma_runtime_binding_sha256(expected_runtime)
        or metadata.get("hidden_input_binding_sha256")
        != _canonical_sha256(observed_input)
        or observed_without_forward != expected_without_forward
    ):
        raise ValueError(
            "Gemma waypoint cache differs beyond its forward-contract source hash"
        )
    values = load_file(str(tensor_path), device="cpu")
    if set(values) != {"train_hidden", "validation_hidden"}:
        raise ValueError("Gemma waypoint revalidation cache tensor keys differ")
    train_hidden = values["train_hidden"].float()
    validation_hidden = values["validation_hidden"].float()
    if tuple(train_hidden.shape) != (len(train_samples), hidden_size) or tuple(
        validation_hidden.shape
    ) != (len(validation_samples), hidden_size):
        raise ValueError("Gemma waypoint revalidation cache shapes differ")
    if not torch.isfinite(train_hidden).all() or not torch.isfinite(
        validation_hidden
    ).all():
        raise ValueError("Gemma waypoint revalidation cache is non-finite")
    return train_hidden, validation_hidden, metadata


def _controller_state(controller: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in controller.state_dict().items()
    }
    if not state:
        raise ValueError("Waypoint controller has no checkpointable tensors")
    blocked_fragments = ("language_model", "decoder", "vision_tower", "tokenizer")
    for name, tensor in state.items():
        if any(fragment in name.casefold() for fragment in blocked_fragments):
            raise ValueError(f"Waypoint checkpoint tries to include frozen model tensor {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Waypoint checkpoint tensor {name} is non-finite")
    return state


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


def save_waypoint_checkpoint(
    destination: str | Path,
    controller: nn.Module,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    history_projector = getattr(controller, "history_projector", None)
    controller_history_dim = getattr(history_projector, "feature_dim", None)
    history_dim = metadata.get("history_dim", controller_history_dim)
    history_parameterization = metadata.get(
        "history_parameterization", HISTORY_PARAMETERIZATION_V1
    )
    checkpoint_schema = _checkpoint_schema_for_history(
        history_dim, history_parameterization
    )
    if controller_history_dim != history_dim:
        raise ValueError("Waypoint checkpoint controller history dimension differs")
    root = _rooted(destination)
    _guard_runtime_checkpoint_path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = {item.name for item in root.iterdir()}
    if existing - _CHECKPOINT_FILES:
        raise ValueError("Waypoint checkpoint directory contains unexpected files")
    state = _controller_state(controller)
    weights = root / "policy.safetensors"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".policy.", suffix=".safetensors", dir=root
    )
    os.close(descriptor)
    try:
        save_file(state, temporary_name)
        os.replace(temporary_name, weights)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    sanitized = {
        "action_names": list(ACTION_NAMES),
        "weights_sha256": _sha256_file(weights),
        "saved_controller_tensors_only": True,
        "frozen_gemma_weights_saved": False,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "runtime_required_files": ["policy.safetensors", "runtime_metadata.json"],
        **dict(metadata),
        # These fields bind the saved numeric-history tensors to the exact
        # canonicalization used by the live controller. Callers cannot
        # accidentally override them with an older checkpoint contract.
        "schema": checkpoint_schema,
        "history_dim": history_dim,
        "history_parameterization": history_parameterization,
    }
    serialized = json.dumps(sanitized, sort_keys=True, allow_nan=False).casefold()
    for forbidden in ('"instruction"', '"scene_ids"', '"object_labels"', '"oracle_path"'):
        if forbidden in serialized:
            raise ValueError(f"Waypoint checkpoint metadata contains forbidden field {forbidden}")
    _atomic_json(root / "runtime_metadata.json", sanitized)
    return sanitized


def load_waypoint_checkpoint(
    source: str | Path,
    controller: nn.Module,
) -> dict[str, Any]:
    root = _rooted(source)
    _guard_runtime_checkpoint_path(root)
    if not root.is_dir() or {item.name for item in root.iterdir()} != _CHECKPOINT_FILES:
        raise ValueError("Waypoint checkpoint must contain exactly two sanitized files")
    metadata = json.loads((root / "runtime_metadata.json").read_text(encoding="utf-8"))
    weights = root / "policy.safetensors"
    if metadata.get("weights_sha256") != _sha256_file(weights):
        raise ValueError("Waypoint checkpoint weights changed")
    try:
        expected_schema = _checkpoint_schema_for_history(
            metadata.get("history_dim"), metadata.get("history_parameterization")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Waypoint checkpoint runtime contract differs") from exc
    history_projector = getattr(controller, "history_projector", None)
    if (
        metadata.get("schema") != expected_schema
        or getattr(history_projector, "feature_dim", None)
        != metadata.get("history_dim")
        or metadata.get("frozen_gemma_weights_saved") is not False
    ):
        raise ValueError("Waypoint checkpoint runtime contract differs")
    state = load_file(str(weights), device="cpu")
    controller.load_state_dict(state, strict=True)
    return metadata


def _same_retention_sample(
    current: WaypointTraceSample, reference: WaypointTraceSample
) -> bool:
    """Require an exact label, metadata, and continuous-input row match."""

    return (
        current.scene_id == reference.scene_id
        and current.split == reference.split
        and current.instruction == reference.instruction
        and current.action_index == reference.action_index
        and current.heading_degrees == reference.heading_degrees
        and torch.equal(current.state, reference.state)
        and torch.equal(current.history, reference.history)
        and torch.equal(
            current.waypoint_delta_robot_m, reference.waypoint_delta_robot_m
        )
    )


def _retention_sample_fingerprint(sample: WaypointTraceSample) -> str:
    """Hash every retained input/target field except the unstable row ID."""

    digest = hashlib.sha256()

    def frame(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        digest.update(value)

    frame(b"semantic_3d_chat.waypoint_retention_row.v1")
    frame(
        json.dumps(
            {
                "scene_id": sample.scene_id,
                "split": sample.split,
                "instruction": sample.instruction,
                "action_index": sample.action_index,
                "heading_degrees": sample.heading_degrees,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )
    for name, tensor in (
        ("state", sample.state),
        ("history", sample.history),
        ("waypoint_delta_robot_m", sample.waypoint_delta_robot_m),
    ):
        canonical = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        frame(name.encode("ascii"))
        frame(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"))
        frame(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _retention_fingerprint_buckets(
    samples: Sequence[WaypointTraceSample], *, source: str
) -> dict[str, list[WaypointTraceSample]]:
    """Build exact-row buckets while failing closed on a hash collision."""

    buckets: dict[str, list[WaypointTraceSample]] = {}
    for sample in samples:
        fingerprint = _retention_sample_fingerprint(sample)
        bucket = buckets.setdefault(fingerprint, [])
        if bucket and not _same_retention_sample(sample, bucket[0]):
            raise ValueError(f"{source} retention row fingerprint collision")
        bucket.append(sample)
    return buckets


@torch.inference_mode()
def _cached_outputs_in_batches(
    controller: nn.Module,
    hidden: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> WaypointPolicyTensors:
    if hidden.ndim != 2 or hidden.shape[0] < 1 or not torch.isfinite(hidden).all():
        raise ValueError("retention hidden states are invalid")
    values: list[WaypointPolicyTensors] = []
    controller.eval()
    for offset in range(0, int(hidden.shape[0]), batch_size):
        values.append(
            heads_from_cached_hidden(
                controller, hidden[offset : offset + batch_size].to(device)
            )
        )
    return _concatenate_outputs(values)


def load_waypoint_retention_reference(
    config: Mapping[str, Any],
    controller: nn.Module,
    dataset: WaypointTraceDataset,
    train_samples: Sequence[WaypointTraceSample],
    train_hidden: torch.Tensor,
    *,
    settings: Mapping[str, Any],
    gemma_runtime_binding: Mapping[str, Any],
    device: torch.device,
) -> WaypointRetentionReference | None:
    """Warm-start from and distill an authenticated prior waypoint policy.

    The prior trace dataset authenticates shared rows by exact input/target
    fingerprints rather than unstable generated sample IDs. Every old training
    row occurrence must still exist with byte-identical numeric inputs and the
    same target. The prior checkpoint is loaded through the normal fail-closed
    runtime loader, and its frozen context tensors must match the controller
    used to authenticate the current hidden cache.
    """

    checkpoint_path = settings.get("retention_reference_checkpoint")
    trace_path = settings.get("retention_reference_trace_dataset")
    if checkpoint_path is None and trace_path is None:
        return None
    if not isinstance(checkpoint_path, str) or not isinstance(trace_path, str):
        raise TypeError("waypoint retention requires checkpoint and trace dataset")

    reference_dataset = load_waypoint_trace_jsonl(
        trace_path,
        state_dim=int(settings["state_dim"]),
        history_dim=int(settings["history_dim"]),
        history_parameterization=str(settings["history_parameterization"]),
        max_history_tokens=int(settings["max_history_tokens"]),
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
    )
    reference_train = reference_dataset.split("train")
    context_before = {
        name: value.detach().cpu().clone()
        for name, value in controller.state_dict().items()
        if not name.startswith("numeric_heads.")
    }
    metadata = load_waypoint_checkpoint(checkpoint_path, controller)
    for name, expected in context_before.items():
        if not torch.equal(controller.state_dict()[name].detach().cpu(), expected):
            raise ValueError("retention checkpoint context differs from hidden cache")

    expected_contract = {
        "model_id": str(config["language"]["model_id"]),
        "model_revision": str(config["language"]["revision"]),
        "gemma_runtime_binding": dict(gemma_runtime_binding),
        "gemma_runtime_binding_sha256": gemma_runtime_binding_sha256(
            gemma_runtime_binding
        ),
        "dataset_sha256": reference_dataset.sha256,
        "training_traces_sha256": reference_dataset.traces_sha256,
        "training_sample_count": len(reference_train),
        "scene_token_count": int(settings["scene_token_count"]),
        "robot_token_count": int(settings["robot_token_count"]),
        "hidden_size": int(settings["hidden_size"]),
        "state_dim": int(settings["state_dim"]),
        "history_dim": int(settings["history_dim"]),
        "history_parameterization": str(settings["history_parameterization"]),
        "max_history_tokens": int(settings["max_history_tokens"]),
        "context_token_count": int(settings["context_token_count"]),
        "head_hidden_dim": int(settings["head_hidden_dim"]),
        "max_waypoint_step_m": float(settings["max_waypoint_step_m"]),
        "max_turn_delta_degrees": float(settings["max_turn_delta_degrees"]),
        "heading_parameterization": "robot_relative_bounded_scalar_tanh",
    }
    if any(metadata.get(name) != value for name, value in expected_contract.items()):
        raise ValueError("retention checkpoint model, dataset, or prefix contract differs")

    reference_buckets = _retention_fingerprint_buckets(
        reference_train, source="reference"
    )
    current_buckets = _retention_fingerprint_buckets(train_samples, source="current")
    consumed: dict[str, int] = {}
    shared_values: list[bool] = []
    for sample in train_samples:
        fingerprint = _retention_sample_fingerprint(sample)
        reference_bucket = reference_buckets.get(fingerprint, [])
        offset = consumed.get(fingerprint, 0)
        if offset >= len(reference_bucket):
            shared_values.append(False)
            continue
        reference = reference_bucket[offset]
        if not _same_retention_sample(sample, reference):
            raise ValueError("retention row fingerprint collision")
        consumed[fingerprint] = offset + 1
        shared_values.append(True)
    missing_occurrences = sum(
        len(bucket) - consumed.get(fingerprint, 0)
        for fingerprint, bucket in reference_buckets.items()
    )
    if missing_occurrences:
        raise ValueError("current training data removed or changed reference retention rows")
    shared_mask = torch.tensor(shared_values, dtype=torch.bool)
    if int(shared_mask.sum()) != len(reference_train):
        raise ValueError("retention shared-row authentication differs")
    if sum(len(bucket) for bucket in current_buckets.values()) != len(train_samples):
        raise RuntimeError("current retention row occurrence accounting differs")

    teacher_outputs = _cached_outputs_in_batches(
        controller,
        train_hidden,
        device=device,
        batch_size=int(settings["head_batch_size"]),
    )
    numeric_heads = getattr(controller, "numeric_heads", None)
    reference_action = getattr(numeric_heads, "action", None)
    if not isinstance(reference_action, nn.Linear):
        raise TypeError("retention checkpoint has no standard linear action head")
    new_sample_weight = float(settings["retention_new_sample_weight"])
    sample_weights = torch.where(
        shared_mask,
        torch.ones(len(train_samples), dtype=torch.float32),
        torch.full((len(train_samples),), new_sample_weight, dtype=torch.float32),
    )
    report = {
        "enabled": True,
        "training_only": True,
        "runtime_architecture_changed": False,
        "reference_checkpoint_weights_sha256": metadata["weights_sha256"],
        "reference_dataset_sha256": reference_dataset.sha256,
        "reference_training_traces_sha256": reference_dataset.traces_sha256,
        "current_dataset_sha256": dataset.sha256,
        "shared_training_rows": int(shared_mask.sum()),
        "new_training_rows": int((~shared_mask).sum()),
        "all_reference_rows_preserved_exactly": True,
        "all_reference_row_occurrences_consumed_once": True,
        "identity_uses_generated_sample_id": False,
        "identity_fingerprint": "sha256_exact_inputs_targets_metadata_v1",
        "one_to_one_reference_row_occurrence_matching": True,
        "duplicate_exact_rows_authenticated_by_multiplicity": True,
        "reference_unique_fingerprint_buckets": len(reference_buckets),
        "reference_duplicate_row_occurrences": len(reference_train)
        - len(reference_buckets),
        "current_unique_fingerprint_buckets": len(current_buckets),
        "current_duplicate_row_occurrences": len(train_samples)
        - len(current_buckets),
        "warm_started_from_reference": True,
        "new_sample_weight": new_sample_weight,
    }
    return WaypointRetentionReference(
        outputs=teacher_outputs,
        shared_mask=shared_mask,
        sample_weights=sample_weights,
        action_weight=reference_action.weight.detach().float().cpu().clone(),
        action_bias=reference_action.bias.detach().float().cpu().clone(),
        metadata=report,
    )


def validate_waypoint_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("gemma_waypoint_policy")
    if not isinstance(value, dict):
        raise TypeError("Config has no gemma_waypoint_policy mapping")
    settings = dict(value)
    for name in (
        "scene_token_count",
        "robot_token_count",
        "hidden_size",
        "state_dim",
        "history_dim",
        "max_history_tokens",
        "head_hidden_dim",
        "context_token_count",
        "epochs",
        "gradient_accumulation_steps",
        "head_batch_size",
        "seed",
    ):
        number = settings.get(name)
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError(f"gemma_waypoint_policy.{name} must be a positive integer")
    history_parameterization = settings.get(
        "history_parameterization", HISTORY_PARAMETERIZATION_V1
    )
    _checkpoint_schema_for_history(
        settings["history_dim"], history_parameterization
    )
    settings["history_parameterization"] = history_parameterization
    action_refit_max_iter = settings.get("action_refit_max_iter")
    if (
        isinstance(action_refit_max_iter, bool)
        or not isinstance(action_refit_max_iter, int)
        or action_refit_max_iter < 0
    ):
        raise ValueError(
            "gemma_waypoint_policy.action_refit_max_iter must be a nonnegative integer"
        )
    # Existing experiment configs predate the regularized refit.  Zero is an
    # explicit compatibility default and follows the original LBFGS objective.
    settings.setdefault("action_refit_l2_weight", 0.0)
    settings.setdefault("action_constrained_refit_margin", 0.001)
    settings.setdefault("action_constrained_refit_covariance_ridge", 1e-9)
    settings.setdefault("action_constrained_refit_feasibility_tolerance", 1e-7)
    settings.setdefault("action_constrained_refit_shared_cut_iterations", 16)
    # The branch-isolated refit is opt-in so historical experiments retain
    # their exact joint-training behavior. Authenticated retention experiments
    # can enable it to repair numeric MOVE_TO targets without changing action,
    # heading, normalization, or projection parameters.
    settings.setdefault("waypoint_branch_refit_enabled", False)
    settings.setdefault("waypoint_branch_refit_steps", 300)
    settings.setdefault("waypoint_branch_refit_learning_rate", 1e-4)
    settings.setdefault("waypoint_branch_refit_weight_decay", 1e-4)
    settings.setdefault("waypoint_branch_refit_gradient_clip_norm", 1.0)
    settings.setdefault("waypoint_branch_refit_new_error_tolerance_m", 0.025)
    settings.setdefault(
        "waypoint_branch_refit_minimum_new_within_tolerance_fraction", 1.0
    )
    # An authenticated warm-start may skip the generic joint AdamW phase and
    # proceed directly through independently gated branch repairs. ``None``
    # preserves the historical ``epochs`` behavior for existing experiments.
    settings.setdefault("retention_joint_training_epochs", None)
    # Retention is opt-in. With no authenticated reference these defaults
    # reproduce the legacy fresh-head training objective exactly.
    settings.setdefault("retention_reference_checkpoint", None)
    settings.setdefault("retention_reference_trace_dataset", None)
    settings.setdefault("retention_logit_weight", 0.0)
    settings.setdefault("retention_waypoint_weight", 0.0)
    settings.setdefault("retention_heading_weight", 0.0)
    settings.setdefault("retention_new_sample_weight", 1.0)
    settings.setdefault("retention_freeze_input_norm", False)
    settings.setdefault("retention_minimum_shared_action_agreement", 0.0)
    settings.setdefault("retention_minimum_new_action_accuracy", 0.0)
    settings.setdefault("retention_maximum_shared_centered_logit_rmse", None)
    settings.setdefault("retention_maximum_shared_waypoint_drift_m", None)
    settings.setdefault("retention_maximum_shared_heading_drift_degrees", None)
    reference_values = (
        settings["retention_reference_checkpoint"],
        settings["retention_reference_trace_dataset"],
    )
    if any(value is not None for value in reference_values) and not all(
        isinstance(value, str) and bool(value.strip()) for value in reference_values
    ):
        raise ValueError(
            "retention_reference_checkpoint and retention_reference_trace_dataset "
            "must be configured together"
        )
    if not isinstance(settings["retention_freeze_input_norm"], bool):
        raise TypeError("gemma_waypoint_policy.retention_freeze_input_norm must be bool")
    if not isinstance(settings["waypoint_branch_refit_enabled"], bool):
        raise TypeError(
            "gemma_waypoint_policy.waypoint_branch_refit_enabled must be bool"
        )
    retention_joint_epochs = settings["retention_joint_training_epochs"]
    if retention_joint_epochs is not None and (
        isinstance(retention_joint_epochs, bool)
        or not isinstance(retention_joint_epochs, int)
        or retention_joint_epochs < 0
    ):
        raise ValueError(
            "gemma_waypoint_policy.retention_joint_training_epochs must be null "
            "or nonnegative"
        )
    shared_cut_iterations = settings["action_constrained_refit_shared_cut_iterations"]
    if (
        isinstance(shared_cut_iterations, bool)
        or not isinstance(shared_cut_iterations, int)
        or shared_cut_iterations < 1
    ):
        raise ValueError(
            "gemma_waypoint_policy.action_constrained_refit_shared_cut_iterations "
            "must be positive"
        )
    waypoint_refit_steps = settings["waypoint_branch_refit_steps"]
    if (
        isinstance(waypoint_refit_steps, bool)
        or not isinstance(waypoint_refit_steps, int)
        or waypoint_refit_steps < 1
    ):
        raise ValueError(
            "gemma_waypoint_policy.waypoint_branch_refit_steps must be positive"
        )
    heading_refit_steps = settings.get("heading_refit_steps")
    if (
        isinstance(heading_refit_steps, bool)
        or not isinstance(heading_refit_steps, int)
        or heading_refit_steps < 0
    ):
        raise ValueError(
            "gemma_waypoint_policy.heading_refit_steps must be a nonnegative integer"
        )
    for name in (
        "learning_rate",
        "weight_decay",
        "waypoint_loss_weight",
        "heading_loss_weight",
        "action_class_weight_power",
        "gradient_clip_norm",
        "max_waypoint_step_m",
        "max_turn_delta_degrees",
        "action_refit_learning_rate",
        "action_refit_l2_weight",
        "action_constrained_refit_margin",
        "action_constrained_refit_covariance_ridge",
        "action_constrained_refit_feasibility_tolerance",
        "waypoint_branch_refit_learning_rate",
        "waypoint_branch_refit_weight_decay",
        "waypoint_branch_refit_gradient_clip_norm",
        "waypoint_branch_refit_new_error_tolerance_m",
        "waypoint_branch_refit_minimum_new_within_tolerance_fraction",
        "retention_logit_weight",
        "retention_waypoint_weight",
        "retention_heading_weight",
        "retention_new_sample_weight",
        "retention_minimum_shared_action_agreement",
        "retention_minimum_new_action_accuracy",
        "minimum_training_action_accuracy",
        "heading_refit_learning_rate",
        "minimum_training_turn_margin_degrees",
    ):
        number = settings.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise TypeError(f"gemma_waypoint_policy.{name} must be numeric")
        if not math.isfinite(float(number)) or float(number) < 0.0:
            raise ValueError(f"gemma_waypoint_policy.{name} must be finite and nonnegative")
    if float(settings["action_class_weight_power"]) > 2.0:
        raise ValueError("gemma_waypoint_policy.action_class_weight_power must be <= 2")
    if float(settings["retention_new_sample_weight"]) <= 0.0:
        raise ValueError(
            "gemma_waypoint_policy.retention_new_sample_weight must be positive"
        )
    for name in (
        "action_constrained_refit_margin",
        "action_constrained_refit_covariance_ridge",
        "action_constrained_refit_feasibility_tolerance",
        "waypoint_branch_refit_learning_rate",
        "waypoint_branch_refit_gradient_clip_norm",
        "waypoint_branch_refit_new_error_tolerance_m",
    ):
        if float(settings[name]) <= 0.0:
            raise ValueError(f"gemma_waypoint_policy.{name} must be positive")
    for name in (
        "retention_minimum_shared_action_agreement",
        "retention_minimum_new_action_accuracy",
        "waypoint_branch_refit_minimum_new_within_tolerance_fraction",
    ):
        if not 0.0 <= float(settings[name]) <= 1.0:
            raise ValueError(f"gemma_waypoint_policy.{name} must be in [0,1]")
    for name in (
        "retention_maximum_shared_centered_logit_rmse",
        "retention_maximum_shared_waypoint_drift_m",
        "retention_maximum_shared_heading_drift_degrees",
    ):
        value = settings[name]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"gemma_waypoint_policy.{name} must be null or nonnegative")
    retention_active = (
        any(float(settings[name]) > 0.0 for name in (
            "retention_logit_weight",
            "retention_waypoint_weight",
            "retention_heading_weight",
        ))
        or float(settings["retention_new_sample_weight"]) != 1.0
        or settings["retention_freeze_input_norm"] is True
        or float(settings["retention_minimum_shared_action_agreement"]) > 0.0
        or float(settings["retention_minimum_new_action_accuracy"]) > 0.0
        or any(
            settings[name] is not None
            for name in (
                "retention_maximum_shared_centered_logit_rmse",
                "retention_maximum_shared_waypoint_drift_m",
                "retention_maximum_shared_heading_drift_degrees",
            )
        )
    )
    if retention_active and reference_values[0] is None:
        raise ValueError("waypoint retention weights require an authenticated reference")
    if retention_joint_epochs is not None and reference_values[0] is None:
        raise ValueError(
            "retention_joint_training_epochs requires an authenticated reference"
        )
    if settings["waypoint_branch_refit_enabled"]:
        if reference_values[0] is None:
            raise ValueError(
                "waypoint_branch_refit_enabled requires an authenticated reference"
            )
        if settings["retention_maximum_shared_waypoint_drift_m"] is None:
            raise ValueError(
                "waypoint_branch_refit_enabled requires "
                "retention_maximum_shared_waypoint_drift_m"
            )
    if not 0.0 < float(settings["max_turn_delta_degrees"]) < 180.0:
        raise ValueError(
            "gemma_waypoint_policy.max_turn_delta_degrees must be in (0,180)"
        )
    if action_refit_max_iter > 0 and float(settings["action_refit_learning_rate"]) <= 0.0:
        raise ValueError(
            "gemma_waypoint_policy.action_refit_learning_rate must be positive when refit is enabled"
        )
    if heading_refit_steps > 0 and float(settings["heading_refit_learning_rate"]) <= 0.0:
        raise ValueError(
            "gemma_waypoint_policy.heading_refit_learning_rate must be positive when refit is enabled"
        )
    if not 0.0 <= float(settings["minimum_training_action_accuracy"]) <= 1.0:
        raise ValueError(
            "gemma_waypoint_policy.minimum_training_action_accuracy must be in [0,1]"
        )
    if settings.get("checkpoint_selection") not in {
        "heldout_validation",
        "final_training_epoch",
    }:
        raise ValueError(
            "gemma_waypoint_policy.checkpoint_selection must be "
            "heldout_validation or final_training_epoch"
        )
    if settings.get("batch_size") != 1:
        raise ValueError("Actual Gemma waypoint training currently requires batch_size: 1")
    control_limit = settings.get("control_sample_limit")
    if control_limit is not None and (
        isinstance(control_limit, bool)
        or not isinstance(control_limit, int)
        or control_limit < 2
    ):
        raise ValueError("gemma_waypoint_policy.control_sample_limit must be at least two")
    for name in ("training_sample_limit", "validation_sample_limit"):
        sample_limit = settings.get(name)
        if sample_limit is not None and (
            isinstance(sample_limit, bool)
            or not isinstance(sample_limit, int)
            or sample_limit < len(ACTION_NAMES)
        ):
            raise ValueError(f"gemma_waypoint_policy.{name} must cover every action")
    return settings


def train_waypoint_controller(
    config: Mapping[str, Any],
    language: LocalLanguageModel,
    controller: nn.Module,
    robot_state_encoder: nn.Module,
    dataset: WaypointTraceDataset,
    cache: ScenePrefixCache,
    *,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end train the numeric projector/heads through frozen Gemma."""

    settings = validate_waypoint_settings(config)
    if (
        dataset.history_dim != int(settings["history_dim"])
        or dataset.history_parameterization
        != str(settings["history_parameterization"])
    ):
        raise ValueError("Waypoint dataset history contract differs from config")
    seed = int(settings["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    train = select_balanced_waypoint_samples(
        dataset.split("train"), settings.get("training_sample_limit")
    )
    validation = select_balanced_waypoint_samples(
        dataset.split("validation"), settings.get("validation_sample_limit")
    )
    required_scenes = {sample.scene_id for sample in (*train, *validation)}
    if not required_scenes <= set(cache.prefixes):
        raise FileNotFoundError("Waypoint cache does not contain every training scene")
    runner = ActualGemmaWaypointForward(
        language,
        controller,
        robot_state_encoder,
        scene_token_count=int(settings["scene_token_count"]),
        robot_token_count=int(settings["robot_token_count"]),
        hidden_size=int(settings["hidden_size"]),
        state_dim=int(settings["state_dim"]),
        history_dim=int(settings["history_dim"]),
    )
    controller.to(language.device)
    robot_state_encoder.to(language.device)
    robot_state_encoder.requires_grad_(False)
    robot_state_encoder.eval()
    language.model.requires_grad_(False)
    language.model.eval()
    gemma_binding = language_gemma_runtime_binding(language)
    if getattr(controller, "context_projection_frozen", None) is not True:
        raise ValueError(
            "Cached Gemma training requires a frozen history projector and decision token"
        )
    numeric_heads = getattr(controller, "numeric_heads", None)
    if not isinstance(numeric_heads, nn.Module):
        raise TypeError("Waypoint controller exposes no numeric_heads module")
    hidden_input_binding = gemma_hidden_input_binding(
        language,
        controller,
        robot_state_encoder,
        cache,
        (*train, *validation),
        history_parameterization=str(settings["history_parameterization"]),
    )
    hidden_cache_path = settings.get("hidden_cache")
    if hidden_cache_path is None:
        train_hidden = cache_actual_gemma_decision_hidden(runner, cache, train)
        validation_hidden = cache_actual_gemma_decision_hidden(runner, cache, validation)
        hidden_cache_metadata = None
    else:
        train_hidden, validation_hidden, hidden_cache_metadata = load_gemma_hidden_cache(
            str(hidden_cache_path),
            train_samples=train,
            validation_samples=validation,
            dataset_sha256=dataset.sha256,
            hidden_size=int(settings["hidden_size"]),
            expected_gemma_runtime_binding=gemma_binding,
            expected_hidden_input_binding=hidden_input_binding,
        )
    retention = load_waypoint_retention_reference(
        config,
        controller,
        dataset,
        train,
        train_hidden,
        settings=settings,
        gemma_runtime_binding=gemma_binding,
        device=language.device,
    )
    if retention is not None and settings["retention_freeze_input_norm"]:
        input_norm = getattr(numeric_heads, "input_norm", None)
        if not isinstance(input_norm, nn.LayerNorm):
            raise TypeError("Waypoint retention requires the standard input normalization")
        input_norm.requires_grad_(False)
    trainable = [
        parameter for parameter in numeric_heads.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Waypoint controller has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    action_counts = torch.bincount(
        torch.tensor([sample.action_index for sample in train]), minlength=len(ACTION_NAMES)
    ).float()
    if torch.any(action_counts == 0):
        raise ValueError("Waypoint training split must cover MOVE_TO, FACE, and STOP")
    class_weights = action_counts.pow(-float(settings["action_class_weight_power"]))
    class_weights /= class_weights.mean()
    max_waypoint_step_m = float(settings["max_waypoint_step_m"])
    accumulation = int(settings["gradient_accumulation_steps"])
    head_batch_size = int(settings["head_batch_size"])
    generator = random.Random(seed)
    configured_joint_epochs = settings["retention_joint_training_epochs"]
    joint_training_epochs = (
        int(settings["epochs"])
        if retention is None or configured_joint_epochs is None
        else int(configured_joint_epochs)
    )
    best_state: dict[str, torch.Tensor] | None = (
        _controller_state(controller) if joint_training_epochs == 0 else None
    )
    best_score = -math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(1, joint_training_epochs + 1):
        controller.train()
        indices = list(range(len(train)))
        generator.shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        retention_loss_sum = 0.0
        offsets = range(0, len(indices), head_batch_size)
        for update_count, offset in enumerate(offsets, start=1):
            batch_indices = indices[offset : offset + head_batch_size]
            batch_samples = tuple(train[index] for index in batch_indices)
            output = heads_from_cached_hidden(
                controller,
                train_hidden[batch_indices].to(language.device),
            )
            loss, _parts = waypoint_loss(
                output,
                batch_samples,
                max_waypoint_step_m=max_waypoint_step_m,
                max_turn_delta_degrees=float(settings["max_turn_delta_degrees"]),
                action_weights=class_weights,
                sample_weights=(
                    None
                    if retention is None
                    else retention.sample_weights[batch_indices]
                ),
                waypoint_weight=float(settings["waypoint_loss_weight"]),
                heading_weight=float(settings["heading_loss_weight"]),
            )
            if retention is not None:
                reference_batch = WaypointPolicyTensors(
                    action_logits=retention.outputs.action_logits[batch_indices],
                    waypoint_delta_robot_m=(
                        retention.outputs.waypoint_delta_robot_m[batch_indices]
                    ),
                    turn_delta_degrees=(
                        retention.outputs.turn_delta_degrees[batch_indices]
                    ),
                )
                retained, retention_parts = waypoint_retention_loss(
                    output,
                    reference_batch,
                    batch_samples,
                    retention.shared_mask[batch_indices],
                    max_waypoint_step_m=max_waypoint_step_m,
                    max_turn_delta_degrees=float(settings["max_turn_delta_degrees"]),
                    logit_weight=float(settings["retention_logit_weight"]),
                    waypoint_weight=float(settings["retention_waypoint_weight"]),
                    heading_weight=float(settings["retention_heading_weight"]),
                )
                loss = loss + retained
                retention_loss_sum += retention_parts["total"] * len(batch_indices)
            (loss / accumulation).backward()
            loss_sum += float(loss.detach()) * len(batch_indices)
            if update_count % accumulation == 0 or offset + head_batch_size >= len(indices):
                torch.nn.utils.clip_grad_norm_(
                    trainable, float(settings["gradient_clip_norm"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        controller.eval()
        with torch.inference_mode():
            validation_outputs = heads_from_cached_hidden(
                controller, validation_hidden.to(language.device)
            )
        validation_metrics = waypoint_metrics(
            validation_outputs, validation
        )
        waypoint_error = validation_metrics["waypoint_error_m_mean"]
        heading_error = validation_metrics["heading_error_degrees_mean"]
        score = (
            0.5 * float(validation_metrics["action_accuracy"])
            + 0.5 * float(validation_metrics["action_macro_recall"])
            + 0.25 * float(validation_metrics["stop_recall"])
            - 0.10 * (0.0 if waypoint_error is None else float(waypoint_error))
            - 0.001 * (0.0 if heading_error is None else float(heading_error))
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(train),
                "retention_loss": retention_loss_sum / len(train),
                "validation": validation_metrics,
                "selector_score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = _controller_state(controller)
    if settings["checkpoint_selection"] == "final_training_epoch":
        # The operator profile is an explicit live-room specialization.  Its
        # scene-disjoint validation scores remain reported, but must not select
        # an early epoch that sacrifices the demonstrated room's numeric
        # waypoint/heading fit.  The general multi-scene profile continues to
        # use held-out validation selection.
        best_state = _controller_state(controller)
        best_epoch = joint_training_epochs
    if best_state is None:
        raise RuntimeError("Waypoint training produced no checkpoint candidate")
    controller.load_state_dict(best_state, strict=True)
    action_refit_max_iter = int(settings["action_refit_max_iter"])
    if action_refit_max_iter:
        if retention is None:
            action_refit = refit_waypoint_action_classifier(
                controller,
                train_hidden,
                train,
                max_iter=action_refit_max_iter,
                learning_rate=float(settings["action_refit_learning_rate"]),
                l2_weight=float(settings["action_refit_l2_weight"]),
                action_weights=class_weights,
            )
        else:
            action_refit = refit_waypoint_action_classifier_constrained(
                controller,
                train_hidden,
                train,
                reference_logits=retention.outputs.action_logits,
                reference_action_weight=retention.action_weight,
                reference_action_bias=retention.action_bias,
                retention_mask=retention.shared_mask,
                positive_margin=float(settings["action_constrained_refit_margin"]),
                covariance_ridge=float(
                    settings["action_constrained_refit_covariance_ridge"]
                ),
                feasibility_tolerance=float(
                    settings["action_constrained_refit_feasibility_tolerance"]
                ),
                maximum_centered_logit_rmse=settings[
                    "retention_maximum_shared_centered_logit_rmse"
                ],
                max_active_set_iterations=action_refit_max_iter,
                max_shared_cut_iterations=int(
                    settings["action_constrained_refit_shared_cut_iterations"]
                ),
            )
    else:
        action_refit = {
            "enabled": False,
            "training_rows_only": True,
            "l2_weight": float(settings["action_refit_l2_weight"]),
            "shared_norm_and_numeric_branches_unchanged": True,
        }
    waypoint_branch_refit_enabled = bool(settings["waypoint_branch_refit_enabled"])
    if waypoint_branch_refit_enabled:
        if retention is None:
            raise RuntimeError(
                "Waypoint branch refit requires the authenticated retention reference"
            )
        waypoint_refit = refit_waypoint_branch(
            controller,
            train_hidden,
            train,
            steps=int(settings["waypoint_branch_refit_steps"]),
            learning_rate=float(settings["waypoint_branch_refit_learning_rate"]),
            weight_decay=float(settings["waypoint_branch_refit_weight_decay"]),
            gradient_clip_norm=float(
                settings["waypoint_branch_refit_gradient_clip_norm"]
            ),
            sample_weights=retention.sample_weights,
            reference_waypoints=retention.outputs.waypoint_delta_robot_m,
            retention_mask=retention.shared_mask,
            retention_weight=float(settings["retention_waypoint_weight"]),
            error_tolerance_m=float(
                settings["waypoint_branch_refit_new_error_tolerance_m"]
            ),
            # CPU full-batch AdamW is deterministic for this small isolated
            # branch and avoids backend-dependent MPS reduction drift.
            device="cpu",
        )
        observed_new_fraction = waypoint_refit[
            "new_move_within_tolerance_fraction"
        ]
        required_new_fraction = float(
            settings[
                "waypoint_branch_refit_minimum_new_within_tolerance_fraction"
            ]
        )
        observed_shared_drift = waypoint_refit["shared_waypoint_drift_m"]["max"]
        required_shared_drift = float(
            settings["retention_maximum_shared_waypoint_drift_m"]
        )
        waypoint_refit_gates = {
            "new_move_within_tolerance_fraction": {
                "observed": observed_new_fraction,
                "comparison": ">=",
                "required": required_new_fraction,
                "passed": (
                    isinstance(observed_new_fraction, (int, float))
                    and float(observed_new_fraction) >= required_new_fraction
                ),
            },
            "shared_waypoint_drift_m_max": {
                "observed": observed_shared_drift,
                "comparison": "<=",
                "required": required_shared_drift,
                "passed": (
                    isinstance(observed_shared_drift, (int, float))
                    and float(observed_shared_drift) <= required_shared_drift
                ),
            },
        }
        waypoint_refit_gates["passed"] = all(
            bool(gate["passed"])
            for gate in waypoint_refit_gates.values()
            if isinstance(gate, Mapping)
        )
        waypoint_refit["gates"] = waypoint_refit_gates
        if not waypoint_refit_gates["passed"]:
            failures = {
                name: gate
                for name, gate in waypoint_refit_gates.items()
                if isinstance(gate, Mapping) and not gate["passed"]
            }
            raise RuntimeError(
                "Waypoint branch refit failed before checkpoint mutation: "
                + json.dumps(failures, sort_keys=True, separators=(",", ":"))
            )
    else:
        waypoint_refit = {
            "enabled": False,
            "training_rows_only": True,
            "waypoint_branch_only": True,
        }
    controller.to(language.device)
    heading_refit_steps = int(settings["heading_refit_steps"])
    if heading_refit_steps:
        robot_settings = config.get("robot")
        if not isinstance(robot_settings, Mapping):
            raise TypeError("Waypoint heading refit requires robot runtime bounds")
        heading_refit = refit_waypoint_heading_head(
            controller,
            train_hidden,
            train,
            steps=heading_refit_steps,
            learning_rate=float(settings["heading_refit_learning_rate"]),
            max_turn_degrees=float(robot_settings["max_turn_degrees"]),
            device=language.device,
            sample_weights=(None if retention is None else retention.sample_weights),
            reference_turn_deltas=(
                None if retention is None else retention.outputs.turn_delta_degrees
            ),
            retention_mask=(None if retention is None else retention.shared_mask),
            retention_weight=float(settings["retention_heading_weight"]),
        )
    else:
        heading_refit = {
            "enabled": False,
            "training_rows_only": True,
            "action_waypoint_and_shared_norm_unchanged": True,
        }
    minimum_turn_margin = float(settings["minimum_training_turn_margin_degrees"])
    observed_turn_margin = heading_refit.get("minimum_turn_margin_degrees")
    if heading_refit_steps and (
        not isinstance(observed_turn_margin, (int, float))
        or float(observed_turn_margin) < minimum_turn_margin
    ):
        raise RuntimeError(
            "Waypoint heading refit missed the configured executor-margin gate: "
            f"observed={observed_turn_margin!r} required={minimum_turn_margin:.9f}"
        )
    controller.eval()
    with torch.inference_mode():
        training_outputs = heads_from_cached_hidden(
            controller, train_hidden.to(language.device)
        )
    training_metrics = waypoint_metrics(training_outputs, train)
    if retention is None:
        retention_report: dict[str, Any] = {
            "enabled": False,
            "runtime_architecture_changed": False,
        }
    else:
        final_retention = waypoint_retention_metrics(
            training_outputs, retention.outputs, train, retention.shared_mask
        )
        retention_gates = _waypoint_retention_gate_report(final_retention, settings)
        retention_report = {
            **dict(retention.metadata),
            "freeze_input_norm": bool(settings["retention_freeze_input_norm"]),
            "logit_weight": float(settings["retention_logit_weight"]),
            "waypoint_weight": float(settings["retention_waypoint_weight"]),
            "heading_weight": float(settings["retention_heading_weight"]),
            "final": final_retention,
            "gates": retention_gates,
        }
        if not retention_gates["passed"]:
            raise RuntimeError(
                _waypoint_retention_gate_failure_message(retention_gates)
            )
    minimum_training_accuracy = float(settings["minimum_training_action_accuracy"])
    if float(training_metrics["action_accuracy"]) < minimum_training_accuracy:
        raise RuntimeError(
            "Waypoint action classifier missed the configured training-accuracy gate: "
            f"observed={training_metrics['action_accuracy']:.9f} "
            f"required={minimum_training_accuracy:.9f}"
        )
    controls = evaluate_waypoint_controls(
        runner,
        cache,
        validation,
        sample_limit=settings.get("control_sample_limit"),
    )
    checkpoint_root = checkpoint or settings["checkpoint_output"]
    metadata = save_waypoint_checkpoint(
        checkpoint_root,
        controller,
        metadata={
            "architecture": "gemma_final_hidden_numeric_waypoint_policy",
            "model_id": str(config["language"]["model_id"]),
            "model_revision": str(config["language"]["revision"]),
            "gemma_runtime_binding": gemma_binding,
            "gemma_runtime_binding_sha256": gemma_runtime_binding_sha256(gemma_binding),
            "dataset_sha256": dataset.sha256,
            "training_traces_sha256": dataset.traces_sha256,
            "training_sample_count": len(train),
            "validation_sample_count": len(validation),
            "training_scene_count": len(dataset.scene_splits["train"]),
            "validation_scene_count": len(dataset.scene_splits["validation"]),
            "scene_splits_disjoint": True,
            "scene_token_count": int(settings["scene_token_count"]),
            "robot_token_count": int(settings["robot_token_count"]),
            "hidden_size": int(settings["hidden_size"]),
            "state_dim": int(settings["state_dim"]),
            "history_dim": int(settings["history_dim"]),
            "history_parameterization": str(settings["history_parameterization"]),
            "max_history_tokens": int(settings["max_history_tokens"]),
            "context_token_count": int(settings["context_token_count"]),
            "head_hidden_dim": int(settings["head_hidden_dim"]),
            "max_waypoint_step_m": float(settings["max_waypoint_step_m"]),
            "max_turn_delta_degrees": float(settings["max_turn_delta_degrees"]),
            "heading_parameterization": "robot_relative_bounded_scalar_tanh",
            "history_projector_initialization_seed": int(
                controller.history_projector.initialization_seed
            ),
            "numeric_heads_initialization_seed": int(
                controller.numeric_heads.initialization_seed
            ),
            "action_refit_l2_weight": float(settings["action_refit_l2_weight"]),
            "context_projection_frozen_during_training": True,
            "actual_gemma_causal_forward": True,
            "gemma_output_hidden_states": True,
            "complete_scene_prefix_required": True,
            "every_scene_token_processed": True,
            "numeric_state_and_history_required": True,
            "deterministic_route_planner_allowed_at_runtime": False,
            "model_selects_every_waypoint_and_heading": True,
        },
    )
    return {
        "schema": "semantic_3d_chat.gemma_waypoint_training.v1",
        "checkpoint": str(_rooted(checkpoint_root)),
        "checkpoint_weights_sha256": metadata["weights_sha256"],
        "dataset_sha256": dataset.sha256,
        "training_traces_sha256": dataset.traces_sha256,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "joint_training": {
            "legacy_configured_epochs": int(settings["epochs"]),
            "retention_override_epochs": configured_joint_epochs,
            "epochs_executed": joint_training_epochs,
            "skipped_for_authenticated_retention": (
                retention is not None and joint_training_epochs == 0
            ),
            "branch_refits_receive_authenticated_warm_start": (
                retention is not None and joint_training_epochs == 0
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "training_metrics": training_metrics,
        "action_refit": action_refit,
        "waypoint_refit": waypoint_refit,
        "heading_refit": heading_refit,
        "retention": retention_report,
        "minimum_training_turn_margin_degrees": minimum_turn_margin,
        "minimum_training_action_accuracy": minimum_training_accuracy,
        "action_refit_l2_weight": float(settings["action_refit_l2_weight"]),
        "checkpoint_selection": str(settings["checkpoint_selection"]),
        "controls": controls,
        "runtime_contract": metadata,
        "gemma_hidden_cache": {
            "training_rows": int(train_hidden.shape[0]),
            "validation_rows": int(validation_hidden.shape[0]),
            "hidden_size": int(train_hidden.shape[1]),
            "one_actual_gemma_forward_per_cached_row": True,
            "cache_persisted_to_runtime_checkpoint": False,
            "loaded_authenticated_persistent_cache": hidden_cache_metadata is not None,
        },
    }


def load_waypoint_data_from_config(
    config: Mapping[str, Any],
    *,
    dataset_path: str | Path | None = None,
) -> tuple[WaypointTraceDataset, ScenePrefixCache]:
    settings = validate_waypoint_settings(config)
    dataset = load_waypoint_trace_jsonl(
        dataset_path or str(settings["trace_dataset"]),
        state_dim=int(settings["state_dim"]),
        history_dim=int(settings["history_dim"]),
        history_parameterization=str(settings["history_parameterization"]),
        max_history_tokens=int(settings["max_history_tokens"]),
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
    )
    cache = load_scene_prefix_cache(
        str(settings["prefix_cache_root"]),
        (sample.scene_id for sample in dataset.samples),
        expected_token_count=int(settings["scene_token_count"]),
        expected_hidden_size=int(settings["hidden_size"]),
    )
    return dataset, cache


def load_actual_waypoint_stack(
    config: Mapping[str, Any],
    *,
    checkpoint: str | Path | None = None,
) -> tuple[LocalLanguageModel, nn.Module, nn.Module, str]:
    """Construct the exact local Gemma/controller/state-encoder training stack."""

    from semantic_3d_chat.language.local_lm import load_local_language_model
    from semantic_3d_chat.robot.gemma_waypoint_policy import ActualGemmaWaypointPolicy
    from semantic_3d_chat.robot.state_checkpoint import load_robot_state_checkpoint

    settings = validate_waypoint_settings(config)
    language_settings = config["language"]
    runtime_base_checkpoint = settings.get("runtime_aligned_base_checkpoint")
    runtime_control_config = settings.get("runtime_aligned_control_config")
    runtime_control_checkpoint = settings.get("runtime_aligned_control_checkpoint")
    aligned_values = (
        runtime_base_checkpoint,
        runtime_control_config,
        runtime_control_checkpoint,
    )
    if any(value is not None for value in aligned_values) and not all(
        value is not None for value in aligned_values
    ):
        raise ValueError(
            "runtime_aligned_base_checkpoint, runtime_aligned_control_config, and "
            "runtime_aligned_control_checkpoint must be configured together"
        )
    if runtime_base_checkpoint is not None:
        # The deployed question-controlled runtime installs checkpointed Gemma
        # LoRA banks.  Raw-HF states are not numerically interchangeable.
        # Caching against a raw HF model gives a different decision hidden even
        # when the continuous prefix and instruction hashes are identical.
        from semantic_3d_chat.chat.question_control_runtime import (
            QuestionControlledChatRuntime,
        )
        from semantic_3d_chat.chat.runtime_config import load_runtime_config

        runtime_config = load_runtime_config(str(runtime_control_config))
        aligned_scene = str(settings.get("runtime_aligned_scene_id", "scene_000001"))
        aligned_runtime = QuestionControlledChatRuntime.load(
            runtime_config,
            aligned_scene,
            base_checkpoint=str(runtime_base_checkpoint),
            control_checkpoint=str(runtime_control_checkpoint),
        )
        language = aligned_runtime.base.language
        binding = question_controlled_gemma_runtime_binding(
            aligned_runtime,
            runtime_config,
            base_checkpoint=str(runtime_base_checkpoint),
            control_checkpoint=str(runtime_control_checkpoint),
        )
        attach_gemma_runtime_binding(language, binding)
        # Retain the owner for the duration of caching/training; the language
        # model itself owns the installed LoRA modules, while this reference
        # also makes the exact source stack explicit during debugging.
        language._waypoint_aligned_runtime = aligned_runtime
    else:
        language = load_local_language_model(
            str(language_settings["model_id"]),
            revision=str(language_settings["revision"]),
            requested_dtype=str(language_settings.get("dtype", "bfloat16")),
            freeze=True,
            local_files_only=True,
            backend="gemma4",
            decoder_gradient_checkpointing=False,
        )
        attach_gemma_runtime_binding(
            language,
            raw_hf_gemma_runtime_binding(
                model_id=str(language_settings["model_id"]),
                model_revision=str(language_settings["revision"]),
                language_dtype=str(language_settings.get("dtype", "bfloat16")),
            ),
        )
    controller = ActualGemmaWaypointPolicy(
        hidden_size=int(settings["hidden_size"]),
        scene_token_count=int(settings["scene_token_count"]),
        robot_token_count=int(settings["robot_token_count"]),
        history_feature_dim=int(settings["history_dim"]),
        max_history_tokens=int(settings["max_history_tokens"]),
        head_hidden_dim=int(settings["head_hidden_dim"]),
        max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
        max_turn_delta_degrees=float(settings["max_turn_delta_degrees"]),
        freeze_context_projection=True,
    ).to(language.device)
    state_encoder, state_hash, state_metadata = load_robot_state_checkpoint(
        str(settings["robot_state_checkpoint"]),
        expected_output_dim=int(settings["hidden_size"]),
        device=language.device,
    )
    if state_metadata.get("token_count") != int(settings["robot_token_count"]):
        raise ValueError("Robot-state checkpoint token count differs from waypoint config")
    if checkpoint is not None:
        metadata = load_waypoint_checkpoint(checkpoint, controller)
        expected_binding = language_gemma_runtime_binding(language)
        if (
            metadata.get("model_id") != str(language_settings["model_id"])
            or metadata.get("model_revision") != str(language_settings["revision"])
            or metadata.get("scene_token_count") != int(settings["scene_token_count"])
            or metadata.get("robot_token_count") != int(settings["robot_token_count"])
            or metadata.get("heading_parameterization")
            != "robot_relative_bounded_scalar_tanh"
            or metadata.get("history_dim") != int(settings["history_dim"])
            or metadata.get("history_parameterization")
            != str(settings["history_parameterization"])
            or float(metadata.get("max_turn_delta_degrees", math.nan))
            != float(settings["max_turn_delta_degrees"])
            or metadata.get("gemma_runtime_binding") != expected_binding
            or metadata.get("gemma_runtime_binding_sha256")
            != gemma_runtime_binding_sha256(expected_binding)
        ):
            raise ValueError("Waypoint checkpoint model or prefix contract differs")
    return language, controller, state_encoder, state_hash


__all__ = [
    "ACTION_NAMES",
    "ACTION_TO_INDEX",
    "ActualGemmaWaypointForward",
    "GemmaWaypointControllerProtocol",
    "ScenePrefixCache",
    "WaypointPolicyTensors",
    "WaypointRetentionReference",
    "WaypointTraceDataset",
    "WaypointTraceSample",
    "assemble_demo_scene_prefix_cache",
    "cache_actual_gemma_decision_hidden",
    "evaluate_waypoint_condition",
    "evaluate_waypoint_controls",
    "gemma_hidden_input_binding",
    "heads_from_cached_hidden",
    "load_actual_waypoint_stack",
    "load_gemma_hidden_cache",
    "load_gemma_hidden_cache_for_forward_revalidation",
    "load_scene_prefix_cache",
    "load_waypoint_checkpoint",
    "load_waypoint_data_from_config",
    "load_waypoint_retention_reference",
    "load_waypoint_trace_jsonl",
    "normalize_policy_output",
    "refit_waypoint_action_classifier",
    "refit_waypoint_action_classifier_constrained",
    "refit_waypoint_heading_head",
    "save_gemma_hidden_cache",
    "save_waypoint_checkpoint",
    "select_balanced_waypoint_samples",
    "train_waypoint_controller",
    "validate_waypoint_settings",
    "waypoint_loss",
    "waypoint_metrics",
    "waypoint_retention_loss",
    "waypoint_retention_metrics",
]

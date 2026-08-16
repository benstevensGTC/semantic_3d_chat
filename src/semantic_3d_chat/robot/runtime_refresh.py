"""Atomic semantic-map to question-independent scene-prefix refresh.

This module binds the embodied map transaction to the same full-scene encoder
used by static chat.  A candidate prefix is built from every voxel in the
staged numeric map before that map is committed; failed fusion or tokenization
leaves both the prior map and prior runtime current.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.config import PROJECT_ROOT, project_path
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.semantic_mapping import (
    PersistentSemanticMapUpdater,
    persistent_semantic_map_identity,
    semantic_map_content_hash,
)
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator
from semantic_3d_chat.robot.state_encoder import (
    NumericRobotState,
    RobotStateEncoder,
    insert_robot_state_tokens,
    robot_state_vector,
    robot_state_vector_sha256,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.vision.encoder import (
    DenseImageEncoder,
    load_configured_dense_image_encoder,
)

_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_BINDING_SCHEMA: Final[str] = "semantic_3d_chat.scene_prefix_binding.v2"
_PROTECTED_MAP_COMPONENTS: Final[frozenset[str]] = frozenset(
    {
        "experiments",
        "features",
        "oracle",
        "predictions",
        "qa",
        "questions",
        "rendered",
        "scorer",
        "scorer_only",
        "scorer-only",
        "training",
    }
)
_AUTO_SCAN_ACTIONS: Final[frozenset[str]] = frozenset(
    {"look", "turn", "move_forward", "move_backward", "move_to"}
)

RuntimeBuilder = Callable[[Any, MapTensorData], Any]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_map_path(path: str | Path, *, purpose: str) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    unresolved = Path(os.path.abspath(rooted))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} cannot use a symbolic-link path")
    if _PROTECTED_MAP_COMPONENTS.intersection(
        part.casefold() for part in unresolved.parts
    ):
        raise ValueError(
            f"{purpose} cannot use an oracle or QA path or another protected runtime path"
        )
    return unresolved


def robot_state_encoder_sha256(encoder: RobotStateEncoder) -> str:
    """Hash every numeric state-encoder parameter and buffer deterministically."""

    digest = hashlib.sha256()
    state = encoder.state_dict()
    if not state:
        raise ValueError("Robot-state encoder has no checkpointable state")
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError("Robot-state encoder contains NaN or infinity")
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(torch.tensor(tensor.shape, dtype=torch.int64).numpy().tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ScenePrefixBinding:
    scene_id: str
    map_version: int
    map_sha256: str
    scene_prefix_sha256: str
    scene_control_signature_sha256: str | None
    source_voxels: int
    processed_voxels: int

    def identity(self) -> dict[str, int | str | None]:
        return {
            "schema": _BINDING_SCHEMA,
            "scene_id": self.scene_id,
            "map_version": self.map_version,
            "map_sha256": self.map_sha256,
            "scene_prefix_sha256": self.scene_prefix_sha256,
            "scene_control_signature_sha256": self.scene_control_signature_sha256,
            "source_voxels": self.source_voxels,
            "processed_voxels": self.processed_voxels,
        }

    @property
    def binding_sha256(self) -> str:
        return _canonical_sha256(self.identity())

    def as_dict(self) -> dict[str, int | str | None]:
        return {**self.identity(), "binding_sha256": self.binding_sha256}


@dataclass(frozen=True)
class _PreparedPrefixCommit:
    prior_runtime: Any
    prior_binding: ScenePrefixBinding
    candidate_runtime: Any
    candidate_binding: ScenePrefixBinding


def _static_base(runtime: Any) -> Any:
    """Return the immutable full-scene runtime beneath an optional controller."""

    if _is_signature_control_runtime(runtime) or _is_fixed_scene_memory_runtime(
        runtime
    ):
        return runtime.base
    return runtime


def _is_fixed_scene_memory_runtime(runtime: Any) -> bool:
    """Recognize a pre-question full-memory runtime without importing its version.

    V81/V83 and the gated V96 successor keep the map-derived 258-token base on
    ``runtime.base`` while supplying a separately authenticated 738-token
    memory to Gemma.  Structural recognition avoids a chat/robot import cycle,
    and the invariant checks below still fail closed on every tensor and hash.
    """

    memory = getattr(runtime, "fixed_scene_memory", None)
    return bool(
        getattr(runtime, "base", None) is not None
        and isinstance(memory, torch.Tensor)
        and memory.ndim == 3
        and memory.shape[0] == 1
        and isinstance(getattr(runtime, "base_scene_prefix_hash", None), str)
        and callable(getattr(runtime, "current_prefix_hash", None))
        and callable(getattr(runtime, "assert_prefix_unchanged", None))
    )


def _is_signature_control_runtime(runtime: Any) -> bool:
    """Recognize a cached full-scene controller without naming its subclass.

    Older controllers cache one scene signature.  The V74/V75 dense reader
    caches a key/value pair instead.  Both are built before user text arrives
    and must be rebuilt transactionally whenever the semantic map changes.
    """

    control = getattr(runtime, "control", None)
    common = (
        getattr(runtime, "base", None) is not None
        and isinstance(getattr(runtime, "control_metadata", None), Mapping)
        and callable(getattr(control, "encode_scene", None))
    )
    signature_cache = callable(
        getattr(control, "forward_from_signature", None)
    ) and hasattr(runtime, "_scene_control_signature")
    dense_key_value_cache = callable(
        getattr(control, "forward_encoded", None)
    ) and all(
        hasattr(runtime, field)
        for field in ("_scene_control_key", "_scene_control_value")
    )
    return bool(common and (signature_cache or dense_key_value_cache))


def _control_signature_sha256(
    runtime: Any,
    *,
    verify_against_prefix: bool = False,
) -> str | None:
    """Attest the question-independent signature cached by a control runtime."""

    if _is_fixed_scene_memory_runtime(runtime):
        runtime.assert_prefix_unchanged()
        observed = runtime.current_prefix_hash()
        if not isinstance(observed, str) or _SHA256.fullmatch(observed) is None:
            raise RuntimeError("Fixed scene memory has an invalid canonical hash")
        base = runtime.base
        if verify_against_prefix and (
            base.current_prefix_hash() != runtime.base_scene_prefix_hash
            or prefix_sha256(runtime.scene_prefix)
            != runtime.base_scene_prefix_hash
            or tuple(runtime.scene_prefix.shape) != tuple(base.scene_prefix.shape)
            or not torch.equal(runtime.scene_prefix.to(base.scene_prefix), base.scene_prefix)
        ):
            raise RuntimeError(
                "Fixed scene memory does not reconstruct the refreshed map-derived prefix"
            )
        return observed
    if not _is_signature_control_runtime(runtime):
        return None
    signature = getattr(runtime, "_scene_control_signature", None)
    key = getattr(runtime, "_scene_control_key", None)
    value = getattr(runtime, "_scene_control_value", None)
    dense = isinstance(key, torch.Tensor) or isinstance(value, torch.Tensor)
    if dense:
        if (
            not isinstance(key, torch.Tensor)
            or not isinstance(value, torch.Tensor)
            or key.ndim != 3
            or value.shape != key.shape
            or key.shape[0] != 1
            or not torch.isfinite(key).all()
            or not torch.isfinite(value).all()
        ):
            raise RuntimeError("Question-controlled runtime has an invalid scene K/V cache")
        observed = prefix_sha256(torch.cat((key, value), dim=-1))
    else:
        if not isinstance(signature, torch.Tensor) or signature.ndim != 3:
            raise RuntimeError("Question-controlled runtime lacks a cached scene signature")
        if signature.shape[0] != 1 or not torch.isfinite(signature).all():
            raise RuntimeError("Question-controlled scene signature is invalid")
        observed = prefix_sha256(signature)
    if verify_against_prefix:
        with torch.inference_mode():
            recomputed = runtime.control.encode_scene(
                runtime.base.scene_prefix.float()
            )
        if dense:
            if (
                not isinstance(recomputed, tuple)
                or len(recomputed) != 2
                or not all(isinstance(item, torch.Tensor) for item in recomputed)
            ):
                raise RuntimeError("Recomputed question-control scene K/V is invalid")
            recomputed_key, recomputed_value = recomputed
            recomputed_hash = prefix_sha256(
                torch.cat((recomputed_key, recomputed_value), dim=-1)
            )
            shape_matches = (
                recomputed_key.shape == key.shape
                and recomputed_value.shape == value.shape
            )
        else:
            if not isinstance(recomputed, torch.Tensor):
                raise RuntimeError("Recomputed question-control scene signature is invalid")
            recomputed_hash = prefix_sha256(recomputed)
            shape_matches = recomputed.shape == signature.shape
        if (
            not shape_matches
            or not all(torch.isfinite(item).all() for item in (
                recomputed if isinstance(recomputed, tuple) else (recomputed,)
            ))
            or recomputed_hash != observed
        ):
            raise RuntimeError(
                "Cached question-control signature does not match the refreshed full prefix"
            )
    return observed


def _rebuild_static_base(previous: StaticChatRuntime, map_data: MapTensorData) -> StaticChatRuntime:
    """Re-run the immutable full-scene stack without reloading model weights."""

    return StaticChatRuntime(
        config=previous.config,
        scene_id=previous.scene_id,
        checkpoint_path=previous.checkpoint_path,
        checkpoint_metadata=previous.checkpoint_metadata,
        language=previous.language,
        map_data=map_data,
        scene_model=previous.scene_model,
        dense_aligner=previous.dense_aligner,
        dense_sidecar_adapter=previous.dense_sidecar_adapter,
        block_cross_residual=previous.block_cross_residual,
        global_scene_residual=previous.global_scene_residual,
        signed_x_scene_residual=previous.signed_x_scene_residual,
        composer=previous.composer,
        grounding=previous.grounding,
        warnings=previous.warnings,
        generation_function=previous._generation_function,
    )


def _rebuild_runtime(previous: Any, map_data: MapTensorData) -> Any:
    """Rebuild a base prefix and, when present, its full-scene control cache."""

    if _is_fixed_scene_memory_runtime(previous):
        raise RuntimeError(
            "A fixed scene-memory runtime requires an explicit question-free "
            "memory compiler for embodied refresh"
        )
    if _is_signature_control_runtime(previous):
        candidate_base = _rebuild_static_base(previous.base, map_data)
        # Construction calls the controller's existing ``encode_scene``
        # interface.  V3/V4/V5 and V6 all share this contract, so no
        # architecture-specific routing or semantic policy is introduced by
        # embodied refresh.
        grounding = getattr(previous, "grounding_sidecar", None)
        if grounding is None:
            return type(previous)(
                candidate_base,
                previous.control,
                previous.control_metadata,
            )
        refreshed_grounding = type(grounding)(
            checkpoint=grounding.checkpoint,
            model=grounding.model,
            metadata=grounding.metadata,
            scene_prefix=candidate_base.scene_prefix,
            room_min=candidate_base.map_data.room_min,
            room_max=candidate_base.map_data.room_max,
        )
        return type(previous)(
            candidate_base,
            previous.control,
            previous.control_metadata,
            grounding_sidecar=refreshed_grounding,
        )
    return _rebuild_static_base(previous, map_data)


class QuestionIndependentPrefixRefresher:
    """Transactional consumer of staged numeric semantic maps."""

    def __init__(
        self,
        runtime: Any,
        *,
        base_map_path: str | Path,
        runtime_builder: RuntimeBuilder = _rebuild_runtime,
        audit: FileAccessAudit | None = None,
    ) -> None:
        base = _static_base(runtime)
        if not isinstance(base.scene_id, str) or _SCENE_ID.fullmatch(base.scene_id) is None:
            raise ValueError("Runtime scene_id must be opaque")
        self._lock = threading.RLock()
        self._runtime_builder = runtime_builder
        self._audit = audit
        self._runtime = runtime
        source = _safe_map_path(base_map_path, purpose="scene-prefix base map")
        if not source.is_file():
            raise FileNotFoundError(f"Scene-prefix base map is unavailable: {source}")
        if self._audit is not None:
            self._audit.record(source)
        base_hash = semantic_map_content_hash(source)
        self._binding = self._binding_for(runtime, map_version=0, map_sha256=base_hash)
        self._base_runtime = runtime
        self._base_binding = self._binding

    @staticmethod
    def _binding_for(
        runtime: Any,
        *,
        map_version: int,
        map_sha256: str,
    ) -> ScenePrefixBinding:
        if isinstance(map_version, bool) or not isinstance(map_version, int) or map_version < 0:
            raise ValueError("Scene-prefix map version must be nonnegative")
        if not isinstance(map_sha256, str) or _SHA256.fullmatch(map_sha256) is None:
            raise ValueError("Scene-prefix map hash must be lowercase SHA-256")
        base = _static_base(runtime)
        prefix_hash = base.current_prefix_hash()
        if prefix_hash != base.scene_prefix_hash:
            raise RuntimeError("Candidate scene prefix changed during construction")
        if (
            _is_signature_control_runtime(runtime)
            and runtime.scene_prefix_hash != prefix_hash
        ):
            raise RuntimeError("Question-control wrapper differs from its base prefix")
        if _is_fixed_scene_memory_runtime(runtime) and (
            runtime.base_scene_prefix_hash != prefix_hash
            or prefix_sha256(runtime.scene_prefix) != prefix_hash
        ):
            raise RuntimeError(
                "Fixed scene memory differs from its map-derived base prefix"
            )
        control_signature_hash = _control_signature_sha256(
            runtime,
            verify_against_prefix=True,
        )
        source_voxels = int(base.map_data.source_voxel_count)
        processed_voxels = int(base.map_data.voxel_count)
        if source_voxels < 1 or processed_voxels < 1:
            raise ValueError("Scene-prefix runtime cannot bind an empty map")
        return ScenePrefixBinding(
            scene_id=base.scene_id,
            map_version=map_version,
            map_sha256=map_sha256,
            scene_prefix_sha256=prefix_hash,
            scene_control_signature_sha256=control_signature_hash,
            source_voxels=source_voxels,
            processed_voxels=processed_voxels,
        )

    @property
    def runtime(self) -> Any:
        with self._lock:
            return self._runtime

    @property
    def binding(self) -> ScenePrefixBinding:
        with self._lock:
            return self._binding

    def _candidate(
        self,
        map_path: Path,
        *,
        map_version: int,
        map_sha256: str,
    ) -> Any:
        base = _static_base(self._runtime)
        if self._audit is not None:
            self._audit.record(map_path)
        map_data = load_map_tensors(
            map_path,
            base.config["scene"]["room_size_m"],
            device=base.map_data.semantic.device,
            input_voxel_size_m=base.config["scene_encoder"].get(
                "input_voxel_size_m"
            ),
        )
        expected_dim = int(base.checkpoint_metadata.get("semantic_dim", map_data.feature_dim))
        if map_data.feature_dim != expected_dim:
            raise ValueError("Updated semantic map differs from checkpoint semantic dimension")
        candidate = self._runtime_builder(self._runtime, map_data)
        if _static_base(candidate).scene_id != base.scene_id:
            raise ValueError("Refreshed runtime changed its opaque scene ID")
        binding = self._binding_for(
            candidate,
            map_version=map_version,
            map_sha256=map_sha256,
        )
        return candidate, binding

    def prepare_map_commit(
        self,
        staged_map_path: Path,
        receipt: Mapping[str, Any],
    ) -> object:
        """Build the full prefix before the staged map becomes visible."""

        with self._lock:
            identity = persistent_semantic_map_identity(
                staged_map_path,
                _static_base(self._runtime).scene_id,
            )
            required = {
                "map_version": self._binding.map_version + 1,
                "map_sha256": identity["map_sha256"],
            }
            if identity["map_version"] != required["map_version"]:
                raise ValueError("Staged semantic-map version is not the next prefix version")
            if any(receipt.get(key) != value for key, value in required.items()):
                raise ValueError("Semantic-map receipt differs from staged numeric content")
            candidate, binding = self._candidate(
                staged_map_path,
                map_version=int(identity["map_version"]),
                map_sha256=str(identity["map_sha256"]),
            )
            return _PreparedPrefixCommit(
                prior_runtime=self._runtime,
                prior_binding=self._binding,
                candidate_runtime=candidate,
                candidate_binding=binding,
            )

    def commit_map(self, prepared: object) -> None:
        if not isinstance(prepared, _PreparedPrefixCommit):
            raise TypeError("Invalid prepared scene-prefix transaction")
        with self._lock:
            if self._runtime is not prepared.prior_runtime or self._binding != prepared.prior_binding:
                raise RuntimeError("Scene-prefix runtime changed during map transaction")
            # These two assignments are the only in-memory commit operation.
            self._runtime = prepared.candidate_runtime
            self._binding = prepared.candidate_binding

    def rollback_map(self, prepared: object) -> None:
        if not isinstance(prepared, _PreparedPrefixCommit):
            raise TypeError("Invalid scene-prefix rollback transaction")
        with self._lock:
            self._runtime = prepared.prior_runtime
            self._binding = prepared.prior_binding

    def prepare_base_reset(self) -> object:
        """Prepare restoring the authenticated static map and its cached prefix."""

        with self._lock:
            self._binding_for(
                self._base_runtime,
                map_version=0,
                map_sha256=self._base_binding.map_sha256,
            )
            return _PreparedPrefixCommit(
                prior_runtime=self._runtime,
                prior_binding=self._binding,
                candidate_runtime=self._base_runtime,
                candidate_binding=self._base_binding,
            )

    def commit_base_reset(self, prepared: object) -> None:
        self.commit_map(prepared)

    def rollback_base_reset(self, prepared: object) -> None:
        self.rollback_map(prepared)

    def load_existing(self, path: str | Path) -> None:
        """Rebuild from a previously committed persistent map at startup."""

        source = _safe_map_path(path, purpose="existing persistent semantic map")
        if self._audit is not None:
            self._audit.record(source)
        identity = persistent_semantic_map_identity(
            source,
            _static_base(self._runtime).scene_id,
        )
        with self._lock:
            candidate, binding = self._candidate(
                source,
                map_version=int(identity["map_version"]),
                map_sha256=str(identity["map_sha256"]),
            )
            self._runtime = candidate
            self._binding = binding


@dataclass(frozen=True)
class ActivePrefixBinding:
    scene: ScenePrefixBinding
    active_prefix_sha256: str
    robot_state_sha256: str | None
    robot_tokens_sha256: str | None
    robot_state_encoder_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            **self.scene.as_dict(),
            "active_prefix_sha256": self.active_prefix_sha256,
            "robot_state_sha256": self.robot_state_sha256,
            "robot_tokens_sha256": self.robot_tokens_sha256,
            "robot_state_encoder_sha256": self.robot_state_encoder_sha256,
        }
        payload["active_binding_sha256"] = _canonical_sha256(payload)
        return payload


class CheckpointBoundRobotStateTokens:
    """Continuous numeric robot tokens accepted only from an exact state hash."""

    def __init__(self, encoder: RobotStateEncoder, *, expected_sha256: str) -> None:
        if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
            raise ValueError("Expected robot-state encoder hash must be lowercase SHA-256")
        observed = robot_state_encoder_sha256(encoder)
        if observed != expected_sha256:
            raise ValueError("Robot-state encoder checkpoint hash mismatch")
        self.encoder = encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.state_sha256 = observed

    @torch.inference_mode()
    def encode(
        self,
        simulator: EmbodiedCameraSimulator,
        map_data: MapTensorData,
        scene_prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, str, str]:
        try:
            device = next(self.encoder.parameters()).device
        except StopIteration:  # pragma: no cover - RobotStateEncoder always has parameters
            device = torch.device("cpu")
        vector = robot_state_vector(
            simulator.numeric_state(),
            map_data.room_min,
            map_data.room_max,
            device=device,
        )
        vector_hash = robot_state_vector_sha256(vector)
        tokens = self.encoder(vector)
        if tokens.shape[-1] != scene_prefix.shape[-1]:
            raise ValueError("Robot-state token width differs from language hidden size")
        # The active prefix stores robot tokens after matching the scene
        # prefix's device and dtype.  Bind the hash to those exact inserted
        # bytes, not to the encoder's pre-cast output (normally float32).  A
        # bfloat16 Gemma runtime would otherwise publish a robot-token digest
        # that could never be reconstructed from its own active prefix.
        bound_tokens = tokens.to(scene_prefix)
        token_hash = prefix_sha256(bound_tokens)
        return bound_tokens, vector_hash, token_hash


class RefreshingEmbodiedChatRuntime:
    """Serialized robot actions, map commits, prefix refreshes, and chat."""

    def __init__(
        self,
        simulator: EmbodiedCameraSimulator,
        map_updater: PersistentSemanticMapUpdater,
        prefix_refresher: QuestionIndependentPrefixRefresher,
        *,
        robot_tokens: CheckpointBoundRobotStateTokens | None = None,
        auto_scan_after_motion: bool = False,
    ) -> None:
        if not isinstance(auto_scan_after_motion, bool):
            raise TypeError("auto_scan_after_motion must be boolean")
        self.simulator = simulator
        self.map_updater = map_updater
        self.prefix_refresher = prefix_refresher
        self.robot_tokens = robot_tokens
        self.auto_scan_after_motion = auto_scan_after_motion
        self._lock = threading.RLock()
        self._active_prefix = torch.empty(0)
        self._active_binding: ActivePrefixBinding
        self._refresh_active_prefix()

    def _assert_scene_memory_current(self) -> None:
        runtime = self.prefix_refresher.runtime
        base = _static_base(runtime)
        binding = self.prefix_refresher.binding
        if base.current_prefix_hash() != binding.scene_prefix_sha256:
            raise RuntimeError("Active base prefix differs from its semantic-map binding")
        if _control_signature_sha256(runtime) != (
            binding.scene_control_signature_sha256
        ):
            raise RuntimeError(
                "Active question-control signature differs from its semantic-map binding"
            )

    def _refresh_active_prefix(self) -> None:
        self._assert_scene_memory_current()
        runtime = self.prefix_refresher.runtime
        base = _static_base(runtime)
        scene = self.prefix_refresher.binding
        prefix = (
            runtime.fixed_scene_memory.detach()
            if _is_fixed_scene_memory_runtime(runtime)
            else base.scene_prefix.detach()
        )
        robot_state_hash: str | None = None
        robot_token_hash: str | None = None
        encoder_hash: str | None = None
        if self.robot_tokens is not None:
            state_tokens, robot_state_hash, robot_token_hash = self.robot_tokens.encode(
                self.simulator,
                base.map_data,
                prefix,
            )
            prefix = insert_robot_state_tokens(prefix, state_tokens).detach()
            encoder_hash = self.robot_tokens.state_sha256
        self._active_prefix = prefix
        self._active_binding = ActivePrefixBinding(
            scene=scene,
            active_prefix_sha256=prefix_sha256(prefix),
            robot_state_sha256=robot_state_hash,
            robot_tokens_sha256=robot_token_hash,
            robot_state_encoder_sha256=encoder_hash,
        )

    @property
    def scene_prefix_hash(self) -> str:
        return self.prefix_refresher.binding.scene_prefix_sha256

    @property
    def active_prefix_hash(self) -> str:
        return self._active_binding.active_prefix_sha256

    def prefix_binding(self) -> dict[str, Any]:
        with self._lock:
            return self._active_binding.as_dict()

    def active_prefix_snapshot(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return an immutable action-policy snapshot of scene plus robot tokens.

        The prefix was constructed before any action instruction was supplied.
        Copying it while holding the runtime lock prevents a concurrent robot
        action from pairing stale continuous tokens with a newer numeric pose.
        No map path or environmental text is exposed through this seam.
        """

        with self._lock:
            self._assert_scene_memory_current()
            if self._state_hash_now() != self._active_binding.robot_state_sha256:
                raise RuntimeError("Robot state changed without refreshing its continuous tokens")
            return self._active_prefix.detach().clone(), self._active_binding.as_dict()

    def continuous_action_context_snapshot(
        self,
    ) -> tuple[torch.Tensor, dict[str, Any], NumericRobotState]:
        """Atomically snapshot the action prefix and its exact numeric state.

        Navigation backends derive target-relative geometry and clearance from
        the returned immutable numeric state.  The shared lock prevents an
        action on another thread from mixing a newer pose with older scene and
        robot tokens.
        """

        with self._lock:
            self._assert_scene_memory_current()
            state = self.simulator.numeric_state()
            if self._state_hash_now() != self._active_binding.robot_state_sha256:
                raise RuntimeError("Robot state changed without refreshing its continuous tokens")
            return (
                self._active_prefix.detach().clone(),
                self._active_binding.as_dict(),
                state,
            )

    def _action(self, name: str, *arguments: Any) -> dict[str, Any]:
        with self._lock:
            before = self.prefix_refresher.binding
            result = getattr(self.simulator, name)(*arguments)
            scanned = name == "scan"
            if (
                self.auto_scan_after_motion
                and result["success"]
                and name in _AUTO_SCAN_ACTIONS
            ):
                action_result = result
                result = self.simulator.scan()
                scanned = True
                # Preserve the bounded action receipt while returning the
                # current post-observation numerical state. No semantic text
                # or simulator labels are introduced into the tool payload.
                for field in ("distance_moved", "turn_degrees", "clearance_m"):
                    result[field] = action_result[field]
            after = self.prefix_refresher.binding
            if result["success"] and scanned:
                if (
                    after.map_version != result["scene_version"]
                    or after.map_sha256 != result.get("map_sha256")
                ):
                    raise RuntimeError("Robot/map/prefix transaction receipt mismatch")
            elif scanned and after != before:
                raise RuntimeError("Failed scan changed the scene-prefix binding")
            self._refresh_active_prefix()
            return {**result, **self.prefix_binding()}

    def get_robot_state(self) -> dict[str, Any]:
        with self._lock:
            return {**self.simulator.get_robot_state(), **self.prefix_binding()}

    def look(self, yaw_delta_degrees: Any, pitch_delta_degrees: Any) -> dict[str, Any]:
        return self._action("look", yaw_delta_degrees, pitch_delta_degrees)

    def turn(self, angle_degrees: Any) -> dict[str, Any]:
        return self._action("turn", angle_degrees)

    def move_forward(self, distance_meters: Any) -> dict[str, Any]:
        return self._action("move_forward", distance_meters)

    def move_backward(self, distance_meters: Any) -> dict[str, Any]:
        return self._action("move_backward", distance_meters)

    def move_to(self, x: Any, y: Any) -> dict[str, Any]:
        return self._action("move_to", x, y)

    def scan(self) -> dict[str, Any]:
        return self._action("scan")

    def stop(self) -> dict[str, Any]:
        return self._action("stop")

    def reset_scene(self, scene_id: str, seed: Any) -> dict[str, Any]:
        """Atomically restore this runtime's base map, pose, and scene prefix.

        A loaded language/vision runtime is bound to one opaque scene, so a
        cross-scene reset still fails closed. Resetting that same scene removes
        its exact persistent map file under the updater lock, restores the
        pre-question base prefix, preserves the authenticated renderer, and
        resets its numerical coverage before returning success.
        """

        with self._lock:
            if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
                result = self.simulator.protocol_error("E_SCENE_ID")
                self._refresh_active_prefix()
                return {**result, **self.prefix_binding()}
            if scene_id != self.simulator.state.scene_id:
                result = self.simulator.protocol_error("E_SCENE_UNAVAILABLE")
                self._refresh_active_prefix()
                return {**result, **self.prefix_binding()}
            if isinstance(seed, bool):
                result = self.simulator.protocol_error("E_NUMERIC")
                self._refresh_active_prefix()
                return {**result, **self.prefix_binding()}
            try:
                numeric_seed = float(seed)
            except (TypeError, ValueError):
                numeric_seed = math.nan
            if (
                not math.isfinite(numeric_seed)
                or not numeric_seed.is_integer()
                or numeric_seed < 0
                or numeric_seed > 2**32 - 1
            ):
                result = self.simulator.protocol_error("E_NUMERIC")
                self._refresh_active_prefix()
                return {**result, **self.prefix_binding()}
            try:
                prepared = self.simulator.prepare_scene_reset(scene_id, int(numeric_seed))
                retained_scanner = self.simulator.scanner
                snapshot_episode = getattr(retained_scanner, "snapshot_episode_state", None)
                restore_episode = getattr(retained_scanner, "restore_episode_state", None)
                reset_episode = getattr(retained_scanner, "reset_episode", None)
                if not all(
                    callable(value)
                    for value in (snapshot_episode, restore_episode, reset_episode)
                ):
                    raise TypeError("Semantic scanner lacks transactional reset methods")
                scanner_snapshot = snapshot_episode()
                try:
                    reset_episode()
                    self.map_updater.reset_to_base()
                    result = self.simulator.commit_scene_reset(
                        prepared,
                        scanner_override=retained_scanner,
                        reset_scanner=False,
                    )
                except BaseException:
                    restore_episode(scanner_snapshot)
                    raise
            except (OSError, RuntimeError, TypeError, ValueError):
                result = self.simulator.protocol_error("E_MAP_RESET")
            self.simulator.state.scene_version = self.map_updater.current_version
            self.simulator.state.scan_count = self.map_updater.current_version
            self.simulator.state.map_sha256 = self.map_updater.current_map_sha256
            self._refresh_active_prefix()
            return {**result, **self.prefix_binding()}

    def _state_hash_now(self) -> str | None:
        if self.robot_tokens is None:
            return None
        runtime = self.prefix_refresher.runtime
        base = _static_base(runtime)
        vector = robot_state_vector(
            self.simulator.numeric_state(),
            base.map_data.room_min,
            base.map_data.room_max,
        )
        return robot_state_vector_sha256(vector)

    def answer(self, question: str) -> ChatAnswer:
        """Answer with a prefix cached before this question was supplied."""

        with self._lock:
            self._assert_scene_memory_current()
            if self._state_hash_now() != self._active_binding.robot_state_sha256:
                raise RuntimeError("Robot state changed without refreshing its continuous tokens")
            runtime = self.prefix_refresher.runtime
            if self.robot_tokens is None:
                result = runtime.answer(question)
                self._assert_scene_memory_current()
                return result
            if _is_fixed_scene_memory_runtime(runtime):
                raise RuntimeError(
                    "Robot-state tokens are bound to the numeric MCP/action prefix, but "
                    "this fixed-memory chat runtime has no authenticated direct-generation "
                    "layout for those additional tokens"
                )
            base = _static_base(runtime)
            original_prefix = base.scene_prefix
            original_base_hash = base.scene_prefix_hash
            original_runtime_hash = (
                runtime.scene_prefix_hash
                if _is_signature_control_runtime(runtime)
                else None
            )
            base.scene_prefix = self._active_prefix
            base.scene_prefix_hash = self._active_binding.active_prefix_sha256
            if _is_signature_control_runtime(runtime):
                # Robot state is a separate numeric prefix seam.  It affects
                # generation but never changes the cached environmental scene
                # signature used by the question controller.
                runtime.scene_prefix_hash = self._active_binding.active_prefix_sha256
            try:
                return runtime.answer(question)
            finally:
                base.scene_prefix = original_prefix
                base.scene_prefix_hash = original_base_hash
                if _is_signature_control_runtime(runtime):
                    assert original_runtime_hash is not None
                    runtime.scene_prefix_hash = original_runtime_hash
                self._assert_scene_memory_current()


def build_refreshing_embodied_runtime(
    config: dict[str, Any],
    scene_id: str,
    *,
    checkpoint: str | Path,
    chat_runtime: Any | None = None,
    vision_encoder: DenseImageEncoder | None = None,
    persistent_map_path: str | Path | None = None,
    runtime_builder: RuntimeBuilder = _rebuild_runtime,
    observation_scanner: Any | None = None,
    robot_state_encoder: RobotStateEncoder | None = None,
    robot_state_encoder_sha256: str | None = None,
    robot_state_checkpoint: str | Path | None = None,
    audit: FileAccessAudit | None = None,
    local_files_only: bool = True,
) -> RefreshingEmbodiedChatRuntime:
    """Build map update and complete-prefix refresh with shared local weights."""

    if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
        raise ValueError("scene_id must be opaque")
    if robot_state_checkpoint is not None and robot_state_encoder is not None:
        raise ValueError(
            "Supply a robot-state checkpoint or an in-memory encoder, not both"
        )
    if (robot_state_encoder is None) != (robot_state_encoder_sha256 is None):
        raise ValueError("Robot-state encoder and exact checkpoint hash must be supplied together")
    base_map = _safe_map_path(
        project_path(config, "maps", scene_id, "voxel_map.npz"),
        purpose="embodied base semantic map",
    )
    persistent = _safe_map_path(
        persistent_map_path
        or project_path(config, "robot", scene_id, "semantic_map.npz"),
        purpose="embodied persistent semantic map",
    )
    runtime = chat_runtime or StaticChatRuntime.load(
        config,
        scene_id,
        checkpoint=checkpoint,
        audit=audit,
        local_files_only=local_files_only,
    )
    if robot_state_checkpoint is not None:
        from semantic_3d_chat.robot.state_checkpoint import load_robot_state_checkpoint

        base = _static_base(runtime)
        robot_state_encoder, robot_state_encoder_sha256, _ = (
            load_robot_state_checkpoint(
                robot_state_checkpoint,
                expected_output_dim=int(base.language.hidden_size),
                device=base.language.device,
                audit=audit,
            )
        )
    refresher = QuestionIndependentPrefixRefresher(
        runtime,
        base_map_path=base_map,
        runtime_builder=runtime_builder,
        audit=audit,
    )
    if persistent.exists():
        refresher.load_existing(persistent)
    active_vision = vision_encoder or load_configured_dense_image_encoder(
        config,
        local_files_only=local_files_only,
    )
    updater = PersistentSemanticMapUpdater(
        config,
        scene_id,
        active_vision,
        base_map_path=base_map,
        persistent_map_path=persistent,
        commit_participant=refresher,
    )
    simulator = EmbodiedCameraSimulator(config, scene_id, map_update_hook=updater.update)
    if observation_scanner is not None:
        simulator.scanner = observation_scanner
    simulator.state.scene_version = updater.current_version
    simulator.state.scan_count = updater.current_version
    simulator.state.map_sha256 = updater.current_map_sha256
    state_tokens = (
        None
        if robot_state_encoder is None
        else CheckpointBoundRobotStateTokens(
            robot_state_encoder,
            expected_sha256=str(robot_state_encoder_sha256),
        )
    )
    auto_scan_after_motion = config.get("robot", {}).get(
        "auto_scan_after_motion", False
    )
    if not isinstance(auto_scan_after_motion, bool):
        raise TypeError("robot.auto_scan_after_motion must be boolean")
    return RefreshingEmbodiedChatRuntime(
        simulator,
        updater,
        refresher,
        robot_tokens=state_tokens,
        auto_scan_after_motion=auto_scan_after_motion,
    )


__all__ = [
    "ActivePrefixBinding",
    "CheckpointBoundRobotStateTokens",
    "QuestionIndependentPrefixRefresher",
    "RefreshingEmbodiedChatRuntime",
    "ScenePrefixBinding",
    "build_refreshing_embodied_runtime",
    "robot_state_encoder_sha256",
]

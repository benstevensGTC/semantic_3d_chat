"""Explicit 738-scene-token plus 4-robot-token V96 action bridge.

The accepted V3 controller was trained with a 258-token scene sequence.  Its
learned layers are sequence-length agnostic: every scene token is projected to
keys and values, every token participates in softmax attention, and every token
also enters the global mean branch.  This module reuses those exact frozen
weights over V96's 738-token memory and records that fact as a *sequence-length
transfer*, not as V96 navigation training or held-out navigation evidence.

Environmental semantics still reach the action path only through continuous
scene/map features.  The target phrase is copied from the user's navigation
instruction and grounded against every active voxel, as in the V3 policy.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.v96_strict_multiscene_runtime import (
    PROMOTED_DECISION,
    V96StrictMultisceneChatRuntime,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.action_context import capture_continuous_action_context
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal
from semantic_3d_chat.robot.navigation_policy import split_active_prefix
from semantic_3d_chat.robot.navigation_policy_v3 import (
    GroundedContinuousNavigationControllerV3,
    load_navigation_policy_v3_checkpoint,
)
from semantic_3d_chat.robot.navigation_policy_v3_3 import (
    SemanticGroundedActionBackendV33,
)
from semantic_3d_chat.robot.semantic_agent import ContinuousTextEncoder

SOURCE_SCENE_TOKEN_COUNT: Final[int] = 258
V96_SCENE_TOKEN_COUNT: Final[int] = 738
ROBOT_TOKEN_COUNT: Final[int] = 4
HIDDEN_SIZE: Final[int] = 1536
ACTIVE_TOKEN_COUNT: Final[int] = V96_SCENE_TOKEN_COUNT + ROBOT_TOKEN_COUNT
TRANSFER_MODE: Final[str] = (
    "frozen_v3_sequence_length_transfer_258_to_738_not_retrained_on_v96"
)
V3_POLICY_WEIGHTS_SHA256: Final[str] = (
    "975c7c6c5e103dd1bb055feb2eceff6cc7fe9ab82a3f7f492a8fbdb5a26cc87c"
)
V3_POLICY_METADATA_SHA256: Final[str] = (
    "1350fb92e667a793088c7f4e4a3063f3aad0b218f50fff10c418ae850e5cc6d8"
)
V3_TRAINING_DATASET_SHA256: Final[str] = (
    "d8d97ac248a5821eb971301efb742c25c996627bae22d6417c02755e61d50f9d"
)
V3_NAVIGATION_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/checkpoints/navigation_policy_v3"
)


@dataclass(frozen=True)
class V96NavigationTransferContract:
    """Truthful immutable description of the downstream weight transfer."""

    source_scene_token_count: int
    target_scene_token_count: int
    robot_token_count: int
    hidden_size: int
    source_weights_sha256: str
    source_metadata_sha256: str
    source_training_dataset_sha256: str
    source_training_status: str
    transfer_mode: str = TRANSFER_MODE

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_scene_token_count": self.source_scene_token_count,
            "target_scene_token_count": self.target_scene_token_count,
            "robot_token_count": self.robot_token_count,
            "active_token_count": self.target_scene_token_count
            + self.robot_token_count,
            "hidden_size": self.hidden_size,
            "source_weights_sha256": self.source_weights_sha256,
            "source_metadata_sha256": self.source_metadata_sha256,
            "source_training_dataset_sha256": self.source_training_dataset_sha256,
            "source_training_status": self.source_training_status,
            "transfer_mode": self.transfer_mode,
            "weights_changed": False,
            "retrained_on_v96": False,
            "held_out_navigation_claim": False,
            "every_scene_token_processed_by_attention": True,
            "every_scene_token_processed_by_global_mean": True,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }


class V96SequenceLengthTransferredController(nn.Module):
    """Shape-attested wrapper around the exact frozen V3 controller."""

    def __init__(
        self,
        source_controller: GroundedContinuousNavigationControllerV3,
        contract: V96NavigationTransferContract,
    ) -> None:
        super().__init__()
        if type(source_controller) is not GroundedContinuousNavigationControllerV3:
            raise TypeError("V96 transfer requires the exact V3 controller class")
        if (
            contract.source_scene_token_count != SOURCE_SCENE_TOKEN_COUNT
            or contract.target_scene_token_count != V96_SCENE_TOKEN_COUNT
            or contract.robot_token_count != ROBOT_TOKEN_COUNT
            or contract.hidden_size != HIDDEN_SIZE
            or source_controller.hidden_size != HIDDEN_SIZE
        ):
            raise ValueError("V96 navigation transfer contract changed")
        self.source_controller = source_controller.eval()
        self.contract = contract
        self.hidden_size = source_controller.hidden_size
        self.model_dim = source_controller.model_dim
        self.forward_calls = 0
        self.last_forward_audit: dict[str, Any] | None = None
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(
        self,
        scene_prefix: torch.Tensor,
        robot_tokens: torch.Tensor,
        instruction_embedding: torch.Tensor,
        target_state: torch.Tensor,
        *,
        scene_batch_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(scene_prefix.shape) != (1, V96_SCENE_TOKEN_COUNT, HIDDEN_SIZE):
            raise ValueError("V96 action scene input must be exactly [1,738,1536]")
        if tuple(robot_tokens.shape) != (1, ROBOT_TOKEN_COUNT, HIDDEN_SIZE):
            raise ValueError("V96 action robot input must be exactly [1,4,1536]")
        if not torch.isfinite(scene_prefix.float()).all() or not torch.isfinite(
            robot_tokens.float()
        ).all():
            raise ValueError("V96 action context contains NaN or infinity")
        if scene_batch_indices is not None:
            raise ValueError("V96 live action inference does not accept batch remapping")

        scene_hash = prefix_sha256(scene_prefix)
        robot_hash = prefix_sha256(robot_tokens)
        logits, arguments = self.source_controller(
            scene_prefix,
            robot_tokens,
            instruction_embedding,
            target_state,
        )
        self.forward_calls += 1
        self.last_forward_audit = {
            "schema": "semantic_3d_chat.v96_navigation_transfer_forward.v1",
            "forward_call": self.forward_calls,
            "scene_shape": list(scene_prefix.shape),
            "robot_shape": list(robot_tokens.shape),
            "scene_prefix_sha256": scene_hash,
            "robot_tokens_sha256": robot_hash,
            "scene_tokens_processed": V96_SCENE_TOKEN_COUNT,
            "robot_tokens_processed": ROBOT_TOKEN_COUNT,
            "hidden_size": HIDDEN_SIZE,
            "all_scene_tokens_enter_attention_keys_and_values": True,
            "all_scene_tokens_enter_global_mean": True,
            "robot_tokens_enter_robot_value_mean": True,
            "question_dependent_scene_selection": False,
            "top_k_scene_selection": False,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
            "transfer": self.contract.as_dict(),
        }
        return logits, arguments


def _promoted_runtime(runtime: Any) -> V96StrictMultisceneChatRuntime:
    refresher = getattr(runtime, "prefix_refresher", None)
    wrapped = getattr(refresher, "runtime", None)
    if not isinstance(wrapped, V96StrictMultisceneChatRuntime):
        raise TypeError("V96 action bridge requires a promoted strict chat runtime")
    wrapped.assert_prefix_unchanged()
    if (
        wrapped.runtime_package_mode != "promoted"
        or wrapped.runtime_promotion_authorized is not True
        or wrapped.release_provenance.get("promotion_decision") != PROMOTED_DECISION
    ):
        raise ValueError("V96 action bridge refuses an unpromoted static runtime")
    return wrapped


def _grounding_coverage_audit(
    grounding: object,
    *,
    available_voxels: object,
    map_sha256: object,
) -> dict[str, Any]:
    """Report all-voxel grounding only when the live grounding proves it.

    Instructions such as ``scan`` and ``stop`` legitimately have no semantic
    target.  In that case the audit must say that grounding was not used; a
    missing grounding object is never evidence that all voxels were scored.
    """

    absent = {
        "target_grounding_used": False,
        "all_active_map_voxels_scored_for_target_grounding": None,
        "grounding_scored_voxels": None,
    }
    if grounding is None:
        return absent
    if not isinstance(grounding, Mapping):
        raise TypeError("V96 grounding audit is not a mapping")
    target_available = grounding.get("target_available")
    if target_available is False:
        return absent
    if target_available is not True:
        raise RuntimeError("V96 grounding audit omitted target availability")
    scored = grounding.get("scored_voxels")
    if (
        isinstance(scored, bool)
        or not isinstance(scored, int)
        or isinstance(available_voxels, bool)
        or not isinstance(available_voxels, int)
        or available_voxels < 1
        or scored != available_voxels
        or grounding.get("map_sha256") != map_sha256
    ):
        raise RuntimeError("V96 target grounding did not prove complete active-map coverage")
    return {
        "target_grounding_used": True,
        "all_active_map_voxels_scored_for_target_grounding": True,
        "grounding_scored_voxels": scored,
    }


def _validate_source_metadata(metadata: Mapping[str, Any]) -> None:
    if (
        metadata.get("scene_token_count") != SOURCE_SCENE_TOKEN_COUNT
        or metadata.get("robot_token_count") != ROBOT_TOKEN_COUNT
        or metadata.get("hidden_size") != HIDDEN_SIZE
        or metadata.get("task_trained") is not True
        or metadata.get("every_scene_token_processed") is not True
        or metadata.get("numeric_robot_tokens_required") is not True
        or metadata.get("all_map_voxels_scored_for_grounding") is not True
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("oracle_inputs_at_runtime") is not False
        or metadata.get("weights_sha256") != V3_POLICY_WEIGHTS_SHA256
        or metadata.get("training_dataset_sha256") != V3_TRAINING_DATASET_SHA256
        or metadata.get("train_scene_count") != 14
        or metadata.get("validation_scene_count") != 8
    ):
        raise ValueError("Source V3 navigation checkpoint contract changed")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v96_sequence_length_transfer(
    checkpoint: str | Path,
    *,
    expected_model_id: str,
    expected_model_revision: str,
    device: torch.device | str,
    audit: FileAccessAudit | None = None,
) -> tuple[
    V96SequenceLengthTransferredController,
    dict[str, Any],
    V96NavigationTransferContract,
]:
    """Load exact V3 weights, then change only the input-length contract."""

    candidate = Path(checkpoint).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    source_root = Path(os.path.abspath(rooted))
    canonical_root = Path(os.path.abspath(V3_NAVIGATION_CHECKPOINT))
    current = Path(source_root.anchor)
    for component in source_root.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("V96 navigation checkpoint path cannot contain symbolic links")
    weights_path = source_root / "policy.safetensors"
    metadata_path = source_root / "runtime_metadata.json"
    if (
        source_root != canonical_root
        or
        source_root.is_symlink()
        or not source_root.is_dir()
        or {entry.name for entry in source_root.iterdir()}
        != {"policy.safetensors", "runtime_metadata.json"}
        or any(path.is_symlink() or not path.is_file() for path in (weights_path, metadata_path))
        or _sha256_file(weights_path) != V3_POLICY_WEIGHTS_SHA256
        or _sha256_file(metadata_path) != V3_POLICY_METADATA_SHA256
    ):
        raise ValueError("V96 requires the exact immutable V3 navigation checkpoint")

    source, source_metadata = load_navigation_policy_v3_checkpoint(
        source_root,
        expected_hidden_size=HIDDEN_SIZE,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
        device=device,
        audit=audit,
    )
    _validate_source_metadata(source_metadata)
    contract = V96NavigationTransferContract(
        source_scene_token_count=SOURCE_SCENE_TOKEN_COUNT,
        target_scene_token_count=V96_SCENE_TOKEN_COUNT,
        robot_token_count=ROBOT_TOKEN_COUNT,
        hidden_size=HIDDEN_SIZE,
        source_weights_sha256=str(source_metadata["weights_sha256"]),
        source_metadata_sha256=V3_POLICY_METADATA_SHA256,
        source_training_dataset_sha256=V3_TRAINING_DATASET_SHA256,
        source_training_status=(
            "supervised_continuous_semantic_grounded_navigation_policy_v3"
        ),
    )
    transferred = V96SequenceLengthTransferredController(source, contract)
    # V3 metadata has an exact schema.  Keep the source object untouched and
    # create the exact schema consumed by the inherited runtime backend.
    bridge_metadata = dict(source_metadata)
    bridge_metadata["scene_token_count"] = V96_SCENE_TOKEN_COUNT
    return transferred.to(device).eval(), bridge_metadata, contract


class V96ReleaseSemanticGroundedActionBackend(
    SemanticGroundedActionBackendV33
):
    """V3.3 navigation with an authenticated complete V96 action prefix."""

    def __init__(
        self,
        runtime: Any,
        controller: V96SequenceLengthTransferredController,
        metadata: Mapping[str, Any],
        config: Mapping[str, Any],
        *,
        text_encoder: ContinuousTextEncoder | None = None,
    ) -> None:
        _promoted_runtime(runtime)
        if not isinstance(controller, V96SequenceLengthTransferredController):
            raise TypeError("V96 action backend requires the transfer wrapper")
        super().__init__(
            runtime,
            controller,
            dict(metadata),
            dict(config),
            text_encoder=text_encoder,
        )
        self.transfer_controller = controller
        self.transfer_contract = controller.contract
        self.last_v96_context_audit: dict[str, Any] | None = None
        self._require_active_layout()

    def _require_active_layout(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        context = capture_continuous_action_context(
            self.runtime,
            self.metadata["room_size_m"],
        )
        if tuple(context.active_prefix.shape) != (1, ACTIVE_TOKEN_COUNT, HIDDEN_SIZE):
            raise ValueError("V96 active action prefix must be exactly [1,742,1536]")
        scene, robot = split_active_prefix(
            context.active_prefix,
            scene_token_count=V96_SCENE_TOKEN_COUNT,
            robot_token_count=ROBOT_TOKEN_COUNT,
        )
        binding = context.binding
        if (
            prefix_sha256(scene) != binding.get("scene_control_signature_sha256")
            or prefix_sha256(robot) != binding.get("robot_tokens_sha256")
        ):
            raise RuntimeError("V96 738+4 split differs from its runtime binding")
        return scene, robot, binding

    @torch.inference_mode()
    def generate(
        self,
        instruction: str,
        *,
        correction_code: str | None,
    ) -> GeneratedToolProposal:
        scene, robot, before = self._require_active_layout()
        call_count = self.transfer_controller.forward_calls
        proposal = super().generate(
            instruction,
            correction_code=correction_code,
        )
        _scene_after, _robot_after, after = self._require_active_layout()
        forward = self.transfer_controller.last_forward_audit
        if (
            self.transfer_controller.forward_calls != call_count + 1
            or not isinstance(forward, Mapping)
            or forward.get("scene_prefix_sha256") != prefix_sha256(scene)
            or forward.get("robot_tokens_sha256") != prefix_sha256(robot)
            or proposal.active_prefix_sha256 != before.get("active_prefix_sha256")
            or before.get("active_prefix_sha256") != after.get("active_prefix_sha256")
        ):
            raise RuntimeError("V96 action proposal is not bound to the 738+4 forward pass")
        grounding_audit = _grounding_coverage_audit(
            self.last_grounding,
            available_voxels=before.get("source_voxels"),
            map_sha256=before.get("map_sha256"),
        )
        self.last_v96_context_audit = {
            "schema": "semantic_3d_chat.v96_release_action_context.v1",
            "active_prefix_shape": [1, ACTIVE_TOKEN_COUNT, HIDDEN_SIZE],
            "active_prefix_sha256": before["active_prefix_sha256"],
            "full_scene_memory_sha256": before["scene_control_signature_sha256"],
            "base_scene_prefix_sha256": before["scene_prefix_sha256"],
            "robot_tokens_sha256": before["robot_tokens_sha256"],
            "map_sha256": before["map_sha256"],
            "scene_tokens_consumed": V96_SCENE_TOKEN_COUNT,
            "robot_tokens_consumed": ROBOT_TOKEN_COUNT,
            "policy_consumed_738_scene_tokens": True,
            "policy_consumed_4_robot_tokens": True,
            "complete_scene_memory_used": True,
            "question_dependent_scene_retrieval": False,
            **grounding_audit,
            "source_policy_was_retrained_on_v96": False,
            "transfer": self.transfer_contract.as_dict(),
            "forward": dict(forward),
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }
        # Preserve the accepted V3 training-status vocabulary used by the
        # strict validator; the separate transfer audit prevents overclaiming.
        return replace(
            proposal,
            training_status=self.transfer_contract.source_training_status,
        )


def build_v96_release_action_backend(
    runtime: Any,
    config: Mapping[str, Any],
    *,
    navigation_checkpoint: str | Path,
    text_encoder: ContinuousTextEncoder | None = None,
    audit: FileAccessAudit | None = None,
) -> V96ReleaseSemanticGroundedActionBackend:
    """Attach the exact V3 weights to one already-promoted V96 robot runtime."""

    promoted = _promoted_runtime(runtime)
    language = promoted.base.language
    controller, metadata, _contract = load_v96_sequence_length_transfer(
        navigation_checkpoint,
        expected_model_id=str(promoted.config["language"]["model_id"]),
        expected_model_revision=str(promoted.config["language"]["revision"]),
        device=language.device,
        audit=audit,
    )
    return V96ReleaseSemanticGroundedActionBackend(
        runtime,
        controller,
        metadata,
        config,
        text_encoder=text_encoder,
    )


__all__ = [
    "ACTIVE_TOKEN_COUNT",
    "HIDDEN_SIZE",
    "ROBOT_TOKEN_COUNT",
    "SOURCE_SCENE_TOKEN_COUNT",
    "TRANSFER_MODE",
    "V3_NAVIGATION_CHECKPOINT",
    "V3_POLICY_METADATA_SHA256",
    "V3_POLICY_WEIGHTS_SHA256",
    "V3_TRAINING_DATASET_SHA256",
    "V96_SCENE_TOKEN_COUNT",
    "V96NavigationTransferContract",
    "V96ReleaseSemanticGroundedActionBackend",
    "V96SequenceLengthTransferredController",
    "build_v96_release_action_backend",
    "load_v96_sequence_length_transfer",
]

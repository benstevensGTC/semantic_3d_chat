"""Runtime backend for the trained continuous-context Gemma-4 V2 decoder."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_PROJECTED_DIM,
    GEMMA4_PROJECTED_START,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    NumericToolContextProjectorV2,
    prepare_tool_decoder_inputs,
    tool_decoder_system_prompt,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2_checkpoint import TRAINING_STATUS
from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.action_context import (
    ContinuousActionContext,
    capture_continuous_action_context,
    require_grounding_map_binding,
)
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal
from semantic_3d_chat.robot.navigation_policy_v3 import (
    grounded_target_state,
    target_text_from_navigation_instruction,
)
from semantic_3d_chat.robot.navigation_policy_v4 import robot_frame_clearance_state
from semantic_3d_chat.robot.semantic_agent import (
    ContinuousSemanticGrounding,
    ContinuousSemanticTargetGrounder,
    ContinuousTextEncoder,
    GemmaProjectedTextEncoder,
)

_BLOCKED = frozenset({"oracle", "qa", "training", "scorer_only"})


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _active_map_path(runtime: Any) -> Path:
    updater = getattr(runtime, "map_updater", None)
    if updater is None:
        raise TypeError("V2 Gemma runtime has no semantic map updater")
    persistent = Path(updater.persistent_map_path)
    base = Path(updater.base_map_path)
    selected = _rooted(persistent if persistent.is_file() else base)
    if _BLOCKED & {part.casefold() for part in selected.parts}:
        raise ValueError("V2 Gemma runtime cannot read a blocked data tree")
    current = Path(selected.anchor)
    for component in selected.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("V2 Gemma runtime map paths cannot contain symlinks")
    if not selected.is_file():
        raise FileNotFoundError("V2 Gemma active semantic map is unavailable")
    return selected


def _literal_instruction(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("V2 Gemma navigation instruction must be text")
    prefix = "User navigation instruction: "
    stripped = value.strip()
    first = stripped.splitlines()[0] if stripped else ""
    literal = first[len(prefix) :].strip() if first.startswith(prefix) else stripped
    if not literal or len(literal) > 1024:
        raise ValueError("V2 Gemma navigation instruction is empty or too long")
    return literal


class ContinuousGemmaToolDecoderBackendV2:
    """Decode JSON from scene, robot, grounded-target, and free-space tokens."""

    def __init__(
        self,
        runtime: Any,
        projector: NumericToolContextProjectorV2,
        checkpoint_metadata: Mapping[str, Any],
        config: Mapping[str, Any],
        *,
        text_encoder: ContinuousTextEncoder | None = None,
        max_new_tokens: int = 24,
        allow_staged_runtime_probe: bool = False,
    ) -> None:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("V2 Gemma max_new_tokens must be an integer")
        if not 1 <= max_new_tokens <= 128:
            raise ValueError("V2 Gemma max_new_tokens must be in [1,128]")
        if checkpoint_metadata.get("max_new_tokens") != max_new_tokens:
            raise ValueError("V2 Gemma generation bound differs from its checkpoint")
        if not isinstance(allow_staged_runtime_probe, bool):
            raise TypeError("V2 staged-runtime-probe flag must be a boolean")
        if checkpoint_metadata.get("training_status") != TRAINING_STATUS:
            raise ValueError("V2 Gemma backend requires its promoted training status")
        promoted = bool(
            checkpoint_metadata.get("status") == "promoted_runtime"
            and checkpoint_metadata.get("promotion_gates_passed") is True
            and checkpoint_metadata.get("saved_runtime_execution_gate_passed") is True
        )
        staged_probe = bool(
            allow_staged_runtime_probe
            and checkpoint_metadata.get("status") == "staged_runtime_probe_only"
            and checkpoint_metadata.get("promotion_gates_passed") is False
            and checkpoint_metadata.get("saved_runtime_execution_gate_passed") is False
        )
        if (
            not (promoted or staged_probe)
            or checkpoint_metadata.get("environmental_text_inputs") != []
            or checkpoint_metadata.get("oracle_inputs_at_runtime") is not False
        ):
            raise ValueError("V2 Gemma checkpoint is not promoted or inference safe")
        if not callable(getattr(runtime, "active_prefix_snapshot", None)):
            raise TypeError("V2 Gemma runtime lacks an active-prefix snapshot")
        prefix_refresher = getattr(runtime, "prefix_refresher", None)
        wrapped = getattr(prefix_refresher, "runtime", None)
        self.base = getattr(wrapped, "base", wrapped)
        if self.base is None or getattr(self.base, "language", None) is None:
            raise TypeError("V2 Gemma runtime lacks a loaded local language backend")
        self.language = self.base.language
        if (
            self.language.backend_name != "gemma4"
            or self.language.prefix_backend is None
            or self.language.hidden_size != 1536
        ):
            raise ValueError("V2 backend requires pinned local Gemma-4 E2B")
        self.runtime = runtime
        self.projector = projector.eval().to(self.language.device)
        self.metadata = dict(checkpoint_metadata)
        self.config = dict(config)
        self.max_new_tokens = max_new_tokens
        self.text_encoder = text_encoder or GemmaProjectedTextEncoder.from_config(config)
        if self.text_encoder.output_dim != GEMMA4_PROJECTED_DIM:
            raise ValueError("V2 target-grounding text encoder width changed")
        robot = config.get("robot")
        scene = config.get("scene")
        if not isinstance(robot, Mapping) or not isinstance(scene, Mapping):
            raise TypeError("V2 runtime config has no scene or robot mapping")
        self.max_turn_degrees = float(robot["max_turn_degrees"])
        self.max_move_m = float(robot["max_move_m"])
        self.room_size_m = [float(value) for value in scene["room_size_m"]]
        if (
            len(self.room_size_m) != 3
            or any(not math.isfinite(value) or value <= 0.0 for value in self.room_size_m)
        ):
            raise ValueError("V2 runtime room dimensions are invalid")
        self.last_context: dict[str, Any] | None = None

    def _ground(
        self,
        target_text: str | None,
        state_features: torch.Tensor,
        *,
        context: ContinuousActionContext | None = None,
    ) -> tuple[torch.Tensor, ContinuousSemanticGrounding | None]:
        if target_text is None:
            return (
                grounded_target_state(
                    torch.zeros(3),
                    state_features,
                    torch.tensor(0.0),
                    room_size_m=self.room_size_m,
                ),
                None,
            )
        grounder = ContinuousSemanticTargetGrounder(
            _active_map_path(self.runtime),
            self.text_encoder,
            room_size_m=self.room_size_m,
            feature_start=GEMMA4_PROJECTED_START,
            feature_dim=GEMMA4_PROJECTED_DIM,
        )
        grounding = grounder.ground(target_text)
        if context is None:
            if grounding.scored_voxels != len(grounder.xyz):
                raise RuntimeError("V2 target grounding did not score every active voxel")
        else:
            require_grounding_map_binding(
                context,
                grounding_map_sha256=grounding.map_sha256,
                scored_voxels=grounding.scored_voxels,
                available_voxels=len(grounder.xyz),
            )
        if grounder.scene_id != self.runtime.simulator.state.scene_id:
            raise RuntimeError("V2 target-grounding map differs from robot scene")
        return (
            grounded_target_state(
                torch.tensor(grounding.target_xyz_m),
                state_features,
                torch.tensor(1.0),
                room_size_m=self.room_size_m,
            ),
            grounding,
        )

    @torch.inference_mode()
    def generate(
        self,
        instruction: str,
        *,
        correction_code: str | None,
    ) -> GeneratedToolProposal:
        context = capture_continuous_action_context(
            self.runtime,
            self.room_size_m,
        )
        active = context.active_prefix
        binding = context.binding
        if not isinstance(active, torch.Tensor) or active.shape != (1, 262, 1536):
            raise RuntimeError("V2 runtime active prefix shape changed")
        if not torch.isfinite(active.float()).all():
            raise RuntimeError("V2 runtime active prefix contains NaN or infinity")
        active_digest = prefix_sha256(active)
        if binding.get("active_prefix_sha256") != active_digest:
            raise RuntimeError("V2 active prefix differs from its runtime binding")
        literal = _literal_instruction(instruction)
        target_text = target_text_from_navigation_instruction(literal)
        state_features = context.state_features
        target_state, grounding = self._ground(
            target_text,
            state_features,
            context=context,
        )
        simulator = self.runtime.simulator
        clearance = robot_frame_clearance_state(
            simulator.collision_map,
            context.numeric_state.position_m[:2],
            context.numeric_state.body_yaw_degrees,
            ray_count=24,
            max_range_m=1.0,
        ).unsqueeze(0)
        retry = (
            ""
            if correction_code is None
            else (
                "\nThe previous JSON proposal failed protocol validation with code "
                f"{correction_code}. Return a fresh valid JSON object."
            )
        )
        prompt = prompt_token_ids(
            self.language.tokenizer,
            tool_decoder_system_prompt(
                max_turn_degrees=self.max_turn_degrees,
                max_move_m=self.max_move_m,
            ),
            literal + retry,
            self.language.device,
        )
        prepared = prepare_tool_decoder_inputs(
            self.language.prefix_backend,
            active.to(self.language.device),
            prompt,
            self.projector,
            target_state.to(self.language.device),
            clearance.to(self.language.device),
        )
        generated = self.language.prefix_backend.generate(
            prepared,
            max_new_tokens=self.max_new_tokens,
            eos_token_ids=self.language.tokenizer.eos_token_id,
        )
        text = self.language.tokenizer.decode(
            generated[0].detach().cpu().tolist(), skip_special_tokens=True
        ).strip()
        target_digest = hashlib.sha256(
            target_state.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        clearance_digest = hashlib.sha256(
            clearance.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        self.last_context = {
            "target_state_sha256": target_digest,
            "clearance_state_sha256": clearance_digest,
            "target_available": grounding is not None,
            "scored_voxels": None if grounding is None else grounding.scored_voxels,
            "active_prefix_sha256": active_digest,
            "scene_prefix_sha256": binding["scene_prefix_sha256"],
            "map_sha256": binding["map_sha256"],
            "robot_state_sha256": binding["robot_state_sha256"],
            "robot_tokens_sha256": binding["robot_tokens_sha256"],
            "continuous_context_verified": True,
            "environmental_text_inputs": [],
            "oracle_inputs_loaded": False,
        }
        scene_hash = binding.get("scene_prefix_sha256")
        robot_hash = binding.get("robot_tokens_sha256")
        return GeneratedToolProposal(
            text=text,
            active_prefix_sha256=active_digest,
            scene_prefix_sha256=scene_hash if isinstance(scene_hash, str) else "",
            robot_tokens_sha256=robot_hash if isinstance(robot_hash, str) else None,
            local_inference=True,
            used_continuous_scene_prefix=True,
            used_continuous_robot_tokens=True,
            training_status=TRAINING_STATUS,
        )


__all__ = ["ContinuousGemmaToolDecoderBackendV2"]

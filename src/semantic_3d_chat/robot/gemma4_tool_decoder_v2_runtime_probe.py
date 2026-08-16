"""Saved-checkpoint execution probe for the continuous Gemma tool decoder.

The probe is intentionally in-process: the already resident frozen Gemma model
is reused after training, while the just-written two-file checkpoint is loaded
strictly back into its LoRA/projector surfaces.  This prevents a second 4.6B
parameter model load and still proves that the serialized bytes can drive the
real runtime decoder, validator, and numerical simulator interlock.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import numpy as np
import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma4_semantic_sanity import GEMMA4_PROJECTED_DIM
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import NumericToolContextProjectorV2
from semantic_3d_chat.language.gemma4_tool_decoder_v2_checkpoint import (
    load_runtime_checkpoint_v2,
)
from semantic_3d_chat.language.lora import LoRAInstallation
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.gemma4_tool_decoder_v2_backend import (
    ContinuousGemmaToolDecoderBackendV2,
)
from semantic_3d_chat.robot.llm_tool_policy import (
    execute_validated_tool_call,
    validate_tool_call_text,
)
from semantic_3d_chat.robot.navigation_policy_v4 import robot_frame_clearance_state
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator
from semantic_3d_chat.robot.state_encoder import (
    insert_robot_state_tokens,
    robot_state_vector,
)
from semantic_3d_chat.training.gemma4_tool_decoder_v2_data import (
    ToolDecoderDatasetV2,
    ToolDecoderSampleV2,
)

PROBE_SAMPLE_ID: Final[str] = "g_00004208"
PROBE_SCENE_ID: Final[str] = "scene_000031"
PROBE_ACTION: Final[str] = "stop"
PROBE_INSTRUCTION: Final[str] = "Move forward 0.25 meters, then stop."


class _NoTargetTextEncoder:
    """Fail closed if the no-target saved-runtime probe unexpectedly grounds text."""

    output_dim = GEMMA4_PROJECTED_DIM

    def encode(self, _text: str) -> torch.Tensor:
        raise RuntimeError("V2 no-target runtime probe unexpectedly requested text grounding")


class _ProbeRuntime:
    """Small production-interface adapter around the real numerical simulator."""

    def __init__(
        self,
        *,
        language: Any,
        simulator: EmbodiedCameraSimulator,
        scene_prefix: torch.Tensor,
        robot_tokens: torch.Tensor,
    ) -> None:
        if scene_prefix.shape != (1, 258, 1536):
            raise ValueError("V2 probe scene prefix shape changed")
        if robot_tokens.shape != (1, 4, 1536):
            raise ValueError("V2 probe robot-token shape changed")
        self.simulator = simulator
        self._scene_prefix = scene_prefix.detach().cpu().contiguous()
        self._robot_tokens = robot_tokens.detach().cpu().contiguous()
        self._active_prefix = insert_robot_state_tokens(
            self._scene_prefix, self._robot_tokens
        ).contiguous()
        self._binding = {
            "active_prefix_sha256": prefix_sha256(self._active_prefix),
            "scene_prefix_sha256": prefix_sha256(self._scene_prefix),
            "robot_tokens_sha256": prefix_sha256(self._robot_tokens),
        }
        self.prefix_refresher = SimpleNamespace(
            runtime=SimpleNamespace(base=SimpleNamespace(language=language))
        )

    def active_prefix_snapshot(self) -> tuple[torch.Tensor, dict[str, str]]:
        return self._active_prefix.clone(), dict(self._binding)

    def get_robot_state(self) -> dict[str, Any]:
        return self.simulator.get_robot_state()

    def look(self, yaw_delta_degrees: Any, pitch_delta_degrees: Any) -> dict[str, Any]:
        return self.simulator.look(yaw_delta_degrees, pitch_delta_degrees)

    def turn(self, angle_degrees: Any) -> dict[str, Any]:
        return self.simulator.turn(angle_degrees)

    def move_forward(self, distance_meters: Any) -> dict[str, Any]:
        return self.simulator.move_forward(distance_meters)

    def move_backward(self, distance_meters: Any) -> dict[str, Any]:
        return self.simulator.move_backward(distance_meters)

    def move_to(self, x: Any, y: Any) -> dict[str, Any]:
        return self.simulator.move_to(x, y)

    def scan(self) -> dict[str, Any]:
        return self.simulator.scan()

    def stop(self) -> dict[str, Any]:
        return self.simulator.stop()

    def reset_scene(self, scene_id: str, seed: Any) -> dict[str, Any]:
        return self.simulator.reset_scene(scene_id, seed)


def _probe_sample(dataset: ToolDecoderDatasetV2) -> ToolDecoderSampleV2:
    matches = [sample for sample in dataset.samples if sample.sample_id == PROBE_SAMPLE_ID]
    if len(matches) != 1:
        raise ValueError("V2 saved-runtime probe sample identity changed")
    sample = matches[0]
    if (
        sample.scene_id != PROBE_SCENE_ID
        or sample.split != "validation"
        or sample.family != "stop"
        or sample.action_name != PROBE_ACTION
        or sample.instruction != PROBE_INSTRUCTION
        or torch.count_nonzero(sample.target_state).item() != 0
        or sample.canonical_answer != '{"arguments":{},"tool":"stop"}'
    ):
        raise ValueError("V2 saved-runtime probe sample contract changed")
    return sample


def _degrees(sine: float, cosine: float) -> float:
    if not math.isfinite(sine) or not math.isfinite(cosine):
        raise ValueError("V2 probe orientation contains NaN or infinity")
    return math.degrees(math.atan2(sine, cosine))


def restore_simulator_state_v2(
    simulator: EmbodiedCameraSimulator,
    state_features: torch.Tensor,
    config: Mapping[str, Any],
) -> None:
    """Invert the public 18-value numeric state encoding for one held-out row."""

    values = state_features.detach().float().cpu()
    if values.shape != (18,) or not torch.isfinite(values).all():
        raise ValueError("V2 probe state features are invalid")
    room = config.get("scene", {}).get("room_size_m")
    if not isinstance(room, (list, tuple)) or len(room) != 3:
        raise ValueError("V2 probe room dimensions are unavailable")
    span = np.asarray(room, dtype=np.float64)
    if not np.isfinite(span).all() or np.any(span <= 0.0):
        raise ValueError("V2 probe room dimensions are invalid")
    minimum = np.asarray([-span[0] / 2.0, -span[1] / 2.0, 0.0])
    raw = values.numpy().astype(np.float64)
    position = (raw[:3] + 1.0) * 0.5 * span + minimum
    body_yaw = _degrees(raw[3], raw[4])
    camera_yaw = _degrees(raw[5], raw[6])
    camera_offset = (camera_yaw - body_yaw + 180.0) % 360.0 - 180.0
    state = simulator.state
    state.position_xy_m = position[:2].copy()
    state.body_yaw_degrees = body_yaw
    state.camera_yaw_offset_degrees = camera_offset
    state.pitch_degrees = _degrees(raw[7], raw[8])
    state.linear_velocity_xy_m = raw[9:11].copy()
    state.angular_velocity_degrees = float(raw[11] * 180.0)
    state.collision = bool(raw[12] >= 0.5)
    state.last_movement_delta_m = (raw[13:16] * span).copy()
    state.scan_coverage = float(raw[16])
    state.stopped = bool(raw[17] >= 0.5)
    minimum_tensor = torch.tensor(minimum, dtype=torch.float32)
    maximum_tensor = torch.tensor(minimum + span, dtype=torch.float32)
    restored = robot_state_vector(
        simulator.numeric_state(), minimum_tensor, maximum_tensor
    )
    if not torch.allclose(restored, values, atol=2e-6, rtol=2e-6):
        maximum_error = float((restored - values).abs().max())
        raise RuntimeError(
            f"V2 probe could not restore the held-out numeric state: {maximum_error}"
        )
    collision = simulator.collision_map.point_check(state.position_xy_m)
    if collision.collision:
        raise RuntimeError("V2 probe held-out pose intersects simulator geometry")


def numeric_robot_state_for_gate_v2(simulator: EmbodiedCameraSimulator) -> dict[str, Any]:
    """Return only finite numeric values; scene identifiers stay outside the gate."""

    state = simulator.numeric_state()
    return {
        "position_m": list(state.position_m),
        "body_yaw_degrees": state.body_yaw_degrees,
        "camera_yaw_degrees": state.camera_yaw_degrees,
        "pitch_degrees": state.pitch_degrees,
        "linear_velocity_xy_m": list(state.linear_velocity_xy_m),
        "angular_velocity_degrees": state.angular_velocity_degrees,
        "collision": state.collision,
        "last_movement_delta_m": list(state.last_movement_delta_m),
        "scan_coverage": state.scan_coverage,
        "stopped": state.stopped,
    }


def build_saved_runtime_probe_v2(
    *,
    language: Any,
    installation: LoRAInstallation,
    projector: NumericToolContextProjectorV2,
    dataset: ToolDecoderDatasetV2,
    provenance: Mapping[str, str],
    config: Mapping[str, Any],
) -> Callable[[Path], Mapping[str, Any]]:
    """Build the one-shot callback consumed by atomic checkpoint publication."""

    sample = _probe_sample(dataset)
    scene_prefix = dataset.prefixes[sample.scene_id].detach().cpu().contiguous()
    robot_tokens = sample.robot_tokens.unsqueeze(0).detach().cpu().contiguous()
    expected_clearance = sample.clearance_state.detach().float().cpu().contiguous()
    expected_provenance = dict(provenance)

    def probe(staged_checkpoint: Path) -> Mapping[str, Any]:
        audit = FileAccessAudit(
            forbidden_roots=[
                PROJECT_ROOT / "data" / "oracle",
                PROJECT_ROOT / "data_gemma4" / "oracle",
                PROJECT_ROOT / "data_gemma4" / "qa",
                PROJECT_ROOT / "data_gemma4" / "training",
            ],
            forbidden_component_names={"oracle", "qa", "training", "scorer_only"},
            block_forbidden=True,
        )
        with audit:
            metadata = load_runtime_checkpoint_v2(
                staged_checkpoint,
                installation,
                projector,
                expected_provenance=expected_provenance,
                require_promoted=False,
            )
            simulator = EmbodiedCameraSimulator(
                dict(config), sample.scene_id, seed=int(config["seed"])
            )
            restore_simulator_state_v2(simulator, sample.state_features, config)
            observed_clearance = robot_frame_clearance_state(
                simulator.collision_map,
                simulator.state.position_xy_m,
                simulator.state.body_yaw_degrees,
                ray_count=24,
                max_range_m=1.0,
            )
            if not torch.allclose(
                observed_clearance.float(), expected_clearance, atol=1e-5, rtol=1e-5
            ):
                maximum_error = float(
                    (observed_clearance.float() - expected_clearance).abs().max()
                )
                raise RuntimeError(
                    f"V2 probe clearance differs from sealed held-out row: {maximum_error}"
                )
            runtime = _ProbeRuntime(
                language=language,
                simulator=simulator,
                scene_prefix=scene_prefix,
                robot_tokens=robot_tokens,
            )
            backend = ContinuousGemmaToolDecoderBackendV2(
                runtime,
                projector,
                metadata,
                config,
                text_encoder=_NoTargetTextEncoder(),
                max_new_tokens=24,
                allow_staged_runtime_probe=True,
            )
            proposal = backend.generate(sample.instruction, correction_code=None)
            before = numeric_robot_state_for_gate_v2(simulator)
            validation = validate_tool_call_text(
                proposal.text, config, robot_state=runtime.get_robot_state()
            )
            if validation.call is None or validation.error_code is not None:
                result: Mapping[str, Any] = {}
                attempted = False
                collision_checked = False
            else:
                result = execute_validated_tool_call(
                    runtime, validation.call, config=config
                )
                attempted = True
                collision_checked = True
        audit.assert_clean()
        return {
            "saved_checkpoint_loaded": True,
            "generated_text": proposal.text,
            "numeric_robot_state": before,
            "tool_execution_attempted": attempted,
            "tool_execution_result": dict(result),
            "collision_interlock_checked": collision_checked,
            "oracle_inputs_loaded": False,
            "environmental_text_inputs": [],
        }

    return probe


__all__ = [
    "PROBE_ACTION",
    "PROBE_INSTRUCTION",
    "PROBE_SAMPLE_ID",
    "PROBE_SCENE_ID",
    "build_saved_runtime_probe_v2",
    "numeric_robot_state_for_gate_v2",
    "restore_simulator_state_v2",
]

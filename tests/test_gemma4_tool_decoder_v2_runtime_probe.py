from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.gemma4_tool_decoder_training_authorization_v2_2 import (
    BOUND_SOURCE_PATHS,
    build_cpu_authorization_v2_2,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import NumericToolContextProjectorV2
from semantic_3d_chat.language.gemma4_tool_decoder_v2_checkpoint import TRAINING_STATUS
from semantic_3d_chat.robot.gemma4_tool_decoder_v2_backend import (
    ContinuousGemmaToolDecoderBackendV2,
)
from semantic_3d_chat.robot.gemma4_tool_decoder_v2_runtime_probe import (
    PROBE_INSTRUCTION,
    PROBE_SAMPLE_ID,
    PROBE_SCENE_ID,
    build_saved_runtime_probe_v2,
    restore_simulator_state_v2,
)
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal
from semantic_3d_chat.robot.state_encoder import NumericRobotState
from semantic_3d_chat.training.gemma4_tool_decoder_v2_data import (
    ToolDecoderDatasetV2,
    ToolDecoderSampleV2,
)


class _CollisionMap:
    def point_check(self, _position: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(collision=False, clearance_m=1.0)


class _Simulator:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.collision_map = _CollisionMap()
        self.state = SimpleNamespace(
            scene_id=PROBE_SCENE_ID,
            position_xy_m=np.zeros(2, dtype=np.float64),
            body_yaw_degrees=0.0,
            camera_yaw_offset_degrees=0.0,
            pitch_degrees=0.0,
            linear_velocity_xy_m=np.zeros(2, dtype=np.float64),
            angular_velocity_degrees=0.0,
            collision=False,
            last_movement_delta_m=np.zeros(3, dtype=np.float64),
            scan_coverage=0.0,
            stopped=False,
        )

    def numeric_state(self) -> NumericRobotState:
        state = self.state
        return NumericRobotState(
            position_m=(
                float(state.position_xy_m[0]),
                float(state.position_xy_m[1]),
                0.0,
            ),
            body_yaw_degrees=float(state.body_yaw_degrees),
            camera_yaw_degrees=float(
                state.body_yaw_degrees + state.camera_yaw_offset_degrees
            ),
            pitch_degrees=float(state.pitch_degrees),
            linear_velocity_xy_m=tuple(float(value) for value in state.linear_velocity_xy_m),
            angular_velocity_degrees=float(state.angular_velocity_degrees),
            collision=bool(state.collision),
            last_movement_delta_m=tuple(
                float(value) for value in state.last_movement_delta_m
            ),
            scan_coverage=float(state.scan_coverage),
            stopped=bool(state.stopped),
        )

    def get_robot_state(self) -> dict[str, object]:
        state = self.numeric_state()
        return {
            "success": True,
            "scene_id": PROBE_SCENE_ID,
            "position_m": list(state.position_m),
            "collision": state.collision,
            "stopped": state.stopped,
        }

    def stop(self) -> dict[str, object]:
        self.state.stopped = True
        return {
            "success": True,
            "scene_id": PROBE_SCENE_ID,
            "position_m": [
                float(self.state.position_xy_m[0]),
                float(self.state.position_xy_m[1]),
                0.0,
            ],
            "collision": False,
            "stopped": True,
        }


def _state_features() -> torch.Tensor:
    return torch.tensor(
        [
            0.0,
            0.1,
            -1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.05,
            0.0,
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )


def _dataset() -> ToolDecoderDatasetV2:
    sample = ToolDecoderSampleV2(
        sample_id=PROBE_SAMPLE_ID,
        scene_id=PROBE_SCENE_ID,
        split="validation",
        family="stop",
        instruction=PROBE_INSTRUCTION,
        action_index=0,
        action_name="stop",
        normalized_argument=0.0,
        state_features=_state_features(),
        robot_tokens=torch.zeros(4, 1536),
        target_state=torch.zeros(10),
        clearance_state=torch.linspace(0.1, 1.0, 24),
        collision_targets=torch.ones(8),
        canonical_answer='{"arguments":{},"tool":"stop"}',
    )
    return ToolDecoderDatasetV2(
        prefixes={PROBE_SCENE_ID: torch.zeros(1, 258, 1536, dtype=torch.bfloat16)},
        samples=(sample,),
        train_indices=(),
        validation_indices=(0,),
        prefix_inventory_sha256="0" * 64,
        clearance_cache_sha256="1" * 64,
        trace_rows_sha256="2" * 64,
    )


def test_restore_simulator_state_round_trips_public_numeric_encoding() -> None:
    simulator = _Simulator()
    config = load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")
    restore_simulator_state_v2(simulator, _state_features(), config)
    assert np.allclose(simulator.state.position_xy_m, [0.0, 0.25])
    assert simulator.state.body_yaw_degrees == pytest.approx(0.0)
    assert simulator.state.linear_velocity_xy_m.tolist() == pytest.approx([0.0, 0.25])
    assert simulator.state.last_movement_delta_m.tolist() == pytest.approx([0.0, 0.25, 0.0])


def test_backend_accepts_staged_metadata_only_for_explicit_private_probe() -> None:
    language = SimpleNamespace(
        backend_name="gemma4",
        prefix_backend=object(),
        hidden_size=1536,
        device=torch.device("cpu"),
    )
    runtime = SimpleNamespace(
        active_prefix_snapshot=lambda: None,
        prefix_refresher=SimpleNamespace(
            runtime=SimpleNamespace(base=SimpleNamespace(language=language))
        ),
    )
    metadata = {
        "training_status": TRAINING_STATUS,
        "status": "staged_runtime_probe_only",
        "promotion_gates_passed": False,
        "saved_runtime_execution_gate_passed": False,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "max_new_tokens": 24,
    }
    config = load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")
    text_encoder = SimpleNamespace(output_dim=1536)
    with pytest.raises(ValueError, match="not promoted"):
        ContinuousGemmaToolDecoderBackendV2(
            runtime,
            NumericToolContextProjectorV2(),
            metadata,
            config,
            text_encoder=text_encoder,
        )
    backend = ContinuousGemmaToolDecoderBackendV2(
        runtime,
        NumericToolContextProjectorV2(),
        metadata,
        config,
        text_encoder=text_encoder,
        allow_staged_runtime_probe=True,
    )
    assert backend.metadata["status"] == "staged_runtime_probe_only"


def test_default_saved_runtime_probe_strict_loads_generates_and_executes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import semantic_3d_chat.robot.gemma4_tool_decoder_v2_runtime_probe as module

    calls: dict[str, object] = {}
    staged_metadata = {
        "training_status": TRAINING_STATUS,
        "status": "staged_runtime_probe_only",
    }

    def fake_load(
        checkpoint: Path,
        _installation: object,
        _projector: object,
        *,
        expected_provenance: object,
        require_promoted: bool,
    ) -> dict[str, object]:
        calls["checkpoint"] = checkpoint
        calls["provenance"] = expected_provenance
        calls["require_promoted"] = require_promoted
        return staged_metadata

    class FakeBackend:
        def __init__(self, runtime: object, *_args: object, **kwargs: object) -> None:
            calls["runtime"] = runtime
            calls["allow_staged"] = kwargs["allow_staged_runtime_probe"]

        def generate(
            self, instruction: str, *, correction_code: str | None
        ) -> GeneratedToolProposal:
            calls["instruction"] = instruction
            calls["correction"] = correction_code
            return GeneratedToolProposal(
                text='{"arguments":{},"tool":"stop"}',
                active_prefix_sha256="a" * 64,
                scene_prefix_sha256="b" * 64,
                robot_tokens_sha256="c" * 64,
                local_inference=True,
                used_continuous_scene_prefix=True,
                used_continuous_robot_tokens=True,
                training_status=TRAINING_STATUS,
            )

    monkeypatch.setattr(module, "load_runtime_checkpoint_v2", fake_load)
    monkeypatch.setattr(module, "EmbodiedCameraSimulator", _Simulator)
    monkeypatch.setattr(module, "ContinuousGemmaToolDecoderBackendV2", FakeBackend)
    monkeypatch.setattr(
        module,
        "robot_frame_clearance_state",
        lambda *_args, **_kwargs: torch.linspace(0.1, 1.0, 24),
    )
    config = load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")
    provenance = {
        "base_checkpoint_sha256": "0" * 64,
        "preregistration_sha256": "1" * 64,
        "cpu_preflight_sha256": "2" * 64,
        "training_authorization_sha256": "3" * 64,
        "clearance_cache_sha256": "4" * 64,
        "prefix_inventory_sha256": "5" * 64,
    }
    callback = build_saved_runtime_probe_v2(
        language=object(),
        installation=object(),  # type: ignore[arg-type]
        projector=NumericToolContextProjectorV2(),
        dataset=_dataset(),
        provenance=provenance,
        config=config,
    )
    probe = callback(tmp_path)
    assert calls == {
        "checkpoint": tmp_path,
        "provenance": provenance,
        "require_promoted": False,
        "runtime": calls["runtime"],
        "allow_staged": True,
        "instruction": PROBE_INSTRUCTION,
        "correction": None,
    }
    assert probe["saved_checkpoint_loaded"] is True
    assert probe["tool_execution_attempted"] is True
    assert probe["collision_interlock_checked"] is True
    assert probe["tool_execution_result"]["success"] is True
    assert probe["oracle_inputs_loaded"] is False
    assert probe["environmental_text_inputs"] == []


def test_v2_2_cpu_authorization_binds_runner_probe_and_denies_heavy_work() -> None:
    payload = build_cpu_authorization_v2_2()
    bindings = payload["bound_source_sha256"]
    assert set(bindings) == set(BOUND_SOURCE_PATHS)
    assert "scripts/run_gemma4_tool_decoder_v2_2_training.py" in bindings
    assert (
        "src/semantic_3d_chat/robot/gemma4_tool_decoder_v2_runtime_probe.py"
        in bindings
    )
    assert payload["full_model_mps_microbatch_authorized"] is False
    assert payload["multi_update_training_authorized"] is False
    assert payload["execution"]["optimizer_steps"] == 0
    assert payload["resource_contract"]["default_training_runner_bound"] is True

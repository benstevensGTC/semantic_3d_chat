from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_PROJECTED_DIM,
    GEMMA4_PROJECTED_START,
)
from semantic_3d_chat.mapping.voxel_map import SparseVoxelMap
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.navigation_policy import ACTION_NAMES
from semantic_3d_chat.robot.navigation_policy_v3 import (
    RUNTIME_INTERLOCK_VERSION,
    TARGET_STATE_DIM,
    GroundedContinuousNavigationControllerV3,
    SemanticGroundedActionBackendV3,
    apply_collision_limited_approach_interlock,
    apply_numeric_alignment_interlock,
    apply_numeric_approach_interlock,
    grounded_target_state,
    load_navigation_policy_v3_checkpoint,
    save_navigation_policy_v3_checkpoint,
    target_text_from_navigation_instruction,
)


def _alignment_state(error_degrees: float, *, available: bool = True) -> torch.Tensor:
    state = torch.zeros(1, TARGET_STATE_DIM)
    state[0, 0] = float(available)
    state[0, 8] = math.sin(math.radians(error_degrees)) * float(available)
    state[0, 9] = math.cos(math.radians(error_degrees)) * float(available)
    return state


def _target_for_yaw(yaw_degrees: float) -> tuple[float, float, float]:
    radians = math.radians(yaw_degrees)
    return (-math.sin(radians), math.cos(radians), 0.5)


def test_tool_policy_accepts_v3_training_attestation() -> None:
    from semantic_3d_chat.robot.llm_tool_policy import (
        GeneratedToolProposal,
        LocalGemmaToolPolicy,
    )

    digest = "a" * 64
    proposal = GeneratedToolProposal(
        text='{"tool":"stop","arguments":{}}',
        active_prefix_sha256=digest,
        scene_prefix_sha256=digest,
        robot_tokens_sha256=digest,
        local_inference=True,
        used_continuous_scene_prefix=True,
        used_continuous_robot_tokens=True,
        training_status=("supervised_continuous_semantic_grounded_navigation_policy_v3"),
    )
    assert LocalGemmaToolPolicy._context_error(proposal) is None


from semantic_3d_chat.robot.semantic_mapping import semantic_map_content_hash
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator


def _metadata() -> dict[str, object]:
    return {
        "scene_token_count": 6,
        "robot_token_count": 2,
        "model_id": "local/test-model",
        "model_revision": "1" * 40,
        "max_turn_degrees": 45.0,
        "max_move_m": 0.5,
        "room_size_m": [6.0, 5.0, 3.0],
        "grounding_feature_start": GEMMA4_PROJECTED_START,
        "grounding_feature_dim": GEMMA4_PROJECTED_DIM,
        "task_trained": True,
        "training_dataset_sha256": "2" * 64,
        "train_scene_count": 2,
        "validation_scene_count": 1,
        "scene_splits_disjoint": True,
        "complete_scene_prefix_required": True,
        "question_independent_static_scene_prefix_required": True,
        "every_scene_token_processed": True,
        "numeric_robot_tokens_required": True,
        "continuous_semantic_grounding_required": True,
        "all_map_voxels_scored_for_grounding": True,
        "query_dependent_grounding_navigation_only": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "runtime_required_files": ["policy.safetensors", "runtime_metadata.json"],
        "collision_interlock_required": True,
    }


@pytest.mark.parametrize(
    ("instruction", "target"),
    [
        ("Face the chair, then stop.", "chair"),
        ("Move closer to the table, then stop.", "table"),
        ("Move forward 0.25 meters, then stop.", None),
        ("Go around the chair and stop beside the bowl.", "bowl"),
        ("Turn toward the bowl using the shorter direction, then stop.", "bowl"),
        ("Scan the room, then move closer to the floor lamp and stop.", "floor lamp"),
        ("Stop immediately because movement was blocked.", None),
    ],
)
def test_target_parser_uses_only_user_text_protocol(instruction: str, target: str | None) -> None:
    assert target_text_from_navigation_instruction(instruction) == target


def test_numeric_grounded_target_state_is_relative_and_zeroable() -> None:
    state = torch.zeros(2, 18)
    state[:, 4] = 1.0  # yaw 0: sin=0, cos=1
    targets = torch.tensor([[1.0, 0.5, 0.4], [-1.0, 0.5, 0.4]])
    encoded = grounded_target_state(targets, state, torch.ones(2), room_size_m=[6.0, 5.0, 3.0])
    zero = grounded_target_state(targets, state, torch.zeros(2), room_size_m=[6.0, 5.0, 3.0])

    assert encoded.shape == (2, TARGET_STATE_DIM)
    assert torch.equal(encoded[:, 0], torch.ones(2))
    assert not torch.equal(encoded[0], encoded[1])
    assert torch.equal(zero, torch.zeros_like(zero))
    assert encoded[0, 8].sign() != encoded[1, 8].sign()


def test_numeric_alignment_interlock_corrects_a_stalled_turn_then_stops() -> None:
    correction, first = apply_numeric_alignment_interlock(
        "Face the target, then stop.",
        _alignment_state(10.89),
        {"tool": "turn", "arguments": {"angle_degrees": 0.09}},
        target_xyz_m=_target_for_yaw(10.89),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        deadband_degrees=3.0,
        stalled_turn_degrees=1.0,
    )
    stopped, second = apply_numeric_alignment_interlock(
        "Face the target, then stop.",
        _alignment_state(1.25),
        {"tool": "turn", "arguments": {"angle_degrees": 0.4}},
        target_xyz_m=_target_for_yaw(11.25),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=10.0,
        max_turn_degrees=45.0,
        deadband_degrees=3.0,
        stalled_turn_degrees=1.0,
    )

    assert correction["tool"] == "turn"
    assert correction["arguments"]["angle_degrees"] == pytest.approx(10.89)
    assert first["correction_applied"] is True
    assert first["reason"] == "stalled_learned_turn"
    assert first["target_xyz_m"] == pytest.approx(_target_for_yaw(10.89))
    assert first["robot_yaw_degrees"] == 0.0
    assert first["desired_yaw_degrees"] == pytest.approx(10.89)
    assert stopped == {"tool": "stop", "arguments": {}}
    assert second["stop_applied"] is True
    assert second["reason"] == "fresh_grounding_inside_deadband"
    assert first["environmental_text_inputs"] == []
    assert first["oracle_inputs_at_runtime"] is False


def test_numeric_alignment_interlock_rejects_premature_stop_and_bounds_correction() -> None:
    corrected, audit = apply_numeric_alignment_interlock(
        "Turn toward the target using the shorter direction, then stop.",
        _alignment_state(-80.0),
        {"tool": "stop", "arguments": {}},
        target_xyz_m=_target_for_yaw(-80.0),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
    )

    assert corrected == {"tool": "turn", "arguments": {"angle_degrees": -45.0}}
    assert audit["reason"] == "premature_learned_stop"
    assert audit["correction_applied"] is True


@pytest.mark.parametrize(
    "instruction",
    [
        "Face the target.",
        "Move closer to the target, then stop.",
        "Move forward 0.25 meters, then stop.",
    ],
)
def test_numeric_alignment_interlock_does_not_change_other_action_modes(
    instruction: str,
) -> None:
    learned = {"tool": "turn", "arguments": {"angle_degrees": 0.05}}
    output, audit = apply_numeric_alignment_interlock(
        instruction,
        _alignment_state(12.0),
        learned,
        target_xyz_m=_target_for_yaw(12.0),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
    )

    assert output == learned
    assert audit["terminal_alignment_requested"] is False
    assert audit["correction_applied"] is False
    assert audit["stop_applied"] is False


def test_numeric_approach_interlock_rejects_stop_until_translation_and_standoff() -> None:
    state = _alignment_state(0.0)
    move, first = apply_numeric_approach_interlock(
        "Move closer to the target, then stop.",
        state,
        {"tool": "stop", "arguments": {}},
        target_xyz_m=(0.0, 0.65, 0.5),
        initial_robot_position_xy_m=(0.0, 0.0),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
        target_standoff_m=0.5,
        minimum_progress_m=0.15,
    )
    stopped, second = apply_numeric_approach_interlock(
        "Move closer to the target, then stop.",
        state,
        {"tool": "stop", "arguments": {}},
        target_xyz_m=(0.0, 0.50, 0.5),
        initial_robot_position_xy_m=(0.0, 0.0),
        robot_position_xy_m=(0.0, 0.15),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
        target_standoff_m=0.5,
        minimum_progress_m=0.15,
    )

    assert move["tool"] == "move_forward"
    assert move["arguments"]["distance_meters"] == pytest.approx(0.15)
    assert first["reason"] == "premature_stop_forward_progress"
    assert first["goal_satisfied"] is False
    assert stopped == {"tool": "stop", "arguments": {}}
    assert second["reason"] == "fresh_grounding_approach_goal_satisfied"
    assert second["actual_progress_m"] == pytest.approx(0.15)
    assert second["environmental_text_inputs"] == []
    assert second["oracle_inputs_at_runtime"] is False


def test_v3_1_scan_then_approach_uses_numeric_completion_interlock() -> None:
    """Regression for the historical update-after-scan premature stop."""

    instruction = "Scan the room, then move closer to the arbitrary-7f target and stop."
    move, before = apply_numeric_approach_interlock(
        instruction,
        _alignment_state(0.0),
        {"tool": "stop", "arguments": {}},
        target_xyz_m=(0.0, 1.9, 0.5),
        initial_robot_position_xy_m=(0.0, -0.8),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
        target_standoff_m=0.5,
        minimum_progress_m=0.15,
    )
    stopped, after = apply_numeric_approach_interlock(
        instruction,
        _alignment_state(0.0),
        {"tool": "stop", "arguments": {}},
        target_xyz_m=(0.0, 0.5, 0.5),
        initial_robot_position_xy_m=(0.0, -0.8),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
        target_standoff_m=0.5,
        minimum_progress_m=0.15,
    )

    assert move == {"tool": "move_forward", "arguments": {"distance_meters": 0.5}}
    assert before["terminal_approach_requested"] is True
    assert before["runtime_interlock_version"] == RUNTIME_INTERLOCK_VERSION == "v3.1"
    assert before["reason"] == "premature_stop_forward_progress"
    assert before["environmental_text_inputs"] == []
    assert stopped == {"tool": "stop", "arguments": {}}
    assert after["goal_satisfied"] is True


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("Scan, then approach anything-opaque and stop.", True),
        ("Look around, then walk toward any phrase, then stop.", True),
        ("Scan the room, then move toward x9 and stop.", True),
        ("Scan the room, then stop.", False),
        ("Scan the room, then move closer to x9.", False),
        ("Face x9, then stop.", False),
    ],
)
def test_v3_1_compound_approach_grammar_is_action_only(
    instruction: str,
    expected: bool,
) -> None:
    _, audit = apply_numeric_approach_interlock(
        instruction,
        _alignment_state(0.0),
        {"tool": "stop", "arguments": {}},
        target_xyz_m=(0.0, 1.0, 0.5),
        initial_robot_position_xy_m=(0.0, 0.0),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
    )

    assert audit["terminal_approach_requested"] is expected
    assert audit["environmental_text_inputs"] == []
    assert audit["oracle_inputs_at_runtime"] is False


def test_numeric_approach_interlock_corrects_heading_before_translation() -> None:
    corrected, audit = apply_numeric_approach_interlock(
        "Approach the target, then stop.",
        _alignment_state(35.0),
        {"tool": "stop", "arguments": {}},
        target_xyz_m=_target_for_yaw(35.0),
        initial_robot_position_xy_m=(0.0, 0.0),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
    )

    assert corrected["tool"] == "turn"
    assert corrected["arguments"]["angle_degrees"] == pytest.approx(35.0)
    assert audit["reason"] == "premature_stop_heading_correction"


def test_numeric_approach_interlock_does_not_change_nonapproach_mode() -> None:
    learned = {"tool": "stop", "arguments": {}}
    output, audit = apply_numeric_approach_interlock(
        "Face the target, then stop.",
        _alignment_state(0.0),
        learned,
        target_xyz_m=(0.0, 0.3, 0.5),
        initial_robot_position_xy_m=(0.0, 0.0),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
    )

    assert output == learned
    assert audit["terminal_approach_requested"] is False


def _single_obstacle_collision_map() -> NumericCollisionMap:
    return NumericCollisionMap(
        np.asarray([[0.0, 0.70]], dtype=np.float32),
        room_min_xy_m=(-3.0, -2.5),
        room_max_xy_m=(3.0, 2.5),
        robot_radius_m=0.20,
        surface_padding_m=0.0,
    )


def test_collision_limited_approach_clips_then_stops_without_contact() -> None:
    collision_map = _single_obstacle_collision_map()
    learned = {"tool": "move_forward", "arguments": {"distance_meters": 0.5}}
    _, first = apply_numeric_approach_interlock(
        "Move closer to the target, then stop.",
        _alignment_state(0.0),
        learned,
        target_xyz_m=(0.0, 1.0, 0.5),
        initial_robot_position_xy_m=(0.0, 0.0),
        robot_position_xy_m=(0.0, 0.4),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
        target_standoff_m=0.5,
        minimum_progress_m=0.15,
    )
    clipped, clipped_audit = apply_collision_limited_approach_interlock(
        learned,
        first,
        collision_map=collision_map,
        robot_position_xy_m=(0.0, 0.4),
        robot_yaw_degrees=0.0,
        minimum_safe_step_m=0.02,
    )
    clipped_distance = clipped["arguments"]["distance_meters"]
    clipped_end = np.asarray([0.0, 0.4 + clipped_distance])

    assert clipped["tool"] == "move_forward"
    assert 0.08 < clipped_distance < 0.10
    assert collision_map.segment_check((0.0, 0.4), clipped_end).collision is False
    assert clipped_audit["reason"] == "collision_limited_safe_progress"
    assert clipped_audit["collision_limited_interlock"]["collision_predicted"] is True

    _, second = apply_numeric_approach_interlock(
        "Move closer to the target, then stop.",
        _alignment_state(0.0),
        learned,
        target_xyz_m=(0.0, 1.0, 0.5),
        initial_robot_position_xy_m=(0.0, 0.0),
        robot_position_xy_m=clipped_end,
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
        target_standoff_m=0.5,
        minimum_progress_m=0.15,
    )
    stopped, stopped_audit = apply_collision_limited_approach_interlock(
        learned,
        second,
        collision_map=collision_map,
        robot_position_xy_m=clipped_end,
        robot_yaw_degrees=0.0,
        minimum_safe_step_m=0.02,
    )

    assert stopped == {"tool": "stop", "arguments": {}}
    assert stopped_audit["goal_satisfied"] is False
    assert stopped_audit["completion_satisfied"] is True
    assert stopped_audit["completion_mode"] == "collision_limited_safe_stop"
    assert stopped_audit["collision_limited_interlock"]["safe_closest_reachable"] is True
    assert stopped_audit["environmental_text_inputs"] == []
    assert stopped_audit["oracle_inputs_at_runtime"] is False


def test_collision_limited_approach_does_not_claim_blocked_zero_progress() -> None:
    collision_map = NumericCollisionMap(
        np.asarray([[0.0, 0.21]], dtype=np.float32),
        room_min_xy_m=(-3.0, -2.5),
        room_max_xy_m=(3.0, 2.5),
        robot_radius_m=0.20,
        surface_padding_m=0.0,
    )
    learned = {"tool": "move_forward", "arguments": {"distance_meters": 0.3}}
    _, approach = apply_numeric_approach_interlock(
        "Approach the target, then stop.",
        _alignment_state(0.0),
        learned,
        target_xyz_m=(0.0, 1.0, 0.5),
        initial_robot_position_xy_m=(0.0, 0.0),
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
    )
    output, audit = apply_collision_limited_approach_interlock(
        learned,
        approach,
        collision_map=collision_map,
        robot_position_xy_m=(0.0, 0.0),
        robot_yaw_degrees=0.0,
        minimum_safe_step_m=0.02,
    )

    assert output == learned
    assert audit["completion_satisfied"] is False
    assert audit["collision_rejection_deferred_to_exact_simulator"] is True
    assert audit["collision_limited_interlock"]["reason"] == "blocked_before_minimum_progress"


def test_v3_controller_has_gradients_from_every_scene_and_target_value() -> None:
    torch.manual_seed(9)
    controller = GroundedContinuousNavigationControllerV3(16, model_dim=8)
    scene = torch.randn(2, 7, 16, requires_grad=True)
    robot = torch.randn(2, 3, 16, requires_grad=True)
    instruction = torch.randn(2, 16, requires_grad=True)
    target = torch.randn(2, TARGET_STATE_DIM, requires_grad=True)
    logits, arguments = controller(scene, robot, instruction, target)

    assert logits.shape == (2, len(ACTION_NAMES))
    assert arguments.shape == (2, len(ACTION_NAMES))
    (logits.square().sum() + arguments.square().sum()).backward()
    assert scene.grad is not None and torch.all(scene.grad.abs().sum(dim=-1) > 0)
    assert robot.grad is not None and torch.all(robot.grad.abs().sum(dim=-1) > 0)
    assert target.grad is not None and torch.all(target.grad.abs().sum(dim=-1) > 0)


def test_v3_checkpoint_is_versioned_sanitized_and_blocks_training_tree(
    tmp_path: Path,
) -> None:
    controller = GroundedContinuousNavigationControllerV3(16, model_dim=8)
    checkpoint = tmp_path / "checkpoints" / "navigation_policy_v3"
    saved = save_navigation_policy_v3_checkpoint(
        checkpoint, controller, runtime_metadata=_metadata()
    )
    loaded, metadata = load_navigation_policy_v3_checkpoint(
        checkpoint,
        expected_hidden_size=16,
        expected_model_id="local/test-model",
        expected_model_revision="1" * 40,
    )

    assert metadata["schema_version"] == 3
    assert metadata["continuous_semantic_grounding_required"] is True
    assert metadata["query_dependent_grounding_navigation_only"] is True
    assert metadata["oracle_inputs_at_runtime"] is False
    assert metadata["weights_sha256"] == saved["weights_sha256"]
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    encoded = json.dumps(metadata, sort_keys=True).casefold()
    for forbidden in ("chair", "table", "bowl", "floor lamp", "instance_id"):
        assert forbidden not in encoded
    with pytest.raises(ValueError, match="blocked"):
        save_navigation_policy_v3_checkpoint(
            tmp_path / "training" / "navigation_policy_v3",
            controller,
            runtime_metadata=_metadata(),
        )


class _ProjectedTextEncoder:
    output_dim = GEMMA4_PROJECTED_DIM

    def encode_queries(self, queries: list[str] | tuple[str, ...]) -> np.ndarray:
        output = np.zeros((len(queries), self.output_dim), dtype=np.float32)
        output[:, 0] = 1.0
        return output


def _write_semantic_map(path: Path) -> None:
    target = np.asarray(
        [
            [1.45, -0.05, 0.45],
            [1.50, -0.05, 0.50],
            [1.55, -0.05, 0.55],
            [1.45, 0.05, 0.50],
            [1.50, 0.05, 0.55],
            [1.55, 0.05, 0.45],
        ],
        dtype=np.float32,
    )
    distractor = target.copy()
    distractor[:, 0] *= -1.0
    points = np.concatenate((target, distractor))
    features = np.zeros((len(points), GEMMA4_PROJECTED_START + GEMMA4_PROJECTED_DIM))
    features[: len(target), GEMMA4_PROJECTED_START] = 1.0
    features[len(target) :, GEMMA4_PROJECTED_START + 1] = 1.0
    voxel_map = SparseVoxelMap(0.05, feature_dim=features.shape[1])
    voxel_map.add_observations(
        points,
        features.astype(np.float32),
        rgb=np.tile(np.asarray([[80.0, 120.0, 160.0]], dtype=np.float32), (len(points), 1)),
        frame_id="f_000001",
    )
    voxel_map.save(path, metadata={"scene_id": "scene_000001"})


def _runtime_config(tmp_path: Path, map_path: Path) -> dict:
    return {
        "seed": 7,
        "paths": {
            "data_root": str(tmp_path / "runtime"),
            "maps_root": str(map_path.parents[1]),
        },
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "render": {"resolution": [16, 16], "horizontal_fov_degrees": 72.0},
        "robot": {
            "radius_m": 0.20,
            "camera_height_m": 1.20,
            "max_move_m": 0.50,
            "max_move_to_m": 0.50,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.80,
            "surface_padding_m": 0.02,
            "scan_depth_min_m": 0.10,
            "scan_depth_max_m": 6.0,
            "initial_position_xy_m": [0.0, 0.0],
            "history_length": 16,
        },
    }


def test_v3_backend_grounding_scores_all_active_voxels_without_labels(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    _write_semantic_map(map_path)
    simulator = EmbodiedCameraSimulator(_runtime_config(tmp_path, map_path), "scene_000001")

    class Runtime:
        def __init__(self) -> None:
            self.simulator = simulator
            self.map_updater = SimpleNamespace(
                persistent_map_path=tmp_path / "not_created.npz",
                base_map_path=map_path,
            )
            language = SimpleNamespace(hidden_size=16)
            base = SimpleNamespace(language=language)
            self.prefix_refresher = SimpleNamespace(runtime=base)

        def prefix_binding(self) -> dict[str, object]:
            return {"map_sha256": semantic_map_content_hash(map_path)}

    controller = GroundedContinuousNavigationControllerV3(16, model_dim=8)
    backend = SemanticGroundedActionBackendV3(
        Runtime(),
        controller,
        {
            **_metadata(),
            "schema_version": 3,
            "architecture": "continuous_semantic_grounded_navigation_controller_v3",
            "hidden_size": 16,
            "model_dim": 8,
            "target_state_dim": TARGET_STATE_DIM,
            "action_names": list(ACTION_NAMES),
            "weights_sha256": "3" * 64,
        },
        {},
        text_encoder=_ProjectedTextEncoder(),
    )
    target_state, grounding = backend._ground("fixture")

    assert grounding is not None
    assert grounding.scored_voxels == 12
    assert target_state.shape == (1, TARGET_STATE_DIM)
    assert target_state[0, 0] == 1.0
    assert backend.last_grounding is not None
    assert backend.last_grounding["scored_voxels"] == 12
    assert "fixture" not in json.dumps(backend.last_grounding, sort_keys=True)

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_PROJECTED_DIM,
    GEMMA4_PROJECTED_START,
)
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.navigation_policy import ACTION_NAMES, ACTION_TO_INDEX
from semantic_3d_chat.robot.navigation_policy_v3 import (
    TARGET_STATE_DIM,
    GroundedContinuousNavigationControllerV3,
)
from semantic_3d_chat.robot.navigation_policy_v4 import (
    ARCHITECTURE,
    CLEARANCE_MAX_RANGE_M,
    CLEARANCE_RAY_COUNT,
    COLLISION_PROBE_DISTANCES_M,
    COLLISION_RISK_DIM,
    ClearanceAwareNavigationControllerV4,
    counterfactual_motion_collision_targets,
    load_navigation_policy_v4_checkpoint,
    robot_frame_clearance_state,
    save_navigation_policy_v4_checkpoint,
    select_highest_safe_nonterminal_action,
)
from semantic_3d_chat.training.train_navigation_policy_v3 import PreparedSamplesV3
from semantic_3d_chat.training.train_navigation_policy_v4 import (
    PreparedSamplesV4,
    evaluate_prepared_v4,
)


def _map(*points: tuple[float, float]) -> NumericCollisionMap:
    return NumericCollisionMap(
        np.asarray(points, dtype=np.float32),
        room_min_xy_m=(-3.0, -2.5),
        room_max_xy_m=(3.0, 2.5),
        robot_radius_m=0.20,
        surface_padding_m=0.0,
    )


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
        "preregistration_sha256": "4" * 64,
        "preregistered_single_arm": True,
        "v3_initialization_weights_sha256": "3" * 64,
        "train_scene_count": 14,
        "validation_scene_count": 8,
        "scene_splits_disjoint": True,
        "complete_scene_prefix_required": True,
        "question_independent_static_scene_prefix_required": True,
        "every_scene_token_processed": True,
        "numeric_robot_tokens_required": True,
        "continuous_semantic_grounding_required": True,
        "all_map_voxels_scored_for_grounding": True,
        "numeric_clearance_state_required": True,
        "clearance_from_sanitized_geometry_only": True,
        "exact_collision_mask_required": True,
        "unsafe_motion_fallback": "highest_safe_nonterminal_action",
        "query_dependent_grounding_navigation_only": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "runtime_required_files": ["policy.safetensors", "runtime_metadata.json"],
        "collision_interlock_required": True,
    }


def test_clearance_rays_follow_robot_frame_and_exact_inflated_geometry() -> None:
    collision_map = _map((0.0, 0.70), (0.80, 0.0))
    state = robot_frame_clearance_state(collision_map, (0.0, 0.0), 0.0)

    assert state.shape == (CLEARANCE_RAY_COUNT,)
    # yaw=0 points toward +Y; obstacle contact is 0.70 - radius 0.20.
    assert float(state[0]) == pytest.approx(0.50 / CLEARANCE_MAX_RANGE_M, abs=1e-6)
    assert float(state[CLEARANCE_RAY_COUNT // 2]) == pytest.approx(1.0)
    # Ray 18 is +X under the simulator's [-sin(yaw), cos(yaw)] convention.
    assert float(state[18]) == pytest.approx(0.60 / CLEARANCE_MAX_RANGE_M, abs=1e-6)
    assert torch.all((state >= 0.0) & (state <= 1.0))


def test_counterfactual_collision_targets_use_forward_and_backward_rays() -> None:
    clearance = torch.ones(CLEARANCE_RAY_COUNT)
    clearance[0] = 0.25
    clearance[CLEARANCE_RAY_COUNT // 2] = 0.40
    targets = counterfactual_motion_collision_targets(clearance)

    assert targets.shape == (COLLISION_RISK_DIM,)
    assert targets[:4].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert targets[4:].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert list(COLLISION_PROBE_DISTANCES_M) == [0.125, 0.25, 0.375, 0.5]


def test_v4_zero_residual_initialization_exactly_preserves_v3() -> None:
    torch.manual_seed(19)
    v3 = GroundedContinuousNavigationControllerV3(16, model_dim=8)
    v4 = ClearanceAwareNavigationControllerV4(16, model_dim=8)
    v4.initialize_from_v3(v3)
    scene = torch.randn(2, 7, 16)
    robot = torch.randn(2, 3, 16)
    instruction = torch.randn(2, 16)
    target = torch.randn(2, TARGET_STATE_DIM)
    clearance = torch.rand(2, CLEARANCE_RAY_COUNT, requires_grad=True)

    v3_logits, v3_arguments = v3(scene, robot, instruction, target)
    v4_logits, v4_arguments, risks = v4(
        scene, robot, instruction, target, clearance
    )
    assert torch.equal(v3_logits, v4_logits)
    assert torch.equal(v3_arguments, v4_arguments)
    assert risks.shape == (2, COLLISION_RISK_DIM)
    risks.square().sum().backward()
    assert clearance.grad is not None and torch.all(clearance.grad.abs().sum(dim=1) > 0)


def test_unsafe_motion_selects_highest_safe_nonterminal_action() -> None:
    collision_map = _map((0.0, 0.70))
    runtime = SimpleNamespace(
        simulator=SimpleNamespace(
            collision_map=collision_map,
            state=SimpleNamespace(
                position_xy_m=np.asarray([0.0, 0.0]),
                body_yaw_degrees=0.0,
            ),
        )
    )
    logits = torch.tensor([-5.0, -4.0, 4.0, 5.0, 1.0])
    arguments = torch.tensor([0.0, 0.0, 0.5, 1.0, -1.0])
    selection = select_highest_safe_nonterminal_action(
        runtime,
        logits,
        arguments,
        max_turn_degrees=45.0,
        max_move_m=0.5,
    )

    assert selection.raw_call == {
        "tool": "move_forward",
        "arguments": {"distance_meters": 0.5},
    }
    assert selection.unsafe_motion_masked is True
    assert selection.selected_action_index == ACTION_TO_INDEX["turn"]
    assert selection.selected_call["tool"] == "turn"
    assert selection.selected_call["tool"] != "stop"


def test_safe_motion_and_explicit_stop_are_not_reinterpreted() -> None:
    runtime = SimpleNamespace(
        simulator=SimpleNamespace(
            collision_map=_map((2.0, 2.0)),
            state=SimpleNamespace(
                position_xy_m=np.asarray([0.0, 0.0]),
                body_yaw_degrees=0.0,
            ),
        )
    )
    arguments = torch.zeros(len(ACTION_NAMES))
    safe = select_highest_safe_nonterminal_action(
        runtime,
        torch.tensor([-5.0, -4.0, 1.0, 5.0, 0.0]),
        arguments,
        max_turn_degrees=45.0,
        max_move_m=0.5,
    )
    stopped = select_highest_safe_nonterminal_action(
        runtime,
        torch.tensor([5.0, -4.0, 1.0, 0.0, -1.0]),
        arguments,
        max_turn_degrees=45.0,
        max_move_m=0.5,
    )
    assert safe.selected_call["tool"] == "move_forward"
    assert safe.unsafe_motion_masked is False
    assert stopped.selected_call == {"tool": "stop", "arguments": {}}
    assert stopped.unsafe_motion_masked is False


def test_v4_checkpoint_is_sanitized_versioned_and_blocks_training_tree(
    tmp_path: Path,
) -> None:
    controller = ClearanceAwareNavigationControllerV4(16, model_dim=8)
    checkpoint = tmp_path / "checkpoints" / "navigation_policy_v4"
    saved = save_navigation_policy_v4_checkpoint(
        checkpoint, controller, runtime_metadata=_metadata()
    )
    loaded, metadata = load_navigation_policy_v4_checkpoint(
        checkpoint,
        expected_hidden_size=16,
        expected_model_id="local/test-model",
        expected_model_revision="1" * 40,
    )

    assert metadata["schema_version"] == 4
    assert metadata["architecture"] == ARCHITECTURE
    assert metadata["numeric_clearance_state_required"] is True
    assert metadata["preregistered_single_arm"] is True
    assert metadata["clearance_from_sanitized_geometry_only"] is True
    assert metadata["query_dependent_grounding_navigation_only"] is True
    assert metadata["question_independent_static_scene_prefix_required"] is True
    assert metadata["unsafe_motion_fallback"] == "highest_safe_nonterminal_action"
    assert metadata["weights_sha256"] == saved["weights_sha256"]
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    encoded = json.dumps(metadata, sort_keys=True).casefold()
    for forbidden in ("chair", "table", "bowl", "floor lamp", "instance_id"):
        assert forbidden not in encoded
    with pytest.raises(ValueError, match="blocked"):
        save_navigation_policy_v4_checkpoint(
            tmp_path / "training" / "navigation_policy_v4",
            controller,
            runtime_metadata=_metadata(),
        )


@pytest.mark.parametrize("all_targeted", [False, True])
def test_v4_1_empty_target_subgroup_metrics_are_finite_json(
    all_targeted: bool,
) -> None:
    sample_count = 4
    base = PreparedSamplesV3(
        scene_indices=torch.zeros(sample_count, dtype=torch.long),
        robot_tokens=torch.randn(sample_count, 2, 16),
        instruction_embeddings=torch.randn(sample_count, 16),
        state_features=torch.zeros(sample_count, 18),
        target_xyz_m=torch.zeros(sample_count, 3),
        target_available=torch.full(
            (sample_count,), all_targeted, dtype=torch.bool
        ),
        target_states=torch.zeros(sample_count, TARGET_STATE_DIM),
        action_targets=torch.tensor(
            [
                ACTION_TO_INDEX["stop"],
                ACTION_TO_INDEX["turn"],
                ACTION_TO_INDEX["move_forward"],
                ACTION_TO_INDEX["move_backward"],
            ],
            dtype=torch.long,
        ),
        argument_targets=torch.tensor([0.0, 0.25, 0.5, 0.5]),
        families=("finite",) * sample_count,
        scene_ids=("scene_000001",) * sample_count,
    )
    samples = PreparedSamplesV4(
        base=base,
        clearance_states=torch.ones(sample_count, CLEARANCE_RAY_COUNT),
        collision_targets=torch.zeros(sample_count, COLLISION_RISK_DIM),
    )
    controller = ClearanceAwareNavigationControllerV4(16, model_dim=8)
    metrics = evaluate_prepared_v4(
        controller,
        torch.randn(1, 3, 16),
        samples,
        batch_size=4,
        max_turn_degrees=45.0,
        max_move_m=0.5,
    )

    assert metrics["targeted_sample_count"] == (
        sample_count if all_targeted else 0
    )
    empty_metric = (
        metrics["targetless_action_accuracy"]
        if all_targeted
        else metrics["targeted_action_accuracy"]
    )
    assert empty_metric == 0.0
    json.dumps(metrics, sort_keys=True, allow_nan=False)

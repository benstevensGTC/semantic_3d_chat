from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION_V2,
)
from semantic_3d_chat.training.gemma_waypoint_trace_generator import (
    _canonical_sha256,
    _effective_settings,
    _live_object_goal_dagger_augmentations,
    load_gemma_waypoint_trace_dataset,
)

CONFIG = "configs/experiments/gemma_waypoint_policy_operator_dagger_v13.yaml"
V12_DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v2_operator_dagger_v12"
)
V13_DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v2_operator_dagger_v13"
)
V12_CHECKPOINT = (
    "data_gemma4/checkpoints/"
    "gemma_waypoint_policy_v2_operator_dagger_v12_runtime_aligned"
)
V13_HIDDEN_CACHE = (
    "data_gemma4/training/"
    "gemma_waypoint_policy_v2_operator_dagger_v13_runtime_aligned_hidden_branch_refit_v1"
)
V13_CHECKPOINT = (
    "data_gemma4/checkpoints/"
    "gemma_waypoint_policy_v2_operator_dagger_v13_runtime_aligned"
)
V10_OBJECT_RUNTIME_SHA256 = (
    "a42a7445e1cdfa424b184f1a5958db197b5f7dcc1a0f15ed5ff4dffe521e71e6"
)
V11_OBJECT_RUNTIME_SHA256 = (
    "a853478a05e07ef5e1b8acec14f014fec41172dcf676ac8eeb8a30e46ee52aad"
)
V12_OBJECT_RUNTIME_SHA256 = (
    "a9d4069ce16ac4863260f6537db49c00828cdec95921c5c5b31bea217488d9de"
)
V12_OBJECT_SCORE_SHA256 = (
    "cb714a7b64f9c7f39fd8e2f8b7552019946f6f6d60294b2a55ce70129bed3e17"
)
V12_CHECKPOINT_SHA256 = (
    "95bbc0cd68a73f81dc0bc1d5875993cdcb7b6ccf092d4f13811f005e8ca1e9be"
)
SCENE_PREFIX_SHA256 = (
    "52c33298140845d341fa2b4568f2c6e960279495890e08455caafa7d5bbc9c95"
)
D2_INPUT_SHA256 = (
    "f690c67f625b2165c32a8f8634aa9e69f91332045c8a23a4d33ece25aeadccec"
)


def _input_sha256(row: dict[str, Any]) -> str:
    return _canonical_sha256(
        {"state_features": row["state_features"], "history": row["history"]}
    )


def _retained_row_sha256(row: dict[str, Any]) -> str:
    # Sample IDs are reassigned by final row position when an append-only source
    # is inserted before validation scenes. Every substantive field must remain.
    return _canonical_sha256(
        {key: value for key, value in row.items() if key != "sample_id"}
    )


@pytest.fixture(scope="module")
def artifacts():
    config = load_config(CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    v12_manifest, v12_rows = load_gemma_waypoint_trace_dataset(
        PROJECT_ROOT / V12_DATASET
    )
    v13_manifest, v13_rows = load_gemma_waypoint_trace_dataset(
        PROJECT_ROOT / V13_DATASET
    )
    return config, settings, v12_manifest, v12_rows, v13_manifest, v13_rows


def test_v13_cumulatively_preserves_all_object_goal_sources(artifacts) -> None:
    config, settings, _v12_manifest, _v12_rows, manifest, _rows = artifacts
    sources = _live_object_goal_dagger_augmentations(settings)
    expected_digests = [
        V10_OBJECT_RUNTIME_SHA256,
        V11_OBJECT_RUNTIME_SHA256,
        V12_OBJECT_RUNTIME_SHA256,
    ]

    assert [source.runtime_report_sha256 for source in sources] == expected_digests
    assert len(set(expected_digests)) == 3
    latest = sources[-1]
    assert latest.score_report_sha256 == V12_OBJECT_SCORE_SHA256
    assert latest.checkpoint_sha256 == V12_CHECKPOINT_SHA256
    assert latest.scene_prefix_sha256 == SCENE_PREFIX_SHA256
    assert latest.correction_decision_step == 2
    assert latest.observed_model_action == "move_to"
    assert latest.observed_action_accepted is True
    assert latest.expected_expert_first_action == "FACE"
    assert latest.expected_pose_xy_yaw == pytest.approx(
        (-0.5, -0.25, -52.9210205078125)
    )
    assert latest.expected_input_sha256 == D2_INPUT_SHA256
    assert manifest["live_object_goal_dagger_runtime_report_sha256"] == expected_digests
    assert manifest["live_object_goal_dagger_all_sources_retained"] is True
    policy = config["gemma_waypoint_policy"]
    assert policy["trace_dataset"] == str(V13_DATASET)
    assert policy["hidden_cache"] == V13_HIDDEN_CACHE
    assert policy["checkpoint_output"] == V13_CHECKPOINT
    assert policy["retention_reference_checkpoint"] == V12_CHECKPOINT
    assert policy["retention_reference_trace_dataset"] == str(V12_DATASET)
    assert policy["retention_joint_training_epochs"] == 0
    assert policy["retention_freeze_input_norm"] is True
    assert policy["retention_new_sample_weight"] == pytest.approx(32.0)
    assert policy["retention_minimum_shared_action_agreement"] == pytest.approx(1.0)
    assert policy["retention_minimum_new_action_accuracy"] == pytest.approx(1.0)
    assert policy["retention_maximum_shared_centered_logit_rmse"] == pytest.approx(
        0.25
    )
    assert policy["waypoint_branch_refit_enabled"] is True
    assert policy["waypoint_branch_refit_steps"] == 1000
    assert policy["waypoint_branch_refit_new_error_tolerance_m"] == pytest.approx(
        0.025
    )
    assert policy[
        "waypoint_branch_refit_minimum_new_within_tolerance_fraction"
    ] == pytest.approx(1.0)
    assert policy["retention_maximum_shared_waypoint_drift_m"] == pytest.approx(
        0.025
    )
    assert policy["heading_refit_steps"] == 1500
    assert policy["minimum_training_turn_margin_degrees"] == pytest.approx(3.0)
    assert policy["retention_maximum_shared_heading_drift_degrees"] == pytest.approx(
        1.0
    )


def test_v13_exact_v12_d2_input_is_face_corrected(artifacts) -> None:
    _config, _settings, _v12_manifest, _v12_rows, manifest, rows = artifacts
    correction = [
        row
        for row in rows
        if row.get("source_sample_sha256") == V12_OBJECT_RUNTIME_SHA256
        and row.get("family") == "object_goal_live_divergence_correction"
    ]

    assert len(correction) == 7
    assert [row["expert_action"] for row in correction] == [
        "FACE",
        "FACE",
        "FACE",
        "FACE",
        "MOVE_TO",
        "MOVE_TO",
        "STOP",
    ]
    first = correction[0]
    assert _input_sha256(first) == D2_INPUT_SHA256
    assert first["history_pose_xy_yaw"][-1] == pytest.approx(
        [-0.5, -0.25, -52.9210205078125]
    )
    assert first["expert_heading_degrees"] == pytest.approx(-12.9210205078125)
    assert first["waypoint_delta_robot_m"] is None
    assert len(first["history"]) == 1
    assert len(first["history"][0]) == HISTORY_FEATURE_DIM_V2
    metrics = manifest["live_object_goal_dagger_metrics_by_report_sha256"][
        V12_OBJECT_RUNTIME_SHA256
    ]
    assert metrics["first_correction_input_sha256"] == D2_INPUT_SHA256
    assert metrics["expert_first_action"] == "FACE"
    assert metrics["exact_prefix_transition_count"] == 1
    assert metrics["continuation_sample_count"] == 7
    assert metrics["continuation_episode_count"] == 1
    assert metrics["total_decision_count"] == 8
    assert metrics["runtime_planner_available"] is False
    assert metrics["runtime_oracle_available"] is False


def test_v13_retains_every_v12_row_and_has_no_exact_input_contradictions(
    artifacts,
) -> None:
    (
        _config,
        _settings,
        v12_manifest,
        v12_rows,
        v13_manifest,
        v13_rows,
    ) = artifacts
    old_rows = Counter(_retained_row_sha256(row) for row in v12_rows)
    new_rows = Counter(_retained_row_sha256(row) for row in v13_rows)

    assert not old_rows - new_rows
    assert len(v13_rows) - len(v12_rows) == 7
    assert v12_manifest["sample_count"] == 7816
    assert v13_manifest["sample_count"] == 7823
    assert v12_manifest["episode_count"] == 209
    assert v13_manifest["episode_count"] == 210
    assert v12_manifest["live_object_goal_dagger_sample_count"] == 12
    assert v13_manifest["live_object_goal_dagger_sample_count"] == 19
    assert v12_manifest["live_object_goal_dagger_episode_count"] == 6
    assert v13_manifest["live_object_goal_dagger_episode_count"] == 7
    assert v13_manifest["live_failure_dagger_sample_count"] == 355
    assert v13_manifest["live_failure_dagger_episode_count"] == 7
    assert v13_manifest["contradictory_exact_input_count"] == 0
    assert v13_manifest["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert v13_manifest["history_feature_dim"] == HISTORY_FEATURE_DIM_V2
    assert Counter(row["split"] for row in v12_rows) == {
        "train": 7105,
        "validation": 711,
    }
    assert Counter(row["split"] for row in v13_rows) == {
        "train": 7112,
        "validation": 711,
    }

    input_targets: dict[str, str] = {}
    for row in v13_rows:
        input_digest = _canonical_sha256(
            {
                "scene_id": row["scene_id"],
                "instruction": row["instruction"],
                "state_features": row["state_features"],
                "history": row["history"],
            }
        )
        target_digest = _canonical_sha256(
            {
                "expert_action": row["expert_action"],
                "waypoint_delta_robot_m": row["waypoint_delta_robot_m"],
                "heading_degrees": row["heading_degrees"],
            }
        )
        assert input_targets.setdefault(input_digest, target_digest) == target_digest

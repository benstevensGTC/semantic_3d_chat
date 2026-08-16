from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import replace
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
    _load_authenticated_live_object_goal_reports,
    load_gemma_waypoint_trace_dataset,
)

CONFIG = "configs/experiments/gemma_waypoint_policy_operator_dagger_v14.yaml"
V13_DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v2_operator_dagger_v13"
)
V14_DATASET = Path(
    "data_gemma4/training/gemma_waypoint_policy_v2_operator_dagger_v14"
)
V13_CHECKPOINT = (
    "data_gemma4/checkpoints/"
    "gemma_waypoint_policy_v2_operator_dagger_v13_runtime_aligned"
)
V14_HIDDEN_CACHE = (
    "data_gemma4/training/"
    "gemma_waypoint_policy_v2_operator_dagger_v14_runtime_aligned_hidden_branch_refit_v1"
)
V14_CHECKPOINT = (
    "data_gemma4/checkpoints/"
    "gemma_waypoint_policy_v2_operator_dagger_v14_runtime_aligned"
)
V10_RUNTIME_SHA256 = (
    "a42a7445e1cdfa424b184f1a5958db197b5f7dcc1a0f15ed5ff4dffe521e71e6"
)
V11_RUNTIME_SHA256 = (
    "a853478a05e07ef5e1b8acec14f014fec41172dcf676ac8eeb8a30e46ee52aad"
)
V12_RUNTIME_SHA256 = (
    "a9d4069ce16ac4863260f6537db49c00828cdec95921c5c5b31bea217488d9de"
)
V13_RUNTIME_SHA256 = (
    "2fb6fb805094205d10a98967ef50ed801d85a9eac48e28f4b66b76468d28bf50"
)
V13_SCORE_SHA256 = (
    "da3bc1ae9d61d5b48db8f38eb2fa41fb26635e853283575140ea8d08a85d4c54"
)
V13_CHECKPOINT_SHA256 = (
    "2a56e655fcb77f7e261f3b45d271ac5a3c97106c66359c6a9104ed00b74b16f3"
)
SCENE_PREFIX_SHA256 = (
    "52c33298140845d341fa2b4568f2c6e960279495890e08455caafa7d5bbc9c95"
)
D6_INPUT_SHA256 = (
    "46d272f5fe3cba7f1c924283eeffb0fe0ef989f37833950a4c069b2dd62fd979"
)
V14_DATASET_SHA256 = (
    "e3b9aa0143f8c45847c34d9cf740e797cd3909f54d9cd38eb3fa3d9af63905f7"
)
V14_TRACES_SHA256 = (
    "35af5ad24751a0bf9fdb3b8e770f9239528bd237e11aade9ca6c4eb093b30119"
)


def _input_sha256(row: dict[str, Any]) -> str:
    return _canonical_sha256(
        {"state_features": row["state_features"], "history": row["history"]}
    )


def _retained_row_sha256(row: dict[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in row.items() if key != "sample_id"}
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def artifacts():
    config = load_config(CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    sources = _live_object_goal_dagger_augmentations(settings)
    v13_manifest, v13_rows = load_gemma_waypoint_trace_dataset(
        PROJECT_ROOT / V13_DATASET
    )
    v14_manifest, v14_rows = load_gemma_waypoint_trace_dataset(
        PROJECT_ROOT / V14_DATASET
    )
    return config, sources, v13_manifest, v13_rows, v14_manifest, v14_rows


def test_v14_uses_v13_as_exact_retention_reference(artifacts) -> None:
    config, sources, _v13_manifest, _v13_rows, manifest, _v14_rows = artifacts
    expected_sources = [
        V10_RUNTIME_SHA256,
        V11_RUNTIME_SHA256,
        V12_RUNTIME_SHA256,
        V13_RUNTIME_SHA256,
    ]
    assert [source.runtime_report_sha256 for source in sources] == expected_sources
    assert manifest["live_object_goal_dagger_runtime_report_sha256"] == expected_sources
    assert manifest["live_object_goal_dagger_all_sources_retained"] is True

    source = sources[-1]
    assert source.score_report_sha256 == V13_SCORE_SHA256
    assert source.checkpoint_sha256 == V13_CHECKPOINT_SHA256
    assert source.scene_prefix_sha256 == SCENE_PREFIX_SHA256
    assert source.correction_decision_step == 6
    assert source.observed_model_action == "stop"
    assert source.observed_action_accepted is True
    assert source.expected_expert_first_action == "FACE"
    assert source.offline_expert_standoff_m == pytest.approx(0.76)
    assert all(item.offline_expert_standoff_m is None for item in sources[:-1])
    assert source.expected_pose_xy_yaw == pytest.approx(
        (-0.458561488760024, 0.14776059329035243, 66.31795501708984)
    )
    assert source.expected_input_sha256 == D6_INPUT_SHA256

    policy = config["gemma_waypoint_policy"]
    assert policy["trace_dataset"] == str(V14_DATASET)
    assert policy["hidden_cache"] == V14_HIDDEN_CACHE
    assert policy["checkpoint_output"] == V14_CHECKPOINT
    assert policy["retention_reference_checkpoint"] == V13_CHECKPOINT
    assert policy["retention_reference_trace_dataset"] == str(V13_DATASET)
    assert policy["retention_joint_training_epochs"] == 0
    assert policy["retention_minimum_shared_action_agreement"] == pytest.approx(1.0)
    assert policy["retention_minimum_new_action_accuracy"] == pytest.approx(1.0)
    assert policy["retention_maximum_shared_waypoint_drift_m"] == pytest.approx(
        0.025
    )
    assert policy["retention_maximum_shared_heading_drift_degrees"] == pytest.approx(
        1.0
    )


def test_v14_exact_d6_state_is_face_move_stop_supervised(artifacts) -> None:
    _config, _sources, _v13_manifest, _v13_rows, manifest, rows = artifacts
    correction = [
        row for row in rows if row.get("source_sample_sha256") == V13_RUNTIME_SHA256
    ]

    assert [row["expert_action"] for row in correction] == [
        "FACE",
        "MOVE_TO",
        "STOP",
    ]
    first = correction[0]
    assert _input_sha256(first) == D6_INPUT_SHA256
    assert first["history_action_codes"] == [1, 1, 0, 1, 1]
    assert len(first["history"]) == 5
    assert all(len(row) == HISTORY_FEATURE_DIM_V2 for row in first["history"])
    assert first["history_pose_xy_yaw"][-1] == pytest.approx(
        [-0.458561488760024, 0.14776059329035243, 66.31795501708984]
    )
    assert first["expert_heading_degrees"] == pytest.approx(64.24777710205993)
    assert correction[1]["expert_xy_m"] == pytest.approx(
        [-0.5528910249433068, 0.19326427153390802]
    )
    assert correction[1]["waypoint_delta_robot_m"] == pytest.approx(
        [1.3877787807814457e-17, 0.10473130444258912]
    )

    metrics = manifest["live_object_goal_dagger_metrics_by_report_sha256"][
        V13_RUNTIME_SHA256
    ]
    assert metrics["first_correction_input_sha256"] == D6_INPUT_SHA256
    assert metrics["exact_prefix_transition_count"] == 5
    assert metrics["continuation_sample_count"] == 3
    assert metrics["continuation_episode_count"] == 1
    assert metrics["total_decision_count"] == 8
    assert metrics["target_center_progress_m"] == pytest.approx(0.3086563958457009)
    assert metrics["final_oracle_bbox_standoff_m"] == pytest.approx(
        0.4186893264999066
    )
    assert metrics["minimum_padded_map_clearance_m"] == pytest.approx(
        0.0949496915144673
    )
    assert metrics["minimum_padded_map_clearance_m"] >= 0.05
    assert metrics["offline_expert_standoff_m"] == pytest.approx(0.76)
    assert metrics["offline_expert_standoff_override"] is True
    assert metrics["offline_planner_used_for_labels_only"] is True
    assert metrics["runtime_planner_available"] is False
    assert metrics["runtime_oracle_available"] is False


def test_v14_retains_every_v13_row_and_adds_only_three_training_rows(
    artifacts,
) -> None:
    (
        _config,
        _sources,
        v13_manifest,
        v13_rows,
        v14_manifest,
        v14_rows,
    ) = artifacts
    old_rows = Counter(_retained_row_sha256(row) for row in v13_rows)
    new_rows = Counter(_retained_row_sha256(row) for row in v14_rows)

    assert not old_rows - new_rows
    assert len(v14_rows) - len(v13_rows) == 3
    assert v13_manifest["sample_count"] == 7823
    assert v14_manifest["sample_count"] == 7826
    assert v13_manifest["episode_count"] == 210
    assert v14_manifest["episode_count"] == 211
    assert v14_manifest["dataset_sha256"] == V14_DATASET_SHA256
    assert v14_manifest["traces_sha256"] == V14_TRACES_SHA256
    assert v14_manifest["live_object_goal_dagger_sample_count"] == 22
    assert v14_manifest["live_object_goal_dagger_episode_count"] == 8
    assert v14_manifest["contradictory_exact_input_count"] == 0
    assert v14_manifest["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert Counter(row["split"] for row in v13_rows) == {
        "train": 7112,
        "validation": 711,
    }
    assert Counter(row["split"] for row in v14_rows) == {
        "train": 7115,
        "validation": 711,
    }


def test_v14_offline_standoff_parser_is_narrow_and_fail_closed() -> None:
    config = load_config(CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    additional = settings["additional_live_object_goal_dagger_augmentations"]

    for invalid in (True, 0.0, -0.1, float("inf"), "0.76"):
        mutated = copy.deepcopy(settings)
        mutated["additional_live_object_goal_dagger_augmentations"][-1][
            "offline_expert_standoff_m"
        ] = invalid
        with pytest.raises(ValueError, match="object-goal DAgger.*invalid"):
            _live_object_goal_dagger_augmentations(mutated)

    stop_mutated = copy.deepcopy(settings)
    assert additional[0]["expected_expert_first_action"] == "STOP"
    stop_mutated["additional_live_object_goal_dagger_augmentations"][0][
        "offline_expert_standoff_m"
    ] = 0.76
    with pytest.raises(ValueError, match="object-goal DAgger.*invalid"):
        _live_object_goal_dagger_augmentations(stop_mutated)


def _mutated_score_source(source, tmp_path: Path, score: dict[str, Any]):
    path = tmp_path / "score.json"
    path.write_text(json.dumps(score, sort_keys=True), encoding="utf-8")
    return replace(
        source,
        score_report_path=path,
        score_report_sha256=_file_sha256(path),
    )


def test_failed_score_loader_accepts_exactly_one_failed_geometry_gate(
    artifacts,
) -> None:
    _config, sources, _v13_manifest, _v13_rows, _manifest, _rows = artifacts
    _runtime, _score, goal = _load_authenticated_live_object_goal_reports(sources[-1])
    assert goal["passed"] is False
    assert goal["checks"] == {
        "accepted_gemma_stop": True,
        "maximum_oracle_bbox_standoff": True,
        "minimum_oracle_center_progress": False,
    }


def test_failed_score_loader_rejects_no_failed_geometry_gate(
    artifacts,
    tmp_path: Path,
) -> None:
    _config, sources, _v13_manifest, _v13_rows, _manifest, _rows = artifacts
    score = json.loads(sources[-1].score_report_path.read_text(encoding="utf-8"))
    goal = score["goals"][0]
    goal["target_center_progress_m"] = goal["minimum_center_progress_m"]
    goal["checks"]["minimum_oracle_center_progress"] = True
    mutated = _mutated_score_source(sources[-1], tmp_path, score)

    with pytest.raises(ValueError, match="failed score differs"):
        _load_authenticated_live_object_goal_reports(mutated)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda goal: goal["checks"].pop("maximum_oracle_bbox_standoff"),
        lambda goal: goal["checks"].update(
            {"minimum_oracle_center_progress": "false"}
        ),
        lambda goal: goal.update({"target_center_progress_m": "not-a-number"}),
    ],
)
def test_failed_score_loader_rejects_malformed_checks_or_metrics(
    artifacts,
    tmp_path: Path,
    mutation,
) -> None:
    _config, sources, _v13_manifest, _v13_rows, _manifest, _rows = artifacts
    score = copy.deepcopy(
        json.loads(sources[-1].score_report_path.read_text(encoding="utf-8"))
    )
    mutation(score["goals"][0])
    mutated = _mutated_score_source(sources[-1], tmp_path, score)

    with pytest.raises(ValueError, match="failed score differs"):
        _load_authenticated_live_object_goal_reports(mutated)

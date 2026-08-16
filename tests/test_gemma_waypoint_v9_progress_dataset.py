from __future__ import annotations

from semantic_3d_chat.config import load_config
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION_V2,
)
from semantic_3d_chat.training.gemma_waypoint_trace_generator import (
    _canonical_sha256,
    _effective_settings,
    _HistoryEncoding,
    _live_failure_dagger_augmentations,
    _live_pre_divergence_dagger_augmentations,
    _load_authenticated_live_failure_report,
    _numeric_state,
    _replay_authenticated_live_prefix,
)

CONFIG = "configs/experiments/gemma_waypoint_policy_operator_dagger_v9.yaml"
V8_REPORT_SHA256 = (
    "1f09d852bb59c2f99cf3b3f049b02512a763a34cee293f030b38677ce2abf5ec"
)
EXPECTED_V2_INPUTS = {
    "pre_d21_face_wp10": (
        "e189a520c86951efe41b735c7059ccf273c24de40e91e983560a9c94427fa243"
    ),
    "pre_d72_face_recover_wp43": (
        "9fad78458c78a8e26ff3bea9bdc88c7eee06e015671d3b269fbabf36f0448c33"
    ),
}


def test_v9_retains_all_sources_and_both_v8_divergence_branches() -> None:
    config = load_config(CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    sources = _live_failure_dagger_augmentations(settings)
    branches = _live_pre_divergence_dagger_augmentations(settings, sources)

    assert [source.report_sha256 for source in sources][-1] == V8_REPORT_SHA256
    assert len(sources) == 5
    assert {branch.branch_id for branch in branches} == {
        "pre_d45_move_wp25",
        "pre_d14_move_wp05",
        *EXPECTED_V2_INPUTS,
    }
    assert len({(branch.source.report_sha256, branch.branch_id) for branch in branches}) == 4
    d21 = next(branch for branch in branches if branch.branch_id.startswith("pre_d21"))
    d72 = next(branch for branch in branches if branch.branch_id.startswith("pre_d72"))
    assert (d21.observed_model_action, d21.expected_expert_first_action) == (
        "move_to",
        "FACE",
    )
    assert (d72.observed_model_action, d72.expected_expert_first_action) == (
        "stop",
        "FACE",
    )
    assert d72.recovery_plan_to_resume is True
    policy = config["gemma_waypoint_policy"]
    assert policy["history_dim"] == HISTORY_FEATURE_DIM_V2
    assert policy["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
    assert "v2_operator_dagger_v9" in policy["trace_dataset"]
    assert "v2_operator_dagger_v9" in policy["hidden_cache"]


def test_v8_branch_hashes_are_rederived_from_shared_v2_numeric_ledger() -> None:
    config = load_config(CONFIG)
    settings, _, _ = _effective_settings(config, "operator")
    sources = _live_failure_dagger_augmentations(settings)
    branches = _live_pre_divergence_dagger_augmentations(settings, sources)
    source = next(value for value in sources if value.report_sha256 == V8_REPORT_SHA256)
    report = _load_authenticated_live_failure_report(source)
    decisions = report["runtime_snapshot"]["model_decisions"]
    history_length = int(settings["history_length"])
    encoding = _HistoryEncoding(
        parameterization=HISTORY_PARAMETERIZATION_V2,
        feature_dim=HISTORY_FEATURE_DIM_V2,
        rejection_streak_scale=history_length,
    )

    for branch in branches:
        if branch.source.report_sha256 != V8_REPORT_SHA256:
            continue
        replay = _replay_authenticated_live_prefix(
            augmentation=source,
            decisions=decisions,
            transition_count=branch.correction_decision_step - 1,
            terminal_rejection_error_code=None,
            room_size_m=config["scene"]["room_size_m"],
            max_waypoint_step_m=float(settings["max_waypoint_step_m"]),
            fixed_face_step_degrees=float(settings["lap_fixed_face_step_degrees"]),
            history_length=history_length,
            history_encoding=encoding,
        )
        payload = {
            "state_features": _numeric_state(
                replay.pose,
                config["scene"]["room_size_m"],
            ),
            "history": [
                list(row) for row in replay.numeric_history[-history_length:]
            ],
        }
        assert _canonical_sha256(payload) == EXPECTED_V2_INPUTS[branch.branch_id]
        assert all(len(row) == HISTORY_FEATURE_DIM_V2 for row in payload["history"])

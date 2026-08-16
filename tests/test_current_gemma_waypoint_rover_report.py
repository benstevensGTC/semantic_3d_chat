from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _write_json(path: Path, value: object) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def summary() -> dict[str, object]:
    return BUILDER["build_summary"]()


def test_v14_rover_summary_authenticates_exact_model_only_result(
    summary: dict[str, object],
) -> None:
    rover = summary["gemma_waypoint_rover"]
    assert isinstance(rover, dict)
    assert rover["schema"] == "semantic_3d_chat.current_gemma_waypoint_rover.v1"
    assert rover["status"] == (
        "authenticated_one_scene_live_acceptance_passed_"
        "current_practical_rover_default"
    )
    assert rover["evidence_authenticated"] is True
    assert all(rover["checks"].values())
    assert rover["current_practical_rover_default"] is True
    assert rover["held_out_or_cross_scene_navigation_claim"] is False
    assert rover["project_wide_final_acceptance_claim"] is False
    assert rover["checkpoint_sha256"] == (
        "149f5e04de1d8305e642909443f03b96894edc3ece67e4500eacec8f5ca81e7c"
    )
    assert rover["scene_prefix_sha256"] == (
        "52c33298140845d341fa2b4568f2c6e960279495890e08455caafa7d5bbc9c95"
    )

    assert rover["architecture"] == {
        "scene_token_count": 258,
        "scene_token_dim": 1536,
        "robot_token_count": 4,
        "history_feature_dim": 16,
        "history_parameterization": "selected_action_parameters_goal_progress_v2",
        "complete_scene_prefix_required": True,
        "every_scene_token_processed_per_decision": True,
    }
    contract = rover["control_contract"]
    assert contract["actual_local_gemma_causal_forward_every_decision"] is True
    assert contract["model_selects_every_action_waypoint_heading_and_stop"] is True
    assert contract["deterministic_executor_only"] is True
    assert contract["deterministic_route_planner_used"] is False
    assert contract["fallback_used"] is False
    assert contract["substitution_applied"] is False
    assert contract["synthetic_stop_applied"] is False
    assert contract["cloud_inference_used"] is False
    assert contract["runtime_oracle_access_count"] == 0
    assert contract["environmental_text_inputs"] == []

    lap = rover["lap"]
    assert lap["passed"] is True
    assert lap["model_decision_count"] == 76
    assert lap["action_counts"] == {"face": 29, "move_to": 46, "stop": 1}
    assert lap["rejected_decision_count"] == 0
    assert lap["path_length_m"] == pytest.approx(18.715407828802974)
    assert lap["abs_winding_area_m2"] == pytest.approx(4.2729619440952105)
    assert lap["return_error_m"] == pytest.approx(0.04873323180807962)

    approach = rover["object_goals"]["approach_chair"]
    assert approach["passed"] is True
    assert approach["model_decision_count"] == 16
    assert approach["accepted_decision_count"] == 8
    assert approach["rejected_collision_attempt_count"] == 8
    assert approach["center_progress_m"] == pytest.approx(0.2636740957485727)
    assert approach["bbox_standoff_m"] == pytest.approx(0.43145638539269443)
    face = rover["object_goals"]["face_cube"]
    assert face["passed"] is True
    assert face["model_decision_count"] == 2
    assert face["yaw_error_degrees"] == pytest.approx(0.2015617605461273)


def test_v14_rover_summary_keeps_training_and_generalization_scope_exact(
    summary: dict[str, object],
) -> None:
    rover = summary["gemma_waypoint_rover"]
    training = rover["training"]
    assert training["scene_count"] == 1
    assert training["sample_count"] == 7115
    assert training["dataset_sample_count"] == 7826
    assert training["dataset_episode_count"] == 211
    assert training["action_accuracy"] == pytest.approx(0.9997189044952393)

    validation = rover["scene_disjoint_validation_control"]
    assert validation["scene_count"] == 2
    assert validation["scene_ids"] == ["scene_000031", "scene_000032"]
    assert validation["cached_sample_count"] == 96
    assert validation["evaluated_sample_count"] == 24
    assert validation["action_accuracy"] == pytest.approx(0.125)
    assert validation["mean_waypoint_error_m"] == pytest.approx(
        0.12276534736156464
    )
    assert validation["mean_heading_error_degrees"] == pytest.approx(
        29.524070739746094
    )
    assert validation["stop_recall"] == 0.0

    scope = summary["claim_scope"]
    assert scope["gemma_waypoint_rover_is_one_scene_live_acceptance"] is True
    assert scope["gemma_waypoint_rover_current_practical_default"] is True
    assert scope["gemma_waypoint_rover_held_out_or_cross_scene_measured"] is False
    assert scope["gemma_waypoint_rover_project_wide_final_acceptance_claimed"] is False


def test_v14_blender_and_isolation_evidence_is_current_and_bound(
    summary: dict[str, object],
) -> None:
    rover = summary["gemma_waypoint_rover"]
    isolation = rover["oracle_isolation"]
    assert isolation == {
        "passed": True,
        "source_oracle_directory_physically_unavailable": True,
        "forbidden_access_count": 0,
        "scene_prefix_built_before_questions": True,
        "scene_prefix_hash_invariant": True,
    }

    blender = rover["blender_live_ui"]
    assert blender["passed"] is True
    assert blender["goal"] == "Move closer to the chair and stop."
    assert blender["model_decision_count"] == 17
    assert blender["action_counts"] == {"face": 6, "move_to": 10, "stop": 1}
    assert blender["accepted_decision_count"] == 9
    assert blender["rejected_collision_attempt_count"] == 8
    assert blender["model_selected_terminal_stop"] is True
    assert blender["real_3d_furnished_room_visible"] is True
    assert blender["toy_rover_visible_in_3d_viewport"] is True
    assert blender["direct_driving_controls_absent"] is True
    assert blender["historical_hybrid_visual"] is False

    goal_mcp = rover["mcp_goal_interface"]
    assert goal_mcp == {
        "passed": True,
        "status": (
            "model_free_preflight_passed_official_sdk_dispatch_tested_in_process"
        ),
        "tool_names": ["navigate"],
        "tool_count": 1,
        "direct_motor_tools_exposed": False,
        "goal_is_passed_verbatim_to_gemma": True,
        "official_sdk_dispatch_tested_in_process": True,
        "model_free_preflight": True,
        "second_live_gemma_mcp_process_launched_during_blender_session": False,
        "same_controller_live_in_blender_verified": True,
        "response_contains_environmental_text": False,
        "legacy_numeric_mcp_separate": True,
    }

    assert rover["evidence_sha256"] == {
        path.as_posix(): digest
        for path, digest in BUILDER["GEMMA_WAYPOINT_V14_EVIDENCE_SHA256"].items()
    }
    for path, digest in rover["evidence_sha256"].items():
        assert summary["source_artifacts"][path] == digest


def test_v14_markdown_is_current_and_rejects_stale_hybrid_status(
    summary: dict[str, object],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "### Current model-only Gemma waypoint rover" in markdown
    assert "current practical-rover default" in collapsed
    assert "76 Gemma decisions" in collapsed
    assert "18.715408 m" in collapsed
    assert "4.272962 m²" in collapsed
    assert "0.048733 m" in collapsed
    assert "0.201562° yaw error" in collapsed
    assert "0.263674 m center progress" in collapsed
    assert "0.431456 m bounding-box standoff" in collapsed
    assert "configured 24-row control slice" in collapsed
    assert "only 12.50% action accuracy" in collapsed
    assert "not broad unseen-room navigation generalization" in collapsed
    assert "historical 47-waypoint hybrid/planner screenshots" in collapsed
    assert "exposes exactly one high-level tool, `navigate(goal)`" in collapsed
    assert "no motor primitives" in collapsed
    assert "model-free preflight with in-process SDK dispatch" in collapsed
    assert "No second heavy live Gemma MCP process was launched" in collapsed
    assert "blender_rover_v14_approach_chair_before.png" in markdown
    assert "blender_rover_v14_approach_chair_complete.png" in markdown

    stale_phrases = (
        "This is the strongest current embodied integration proof",
        "A current two-scene semantic face-target",
        "Current hybrid semantic navigation:",
        "dagger_v1_live_lap_failed_not_promoted",
        "has not passed live navigation acceptance",
    )
    assert not any(phrase in markdown for phrase in stale_phrases)


def test_v14_rover_inspector_fails_closed_on_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_gemma_waypoint_rover_v14"]
    hashes = dict(inspector.__globals__["GEMMA_WAYPOINT_V14_EVIDENCE_SHA256"])
    first_path = next(iter(hashes))
    hashes[first_path] = "0" * 64
    monkeypatch.setitem(
        inspector.__globals__, "GEMMA_WAYPOINT_V14_EVIDENCE_SHA256", hashes
    )

    rejected = inspector()
    assert rejected["evidence_authenticated"] is False
    assert rejected["current_practical_rover_default"] is False
    assert rejected["held_out_or_cross_scene_navigation_claim"] is False
    assert rejected["project_wide_final_acceptance_claim"] is False


def test_v14_rover_inspector_rejects_rehashed_fallback_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_gemma_waypoint_rover_v14"]
    globals_ = inspector.__globals__
    source_lap = ROOT / globals_["GEMMA_WAYPOINT_V14_LIVE_ACCEPTANCE"]
    source_release = ROOT / globals_["GEMMA_WAYPOINT_V14_RELEASE_SUMMARY"]

    lap = json.loads(source_lap.read_text(encoding="utf-8"))
    lap["fallback_used"] = True
    tampered_lap = tmp_path / "lap.json"
    lap_sha256 = _write_json(tampered_lap, lap)

    release = json.loads(source_release.read_text(encoding="utf-8"))
    release["evidence"]["lap_report"] = {
        "path": tampered_lap.as_posix(),
        "sha256": lap_sha256,
    }
    tampered_release = tmp_path / "release.json"
    release_sha256 = _write_json(tampered_release, release)

    hashes = dict(globals_["GEMMA_WAYPOINT_V14_EVIDENCE_SHA256"])
    del hashes[globals_["GEMMA_WAYPOINT_V14_LIVE_ACCEPTANCE"]]
    del hashes[globals_["GEMMA_WAYPOINT_V14_RELEASE_SUMMARY"]]
    hashes[tampered_lap] = lap_sha256
    hashes[tampered_release] = release_sha256
    monkeypatch.setitem(globals_, "GEMMA_WAYPOINT_V14_LIVE_ACCEPTANCE", tampered_lap)
    monkeypatch.setitem(globals_, "GEMMA_WAYPOINT_V14_RELEASE_SUMMARY", tampered_release)
    monkeypatch.setitem(globals_, "GEMMA_WAYPOINT_V14_EVIDENCE_SHA256", hashes)

    rejected = inspector()
    assert rejected["evidence_authenticated"] is False
    assert rejected["current_practical_rover_default"] is False
    assert "lap evidence differs" in rejected["measurement_evidence_error"]


def test_v14_rover_inspector_rejects_rehashed_goal_mcp_motor_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspector = BUILDER["_inspect_gemma_waypoint_rover_v14"]
    globals_ = inspector.__globals__
    source = ROOT / globals_["GEMMA_WAYPOINT_V14_GOAL_MCP_PREFLIGHT"]
    preflight = json.loads(source.read_text(encoding="utf-8"))
    preflight["tool_names"] = ["navigate", "move_forward"]
    preflight["tool_count"] = 2
    preflight["direct_motor_tools_exposed"] = True
    tampered = tmp_path / "goal_mcp_preflight.json"
    tampered_sha256 = _write_json(tampered, preflight)

    hashes = dict(globals_["GEMMA_WAYPOINT_V14_EVIDENCE_SHA256"])
    del hashes[globals_["GEMMA_WAYPOINT_V14_GOAL_MCP_PREFLIGHT"]]
    hashes[tampered] = tampered_sha256
    monkeypatch.setitem(
        globals_, "GEMMA_WAYPOINT_V14_GOAL_MCP_PREFLIGHT", tampered
    )
    monkeypatch.setitem(globals_, "GEMMA_WAYPOINT_V14_EVIDENCE_SHA256", hashes)

    rejected = inspector()
    assert rejected["evidence_authenticated"] is False
    assert rejected["current_practical_rover_default"] is False
    assert "goal-only MCP preflight evidence differs" in rejected[
        "measurement_evidence_error"
    ]

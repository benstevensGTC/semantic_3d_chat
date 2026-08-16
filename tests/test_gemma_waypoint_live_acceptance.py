from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = runpy.run_path(str(ROOT / "scripts/verify_gemma_waypoint_rover_live.py"))
PREFIX_SHA256 = "a" * 64
NAVIGATION_SHA256 = "c" * 64
RUNTIME_BINDING_SHA256 = "d" * 64


def _action(x: float, y: float) -> dict[str, Any]:
    return {
        "position_m": [x, y, 0.0],
        "distance_moved": 0.0,
        "success": True,
    }


def _control(*, attempted: bool) -> dict[str, Any]:
    return {
        "control_mode": "actual_local_gemma_model_only_waypoint_policy",
        "gemma_attempted": attempted,
        "gemma_accepted": attempted,
        "fallback_used": False,
        "local_inference": True,
        "cloud_model_used": False,
        "high_level_natural_language_only": True,
        "task_trained_navigation": True,
        "untrained_json_backend_enabled": False,
        "static_precomputed_scene_memory": True,
        "camera_control_input": False,
        "navigation_checkpoint_sha256": NAVIGATION_SHA256,
        "gemma_runtime_binding_sha256": RUNTIME_BINDING_SHA256,
    }


def _memory() -> dict[str, Any]:
    return {
        "sha256": "b" * 64,
        "tensor_shape": [1, 258, 1536],
        "active_tensor_shape": [1, 262, 1536],
        "source_voxels": 74_699,
        "processed_voxels": 74_699,
        "semantic_feature_dim": 3_072,
        "question_dependent_scene_retrieval": False,
        "loaded_file_audit": {"forbidden_access_count": 0, "passed": True},
    }


def _state(position: tuple[float, float], *, action_count: int) -> dict[str, Any]:
    return {
        "scene_id": "scene_000001",
        "position_xy_m": list(position),
        "scene_prefix_hash": PREFIX_SHA256,
        "map_version": 1,
        "scan_count": 0,
        "action_count": action_count,
    }


def _decision(
    step: int,
    *,
    terminal: bool,
    target_xy: tuple[float, float] | None = None,
    accepted: bool = True,
    executed: bool | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    if executed is None:
        executed = accepted
    action = "stop" if terminal else "move_to"
    return {
        "step": step,
        "model_action": action,
        "primitive_tool": "stop" if terminal else "move_to",
        "derived_world_waypoint_xy_m": (None if terminal else list(target_xy or (0.0, 0.0))),
        "accepted": accepted,
        "executed": executed,
        "error_code": error_code,
        "actual_gemma_causal_forward": True,
        "checkpoint_sha256": NAVIGATION_SHA256,
        "scene_prefix_sha256": PREFIX_SHA256,
        "model_selected_every_waypoint_and_heading": True,
        "deterministic_route_planner_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
    }


def _payloads(actions: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    startup = {
        "state": _state((0.0, 0.0), action_count=0),
        "scene_memory": _memory(),
        "control": _control(attempted=False),
    }
    final_xy = tuple(actions[-1]["position_m"][:2]) if actions else (0.0, 0.0)
    # STOP terminates the Gemma decision protocol but does not append a
    # simulator motion receipt.
    action_receipts = list(actions)
    decisions = [
        _decision(
            index,
            terminal=False,
            target_xy=tuple(action["position_m"][:2]),
        )
        for index, action in enumerate(actions, start=1)
    ]
    decisions.append(_decision(len(decisions) + 1, terminal=True))
    result = {
        "state": _state(final_xy, action_count=len(action_receipts)),
        "scene_memory": _memory(),
        "control": _control(attempted=True),
        "actions": action_receipts,
        "model_decisions": decisions,
        "reply": "done",
    }
    return startup, result


def _install_requests(
    monkeypatch: pytest.MonkeyPatch,
    startup: dict[str, Any],
    result: dict[str, Any],
) -> None:
    def request(
        _base_url: str,
        path: str,
        _payload: object = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        return result if path == "/api/instruction" else startup

    monkeypatch.setitem(VERIFIER["run_acceptance"].__globals__, "_request", request)


def test_real_closed_loop_passes_area_gate_and_preserves_model_only_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup, result = _payloads(
        [_action(2.0, 0.0), _action(2.0, 2.0), _action(0.0, 2.0), _action(0.0, 0.0)]
    )
    _install_requests(monkeypatch, startup, result)

    report = VERIFIER["run_acceptance"](
        "http://127.0.0.1:8770",
        "Do a lap around the room.",
        expected_navigation_checkpoint_sha256=NAVIGATION_SHA256,
    )

    assert report["passed"] is True
    assert report["path_length_m"] == pytest.approx(8.0)
    assert report["signed_winding_area_m2"] == pytest.approx(4.0)
    assert report["abs_winding_area_m2"] == pytest.approx(4.0)
    assert report["deterministic_route_planner_used"] is False
    assert report["fallback_used"] is False
    assert report["substitution_applied"] is False
    assert report["synthetic_stop_applied"] is False
    assert report["navigation_checkpoint_sha256"] == NAVIGATION_SHA256
    assert report["gemma_runtime_binding_sha256"] == RUNTIME_BINDING_SHA256


def test_acceptance_rejects_a_different_live_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup, result = _payloads(
        [_action(2.0, 0.0), _action(2.0, 2.0), _action(0.0, 2.0), _action(0.0, 0.0)]
    )
    _install_requests(monkeypatch, startup, result)

    with pytest.raises(AssertionError, match="differs from the requested candidate"):
        VERIFIER["run_acceptance"](
            "http://127.0.0.1:8770",
            "Do a lap around the room.",
            expected_navigation_checkpoint_sha256="e" * 64,
        )


def test_long_out_and_back_cannot_pass_as_lap(monkeypatch: pytest.MonkeyPatch) -> None:
    startup, result = _payloads([_action(3.0, 0.0), _action(0.0, 0.0)])
    _install_requests(monkeypatch, startup, result)

    with pytest.raises(AssertionError, match="lap geometry failed"):
        VERIFIER["run_acceptance"]("http://127.0.0.1:8770", "Do a lap around the room.")

    metrics = VERIFIER["_trajectory_metrics"]((0.0, 0.0), result["actions"])
    assert metrics["path_length_m"] == pytest.approx(6.0)
    assert metrics["signed_winding_area_m2"] == pytest.approx(0.0)


def test_degenerate_collinear_closed_path_has_zero_winding_area() -> None:
    metrics = VERIFIER["_trajectory_metrics"](
        (0.0, 0.0),
        [_action(2.0, 0.0), _action(4.0, 0.0), _action(0.0, 0.0)],
    )

    assert metrics["path_length_m"] == pytest.approx(8.0)
    assert metrics["abs_winding_area_m2"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "actions",
    [
        [{}],
        [{"position_m": [0.0, float("nan"), 0.0]}],
        [{"position_m": [0.0, 0.0]}],
        [{"position_xy_m": [0.0, 0.0], "position_m": [1.0, 0.0, 0.0]}],
    ],
)
def test_malformed_or_nonfinite_action_pose_fails_closed(
    actions: list[dict[str, Any]],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        VERIFIER["_trajectory_metrics"]((0.0, 0.0), actions)


def test_failure_artifact_includes_zero_area_metric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    startup, result = _payloads([_action(3.0, 0.0), _action(0.0, 0.0)])
    calls = 0

    def request(
        _base_url: str,
        path: str,
        _payload: object = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if path == "/api/instruction" or calls >= 4:
            return result
        return startup

    globals_ = VERIFIER["main"].__globals__
    monkeypatch.setitem(globals_, "_request", request)
    output = tmp_path / "failed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", "--output", str(output)],
    )

    assert VERIFIER["main"]() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["path_length_m"] == pytest.approx(6.0)
    assert report["signed_winding_area_m2"] == pytest.approx(0.0)
    assert report["abs_winding_area_m2"] == pytest.approx(0.0)
    assert report["trajectory_metric_failure"] is None
    assert report["deterministic_route_planner_used"] is False
    assert report["fallback_used"] is False
    assert report["substitution_applied"] is False


def test_decision_provenance_rejects_substitution() -> None:
    decision = _decision(1, terminal=True)
    decision["substitution_applied"] = True

    with pytest.raises(AssertionError, match="failed provenance"):
        VERIFIER["_verify_decisions"](
            {"model_decisions": [decision]},
            expected_scene_prefix_sha256=PREFIX_SHA256,
        )


def test_complete_decision_trajectory_is_not_truncated_by_63_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outbound = [_action(index / 10.0, 0.0) for index in range(1, 41)]
    inbound = [_action(index / 10.0, 0.0) for index in range(39, -1, -1)]
    startup, result = _payloads([*outbound, *inbound])
    assert len(result["model_decisions"]) == 81
    result["actions"] = result["actions"][-63:]
    _install_requests(monkeypatch, startup, result)

    report = VERIFIER["run_acceptance"](
        "http://127.0.0.1:8770",
        "Traverse the route and return.",
    )

    assert report["passed"] is True
    assert report["trajectory_source"] == "complete_model_decisions"
    assert report["numeric_action_receipt_count"] == 63
    assert report["model_decision_count"] == 81
    assert report["trajectory_point_count"] == 82
    assert report["path_length_m"] == pytest.approx(8.0)
    assert report["return_error_m"] == pytest.approx(0.0)


def test_v8_terminal_stop_is_not_misaligned_as_a_motion_receipt() -> None:
    artifact = json.loads(
        (
            ROOT
            / "reports/gemma4/metrics/gemma_waypoint_dagger_v8_live_acceptance.json"
        ).read_text(encoding="utf-8")
    )
    snapshot = artifact["runtime_snapshot"]
    decisions = VERIFIER["_verify_decisions"](
        snapshot,
        expected_scene_prefix_sha256=(snapshot["scene_prefix_hash"]),
        expected_navigation_checkpoint_sha256=(
            snapshot["control"]["navigation_checkpoint_sha256"]
        ),
    )
    metrics = VERIFIER["_decision_trajectory_metrics"](
        artifact["initial_position_xy_m"],
        decisions,
        final_xy=snapshot["state"]["position_xy_m"],
    )

    assert len(metrics["executed_completion_positions"]) == 63
    assert len(metrics["receipt_completion_positions"]) == 62
    assert len(snapshot["actions"]) == 62
    VERIFIER["_verify_action_receipt_suffix"](
        snapshot["actions"], metrics["receipt_completion_positions"]
    )
    assert metrics["trajectory_point_count"] == 64
    assert metrics["path_length_m"] == pytest.approx(13.710212594574054)
    assert metrics["signed_winding_area_m2"] == pytest.approx(
        -3.4644705506085653
    )
    assert metrics["reconstructed_final_position_xy_m"] == pytest.approx(
        [-1.3990058852065077, -1.9595794292964281]
    )

    with pytest.raises(ValueError, match="longer than the receipt-bearing"):
        VERIFIER["_verify_action_receipt_suffix"](
            [*snapshot["actions"], snapshot["actions"][-1]],
            metrics["receipt_completion_positions"],
        )


def test_receipt_mismatch_does_not_erase_authenticated_decision_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    startup, result = _payloads(
        [_action(2.0, 0.0), _action(2.0, 2.0), _action(0.0, 2.0), _action(0.0, 0.0)]
    )
    result["actions"][-1] = _action(9.0, 9.0)
    calls = 0

    def request(
        _base_url: str,
        path: str,
        _payload: object = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if path == "/api/instruction" or calls >= 4:
            return result
        return startup

    globals_ = VERIFIER["main"].__globals__
    monkeypatch.setitem(globals_, "_request", request)
    output = tmp_path / "receipt-mismatch.json"
    monkeypatch.setattr(sys, "argv", ["verify", "--output", str(output)])

    assert VERIFIER["main"]() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["model_decisions_authenticated"] is True
    assert report["decision_authentication_failure"] is None
    assert report["trajectory_metric_failure"] is None
    assert report["path_length_m"] == pytest.approx(8.0)
    assert report["signed_winding_area_m2"] == pytest.approx(4.0)
    assert report["action_receipt_alignment_passed"] is False
    assert "differs from its reconstructed decision pose" in report[
        "action_receipt_alignment_failure"
    ]


@pytest.mark.parametrize("executed, expected_points", [(False, 2), (True, 3)])
def test_rejected_move_does_not_mutate_reconstructed_pose(
    executed: bool,
    expected_points: int,
) -> None:
    rejected = _decision(
        1,
        terminal=False,
        target_xy=(4.0, 3.0),
        accepted=False,
        executed=executed,
        error_code="E_MODEL_COLLISION",
    )
    stopped = _decision(2, terminal=True)

    metrics = VERIFIER["_decision_trajectory_metrics"](
        (0.0, 0.0),
        [rejected, stopped],
        final_xy=(0.0, 0.0),
    )

    assert metrics["trajectory_point_count"] == expected_points
    assert metrics["path_length_m"] == pytest.approx(0.0)
    assert metrics["abs_winding_area_m2"] == pytest.approx(0.0)
    assert metrics["reconstructed_final_position_xy_m"] == [0.0, 0.0]


def test_face_and_stop_do_not_translate_reconstructed_pose() -> None:
    face = _decision(1, terminal=False, target_xy=(9.0, 9.0))
    face.update(
        {
            "model_action": "face",
            "primitive_tool": "turn",
            "derived_world_waypoint_xy_m": None,
        }
    )
    stopped = _decision(2, terminal=True)

    metrics = VERIFIER["_decision_trajectory_metrics"](
        (1.25, -0.75),
        [face, stopped],
        final_xy=(1.25, -0.75),
    )

    assert metrics["trajectory_point_count"] == 3
    assert metrics["path_length_m"] == pytest.approx(0.0)
    assert metrics["reconstructed_final_position_xy_m"] == [1.25, -0.75]


def test_non_move_waypoint_and_final_pose_mismatch_fail_closed() -> None:
    face = _decision(1, terminal=False, target_xy=(1.0, 1.0))
    face.update({"model_action": "face", "primitive_tool": "turn"})
    with pytest.raises(ValueError, match="non-MOVE action carries a waypoint"):
        VERIFIER["_decision_trajectory_metrics"]((0.0, 0.0), [face])

    move = _decision(1, terminal=False, target_xy=(1.0, 0.0))
    with pytest.raises(ValueError, match="differs from runtime state"):
        VERIFIER["_decision_trajectory_metrics"](
            (0.0, 0.0),
            [move],
            final_xy=(0.0, 0.0),
        )

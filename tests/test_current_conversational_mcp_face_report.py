from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_current_report.py"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return BUILDER["build_summary"]()


def test_current_summary_authenticates_live_semantic_conversational_mcp(
    summary: dict[str, Any],
) -> None:
    evidence = summary["conversational_mcp_face_runtime"]

    assert evidence["status"] == (
        "authenticated_single_scene_selective_gemma_numeric_v3_official_mcp_face_passed"
    )
    assert evidence["evidence_authenticated"] is True
    assert evidence["development_only"] is True
    assert evidence["official_validation"] is False
    assert evidence["runtime_promotion_authorized"] is False
    assert evidence["scene_id"] == "scene_000001"
    assert evidence["instruction"] == "Face the chair, then stop."
    assert evidence["instruction_source"] == "user_text_only"
    assert all(evidence["checks"].values())

    policy = evidence["policy"]
    assert policy == {
        "name": "selective_gemma_all_voxel_v3_numeric_alignment",
        "continuous_grounding": "selective_local_gemma_tied_token_embeddings",
        "grounding_scope": "every_active_map_voxel",
        "action_controller": "v3_numeric_alignment_convergence_interlock",
        "learned_v3_action_head_used": False,
        "gemma_native_function_calling_used": False,
        "direct_function_action_execution_used": False,
        "neutral_stop_sentinel_used": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }
    assert evidence["transport"] == {
        "implementation": "official_python_mcp_sdk_stdio",
        "mcp_sdk_version": "2.0.0",
        "process_boundary": True,
        "numeric_structured_output_only": True,
        "tool_count": 9,
    }

    runtime = evidence["runtime"]
    assert runtime["passed"] is True
    assert runtime["action_sequence"] == ["scan", "turn", "turn", "stop"]
    assert runtime["turn_degrees"] == pytest.approx([45.0, 21.923440783540574])
    assert runtime["final_body_yaw_degrees"] == pytest.approx(66.92344078354057)
    assert runtime["final_continuous_grounding_residual_degrees"] == pytest.approx(
        -0.32455355627810434
    )
    assert runtime["collision_count"] == 0
    assert runtime["all_decisions_used_fresh_all_voxel_grounding"] is True
    assert runtime["scored_voxels_by_decision"] == [74897, 75468, 76220]
    assert runtime["prefix_binding_refresh_count"] == 5
    assert runtime["environmental_text_inputs"] == []
    assert runtime["semantic_leaks_in_numeric_tool_receipts"] == []
    assert runtime["target_phrase_retained_in_tool_output"] is False
    assert runtime["oracle_inputs_at_runtime"] is False
    assert runtime["client_loaded_file_count"] == 93
    assert runtime["server_loaded_file_count"] == 4185
    assert runtime["client_forbidden_access_count"] == 0
    assert runtime["server_forbidden_access_count"] == 0

    score = evidence["evaluation_only_oracle_score"]
    assert score == {
        "passed": True,
        "absolute_heading_error_degrees": pytest.approx(0.14641915347542067),
        "maximum_heading_error_degrees": 20.0,
        "collision_count": 0,
        "runtime_validated_before_oracle_open": True,
        "runtime_process_read_oracle": False,
        "oracle_geometry_loaded_by_scorer_only": True,
        "score_fed_back_to_runtime": False,
        "oracle_files_opened_by_report_builder": False,
    }
    for path, digest in BUILDER["CONVERSATIONAL_MCP_FACE_EVIDENCE_SHA256"].items():
        assert summary["source_artifacts"][path.as_posix()] == digest


def test_current_markdown_states_exact_conversational_mcp_boundary(
    summary: dict[str, Any],
) -> None:
    markdown = BUILDER["render_markdown"](summary)
    collapsed = " ".join(markdown.split())

    assert "A separate live conversational MCP episode" in collapsed
    assert "`Face the chair, then stop.`" in collapsed
    assert "initial scan, turns of 45.000 and 21.923 degrees, and stop" in collapsed
    assert "Every decision rescored every active map voxel" in collapsed
    assert "final continuous grounding residual was 0.325 degrees" in collapsed
    assert "0.146 degrees physical heading error" in collapsed
    assert "selective local Gemma tied- token embeddings" in collapsed
    assert "deterministic V3 numeric alignment interlock" in collapsed
    assert "does **not** use Gemma native function calling" in collapsed
    assert "does **not** execute the learned V3 action head" in collapsed
    assert "does not claim that a learned action decoder selected these calls" in collapsed
    assert "one deterministic development scene and one instruction family" in collapsed
    assert "not held-out or general conversational-navigation evidence" in collapsed


def test_conversational_mcp_inspector_fails_closed_on_audit_digest_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_conversational_mcp_face_runtime"]
    globals_ = inspector.__globals__
    original = ROOT / BUILDER["CONVERSATIONAL_MCP_FACE_CLIENT_ACCESS"]
    tampered = tmp_path / original.name
    tampered.write_bytes(original.read_bytes() + b"tamper")
    evidence = {
        (tampered if path == BUILDER["CONVERSATIONAL_MCP_FACE_CLIENT_ACCESS"] else path): digest
        for path, digest in BUILDER["CONVERSATIONAL_MCP_FACE_EVIDENCE_SHA256"].items()
    }
    monkeypatch.setitem(globals_, "CONVERSATIONAL_MCP_FACE_CLIENT_ACCESS", tampered)
    monkeypatch.setitem(globals_, "CONVERSATIONAL_MCP_FACE_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "evidence digest differs" in result["measurement_evidence_error"]


def test_conversational_mcp_inspector_rejects_rehashed_learned_head_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_conversational_mcp_face_runtime"]
    globals_ = inspector.__globals__
    original = ROOT / BUILDER["CONVERSATIONAL_MCP_FACE_RUNTIME"]
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["policy"]["learned_v3_action_head_used"] = True
    tampered = tmp_path / original.name
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    evidence = {
        (tampered if path == BUILDER["CONVERSATIONAL_MCP_FACE_RUNTIME"] else path): (
            _sha256(tampered) if path == BUILDER["CONVERSATIONAL_MCP_FACE_RUNTIME"] else digest
        )
        for path, digest in BUILDER["CONVERSATIONAL_MCP_FACE_EVIDENCE_SHA256"].items()
    }
    monkeypatch.setitem(globals_, "CONVERSATIONAL_MCP_FACE_RUNTIME", tampered)
    monkeypatch.setitem(globals_, "CONVERSATIONAL_MCP_FACE_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "live_conversational_mcp_runtime" in result["measurement_evidence_error"]


def test_conversational_mcp_inspector_rejects_rehashed_oracle_feedback_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = BUILDER["_inspect_conversational_mcp_face_runtime"]
    globals_ = inspector.__globals__
    original = ROOT / BUILDER["CONVERSATIONAL_MCP_FACE_ORACLE_SCORE"]
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["oracle_only_scorer_attestation"]["score_fed_back_to_runtime"] = True
    tampered = tmp_path / original.name
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    evidence = {
        (tampered if path == BUILDER["CONVERSATIONAL_MCP_FACE_ORACLE_SCORE"] else path): (
            _sha256(tampered) if path == BUILDER["CONVERSATIONAL_MCP_FACE_ORACLE_SCORE"] else digest
        )
        for path, digest in BUILDER["CONVERSATIONAL_MCP_FACE_EVIDENCE_SHA256"].items()
    }
    monkeypatch.setitem(globals_, "CONVERSATIONAL_MCP_FACE_ORACLE_SCORE", tampered)
    monkeypatch.setitem(globals_, "CONVERSATIONAL_MCP_FACE_EVIDENCE_SHA256", evidence)

    result = inspector()

    assert result["status"] == "artifact_present_authentication_failed"
    assert result["evidence_authenticated"] is False
    assert "separate_oracle_only_score" in result["measurement_evidence_error"]

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.conversational_mcp_face_score import (
    POLICY_NAME,
    SCHEMA,
    build_score,
)


def test_real_scene1_conversational_mcp_face_score_passes() -> None:
    result = build_score(
        "reports/gemma4/metrics/conversational_mcp_face_scene_000001.json",
        "data/oracle/scene_000001/oracle.json",
        "configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json",
    )

    assert result["schema"] == SCHEMA
    assert result["runtime_result"]["policy"] == POLICY_NAME
    assert result["runtime_result"]["official_mcp_sdk_stdio_action_execution"] is True
    assert result["runtime_result"]["learned_v3_action_head_used"] is False
    assert result["runtime_result"]["gemma_native_function_calling_used"] is False
    assert result["final_pose"] == {
        "position_xyz_m": [0.0, 0.0, 0.0],
        "body_yaw_degrees": pytest.approx(66.92344078354057),
        "stopped": True,
    }
    assert result["oracle_target"]["center_xyz_m"] == pytest.approx(
        [-1.2374088764190674, 0.5234692245721817, 0.6299999952316284]
    )
    assert result["oracle_target"]["desired_yaw_degrees"] == pytest.approx(67.06985993701599)
    assert result["heading"]["absolute_error_degrees"] == pytest.approx(0.14641915347541823)
    assert result["heading"]["maximum_error_degrees"] == 20.0
    assert result["collision"] == {"count": 0, "maximum_count": 0, "passed": True}
    assert result["passed"] is True
    attestation = result["oracle_only_scorer_attestation"]
    assert attestation["runtime_validated_before_oracle_open"] is True
    assert attestation["runtime_process_read_oracle"] is False
    assert attestation["oracle_geometry_loaded_by_scorer_only"] is True
    assert attestation["score_fed_back_to_runtime"] is False


def test_invalid_runtime_fails_before_missing_oracle_is_opened(tmp_path: Path) -> None:
    source = Path("reports/gemma4/metrics/conversational_mcp_face_scene_000001.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["passed"] = False
    invalid = tmp_path / "runtime.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime result lacks"):
        build_score(
            invalid,
            tmp_path / "missing_oracle.json",
            tmp_path / "missing_spec.json",
        )

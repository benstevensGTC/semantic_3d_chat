from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.embodied_hybrid_oracle_score import (
    ARTIFACT,
    build_face_oracle_score,
    create_face_oracle_score,
)


def test_real_scene1_hybrid_oracle_score_is_separated_and_passes() -> None:
    result = build_face_oracle_score(
        "reports/gemma4/metrics/embodied_conversation_hybrid_scene_000001.json",
        "data/oracle/scene_000001/oracle.json",
        "configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json",
    )

    assert result["artifact"] == ARTIFACT
    assert result["final_pose"]["body_yaw_degrees"] == pytest.approx(
        60.490709645513306
    )
    assert result["oracle_target"]["center_xyz_m"] == pytest.approx(
        [-1.2374088764190674, 0.5234692245721817, 0.6299999952316284]
    )
    assert result["oracle_target"]["desired_yaw_degrees"] == pytest.approx(
        67.06985993701599
    )
    assert result["heading"]["absolute_error_degrees"] == pytest.approx(
        6.579150291502685
    )
    assert result["heading"]["maximum_error_degrees"] == 20.0
    assert result["collision"] == {"count": 0, "maximum_count": 0, "passed": True}
    assert result["passed"] is True
    attestation = result["oracle_only_scorer_attestation"]
    assert attestation["oracle_geometry_loaded_by_primary_runtime"] is False
    assert attestation["oracle_geometry_loaded_by_scorer_only"] is True
    assert attestation["score_fed_back_to_runtime"] is False


def test_oracle_score_is_create_once(tmp_path: Path) -> None:
    output = tmp_path / "score.json"
    first = create_face_oracle_score(
        "reports/gemma4/metrics/embodied_conversation_hybrid_scene_000001.json",
        "data/oracle/scene_000001/oracle.json",
        "configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json",
        output,
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == first

    with pytest.raises(FileExistsError):
        create_face_oracle_score(
            "reports/gemma4/metrics/embodied_conversation_hybrid_scene_000001.json",
            "data/oracle/scene_000001/oracle.json",
            "configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json",
            output,
        )
    assert json.loads(output.read_text(encoding="utf-8")) == first

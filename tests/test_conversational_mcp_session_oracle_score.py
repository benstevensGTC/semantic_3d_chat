from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.conversational_mcp_session_oracle_score import (
    SCHEMA,
    build_score,
)

RUNTIME = Path("reports/gemma4/metrics/conversational_mcp_session_smoke_scene_000001.json")
INSPECTION = Path(
    "reports/gemma4/metrics/conversational_mcp_session_smoke_inspection_scene_000001.json"
)
ORACLE = Path("data/oracle/scene_000001/oracle.json")
SPEC = Path("configs/benchmarks/oracle/conversational_mcp_session_scene_000001.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_completed_session_oracle_score_passes_and_binds_all_evidence() -> None:
    result = build_score(RUNTIME, INSPECTION, ORACLE, SPEC)

    assert result["schema"] == SCHEMA
    assert result["passed"] is True
    assert all(result["gates"].values())
    assert result["distance"]["initial_target_distance_m"] == pytest.approx(1.343577596015541)
    assert result["distance"]["final_target_distance_m"] == pytest.approx(0.5978606003207539)
    assert result["distance"]["target_progress_m"] == pytest.approx(0.745716995694787)
    assert result["distance"]["robot_displacement_m"] == pytest.approx(0.7457224678993225)
    assert result["heading"]["face_absolute_error_degrees"] == pytest.approx(0.14641915347541823)
    assert result["heading"]["final_absolute_error_degrees"] == pytest.approx(0.3290505536254229)
    assert result["collision"] == {
        "receipt_count": 11,
        "collision_count": 0,
        "maximum_collisions": 0,
    }
    evidence = result["runtime_evidence"]
    assert evidence["runtime_result_sha256"] == _sha256(RUNTIME)
    assert evidence["inspection_sha256"] == _sha256(INSPECTION)
    assert evidence["inspection_runtime_sha256"] == _sha256(RUNTIME)
    assert evidence["client_forbidden_access_count"] == 0
    assert evidence["server_forbidden_access_count"] == 0
    attestation = result["oracle_only_scorer_attestation"]
    assert attestation["runtime_and_inspection_validated_before_oracle_open"] is True
    assert attestation["runtime_process_read_oracle"] is False
    assert attestation["oracle_geometry_loaded_by_scorer_only"] is True
    assert attestation["score_fed_back_to_runtime"] is False
    assert attestation["runtime_result_modified"] is False
    assert attestation["inspection_result_modified"] is False
    assert attestation["scene_oracle_sha256"] == _sha256(ORACLE)
    assert attestation["scoring_spec_sha256"] == _sha256(SPEC)


def test_invalid_runtime_fails_before_missing_oracle_is_opened(tmp_path: Path) -> None:
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    runtime["passed"] = False
    invalid = tmp_path / "runtime.json"
    invalid.write_text(json.dumps(runtime), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime did not pass"):
        build_score(
            invalid,
            INSPECTION,
            tmp_path / "missing_oracle.json",
            tmp_path / "missing_spec.json",
        )


def test_rehashed_inspection_tamper_fails_before_oracle_open(tmp_path: Path) -> None:
    inspection = json.loads(INSPECTION.read_text(encoding="utf-8"))
    inspection["transcript"]["move_count"] = 999
    invalid = tmp_path / "inspection.json"
    invalid.write_text(json.dumps(inspection), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from a fresh authentication"):
        build_score(
            RUNTIME,
            invalid,
            tmp_path / "missing_oracle.json",
            tmp_path / "missing_spec.json",
        )


def test_oracle_score_does_not_modify_runtime_or_inspection() -> None:
    before_runtime = _sha256(RUNTIME)
    before_inspection = _sha256(INSPECTION)

    build_score(RUNTIME, INSPECTION, ORACLE, SPEC)

    assert _sha256(RUNTIME) == before_runtime
    assert _sha256(INSPECTION) == before_inspection

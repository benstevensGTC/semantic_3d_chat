from __future__ import annotations

import json

import pytest

from semantic_3d_chat.evaluation import navigation_policy_v3_2_preregistration as v32
from semantic_3d_chat.evaluation.navigation_policy_v3_2_preregistration import (
    ACCEPTANCE_GATES,
    CALIBRATION,
    RUNTIME_VERSION,
    authenticate_historical_result,
    authenticate_preregistration,
    build_preregistration,
)


def test_v3_2_preregistration_is_development_only_and_fixes_gates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {
        name: str(tmp_path / f"future_{name}.json")
        for name in v32.SUCCESSOR_OUTPUTS
    }
    monkeypatch.setattr(v32, "SUCCESSOR_OUTPUTS", outputs)
    monkeypatch.setattr(v32, "DEFAULT_RESULT_OUTPUT", str(tmp_path / "future_result.json"))
    payload = build_preregistration()
    destination = tmp_path / "preregistration.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    authenticated = authenticate_preregistration(destination)

    assert authenticated["runtime_version"] == RUNTIME_VERSION == "v3.2"
    assert authenticated["acceptance_gates"] == ACCEPTANCE_GATES
    assert authenticated["calibration"] == CALIBRATION
    assert authenticated["claim_scope"] == {
        "development_calibration": True,
        "same_benchmark_used_for_diagnosis": True,
        "held_out_claim": False,
        "generalization_claim": False,
    }
    assert authenticated["authorized_change"]["task_id_special_case"] is False
    assert authenticated["authorized_change"]["object_vocabulary_added"] is False
    assert authenticated["authorized_change"]["oracle_coordinate_added"] is False
    assert authenticated["authorized_change"]["environmental_text_inputs"] == []
    assert authenticated["benchmark_rerun_completed"] is False
    assert authenticated["runtime_promotion_authorized"] is False


def test_v3_2_production_builder_refuses_existing_sealed_outputs() -> None:
    with pytest.raises(FileExistsError, match="V3.2 output already exists"):
        build_preregistration()


def test_v3_2_post_run_rejection_authenticates_without_current_source_claim() -> None:
    result = authenticate_historical_result()

    assert result["status"] == "rejected"
    assert result["passed"] is False
    assert result["runtime_promotion_authorized"] is False
    assert result["claim_scope"]["held_out_claim"] is False

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v66_internal_validation_preregistration as contract

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs/experiments/v66_internal_validation_preregistration.json"


def test_v66_internal_validation_preregistration_matches_public_pin() -> None:
    value = contract.validate_v66_internal_validation_preregistration(PREREGISTRATION)

    assert value["thresholds"]["internal_validation"] == (contract.INTERNAL_VALIDATION_THRESHOLDS)
    assert value["thresholds"]["scene_swap"] == contract.SCENE_SWAP_THRESHOLDS
    assert value["candidate_contract"]["control_runtime_schema_version"] == 7
    assert value["candidate_contract"]["expected_checkpoint_path"] == (
        "data_gemma4/checkpoints/gemma4_v66b_allrow_always_on_control"
    )
    assert value["candidate_contract"]["expected_training_report_path"] == (
        "reports/gemma4/metrics/v66b_allrow_always_on_distillation.json"
    )
    assert value["controls"]["reference_file_is_last_input_opened"] is True
    assert value["controls"]["retry_permitted"] is False


def test_v66_internal_validation_preregistration_fails_closed_on_mutation(
    tmp_path: Path,
) -> None:
    value = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    value["thresholds"]["internal_validation"]["canonical_exact"]["minimum"] -= 1
    changed = tmp_path / "preregistration.json"
    changed.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its pin"):
        contract.validate_v66_internal_validation_preregistration(changed)


def test_v66_internal_validation_preregistration_exposes_no_evaluation_input() -> None:
    module_source = Path(contract.__file__).read_text(encoding="utf-8")

    assert "scorer_only/" not in module_source
    assert "data_gemma4/maps/" not in module_source
    assert "data_gemma4/qa/" not in module_source
    assert "data_gemma4/oracle/" not in module_source
    assert "predicted_answer" not in module_source

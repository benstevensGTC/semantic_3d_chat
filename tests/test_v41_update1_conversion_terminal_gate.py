from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v41_update1_conversion_terminal_gate as gate


def test_retry_terminal_authenticates_failure_and_denies_restricted_access() -> None:
    report = gate.load_materialized_report()
    assert report["artifact"] == "v41_update1_conversion_terminal_gate"
    assert report["seal_revision"] == 4
    assert report["passed"] is True
    assert report["only_exact_successor_authorized"] == (
        "v41_retry1_train_only_projected_gradient_continuation"
    )
    assert report[
        "v41_retry1_train_only_projected_gradient_continuation_authorized"
    ] is True
    assert report["validation_access_authorized"] is False
    assert report["oracle_access_authorized"] is False
    assert report["final_test_access_authorized"] is False
    assert report["selector_execution_authorized"] is False
    predecessor = report["predecessor_failure"]
    assert predecessor["failed_before_optimizer_step"] == 1
    assert predecessor["optimizer_step_executed"] is False
    assert predecessor["checkpoint_written"] is False
    assert predecessor["target_and_frozen_state_unchanged"] is True
    assert predecessor["raw_cpu_first_diagnostic_finite"] is True
    assert predecessor["combined_mps_to_cpu_float64_projection_nonfinite"] is True
    assert predecessor["file_sha256"] == gate.FAILED_FILES


def test_retry_terminal_binds_cpu_first_mps_repair_and_sibling_root() -> None:
    authorization = gate.load_materialized_report()["conditional_successor_authorization"]
    assert authorization["authorization_id"] == (
        "v41_retry1_cpu_first_projected_gradient_l14_lora_b"
    )
    assert authorization["authorized_output_root"] == str(gate.RETRY_ROOT)
    assert authorization["cpu_first_mps_conversion_required"] is True
    assert authorization["mps_regression"] == {
        "full_suite_collected_and_passed": 19,
        "non_source_retry_regression_subset_passed": 18,
        "raw_feasible_mask_zero_bit_exact_tested": True,
        "conflicting_nonzero_mask_cpu_projection_mps_cast_clip_tested": True,
        "live_shape_4096_by_4_tested": True,
    }
    assert authorization["fixed_repair_file_sha256"] == {
        str(gate.FIXED_TRAINER): gate.FIXED_TRAINER_SHA256,
        str(gate.FIXED_TRAINING_TEST): gate.FIXED_TRAINING_TEST_SHA256,
    }
    retry_root = gate._resolve(gate.RETRY_ROOT)
    assert retry_root.is_dir()
    assert sorted(path.name for path in retry_root.iterdir()) == ["update_000", "update_008"]


def test_retry_terminal_refuses_reauthorization_after_successor_ran(
    tmp_path: Path,
) -> None:
    rev3 = gate._resolve(gate.REV3_REPORT)
    before = gate._sha256(rev3)
    output = tmp_path / "retry_terminal.json"
    with pytest.raises(ValueError, match="nonempty"):
        gate.write_report(output)
    assert not output.exists()
    assert gate._sha256(rev3) == before == gate.REV3_SHA256
    assert list(tmp_path.iterdir()) == []


def test_materialized_retry_terminal_is_hash_pinned() -> None:
    path = gate._resolve(gate.DEFAULT_OUTPUT)
    assert gate._sha256(path) == gate.MATERIALIZED_REPORT_SHA256
    assert json.loads(path.read_text(encoding="utf-8")) == gate.load_materialized_report()


def test_retry_terminal_fails_closed_when_fixed_trainer_pin_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "FIXED_TRAINER_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="repair source"):
        gate.build_report()

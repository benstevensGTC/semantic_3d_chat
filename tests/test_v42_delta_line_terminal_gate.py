from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v42_delta_line_terminal_gate as gate


def test_terminal_authenticates_negative_screen() -> None:
    report = gate.build_report()
    assert report["passed"] is True
    assert report["negative_result"] == {
        "fixed_alpha_grid": gate._ALPHAS,
        "candidate_count": 9,
        "teacher_eligible_candidate_count": 0,
        "endpoint_replay_exact": True,
        "all_candidates_restored_u0": True,
        "no_optimizer_gradient_checkpoint_or_restricted_access": True,
    }
    assert report["training_authorized"] is False
    assert report["validation_access_authorized"] is False
    assert report["selector_execution_authorized"] is False


def test_only_exact_v43_no_step_screen_is_authorized() -> None:
    report = gate.build_report()
    auth = report["conditional_successor_authorization"]
    assert report["only_exact_successor_authorized"] == (
        "v43_aggregate_projected_train_only_no_step_screen"
    )
    assert auth["gradient_surface"] == {
        "target_tensor": "layer14_q_proj_lora_b_only",
        "target_parameter_count": 16_384,
        "broad_component": "mean_48_unchanged_rows_times_1",
        "answer_component": "mean_8_priority_pair_answer_nll_times_0.5",
        "side_component": "mean_8_priority_pair_side_hinge_times_8",
        "cross_component": "mean_all_25_pair_cross_hinge_times_56",
        "autograd_api": "torch.autograd.grad",
        "parameter_grad_accumulation_authorized": False,
        "optimizer_authorized": False,
    }
    assert auth["projection"]["fixed_scalar_steps"] == gate._V43_STEPS
    assert auth["projection"]["adaptive_refinement_authorized"] is False
    scope = auth["diagnostic_scope"]
    assert scope["temporary_target_substitution_authorized"] is True
    assert scope["exact_u0_restoration_after_every_candidate"] is True
    assert scope["optimizer_step_authorized"] is False
    assert scope["checkpoint_write_authorized"] is False
    assert scope["validation_access_authorized"] is False


def test_materialized_terminal_replays_exactly() -> None:
    path = gate._resolve(gate.DEFAULT_OUTPUT)
    if not path.exists():
        pytest.skip("terminal not materialized yet")
    assert json.loads(path.read_text(encoding="utf-8")) == gate.build_report()


def test_atomic_write_does_not_change_inputs(tmp_path: Path) -> None:
    before = {name: gate._sha256(gate._resolve(name)) for name in gate._PINS}
    output = tmp_path / "gate.json"
    assert gate.write_report(output)["passed"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    after = {name: gate._sha256(gate._resolve(name)) for name in gate._PINS}
    assert before == after == gate._PINS


def test_changed_screen_pin_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(gate._PINS, str(gate.DEFAULT_SCREEN), "0" * 64)
    with pytest.raises(ValueError, match="changed"):
        gate.build_report()

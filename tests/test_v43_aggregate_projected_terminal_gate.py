from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import (
    v43_aggregate_projected_terminal_gate as gate,
)


def test_terminal_authenticates_exact_negative_screen() -> None:
    report = gate.build_report()
    assert report["passed"] is True
    assert report["screen_sha256"] == gate._PINS[str(gate.DEFAULT_SCREEN)]
    assert report["negative_result"] == {
        "fixed_scalar_steps": gate._STEPS,
        "candidate_count": 8,
        "teacher_eligible_candidate_count": 0,
        "update_zero_endpoint_replay_exact": True,
        "all_candidates_restored_exact_u0": True,
        "gradient_measurement_left_source_exact": True,
        "no_optimizer_checkpoint_selector_or_restricted_access": True,
    }
    assert report["validation_access_authorized"] is False
    assert report["selector_execution_authorized"] is False
    assert report["runtime_promotion_authorized"] is False


def test_only_exact_bounded_v44_surface_is_authorized() -> None:
    auth = gate.build_report()["conditional_successor_authorization"]
    surface = auth["trainable_surface"]
    assert surface["parameter_names"] == [
        "block_cross_residual.w_o",
        "lora_banks.extension_v28_stage_b_query.adapters.1.lora_a",
        "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b",
    ]
    assert surface["parameter_shapes"] == [[256, 1536], [4, 1536], [4096, 4]]
    assert surface["total_parameter_count"] == 415_744
    assert auth["optimizer"] == {
        "implementation": "fresh_torch_adamw_two_groups",
        "source_optimizer_loaded": False,
        "scene_readout_learning_rate": 2.5e-5,
        "query_learning_rate": 2e-5,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "per_group_gradient_clip_norm": 1.0,
    }
    assert auth["schedule"]["checkpoint_steps"] == [0, 4, 8, 16]
    assert auth["schedule"]["update8_must_pass_before_updates_9_through_16"] is True
    assert auth["scope"]["validation_access_authorized"] is False
    assert auth["scope"]["runtime_promotion_authorized"] is False


def test_materialized_terminal_replays_exactly() -> None:
    path = gate._resolve(gate.DEFAULT_OUTPUT)
    if not path.exists():
        pytest.skip("terminal not materialized yet")
    assert json.loads(path.read_text(encoding="utf-8")) == gate.build_report()


def test_write_refuses_unpinned_or_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="pinned"):
        gate.write_report(tmp_path / "other.json")
    existing = tmp_path / "terminal.json"
    existing.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", existing)
    with pytest.raises(FileExistsError, match="one-shot"):
        gate.write_report(existing)


def test_changed_screen_pin_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(gate._PINS, str(gate.DEFAULT_SCREEN), "0" * 64)
    with pytest.raises(ValueError, match="changed"):
        gate.build_report()

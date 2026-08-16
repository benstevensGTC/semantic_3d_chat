from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v34_terminal_gate import (
    _validate_checkpoint_sequence,
    audit_v34_update32,
)

CONFIG = Path("configs/experiments/gemma4_diverse28_base_surface_v34.yaml")
CHECKPOINT_ROOT = Path("data_gemma4/checkpoints/gemma4_v34_diverse28_base_surface")
REPORT = Path("reports/gemma4/metrics/v34_update32_terminal_gate.json")


def test_v34_terminal_gate_replays_exact_update32_without_environment_inputs() -> None:
    report = audit_v34_update32()
    assert report["passed"] is True
    assert report["gemma_loaded"] is False
    assert report["scene_maps_loaded"] is False
    assert report["qa_loaded"] is False
    assert report["oracle_loaded"] is False
    assert report["final_test_scenes_touched"] is False
    assert report["observed_saved_optimizer_steps"] == [0, 8, 16, 24, 32]
    assert report["stopped_at_optimizer_step"] == 32
    assert report["no_update_040_or_later"] is True
    gate = report["update32_gate_evidence"]
    assert gate["changed_selectivity_ratio_geometric_mean"] == 1.00003981590271
    assert gate["changed_selectivity_over_1_02_count"] == 0
    assert gate["passed"] is False


def test_v34_terminal_gate_proves_only_exact_four_tensor_surface_changed() -> None:
    transition = audit_v34_update32()["tensor_transition"]
    assert transition["changed_tensor_names"] == [
        "dense_sidecar_adapter.base_norm.bias",
        "dense_sidecar_adapter.base_norm.weight",
        "dense_sidecar_adapter.base_projection.bias",
        "dense_sidecar_adapter.base_projection.weight",
    ]
    assert transition["changed_tensor_count"] == 4
    assert transition["changed_parameter_count"] == 199_808
    assert transition["all_inherited_tensors_frozen_at_every_saved_arm"] is True


def test_v34_terminal_gate_authorizes_only_v35_block_cross_residual() -> None:
    report = audit_v34_update32()
    authorization = report["conditional_authorization"]
    assert authorization == {
        "authorized": True,
        "stage": "v35_block_cross_residual",
        "scope": "exact_zero_block_token_cross_residual_only",
        "all_other_followup_architectures_authorized": False,
        "chat_promotion_authorized": False,
        "final_test_access_authorized": False,
    }
    assert report["conditional_v35_block_cross_residual_authorized"] is True
    assert report["v34_chat_promotion_eligible"] is False


def test_v34_terminal_report_is_exact_replay() -> None:
    assert json.loads(REPORT.read_text(encoding="utf-8")) == audit_v34_update32()
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == (
        "631d5cee0253efef9060d66bcb66941f3fbcfdae7c38039b80cb88db2d737695"
    )


def test_v34_terminal_gate_rejects_changed_config_bytes(tmp_path: Path) -> None:
    changed = tmp_path / "v34.yaml"
    changed.write_bytes(CONFIG.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="config bytes differ"):
        audit_v34_update32(changed, CHECKPOINT_ROOT)


def test_v34_terminal_gate_rejects_any_later_update_directory(tmp_path: Path) -> None:
    for step in (0, 8, 16, 24, 32, 40):
        (tmp_path / f"update_{step:03d}").mkdir()
    with pytest.raises(ValueError, match="stopped at its contiguous update-32"):
        _validate_checkpoint_sequence(tmp_path)


def test_v34_terminal_make_target_has_no_selection_or_final_dependency() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("gemma4-v34-seal-update32:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "semantic_3d_chat.evaluation.v34_terminal_gate" in recipe
    assert "select" not in recipe
    assert "final" not in recipe

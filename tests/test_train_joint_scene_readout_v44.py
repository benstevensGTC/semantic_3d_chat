from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.training import train_joint_scene_readout_v44 as v44

CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/gemma4_diverse28_joint_scene_readout_v44.yaml"
)
TERMINAL_SHA256 = (
    "013fbe79ac42e842e83989e33f132b9ff3529746a8045feb212ded32e50a2cc2"
)


def _pair_metrics(
    *, complete: int, positive: int, cross: int, physical: int
) -> dict[str, object]:
    rows = []
    families = ["book_support"] * 4 + ["picture_support"] * 4 + ["mirror_lr"] * 17
    for index, family in enumerate(families):
        rows.append(
            {
                "pair_id": f"pair_{index:02d}",
                "question_key": f"question_{index:02d}",
                "family": family,
                "side_margins": [0.5, 0.5],
            }
        )
    return {
        "complete_units": complete,
        "positive_sides": positive,
        "cross_prefix_complete_units": cross,
        "complete_physical_pair_coverage": physical,
        "complete_units_by_family": {
            "book_support": 1,
            "picture_support": 0,
            "mirror_lr": max(0, complete - 1),
        },
        "units": rows,
    }


def test_exact_config_surface_objective_and_source_are_locked() -> None:
    config = load_config(CONFIG)
    settings = v44.v44_settings(config)
    contract = v44.v44_contract(config)
    assert settings.optimizer_steps == 16
    assert settings.checkpoint_steps == (0, 4, 8, 16)
    assert settings.scene_readout_learning_rate == 2.5e-5
    assert settings.query_learning_rate == 2e-5
    assert settings.source_prefix_trust_weight == 0.001
    assert contract.configured_terminal_sha256 == TERMINAL_SHA256
    assert contract.source_checkpoint.name == "update_000"
    assert "v41_retry1" in str(contract.source_checkpoint)
    assert contract.authorized_parameter_shapes == (
        (256, 1536),
        (4, 1536),
        (4096, 4),
    )
    assert contract.total_parameter_count == 415_744


def test_v43_terminal_deep_authorization_is_exact() -> None:
    config = load_config(CONFIG)
    terminal = v44.require_v43_terminal_gate(
        config, expected_sha256=TERMINAL_SHA256
    )
    authorization = terminal["authorization"]
    assert terminal["exact_authorization_fields_verified"] is True
    assert authorization["only_exact_action"] == (
        "one_bounded_v44_joint_scene_readout_training_pilot"
    )
    assert authorization["trainable_surface"]["total_parameter_count"] == 415_744
    assert authorization["scope"]["validation_access_authorized"] is False
    assert authorization["optimizer"]["per_group_gradient_clip_norm"] == 1.0
    with pytest.raises(ValueError, match="exact pinned"):
        v44.require_v43_terminal_gate(config, expected_sha256="0" * 64)


def test_contract_rejects_any_objective_change() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["training"]["v44_joint_scene_readout"]["side_hinge_weight"] = 7.0
    with pytest.raises(ValueError, match="settings changed"):
        v44.v44_settings(config)


def test_authenticated_retry1_source_inventory_and_hashes() -> None:
    config = load_config(CONFIG)
    tensors, metadata = v44._source_tensors(v44.v44_contract(config))
    assert len(tensors) == 179
    assert metadata["optimizer_step"] == 0
    assert v44.tensor_state_sha256(tensors) == v44._SOURCE_FULL_SHA256
    authorized = {name: tensors[name] for name in v44._PARAMETER_NAMES}
    assert v44.tensor_state_sha256(authorized) == v44._SOURCE_AUTHORIZED_SHA256


def test_fresh_adamw_has_two_exact_groups_and_empty_state() -> None:
    config = load_config(CONFIG)
    settings = v44.v44_settings(config)
    scene = torch.nn.Parameter(torch.zeros((256, 1536)))
    query_a = torch.nn.Parameter(torch.zeros((4, 1536)))
    query_b = torch.nn.Parameter(torch.zeros((4096, 4)))
    optimizer = v44.v44_optimizer([scene], [query_a, query_b], settings)
    audit = v44.v44_optimizer_audit(optimizer)
    assert optimizer.state == {}
    assert audit["group_names"] == ["scene_readout", "layer14_query"]
    assert audit["learning_rates"] == [2.5e-5, 2e-5]
    assert audit["parameter_counts"] == [393_216, 22_528]
    assert audit["gradient_clip_method"] == "independent_per_optimizer_group"
    assert audit["source_optimizer_loaded"] is False


def test_update8_requires_both_parameter_groups_to_change() -> None:
    metrics = _pair_metrics(complete=9, positive=34, cross=17, physical=5)
    passed = v44.v44_update8_gate(
        pair_metrics=metrics,
        broad_nll=2.0,
        source_broad_nll=2.0,
        source_priority_deficit=1.0,
        scene_readout_state_changed=True,
        query_state_changed=True,
        frozen_state_exact=True,
        trust_rms=0.2,
    )
    assert passed["passed"] is True
    assert "source_prefix_trust_rms_at_most_0_05" not in passed["checks"]
    failed = v44.v44_update8_gate(
        pair_metrics=metrics,
        broad_nll=2.0,
        source_broad_nll=2.0,
        source_priority_deficit=1.0,
        scene_readout_state_changed=False,
        query_state_changed=True,
        frozen_state_exact=True,
        trust_rms=0.0,
    )
    assert failed["both_authorized_parameter_groups_changed"] is False
    assert failed["passed"] is False


def test_update16_uses_exact_v43_teacher_and_greedy_thresholds() -> None:
    metrics = _pair_metrics(complete=10, positive=35, cross=17, physical=5)
    greedy = {"complete_units": 5, "broad_exact_correct": 23, "broad_row_count": 48}
    gate = v44.v44_update16_gate(
        update8_gate={"passed": True},
        pair_metrics=metrics,
        broad_nll=2.02,
        source_broad_nll=2.0,
        source_priority_deficit=0.5,
        greedy_metrics=greedy,
        scene_readout_state_changed=True,
        query_state_changed=True,
        frozen_state_exact=True,
        trust_rms=0.3,
    )
    assert gate["passed"] is True
    assert gate["priority_teacher_deficit_improved_at_least_0_5"] is True
    greedy["broad_exact_correct"] = 22
    failed = v44.v44_update16_gate(
        update8_gate={"passed": True},
        pair_metrics=metrics,
        broad_nll=2.0,
        source_broad_nll=2.0,
        source_priority_deficit=0.5,
        greedy_metrics=greedy,
        scene_readout_state_changed=True,
        query_state_changed=True,
        frozen_state_exact=True,
        trust_rms=0.0,
    )
    assert failed["broad_greedy_exact_correct_at_least_23_of_48"] is False
    assert failed["passed"] is False


def test_failed_gate_produces_auditable_bounded_stop_reason() -> None:
    assert (
        v44.v44_stop_reason(
            8, update8_gate={"passed": False}, update16_gate=None
        )
        == "update8_train_only_gate_failed"
    )
    assert (
        v44.v44_stop_reason(
            8, update8_gate={"passed": True}, update16_gate=None
        )
        is None
    )
    assert (
        v44.v44_stop_reason(
            16,
            update8_gate={"passed": True},
            update16_gate={"passed": False},
        )
        == "update16_train_only_gate_failed"
    )
    with pytest.raises(RuntimeError, match="lacks"):
        v44.v44_stop_reason(8, update8_gate=None, update16_gate=None)


def test_saved_optimizer_steps_reports_only_materialized_checkpoints() -> None:
    stopped_history = [
        {"optimizer_update": 0, "saved_checkpoint": True},
        {"optimizer_update": 1, "saved_checkpoint": False},
        {"optimizer_update": 4, "saved_checkpoint": True},
        {"optimizer_update": 8, "saved_checkpoint": True},
    ]
    assert v44.v44_saved_optimizer_steps(stopped_history) == [0, 4, 8]

    completed_history = [
        *stopped_history,
        {"optimizer_update": 9, "saved_checkpoint": False},
        {"optimizer_update": 16, "saved_checkpoint": True},
    ]
    assert v44.v44_saved_optimizer_steps(completed_history) == [0, 4, 8, 16]


def test_preflight_loads_no_model_map_optimizer_or_restricted_data() -> None:
    result = v44.preflight_v44(CONFIG, v43_terminal_sha256=TERMINAL_SHA256)
    assert result["passed"] is True
    assert result["source_tensor_count"] == 179
    assert result["train_question_count"] == 384
    assert result["train_pair_unit_count"] == 25
    assert result["bounded_schedule_steps"] == list(range(1, 17))
    assert result["gemma_loaded"] is False
    assert result["scene_maps_loaded"] is False
    assert result["validation_qa_loaded"] is False
    assert result["oracle_loaded"] is False
    assert result["final_test_scenes_touched"] is False
    assert result["forbidden_file_accesses"] == []
    assert not any("/maps/" in path for path in result["loaded_files"])


def test_preflight_rejects_an_equivalent_config_at_an_unapproved_path(
    tmp_path: Path,
) -> None:
    copied = tmp_path / CONFIG.name
    copied.write_bytes(CONFIG.read_bytes())
    with pytest.raises(ValueError, match="path or bytes"):
        v44.preflight_v44(copied, v43_terminal_sha256=TERMINAL_SHA256)


def test_trainer_loads_authenticated_v44_source_not_v40_contract_path() -> None:
    path = Path(v44.__file__)
    source = path.read_text(encoding="utf-8")
    assert "config, approved, contract.source_checkpoint, source_tensors" in source
    assert "v41_contract(config).source_checkpoint" not in source


def test_checkpoint_metadata_refreshes_both_runtime_enforced_hashes() -> None:
    path = Path(v44.__file__)
    source = path.read_text(encoding="utf-8")
    assert '"block_cross_residual_state_sha256": block_core.state_sha256()' in source
    assert '"frozen_block_cross_source_stack_state_sha256": (' in source
    core = v44.BlockCrossResidual()
    other = torch.nn.Linear(3, 2)
    bundle = type(
        "Bundle",
        (),
        {"checkpoint_modules": {"block_cross_residual": core, "other": other}},
    )()
    before = v44.block_source_stack_state_sha256(bundle, core)
    with torch.no_grad():
        other.weight.add_(1.0)
    after = v44.block_source_stack_state_sha256(bundle, core)
    assert before != after


def test_decoder_recomputation_mode_is_preserved_for_training() -> None:
    decoder = torch.nn.Linear(2, 2)
    decoder.is_gradient_checkpointing = True
    bundle = SimpleNamespace(
        language=SimpleNamespace(
            decoder_module=decoder,
            decoder_gradient_checkpointing_enabled=True,
        )
    )
    decoder.eval()
    audit = v44.configure_v44_decoder_training_mode(bundle)
    assert decoder.training is True
    assert audit == {
        "decoder_training": True,
        "decoder_gradient_checkpointing_enabled": True,
        "recomputation_active": True,
    }

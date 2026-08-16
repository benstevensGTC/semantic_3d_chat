from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.training import train_retention_repair_v45 as v45
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
from semantic_3d_chat.training.train_block_cross_v35 import load_v35_train_qa_records
from semantic_3d_chat.training.train_projected_gradient_v41 import v41_loader_config

CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/gemma4_diverse28_retention_repair_v45.yaml"
)
TERMINAL_SHA256 = (
    "b968c46c686051e864417b7539db7e90160a1f0b4639af031d02aab005643b67"
)


def test_exact_config_source_surface_schedule_and_gates_are_locked() -> None:
    config = load_config(CONFIG)
    settings = v45.v45_settings(config)
    contract = v45.v45_contract(config)
    assert settings.optimizer_steps == 8
    assert settings.checkpoint_steps == (0, 2, 4, 6, 8)
    assert settings.scene_readout_learning_rate == 1.0e-5
    assert settings.query_learning_rate == 8.0e-6
    assert settings.retention_weight == 8.0
    assert settings.retention_side_floor == 0.125
    assert settings.retention_book_cross_floor == 0.025
    assert contract.configured_terminal_sha256 == TERMINAL_SHA256
    assert contract.source_checkpoint.name == "update_008"
    assert "v44_joint_scene_readout" in str(contract.source_checkpoint)
    assert contract.construction_source_checkpoint.name == "update_000"
    assert "v41_retry1" in str(contract.construction_source_checkpoint)
    assert contract.authorized_parameter_shapes == (
        (256, 1536),
        (4, 1536),
        (4096, 4),
    )
    assert contract.total_parameter_count == 415_744


def test_v44_terminal_deep_authorization_is_exact() -> None:
    config = load_config(CONFIG)
    terminal = v45.require_v44_terminal_gate(
        config, expected_sha256=TERMINAL_SHA256
    )
    authorization = terminal["authorization"]
    assert terminal["exact_authorization_fields_verified"] is True
    assert authorization == v45._expected_v45_authorization()
    assert authorization["only_exact_action"] == (
        "one_bounded_v45_train_only_retention_repair_pilot"
    )
    assert authorization["scope"]["validation_access_authorized"] is False
    assert authorization["source_optimizer_policy"] == {
        "source_optimizer_file_present_and_authenticated": True,
        "source_optimizer_file_open_authorized_by_v45": False,
        "source_optimizer_deserialization_authorized": False,
        "source_optimizer_state_loading_authorized": False,
        "fresh_optimizer_required": True,
    }
    with pytest.raises(ValueError, match="exact pinned"):
        v45.require_v44_terminal_gate(config, expected_sha256="0" * 64)


def test_contract_rejects_objective_schedule_anchor_and_gate_changes() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["training"]["v45_retention_repair"]["retention_weight"] = 7.0
    with pytest.raises(ValueError, match="settings changed"):
        v45.v45_settings(config)

    config = copy.deepcopy(load_config(CONFIG))
    config["v45_retention_repair"]["target_question_key_schedule"][0] = "cfq_bad"
    with pytest.raises(ValueError, match="contract changed"):
        v45.v45_contract(config)

    config = copy.deepcopy(load_config(CONFIG))
    config["v45_retention_repair"]["fragile_side_constraints"][0][
        "retention_floor"
    ] = 0.124
    with pytest.raises(ValueError, match="contract changed"):
        v45.v45_contract(config)

    config = copy.deepcopy(load_config(CONFIG))
    config["v45_retention_repair"]["update8_gate"][
        "u8_prefix_trust_rms_maximum"
    ] = 0.003
    with pytest.raises(ValueError, match="contract changed"):
        v45.v45_contract(config)


def test_authenticated_v44_u8_source_inventory_hashes_and_no_optimizer_read() -> None:
    config = load_config(CONFIG)
    contract = v45.v45_contract(config)
    tensors, metadata = v45._v44_u8_source(contract)
    assert len(tensors) == 179
    assert metadata["optimizer_step"] == 8
    assert v45.tensor_state_sha256(tensors) == v45._SOURCE_FULL_SHA256
    authorized = {name: tensors[name] for name in v45._PARAMETER_NAMES}
    assert v45.tensor_state_sha256(authorized) == v45._SOURCE_AUTHORIZED_SHA256
    assert "optimizer.pt" not in contract.source_file_sha256
    source = Path(v45.__file__).read_text(encoding="utf-8")
    function = source[source.index("def _v44_u8_source"):source.index("def _unit_index")]
    assert "torch.load" not in function
    assert "optimizer.pt" not in function


def test_retention_formula_uses_two_separate_means_and_gradients() -> None:
    side = torch.zeros(8, requires_grad=True)
    cross = torch.zeros(4, requires_grad=True)
    total, side_mean, cross_mean = v45.retention_hinge_from_selected_margins(
        side, cross
    )
    assert float(side_mean.detach()) == pytest.approx(0.125)
    assert float(cross_mean.detach()) == pytest.approx(0.025)
    assert float(total.detach()) == pytest.approx(0.15)
    (8.0 * total).backward()
    assert torch.allclose(side.grad, torch.full((8,), -1.0))
    assert torch.allclose(cross.grad, torch.full((4,), -2.0))
    with pytest.raises(ValueError, match="shapes"):
        v45.retention_hinge_from_selected_margins(torch.zeros(7), cross.detach())


def test_exact_fixed_schedule_and_anchor_inventory() -> None:
    config = load_config(CONFIG)
    loader = v41_loader_config(config)
    records, _audit = load_v35_train_qa_records(loader)
    units = build_exact_question_pair_units(records)
    unit_audit = v45.validate_v45_unit_inventory(units)
    schedule, schedule_audit, broad = v45.build_v45_schedule(
        records, units, config=config
    )
    assert len(broad) == 48
    assert [row.optimizer_step for row in schedule] == list(range(1, 9))
    assert [row.target_unit.question_key for row in schedule] == list(
        v45._TARGET_QUESTION_KEYS
    )
    assert [row.broad_record.question_id for row in schedule] == list(
        v45._BROAD_QUESTION_IDS
    )
    assert schedule_audit["fixed_broad_rows"] == list(range(9, 17))
    assert unit_audit["unit_count"] == 25
    assert len(unit_audit["fragile_side_constraints"]) == 8
    assert len(unit_audit["book_cross_constraints"]) == 4


def test_live_update_zero_attestation_matches_pinned_v44_u8_diagnostics() -> None:
    metadata = json.loads(
        (
            PROJECT_ROOT
            / "data_gemma4/checkpoints/"
            "gemma4_v44_joint_scene_readout_l14_query/update_008/metadata.json"
        ).read_text(encoding="utf-8")
    )
    row = metadata["history"][-1]
    attestation = v45.validate_v45_update_zero_baseline(
        pair_metrics=row["pair_metrics"], broad_nll=row["broad_diagnostic_nll"]
    )
    assert attestation["passed"] is True
    assert attestation["both_lost_side_margins_exact_zero"] is True
    assert attestation["computed_before_optimizer_construction"] is True
    changed = copy.deepcopy(row["pair_metrics"])
    changed["positive_sides"] = 33
    with pytest.raises(RuntimeError, match="baseline changed"):
        v45.validate_v45_update_zero_baseline(
            pair_metrics=changed, broad_nll=row["broad_diagnostic_nll"]
        )


def _gate_metrics(*, lost_positive: bool = True) -> dict[str, object]:
    required = {
        "cfq_a578dc166be9a217": ("pair_000005", "other"),
        "cfq_0a79d507273195ef": ("pair_000006", "other"),
        "cfq_5c84a2c27d2be251": ("pair_000006", "other"),
        "cfq_736067b51ce93c49": ("pair_000007", "other"),
        "cfq_997610c185204121": ("pair_000007", "other"),
        "cfq_699675ceeaf65406": ("pair_000016", "mirror_lr"),
        "cfq_90b3d9852a93ce2a": ("pair_000018", "other"),
        "cfq_13b1138d14c52a7c": ("pair_000015", "book_support"),
        "cfq_a1c673a1197a0961": ("pair_000015", "book_support"),
    }
    rows: list[dict[str, object]] = []
    for key, (pair_id, family) in required.items():
        rows.append(
            {
                "pair_id": pair_id,
                "question_key": key,
                "family": family,
                "side_margins": [0.5, 0.5],
                "cross_prefix_margins": [0.5, 0.5],
            }
        )
    for index in range(16):
        family = (
            "book_support"
            if index < 2
            else "picture_support"
            if index < 6
            else "other"
        )
        rows.append(
            {
                "pair_id": f"pair_fill_{index:02d}",
                "question_key": f"cfq_fill_{index:02d}",
                "family": family,
                "side_margins": [0.5, 0.5],
                "cross_prefix_margins": [0.5, 0.5],
            }
        )
    assert len(rows) == 25
    if not lost_positive:
        next(
            row
            for row in rows
            if row["question_key"] == "cfq_699675ceeaf65406"
        )["side_margins"] = [0.5, 0.0]
    return {
        "unit_count": 25,
        "complete_units": 10,
        "positive_sides": 35,
        "cross_prefix_complete_units": 17,
        "complete_physical_pair_coverage": 5,
        "complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 2,
            "picture_support": 0,
        },
        "cross_prefix_complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 2,
            "picture_support": 0,
        },
        "units": rows,
    }


def test_gate4_and_gate8_apply_full_train_thresholds_and_lost_side_reassertion() -> None:
    metrics = _gate_metrics()
    gate4 = v45.v45_update4_gate(
        pair_metrics=metrics,
        broad_nll=2.9,
        scene_readout_state_changed=True,
        query_state_changed=True,
        frozen_state_exact=True,
        trust_rms=0.001,
    )
    assert gate4["passed"] is True
    greedy = {"complete_units": 5, "broad_exact_correct": 23, "broad_row_count": 48}
    gate8 = v45.v45_update8_gate(
        update4_gate=gate4,
        pair_metrics=metrics,
        broad_nll=2.9,
        greedy_metrics=greedy,
        scene_readout_state_changed=True,
        query_state_changed=True,
        frozen_state_exact=True,
        trust_rms=0.002,
    )
    assert gate8["passed"] is True
    assert gate8["recorded_update4_gate_passed"] is True
    assert gate8["full_train_pair_unit_count"] == 25
    assert gate8["full_broad_nll_row_count"] == 48

    failed = v45.v45_update8_gate(
        update4_gate=gate4,
        pair_metrics=_gate_metrics(lost_positive=False),
        broad_nll=2.9,
        greedy_metrics=greedy,
        scene_readout_state_changed=True,
        query_state_changed=True,
        frozen_state_exact=True,
        trust_rms=0.001,
    )
    assert failed["both_lost_side_margins_remain_strictly_positive"] is False
    assert failed["passed"] is False


def test_fresh_adamw_has_two_exact_groups_and_empty_state() -> None:
    settings = v45.v45_settings(load_config(CONFIG))
    scene = torch.nn.Parameter(torch.zeros((256, 1536)))
    query_a = torch.nn.Parameter(torch.zeros((4, 1536)))
    query_b = torch.nn.Parameter(torch.zeros((4096, 4)))
    optimizer = v45.v45_optimizer([scene], [query_a, query_b], settings)
    audit = v45.v45_optimizer_audit(optimizer)
    assert optimizer.state == {}
    assert audit["learning_rates"] == [1.0e-5, 8.0e-6]
    assert audit["parameter_counts"] == [393_216, 22_528]
    assert audit["source_optimizer_loaded"] is False


def test_graceful_stop_and_truthful_saved_steps() -> None:
    assert v45.v45_stop_reason(
        4, update4_gate={"passed": False}, update8_gate=None
    ) == "update4_train_only_gate_failed"
    assert v45.v45_stop_reason(
        4, update4_gate={"passed": True}, update8_gate=None
    ) is None
    assert v45.v45_stop_reason(
        8, update4_gate={"passed": True}, update8_gate={"passed": False}
    ) == "update8_train_only_gate_failed"
    history = [
        {"optimizer_update": 0, "saved_checkpoint": True},
        {"optimizer_update": 1, "saved_checkpoint": False},
        {"optimizer_update": 2, "saved_checkpoint": True},
        {"optimizer_update": 3, "saved_checkpoint": False},
        {"optimizer_update": 4, "saved_checkpoint": True},
    ]
    assert v45.v45_saved_optimizer_steps(history) == [0, 2, 4]


def test_preflight_loads_no_model_map_optimizer_or_restricted_data() -> None:
    result = v45.preflight_v45(CONFIG, v44_terminal_sha256=TERMINAL_SHA256)
    assert result["passed"] is True
    assert result["source_tensor_count"] == 179
    assert result["source_metadata_optimizer_step"] == 8
    assert result["train_question_count"] == 384
    assert result["train_pair_unit_count"] == 25
    assert result["broad_gate_row_count"] == 48
    assert result["bounded_schedule_steps"] == list(range(1, 9))
    assert result["gemma_loaded"] is False
    assert result["scene_maps_loaded"] is False
    assert result["optimizer_file_opened"] is False
    assert result["validation_qa_loaded"] is False
    assert result["oracle_loaded"] is False
    assert result["final_test_scenes_touched"] is False
    assert result["forbidden_file_accesses"] == []
    assert not any("/maps/" in path for path in result["loaded_files"])
    assert not any(path.endswith("/optimizer.pt") for path in result["loaded_files"])


def test_preflight_rejects_equivalent_config_at_unapproved_path(tmp_path: Path) -> None:
    copied = tmp_path / CONFIG.name
    copied.write_bytes(CONFIG.read_bytes())
    with pytest.raises(ValueError, match="path or bytes"):
        v45.preflight_v45(copied, v44_terminal_sha256=TERMINAL_SHA256)


def test_runtime_metadata_hashes_are_dynamic_and_construction_is_strict() -> None:
    source = Path(v45.__file__).read_text(encoding="utf-8")
    assert '"block_cross_residual_state_sha256": block_core.state_sha256()' in source
    assert '"frozen_block_cross_source_stack_state_sha256": (' in source
    assert "exact_original_v41_then_strict_v44_u8_overlay" in source
    assert "load_adapter_checkpoint(" in source
    assert "source_optimizer_state_loaded\": False" in source

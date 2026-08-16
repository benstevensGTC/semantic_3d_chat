from __future__ import annotations

import copy
import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.evaluation.v33_terminal_gate import audit_v33_update64
from semantic_3d_chat.evaluation.v34_base_surface_selector import (
    _approved_v29_runtime_tensor_envelope,
    _promotion,
)
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import DenseSidecarAdapter
from semantic_3d_chat.training.pair_curriculum import CounterfactualPairUnit
from semantic_3d_chat.training.train_base_surface_v34 import (
    PrefixSeparationReference,
    _optimizer,
    assert_v34_trainable_surface,
    build_v34_schedule,
    freeze_for_v34,
    physical_pair_sets,
    require_v33_terminal_gate,
    separation_loss_and_diagnostics,
    summarize_separation,
    v34_contract,
    v34_early_training_gate,
    v34_settings,
)

V34_CONFIG = Path("configs/experiments/gemma4_diverse28_base_surface_v34.yaml")
TERMINAL_REPORT = Path("reports/gemma4/metrics/v33_update64_terminal_gate.json")


def _record(
    index: int,
    *,
    scene_id: str,
    answer_type: str = "spatial_relation",
    pair_id: str | None = None,
    question_key: str | None = None,
    role: str | None = None,
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=f"q_{index:04d}",
        question=f"opaque training question {index}",
        answer="left" if role != "counterfactual" else "right",
        answer_type=answer_type,
        target_xyz=None,
        counterfactual_pair_id=pair_id,
        counterfactual_question_key=question_key,
        counterfactual_expected_change=pair_id is not None,
        counterfactual_role=role,
        counterfactual_change_type="opaque_change" if pair_id else None,
    )


def _schedule_fixture() -> tuple[list[QARecord], list[CounterfactualPairUnit]]:
    records = [
        _record(
            index,
            scene_id=f"scene_{11 + index % 8:06d}",
            answer_type=("presence", "count", "attribute", "spatial_relation")[index % 4],
        )
        for index in range(80)
    ]
    repetitions = (1, 4, 4, 1, 4, 4, 4, 3)
    units: list[CounterfactualPairUnit] = []
    unit_index = 0
    for pair_index, count in enumerate(repetitions):
        pair_id = f"pair_{pair_index:06d}"
        left_scene = f"scene_{11 + 2 * pair_index:06d}"
        right_scene = f"scene_{12 + 2 * pair_index:06d}"
        for question_index in range(count):
            key = f"unit_{unit_index:03d}"
            reference = _record(
                1_000 + 2 * unit_index,
                scene_id=left_scene,
                pair_id=pair_id,
                question_key=key,
                role="reference",
            )
            counterfactual = _record(
                1_001 + 2 * unit_index,
                scene_id=right_scene,
                pair_id=pair_id,
                question_key=key,
                role="counterfactual",
            )
            units.append(
                CounterfactualPairUnit(
                    pair_id, key, reference, counterfactual
                )
            )
            records.extend((reference, counterfactual))
            unit_index += 1
    assert unit_index == 25
    return records, units


def test_v33_terminal_gate_replays_exact_stopped_u64_without_model_or_scene_data() -> None:
    report = audit_v33_update64()
    assert report["passed"] is True
    assert report["gemma_loaded"] is False
    assert report["scene_maps_loaded"] is False
    assert report["qa_loaded"] is False
    assert report["observed_saved_optimizer_steps"] == list(range(0, 65, 8))
    assert report["no_update_072_or_later"] is True
    assert report["update64_gate_evidence"]["nonmirror_teacher_complete_units"] == 0
    assert report["update64_gate_evidence"]["passed"] is False
    assert report["conditional_v34_base_surface_authorized"] is True
    assert hashlib.sha256(TERMINAL_REPORT.read_bytes()).hexdigest() == (
        "703525975c7a03a9b995c6f950dda92ed2945bd1857008196a1086e2a6c19a49"
    )


def test_v34_contract_locks_source_surface_objective_and_train_only_gate() -> None:
    config = load_config(V34_CONFIG)
    settings = v34_settings(config)
    contract = v34_contract(config)
    terminal = require_v33_terminal_gate(config)
    assert settings.saved_optimizer_steps == tuple(range(0, 65, 8))
    assert settings.separation_rank_margin == pytest.approx(torch.log(torch.tensor(1.02)).item())
    assert settings.base_norm_learning_rate == 2.5e-5
    assert settings.base_projection_learning_rate == 1e-4
    assert contract.early_gate_changed_pair_coverage_minimum == 6
    assert contract.early_gate_unrelated_median_ratio_maximum == 1.02
    assert terminal["sha256"] == (
        "703525975c7a03a9b995c6f950dda92ed2945bd1857008196a1086e2a6c19a49"
    )
    assert contract.v31.deferred_final_scene_ids == tuple(
        f"scene_{index:06d}" for index in range(25, 31)
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training.v34_base_surface", "optimizer_steps", 65),
        ("training.v34_base_surface", "separation_rank_margin", 0.05),
        ("training.v34_base_surface", "base_norm_learning_rate", 1e-4),
        ("v34_base_surface", "source_optimizer_step", 56),
        ("v34_base_surface", "exact_trainable_parameter_count", 404_608),
        ("v34_base_surface", "early_gate_uses_training_scenes_only", False),
        ("v34_base_surface", "separation_all_nonchanged_train_scene_pair_count", 111),
        ("v34_base_surface", "v33_terminal_gate_report_sha256", "0" * 64),
    ],
)
def test_v34_contract_fails_closed(section: str, field: str, value: object) -> None:
    config = copy.deepcopy(load_config(V34_CONFIG))
    target = config
    for key in section.split("."):
        target = target[key]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        v34_contract(config)


def test_v34_schedule_balances_25_qa_units_while_geometry_uses_8_and_112() -> None:
    records, units = _schedule_fixture()
    settings = v34_settings(load_config(V34_CONFIG))
    schedule, audit = build_v34_schedule(records, units, settings=settings, seed=34034)
    appearances = Counter(
        (unit.pair_id, unit.question_key) for row in schedule for unit in row.pair_units
    )
    changed, unrelated = physical_pair_sets(units)
    assert len(schedule) == 64
    assert Counter(appearances.values()) == Counter({2: 11, 3: 14})
    assert len(changed) == 8
    assert len(unrelated) == 112
    assert audit["pair_units_with_third_recurrence"] == 14
    assert audit["true_optimizer_step_per_schedule_row"] is True


def test_v34_log_selectivity_loss_is_bounded_and_question_free() -> None:
    settings = v34_settings(load_config(V34_CONFIG))
    source = {
        "a": torch.tensor([[[0.0]]]),
        "b": torch.tensor([[[1.0]]]),
        "c": torch.tensor([[[0.0]]]),
        "d": torch.tensor([[[1.0]]]),
    }
    reference = PrefixSeparationReference(
        source_prefixes=source,
        changed_pairs={"opaque_pair": ("a", "b")},
        unrelated_pairs=(("c", "d"),),
        changed_rms={"opaque_pair": 1.0},
        unrelated_rms={("c", "d"): 1.0},
        audit_sha256="a" * 64,
    )
    prefixes = {
        "a": torch.tensor([[[0.0]]], requires_grad=True),
        "b": torch.tensor([[[1.03]]], requires_grad=True),
        "c": torch.tensor([[[0.0]]], requires_grad=True),
        "d": torch.tensor([[[1.0]]], requires_grad=True),
    }
    loss, raw = separation_loss_and_diagnostics(
        prefixes=prefixes, reference=reference, settings=settings
    )
    summary = summarize_separation(raw)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in prefixes.values())
    assert summary["changed_selectivity_ratio_geometric_mean"] == pytest.approx(1.03)
    assert summary["unrelated_ratio_median"] == pytest.approx(1.0)
    assert summary["question_or_answer_text_used"] is False
    assert summary["oracle_environment_inputs_used"] is False
    assert summary["validation_scenes_used"] is False


def test_v34_early_gate_is_training_only_and_uses_two_sided_unrelated_limits() -> None:
    contract = v34_contract(load_config(V34_CONFIG))
    passing = {
        "changed_selectivity_ratio_geometric_mean": 1.025,
        "changed_selectivity_over_1_02_count": 6,
        "changed_selectivity_ratio_minimum": 0.99,
        "unrelated_ratio_median": 1.0,
        "unrelated_abs_log_ratio_p90": 0.01,
    }
    gate = v34_early_training_gate(passing, contract)
    assert gate["passed"] is True
    assert gate["training_scenes_only"] is True
    assert not any("validation" in key for key in gate)
    passing["unrelated_ratio_median"] = 0.97
    assert v34_early_training_gate(passing, contract)["passed"] is False


class _FakeInstallation(torch.nn.Module):
    pass


def _bundle() -> SimpleNamespace:
    sidecar = DenseSidecarAdapter(
        scene_dim=1536,
        latent_count=256,
        width=128,
        fourier_bands=8,
        max_direct_scale=0.25,
        initialization_seed=28028,
    )
    decoder = torch.nn.Linear(2, 2)
    bank_module = _FakeInstallation()
    bank_module.weight = torch.nn.Parameter(torch.ones(3))
    bank = SimpleNamespace(
        settings=SimpleNamespace(name="opaque_bank"), installation=bank_module
    )
    return SimpleNamespace(
        language=SimpleNamespace(model=decoder),
        dense_sidecar_adapter=sidecar,
        checkpoint_modules={
            "dense_sidecar_adapter": sidecar,
            "opaque_decoder_bank": bank_module,
        },
        lora_installation=SimpleNamespace(banks=(bank,)),
    )


def test_v34_freeze_and_optimizer_expose_only_four_base_tensors_in_two_groups() -> None:
    bundle = _bundle()
    trainable = freeze_for_v34(bundle)
    surface = assert_v34_trainable_surface(bundle)
    optimizer = _optimizer(bundle, v34_settings(load_config(V34_CONFIG)))
    assert_v34_trainable_surface(bundle, optimizer)
    assert sum(parameter.numel() for parameter in trainable) == 199_808
    assert surface["group_parameter_counts"] == {
        "base_norm": 3_072,
        "base_projection": 196_736,
    }
    assert [group["name"] for group in optimizer.param_groups] == [
        "dense_sidecar_adapter.base_norm",
        "dense_sidecar_adapter.base_projection",
    ]
    assert not any(parameter.requires_grad for parameter in bundle.language.model.parameters())
    sidecar_named = dict(bundle.dense_sidecar_adapter.named_parameters())
    assert all(
        not sidecar_named[name].requires_grad
        for name in (
            "output_projection.weight",
            "channel_gain",
            "sidecar_norm.weight",
            "sidecar_norm.bias",
            "sidecar_projection.weight",
            "sidecar_projection.bias",
            "position_projection.weight",
            "position_projection.bias",
        )
    )


def test_v34_promotion_has_exact_final_once_outward_check_shape() -> None:
    selected = {
        "checkpoint": "/tmp/update_064",
        "optimizer_step": 64,
        "greedy_exact_complete_units_correct": 6,
        "greedy_complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 4,
            "picture_support": 1,
        },
        "color_full_vocab_sides": 12,
        "mirror_full_vocab_sides": 10,
        "new_negative_sides_vs_approved_v29": [],
        "checks": {"rich_internal_requirement": True},
    }
    promotion = _promotion(
        selected,
        approved_v29_aggregate=(216, 81),
        selected_aggregate=(216, 81),
    )
    assert set(promotion["checks"]) == {
        "development_checkpoint_selected",
        "changed_complete_pair_threshold_met",
        "aggregate_validation_exact_accuracy_retained",
    }
    assert all(promotion["checks"].values())
    assert all(promotion["audited_internal_requirements"].values())
    assert promotion["eligible"] is True


def test_v34_approved_v29_retention_envelope_allows_only_exact_zero_fresh_bank() -> None:
    v29 = {"dense_sidecar_adapter.base_norm.weight": torch.ones(2)}
    update0 = {
        **v29,
        **{
            f"lora_banks.extension_v30_joint_pair_query.adapters.{index}.lora_a": torch.ones(1)
            for index in range(4)
        },
        **{
            f"lora_banks.extension_v30_joint_pair_query.adapters.{index}.lora_b": torch.zeros(1)
            for index in range(4)
        },
    }
    merged = _approved_v29_runtime_tensor_envelope(update0, v29)
    assert set(merged) == set(update0)
    forged = dict(update0)
    forged[
        "lora_banks.extension_v30_joint_pair_query.adapters.0.lora_b"
    ] = torch.ones(1)
    with pytest.raises(ValueError, match="not exact-zero output"):
        _approved_v29_runtime_tensor_envelope(forged, v29)


def test_v34_docs_and_make_targets_have_no_final_bypass() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    for target in (
        "gemma4-v33-seal-update64",
        "gemma4-v34-preflight-base-surface",
        "gemma4-v34-train-base-surface",
        "gemma4-v34-select-base-surface",
    ):
        assert target in makefile
    assert "gemma4-v34-evaluate-final" not in makefile
    assert "V34 bounded base-route" in readme

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.v19_optimizer_state import (
    validate_v19_adamw_state_manifest,
)
from semantic_3d_chat.evaluation.v21_phase_aware_precision import (
    PHASE_AWARE_PRECISION_PAIR_V1,
)
from semantic_3d_chat.evaluation.v21_structural_preflight import (
    EXPECTED_ORDERED_UNIT_SHA256,
    EXPECTED_PAIR_UNIT_SELECTION_SHA256,
    EXPECTED_RESOLVED_CONFIG_HASH,
    EXPECTED_SELECTION_SHA256,
    SIGNED_X_OPTIMIZER_GROUP_NAME,
    V21StructuralPreflightViolation,
    atomic_write_json,
    evaluate_v21_structural_gate,
    exact_clone_adamw_evidence,
    functional_local_field_delta,
    local_dependence_evidence,
    ordered_curriculum_evidence,
    pair_unit_selection_evidence,
    precision_cast_audit,
    signed_x_residual_state_sha256,
    spatial_rank_evidence,
    validate_v21_config_contract,
)
from semantic_3d_chat.scene_encoder.signed_x_local_field import (
    SignedXLocalFieldSceneResidual,
)
from semantic_3d_chat.scene_encoder.signed_x_residual import SignedXSceneResidual
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_local_field_phase_aware_v21.yaml"


def _optimizer_contract() -> dict[str, object]:
    return {
        "name": "AdamW",
        "learning_rate": 1.0e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
        "amsgrad": False,
        "gradient_clip_norm": 1.0,
        "accumulation_divisor": 12,
        "step_index": 1,
    }


def _record(scene_id: str, question_id: str) -> SimpleNamespace:
    return SimpleNamespace(scene_id=scene_id, question_id=question_id)


def _unit(index: int, pair_id: str = "pair_000001") -> SimpleNamespace:
    reference = _record("scene_000003", f"q_{index:04d}")
    counterfactual = _record("scene_000004", f"q_{index + 20:04d}")
    return SimpleNamespace(
        pair_id=pair_id,
        question_key=f"cfq_{index:04d}",
        reference=reference,
        counterfactual=counterfactual,
        scene_ids=(reference.scene_id, counterfactual.scene_id),
        records=(reference, counterfactual),
    )


def test_current_v21_config_satisfies_strict_no_step_contract() -> None:
    contract = validate_v21_config_contract(load_config(CONFIG_PATH))

    assert (
        contract["role"]
        == "v21_exact_ordered_signed_x_local_field_phase_aware_structural_preflight"
    )
    assert contract["optimizer"] == _optimizer_contract()
    assert contract["resolved_config_hash"] == EXPECTED_RESOLVED_CONFIG_HASH
    assert contract["expected_hashes"]["selection_sha256"] == EXPECTED_SELECTION_SHA256
    assert contract["expected_hashes"]["ordered_unit_sha256"] == EXPECTED_ORDERED_UNIT_SHA256
    assert (
        contract["expected_hashes"]["pair_unit_selection_sha256"]
        == EXPECTED_PAIR_UNIT_SELECTION_SHA256
    )
    assert contract["structural_preflight_requires"] == {
        "maximum_per_scene_raw_and_effective_delta_to_core_rms_ratio": 0.01,
        "minimum_raw_mirror_residual_to_core_rms_ratio": 0.01,
        "minimum_raw_mirror_to_color_normalized_selectivity": 1.5,
        "minimum_mirror_effective_residual_to_core_rms_ratio": 0.01,
        "legacy_effective_total_norm_selectivity_diagnostic_only": True,
        "minimum_mirror_effective_raw_pair_cosine": 0.8,
        "minimum_mirror_effective_raw_aligned_gain": 0.75,
        "maximum_mirror_effective_raw_aligned_gain": 1.25,
        "minimum_mirror_signal_to_orthogonal_noise_ratio": 1.0,
        "minimum_local_hidden_spatial_rank": 2,
    }
    assert len(contract["contract_sha256"]) == 64


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config["training"]["optimizer"].update(learning_rate=2.0e-4),
        lambda config: config["training"]["pair_objectives"]["by_pair"]["pair_000001"].update(
            candidate_margin=1.0
        ),
        lambda config: config.update(structural_preflight={"required": True}),
        lambda config: config["experiment"].update(source_checkpoint_epoch=3),
        lambda config: config["v21_screen"]["structural_preflight_requires"].update(
            minimum_local_hidden_spatial_rank=1
        ),
        lambda config: config["v21_screen"]["predicted_update_requires"].update(
            expected_units_per_pair=5
        ),
        lambda config: config["scene_encoder"]["signed_x_scene_residual"].update(
            architecture_version="signed_x_moment_v1"
        ),
        lambda config: config["language"].update(system_prompt="changed gradient prompt"),
        lambda config: config["language"].update(dtype="float16"),
        lambda config: config["scene_encoder"].update(query_identity_scale=99.0),
    ],
)
def test_v21_contract_fails_closed_on_gradient_or_source_mutation(mutate) -> None:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    mutate(config)

    with pytest.raises(V21StructuralPreflightViolation):
        validate_v21_config_contract(config)


def test_pair_unit_and_order_hashes_are_canonical_and_distinct() -> None:
    first = _unit(1)
    second = _unit(2, "pair_000003")
    first_batch = SimpleNamespace(kind="pair", pair_units=(first,))
    second_batch = SimpleNamespace(kind="pair", pair_units=(second,))

    selection, selection_hash = pair_unit_selection_evidence((second, first))
    repeated_selection, repeated_hash = pair_unit_selection_evidence((first, second))
    order, order_hash = ordered_curriculum_evidence((first_batch, second_batch))
    _reversed_order, reversed_hash = ordered_curriculum_evidence((second_batch, first_batch))

    assert selection == repeated_selection
    assert selection_hash == repeated_hash
    assert set(selection[0]) == {"pair_id", "question_key", "scene_ids", "question_ids"}
    assert order[0]["microstep"] == 1
    assert order_hash != reversed_hash


def test_exact_clone_adamw_predicts_state_without_mutating_live_module() -> None:
    module = SignedXLocalFieldSceneResidual(scene_dim=1536, latent_count=8, content_dim=128)
    gradient = torch.linspace(
        -0.5,
        0.5,
        module.output_projection.weight.numel(),
        dtype=torch.float32,
    ).reshape_as(module.output_projection.weight)
    module.output_projection.weight.grad = gradient
    live_hash_before = module_collection_state_sha256({"signed_x_scene_residual": module})

    predicted_weight, evidence = exact_clone_adamw_evidence(module, _optimizer_contract())

    assert module_collection_state_sha256({"signed_x_scene_residual": module}) == live_hash_before
    assert torch.count_nonzero(module.output_projection.weight) == 0
    assert torch.count_nonzero(predicted_weight) > 0
    assert evidence["changed_parameter_keys"] == ["output_projection.weight"]
    assert evidence["finite_update"] is True
    assert evidence["predicted_signed_x_scene_residual_state_sha256"] == (
        signed_x_residual_state_sha256(module, predicted_weight)
    )
    assert evidence["canonical_adamw_state_sha256"] == validate_v19_adamw_state_manifest(
        evidence["canonical_adamw_state_manifest"], _optimizer_contract()
    )
    assert (
        evidence["canonical_adamw_state_manifest"]["param_groups"][0]["name"]
        == SIGNED_X_OPTIMIZER_GROUP_NAME
    )


def test_local_dependence_proves_no_global_moment_reduction() -> None:
    local = SignedXLocalFieldSceneResidual(scene_dim=16, latent_count=256, content_dim=8)
    moment = SignedXSceneResidual(scene_dim=16, latent_count=256, content_dim=8)

    local_evidence = local_dependence_evidence(local)
    moment_evidence = local_dependence_evidence(moment)

    assert local_evidence["changed_slot_union_count"] == 256
    assert local_evidence["exact_two_slot_local_support"] is True
    assert local_evidence["no_global_moment_broadcast"] is True
    assert moment_evidence["exact_two_slot_local_support"] is False
    assert moment_evidence["no_global_moment_broadcast"] is False


def test_local_hidden_has_spatial_rank_greater_than_one() -> None:
    module = SignedXLocalFieldSceneResidual(scene_dim=16, latent_count=256, content_dim=8)
    positions = torch.linspace(-1.0, 1.0, 256)
    centered = torch.stack(
        (
            positions,
            positions.square() - positions.square().mean(),
            torch.sin(positions * 3.0),
            torch.cos(positions * 5.0) - torch.cos(positions * 5.0).mean(),
        ),
        dim=-1,
    )
    centered = torch.nn.functional.pad(centered, (0, 4)).unsqueeze(0)

    rank = spatial_rank_evidence(module.hidden_values(centered))

    assert rank["minimum_spatial_rank"] > 1


def test_bfloat16_audit_matches_direct_module_forward_arithmetic() -> None:
    module = SignedXLocalFieldSceneResidual(scene_dim=4, latent_count=4, content_dim=2)
    centered = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0], [-0.5, 0.0]]])
    predicted_weight = torch.full_like(module.output_projection.weight, 5.0e-3)
    base = torch.full((1, 4, 4), 0.5001500248908997, dtype=torch.float32)

    raw, effective = functional_local_field_delta(
        module,
        centered,
        predicted_weight,
        base_tokens=base,
    )
    audit = precision_cast_audit(base, raw, effective, model_dtype=torch.bfloat16)
    direct = copy.deepcopy(module)
    direct.output_projection.weight.data.copy_(predicted_weight)
    direct_effective = (
        direct(base, centered).to(torch.bfloat16).float() - base.to(torch.bfloat16).float()
    )
    incorrectly_precast_effective = (
        base.to(torch.bfloat16) + raw.to(torch.bfloat16)
    ).float() - base.to(torch.bfloat16).float()

    assert torch.count_nonzero(raw) > 0
    assert torch.count_nonzero(raw.to(torch.bfloat16)) > 0
    assert torch.equal(effective, direct_effective)
    assert not torch.equal(effective, incorrectly_precast_effective)
    assert audit["changed_element_count"] == torch.count_nonzero(direct_effective)
    assert audit["algorithm"] == "bfloat16_cast_of_fp32_base_plus_fp32_delta"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_bfloat16_audit_moves_mps_diagnostics_to_cpu_before_float64() -> None:
    base = torch.ones(1, 2, 3, device="mps", dtype=torch.float32)
    raw = torch.full_like(base, 1.0e-3)
    effective = torch.zeros_like(base)

    audit = precision_cast_audit(base, raw, effective, model_dtype=torch.bfloat16)

    assert audit["comparison_dtype"] == "float64"
    assert audit["raw_delta_rms"] == pytest.approx(1.0e-3)


def _scene_metric(ratio: float) -> dict[str, object]:
    return {
        "positive_finite_total_energy": True,
        "across_slot_mean_energy_fraction": 0.0,
        "slot_varying_energy_fraction": 1.0,
        "delta_to_core_rms_ratio": ratio,
    }


def _pair_metric(ratio: float) -> dict[str, float]:
    return {"residual_to_core_pair_difference_ratio": ratio}


def _phase_pair(*, cosine: float = 0.95) -> dict[str, object]:
    return {
        "algorithm_family": PHASE_AWARE_PRECISION_PAIR_V1,
        "algorithm": "phase_aware_bfloat16_pair_v1",
        "model_dtype": "bfloat16",
        "actual_pair": {
            "raw_effective_cosine": cosine,
            "aligned_gain": 1.0,
            "aligned_effective_rms": 0.02,
            "orthogonal_quantization_rms": 0.005,
        },
        "common_delta_null": {"response_rms": 0.001},
        "shared_base": {"phase_spread_rms": 0.001},
    }


def test_structural_gate_requires_rank_bfloat16_signal_and_phase_alignment() -> None:
    module = SignedXLocalFieldSceneResidual(scene_dim=16, latent_count=256, content_dim=8)
    raw = {scene: _scene_metric(0.006) for scene in ("a", "b")}
    effective = {scene: _scene_metric(0.005) for scene in ("a", "b")}
    audits = {
        scene: {"changed_element_count": 5, "quantization_error_rms": 0.001} for scene in ("a", "b")
    }
    ranks = {scene: {"minimum_spatial_rank": 3} for scene in ("a", "b")}
    local = local_dependence_evidence(module)
    requirements = {
        "maximum_per_scene_raw_and_effective_delta_to_core_rms_ratio": 0.01,
        "minimum_raw_mirror_residual_to_core_rms_ratio": 0.01,
        "minimum_raw_mirror_to_color_normalized_selectivity": 1.5,
        "minimum_mirror_effective_residual_to_core_rms_ratio": 0.01,
        "legacy_effective_total_norm_selectivity_diagnostic_only": True,
        "minimum_mirror_effective_raw_pair_cosine": 0.8,
        "minimum_mirror_effective_raw_aligned_gain": 0.75,
        "maximum_mirror_effective_raw_aligned_gain": 1.25,
        "minimum_mirror_signal_to_orthogonal_noise_ratio": 1.0,
        "minimum_local_hidden_spatial_rank": 2,
    }
    raw_pairs = {
        "pair_000001": _pair_metric(0.008),
        "pair_000003": _pair_metric(0.016),
    }
    effective_pairs = {
        "pair_000001": _pair_metric(0.04),
        "pair_000003": _pair_metric(0.02),
    }
    phase_pairs = {
        "pair_000001": _phase_pair(),
        "pair_000003": _phase_pair(),
    }

    gate = evaluate_v21_structural_gate(
        raw,
        effective,
        raw_pair_metrics=raw_pairs,
        effective_pair_metrics=effective_pairs,
        phase_pair_diagnostics=phase_pairs,
        precision_audits=audits,
        structural_state=module.validate_structural_state(),
        local_dependence=local,
        local_hidden_ranks=ranks,
        requirements=requirements,
    )
    assert gate["passed"] is True
    assert gate["model_effective_pair_selectivity"]["mirror_to_color_normalized_selectivity"] == 0.5
    assert gate["legacy_effective_total_norm_selectivity"] == {
        "diagnostic_only": True,
        "historical_reference_threshold": 1.5,
        "observed_mirror_to_color_normalized_selectivity": 0.5,
        "would_have_passed_historical_v20_threshold": False,
        "excluded_from_v21_authorization": True,
    }

    ranks["a"]["minimum_spatial_rank"] = 1
    failed = evaluate_v21_structural_gate(
        raw,
        effective,
        raw_pair_metrics=raw_pairs,
        effective_pair_metrics=effective_pairs,
        phase_pair_diagnostics=phase_pairs,
        precision_audits=audits,
        structural_state=module.validate_structural_state(),
        local_dependence=local,
        local_hidden_ranks=ranks,
        requirements=requirements,
    )
    assert failed["passed"] is False

    ranks["a"]["minimum_spatial_rank"] = 3
    phase_pairs["pair_000003"] = _phase_pair(cosine=0.5)
    phase_failed = evaluate_v21_structural_gate(
        raw,
        effective,
        raw_pair_metrics=raw_pairs,
        effective_pair_metrics=effective_pairs,
        phase_pair_diagnostics=phase_pairs,
        precision_audits=audits,
        structural_state=module.validate_structural_state(),
        local_dependence=local,
        local_hidden_ranks=ranks,
        requirements=requirements,
    )
    assert phase_failed["passed"] is False


def test_atomic_json_writer_replaces_complete_document(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "report.json"

    atomic_write_json(destination, {"version": 1, "authorized": False})
    atomic_write_json(destination, {"version": 2, "authorized": True})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "authorized": True,
        "version": 2,
    }
    assert list(destination.parent.glob("*.tmp")) == []

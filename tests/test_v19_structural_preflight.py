from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.v18_structural_preflight import fp64_delta_metrics
from semantic_3d_chat.evaluation.v19_optimizer_state import (
    validate_v19_adamw_state_manifest,
)
from semantic_3d_chat.evaluation.v19_structural_preflight import (
    EXPECTED_ORDERED_UNIT_SHA256,
    EXPECTED_SELECTION_SHA256,
    SIGNED_X_OPTIMIZER_GROUP_NAME,
    V19StructuralPreflightViolation,
    atomic_write_json,
    evaluate_v19_structural_gate,
    exact_clone_adamw_evidence,
    functional_signed_x_delta,
    ordered_curriculum_evidence,
    pair_unit_selection_evidence,
    signed_x_residual_state_sha256,
    validate_v19_config_contract,
)
from semantic_3d_chat.scene_encoder.signed_x_residual import SignedXSceneResidual
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_moment_v19.yaml"


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
    return SimpleNamespace(
        pair_id=pair_id,
        question_key=f"cfq_{index:04d}",
        reference=_record("scene_000003", f"q_{index:04d}"),
        counterfactual=_record("scene_000004", f"q_{index + 20:04d}"),
    )


def test_current_v19_config_satisfies_strict_no_step_contract() -> None:
    contract = validate_v19_config_contract(load_config(CONFIG_PATH))

    assert contract["role"] == "v19_exact_ordered_epoch1_signed_x_structural_preflight"
    assert contract["optimizer"] == _optimizer_contract()
    assert contract["expected_hashes"]["selection_sha256"] == EXPECTED_SELECTION_SHA256
    assert (
        contract["expected_hashes"]["ordered_unit_sha256"]
        == EXPECTED_ORDERED_UNIT_SHA256
    )
    assert contract["maximum_delta_to_core_rms_ratio"] == 0.01
    assert len(contract["contract_sha256"]) == 64


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config["training"]["optimizer"].update(learning_rate=2.0e-4),
        lambda config: config["training"]["pair_objectives"]["by_pair"][
            "pair_000001"
        ].update(candidate_margin=1.0),
        lambda config: config.update(structural_preflight={"required": True}),
        lambda config: config["experiment"].update(source_checkpoint_epoch=3),
    ],
)
def test_v19_contract_fails_closed_on_gradient_or_source_mutation(mutate) -> None:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    mutate(config)

    with pytest.raises(V19StructuralPreflightViolation):
        validate_v19_config_contract(config)


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
    assert order[0]["microstep"] == 1
    assert order_hash != reversed_hash


def test_exact_clone_adamw_predicts_state_without_mutating_live_module() -> None:
    module = SignedXSceneResidual(scene_dim=1536, latent_count=8, content_dim=128)
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


def test_fp32_signed_delta_is_centered_all_slot_and_bounded() -> None:
    torch.manual_seed(19)
    module = SignedXSceneResidual(scene_dim=16, latent_count=256, content_dim=8)
    centered_content = torch.randn(1, 256, 8)
    centered_content = centered_content - centered_content.mean(dim=1, keepdim=True)
    predicted_weight = torch.randn_like(module.output_projection.weight) * 1.0e-5
    base = torch.ones(1, 256, 16)

    raw_delta, effective_delta = functional_signed_x_delta(
        module,
        centered_content,
        predicted_weight,
        base_tokens=base,
    )
    raw = {"scene": fp64_delta_metrics(base, raw_delta)}
    effective = {"scene": fp64_delta_metrics(base, effective_delta)}
    structural = module.validate_structural_state()
    gate = evaluate_v19_structural_gate(
        raw,
        effective,
        structural_state=structural,
        maximum_delta_to_core_rms_ratio=0.01,
    )

    assert raw_delta.dtype == torch.float32
    assert raw_delta.mean(dim=1).abs().max() < 1.0e-10
    assert structural["accounted_slot_count"] == 256
    assert gate["all_slots_accounted"] is True
    assert gate["passed"] is True


def test_atomic_json_writer_replaces_complete_document(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "report.json"

    atomic_write_json(destination, {"version": 1, "authorized": False})
    atomic_write_json(destination, {"version": 2, "authorized": True})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "authorized": True,
        "version": 2,
    }
    assert list(destination.parent.glob("*.tmp")) == []

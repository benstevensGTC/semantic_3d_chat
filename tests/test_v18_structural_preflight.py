from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest
import torch

from semantic_3d_chat.evaluation.v18_structural_preflight import (
    COLOR_PAIR_ID,
    EXPECTED_CONTINUATION,
    EXPECTED_ELIGIBILITY,
    EXPECTED_FULL_TEACHER_GATE,
    EXPECTED_RANKING_FIELDS,
    MIRROR_PAIR_ID,
    STRUCTURAL_PREFLIGHT_ROLE,
    V18_SCREEN_ROLE,
    StructuralThresholds,
    V18StructuralPreflightViolation,
    _exact_clone_adamw_evidence,
    capture_rng_states,
    evaluate_structural_gate,
    fp64_delta_metrics,
    fp64_pair_delta_metrics,
    functional_simulated_deltas,
    ordered_curriculum_evidence,
    restore_rng_states,
    rng_state_evidence,
    simulated_residual_state_sha256,
    validate_v18_config_contract,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    GLOBAL_MEAN_V1,
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
)
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256

SHA = "a" * 64


def _config() -> dict:
    expected_hashes = {
        "ordered_unit_sha256": SHA,
        "source_adapter_sha256": "b" * 64,
        "source_metadata_sha256": "c" * 64,
        "frozen_scene_state_sha256": "d" * 64,
        "frozen_lora_bank_state_sha256": {
            "inherited_v12": "e" * 64,
            "extension_v13": "f" * 64,
        },
        "initial_residual_state_sha256": "1" * 64,
        "position_features_sha256": "0" * 64,
        "selection_sha256": "2" * 64,
        "pair_membership_sha256": "3" * 64,
        "core_prefix_sha256": {
            "scene_000003": "4" * 64,
            "scene_000004": "5" * 64,
            "scene_000007": "6" * 64,
            "scene_000008": "7" * 64,
        },
        "v16_gradient_audit_sha256": "8" * 64,
        "v17_lr_response_sha256": "9" * 64,
    }
    optimizer = {
        "name": "AdamW",
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
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
    return {
        "scene_encoder": {
            "global_latents": 256,
            "global_residual": None,
            "global_scene_residual": {
                "enabled": True,
                "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
                "expected_initial_state_sha256": expected_hashes["initial_residual_state_sha256"],
            },
        },
        "training": {
            "batch_size": 2,
            "max_questions_per_scene": 6,
            "language_decoder_gradient_checkpointing": True,
            "initialize_legacy_lora_into_bank": None,
            "initialize_named_lora_freeze_transition": True,
            "train_global_scene_residual_only": True,
            "freeze_scene_adapter": True,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "gradient_accumulation": 12,
            "pair_steps_per_epoch": 12,
            "epochs": 4,
            "pair_only_mode": True,
            "pair_batch_fraction": 1.0,
            "pair_units_per_batch": 1,
            "pair_max_units_per_pair": 6,
            "pair_only_scene_ids": [
                "scene_000003",
                "scene_000004",
                "scene_000007",
                "scene_000008",
            ],
            "pair_ranking_weight": 8.0,
            "pair_ranking_margin": 1.0,
            "pair_ranking_mode": "candidate_logit",
            "pair_full_vocab_ranking_weight": 2.0,
            "pair_full_vocab_ranking_margin": 1.0,
            "grounding_weight": 0.0,
            "grounding_anchor_weight": 0.0,
            "latent_diversity_weight": 0.0,
            "paired_scene_separation_weight": 0.0,
            "spatial_answer_contrastive_weight": 0.0,
            "spatial_answer_warmup_steps": 0,
            "spatial_relation_contrastive_weight": 0.0,
            "spatial_relation_warmup_steps": 0,
            "pair_gate_enabled": True,
            "pair_gate_every_epochs": 1,
            "pair_gate_changed_unit_accuracy": 0.95,
            "pair_gate_prediction_flip_rate": 1.0,
            "pair_gate_wrong_prefix_flip_rate": 1.0,
            "pair_gate_first_answer_token_top1_accuracy": 1.0,
            "pair_gate_stop_when_passed": False,
            "early_stopping_patience": 0,
            "initialize_expected_adapter_sha256": expected_hashes["source_adapter_sha256"],
            "initialize_expected_metadata_sha256": expected_hashes["source_metadata_sha256"],
            "optimizer": copy.deepcopy(optimizer),
        },
        "language": {
            "model_id": "google/gemma-4-E2B-it",
            "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "backend": "gemma4",
            "dtype": "bfloat16",
            "scene_prefix_after_bos": True,
            "scene_boundary_mode": "gemma4_native_image",
            "system_prompt": (
                "You answer using only the continuous 3D scene memory supplied before this "
                "conversation. Do not invent objects or relationships unsupported by the "
                "scene. If there is not enough evidence, answer unknown."
            ),
        },
        "experiment": {"residual_parameter_count": 400_128},
        "structural_preflight": {
            "schema_version": 1,
            "required": True,
            "role": STRUCTURAL_PREFLIGHT_ROLE,
            "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
            "spatial_centering": "all_slots_fp32",
            "content_gate": "bias_free_scalar_sigmoid_centered_content",
            "implementation_source_sha256": SHA,
            "source_must_be_clean": True,
            "latent_count": 256,
            "scene_dim": 1536,
            "residual_parameter_count": 400_128,
            "exact_epoch": 1,
            "microsteps": 12,
            "optimizer": optimizer,
            "thresholds": StructuralThresholds().__dict__,
            "expected_hashes": expected_hashes,
            "evidence_paths": {
                "v16_gradient_audit": "reports/v16.json",
                "v17_lr_response": "reports/v17.json",
            },
        },
        "v18_screen": {
            "schema_version": 1,
            "role": V18_SCREEN_ROLE,
            "learning_rate": 0.001,
            "screen_optimizer_updates": 4,
            "conditional_max_optimizer_updates": 12,
            "epoch_tiebreaker": "lower_epoch",
            "execution_stages": {
                "stage_1_exact_v14_restart_updates": 1,
                "stage_1_stop_required": True,
                "predicted_preflight_state_must_match_epoch_001": True,
                "stage_2_resume_from_epoch": 1,
                "stage_2_load_optimizer_state": True,
                "stage_2_load_history": True,
                "stage_2_target_total_optimizer_updates": 4,
            },
            "eligibility_requires": EXPECTED_ELIGIBILITY,
            "ranking_descending": list(EXPECTED_RANKING_FIELDS),
            "continuation_requires": EXPECTED_CONTINUATION,
            "full_teacher_gate_requires": EXPECTED_FULL_TEACHER_GATE,
            "greedy_audit_only_after_full_teacher_gate": True,
        },
    }


def test_v18_config_contract_accepts_only_pinned_architecture_optimizer_and_stages() -> None:
    observed = validate_v18_config_contract(_config(), implementation_source_sha256=SHA)

    assert observed["architecture_version"] == ZERO_SPATIAL_MEAN_CONTENT_GATE_V1
    assert observed["optimizer"]["accumulation_divisor"] == 12
    assert observed["v18_screen"]["screen_optimizer_updates"] == 4
    assert observed["v18_screen"]["execution_stages"]["stage_2_load_history"] is True
    assert len(observed["contract_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda value: value["scene_encoder"]["global_scene_residual"].update(
                architecture_version=GLOBAL_MEAN_V1
            ),
            "architecture",
        ),
        (
            lambda value: value["structural_preflight"]["optimizer"].update(learning_rate=3e-4),
            "AdamW",
        ),
        (
            lambda value: value["structural_preflight"]["thresholds"].update(
                maximum_effective_mean_energy_fraction=1e-6
            ),
            "thresholds",
        ),
        (
            lambda value: value["v18_screen"]["execution_stages"].update(
                stage_2_load_optimizer_state=False
            ),
            "policy",
        ),
        (
            lambda value: value["v18_screen"].update(epoch_tiebreaker="higher_epoch"),
            "policy",
        ),
        (
            lambda value: value["experiment"].update(residual_parameter_count=400_000),
            "residual_parameter_count",
        ),
        (
            lambda value: value["training"]["optimizer"].update(foreach=None),
            "training contract",
        ),
        (
            lambda value: value["training"].update(pair_ranking_weight=4.0),
            "gradient-defining objective",
        ),
        (
            lambda value: value["training"].update(
                pair_full_vocab_ranking_weight=1.0
            ),
            "gradient-defining objective",
        ),
        (
            lambda value: value["training"].update(pair_ranking_mode="nll"),
            "gradient-defining objective",
        ),
        (
            lambda value: value["training"].update(pair_ranking_margin=0.5),
            "gradient-defining objective",
        ),
        (
            lambda value: value["training"].update(
                pair_full_vocab_ranking_margin=0.5
            ),
            "gradient-defining objective",
        ),
        (
            lambda value: value["training"].update(
                language_decoder_gradient_checkpointing=False
            ),
            "gradient-defining objective",
        ),
    ],
)
def test_v18_config_contract_rejects_legacy_or_unreviewed_changes(mutator, match: str) -> None:
    config = _config()
    mutator(config)

    with pytest.raises(V18StructuralPreflightViolation, match=match):
        validate_v18_config_contract(config, implementation_source_sha256=SHA)


def test_v18_config_contract_rejects_implementation_hash_mismatch_and_unknown_fields() -> None:
    with pytest.raises(V18StructuralPreflightViolation, match="implementation source hash"):
        validate_v18_config_contract(_config(), implementation_source_sha256="0" * 64)

    config = _config()
    config["structural_preflight"]["post_hoc_override"] = True
    with pytest.raises(V18StructuralPreflightViolation, match="unknown"):
        validate_v18_config_contract(config, implementation_source_sha256=SHA)


def test_fp64_delta_metrics_distinguishes_raw_centering_and_effective_quantization() -> None:
    core = torch.ones(1, 4, 2)
    raw = torch.tensor([[[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]])
    effective = raw + 0.02

    raw_metrics = fp64_delta_metrics(core, raw)
    effective_metrics = fp64_delta_metrics(core, effective)

    assert raw_metrics["positive_finite_total_energy"] is True
    assert raw_metrics["across_slot_mean_energy_fraction"] == pytest.approx(0.0)
    assert raw_metrics["slot_varying_energy_fraction"] == pytest.approx(1.0)
    assert effective_metrics["across_slot_mean_energy_fraction"] > 0.0
    assert effective_metrics["energy_closure_absolute_error"] < 1e-12


def _passing_scene_metrics() -> tuple[dict, dict]:
    raw = {
        scene: {
            "positive_finite_total_energy": True,
            "across_slot_mean_energy_fraction": 1e-9,
            "slot_varying_energy_fraction": 1.0 - 1e-9,
            "delta_to_core_rms_ratio": 0.01,
        }
        for scene in ("s1", "s2", "s3", "s4")
    }
    effective = {
        scene: {
            "positive_finite_total_energy": True,
            "across_slot_mean_energy_fraction": 3.11e-4,
            "slot_varying_energy_fraction": 1.0 - 3.11e-4,
            "delta_to_core_rms_ratio": 0.04,
        }
        for scene in raw
    }
    return raw, effective


def _passing_pairs() -> dict:
    return {
        COLOR_PAIR_ID: {"positive_finite_pair_delta": True},
        MIRROR_PAIR_ID: {"positive_finite_pair_delta": True},
    }


def test_structural_gate_accepts_bf16_guard_and_rejects_noop_or_pair_collapse() -> None:
    raw, effective = _passing_scene_metrics()
    gate = evaluate_structural_gate(
        raw,
        effective,
        _passing_pairs(),
        _passing_pairs(),
        StructuralThresholds(),
    )
    assert gate["passed"] is True
    assert gate["maximum_observed_effective_mean_energy_fraction"] == pytest.approx(3.11e-4)

    no_op = copy.deepcopy(raw)
    no_op["s1"]["positive_finite_total_energy"] = False
    gate = evaluate_structural_gate(
        no_op,
        effective,
        _passing_pairs(),
        _passing_pairs(),
        StructuralThresholds(),
    )
    assert gate["passed"] is False
    assert gate["scene_checks"]["s1"]["raw_positive_finite_total_energy"] is False

    collapsed_pairs = _passing_pairs()
    collapsed_pairs[MIRROR_PAIR_ID]["positive_finite_pair_delta"] = False
    gate = evaluate_structural_gate(
        raw,
        effective,
        collapsed_pairs,
        _passing_pairs(),
        StructuralThresholds(),
    )
    assert gate["passed"] is False


def test_fp64_pair_metrics_rejects_scene_independent_delta_as_nonzero_evidence() -> None:
    first_core = torch.tensor([[[2.0], [4.0]]])
    second_core = torch.tensor([[[1.0], [2.0]]])
    shared_delta = torch.tensor([[[0.2], [-0.2]]])

    metrics = fp64_pair_delta_metrics(
        first_core,
        second_core,
        shared_delta,
        shared_delta,
    )

    assert metrics["positive_finite_core_difference"] is True
    assert metrics["positive_finite_pair_delta"] is False
    assert metrics["residual_pair_difference_rms"] == pytest.approx(0.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS portability regression"
)
def test_fp64_scene_and_pair_metrics_transfer_to_cpu_before_float64_cast() -> None:
    first_core = torch.tensor([[[2.0], [4.0]]], device="mps")
    second_core = torch.tensor([[[1.0], [2.0]]], device="mps")
    first_delta = torch.tensor([[[0.2], [-0.2]]], device="mps")
    second_delta = torch.tensor([[[-0.1], [0.1]]], device="mps")

    scene_metrics = fp64_delta_metrics(first_core, first_delta)
    pair_metrics = fp64_pair_delta_metrics(
        first_core,
        second_core,
        first_delta,
        second_delta,
    )

    assert scene_metrics["positive_finite_core_rms"] is True
    assert scene_metrics["positive_finite_total_energy"] is True
    assert scene_metrics["core_rms"] > 0.0
    assert scene_metrics["delta_rms"] > 0.0
    assert pair_metrics["positive_finite_core_difference"] is True
    assert pair_metrics["positive_finite_pair_delta"] is True


def test_functional_simulation_reports_raw_and_effective_without_live_mutation() -> None:
    module = GlobalSceneResidual(
        scene_dim=8,
        latent_count=4,
        width=3,
        fourier_bands=2,
        initialization_seed=18018,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        gate_temperature=1.0,
    )
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randn(1, 4, 8, generator=generator)
    simulated_weight = (
        torch.randn(module.output_projection.weight.shape, generator=generator) * 1e-3
    )
    before = module_collection_state_sha256({"global_scene_residual": module})

    raw, effective = functional_simulated_deltas(module, tokens, simulated_weight)
    predicted = simulated_residual_state_sha256(module, simulated_weight)

    assert raw.dtype == torch.float32
    assert raw.mean(dim=1).abs().max().item() <= 1e-7
    assert torch.allclose(raw, effective, atol=2e-7, rtol=1e-5)
    assert predicted != before
    assert module_collection_state_sha256({"global_scene_residual": module}) == before
    assert torch.count_nonzero(module.output_projection.weight).item() == 0


def test_exact_clone_adamw_matches_real_optimizer_bitwise_without_live_mutation() -> None:
    module = GlobalSceneResidual(
        scene_dim=8,
        latent_count=4,
        width=3,
        fourier_bands=2,
        initialization_seed=18018,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        gate_temperature=1.0,
    )
    tokens = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(18))
    slot_weights = torch.tensor([1.0, -0.5, 0.25, -0.75]).view(1, 4, 1)
    (module(tokens) * slot_weights).sum().backward()
    reference = copy.deepcopy(module)
    for source, target in zip(module.parameters(), reference.parameters(), strict=True):
        target.grad = None if source.grad is None else source.grad.detach().clone()
    live_state_before = module_collection_state_sha256({"global_scene_residual": module})
    live_gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    }
    contract = _config()["structural_preflight"]["optimizer"]

    predicted_weight, evidence = _exact_clone_adamw_evidence(module, contract)

    reference_parameters = list(reference.parameters())
    torch.nn.utils.clip_grad_norm_(reference_parameters, 1.0)
    optimizer = torch.optim.AdamW(
        reference_parameters,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
        capturable=False,
        maximize=False,
        amsgrad=False,
    )
    optimizer.step()

    assert torch.equal(predicted_weight, reference.output_projection.weight)
    assert evidence["implementation"] == "isolated_full_residual_torch_adamw_clone"
    assert evidence["changed_parameter_keys"] == ["output_projection.weight"]
    assert evidence["clone_optimizer_state_parameter_count"] > 0
    assert module_collection_state_sha256({"global_scene_residual": module}) == live_state_before
    for name, parameter in module.named_parameters():
        expected = live_gradients_before[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected)


@dataclass(frozen=True)
class _Record:
    scene_id: str
    question_id: str


@dataclass(frozen=True)
class _Unit:
    pair_id: str
    question_key: str
    reference: _Record
    counterfactual: _Record


@dataclass(frozen=True)
class _Batch:
    kind: str
    pair_units: tuple[_Unit, ...]


def test_ordered_curriculum_hash_is_deterministic_and_order_sensitive() -> None:
    units = [
        _Unit("p1", "q1", _Record("s1", "r1"), _Record("s2", "c1")),
        _Unit("p2", "q2", _Record("s3", "r2"), _Record("s4", "c2")),
    ]
    curriculum = [_Batch("pair", (unit,)) for unit in units]

    entries, first = ordered_curriculum_evidence(curriculum)
    _, repeated = ordered_curriculum_evidence(curriculum)
    _, reversed_hash = ordered_curriculum_evidence(list(reversed(curriculum)))

    assert entries[0]["microstep"] == 1
    assert first == repeated
    assert first != reversed_hash
    assert len(first) == 64


def test_ordered_curriculum_rejects_standard_or_multiunit_microsteps() -> None:
    unit = _Unit("p1", "q1", _Record("s1", "r1"), _Record("s2", "c1"))
    with pytest.raises(V18StructuralPreflightViolation, match="not a pair"):
        ordered_curriculum_evidence([_Batch("standard", (unit,))])
    with pytest.raises(V18StructuralPreflightViolation, match="exactly one"):
        ordered_curriculum_evidence([_Batch("pair", (unit, unit))])


def test_rng_evidence_detects_and_restores_cpu_state_mutation() -> None:
    before = capture_rng_states(require_mps=False)
    try:
        torch.rand(4)
        after = capture_rng_states(require_mps=False)
        evidence = rng_state_evidence(before, after)
        assert evidence["domains"]["cpu"]["unchanged"] is False
        assert evidence["all_available_domains_unchanged"] is False
    finally:
        restore_rng_states(before)

    restored = capture_rng_states(require_mps=False)
    assert rng_state_evidence(before, restored)["domains"]["cpu"]["unchanged"] is True

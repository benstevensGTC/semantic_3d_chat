from __future__ import annotations

from copy import deepcopy

import pytest

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.residual_lr_response import (
    EXPECTED_ARM_SPECS,
    ResidualLRResponseViolation,
    summarize_residual_lr_response,
)

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SOURCE_COMMIT = "1" * 40
SOURCE_TREE = "2" * 40


@pytest.fixture
def configs() -> dict[str, dict]:
    return {
        name: load_config(
            f"configs/experiments/gemma4_color_mirror_global_scene_residual_v17_{name}.yaml"
        )
        for name in EXPECTED_ARM_SPECS
    }


def _source() -> dict:
    return {
        "schema_version": 1,
        "scope": "repository_excluding_generated_artifacts_v1",
        "available": True,
        "head_commit": SOURCE_COMMIT,
        "head_tree": SOURCE_TREE,
        "is_clean": True,
        "tracked_diff_sha256": EMPTY_SHA256,
    }


def _pair(
    *,
    full_sides: int,
    full_units: int,
    candidate_sides: int | None = None,
    candidate_units: int | None = None,
    mean_full: float = 1.0,
    minimum_full: float = 0.5,
    mean_candidate: float = 1.0,
    minimum_candidate: float = 0.5,
) -> dict:
    candidate_sides = full_sides if candidate_sides is None else candidate_sides
    candidate_units = full_units if candidate_units is None else candidate_units
    return {
        "ranking_mode": "candidate_logit",
        "same_next_token_distribution": True,
        "shared_candidate_tokens_excluded": True,
        "free_generation_evaluated": False,
        "first_answer_token_full_vocab_evaluated": True,
        "unit_count": 6,
        "side_count": 12,
        "first_answer_token_top1_unit_accuracy": full_units / 6,
        "first_answer_token_top1_accuracy": full_sides / 12,
        "changed_unit_accuracy": candidate_units / 6,
        "side_accuracy": candidate_sides / 12,
        "mean_first_answer_token_target_vs_best_other_logit_margin": mean_full,
        "minimum_first_answer_token_target_vs_best_other_logit_margin": minimum_full,
        "mean_own_vs_alternate_candidate_logit_margin": mean_candidate,
        "minimum_own_vs_alternate_candidate_logit_margin": minimum_candidate,
    }


def _gate(color: dict, mirror: dict) -> dict:
    return {"by_pair": {"pair_000001": color, "pair_000003": mirror}}


def _report(config: dict, mirrors: list[dict], *, colors: list[dict] | None = None) -> dict:
    response = config["lr_response"]
    source = _source()
    colors = colors or [_pair(full_sides=12, full_units=6) for _ in range(4)]
    history = [
        {"epoch": epoch, "pair_candidate_gate": _gate(color, mirror)}
        for epoch, (color, mirror) in enumerate(zip(colors, mirrors, strict=True), start=1)
    ]
    banks = {
        "inherited_v12": response["expected_frozen_inherited_bank_sha256"],
        "extension_v13": response["expected_frozen_extension_bank_sha256"],
    }
    residual_hash = response["expected_initial_residual_state_sha256"]
    prefixes = {
        f"scene_{index:06d}": {
            "core_prefix_sha256": str(index) * 64,
            "adapted_prefix_sha256": str(index) * 64,
        }
        for index in range(3, 7)
    }
    return {
        "output_namespace": config["training"]["output_namespace"],
        "optimizer_steps": 4,
        "epochs": 4,
        "target_epochs": 4,
        "steps": 48,
        "gradient_accumulation": 12,
        "stopped_early": False,
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": True,
        "question_dependent_scene_processing": False,
        "global_scene_residual_parameter_count": 400000,
        "lora_trainable_parameter_count": 0,
        "lora_optimizer": None,
        "selection": {
            "train": {
                "selected_ids_sha256": response["expected_selection_sha256"],
                "selected_count": 24,
            },
            "training_counterfactual_pair_membership_sha256": response[
                "expected_pair_membership_sha256"
            ],
            "source_provenance": source,
        },
        "training_counterfactual_pair_membership_sha256": response[
            "expected_pair_membership_sha256"
        ],
        "counterfactual_pair_unit_count": 12,
        "source_provenance": source,
        "initialize_expected_adapter_sha256": response["expected_source_adapter_sha256"],
        "initialize_expected_metadata_sha256": response["expected_source_metadata_sha256"],
        "initialization_provenance": {
            "schema_version": 3,
            "mode": "named_lora_banks_frozen_plus_zero_output_scene_residual",
            "adapter_sha256": response["expected_source_adapter_sha256"],
            "metadata_sha256": response["expected_source_metadata_sha256"],
            "expected_adapter_sha256": response["expected_source_adapter_sha256"],
            "expected_metadata_sha256": response["expected_source_metadata_sha256"],
            "checkpoint_epoch": 7,
            "optimizer_state_loaded": False,
            "history_loaded": False,
            "all_source_lora_banks_frozen": True,
            "global_scene_residual_zero_output": True,
            "source_lora_bank_state_sha256": banks,
            "global_scene_residual_initial_state_sha256": residual_hash,
        },
        "frozen_scene_state_sha256": response["expected_frozen_scene_state_sha256"],
        "frozen_lora_bank_state_sha256": banks,
        "lora_bank_state_sha256": banks,
        "global_scene_residual_initial_state_sha256": residual_hash,
        "global_scene_residual_state_sha256": "f" * 64,
        "global_scene_residual": {"expected_initial_state_sha256": residual_hash},
        "global_scene_residual_zero_output_equivalence": {
            "verified": True,
            "question_dependent_scene_processing": False,
            "scene_prefixes": prefixes,
        },
        "history": history,
        "pair_candidate_gate": history[-1]["pair_candidate_gate"],
    }


def _default_reports(configs: dict[str, dict]) -> dict[str, dict]:
    weak = [
        _pair(
            full_sides=6,
            full_units=0,
            mean_full=-0.1 - epoch,
            minimum_full=-2.0,
            minimum_candidate=-2.0,
        )
        for epoch in range(4)
    ]
    return {name: _report(config, deepcopy(weak)) for name, config in configs.items()}


def test_selects_best_color_eligible_epoch_and_authorizes_only_continuation(
    configs: dict[str, dict],
) -> None:
    reports = _default_reports(configs)
    reports["lr1e4"] = _report(
        configs["lr1e4"],
        [
            _pair(full_sides=6, full_units=0, mean_full=-0.1, minimum_full=-2.0),
            _pair(full_sides=7, full_units=1, mean_full=0.1, minimum_full=-1.0),
            _pair(full_sides=7, full_units=1, mean_full=0.2, minimum_full=-0.8),
            _pair(full_sides=5, full_units=0, mean_full=0.3, minimum_full=-0.5),
        ],
    )
    reports["lr3e4"] = _report(
        configs["lr3e4"],
        [
            _pair(full_sides=7, full_units=1, mean_full=0.4, minimum_full=-0.8),
            _pair(full_sides=8, full_units=2, mean_full=0.1, minimum_full=-0.7),
            _pair(full_sides=8, full_units=2, mean_full=0.2, minimum_full=-0.6),
            _pair(full_sides=6, full_units=0, mean_full=0.5, minimum_full=-0.4),
        ],
    )

    summary = summarize_residual_lr_response(configs, reports)

    assert summary["selected_arm"] == "lr3e4"
    assert summary["selected_epoch"] == 3
    assert summary["continuation_authorized"] is True
    assert summary["full_teacher_gate_passed"] is False
    assert summary["greedy_audit_authorized"] is False
    assert summary["decision"] == "continue_selected_arm_no_greedy_audit"


def test_lower_learning_rate_is_exact_final_tiebreaker(configs: dict[str, dict]) -> None:
    reports = _default_reports(configs)
    tied = _pair(full_sides=8, full_units=2, mean_full=0.25, minimum_full=-0.5)
    for name in reports:
        reports[name] = _report(
            configs[name],
            [
                deepcopy(tied),
                _pair(full_sides=6, full_units=0, mean_full=-0.1),
                _pair(full_sides=5, full_units=0, mean_full=-0.2),
                _pair(full_sides=4, full_units=0, mean_full=-0.3),
            ],
        )

    summary = summarize_residual_lr_response(configs, reports)

    assert summary["selected_arm"] == "lr1e4"
    assert summary["selected_epoch"] == 1


def test_greedy_audit_requires_selected_full_teacher_gate(configs: dict[str, dict]) -> None:
    reports = _default_reports(configs)
    perfect = _pair(
        full_sides=12,
        full_units=6,
        candidate_sides=12,
        candidate_units=6,
        mean_full=2.0,
        minimum_full=0.25,
        minimum_candidate=0.25,
    )
    reports["lr3e4"] = _report(
        configs["lr3e4"],
        [
            deepcopy(perfect),
            _pair(full_sides=6, full_units=0, mean_full=-0.1),
            _pair(full_sides=5, full_units=0, mean_full=-0.2),
            _pair(full_sides=4, full_units=0, mean_full=-0.3),
        ],
    )

    summary = summarize_residual_lr_response(configs, reports)

    assert summary["selected_arm"] == "lr3e4"
    assert summary["full_teacher_gate_passed"] is True
    assert summary["continuation_authorized"] is True
    assert summary["greedy_audit_authorized"] is True


def test_ambiguous_same_lr_epoch_tie_authorizes_nothing(configs: dict[str, dict]) -> None:
    reports = _default_reports(configs)
    best = _pair(full_sides=9, full_units=3, mean_full=0.25, minimum_full=-0.5)
    reports["lr1e4"] = _report(
        configs["lr1e4"],
        [deepcopy(best), deepcopy(best), _pair(full_sides=6, full_units=0)] * 1
        + [_pair(full_sides=5, full_units=0)],
    )

    summary = summarize_residual_lr_response(configs, reports)

    assert summary["selection_ambiguous"] is True
    assert summary["selected_arm"] is None
    assert summary["continuation_authorized"] is False
    assert summary["greedy_audit_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda reports: reports["lr3e4"].update(optimizer_steps=3), "four completed"),
        (
            lambda reports: reports["lr3e4"]["source_provenance"].update(head_tree="3" * 40),
            "exact clean source",
        ),
        (
            lambda reports: reports["lr1e4"]["selection"]["train"].update(
                selected_ids_sha256="0" * 64
            ),
            "selection hash",
        ),
        (
            lambda reports: reports["lr1e4"].update(frozen_scene_state_sha256="0" * 64),
            "frozen scene hash",
        ),
        (
            lambda reports: reports["lr1e4"].update(
                global_scene_residual_initial_state_sha256="0" * 64
            ),
            "residual initial hash",
        ),
    ],
)
def test_rejects_protocol_or_provenance_mutation(
    configs: dict[str, dict], mutation, message: str
) -> None:
    reports = _default_reports(configs)
    mutation(reports)

    with pytest.raises(ResidualLRResponseViolation, match=message):
        summarize_residual_lr_response(configs, reports)


def test_rejects_missing_or_extra_arm(configs: dict[str, dict]) -> None:
    reports = _default_reports(configs)

    with pytest.raises(ResidualLRResponseViolation, match="exactly"):
        summarize_residual_lr_response({"lr1e4": configs["lr1e4"]}, reports)


def test_rejects_learning_rate_or_config_hash_change(configs: dict[str, dict]) -> None:
    reports = _default_reports(configs)
    modified = deepcopy(configs)
    modified["lr1e4"]["training"]["learning_rate"] = 2.0e-4

    with pytest.raises(ResidualLRResponseViolation, match="config hash mismatch"):
        summarize_residual_lr_response(modified, reports)

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v5 import (
    NormalizedFactorizedSceneQuestionControlV5,
)
from semantic_3d_chat.training import compose_question_control_v64 as v64


def _v3(seed: int = 64) -> TeacherBasisFullSceneQuestionControlV3:
    torch.manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(8, 4)).Q.T.contiguous()
    return TeacherBasisFullSceneQuestionControlV3(
        8,
        basis,
        control_tokens=2,
        expected_environment_latents=4,
        moment_count=2,
        interaction_dim=3,
        trunk_dim=5,
        maximum_control_rms=0.3,
        initial_control_rms=0.1,
    ).eval()


def _sources() -> tuple[
    TeacherBasisFullSceneQuestionControlV3,
    NormalizedFactorizedSceneQuestionControlV5,
]:
    value = _v3()
    old_value = copy.deepcopy(value)
    with torch.no_grad():
        old_value.coefficient_output.weight.add_(0.125)
        old_value.magnitude_output.bias.sub_(0.25)
    route = NormalizedFactorizedSceneQuestionControlV5.from_v60(
        old_value,
        route_factor_rank=3,
    )
    torch.manual_seed(6401)
    with torch.no_grad():
        for parameter in route.factorized_route.parameters():
            parameter.copy_(torch.randn_like(parameter))
    return value, route.eval()


def _metadata(*, route: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "hidden_size": 1536,
        "control_tokens": 4,
        "expected_environment_latents": 256,
        "moment_count": 8,
        "base_checkpoint_sha256": "a" * 64,
        "base_runtime_config_sha256": "b" * 64,
        "weights_sha256": "c" * 64 if route else "d" * 64,
    }
    if route:
        result["route_factor_rank"] = 32
        result["source_v60_checkpoint_sha256"] = "3" * 64
        result["inherited_value_state_sha256"] = "4" * 64
    else:
        result["output_basis_rank"] = 128
    return result


def _common() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    authorization: dict[str, object] = {
        "baseline_lock_sha256": "e" * 64,
        "preregistration_sha256": v64.PINNED_V62_PREREGISTRATION_SHA256,
    }
    base: dict[str, object] = {
        "checkpoint_sha256": "a" * 64,
        "runtime_config_effective_sha256": "b" * 64,
    }
    inputs: dict[str, object] = {
        "filtered_training_qa_sha256": v64._PINNED_FILTERED_TRAIN_SHA256,
        "prefix_cache_manifest_sha256": "f" * 64,
        "training_scene_ids": [f"scene_{index:06d}" for index in range(1, 25)],
    }
    return authorization, base, inputs


def _route_report() -> dict[str, object]:
    authorization, base, inputs = _common()
    inputs.update(
        {
            "training_pair_ids": [f"pair_{index:06d}" for index in range(1, 13)],
            "natural_row_count": 576,
            "natural_changed_count": 80,
        }
    )
    return {
        "artifact": "v62_normalized_factorized_route_training",
        "offline_checks_passed": True,
        "promotion_eligible": False,
        "terminal_reason": "train_only_gates_passed_checkpoint_saved",
        "authorization": authorization,
        "source": {
            "v60_checkpoint_sha256": "3" * 64,
            "v60_weights_sha256": "5" * 64,
            "v60_state_sha256": "4" * 64,
        },
        "base": base,
        "inputs": inputs,
        "architecture": {
            "name": "normalized_factorized_scene_question_route_v5",
            "route_factor_rank": 32,
        },
        "cross_validation": {
            "method": "deterministic_leave_one_counterfactual_pair_out",
            "fold_count": 12,
            "pair_disjoint": True,
            "folds": [{} for _ in range(12)],
            "aggregate": {"passed": True, "checks": {"held": True}},
        },
        "final_fit": {"checks": {"routes": True, "reload": True}},
        "scope": {
            "gemma_backward_used": False,
            "gemma_generation_used": False,
            "only_factorized_route_trained": True,
            "v60_values_frozen": True,
            "question_dependent_scene_retrieval": False,
            "internal_validation_questions_loaded": False,
            "scorer_references_loaded": False,
            "prediction_answers_loaded": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
        },
        "checkpoint": {
            "weights_sha256": "c" * 64,
            "runtime_metadata_sha256": "1" * 64,
        },
    }


def _value_report() -> dict[str, object]:
    authorization, base, inputs = _common()
    inputs.update(
        {
            "training_record_count": 576,
            "counterfactual_pair_count": 12,
            "changed_teacher_side_count": 80,
            "changed_paired_unit_count": 40,
        }
    )
    return {
        "artifact": "v63_pair_disjoint_expanded_value_distillation",
        "offline_checks_passed": True,
        "promotion_eligible": False,
        "successor_factorized_route_required": True,
        "authorization": authorization,
        "source_v60": {
            "checkpoint_sha256": "3" * 64,
            "weights_sha256": "5" * 64,
            "runtime_metadata_sha256": "6" * 64,
            "architecture": "teacher_basis_full_scene_question_control_v3",
            "question_norm_sha256": "7" * 64,
            "question_norm_copied_tensor_exact": True,
            "question_norm_frozen_in_every_fit": True,
        },
        "base": base,
        "inputs": inputs,
        "architecture": {
            "name": "teacher_basis_full_scene_question_control_v3",
            "runtime_schema_version": 3,
            "hidden_size": 1536,
            "control_tokens": 4,
            "basis_rank_effective": 128,
            "scene_moment_count": 8,
            "all_256_scene_latents_used": True,
            "source_v60_question_norm_frozen": True,
        },
        "cross_validation": {
            "protocol": "deterministic_leave_one_counterfactual_pair_out",
            "pair_count": 12,
            "fold_specific_output_basis": True,
            "heldout_teacher_used_in_fold_basis": False,
            "heldout_teacher_used_in_fold_optimization": False,
            "each_teacher_evaluated_once": True,
            "passed": True,
            "checks": {"cosine": True},
            "aggregate": {
                "teacher_side_count": 80,
                "changed_pair_unit_count": 40,
            },
        },
        "final_fit": {
            "checks": {"reconstruction": True},
            "summary": {
                "teacher_side_count": 80,
                "changed_pair_unit_count": 40,
            },
        },
        "scope": {
            "gemma_backward_used": False,
            "gemma_generation_used": False,
            "numeric_teacher_cache_only": True,
            "teacher_cache_runtime_access": False,
            "complete_scene_prefix_retained": True,
            "question_dependent_scene_retrieval": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
            "report_contains_question_or_answer_text": False,
        },
        "checkpoint": {
            "weights_sha256": "d" * 64,
            "runtime_metadata_sha256": "2" * 64,
        },
    }


def test_parser_exposes_no_qa_validation_scorer_or_prediction_input() -> None:
    destinations = {action.dest for action in v64._parser()._actions}

    assert destinations == {
        "help",
        "baseline_lock",
        "v63_value_checkpoint",
        "v63_value_report",
        "v62_route_checkpoint",
        "v62_route_report",
        "output_checkpoint",
        "composition_report",
    }


def test_composer_copies_only_route_and_preserves_both_functions_exactly() -> None:
    value, route = _sources()
    value_before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
    route_before = {
        name: tensor.clone() for name, tensor in route.factorized_route.state_dict().items()
    }

    candidate, proof = v64.compose_controller(value, route)
    forward = v64.randomized_forward_equivalence(
        value_control=value,
        route_control=route,
        candidate=candidate,
        probe_count=2,
        seed=65,
    )

    assert proof["copied_state_prefixes"] == ["factorized_route."]
    assert proof["inherited_v63_tensors_exact"] is True
    assert proof["factorized_route_tensors_exact"] is True
    assert forward["value_outputs_exact"] is True
    assert forward["route_logits_exact"] is True
    assert all(
        torch.equal(candidate.state_dict()[name], tensor)
        for name, tensor in value_before.items()
    )
    assert all(
        torch.equal(candidate.factorized_route.state_dict()[name], tensor)
        for name, tensor in route_before.items()
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in candidate.named_parameters()
        if not name.startswith("factorized_route.")
    )


def test_composer_refuses_question_norm_mismatch_needed_for_route_identity() -> None:
    value, route = _sources()
    with torch.no_grad():
        value.question_norm.weight[0].add_(0.01)

    with pytest.raises(ValueError, match="question_norm differs"):
        v64.compose_controller(value, route)


def test_route_and_value_report_validators_return_identical_locked_provenance() -> None:
    route = v64.validate_v62_route_report(_route_report(), _metadata(route=True))
    value = v64.validate_v63_value_report(_value_report(), _metadata(route=False))

    assert route == value
    assert route["training_row_count"] == 576
    assert route["training_scene_count"] == 24
    assert route["training_pair_count"] == 12


def test_report_validator_rejects_failed_upstream_gate() -> None:
    report = _value_report()
    report["cross_validation"]["checks"]["cosine"] = False  # type: ignore[index]

    with pytest.raises(ValueError, match="value evidence failed"):
        v64.validate_v63_value_report(report, _metadata(route=False))


def test_output_isolation_refuses_source_ancestor(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="overlaps a source"):
        v64._output_paths(
            checkpoint=source / "candidate",
            report=tmp_path / "composition.json",
            inputs=(source,),
        )


def test_publisher_seals_runtime_minimal_schema5_checkpoint_create_once(
    tmp_path: Path,
) -> None:
    value, route = _sources()
    candidate, _proof = v64.compose_controller(value, route)
    checkpoint = tmp_path / "candidate"
    report_path = tmp_path / "reports" / "composition.json"
    sources = v64.V64Sources(
        baseline_lock_sha256="0" * 64,
        baseline_authorization={},
        value_checkpoint=tmp_path / "value",
        value_checkpoint_sha256="1" * 64,
        value_control=value,
        value_metadata={
            "base_checkpoint_sha256": "2" * 64,
            "base_runtime_config_sha256": "3" * 64,
        },
        value_report={},
        value_report_path=tmp_path / "value.json",
        value_report_sha256="4" * 64,
        route_checkpoint=tmp_path / "route",
        route_checkpoint_sha256="5" * 64,
        route_control=route,
        route_metadata={},
        route_report={},
        route_report_path=tmp_path / "route.json",
        route_report_sha256="6" * 64,
        output_checkpoint=checkpoint,
        composition_report=report_path,
    )

    sealed = v64._publish(
        sources=sources,
        candidate=candidate,
        report_without_checkpoint={"artifact": "synthetic_v64"},
    )

    assert sealed["checkpoint"]["inherited_value_state_sha256"] == v64._state_sha256(
        value.state_dict()
    )
    assert {item.name for item in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    assert report_path.is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        v64._publish(
            sources=sources,
            candidate=candidate,
            report_without_checkpoint={"artifact": "synthetic_v64"},
        )

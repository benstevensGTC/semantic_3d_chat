"""Locked design contract for a future Gemma-4 fixed-prefix PLE reader.

This module deliberately contains no training entry point.  It specifies and
validates the only allowed trainable surface and loss for an experiment that
may begin only after an independently accepted fixed-prefix atlas-v2 artifact
has been supplied by content hash.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import nn

from semantic_3d_chat.language.lora import LoRASettings

MODEL_ID: Final[str] = "google/gemma-4-E2B-it"
MODEL_REVISION: Final[str] = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
TARGET_MODULE: Final[str] = "model.language_model.per_layer_model_projection"
PROJECTION_IN_FEATURES: Final[int] = 1536
PROJECTION_OUT_FEATURES: Final[int] = 35 * 256
LORA_RANK: Final[int] = 4
LORA_ALPHA: Final[float] = 8.0
LORA_PARAMETER_COUNT: Final[int] = LORA_RANK * (
    PROJECTION_IN_FEATURES + PROJECTION_OUT_FEATURES
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def reader_lora_settings() -> LoRASettings:
    """Return the complete, exact-path, unmerged reader adapter contract."""

    return LoRASettings(
        enabled=True,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=0.0,
        target_modules=(TARGET_MODULE,),
    )


def validate_projection_surface(model: nn.Module) -> nn.Linear:
    """Reject any loaded model whose PLE projection differs from preregistration."""

    try:
        projection = model.get_submodule(TARGET_MODULE)
    except AttributeError as exc:
        raise ValueError(f"Missing preregistered PLE projection: {TARGET_MODULE}") from exc
    if not isinstance(projection, nn.Linear):
        raise TypeError("Preregistered PLE projection must be torch.nn.Linear")
    observed = (projection.in_features, projection.out_features, projection.bias is None)
    expected = (PROJECTION_IN_FEATURES, PROJECTION_OUT_FEATURES, True)
    if observed != expected:
        raise ValueError(f"PLE projection contract changed: {observed} != {expected}")
    return projection


def validate_launch_authorization(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable atlas-v2 evidence before loading Gemma or any QA rows."""

    required_hashes = (
        "atlas_v2_checkpoint_sha256",
        "atlas_v2_weights_sha256",
        "atlas_v2_runtime_metadata_sha256",
        "atlas_v2_acceptance_report_sha256",
    )
    hashes: dict[str, str] = {}
    for key in required_hashes:
        digest = value.get(key)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{key} must be a concrete lowercase SHA-256 digest")
        if digest == "0" * 64:
            raise ValueError(f"{key} cannot be a placeholder digest")
        hashes[key] = digest
    required_true = (
        "atlas_v2_accepted",
        "two_file_numeric_checkpoint_only",
        "compiler_frozen",
        "compiler_absent_at_runtime",
        "base_checkpoint_frozen",
        "question_independent_prefix",
        "complete_base_scene_prefix_preserved",
        "oracle_free_runtime",
    )
    for key in required_true:
        if value.get(key) is not True:
            raise ValueError(f"Launch authorization requires {key}=true")
    token_count = value.get("fixed_prefix_tokens")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 258:
        raise ValueError("fixed_prefix_tokens must preserve at least the 258-token base prefix")
    return {**hashes, **{key: True for key in required_true}, "fixed_prefix_tokens": token_count}


def answer_only_wrong_prefix_objective(
    correct_prefix_answer_nll: torch.Tensor,
    wrong_prefix_answer_nll: torch.Tensor,
    *,
    margin: float = 0.25,
    answer_ce_weight: float = 1.0,
    wrong_prefix_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Answer CE plus a same-question wrong-prefix ranking hinge.

    Each input is one answer-token-normalized NLL per row.  The wrong-prefix
    row must use the *same question and candidate answer* with only the fixed
    scene prefix changed.  Token labels outside the answer suffix are required
    to be ``-100`` by the caller and are independently audited at launch.
    """

    if (
        correct_prefix_answer_nll.ndim != 1
        or wrong_prefix_answer_nll.shape != correct_prefix_answer_nll.shape
        or correct_prefix_answer_nll.numel() < 1
    ):
        raise ValueError("Correct and wrong-prefix NLLs must be equal nonempty vectors")
    scalars = (margin, answer_ce_weight, wrong_prefix_weight)
    if any(not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in scalars):
        raise ValueError("Objective weights and margin must be finite")
    if margin <= 0 or answer_ce_weight <= 0 or wrong_prefix_weight <= 0:
        raise ValueError("Objective weights and margin must be positive")
    if not torch.isfinite(correct_prefix_answer_nll).all() or not torch.isfinite(
        wrong_prefix_answer_nll
    ).all():
        raise ValueError("Reader objective received NaN or infinity")
    prefix_margins = wrong_prefix_answer_nll - correct_prefix_answer_nll
    hinge = torch.relu(float(margin) - prefix_margins).mean()
    answer_ce = correct_prefix_answer_nll.mean()
    total = float(answer_ce_weight) * answer_ce + float(wrong_prefix_weight) * hinge
    return total, {
        "answer_only_ce": answer_ce,
        "wrong_prefix_hinge": hinge,
        "wrong_prefix_margins": prefix_margins,
    }


def build_ple_reader_preregistration() -> dict[str, Any]:
    """Return the immutable future-experiment design; this performs no training."""

    return {
        "schema_version": 1,
        "artifact": "gemma4_fixed_prefix_ple_reader_preregistration_v1",
        "status": "design_locked_training_not_authorized",
        "research_question": (
            "Can a rank-4 adapter on Gemma-4's per-layer model projection learn "
            "to read an accepted question-independent continuous 3D atlas prefix?"
        ),
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_inference_only": True,
            "projection_path": TARGET_MODULE,
            "projection_shape": [PROJECTION_OUT_FEATURES, PROJECTION_IN_FEATURES],
            "gemma4_layers": 35,
            "ple_dimension_per_layer": 256,
        },
        "trainable_surface": {
            "type": "strict_unmerged_fp32_lora",
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": 0.0,
            "parameter_count": LORA_PARAMETER_COUNT,
            "exact_target_modules": [TARGET_MODULE],
            "base_model_frozen": True,
            "scene_compiler_frozen": True,
            "atlas_frozen": True,
            "scene_tokenizer_frozen": True,
            "no_lora_merge": True,
        },
        "atlas_v2_launch_authorization": {
            "all_digest_values_required_at_launch": True,
            "digest_placeholders_permitted": False,
            "required_digests": [
                "atlas_v2_checkpoint_sha256",
                "atlas_v2_weights_sha256",
                "atlas_v2_runtime_metadata_sha256",
                "atlas_v2_acceptance_report_sha256",
            ],
            "atlas_v2_must_be_accepted_before_training": True,
            "no_mutable_atlas_v2_source_hash_is_preregistered": True,
            "compile_prefix_before_question": True,
            "reuse_identical_prefix_for_every_unchanged_scene_question": True,
            "compiler_absent_at_chat_runtime": True,
        },
        "data": {
            "training_jsonl": "data_gemma4/training/v62_pair_disjoint/train.jsonl",
            "training_sha256": "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1",
            "selection_manifest": "reports/gemma4/questions/v62_internal_validation.json",
            "selection_sha256": "078f65e1402e6e382a7bfdb2ad4b8a65d58e3164705a8a46cd222503aa201052",
            "test_manifest": "reports/gemma4/questions/test.json",
            "test_sha256": "785f41a071853b82d604cfd3b6906970ad0e3f085c040e9bec74a1c8078735e0",
            "scene_disjoint_splits_required": True,
            "test_manifest_unavailable_until_single_final_evaluation": True,
            "oracle_fields_allowed_only_in_offline_training_and_scoring": True,
        },
        "objective": {
            "answer_token_normalized_ce_weight": 1.0,
            "same_question_wrong_prefix_hinge_weight": 1.0,
            "same_question_wrong_prefix_margin_nats_per_answer_token": 0.25,
            "labels_before_answer_suffix_are_ignore_index": -100,
            "wrong_prefix_changes_only_scene_prefix": True,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
        },
        "optimization": {
            "seed": 710071,
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "batch_size": 1,
            "gradient_accumulation": 12,
            "gradient_clip_l2": 1.0,
            "maximum_updates": 96,
            "validate_every_updates": 8,
            "early_stopping_patience_validations": 4,
            "adapter_dtype": "float32",
            "base_dtype": "bfloat16_or_float32_fallback",
            "mps_safe": {
                "no_bitsandbytes": True,
                "no_cuda_requirement": True,
                "loss_and_log_softmax_float32": True,
                "finite_loss_gradient_and_parameter_checks_each_update": True,
                "cpu_fallback_on_unsupported_mps_operation": True,
            },
        },
        "hard_promotion_gates": {
            "selection_exact_accuracy_delta_minimum": 0.02,
            "selection_wrong_prefix_positive_margin_rate_minimum": 0.70,
            "selection_counterfactual_consistency_not_below_frozen_atlas": True,
            "text_retention_mean_ce_increase_nats_maximum": 0.03,
            "text_retention_mean_kl_nats_maximum": 0.02,
            "text_retention_next_token_top1_agreement_minimum": 0.98,
            "empty_prefix_scene_qa_drop_minimum": 0.10,
            "wrong_scene_prefix_answer_follow_rate_minimum": 0.60,
            "all_required_gates_must_pass": True,
            "failed_run_publishes_no_runtime_checkpoint": True,
        },
        "text_retention_controls": {
            "reason": "The target projection also processes ordinary text-token embeddings.",
            "corpus": "frozen_non_environmental_text_retention_set",
            "corpus_digest_required_before_training": True,
            "compare_against_exact_frozen_atlas_model": True,
            "measure_ce_kl_and_next_token_top1_agreement": True,
            "scene_questions_or_answers_forbidden_in_retention_corpus": True,
        },
        "runtime_and_leakage": {
            "runtime_may_load": [
                "sanitized_continuous_fixed_scene_prefix",
                "reader_lora_weights",
                "frozen_model_weights",
                "user_question_tokens",
            ],
            "runtime_may_not_load": [
                "oracle",
                "training_qa",
                "evaluation_answers",
                "scene_labels",
                "captions",
                "compiler",
            ],
            "loaded_file_audit_required": True,
            "oracle_directory_removal_test_required": True,
            "prefix_hash_before_first_question_required": True,
            "one_prefix_hash_per_unchanged_scene_required": True,
            "environmental_text_inputs": [],
        },
        "required_controls": [
            "frozen_reader_zero_lora",
            "correct_prefix",
            "wrong_scene_prefix",
            "empty_prefix",
            "same_prefix_different_questions",
            "oracle_directory_removed",
            "non_environmental_text_retention",
        ],
        "execution": {
            "training_executed": False,
            "gemma_generation_executed": False,
            "checkpoint_published": False,
            "implementation_must_be_new_files_or_explicitly_unsealed_before_run": True,
        },
    }


__all__ = [
    "LORA_PARAMETER_COUNT",
    "MODEL_ID",
    "MODEL_REVISION",
    "PROJECTION_IN_FEATURES",
    "PROJECTION_OUT_FEATURES",
    "TARGET_MODULE",
    "answer_only_wrong_prefix_objective",
    "build_ple_reader_preregistration",
    "reader_lora_settings",
    "validate_launch_authorization",
    "validate_projection_surface",
]

"""Unsealed CPU-only preregistration draft for Gemma-4 tool decoder V3.

The draft diagnoses the terminal V2.2 aggregate result and fixes one successor
arm.  It does not authorize a model load, MPS use, optimizer construction,
held-out evaluation, generation, or checkpoint publication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.training.gemma4_tool_decoder_v3_design import (
    GRADIENT_ACCUMULATION,
    MICROBATCH_COUNT,
    OPTIMIZER_UPDATES,
    SCHEDULE_SEED,
    TOKEN_ROLE_WEIGHTS,
    TRAIN_PREFIX_BYTES_SHA256,
    TRAIN_ROW_COUNT,
    V2_TERMINAL_PATH,
    V2_TERMINAL_SHA256,
    authenticate_v2_2_terminal_negative,
)

MODEL_ID: Final[str] = "google/gemma-4-E2B-it"
MODEL_REVISION: Final[str] = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
V3_BANK_NAME: Final[str] = "embodied_tool_decoder_v3_weighted_final_down"
V3_TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.down_proj"


def build_tool_decoder_v3_preregistration(project_root: Path) -> dict[str, Any]:
    """Return the fixed but explicitly unsealed V3 experiment contract."""

    terminal = authenticate_v2_2_terminal_negative(project_root)
    metrics = terminal["aggregate_metrics"]
    mean_answer_tokens = metrics["answer_token_count"] / metrics["sample_count"]
    return {
        "schema_version": "3.0-draft",
        "artifact": "gemma4_embodied_tool_decoder_v3_preregistration_draft",
        "status": "unsealed_cpu_design_only_training_unauthorized",
        "authorization": {
            "sealed": False,
            "training_authorized": False,
            "full_model_load_authorized": False,
            "mps_authorized": False,
            "optimizer_construction_authorized": False,
            "heldout_rows_authorized": False,
            "greedy_generation_authorized": False,
            "checkpoint_write_authorized": False,
        },
        "v2_2_terminal_negative": terminal,
        "diagnosis": {
            "evidence_scope": (
                "published aggregate V2.2 metrics only; no held-out row prediction "
                "or held-out token error was read or regenerated"
            ),
            "mean_answer_tokens_including_eos": mean_answer_tokens,
            "answer_token_nll": metrics["answer_token_nll"],
            "answer_token_accuracy": metrics["answer_token_accuracy"],
            "exact_sequence_accuracy": metrics["exact_sequence_accuracy"],
            "valid_schema_rate": metrics[
                "teacher_forced_argmax_valid_schema_rate"
            ],
            "tool_accuracy": metrics["teacher_forced_argmax_tool_accuracy"],
            "token_accuracy_minus_exact_sequence": (
                metrics["answer_token_accuracy"]
                - metrics["exact_sequence_accuracy"]
            ),
            "token_accuracy_minus_tool_accuracy": (
                metrics["answer_token_accuracy"]
                - metrics["teacher_forced_argmax_tool_accuracy"]
            ),
            "mechanistic_hypothesis_fixed_before_v3_training": (
                "The token-normalized V2 objective was dominated by repeated JSON "
                "punctuation and schema tokens. It learned common local syntax well "
                "enough to reach low mean NLL and high token accuracy, while the few "
                "action-name and numeric-value tokens remained underlearned; one "
                "wrong high-consequence token then destroyed sequence, schema, or tool "
                "accuracy. V3 tests this diagnosis with semantic-role weighting and "
                "argument-bin balance, not a post-hoc held-out error analysis."
            ),
        },
        "single_fixed_arm": {
            "arm_count": 1,
            "arm_id": "v3_weighted_final_down_r4_u100",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "base_checkpoint": (
                "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000"
            ),
            "frozen_language_model_except_fresh_lora": True,
            "fresh_bank_name": V3_BANK_NAME,
            "target_module": V3_TARGET_MODULE,
            "target_shape": [1536, 12288],
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "lora_dropout": 0.0,
            "lora_parameter_count": 55296,
            "numeric_context_projector_parameter_count": 110592,
            "total_trainable_parameter_count": 165888,
            "initialization_seed": SCHEDULE_SEED,
            "lora_learning_rate": 0.0001,
            "numeric_projector_learning_rate": 0.0002,
            "weight_decay": 0.0,
            "gradient_clip_l2": 1.0,
            "microbatch_size": 1,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "microbatch_count": MICROBATCH_COUNT,
            "optimizer_updates": OPTIMIZER_UPDATES,
            "checkpoint_selection": "fixed_final_update_no_posthoc_selection",
            "no_classifier_or_non_lm_action_head": True,
            "no_grammar_constraint_in_primary_evaluation": True,
        },
        "surface_isolation": {
            "static_v6_reserved_modules": [
                "model.language_model.layers.32.mlp.down_proj",
                "model.language_model.layers.33.mlp.down_proj",
            ],
            "v3_module": V3_TARGET_MODULE,
            "disjoint_from_static_v6": True,
            "v2_module_reused_only_after_terminal_nonpublication": True,
            "v2_runtime_checkpoint_absent": True,
            "v2_and_v3_banks_must_never_be_installed_together": True,
            "joint_static_v6_plus_tool_v3_runtime_requires_new_composition_probe": True,
        },
        "training_data": {
            "source": "data_gemma4/training/navigation_policy_v3/traces.jsonl",
            "permitted_prefix_only_rows": TRAIN_ROW_COUNT,
            "permitted_prefix_bytes_sha256": TRAIN_PREFIX_BYTES_SHA256,
            "train_scene_count": 14,
            "heldout_rows_read_during_design": 0,
            "heldout_predictions_read_during_design": 0,
            "oracle_training_targets_permitted": True,
            "oracle_inputs_at_runtime": False,
        },
        "deterministic_sampler": {
            "seed": SCHEDULE_SEED,
            "global_action_round_robin": True,
            "action_count_difference_maximum": 0,
            "within_action_occupied_argument_bin_round_robin": True,
            "argument_bin_edges": [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            "argument_free_bin": "none",
            "empty_bins_are_not_synthesized": True,
            "schedule_is_fixed_before_model_load": True,
        },
        "weighted_answer_tail_objective": {
            "labels_before_answer_suffix": -100,
            "full_sequence_vocabulary_logits_materialized": False,
            "logits_to_keep": "answer_label_positions_minus_one",
            "cross_entropy_dtype": "float32",
            "token_roles": list(TOKEN_ROLE_WEIGHTS),
            "role_weights": dict(TOKEN_ROLE_WEIGHTS),
            "token_role_assignment": (
                "fast-tokenizer character offsets; priority action, argument_value, "
                "argument_key, schema_key, then structure; EOS is explicit"
            ),
            "normalization": "sum(weight_i * CE_i) / sum(weight_i) per answer",
            "then": "mean over gradient-accumulation microbatches",
            "unweighted_nll_still_reported": True,
            "weighted_role_nll_diagnostics_reported": True,
            "objective_cannot_change_after_cpu_seal": True,
        },
        "unchanged_early_teacher_forced_gates": {
            "all_heldout_sample_count": 2268,
            "all_heldout_scene_count": 8,
            "answer_token_nll_maximum": 2.0,
            "answer_token_accuracy_minimum": 0.8,
            "exact_sequence_accuracy_minimum": 0.3,
            "teacher_forced_argmax_valid_schema_rate_minimum": 0.8,
            "teacher_forced_argmax_tool_accuracy_minimum": 0.7,
            "must_pass_before_controls_or_greedy": True,
        },
        "unchanged_causal_controls": {
            "conditions": [
                "primary",
                "wrong_scene",
                "zero_scene",
                "wrong_robot",
                "zero_robot",
                "wrong_target",
                "zero_target",
                "wrong_clearance",
                "zero_clearance",
            ],
            "teacher_rows_per_condition": 448,
            "wrong_scene_nll_increase_minimum": 0.01,
            "zero_scene_nll_increase_minimum": 0.01,
            "wrong_robot_targeted_nll_increase_minimum": 0.01,
            "zero_robot_targeted_nll_increase_minimum": 0.01,
            "wrong_target_targeted_nll_increase_minimum": 0.02,
            "zero_target_targeted_nll_increase_minimum": 0.02,
            "wrong_clearance_targeted_nll_increase_minimum": 0.01,
            "zero_clearance_targeted_nll_increase_minimum": 0.01,
            "all_must_pass_before_greedy": True,
        },
        "unchanged_primary_greedy_gates": {
            "decoder": "raw_deterministic_argmax",
            "maximum_new_tokens": 24,
            "primary_sample_count": 448,
            "control_sample_count_per_alteration": 56,
            "total_unique_sequences": 896,
            "exact_json_accuracy_minimum": 0.6,
            "valid_schema_rate_minimum": 0.95,
            "tool_accuracy_minimum": 0.8,
            "turn_sign_accuracy_minimum": 0.8,
            "argument_mae_normalized_maximum": 0.25,
            "unsafe_motion_count_maximum": 0,
            "wrong_clearance_output_change_rate_minimum": 0.1,
            "zero_clearance_output_change_rate_minimum": 0.1,
            "grammar_constrained_secondary_diagnostic_cannot_rescue_failure": True,
        },
        "resource_budget_after_future_release": {
            "staged_full_model_mps_backward_smokes": 1,
            "training_microbatches": MICROBATCH_COUNT,
            "optimizer_updates": OPTIMIZER_UPDATES,
            "teacher_forced_unique_forwards_maximum": 5852,
            "greedy_unique_sequences_maximum": 896,
            "greedy_decode_tokens_maximum": 21504,
            "driver_memory_gib_maximum": 12.5,
            "process_peak_rss_gib_maximum": 8.0,
            "wall_clock_seconds_maximum": 7200,
            "stop_on_nonfinite_loss_or_gradient": True,
            "stop_before_greedy_on_teacher_gate_failure": True,
        },
        "runtime_seam": {
            "input_order": [
                "BOS",
                "native_BOI",
                "256_complete_question_independent_scene_latents",
                "4_numeric_robot_state_tokens",
                "native_EOI",
                "environment_free_protocol_and_literal_user_instruction",
                "2_continuous_grounded_target_tokens",
                "2_anonymous_numeric_clearance_tokens",
            ],
            "scene_prefix_built_before_instruction": True,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "tool_decoder": "Gemma causal LM raw deterministic argmax",
            "tool_call_validation": "existing strict numeric JSON schema",
            "collision_interlock": "existing numeric geometry interlock",
            "checkpoint_files_if_and_only_if_promoted": [
                "adapter.safetensors",
                "metadata.json",
            ],
            "checkpoint_must_bind": [
                "V3 terminal result",
                "V3 authorization ancestry",
                "fresh V3 bank tensor hash",
                "numeric projector tensor hash",
                "objective and schedule hashes",
                "all gate metrics",
                "saved-runtime probe",
            ],
            "saved_runtime_probe_required_before_atomic_publication": True,
        },
        "future_fail_closed_stage_chain": [
            "root audit of this mutable draft and CPU preflight",
            "separate immutable CPU authorization binding every executable source",
            "separate one-shot full-model MPS backward smoke release",
            "separate multi-update release only after exact smoke pass",
            "fixed final update evaluation with unchanged gates",
            "saved-runtime numeric execution probe",
            "atomic publication only if every gate passes",
        ],
        "current_execution": {
            "full_model_loaded": False,
            "tokenizer_loaded_by_preregistration": False,
            "mps_used": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "heldout_rows_read": 0,
            "heldout_predictions_read": 0,
            "greedy_generations": 0,
            "checkpoint_written": False,
            "training_authorized": False,
        },
        "parent_evidence": {
            "path": str(V2_TERMINAL_PATH),
            "sha256": V2_TERMINAL_SHA256,
        },
    }


__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "V3_BANK_NAME",
    "V3_TARGET_MODULE",
    "build_tool_decoder_v3_preregistration",
]

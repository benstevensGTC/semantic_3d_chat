from __future__ import annotations

from pathlib import Path

import pytest
import torch

from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration import (
    CONFIG_SHA256,
    RESERVED_TOOL_TARGET,
    _authenticate_prefix_cache,
    _initial_state_contract,
    _shape_only_joint_install_roundtrip,
    _ShapeOnlyDecoder,
    answer_varying_inventory,
    answer_varying_wrong_prefixes,
    authenticate_frozen_inputs_and_sources,
    build_preregistration_draft,
    build_v6_schedule,
    decoder_contrastive_objective,
    fixed_greedy_subset_contract,
    learning_rate_v6,
    measure_sequence_lengths_v6,
    sequence_length_contract,
    structural_preflight,
    tiny_cpu_gradient_architecture_smoke,
)
from semantic_3d_chat.language.fixed_prefix_decoder_reader_v6 import (
    EXPECTED_LAYER_TYPES,
    INITIAL_STATE_SHA256,
    LORA_PARAMETER_COUNT,
    LORA_PARAMETER_COUNT_PER_MODULE,
    SLIDING_WINDOW_TOKENS,
    TARGET_MODULES,
    decoder_reader_lora_settings_v6,
    validate_decoder_reader_surface_v6,
)
from semantic_3d_chat.training.train_fixed_prefix_ple_v54 import (
    load_training_records,
    load_validation_records,
)


def test_v6_surface_is_two_fresh_upper_decoder_down_projections() -> None:
    assert TARGET_MODULES == (
        "model.language_model.layers.32.mlp.down_proj",
        "model.language_model.layers.33.mlp.down_proj",
    )
    assert RESERVED_TOOL_TARGET == "model.language_model.layers.34.mlp.down_proj"
    assert RESERVED_TOOL_TARGET not in TARGET_MODULES
    assert LORA_PARAMETER_COUNT_PER_MODULE == 55_296
    assert LORA_PARAMETER_COUNT == 110_592
    settings = decoder_reader_lora_settings_v6()
    assert settings.rank == 4
    assert settings.alpha == 8.0
    assert settings.dropout == 0.0
    assert settings.target_modules == TARGET_MODULES
    projections = validate_decoder_reader_surface_v6(_ShapeOnlyDecoder())
    assert len(projections) == 2
    assert all((item.in_features, item.out_features) == (12_288, 1_536) for item in projections)
    assert len(EXPECTED_LAYER_TYPES) == 35
    assert EXPECTED_LAYER_TYPES == (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ) * 7
    assert SLIDING_WINDOW_TOKENS == 512

    wrong_layers = _ShapeOnlyDecoder()
    wrong_layers.config.text_config.layer_types = (*EXPECTED_LAYER_TYPES[:-1], "sliding_attention")
    with pytest.raises(ValueError, match="all 35"):
        validate_decoder_reader_surface_v6(wrong_layers)
    wrong_window = _ShapeOnlyDecoder()
    wrong_window.config.text_config.sliding_window = 511
    with pytest.raises(ValueError, match="512-token"):
        validate_decoder_reader_surface_v6(wrong_window)


def test_v6_initial_state_is_deterministic_exact_zero_and_auditable() -> None:
    first = _initial_state_contract()
    second = _initial_state_contract()

    assert first == second
    assert first["parameter_count"] == 110_592
    assert first["parameter_counts"] == {target: 55_296 for target in TARGET_MODULES}
    assert first["initial_state_sha256"] == INITIAL_STATE_SHA256
    assert first["exact_zero_b"] is True
    assert first["only_authorized_parameters_trainable"] is True


def test_v6_expands_exact_question_controls_with_strict_scope_reporting() -> None:
    train = load_training_records()
    validation = load_validation_records()
    train_inventory = answer_varying_inventory(train)
    validation_inventory = answer_varying_inventory(validation)

    assert train_inventory["answer_varying_exact_question_groups"] == 35
    assert train_inventory["answer_varying_rows"] == 288
    assert train_inventory["nonvarying_rows"] == 288
    assert train_inventory["curated_changed_rows"] == 80
    assert train_inventory["candidate_scope_counts"] == {
        "cross_pair_candidates_only": 208,
        "same_and_cross_pair_candidates": 53,
        "same_pair_candidates_only": 27,
    }
    assert train_inventory["selected_negative_scope_counts"] == {
        "cross_pair": 208,
        "same_counterfactual_pair": 80,
    }
    assert train_inventory["selected_answer_type_family_distribution"] == {
        "attribute": 70,
        "count": 28,
        "metric": 24,
        "orientation": 22,
        "presence": 12,
        "spatial_relation": 62,
        "support": 70,
    }
    assert sum(train_inventory["selected_correct_to_wrong_answer_pair_distribution"].values()) == 288
    assert train_inventory["answer_frequency_per_group_cell_minimum"] == 1
    assert train_inventory["answer_frequency_per_group_cell_maximum"] == 23
    assert train_inventory["eligible_candidate_count_per_row_minimum"] == 1
    assert train_inventory["eligible_candidate_count_per_row_maximum"] == 23
    assert train_inventory["selected_wrong_scene_reuse_within_question_minimum"] == 1
    assert train_inventory["selected_wrong_scene_reuse_within_question_maximum"] == 23
    assert validation_inventory["answer_varying_exact_question_groups"] == 29
    assert validation_inventory["answer_varying_rows"] == 170
    assert validation_inventory["nonvarying_rows"] == 214
    assert validation_inventory["curated_changed_rows"] == 52
    assert validation_inventory["candidate_scope_counts"] == {
        "cross_pair_candidates_only": 118,
        "same_and_cross_pair_candidates": 23,
        "same_pair_candidates_only": 29,
    }
    assert validation_inventory["selected_negative_scope_counts"] == {
        "cross_pair": 118,
        "same_counterfactual_pair": 52,
    }
    assert validation_inventory["selected_answer_type_family_distribution"] == {
        "attribute": 44,
        "count": 12,
        "metric": 16,
        "orientation": 14,
        "presence": 8,
        "spatial_relation": 32,
        "support": 44,
    }
    assert (
        sum(validation_inventory["selected_correct_to_wrong_answer_pair_distribution"].values())
        == 170
    )
    assert validation_inventory["answer_frequency_per_group_cell_minimum"] == 1
    assert validation_inventory["answer_frequency_per_group_cell_maximum"] == 15
    assert validation_inventory["eligible_candidate_count_per_row_minimum"] == 1
    assert validation_inventory["eligible_candidate_count_per_row_maximum"] == 15
    assert validation_inventory["selected_wrong_scene_reuse_within_question_minimum"] == 1
    assert validation_inventory["selected_wrong_scene_reuse_within_question_maximum"] == 15
    assert "imbalanced" in train_inventory["frequency_imbalance_warning"]
    assert "cross-scene exact-question causality" in train_inventory["scientific_scope"]


def test_v6_wrong_prefix_assignment_is_deterministic_and_answer_different() -> None:
    rows = load_training_records()
    first = answer_varying_wrong_prefixes(rows)
    second = answer_varying_wrong_prefixes(rows)
    index = {(row.scene_id, row.question_id): row for row in rows}
    by_scene_question = {(row.scene_id, row.question): row for row in rows}

    assert first == second
    assert len(first) == 288
    for key, wrong_scene in first.items():
        row = index[key]
        wrong = by_scene_question[(wrong_scene, row.question)]
        assert wrong_scene != row.scene_id
        assert wrong.answer != row.answer
        if row.changed:
            assert wrong_scene == row.paired_scene_id


def test_v6_schedule_uses_every_training_row_once_in_96_fixed_updates() -> None:
    rows = load_training_records()
    schedule = build_v6_schedule(rows)
    observed = [
        (row.scene_id, row.question_id)
        for update in schedule
        for row in (*update.contrastive, *update.broad)
    ]

    assert len(schedule) == 96
    assert all(len(update.contrastive) == 3 for update in schedule)
    assert all(len(update.broad) == 3 for update in schedule)
    assert len(observed) == len(set(observed)) == 576
    assert build_v6_schedule(rows) == schedule


def test_v6_learning_rate_formula_and_endpoints_are_exact() -> None:
    assert learning_rate_v6(1) == 0.0000125
    assert learning_rate_v6(8) == 0.0001
    assert learning_rate_v6(96) == 0.00001
    assert all(
        learning_rate_v6(update) > learning_rate_v6(update + 1)
        for update in range(8, 96)
    )
    with pytest.raises(ValueError, match=r"\[1, 96\]"):
        learning_rate_v6(0)


def test_v6_objective_preserves_v5_weights_exactly() -> None:
    correct = torch.tensor([0.7, 0.9, 1.2], requires_grad=True)
    wrong = torch.tensor([0.8, 1.6, 1.1], requires_grad=True)
    broad = torch.tensor([2.0, 1.5, 1.0], requires_grad=True)
    retention = torch.tensor(0.02, requires_grad=True)
    loss, diagnostics = decoder_contrastive_objective(correct, wrong, broad, retention)
    expected = (
        0.5 * correct.mean()
        + 4.0 * torch.relu(0.5 - (wrong - correct)).mean()
        + 0.5 * broad.mean()
        + 0.5 * retention
    )

    assert torch.equal(loss, expected)
    assert torch.equal(diagnostics["wrong_prefix_margins"], wrong - correct)
    loss.backward()
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (correct, wrong, broad, retention)
    )
    with pytest.raises(ValueError, match="shapes"):
        decoder_contrastive_objective(correct, wrong[:2], broad, retention)


def test_v6_draft_is_unsealed_fail_closed_and_keeps_both_gate_families() -> None:
    draft = build_preregistration_draft()
    gates = draft["promotion_gates"]
    execution = draft["execution"]

    assert draft["status"] == "unsealed_cpu_only_draft_training_not_authorized"
    assert "structural target disjointness only" in draft["trainable_surface"][
        "layer_34_claim_scope"
    ]
    joint = draft["trainable_surface"]["coexistence"][
        "shape_only_joint_install_and_state_roundtrip"
    ]
    assert joint["strict_state_load"] is True
    assert joint["real_gemma_checkpoint_loaded"] is False
    assert joint["runtime_semantic_or_tool_behavior_proven"] is False
    assert draft["optimization"]["final_state_after_update_96_is_only_candidate"]
    assert draft["optimization"]["intermediate_or_best_loss_selection"] is False
    curated = gates["v5_curated_52_side_gates_unchanged"]
    assert curated["positive_margin_rate_minimum"] == 0.65
    assert curated["positive_margin_rate_delta_minimum"] == 0.10
    assert curated["complete_pair_unit_delta_minimum"] == 3
    expanded = gates["expanded_170_side_gates"]
    assert expanded["positive_margin_rate_minimum"] == 0.65
    assert expanded["positive_margin_rate_delta_minimum"] == 0.10
    assert expanded["answer_type_strata"]["all_7_families_required"] is True
    assert expanded["answer_type_strata"]["macro_positive_margin_rate_minimum"] == 0.65
    assert expanded["selected_negative_scope_strata"]["both_scopes_required"] is True
    assert gates["greedy"]["row_count"] == 96
    assert gates["greedy"]["opaque_row_key_sha256"] == (
        "1d06ad2292635f438af38bfb31f05d0502972244b2c46d9691067fc8fb6756cd"
    )
    assert draft["configuration"]["sha256"] == CONFIG_SHA256
    assert draft["failure_driven_transition"]["exact_inheritance_claim"] is False
    assert draft["objective"]["objective_is_not_exactly_inherited_from_v5"] is True
    assert execution == {
        "preregistration_sealed": False,
        "full_gemma_loaded": False,
        "mps_used": False,
        "optimizer_constructed": False,
        "training_authorized": False,
        "training_executed": False,
        "checkpoint_published": False,
    }


def test_v6_authenticates_every_input_prefix_and_critical_source_fail_closed() -> None:
    authenticated = authenticate_frozen_inputs_and_sources()
    assert len(authenticated["listed_inputs"]) == 10
    assert authenticated["prefix_cache"] == _authenticate_prefix_cache()
    assert authenticated["prefix_cache"]["scene_count"] == 40
    assert authenticated["prefix_cache"]["all_40_prefix_files_authenticated"] is True
    assert len(authenticated["critical_sources"]) == 14


def test_v6_sequence_and_greedy_populations_are_fully_bound() -> None:
    lengths = sequence_length_contract()
    assert lengths["maximum_train_teacher_sequence_tokens"] == 324
    assert lengths["maximum_validation_teacher_sequence_tokens"] == 325
    assert lengths["maximum_retention_prompt_tokens"] == 13
    assert lengths["maximum_validation_greedy_sequence_tokens"] == 354
    assert lengths["all_teacher_and_preregistered_greedy_sequences_within_window"] is True
    greedy = fixed_greedy_subset_contract(load_validation_records())
    assert greedy["row_count"] == 96
    assert greedy["maximum_new_tokens"] == 32


def test_v6_pinned_tokenizer_recomputes_exact_sequence_maxima_without_model() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/gemma-4-E2B-it",
            revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            local_files_only=True,
        )
    except (AttributeError, ImportError) as exc:
        pytest.skip(f"environment tokenizer loader is incompatible: {exc}")
    assert measure_sequence_lengths_v6(tokenizer) == {
        "maximum_greedy_new_tokens": 32,
        "maximum_retention_prompt_tokens": 13,
        "maximum_train_answer_tokens": 4,
        "maximum_train_prompt_tokens": 63,
        "maximum_train_teacher_sequence_tokens": 324,
        "maximum_validation_answer_tokens": 4,
        "maximum_validation_greedy_sequence_tokens": 354,
        "maximum_validation_prompt_tokens": 64,
        "maximum_validation_teacher_sequence_tokens": 325,
    }


def test_v6_shape_only_joint_install_roundtrip_is_explicitly_not_behavioral() -> None:
    report = _shape_only_joint_install_roundtrip()
    assert report["device"] == "cpu"
    assert report["target_sets_disjoint"] is True
    assert report["strict_state_load"] is True
    assert report["real_gemma_checkpoint_loaded"] is False
    assert report["runtime_semantic_or_tool_behavior_proven"] is False


def test_v6_structural_preflight_loads_no_model_and_authorizes_nothing() -> None:
    report = structural_preflight()

    assert report["passed"] is True
    assert report["status"] == "passed_cpu_no_model_draft_training_not_authorized"
    assert report["targets"] == list(TARGET_MODULES)
    assert report["trainable_parameter_count"] == 110_592
    assert report["training_answer_varying_rows"] == 288
    assert report["validation_answer_varying_rows"] == 170
    assert report["schedule_updates"] == 96
    assert report["full_checkpoint_loaded"] is False
    assert report["mps_used"] is False
    assert report["optimizer_constructed"] is False
    assert report["training_authorized"] is False


def test_v6_tiny_true_gemma_cpu_smoke_proves_both_gradients_and_coexistence() -> None:
    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    report = tiny_cpu_gradient_architecture_smoke()

    assert report["passed"] is True
    assert report["device"] == "cpu"
    assert report["full_checkpoint_loaded"] is False
    assert report["mps_used"] is False
    assert report["optimizer_constructed"] is False
    assert report["zero_output_exact_noop"] is True
    assert report["answer_logit_positions_only"] is True
    assert report["base_or_reserved_layer34_trainable_parameter_count"] == 0
    assert set(report["lora_b_gradient_l2_by_target"]) == set(TARGET_MODULES)
    assert all(value > 0.0 for value in report["lora_b_gradient_l2_by_target"].values())
    assert set(report["lora_a_gradient_l2_expected_zero_by_target"].values()) == {0.0}
    assert report["nonzero_adapter_maximum_answer_logit_change"] > 0.0


def test_v6_proposal_source_has_no_heavy_or_mutating_entrypoint() -> None:
    source = Path(
        "src/semantic_3d_chat/evaluation/fixed_prefix_decoder_reader_v6_preregistration.py"
    ).read_text(encoding="utf-8")

    assert 'choices=("draft", "preflight", "tiny-smoke")' in source
    assert "from_pretrained" not in source
    assert "torch.optim" not in source
    assert "save_file" not in source
    assert "write_preregistration" not in source
    assert "reports/gemma4/questions/test.json" not in source

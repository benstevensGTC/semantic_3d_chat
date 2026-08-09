from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import semantic_3d_chat.evaluation.scene_signal_audit as audit_module
from semantic_3d_chat.evaluation.scene_signal_audit import (
    _canonical_relation_generation_summary,
    _checkpoint_epoch_fields,
    _configured_runtime_dtype,
    _construct_audit_composer,
    _encode_scene,
    _generation_answer_and_provenance,
    _generation_audit,
    _install_checkpoint_lora,
    _left_right_reference_orientation_summary,
    _normalized_generation_result,
    _normalized_generation_summary,
    _summary_findings,
    _training_selected_pair_keys,
    _unvalidated_runtime_prefix_status,
    _validate_runtime_prefix_against_loaded_model,
)
from semantic_3d_chat.language.lora import (
    install_lora_adapters,
    lora_checkpoint_contract,
    lora_optimizer_settings,
    lora_settings,
)
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    ContinuousPrefixComposer,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.training.checkpointing import (
    load_adapter_checkpoint,
    module_collection_state_sha256,
    save_adapter_checkpoint,
)


def _native_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_revision": "pinned-revision",
        "bos_token_id": 2,
        "pad_token_id": 0,
        "boi_token_id": 10,
        "image_token_id": 11,
        "eoi_token_id": 12,
        "use_bidirectional_attention": None,
    }


def _native_config(dtype: str = "bfloat16") -> dict:
    return {
        "paths": {"data_root": "data", "maps_root": "data/maps"},
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "scene_encoder": {"input_voxel_size_m": 0.15},
        "language": {
            "backend": "gemma4",
            "revision": "pinned-revision",
            "dtype": dtype,
            "scene_prefix_after_bos": True,
            "scene_boundary_mode": SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
            "gemma4_native_image_contract": _native_contract(),
        },
    }


def _tiny_map() -> MapTensorData:
    return MapTensorData(
        semantic=torch.zeros(2, 3),
        xyz=torch.zeros(2, 3),
        rgb=torch.zeros(2, 3),
        normal=torch.zeros(2, 3),
        confidence=torch.ones(2),
        observation_count=torch.ones(2),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=2,
        input_voxel_size_m=0.15,
    )


class _SceneModel(nn.Module):
    def forward(self, semantic, *_args):
        return SimpleNamespace(
            scene_tokens=torch.tensor(
                [[[0.33333334, -0.7777778], [1.125, -1.875]]],
                device=semantic.device,
            ),
            native_latents=torch.zeros(1, 2, 2, device=semantic.device),
            block_tokens=torch.zeros(1, 2, device=semantic.device),
            audit={"block_indices": torch.zeros(1, 3, dtype=torch.long)},
        )


class _LoadedLanguage:
    def __init__(self, contract, boundaries, dtype: torch.dtype) -> None:
        self._contract = contract
        self._boundaries = boundaries
        self.model = nn.Linear(2, 2, bias=False).to(dtype=dtype)

    def scene_boundary_contract(self, _mode):
        return self._contract

    def scene_boundary_embeddings(self, _mode):
        return self._boundaries


def test_audit_projection_uses_supplied_effective_runtime_dtype_and_neutral_names(
    monkeypatch,
) -> None:
    monkeypatch.setattr(audit_module, "load_map_tensors", lambda *_args, **_kwargs: _tiny_map())
    composer = ContinuousPrefixComposer(2)
    _, representation = _encode_scene(
        _native_config(),
        "scene_000001",
        _SceneModel(),
        composer,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert representation["projected_scene_tokens_runtime_dtype"].dtype is torch.bfloat16
    assert representation["final_prefix_runtime_dtype"].dtype is torch.bfloat16
    assert "projected_scene_tokens_runtime_float16" not in representation
    assert "final_prefix_runtime_float16" not in representation


def test_skip_generation_explicitly_disclaims_model_and_boundary_validation(
    monkeypatch,
) -> None:
    captured = {}

    def fake_safe_dtype(device, requested):
        captured.update(device=device, requested=requested)
        return torch.bfloat16

    monkeypatch.setattr(audit_module, "safe_dtype", fake_safe_dtype)
    config = _native_config()
    runtime_dtype = _configured_runtime_dtype(config, torch.device("mps"))
    status = _unvalidated_runtime_prefix_status(config, runtime_dtype)

    assert runtime_dtype is torch.bfloat16
    assert captured == {"device": torch.device("mps"), "requested": "bfloat16"}
    assert status["status"] == "checkpoint_projected_not_model_validated"
    assert status["configured_runtime_dtype"] == "bfloat16"
    assert status["base_model_loaded"] is False
    assert status["native_boundary_validation_required"] is True
    assert status["native_boundary_embeddings_validated"] is False
    assert status["runtime_prefix_parity_validated"] is False


def test_loaded_model_validation_checks_dtype_contract_and_checkpoint_boundaries() -> None:
    torch.manual_seed(101)
    native = (
        torch.randn(1, 1, 2, dtype=torch.bfloat16),
        torch.randn(1, 1, 2, dtype=torch.bfloat16),
    )
    composer = ContinuousPrefixComposer(
        2,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=native,
    )
    language = _LoadedLanguage(_native_contract(), native, torch.bfloat16)
    status = _validate_runtime_prefix_against_loaded_model(
        _native_config(), language, composer, torch.bfloat16
    )

    assert status["status"] == "model_validated_runtime_prefix"
    assert status["runtime_dtype_validated_against_loaded_model"] is True
    assert status["native_boundary_embeddings_validated"] is True
    assert status["runtime_prefix_parity_validated"] is True

    with torch.no_grad():
        composer.scene_start.add_(1)
    with pytest.raises(ValueError, match="BOI boundary embedding"):
        _validate_runtime_prefix_against_loaded_model(
            _native_config(), language, composer, torch.bfloat16
        )

    fresh = ContinuousPrefixComposer(
        2,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=native,
    )
    with pytest.raises(ValueError, match="dtype"):
        _validate_runtime_prefix_against_loaded_model(
            _native_config(), language, fresh, torch.float16
        )


def test_audit_composer_preserves_native_checkpoint_dtype_and_legacy_learned_mode(
    tmp_path,
) -> None:
    native = (
        torch.tensor([[[0.25, -0.5]]], dtype=torch.bfloat16),
        torch.tensor([[[-0.75, 1.0]]], dtype=torch.bfloat16),
    )
    source = ContinuousPrefixComposer(
        2,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=native,
    )
    expected_hash = module_collection_state_sha256({"composer": source})
    checkpoint = save_adapter_checkpoint(
        tmp_path / "native_composer",
        {"composer": source},
        {"frozen_scene_state_sha256": expected_hash},
    )
    language = _LoadedLanguage(_native_contract(), native, torch.bfloat16)
    language.bos_token_id = 2

    restored = _construct_audit_composer(
        _native_config(), 2, torch.device("cpu"), torch.bfloat16, language
    )
    load_adapter_checkpoint(checkpoint, {"composer": restored}, device="cpu")

    assert restored.scene_start.dtype is torch.bfloat16
    assert restored.scene_end.dtype is torch.bfloat16
    assert module_collection_state_sha256({"composer": restored}) == expected_hash

    model_free = _construct_audit_composer(
        _native_config(), 2, torch.device("cpu"), torch.bfloat16
    )
    load_adapter_checkpoint(checkpoint, {"composer": model_free}, device="cpu")
    assert module_collection_state_sha256({"composer": model_free}) == expected_hash

    learned_config = {
        "language": {
            "dtype": "float16",
            "scene_prefix_after_bos": False,
            "scene_boundary_mode": "learned",
        }
    }
    learned = _construct_audit_composer(
        learned_config, 2, torch.device("cpu"), torch.float16
    )
    assert learned.scene_start.dtype is torch.float32
    assert learned.scene_end.dtype is torch.float32


def test_audit_installs_and_hash_validates_checkpoint_lora_before_forward(tmp_path) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.proj = nn.Linear(3, 5, bias=False)

    config = _native_config()
    config["language"].update(
        {
            "lora": {
                "enabled": True,
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.0,
                "target_modules": ["decoder.proj"],
            }
        }
    )
    config["training"] = {
        "lora_learning_rate": 1e-4,
        "lora_weight_decay": 0.0,
    }
    settings = lora_settings(config)
    optimizer_settings = lora_optimizer_settings(config, settings)
    source_model = TinyModel().requires_grad_(False)
    source = install_lora_adapters(source_model, settings)
    assert source is not None and optimizer_settings is not None
    with torch.no_grad():
        source.adapters[0].lora_b.fill_(0.375)
    metadata = {
        "lora": lora_checkpoint_contract(settings, optimizer_settings, source.parameter_count),
        "lora_wrapped_modules": list(source.target_names),
        "lora_trainable_parameter_counts": source.parameter_counts,
        "lora_trainable_parameter_count": source.parameter_count,
        "lora_state_sha256": source.state_sha256(),
    }
    checkpoint = save_adapter_checkpoint(
        tmp_path / "checkpoint", {"lora": source.state_module}, metadata
    )
    runtime_model = TinyModel().requires_grad_(False)
    language = SimpleNamespace(model=runtime_model)

    restored = _install_checkpoint_lora(config, language, checkpoint, metadata)

    assert restored is not None
    assert torch.equal(restored.adapters[0].lora_b, source.adapters[0].lora_b)
    assert restored.training is False
    assert all(not parameter.requires_grad for parameter in runtime_model.parameters())


def test_generation_audit_reports_exact_normalized_side_and_complete_unit_accuracy() -> None:
    first = _normalized_generation_result(
        "The RED!",
        "orange",
        "red",
        "blue",
    )
    second = _normalized_generation_result(
        "on the left",
        "To the RIGHT.",
        "on left",
        "to right",
    )

    summary = _normalized_generation_summary([first, second])

    assert {
        key: first[key]
        for key in (
            "normalized_prediction_a",
            "normalized_prediction_b",
            "normalized_expected_a",
            "normalized_expected_b",
            "normalized_exact_correct_a",
            "normalized_exact_correct_b",
            "normalized_exact_complete_unit_correct",
        )
    } == {
        "normalized_prediction_a": "red",
        "normalized_prediction_b": "orange",
        "normalized_expected_a": "red",
        "normalized_expected_b": "blue",
        "normalized_exact_correct_a": True,
        "normalized_exact_correct_b": False,
        "normalized_exact_complete_unit_correct": False,
    }
    assert first["canonical_relation_secondary_eligible_a"] is False
    assert first["canonical_relation_secondary_eligible_b"] is False
    assert second["normalized_exact_correct_a"] is True
    assert second["normalized_exact_correct_b"] is True
    assert second["normalized_exact_complete_unit_correct"] is True
    assert summary == {
        "normalized_exact_side_count": 4,
        "normalized_exact_correct_side_count": 3,
        "normalized_exact_side_accuracy": 0.75,
        "normalized_exact_complete_unit_count": 2,
        "normalized_exact_complete_unit_correct_count": 1,
        "normalized_exact_complete_unit_accuracy": 0.5,
    }


def test_canonical_relation_score_is_secondary_and_does_not_relax_exact_match() -> None:
    result = _normalized_generation_result(
        "The object is to the left of the chair.",
        "It is on the right.",
        "left",
        "right",
    )
    summary = _canonical_relation_generation_summary([result])

    assert result["normalized_exact_correct_a"] is False
    assert result["normalized_exact_correct_b"] is False
    assert result["normalized_exact_complete_unit_correct"] is False
    assert result["canonical_relation_prediction_a"] == "left"
    assert result["canonical_relation_prediction_b"] == "right"
    assert result["canonical_relation_secondary_complete_unit_correct"] is True
    assert summary == {
        "canonical_relation_secondary_only": True,
        "canonical_relation_secondary_side_count": 2,
        "canonical_relation_secondary_correct_side_count": 2,
        "canonical_relation_secondary_side_accuracy": 1.0,
        "canonical_relation_secondary_complete_unit_count": 1,
        "canonical_relation_secondary_complete_unit_correct_count": 1,
        "canonical_relation_secondary_complete_unit_accuracy": 1.0,
    }


def test_canonical_relation_score_excludes_non_relation_targets_and_ambiguous_output() -> None:
    color = _normalized_generation_result("red", "blue", "red", "blue")
    ambiguous = _normalized_generation_result("left or right", "right", "left", "right")

    color_summary = _canonical_relation_generation_summary([color])
    ambiguous_summary = _canonical_relation_generation_summary([ambiguous])

    assert color_summary["canonical_relation_secondary_side_count"] == 0
    assert color_summary["canonical_relation_secondary_side_accuracy"] is None
    assert ambiguous["canonical_relation_prediction_a"] is None
    assert ambiguous_summary["canonical_relation_secondary_side_accuracy"] == 0.5
    assert ambiguous_summary["canonical_relation_secondary_complete_unit_accuracy"] == 0.0


def test_generation_provenance_distinguishes_eos_only_fallback_from_literal_unknown() -> None:
    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [99]
            assert skip_special_tokens is True
            return ""

    answer, provenance = _generation_answer_and_provenance(
        Tokenizer(),
        torch.tensor([[99]]),
        eos_token_ids=[2, 99],
        max_new_tokens=8,
    )

    assert answer == "unknown"
    assert provenance == {
        "generated_token_ids": [99],
        "generated_token_count": 1,
        "max_new_tokens": 8,
        "eos_token_ids": [2, 99],
        "termination_reason": "eos_token",
        "terminated_by_eos": True,
        "termination_token_id": 99,
        "token_budget_exhausted": False,
        "decoded_text_skip_special_tokens": "",
        "decoded_text_after_stripping": "",
        "decoded_text_was_empty": True,
        "fallback_answer": "unknown",
        "fallback_answer_used": True,
    }


def test_generation_provenance_records_token_budget_termination() -> None:
    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [4, 5]
            assert skip_special_tokens is True
            return " left "

    answer, provenance = _generation_answer_and_provenance(
        Tokenizer(),
        torch.tensor([[4, 5]]),
        eos_token_ids=2,
        max_new_tokens=2,
    )

    assert answer == "left"
    assert provenance["generated_token_ids"] == [4, 5]
    assert provenance["termination_reason"] == "max_new_tokens"
    assert provenance["terminated_by_eos"] is False
    assert provenance["termination_token_id"] is None
    assert provenance["token_budget_exhausted"] is True
    assert provenance["fallback_answer_used"] is False


def test_generation_accuracy_is_distinct_from_prediction_change() -> None:
    result = _normalized_generation_result("orange", "purple", "red", "blue")

    assert result["normalized_prediction_a"] != result["normalized_prediction_b"]
    assert result["normalized_exact_complete_unit_correct"] is False
    assert "prediction_changed" not in result


def test_left_right_reference_orientation_macro_accuracy_balances_directions() -> None:
    examples = [
        {
            **_normalized_generation_result("left", "right", "left", "right"),
            "prediction_changed": True,
        },
        {
            **_normalized_generation_result("left", "right", "left", "right"),
            "prediction_changed": True,
        },
        {
            **_normalized_generation_result("left", "right", "right", "left"),
            "prediction_changed": True,
        },
    ]

    summary = _left_right_reference_orientation_summary(examples)

    assert summary["eligible_complete_unit_count"] == 3
    assert summary["both_reference_orientations_present"] is True
    assert summary["normalized_exact_reference_orientation_macro_accuracy"] == 0.5
    assert summary["normalized_exact_complete_unit_reference_orientation_macro_accuracy"] == 0.5
    assert (
        summary["by_reference_expected_orientation"]["left"][
            "normalized_exact_reference_side_accuracy"
        ]
        == 1.0
    )
    assert (
        summary["by_reference_expected_orientation"]["right"][
            "normalized_exact_reference_side_accuracy"
        ]
        == 0.0
    )


def test_training_pair_selection_reconstructs_seeded_complete_unit_cap(tmp_path) -> None:
    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    records = []
    for pair_id, scene_ids in (
        ("pair_a", ("scene_a", "scene_b")),
        ("pair_b", ("scene_c", "scene_d")),
    ):
        for index in range(3):
            key = f"question_{index}"
            for role, scene_id, answer in (
                ("reference", scene_ids[0], "left"),
                ("counterfactual", scene_ids[1], "right"),
            ):
                records.append(
                    {
                        "scene_id": scene_id,
                        "question_id": f"{pair_id}_{scene_id}_{index}",
                        "question": f"Question {index}?",
                        "answer": answer,
                        "answer_type": "spatial_relation",
                        "target_xyz": [0.0, 0.0, 0.0],
                        "counterfactual_pair_id": pair_id,
                        "counterfactual_question_key": key,
                        "counterfactual_expected_change": True,
                        "counterfactual_role": role,
                        "counterfactual_change_type": "mirror_lr",
                    }
                )
    (qa_root / "train.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    config = {
        "seed": 17,
        "paths": {"data_root": str(tmp_path), "qa_root": str(qa_root)},
        "training": {
            "pair_only_mode": True,
            "pair_only_scene_ids": ["scene_a", "scene_b", "scene_c", "scene_d"],
            "pair_max_units_per_pair": 2,
            "pair_ranking_weight": 1.0,
            "pair_batch_fraction": 1.0,
        },
    }

    selected = _training_selected_pair_keys(config)

    assert selected == _training_selected_pair_keys(config)
    assert len(selected) == 4
    assert {pair_id for pair_id, _ in selected} == {"pair_a", "pair_b"}
    assert all(
        sum(pair_id == expected for pair_id, _ in selected) == 2
        for expected in ("pair_a", "pair_b")
    )


def test_generation_audit_attaches_exact_scores_to_each_pair(monkeypatch, tmp_path) -> None:
    class Tokenizer:
        def __init__(self) -> None:
            self.token_ids = {"red": 0, "blue": 1}

        def __call__(self, text, **_kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[self.token_ids[text.strip()]]]))

        def decode(self, token_ids, **_kwargs):
            inverse = {value: key for key, value in self.token_ids.items()}
            return inverse[int(token_ids[0])]

    record_a = {
        "counterfactual_question_key": "color",
        "question": "What color?",
        "answer": "red",
    }
    record_b = {
        "counterfactual_question_key": "color",
        "question": "What color?",
        "answer": "blue",
    }
    monkeypatch.setattr(
        audit_module,
        "_changed_question_pairs",
        lambda _path, _pair_id: [(record_a, record_b)],
    )
    monkeypatch.setattr(
        audit_module,
        "_training_selected_pair_keys",
        lambda _config: {("pair_000001", "color")},
    )
    answers = iter(
        [
            (
                torch.tensor([4.0, 1.0]),
                "The red!",
                {"generated_token_ids": [0], "termination_reason": "eos_token"},
            ),
            (
                torch.tensor([1.0, 4.0]),
                "orange",
                {"generated_token_ids": [1], "termination_reason": "max_new_tokens"},
            ),
        ]
    )
    monkeypatch.setattr(
        audit_module,
        "_question_logits_and_answer",
        lambda *_args, **_kwargs: next(answers),
    )
    validation = {"runtime_prefix_parity_validated": True}
    monkeypatch.setattr(
        audit_module,
        "_validate_runtime_prefix_against_loaded_model",
        lambda *_args, **_kwargs: validation,
    )
    spec = {
        "pair_id": "pair_000001",
        "change_type": "color_swap",
        "split": "train",
        "scene_a": "scene_a",
        "scene_b": "scene_b",
    }
    representations = {
        "scene_a": {"final_prefix_runtime_dtype": torch.zeros(1, 2, 2)},
        "scene_b": {"final_prefix_runtime_dtype": torch.ones(1, 2, 2)},
    }
    language = SimpleNamespace(tokenizer=Tokenizer())

    generation, returned_validation = _generation_audit(
        _native_config(),
        representations,
        (spec,),
        ContinuousPrefixComposer(2),
        torch.bfloat16,
        tmp_path,
        {},
        language=language,
    )

    pair = generation["pairs"][0]
    assert returned_validation is validation
    assert pair["prediction_changed_count"] == 1
    assert pair["prediction_changed_rate"] == 1.0
    assert pair["normalized_exact_correct_side_count"] == 1
    assert pair["normalized_exact_side_accuracy"] == 0.5
    assert pair["normalized_exact_complete_unit_correct_count"] == 0
    assert pair["normalized_exact_complete_unit_accuracy"] == 0.0
    assert pair["examples"][0]["training_selected"] is True
    assert pair["training_selection_breakdown"]["selected"]["normalized_exact_side_accuracy"] == 0.5
    assert (
        pair["training_selection_breakdown"]["unselected"]["normalized_exact_side_accuracy"] is None
    )
    assert generation["training_pair_selection"]["selected_complete_unit_count"] == 1
    assert generation["scoring_policy"] == {
        "primary_promotion_metric": "normalized_exact_complete_unit_accuracy",
        "primary_side_metric": "normalized_exact_side_accuracy",
        "canonical_relation_metric_role": "secondary_diagnostic_only",
        "canonical_relation_does_not_satisfy_promotion": True,
    }
    assert pair["examples"][0]["normalized_exact_correct_a"] is True
    assert pair["examples"][0]["normalized_exact_correct_b"] is False
    assert pair["examples"][0]["generation_a"]["generated_token_ids"] == [0]
    assert pair["examples"][0]["generation_b"]["termination_reason"] == "max_new_tokens"


def test_checkpoint_epoch_fields_distinguish_selected_checkpoint_from_best_epoch() -> None:
    assert _checkpoint_epoch_fields({"epoch": 36, "best_epoch": 30}) == {
        "checkpoint_epoch": 36,
        "checkpoint_best_epoch": 30,
    }


def _summary_pair(
    *,
    blocks: float,
    native: float,
    projected: float,
    native_over_blocks: float,
    latent_cosine: float,
    runtime_changed: float = 0.8,
) -> dict:
    return {
        "raw_map": {"semantic": {"relative_l2": 0.1}},
        "block_tokens": {"common_block_tokens": {"relative_l2": blocks}},
        "native_latents": {"relative_l2": native},
        "projected_scene_tokens_float32": {"relative_l2": projected},
        "final_prefix_runtime_dtype": {"changed_element_fraction_at_1e-6": runtime_changed},
        "latent_diversity": {
            "scene_a_native": {"mean_off_diagonal_cosine": latent_cosine},
            "scene_b_native": {"mean_off_diagonal_cosine": latent_cosine},
        },
        "signal_retention": {
            "native_latents_over_blocks": native_over_blocks,
            "projected_over_native_latents": projected / native,
        },
    }


def test_summary_diagnosis_does_not_repeat_stale_collapse_claim() -> None:
    summary = _summary_findings(
        [
            _summary_pair(
                blocks=0.2,
                native=0.15,
                projected=0.12,
                native_over_blocks=0.75,
                latent_cosine=0.93,
            )
        ],
        "bfloat16",
    )

    assert summary["evidence_flags"] == {
        "raw_signal_present_for_all_pairs": True,
        "severe_native_attenuation_for_all_pairs_at_0_1": False,
        "near_duplicate_native_latents_for_all_scenes_at_0_99": False,
        "runtime_cast_erased_any_pair_at_1e_6": False,
    }
    assert (
        "do not support the historical global Perceiver-collapse diagnosis" in summary["diagnosis"]
    )
    assert "Do not infer a resampler defect" in summary["recommended_fix"]


def test_summary_reports_collapse_only_when_computed_thresholds_support_it() -> None:
    summary = _summary_findings(
        [
            _summary_pair(
                blocks=0.2,
                native=0.01,
                projected=0.005,
                native_over_blocks=0.05,
                latent_cosine=0.995,
            )
        ],
        "float16",
    )

    assert summary["evidence_flags"]["severe_native_attenuation_for_all_pairs_at_0_1"]
    assert summary["evidence_flags"]["near_duplicate_native_latents_for_all_scenes_at_0_99"]
    assert "support severe global-resampler attenuation" in summary["diagnosis"]

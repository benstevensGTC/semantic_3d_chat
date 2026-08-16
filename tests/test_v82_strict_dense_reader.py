from __future__ import annotations

from pathlib import Path

import pytest
import torch

from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    ATLAS_MEMORY_TOKENS,
    HIDDEN_SIZE,
    bind_fixed_prefix_before_question_v81,
    split_v75_v2_prefix_v81,
)
from semantic_3d_chat.language.v82_dense_learned_reader import (
    CANDIDATE_TENSOR_NAMES,
    TRAINABLE_PARAMETER_COUNT,
    DenseLearnedSceneReaderV82,
    wrong_scene_contrast_loss_v82,
)
from semantic_3d_chat.training.v82_reader_artifacts import (
    load_v82_cache,
    load_v82_candidate,
    save_v82_cache,
    save_v82_candidate,
)


def _memory(seed: int, batch_size: int = 2) -> torch.Tensor:
    return (
        torch.randn(
            batch_size,
            738,
            HIDDEN_SIZE,
            generator=torch.Generator().manual_seed(seed),
        )
        * 0.03
    )


def _zero_payload(memory: torch.Tensor) -> torch.Tensor:
    banks = split_v75_v2_prefix_v81(memory)
    atlas = torch.cat(
        (banks.probe_keys.unsqueeze(2), torch.zeros_like(banks.atlas_values)), dim=2
    ).reshape(memory.shape[0], ATLAS_MEMORY_TOKENS, HIDDEN_SIZE)
    return torch.cat(
        (banks.boi, atlas, torch.zeros_like(banks.base_latents), banks.eoi), dim=1
    )


def test_v82_parameter_and_payload_claim_is_exact() -> None:
    model = DenseLearnedSceneReaderV82()
    assert model.trainable_parameter_count == TRAINABLE_PARAMETER_COUNT == 688_130
    assert set(model.state_dict()) == CANDIDATE_TENSOR_NAMES
    audit = model.audit().as_dict()
    assert audit["every_atlas_value_participates"] is True
    assert audit["every_base_latent_participates"] is True
    assert audit["question_dependent_retrieval"] is False
    assert audit["semantic_or_spatial_top_k_selection"] is False
    assert audit["environmental_text_inputs"] == []


def test_v82_dense_positive_floors_and_fixed_binding() -> None:
    model = DenseLearnedSceneReaderV82()
    memory = _memory(1)
    query = torch.randn(2, HIDDEN_SIZE, generator=torch.Generator().manual_seed(2))
    binding = bind_fixed_prefix_before_question_v81(memory)
    output = model(memory, query, binding=binding)
    assert output.controls.shape == (2, 4, HIDDEN_SIZE)
    assert float(output.atlas_weights.detach().min()) >= 0.05 / 96 - 1e-8
    assert float(output.base_weights.detach().min()) >= 0.10 / 256 - 1e-8
    assert output.all_384_atlas_values_positive is True
    assert output.all_256_base_latents_positive is True
    assert torch.allclose(output.atlas_attention_sums, torch.ones(2), atol=1e-6)
    assert torch.allclose(output.base_attention_sums, torch.ones(2), atol=1e-6)


def test_v82_zero_environment_is_exact_before_and_after_backward() -> None:
    model = DenseLearnedSceneReaderV82()
    memory = _zero_payload(_memory(3, 1))
    query = torch.randn(1, HIDDEN_SIZE, generator=torch.Generator().manual_seed(4))
    output = model(
        memory, query, binding=bind_fixed_prefix_before_question_v81(memory)
    )
    assert output.zero_environmental_payload is True
    assert torch.count_nonzero(output.controls) == 0
    assert torch.count_nonzero(output.residual) == 0


def test_v82_binding_rejects_memory_mutation() -> None:
    model = DenseLearnedSceneReaderV82()
    memory = _memory(5, 1)
    binding = bind_fixed_prefix_before_question_v81(memory)
    changed = memory.clone()
    changed[:, 10, 10] += 1.0
    query = torch.randn(1, HIDDEN_SIZE, generator=torch.Generator().manual_seed(6))
    with pytest.raises(ValueError, match="changed after prequestion binding"):
        model(changed, query, binding=binding)


def test_v82_wrong_scene_contrast_reaches_trainable_parameters() -> None:
    model = DenseLearnedSceneReaderV82()
    memory = _memory(7)
    query = torch.randn(2, HIDDEN_SIZE, generator=torch.Generator().manual_seed(8))
    output = model(
        memory, query, binding=bind_fixed_prefix_before_question_v81(memory)
    )
    own = torch.randn(output.controls.shape, generator=torch.Generator().manual_seed(9))
    wrong = torch.randn(output.controls.shape, generator=torch.Generator().manual_seed(10))
    loss, preference = wrong_scene_contrast_loss_v82(
        output.controls, own, wrong, margin=3.0
    )
    loss.backward()
    assert preference.shape == (2,)
    assert float(loss.detach()) > 0.0
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients
    assert any(float(value.norm()) > 0.0 for value in gradients.values())
    assert all(bool(torch.isfinite(value).all()) for value in gradients.values())


def _cache_tensors() -> dict[str, torch.Tensor]:
    memories = _memory(11).to(torch.bfloat16)
    queries = torch.randn(3, HIDDEN_SIZE, generator=torch.Generator().manual_seed(12))
    targets = torch.randn(4, 4, HIDDEN_SIZE, generator=torch.Generator().manual_seed(13))
    return {
        "scene_memories": memories,
        "question_queries": queries,
        "row_scene_indices": torch.tensor([0, 1, 0, 1]),
        "row_paired_scene_indices": torch.tensor([1, 0, 1, 0]),
        "row_query_indices": torch.tensor([0, 1, 2, 0]),
        "row_expected_change": torch.tensor([True, True, False, False]),
        "target_controls": targets.to(torch.bfloat16),
        "paired_target_controls": targets.roll(1, dims=0).to(torch.bfloat16),
    }


def test_v82_cache_round_trip_is_numeric_and_create_once(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "cache"
    metadata = save_v82_cache(
        root,
        _cache_tensors(),
        split_role="historical_optimization_fold",
        scene_ids=["scene_000001", "scene_000002"],
        source_qa_sha256="0" * 64,
        source_v73_config_sha256="1" * 64,
        source_prefix_manifest_sha256="2" * 64,
        source_controller_sha256="3" * 64,
        source_probe_tensor_sha256="4" * 64,
    )
    loaded = load_v82_cache(root)
    assert loaded.metadata == metadata
    assert loaded.metadata["questions_or_answers_serialized"] is False
    assert loaded.metadata["environmental_text_serialized"] is False
    with pytest.raises(FileExistsError):
        save_v82_cache(
            root,
            _cache_tensors(),
            split_role="historical_optimization_fold",
            scene_ids=["scene_000001", "scene_000002"],
            source_qa_sha256="0" * 64,
            source_v73_config_sha256="1" * 64,
            source_prefix_manifest_sha256="2" * 64,
            source_controller_sha256="3" * 64,
            source_probe_tensor_sha256="4" * 64,
        )


def test_v82_candidate_round_trip_has_narrow_payload_claim(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "candidate"
    model = DenseLearnedSceneReaderV82()
    metadata = save_v82_candidate(
        root,
        model,
        training_cache_sha256="5" * 64,
        training_cache_metadata_sha256="6" * 64,
        fit_summary={
            "optimizer_updates": 1,
            "zero_environment_maximum_absolute_control": 0.0,
            "training_fold_only": True,
        },
    )
    loaded = load_v82_candidate(root)
    assert loaded.metadata == metadata
    assert metadata["all_384_atlas_values_and_256_base_latents_positive_floor"] is True
    assert metadata["boi_eoi_and_96_probe_keys_are_not_payload"] is True
    assert "all_environmental_tokens_positive_floor" not in metadata
    memory = _zero_payload(_memory(14, 1))
    query = torch.randn(1, HIDDEN_SIZE, generator=torch.Generator().manual_seed(15))
    output = loaded.model(
        memory, query, binding=bind_fixed_prefix_before_question_v81(memory)
    )
    assert torch.count_nonzero(output.controls) == 0

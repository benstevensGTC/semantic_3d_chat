from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from semantic_3d_chat.evaluation.fixed_prefix_atlas_v2_exposure import (
    GEMMA4_E2B_MODEL_ID,
    GEMMA4_E2B_SLIDING_WINDOW,
    final_prompt_sliding_exposure,
    gemma4_e2b_prompt_exposure_table,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import compile_fixed_scene_atlas
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v2 import (
    compile_fixed_scene_atlas_v2,
    reorder_compiled_scene_atlas_v2,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)


def _controller() -> AlwaysOnTeacherBasisFullSceneQuestionControlV7:
    generator = torch.Generator().manual_seed(2701)
    basis, _ = torch.linalg.qr(torch.randn(8, 4, generator=generator))
    return AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        8,
        basis.T.contiguous(),
        control_tokens=2,
        expected_environment_latents=4,
        moment_count=2,
        interaction_dim=3,
        trunk_dim=5,
        maximum_control_rms=0.2,
        initial_control_rms=0.05,
    ).eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2702)
    return (
        torch.randn(1, 6, 8, generator=generator),
        torch.randn(3, 8, generator=generator),
    )


def test_v2_compiler_has_no_question_argument_and_is_deterministic() -> None:
    assert "question" not in inspect.signature(compile_fixed_scene_atlas_v2).parameters
    prefix, probes = _inputs()
    controller = _controller()
    first = compile_fixed_scene_atlas_v2(prefix, controller, probes)
    second = compile_fixed_scene_atlas_v2(prefix, controller, probes)
    assert torch.equal(first.scene_prefix, second.scene_prefix)
    assert first.audit == second.audit
    assert first.audit.compiled_before_user_question is True
    assert first.audit.user_question_inputs_used_for_compilation is False
    assert first.audit.question_dependent_scene_processing is False
    assert first.audit.question_dependent_retrieval is False
    assert first.audit.semantic_or_spatial_top_k_selection is False
    assert first.audit.environmental_text_inputs == ()


def test_v2_reorders_every_v1_tensor_without_numeric_change() -> None:
    prefix, probes = _inputs()
    controller = _controller()
    v1 = compile_fixed_scene_atlas(prefix, controller, probes)
    v2 = compile_fixed_scene_atlas_v2(prefix, controller, probes)

    # Toy V1: BOI, four base latents, nine atlas tokens, EOI.
    v1_base = v1.scene_prefix[:, 1:5]
    v1_atlas = v1.scene_prefix[:, 5:-1]
    assert v1.scene_prefix.shape == v2.scene_prefix.shape == (1, 15, 8)
    assert torch.equal(v2.scene_prefix[:, :1], v1.scene_prefix[:, :1])
    assert torch.equal(v2.scene_prefix[:, 1:10], v1_atlas)
    assert torch.equal(v2.scene_prefix[:, 10:14], v1_base)
    assert torch.equal(v2.scene_prefix[:, -1:], v1.scene_prefix[:, -1:])

    assert torch.equal(v2.scene_signature, v1.scene_signature)
    assert torch.equal(v2.atlas_keys, v1.atlas_keys)
    assert torch.equal(v2.atlas_values, v1.atlas_values)
    assert v2.audit.atlas_key_sha256 == v1.audit.atlas_key_sha256
    assert v2.audit.atlas_value_sha256 == v1.audit.atlas_value_sha256
    assert v2.audit.base_environment_tokens_preserved_exactly is True
    assert v2.audit.atlas_key_value_tokens_preserved_exactly is True
    assert v2.audit.boundary_tokens_preserved_exactly is True
    assert v2.audit.complete_atlas_included is True
    assert v2.audit.layout == (
        "boi",
        "all_atlas_key_value_tokens",
        "all_base_scene_latents",
        "eoi",
    )


def test_public_reorder_helper_accepts_compiled_numeric_atlas_only() -> None:
    prefix, probes = _inputs()
    controller = _controller()
    source = compile_fixed_scene_atlas(prefix, controller, probes)
    reordered = reorder_compiled_scene_atlas_v2(source)
    wrapped = compile_fixed_scene_atlas_v2(prefix, controller, probes)

    assert "question" not in inspect.signature(reorder_compiled_scene_atlas_v2).parameters
    assert "controller" not in inspect.signature(reorder_compiled_scene_atlas_v2).parameters
    assert torch.equal(reordered.scene_prefix, wrapped.scene_prefix)
    assert reordered.audit == wrapped.audit
    with pytest.raises(TypeError, match="FixedPrefixAtlasOutput"):
        reorder_compiled_scene_atlas_v2(object())  # type: ignore[arg-type]


def test_public_reorder_helper_rejects_tampered_compiled_tensors() -> None:
    prefix, probes = _inputs()
    source = compile_fixed_scene_atlas(prefix, _controller(), probes)
    source.scene_prefix[:, 2] += 1
    with pytest.raises(ValueError, match="audited hash"):
        reorder_compiled_scene_atlas_v2(source)


@pytest.mark.parametrize("prompt_tokens", range(57, 65))
def test_gemma4_e2b_final_prompt_sliding_exposure_is_0_in_v1_256_in_v2(
    prompt_tokens: int,
) -> None:
    v1 = final_prompt_sliding_exposure(
        "v1", prompt_token_count_including_bos=prompt_tokens
    )
    v2 = final_prompt_sliding_exposure(
        "v2", prompt_token_count_including_bos=prompt_tokens
    )
    expected_query = 737 + prompt_tokens
    expected_first_visible = expected_query - GEMMA4_E2B_SLIDING_WINDOW + 1

    assert v1.model_id == v2.model_id == GEMMA4_E2B_MODEL_ID
    assert v1.sliding_window == v2.sliding_window == 512
    assert v1.final_prompt_query_position == v2.final_prompt_query_position == expected_query
    assert v1.first_visible_key_position == expected_first_visible
    assert v2.first_visible_key_position == expected_first_visible
    assert (v1.base_first_position, v1.base_last_position) == (2, 257)
    assert (v2.base_first_position, v2.base_last_position) == (482, 737)
    assert v1.visible_base_latent_count == 0
    assert v2.visible_base_latent_count == v2.total_base_latent_count == 256
    assert v1.attention_layer_kind == v2.attention_layer_kind == "sliding_attention"
    assert v1.mask_predicate == "key_position > query_position - sliding_window"
    # EOI remains directly visible; BOI is outside the local window in both layouts.
    assert v1.boi_visible is v2.boi_visible is False
    assert v1.eoi_visible is v2.eoi_visible is True


def test_exposure_table_quantifies_every_inclusive_prompt_length() -> None:
    table = gemma4_e2b_prompt_exposure_table()
    assert len(table) == 8
    assert [pair[0].prompt_token_count_including_bos for pair in table] == list(range(57, 65))
    assert [pair[0].final_prompt_query_position for pair in table] == list(range(794, 802))
    assert [pair[0].first_visible_key_position for pair in table] == list(range(283, 291))
    assert {pair[0].visible_base_latent_count for pair in table} == {0}
    assert {pair[1].visible_base_latent_count for pair in table} == {256}
    # The relocation is lossless, not retrieval: all 480 atlas tokens remain in V2.
    assert {pair[1].total_atlas_token_count for pair in table} == {480}
    assert [pair[1].visible_atlas_token_count for pair in table] == list(range(199, 191, -1))


def test_exposure_uses_transformers_strict_lower_window_boundary() -> None:
    exposure = final_prompt_sliding_exposure(
        "v2", prompt_token_count_including_bos=57
    )
    query = exposure.final_prompt_query_position
    window = exposure.sliding_window
    assert exposure.first_visible_key_position == query - window + 1
    assert exposure.first_visible_key_position > query - window
    assert not (exposure.first_visible_key_position - 1 > query - window)


def test_v2_config_is_disabled_and_makes_no_behavioral_claim() -> None:
    config = yaml.safe_load(
        Path("configs/experiments/gemma4_strict_fixed_prefix_atlas_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    atlas = config["strict_fixed_prefix_atlas"]
    exposure = config["sliding_attention_exposure"]
    evaluation = config["evaluation"]
    assert atlas["architecture"] == "fixed_scene_key_value_atlas_v2"
    assert atlas["layout"] == [
        "boi",
        "all_atlas_key_value_tokens",
        "all_base_scene_latents",
        "eoi",
    ]
    assert atlas["compiled_fixed_prefix_tokens"] == 738
    assert atlas["compilation_enabled"] is False
    assert atlas["source_sealed_controller"] is None
    assert exposure["sliding_window_tokens"] == 512
    assert exposure["prompt_token_range_including_bos"] == [57, 64]
    assert exposure["final_prompt_token_v1_visible_base_latents"] == 0
    assert exposure["final_prompt_token_v2_visible_base_latents"] == 256
    assert evaluation["behavioral_accuracy_measured"] is False
    assert evaluation["behavioral_improvement_claimed"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layout_version": "v3", "prompt_token_count_including_bos": 57}, "Unknown"),
        ({"layout_version": "v2", "prompt_token_count_including_bos": 0}, "Prompt"),
        (
            {
                "layout_version": "v2",
                "prompt_token_count_including_bos": 57,
                "sliding_window": 0,
            },
            "Sliding",
        ),
    ],
)
def test_exposure_rejects_invalid_contracts(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        final_prompt_sliding_exposure(**kwargs)  # type: ignore[arg-type]

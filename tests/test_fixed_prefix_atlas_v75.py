from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75,
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)


def _controller() -> DenseFullSceneContinuousControlV75:
    torch.manual_seed(8075)
    return DenseFullSceneContinuousControlV75(
        16,
        torch.eye(8, 16),
        environment_latents=6,
        query_count=3,
        model_dimension=8,
        coefficient_decoder_hidden_dimension=12,
        uniform_floor_mass=0.1,
        maximum_control_rms=0.25,
    ).eval()


def test_v75_atlas_compiles_every_probe_and_preserves_every_base_tensor() -> None:
    torch.manual_seed(8076)
    controller = _controller()
    base = torch.randn(1, 8, 16, dtype=torch.float32)
    probes = torch.randn(5, 16)

    compiled = compile_fixed_scene_atlas_v75(base, controller, probes)

    assert compiled.scene_prefix.shape == (1, 28, 16)
    assert torch.equal(compiled.scene_prefix[:, :7], base[:, :-1])
    assert torch.equal(compiled.scene_prefix[:, -1:], base[:, -1:])
    assert compiled.atlas_keys.shape == (5, 16)
    assert compiled.atlas_values.shape == (5, 3, 16)
    assert compiled.scene_signature.shape == (1, 6, 16)
    assert compiled.audit.environment_latent_count == 6
    assert compiled.audit.probe_count == 5
    assert compiled.audit.values_per_probe == 3
    assert compiled.audit.atlas_memory_token_count == 20
    assert compiled.audit.every_environment_latent_influenced_signature is True
    assert compiled.audit.every_probe_processed is True
    assert compiled.audit.complete_atlas_appended is True
    assert compiled.audit.compiled_before_user_question is True
    assert compiled.audit.user_question_inputs_used_for_compilation is False
    assert compiled.audit.question_dependent_scene_processing is False
    assert compiled.audit.question_dependent_retrieval is False
    assert compiled.audit.semantic_or_spatial_top_k_selection is False
    assert compiled.audit.environmental_text_inputs == ()
    assert prefix_sha256(compiled.scene_prefix) == (
        compiled.audit.fixed_scene_prefix_sha256
    )
    assert tensor_sha256(compiled.atlas_keys) == compiled.audit.atlas_key_sha256
    assert tensor_sha256(compiled.atlas_values) == compiled.audit.atlas_value_sha256


def test_v75_atlas_zero_scene_has_no_question_only_value_path() -> None:
    controller = _controller()
    base = torch.randn(1, 8, 16)
    base[:, 1:-1] = 0
    probes = torch.randn(4, 16)

    compiled = compile_fixed_scene_atlas_v75(base, controller, probes)

    assert torch.count_nonzero(compiled.atlas_values) == 0
    assert torch.equal(compiled.scene_prefix[:, :7], base[:, :-1])
    assert torch.equal(compiled.scene_prefix[:, -1:], base[:, -1:])


def test_v75_v2_reordering_is_lossless_and_still_question_independent() -> None:
    torch.manual_seed(8077)
    controller = _controller()
    base = torch.randn(1, 8, 16)
    probes = torch.randn(5, 16)
    v1 = compile_fixed_scene_atlas_v75(base, controller, probes)
    v2 = compile_fixed_scene_atlas_v75_v2(base, controller, probes)

    assert v2.scene_prefix.shape == v1.scene_prefix.shape
    assert torch.equal(v2.scene_prefix[:, :1], base[:, :1])
    assert torch.equal(v2.scene_prefix[:, 1:21], v1.scene_prefix[:, 7:-1])
    assert torch.equal(v2.scene_prefix[:, 21:-1], base[:, 1:-1])
    assert torch.equal(v2.scene_prefix[:, -1:], base[:, -1:])
    assert v2.audit.base_environment_tokens_preserved_exactly is True
    assert v2.audit.atlas_key_value_tokens_preserved_exactly is True
    assert v2.audit.user_question_inputs_used_for_compilation is False
    assert v2.audit.question_dependent_scene_processing is False
    assert v2.audit.question_dependent_retrieval is False


def test_v75_atlas_rejects_wrong_or_invalid_inputs() -> None:
    controller = _controller()
    probes = torch.randn(3, 16)
    with pytest.raises(ValueError, match="BOI"):
        compile_fixed_scene_atlas_v75(torch.randn(1, 7, 16), controller, probes)
    invalid = torch.randn(1, 8, 16)
    invalid[0, 2, 4] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compile_fixed_scene_atlas_v75(invalid, controller, probes)
    with pytest.raises(ValueError, match="hidden size"):
        compile_fixed_scene_atlas_v75(
            torch.randn(1, 8, 16), controller, torch.randn(3, 15)
        )
    with pytest.raises(TypeError, match="exact V75"):
        compile_fixed_scene_atlas_v75(  # type: ignore[arg-type]
            torch.randn(1, 8, 16), torch.nn.Identity(), probes
        )

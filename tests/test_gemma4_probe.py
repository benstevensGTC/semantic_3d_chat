from __future__ import annotations

import importlib.metadata

import pytest
import torch

from semantic_3d_chat.vision.gemma4_probe import (
    GEMMA4_CONFIG_SHA256,
    GEMMA4_REQUIREMENTS_TRANSFORMERS,
    GEMMA4_REVISION,
    derive_vision_grid_mapping,
    model_shape_and_memory_report,
    patchify_complete_image,
    resolve_config_path,
    run_processor_grid_probe,
    run_tiny_decoder_probe,
    run_tiny_vision_probe,
)


def _has_probe_transformers() -> bool:
    try:
        return importlib.metadata.version("transformers") == GEMMA4_REQUIREMENTS_TRANSFORMERS
    except importlib.metadata.PackageNotFoundError:
        return False


def test_patchify_complete_image_preserves_row_major_locations() -> None:
    image = torch.arange(1 * 3 * 4 * 6, dtype=torch.float32).reshape(1, 3, 4, 6)
    image /= image.max()
    patches, positions = patchify_complete_image(image, patch_size=2)

    assert patches.shape == (1, 6, 12)
    assert positions.tolist() == [
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    ]
    # Each flattened patch keeps RGB as the innermost axis, matching the
    # official Gemma4 image processor.
    assert torch.equal(patches[0, 0].reshape(2, 2, 3), image[0, :, :2, :2].permute(1, 2, 0))


def test_grid_mapping_reproduces_three_by_three_pooling_with_padding() -> None:
    y, x = torch.meshgrid(torch.arange(6), torch.arange(9), indexing="ij")
    valid = torch.stack((x, y), dim=-1).reshape(-1, 2)
    padded = torch.cat((valid, torch.full((9, 2), -1)), dim=0).unsqueeze(0)
    mapping = derive_vision_grid_mapping(padded, pooling_kernel_size=3)

    assert (mapping.pre_grid_height, mapping.pre_grid_width) == (6, 9)
    assert (mapping.post_grid_height, mapping.post_grid_width) == (2, 3)
    assert mapping.pre_token_count == 54
    assert mapping.post_token_count == 6
    assert mapping.pre_to_post_token[:9] == (0, 0, 0, 1, 1, 1, 2, 2, 2)
    assert mapping.pre_to_post_token[27:36] == (3, 3, 3, 4, 4, 4, 5, 5, 5)
    assert mapping.post_token_pixel_bounds(4, patch_size=16) == (48, 48, 96, 96)


def test_grid_mapping_rejects_missing_patch() -> None:
    positions = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="rectangular"):
        derive_vision_grid_mapping(positions, pooling_kernel_size=1)


@pytest.mark.skipif(
    not _has_probe_transformers(),
    reason="Gemma 4 model probes run in .venv-gemma4 with Transformers 5.14.1",
)
def test_one_full_image_call_exposes_pre_and_post_pool_spatial_states() -> None:
    result = run_tiny_vision_probe()

    assert result["passed"] is True
    assert result["vision_forward_calls"] == 1
    assert result["complete_image_shape"] == [1, 3, 96, 96]
    assert result["encoder_input_shape"] == [1, 36, 768]
    assert result["pre_pool_layer_state_shape"] == [1, 36, 16]
    assert result["post_pool_state_shape"] == [4, 16]
    assert result["pre_pool_spatial_variance"] > 0
    assert result["post_pool_spatial_variance"] > 0


@pytest.mark.skipif(
    not _has_probe_transformers(),
    reason="Gemma 4 model probes run in .venv-gemma4 with Transformers 5.14.1",
)
def test_arbitrary_prefix_uses_explicit_ple_and_extends_cache() -> None:
    result = run_tiny_decoder_probe()

    assert result["passed"] is True
    assert result["arbitrary_scene_prefix_shape"] == [1, 5, 32]
    assert result["explicit_per_layer_inputs_shape"] == [1, 8, 3, 8]
    assert result["scene_ple_identity"] == "pad_token_native_multimodal_convention"
    assert result["missing_ple_rejected"] is True
    assert result["cache_length_before_increment"] == 8
    assert result["cache_length_after_increment"] == 9
    assert result["first_logits_shape"] == [1, 1, 64]


@pytest.mark.skipif(
    not _has_probe_transformers(),
    reason="Gemma 4 config probe runs in .venv-gemma4 with Transformers 5.14.1",
)
def test_pinned_e2b_config_shapes_and_memory_are_reproducible() -> None:
    from transformers import AutoConfig

    config_path, metadata = resolve_config_path(allow_network=False)
    config = AutoConfig.from_pretrained(config_path.parent, local_files_only=True)
    memory = model_shape_and_memory_report(config, metadata["checkpoint_size_bytes"])
    grid = run_processor_grid_probe(config)

    assert metadata["resolved_revision"] == GEMMA4_REVISION
    assert metadata["config_sha256"] == GEMMA4_CONFIG_SHA256
    assert metadata["weights_downloaded_by_probe"] is False
    assert memory["total_parameters"] == 5_104_297_504
    assert memory["text"]["hidden_size"] == 1536
    assert memory["vision"]["hidden_size"] == 768
    assert 9.50 < memory["weights_only"]["bf16_or_fp16_gib"] < 9.51
    assert grid["pre_pool_grid_hw"] == [48, 48]
    assert grid["post_pool_grid_hw"] == [16, 16]
    assert grid["valid_pre_pool_tokens"] == 2304
    assert grid["valid_post_pool_tokens"] == 256

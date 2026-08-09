from pathlib import Path

import numpy as np
import torch

from semantic_3d_chat.vision.patch_features import (
    ALIGNED_FEATURE_KEY,
    COMPONENT_NAMES,
    LATE_FEATURE_KEY,
    MIDDLE_FEATURE_KEY,
    NATIVE_FEATURE_KEY,
    SPATIAL_FEATURE_KEY,
    DensePatchFeatures,
    clip_patch_grid,
)


def test_clip_patch_grid_explicitly_removes_cls_and_preserves_row_major_order() -> None:
    sequence = torch.arange(197, dtype=torch.float32).reshape(1, 197, 1)
    sequence[:, 0] = -1000  # A CLS sentinel that must not enter the patch field.

    grid = clip_patch_grid(sequence, (14, 14))

    assert grid.shape == (1, 14, 14, 1)
    assert grid[0, 0, 0, 0].item() == 1
    assert grid[0, 0, 13, 0].item() == 14
    assert grid[0, 1, 0, 0].item() == 15
    assert grid[0, 13, 13, 0].item() == 196
    assert not torch.any(grid == -1000)


def test_clip_patch_grid_rejects_non_197_token_sequence() -> None:
    with torch.no_grad():
        sequence = torch.zeros(1, 196, 8)
    try:
        clip_patch_grid(sequence, (14, 14))
    except ValueError as exc:
        assert "one CLS plus 14x14" in str(exc)
    else:
        raise AssertionError("A missing CLS token must be rejected")


def _features() -> DensePatchFeatures:
    generator = torch.Generator().manual_seed(7)
    return DensePatchFeatures(
        native_middle=torch.randn(14, 14, 768, generator=generator, dtype=torch.float16),
        native_late=torch.randn(14, 14, 768, generator=generator, dtype=torch.float16),
        clip_aligned=torch.randn(14, 14, 512, generator=generator, dtype=torch.float16),
    )


def test_dense_patch_artifact_retains_named_components_and_fusion_layout(tmp_path: Path) -> None:
    features = _features()
    output = tmp_path / "frame_000000.npz"

    features.save(output, {"cache_signature": "abc123", "frame_key": "frame_000000"})
    loaded, metadata = DensePatchFeatures.load(output)

    assert loaded.native_middle.shape == (14, 14, 768)
    assert loaded.native_late.shape == (14, 14, 768)
    assert loaded.native_middle_late.shape == (14, 14, 1536)
    assert loaded.clip_aligned.shape == (14, 14, 512)
    assert loaded.spatial_features.shape == (14, 14, 2048)
    assert loaded.component_offsets == (0, 768, 1536, 2048)
    assert loaded.native_middle.dtype == torch.float16
    assert torch.equal(loaded.spatial_features[..., :768], loaded.native_middle)
    assert torch.equal(loaded.spatial_features[..., 768:1536], loaded.native_late)
    assert torch.equal(loaded.spatial_features[..., 1536:], loaded.clip_aligned)
    assert metadata == {"cache_signature": "abc123", "frame_key": "frame_000000"}

    with np.load(output, allow_pickle=False) as archive:
        assert {
            SPATIAL_FEATURE_KEY,
            NATIVE_FEATURE_KEY,
            MIDDLE_FEATURE_KEY,
            LATE_FEATURE_KEY,
            ALIGNED_FEATURE_KEY,
            "component_names",
            "component_offsets",
            "metadata_json",
        } <= set(archive.files)
        assert archive[SPATIAL_FEATURE_KEY].shape == (14, 14, 2048)
        assert tuple(archive["component_names"].tolist()) == COMPONENT_NAMES
        assert tuple(archive["component_offsets"].tolist()) == (0, 768, 1536, 2048)

"""Explicit patch-token reshaping and lossless dense-feature artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from semantic_3d_chat.vision.model_registry import DenseVisionModelSpec

SPATIAL_FEATURE_KEY = "spatial_features"
NATIVE_FEATURE_KEY = "native_middle_late"
ALIGNED_FEATURE_KEY = "clip_aligned"
MIDDLE_FEATURE_KEY = "native_middle"
LATE_FEATURE_KEY = "native_late"
COMPONENT_NAMES = (MIDDLE_FEATURE_KEY, LATE_FEATURE_KEY, ALIGNED_FEATURE_KEY)


def clip_patch_grid(sequence: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Remove CLIP's CLS token and reshape the remaining tokens spatially.

    Args:
        sequence: Tensor shaped ``[batch, 1 + grid_h * grid_w, feature_dim]``.
        grid_size: Patch-grid height and width.

    Returns:
        Tensor shaped ``[batch, grid_h, grid_w, feature_dim]``.

    This function is intentionally small and explicit: token zero is the CLS
    token, while token ``1 + row * grid_w + column`` maps to ``[row, column]``.
    """

    if sequence.ndim != 3:
        raise ValueError(f"Expected [batch, sequence, dim], got {tuple(sequence.shape)}")
    grid_h, grid_w = grid_size
    expected_sequence = 1 + grid_h * grid_w
    if sequence.shape[1] != expected_sequence:
        raise ValueError(
            f"Expected one CLS plus {grid_h}x{grid_w} patches "
            f"({expected_sequence} tokens), got {sequence.shape[1]}"
        )
    patch_tokens = sequence[:, 1:, :]  # Explicitly discard CLIP's CLS token.
    return patch_tokens.reshape(sequence.shape[0], grid_h, grid_w, sequence.shape[-1])


@dataclass(frozen=True)
class DensePatchFeatures:
    """Native middle/late and language-aligned grids for one complete image.

    The ``clip_aligned`` field name is retained in schema version 1 so existing
    CLIP maps remain byte-compatible. Gemma 4 stores its native multimodal
    projector stream in the same generic aligned slot.
    """

    native_middle: torch.Tensor
    native_late: torch.Tensor
    clip_aligned: torch.Tensor

    def __post_init__(self) -> None:
        middle = self.native_middle
        late = self.native_late
        aligned = self.clip_aligned
        streams = (middle, late, aligned)
        if any(stream.ndim != 3 for stream in streams):
            raise ValueError("Patch artifacts must be [grid_h, grid_w, feature_dim]")
        if middle.shape != late.shape:
            raise ValueError(
                f"Middle/late native shape mismatch: {middle.shape} vs {late.shape}"
            )
        if middle.shape[:2] != aligned.shape[:2]:
            raise ValueError(
                f"Feature-grid mismatch: native {middle.shape[:2]} vs aligned {aligned.shape[:2]}"
            )
        if any(not stream.is_floating_point() for stream in streams):
            raise TypeError("Patch features must use floating-point tensors")
        if any(not torch.isfinite(stream).all() for stream in streams):
            raise ValueError("Patch features contain NaN or infinite values")
        if any(torch.linalg.vector_norm(stream.float(), dim=-1).eq(0).any() for stream in streams):
            raise ValueError("A feature stream contains a zero-norm patch")

    @property
    def grid_size(self) -> tuple[int, int]:
        return int(self.native_middle.shape[0]), int(self.native_middle.shape[1])

    @property
    def native_middle_late(self) -> torch.Tensor:
        """The retained native 1,536D stream, ordered middle then late."""

        return torch.cat((self.native_middle, self.native_late), dim=-1)

    @property
    def spatial_features(self) -> torch.Tensor:
        """The fusion field: ``[native middle, native late, aligned]``."""

        return torch.cat((self.native_middle, self.native_late, self.clip_aligned), dim=-1)

    @property
    def aligned(self) -> torch.Tensor:
        """Architecture-neutral alias for the schema's aligned feature slot."""

        return self.clip_aligned

    @property
    def component_offsets(self) -> tuple[int, int, int, int]:
        middle_end = int(self.native_middle.shape[-1])
        late_end = middle_end + int(self.native_late.shape[-1])
        aligned_end = late_end + int(self.clip_aligned.shape[-1])
        return 0, middle_end, late_end, aligned_end

    def save(self, path: str | Path, metadata: Mapping[str, str] | None = None) -> Path:
        """Atomically save fusion and named component arrays without lossy coding."""

        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError(f"Dense patch cache must use .npz, got {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}.tmp.npz")
        arrays = {
            SPATIAL_FEATURE_KEY: self.spatial_features.detach().cpu().numpy(),
            NATIVE_FEATURE_KEY: self.native_middle_late.detach().cpu().numpy(),
            MIDDLE_FEATURE_KEY: self.native_middle.detach().cpu().numpy(),
            LATE_FEATURE_KEY: self.native_late.detach().cpu().numpy(),
            ALIGNED_FEATURE_KEY: self.clip_aligned.detach().cpu().numpy(),
            "component_names": np.asarray(COMPONENT_NAMES),
            "component_offsets": np.asarray(self.component_offsets, dtype=np.int32),
            "metadata_json": np.asarray(
                json.dumps(
                    {str(key): str(value) for key, value in (metadata or {}).items()},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        }
        try:
            # np.savez uses ZIP storage but no lossy semantic compression.
            with temporary.open("wb") as handle:
                np.savez(handle, **arrays)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> tuple[DensePatchFeatures, dict[str, str]]:
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            required = {
                SPATIAL_FEATURE_KEY,
                NATIVE_FEATURE_KEY,
                MIDDLE_FEATURE_KEY,
                LATE_FEATURE_KEY,
                ALIGNED_FEATURE_KEY,
                "component_names",
                "component_offsets",
                "metadata_json",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Missing feature arrays in {source}: {sorted(missing)}")
            features = cls(
                native_middle=torch.from_numpy(archive[MIDDLE_FEATURE_KEY].copy()),
                native_late=torch.from_numpy(archive[LATE_FEATURE_KEY].copy()),
                clip_aligned=torch.from_numpy(archive[ALIGNED_FEATURE_KEY].copy()),
            )
            stored_spatial = torch.from_numpy(archive[SPATIAL_FEATURE_KEY].copy())
            stored_native = torch.from_numpy(archive[NATIVE_FEATURE_KEY].copy())
            names = tuple(str(value) for value in archive["component_names"].tolist())
            offsets = tuple(int(value) for value in archive["component_offsets"].tolist())
            metadata_value = archive["metadata_json"].item()
        if names != COMPONENT_NAMES:
            raise ValueError(f"Unexpected feature component order in {source}: {names}")
        if offsets != features.component_offsets:
            raise ValueError(f"Unexpected feature offsets in {source}: {offsets}")
        if not torch.equal(stored_native, features.native_middle_late):
            raise ValueError(f"Stored native feature field is inconsistent in {source}")
        if not torch.equal(stored_spatial, features.spatial_features):
            raise ValueError(f"Stored spatial feature field is inconsistent in {source}")
        metadata = json.loads(str(metadata_value))
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise ValueError(f"Invalid feature metadata in {source}")
        return features, metadata


def extract_clip_streams(
    vision_outputs: Any,
    model: Any,
    spec: DenseVisionModelSpec,
    middle_layer: int,
    late_layer: int,
    storage_dtype: torch.dtype = torch.float16,
    aligned_method: str = "tokenwise_projection",
) -> DensePatchFeatures:
    """Build native and projected patch streams from one CLIP vision forward."""

    spec.validate_layers(middle_layer, late_layer)
    hidden_states = vision_outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Vision model did not return hidden states")
    required_states = spec.num_hidden_layers + 1
    if len(hidden_states) < required_states:
        raise ValueError(f"Expected at least {required_states} hidden states, got {len(hidden_states)}")

    middle_sequence = hidden_states[middle_layer]
    late_sequence = hidden_states[late_layer]
    if middle_sequence.shape[-1] != spec.native_dim or late_sequence.shape[-1] != spec.native_dim:
        raise ValueError(
            f"Expected native dimension {spec.native_dim}, got "
            f"{middle_sequence.shape[-1]} and {late_sequence.shape[-1]}"
        )

    middle_grid = clip_patch_grid(middle_sequence, spec.grid_size)
    late_grid = clip_patch_grid(late_sequence, spec.grid_size)
    if aligned_method == "tokenwise_projection":
        aligned_sequence = late_sequence[:, 1:, :]
    elif aligned_method == "maskclip_value":
        # The final-block input has already jointly processed the complete image
        # through eleven transformer blocks. Its learned value/output path keeps
        # stronger spatial localization than another globally mixing attention.
        final_layer = model.vision_model.encoder.layers[-1]
        local_sequence = final_layer.layer_norm1(hidden_states[-2])
        value_sequence = final_layer.self_attn.v_proj(local_sequence)
        aligned_sequence = final_layer.self_attn.out_proj(value_sequence)[:, 1:, :]
    else:
        raise ValueError(
            f"Unknown aligned_method {aligned_method!r}; use tokenwise_projection or maskclip_value"
        )
    post_layernorm = model.vision_model.post_layernorm(aligned_sequence)
    projected = model.visual_projection(post_layernorm)
    if projected.shape[-1] != spec.aligned_dim:
        raise ValueError(
            f"Expected projected dimension {spec.aligned_dim}, got {projected.shape[-1]}"
        )
    aligned_grid = projected.reshape(
        projected.shape[0], spec.grid_size[0], spec.grid_size[1], spec.aligned_dim
    )

    if middle_grid.shape[0] != 1:
        raise ValueError("Per-frame extraction requires exactly one complete image")
    return DensePatchFeatures(
        native_middle=middle_grid[0].detach().to(device="cpu", dtype=storage_dtype),
        native_late=late_grid[0].detach().to(device="cpu", dtype=storage_dtype),
        clip_aligned=aligned_grid[0].detach().to(device="cpu", dtype=storage_dtype),
    )

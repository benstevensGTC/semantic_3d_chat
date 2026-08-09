"""Dense full-image Gemma 4 feature extraction for persistent 3D mapping.

One call to the vision tower processes the complete frame. The extractor keeps
context-aware hidden states before Gemma 4's 3x3 spatial pooler, preserving the
48x48 native grid for a square input under the pinned E2B processor. The native
vision-to-language projector is applied only to its trained post-pool 16x16
tokens; each result is then broadcast back to its owning 3x3 pre-pool cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from semantic_3d_chat.device import safe_dtype, select_device
from semantic_3d_chat.vision.gemma4_probe import derive_vision_grid_mapping
from semantic_3d_chat.vision.model_registry import DenseVisionModelSpec, get_model_spec
from semantic_3d_chat.vision.patch_features import DensePatchFeatures


class Gemma4VisionBundle(torch.nn.Module):
    """Only the two checkpoint modules required for dense feature extraction."""

    def __init__(self, vision_tower: torch.nn.Module, embed_vision: torch.nn.Module) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision


def load_selective_gemma4_vision_bundle(
    config: Any,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Gemma4VisionBundle:
    """Strictly load only ``vision_tower`` and ``embed_vision`` safetensors.

    The E2B checkpoint is a single ~10.25 GB file, but the selected tensors are
    only 168.5M parameters. Strict ``assign=True`` loading avoids ever
    materializing the 4.6B-parameter language/audio modules. The small modules
    are constructed normally so non-persistent RoPE buffers are initialized.
    """

    try:
        from safetensors import safe_open
        from transformers import Gemma4VisionModel
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder
    except ImportError as exc:  # pragma: no cover - main baseline environment
        raise RuntimeError("Selective Gemma 4 loading requires Transformers 5 and safetensors") from exc

    source = Path(checkpoint_path)
    if not source.is_file() or source.suffix != ".safetensors":
        raise FileNotFoundError(f"Gemma 4 safetensors checkpoint not found: {source}")
    vision_tower = Gemma4VisionModel(config.vision_config)
    embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)

    module_specs = {
        "model.vision_tower.": vision_tower,
        "model.embed_vision.": embed_vision,
    }
    selected_states: dict[str, dict[str, torch.Tensor]] = {
        prefix: {} for prefix in module_specs
    }
    with safe_open(source, framework="pt", device="cpu") as archive:
        checkpoint_keys = list(archive.keys())
        for checkpoint_key in checkpoint_keys:
            for prefix in module_specs:
                if checkpoint_key.startswith(prefix):
                    selected_states[prefix][checkpoint_key.removeprefix(prefix)] = archive.get_tensor(
                        checkpoint_key
                    )
                    break

    for prefix, module in module_specs.items():
        expected = set(module.state_dict())
        observed = set(selected_states[prefix])
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing or unexpected:
            raise RuntimeError(
                f"Selective Gemma 4 state mismatch for {prefix}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        result = module.load_state_dict(selected_states[prefix], strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Strict Gemma 4 state assignment failed for {prefix}: {result}")

    bundle = Gemma4VisionBundle(vision_tower, embed_vision)
    return bundle.to(device=device, dtype=dtype).eval()


def _grid_from_positions(
    sequence: torch.Tensor,
    *,
    valid_token_indices: tuple[int, ...],
    positions_xy: tuple[tuple[int, int], ...],
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """Scatter one padded row-major sequence into its explicit x/y grid."""

    if sequence.ndim != 3 or sequence.shape[0] != 1:
        raise ValueError(f"Expected one hidden sequence [1, tokens, dim], got {sequence.shape}")
    if len(valid_token_indices) != len(positions_xy):
        raise ValueError("Position and token-index counts disagree")
    grid_h, grid_w = grid_size
    if len(valid_token_indices) != grid_h * grid_w:
        raise ValueError("Valid Gemma 4 tokens do not fill the configured spatial grid")
    indices = torch.tensor(valid_token_indices, dtype=torch.long, device=sequence.device)
    valid = sequence.index_select(1, indices)[0]
    grid = torch.empty((grid_h, grid_w, sequence.shape[-1]), dtype=sequence.dtype, device=sequence.device)
    x = torch.tensor([position[0] for position in positions_xy], device=sequence.device)
    y = torch.tensor([position[1] for position in positions_xy], device=sequence.device)
    grid[y, x] = valid
    return grid


def _broadcast_pooled_aligned_grid(
    projected_post_pool: torch.Tensor,
    *,
    pre_to_post_token: tuple[int, ...],
    positions_xy: tuple[tuple[int, int], ...],
    pre_grid_size: tuple[int, int],
) -> torch.Tensor:
    """Broadcast every native post-pool token to its exact owning pre-pool cells."""

    if projected_post_pool.ndim == 3:
        if projected_post_pool.shape[0] != 1:
            raise ValueError("Expected projected post-pool tokens for one image")
        projected_post_pool = projected_post_pool[0]
    if projected_post_pool.ndim != 2:
        raise ValueError("projected_post_pool must have shape [tokens, dim]")
    if len(pre_to_post_token) != len(positions_xy):
        raise ValueError("Pre/post ownership and position counts disagree")
    if not pre_to_post_token or max(pre_to_post_token) >= projected_post_pool.shape[0]:
        raise ValueError("Pre-pool ownership references a missing projected token")
    ownership = torch.tensor(
        pre_to_post_token,
        dtype=torch.long,
        device=projected_post_pool.device,
    )
    expanded = projected_post_pool.index_select(0, ownership)
    grid_h, grid_w = pre_grid_size
    if expanded.shape[0] != grid_h * grid_w:
        raise ValueError("Broadcast aligned tokens do not fill the pre-pool grid")
    grid = torch.empty(
        (grid_h, grid_w, projected_post_pool.shape[-1]),
        dtype=projected_post_pool.dtype,
        device=projected_post_pool.device,
    )
    x = torch.tensor([position[0] for position in positions_xy], device=grid.device)
    y = torch.tensor([position[1] for position in positions_xy], device=grid.device)
    grid[y, x] = expanded
    return grid


class DenseGemma4Encoder:
    """Extract middle, late, and projected tokens from one complete image call."""

    def __init__(
        self,
        model: Any,
        image_processor: Any,
        spec: DenseVisionModelSpec,
        *,
        device: torch.device,
        compute_dtype: torch.dtype,
        storage_dtype: torch.dtype = torch.float16,
        middle_layer: int | None = None,
        late_layer: int | None = None,
        aligned_method: str = "pooled_native_projector_broadcast",
    ) -> None:
        if spec.architecture != "gemma4":
            raise ValueError(f"DenseGemma4Encoder cannot load architecture {spec.architecture!r}")
        if aligned_method != "pooled_native_projector_broadcast":
            raise ValueError(
                "Gemma 4 supports aligned_method=pooled_native_projector_broadcast"
            )
        self.model = model.eval()
        self.image_processor = image_processor
        self.spec = spec
        self.device = device
        self.compute_dtype = compute_dtype
        self.storage_dtype = storage_dtype
        self.middle_layer = spec.default_middle_layer if middle_layer is None else int(middle_layer)
        self.late_layer = spec.default_late_layer if late_layer is None else int(late_layer)
        self.aligned_method = aligned_method
        spec.validate_layers(self.middle_layer, self.late_layer)
        self._vision_tower, self._vision_projector = self._resolve_modules(model)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _resolve_modules(model: Any) -> tuple[Any, Any]:
        base = getattr(model, "model", model)
        vision_tower = getattr(base, "vision_tower", None)
        projector = getattr(base, "embed_vision", None)
        if vision_tower is None or projector is None:
            raise TypeError("Gemma 4 model must expose model.vision_tower and model.embed_vision")
        return vision_tower, projector

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str = "main",
        device: torch.device | None = None,
        requested_dtype: str = "float16",
        storage_dtype: torch.dtype = torch.float16,
        middle_layer: int | None = None,
        late_layer: int | None = None,
        aligned_method: str = "pooled_native_projector_broadcast",
        local_files_only: bool = False,
    ) -> DenseGemma4Encoder:
        """Load the pinned local checkpoint through the isolated Transformers 5 env."""

        try:
            from huggingface_hub import hf_hub_download
            from transformers import AutoConfig, Gemma4ImageProcessor
        except ImportError as exc:  # pragma: no cover - exercised by the main v4 environment
            raise RuntimeError(
                "Gemma 4 extraction requires the isolated Transformers 5 environment; "
                "run `make setup-gemma4-probe`"
            ) from exc

        spec = get_model_spec(model_id)
        selected_device = device or select_device()
        compute_dtype = safe_dtype(selected_device, requested_dtype)
        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        image_processor = Gemma4ImageProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        checkpoint = hf_hub_download(
            model_id,
            "model.safetensors",
            revision=revision,
            local_files_only=local_files_only,
        )
        model = load_selective_gemma4_vision_bundle(
            config,
            checkpoint,
            device=selected_device,
            dtype=compute_dtype,
        )
        return cls(
            model,
            image_processor,
            spec,
            device=selected_device,
            compute_dtype=compute_dtype,
            storage_dtype=storage_dtype,
            middle_layer=middle_layer,
            late_layer=late_layer,
            aligned_method=aligned_method,
        )

    def _prepare_image(self, image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("RGB arrays must have shape [H, W, 3] and dtype uint8")
            complete = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            complete = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")
        expected = (self.spec.image_size, self.spec.image_size)
        if complete.size != expected:
            raise ValueError(
                f"Dense spatial mapping requires a complete {expected[0]}x{expected[1]} render; "
                f"got {complete.width}x{complete.height}. Re-render rather than crop."
            )
        return complete

    def encode_image(self, image: Image.Image | np.ndarray) -> DensePatchFeatures:
        """Encode one complete frame and retain the unpooled localized grid."""

        complete = self._prepare_image(image)
        processed = self.image_processor(images=complete, return_tensors="pt")
        required = {"pixel_values", "image_position_ids"}
        if missing := required - set(processed):
            raise ValueError(f"Gemma 4 image processor omitted required fields: {sorted(missing)}")
        pixel_values = processed["pixel_values"]
        position_ids = processed["image_position_ids"]
        if pixel_values.ndim != 3 or pixel_values.shape[0] != 1:
            raise ValueError(
                "Gemma 4 must receive one complete patchified image [1, patches, patch_pixels]"
            )
        expected_patch_pixels = 3 * self.spec.patch_size**2
        if pixel_values.shape[-1] != expected_patch_pixels:
            raise ValueError(
                f"Expected {expected_patch_pixels} values per patch, got {pixel_values.shape[-1]}"
            )
        mapping = derive_vision_grid_mapping(
            position_ids,
            pooling_kernel_size=self.spec.pooling_kernel_size,
        )
        if (mapping.pre_grid_height, mapping.pre_grid_width) != self.spec.grid_size:
            raise ValueError(
                f"Processor produced {mapping.pre_grid_height}x{mapping.pre_grid_width}, "
                f"expected {self.spec.grid_size} for a complete square frame"
            )
        pixel_values = pixel_values.to(self.device, dtype=self.compute_dtype)
        position_ids = position_ids.to(self.device)

        with torch.inference_mode():
            # The only vision-encoder call. Middle and late grids come from
            # captured pre-pool layer states in this same complete-image pass.
            output = self._vision_tower(
                pixel_values=pixel_values,
                pixel_position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise ValueError("Gemma 4 vision tower did not expose hidden states")
            middle_index = self.spec.hidden_state_index(self.middle_layer)
            late_index = self.spec.hidden_state_index(self.late_layer)
            if max(middle_index, late_index) >= len(hidden_states) - 1:
                raise ValueError(
                    "Gemma 4 hidden-state tuple lacks the requested pre-pool layer states"
                )
            middle_sequence = hidden_states[middle_index]
            late_sequence = hidden_states[late_index]
            if middle_sequence.shape[-1] != self.spec.native_dim:
                raise ValueError("Gemma 4 middle feature dimension disagrees with model registry")
            if late_sequence.shape[-1] != self.spec.native_dim:
                raise ValueError("Gemma 4 late feature dimension disagrees with model registry")
            middle_grid = _grid_from_positions(
                middle_sequence,
                valid_token_indices=mapping.valid_pre_token_indices,
                positions_xy=mapping.pre_xy,
                grid_size=self.spec.grid_size,
            )
            late_grid = _grid_from_positions(
                late_sequence,
                valid_token_indices=mapping.valid_pre_token_indices,
                positions_xy=mapping.pre_xy,
                grid_size=self.spec.grid_size,
            )
            # This follows Gemma4Model.get_image_features exactly: the trained
            # embed_vision module consumes the vision tower's *post-pool*
            # last_hidden_state. Projecting pre-pool late states would be merely
            # shape-compatible and outside the trained native path.
            projected_post_pool = self._vision_projector(inputs_embeds=output.last_hidden_state)
            if projected_post_pool.shape[-1] != self.spec.aligned_dim:
                raise ValueError(
                    "Gemma 4 native projector returned an unexpected aligned feature shape: "
                    f"{tuple(projected_post_pool.shape)}"
                )
            post_count = mapping.post_token_count
            if projected_post_pool.numel() != post_count * self.spec.aligned_dim:
                raise ValueError(
                    "Gemma 4 projected post-pool token count disagrees with the spatial mapping"
                )
            projected_post_pool = projected_post_pool.reshape(post_count, self.spec.aligned_dim)
            aligned_grid = _broadcast_pooled_aligned_grid(
                projected_post_pool,
                pre_to_post_token=mapping.pre_to_post_token,
                positions_xy=mapping.pre_xy,
                pre_grid_size=self.spec.grid_size,
            )

        return DensePatchFeatures(
            native_middle=middle_grid.detach().to(device="cpu", dtype=self.storage_dtype),
            native_late=late_grid.detach().to(device="cpu", dtype=self.storage_dtype),
            clip_aligned=aligned_grid.detach().to(device="cpu", dtype=self.storage_dtype),
        )


__all__ = [
    "DenseGemma4Encoder",
    "Gemma4VisionBundle",
    "_broadcast_pooled_aligned_grid",
    "load_selective_gemma4_vision_bundle",
]

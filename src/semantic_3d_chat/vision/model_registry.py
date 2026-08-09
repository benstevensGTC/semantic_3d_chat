"""Dense vision model specifications used by the runtime.

The preserved CLIP baseline and the isolated Gemma 4 candidate each have an
explicit spatial contract. Keeping those differences in one registry makes
future SigLIP/DINO ablations explicit instead of relying on model-name
heuristics spread through the extractor.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseVisionModelSpec:
    """Static tensor-shape contract for a dense full-image encoder."""

    model_id: str
    architecture: str
    image_size: int
    patch_size: int
    native_dim: int
    aligned_dim: int
    num_hidden_layers: int
    default_middle_layer: int
    default_late_layer: int
    has_cls_token: bool
    license_name: str
    processed_grid_size: tuple[int, int] | None = None
    pooling_kernel_size: int = 1
    hidden_states_include_input_embedding: bool = True

    @property
    def grid_size(self) -> tuple[int, int]:
        if self.processed_grid_size is not None:
            return self.processed_grid_size
        if self.image_size % self.patch_size:
            raise ValueError(
                f"Image size {self.image_size} is not divisible by patch size {self.patch_size}"
            )
        side = self.image_size // self.patch_size
        return side, side

    @property
    def patch_count(self) -> int:
        grid_h, grid_w = self.grid_size
        return grid_h * grid_w

    @property
    def sequence_length(self) -> int:
        return self.patch_count + int(self.has_cls_token)

    def validate_layers(self, middle_layer: int, late_layer: int) -> None:
        """Validate indices into Transformers' hidden-state tuple.

        CLIP returns the input embedding state at index zero followed by one
        entry for every transformer layer.  A 12-layer model therefore accepts
        layer indices in ``[0, 12]``.
        """

        minimum = 0 if self.hidden_states_include_input_embedding else 1
        for name, layer in (("middle_layer", middle_layer), ("late_layer", late_layer)):
            if not minimum <= layer <= self.num_hidden_layers:
                raise ValueError(
                    f"{name}={layer} is outside [{minimum}, {self.num_hidden_layers}] "
                    f"for {self.model_id}"
                )
        if middle_layer >= late_layer:
            raise ValueError("middle_layer must be earlier than late_layer")

    def hidden_state_index(self, layer: int) -> int:
        """Map a human layer number to the model output's hidden-state tuple."""

        self.validate_layer(layer)
        return layer if self.hidden_states_include_input_embedding else layer - 1

    def validate_layer(self, layer: int) -> None:
        minimum = 0 if self.hidden_states_include_input_embedding else 1
        if not minimum <= layer <= self.num_hidden_layers:
            raise ValueError(
                f"layer={layer} is outside [{minimum}, {self.num_hidden_layers}] "
                f"for {self.model_id}"
            )


CLIP_VIT_BASE_PATCH16_224 = DenseVisionModelSpec(
    model_id="openai/clip-vit-base-patch16",
    architecture="clip",
    image_size=224,
    patch_size=16,
    native_dim=768,
    aligned_dim=512,
    num_hidden_layers=12,
    default_middle_layer=6,
    default_late_layer=12,
    has_cls_token=True,
    license_name="MIT",
)


GEMMA4_E2B = DenseVisionModelSpec(
    model_id="google/gemma-4-E2B-it",
    architecture="gemma4",
    # Runtime RGB/depth frames remain 224x224. The official aspect-preserving
    # processor resizes the complete square image to 768x768, then patchifies it.
    image_size=224,
    patch_size=16,
    native_dim=768,
    aligned_dim=1536,
    num_hidden_layers=16,
    default_middle_layer=8,
    default_late_layer=16,
    has_cls_token=False,
    license_name="Apache-2.0",
    processed_grid_size=(48, 48),
    pooling_kernel_size=3,
    hidden_states_include_input_embedding=False,
)


_MODEL_REGISTRY = {
    CLIP_VIT_BASE_PATCH16_224.model_id: CLIP_VIT_BASE_PATCH16_224,
    GEMMA4_E2B.model_id: GEMMA4_E2B,
}


def get_model_spec(model_id: str) -> DenseVisionModelSpec:
    """Return a supported dense model specification or fail loudly."""

    try:
        return _MODEL_REGISTRY[model_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unsupported dense vision model {model_id!r}; supported: {supported}") from exc


def registered_model_ids() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_REGISTRY))

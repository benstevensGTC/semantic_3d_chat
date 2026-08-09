"""Weight-free Gemma 4 E2B capability probe for the continuous-scene path.

This module deliberately does not load checkpoint weights.  It inspects the
pinned public configuration, builds tiny random Gemma 4 models on CPU, and
tests two contracts needed by this project:

* a complete image is patchified once and one vision forward exposes the
  localized, pre-pooling layer states as well as the spatially pooled states;
* arbitrary continuous scene-prefix embeddings can enter the decoder when
  Gemma 4's per-layer embedding (PLE) side input is supplied explicitly, and
  the resulting prefix can be reused through the causal KV cache.

The proposed primary path remains image -> dense features -> 3D map -> global
scene resampler -> decoder embeddings.  It never passes images directly to the
decoder and never turns the environment into text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.config import PROJECT_ROOT

GEMMA4_MODEL_ID = "google/gemma-4-E2B-it"
GEMMA4_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
GEMMA4_CONFIG_SHA256 = "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330"
GEMMA4_LICENSE = "apache-2.0"
GEMMA4_CHECKPOINT_BYTES = 10_246_621_918
GEMMA4_DECLARED_EFFECTIVE_PARAMETERS = 2_300_000_000
GEMMA4_REQUIREMENTS_TRANSFORMERS = "5.14.1"


@dataclass(frozen=True)
class VisionGridMapping:
    """Exact row-major mapping across Gemma 4's spatial average pooler."""

    pre_grid_height: int
    pre_grid_width: int
    post_grid_height: int
    post_grid_width: int
    pooling_kernel_size: int
    valid_pre_token_indices: tuple[int, ...]
    pre_xy: tuple[tuple[int, int], ...]
    pre_to_post_token: tuple[int, ...]
    post_xy: tuple[tuple[int, int], ...]

    @property
    def pre_token_count(self) -> int:
        return len(self.valid_pre_token_indices)

    @property
    def post_token_count(self) -> int:
        return len(self.post_xy)

    def post_token_pixel_bounds(
        self,
        post_token_index: int,
        *,
        patch_size: int,
    ) -> tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` in the resized complete image."""

        if not 0 <= post_token_index < self.post_token_count:
            raise IndexError(f"post_token_index outside [0, {self.post_token_count})")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        x, y = self.post_xy[post_token_index]
        span = patch_size * self.pooling_kernel_size
        return x * span, y * span, (x + 1) * span, (y + 1) * span


def _single_position_grid(pixel_position_ids: torch.Tensor) -> torch.Tensor:
    if pixel_position_ids.ndim == 3:
        if pixel_position_ids.shape[0] != 1:
            raise ValueError("Grid mapping currently requires exactly one complete image")
        pixel_position_ids = pixel_position_ids[0]
    if pixel_position_ids.ndim != 2 or pixel_position_ids.shape[-1] != 2:
        raise ValueError(
            "pixel_position_ids must have shape [tokens, 2] or [1, tokens, 2]"
        )
    return pixel_position_ids.detach().to(device="cpu", dtype=torch.long)


def derive_vision_grid_mapping(
    pixel_position_ids: torch.Tensor,
    *,
    pooling_kernel_size: int,
) -> VisionGridMapping:
    """Reproduce the official pooler's x/y-to-token mapping exactly.

    Position IDs are ``(x, y)`` and padding is ``(-1, -1)``.  Valid positions
    must form a complete rectangle.  Gemma 4 assigns each pre-pool patch to
    ``(x // k, y // k)`` and emits valid pooled tokens in row-major order after
    stripping the unused slots from its fixed soft-token budget.
    """

    if pooling_kernel_size <= 0:
        raise ValueError("pooling_kernel_size must be positive")
    positions = _single_position_grid(pixel_position_ids)
    padding = (positions == -1).all(dim=-1)
    partially_padded = (positions == -1).any(dim=-1) & ~padding
    if partially_padded.any():
        raise ValueError("Padding position IDs must use (-1, -1), not partial padding")
    valid_indices = (~padding).nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("No valid vision patch positions")
    valid = positions[valid_indices]
    if (valid < 0).any():
        raise ValueError("Valid vision patch positions must be non-negative")

    max_x = int(valid[:, 0].max().item())
    max_y = int(valid[:, 1].max().item())
    width = max_x + 1
    height = max_y + 1
    if width % pooling_kernel_size or height % pooling_kernel_size:
        raise ValueError(
            f"Pre-pool grid {height}x{width} is not divisible by pooling kernel "
            f"{pooling_kernel_size}"
        )

    observed = {(int(x), int(y)) for x, y in valid.tolist()}
    expected = {(x, y) for y in range(height) for x in range(width)}
    if observed != expected or len(observed) != valid.shape[0]:
        raise ValueError("Valid patch positions must form one duplicate-free rectangular grid")

    post_width = width // pooling_kernel_size
    post_height = height // pooling_kernel_size
    pre_to_post: list[int] = []
    for x, y in valid.tolist():
        pooled_x = int(x) // pooling_kernel_size
        pooled_y = int(y) // pooling_kernel_size
        pre_to_post.append(pooled_x + post_width * pooled_y)
    post_xy = tuple((x, y) for y in range(post_height) for x in range(post_width))
    if set(pre_to_post) != set(range(len(post_xy))):
        raise ValueError("Pooling did not cover every expected post-pool token")

    return VisionGridMapping(
        pre_grid_height=height,
        pre_grid_width=width,
        post_grid_height=post_height,
        post_grid_width=post_width,
        pooling_kernel_size=pooling_kernel_size,
        valid_pre_token_indices=tuple(int(index) for index in valid_indices.tolist()),
        pre_xy=tuple((int(x), int(y)) for x, y in valid.tolist()),
        pre_to_post_token=tuple(pre_to_post),
        post_xy=post_xy,
    )


def patchify_complete_image(
    image: torch.Tensor,
    *,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchify one whole image without any crop-level encoder calls.

    Args:
        image: Float tensor shaped ``[1, 3, height, width]`` in ``[0, 1]``.
        patch_size: Square patch side in pixels.

    Returns:
        Flattened patch pixels ``[1, grid_h*grid_w, 3*patch_size**2]`` and
        row-major ``(x, y)`` position IDs ``[1, grid_h*grid_w, 2]``.
    """

    if image.ndim != 4 or tuple(image.shape[:2]) != (1, 3):
        raise ValueError(f"Expected one RGB image [1, 3, H, W], got {tuple(image.shape)}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not image.dtype.is_floating_point:
        raise TypeError("Complete image must be a floating-point tensor")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Complete image must contain finite values in [0, 1]")
    height, width = image.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("Complete image dimensions must be divisible by patch_size")

    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = image.reshape(1, 3, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(
        1, grid_h * grid_w, 3 * patch_size**2
    )
    y, x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    positions = torch.stack((x, y), dim=-1).reshape(1, grid_h * grid_w, 2)
    return patches, positions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_summary(mapping: VisionGridMapping) -> dict[str, Any]:
    """Serialize an auditable compact mapping instead of thousands of coordinates."""

    payload = json.dumps(asdict(mapping), sort_keys=True, separators=(",", ":")).encode()
    return {
        "pre_grid_hw": [mapping.pre_grid_height, mapping.pre_grid_width],
        "post_grid_hw": [mapping.post_grid_height, mapping.post_grid_width],
        "pre_token_count": mapping.pre_token_count,
        "post_token_count": mapping.post_token_count,
        "pooling_kernel_size": mapping.pooling_kernel_size,
        "mapping_sha256": hashlib.sha256(payload).hexdigest(),
        "first_pre_xy": [list(value) for value in mapping.pre_xy[:6]],
        "first_pre_to_post": list(mapping.pre_to_post_token[:12]),
        "first_post_xy": [list(value) for value in mapping.post_xy[:6]],
        "last_post_xy": list(mapping.post_xy[-1]),
    }


def resolve_config_path(*, allow_network: bool) -> tuple[Path, dict[str, Any]]:
    """Resolve only ``config.json`` at the pinned revision, never model weights."""

    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache

    hub_metadata: dict[str, Any] = {
        "model_id": GEMMA4_MODEL_ID,
        "requested_revision": GEMMA4_REVISION,
        "resolved_revision": GEMMA4_REVISION,
        "gated": False,
        "private": False,
        "license": GEMMA4_LICENSE,
        "checkpoint_size_bytes": GEMMA4_CHECKPOINT_BYTES,
    }
    if allow_network:
        info = HfApi().model_info(GEMMA4_MODEL_ID, revision=GEMMA4_REVISION, files_metadata=True)
        files = {item.rfilename: item.size for item in info.siblings}
        config_path = Path(
            hf_hub_download(GEMMA4_MODEL_ID, "config.json", revision=GEMMA4_REVISION)
        )
        hub_metadata.update(
            {
                "resolved_revision": info.sha,
                "gated": bool(info.gated),
                "private": bool(info.private),
                "license": (info.card_data or {}).get("license", GEMMA4_LICENSE),
                "checkpoint_size_bytes": int(
                    files.get("model.safetensors") or GEMMA4_CHECKPOINT_BYTES
                ),
            }
        )
    else:
        cached = try_to_load_from_cache(
            GEMMA4_MODEL_ID,
            "config.json",
            revision=GEMMA4_REVISION,
        )
        if not isinstance(cached, str):
            raise FileNotFoundError(
                "Pinned Gemma 4 config is not cached; run `make download-gemma4-config`"
            )
        config_path = Path(cached)

    actual_sha = _sha256_file(config_path)
    if actual_sha != GEMMA4_CONFIG_SHA256:
        raise RuntimeError(
            f"Pinned Gemma 4 config hash mismatch: expected {GEMMA4_CONFIG_SHA256}, "
            f"got {actual_sha}"
        )
    if hub_metadata["resolved_revision"] != GEMMA4_REVISION:
        raise RuntimeError("The model hub resolved a different revision than the pinned commit")
    hub_metadata["config_sha256"] = actual_sha
    hub_metadata["weights_downloaded_by_probe"] = False
    return config_path, hub_metadata


def _meta_parameter_counts(config: Any) -> tuple[int, dict[str, int]]:
    """Count the actual installed architecture on the meta device (zero weight RAM)."""

    from transformers import Gemma4ForConditionalGeneration

    with torch.device("meta"):
        model = Gemma4ForConditionalGeneration(config)
    components: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        component = parts[1] if parts[0] == "model" and len(parts) > 1 else parts[0]
        components[component] = components.get(component, 0) + parameter.numel()
    return sum(components.values()), dict(sorted(components.items()))


def _text_kv_bytes_per_token(text_config: Any, *, bytes_per_element: int = 2) -> int:
    """Estimate the non-shared decoder KV-cache growth before sliding eviction."""

    nonshared_layers = text_config.num_hidden_layers - text_config.num_kv_shared_layers
    total_elements = 0
    for layer_type in text_config.layer_types[:nonshared_layers]:
        is_sliding = layer_type == "sliding_attention"
        head_dim = text_config.head_dim if is_sliding else text_config.global_head_dim
        kv_heads = text_config.num_key_value_heads
        total_elements += 2 * kv_heads * head_dim
    return int(total_elements * bytes_per_element)


def model_shape_and_memory_report(config: Any, checkpoint_size_bytes: int) -> dict[str, Any]:
    """Create shape facts and conservative, explicitly labeled memory lower bounds."""

    total_parameters, components = _meta_parameter_counts(config)
    text = config.text_config
    vision = config.vision_config
    bf16_bytes = total_parameters * 2
    fp32_bytes = total_parameters * 4
    kv_per_token = _text_kv_bytes_per_token(text)
    return {
        "model_card_declared_effective_parameters": GEMMA4_DECLARED_EFFECTIVE_PARAMETERS,
        "total_parameters": total_parameters,
        "component_parameters": components,
        "checkpoint_bytes": checkpoint_size_bytes,
        "weights_only": {
            "bf16_or_fp16_bytes": bf16_bytes,
            "bf16_or_fp16_gib": bf16_bytes / 2**30,
            "fp32_bytes": fp32_bytes,
            "fp32_gib": fp32_bytes / 2**30,
        },
        "kv_cache_estimate": {
            "assumptions": (
                "bf16/fp16, batch=1, only non-shared KV layers, before sliding-window "
                "eviction; excludes activations and allocator overhead"
            ),
            "bytes_per_token": kv_per_token,
            "mib_for_512_tokens": kv_per_token * 512 / 2**20,
            "mib_for_1024_tokens": kv_per_token * 1024 / 2**20,
        },
        "warning": (
            "Weights-only and KV figures are lower bounds, not a peak-MPS-memory promise. "
            "Training through a frozen decoder retains activations and needs substantially more."
        ),
        "text": {
            "hidden_size": text.hidden_size,
            "layers": text.num_hidden_layers,
            "per_layer_embedding_dim": text.hidden_size_per_layer_input,
            "kv_shared_layers": text.num_kv_shared_layers,
            "sliding_window": text.sliding_window,
        },
        "vision": {
            "hidden_size": vision.hidden_size,
            "layers": vision.num_hidden_layers,
            "patch_size": vision.patch_size,
            "pooling_kernel_size": vision.pooling_kernel_size,
        },
    }


def _tiny_vision_config() -> Any:
    from transformers import Gemma4VisionConfig

    return Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        pooling_kernel_size=3,
        patch_size=16,
        position_embedding_size=64,
        use_clipped_linears=False,
        standardize=False,
    )


def run_tiny_vision_probe() -> dict[str, Any]:
    """Exercise one complete-image vision call with random tiny weights on CPU."""

    from transformers import Gemma4VisionModel

    torch.manual_seed(1307)
    config = _tiny_vision_config()
    model = Gemma4VisionModel(config).to(device="cpu").eval()

    # A single 96x96 complete image gives a 6x6 pre-pool grid and 2x2
    # post-pool grid.  The horizontal and vertical gradients make spatial
    # non-uniformity deterministic without encoding any semantic label.
    horizontal = torch.linspace(0.0, 1.0, 96).view(1, 1, 1, 96).expand(1, 1, 96, 96)
    vertical = torch.linspace(0.0, 1.0, 96).view(1, 1, 96, 1).expand(1, 1, 96, 96)
    image = torch.cat((horizontal, vertical, (horizontal + vertical) / 2), dim=1)
    pixel_values, position_ids = patchify_complete_image(image, patch_size=config.patch_size)
    mapping = derive_vision_grid_mapping(
        position_ids,
        pooling_kernel_size=config.pooling_kernel_size,
    )

    calls = 0

    def count_call(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal calls
        calls += 1

    hook = model.register_forward_hook(count_call)
    with torch.inference_mode():
        output = model(
            pixel_values=pixel_values,
            pixel_position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    hook.remove()

    hidden_shapes = [list(tensor.shape) for tensor in output.hidden_states or ()]
    pre_pool = output.hidden_states[0] if output.hidden_states else None
    post_pool = output.last_hidden_state
    expected_pre = (1, mapping.pre_token_count, config.hidden_size)
    expected_post = (mapping.post_token_count, config.hidden_size)
    if pre_pool is None or tuple(pre_pool.shape) != expected_pre:
        raise RuntimeError(f"Unexpected pre-pool state shape; expected {expected_pre}")
    if tuple(post_pool.shape) != expected_post:
        raise RuntimeError(f"Unexpected post-pool state shape; expected {expected_post}")
    if calls != 1:
        raise RuntimeError(f"Complete image required exactly one vision call, observed {calls}")
    pre_variance = float(pre_pool.float().var(dim=1).mean().item())
    post_variance = float(post_pool.float().var(dim=0).mean().item())
    if not math.isfinite(pre_variance + post_variance) or min(pre_variance, post_variance) <= 0:
        raise RuntimeError("Gemma 4 vision probe did not retain spatially distinct features")

    return {
        "passed": True,
        "device": "cpu",
        "complete_image_shape": list(image.shape),
        "vision_forward_calls": calls,
        "encoder_input_shape": list(pixel_values.shape),
        "hidden_state_shapes": hidden_shapes,
        "pre_pool_layer_state_shape": list(pre_pool.shape),
        "post_pool_state_shape": list(post_pool.shape),
        "pre_pool_spatial_variance": pre_variance,
        "post_pool_spatial_variance": post_variance,
        "mapping": _mapping_summary(mapping),
        "contract": (
            "Each hidden state before the final tuple entry is localized on image_position_ids; "
            "the final/last_hidden_state is pooled and padding-stripped, with batch flattened."
        ),
    }


def run_processor_grid_probe(config: Any) -> dict[str, Any]:
    """Measure the official default mapping for one complete 224x224 image."""

    from transformers import Gemma4ImageProcessor

    image_processor = Gemma4ImageProcessor(
        patch_size=config.vision_config.patch_size,
        max_soft_tokens=config.vision_soft_tokens_per_image,
        pooling_kernel_size=config.vision_config.pooling_kernel_size,
    )
    image = torch.zeros((3, 224, 224), dtype=torch.float32)
    processed = image_processor(image, return_tensors="pt")
    mapping = derive_vision_grid_mapping(
        processed["image_position_ids"],
        pooling_kernel_size=config.vision_config.pooling_kernel_size,
    )
    resized_height = mapping.pre_grid_height * config.vision_config.patch_size
    resized_width = mapping.pre_grid_width * config.vision_config.patch_size
    expected_soft = int(processed["num_soft_tokens_per_image"][0])
    if expected_soft != mapping.post_token_count:
        raise RuntimeError("Processor soft-token count disagrees with derived pooling grid")
    return {
        "input_complete_image_hw": [224, 224],
        "aspect_ratio_preserved": True,
        "crop_used": False,
        "resized_complete_image_hw": [resized_height, resized_width],
        "padded_encoder_input_shape": list(processed["pixel_values"].shape),
        "valid_pre_pool_tokens": mapping.pre_token_count,
        "valid_post_pool_tokens": mapping.post_token_count,
        "pre_pool_grid_hw": [mapping.pre_grid_height, mapping.pre_grid_width],
        "post_pool_grid_hw": [mapping.post_grid_height, mapping.post_grid_width],
        "post_token_pixel_span": (
            config.vision_config.patch_size * config.vision_config.pooling_kernel_size
        ),
        "mapping": _mapping_summary(mapping),
    }


def _tiny_conditional_config() -> Any:
    from transformers import Gemma4Config, Gemma4TextConfig

    text = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=64,
        layer_types=["sliding_attention", "full_attention", "full_attention"],
        sliding_window=8,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        use_double_wide_mlp=False,
        num_kv_shared_layers=1,
    )
    return Gemma4Config(
        text_config=text,
        vision_config=None,
        audio_config=None,
        image_token_id=60,
        video_token_id=61,
        audio_token_id=62,
        boi_token_id=58,
        eoi_token_id=59,
    )


def run_tiny_decoder_probe() -> dict[str, Any]:
    """Validate arbitrary continuous prefix + explicit PLE + cached decoding."""

    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(2053)
    config = _tiny_conditional_config()
    model = Gemma4ForConditionalGeneration(config).to(device="cpu").eval()
    text_model = model.model.language_model
    scene_length = 5
    prompt_ids = torch.tensor([[2, 11, 17]], dtype=torch.long)
    prompt_embeds = model.get_input_embeddings()(prompt_ids)
    scene_embeds = torch.randn(1, scene_length, config.text_config.hidden_size)
    if any(
        torch.equal(scene_embeds[0, index], row)
        for index in range(scene_length)
        for row in model.get_input_embeddings().weight
    ):
        raise RuntimeError("Tiny random scene prefix unexpectedly equals a token embedding")
    inputs_embeds = torch.cat((scene_embeds, prompt_embeds), dim=1)

    # Gemma 4 E2B uses per-layer embeddings. Native visual soft tokens retain
    # the PAD token's PLE identity after their placeholder embeddings are
    # replaced, so we use that same non-semantic identity for scene latents.
    # Real prompt tokens keep their own learned PLE lookup. The text model adds
    # its context-aware PLE projection internally for both pieces.
    scene_placeholder_ids = torch.full(
        (1, scene_length),
        config.text_config.pad_token_id,
        dtype=torch.long,
    )
    scene_placeholder_embeds = model.get_input_embeddings()(scene_placeholder_ids)
    scene_ple = text_model.get_per_layer_inputs(
        scene_placeholder_ids,
        scene_placeholder_embeds,
    )
    prompt_ple = text_model.get_per_layer_inputs(prompt_ids, prompt_embeds)
    per_layer_inputs = torch.cat((scene_ple, prompt_ple), dim=1)
    prefix_length = inputs_embeds.shape[1]
    mm_token_type_ids = torch.zeros((1, prefix_length), dtype=torch.long)
    with torch.inference_mode():
        first = model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_mask=torch.ones((1, prefix_length), dtype=torch.long),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
    cache_length_before = first.past_key_values.get_seq_length()

    next_ids = torch.tensor([[23]], dtype=torch.long)
    next_embeds = model.get_input_embeddings()(next_ids)
    next_ple = text_model.get_per_layer_inputs(next_ids, next_embeds)
    with torch.inference_mode():
        second = model(
            inputs_embeds=next_embeds,
            per_layer_inputs=next_ple,
            attention_mask=torch.ones((1, prefix_length + 1), dtype=torch.long),
            mm_token_type_ids=torch.zeros((1, 1), dtype=torch.long),
            past_key_values=first.past_key_values,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
    cache_length_after = second.past_key_values.get_seq_length()
    if cache_length_before != prefix_length or cache_length_after != prefix_length + 1:
        raise RuntimeError("Gemma 4 KV cache did not preserve and extend the continuous prefix")
    if tuple(first.logits.shape) != (1, 1, config.text_config.vocab_size):
        raise RuntimeError("Unexpected first-step logits shape")
    if not torch.isfinite(first.logits).all() or not torch.isfinite(second.logits).all():
        raise RuntimeError("Gemma 4 decoder probe produced non-finite logits")

    missing_ple_error = ""
    try:
        with torch.inference_mode():
            model(inputs_embeds=scene_embeds, use_cache=False, return_dict=True)
    except RuntimeError as exc:
        missing_ple_error = str(exc)
    if "per-layer" not in missing_ple_error and "exactly match" not in missing_ple_error:
        raise RuntimeError("Probe did not confirm that arbitrary E2B embeddings require explicit PLE")

    return {
        "passed": True,
        "device": "cpu",
        "arbitrary_scene_prefix_shape": list(scene_embeds.shape),
        "prompt_token_count": prompt_ids.shape[1],
        "explicit_per_layer_inputs_shape": list(per_layer_inputs.shape),
        "scene_ple_identity": "pad_token_native_multimodal_convention",
        "mm_token_type_for_scene_prefix": 0,
        "first_logits_shape": list(first.logits.shape),
        "cache_class": type(first.past_key_values).__name__,
        "cache_length_before_increment": cache_length_before,
        "cache_length_after_increment": cache_length_after,
        "missing_ple_rejected": True,
        "contract": (
            "Use the native multimodal PAD-token PLE identity for arbitrary scene latents, "
            "learned token-identity PLE for prompt tokens, and supply the concatenated "
            "per_layer_inputs explicitly. "
            "On cached steps pass only the new token embedding/PLE plus a full-length attention mask."
        ),
    }


def build_capability_report(*, allow_network: bool) -> dict[str, Any]:
    """Build a reproducible report without downloading or loading model weights."""

    import transformers
    from transformers import AutoConfig

    if transformers.__version__ != GEMMA4_REQUIREMENTS_TRANSFORMERS:
        raise RuntimeError(
            f"Gemma 4 probe requires Transformers {GEMMA4_REQUIREMENTS_TRANSFORMERS}, "
            f"found {transformers.__version__}. Use .venv-gemma4."
        )
    config_path, hub = resolve_config_path(allow_network=allow_network)
    config = AutoConfig.from_pretrained(config_path.parent, local_files_only=True)
    modeling_path = Path(__import__(
        "transformers.models.gemma4.modeling_gemma4",
        fromlist=["__file__"],
    ).__file__)
    report = {
        "format_version": 1,
        "status": "capability_validated_without_checkpoint_weights",
        "model": hub,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "modeling_source_sha256": _sha256_file(modeling_path),
        },
        "shape_and_memory": model_shape_and_memory_report(
            config,
            checkpoint_size_bytes=int(hub["checkpoint_size_bytes"]),
        ),
        "processor_grid_224_square": run_processor_grid_probe(config),
        "tiny_cpu_vision": run_tiny_vision_probe(),
        "tiny_cpu_decoder": run_tiny_decoder_probe(),
        "primary_integration_design": {
            "path": [
                "complete RGB image",
                "one Gemma 4 vision-tower call",
                "pre-pool localized patch hidden states + image_position_ids",
                "depth projection and persistent 3D voxel fusion",
                "question-independent full-scene resampler",
                "projected continuous scene prefix with explicit PLE side input",
                "Gemma 4 causal decoder",
            ],
            "direct_image_chat_primary": False,
            "environmental_text_or_labels": False,
            "question_dependent_retrieval": False,
            "checkpoint_weights_downloaded_by_probe": False,
            "checkpoint_weights_loaded_by_probe": False,
        },
        "decision": {
            "promising": True,
            "reason": (
                "Gemma 4 E2B provides an integrated 768D dense vision tower and 1536D local "
                "decoder, and both required low-level contracts pass with tiny CPU models."
            ),
            "remaining_gate": (
                "Download the 10.25 GB checkpoint only after the current baseline is preserved, "
                "then measure real MPS forward stability, peak unified memory, dense feature "
                "quality, and adapter gradients before making it the primary model."
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-config",
        action="store_true",
        help="Allow network access for config.json and hub file metadata only; never downloads weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "metrics" / "gemma4_e2b_capability_probe.json",
    )
    args = parser.parse_args()
    report = build_capability_report(allow_network=args.download_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": report["status"],
        "model": report["model"]["model_id"],
        "revision": report["model"]["resolved_revision"],
        "parameters": report["shape_and_memory"]["total_parameters"],
        "checkpoint_gib": report["model"]["checkpoint_size_bytes"] / 2**30,
        "vision_passed": report["tiny_cpu_vision"]["passed"],
        "decoder_prefix_passed": report["tiny_cpu_decoder"]["passed"],
        "weights_loaded_by_probe": report["primary_integration_design"][
            "checkpoint_weights_loaded_by_probe"
        ],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

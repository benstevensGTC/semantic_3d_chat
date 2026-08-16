"""Exact Gemma 4 E2B sliding-attention exposure for atlas layouts.

The calculation follows Transformers' causal sliding mask predicate
``key_position > query_position - sliding_window``.  It describes the direct
key exposure of the *final prompt token in a sliding-attention layer*.  Gemma's
periodic full-attention layers are deliberately outside this local-window
calculation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal

GEMMA4_E2B_MODEL_ID: Final[str] = "google/gemma-4-E2B-it"
GEMMA4_E2B_SLIDING_WINDOW: Final[int] = 512
AtlasLayout = Literal["v1", "v2"]


@dataclass(frozen=True)
class FinalPromptSlidingExposure:
    """Direct sliding-attention visibility at the final prompt position."""

    model_id: str
    layout_version: AtlasLayout
    prompt_token_count_including_bos: int
    sliding_window: int
    final_prompt_query_position: int
    first_visible_key_position: int
    last_visible_key_position: int
    base_first_position: int
    base_last_position: int
    visible_base_latent_count: int
    total_base_latent_count: int
    atlas_first_position: int
    atlas_last_position: int
    visible_atlas_token_count: int
    total_atlas_token_count: int
    boi_position: int
    eoi_position: int
    boi_visible: bool
    eoi_visible: bool
    mask_predicate: str
    attention_layer_kind: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _closed_intersection_count(
    first_a: int,
    last_a: int,
    first_b: int,
    last_b: int,
) -> int:
    return max(0, min(last_a, last_b) - max(first_a, first_b) + 1)


def final_prompt_sliding_exposure(
    layout_version: AtlasLayout,
    *,
    prompt_token_count_including_bos: int,
    base_latent_count: int = 256,
    atlas_memory_token_count: int = 480,
    sliding_window: int = GEMMA4_E2B_SLIDING_WINDOW,
) -> FinalPromptSlidingExposure:
    """Calculate exact local-window visibility for a BOS-first scene prefix.

    The full sequence is ``[BOS][scene prefix][remaining prompt]``.  The scene
    prefix contains one BOI, both variable blocks, and one EOI.  Prompt length
    includes the BOS already present at position zero.
    """

    if layout_version not in ("v1", "v2"):
        raise ValueError(f"Unknown fixed-prefix atlas layout: {layout_version!r}")
    if prompt_token_count_including_bos < 1:
        raise ValueError("Prompt token count must include at least BOS")
    if base_latent_count < 1 or atlas_memory_token_count < 1:
        raise ValueError("Base-latent and atlas-memory counts must be positive")
    if sliding_window < 1:
        raise ValueError("Sliding window must be positive")

    scene_prefix_tokens = base_latent_count + atlas_memory_token_count + 2
    final_query = scene_prefix_tokens + prompt_token_count_including_bos - 1
    first_visible = max(0, final_query - sliding_window + 1)
    last_visible = final_query
    boi_position = 1
    eoi_position = scene_prefix_tokens

    if layout_version == "v1":
        base_first = 2
        base_last = base_first + base_latent_count - 1
        atlas_first = base_last + 1
        atlas_last = atlas_first + atlas_memory_token_count - 1
    else:
        atlas_first = 2
        atlas_last = atlas_first + atlas_memory_token_count - 1
        base_first = atlas_last + 1
        base_last = base_first + base_latent_count - 1

    return FinalPromptSlidingExposure(
        model_id=GEMMA4_E2B_MODEL_ID,
        layout_version=layout_version,
        prompt_token_count_including_bos=prompt_token_count_including_bos,
        sliding_window=sliding_window,
        final_prompt_query_position=final_query,
        first_visible_key_position=first_visible,
        last_visible_key_position=last_visible,
        base_first_position=base_first,
        base_last_position=base_last,
        visible_base_latent_count=_closed_intersection_count(
            base_first, base_last, first_visible, last_visible
        ),
        total_base_latent_count=base_latent_count,
        atlas_first_position=atlas_first,
        atlas_last_position=atlas_last,
        visible_atlas_token_count=_closed_intersection_count(
            atlas_first, atlas_last, first_visible, last_visible
        ),
        total_atlas_token_count=atlas_memory_token_count,
        boi_position=boi_position,
        eoi_position=eoi_position,
        boi_visible=first_visible <= boi_position <= last_visible,
        eoi_visible=first_visible <= eoi_position <= last_visible,
        mask_predicate="key_position > query_position - sliding_window",
        attention_layer_kind="sliding_attention",
    )


def gemma4_e2b_prompt_exposure_table(
    *,
    minimum_prompt_tokens: int = 57,
    maximum_prompt_tokens: int = 64,
) -> tuple[tuple[FinalPromptSlidingExposure, FinalPromptSlidingExposure], ...]:
    """Return paired V1/V2 exposure records over an inclusive prompt range."""

    if maximum_prompt_tokens < minimum_prompt_tokens:
        raise ValueError("Maximum prompt length must not be below minimum")
    return tuple(
        (
            final_prompt_sliding_exposure(
                "v1", prompt_token_count_including_bos=prompt_tokens
            ),
            final_prompt_sliding_exposure(
                "v2", prompt_token_count_including_bos=prompt_tokens
            ),
        )
        for prompt_tokens in range(minimum_prompt_tokens, maximum_prompt_tokens + 1)
    )


__all__ = [
    "GEMMA4_E2B_MODEL_ID",
    "GEMMA4_E2B_SLIDING_WINDOW",
    "AtlasLayout",
    "FinalPromptSlidingExposure",
    "final_prompt_sliding_exposure",
    "gemma4_e2b_prompt_exposure_table",
]

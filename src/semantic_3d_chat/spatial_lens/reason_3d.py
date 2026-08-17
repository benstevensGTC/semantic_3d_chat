"""Ask Gemma about a room by handing it the 3D scene, not a description.

The tokens produced by :mod:`scene_tokens_3d` go into the decoder through the
same continuous pathway Gemma uses for image tokens, wrapped in its native
image-boundary embeddings.  The question follows as ordinary text.  There is no
scene graph, no object list and no caption anywhere in the prompt: if the model
says anything true about the room, it read it out of the 3D field.

Because that claim is easy to make and hard to earn, every question is also run
against a scrambled-layout copy and a zeroed copy of the same tokens.  An answer
that does not change under those controls was not grounded in the scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from semantic_3d_chat.spatial_lens.scene_tokens_3d import SceneTokens3D

SYSTEM = (
    "You are looking at a three-dimensional scan of a room, supplied as visual "
    "tokens laid out as a bird's-eye grid: the first tokens are the far side of "
    "the room and the last are the near side, left to right. Answer the question "
    "about what is in the room and where. Be concise."
)


@dataclass(frozen=True)
class Answer3D:
    question: str
    answer: str
    condition: str


def _prefix_embeddings(
    backend: Any, tokens: SceneTokens3D, device: torch.device
) -> torch.Tensor:
    """Wrap the grid in Gemma's own begin/end-of-image embeddings."""

    grid = torch.from_numpy(np.ascontiguousarray(tokens.tokens)).to(
        device=device, dtype=torch.float32
    )
    boi, eoi = backend.native_boundary_embeddings()
    return torch.cat(
        [
            boi.to(device=device, dtype=torch.float32).reshape(1, 1, -1),
            grid.unsqueeze(0),
            eoi.to(device=device, dtype=torch.float32).reshape(1, 1, -1),
        ],
        dim=1,
    )


def ask_3d(
    language: Any,
    tokens: SceneTokens3D,
    question: str,
    *,
    max_new_tokens: int = 96,
) -> str:
    """Answer one question with the 3D scene as the only evidence."""

    from semantic_3d_chat.language.local_lm import prompt_token_ids
    from semantic_3d_chat.language.prefix_injection import (
        SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    )

    backend = language.prefix_backend
    device = language.device
    prefix = _prefix_embeddings(backend, tokens, device)
    prompt_ids = prompt_token_ids(language.tokenizer, SYSTEM, question, device)
    reference = backend.native_boundary_embeddings()[0]
    prepared = backend.prepare(
        prefix.to(dtype=reference.dtype),
        prompt_ids,
        scene_prefix_after_bos=True,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    )
    # The backend's own generate() carries Gemma 4's auxiliary per-layer input
    # stream through prefill and every cached step; the plain embeddings path
    # would make the decoder re-derive it and blow up.
    produced = backend.generate(
        prepared,
        max_new_tokens=max_new_tokens,
        eos_token_ids=getattr(language.tokenizer, "eos_token_id", None),
    )
    return language.tokenizer.decode(
        produced[0].detach().cpu(), skip_special_tokens=True
    ).strip()


__all__ = ["SYSTEM", "Answer3D", "ask_3d"]

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
    "tokens laid out as a bird's-eye grid of the floor, read row by row: the "
    "first tokens are one edge of the room and the last are the opposite edge, "
    "each row running left to right. Answer the question about what is in the "
    "room and where. Be concise."
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
    system: str | None = None,
    rope3d: bool = False,
    span_units: float = 256.0,
) -> str:
    """Answer one question with the 3D scene as the only evidence.

    With ``rope3d`` the decoder's own rotary position encoding is driven by each
    token's position in the room rather than by its index in the sequence, so
    attention between two scene tokens depends on the displacement between the
    two places. The raster convention in the system prompt then becomes
    unnecessary, and no coordinate is written out as text either way.
    """

    from semantic_3d_chat.language.local_lm import prompt_token_ids
    from semantic_3d_chat.language.prefix_injection import (
        SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    )

    backend = language.prefix_backend
    device = language.device
    prefix = _prefix_embeddings(backend, tokens, device)
    prompt_ids = prompt_token_ids(
        language.tokenizer, system or SYSTEM, question, device
    )
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
    def _generate() -> Any:
        return backend.generate(
            prepared,
            max_new_tokens=max_new_tokens,
            eos_token_ids=getattr(language.tokenizer, "eos_token_id", None),
        )

    if rope3d:
        from semantic_3d_chat.language.rope3d_patch import (
            ScenePositions,
            attach_rope3d,
            scene_span_from_mask,
        )

        if tokens.centroids_m is None:
            raise ValueError("rope3d needs token centroids; rebuild the scene tokens")
        marks = getattr(prepared, "mm_token_type_ids", None)
        if marks is None:
            raise ValueError("rope3d needs the prepared batch's multimodal mask")
        start, stop = scene_span_from_mask(marks)
        places = torch.from_numpy(tokens.centroids_m).to(device=device, dtype=torch.float32)
        if stop - start != places.shape[0]:
            raise ValueError(
                f"scene span is {stop - start} tokens but {places.shape[0]} centroids"
            )
        with attach_rope3d(language.model, ScenePositions(start, places), span_units=span_units):
            produced = _generate()
    else:
        produced = _generate()
    return language.tokenizer.decode(
        produced[0].detach().cpu(), skip_special_tokens=True
    ).strip()


__all__ = ["LOCATE_SYSTEM", "SYSTEM", "Answer3D", "ask_3d", "locate_3d"]


# The grid is emitted with row 0 at y = -depth/2 and column 0 at x = -width/2,
# so the convention stated here must match cell_center_m exactly. An earlier
# version described row 0 as the far side, which is the opposite of the code.
LOCATE_SYSTEM = (
    "You are looking at a three-dimensional scan of a room, supplied as visual "
    "tokens arranged as a bird's-eye grid of {grid} rows by {grid} columns "
    "covering the whole floor.\n"
    "Row 0 is the y = {y_min:+.1f} m edge and row {last} is the y = {y_max:+.1f} m "
    "edge. Column 0 is the x = {x_min:+.1f} m edge and column {last} is the "
    "x = {x_max:+.1f} m edge.\n"
    "The user names one object. Report which grid cell it occupies.\n"
    'Reply with ONE json object and nothing else: {{"row": <0-{last}>, '
    '"col": <0-{last}>, "found": true|false}}. Set found=false if that object '
    "is not in the scan."
)


def locate_3d(
    language: Any,
    tokens: SceneTokens3D,
    name: str,
    *,
    max_new_tokens: int = 48,
    rope3d: bool = False,
) -> tuple[float, float] | None:
    """Ask where an object is, using only the 3D field, and return metres.

    This is what keeps navigation grounded in the scene rather than in a list of
    coordinates: the target's position is read out of the same tokens the model
    would use to describe the room.
    """

    import json as _json
    import re as _re

    grid = tokens.grid
    width, depth, _ = tokens.room_size_m
    system = LOCATE_SYSTEM.format(
        grid=grid,
        last=grid - 1,
        x_min=-width / 2.0,
        x_max=width / 2.0,
        y_min=-depth / 2.0,
        y_max=depth / 2.0,
    )
    reply = ask_3d(
        language,
        tokens,
        f"Where is the {name}?",
        max_new_tokens=max_new_tokens,
        system=system,
        rope3d=rope3d,
    )
    match = _re.search(r"\{.*\}", reply, _re.DOTALL)
    if match is None:
        return None
    try:
        payload = _json.loads(match.group(0))
    except ValueError:
        return None
    if payload.get("found") is False:
        return None
    try:
        row = int(payload["row"])
        column = int(payload["col"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= row < grid and 0 <= column < grid):
        return None
    return tokens.cell_center_m(row * grid + column)

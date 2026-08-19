"""Give Gemma's own attention a three-dimensional sense of where points are.

Gemma computes one ``(cos, sin)`` pair per position and hands it to every
attention layer, which rotates queries and keys by it. That is the mechanism
that makes an attention score depend on the distance between two tokens instead
of on their absolute indices, and it is the only place position enters the
decoder at all.

The substitution here keeps that mechanism and changes what it is a function of.
For the span of scene tokens, the frequency slots are dealt out among x, y and z
and each slot is driven by its own axis, so a score between two scene tokens
becomes a function of the displacement between the two points in the room. Text
tokens keep ordinary sequence positions. This is Qwen2-VL's M-RoPE with metres
in place of time, height and width.

Nothing is trained and no coordinate is ever written down for the model to read:
position stays a rotation, which is the form the decoder's pretrained attention
already knows how to use.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ScenePositions:
    """Where each scene token sits, and where it sits in the sequence."""

    start: int
    positions: torch.Tensor  # [n, 3] metres

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("scene positions must be [n, 3]")


def index_units(
    positions: torch.Tensor, span_units: float = 256.0, *, centred: bool = False
) -> torch.Tensor:
    """Rescale metres into the numeric range Gemma's rotary frequencies expect.

    The decoder's frequencies were fitted against integer token indices, and a
    Gemma image occupies 256 of them. Stretching a room across the same range
    keeps every rotation inside the regime the frozen weights were trained in;
    feeding raw metres would leave all but the highest frequency band barely
    turning at all. One scale is used for all three axes so the room stays rigid
    rather than being stretched along its longest side.

    With ``centred`` the result is symmetric about zero, which is what the
    anchored mode wants: the room becomes an offset around wherever the scene
    sits in the sequence, instead of replacing that position outright.
    """

    extent = positions.reshape(-1, 3)
    low = extent.min(dim=0).values
    high = extent.max(dim=0).values
    size = (high - low).clamp(min=1e-3)
    # Centre on each axis's own midpoint. Subtracting one global half-span
    # instead centres only the longest axis and leaves the others sitting
    # entirely to one side of zero -- for a room-shaped scan that put the whole
    # height band about ninety index units from where it belonged.
    origin = (low + high) / 2.0 if centred else low
    return (positions - origin) * (span_units / size.max())


class Rope3DRotary(nn.Module):
    """Wraps Gemma's rotary module and overrides the scene span with 3D angles."""

    def __init__(
        self,
        inner: nn.Module,
        scene: ScenePositions,
        *,
        span_units: float = 256.0,
        anchored: bool = True,
        axes: str = "xyz",
    ) -> None:
        super().__init__()
        self.inner = inner
        self.scene = scene
        self.span_units = float(span_units)
        # Anchored keeps the scene where it sits in the sequence and lets the
        # room be an offset around that point. Unanchored throws the sequence
        # position away, which also throws away how far the question is from
        # the scene -- the decoder can no longer tell the two apart in the way
        # its pretrained attention expects.
        self.anchored = bool(anchored)
        if axes not in {"xyz", "z_only"}:
            raise ValueError(f"unknown axes mode: {axes}")
        # What the decoder already knows about a block of image tokens is that
        # they are a 2D raster; that prior is why the flat grid can be described
        # at all. Replacing every axis discards it. 'z_only' keeps the raster
        # order driving most frequency slots and spends the rest on height,
        # which raster order cannot express, since the pooling averages a whole
        # floor column into one token.
        self.axes = axes
        self.enabled = True

    def forward(self, x, position_ids, layer_type=None):
        cos, sin = self.inner(x, position_ids, layer_type)
        if not self.enabled:
            return cos, sin
        count = self.scene.positions.shape[0]
        stop = self.scene.start + count
        # Only prefill sees the scene tokens. Every later step reads them from
        # the cache, where the rotation has already been applied.
        if position_ids.shape[1] < stop:
            return cos, sin

        inv_freq = getattr(self.inner, f"{layer_type}_inv_freq")
        scaling = getattr(self.inner, f"{layer_type}_attention_scaling")
        slots = inv_freq.shape[0]
        coordinates = index_units(
            self.scene.positions.to(device=inv_freq.device, dtype=torch.float32),
            self.span_units,
            centred=self.anchored,
        )
        if self.anchored:
            # The middle of the span the scene tokens would have occupied.
            coordinates = coordinates + (self.scene.start + count / 2.0)
        # Interleaved rather than blocked: dealing the slots round-robin gives
        # every axis the full spread of wavelengths, so each one can express
        # both a coarse side-of-the-room offset and a fine one.
        if self.axes == "xyz":
            axis = torch.arange(slots, device=inv_freq.device) % 3
            driver = coordinates[:, axis]
        else:
            sequential = torch.arange(
                self.scene.start, stop, device=inv_freq.device, dtype=torch.float32
            )
            # Every third slot carries height; the rest keep raster order.
            height = torch.arange(slots, device=inv_freq.device) % 3 == 2
            driver = sequential.unsqueeze(1).expand(count, slots).clone()
            driver[:, height] = coordinates[:, 2:3].expand(count, int(height.sum()))
        angles = driver * inv_freq.float().unsqueeze(0)
        block = torch.cat((angles, angles), dim=-1)
        new_cos = (block.cos() * scaling).to(cos.dtype)
        new_sin = (block.sin() * scaling).to(sin.dtype)
        cos = cos.clone()
        sin = sin.clone()
        cos[:, self.scene.start : stop, :] = new_cos
        sin[:, self.scene.start : stop, :] = new_sin
        return cos, sin


def scene_span_from_mask(mm_token_type_ids: torch.Tensor) -> tuple[int, int]:
    """Locate the injected image tokens in the composed sequence."""

    marked = (mm_token_type_ids[0] != 0).nonzero().flatten()
    if marked.numel() == 0:
        raise ValueError("no multimodal tokens found in the prepared batch")
    return int(marked[0]), int(marked[-1]) + 1


class attach_rope3d:  # A context manager at the call site, so it reads as a verb.
    """Temporarily give a Gemma text model 3D rotary position for one span."""

    def __init__(
        self,
        model: nn.Module,
        scene: ScenePositions,
        *,
        span_units: float = 256.0,
        anchored: bool = True,
        axes: str = "xyz",
    ):
        self.text = _text_model(model)
        self.scene = scene
        self.span_units = span_units
        self.anchored = anchored
        self.axes = axes
        self.original: nn.Module | None = None

    def __enter__(self) -> Rope3DRotary:
        self.original = self.text.rotary_emb
        patched = Rope3DRotary(
            self.original,
            self.scene,
            span_units=self.span_units,
            anchored=self.anchored,
            axes=self.axes,
        )
        self.text.rotary_emb = patched
        return patched

    def __exit__(self, *exception: object) -> None:
        if self.original is not None:
            self.text.rotary_emb = self.original


def _text_model(model: nn.Module) -> nn.Module:
    node: object = model
    for _ in range(4):
        if hasattr(node, "rotary_emb"):
            return node  # type: ignore[return-value]
        for attribute in ("model", "language_model", "text_model"):
            child = getattr(node, attribute, None)
            if child is not None:
                node = child
                break
        else:
            break
    raise ValueError("could not find the text model's rotary embedding")


__all__ = [
    "Rope3DRotary",
    "ScenePositions",
    "attach_rope3d",
    "index_units",
    "scene_span_from_mask",
]

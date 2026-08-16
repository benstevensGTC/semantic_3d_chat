"""Name discovered objects by asking Gemma what it sees.

Each anonymous proposal is projected back into the scan frames that saw it best.
Rather than cropping tightly -- which strips away the context a vision model
needs, and reliably produces answers like "Shape" -- the object is highlighted
with a box inside a generous crop that keeps the floor, walls and neighbouring
furniture visible.  Gemma is then asked what is inside the box.

Several views vote, so a single bad angle cannot decide a label.  No weights are
trained and no vocabulary is supplied: the answer is whatever Gemma says.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

NAME_PROMPT = (
    "This is a photo of a room. What is the single object outlined by the red "
    "rectangle? Answer with just the object's common name, one or two words, "
    "and nothing else."
)

# Answers that carry no information; if a view returns one of these it does not
# get a vote.
_REJECTED = frozenset(
    {
        "shape",
        "object",
        "objects",
        "thing",
        "item",
        "box",
        "rectangle",
        "red rectangle",
        "unknown",
        "nothing",
        "room",
        "image",
        "photo",
        "picture",
        "none",
        "n/a",
    }
)
_CLEAN = re.compile(r"[^a-z0-9 \-]+")


@dataclass(frozen=True)
class ViewCrop:
    frame_index: int
    box: tuple[int, int, int, int]
    visible_points: int
    mean_depth_m: float


@dataclass(frozen=True)
class NamedObject:
    proposal_id: str
    name: str
    votes: dict[str, int]
    view_count: int
    answers: tuple[str, ...]

    @property
    def confidence(self) -> float:
        if not self.view_count:
            return 0.0
        return self.votes.get(self.name, 0) / float(self.view_count)


def normalize_answer(raw: str) -> str:
    """Reduce a free-form answer to a short, comparable noun phrase."""

    text = str(raw).strip().lower()
    text = text.split("\n")[0]
    # Models like to answer in a sentence; keep the tail after a copula.
    for lead in (" is a ", " is an ", " is the ", "this is a ", "this is an "):
        if lead in text:
            text = text.split(lead)[-1]
    text = _CLEAN.sub("", text).strip()
    for article in ("a ", "an ", "the "):
        text = text.removeprefix(article)
    words = text.split()
    if not words:
        return ""
    # Two words is enough for "floor lamp" or "coffee table".
    return " ".join(words[:2]).strip()


def select_views(
    proposal_points: np.ndarray,
    frames: list[dict[str, Any]],
    *,
    image_size: tuple[int, int],
    max_views: int,
    min_visible_points: int = 25,
    margin_fraction: float = 0.55,
) -> list[ViewCrop]:
    """Pick the frames that see this object largest and most completely."""

    from semantic_3d_chat.mapping.depth_projection import project_world_points_to_pixels

    width, height = image_size
    candidates: list[ViewCrop] = []
    for index, frame in enumerate(frames):
        intrinsics = np.asarray(frame["intrinsics"], dtype=np.float64)
        camera_to_world = np.asarray(frame["camera_to_world"], dtype=np.float64)
        pixels, depth = project_world_points_to_pixels(
            proposal_points, intrinsics, camera_to_world
        )
        inside = (
            np.isfinite(pixels).all(axis=1)
            & (depth > 0.15)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        visible = int(inside.sum())
        if visible < min_visible_points:
            continue
        seen = pixels[inside]
        x0, y0 = seen.min(axis=0)
        x1, y1 = seen.max(axis=0)
        # Expand the tight hull so the model keeps surrounding context.
        pad_x = max((x1 - x0) * margin_fraction, 24.0)
        pad_y = max((y1 - y0) * margin_fraction, 24.0)
        candidates.append(
            ViewCrop(
                frame_index=index,
                box=(
                    int(max(0, np.floor(x0 - pad_x))),
                    int(max(0, np.floor(y0 - pad_y))),
                    int(min(width, np.ceil(x1 + pad_x))),
                    int(min(height, np.ceil(y1 + pad_y))),
                ),
                visible_points=visible,
                mean_depth_m=float(depth[inside].mean()),
            )
        )
    candidates.sort(key=lambda item: item.visible_points, reverse=True)
    return candidates[:max_views]


def highlight(image: Any, box: tuple[int, int, int, int], crop: tuple[int, int, int, int]) -> Any:
    """Draw the object's box, then crop around it keeping context."""

    from PIL import ImageDraw

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width = max(2, int(0.006 * max(annotated.size)))
    draw.rectangle(box, outline=(255, 0, 0), width=width)
    return annotated.crop(crop)


def tight_box(
    proposal_points: np.ndarray,
    frame: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    from semantic_3d_chat.mapping.depth_projection import project_world_points_to_pixels

    width, height = image_size
    pixels, depth = project_world_points_to_pixels(
        proposal_points,
        np.asarray(frame["intrinsics"], dtype=np.float64),
        np.asarray(frame["camera_to_world"], dtype=np.float64),
    )
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (depth > 0.15)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    seen = pixels[inside]
    x0, y0 = seen.min(axis=0)
    x1, y1 = seen.max(axis=0)
    return (int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)))


# Hue centres in degrees. Nearest-neighbour in raw RGB is dominated by
# brightness -- it calls a tan surface "pink" and a teal one "gray" -- so hue,
# saturation and value are separated first.
_HUE_NAMES: tuple[tuple[float, str], ...] = (
    (0.0, "red"),
    (25.0, "orange"),
    (55.0, "yellow"),
    (110.0, "green"),
    (175.0, "teal"),
    (225.0, "blue"),
    (280.0, "purple"),
    (325.0, "pink"),
    (360.0, "red"),
)


def color_word(rgb: tuple[float, float, float]) -> str:
    """Common colour name for a perceived mean RGB in [0, 1]."""

    import colorsys

    red, green, blue = (float(min(max(channel, 0.0), 1.0)) for channel in rgb)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    degrees = hue * 360.0
    if saturation < 0.16 or lightness < 0.08:
        if lightness < 0.22:
            return "black"
        return "gray" if lightness < 0.72 else "white"
    name = min(_HUE_NAMES, key=lambda item: abs(item[0] - degrees))[1]
    # Dark, desaturated warm hues read as brown rather than orange or red.
    if name in {"orange", "red", "yellow"} and lightness < 0.45:
        return "brown"
    if name == "orange" and saturation < 0.55 and lightness > 0.45:
        return "tan"
    return name


def disambiguate(names: list[str], colors: list[tuple[float, float, float]]) -> list[str]:
    """Make repeated names addressable without inventing information.

    Two blobs that Gemma both called "table" cannot be referred to by name.  The
    qualifier added here is the object's own perceived colour, and only when
    that actually separates them; otherwise a positional index is appended so
    every object still has a unique handle.
    """

    result = list(names)
    for name in {value for value in names if names.count(value) > 1}:
        positions = [index for index, value in enumerate(names) if value == name]
        qualified = [f"{color_word(colors[index])} {name}" for index in positions]
        if len(set(qualified)) == len(qualified):
            for index, value in zip(positions, qualified, strict=True):
                result[index] = value
        else:
            for order, index in enumerate(positions, start=1):
                result[index] = f"{name} {order}"
    return result


def vote(answers: list[str]) -> tuple[str, dict[str, int]]:
    """Majority vote over normalized answers, ignoring empty ones."""

    usable = [answer for answer in answers if answer and answer not in _REJECTED]
    if not usable:
        return "unidentified object", {}
    counts = Counter(usable)
    best = max(counts.items(), key=lambda item: (item[1], -usable.index(item[0])))
    return best[0], dict(counts)


__all__ = [
    "NAME_PROMPT",
    "NamedObject",
    "ViewCrop",
    "color_word",
    "disambiguate",
    "highlight",
    "normalize_answer",
    "select_views",
    "tight_box",
    "vote",
]

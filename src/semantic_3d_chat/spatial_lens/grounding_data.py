"""Build grounding supervision out of perception's own output.

For every scanned room the discovery stage already knows which voxels form each
object, and the naming stage already asked Gemma what that object is. Pooling
those voxels onto the bird's-eye grid turns each object into a target footprint,
and the name it was given becomes the phrase. Nothing here reads an oracle, a
human annotation, or the author's room spec.

Phrases are varied a little -- bare name, "the <name>", "a <colour> <name>" --
so the head learns to ground language rather than to memorise seven strings.
"""

from __future__ import annotations

import json

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.discover import discover_objects
from semantic_3d_chat.spatial_lens.grounding import GroundingExample
from semantic_3d_chat.spatial_lens.naming import color_word
from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.scene_tokens_3d import build_scene_tokens_3d


def _phrases(name: str, rgb: tuple[float, float, float]) -> list[str]:
    colour = color_word(rgb)
    variants = [name, f"the {name}", f"a {name}"]
    if colour not in name:
        variants.append(f"the {colour} {name}")
    return variants


def room_examples(
    room: str,
    *,
    grid: int = 16,
    min_cells: int = 1,
) -> list[GroundingExample]:
    """Every (phrase, footprint) pair a single scanned room provides."""

    root = PROJECT_ROOT / "data" / "spatial_lens" / room
    cloud = SemanticCloud.load(root / "point_cloud.npz")
    tokens = build_scene_tokens_3d(cloud, grid=grid)

    graph_path = root / "scene_graph.json"
    named: dict[str, str] = {}
    if graph_path.is_file():
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        named = {item["object_id"]: item["name"] for item in payload["objects"]}

    width, depth, _ = cloud.room_size_m
    centers = np.asarray(cloud.centers_m, dtype=np.float64)
    examples: list[GroundingExample] = []
    for proposal in discover_objects(cloud):
        phrase_root = named.get(proposal.proposal_id)
        if not phrase_root or phrase_root == "unidentified object":
            continue
        points = centers[proposal.voxel_indices]
        columns = np.clip(((points[:, 0] + width / 2) / width * grid).astype(int), 0, grid - 1)
        rows = np.clip(((points[:, 1] + depth / 2) / depth * grid).astype(int), 0, grid - 1)
        footprint = np.zeros(grid * grid, dtype=np.float32)
        np.add.at(footprint, rows * grid + columns, 1.0)
        if int((footprint > 0).sum()) < min_cells or footprint.sum() <= 0:
            continue
        footprint /= footprint.sum()
        for phrase in _phrases(phrase_root, proposal.mean_rgb):
            examples.append(
                GroundingExample(
                    room=room,
                    phrase=phrase,
                    scene=tokens.tokens,
                    target=footprint,
                    grid=grid,
                    room_size_m=cloud.room_size_m,
                )
            )
    return examples


def collect(rooms: list[str], *, grid: int = 16) -> list[GroundingExample]:
    gathered: list[GroundingExample] = []
    for room in rooms:
        if not (PROJECT_ROOT / "data" / "spatial_lens" / room / "point_cloud.npz").is_file():
            continue
        gathered.extend(room_examples(room, grid=grid))
    return gathered


def available_rooms() -> list[str]:
    root = PROJECT_ROOT / "data" / "spatial_lens"
    if not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if (path / "point_cloud.npz").is_file() and (path / "scene_graph.json").is_file()
    )


def embed_phrases(language: object, phrases: list[str]) -> np.ndarray:
    """Mean-pooled Gemma input embeddings for each phrase, frozen."""

    import torch

    tokenizer = language.tokenizer  # type: ignore[attr-defined]
    embeddings = language.model.get_input_embeddings()  # type: ignore[attr-defined]
    device = next(embeddings.parameters()).device
    out = np.zeros((len(phrases), embeddings.weight.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for index, phrase in enumerate(phrases):
            ids = torch.tensor(
                [tokenizer.encode(phrase, add_special_tokens=False)],
                dtype=torch.long, device=device,
            )
            if ids.shape[1] == 0:
                continue
            out[index] = embeddings(ids)[0].float().mean(0).cpu().numpy()
    return out


def embed_phrase_tokens(
    language: object, phrases: list[str], max_tokens: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Per-token Gemma embeddings, padded, with a mask.

    Mean-pooling is fine for "the blue cabinet", where every word narrows the
    same thing down. It is not fine for "the object nearest the blue cabinet":
    averaging blends the relation into the thing it relates to, and what comes
    out cannot express *nearest(cabinet)* at all. A model asked to ground a
    compositional phrase from that vector is being asked to recover structure
    that was destroyed before it was called.
    """

    import torch

    tokenizer = language.tokenizer  # type: ignore[attr-defined]
    embeddings = language.model.get_input_embeddings()  # type: ignore[attr-defined]
    device = next(embeddings.parameters()).device
    width = embeddings.weight.shape[1]
    out = np.zeros((len(phrases), max_tokens, width), dtype=np.float32)
    mask = np.zeros((len(phrases), max_tokens), dtype=bool)
    with torch.no_grad():
        for index, phrase in enumerate(phrases):
            ids = tokenizer.encode(phrase, add_special_tokens=False)[:max_tokens]
            if not ids:
                continue
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            out[index, : len(ids)] = embeddings(tensor)[0].float().cpu().numpy()
            mask[index, : len(ids)] = True
    return out, mask


__all__ = [
    "available_rooms",
    "collect",
    "embed_phrase_tokens",
    "embed_phrases",
    "room_examples",
]

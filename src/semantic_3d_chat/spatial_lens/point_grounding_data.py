"""Turn scanned rooms into point tokens with 3D positions and phrase targets.

The grid builder pooled the cloud into cells before the model ever saw it, which
is where position stopped being a coordinate and became an index. Here the cloud
stays a cloud: points keep their metres, and the only reduction is a coarse
voxel downsample so a room fits in a fixed token budget without losing coverage.

Supervision comes from perception, as before -- discovery knows which voxels
form an object, naming knows what Gemma called it -- so a target is simply the
subset of sampled points that fall inside the object.
"""

from __future__ import annotations

import json

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.discover import discover_objects
from semantic_3d_chat.spatial_lens.naming import color_word
from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.point_grounding import PointExample


def _phrases(name: str, rgb: tuple[float, float, float]) -> list[str]:
    colour = color_word(rgb)
    variants = [name, f"the {name}", f"a {name}"]
    if colour not in name:
        variants.append(f"the {colour} {name}")
    return variants


def downsample(
    cloud: SemanticCloud, *, token_budget: int, cell_m: float, seed: int
) -> np.ndarray:
    """Indices of a spatially even subset of the cloud, at most token_budget.

    One representative per coarse voxel keeps thin objects from being drowned by
    the floor, which plain uniform sampling would do: the floor has far more
    points but occupies no more space than the furniture standing on it.
    """

    centers = np.asarray(cloud.centers_m, dtype=np.float64)
    keys = np.floor(centers / float(cell_m)).astype(np.int64)
    _, representative = np.unique(keys, axis=0, return_index=True)
    representative = np.sort(representative)
    if representative.size <= token_budget:
        return representative
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(representative, size=token_budget, replace=False))


def room_examples(
    room: str,
    *,
    token_budget: int = 1024,
    cell_m: float = 0.14,
    min_points: int = 3,
    seed: int = 0,
) -> list[PointExample]:
    """Every (phrase, point-set) pair one scanned room provides."""

    root = PROJECT_ROOT / "data" / "spatial_lens" / room
    cloud = SemanticCloud.load(root / "point_cloud.npz")
    chosen = downsample(cloud, token_budget=token_budget, cell_m=cell_m, seed=seed)
    points = np.asarray(cloud.centers_m, dtype=np.float32)[chosen]
    features = np.asarray(cloud.features, dtype=np.float32)[chosen]

    graph_path = root / "scene_graph.json"
    named: dict[str, str] = {}
    if graph_path.is_file():
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        named = {item["object_id"]: item["name"] for item in payload["objects"]}

    membership = np.full(len(cloud), -1, dtype=np.int64)
    examples: list[PointExample] = []
    for index, proposal in enumerate(discover_objects(cloud)):
        membership[proposal.voxel_indices] = index
        phrase_root = named.get(proposal.proposal_id)
        if not phrase_root or phrase_root == "unidentified object":
            continue
        inside = membership[chosen] == index
        if int(inside.sum()) < min_points:
            continue
        target = inside.astype(np.float32)
        target /= target.sum()
        footprint = np.asarray(cloud.centers_m, dtype=np.float32)[proposal.voxel_indices]
        for phrase in _phrases(phrase_root, proposal.mean_rgb):
            examples.append(
                PointExample(
                    room=room,
                    phrase=phrase,
                    points=points,
                    features=features,
                    target=target,
                    room_size_m=cloud.room_size_m,
                    footprint=footprint,
                )
            )
    return examples


def relational_examples(
    room: str,
    *,
    token_budget: int = 1024,
    cell_m: float = 0.14,
    min_points: int = 3,
    min_margin_m: float = 0.8,
    seed: int = 0,
) -> list[PointExample]:
    """Phrases that name an object by where it is relative to another one.

    "The chair" can be answered by semantics alone: find the points that look
    like a chair. "The chair nearest the bookshelf" cannot -- it needs the
    distance between two objects, which exists only if the model can relate two
    positions. These are the queries that separate a semantic point cloud from a
    spatial one, and like every other label here they are read off perception's
    own output rather than from an oracle.

    A pair is only used when the two candidates differ clearly in distance, so a
    near-tie is never scored as though it had a right answer.
    """

    root = PROJECT_ROOT / "data" / "spatial_lens" / room
    cloud = SemanticCloud.load(root / "point_cloud.npz")
    chosen = downsample(cloud, token_budget=token_budget, cell_m=cell_m, seed=seed)
    points = np.asarray(cloud.centers_m, dtype=np.float32)[chosen]
    features = np.asarray(cloud.features, dtype=np.float32)[chosen]

    graph_path = root / "scene_graph.json"
    named: dict[str, str] = {}
    if graph_path.is_file():
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        named = {item["object_id"]: item["name"] for item in payload["objects"]}

    centers = np.asarray(cloud.centers_m, dtype=np.float64)
    found = []
    for index, proposal in enumerate(discover_objects(cloud)):
        name = named.get(proposal.proposal_id)
        if not name or name == "unidentified object":
            continue
        inside = np.zeros(len(cloud), dtype=bool)
        inside[proposal.voxel_indices] = True
        picked = inside[chosen]
        if int(picked.sum()) < min_points:
            continue
        found.append((
            index,
            name,
            centers[proposal.voxel_indices].mean(axis=0),
            picked,
            np.asarray(cloud.centers_m, dtype=np.float32)[proposal.voxel_indices],
        ))

    examples: list[PointExample] = []
    for anchor_index, anchor_name, anchor_mid, _, _ in found:
        others = [item for item in found if item[0] != anchor_index]
        if len(others) < 2:
            continue
        gaps = sorted(
            (float(np.linalg.norm(mid[:2] - anchor_mid[:2])), name, picked, voxels)
            for _, name, mid, picked, voxels in others
        )
        near_gap, near_name, near_mask, near_voxels = gaps[0]
        far_gap, far_name, far_mask, far_voxels = gaps[-1]
        if far_gap - near_gap < min_margin_m:
            continue
        for phrase, mask, voxels in (
            (f"the {near_name} nearest the {anchor_name}", near_mask, near_voxels),
            (f"the {far_name} furthest from the {anchor_name}", far_mask, far_voxels),
        ):
            target = mask.astype(np.float32)
            target /= target.sum()
            examples.append(
                PointExample(
                    room=room,
                    phrase=phrase,
                    points=points,
                    features=features,
                    target=target,
                    room_size_m=cloud.room_size_m,
                    footprint=voxels,
                )
            )
    return examples


def collect(rooms: list[str], **kwargs: object) -> list[PointExample]:
    gathered: list[PointExample] = []
    for room in rooms:
        if not (PROJECT_ROOT / "data" / "spatial_lens" / room / "point_cloud.npz").is_file():
            continue
        gathered.extend(room_examples(room, **kwargs))  # type: ignore[arg-type]
    return gathered


def collect_relational(rooms: list[str], **kwargs: object) -> list[PointExample]:
    gathered: list[PointExample] = []
    for room in rooms:
        if not (PROJECT_ROOT / "data" / "spatial_lens" / room / "point_cloud.npz").is_file():
            continue
        gathered.extend(relational_examples(room, **kwargs))  # type: ignore[arg-type]
    return gathered


__all__ = [
    "collect",
    "collect_relational",
    "downsample",
    "relational_examples",
    "room_examples",
]

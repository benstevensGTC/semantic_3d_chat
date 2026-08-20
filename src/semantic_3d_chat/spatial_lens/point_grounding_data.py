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
import re

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
    colours = np.asarray(cloud.rgb, dtype=np.float32)[chosen]

    graph_path = root / "scene_graph.json"
    named: dict[str, str] = {}
    if graph_path.is_file():
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        named = {item["object_id"]: item["name"] for item in payload["objects"]}

    membership = np.full(len(cloud), -1, dtype=np.int64)
    eligible: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for index, proposal in enumerate(discover_objects(cloud)):
        membership[proposal.voxel_indices] = index
        phrase_root = named.get(proposal.proposal_id)
        if not phrase_root or phrase_root == "unidentified object":
            continue
        inside = membership[chosen] == index
        if int(inside.sum()) < min_points:
            continue
        eligible.append((
            index,
            phrase_root,
            proposal.mean_rgb,
            inside,
            np.asarray(cloud.centers_m, dtype=np.float32)[proposal.voxel_indices],
        ))

    # Which points lie on any object at all. A guesser that knows only "the
    # answer is an object, not floor or wall" already does far better than a
    # uniformly random point, and that is the null worth reporting.
    union = np.zeros(chosen.shape[0], dtype=bool)
    for _, _, _, inside, _ in eligible:
        union |= inside

    examples: list[PointExample] = []
    for _, phrase_root, mean_rgb, inside, footprint in eligible:
        target = inside.astype(np.float32)
        target /= target.sum()
        for phrase in _phrases(phrase_root, mean_rgb):
            examples.append(
                PointExample(
                    room=room,
                    phrase=phrase,
                    points=points,
                    features=features,
                    target=target,
                    room_size_m=cloud.room_size_m,
                    rgb=colours,
                    candidates=union,
                    candidate_count=len(eligible),
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
    min_margin_m: float = 0.5,
    seed: int = 0,
) -> list[PointExample]:
    """Phrases that identify an object purely by where it is.

    "The chair" is answerable by semantics alone: find the points that look like
    a chair. So is "the chair nearest the bookshelf", as long as the room holds
    only one chair -- and none of these rooms repeats a name, so naming the
    target would have handed the answer over and measured nothing.

    The target is therefore left unnamed. "The object nearest the bookshelf"
    can only be resolved by finding the bookshelf, measuring to everything else,
    and comparing -- which needs the displacement between two positions and
    cannot be reached from semantics at all. The anchor is still named, because
    something has to be found by meaning before anything can be measured from it.

    A label is only emitted when the runner-up is at least ``min_margin_m``
    further away than the answer, and when no object in the cloud at all --
    including the ones too small or too anonymous to be a target -- contradicts
    the phrase. Checking the spread between nearest and furthest instead, as an
    earlier version did, measures how big the room is and lets an exact tie be
    scored as though it had a unique right answer; two thirds of the labels it
    produced had a runner-up inside the margin it claimed to enforce.
    """

    root = PROJECT_ROOT / "data" / "spatial_lens" / room
    cloud = SemanticCloud.load(root / "point_cloud.npz")
    chosen = downsample(cloud, token_budget=token_budget, cell_m=cell_m, seed=seed)
    points = np.asarray(cloud.centers_m, dtype=np.float32)[chosen]
    features = np.asarray(cloud.features, dtype=np.float32)[chosen]
    colours = np.asarray(cloud.rgb, dtype=np.float32)[chosen]

    graph_path = root / "scene_graph.json"
    named: dict[str, str] = {}
    if graph_path.is_file():
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        named = {item["object_id"]: item["name"] for item in payload["objects"]}

    centers = np.asarray(cloud.centers_m, dtype=np.float64)
    # Every proposal, including the ones that cannot be a target. They are still
    # physically in the cloud the model is scored on, so "the object nearest the
    # shelf" is a false statement if one of them is nearer than the answer.
    every: list[np.ndarray] = []
    found = []
    for index, proposal in enumerate(discover_objects(cloud)):
        every.append(centers[proposal.voxel_indices].mean(axis=0))
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
            (float(np.linalg.norm(mid[:2] - anchor_mid[:2])), index, picked, voxels)
            for index, _, mid, picked, voxels in others
        )
        # Distance from the anchor to every proposal in the room, eligible to be
        # a target or not, so the phrase can be checked against the whole cloud.
        world = sorted(
            float(np.linalg.norm(mid[:2] - anchor_mid[:2]))
            for index, mid in enumerate(every)
            if index != anchor_index
        )
        union = np.zeros(chosen.shape[0], dtype=bool)
        for _, _, picked, _ in gaps:
            union |= picked

        wording: list[tuple[str, np.ndarray, np.ndarray]] = []
        # The margin that decides "nearest" is the one to the runner-up, and
        # the runner-up is whatever is actually second closest in the cloud --
        # not the second closest thing that happened to be nameable. Measuring
        # it against the eligible candidates alone lets an unnamed object sit
        # between the answer and the runner-up and go uncounted, which leaves a
        # label that is ambiguous in the room even though the filter passed.
        if (
            gaps[0][0] <= world[0] + 1e-6
            and len(world) > 1
            and world[1] - world[0] >= min_margin_m
        ):
            _, _, near_mask, near_voxels = gaps[0]
            wording += [
                (f"the object nearest the {anchor_name}", near_mask, near_voxels),
                (f"the object closest to the {anchor_name}", near_mask, near_voxels),
            ]
        if (
            gaps[-1][0] >= world[-1] - 1e-6
            and len(world) > 1
            and world[-1] - world[-2] >= min_margin_m
        ):
            _, _, far_mask, far_voxels = gaps[-1]
            wording += [
                (f"the object furthest from the {anchor_name}", far_mask, far_voxels),
                (f"the object farthest from the {anchor_name}", far_mask, far_voxels),
            ]
        for phrase, mask, voxels in wording:
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
                    rgb=colours,
                    candidates=union,
                    candidate_count=len(others),
                    footprint=voxels,
                )
            )
    return examples


def disambiguation_examples(
    room: str,
    *,
    token_budget: int = 1024,
    cell_m: float = 0.14,
    min_points: int = 3,
    min_margin_m: float = 0.5,
    seed: int = 0,
) -> list[PointExample]:
    """"The cabinet nearest the sofa", in rooms holding more than one cabinet.

    This is the form the referring-expression literature uses, and the primitive
    corpus could not pose it: no room there held two of anything, so the category
    alone always identified the target and the relation was decoration.

    It is a sharper test than the unnamed version. Semantics narrows the field to
    the cabinets and can go no further; only the distance from the sofa decides
    which one is meant. A model that ignores position scores one in k however
    good its features are, and k is usually two, so the baseline is 50% rather
    than something a bag of semantics can drift above.

    Both the category and the anchor come from what Gemma called things. Nothing
    here reads the composer's own labels.
    """

    root = PROJECT_ROOT / "data" / "spatial_lens" / room
    cloud = SemanticCloud.load(root / "point_cloud.npz")
    chosen = downsample(cloud, token_budget=token_budget, cell_m=cell_m, seed=seed)
    points = np.asarray(cloud.centers_m, dtype=np.float32)[chosen]
    features = np.asarray(cloud.features, dtype=np.float32)[chosen]
    colours = np.asarray(cloud.rgb, dtype=np.float32)[chosen]

    graph_path = root / "scene_graph.json"
    if not graph_path.is_file():
        return []
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    named = {item["object_id"]: item["name"] for item in payload["objects"]}

    centers = np.asarray(cloud.centers_m, dtype=np.float64)
    found: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for proposal in discover_objects(cloud):
        name = named.get(proposal.proposal_id)
        if not name or name == "unidentified object":
            continue
        inside = np.zeros(len(cloud), dtype=bool)
        inside[proposal.voxel_indices] = True
        picked = inside[chosen]
        if int(picked.sum()) < min_points:
            continue
        # "cabinet 2" is the naming stage disambiguating; the category is
        # "cabinet", and that is what a person would say.
        category = re.sub(r"\s+\d+$", "", name)
        found.append((
            category,
            name,
            centers[proposal.voxel_indices].mean(axis=0),
            picked,
            np.asarray(cloud.centers_m, dtype=np.float32)[proposal.voxel_indices],
        ))

    counts: dict[str, int] = {}
    for category, *_ in found:
        counts[category] = counts.get(category, 0) + 1
    repeated = {c for c, n in counts.items() if n >= 2}
    singles = [item for item in found if counts[item[0]] == 1]
    if not repeated or not singles:
        return []

    examples: list[PointExample] = []
    for category in sorted(repeated):
        members = [item for item in found if item[0] == category]
        union = np.zeros(chosen.shape[0], dtype=bool)
        for _c, _n, _mid, picked, _v in members:
            union |= picked
        for _ac, anchor_name, anchor_mid, _ap, _av in singles:
            if anchor_name == category:
                continue
            ranked = sorted(
                (float(np.linalg.norm(mid[:2] - anchor_mid[:2])), index)
                for index, (_c, _n, mid, _p, _v) in enumerate(members)
            )
            near_gap, near_index = ranked[0]
            far_gap, far_index = ranked[-1]
            if far_gap - near_gap < min_margin_m:
                continue
            for phrase, index in (
                (f"the {category} nearest the {anchor_name}", near_index),
                (f"the {category} closest to the {anchor_name}", near_index),
                (f"the {category} furthest from the {anchor_name}", far_index),
            ):
                _c, _n, _mid, picked, voxels = members[index]
                target = picked.astype(np.float32)
                target /= target.sum()
                examples.append(
                    PointExample(
                        room=room,
                        phrase=phrase,
                        points=points,
                        features=features,
                        target=target,
                        room_size_m=cloud.room_size_m,
                        rgb=colours,
                        candidates=union,
                        candidate_count=len(members),
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


def collect_disambiguation(rooms: list[str], **kwargs: object) -> list[PointExample]:
    gathered: list[PointExample] = []
    for room in rooms:
        if not (PROJECT_ROOT / "data" / "spatial_lens" / room / "point_cloud.npz").is_file():
            continue
        gathered.extend(disambiguation_examples(room, **kwargs))  # type: ignore[arg-type]
    return gathered


__all__ = [
    "collect",
    "collect_disambiguation",
    "collect_relational",
    "disambiguation_examples",
    "downsample",
    "relational_examples",
    "room_examples",
]

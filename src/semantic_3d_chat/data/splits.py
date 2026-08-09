from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Mapping


def _atomic_scene_groups(
    scene_ids: list[str], scene_groups: Mapping[str, str] | None
) -> list[list[str]]:
    """Return indivisible scene groups, using singleton groups when unspecified."""

    known = set(scene_ids)
    groups = dict(scene_groups or {})
    unknown = set(groups) - known
    if unknown:
        raise ValueError(f"Scene groups reference unknown scenes: {sorted(unknown)}")
    if any(not isinstance(group_id, str) or not group_id for group_id in groups.values()):
        raise ValueError("Scene group IDs must be non-empty strings")

    members: dict[str, list[str]] = defaultdict(list)
    for scene_id in scene_ids:
        # A namespace prefix prevents a user-supplied group ID from colliding
        # with the implicit singleton identifier.
        group_id = groups.get(scene_id)
        key = f"group:{group_id}" if group_id is not None else f"scene:{scene_id}"
        members[key].append(scene_id)
    return [sorted(members[key]) for key in sorted(members)]


def scene_level_splits(
    scene_ids: list[str],
    seed: int,
    scene_groups: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Create deterministic scene-disjoint splits without dividing atomic groups.

    ``scene_groups`` maps a scene to an opaque grouping key. Counterfactual pair
    members use the same key, so both variants always land in the same split.
    Scenes absent from the mapping remain independent singleton groups.
    """

    unique = sorted(set(scene_ids))
    if len(unique) != len(scene_ids):
        raise ValueError("scene_ids must be unique")
    count = len(unique)
    if count < 3:
        return {"train": unique, "validation": [], "test": []}

    atomic_groups = _atomic_scene_groups(unique, scene_groups)
    rng = random.Random(seed)
    rng.shuffle(atomic_groups)
    validation_count = max(1, round(count * 0.15))
    test_count = max(1, round(count * 0.20))
    if validation_count + test_count >= count:
        validation_count = 1
        test_count = 1
    train_count = count - validation_count - test_count
    targets = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    splits = {name: [] for name in targets}
    split_priority = {"train": 2, "validation": 1, "test": 0}
    for group in atomic_groups:
        # Assign the next indivisible group to the split with the largest
        # relative deficit. This reaches the ordinary 70/15/20 allocation for
        # singleton scenes and degrades predictably when large groups make the
        # exact target impossible.
        destination = max(
            targets,
            key=lambda name: (
                (targets[name] - len(splits[name])) / max(targets[name], 1),
                -len(splits[name]),
                split_priority[name],
            ),
        )
        splits[destination].extend(group)
    for values in splits.values():
        values.sort()
    assert_scene_disjoint(splits)
    assert_group_disjoint(splits, scene_groups or {})
    return splits


def split_fingerprint(splits: dict[str, list[str]]) -> str:
    lines = [f"{split}:{scene}" for split in sorted(splits) for scene in splits[split]]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def assert_scene_disjoint(splits: dict[str, list[str]]) -> None:
    seen: set[str] = set()
    for name, scene_ids in splits.items():
        overlap = seen & set(scene_ids)
        if overlap:
            raise ValueError(f"Scene leakage into {name}: {sorted(overlap)}")
        seen.update(scene_ids)


def assert_group_disjoint(
    splits: Mapping[str, list[str]], scene_groups: Mapping[str, str]
) -> None:
    """Raise when members of one atomic group appear in different splits."""

    group_locations: dict[str, str] = {}
    for split_name, scene_ids in splits.items():
        for scene_id in scene_ids:
            group_id = scene_groups.get(scene_id)
            if group_id is None:
                continue
            previous = group_locations.setdefault(group_id, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Atomic scene group {group_id!r} crosses {previous} and {split_name}"
                )

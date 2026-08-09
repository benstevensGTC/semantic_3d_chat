from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.data.splits import (
    assert_group_disjoint,
    assert_scene_disjoint,
    scene_level_splits,
    split_fingerprint,
)

DISTRACTOR_CATEGORIES = ("sofa", "television", "bed", "sink")
RELATION_QUESTIONS = {
    "left_of": ("Is the {a} left or right of the {b}?", "left"),
    "right_of": ("Is the {a} left or right of the {b}?", "right"),
    "above": ("Is the {a} above or below the {b}?", "above"),
    "below": ("Is the {a} above or below the {b}?", "below"),
    "in_front_of": ("Is the {a} in front of or behind the {b}?", "in front"),
    "behind": ("Is the {a} in front of or behind the {b}?", "behind"),
}


def _record(
    scene_id: str,
    question: str,
    answer: str,
    answer_type: str,
    target: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "scene_id": scene_id,
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "target_xyz": None if target is None else target["expected_center_xyz_m"],
        "target_instance": None if target is None else target["instance_id"],
    }
    result.update(extra)
    return result


def generate_scene_questions(
    oracle: dict[str, Any],
    seed: int,
    *,
    category_universe: set[str] | None = None,
) -> list[dict[str, Any]]:
    scene_id = oracle["scene_id"]
    instances = oracle["instances"]
    objects = [item for item in instances if item["kind"] == "object"]
    by_id = {item["instance_id"]: item for item in instances}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        by_category[item["category"]].append(item)
    records: list[dict[str, Any]] = []

    for category, members in sorted(by_category.items()):
        records.append(_record(scene_id, f"Is there a {category} in the room?", "yes", "presence"))
        records.append(_record(scene_id, f"Can you find a {category}?", "yes", "presence"))
        records.append(
            _record(
                scene_id,
                f"How many {category}s are present?",
                str(len(members)),
                "count",
                count=len(members),
            )
        )
        if len(members) == 1 and members[0]["color"]["name"] != "neutral":
            member = members[0]
            color = member["color"]["name"]
            records.append(
                _record(scene_id, f"What color is the {category}?", color, "attribute", member)
            )
            records.append(
                _record(scene_id, f"Tell me the {category}'s color.", color, "attribute", member)
            )

    absent_categories = set(DISTRACTOR_CATEGORIES)
    if category_universe is not None:
        absent_categories.update(category_universe)
    for category in sorted(absent_categories):
        if category not in by_category:
            records.append(
                _record(scene_id, f"Is there a {category} in the room?", "no", "presence")
            )
            # Pair-derived categories get matching questions on an
            # object-removal side, making the changed fact directly scoreable.
            if category_universe is not None and category in category_universe:
                records.append(_record(scene_id, f"Can you find a {category}?", "no", "presence"))
                records.append(
                    _record(
                        scene_id,
                        f"How many {category}s are present?",
                        "0",
                        "count",
                        count=0,
                    )
                )

    # One canonical directional question per unordered pair avoids contradictory duplicates.
    seen_pairs: set[tuple[str, str, str]] = set()
    for relation in oracle.get("relationships", []):
        predicate = relation["predicate"]
        if predicate not in RELATION_QUESTIONS:
            continue
        subject = by_id[relation["subject_instance_id"]]
        object_ = by_id[relation["object_instance_id"]]
        if subject["kind"] != "object" or object_["kind"] != "object":
            continue
        if len(by_category[subject["category"]]) != 1 or len(by_category[object_["category"]]) != 1:
            continue
        axis = {
            "left_of": "horizontal",
            "right_of": "horizontal",
            "above": "vertical",
            "below": "vertical",
            "in_front_of": "depth",
            "behind": "depth",
        }[predicate]
        key = tuple(sorted((subject["instance_id"], object_["instance_id"]))) + (axis,)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        template, answer = RELATION_QUESTIONS[predicate]
        records.append(
            _record(
                scene_id,
                template.format(a=subject["category"], b=object_["category"]),
                answer,
                "spatial_relation",
                subject,
                reference_instance=object_["instance_id"],
                reference_xyz=list(object_["expected_center_xyz_m"]),
                predicate=predicate,
            )
        )

    supported: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        if item.get("support_surface"):
            supported[item["support_surface"]].append(item)
    for support_id, members in supported.items():
        support = by_id[support_id]
        if len([item for item in instances if item["category"] == support["category"]]) != 1:
            continue
        names = sorted(item["category"] for item in members)
        answer = ", ".join(names)
        records.append(
            _record(
                scene_id,
                f"What is on the {support['category']}?",
                answer,
                "support",
                members[0] if len(members) == 1 else support,
                answer_items=names,
            )
        )

    camera = oracle.get("scan_origin_xyz_m", [0.0, 0.0, 1.4])
    if objects:

        def distance(item: dict[str, Any]) -> float:
            return math.dist(item["expected_center_xyz_m"], camera)

        nearest = min(objects, key=distance)
        if len(by_category[nearest["category"]]) == 1:
            records.append(
                _record(
                    scene_id,
                    "Which object is closest to the camera?",
                    nearest["category"],
                    "metric",
                    nearest,
                    distance_m=distance(nearest),
                )
            )

    rng = random.Random(seed)
    rng.shuffle(records)
    for index, record in enumerate(records):
        record["question_id"] = f"q_{index:06d}"
    return records


def _counterfactual_metadata(oracle: Mapping[str, Any]) -> Mapping[str, Any] | None:
    generation = oracle.get("generation")
    if not isinstance(generation, Mapping):
        return None
    pair = generation.get("counterfactual_pair")
    return pair if isinstance(pair, Mapping) else None


def counterfactual_scene_groups(oracles: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Extract and validate oracle-only counterfactual grouping metadata."""

    groups: dict[str, str] = {}
    pair_members: dict[str, list[str]] = defaultdict(list)
    for scene_id, oracle in oracles.items():
        pair = _counterfactual_metadata(oracle)
        if pair is None:
            continue
        pair_id = pair.get("pair_id")
        paired_scene_id = pair.get("paired_scene_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{scene_id} has an invalid counterfactual pair_id")
        if not isinstance(paired_scene_id, str) or paired_scene_id not in oracles:
            raise ValueError(f"{scene_id} references unavailable pair member {paired_scene_id!r}")
        groups[scene_id] = pair_id
        pair_members[pair_id].append(scene_id)

    for pair_id, members in pair_members.items():
        if len(members) != 2:
            raise ValueError(f"{pair_id} must have exactly two available scene members")
        first, second = sorted(members)
        first_pair = _counterfactual_metadata(oracles[first])
        second_pair = _counterfactual_metadata(oracles[second])
        if first_pair is None or second_pair is None:
            raise AssertionError("pair metadata disappeared during validation")
        if (
            first_pair.get("paired_scene_id") != second
            or second_pair.get("paired_scene_id") != first
        ):
            raise ValueError(f"{pair_id} scene references must be reciprocal")
        if first_pair.get("change_type") != second_pair.get("change_type"):
            raise ValueError(f"{pair_id} members disagree on change_type")
        if {first_pair.get("role"), second_pair.get("role")} != {
            "reference",
            "counterfactual",
        }:
            raise ValueError(f"{pair_id} must contain reference and counterfactual roles")
    return groups


def _canonical_question_key(record: Mapping[str, Any]) -> tuple[str, str]:
    question = " ".join(str(record["question"]).casefold().split())
    return str(record["answer_type"]), question


def _canonical_target(record: Mapping[str, Any]) -> str:
    items = record.get("answer_items")
    if isinstance(items, list):
        return json.dumps(sorted(str(item).casefold() for item in items), separators=(",", ":"))
    return " ".join(str(record["answer"]).casefold().split())


def annotate_counterfactual_questions(
    records_by_scene: Mapping[str, list[dict[str, Any]]],
    oracles: Mapping[str, Mapping[str, Any]],
) -> int:
    """Annotate identical paired questions with stable keys and change truth.

    Pair metadata is attached only to a question that exists exactly once in
    both scenes. This prevents evaluation from treating unpaired, scene-specific
    questions as malformed counterfactual examples.
    """

    annotated = 0
    processed_pairs: set[str] = set()
    for scene_id in sorted(oracles):
        pair = _counterfactual_metadata(oracles[scene_id])
        if pair is None:
            continue
        pair_id = str(pair["pair_id"])
        if pair_id in processed_pairs:
            continue
        paired_scene_id = str(pair["paired_scene_id"])
        processed_pairs.add(pair_id)
        first_records = records_by_scene[scene_id]
        second_records = records_by_scene[paired_scene_id]
        first_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        second_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in first_records:
            first_index[_canonical_question_key(record)].append(record)
        for record in second_records:
            second_index[_canonical_question_key(record)].append(record)

        for canonical_key in sorted(first_index.keys() & second_index.keys()):
            first_matches = first_index[canonical_key]
            second_matches = second_index[canonical_key]
            if len(first_matches) != 1 or len(second_matches) != 1:
                continue
            digest_input = f"{pair_id}\0{canonical_key[0]}\0{canonical_key[1]}".encode()
            question_key = f"cfq_{hashlib.sha256(digest_input).hexdigest()[:16]}"
            expected_change = _canonical_target(first_matches[0]) != _canonical_target(
                second_matches[0]
            )
            for current_scene_id, other_scene_id, record in (
                (scene_id, paired_scene_id, first_matches[0]),
                (paired_scene_id, scene_id, second_matches[0]),
            ):
                current_pair = _counterfactual_metadata(oracles[current_scene_id])
                if current_pair is None:
                    raise AssertionError("pair metadata disappeared during annotation")
                record.update(
                    {
                        "counterfactual_pair_id": pair_id,
                        "counterfactual_paired_scene_id": other_scene_id,
                        "counterfactual_change_type": current_pair["change_type"],
                        "counterfactual_role": current_pair["role"],
                        "counterfactual_question_key": question_key,
                        "counterfactual_expected_change": expected_change,
                    }
                )
                annotated += 1
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scene", action="append", dest="scenes")
    args = parser.parse_args()
    config = load_config(args.config)
    oracle_root = PROJECT_ROOT / config["paths"]["data_root"] / "oracle"
    scene_ids = args.scenes or sorted(path.name for path in oracle_root.glob("scene_*"))
    if not scene_ids:
        raise SystemExit("No oracle scenes found; render a scene first")
    oracles = {
        scene_id: json.loads((oracle_root / scene_id / "oracle.json").read_text(encoding="utf-8"))
        for scene_id in scene_ids
    }
    scene_groups = counterfactual_scene_groups(oracles)
    splits = scene_level_splits(scene_ids, int(config["seed"]), scene_groups)
    assert_scene_disjoint(splits)
    assert_group_disjoint(splits, scene_groups)
    qa_root = PROJECT_ROOT / config["paths"]["data_root"] / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter()
    category_universes: dict[str, set[str]] = {}
    for scene_id, pair_id in scene_groups.items():
        members = sorted(member for member, group in scene_groups.items() if group == pair_id)
        category_universes[scene_id] = {
            str(instance["category"])
            for member in members
            for instance in oracles[member]["instances"]
            if instance["kind"] == "object"
        }

    records_by_scene: dict[str, list[dict[str, Any]]] = {}
    for scene_id in sorted(scene_ids):
        scene_seed = int(config["seed"]) + int(scene_id.removeprefix("scene_"))
        records_by_scene[scene_id] = generate_scene_questions(
            oracles[scene_id],
            scene_seed,
            category_universe=category_universes.get(scene_id),
        )
    annotated_records = annotate_counterfactual_questions(records_by_scene, oracles)
    for split_name, split_scenes in splits.items():
        output = qa_root / f"{split_name}.jsonl"
        lines = []
        for scene_id in split_scenes:
            records = records_by_scene[scene_id]
            lines.extend(json.dumps(item, sort_keys=True) for item in records)
            totals[split_name] += len(records)
        output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "seed": config["seed"],
        "splits": splits,
        "counterfactual_scene_groups": scene_groups,
        "counterfactual_annotated_records": annotated_records,
        "fingerprint": split_fingerprint(splits),
        "question_counts": dict(totals),
    }
    (qa_root / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

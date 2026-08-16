from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config
from semantic_3d_chat.data.scene_variants import (
    batch_scene_plans,
    batch_scene_splits,
    validate_visibility_evidence,
)
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
VISIBILITY_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "scene_id",
        "method",
        "minimum_visible_pixels",
        "expected_instance_ids",
        "visible_pixel_counts",
        "all_required_visible",
    }
)
VIEWPOINT_CONVENTION = "x_right_y_forward_z_up_yaw_0"


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


def _visibility_for_qa(
    scene_id: str,
    objects: list[dict[str, Any]],
    evidence: Mapping[str, Any] | None,
) -> tuple[dict[str, int], int] | None:
    """Validate exact ray-hit evidence while permitting genuinely hidden objects.

    ``validate_visibility_evidence`` intentionally fails when any object is
    hidden because it protects the original all-visible dataset contract. QA
    uncertainty examples need the complementary case, so this offline-only
    path applies the same schema checks but accepts a truthful
    ``all_required_visible=false`` result.
    """

    if evidence is None:
        return None
    if not isinstance(evidence, Mapping):
        raise TypeError("visibility_evidence must be a mapping")
    unexpected = set(evidence) - VISIBILITY_EVIDENCE_KEYS
    if unexpected:
        raise ValueError(f"Unexpected visibility evidence keys: {sorted(unexpected)}")
    if evidence.get("schema_version") != 1:
        raise ValueError("visibility evidence schema_version must be 1")
    if evidence.get("scene_id") != scene_id:
        raise ValueError("visibility evidence scene_id does not match the oracle")
    if evidence.get("method") != "exact_depth_raycast":
        raise ValueError("visibility evidence must come from exact_depth_raycast")
    minimum = evidence.get("minimum_visible_pixels")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("minimum_visible_pixels must be a positive integer")
    expected = evidence.get("expected_instance_ids")
    counts = evidence.get("visible_pixel_counts")
    if not isinstance(expected, list) or not isinstance(counts, Mapping):
        raise TypeError("visibility evidence requires instance IDs and pixel counts")
    expected_ids = [str(value) for value in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected_instance_ids contains duplicates")
    object_ids = {str(item["instance_id"]) for item in objects}
    if set(expected_ids) != object_ids or set(counts) != object_ids:
        raise ValueError("visibility evidence must cover every oracle object exactly")
    normalized: dict[str, int] = {}
    for instance_id in expected_ids:
        count = counts[instance_id]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"Invalid visibility count for {instance_id}")
        normalized[instance_id] = count
    observed_all_visible = all(count >= minimum for count in normalized.values())
    if not isinstance(evidence.get("all_required_visible"), bool):
        raise TypeError("all_required_visible must be boolean")
    if evidence["all_required_visible"] is not observed_all_visible:
        raise ValueError("all_required_visible disagrees with measured counts")
    return normalized, minimum


def _room_reference(oracle: Mapping[str, Any]) -> tuple[list[float], float] | None:
    room = oracle.get("room")
    if not isinstance(room, Mapping):
        return None
    lower = room.get("bounds_min_m")
    upper = room.get("bounds_max_m")
    if not isinstance(lower, list) or not isinstance(upper, list):
        return None
    if len(lower) != 3 or len(upper) != 3:
        return None
    try:
        minimum = [float(value) for value in lower]
        maximum = [float(value) for value in upper]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (*minimum, *maximum)):
        return None
    width = maximum[0] - minimum[0]
    if width <= 0.0 or any(maximum[index] <= minimum[index] for index in range(1, 3)):
        return None
    center = [(minimum[index] + maximum[index]) / 2.0 for index in range(3)]
    return center, width


def _append_object_location_questions(
    records: list[dict[str, Any]],
    scene_id: str,
    oracle: Mapping[str, Any],
    by_category: Mapping[str, list[dict[str, Any]]],
    observable_ids: set[str],
) -> None:
    room_reference = _room_reference(oracle)
    if room_reference is None:
        return
    room_center, room_width = room_reference
    boundary = room_width / 6.0
    ambiguity_margin = max(0.05, room_width * 0.01)
    for category, members in sorted(by_category.items()):
        if len(members) != 1 or members[0]["instance_id"] not in observable_ids:
            continue
        item = members[0]
        offset_x = float(item["expected_center_xyz_m"][0]) - room_center[0]
        if abs(abs(offset_x) - boundary) <= ambiguity_margin:
            continue
        region = "left" if offset_x < -boundary else "right" if offset_x > boundary else "center"
        records.append(
            _record(
                scene_id,
                f"Where is the {category} relative to the room center?",
                region,
                "object_location",
                item,
                reference_instance=None,
                reference_xyz=list(room_center),
                reference_frame="room",
                room_region=region,
            )
        )


def _append_containment_questions(
    records: list[dict[str, Any]],
    scene_id: str,
    relationships: list[dict[str, Any]],
    by_id: Mapping[str, dict[str, Any]],
    by_category: Mapping[str, list[dict[str, Any]]],
    observable_ids: set[str],
) -> None:
    contained_by_container: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for relation in relationships:
        predicate = str(relation.get("predicate", ""))
        subject = by_id.get(str(relation.get("subject_instance_id", "")))
        object_ = by_id.get(str(relation.get("object_instance_id", "")))
        if subject is None or object_ is None:
            continue
        if subject.get("kind") != "object" or object_.get("kind") != "object":
            continue
        if predicate in {"inside", "inside_of", "in"}:
            member, container = subject, object_
        elif predicate == "contains":
            member, container = object_, subject
        else:
            continue
        if (
            member["instance_id"] not in observable_ids
            or container["instance_id"] not in observable_ids
        ):
            continue
        contained_by_container[container["instance_id"]][member["instance_id"]] = member

    for container_id, member_index in sorted(contained_by_container.items()):
        container = by_id[container_id]
        container_category = str(container["category"])
        if len(by_category[container_category]) != 1:
            continue
        members = sorted(member_index.values(), key=lambda item: str(item["category"]))
        names = [str(member["category"]) for member in members]
        if len(names) != len(set(names)) or any(len(by_category[name]) != 1 for name in names):
            continue
        target = members[0] if len(members) == 1 else None
        records.append(
            _record(
                scene_id,
                f"What is inside the {container_category}?",
                ", ".join(names),
                "containment",
                target,
                answer_items=names,
                target_instances=[member["instance_id"] for member in members],
                target_xyzs=[list(member["expected_center_xyz_m"]) for member in members],
                reference_instance=container_id,
                reference_xyz=list(container["expected_center_xyz_m"]),
                predicate="inside",
            )
        )
        for member in members:
            records.append(
                _record(
                    scene_id,
                    f"Is the {member['category']} inside the {container_category}?",
                    "yes",
                    "containment",
                    member,
                    answer_items=["yes"],
                    reference_instance=container_id,
                    reference_xyz=list(container["expected_center_xyz_m"]),
                    predicate="inside",
                )
            )


def _append_viewpoint_questions(
    records: list[dict[str, Any]],
    scene_id: str,
    camera: list[float],
    by_category: Mapping[str, list[dict[str, Any]]],
    observable_ids: set[str],
) -> None:
    position_margin_m = 0.25
    angular_margin_degrees = 10.0
    for category, members in sorted(by_category.items()):
        if len(members) != 1 or members[0]["instance_id"] not in observable_ids:
            continue
        item = members[0]
        center = item["expected_center_xyz_m"]
        delta_x = float(center[0]) - camera[0]
        delta_y = float(center[1]) - camera[1]
        common = {
            "reference_instance": None,
            "reference_xyz": list(camera),
            "reference_frame": "camera",
            "viewpoint_yaw_degrees": 0.0,
            "viewpoint_convention": VIEWPOINT_CONVENTION,
        }
        if abs(delta_x) >= position_margin_m:
            answer = "right" if delta_x > 0.0 else "left"
            records.append(
                _record(
                    scene_id,
                    f"Is the {category} to the left or right of the current viewpoint?",
                    answer,
                    "viewpoint_relative",
                    item,
                    predicate=f"viewpoint_{answer}",
                    **common,
                )
            )
        if abs(delta_y) >= position_margin_m:
            answer = "in front" if delta_y > 0.0 else "behind"
            records.append(
                _record(
                    scene_id,
                    f"Is the {category} in front of or behind the camera?",
                    answer,
                    "viewpoint_relative",
                    item,
                    predicate="viewpoint_in_front" if delta_y > 0.0 else "viewpoint_behind",
                    **common,
                )
            )
        if math.hypot(delta_x, delta_y) < position_margin_m:
            continue
        bearing = math.degrees(math.atan2(delta_x, delta_y))
        if abs(abs(bearing) - 180.0) <= angular_margin_degrees:
            continue
        turn_direction = (
            "straight ahead"
            if abs(bearing) <= angular_margin_degrees
            else "right"
            if bearing > 0.0
            else "left"
        )
        records.append(
            _record(
                scene_id,
                f"Which way should the camera turn to face the {category}?",
                turn_direction,
                "viewpoint_relative",
                item,
                bearing_degrees=round(bearing, 6),
                **common,
            )
        )


def _append_metric_questions(
    records: list[dict[str, Any]],
    scene_id: str,
    camera: list[float],
    by_category: Mapping[str, list[dict[str, Any]]],
    observable_ids: set[str],
) -> None:
    unique_objects = [
        members[0]
        for _category, members in sorted(by_category.items())
        if len(members) == 1 and members[0]["instance_id"] in observable_ids
    ]

    def distance(item: dict[str, Any]) -> float:
        return math.dist(item["expected_center_xyz_m"], camera)

    ranked = sorted(unique_objects, key=lambda item: (distance(item), item["instance_id"]))
    if ranked and (len(ranked) == 1 or distance(ranked[1]) - distance(ranked[0]) >= 0.25):
        nearest = ranked[0]
        records.append(
            _record(
                scene_id,
                "Which object is closest to the camera?",
                nearest["category"],
                "metric",
                nearest,
                distance_m=distance(nearest),
                metric_kind="nearest",
                reference_instance=None,
                reference_xyz=list(camera),
            )
        )

    for item in unique_objects:
        exact_distance = distance(item)
        rounded_distance = math.floor(exact_distance * 2.0 + 0.5) / 2.0
        records.append(
            _record(
                scene_id,
                f"Approximately how far is the {item['category']} from the camera?",
                f"{rounded_distance:.1f} meters",
                "metric",
                item,
                distance_m=exact_distance,
                approximate_distance_m=rounded_distance,
                tolerance_m=0.25,
                metric_kind="distance",
                reference_instance=None,
                reference_xyz=list(camera),
            )
        )

    for first_index, first in enumerate(unique_objects):
        for second in unique_objects[first_index + 1 :]:
            first_distance = distance(first)
            second_distance = distance(second)
            if abs(first_distance - second_distance) < 0.25:
                continue
            farther, nearer = (
                (first, second) if first_distance > second_distance else (second, first)
            )
            records.append(
                _record(
                    scene_id,
                    "Which is farther from the camera, "
                    f"the {first['category']} or the {second['category']}?",
                    farther["category"],
                    "metric",
                    farther,
                    reference_instance=nearer["instance_id"],
                    reference_xyz=list(nearer["expected_center_xyz_m"]),
                    candidate_instances=[first["instance_id"], second["instance_id"]],
                    candidate_distances_m=[first_distance, second_distance],
                    metric_kind="farther_comparison",
                )
            )


def _append_uncertainty_questions(
    records: list[dict[str, Any]],
    scene_id: str,
    by_category: Mapping[str, list[dict[str, Any]]],
    visibility: tuple[dict[str, int], int] | None,
) -> None:
    if visibility is None:
        return
    counts, minimum = visibility
    for category, members in sorted(by_category.items()):
        if len(members) != 1:
            continue
        item = members[0]
        count = counts[item["instance_id"]]
        sufficient = count >= minimum
        records.append(
            _record(
                scene_id,
                f"Is there enough visual evidence to locate the {category}?",
                "yes" if sufficient else "no",
                "uncertainty",
                item if sufficient else None,
                target_instance=item["instance_id"],
                target_xyz=(list(item["expected_center_xyz_m"]) if sufficient else None),
                visibility_method="exact_depth_raycast",
                visibility_pixels=count,
                minimum_visibility_pixels=minimum,
                evidence_sufficient=sufficient,
            )
        )


def generate_scene_questions(
    oracle: dict[str, Any],
    seed: int,
    *,
    category_universe: set[str] | None = None,
    visibility_evidence: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    scene_id = oracle["scene_id"]
    instances = oracle["instances"]
    objects = [item for item in instances if item["kind"] == "object"]
    by_id = {item["instance_id"]: item for item in instances}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        by_category[item["category"]].append(item)
    visibility = _visibility_for_qa(scene_id, objects, visibility_evidence)
    observable_ids = (
        {item["instance_id"] for item in objects}
        if visibility is None
        else {
            instance_id
            for instance_id, count in visibility[0].items()
            if count >= visibility[1]
        }
    )
    records: list[dict[str, Any]] = []

    for category, members in sorted(by_category.items()):
        if visibility is not None and any(
            member["instance_id"] not in observable_ids for member in members
        ):
            continue
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

    chairs = by_category.get("chair", [])
    if len(chairs) == 1 and chairs[0]["instance_id"] in observable_ids:
        rotation = chairs[0].get("pose", {}).get("rotation_euler_degrees", [0.0, 0.0, 0.0])
        upside_down = (
            isinstance(rotation, list)
            and len(rotation) == 3
            and abs(abs(float(rotation[0])) - 180.0) < 1.0
        )
        records.append(
            _record(
                scene_id,
                "Is the chair upright or upside down?",
                "upside down" if upside_down else "upright",
                "orientation",
                chairs[0],
            )
        )

    picture_frames = by_category.get("picture frame", [])
    if len(picture_frames) == 1 and picture_frames[0]["instance_id"] in observable_ids:
        picture = picture_frames[0]
        support = by_id.get(str(picture.get("support_surface")))
        if support is not None and support.get("category") in {"wall", "floor"}:
            records.append(
                _record(
                    scene_id,
                    "Is the picture frame on the wall or on the floor?",
                    str(support["category"]),
                    "support",
                    picture,
                )
            )

    books = by_category.get("book", [])
    tables = by_category.get("table", [])
    if (
        len(books) == 1
        and len(tables) == 1
        and books[0]["instance_id"] in observable_ids
        and tables[0]["instance_id"] in observable_ids
    ):
        book = books[0]
        under_table = any(
            relation.get("subject_instance_id") == book["instance_id"]
            and relation.get("predicate") == "under"
            and relation.get("object_instance_id") == tables[0]["instance_id"]
            for relation in oracle.get("relationships", [])
        )
        records.append(
            _record(
                scene_id,
                "Is the book on the table or under the table?",
                "under" if under_table else "on",
                "support",
                book,
                reference_instance=tables[0]["instance_id"],
                reference_xyz=list(tables[0]["expected_center_xyz_m"]),
            )
        )

    bowls = by_category.get("bowl", [])
    if len(bowls) == 1 and bowls[0]["instance_id"] in observable_ids:
        bowl = bowls[0]
        support = by_id.get(str(bowl.get("support_surface")))
        if support is not None and support.get("category") in {"floor", "table"}:
            records.append(
                _record(
                    scene_id,
                    "Is the bowl on the floor or on the table?",
                    str(support["category"]),
                    "support",
                    bowl,
                )
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
        if (
            subject["instance_id"] not in observable_ids
            or object_["instance_id"] not in observable_ids
        ):
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
        if visibility is not None and (
            any(member["instance_id"] not in observable_ids for member in members)
            or (support.get("kind") == "object" and support_id not in observable_ids)
        ):
            continue
        if support["category"] in {"floor", "wall", "ceiling"}:
            continue
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

    camera = [float(value) for value in oracle.get("scan_origin_xyz_m", [0.0, 0.0, 1.4])]
    if len(camera) != 3 or not all(math.isfinite(value) for value in camera):
        raise ValueError("scan_origin_xyz_m must contain three finite coordinates")
    _append_object_location_questions(
        records,
        scene_id,
        oracle,
        by_category,
        observable_ids,
    )
    _append_containment_questions(
        records,
        scene_id,
        list(oracle.get("relationships", [])),
        by_id,
        by_category,
        observable_ids,
    )
    _append_viewpoint_questions(records, scene_id, camera, by_category, observable_ids)
    _append_metric_questions(records, scene_id, camera, by_category, observable_ids)
    _append_uncertainty_questions(records, scene_id, by_category, visibility)

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


def _selection_hash(seed: int, *parts: object) -> str:
    encoded = "\0".join((str(seed), *(str(part) for part in parts))).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_balanced_split_records(
    records_by_scene: Mapping[str, list[dict[str, Any]]],
    scene_ids: list[str],
    *,
    per_scene_limit: int,
    seed: int,
    max_changed_units_per_pair: int,
) -> dict[str, list[dict[str, Any]]]:
    """Select a deterministic, answer-balanced subset without splitting changes.

    Every selected counterfactual unit contains both physical scene sides.
    Answer-changing units are selected first, up to the configured per-pair
    cap. Stable paired units are then selected atomically, and any remaining
    per-scene capacity is filled only from questions without pair metadata.
    Question text and oracle semantics do not influence split placement.
    """

    if per_scene_limit < 1:
        raise ValueError("per_scene_limit must be positive")
    if max_changed_units_per_pair < 0:
        raise ValueError("max_changed_units_per_pair must be non-negative")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("scene_ids contains duplicates")
    missing_scenes = set(scene_ids) - set(records_by_scene)
    if missing_scenes:
        raise ValueError(f"Records are unavailable for scenes: {sorted(missing_scenes)}")

    selected_keys: dict[str, set[tuple[str, str]]] = {scene_id: set() for scene_id in scene_ids}
    paired_units: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for scene_id in scene_ids:
        for record in records_by_scene[scene_id]:
            pair_fields = (
                record.get("counterfactual_pair_id"),
                record.get("counterfactual_question_key"),
                record.get("counterfactual_expected_change"),
            )
            if all(value is None for value in pair_fields):
                continue
            pair_id, question_key, expected_change = pair_fields
            if (
                not isinstance(pair_id, str)
                or not isinstance(question_key, str)
                or not isinstance(expected_change, bool)
            ):
                raise TypeError(
                    "Counterfactual records require a pair ID, stable question key, "
                    "and boolean expected-change flag"
                )
            paired_units[(pair_id, question_key)].append((scene_id, record))

    units_by_pair: dict[str, list[tuple[str, bool, list[tuple[str, dict[str, Any]]]]]] = (
        defaultdict(list)
    )
    for (pair_id, question_key), members in paired_units.items():
        member_scenes = {scene_id for scene_id, _ in members}
        if len(members) != 2 or len(member_scenes) != 2:
            raise ValueError(f"Counterfactual unit {pair_id}/{question_key} must have two sides")
        expected_flags = {bool(record["counterfactual_expected_change"]) for _, record in members}
        if len(expected_flags) != 1:
            raise ValueError(
                f"Counterfactual unit {pair_id}/{question_key} disagrees on expected change"
            )
        expected_change = expected_flags.pop()
        targets = {_canonical_target(record) for _, record in members}
        if expected_change != (len(targets) > 1):
            raise ValueError(
                f"Counterfactual unit {pair_id}/{question_key} has inconsistent targets"
            )
        units_by_pair[pair_id].append((question_key, expected_change, members))

    for pair_id, units in sorted(units_by_pair.items()):
        ranked = sorted(
            (unit for unit in units if unit[1]),
            key=lambda item: _selection_hash(seed, pair_id, item[0]),
        )
        for question_key, expected_change, members in ranked[:max_changed_units_per_pair]:
            del question_key
            assert expected_change
            for scene_id, record in members:
                record_key = (str(record["question_id"]), str(record["question"]))
                selected_keys[scene_id].add(record_key)

    type_counts_by_scene: dict[str, Counter[str]] = {}
    answer_counts_by_scene: dict[str, Counter[str]] = {}
    for scene_id in scene_ids:
        preselected = [
            record
            for record in records_by_scene[scene_id]
            if (str(record["question_id"]), str(record["question"])) in selected_keys[scene_id]
        ]
        if len(preselected) > per_scene_limit:
            raise ValueError(
                f"Changed pair units exceed the per-scene limit for {scene_id}: "
                f"{len(preselected)} > {per_scene_limit}"
            )
        type_counts_by_scene[scene_id] = Counter(
            str(record["answer_type"]) for record in preselected
        )
        answer_counts_by_scene[scene_id] = Counter(
            _canonical_target(record) for record in preselected
        )

    # Keep stable pair controls usable for consistency scoring by selecting
    # both sides together. Each pair owns two scenes, so this preserves equal
    # capacity on its two members.
    for pair_id, units in sorted(units_by_pair.items()):
        remaining_units = [unit for unit in units if not unit[1]]
        while remaining_units:
            eligible = [
                unit
                for unit in remaining_units
                if all(len(selected_keys[scene_id]) < per_scene_limit for scene_id, _ in unit[2])
            ]
            if not eligible:
                break
            best = min(
                eligible,
                key=lambda unit: (
                    sum(
                        type_counts_by_scene[scene_id][str(record["answer_type"])]
                        for scene_id, record in unit[2]
                    ),
                    sum(
                        answer_counts_by_scene[scene_id][_canonical_target(record)]
                        for scene_id, record in unit[2]
                    ),
                    _selection_hash(seed, pair_id, unit[0]),
                ),
            )
            remaining_units.remove(best)
            for scene_id, record in best[2]:
                record_key = (str(record["question_id"]), str(record["question"]))
                selected_keys[scene_id].add(record_key)
                type_counts_by_scene[scene_id][str(record["answer_type"])] += 1
                answer_counts_by_scene[scene_id][_canonical_target(record)] += 1

    selected: dict[str, list[dict[str, Any]]] = {}
    for scene_id in scene_ids:
        records = records_by_scene[scene_id]
        preselected = [
            record
            for record in records
            if (str(record["question_id"]), str(record["question"])) in selected_keys[scene_id]
        ]
        type_counts = type_counts_by_scene[scene_id]
        answer_counts = answer_counts_by_scene[scene_id]
        remaining = [
            record
            for record in records
            if (str(record["question_id"]), str(record["question"])) not in selected_keys[scene_id]
            and record.get("counterfactual_pair_id") is None
            and record.get("counterfactual_question_key") is None
            and record.get("counterfactual_expected_change") is None
        ]
        chosen = list(preselected)
        while remaining and len(chosen) < per_scene_limit:
            best = min(
                remaining,
                key=lambda record: (
                    type_counts[str(record["answer_type"])],
                    answer_counts[_canonical_target(record)],
                    _selection_hash(
                        seed,
                        scene_id,
                        record["question_id"],
                        record["question"],
                    ),
                ),
            )
            remaining.remove(best)
            chosen.append(best)
            type_counts[str(best["answer_type"])] += 1
            answer_counts[_canonical_target(best)] += 1
        selected[scene_id] = sorted(
            chosen,
            key=lambda record: _selection_hash(
                seed, scene_id, record["question_id"], record["question"]
            ),
        )
    return selected


def validate_exact_visibility_files(
    oracle_root: Path,
    scene_ids: list[str],
) -> dict[str, dict[str, int]]:
    """Fail closed unless every QA scene has exact ray-hit visibility proof."""

    results: dict[str, dict[str, int]] = {}
    for scene_id in scene_ids:
        path = oracle_root / scene_id / "visibility.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Exact visibility evidence is unavailable for {scene_id}: {path}"
            )
        try:
            evidence = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Exact visibility evidence is invalid JSON for {scene_id}: {path}"
            ) from error
        counts = validate_visibility_evidence(evidence)
        if evidence.get("scene_id") != scene_id:
            raise ValueError(
                f"Exact visibility evidence scene mismatch: expected {scene_id}, "
                f"found {evidence.get('scene_id')!r}"
            )
        results[scene_id] = counts
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument(
        "--include-deferred-test",
        action="store_true",
        help="Explicitly include scenes in batch.deferred_splits",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    oracle_root = PROJECT_ROOT / config["paths"]["data_root"] / "oracle"
    available_scene_ids = {
        path.name for path in oracle_root.glob("scene_*") if (path / "oracle.json").is_file()
    }
    explicit_splits = None
    if isinstance(config.get("batch"), dict):
        plans = batch_scene_plans(config)
        explicit_splits = batch_scene_splits(config, plans)
    deferred_splits = set(config.get("batch", {}).get("deferred_splits", []))
    if deferred_splits and explicit_splits is None:
        raise ValueError("batch.deferred_splits requires explicit batch.splits")
    locked_scene_ids = (
        set()
        if args.include_deferred_test or explicit_splits is None
        else {
            scene_id for split_name in deferred_splits for scene_id in explicit_splits[split_name]
        }
    )
    if args.scenes:
        requested = set(args.scenes)
        unavailable = requested - available_scene_ids
        if unavailable:
            raise FileNotFoundError(f"Oracle scenes are unavailable: {sorted(unavailable)}")
        locked = requested & locked_scene_ids
        if locked:
            raise ValueError(
                f"Deferred test scenes require --include-deferred-test: {sorted(locked)}"
            )
        scene_ids = sorted(requested)
    elif explicit_splits is not None:
        ordered = [
            scene_id
            for split_name in ("train", "validation", "test")
            for scene_id in explicit_splits[split_name]
        ]
        required_scene_ids = [scene_id for scene_id in ordered if scene_id not in locked_scene_ids]
        if bool(config.get("batch", {}).get("require_visibility_evidence", False)):
            unavailable = set(required_scene_ids) - available_scene_ids
            if unavailable:
                raise FileNotFoundError(
                    "Exact-visibility QA requires every selected batch scene to be "
                    f"materialized: {sorted(unavailable)}"
                )
        scene_ids = [
            scene_id
            for scene_id in required_scene_ids
            if scene_id in available_scene_ids and scene_id not in locked_scene_ids
        ]
    else:
        scene_ids = sorted(available_scene_ids)
    if not scene_ids:
        raise SystemExit("No oracle scenes found; render a scene first")
    if bool(config.get("batch", {}).get("require_visibility_evidence", False)):
        validate_exact_visibility_files(oracle_root, scene_ids)
    oracles = {
        scene_id: json.loads((oracle_root / scene_id / "oracle.json").read_text(encoding="utf-8"))
        for scene_id in scene_ids
    }
    visibility_evidence_by_scene: dict[str, Mapping[str, Any]] = {}
    for scene_id in scene_ids:
        visibility_path = oracle_root / scene_id / "visibility.json"
        if visibility_path.is_file():
            try:
                evidence = json.loads(visibility_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Exact visibility evidence is invalid JSON for {scene_id}: {visibility_path}"
                ) from error
            if not isinstance(evidence, Mapping):
                raise TypeError(f"Visibility evidence must be a mapping: {visibility_path}")
            visibility_evidence_by_scene[scene_id] = evidence
    scene_groups = counterfactual_scene_groups(oracles)
    if explicit_splits is None:
        splits = scene_level_splits(scene_ids, int(config["seed"]), scene_groups)
    else:
        selected_scene_ids = set(scene_ids)
        splits = {
            split_name: [
                scene_id
                for scene_id in explicit_splits[split_name]
                if scene_id in selected_scene_ids
            ]
            for split_name in ("train", "validation", "test")
        }
    assert_scene_disjoint(splits)
    assert_group_disjoint(splits, scene_groups)
    qa_root = artifact_root(config, "qa")
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
            visibility_evidence=visibility_evidence_by_scene.get(scene_id),
        )
    annotated_records = annotate_counterfactual_questions(records_by_scene, oracles)
    balanced_config = config.get("qa", {}).get("balanced_selection", {})
    selection_manifest: dict[str, Any] = {"enabled": False}
    if bool(balanced_config.get("enabled", False)):
        per_scene = balanced_config.get("per_scene")
        if not isinstance(per_scene, Mapping):
            raise TypeError("qa.balanced_selection.per_scene must be a mapping")
        selected_records: dict[str, list[dict[str, Any]]] = {}
        for split_name, split_scenes in splits.items():
            if not split_scenes:
                continue
            split_selected = select_balanced_split_records(
                records_by_scene,
                split_scenes,
                per_scene_limit=int(per_scene[split_name]),
                seed=int(config["seed"]),
                max_changed_units_per_pair=int(
                    balanced_config.get("max_changed_units_per_pair", 4)
                ),
            )
            selected_records.update(split_selected)
        records_by_scene = selected_records
        selection_manifest = {
            "enabled": True,
            "per_scene": {key: int(value) for key, value in per_scene.items()},
            "max_changed_units_per_pair": int(balanced_config.get("max_changed_units_per_pair", 4)),
        }
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
        "balanced_selection": selection_manifest,
        "fingerprint": split_fingerprint(splits),
        "question_counts": dict(totals),
    }
    (qa_root / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

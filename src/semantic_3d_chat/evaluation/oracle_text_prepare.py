"""Prepare question-independent exact scene text for a prohibited upper bound.

This is the only stage of the benchmark that opens simulator oracle JSON.  It
copies no QA answers or target-instance IDs.  Its output is an explicitly
evaluation-only textual scene artifact and must never be used by primary chat.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.baseline_io import sha256_file
from semantic_3d_chat.evaluation.oracle_text_artifacts import (
    SceneTextRecord,
    atomic_write_json,
    build_scene_text_bundle,
    text_sha256,
    validate_v55_development_scope,
)
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest

DEFAULT_CONFIG = Path("configs/experiments/gemma4_oracle_text_v55.yaml")
DEFAULT_QUESTIONS = Path("reports/gemma4/questions/v55_development_validation.json")
DEFAULT_ORACLE_ROOT = Path("data/oracle")
DEFAULT_OUTPUT = Path(
    "reports/gemma4/evaluation_only/oracle_text_upper_bound/v55_scene_descriptions.json"
)


def _finite_triplet(value: object, field: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{field} must contain three numeric values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain finite values")
    return result  # type: ignore[return-value]


def _number(value: float) -> str:
    return f"{float(value):.3f}"


def _orientation(instance: Mapping[str, Any]) -> str:
    pose = instance.get("pose")
    if not isinstance(pose, Mapping):
        return "unknown"
    rotation = _finite_triplet(pose.get("rotation_euler_degrees"), "pose rotation")
    # The synthetic chair counterfactual is constructed by a half-turn about a
    # horizontal axis.  Derive the state from pose rather than generation labels.
    horizontal_tilt = max(abs(((angle + 180.0) % 360.0) - 180.0) for angle in rotation[:2])
    return "upside down" if horizontal_tilt >= 135.0 else "upright"


def _display_names(instances: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    category_counts = Counter(str(instance.get("category", "unknown")) for instance in instances)
    category_indices: defaultdict[str, int] = defaultdict(int)
    names: dict[str, str] = {}
    for instance in sorted(instances, key=lambda item: str(item.get("instance_id", ""))):
        instance_id = str(instance.get("instance_id", ""))
        category = str(instance.get("category", "unknown"))
        category_indices[category] += 1
        names[instance_id] = (
            category
            if category_counts[category] == 1
            else f"{category} {category_indices[category]}"
        )
    return names


def oracle_scene_text(
    oracle: Mapping[str, Any],
    *,
    camera_position_m: Sequence[float],
    camera_yaw_degrees: float,
    camera_pitch_degrees: float,
) -> str:
    """Create a complete scene-level fact sheet with no simulator instance IDs."""

    raw_instances = oracle.get("instances")
    if not isinstance(raw_instances, list) or not raw_instances:
        raise ValueError("Oracle requires a non-empty instances list")
    instances: list[Mapping[str, Any]] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(raw_instances):
        if not isinstance(value, Mapping):
            raise TypeError(f"Oracle instance {index} must be an object")
        instance_id = value.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"Oracle instance {index} lacks an instance ID")
        if instance_id in by_id:
            raise ValueError(f"Duplicate oracle instance ID: {instance_id}")
        instances.append(value)
        by_id[instance_id] = value
    names = _display_names(instances)
    camera = _finite_triplet(camera_position_m, "camera_position_m")
    if not all(math.isfinite(value) for value in (camera_yaw_degrees, camera_pitch_degrees)):
        raise ValueError("Camera yaw and pitch must be finite")
    yaw = math.radians(float(camera_yaw_degrees))

    room = oracle.get("room")
    if not isinstance(room, Mapping):
        raise TypeError("Oracle room must be an object")
    room_size = _finite_triplet(room.get("size_m"), "room size")
    room_min = _finite_triplet(room.get("bounds_min_m"), "room minimum bounds")
    room_max = _finite_triplet(room.get("bounds_max_m"), "room maximum bounds")

    objects = [instance for instance in instances if instance.get("kind") == "object"]
    counts = Counter(str(instance.get("category", "unknown")) for instance in objects)
    lines = [
        "EVALUATION-ONLY ORACLE TEXT; prohibited as input to the primary 3D-memory model.",
        "This fact sheet describes the complete unchanged scene and is independent of the question.",
        "Coordinates are meters in world axes X=right, Y=forward, Z=up.",
        (
            "Reference camera pose: position="
            f"({_number(camera[0])},{_number(camera[1])},{_number(camera[2])}); "
            f"yaw={_number(camera_yaw_degrees)} degrees; "
            f"pitch={_number(camera_pitch_degrees)} degrees."
        ),
        (
            "Room: size="
            f"({_number(room_size[0])},{_number(room_size[1])},{_number(room_size[2])}); "
            f"bounds_min=({_number(room_min[0])},{_number(room_min[1])},{_number(room_min[2])}); "
            f"bounds_max=({_number(room_max[0])},{_number(room_max[1])},{_number(room_max[2])})."
        ),
        "Exact present-object counts: "
        + "; ".join(f"{category}={count}" for category, count in sorted(counts.items()))
        + ". Categories absent from this inventory have count zero.",
        "Exact object facts:",
    ]

    for instance in sorted(objects, key=lambda item: names[str(item["instance_id"])]):
        instance_id = str(instance["instance_id"])
        center = _finite_triplet(instance.get("expected_center_xyz_m"), "object center")
        dimensions = _finite_triplet(instance.get("dimensions_m"), "object dimensions")
        color_value = instance.get("color")
        color = (
            str(color_value.get("name", "unknown"))
            if isinstance(color_value, Mapping)
            else "unknown"
        )
        support_id = instance.get("support_surface")
        support = names.get(str(support_id), "none") if support_id is not None else "none"
        dx, dy, dz = (center[index] - camera[index] for index in range(3))
        camera_right = math.cos(yaw) * dx + math.sin(yaw) * dy
        camera_forward = -math.sin(yaw) * dx + math.cos(yaw) * dy
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        visible = "yes" if instance.get("visible_from_center_scan") is True else "no"
        lines.append(
            f"- {names[instance_id]}: category={instance.get('category')}; color={color}; "
            f"center=({_number(center[0])},{_number(center[1])},{_number(center[2])}); "
            f"dimensions=({_number(dimensions[0])},{_number(dimensions[1])},"
            f"{_number(dimensions[2])}); orientation={_orientation(instance)}; "
            f"supported_by={support}; center_scan_visible={visible}; "
            f"camera_distance={_number(distance)}; camera_right={_number(camera_right)}; "
            f"camera_forward={_number(camera_forward)}."
        )

    raw_relationships = oracle.get("relationships", [])
    if not isinstance(raw_relationships, list):
        raise TypeError("Oracle relationships must be a list")
    relationship_lines: set[str] = set()
    for index, value in enumerate(raw_relationships):
        if not isinstance(value, Mapping):
            raise TypeError(f"Oracle relationship {index} must be an object")
        subject_id = str(value.get("subject_instance_id", ""))
        object_id = str(value.get("object_instance_id", ""))
        subject = by_id.get(subject_id)
        object_ = by_id.get(object_id)
        if subject is None or object_ is None or subject.get("kind") != "object":
            continue
        predicate = str(value.get("predicate", "")).replace("_", " ").strip()
        if not predicate:
            continue
        relationship_lines.add(f"- {names[subject_id]} | {predicate} | {names[object_id]}.")
    lines.append("Exact directed relationships (subject | relation | object):")
    lines.extend(sorted(relationship_lines))
    result = "\n".join(lines).strip()
    leaked_ids = [instance_id for instance_id in by_id if instance_id in result]
    if leaked_ids:
        raise AssertionError("Sanitized oracle text leaked simulator instance IDs")
    return result


def prepare_scene_text_bundle(
    config: Mapping[str, Any],
    question_manifest_path: str | Path,
    oracle_root: str | Path,
    output_path: str | Path,
    *,
    require_v55_development: bool = True,
) -> dict[str, Any]:
    """Declassify exact scene facts into the isolated evaluation-control area."""

    question_manifest = load_question_manifest(question_manifest_path)
    if question_manifest.manifest_sha256 is None:
        raise AssertionError("Loaded question manifest lacks its file hash")
    scene_ids = sorted(question_manifest.by_scene())
    validate_v55_development_scope(
        scene_ids,
        question_manifest.question_count,
        required=require_v55_development,
    )
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise TypeError("Config requires runtime settings")
    viewpoint = runtime.get("reference_viewpoint")
    if not isinstance(viewpoint, Mapping):
        raise TypeError("Config requires runtime.reference_viewpoint")
    position = _finite_triplet(viewpoint.get("position_m"), "reference camera position")
    yaw = float(viewpoint.get("yaw_degrees"))
    pitch = float(viewpoint.get("pitch_degrees"))

    oracle_directory = Path(oracle_root).expanduser().resolve()
    if oracle_directory.name.casefold() != "oracle":
        raise ValueError("Preparation oracle_root must be the exact isolated oracle directory")
    records: list[SceneTextRecord] = []
    for scene_id in scene_ids:
        oracle_path = oracle_directory / scene_id / "oracle.json"
        if not oracle_path.is_file() or oracle_path.is_symlink():
            raise FileNotFoundError(f"Oracle scene is unavailable or unsafe: {oracle_path}")
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        if not isinstance(oracle, Mapping) or oracle.get("scene_id") != scene_id:
            raise ValueError(f"Oracle scene identity mismatch: {oracle_path}")
        scene_text = oracle_scene_text(
            oracle,
            camera_position_m=position,
            camera_yaw_degrees=yaw,
            camera_pitch_degrees=pitch,
        )
        records.append(
            SceneTextRecord(
                scene_id=scene_id,
                scene_text=scene_text,
                scene_text_sha256=text_sha256(scene_text),
                source_oracle_sha256=sha256_file(oracle_path),
            )
        )
    bundle = build_scene_text_bundle(
        records,
        question_manifest_sha256=question_manifest.manifest_sha256,
        questions_sha256=question_manifest.questions_sha256,
        source_qa_sha256=question_manifest.source_qa_sha256,
    )
    destination = Path(output_path).expanduser().resolve()
    atomic_write_json(destination, bundle.as_dict())
    return {
        "artifact": "oracle_text_scene_preparation",
        "evaluation_only": True,
        "primary_path_eligible": False,
        "prohibited_primary_input": True,
        "question_independent": True,
        "question_count": question_manifest.question_count,
        "scene_count": len(records),
        "scene_ids": scene_ids,
        "question_manifest_path": str(question_manifest.manifest_path),
        "question_manifest_sha256": question_manifest.manifest_sha256,
        "scene_descriptions_path": str(destination),
        "scene_descriptions_sha256": sha256_file(destination),
        "scene_descriptions_content_sha256": bundle.scene_descriptions_sha256,
        "source_oracle_hashes": {
            record.scene_id: record.source_oracle_sha256 for record in bundle.scenes
        },
    }


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-non-v55-scope",
        action="store_true",
        help="Explicitly permit a non-V55 development manifest for isolated tests.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = _project_path(args.config)
    report = prepare_scene_text_bundle(
        load_config(config_path),
        _project_path(args.questions),
        _project_path(args.oracle_root),
        _project_path(args.output),
        require_v55_development=not args.allow_non_v55_scope,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

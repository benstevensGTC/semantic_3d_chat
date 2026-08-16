"""Fail-closed behavioral promotion for the production Gemma chat path.

Promotion is created offline, where selector and evaluation reports are
allowed to be read.  The chat process reads only the resulting numeric/hash
attestation; it never opens those training/evaluation artifacts itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.model_snapshot import local_model_snapshot_identity
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
    runtime_config_file_sha256,
    validate_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root
from semantic_3d_chat.evaluation.prediction_artifacts import (
    PROVENANCE_SCHEMA_VERSION,
    scene_map_manifest_sha256,
    validate_scene_map_manifest,
)

PROMOTION_FILENAME: Final[str] = "promotion.json"
PROMOTION_SCHEMA_VERSION: Final[int] = 4
PRIMARY_POINTER_SCHEMA_VERSION: Final[int] = 1
FINAL_EVIDENCE_SCHEMA_VERSION: Final[int] = 3
MIN_FINAL_COUNTERFACTUAL_PAIR_ACCURACY: Final[float] = 0.5
MIN_FINAL_COUNTERFACTUAL_CHANGED_RATE: Final[float] = 0.5
MIN_FINAL_GROUNDING_COVERAGE: Final[float] = 1.0
MIN_FINAL_GROUNDING_RELATIVE_IMPROVEMENT: Final[float] = 0.1
MAX_FINAL_GROUNDING_ROOM_DIAGONAL_FRACTION: Final[float] = 0.5
PRIMARY_POINTER_DEFAULT: Final[Path] = PROJECT_ROOT / "configs/runtime/primary.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_OPAQUE_QUESTION_ID = re.compile(r"q_[0-9]{6}")
_FORBIDDEN_RUNTIME_PATH_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "rendered", "features"}
)
_FINAL_EVIDENCE_PATH_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("metrics_path", "metrics_sha256"),
    ("predictions_path", "predictions_sha256"),
    ("prediction_provenance_path", "prediction_provenance_sha256"),
    ("chance_metrics_path", "chance_metrics_sha256"),
    ("chance_predictions_path", "chance_predictions_sha256"),
    ("chance_prediction_provenance_path", "chance_prediction_provenance_sha256"),
    ("split_manifest_path", "split_manifest_sha256"),
)
_FINAL_PERFORMANCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "counterfactual_pair_accuracy",
        "chance_counterfactual_pair_accuracy",
        "counterfactual_changed_when_expected_rate",
        "chance_counterfactual_changed_when_expected_rate",
        "grounding_coverage",
        "chance_grounding_coverage",
        "grounding_mean_coordinate_error_m",
        "chance_grounding_mean_coordinate_error_m",
        "minimum_counterfactual_pair_accuracy",
        "minimum_counterfactual_changed_when_expected_rate",
        "minimum_grounding_coverage",
        "maximum_grounding_mean_coordinate_error_m",
        "minimum_grounding_relative_improvement",
    }
)


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_identity_sha256(path: str | Path) -> str:
    return hashlib.sha256(str(_rooted(path)).encode("utf-8")).hexdigest()


def _json_object(path: str | Path) -> dict[str, Any]:
    unresolved = _unresolved_rooted(path)
    _reject_symlink_components(unresolved, "Attestation input")
    source = unresolved.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Required attestation input is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {source}")
    return payload


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _unresolved_rooted(path: str | Path) -> Path:
    """Return an absolute path without following the final path component."""

    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _reject_symlink_components(path: Path, field: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{field} must not use symbolic-link path components: {current}")


def _checkpoint_files(checkpoint: str | Path) -> tuple[Path, Path, Path]:
    unresolved_root = _unresolved_rooted(checkpoint)
    _reject_symlink_components(unresolved_root, "Checkpoint directory")
    root = unresolved_root.resolve()
    forbidden = _FORBIDDEN_RUNTIME_PATH_COMPONENTS.intersection(
        part.casefold() for part in root.parts
    )
    if forbidden:
        raise ValueError(
            "Checkpoint directory is inside a forbidden runtime path: "
            f"{root}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    adapter = root / "adapter.safetensors"
    runtime_metadata = root / "runtime_metadata.json"
    promotion = root / PROMOTION_FILENAME
    for path in (adapter, runtime_metadata):
        _reject_symlink_components(path, "Checkpoint runtime input")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint runtime input is missing: {path}")
    return adapter, runtime_metadata, promotion


def _require_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _regular_file(path: str | Path, field: str) -> Path:
    unresolved = _unresolved_rooted(path)
    _reject_symlink_components(unresolved, field)
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {resolved}")
    return resolved


def _validate_scene_runtime_manifest(
    value: object,
    *,
    runtime_config: Mapping[str, Any] | None = None,
    expected_prefix_hashes: Mapping[str, str] | None = None,
) -> dict[str, dict[str, int | str]]:
    """Validate the sanitized scene-to-map/prefix identity used by chat.

    The manifest intentionally contains no source paths, labels, captions, or
    simulator identifiers: only opaque scene IDs, byte identity, and the
    continuous prefix identity produced from those bytes.
    """

    if not isinstance(value, Mapping) or not value:
        raise ValueError("Scene runtime manifest must be a nonempty mapping")
    result: dict[str, dict[str, int | str]] = {}
    expected_entry_fields = {
        "voxel_map_sha256",
        "voxel_map_size_bytes",
        "scene_prefix_sha256",
    }
    for scene_id, raw_entry in value.items():
        if not isinstance(scene_id, str) or _OPAQUE_SCENE_ID.fullmatch(scene_id) is None:
            raise ValueError("Scene runtime manifest keys must be opaque scene IDs")
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != expected_entry_fields:
            raise ValueError(
                f"Scene runtime manifest entry has invalid fields: {scene_id}"
            )
        map_sha256 = _require_sha(
            raw_entry.get("voxel_map_sha256"),
            f"scene runtime manifest {scene_id} voxel_map_sha256",
        )
        prefix_sha256 = _require_sha(
            raw_entry.get("scene_prefix_sha256"),
            f"scene runtime manifest {scene_id} scene_prefix_sha256",
        )
        size_bytes = raw_entry.get("voxel_map_size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
            raise ValueError(
                f"Scene runtime manifest {scene_id} voxel_map_size_bytes must be positive"
            )
        result[scene_id] = {
            "voxel_map_sha256": map_sha256,
            "voxel_map_size_bytes": size_bytes,
            "scene_prefix_sha256": prefix_sha256,
        }

    if expected_prefix_hashes is not None:
        expected = dict(expected_prefix_hashes)
        if set(result) != set(expected):
            raise ValueError(
                "Scene runtime manifest scenes do not match primary prediction scenes"
            )
        prefix_mismatches = {
            scene_id: {
                "manifest": result[scene_id]["scene_prefix_sha256"],
                "predictions": expected[scene_id],
            }
            for scene_id in sorted(expected)
            if result[scene_id]["scene_prefix_sha256"] != expected[scene_id]
        }
        if prefix_mismatches:
            raise ValueError(
                "Scene runtime manifest prefix hashes do not match primary predictions: "
                f"{prefix_mismatches}"
            )

    if runtime_config is not None:
        maps_root = artifact_root(dict(runtime_config), "maps").resolve()
        for scene_id, entry in result.items():
            map_path = _regular_file(
                maps_root / scene_id / "voxel_map.npz",
                f"scene runtime map {scene_id}",
            )
            observed_size = map_path.stat().st_size
            observed_sha256 = sha256_file(map_path)
            if (
                observed_size != entry["voxel_map_size_bytes"]
                or observed_sha256 != entry["voxel_map_sha256"]
            ):
                raise ValueError(
                    "Scene runtime manifest voxel-map bytes changed: "
                    f"scene={scene_id} expected_size={entry['voxel_map_size_bytes']} "
                    f"observed_size={observed_size} "
                    f"expected_sha256={entry['voxel_map_sha256']} "
                    f"observed_sha256={observed_sha256}"
                )
    return {scene_id: result[scene_id] for scene_id in sorted(result)}


def _build_scene_runtime_manifest(
    runtime_config: Mapping[str, Any], predictions_path: Path
) -> dict[str, dict[str, int | str]]:
    prefix_hashes: dict[str, set[str]] = {}
    for record in _jsonl_objects(predictions_path):
        scene_id = str(record.get("scene_id", ""))
        if _OPAQUE_SCENE_ID.fullmatch(scene_id) is None:
            raise ValueError("Primary predictions contain a non-opaque scene ID")
        prefix_hash = _require_sha(
            record.get("prefix_hash"), f"primary prediction {scene_id} prefix_hash"
        )
        prefix_hashes.setdefault(scene_id, set()).add(prefix_hash)
    if not prefix_hashes or any(len(values) != 1 for values in prefix_hashes.values()):
        raise ValueError("Primary predictions do not provide one invariant prefix per scene")
    maps_root = artifact_root(dict(runtime_config), "maps").resolve()
    manifest: dict[str, dict[str, int | str]] = {}
    for scene_id, values in sorted(prefix_hashes.items()):
        map_path = _regular_file(
            maps_root / scene_id / "voxel_map.npz",
            f"scene runtime map {scene_id}",
        )
        manifest[scene_id] = {
            "voxel_map_sha256": sha256_file(map_path),
            "voxel_map_size_bytes": map_path.stat().st_size,
            "scene_prefix_sha256": next(iter(values)),
        }
    return _validate_scene_runtime_manifest(
        manifest,
        runtime_config=runtime_config,
        expected_prefix_hashes={
            scene_id: next(iter(values)) for scene_id, values in prefix_hashes.items()
        },
    )


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_checkpoint_runtime_metadata(path: Path) -> None:
    """Prove offline that the metadata hash bound into promotion is runtime-safe."""

    from semantic_3d_chat.training.checkpointing import (
        validate_runtime_checkpoint_metadata,
    )

    validate_runtime_checkpoint_metadata(_json_object(path))


def _prediction_provenance_sha256(report: Mapping[str, Any]) -> str:
    identity_fields = {
        "schema_version",
        "config_sha256",
        "config_file_sha256",
        "checkpoint_sha256",
        "references_sha256",
        "scene_map_manifest_sha256",
        "split",
        "run_kind",
        "condition",
    }
    return _canonical_json_sha256({key: report.get(key) for key in identity_fields})


def _validate_prediction_provenance(
    report: Mapping[str, Any],
    *,
    runtime_config_path: Path,
    runtime_config_sha256: str,
    checkpoint: Path,
    adapter_sha256: str,
    run_kind: str,
    condition: str,
) -> str:
    expected = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "config_sha256": effective_runtime_config_sha256(
            load_runtime_config(runtime_config_path)
        ),
        "config_file_sha256": runtime_config_sha256,
        "config_path": str(runtime_config_path),
        "checkpoint_path": str(checkpoint),
        "split": "test",
        "run_kind": run_kind,
    }
    mismatches = {
        key: {"provenance": report.get(key), "required": value}
        for key, value in expected.items()
        if report.get(key) != value
    }
    observed_condition = report.get("condition")
    if condition == "all_questions":
        if observed_condition != condition:
            mismatches["condition"] = {
                "provenance": observed_condition,
                "required": condition,
            }
    else:
        try:
            parsed_condition = json.loads(str(observed_condition))
        except json.JSONDecodeError:
            parsed_condition = None
        if not isinstance(parsed_condition, Mapping) or (
            parsed_condition.get("condition") != condition
            or parsed_condition.get("max_questions_per_scene") is not None
        ):
            mismatches["condition"] = {
                "provenance": observed_condition,
                "required": f"complete {condition} control",
            }
    files = report.get("checkpoint_files")
    adapter_entries = (
        [entry for entry in files if isinstance(entry, Mapping) and entry.get("path") == "adapter.safetensors"]
        if isinstance(files, list)
        else []
    )
    if len(adapter_entries) != 1 or adapter_entries[0].get("sha256") != adapter_sha256:
        mismatches["checkpoint_files"] = "adapter digest is absent or mismatched"
    runtime_metadata = checkpoint / "runtime_metadata.json"
    runtime_metadata_entries = (
        [
            entry
            for entry in files
            if isinstance(entry, Mapping)
            and entry.get("path") == "runtime_metadata.json"
        ]
        if isinstance(files, list)
        else []
    )
    if (
        len(runtime_metadata_entries) != 1
        or runtime_metadata_entries[0].get("sha256")
        != sha256_file(runtime_metadata)
        or runtime_metadata_entries[0].get("size_bytes")
        != runtime_metadata.stat().st_size
    ):
        mismatches["checkpoint_runtime_metadata"] = (
            "runtime metadata digest/size is absent or mismatched"
        )
    if not isinstance(files, list) or report.get("checkpoint_sha256") != _canonical_json_sha256(
        files
    ):
        mismatches["checkpoint_sha256"] = "checkpoint file envelope digest is stale"
    try:
        scene_map_manifest = validate_scene_map_manifest(
            report.get("scene_map_manifest"),
            config=load_runtime_config(runtime_config_path),
        )
        observed_map_manifest_hash = scene_map_manifest_sha256(scene_map_manifest)
        if report.get("scene_map_manifest_sha256") != observed_map_manifest_hash:
            mismatches["scene_map_manifest_sha256"] = {
                "provenance": report.get("scene_map_manifest_sha256"),
                "required": observed_map_manifest_hash,
            }
    except (OSError, TypeError, ValueError) as exc:
        scene_map_manifest = None
        mismatches["scene_map_manifest"] = str(exc)
    references_path = report.get("references_path")
    if not isinstance(references_path, str):
        mismatches["references_path"] = "must be a path string"
    else:
        try:
            questions = _regular_file(references_path, "prediction questions manifest")
            if report.get("references_sha256") != sha256_file(questions):
                mismatches["references_sha256"] = "questions manifest digest is stale"
            from semantic_3d_chat.evaluation.question_manifest import (
                load_question_manifest,
            )

            question_manifest = load_question_manifest(questions)
            question_scene_ids = {
                record.scene_id for record in question_manifest.questions
            }
            if (
                scene_map_manifest is not None
                and set(scene_map_manifest) != question_scene_ids
            ):
                mismatches["scene_map_manifest_scenes"] = {
                    "provenance": sorted(scene_map_manifest),
                    "questions": sorted(question_scene_ids),
                }
        except (OSError, TypeError, ValueError) as exc:
            mismatches["references_path"] = str(exc)
    expected_provenance_hash = _prediction_provenance_sha256(report)
    if report.get("provenance_sha256") != expected_provenance_hash:
        mismatches["provenance_sha256"] = {
            "provenance": report.get("provenance_sha256"),
            "required": expected_provenance_hash,
        }
    if mismatches:
        raise ValueError(f"Prediction provenance is not promotion-safe: {mismatches}")
    return expected_provenance_hash


def _validate_question_manifest_binding(
    question_manifest_path: Path,
    references_path: Path,
) -> None:
    """Bind inference questions to the exact answer-bearing held-out records.

    Prediction provenance already content-addresses the sanitized manifest.  A
    separate check is still required to prove that its opaque IDs and question
    text are the projection of the references that are actually scored.  Without
    this binding, two files with the same opaque keys but different questions
    could be mixed while all prediction and metric hashes remained internally
    consistent.
    """

    # This import remains evaluation-side.  Production chat validation reads
    # only promotion.json and never imports or opens a question manifest.
    from semantic_3d_chat.evaluation.question_manifest import load_question_manifest

    manifest = load_question_manifest(question_manifest_path)
    references_hash = sha256_file(references_path)
    if manifest.source_qa_sha256 != references_hash:
        raise ValueError(
            "Question manifest is not derived from the scored held-out references"
        )
    reference_records = _jsonl_objects(references_path)
    projected_references = tuple(
        (
            str(record.get("scene_id", "")),
            str(record.get("question_id", "")),
            record.get("question"),
        )
        for record in reference_records
    )
    projected_manifest = tuple(
        (record.scene_id, record.question_id, record.question)
        for record in manifest.questions
    )
    if projected_manifest != projected_references:
        raise ValueError(
            "Question manifest does not exactly match the ordered held-out "
            "scene/question projection"
        )


def _validate_prediction_records(
    path: Path,
    *,
    provenance_sha256: str,
    require_prefix_hash: bool = False,
) -> list[dict[str, Any]]:
    records = _jsonl_objects(path)
    keys: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record.get("scene_id", "")), str(record.get("question_id", "")))
        if (
            _OPAQUE_SCENE_ID.fullmatch(key[0]) is None
            or _OPAQUE_QUESTION_ID.fullmatch(key[1]) is None
            or key in keys
        ):
            raise ValueError(f"Prediction file has a missing or duplicate opaque key: {key}")
        if record.get("provenance_sha256") != provenance_sha256:
            raise ValueError(f"Prediction record has stale provenance: {key}")
        if require_prefix_hash:
            try:
                _require_sha(record.get("prefix_hash"), f"prediction {key} prefix_hash")
            except ValueError as exc:
                raise ValueError(
                    f"Primary prediction record has an invalid scene-prefix hash: {key}"
                ) from exc
        keys.add(key)
    return records


def _scene_disjoint_test_split(
    split_manifest: Mapping[str, Any], references_path: Path
) -> bool:
    splits = split_manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Final split manifest must contain exactly train/validation/test")
    parsed: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        values = splits.get(name)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise TypeError(f"Final split manifest {name} must be a string list")
        parsed[name] = set(values)
        if len(parsed[name]) != len(values):
            raise ValueError(f"Final split manifest {name} contains duplicates")
        if any(_OPAQUE_SCENE_ID.fullmatch(value) is None for value in values):
            raise ValueError(f"Final split manifest {name} contains non-opaque scene IDs")
    if not parsed["test"]:
        raise ValueError("Final split manifest has no held-out test scenes")
    if any(
        parsed[left] & parsed[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Final split manifest is not scene-disjoint")
    reference_records = _jsonl_objects(references_path)
    reference_scenes = {str(record.get("scene_id", "")) for record in reference_records}
    if any(
        _OPAQUE_SCENE_ID.fullmatch(str(record.get("scene_id", ""))) is None
        or _OPAQUE_QUESTION_ID.fullmatch(str(record.get("question_id", ""))) is None
        for record in reference_records
    ):
        raise ValueError("Held-out references require opaque scene and question IDs")
    if "" in reference_scenes or reference_scenes != parsed["test"]:
        raise ValueError("Scored reference scenes do not exactly match the held-out test split")
    return True


def _selector_attestation(report: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    if report.get("passed") is not True or report.get("development_selection_passed") is not True:
        raise ValueError("Selector did not pass its development selection gate")
    if report.get("chat_promotion_eligible") is not True:
        raise ValueError("Selector explicitly denied chat promotion")
    selected = report.get("selected_checkpoint")
    if not isinstance(selected, str) or _rooted(selected) != checkpoint:
        raise ValueError("Selector-selected checkpoint does not match the promotion target")
    selected_update = report.get("selected_update")
    if (
        isinstance(selected_update, bool)
        or not isinstance(selected_update, int)
        or selected_update < 0
    ):
        raise ValueError("Selector selected_update must be a nonnegative integer")
    expected_checkpoint_suffix = f"update_{selected_update:03d}"
    if checkpoint.name != expected_checkpoint_suffix:
        raise ValueError(
            "Selector selected_update does not match the selected checkpoint suffix: "
            f"selected_update={selected_update} checkpoint={checkpoint.name!r} "
            f"required={expected_checkpoint_suffix!r}"
        )
    promotion = report.get("chat_promotion")
    if not isinstance(promotion, Mapping) or promotion.get("eligible") is not True:
        raise ValueError("Selector chat-promotion audit is not eligible")
    if promotion.get("evaluated") is not True:
        raise ValueError("Selector chat-promotion audit was not evaluated")
    checks = promotion.get("checks")
    required_checks = {
        "development_checkpoint_selected",
        "changed_complete_pair_threshold_met",
        "aggregate_validation_exact_accuracy_retained",
    }
    if not isinstance(checks, Mapping) or set(checks) != required_checks:
        raise ValueError("Selector chat-promotion checks are incomplete")
    if not all(value is True for value in checks.values()):
        raise ValueError("Selector chat-promotion checks did not all pass")
    if report.get("final_test_scenes_touched") is not False:
        raise ValueError("Selector report must attest that final scenes were not used for selection")
    return {"selected_update": selected_update, "checks_passed": len(required_checks)}


def _finite_metric(
    report: Mapping[str, Any], field: str, label: str, *, unit_interval: bool = False
) -> float:
    value = report.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label}.{field} must be a finite number")
    parsed = float(value)
    if unit_interval and not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label}.{field} must be in [0, 1]")
    return parsed


def _final_performance_summary(
    metrics: Mapping[str, Any],
    chance_metrics: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the scene-dependence gates from independently scored metrics."""

    counterfactual = metrics.get("counterfactual")
    chance_counterfactual = chance_metrics.get("counterfactual")
    grounding = metrics.get("grounding")
    chance_grounding = chance_metrics.get("grounding")
    if not isinstance(counterfactual, Mapping) or not isinstance(
        chance_counterfactual, Mapping
    ):
        raise TypeError("Primary and empty-prefix metrics require counterfactual mappings")
    if not isinstance(grounding, Mapping) or not isinstance(chance_grounding, Mapping):
        raise TypeError("Primary and empty-prefix metrics require grounding mappings")

    for label, report in (
        ("primary counterfactual", counterfactual),
        ("empty-prefix counterfactual", chance_counterfactual),
    ):
        for field in ("eligible_pairs", "expected_change_pairs"):
            value = report.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label}.{field} must be a positive integer")
        if report.get("malformed_pair_groups") != 0:
            raise ValueError(f"{label} contains malformed counterfactual pair groups")
    if (
        counterfactual.get("eligible_pairs")
        != chance_counterfactual.get("eligible_pairs")
        or counterfactual.get("expected_change_pairs")
        != chance_counterfactual.get("expected_change_pairs")
    ):
        raise ValueError("Primary and empty-prefix counterfactual coverage differs")

    for label, report in (
        ("primary grounding", grounding),
        ("empty-prefix grounding", chance_grounding),
    ):
        target_count = report.get("target_count")
        prediction_count = report.get("prediction_count")
        if (
            isinstance(target_count, bool)
            or not isinstance(target_count, int)
            or target_count < 1
            or isinstance(prediction_count, bool)
            or not isinstance(prediction_count, int)
            or prediction_count != target_count
        ):
            raise ValueError(f"{label} must predict every positive-count target")
    if grounding.get("target_count") != chance_grounding.get("target_count"):
        raise ValueError("Primary and empty-prefix grounding target counts differ")

    primary_pair = _finite_metric(
        counterfactual, "pair_accuracy", "primary counterfactual", unit_interval=True
    )
    chance_pair = _finite_metric(
        chance_counterfactual,
        "pair_accuracy",
        "empty-prefix counterfactual",
        unit_interval=True,
    )
    primary_changed = _finite_metric(
        counterfactual,
        "changed_when_expected_rate",
        "primary counterfactual",
        unit_interval=True,
    )
    chance_changed = _finite_metric(
        chance_counterfactual,
        "changed_when_expected_rate",
        "empty-prefix counterfactual",
        unit_interval=True,
    )
    primary_coverage = _finite_metric(
        grounding, "coverage", "primary grounding", unit_interval=True
    )
    chance_coverage = _finite_metric(
        chance_grounding, "coverage", "empty-prefix grounding", unit_interval=True
    )
    primary_error = _finite_metric(
        grounding, "mean_coordinate_error_m", "primary grounding"
    )
    chance_error = _finite_metric(
        chance_grounding,
        "mean_coordinate_error_m",
        "empty-prefix grounding",
    )
    if primary_error < 0.0 or chance_error < 0.0:
        raise ValueError("Grounding mean coordinate error must be nonnegative")

    scene = runtime_config.get("scene")
    room_size = scene.get("room_size_m") if isinstance(scene, Mapping) else None
    if (
        not isinstance(room_size, (list, tuple))
        or len(room_size) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in room_size
        )
    ):
        raise ValueError("Runtime scene.room_size_m must contain three positive dimensions")
    maximum_error = MAX_FINAL_GROUNDING_ROOM_DIAGONAL_FRACTION * math.sqrt(
        sum(float(value) ** 2 for value in room_size)
    )
    passed = bool(
        primary_pair >= MIN_FINAL_COUNTERFACTUAL_PAIR_ACCURACY
        and primary_pair > chance_pair
        and primary_changed >= MIN_FINAL_COUNTERFACTUAL_CHANGED_RATE
        and primary_changed > chance_changed
        and primary_coverage >= MIN_FINAL_GROUNDING_COVERAGE
        and chance_coverage >= MIN_FINAL_GROUNDING_COVERAGE
        and primary_error <= maximum_error
        and primary_error
        <= chance_error * (1.0 - MIN_FINAL_GROUNDING_RELATIVE_IMPROVEMENT)
    )
    return {
        "counterfactual_pair_accuracy": primary_pair,
        "chance_counterfactual_pair_accuracy": chance_pair,
        "counterfactual_changed_when_expected_rate": primary_changed,
        "chance_counterfactual_changed_when_expected_rate": chance_changed,
        "grounding_coverage": primary_coverage,
        "chance_grounding_coverage": chance_coverage,
        "grounding_mean_coordinate_error_m": primary_error,
        "chance_grounding_mean_coordinate_error_m": chance_error,
        "minimum_counterfactual_pair_accuracy": MIN_FINAL_COUNTERFACTUAL_PAIR_ACCURACY,
        "minimum_counterfactual_changed_when_expected_rate": (
            MIN_FINAL_COUNTERFACTUAL_CHANGED_RATE
        ),
        "minimum_grounding_coverage": MIN_FINAL_GROUNDING_COVERAGE,
        "maximum_grounding_mean_coordinate_error_m": maximum_error,
        "minimum_grounding_relative_improvement": (
            MIN_FINAL_GROUNDING_RELATIVE_IMPROVEMENT
        ),
        "performance_passed": passed,
    }


def _final_attestation(
    report: Mapping[str, Any],
    *,
    runtime_config_path: Path,
    checkpoint: Path,
    adapter_sha256: str,
    runtime_metadata_sha256: str,
    runtime_config_sha256: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "kind",
        "passed",
        "split",
        "scene_disjoint",
        "above_chance",
        "counterfactual_evaluated",
        "grounding_evaluated",
        "checkpoint_adapter_sha256",
        "checkpoint_runtime_metadata_sha256",
        "runtime_config_file_sha256",
        "reference_count",
        "prediction_count",
        "missing_prediction_count",
        "extra_prediction_count",
        "normalized_exact_accuracy",
        "chance_normalized_exact_accuracy",
        "scene_runtime_manifest",
    } | _FINAL_PERFORMANCE_FIELDS | {
        field for pair in _FINAL_EVIDENCE_PATH_FIELDS for field in pair
    }
    if set(report) != expected_fields:
        raise ValueError(
            "Held-out final evidence has an invalid field set: "
            f"missing={sorted(expected_fields - set(report))} "
            f"unexpected={sorted(set(report) - expected_fields)}"
        )
    required = {
        "schema_version": FINAL_EVIDENCE_SCHEMA_VERSION,
        "kind": "held_out_static_evaluation",
        "passed": True,
        "split": "test",
        "scene_disjoint": True,
        "above_chance": True,
        "counterfactual_evaluated": True,
        "grounding_evaluated": True,
        "checkpoint_adapter_sha256": adapter_sha256,
        "checkpoint_runtime_metadata_sha256": runtime_metadata_sha256,
        "runtime_config_file_sha256": runtime_config_sha256,
    }
    mismatches = {
        key: {"evidence": report.get(key), "required": value}
        for key, value in required.items()
        if report.get(key) != value
    }
    artifacts: dict[str, Path] = {}
    for path_field, hash_field in _FINAL_EVIDENCE_PATH_FIELDS:
        path_value = report.get(path_field)
        if not isinstance(path_value, str):
            mismatches[path_field] = "must be a path string"
            continue
        try:
            artifact = _regular_file(path_value, f"final evidence {path_field}")
            expected_hash = _require_sha(report.get(hash_field), f"final evidence {hash_field}")
            observed_hash = sha256_file(artifact)
            if observed_hash != expected_hash:
                mismatches[hash_field] = {
                    "evidence": expected_hash,
                    "observed": observed_hash,
                }
            artifacts[path_field] = artifact
        except (OSError, TypeError, ValueError) as exc:
            mismatches[path_field] = str(exc)
    counts: dict[str, int] = {}
    for field in (
        "reference_count",
        "prediction_count",
        "missing_prediction_count",
        "extra_prediction_count",
    ):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            mismatches[field] = "must be a nonnegative integer"
        else:
            counts[field] = value
    if counts and (
        counts.get("reference_count", 0) < 1
        or counts.get("prediction_count") != counts.get("reference_count")
        or counts.get("missing_prediction_count") != 0
        or counts.get("extra_prediction_count") != 0
    ):
        mismatches["coverage"] = counts
    exact = report.get("normalized_exact_accuracy")
    chance_exact = report.get("chance_normalized_exact_accuracy")
    if (
        isinstance(exact, bool)
        or not isinstance(exact, (int, float))
        or isinstance(chance_exact, bool)
        or not isinstance(chance_exact, (int, float))
        or not (0.0 <= float(exact) <= 1.0)
        or not (0.0 <= float(chance_exact) <= 1.0)
        or float(exact) <= float(chance_exact)
    ):
        mismatches["above_chance"] = {
            "normalized_exact_accuracy": exact,
            "chance_normalized_exact_accuracy": chance_exact,
        }
    if not mismatches:
        metrics = _json_object(artifacts["metrics_path"])
        chance_metrics = _json_object(artifacts["chance_metrics_path"])
        prediction_provenance = _json_object(artifacts["prediction_provenance_path"])
        chance_provenance = _json_object(
            artifacts["chance_prediction_provenance_path"]
        )
        primary_provenance_hash = _validate_prediction_provenance(
            prediction_provenance,
            runtime_config_path=runtime_config_path,
            runtime_config_sha256=runtime_config_sha256,
            checkpoint=checkpoint,
            adapter_sha256=adapter_sha256,
            run_kind="continuous_scene_static",
            condition="all_questions",
        )
        chance_provenance_hash = _validate_prediction_provenance(
            chance_provenance,
            runtime_config_path=runtime_config_path,
            runtime_config_sha256=runtime_config_sha256,
            checkpoint=checkpoint,
            adapter_sha256=adapter_sha256,
            run_kind="continuous_scene_control",
            condition="empty_scene_prefix",
        )
        if (
            prediction_provenance.get("references_path")
            != chance_provenance.get("references_path")
            or prediction_provenance.get("references_sha256")
            != chance_provenance.get("references_sha256")
        ):
            raise ValueError("Primary and empty-prefix runs used different question manifests")
        if (
            prediction_provenance.get("scene_map_manifest")
            != chance_provenance.get("scene_map_manifest")
            or prediction_provenance.get("scene_map_manifest_sha256")
            != chance_provenance.get("scene_map_manifest_sha256")
        ):
            raise ValueError("Primary and empty-prefix runs used different scene-map bytes")
        predictions = _validate_prediction_records(
            artifacts["predictions_path"],
            provenance_sha256=primary_provenance_hash,
            require_prefix_hash=True,
        )
        chance_predictions = _validate_prediction_records(
            artifacts["chance_predictions_path"],
            provenance_sha256=chance_provenance_hash,
        )
        references_value = metrics.get("references_path")
        if not isinstance(references_value, str):
            raise ValueError("Held-out metrics report has no references_path")
        references = _regular_file(references_value, "held-out references")
        references_hash = sha256_file(references)
        question_manifest_value = prediction_provenance.get("references_path")
        if not isinstance(question_manifest_value, str):
            raise ValueError("Prediction provenance has no question-manifest path")
        question_manifest = _regular_file(
            question_manifest_value,
            "prediction question manifest",
        )
        _validate_question_manifest_binding(question_manifest, references)
        split_manifest = _json_object(artifacts["split_manifest_path"])
        _scene_disjoint_test_split(split_manifest, references)
        reference_keys = {
            (str(record.get("scene_id")), str(record.get("question_id")))
            for record in _jsonl_objects(references)
        }
        prediction_keys = {
            (str(record.get("scene_id")), str(record.get("question_id")))
            for record in predictions
        }
        chance_prediction_keys = {
            (str(record.get("scene_id")), str(record.get("question_id")))
            for record in chance_predictions
        }
        if prediction_keys != reference_keys or chance_prediction_keys != reference_keys:
            raise ValueError("Primary/chance prediction keys do not match held-out references")
        # Offline promotion is allowed to open the answer-bearing references.
        # Recompute the structured metrics here so a hand-edited metric JSON
        # cannot promote fluent but unmeasured predictions.
        from semantic_3d_chat.evaluation.metrics import score_predictions

        reference_records = _jsonl_objects(references)
        recomputed_metrics = score_predictions(reference_records, predictions)
        recomputed_chance = score_predictions(reference_records, chance_predictions)
        for field in (
            "reference_count",
            "prediction_count",
            "missing_prediction_count",
            "extra_prediction_count",
            "normalized_exact_accuracy",
            "counterfactual",
            "grounding",
        ):
            if metrics.get(field) != recomputed_metrics.get(field):
                raise ValueError(f"Held-out metric {field} does not recompute exactly")
        for field in (
            "reference_count",
            "prediction_count",
            "missing_prediction_count",
            "extra_prediction_count",
            "normalized_exact_accuracy",
            "counterfactual",
            "grounding",
        ):
            if chance_metrics.get(field) != recomputed_chance.get(field):
                raise ValueError(f"Empty-prefix metric {field} does not recompute exactly")
        observed_metrics = {
            "reference_count": metrics.get("reference_count"),
            "prediction_count": metrics.get("prediction_count"),
            "missing_prediction_count": metrics.get("missing_prediction_count"),
            "extra_prediction_count": metrics.get("extra_prediction_count"),
            "normalized_exact_accuracy": metrics.get("normalized_exact_accuracy"),
            "predictions_path": metrics.get("predictions_path"),
            "predictions_sha256": metrics.get("predictions_sha256"),
            "references_sha256": metrics.get("references_sha256"),
        }
        expected_metrics = {
            **counts,
            "normalized_exact_accuracy": exact,
            "predictions_path": str(artifacts["predictions_path"]),
            "predictions_sha256": report["predictions_sha256"],
            "references_sha256": references_hash,
        }
        if observed_metrics != expected_metrics:
            raise ValueError(
                "Held-out metrics do not match final evidence: "
                f"metrics={observed_metrics} evidence={expected_metrics}"
            )
        if len(predictions) != counts["prediction_count"]:
            raise ValueError("Held-out prediction file count does not match metrics")
        if (
            chance_metrics.get("reference_count") != counts["reference_count"]
            or chance_metrics.get("prediction_count") != len(chance_predictions)
            or chance_metrics.get("missing_prediction_count") != 0
            or chance_metrics.get("extra_prediction_count") != 0
            or chance_metrics.get("references_sha256") != references_hash
            or chance_metrics.get("predictions_path")
            != str(artifacts["chance_predictions_path"])
            or chance_metrics.get("predictions_sha256")
            != report["chance_predictions_sha256"]
            or chance_metrics.get("normalized_exact_accuracy") != chance_exact
        ):
            raise ValueError("Empty-prefix chance metrics are incomplete or mismatched")
        counterfactual = metrics.get("counterfactual")
        if not isinstance(counterfactual, Mapping) or (
            not isinstance(counterfactual.get("eligible_pairs"), int)
            or counterfactual["eligible_pairs"] < 1
            or not isinstance(counterfactual.get("expected_change_pairs"), int)
            or counterfactual["expected_change_pairs"] < 1
            or counterfactual.get("malformed_pair_groups") != 0
        ):
            raise ValueError("Held-out metrics lack valid counterfactual-pair evaluation")
        grounding = metrics.get("grounding")
        if not isinstance(grounding, Mapping) or (
            not isinstance(grounding.get("target_count"), int)
            or grounding["target_count"] < 1
            or not isinstance(grounding.get("prediction_count"), int)
            or grounding["prediction_count"] < 1
        ):
            raise ValueError("Held-out metrics lack grounding evaluation")
        performance = _final_performance_summary(
            metrics,
            chance_metrics,
            load_runtime_config(runtime_config_path),
        )
        reported_performance = {
            field: report.get(field) for field in _FINAL_PERFORMANCE_FIELDS
        }
        expected_performance = {
            field: performance[field] for field in _FINAL_PERFORMANCE_FIELDS
        }
        if reported_performance != expected_performance:
            raise ValueError(
                "Held-out final performance summary does not match recomputed metrics"
            )
        if performance["performance_passed"] is not True:
            raise ValueError(
                "Held-out final evaluation failed counterfactual or grounding gates: "
                f"{expected_performance}"
            )
        prefix_hashes_by_scene: dict[str, set[str]] = {}
        for prediction in predictions:
            scene_id = str(prediction["scene_id"])
            prefix_hashes_by_scene.setdefault(scene_id, set()).add(
                str(prediction["prefix_hash"])
            )
        if any(len(values) != 1 for values in prefix_hashes_by_scene.values()):
            raise ValueError("Primary predictions changed scene prefix within one scene")
        primary_prefix_hash_by_scene = {
            scene_id: next(iter(values))
            for scene_id, values in sorted(prefix_hashes_by_scene.items())
        }
        scene_runtime_manifest = _validate_scene_runtime_manifest(
            report.get("scene_runtime_manifest"),
            runtime_config=load_runtime_config(runtime_config_path),
            expected_prefix_hashes=primary_prefix_hash_by_scene,
        )
        attested_map_manifest = {
            scene_id: {
                "voxel_map_sha256": entry["voxel_map_sha256"],
                "voxel_map_size_bytes": entry["voxel_map_size_bytes"],
            }
            for scene_id, entry in scene_runtime_manifest.items()
        }
        if attested_map_manifest != prediction_provenance.get("scene_map_manifest"):
            raise ValueError(
                "Final scene runtime manifest does not match prediction map provenance"
            )
    if mismatches:
        raise ValueError(f"Held-out final evidence is not promotion-safe: {mismatches}")
    return {
        "reference_count": counts["reference_count"],
        "prediction_count": counts["prediction_count"],
        "primary_prefix_hash_by_scene": primary_prefix_hash_by_scene,
        "scene_runtime_manifest": scene_runtime_manifest,
    }


def create_held_out_final_evidence(
    *,
    runtime_config_path: str | Path,
    checkpoint: str | Path,
    metrics_path: str | Path,
    predictions_path: str | Path,
    prediction_provenance_path: str | Path,
    chance_metrics_path: str | Path,
    chance_predictions_path: str | Path,
    chance_prediction_provenance_path: str | Path,
    split_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build final evidence only from complete, content-addressed test artifacts."""

    config = load_runtime_config(runtime_config_path)
    config_path = Path(str(config["_config_path"]))
    adapter, runtime_metadata, _ = _checkpoint_files(checkpoint)
    checkpoint_path = adapter.parent
    resolved_artifacts = {
        field: _regular_file(value, field)
        for field, value in {
            "metrics_path": metrics_path,
            "predictions_path": predictions_path,
            "prediction_provenance_path": prediction_provenance_path,
            "chance_metrics_path": chance_metrics_path,
            "chance_predictions_path": chance_predictions_path,
            "chance_prediction_provenance_path": chance_prediction_provenance_path,
            "split_manifest_path": split_manifest_path,
        }.items()
    }
    metrics = _json_object(resolved_artifacts["metrics_path"])
    chance_metrics = _json_object(resolved_artifacts["chance_metrics_path"])
    exact = metrics.get("normalized_exact_accuracy")
    chance_exact = chance_metrics.get("normalized_exact_accuracy")
    counterfactual = metrics.get("counterfactual")
    grounding = metrics.get("grounding")
    performance = _final_performance_summary(metrics, chance_metrics, config)
    counts = {
        field: metrics.get(field)
        for field in (
            "reference_count",
            "prediction_count",
            "missing_prediction_count",
            "extra_prediction_count",
        )
    }
    payload: dict[str, Any] = {
        "schema_version": FINAL_EVIDENCE_SCHEMA_VERSION,
        "kind": "held_out_static_evaluation",
        "passed": bool(
            isinstance(exact, (int, float))
            and not isinstance(exact, bool)
            and isinstance(chance_exact, (int, float))
            and not isinstance(chance_exact, bool)
            and float(exact) > float(chance_exact)
            and counts["reference_count"] == counts["prediction_count"]
            and counts["missing_prediction_count"] == 0
            and counts["extra_prediction_count"] == 0
            and performance["performance_passed"] is True
        ),
        "split": "test",
        "scene_disjoint": True,
        "above_chance": bool(
            isinstance(exact, (int, float))
            and not isinstance(exact, bool)
            and isinstance(chance_exact, (int, float))
            and not isinstance(chance_exact, bool)
            and float(exact) > float(chance_exact)
        ),
        "counterfactual_evaluated": bool(
            isinstance(counterfactual, Mapping)
            and isinstance(counterfactual.get("eligible_pairs"), int)
            and counterfactual["eligible_pairs"] > 0
            and isinstance(counterfactual.get("expected_change_pairs"), int)
            and counterfactual["expected_change_pairs"] > 0
            and counterfactual.get("malformed_pair_groups") == 0
        ),
        "grounding_evaluated": bool(
            isinstance(grounding, Mapping)
            and isinstance(grounding.get("target_count"), int)
            and grounding["target_count"] > 0
            and isinstance(grounding.get("prediction_count"), int)
            and grounding["prediction_count"] > 0
        ),
        "checkpoint_adapter_sha256": sha256_file(adapter),
        "checkpoint_runtime_metadata_sha256": sha256_file(runtime_metadata),
        "runtime_config_file_sha256": runtime_config_file_sha256(config_path),
        **counts,
        "normalized_exact_accuracy": exact,
        "chance_normalized_exact_accuracy": chance_exact,
        **{
            field: performance[field]
            for field in sorted(_FINAL_PERFORMANCE_FIELDS)
        },
        "scene_runtime_manifest": _build_scene_runtime_manifest(
            config, resolved_artifacts["predictions_path"]
        ),
    }
    for path_field, hash_field in _FINAL_EVIDENCE_PATH_FIELDS:
        artifact = resolved_artifacts[path_field]
        payload[path_field] = str(artifact)
        payload[hash_field] = sha256_file(artifact)
    _final_attestation(
        payload,
        runtime_config_path=config_path,
        checkpoint=checkpoint_path,
        adapter_sha256=payload["checkpoint_adapter_sha256"],
        runtime_metadata_sha256=payload["checkpoint_runtime_metadata_sha256"],
        runtime_config_sha256=payload["runtime_config_file_sha256"],
    )
    destination = _unresolved_rooted(output_path)
    if destination.is_symlink() or destination.exists():
        raise FileExistsError(f"Refusing to overwrite held-out final evidence: {destination}")
    _atomic_json(destination.resolve(), payload)
    return payload


def _leakage_attestation(
    report: Mapping[str, Any],
    *,
    runtime_config_path: Path,
    checkpoint: Path,
    adapter_sha256: str,
    runtime_metadata_sha256: str,
    runtime_config_sha256: str,
) -> dict[str, Any]:
    runtime_config = load_runtime_config(runtime_config_path)
    expected_oracle = artifact_root(runtime_config, "oracle").resolve()
    required = {
        "schema_version": 1,
        "passed": True,
        "runtime_config": str(runtime_config_path),
        "oracle_was_renamed": True,
        "oracle_unavailable_during_inference": True,
        "oracle_restored": True,
        "prefix_computed_before_first_question": True,
        "prefix_invariant": True,
        "checkpoint_adapter_sha256": adapter_sha256,
        "checkpoint_runtime_metadata_sha256": runtime_metadata_sha256,
        "runtime_config_file_sha256": runtime_config_sha256,
        "failure": None,
    }
    mismatches = {
        key: {"evidence": report.get(key), "required": value}
        for key, value in required.items()
        if report.get(key) != value
    }
    reported_checkpoint = report.get("checkpoint")
    if not isinstance(reported_checkpoint, str) or _rooted(reported_checkpoint) != checkpoint:
        mismatches["checkpoint"] = {
            "evidence": reported_checkpoint,
            "required": str(checkpoint),
        }
    reported_oracle = report.get("oracle_directory")
    if not isinstance(reported_oracle, str) or _rooted(reported_oracle) != expected_oracle:
        mismatches["oracle_directory"] = {
            "evidence": reported_oracle,
            "required": str(expected_oracle),
        }
    scene_id = report.get("scene_id")
    if not isinstance(scene_id, str) or _OPAQUE_SCENE_ID.fullmatch(scene_id) is None:
        mismatches["scene_id"] = "must be one opaque scene ID"
    if report.get("forbidden_accesses") != []:
        mismatches["forbidden_accesses"] = report.get("forbidden_accesses")
    question_count = report.get("question_count")
    valid_question_count = (
        not isinstance(question_count, bool)
        and isinstance(question_count, int)
        and question_count >= 3
    )
    if not valid_question_count:
        mismatches["question_count"] = "at least three questions are required"
    prefix_hash = report.get("prefix_hash")
    try:
        _require_sha(prefix_hash, "leakage prefix_hash")
    except ValueError as exc:
        mismatches["prefix_hash"] = str(exc)
    prefix_hashes = report.get("prefix_hashes")
    if (
        not isinstance(prefix_hashes, list)
        or not valid_question_count
        or len(prefix_hashes) != question_count + 1
        or set(prefix_hashes) != {prefix_hash}
    ):
        mismatches["prefix_hashes"] = "must contain one invariant pre-question hash plus one per answer"
    answers = report.get("answers")
    answer_questions: list[str] = []
    if (
        not isinstance(answers, list)
        or not valid_question_count
        or len(answers) != question_count
        or any(
            not isinstance(answer, Mapping)
            or answer.get("prefix_hash") != prefix_hash
            or not isinstance(answer.get("question"), str)
            or not answer["question"].strip()
            for answer in answers
        )
    ):
        mismatches["answers"] = (
            "every answer must carry one nonempty question and the invariant "
            "scene-prefix hash"
        )
    else:
        answer_questions = [str(answer["question"]).strip() for answer in answers]
        if len(set(answer_questions)) != question_count:
            mismatches["answer_questions"] = "leakage evidence requires distinct questions"
    loaded_files = report.get("loaded_files")
    if not isinstance(loaded_files, list) or not all(
        isinstance(value, str) for value in loaded_files
    ):
        mismatches["loaded_files"] = "must be a string list"
    else:
        forbidden_names = {"oracle", "qa", "rendered", "features"}
        forbidden_loaded = [
            value
            for value in loaded_files
            if forbidden_names.intersection(part.casefold() for part in Path(value).parts)
        ]
        required_loaded = {
            str(runtime_config_path),
            str(checkpoint / "adapter.safetensors"),
            str(checkpoint / "runtime_metadata.json"),
        }
        missing_loaded = sorted(required_loaded - set(loaded_files))
        if forbidden_loaded:
            mismatches["loaded_files_forbidden"] = forbidden_loaded
        if missing_loaded:
            mismatches["loaded_files_missing_runtime_inputs"] = missing_loaded
    if mismatches:
        raise ValueError(f"Leakage evidence is not promotion-safe: {mismatches}")
    return {
        "question_count": question_count,
        "prefix_hash": prefix_hash,
        "scene_id": scene_id,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def create_chat_promotion(
    *,
    runtime_config_path: str | Path,
    checkpoint: str | Path,
    selector_report_path: str | Path,
    final_evidence_path: str | Path,
    leakage_report_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all offline evidence and atomically create a safe attestation."""

    config = load_runtime_config(runtime_config_path)
    config_path = Path(str(config["_config_path"]))
    adapter, runtime_metadata, default_output = _checkpoint_files(checkpoint)
    checkpoint_path = adapter.parent
    destination = default_output if output_path is None else _rooted(output_path)
    if destination != default_output:
        raise ValueError("Promotion must be written beside its exact checkpoint")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing promotion: {destination}")

    adapter_hash = sha256_file(adapter)
    runtime_metadata_hash = sha256_file(runtime_metadata)
    _validate_checkpoint_runtime_metadata(runtime_metadata)
    config_file_hash = runtime_config_file_sha256(config_path)
    selector_path = _regular_file(selector_report_path, "selector report")
    final_path = _regular_file(final_evidence_path, "held-out final evidence")
    leakage_path = _regular_file(leakage_report_path, "leakage report")
    selector_summary = _selector_attestation(_json_object(selector_path), checkpoint_path)
    final_summary = _final_attestation(
        _json_object(final_path),
        runtime_config_path=config_path,
        checkpoint=checkpoint_path,
        adapter_sha256=adapter_hash,
        runtime_metadata_sha256=runtime_metadata_hash,
        runtime_config_sha256=config_file_hash,
    )
    leakage_summary = _leakage_attestation(
        _json_object(leakage_path),
        runtime_config_path=config_path,
        checkpoint=checkpoint_path,
        adapter_sha256=adapter_hash,
        runtime_metadata_sha256=runtime_metadata_hash,
        runtime_config_sha256=config_file_hash,
    )
    primary_prefixes = final_summary["primary_prefix_hash_by_scene"]
    leakage_scene_id = leakage_summary["scene_id"]
    if primary_prefixes.get(leakage_scene_id) != leakage_summary["prefix_hash"]:
        raise ValueError(
            "Leakage scene-prefix hash does not match the primary held-out "
            f"prediction prefix for {leakage_scene_id}"
        )
    model_snapshot = local_model_snapshot_identity(config)
    payload = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "accepted",
        "checkpoint_adapter_sha256": adapter_hash,
        "checkpoint_runtime_metadata_sha256": runtime_metadata_hash,
        "checkpoint_path_sha256": _path_identity_sha256(checkpoint_path),
        "runtime_config_file_sha256": config_file_hash,
        "runtime_config_effective_sha256": effective_runtime_config_sha256(config),
        "runtime_config_path_sha256": _path_identity_sha256(config_path),
        "model_snapshot_sha256": model_snapshot["tree_sha256"],
        "model_snapshot_file_count": model_snapshot["file_count"],
        "selector_report_sha256": sha256_file(selector_path),
        "final_evidence_sha256": sha256_file(final_path),
        "leakage_report_sha256": sha256_file(leakage_path),
        "selector_selected_update": selector_summary["selected_update"],
        "final_reference_count": final_summary["reference_count"],
        "leakage_question_count": leakage_summary["question_count"],
        "scene_prefix_sha256": leakage_summary["prefix_hash"],
        "scene_runtime_manifest": final_summary["scene_runtime_manifest"],
        "selector_attested": True,
        "held_out_final_attested": True,
        "leakage_attested": True,
    }
    _atomic_json(destination, payload)
    return payload


def validate_chat_promotion(
    checkpoint: str | Path,
    runtime_config_path: str | Path,
    config: Mapping[str, Any] | None = None,
    *,
    record_file: Callable[[str | Path], None] | None = None,
) -> dict[str, Any]:
    """Validate only the safe hash attestation; never open evidence reports."""

    # Always establish the canonical, standalone config path from the explicit
    # argument.  Never trust an in-memory ``_config_path`` marker: doing so could
    # make promotion validation hash/open an arbitrary alias before it failed.
    canonical_config = load_runtime_config(runtime_config_path, record_file=record_file)
    config_path = Path(str(canonical_config["_config_path"]))
    if config is None:
        loaded_config = canonical_config
    else:
        config_path_value = config.get("_config_path")
        loaded_config = validate_runtime_config(config)
        if effective_runtime_config_sha256(loaded_config) != effective_runtime_config_sha256(
            canonical_config
        ):
            raise ValueError("In-memory runtime config does not match the explicit config file")
        if config_path_value is not None and _rooted(str(config_path_value)) != config_path:
            raise ValueError(
                "In-memory runtime config _config_path does not match the explicit config file"
            )
        loaded_config["_config_path"] = str(config_path)
    adapter, runtime_metadata, promotion_path = _checkpoint_files(checkpoint)
    if promotion_path.is_symlink():
        raise ValueError("Promotion attestation must not be a symbolic link")
    if not promotion_path.is_file():
        raise FileNotFoundError(f"Checkpoint is not behaviorally promoted: {promotion_path}")
    if record_file is not None:
        record_file(promotion_path)
    promotion = _json_object(promotion_path)
    expected_keys = {
        "schema_version",
        "status",
        "checkpoint_adapter_sha256",
        "checkpoint_runtime_metadata_sha256",
        "checkpoint_path_sha256",
        "runtime_config_file_sha256",
        "runtime_config_effective_sha256",
        "runtime_config_path_sha256",
        "model_snapshot_sha256",
        "model_snapshot_file_count",
        "selector_report_sha256",
        "final_evidence_sha256",
        "leakage_report_sha256",
        "selector_selected_update",
        "final_reference_count",
        "leakage_question_count",
        "scene_prefix_sha256",
        "scene_runtime_manifest",
        "selector_attested",
        "held_out_final_attested",
        "leakage_attested",
    }
    if unknown := sorted(set(promotion) - expected_keys):
        raise ValueError(f"Promotion contains forbidden fields: {unknown}")
    if missing := sorted(expected_keys - set(promotion)):
        raise ValueError(f"Promotion is missing required fields: {missing}")
    model_snapshot = local_model_snapshot_identity(
        loaded_config,
        record_file=record_file,
    )
    expected = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "accepted",
        "checkpoint_adapter_sha256": sha256_file(adapter),
        "checkpoint_runtime_metadata_sha256": sha256_file(runtime_metadata),
        "checkpoint_path_sha256": _path_identity_sha256(checkpoint),
        "runtime_config_file_sha256": runtime_config_file_sha256(config_path),
        "runtime_config_effective_sha256": effective_runtime_config_sha256(loaded_config),
        "runtime_config_path_sha256": _path_identity_sha256(config_path),
        "model_snapshot_sha256": model_snapshot["tree_sha256"],
        "model_snapshot_file_count": model_snapshot["file_count"],
        "selector_attested": True,
        "held_out_final_attested": True,
        "leakage_attested": True,
    }
    mismatches = {
        key: {"promotion": promotion.get(key), "runtime": value}
        for key, value in expected.items()
        if promotion.get(key) != value
    }
    for field in (
        "selector_report_sha256",
        "final_evidence_sha256",
        "leakage_report_sha256",
        "scene_prefix_sha256",
        "model_snapshot_sha256",
    ):
        try:
            _require_sha(promotion.get(field), field)
        except ValueError as exc:
            mismatches[field] = str(exc)
    try:
        parsed_scene_manifest = _validate_scene_runtime_manifest(
            promotion.get("scene_runtime_manifest")
        )
        if promotion.get("scene_prefix_sha256") not in {
            entry["scene_prefix_sha256"] for entry in parsed_scene_manifest.values()
        }:
            mismatches["scene_prefix_sha256"] = (
                "does not match any attested scene prefix"
            )
    except (TypeError, ValueError) as exc:
        mismatches["scene_runtime_manifest"] = str(exc)
    selected_update = promotion.get("selector_selected_update")
    if (
        isinstance(selected_update, bool)
        or not isinstance(selected_update, int)
        or selected_update < 0
    ):
        mismatches["selector_selected_update"] = "must be a nonnegative integer"
    for field in (
        "final_reference_count",
        "leakage_question_count",
        "model_snapshot_file_count",
    ):
        value = promotion.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            mismatches[field] = "must be a positive integer"
    if mismatches:
        raise ValueError(f"Checkpoint promotion is invalid or stale: {mismatches}")
    return promotion


def write_primary_pointer(
    path: str | Path,
    *,
    runtime_config_path: str | Path,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Publish a primary pointer only after validating the exact promotion."""

    unresolved_pointer = _unresolved_rooted(path)
    _reject_symlink_components(unresolved_pointer, "Primary pointer")
    pointer_path = unresolved_pointer.resolve()
    try:
        pointer_path.relative_to((PROJECT_ROOT / "configs/runtime").resolve())
    except ValueError as exc:
        raise ValueError("Primary pointer must live below configs/runtime") from exc
    if pointer_path.exists() or pointer_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing primary pointer: {pointer_path}")
    config = load_runtime_config(runtime_config_path)
    config_path = Path(str(config["_config_path"]))
    promotion = validate_chat_promotion(checkpoint, config_path, config)
    _validate_scene_runtime_manifest(
        promotion["scene_runtime_manifest"], runtime_config=config
    )
    checkpoint_path = _rooted(checkpoint)
    promotion_path = checkpoint_path / PROMOTION_FILENAME
    payload = {
        "schema_version": PRIMARY_POINTER_SCHEMA_VERSION,
        "runtime_config": os.path.relpath(config_path, PROJECT_ROOT),
        "checkpoint": os.path.relpath(checkpoint_path, PROJECT_ROOT),
        "promotion_sha256": sha256_file(promotion_path),
    }
    _atomic_json(pointer_path, payload)
    return payload


def resolve_primary_pointer(
    path: str | Path = PRIMARY_POINTER_DEFAULT,
    *,
    record_file: Callable[[str | Path], None] | None = None,
) -> tuple[Path, Path]:
    unresolved_pointer = _unresolved_rooted(path)
    _reject_symlink_components(unresolved_pointer, "Primary pointer")
    pointer_path = unresolved_pointer.resolve()
    try:
        pointer_path.relative_to((PROJECT_ROOT / "configs/runtime").resolve())
    except ValueError as exc:
        raise ValueError("Primary pointer must live below configs/runtime") from exc
    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"No promoted primary Gemma runtime is installed: {pointer_path}"
        )
    if record_file is not None:
        record_file(pointer_path)
    payload = _json_object(pointer_path)
    expected_keys = {"schema_version", "runtime_config", "checkpoint", "promotion_sha256"}
    if set(payload) != expected_keys:
        raise ValueError("Primary pointer has an invalid field set")
    if payload.get("schema_version") != PRIMARY_POINTER_SCHEMA_VERSION:
        raise ValueError("Primary pointer schema version is unsupported")
    config_value = payload.get("runtime_config")
    checkpoint_value = payload.get("checkpoint")
    if not isinstance(config_value, str) or not isinstance(checkpoint_value, str):
        raise TypeError("Primary pointer paths must be strings")
    config_path = _rooted(config_value)
    config = load_runtime_config(config_path, record_file=record_file)
    validate_chat_promotion(
        checkpoint_value,
        config_path,
        config,
        record_file=record_file,
    )
    checkpoint_path = _rooted(checkpoint_value)
    observed_promotion_hash = sha256_file(checkpoint_path / PROMOTION_FILENAME)
    if payload.get("promotion_sha256") != observed_promotion_hash:
        raise ValueError("Primary pointer promotion hash is stale")
    return config_path, checkpoint_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="Create a promotion from complete evidence")
    create.add_argument("--runtime-config", required=True)
    create.add_argument("--checkpoint", required=True)
    create.add_argument("--selector-report", required=True)
    create.add_argument("--final-evidence", required=True)
    create.add_argument("--leakage-report", required=True)
    create.add_argument("--primary-pointer")
    final = subparsers.add_parser(
        "create-final-evidence",
        help="Build a held-out attestation from complete scored artifacts",
    )
    final.add_argument("--runtime-config", required=True)
    final.add_argument("--checkpoint", required=True)
    final.add_argument("--metrics", required=True)
    final.add_argument("--predictions", required=True)
    final.add_argument("--prediction-provenance", required=True)
    final.add_argument("--chance-metrics", required=True)
    final.add_argument("--chance-predictions", required=True)
    final.add_argument("--chance-prediction-provenance", required=True)
    final.add_argument("--split-manifest", required=True)
    final.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate", help="Validate a promoted checkpoint")
    validate.add_argument("--runtime-config")
    validate.add_argument("--checkpoint")
    validate.add_argument("--primary-pointer", default=str(PRIMARY_POINTER_DEFAULT))
    resolve = subparsers.add_parser("resolve-primary", help="Resolve the promoted primary")
    resolve.add_argument("--primary-pointer", default=str(PRIMARY_POINTER_DEFAULT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create-final-evidence":
        evidence = create_held_out_final_evidence(
            runtime_config_path=args.runtime_config,
            checkpoint=args.checkpoint,
            metrics_path=args.metrics,
            predictions_path=args.predictions,
            prediction_provenance_path=args.prediction_provenance,
            chance_metrics_path=args.chance_metrics,
            chance_predictions_path=args.chance_predictions,
            chance_prediction_provenance_path=args.chance_prediction_provenance,
            split_manifest_path=args.split_manifest,
            output_path=args.output,
        )
        print(json.dumps(evidence, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "create":
        promotion = create_chat_promotion(
            runtime_config_path=args.runtime_config,
            checkpoint=args.checkpoint,
            selector_report_path=args.selector_report,
            final_evidence_path=args.final_evidence,
            leakage_report_path=args.leakage_report,
        )
        if args.primary_pointer:
            write_primary_pointer(
                args.primary_pointer,
                runtime_config_path=args.runtime_config,
                checkpoint=args.checkpoint,
            )
        print(json.dumps(promotion, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "resolve-primary":
        config_path, checkpoint_path = resolve_primary_pointer(args.primary_pointer)
        print(
            json.dumps(
                {"runtime_config": str(config_path), "checkpoint": str(checkpoint_path)},
                sort_keys=True,
            )
        )
        return 0
    if (args.runtime_config is None) != (args.checkpoint is None):
        raise ValueError("Validation requires both --runtime-config and --checkpoint")
    if args.runtime_config is None:
        config_path, checkpoint_path = resolve_primary_pointer(args.primary_pointer)
        config = load_runtime_config(config_path)
    else:
        config_path = _rooted(args.runtime_config)
        checkpoint_path = _rooted(args.checkpoint)
        config = load_runtime_config(config_path)
    promotion = validate_chat_promotion(checkpoint_path, config_path, config)
    print(json.dumps(promotion, sort_keys=True, allow_nan=False))
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    """Render expected fail-closed refusals without an implementation traceback."""

    try:
        return main(argv)
    except (OSError, TypeError, ValueError) as exc:
        print(f"Gemma chat promotion refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())


__all__ = [
    "PRIMARY_POINTER_DEFAULT",
    "PROMOTION_FILENAME",
    "cli",
    "create_chat_promotion",
    "create_held_out_final_evidence",
    "resolve_primary_pointer",
    "sha256_file",
    "validate_chat_promotion",
    "write_primary_pointer",
]

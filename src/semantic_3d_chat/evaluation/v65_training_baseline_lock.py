"""Create the strict hash-only V54 baseline lock for V65 training retention."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.prediction_artifacts import (
    PROVENANCE_SCHEMA_VERSION,
    checkpoint_fingerprint,
    effective_config_sha256,
    scene_map_manifest_sha256,
    validate_scene_map_manifest,
)
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest
from semantic_3d_chat.training.train_question_control_v56 import _sha256_file
from semantic_3d_chat.training.train_question_control_v65 import (
    _EXPECTED_ROWS,
    _EXPECTED_SCENES,
    _PINNED_TRAINING_QUESTIONS_SHA256,
    _PINNED_TRAINING_SCENE_MAP_MANIFEST_SHA256,
    _PINNED_V54_CHECKPOINT_SHA256,
    _PINNED_V54_TRAINING_PREDICTIONS_SHA256,
    _PINNED_V54_TRAINING_PROVENANCE_SHA256,
    _TRAINING_BASELINE_SCHEMA,
    _canonical_sha256,
    _opaque_key_inventory_sha256,
    _write_new_json,
    validate_training_baseline_lock,
)

_PINNED_TRAINING_QUESTIONS_CONTENT_SHA256 = (
    "efb1c20491b03f807b3ab510123806aab66e2295d391f495bbbaf97c245f5ade"
)
_PINNED_FILTERED_TRAIN_SHA256 = "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1"
_PINNED_V54_RUNTIME_CONFIG_FILE_SHA256 = (
    "891c58faaaa5fcd2ed76c7e3871f14c5d8c5ae2e05d9fa4ddd5193773d40e56b"
)
_PINNED_V54_RUNTIME_CONFIG_EFFECTIVE_SHA256 = (
    "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
)
_V54_RUNTIME_CONFIG = PROJECT_ROOT / "configs/runtime/gemma4_v54.yaml"
_EXPECTED_TRAINING_SCENES = frozenset(
    f"scene_{number:06d}" for number in (*range(11, 25), *range(31, 39), 53, 54)
)
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_QUESTION_ID = re.compile(r"q_[0-9]{6}")
_PREDICTION_FIELDS = frozenset(
    {
        "elapsed_seconds",
        "generated_tokens",
        "grounding_confidence",
        "grounding_xyz",
        "predicted_answer",
        "prefix_hash",
        "provenance_sha256",
        "question_id",
        "scene_id",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "config_sha256",
        "config_file_sha256",
        "checkpoint_sha256",
        "references_sha256",
        "scene_map_manifest_sha256",
        "split",
        "run_kind",
        "condition",
        "provenance_sha256",
        "config_path",
        "checkpoint_path",
        "checkpoint_files",
        "references_path",
        "scene_map_manifest",
    }
)
_PROVENANCE_IDENTITY_FIELDS = (
    "schema_version",
    "config_sha256",
    "config_file_sha256",
    "checkpoint_sha256",
    "references_sha256",
    "scene_map_manifest_sha256",
    "split",
    "run_kind",
    "condition",
)


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _question_inventory(path: Path) -> tuple[tuple[str, str], ...]:
    manifest = load_question_manifest(path)
    if (
        manifest.manifest_sha256 != _PINNED_TRAINING_QUESTIONS_SHA256
        or manifest.questions_sha256 != _PINNED_TRAINING_QUESTIONS_CONTENT_SHA256
        or manifest.source_qa_sha256 != _PINNED_FILTERED_TRAIN_SHA256
        or manifest.question_count != _EXPECTED_ROWS
        or manifest.scene_count != _EXPECTED_SCENES
    ):
        raise ValueError("V65 training-question inventory changed")
    keys = tuple((record.scene_id, record.question_id) for record in manifest.questions)
    scene_counts = Counter(scene_id for scene_id, _question_id in keys)
    if set(scene_counts) != _EXPECTED_TRAINING_SCENES or set(scene_counts.values()) != {24}:
        raise ValueError("V65 training questions lost exact 24-by-24 scene coverage")
    return keys


def _canonical_compact_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_prediction_provenance(
    provenance: Mapping[str, Any],
    *,
    question_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_files: Sequence[Mapping[str, Any]],
    expected_scenes: set[str],
) -> tuple[str, str]:
    """Authenticate the complete V54 prediction identity before hashing outputs."""

    if set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("V65 prediction provenance fields changed")
    runtime_config = _V54_RUNTIME_CONFIG.resolve()
    if (
        not runtime_config.is_file()
        or runtime_config.is_symlink()
        or _sha256_file(runtime_config) != _PINNED_V54_RUNTIME_CONFIG_FILE_SHA256
        or effective_config_sha256(load_config(runtime_config))
        != _PINNED_V54_RUNTIME_CONFIG_EFFECTIVE_SHA256
    ):
        raise ValueError("V65 pinned V54 runtime configuration changed")
    try:
        provenance_config = Path(str(provenance["config_path"])).expanduser().resolve()
        provenance_questions = Path(str(provenance["references_path"])).expanduser().resolve()
        provenance_checkpoint = Path(str(provenance["checkpoint_path"])).expanduser().resolve()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("V65 prediction provenance paths are invalid") from error
    scene_manifest = validate_scene_map_manifest(provenance.get("scene_map_manifest"))
    observed_scene_manifest_sha256 = scene_map_manifest_sha256(scene_manifest)
    identity = {field: provenance.get(field) for field in _PROVENANCE_IDENTITY_FIELDS}
    identity_sha256 = _canonical_compact_sha256(identity)
    if (
        provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION
        or provenance.get("config_file_sha256") != _PINNED_V54_RUNTIME_CONFIG_FILE_SHA256
        or provenance.get("config_sha256") != _PINNED_V54_RUNTIME_CONFIG_EFFECTIVE_SHA256
        or provenance.get("references_sha256") != _PINNED_TRAINING_QUESTIONS_SHA256
        or provenance.get("checkpoint_sha256") != checkpoint_sha256
        or provenance.get("checkpoint_files") != list(checkpoint_files)
        or provenance.get("scene_map_manifest_sha256") != observed_scene_manifest_sha256
        or provenance.get("split") != "train"
        or provenance.get("condition") != "all_questions"
        or provenance.get("run_kind") != "continuous_scene_static"
        or provenance.get("provenance_sha256") != identity_sha256
        or provenance_config != runtime_config
        or provenance_questions != question_path
        or provenance_checkpoint != checkpoint_path
        or set(scene_manifest) != expected_scenes
    ):
        raise ValueError(
            "V65 baseline provenance is not bound to exact questions, V54 runtime, "
            "checkpoint, and 24 training maps"
        )
    return identity_sha256, observed_scene_manifest_sha256


def _prediction_inventory(
    path: Path,
    *,
    expected_keys: Sequence[tuple[str, str]],
    expected_provenance_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("V65 baseline predictions are unavailable")
    records: list[dict[str, str]] = []
    prefixes: defaultdict[str, set[str]] = defaultdict(set)
    keys: set[tuple[str, str]] = set()
    ordered_keys: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"V65 prediction line {line_number} must be an object")
            if set(value) != _PREDICTION_FIELDS:
                raise ValueError(f"V65 prediction line {line_number} fields changed")
            scene_id = value["scene_id"]
            question_id = value["question_id"]
            answer = value["predicted_answer"]
            prefix_hash = value["prefix_hash"]
            if (
                not isinstance(scene_id, str)
                or _SCENE_ID.fullmatch(scene_id) is None
                or not isinstance(question_id, str)
                or _QUESTION_ID.fullmatch(question_id) is None
                or not isinstance(answer, str)
            ):
                raise TypeError("V65 prediction identity/answer must be strings")
            if (
                not isinstance(prefix_hash, str)
                or len(prefix_hash) != 64
                or any(character not in "0123456789abcdef" for character in prefix_hash)
            ):
                raise ValueError("V65 prediction prefix hash is invalid")
            key = str(scene_id), str(question_id)
            if key in keys:
                raise ValueError("V65 predictions contain a duplicate key")
            if value["provenance_sha256"] != expected_provenance_sha256:
                raise ValueError("V65 prediction row has the wrong provenance identity")
            keys.add(key)
            ordered_keys.append(key)
            prefixes[str(scene_id)].add(prefix_hash)
            records.append(
                {
                    "scene_id": str(scene_id),
                    "question_id": str(question_id),
                    "raw_output_sha256": hashlib.sha256(str(answer).encode("utf-8")).hexdigest(),
                }
            )
    if ordered_keys != list(expected_keys) or len(records) != _EXPECTED_ROWS:
        raise ValueError("V65 predictions differ from exact training questions")
    if any(len(values) != 1 for values in prefixes.values()):
        raise ValueError("V65 baseline prefix changed between questions")
    fixed = {scene: next(iter(values)) for scene, values in sorted(prefixes.items())}
    if len(fixed) != _EXPECTED_SCENES or len(set(fixed.values())) != _EXPECTED_SCENES:
        raise ValueError("V65 baseline prefixes are not scene-distinct")
    return records, fixed


def build_training_baseline_lock(
    *,
    questions: str | Path,
    predictions: str | Path,
    v54_checkpoint: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    question_path = Path(questions).expanduser().resolve()
    prediction_path = Path(predictions).expanduser().resolve()
    provenance_path = prediction_path.with_suffix(prediction_path.suffix + ".provenance.json")
    checkpoint_path = Path(v54_checkpoint).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("V65 training baseline lock is create-once")
    if _sha256_file(question_path) != _PINNED_TRAINING_QUESTIONS_SHA256:
        raise ValueError("V65 training questions differ from the public pin")
    if _sha256_file(prediction_path) != _PINNED_V54_TRAINING_PREDICTIONS_SHA256:
        raise ValueError("V65 V54 training predictions differ from the completed pin")
    keys = _question_inventory(question_path)
    checkpoint_sha256, checkpoint_files = checkpoint_fingerprint(checkpoint_path)
    if checkpoint_sha256 != _PINNED_V54_CHECKPOINT_SHA256:
        raise ValueError("V65 baseline uses the wrong V54 checkpoint")
    provenance = _load_object(provenance_path, label="V65 prediction provenance")
    if _sha256_file(provenance_path) != _PINNED_V54_TRAINING_PROVENANCE_SHA256:
        raise ValueError("V65 V54 training prediction provenance differs from its pin")
    provenance_identity_sha256, map_manifest_sha256 = _validate_prediction_provenance(
        provenance,
        question_path=question_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_files=checkpoint_files,
        expected_scenes={scene for scene, _question in keys},
    )
    if map_manifest_sha256 != _PINNED_TRAINING_SCENE_MAP_MANIFEST_SHA256:
        raise ValueError("V65 training map-manifest identity differs from its pin")
    output_hashes, prefixes = _prediction_inventory(
        prediction_path,
        expected_keys=keys,
        expected_provenance_sha256=provenance_identity_sha256,
    )
    payload = {
        "schema": _TRAINING_BASELINE_SCHEMA,
        "schema_version": 1,
        "artifact": "v65_v54_training_baseline_lock",
        "status": "locked_before_v65_training",
        "questions_manifest_sha256": _PINNED_TRAINING_QUESTIONS_SHA256,
        "predictions_sha256": _sha256_file(prediction_path),
        "prediction_provenance_sha256": _sha256_file(provenance_path),
        "prediction_provenance_identity_sha256": provenance_identity_sha256,
        "v54_runtime_config_file_sha256": _PINNED_V54_RUNTIME_CONFIG_FILE_SHA256,
        "v54_runtime_config_effective_sha256": (_PINNED_V54_RUNTIME_CONFIG_EFFECTIVE_SHA256),
        "scene_map_manifest_sha256": map_manifest_sha256,
        "v54_checkpoint_sha256": checkpoint_sha256,
        "v54_checkpoint_files": checkpoint_files,
        "question_count": len(output_hashes),
        "scene_count": len(prefixes),
        "question_key_inventory_sha256": _opaque_key_inventory_sha256(tuple(keys)),
        "scene_prefix_hashes": prefixes,
        "one_invariant_prefix_per_scene": True,
        "distinct_prefix_per_scene": True,
        "required_output_hashes": output_hashes,
        "required_output_hashes_sha256": _canonical_sha256(output_hashes),
        "answer_or_question_text_stored": False,
        "training_scenes_only": True,
        "validation_inputs_loaded": False,
        "scorer_inputs_loaded": False,
        "oracle_loaded": False,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    _write_new_json(destination, payload)
    validate_training_baseline_lock(destination)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--v54-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_training_baseline_lock(
        questions=args.questions,
        predictions=args.predictions,
        v54_checkpoint=args.v54_checkpoint,
        output=args.output,
    )
    print(json.dumps({"created": True, "question_count": payload["question_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

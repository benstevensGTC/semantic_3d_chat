"""Read-only authentication of the sealed V75 fixed-atlas behavior result.

The predictor is deliberately absent from this module.  Authentication reads
the already-persisted preparation, prediction, and score artifacts, checks
their immutable SHA-256 pins, recomputes the small behavioral table, and emits
a summary on stdout.  It never loads Gemma, a scene map, or a checkpoint and it
never writes an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

PREPARED_ROOT: Final[Path] = Path("reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1")
PREDICTIONS: Final[Path] = Path(
    "reports/gemma4/predictions/v75_fixed_atlas_historical_internal.json"
)
SCORE: Final[Path] = Path("reports/gemma4/metrics/v75_fixed_atlas_historical_internal_score.json")
PREPARED_FILE_SHA256: Final[dict[str, str]] = {
    "predictor/metadata.json": ("5ec72a44af262f1f0587efdb2fce959bac997c5a37fba2449aabbac9746562e1"),
    "predictor/questions.jsonl": (
        "abfc310564799a87ea576769b55e3a303d70c0881c14b8c50eba8eab22ab1537"
    ),
    "probe_bank/probes.safetensors": (
        "fb32c687dd787f108fab03e9745eefb2273891c2be990d0acf50ca111eb637e8"
    ),
    "probe_bank/runtime_metadata.json": (
        "3e736940f4c83b55e96aa5e36f6774fd007454508722f5b25ddc44f298c2518d"
    ),
    "scorer/metadata.json": ("4f02649c3c368b8cbc920a21656db25a3a5f58c04dad27570f065becf69c2ac0"),
    "scorer/references.jsonl": ("d262633aea021c76061dcbe41db21467c41a0860c7d3e24c89d7787939a4d8b6"),
}
PREDICTIONS_SHA256: Final[str] = "b7c70f2ad49b4c212c3fb63391e960879a097a82267e0cd092e4910d751df46f"
SCORE_SHA256: Final[str] = "224886019172c5080f2bd976de74477d40e37db9a5635aae9c9b7697db53dfd2"
PROBE_TENSOR_SHA256: Final[str] = "5731bbabc15b501007af85254a6bf45dbb0de269469348412196c3f24dce8929"
EXPECTED_SCENES: Final[tuple[str, ...]] = (
    "scene_000039",
    "scene_000040",
    "scene_000041",
    "scene_000042",
    "scene_000043",
    "scene_000044",
    "scene_000045",
    "scene_000046",
    "scene_000047",
    "scene_000048",
    "scene_000049",
    "scene_000050",
    "scene_000051",
    "scene_000052",
    "scene_000055",
    "scene_000056",
)
EXPECTED_ARM_RESULTS: Final[dict[str, tuple[int, int, float]]] = {
    "fixed_v75_atlas": (6, 16, 0.375),
    "direct_exact_v75": (9, 16, 0.5625),
    "frozen_v54": (6, 16, 0.375),
}
EXPECTED_CHANGE_UNITS: Final[dict[str, int]] = {
    "fixed_v75_atlas": 1,
    "direct_exact_v75": 2,
    "frozen_v54": 1,
    "total": 8,
}


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    # Do not call resolve(): authentication must still be able to reject a
    # symlink at the supplied evidence path.
    return Path(os.path.abspath(rooted))


def _regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing, non-regular, or a symlink")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(_regular(path, label).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _regular(path, label).read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{label} contains a non-object row")
        rows.append(value)
    return rows


def _safetensors_header(path: Path) -> dict[str, Any]:
    with _regular(path, "numeric probe bank").open("rb") as handle:
        encoded_size = handle.read(8)
        if len(encoded_size) != 8:
            raise ValueError("numeric probe bank has no safetensors header")
        header_size = struct.unpack("<Q", encoded_size)[0]
        if not 2 <= header_size <= 1_000_000:
            raise ValueError("numeric probe bank safetensors header size is invalid")
        encoded_header = handle.read(header_size)
        if len(encoded_header) != header_size:
            raise ValueError("numeric probe bank safetensors header is truncated")
    value = json.loads(encoded_header)
    if not isinstance(value, dict):
        raise TypeError("numeric probe bank safetensors header is not an object")
    return value


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    characters = [
        character if (character.isalnum() or character.isspace()) else " " for character in text
    ]
    return " ".join(
        token for token in "".join(characters).split() if token not in {"a", "an", "the"}
    )


def _canonical_match(answer_type: str, prediction: object, reference: object) -> bool:
    predicted = _normalize(prediction)
    expected = _normalize(reference)
    if answer_type == "presence":
        positive = {"yes", "present", "true"}
        negative = {"no", "absent", "false"}

        def polarity(text: str) -> bool | None:
            tokens = set(text.split())
            has_positive = bool(tokens & positive)
            has_negative = bool(tokens & negative)
            return has_positive if has_positive != has_negative else None

        return polarity(predicted) is not None and polarity(predicted) == polarity(expected)
    return predicted == expected


def _authenticate_preparation(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("prepared artifact root is missing, non-directory, or a symlink")
    inventory = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if inventory != set(PREPARED_FILE_SHA256):
        raise ValueError("prepared artifact file inventory differs from its immutable pin")
    for relative, expected in PREPARED_FILE_SHA256.items():
        source = _regular(root / relative, f"prepared artifact {relative}")
        if _sha256(source) != expected:
            raise ValueError(f"prepared artifact digest differs: {relative}")

    probe_metadata = _read_object(
        root / "probe_bank/runtime_metadata.json", "probe runtime metadata"
    )
    predictor_metadata = _read_object(root / "predictor/metadata.json", "predictor metadata")
    scorer_metadata = _read_object(root / "scorer/metadata.json", "scorer metadata")
    questions = _read_jsonl(root / "predictor/questions.jsonl", "predictor questions")
    references = _read_jsonl(root / "scorer/references.jsonl", "scorer references")
    header = _safetensors_header(root / "probe_bank/probes.safetensors")
    safe_metadata = header.get("__metadata__")
    tensor = header.get("probe_embeddings")
    if (
        not isinstance(safe_metadata, Mapping)
        or safe_metadata
        != {
            "answer_codebook_serialized": "false",
            "artifact": "v75_fixed_atlas_numeric_probe_bank_v1",
            "environmental_text_serialized": "false",
            "questions_or_answers_serialized": "false",
            "runtime_promotion_authorized": "false",
            "schema_version": "1",
            "tensor_name": "probe_embeddings",
        }
        or tensor != {"data_offsets": [0, 589824], "dtype": "F32", "shape": [96, 1536]}
        or set(header) != {"__metadata__", "probe_embeddings"}
    ):
        raise ValueError("numeric probe safetensors structure changed")
    if (
        probe_metadata.get("artifact") != "v75_fixed_atlas_numeric_probe_bank_v1"
        or probe_metadata.get("probe_file_sha256")
        != PREPARED_FILE_SHA256["probe_bank/probes.safetensors"]
        or probe_metadata.get("probe_tensor_sha256") != PROBE_TENSOR_SHA256
        or probe_metadata.get("probe_count") != 96
        or probe_metadata.get("hidden_size") != 1536
        or probe_metadata.get("source_scope") != "v73_historical_optimization_fold_only"
        or probe_metadata.get("source_train_pair_count") != 12
        or probe_metadata.get("source_train_scene_count") != 24
        or probe_metadata.get("source_train_row_count") != 576
        or probe_metadata.get("source_unique_question_count") != 96
        or probe_metadata.get("questions_or_answers_serialized") is not False
        or probe_metadata.get("answer_codebook_serialized") is not False
        or probe_metadata.get("environmental_text_serialized") is not False
        or probe_metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("numeric probe runtime metadata contract changed")
    if (
        predictor_metadata.get("artifact")
        != "v75_fixed_atlas_historical_smoke_predictor_questions_v1"
        or predictor_metadata.get("questions_file_sha256")
        != PREPARED_FILE_SHA256["predictor/questions.jsonl"]
        or predictor_metadata.get("row_count") != 16
        or predictor_metadata.get("scene_count") != 16
        or tuple(predictor_metadata.get("scene_ids", ())) != EXPECTED_SCENES
        or predictor_metadata.get("questions_are_user_text_only") is not True
        or predictor_metadata.get("answers_or_labels_serialized") is not False
        or predictor_metadata.get("oracle_fields_serialized") is not False
        or predictor_metadata.get("pair_or_change_metadata_serialized") is not False
    ):
        raise ValueError("predictor manifest metadata contract changed")
    if (
        scorer_metadata.get("artifact") != "v75_fixed_atlas_historical_smoke_scorer_references_v1"
        or scorer_metadata.get("references_file_sha256")
        != PREPARED_FILE_SHA256["scorer/references.jsonl"]
        or scorer_metadata.get("row_count") != 16
        or scorer_metadata.get("unit_count") != 8
        or scorer_metadata.get("change_family_count") != 8
        or scorer_metadata.get("model_or_runtime_loaded_by_scorer") is not False
        or scorer_metadata.get("physically_separate_from_predictor_questions") is not True
    ):
        raise ValueError("scorer manifest metadata contract changed")
    if (
        len(questions) != 16
        or len(references) != 16
        or any(set(row) != {"row_id", "scene_id", "question"} for row in questions)
        or any(
            set(row) != {"row_id", "answer", "answer_type", "change_type", "unit_id"}
            for row in references
        )
        or {row["row_id"] for row in questions} != {row["row_id"] for row in references}
        or {row["scene_id"] for row in questions} != set(EXPECTED_SCENES)
    ):
        raise ValueError("prepared predictor/scorer row contract changed")
    return questions, references


def _authenticate_prediction(path: Path, questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if _sha256(_regular(path, "prediction artifact")) != PREDICTIONS_SHA256:
        raise ValueError("prediction artifact digest differs from its immutable pin")
    prediction = _read_object(path, "prediction artifact")
    prefix = prediction.get("scene_prefix")
    leakage = prediction.get("leakage")
    records = prediction.get("records")
    if (
        prediction.get("artifact") != "v75_fixed_atlas_historical_internal_predictions_v1"
        or prediction.get("status") != "behavior_measured_not_promoted"
        or prediction.get("execution_valid") is not True
        or prediction.get("row_count") != 16
        or prediction.get("scene_count") != 16
        or prediction.get("runtime_promotion_authorized") is not False
        or prediction.get("behavioral_accuracy_scored_in_predictor") is not False
        or prediction.get("structural_compiler_implies_behavioral_success") is not False
        or prediction.get("arms") != ["fixed_v75_atlas", "direct_exact_v75", "frozen_v54"]
        or not isinstance(prefix, Mapping)
        or prefix.get("base_environment_latents") != 256
        or prefix.get("fixed_atlas_tokens") != 738
        or prefix.get("atlas_memory_tokens") != 480
        or prefix.get("all_scenes_compiled_before_question_manifest_opened") is not True
        or prefix.get("same_compiled_prefix_reused_for_every_question") is not True
        or prefix.get("prefix_hashes_invariant") is not True
        or prefix.get("question_inputs_used_for_compilation") is not False
        or prefix.get("question_dependent_scene_processing") is not False
        or prefix.get("question_dependent_retrieval") is not False
        or prefix.get("semantic_or_spatial_top_k_selection") is not False
        or prefix.get("all_256_base_latents_preserved") is not True
        or prefix.get("every_probe_processed_for_every_scene") is not True
        or not isinstance(leakage, Mapping)
        or leakage.get("forbidden_access_count") != 0
        or leakage.get("forbidden_accesses") != []
        or leakage.get("scorer_reference_files_loaded") is not False
        or leakage.get("oracle_loaded") is not False
        or leakage.get("training_artifacts_loaded") is not False
        or leakage.get("official_validation_loaded") is not False
        or leakage.get("official_test_loaded") is not False
        or leakage.get("deferred_final_loaded") is not False
        or not isinstance(records, list)
        or len(records) != 16
    ):
        raise ValueError("prediction structural, isolation, or no-promotion contract changed")
    before = prefix.get("prefix_hashes_before")
    after = prefix.get("prefix_hashes_after")
    if not isinstance(before, Mapping) or before != after or set(before) != set(EXPECTED_SCENES):
        raise ValueError("scene-prefix invariance evidence changed")
    question_identity = [(row["row_id"], row["scene_id"]) for row in questions]
    record_identity = [(row.get("row_id"), row.get("scene_id")) for row in records]
    required = {
        "row_id",
        "scene_id",
        "atlas_prefix_sha256",
        "base_prefix_sha256",
        "atlas_prediction",
        "direct_v75_prediction",
        "v54_prediction",
        "direct_v75_control_rms",
        "elapsed_seconds",
    }
    if question_identity != record_identity or any(
        not isinstance(row, Mapping)
        or set(row) != required
        or row["atlas_prefix_sha256"] != before[row["scene_id"]]
        for row in records
    ):
        raise ValueError("prediction row identity or fixed-prefix binding changed")
    return prediction


def _recompute(
    prediction: Mapping[str, Any], references: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, tuple[int, int, float]], dict[str, int]]:
    by_id = {str(row["row_id"]): row for row in references}
    fields = {
        "fixed_v75_atlas": "atlas_prediction",
        "direct_exact_v75": "direct_v75_prediction",
        "frozen_v54": "v54_prediction",
    }
    outcomes: dict[str, tuple[int, int, float]] = {}
    changes: dict[str, int] = {}
    for arm, field in fields.items():
        correct = 0
        by_unit: defaultdict[str, list[str]] = defaultdict(list)
        for record in prediction["records"]:
            reference = by_id[str(record["row_id"])]
            correct += _canonical_match(
                str(reference["answer_type"]), record[field], reference["answer"]
            )
            by_unit[str(reference["unit_id"])].append(_normalize(record[field]))
        if len(by_unit) != 8 or any(len(values) != 2 for values in by_unit.values()):
            raise ValueError("prediction change-unit inventory changed")
        outcomes[arm] = (correct, 16, correct / 16)
        changes[arm] = sum(values[0] != values[1] for values in by_unit.values())
    changes["total"] = 8
    return outcomes, changes


def _authenticate_score(
    path: Path,
    prediction: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if _sha256(_regular(path, "score artifact")) != SCORE_SHA256:
        raise ValueError("score artifact digest differs from its immutable pin")
    score = _read_object(path, "score artifact")
    recomputed_arms, recomputed_changes = _recompute(prediction, references)
    if recomputed_arms != EXPECTED_ARM_RESULTS or recomputed_changes != EXPECTED_CHANGE_UNITS:
        raise ValueError("predictions no longer recompute to the sealed behavioral result")
    for arm, (correct, total, accuracy) in EXPECTED_ARM_RESULTS.items():
        value = score.get(arm)
        if (
            not isinstance(value, Mapping)
            or value.get("correct") != correct
            or value.get("total") != total
            or value.get("accuracy") != accuracy
        ):
            raise ValueError(f"score arm changed: {arm}")
    scope = score.get("scope")
    if (
        score.get("artifact") != "v75_fixed_atlas_historical_internal_score_v1"
        or score.get("status") != "behavior_measured_not_promoted"
        or score.get("execution_valid") is not True
        or score.get("prediction_artifact_sha256") != PREDICTIONS_SHA256
        or score.get("reference_artifact_sha256") != PREPARED_FILE_SHA256["scorer/references.jsonl"]
        or score.get("prediction_change_units") != EXPECTED_CHANGE_UNITS
        or score.get("fixed_atlas_accuracy_gain_over_v54") != 0.0
        or score.get("fixed_atlas_accuracy_gap_to_direct_v75") != -0.1875
        or score.get("prefix_invariance_passed") is not True
        or score.get("predictor_reference_isolation_passed") is not True
        or score.get("behavioral_accuracy_measured") is not True
        or score.get("structural_compiler_implies_behavioral_success") is not False
        or score.get("runtime_promotion_authorized") is not False
        or score.get("protected_evaluation_authorized") is not False
        or not isinstance(scope, Mapping)
        or scope.get("historical_training_pool_only") is not True
        or scope.get("pair_disjoint") is not True
        or scope.get("scene_disjoint") is not True
        or scope.get("question_disjoint") is not False
        or any(
            scope.get(field) is not False
            for field in (
                "official_validation_loaded",
                "official_test_loaded",
                "deferred_final_loaded",
                "oracle_loaded",
            )
        )
    ):
        raise ValueError("score provenance, scope, or conclusion contract changed")
    return score


def _authenticate(
    prepared_root: str | Path,
    predictions_path: str | Path,
    score_path: str | Path,
) -> dict[str, Any]:
    root = _absolute(prepared_root)
    questions, references = _authenticate_preparation(root)
    prediction = _authenticate_prediction(_absolute(predictions_path), questions)
    score = _authenticate_score(_absolute(score_path), prediction, references)
    return {
        "artifact": "v75_fixed_atlas_behavior_result_authentication_v1",
        "status": "authenticated_behavior_measured_not_promoted",
        "measurement_authenticated": True,
        "filesystem_read_only": True,
        "model_loaded": False,
        "inference_executed": False,
        "scene_map_loaded": False,
        "checkpoint_loaded": False,
        "prepared_file_sha256": dict(PREPARED_FILE_SHA256),
        "prediction_sha256": PREDICTIONS_SHA256,
        "score_sha256": SCORE_SHA256,
        "prefix_invariance_passed": True,
        "predictor_reference_isolation_passed": True,
        "forbidden_runtime_access_count": 0,
        "fixed_v75_atlas": dict(score["fixed_v75_atlas"]),
        "direct_exact_v75": dict(score["direct_exact_v75"]),
        "frozen_v54": dict(score["frozen_v54"]),
        "prediction_change_units": dict(score["prediction_change_units"]),
        "conclusion": (
            "fixed atlas tied frozen V54 and trailed direct exact V75; no runtime promotion"
        ),
        "runtime_promotion_authorized": False,
    }


def authenticate_v75_fixed_atlas_behavior_result(
    prepared_root: str | Path = PREPARED_ROOT,
    predictions_path: str | Path = PREDICTIONS,
    score_path: str | Path = SCORE,
) -> dict[str, Any]:
    """Fail closed unless every persisted V75 fixed-atlas artifact authenticates."""

    try:
        return _authenticate(prepared_root, predictions_path, score_path)
    except (KeyError, OSError, TypeError, ValueError, struct.error) as error:
        return {
            "artifact": "v75_fixed_atlas_behavior_result_authentication_v1",
            "status": "authentication_failed",
            "measurement_authenticated": False,
            "filesystem_read_only": True,
            "model_loaded": False,
            "inference_executed": False,
            "runtime_promotion_authorized": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", default=str(PREPARED_ROOT))
    parser.add_argument("--predictions", default=str(PREDICTIONS))
    parser.add_argument("--score", default=str(SCORE))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = authenticate_v75_fixed_atlas_behavior_result(
        args.prepared_root, args.predictions, args.score
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("measurement_authenticated") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PREDICTIONS_SHA256",
    "PREPARED_FILE_SHA256",
    "SCORE_SHA256",
    "authenticate_v75_fixed_atlas_behavior_result",
]

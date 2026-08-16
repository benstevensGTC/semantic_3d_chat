"""Create sanitized artifacts for the V75 fixed-prefix behavioral diagnostic.

This is an offline, historical-training-pool preparation step.  It derives a
fixed bank of numeric Gemma question embeddings from the locked V73
optimization fold and creates physically separated predictor and scorer
manifests for one pair- and scene-disjoint 16-row smoke.  The probe checkpoint
contains no question strings, answers, labels, object names, or codebook.

Nothing in this module authorizes runtime promotion or protected-split use.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import LIST_ANSWER_TYPES
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.training.train_question_control_v73 import (
    EXPECTED_HELD_SCENES,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_TRAIN_ROWS,
    HELD_PAIR_IDS,
    TRAIN_PAIR_IDS,
    RowV73,
    changed_units_v73,
    load_config_v73,
    load_embedding_assets_v73,
    load_training_rows_v73,
    split_rows_v73,
)

ARTIFACT_SCHEMA: Final[int] = 1
PROBE_COUNT: Final[int] = 96
SMOKE_ROW_COUNT: Final[int] = 16
SMOKE_SCENE_COUNT: Final[int] = 16
SMOKE_FAMILY_COUNT: Final[int] = 8
SOURCE_QA_SHA256: Final[str] = (
    "01721bf904b1ab0b65ce8acac6e366287040873cda1356da6c70c4981abe7619"
)
SOURCE_V73_CONFIG_SHA256: Final[str] = (
    "d208f28380e3f1810a688be8ea8a263831b6a741f7f90e667795637f39d841f1"
)
GEMMA_MODEL_FILE_SHA256: Final[str] = (
    "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
)
GEMMA_REVISION: Final[str] = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
EMBEDDING_TENSOR_NAME: Final[str] = "model.language_model.embed_tokens.weight"

_PREP_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "source_v73_config",
        "output_root",
        "expected_source_qa_sha256",
        "expected_v73_config_sha256",
        "expected_model_file_sha256",
        "expected_model_revision",
        "probe_count",
        "smoke_row_count",
        "smoke_scene_count",
        "scope",
    }
)
_SCOPE = {
    "historical_training_pool_only": True,
    "pair_disjoint_smoke": True,
    "scene_disjoint_smoke": True,
    "question_disjoint_smoke": False,
    "official_validation_loaded": False,
    "official_test_loaded": False,
    "deferred_final_loaded": False,
    "oracle_loaded": False,
    "runtime_promotion_authorized": False,
}
_PROBE_SAFE_METADATA = {
    "artifact": "v75_fixed_atlas_numeric_probe_bank_v1",
    "schema_version": "1",
    "tensor_name": "probe_embeddings",
    "questions_or_answers_serialized": "false",
    "answer_codebook_serialized": "false",
    "environmental_text_serialized": "false",
    "runtime_promotion_authorized": "false",
}
_PROBE_RUNTIME_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "probe_file_sha256",
        "probe_tensor_sha256",
        "probe_count",
        "hidden_size",
        "dtype",
        "source_scope",
        "source_train_pair_count",
        "source_train_scene_count",
        "source_train_row_count",
        "source_unique_question_count",
        "source_question_hash_inventory_sha256",
        "source_qa_sha256",
        "source_v73_config_sha256",
        "model_revision",
        "model_file_sha256",
        "embedding_tensor_name",
        "pooling",
        "probe_order",
        "questions_or_answers_serialized",
        "answer_codebook_serialized",
        "environmental_text_serialized",
        "official_validation_loaded",
        "official_test_loaded",
        "deferred_final_loaded",
        "oracle_loaded",
        "runtime_promotion_authorized",
    }
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def load_prepare_config(path: str | Path) -> dict[str, Any]:
    """Load the exact offline preparation contract and reject scope drift."""

    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"V75 atlas preparation config is unavailable: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v75_fixed_atlas_prepare"}:
        raise ValueError("V75 atlas preparation config must contain exactly one mapping")
    config = payload["v75_fixed_atlas_prepare"]
    if not isinstance(config, Mapping) or set(config) != _PREP_CONFIG_KEYS:
        raise ValueError("V75 atlas preparation config fields changed")
    expected = {
        "schema_version": ARTIFACT_SCHEMA,
        "expected_source_qa_sha256": SOURCE_QA_SHA256,
        "expected_v73_config_sha256": SOURCE_V73_CONFIG_SHA256,
        "expected_model_file_sha256": GEMMA_MODEL_FILE_SHA256,
        "expected_model_revision": GEMMA_REVISION,
        "probe_count": PROBE_COUNT,
        "smoke_row_count": SMOKE_ROW_COUNT,
        "smoke_scene_count": SMOKE_SCENE_COUNT,
        "scope": _SCOPE,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"V75 atlas preparation {field} changed")
    if config.get("status") != "historical_internal_diagnostic_not_promoted":
        raise ValueError("V75 atlas preparation status changed")
    output = _resolve(str(config["output_root"]))
    forbidden = {"oracle", "qa", "training", "validation", "test", "deferred", "final"}
    if forbidden.intersection(part.casefold() for part in output.parts):
        raise ValueError("V75 atlas sanitized output crosses a forbidden data boundary")
    return dict(config)


def select_historical_smoke_rows(
    held_rows: Sequence[RowV73],
) -> tuple[RowV73, ...]:
    """Select both sides of the first unit in every held change family."""

    first_by_family: dict[str, Any] = {}
    for unit in changed_units_v73(held_rows):
        first_by_family.setdefault(unit.change_type, unit)
    if len(first_by_family) != SMOKE_FAMILY_COUNT:
        raise ValueError("V75 atlas smoke family inventory changed")
    result = tuple(
        row
        for family in sorted(first_by_family)
        for row in (first_by_family[family].left, first_by_family[family].right)
    )
    if (
        len(result) != SMOKE_ROW_COUNT
        or len({row.key for row in result}) != SMOKE_ROW_COUNT
        or len({row.scene_id for row in result}) != SMOKE_SCENE_COUNT
        or any(row.pair_id not in HELD_PAIR_IDS for row in result)
    ):
        raise RuntimeError("V75 atlas smoke row inventory changed")
    return result


def ordered_probe_questions(train_rows: Sequence[RowV73]) -> tuple[str, ...]:
    """Return the exact 96 unique training-fold questions in opaque hash order."""

    if len(train_rows) != EXPECTED_TRAIN_ROWS or any(
        row.pair_id not in TRAIN_PAIR_IDS for row in train_rows
    ):
        raise ValueError("V75 atlas probes require the complete historical fit fold")
    questions = {row.question for row in train_rows}
    if len(questions) != PROBE_COUNT:
        raise ValueError("V75 atlas unique historical question count changed")
    keyed = [(hashlib.sha256(question.encode("utf-8")).hexdigest(), question) for question in questions]
    if len({digest for digest, _question in keyed}) != len(keyed):
        raise RuntimeError("V75 atlas question SHA-256 collision")
    return tuple(question for _digest, question in sorted(keyed))


def build_numeric_probe_tensor(
    ordered_questions: Sequence[str],
    question_embeddings: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Pool complete token sequences exactly as V75 does at live inference."""

    if len(ordered_questions) != PROBE_COUNT or len(set(ordered_questions)) != PROBE_COUNT:
        raise ValueError("V75 atlas probe question inventory changed")
    if set(ordered_questions) != set(question_embeddings):
        raise ValueError("V75 atlas embedding inventory differs from its questions")
    pooled: list[torch.Tensor] = []
    for question in ordered_questions:
        value = question_embeddings[question]
        if (
            value.ndim != 2
            or value.shape[0] < 1
            or value.shape[1] != EXPECTED_HIDDEN_SIZE
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("V75 atlas source question embedding shape changed")
        pooled.append(value.detach().float().mean(dim=0))
    probes = torch.stack(pooled).float().contiguous()
    if tuple(probes.shape) != (PROBE_COUNT, EXPECTED_HIDDEN_SIZE):
        raise RuntimeError("V75 atlas probe tensor shape changed")
    if not bool(torch.isfinite(probes).all()) or bool((probes.norm(dim=-1) <= 1e-8).any()):
        raise RuntimeError("V75 atlas probe tensor is nonfinite or contains a zero row")
    return probes


def _row_id(row: RowV73) -> str:
    return "row_" + hashlib.sha256(f"{row.scene_id}|{row.question_id}".encode()).hexdigest()[:24]


def _unit_id(row: RowV73) -> str:
    return "unit_" + hashlib.sha256(f"{row.pair_id}|{row.question_key}".encode()).hexdigest()[:24]


def _build_predictor_rows(smoke: Sequence[RowV73]) -> list[dict[str, str]]:
    rows = [
        {"row_id": _row_id(row), "scene_id": row.scene_id, "question": row.question}
        for row in smoke
    ]
    if any(set(row) != {"row_id", "scene_id", "question"} for row in rows):
        raise AssertionError("V75 atlas predictor manifest contains an unexpected field")
    return rows


def _build_reference_rows(smoke: Sequence[RowV73]) -> list[dict[str, str]]:
    if any(row.answer_type in LIST_ANSWER_TYPES for row in smoke):
        raise ValueError("V75 atlas smoke unexpectedly contains list-valued answers")
    return [
        {
            "row_id": _row_id(row),
            "answer": row.answer,
            "answer_type": row.answer_type,
            "change_type": row.change_type,
            "unit_id": _unit_id(row),
        }
        for row in smoke
    ]


def prepare_artifacts(config_path: str | Path) -> dict[str, Any]:
    """Create one no-overwrite probe bank and separated smoke manifests."""

    config = load_prepare_config(config_path)
    v73_path = _resolve(str(config["source_v73_config"]))
    if sha256_file(v73_path) != SOURCE_V73_CONFIG_SHA256:
        raise ValueError("V75 atlas source V73 config changed")
    v73 = load_config_v73(v73_path)
    qa_path = _resolve(v73["training_qa"])
    if sha256_file(qa_path) != SOURCE_QA_SHA256:
        raise ValueError("V75 atlas historical source QA changed")
    all_rows = load_training_rows_v73(qa_path)
    train_rows, held_rows = split_rows_v73(all_rows)
    if len({row.scene_id for row in held_rows}) != EXPECTED_HELD_SCENES:
        raise ValueError("V75 atlas held scene inventory changed")
    smoke = select_historical_smoke_rows(held_rows)
    questions = ordered_probe_questions(train_rows)
    assets = load_embedding_assets_v73(v73["gemma_snapshot"], questions, {})
    if (
        assets.model_file_sha256 != GEMMA_MODEL_FILE_SHA256
        or assets.embedding_tensor_name != EMBEDDING_TENSOR_NAME
        or assets.embedding_shape[1] != EXPECTED_HIDDEN_SIZE
    ):
        raise ValueError("V75 atlas pinned Gemma embedding asset changed")
    probes = build_numeric_probe_tensor(questions, assets.questions)

    destination = _resolve(str(config["output_root"]))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V75 atlas artifact root already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    try:
        probe_root = temporary / "probe_bank"
        predictor_root = temporary / "predictor"
        scorer_root = temporary / "scorer"
        for root in (probe_root, predictor_root, scorer_root):
            root.mkdir()

        probe_path = probe_root / "probes.safetensors"
        save_file(
            {"probe_embeddings": probes},
            probe_path,
            metadata=_PROBE_SAFE_METADATA,
        )
        with safe_open(str(probe_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"probe_embeddings"} or handle.metadata() != _PROBE_SAFE_METADATA:
                raise RuntimeError("V75 atlas serialized probe inventory changed")
        reloaded = load_file(str(probe_path), device="cpu")["probe_embeddings"]
        if not torch.equal(reloaded, probes):
            raise RuntimeError("V75 atlas probe tensor failed exact reload")
        question_hashes = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in questions]
        probe_metadata = {
            "schema_version": ARTIFACT_SCHEMA,
            "artifact": "v75_fixed_atlas_numeric_probe_bank_v1",
            "status": "historical_internal_diagnostic_not_promoted",
            "probe_file_sha256": sha256_file(probe_path),
            "probe_tensor_sha256": tensor_sha256(probes),
            "probe_count": PROBE_COUNT,
            "hidden_size": EXPECTED_HIDDEN_SIZE,
            "dtype": str(probes.dtype),
            "source_scope": "v73_historical_optimization_fold_only",
            "source_train_pair_count": len(TRAIN_PAIR_IDS),
            "source_train_scene_count": len({row.scene_id for row in train_rows}),
            "source_train_row_count": len(train_rows),
            "source_unique_question_count": len(questions),
            "source_question_hash_inventory_sha256": canonical_sha256(question_hashes),
            "source_qa_sha256": SOURCE_QA_SHA256,
            "source_v73_config_sha256": SOURCE_V73_CONFIG_SHA256,
            "model_revision": GEMMA_REVISION,
            "model_file_sha256": GEMMA_MODEL_FILE_SHA256,
            "embedding_tensor_name": EMBEDDING_TENSOR_NAME,
            "pooling": "mean_of_complete_question_token_embedding_sequence",
            "probe_order": "ascending_sha256_of_question_text_not_serialized",
            "questions_or_answers_serialized": False,
            "answer_codebook_serialized": False,
            "environmental_text_serialized": False,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
            "runtime_promotion_authorized": False,
        }
        if set(probe_metadata) != _PROBE_RUNTIME_METADATA_FIELDS:
            raise AssertionError("V75 atlas probe metadata fields changed")
        _write_json(probe_root / "runtime_metadata.json", probe_metadata)

        predictor_rows = _build_predictor_rows(smoke)
        predictor_path = predictor_root / "questions.jsonl"
        _write_jsonl(predictor_path, predictor_rows)
        predictor_metadata = {
            "schema_version": ARTIFACT_SCHEMA,
            "artifact": "v75_fixed_atlas_historical_smoke_predictor_questions_v1",
            "status": "historical_internal_pair_scene_disjoint_smoke",
            "questions_file_sha256": sha256_file(predictor_path),
            "row_count": len(predictor_rows),
            "scene_count": len({row["scene_id"] for row in predictor_rows}),
            "scene_ids": sorted({row["scene_id"] for row in predictor_rows}),
            "questions_are_user_text_only": True,
            "answers_or_labels_serialized": False,
            "oracle_fields_serialized": False,
            "pair_or_change_metadata_serialized": False,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
        }
        _write_json(predictor_root / "metadata.json", predictor_metadata)

        reference_rows = _build_reference_rows(smoke)
        references_path = scorer_root / "references.jsonl"
        _write_jsonl(references_path, reference_rows)
        reference_metadata = {
            "schema_version": ARTIFACT_SCHEMA,
            "artifact": "v75_fixed_atlas_historical_smoke_scorer_references_v1",
            "status": "evaluation_only_never_loaded_by_predictor",
            "references_file_sha256": sha256_file(references_path),
            "row_count": len(reference_rows),
            "unit_count": len({row["unit_id"] for row in reference_rows}),
            "change_family_count": len({row["change_type"] for row in reference_rows}),
            "model_or_runtime_loaded_by_scorer": False,
            "physically_separate_from_predictor_questions": True,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
        }
        _write_json(scorer_root / "metadata.json", reference_metadata)

        os.link(temporary, destination) if temporary.is_file() else os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    result = {
        "artifact": "v75_fixed_atlas_historical_internal_artifact_preparation_v1",
        "output_root": str(destination.relative_to(PROJECT_ROOT)),
        "probe_file_sha256": probe_metadata["probe_file_sha256"],
        "probe_tensor_sha256": probe_metadata["probe_tensor_sha256"],
        "probe_count": PROBE_COUNT,
        "predictor_questions_sha256": predictor_metadata["questions_file_sha256"],
        "scorer_references_sha256": reference_metadata["references_file_sha256"],
        "smoke_rows": SMOKE_ROW_COUNT,
        "smoke_scenes": SMOKE_SCENE_COUNT,
        "historical_train_pairs": len(TRAIN_PAIR_IDS),
        "historical_held_pairs": len(HELD_PAIR_IDS),
        "question_overlap_with_training": sum(
            row.question in set(questions) for row in smoke
        ),
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    return result


__all__ = [
    "ARTIFACT_SCHEMA",
    "GEMMA_MODEL_FILE_SHA256",
    "GEMMA_REVISION",
    "PROBE_COUNT",
    "SMOKE_ROW_COUNT",
    "SMOKE_SCENE_COUNT",
    "SOURCE_QA_SHA256",
    "SOURCE_V73_CONFIG_SHA256",
    "build_numeric_probe_tensor",
    "canonical_sha256",
    "load_prepare_config",
    "ordered_probe_questions",
    "prepare_artifacts",
    "select_historical_smoke_rows",
    "sha256_file",
]

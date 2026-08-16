"""Model-free readiness check for the V75 fixed-atlas diagnostic.

The check authenticates the prepared numeric probes, question-only predictor
manifest, scorer receipt, sixteen cached numeric prefixes, exact V75
controller, minimal V54 runtime checkpoint, sanitized runtime config, and
pinned local Gemma snapshot identity.  It never instantiates or loads Gemma.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v75_fixed_atlas_artifacts import (
    GEMMA_MODEL_FILE_SHA256,
    GEMMA_REVISION,
    load_prepare_config,
)
from semantic_3d_chat.evaluation.v75_fixed_atlas_behavior import (
    HIDDEN_SIZE,
    SCENE_IDS,
    SOURCE_V75_CANDIDATE_SHA256,
    V75_RUNTIME_WEIGHTS_SHA256,
    _guard_regular,
    _load_base_prefixes,
    _load_predictor_questions,
    _load_probe_bank,
    _resolve,
    _sha256_file,
    _strict_json,
    load_behavior_config,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)

V54_ADAPTER_SHA256: Final[str] = (
    "6c627f0a0d9efb7100489c24cbd4acafe10456b79280a64a85399a21cb541daf"
)
V54_METADATA_SHA256: Final[str] = (
    "807515461c71b08c08dfbd08a184a653e791413748530fa69402512eca6f6fdd"
)


def _offline_audit() -> FileAccessAudit:
    forbidden = [
        PROJECT_ROOT / "data_gemma4" / "training",
        PROJECT_ROOT / "data" / "oracle",
        PROJECT_ROOT / "data" / "qa",
        PROJECT_ROOT / "configs" / "benchmarks" / "oracle",
        PROJECT_ROOT
        / "reports"
        / "gemma4"
        / "metrics"
        / "v75_official_validation_score.json",
    ]
    forbidden.extend(PROJECT_ROOT.glob("data*/oracle"))
    forbidden.extend(PROJECT_ROOT.glob("data*/qa"))
    return FileAccessAudit(forbidden, block_forbidden=True)


def _validate_v54_release(root: Path, audit: FileAccessAudit) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V75 atlas V54 release is unavailable: {root}")
    if {path.name for path in root.iterdir()} != {
        "adapter.safetensors",
        "runtime_metadata.json",
    }:
        raise ValueError("V75 atlas V54 release inventory changed")
    adapter = _guard_regular(root / "adapter.safetensors", "V54 adapter")
    metadata = _guard_regular(root / "runtime_metadata.json", "V54 metadata")
    adapter_sha = _sha256_file(adapter, audit)
    metadata_sha = _sha256_file(metadata, audit)
    if adapter_sha != V54_ADAPTER_SHA256 or metadata_sha != V54_METADATA_SHA256:
        raise ValueError("V75 atlas V54 release identity changed")
    return {"adapter_sha256": adapter_sha, "runtime_metadata_sha256": metadata_sha}


def _validate_local_model_identity(
    runtime_config: dict[str, Any], audit: FileAccessAudit
) -> dict[str, Any]:
    vision = runtime_config.get("vision")
    language = runtime_config.get("language")
    if not isinstance(vision, dict) or not isinstance(language, dict):
        raise TypeError("V75 atlas runtime model configuration is incomplete")
    if (
        vision.get("model_id") != "google/gemma-4-E2B-it"
        or language.get("model_id") != "google/gemma-4-E2B-it"
        or vision.get("revision") != GEMMA_REVISION
        or language.get("revision") != GEMMA_REVISION
        or language.get("backend") != "gemma4"
    ):
        raise ValueError("V75 atlas runtime model identity changed")
    snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--google--gemma-4-E2B-it"
        / "snapshots"
        / GEMMA_REVISION
    )
    model_link = snapshot / "model.safetensors"
    if not model_link.exists():
        raise FileNotFoundError("V75 atlas pinned local Gemma model is unavailable")
    resolved = model_link.resolve(strict=True)
    if not resolved.is_file() or resolved.name != GEMMA_MODEL_FILE_SHA256:
        raise ValueError("V75 atlas pinned local Gemma blob identity changed")
    audit.record(resolved)
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (snapshot / name).exists()]
    if missing:
        raise FileNotFoundError(f"V75 atlas pinned model assets are missing: {missing}")
    for name in required:
        audit.record((snapshot / name).resolve(strict=True))
    return {
        "model_id": "google/gemma-4-E2B-it",
        "revision": GEMMA_REVISION,
        "model_blob_sha256_identity": resolved.name,
        "model_loaded": False,
    }


def _validate_scorer_receipt(root: Path, audit: FileAccessAudit) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V75 atlas scorer artifact is unavailable: {root}")
    if {path.name for path in root.iterdir()} != {"references.jsonl", "metadata.json"}:
        raise ValueError("V75 atlas scorer artifact inventory changed")
    metadata = _strict_json(
        _guard_regular(root / "metadata.json", "scorer metadata"), audit
    )
    exact = {
        "schema_version": 1,
        "artifact": "v75_fixed_atlas_historical_smoke_scorer_references_v1",
        "status": "evaluation_only_never_loaded_by_predictor",
        "row_count": 16,
        "unit_count": 8,
        "change_family_count": 8,
        "model_or_runtime_loaded_by_scorer": False,
        "physically_separate_from_predictor_questions": True,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
    }
    if set(metadata) != set(exact) | {"references_file_sha256"} or any(
        metadata.get(field) != value for field, value in exact.items()
    ):
        raise ValueError("V75 atlas scorer receipt metadata changed")
    reference_path = _guard_regular(root / "references.jsonl", "scorer references")
    observed = _sha256_file(reference_path, audit)
    if observed != metadata.get("references_file_sha256"):
        raise ValueError("V75 atlas scorer reference bytes changed")
    return {
        "reference_file_sha256": observed,
        "reference_bytes_hashed_only": True,
        "reference_rows_parsed": False,
        "model_loaded": False,
    }


def preflight(
    behavior_config_path: str | Path,
    prepare_config_path: str | Path,
) -> dict[str, Any]:
    """Authenticate all finite prerequisites without loading the Gemma model."""

    behavior = load_behavior_config(behavior_config_path)
    prepare = load_prepare_config(prepare_config_path)
    prepared_root = _resolve(prepare["output_root"])
    expected_paths = {
        "probe_bank": prepared_root / "probe_bank",
        "predictor_questions": prepared_root / "predictor",
        "scorer_forbidden_root": prepared_root / "scorer",
    }
    for field, expected in expected_paths.items():
        if _resolve(behavior[field]) != expected:
            raise ValueError(f"V75 atlas behavior/preparation {field} differs")
    output = _resolve(behavior["output_predictions"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "V75 atlas prediction output already exists; overwrite is forbidden: "
            f"{output}"
        )

    audit = _offline_audit()
    with audit:
        runtime_config_path = _guard_regular(
            _resolve(behavior["runtime_config"]), "runtime config"
        )
        audit.record(runtime_config_path)
        runtime_config = load_runtime_config(runtime_config_path)
        probes, probe_metadata = _load_probe_bank(expected_paths["probe_bank"], audit)
        prefixes, prefix_manifest = _load_base_prefixes(
            _resolve(behavior["source_prefix_cache"]), SCENE_IDS, audit
        )
        questions, question_metadata = _load_predictor_questions(
            expected_paths["predictor_questions"], audit
        )
        scorer = _validate_scorer_receipt(expected_paths["scorer_forbidden_root"], audit)
        controller, controller_metadata = _load_control_head(
            _resolve(behavior["source_controller"]),
            hidden_size=HIDDEN_SIZE,
            device=torch.device("cpu"),
            audit=audit,
        )
        if (
            type(controller) is not DenseFullSceneContinuousControlV75
            or controller_metadata.get("weights_sha256") != V75_RUNTIME_WEIGHTS_SHA256
            or controller_metadata.get("source_v75_candidate_sha256")
            != SOURCE_V75_CANDIDATE_SHA256
        ):
            raise ValueError("V75 atlas preflight controller identity changed")
        base = _validate_v54_release(_resolve(behavior["base_checkpoint"]), audit)
        model = _validate_local_model_identity(runtime_config, audit)
    audit.assert_clean()

    return {
        "artifact": "v75_fixed_atlas_historical_internal_preflight_v1",
        "passed": True,
        "status": "ready_for_one_bounded_model_bearing_predictor_run",
        "prepared_artifacts_required": True,
        "prepared_artifacts_present": True,
        "prediction_output_absent": True,
        "probe_bank": {
            "shape": list(probes.shape),
            "file_sha256": probe_metadata["probe_file_sha256"],
            "tensor_sha256": probe_metadata["probe_tensor_sha256"],
            "contains_text_or_codebook": False,
        },
        "prefix_cache": {
            "scene_count_loaded": len(prefixes),
            "scene_ids": list(SCENE_IDS),
            "question_inputs_used": prefix_manifest["question_inputs_used"],
            "complete_scene_prefixes": prefix_manifest["complete_scene_prefixes"],
        },
        "predictor_questions": {
            "row_count": len(questions),
            "file_sha256": question_metadata["questions_file_sha256"],
            "answers_or_labels_serialized": False,
        },
        "scorer": scorer,
        "controller": {
            "weights_sha256": controller_metadata["weights_sha256"],
            "source_v75_candidate_sha256": controller_metadata[
                "source_v75_candidate_sha256"
            ],
            "model_loaded": False,
        },
        "base_checkpoint": base,
        "local_model": model,
        "gemma_model_loaded": False,
        "scene_prefix_compiled": False,
        "behavioral_accuracy_measured": False,
        "loaded_file_count": len(audit.unique_paths),
        "forbidden_access_count": len(audit.forbidden_accesses()),
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v75_fixed_prefix_atlas_behavior.yaml",
    )
    parser.add_argument(
        "--prepare-config",
        default="configs/experiments/gemma4_v75_fixed_prefix_atlas_prepare.yaml",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            preflight(args.config, args.prepare_config),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


__all__ = ["main", "preflight"]

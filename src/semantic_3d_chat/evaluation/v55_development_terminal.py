"""Seal the only authorized V55 held-out development evaluation.

The terminal is intentionally model-, map-, and QA-free.  It authenticates
the completed train-only chain and immutable historical development evidence,
then binds the exact code/configuration and one-candidate outputs that may be
used by :mod:`semantic_3d_chat.evaluation.v55_development_selector`.

Creating this artifact does *not* authorize training, another candidate,
final-test access, oracle access, checkpoint mutation, or promotion.  The
selector must create its permanent launch claim before it opens development
QA, a development map, or model checkpoint bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

ARTIFACT: Final[str] = "v55_one_shot_development_terminal"
AUTHORIZATION_ID: Final[str] = "v55_one_shot_development_selector"

V54_REPORT: Final[Path] = Path("reports/gemma4/metrics/v54_semantic_greedy_gate.json")
V54_REPORT_SHA256: Final[str] = (
    "ae3d2ca82a81bd0fa0fb00e4b6b4d87b47019aeeb22001a2bcc43effe2ced048"
)
V54_CHECKPOINT: Final[Path] = Path(
    "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000"
)
V54_CHECKPOINT_FILES: Final[dict[str, str]] = {
    "adapter.safetensors": "6c627f0a0d9efb7100489c24cbd4acafe10456b79280a64a85399a21cb541daf",
    "metadata.json": "db1435f8d38ca587e34dcd55dc4d37532efc0504bfb62bc115838dc0ab7a7ece",
    "runtime_metadata.json": "807515461c71b08c08dfbd08a184a653e791413748530fa69402512eca6f6fdd",
}
V54_TENSOR_HASHES: Final[dict[str, str]] = {
    "full_tensor_state_sha256": "aae9f67451f9f3f4aeb4d1cbb39fc6c9d3cd935521c9a84704b217ac64f18119",
    "authorized_surface_state_sha256": "04cb7ac08c062c921a1711cb87acb57af6fa31637f61cb774cfbcd9a28ce8eef",
    "query_state_sha256": "5144ecc81defa65266e54c3d83a1243d948ebad890cdbed812af1bcc46138249",
    "scene_readout_state_sha256": "4a0b76ada4ba42b076798b91ff8bcbdd414ede1dca2abff75aaba06bbe949baa",
    "frozen_state_sha256": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
}

V29_SELECTION: Final[Path] = Path(
    "reports/gemma4/metrics/v29_diverse_stage_b_selection.json"
)
V29_METRICS: Final[Path] = Path("reports/gemma4/metrics/v29_diverse_validation.json")
V29_PREDICTIONS: Final[Path] = Path(
    "reports/gemma4/predictions/v29_diverse_validation.jsonl"
)
V29_QUESTIONS: Final[Path] = Path(
    "reports/gemma4/questions/v29_diverse_validation.json"
)
V29_ARTIFACT_HASHES: Final[dict[Path, str]] = {
    V29_SELECTION: "d7acbd7173f079f257619510df36ad3c73f953e7cf0123b7bd383ad01ddfe91a",
    V29_METRICS: "21bf97ce8c5afd68d512fa04e5a526701e3dc5e5bed2c9fa745a0dbb0c775e09",
    V29_PREDICTIONS: "25d0cf742c9a0aec409853aa75ddd994e53b1266dd9dddfc4e5b97310f8c8a72",
    V29_QUESTIONS: "a5b6d66fef341c59b4d6ebf9f7071780b1d43447bc12d6c4805678862746a9c6",
}

V47_CONFIG: Final[Path] = Path(
    "configs/experiments/gemma4_diverse28_book_continuation_v47.yaml"
)
V47_CONFIG_SHA256: Final[str] = (
    "6b15813237b217d8daad446c799127215bbb1366f2b442b61e975787efa4f6b7"
)
RUNTIME_CONFIG: Final[Path] = Path("configs/runtime/gemma4_v54.yaml")
RUNTIME_CONFIG_SHA256: Final[str] = (
    "891c58faaaa5fcd2ed76c7e3871f14c5d8c5ae2e05d9fa4ddd5193773d40e56b"
)
PROTECTED_REPORT: Final[Path] = Path(
    "reports/gemma4/metrics/training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
PROTECTED_REPORT_SHA256: Final[str] = (
    "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
)

SELECTOR_SOURCE: Final[Path] = Path(
    "src/semantic_3d_chat/evaluation/v55_development_selector.py"
)
SCORE_SOURCE: Final[Path] = Path(
    "src/semantic_3d_chat/evaluation/v55_development_score.py"
)
SELECTOR_TEST: Final[Path] = Path("tests/test_v55_development_selector.py")
SCORE_TEST: Final[Path] = Path("tests/test_v55_development_score.py")
BOUND_SOURCES: Final[tuple[Path, ...]] = (
    Path("src/semantic_3d_chat/evaluation/v55_development_terminal.py"),
    SELECTOR_SOURCE,
    SCORE_SOURCE,
    SELECTOR_TEST,
    SCORE_TEST,
    Path("src/semantic_3d_chat/evaluation/prepare_questions.py"),
    Path("src/semantic_3d_chat/evaluation/question_manifest.py"),
    Path("src/semantic_3d_chat/evaluation/predict.py"),
    Path("src/semantic_3d_chat/evaluation/prediction_artifacts.py"),
    Path("src/semantic_3d_chat/evaluation/run.py"),
    Path("src/semantic_3d_chat/evaluation/metrics.py"),
    Path("src/semantic_3d_chat/evaluation/baseline_io.py"),
    Path("src/semantic_3d_chat/config.py"),
    Path("src/semantic_3d_chat/device.py"),
    Path("src/semantic_3d_chat/chat/file_audit.py"),
    Path("src/semantic_3d_chat/chat/model_snapshot.py"),
    Path("src/semantic_3d_chat/chat/runtime.py"),
    Path("src/semantic_3d_chat/chat/runtime_config.py"),
    Path("src/semantic_3d_chat/language/generation.py"),
    Path("src/semantic_3d_chat/language/local_lm.py"),
    Path("src/semantic_3d_chat/language/lora.py"),
    Path("src/semantic_3d_chat/language/prefix_injection.py"),
    Path("src/semantic_3d_chat/scene_encoder/block_cross_residual.py"),
    Path("src/semantic_3d_chat/scene_encoder/dense_alignment.py"),
    Path("src/semantic_3d_chat/scene_encoder/dense_sidecar_adapter.py"),
    Path("src/semantic_3d_chat/scene_encoder/global_residual.py"),
    Path("src/semantic_3d_chat/scene_encoder/map_io.py"),
    Path("src/semantic_3d_chat/scene_encoder/perceiver.py"),
    Path("src/semantic_3d_chat/scene_encoder/point_tokens.py"),
    Path("src/semantic_3d_chat/scene_encoder/projector.py"),
    Path("src/semantic_3d_chat/scene_encoder/signed_x_dispatch.py"),
    Path("src/semantic_3d_chat/scene_encoder/signed_x_local_field.py"),
    Path("src/semantic_3d_chat/scene_encoder/signed_x_residual.py"),
    Path("src/semantic_3d_chat/scene_encoder/spatial_blocks.py"),
    Path("src/semantic_3d_chat/training/checkpointing.py"),
    Path("src/semantic_3d_chat/training/losses.py"),
)

DEFAULT_OUTPUT: Final[Path] = Path(
    "reports/gemma4/metrics/v55_development_terminal.json"
)
CLAIM_PATH: Final[Path] = Path(
    "reports/gemma4/metrics/v55_development_launch_claim.json"
)
MODEL_SNAPSHOT_PATH: Final[Path] = Path(
    "reports/gemma4/metrics/v55_development_model_snapshot.json"
)
QUESTIONS_PATH: Final[Path] = Path(
    "reports/gemma4/questions/v55_development_validation.json"
)
PREDICTIONS_PATH: Final[Path] = Path(
    "reports/gemma4/predictions/v55_development_validation.jsonl"
)
PREDICTION_PROVENANCE_PATH: Final[Path] = Path(
    "reports/gemma4/predictions/v55_development_validation.jsonl.provenance.json"
)
SCORE_PATH: Final[Path] = Path(
    "reports/gemma4/metrics/v55_development_score.json"
)
SELECTOR_REPORT_PATH: Final[Path] = Path(
    "reports/gemma4/metrics/v55_development_selector.json"
)

REFERENCE_PATH: Final[Path] = Path("data_diverse28/qa/validation.jsonl")
REFERENCE_SHA256: Final[str] = (
    "67fb14685b3f4cb43f2409db7eb84220ec89d6390205b7bb86eb148b4d4e68b2"
)
QUESTION_MANIFEST_SHA256: Final[str] = (
    "a5b6d66fef341c59b4d6ebf9f7071780b1d43447bc12d6c4805678862746a9c6"
)
QUESTIONS_SHA256: Final[str] = (
    "46517d914868ece84106075c50df0ff1afef1f980f30d8c0f3dd8486bb22ede4"
)
EXPECTED_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(19, 25)
)
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    combined = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(combined))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_file(path: Path, expected: str, label: str) -> None:
    source = _resolve(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"V55 {label} is unavailable or unsafe: {source}")
    observed = _sha256(source)
    if observed != expected:
        raise ValueError(
            f"V55 {label} changed: expected={expected} observed={observed}"
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V55 {label} must be a mapping")
    return value


def _json_object(path: Path) -> Mapping[str, Any]:
    return _mapping(
        json.loads(_resolve(path).read_text(encoding="utf-8")),
        str(path),
    )


def _authenticate_predecessors() -> dict[str, Any]:
    """Authenticate reports/configs only; never open QA, maps, or model files."""

    static = {
        V54_REPORT: V54_REPORT_SHA256,
        V47_CONFIG: V47_CONFIG_SHA256,
        RUNTIME_CONFIG: RUNTIME_CONFIG_SHA256,
        PROTECTED_REPORT: PROTECTED_REPORT_SHA256,
        V29_SELECTION: V29_ARTIFACT_HASHES[V29_SELECTION],
        V29_METRICS: V29_ARTIFACT_HASHES[V29_METRICS],
    }
    for path, digest in static.items():
        _locked_file(path, digest, str(path))

    v54 = _json_object(V54_REPORT)
    checkpoint = _mapping(v54.get("checkpoint"), "V54 checkpoint")
    inventory = _mapping(checkpoint.get("inventory"), "V54 checkpoint inventory")
    v54_checks = {
        "artifact": v54.get("artifact") == "v54_semantic_greedy_gate",
        "passed": v54.get("passed") is True,
        "candidate_checkpoint": checkpoint.get("path") == str(V54_CHECKPOINT),
        "checkpoint_written": checkpoint.get("written") is True,
        "files_exact": inventory.get("file_sha256") == V54_CHECKPOINT_FILES,
        "file_inventory_exact": inventory.get("file_inventory")
        == sorted(V54_CHECKPOINT_FILES),
        "tensors_exact": all(
            inventory.get(field) == digest
            for field, digest in V54_TENSOR_HASHES.items()
        ),
        "tensor_count": inventory.get("tensor_count") == 179,
        "finite": inventory.get("all_tensors_finite") is True,
        "no_optimizer": inventory.get("optimizer_file_written") is False,
        "no_validation_qa": v54.get("validation_qa_loaded") is False,
        "no_validation_maps": v54.get("validation_environment_maps_loaded") is False,
        "no_final": v54.get("final_test_scenes_touched") is False,
        "no_oracle": v54.get("oracle_loaded") is False,
        "no_selector": v54.get("selector_executed") is False,
        "no_promotion": v54.get("chat_promotion_executed") is False,
    }
    if not all(v54_checks.values()):
        raise ValueError(f"V55 V54 predecessor contract changed: {v54_checks}")

    v29_selection = _json_object(V29_SELECTION)
    v29_metrics = _json_object(V29_METRICS)
    v29_checks = {
        "selection_passed": v29_selection.get("passed") is True,
        "historical_selected_update": v29_selection.get("selected_update") == 4,
        "coverage": v29_metrics.get("reference_count") == 216
        and v29_metrics.get("prediction_count") == 216
        and v29_metrics.get("missing_prediction_count") == 0
        and v29_metrics.get("extra_prediction_count") == 0,
        "literal_accuracy": v29_metrics.get("normalized_exact_accuracy") == 0.375,
        "reference_hash": v29_metrics.get("references_sha256") == REFERENCE_SHA256,
        "historical_question_manifest_hash_precommitted": (
            V29_ARTIFACT_HASHES[V29_QUESTIONS] == QUESTION_MANIFEST_SHA256
        ),
        "historical_prediction_hash_precommitted": (
            V29_ARTIFACT_HASHES[V29_PREDICTIONS]
            == "25d0cf742c9a0aec409853aa75ddd994e53b1266dd9dddfc4e5b97310f8c8a72"
        ),
    }
    if not all(v29_checks.values()):
        raise ValueError(f"V55 immutable V29 comparator changed: {v29_checks}")
    return {
        "v54_checks": v54_checks,
        "v29_checks": v29_checks,
        "model_loaded": False,
        "map_loaded": False,
        "qa_loaded": False,
    }


def build_terminal_payload() -> dict[str, Any]:
    predecessor = _authenticate_predecessors()
    source_hashes: dict[str, str] = {}
    for path in BOUND_SOURCES:
        source = _resolve(path)
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"V55 bound source is unavailable: {source}")
        source_hashes[str(path)] = _sha256(source)
    outputs = (
        CLAIM_PATH,
        MODEL_SNAPSHOT_PATH,
        QUESTIONS_PATH,
        PREDICTIONS_PATH,
        PREDICTION_PROVENANCE_PATH,
        SCORE_PATH,
        SELECTOR_REPORT_PATH,
    )
    existing = [str(path) for path in outputs if _resolve(path).exists()]
    if existing:
        raise FileExistsError(
            "V55 one-shot output already exists before terminal sealing: "
            f"{existing}"
        )
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "passed": True,
        "terminal_materialization_authorized": True,
        "authorization": {
            "authorization_id": AUTHORIZATION_ID,
            "only_exact_action": "one_candidate_one_shot_development_selection",
            "explicit_terminal_sha256_required": True,
            "candidate": {
                "checkpoint": str(V54_CHECKPOINT),
                "selected_update": 0,
                "selected_optimizer_step": 0,
                "file_sha256": dict(V54_CHECKPOINT_FILES),
                "tensor_state_sha256": dict(V54_TENSOR_HASHES),
            },
            "development": {
                "split": "validation",
                "scene_ids": list(EXPECTED_SCENES),
                "reference_path": str(REFERENCE_PATH),
                "reference_sha256": REFERENCE_SHA256,
                "question_count": 216,
                "changed_unit_count": 12,
                "changed_side_count": 24,
                "question_manifest_sha256": QUESTION_MANIFEST_SHA256,
                "questions_sha256": QUESTIONS_SHA256,
            },
            "runtime": {
                "config": str(RUNTIME_CONFIG),
                "config_sha256": RUNTIME_CONFIG_SHA256,
            },
            "historical_comparator": {
                str(path): digest for path, digest in V29_ARTIFACT_HASHES.items()
            },
            "immutable_inputs": {
                str(V54_REPORT): V54_REPORT_SHA256,
                str(V47_CONFIG): V47_CONFIG_SHA256,
                str(RUNTIME_CONFIG): RUNTIME_CONFIG_SHA256,
                str(PROTECTED_REPORT): PROTECTED_REPORT_SHA256,
            },
            "bound_sources": source_hashes,
            "outputs": {
                "claim": str(CLAIM_PATH),
                "model_snapshot": str(MODEL_SNAPSHOT_PATH),
                "questions": str(QUESTIONS_PATH),
                "predictions": str(PREDICTIONS_PATH),
                "prediction_provenance": str(PREDICTION_PROVENANCE_PATH),
                "score": str(SCORE_PATH),
                "selector_report": str(SELECTOR_REPORT_PATH),
            },
            "thresholds": {
                "normalized_exact_accuracy_minimum": 0.375,
                "spatial_relation_accuracy_minimum": 0.55,
                "count_accuracy_minimum": 0.80,
                "presence_f1_minimum": 0.15,
                "canonical_complete_units_minimum": 2,
                "canonical_correct_sides_minimum": 12,
                "canonical_prediction_changed_units_minimum": 2,
                "physical_change_families_minimum": 2,
                "canonical_aggregate_correct_minimum": 91,
            },
            "scope": {
                "development_access_authorized_after_launch_claim": True,
                "exactly_one_candidate": True,
                "question_dependent_retrieval_authorized": False,
                "training_access_authorized": False,
                "optimizer_construction_authorized": False,
                "optimizer_state_access_authorized": False,
                "backward_authorized": False,
                "checkpoint_write_authorized": False,
                "oracle_access_authorized": False,
                "final_test_access_authorized": False,
                "runtime_promotion_authorized": False,
                "chat_promotion_authorized": False,
                "embodied_promotion_authorized": False,
            },
        },
        "predecessor_authentication": predecessor,
    }


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_create(path: Path, payload: bytes) -> None:
    destination = _resolve(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def seal_terminal(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = _resolve(output)
    if destination != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V55 terminal output path is pinned")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V55 terminal is immutable and already exists: {destination}")
    report = build_terminal_payload()
    payload = _serialized(report)
    _atomic_create(destination, payload)
    return {
        "path": str(DEFAULT_OUTPUT),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "artifact": ARTIFACT,
        "model_loaded": False,
        "map_loaded": False,
        "qa_loaded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--seal", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.preflight:
        report = build_terminal_payload()
        result = {
            "artifact": ARTIFACT,
            "preflight_passed": True,
            "prospective_sha256": hashlib.sha256(_serialized(report)).hexdigest(),
            "model_loaded": False,
            "map_loaded": False,
            "qa_loaded": False,
        }
    else:
        result = seal_terminal(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "AUTHORIZATION_ID",
    "DEFAULT_OUTPUT",
    "build_terminal_payload",
    "main",
    "seal_terminal",
]

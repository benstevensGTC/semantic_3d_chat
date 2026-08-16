"""Precommit the only authorized V56 fresh-development evaluation.

Sealing hashes the exact trained control head, sanitized question manifest,
numeric scene maps, local model snapshot, runtime configuration, base adapter,
and implementation sources.  It does not parse answer references or run a
model.  The selector must create a permanent launch claim before it opens the
fresh reference file or executes inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.model_snapshot import local_model_snapshot_identity
from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
    runtime_config_file_sha256,
)
from semantic_3d_chat.config import PROJECT_ROOT, project_path
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.prediction_artifacts import (
    build_scene_map_manifest,
    checkpoint_fingerprint,
    scene_map_manifest_sha256,
)
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    EXPECTED_REFERENCE_COUNT,
    EXPECTED_SCENE_IDS,
    FAMILY_PAIR_IDS,
    threshold_contract,
)

ARTIFACT: Final[str] = "v56_sealed_fresh_development_terminal"
AUTHORIZATION_ID: Final[str] = "v56_one_shot_fresh_development"

RUNTIME_CONFIG: Final[Path] = Path(
    "configs/runtime/gemma4_v56_question_control.yaml"
)
RUNTIME_CONFIG_FILE_SHA256: Final[str] = (
    "7cef35a5acbc0da740f79c61e0684fff329713836a4034d4fda024d6bf5372d0"
)
RUNTIME_CONFIG_EFFECTIVE_SHA256: Final[str] = (
    "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
)
V54_REPORT: Final[Path] = Path(
    "reports/gemma4/metrics/v54_semantic_greedy_gate.json"
)
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
V54_CHECKPOINT_SHA256: Final[str] = (
    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
)
V55_SCORE: Final[Path] = Path(
    "reports/gemma4/metrics/v55_development_score.json"
)
V55_SCORE_SHA256: Final[str] = (
    "6f8e041dd2272684e33df987483a1815effa626c7110a259472043426958f8f9"
)
V55_SELECTOR: Final[Path] = Path(
    "reports/gemma4/metrics/v55_development_selector.json"
)
V55_SELECTOR_SHA256: Final[str] = (
    "04f4ae96872a2455ff8c0bc1866d7fbc162df0fe05ab949dce41c463dd093510"
)

QUESTIONS_PATH: Final[Path] = Path(
    "reports/gemma4/questions/v56_fresh_development_validation.json"
)
REFERENCE_PATH: Final[Path] = Path("data_diverse52/qa/validation.jsonl")
DEFAULT_OUTPUT: Final[Path] = Path(
    "reports/gemma4/metrics/v56_fresh_development_terminal.json"
)
CLAIM_PATH: Final[Path] = Path(
    "reports/gemma4/metrics/v56_fresh_development_launch_claim.json"
)
MODEL_SNAPSHOT_PATH: Final[Path] = Path(
    "reports/gemma4/metrics/v56_fresh_development_model_snapshot.json"
)
PREDICTIONS_PATH: Final[Path] = Path(
    "reports/gemma4/predictions/v56_fresh_development_validation.jsonl"
)
PREDICTION_PROVENANCE_PATH: Final[Path] = Path(
    "reports/gemma4/predictions/v56_fresh_development_validation.jsonl.provenance.json"
)
SCORE_PATH: Final[Path] = Path(
    "reports/gemma4/metrics/v56_fresh_development_score.json"
)
SELECTOR_REPORT_PATH: Final[Path] = Path(
    "reports/gemma4/metrics/v56_fresh_development_selector.json"
)

BOUND_SOURCES: Final[tuple[Path, ...]] = (
    Path("src/semantic_3d_chat/evaluation/v56_fresh_development_terminal.py"),
    Path("src/semantic_3d_chat/evaluation/v56_fresh_development_selector.py"),
    Path("src/semantic_3d_chat/evaluation/v56_fresh_development_score.py"),
    Path("tests/test_v56_fresh_development_terminal.py"),
    Path("tests/test_v56_fresh_development_selector.py"),
    Path("tests/test_v56_fresh_development_score.py"),
    Path("src/semantic_3d_chat/training/train_question_control_v56.py"),
    Path("tests/test_train_question_control_v56.py"),
    Path("tests/test_question_control.py"),
    Path("tests/test_question_control_runtime.py"),
    Path("tests/test_predict_question_control.py"),
    Path("tests/test_question_control_runtime_config.py"),
    Path("src/semantic_3d_chat/evaluation/predict_question_control.py"),
    Path("src/semantic_3d_chat/chat/question_control_runtime.py"),
    Path("src/semantic_3d_chat/scene_encoder/question_control.py"),
    Path("src/semantic_3d_chat/evaluation/prediction_artifacts.py"),
    Path("src/semantic_3d_chat/evaluation/question_manifest.py"),
    Path("src/semantic_3d_chat/evaluation/prepare_questions.py"),
    Path("src/semantic_3d_chat/evaluation/metrics.py"),
    Path("src/semantic_3d_chat/evaluation/run.py"),
    Path("src/semantic_3d_chat/evaluation/baseline_io.py"),
    Path("src/semantic_3d_chat/chat/file_audit.py"),
    Path("src/semantic_3d_chat/chat/model_snapshot.py"),
    Path("src/semantic_3d_chat/chat/runtime.py"),
    Path("src/semantic_3d_chat/chat/runtime_config.py"),
    Path("src/semantic_3d_chat/config.py"),
    Path("src/semantic_3d_chat/device.py"),
    Path("src/semantic_3d_chat/language/gemma4_backend.py"),
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
    Path("pyproject.toml"),
    Path("uv.lock"),
)

EXPECTED_TRAIN_SCENES: Final[tuple[str, ...]] = tuple(
    [f"scene_{index:06d}" for index in range(11, 25)]
    + [f"scene_{index:06d}" for index in range(31, 57)]
)
EXPECTED_TRAIN_RECORD_COUNT: Final[int] = 960
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V56 {field} must be a mapping")
    return value


def _reject_symlink_components(path: Path, field: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V56 {field} path contains a symbolic link: {current}")


def _require_pristine_one_shot_outputs() -> None:
    """Reject any prior launch/evidence before opening fresh-development inputs."""

    outputs = (
        DEFAULT_OUTPUT,
        CLAIM_PATH,
        MODEL_SNAPSHOT_PATH,
        PREDICTIONS_PATH,
        PREDICTION_PROVENANCE_PATH,
        SCORE_PATH,
        SELECTOR_REPORT_PATH,
    )
    existing: list[str] = []
    for path in outputs:
        destination = _resolve(path)
        _reject_symlink_components(destination, "one-shot output")
        if destination.exists() or destination.is_symlink():
            existing.append(str(path))
    if existing:
        raise FileExistsError(
            "V56 one-shot output exists before terminal sealing: " f"{existing}"
        )


def _locked_file(path: str | Path, expected: str, field: str) -> None:
    source = _resolve(path)
    _reject_symlink_components(source, field)
    if not source.is_file():
        raise FileNotFoundError(f"V56 {field} is unavailable or unsafe: {source}")
    observed = _sha256(source)
    if observed != expected:
        raise ValueError(
            f"V56 {field} changed: expected={expected} observed={observed}"
        )


def _read_json(path: str | Path, field: str) -> Mapping[str, Any]:
    source = _resolve(path)
    _reject_symlink_components(source, field)
    if not source.is_file():
        raise FileNotFoundError(f"V56 {field} is unavailable or unsafe: {source}")
    return _mapping(json.loads(source.read_text(encoding="utf-8")), field)


def _project_relative(path: Path, field: str) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"V56 {field} must be inside the project") from error


def _authenticate_static_predecessors() -> dict[str, Any]:
    for path, digest, field in (
        (V54_REPORT, V54_REPORT_SHA256, "V54 report"),
        (V55_SCORE, V55_SCORE_SHA256, "V55 score"),
        (V55_SELECTOR, V55_SELECTOR_SHA256, "V55 selector"),
        (RUNTIME_CONFIG, RUNTIME_CONFIG_FILE_SHA256, "runtime config"),
    ):
        _locked_file(path, digest, field)
    runtime = load_runtime_config(RUNTIME_CONFIG)
    if effective_runtime_config_sha256(runtime) != RUNTIME_CONFIG_EFFECTIVE_SHA256:
        raise ValueError("V56 runtime effective configuration changed")
    checkpoint_root = _resolve(V54_CHECKPOINT)
    _reject_symlink_components(checkpoint_root, "V54 base checkpoint")
    if not checkpoint_root.is_dir():
        raise FileNotFoundError("V56 V54 base checkpoint is unavailable")
    inventory = sorted(item.name for item in checkpoint_root.iterdir())
    if inventory != sorted(V54_CHECKPOINT_FILES):
        raise ValueError(f"V56 V54 base checkpoint inventory changed: {inventory}")
    for name in V54_CHECKPOINT_FILES:
        item = checkpoint_root / name
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"V56 V54 base checkpoint entry is unsafe: {item}")
    checkpoint_sha256, checkpoint_files = checkpoint_fingerprint(V54_CHECKPOINT)
    if checkpoint_sha256 != V54_CHECKPOINT_SHA256:
        raise ValueError("V56 V54 base checkpoint fingerprint changed")
    if {entry["path"]: entry["sha256"] for entry in checkpoint_files} != (
        V54_CHECKPOINT_FILES
    ):
        raise ValueError("V56 V54 base checkpoint file bytes changed")

    v54 = _read_json(V54_REPORT, "V54 report")
    v55_score = _read_json(V55_SCORE, "V55 score")
    v55_selector = _read_json(V55_SELECTOR, "V55 selector")
    if (
        v54.get("passed") is not True
        or v54.get("validation_qa_loaded") is not False
        or v54.get("final_test_scenes_touched") is not False
        or v55_score.get("passed") is not False
        or v55_score.get("standard_metrics", {}).get("normalized_exact_accuracy")
        != 0.41203703703703703
        or v55_score.get("standard_metrics", {}).get("count", {}).get("accuracy")
        != 0.6666666666666666
        or v55_score.get("standard_metrics", {}).get("spatial_relation_accuracy")
        != 0.5625
        or v55_selector.get("passed") is not False
        or v55_selector.get("final_test_scenes_touched") is not False
    ):
        raise ValueError("V56 predecessor measurement contract changed")
    return {
        "v54_report_sha256": V54_REPORT_SHA256,
        "v54_checkpoint_sha256": checkpoint_sha256,
        "v55_score_sha256": V55_SCORE_SHA256,
        "v55_selector_sha256": V55_SELECTOR_SHA256,
        "runtime_config_file_sha256": RUNTIME_CONFIG_FILE_SHA256,
        "runtime_config_effective_sha256": RUNTIME_CONFIG_EFFECTIVE_SHA256,
        "fresh_development_opened": False,
        "model_loaded": False,
    }


def _control_checkpoint_identity(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    relative = _project_relative(source, "control checkpoint")
    composite = _control_checkpoint_sha256(source)
    expected_files = ("control.safetensors", "runtime_metadata.json")
    entries = {
        name: {
            "sha256": _sha256(source / name),
            "size_bytes": (source / name).stat().st_size,
        }
        for name in expected_files
    }
    metadata = _read_json(source / "runtime_metadata.json", "control metadata")
    module, strict_metadata = _load_control_head(
        source,
        hidden_size=1536,
        device=torch.device("cpu"),
    )
    if strict_metadata != dict(metadata):
        raise ValueError("V56 control metadata changed during strict loading")
    required = {
        "schema_version",
        "architecture",
        "hidden_size",
        "attention_dim",
        "control_tokens",
        "uniform_floor",
        "output_scale",
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "question_dependent_scene_retrieval",
        "complete_scene_prefix_required",
        "environmental_text_inputs",
    }
    if (
        set(metadata) != required
        or metadata.get("schema_version") != 1
        or metadata.get("architecture") != "full_scene_question_control_v1"
        or metadata.get("hidden_size") != 1536
        or metadata.get("weights_sha256")
        != entries["control.safetensors"]["sha256"]
        or metadata.get("base_checkpoint_sha256") != V54_CHECKPOINT_SHA256
        or metadata.get("base_runtime_config_sha256")
        != RUNTIME_CONFIG_EFFECTIVE_SHA256
        or metadata.get("question_dependent_scene_retrieval") is not False
        or metadata.get("complete_scene_prefix_required") is not True
        or metadata.get("environmental_text_inputs") != []
    ):
        raise ValueError("V56 control checkpoint runtime contract changed")
    return {
        "path": relative,
        "sha256": composite,
        "files": entries,
        "runtime_metadata": dict(metadata),
        "parameter_count": module.parameter_count,
    }


def _training_report_identity(
    path: str | Path,
    control: Mapping[str, Any],
) -> dict[str, Any]:
    source = _resolve(path)
    _reject_symlink_components(source, "training report")
    if {"oracle", "qa"}.intersection(part.casefold() for part in source.parts):
        raise ValueError("V56 training report must be separate from QA/oracle data")
    relative = _project_relative(source, "training report")
    report = _read_json(source, "control training report")
    base = _mapping(report.get("base"), "training base")
    inputs = _mapping(report.get("inputs"), "training inputs")
    curriculum = _mapping(report.get("curriculum"), "training curriculum")
    architecture = _mapping(report.get("architecture"), "training architecture")
    optimization = _mapping(report.get("optimization"), "training optimization")
    checkpoint = _mapping(report.get("checkpoint"), "training checkpoint")
    scope = _mapping(report.get("scope"), "training scope")
    metadata = _mapping(control.get("runtime_metadata"), "control metadata")
    expected_scope = {
        "base_scene_stack_frozen": True,
        "only_control_head_optimized": True,
        "answer_only_cross_entropy": True,
        "paired_two_side_optimizer_steps": True,
        "question_inputs_to_scene_prefix_cache": False,
        "question_dependent_scene_retrieval": False,
        "oracle_loaded": False,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
        "optimizer_state_saved": False,
    }
    expected_top_level = {
        "schema_version",
        "artifact",
        "passed",
        "base",
        "inputs",
        "curriculum",
        "architecture",
        "optimization",
        "checkpoint",
        "scope",
    }
    expected_base_fields = {
        "checkpoint_sha256",
        "checkpoint_files",
        "runtime_config_effective_sha256",
        "runtime_config_file_sha256",
    }
    expected_input_fields = {
        "training_qa_sha256",
        "training_record_count",
        "training_scene_ids",
        "prefix_cache_manifest_sha256",
        "prefix_sha256_by_scene",
        "prefix_cache_created",
    }
    expected_curriculum_fields = {
        "step_count",
        "steps_by_kind",
        "changed_pair_unit_count",
        "schedule_sha256",
        "paired_two_side_optimizer_steps",
    }
    expected_architecture_fields = {
        "name",
        "hidden_size",
        "attention_dim",
        "control_tokens",
        "uniform_floor",
        "output_scale",
        "parameter_count",
    }
    expected_optimization_fields = {
        "seed",
        "epochs",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "optimizer_steps",
        "device",
        "elapsed_seconds",
        "epoch_loss",
        "maximum_preclip_gradient_norm",
    }
    expected_checkpoint_fields = {"weights_sha256", "runtime_metadata_sha256"}
    expected_scope_fields = {*expected_scope, "base_parameter_count"}

    base_files = base.get("checkpoint_files")
    if not isinstance(base_files, list):
        raise TypeError("V56 training base checkpoint_files must be a list")
    observed_base_files: dict[str, str] = {}
    observed_base_sizes: dict[str, int] = {}
    for entry in base_files:
        item = _mapping(entry, "training base checkpoint file")
        path_value = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if (
            set(item) != {"path", "sha256", "size_bytes"}
            or not isinstance(path_value, str)
            or not isinstance(digest, str)
            or _HEX64.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or path_value in observed_base_files
        ):
            raise ValueError("V56 training base checkpoint-file contract changed")
        observed_base_files[path_value] = digest
        observed_base_sizes[path_value] = size

    prefix_hashes = inputs.get("prefix_sha256_by_scene")
    if not isinstance(prefix_hashes, Mapping):
        raise TypeError("V56 training prefix hashes must be a mapping")
    steps_by_kind = curriculum.get("steps_by_kind")
    if not isinstance(steps_by_kind, Mapping):
        raise TypeError("V56 training curriculum counts must be a mapping")
    curriculum_step_count = curriculum.get("step_count")
    curriculum_counts_valid = (
        set(steps_by_kind) == {"broad", "changed_pair", "count_replay"}
        and all(
            not isinstance(value, bool) and isinstance(value, int) and value > 0
            for value in steps_by_kind.values()
        )
        and isinstance(curriculum_step_count, int)
        and not isinstance(curriculum_step_count, bool)
        and curriculum_step_count == sum(int(value) for value in steps_by_kind.values())
    )
    base_parameter_count = scope.get("base_parameter_count")
    epoch_count = optimization.get("epochs")
    optimizer_steps = optimization.get("optimizer_steps")
    epoch_loss = optimization.get("epoch_loss")
    finite_optimization_numbers = (
        optimization.get("learning_rate"),
        optimization.get("weight_decay"),
        optimization.get("gradient_clip_norm"),
        optimization.get("elapsed_seconds"),
        optimization.get("maximum_preclip_gradient_norm"),
    )
    optimization_numbers_valid = all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for value in finite_optimization_numbers
    )
    epoch_loss_valid = (
        isinstance(epoch_loss, list)
        and isinstance(epoch_count, int)
        and not isinstance(epoch_count, bool)
        and epoch_count > 0
        and len(epoch_loss) == epoch_count
    )
    if epoch_loss_valid:
        for expected_epoch, raw_epoch in enumerate(epoch_loss):
            epoch = _mapping(raw_epoch, "training epoch loss")
            if (
                set(epoch)
                != {
                    "epoch",
                    "steps",
                    "mean_answer_ce",
                    "minimum_answer_ce",
                    "maximum_answer_ce",
                }
                or epoch.get("epoch") != expected_epoch
                or isinstance(epoch.get("steps"), bool)
                or not isinstance(epoch.get("steps"), int)
                or int(epoch.get("steps")) < 1
                or any(
                    isinstance(epoch.get(field), bool)
                    or not isinstance(epoch.get(field), (int, float))
                    or not math.isfinite(float(epoch.get(field)))
                    for field in (
                        "mean_answer_ce",
                        "minimum_answer_ce",
                        "maximum_answer_ce",
                    )
                    )
                or float(epoch.get("minimum_answer_ce")) < 0.0
                or float(epoch.get("minimum_answer_ce"))
                > float(epoch.get("mean_answer_ce"))
                or float(epoch.get("mean_answer_ce"))
                > float(epoch.get("maximum_answer_ce"))
            ):
                epoch_loss_valid = False
                break
    if (
        epoch_loss_valid
        and isinstance(optimizer_steps, int)
        and not isinstance(optimizer_steps, bool)
    ):
        epoch_loss_valid = (
            sum(int(_mapping(item, "training epoch loss")["steps"]) for item in epoch_loss)
            == optimizer_steps
        )
    expected_base_sizes = {
        name: (_resolve(V54_CHECKPOINT) / name).stat().st_size
        for name in V54_CHECKPOINT_FILES
    }
    if (
        set(report) != expected_top_level
        or set(base) != expected_base_fields
        or set(inputs) != expected_input_fields
        or set(curriculum) != expected_curriculum_fields
        or set(architecture) != expected_architecture_fields
        or set(optimization) != expected_optimization_fields
        or set(checkpoint) != expected_checkpoint_fields
        or set(scope) != expected_scope_fields
        or report.get("schema_version") != 1
        or report.get("artifact") != "v56_question_control_training"
        or report.get("passed") is not True
        or base.get("checkpoint_sha256") != V54_CHECKPOINT_SHA256
        or observed_base_files != V54_CHECKPOINT_FILES
        or observed_base_sizes != expected_base_sizes
        or base.get("runtime_config_effective_sha256")
        != RUNTIME_CONFIG_EFFECTIVE_SHA256
        or base.get("runtime_config_file_sha256") != RUNTIME_CONFIG_FILE_SHA256
        or not isinstance(inputs.get("training_qa_sha256"), str)
        or _HEX64.fullmatch(str(inputs.get("training_qa_sha256"))) is None
        or inputs.get("training_record_count") != EXPECTED_TRAIN_RECORD_COUNT
        or inputs.get("training_scene_ids") != list(EXPECTED_TRAIN_SCENES)
        or not isinstance(inputs.get("prefix_cache_manifest_sha256"), str)
        or _HEX64.fullmatch(str(inputs.get("prefix_cache_manifest_sha256"))) is None
        or set(prefix_hashes) != set(EXPECTED_TRAIN_SCENES)
        or any(
            not isinstance(value, str) or _HEX64.fullmatch(value) is None
            for value in prefix_hashes.values()
        )
        or not isinstance(inputs.get("prefix_cache_created"), bool)
        or not curriculum_counts_valid
        or isinstance(curriculum.get("changed_pair_unit_count"), bool)
        or not isinstance(curriculum.get("changed_pair_unit_count"), int)
        or int(curriculum.get("changed_pair_unit_count")) < 1
        or not isinstance(curriculum.get("schedule_sha256"), str)
        or _HEX64.fullmatch(str(curriculum.get("schedule_sha256"))) is None
        or curriculum.get("paired_two_side_optimizer_steps") is not True
        or architecture.get("name") != metadata.get("architecture")
        or architecture.get("hidden_size") != metadata.get("hidden_size")
        or architecture.get("attention_dim") != metadata.get("attention_dim")
        or architecture.get("control_tokens") != metadata.get("control_tokens")
        or architecture.get("uniform_floor") != metadata.get("uniform_floor")
        or architecture.get("output_scale") != metadata.get("output_scale")
        or architecture.get("parameter_count") != control.get("parameter_count")
        or checkpoint.get("weights_sha256")
        != control["files"]["control.safetensors"]["sha256"]
        or checkpoint.get("runtime_metadata_sha256")
        != control["files"]["runtime_metadata.json"]["sha256"]
        or not isinstance(base_parameter_count, int)
        or isinstance(base_parameter_count, bool)
        or base_parameter_count < 1
        or {key: scope.get(key) for key in expected_scope} != expected_scope
        or isinstance(optimization.get("seed"), bool)
        or not isinstance(optimization.get("seed"), int)
        or not optimization_numbers_valid
        or float(optimization.get("learning_rate")) <= 0.0
        or float(optimization.get("weight_decay")) < 0.0
        or float(optimization.get("gradient_clip_norm")) <= 0.0
        or float(optimization.get("elapsed_seconds")) < 0.0
        or float(optimization.get("maximum_preclip_gradient_norm")) < 0.0
        or not isinstance(optimizer_steps, int)
        or isinstance(optimizer_steps, bool)
        or optimizer_steps != curriculum_step_count
        or optimization.get("device") not in {"cpu", "mps"}
        or not epoch_loss_valid
    ):
        raise ValueError("V56 control training report contract changed")
    return {
        "path": relative,
        "sha256": _sha256(source),
        "training_qa_sha256": inputs["training_qa_sha256"],
        "training_record_count": EXPECTED_TRAIN_RECORD_COUNT,
        "training_scene_ids": list(EXPECTED_TRAIN_SCENES),
        "prefix_cache_manifest_sha256": inputs["prefix_cache_manifest_sha256"],
        "prefix_sha256_by_scene": dict(prefix_hashes),
        "curriculum_schedule_sha256": curriculum["schedule_sha256"],
        "optimizer_steps": optimizer_steps,
        "scope": dict(scope),
    }


def _question_identity(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    _reject_symlink_components(source, "sanitized questions")
    if source != _resolve(QUESTIONS_PATH):
        raise ValueError("V56 sanitized question-manifest path is pinned")
    manifest = load_question_manifest(source)
    if (
        manifest.question_count != EXPECTED_REFERENCE_COUNT
        or manifest.scene_count != len(EXPECTED_SCENE_IDS)
        or tuple(sorted(manifest.by_scene())) != EXPECTED_SCENE_IDS
        or any(len(records) != 36 for records in manifest.by_scene().values())
    ):
        raise ValueError("V56 question manifest is not the complete fresh split")
    return {
        "path": str(QUESTIONS_PATH),
        "manifest_sha256": manifest.manifest_sha256,
        "questions_sha256": manifest.questions_sha256,
        "reference_sha256": manifest.source_qa_sha256,
        "question_count": manifest.question_count,
        "scene_count": manifest.scene_count,
    }


def _bound_source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in BOUND_SOURCES:
        source = _resolve(path)
        _reject_symlink_components(source, "bound source")
        if not source.is_file():
            raise FileNotFoundError(f"V56 bound source is unavailable: {source}")
        result[str(path)] = _sha256(source)
    return result


def _scene_map_identity(config: Mapping[str, Any]) -> dict[str, dict[str, int | str]]:
    for scene_id in EXPECTED_SCENE_IDS:
        path = _resolve(project_path(dict(config), "maps", scene_id, "voxel_map.npz"))
        _reject_symlink_components(path, f"numeric map {scene_id}")
        if {"oracle", "qa"}.intersection(part.casefold() for part in path.parts):
            raise ValueError("V56 numeric maps must be separate from QA/oracle data")
        if not path.is_file():
            raise FileNotFoundError(f"V56 numeric map is unavailable: {path}")
    return build_scene_map_manifest(config, list(EXPECTED_SCENE_IDS))


def software_identity() -> dict[str, Any]:
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "huggingface-hub",
            "numpy",
            "safetensors",
            "torch",
            "transformers",
        )
    }
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": packages,
    }


def build_terminal_payload(
    *,
    control_checkpoint: str | Path,
    training_report: str | Path,
    questions_manifest: str | Path = QUESTIONS_PATH,
) -> dict[str, Any]:
    # This check deliberately precedes the sanitized fresh question/map reads.
    # A prior claim or evidence file permanently closes terminal materialization.
    _require_pristine_one_shot_outputs()
    predecessor = _authenticate_static_predecessors()
    runtime = load_runtime_config(RUNTIME_CONFIG)
    if runtime_config_file_sha256(RUNTIME_CONFIG) != RUNTIME_CONFIG_FILE_SHA256:
        raise ValueError("V56 runtime config file changed during terminal sealing")
    control = _control_checkpoint_identity(control_checkpoint)
    training = _training_report_identity(training_report, control)
    questions = _question_identity(questions_manifest)
    maps = _scene_map_identity(runtime)
    model_snapshot = local_model_snapshot_identity(runtime)
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "passed": True,
        "terminal_materialization_authorized": True,
        "authorization": {
            "authorization_id": AUTHORIZATION_ID,
            "only_exact_action": "one_control_one_shot_fresh_development",
            "explicit_terminal_sha256_required": True,
            "base_checkpoint": {
                "path": str(V54_CHECKPOINT),
                "sha256": V54_CHECKPOINT_SHA256,
                "file_sha256": dict(V54_CHECKPOINT_FILES),
            },
            "control_checkpoint": control,
            "training_report": training,
            "runtime": {
                "config": str(RUNTIME_CONFIG),
                "file_sha256": RUNTIME_CONFIG_FILE_SHA256,
                "effective_sha256": RUNTIME_CONFIG_EFFECTIVE_SHA256,
            },
            "development": {
                "split": "validation",
                "scene_ids": list(EXPECTED_SCENE_IDS),
                "scene_count": len(EXPECTED_SCENE_IDS),
                "atomic_pair_count": len(FAMILY_PAIR_IDS),
                "question_count": EXPECTED_REFERENCE_COUNT,
                "reference_path": str(REFERENCE_PATH),
                "reference_sha256": questions["reference_sha256"],
                "questions": questions,
                "scene_map_manifest": maps,
                "scene_map_manifest_sha256": scene_map_manifest_sha256(maps),
            },
            "pre_inference_model_snapshot": model_snapshot,
            "software": software_identity(),
            "predecessor_authentication": predecessor,
            "bound_sources": _bound_source_hashes(),
            "outputs": {
                "claim": str(CLAIM_PATH),
                "model_snapshot": str(MODEL_SNAPSHOT_PATH),
                "predictions": str(PREDICTIONS_PATH),
                "prediction_provenance": str(PREDICTION_PROVENANCE_PATH),
                "score": str(SCORE_PATH),
                "selector_report": str(SELECTOR_REPORT_PATH),
            },
            "thresholds": threshold_contract(),
            "scope": {
                "exactly_one_control_checkpoint": True,
                "fresh_development_access_authorized_after_launch_claim": True,
                "question_dependent_scene_retrieval_authorized": False,
                "training_authorized": False,
                "optimizer_authorized": False,
                "backward_authorized": False,
                "checkpoint_write_authorized": False,
                "simulator_oracle_access_authorized": False,
                "deferred_final_access_authorized": False,
                "runtime_promotion_authorized": False,
            },
        },
    }


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_create(path: Path, payload: bytes) -> None:
    destination = _resolve(path)
    _reject_symlink_components(destination, "atomic output")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V56 output is immutable: {destination}")
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


def seal_terminal(
    *,
    control_checkpoint: str | Path,
    training_report: str | Path,
    questions_manifest: str | Path = QUESTIONS_PATH,
    output: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    destination = _resolve(output)
    if destination != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V56 terminal output path is pinned")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V56 terminal is immutable: {destination}")
    report = build_terminal_payload(
        control_checkpoint=control_checkpoint,
        training_report=training_report,
        questions_manifest=questions_manifest,
    )
    payload = _serialized(report)
    _atomic_create(destination, payload)
    return {
        "path": str(DEFAULT_OUTPUT),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "artifact": ARTIFACT,
        "model_inference_executed": False,
        "answer_references_loaded": False,
        "deferred_final_scenes_touched": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--seal", action="store_true")
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--questions-manifest", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.preflight:
        report = build_terminal_payload(
            control_checkpoint=args.control_checkpoint,
            training_report=args.training_report,
            questions_manifest=args.questions_manifest,
        )
        result = {
            "artifact": ARTIFACT,
            "preflight_passed": True,
            "prospective_sha256": hashlib.sha256(_serialized(report)).hexdigest(),
            "model_inference_executed": False,
            "answer_references_loaded": False,
            "deferred_final_scenes_touched": False,
        }
    else:
        result = seal_terminal(
            control_checkpoint=args.control_checkpoint,
            training_report=args.training_report,
            questions_manifest=args.questions_manifest,
            output=args.output,
        )
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
    "software_identity",
]

"""Train one V6.2 upper-decoder reader through the exact full Gemma forward.

V6.1 proved exact prepared-input and final-hidden-state identity, bounded
objective equivalence, and finite nonzero full-reference LoRA gradients.  Its
one smoke attempt failed only because the aggregate gradient from a
shape-specialized selected-logit path did not meet the preregistered comparison
with the full Hugging Face path.  V6.2 removes that comparison entirely: every
QA loss is the token-normalized cross entropy produced from a full Hugging Face
forward with labels and full-sequence logits.

The release and attempt are create-once.  A terminal result, lifetime file
audit, and optional checkpoint are published as one atomic directory, so a
public promoted checkpoint cannot exist without its byte-bound result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import fixed_prefix_decoder_reader_v6_1_release as v61_release
from semantic_3d_chat.evaluation import fixed_prefix_decoder_reader_v6_release as v6_release
from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration import (
    BASE_CHECKPOINT,
    BASE_RUNTIME_CONFIG,
    INITIAL_STATE_SHA256,
    INITIALIZATION_SEED,
    LORA_ALPHA,
    LORA_PARAMETER_COUNT,
    LORA_RANK,
    MODEL_ID,
    MODEL_REVISION,
    TARGET_MODULES,
    answer_varying_wrong_prefixes,
    build_v6_schedule,
    learning_rate_v6,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.language.local_lm import local_model_snapshot_files
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training import train_fixed_prefix_decoder_reader_v6_1 as v61
from semantic_3d_chat.training import train_fixed_prefix_ple_v54 as v1

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_decoder_reader_v6_2"
TRAINING_RELEASE: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_2_training_release.json"
)
TRAINING_ATTEMPT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_2_training_attempt.json"
)
PUBLICATION_ROOT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_decoder_reader_v6_2"
)
RESULT_REPORT: Final[str] = f"{PUBLICATION_ROOT}/terminal_result.json"
FILE_AUDIT_REPORT: Final[str] = f"{PUBLICATION_ROOT}/file_audit.json"
PUBLICATION_MANIFEST: Final[str] = f"{PUBLICATION_ROOT}/publication_manifest.json"
OUTPUT_CHECKPOINT: Final[str] = f"{PUBLICATION_ROOT}/checkpoint"

V6_1_RELEASE: Final[str] = v61_release.MPS_SMOKE_RELEASE
V6_1_ATTEMPT: Final[str] = v61_release.MPS_SMOKE_ATTEMPT
V6_1_FAILURE: Final[str] = v61_release.MPS_SMOKE_REPORT
V6_1_RELEASE_SHA256: Final[str] = (
    "4456ebd11d8cbb154236aa6962bfc5875499580ab326068b1b9581f2127e4b33"
)
V6_1_ATTEMPT_SHA256: Final[str] = (
    "ec462122b737cda9bd111afa2a66f187039711e3f211ea3901f2eaa15986e53a"
)
V6_1_FAILURE_SHA256: Final[str] = (
    "099c1fa684439814b58c17223781b745e406d17cc20c65c402159bd0ede18add"
)

_BASE_CHECKPOINT_FINGERPRINT: Final[str] = (
    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
)
_BASE_RUNTIME_EFFECTIVE_SHA256: Final[str] = (
    "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
)
_QA_FORWARD_PATH: Final[str] = "full_huggingface_forward_token_normalized_ce"
_UPDATES: Final[int] = 96
_ROWS_PER_COMPONENT: Final[int] = 3
_GRADIENT_CLIP: Final[float] = 1.0
_MAX_MPS_DRIVER_BYTES: Final[int] = 25_000_000_000
_TRAIN_WRONG_ASSIGNMENT_SHA256: Final[str] = (
    "875cb3ed4893314494e90d563e1e961358a4fa34ccd6888545a20cfce903c5ff"
)
_VALIDATION_WRONG_ASSIGNMENT_SHA256: Final[str] = (
    "a2eaff713e8a51beec6779fc3d1720f179e2290ecaaf176d13ae1cc8d4362dcd"
)

# Direct and transitive code used to construct the V6.2 training result.  The
# older sealed V6/V6.1 artifacts additionally bind their own complete closure.
_V6_2_DIRECT_BOUND_PATHS: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "requirements-gemma4-probe.txt",
    "uv.lock",
    "src/semantic_3d_chat/chat/file_audit.py",
    "src/semantic_3d_chat/chat/runtime.py",
    "src/semantic_3d_chat/chat/runtime_config.py",
    "src/semantic_3d_chat/config.py",
    "src/semantic_3d_chat/evaluation/baseline_io.py",
    "src/semantic_3d_chat/evaluation/fixed_prefix_decoder_reader_v6_1_release.py",
    "src/semantic_3d_chat/evaluation/fixed_prefix_decoder_reader_v6_preregistration.py",
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_preregistration.py",
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_v2_preregistration.py",
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_v3_preregistration.py",
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_v4_preregistration.py",
    "src/semantic_3d_chat/evaluation/metrics.py",
    "src/semantic_3d_chat/evaluation/ple_reader_preregistration.py",
    "src/semantic_3d_chat/evaluation/prediction_artifacts.py",
    "src/semantic_3d_chat/evaluation/run.py",
    "src/semantic_3d_chat/evaluation/v55_development_score.py",
    "src/semantic_3d_chat/language/gemma4_backend.py",
    "src/semantic_3d_chat/language/generation.py",
    "src/semantic_3d_chat/language/local_lm.py",
    "src/semantic_3d_chat/language/lora.py",
    "src/semantic_3d_chat/language/prefix_injection.py",
    "src/semantic_3d_chat/scene_encoder/question_control.py",
    "src/semantic_3d_chat/scene_encoder/signed_x_local_field.py",
    "src/semantic_3d_chat/scene_encoder/signed_x_residual.py",
    "src/semantic_3d_chat/training/pair_curriculum.py",
    "src/semantic_3d_chat/training/train_adapter.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_decoder_reader_v6_1.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_decoder_reader_v6_2.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v3.py",
    "src/semantic_3d_chat/training/train_question_control_v56.py",
    "scripts/run_gemma4_v54_fixed_prefix_decoder_reader_v6_2.sh",
    "tests/test_train_fixed_prefix_decoder_reader_v6_2.py",
)
TRAINING_BOUND_PATHS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*v6_release.SMOKE_BOUND_PATHS, *_V6_2_DIRECT_BOUND_PATHS))
)


@dataclass(frozen=True)
class StagedCheckpoint:
    directory: Path
    published: dict[str, Any]


@dataclass(frozen=True)
class TrainingOutcome:
    report: dict[str, Any]
    staged_checkpoint: StagedCheckpoint | None


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _lexical_path(path: str | Path) -> Path:
    """Return an absolute path without erasing a final symlink."""

    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _assert_no_symlink_components(path: Path) -> None:
    absolute = _lexical_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"V6.2 path contains a symlink component: {current}")


def _sha256_file(path: str | Path) -> str:
    source = _resolve(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _same(left: object, right: float, *, atol: float = 1e-12) -> bool:
    return _finite(left) and math.isclose(
        float(left), right, rel_tol=1e-12, abs_tol=atol
    )


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"V6.2 JSON contains a duplicate key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"V6.2 JSON contains a non-finite constant: {value}")


def _read_json(path: str | Path) -> dict[str, Any]:
    source = _lexical_path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"V6.2 JSON is missing or unsafe: {source}")
    value = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_json_pairs,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"V6.2 JSON must contain an object: {source}")
    return value


def _write_json_file(path: Path, value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _atomic_create_json(
    path: str | Path, value: Mapping[str, Any]
) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V6.2 create-once artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, hashlib.sha256(payload).hexdigest()


def _safe_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"V6.2 training asset escapes the project: {path}") from exc


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in TRAINING_BOUND_PATHS:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"V6.2 bound source path is unsafe: {raw}")
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V6.2 bound source is missing or linked: {raw}")
        result[relative.as_posix()] = _sha256_file(source)
    return result


def _training_asset_paths() -> tuple[Path, ...]:
    prefix_root = _resolve(v1.PREFIX_CACHE)
    manifest_path = prefix_root / "manifest.json"
    manifest = _read_json(manifest_path)
    scenes = manifest.get("scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != {
        *v1.TRAIN_SCENES,
        *v1.VALIDATION_SCENES,
    }:
        raise ValueError("V6.2 prefix manifest scene inventory changed")
    prefix_files: list[Path] = []
    for scene_id in (*v1.TRAIN_SCENES, *v1.VALIDATION_SCENES):
        entry = scenes.get(scene_id)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("filename"), str):
            raise TypeError(f"V6.2 prefix manifest entry is invalid: {scene_id}")
        source = (prefix_root / str(entry["filename"])).resolve()
        if source.parent != prefix_root:
            raise ValueError(f"V6.2 prefix path escapes its cache: {scene_id}")
        prefix_files.append(source)
    base_checkpoint = _resolve(BASE_CHECKPOINT)
    checkpoint_files = sorted(path for path in base_checkpoint.rglob("*") if path.is_file())
    if not checkpoint_files:
        raise FileNotFoundError("V6.2 frozen base checkpoint is empty")
    fixed = [
        _resolve(v61.V6_CONFIG),
        _resolve(v1.CONFIG),
        _resolve(BASE_RUNTIME_CONFIG),
        _resolve(v1.TRAIN_QA),
        _resolve(v1.VALIDATION_QUESTIONS),
        _resolve(v1.VALIDATION_REFERENCES),
        _resolve(v1.RETENTION),
        _resolve(v1.BASELINE_PREDICTIONS),
        _resolve(V6_1_RELEASE),
        _resolve(V6_1_ATTEMPT),
        _resolve(V6_1_FAILURE),
        manifest_path,
    ]
    paths = sorted({path.resolve() for path in (*fixed, *checkpoint_files, *prefix_files)})
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"V6.2 training input is missing or linked: {path}")
        _safe_relative(path)
    return tuple(paths)


def _training_asset_hashes() -> dict[str, dict[str, int | str]]:
    fingerprint, _inventory = checkpoint_fingerprint(_resolve(BASE_CHECKPOINT))
    if fingerprint != _BASE_CHECKPOINT_FINGERPRINT:
        raise ValueError("V6.2 frozen base checkpoint fingerprint changed")
    runtime_config = load_runtime_config(_resolve(BASE_RUNTIME_CONFIG))
    if effective_runtime_config_sha256(runtime_config) != _BASE_RUNTIME_EFFECTIVE_SHA256:
        raise ValueError("V6.2 effective base runtime configuration changed")
    return {
        _safe_relative(path): {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _training_asset_paths()
    }


def _local_model_snapshot_inventory() -> dict[str, dict[str, int | str]]:
    """Bind every semantic filename and resolved blob in the pinned HF revision."""

    snapshot = _lexical_path(v61_release._model_snapshot())
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise FileNotFoundError("V6.2 local model snapshot is missing or linked")
    model_root = snapshot.parent.parent
    blob_root = (model_root / "blobs").resolve()
    if not blob_root.is_dir() or blob_root.is_symlink():
        raise FileNotFoundError("V6.2 local model blob directory is missing or linked")
    inventory: dict[str, dict[str, int | str]] = {}
    resolved_files: set[Path] = set()
    for logical in sorted(snapshot.rglob("*"), key=lambda path: path.as_posix()):
        if logical.is_dir():
            if logical.is_symlink():
                raise ValueError("V6.2 model snapshot contains a linked directory")
            continue
        if not logical.is_file():
            raise FileNotFoundError(f"V6.2 model snapshot entry is unsafe: {logical}")
        relative = logical.relative_to(snapshot).as_posix()
        resolved = logical.resolve()
        if not resolved.is_file() or resolved.is_symlink():
            raise FileNotFoundError(f"V6.2 model snapshot blob is unsafe: {relative}")
        try:
            blob_relative = resolved.relative_to(blob_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"V6.2 model snapshot entry escapes its blob directory: {relative}"
            ) from exc
        inventory[relative] = {
            "resolved_blob": blob_relative,
            "sha256": _sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
        resolved_files.add(resolved)
    enumerated = set(local_model_snapshot_files(MODEL_ID, MODEL_REVISION))
    if not inventory or resolved_files != enumerated:
        raise ValueError("V6.2 local model snapshot inventory is incomplete")
    required_semantic_files = {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if not required_semantic_files <= set(inventory):
        raise ValueError("V6.2 local model snapshot lost a required semantic file")
    return inventory


def _validate_full_gradient_comparison(
    comparison: object,
) -> dict[str, bool]:
    if not isinstance(comparison, Mapping):
        raise TypeError("V6.2 V6.1 gradient comparison is not a mapping")
    required = {
        "full_norm",
        "tail_norm",
        "cosine_similarity",
        "relative_l2",
        "norm_ratio",
        "full_lora_b_gradient_l2_by_target",
        "tail_lora_b_gradient_l2_by_target",
        "full_lora_a_gradient_l2_by_target",
        "tail_lora_a_gradient_l2_by_target",
        "full_lora_a_exact_zero",
        "tail_lora_a_exact_zero",
        "full_coverage",
        "tail_coverage",
        "coverage_exact",
        "sufficient_statistics",
        "passed",
    }
    if set(comparison) != required:
        raise ValueError("V6.2 V6.1 gradient comparison schema changed")
    full_b = comparison["full_lora_b_gradient_l2_by_target"]
    tail_b = comparison["tail_lora_b_gradient_l2_by_target"]
    full_a = comparison["full_lora_a_gradient_l2_by_target"]
    tail_a = comparison["tail_lora_a_gradient_l2_by_target"]
    evidence = comparison["sufficient_statistics"]
    if not all(isinstance(item, Mapping) for item in (full_b, tail_b, full_a, tail_a, evidence)):
        raise TypeError("V6.2 V6.1 gradient evidence mappings changed")
    targets = set(TARGET_MODULES)
    if any(set(item) != targets for item in (full_b, tail_b, full_a, tail_a)):
        raise ValueError("V6.2 V6.1 gradient target coverage changed")
    if any(not _finite(value) or float(value) <= 0.0 for value in full_b.values()):
        raise ValueError("V6.2 requires finite nonzero full-reference LoRA-B gradients")
    if any(value != 0.0 for value in full_a.values()):
        raise ValueError("V6.2 requires exact zero initialized LoRA-A gradients")
    if any(not _finite(value) or float(value) <= 0.0 for value in tail_b.values()):
        raise ValueError("V6.2 V6.1 comparison lost finite nonzero tail LoRA-B gradients")
    if any(value != 0.0 for value in tail_a.values()):
        raise ValueError("V6.2 V6.1 comparison lost exact zero tail LoRA-A gradients")
    expected_evidence = {
        "element_count",
        "full_vector_sha256",
        "tail_vector_sha256",
        "full_sum_squares",
        "tail_sum_squares",
        "full_tail_dot",
        "difference_sum_squares",
    }
    if set(evidence) != expected_evidence:
        raise ValueError("V6.2 V6.1 sufficient gradient statistics changed")
    if (
        evidence["element_count"] != LORA_PARAMETER_COUNT
        or not _is_sha256(evidence["full_vector_sha256"])
        or not _is_sha256(evidence["tail_vector_sha256"])
        or any(
            not _finite(evidence[key])
            for key in (
                "full_sum_squares",
                "tail_sum_squares",
                "full_tail_dot",
                "difference_sum_squares",
            )
        )
    ):
        raise ValueError("V6.2 V6.1 sufficient gradient values changed")
    full_sq = float(evidence["full_sum_squares"])
    tail_sq = float(evidence["tail_sum_squares"])
    dot = float(evidence["full_tail_dot"])
    diff_sq = float(evidence["difference_sum_squares"])
    if full_sq <= 0.0 or tail_sq <= 0.0 or diff_sq < 0.0:
        raise ValueError("V6.2 V6.1 gradient squared norms are invalid")
    if not _same(diff_sq, full_sq + tail_sq - 2.0 * dot, atol=1e-8):
        raise ValueError("V6.2 V6.1 gradient difference statistics do not close")
    full_norm = math.sqrt(full_sq)
    tail_norm = math.sqrt(tail_sq)
    cosine = dot / (full_norm * tail_norm)
    relative = math.sqrt(diff_sq) / max(full_norm, tail_norm)
    ratio = tail_norm / full_norm
    if not (
        _same(comparison["full_norm"], full_norm)
        and _same(comparison["tail_norm"], tail_norm)
        and _same(comparison["cosine_similarity"], cosine)
        and _same(comparison["relative_l2"], relative)
        and _same(comparison["norm_ratio"], ratio)
        and _same(full_norm, math.sqrt(sum(float(value) ** 2 for value in full_b.values())))
        and _same(tail_norm, math.sqrt(sum(float(value) ** 2 for value in tail_b.values())))
        and comparison["full_lora_a_exact_zero"] is True
        and comparison["tail_lora_a_exact_zero"] is True
        and comparison["full_coverage"] == list(TARGET_MODULES)
        and comparison["tail_coverage"] == list(TARGET_MODULES)
        and comparison["coverage_exact"] is True
    ):
        raise ValueError("V6.2 V6.1 full-reference gradient summaries changed")
    thresholds = v61_release.GRADIENT_EQUIVALENCE_THRESHOLDS
    return {
        "cosine": cosine >= float(thresholds["gradient_cosine_min"]),
        "relative_l2": relative <= float(thresholds["gradient_relative_l2_max"]),
        "norm_ratio": float(thresholds["gradient_norm_ratio_min"])
        <= ratio
        <= float(thresholds["gradient_norm_ratio_max"]),
        "full_coverage": comparison["full_coverage"] == list(TARGET_MODULES),
        "full_lora_a_zero": comparison["full_lora_a_exact_zero"] is True,
        "full_lora_b_nonzero": all(float(value) > 0.0 for value in full_b.values()),
    }


def authenticate_v6_1_terminal_failure() -> dict[str, Any]:
    """Authenticate and interpret the exact consumed V6.1 smoke failure."""

    expected_hashes = {
        V6_1_RELEASE: V6_1_RELEASE_SHA256,
        V6_1_ATTEMPT: V6_1_ATTEMPT_SHA256,
        V6_1_FAILURE: V6_1_FAILURE_SHA256,
    }
    for path, expected in expected_hashes.items():
        if _sha256_file(path) != expected:
            raise ValueError(f"V6.2 sealed V6.1 artifact bytes changed: {path}")
    release = _read_json(V6_1_RELEASE)
    attempt = _read_json(V6_1_ATTEMPT)
    failure = _read_json(V6_1_FAILURE)
    if not (
        release.get("terminal_output") == V6_1_FAILURE
        and release.get("attempt_journal") == V6_1_ATTEMPT
        and release.get("authorized", {}).get("maximum_smoke_runs") == 1
        and release.get("authorized", {}).get("optimizer_construction") is False
        and attempt.get("authorization_sha256") == V6_1_RELEASE_SHA256
        and attempt.get("maximum_attempts") == 1
        and attempt.get("optimizer_steps_authorized") == 0
        and failure.get("authorization_sha256") == V6_1_RELEASE_SHA256
        and failure.get("attempt_sha256") == V6_1_ATTEMPT_SHA256
        and failure.get("status") == "failed_terminal_attempt_consumed"
        and failure.get("passed") is False
        and failure.get("failure_type") == "V61GateFailure"
        and failure.get("failure_stage") == "gradient_equivalence"
        and failure.get("optimizer_constructed") is False
        and failure.get("optimizer_steps") == 0
        and failure.get("training_executed") is False
        and failure.get("checkpoint_published") is False
        and failure.get("file_access_audit_active_for_entire_execution") is True
        and failure.get("forbidden_file_accesses") == []
        and failure.get("deferred_or_final_qa_accessed") is False
    ):
        raise ValueError("V6.2 sealed V6.1 lineage or terminal semantics changed")
    metrics = failure.get("failure_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "objective_equivalence",
        "gradient_equivalence",
    }:
        raise ValueError("V6.2 V6.1 terminal failure metrics changed")
    objective = metrics["objective_equivalence"]
    if not v61_release.objective_equivalence_passes(objective):
        raise ValueError("V6.2 requires independently recomputed V6.1 objective equivalence")
    gradient = metrics["gradient_equivalence"]
    comparisons = gradient.get("gradient_comparisons") if isinstance(gradient, Mapping) else None
    if not isinstance(comparisons, Mapping) or set(comparisons) != {
        "correct",
        "wrong",
        "broad",
        "aggregate",
    }:
        raise ValueError("V6.2 V6.1 gradient branch inventory changed")
    gate_results = {
        name: _validate_full_gradient_comparison(comparison)
        for name, comparison in comparisons.items()
    }
    if any(comparisons[name].get("passed") is not True for name in ("correct", "wrong", "broad")):
        raise ValueError("V6.2 requires every V6.1 full-reference branch smoke to pass")
    if any(not all(gate_results[name].values()) for name in ("correct", "wrong", "broad")):
        raise ValueError("V6.2 recomputed V6.1 branch gradient gate failed")
    aggregate = gate_results["aggregate"]
    failed_aggregate_gates = sorted(name for name, passed in aggregate.items() if not passed)
    if not (
        gradient.get("passed") is False
        and comparisons["aggregate"].get("passed") is False
        and failed_aggregate_gates == ["cosine", "relative_l2"]
    ):
        raise ValueError("V6.2 requires V6.1 to fail only tail/full aggregate comparison")
    return {
        "passed": True,
        "release_sha256": V6_1_RELEASE_SHA256,
        "attempt_sha256": V6_1_ATTEMPT_SHA256,
        "terminal_failure_sha256": V6_1_FAILURE_SHA256,
        "objective_equivalence_recomputed_passed": True,
        "full_reference_gradient_branches": sorted(comparisons),
        "full_reference_gradient_coverage": list(TARGET_MODULES),
        "full_reference_gradients_finite_nonzero": True,
        "full_reference_lora_a_exact_zero": True,
        "sole_failed_comparison": "tail_vs_full_aggregate",
        "failed_aggregate_gates": failed_aggregate_gates,
    }


def _current_model_binding() -> dict[str, Any]:
    weights = v61_release._authenticate_model_blob()
    snapshot = _local_model_snapshot_inventory()
    weight_entry = snapshot.get("model.safetensors")
    if (
        not isinstance(weight_entry, Mapping)
        or weight_entry.get("sha256") != weights["model_weights_blob_sha256"]
        or weight_entry.get("size_bytes") != weights["model_weights_size_bytes"]
    ):
        raise ValueError("V6.2 snapshot and independently streamed model blob differ")
    return {
        "weights": weights,
        "snapshot_files": snapshot,
        "installed_transformers_sources": (
            v61_release._installed_transformers_sources()
        ),
    }


def build_training_release() -> dict[str, Any]:
    lineage = authenticate_v6_1_terminal_failure()
    sources = _source_hashes()
    assets = _training_asset_hashes()
    model = _current_model_binding()
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_training_release",
        "status": "released_exactly_one_full_reference_96_update_training_run",
        "sealed_v6_1_terminal_failure": lineage,
        "qa_forward_contract": {
            "path": _QA_FORWARD_PATH,
            "full_sequence_logits": True,
            "labels_passed_to_huggingface_model": True,
            "token_normalized_answer_cross_entropy": True,
            "shape_specialized_selected_logit_training": False,
        },
        "bound_source_sha256": sources,
        "bound_source_inventory_sha256": _canonical_hash(sources),
        "bound_training_asset_sha256": assets,
        "bound_training_asset_inventory_sha256": _canonical_hash(assets),
        "base_checkpoint_fingerprint": _BASE_CHECKPOINT_FINGERPRINT,
        "base_runtime_effective_sha256": _BASE_RUNTIME_EFFECTIVE_SHA256,
        "local_model_binding": model,
        "authorized": {
            "maximum_training_runs": 1,
            "exact_optimizer_updates": _UPDATES,
            "optimizer_construction": True,
            "intermediate_checkpoint_or_selection": False,
            "atomic_publication_envelope_required": True,
            "deferred_or_final_qa_access": False,
            "oracle_access": False,
        },
        "required_attempt_journal": TRAINING_ATTEMPT,
        "atomic_publication_root": PUBLICATION_ROOT,
        "terminal_result": RESULT_REPORT,
        "file_audit": FILE_AUDIT_REPORT,
        "output_checkpoint": OUTPUT_CHECKPOINT,
    }


def write_training_release(
    destination: str | Path = TRAINING_RELEASE,
) -> tuple[Path, str]:
    return _atomic_create_json(destination, build_training_release())


def authenticate_training_release(path: str | Path = TRAINING_RELEASE) -> dict[str, Any]:
    release = _read_json(path)
    expected = build_training_release()
    if release != expected:
        raise ValueError("V6.2 training release, code, model, or an input changed")
    return {
        "passed": True,
        "path": str(Path(path)),
        "sha256": _sha256_file(path),
        "parent_failure_sha256": V6_1_FAILURE_SHA256,
        "source_count": len(release["bound_source_sha256"]),
        "training_asset_count": len(release["bound_training_asset_sha256"]),
        "local_model_binding_sha256": _canonical_hash(
            release["local_model_binding"]
        ),
        "qa_forward_path": _QA_FORWARD_PATH,
    }


def _assert_current_model_binding(expected_sha256: object) -> None:
    if not _is_sha256(expected_sha256):
        raise ValueError("V6.2 released local model binding digest is invalid")
    observed = _canonical_hash(_current_model_binding())
    if observed != expected_sha256:
        raise ValueError("V6.2 local model snapshot changed after release authentication")


def training_forbidden_roots() -> list[Path]:
    return v61.training_forbidden_roots()


def _expected_attempt_payload(release: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_training_attempt",
        "status": "claimed_before_model_load",
        "training_release_sha256": release["sha256"],
        "parent_v6_1_terminal_failure_sha256": V6_1_FAILURE_SHA256,
        "maximum_optimizer_updates": _UPDATES,
        "qa_forward_path": _QA_FORWARD_PATH,
        "checkpoint_write_authorized_before_internal_gates": False,
        "deferred_or_final_qa_access_authorized": False,
        "oracle_access_authorized": False,
    }


def claim_training_attempt(
    release: Mapping[str, Any] | None = None,
) -> tuple[Path, str]:
    authenticated = authenticate_training_release() if release is None else dict(release)
    if _resolve(TRAINING_ATTEMPT).exists() or _resolve(PUBLICATION_ROOT).exists():
        raise FileExistsError("V6.2 one-shot attempt or publication already exists")
    return _atomic_create_json(
        TRAINING_ATTEMPT,
        _expected_attempt_payload(authenticated),
    )


def authenticate_attempt_state() -> dict[str, Any]:
    """Report a claimed-but-unterminalized hard crash without authorizing resume."""

    release = authenticate_training_release()
    attempt_path = _resolve(TRAINING_ATTEMPT)
    publication = _lexical_path(PUBLICATION_ROOT)
    if not attempt_path.exists():
        if publication.exists():
            raise ValueError("V6.2 publication exists without its create-once attempt")
        return {
            "passed": True,
            "status": "unclaimed",
            "attempt_consumed": False,
            "training_resume_authorized": False,
            "publication_exists": False,
        }
    attempt = _read_json(attempt_path)
    if attempt != _expected_attempt_payload(release):
        raise ValueError("V6.2 attempt journal bytes or lineage changed")
    if publication.exists():
        manifest = _read_json(PUBLICATION_MANIFEST)
        _authenticate_publication_manifest(manifest)
        return {
            "passed": True,
            "status": "terminal_publication_committed",
            "attempt_consumed": True,
            "training_resume_authorized": False,
            "publication_exists": True,
            "attempt_sha256": _sha256_file(attempt_path),
        }
    return {
        "passed": False,
        "status": "claimed_unterminalized_requires_successor_release",
        "attempt_consumed": True,
        "training_resume_authorized": False,
        "publication_exists": False,
        "attempt_sha256": _sha256_file(attempt_path),
    }


def _required_loaded_paths() -> set[str]:
    required = {str(path.resolve()) for path in _training_asset_paths()}
    required.add(str(_resolve(TRAINING_RELEASE)))
    for path in local_model_snapshot_files(MODEL_ID, MODEL_REVISION):
        required.add(str(Path(path).resolve()))
    return required


def _record_required_assets(audit: FileAccessAudit) -> None:
    for path in sorted(_required_loaded_paths()):
        audit.record(path)


def load_base_bundle_v6_2(audit: FileAccessAudit) -> v1.ReaderBundle:
    _record_required_assets(audit)
    bundle = v61.load_base_bundle_v6_1(audit)
    if bundle.installation.state_sha256() != INITIAL_STATE_SHA256:
        raise ValueError("V6.2 initial adapter state differs from sealed V6.1")
    return bundle


def answer_nll(
    bundle: v1.ReaderBundle, prefix: torch.Tensor, row: v1.ReaderRecord
) -> torch.Tensor:
    """Run full-sequence Gemma logits and return token-normalized answer CE."""

    prepared = v1._prepared_batch(bundle, prefix, row)
    if prepared.labels is None:
        raise RuntimeError("V6.2 full-reference QA batch lost its labels")
    output = bundle.language.model(
        inputs_embeds=prepared.inputs_embeds,
        attention_mask=prepared.attention_mask,
        labels=prepared.labels,
        per_layer_inputs=prepared.per_layer_inputs,
        mm_token_type_ids=prepared.mm_token_type_ids,
        use_cache=False,
        return_dict=True,
    )
    logits = output.logits
    if not isinstance(logits, torch.Tensor) or logits.shape[:2] != prepared.labels.shape:
        raise RuntimeError("V6.2 full Hugging Face forward did not return full logits")
    nlls = v1.token_normalized_nll(logits, prepared.labels)
    if nlls.shape != (1,) or not torch.isfinite(nlls).all() or nlls[0] < 0:
        raise RuntimeError("V6.2 full-reference answer NLL is invalid")
    hf_loss = getattr(output, "loss", None)
    if (
        not isinstance(hf_loss, torch.Tensor)
        or hf_loss.ndim != 0
        or not torch.isfinite(hf_loss)
        or abs(float(hf_loss.detach().float().cpu()) - float(nlls[0].detach().float().cpu()))
        > 1e-6
    ):
        raise RuntimeError("V6.2 Hugging Face loss and manual token-normalized CE diverged")
    return nlls[0]


@torch.inference_mode()
def evaluate_teacher_forcing_v6_2(
    bundle: v1.ReaderBundle,
    rows: Sequence[v1.ReaderRecord],
) -> dict[str, Any]:
    """Evaluate all 384 correct and all 170 fixed wrong-prefix examples."""

    if len(rows) != 384:
        raise ValueError("V6.2 teacher evaluation requires all 384 validation rows")
    bundle.installation.eval()
    correct: dict[tuple[str, str], float] = {}
    for row in rows:
        value = answer_nll(bundle, bundle.prefixes[row.scene_id], row)
        correct[v61._row_key(row)] = float(value.detach().cpu())
    assignments = answer_varying_wrong_prefixes(rows)
    assignment_hash = v61._wrong_assignment_hash(assignments)
    if len(assignments) != 170 or assignment_hash != _VALIDATION_WRONG_ASSIGNMENT_SHA256:
        raise ValueError("V6.2 fixed validation wrong-prefix inventory changed")
    wrong: dict[tuple[str, str], float] = {}
    for row in rows:
        key = v61._row_key(row)
        if key in assignments:
            value = answer_nll(bundle, bundle.prefixes[assignments[key]], row)
            wrong[key] = float(value.detach().cpu())
    margins = {key: wrong[key] - correct[key] for key in wrong}
    curated_rows = [row for row in rows if row.changed]
    units: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in curated_rows:
        if row.pair_id is None or row.pair_question_key is None:
            raise ValueError("V6.2 curated validation row lacks pair identity")
        units[(row.pair_id, row.pair_question_key)].append(margins[v61._row_key(row)])
    if len(units) != 26 or any(len(values) != 2 for values in units.values()):
        raise ValueError("V6.2 curated validation units changed")
    rows_by_scene_question = {(row.scene_id, row.question): row for row in rows}
    family_values: defaultdict[str, list[float]] = defaultdict(list)
    scope_values: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = v61._row_key(row)
        if key not in assignments:
            continue
        family_values[row.answer_type].append(margins[key])
        scope = v61._selected_scope(row, assignments[key], rows_by_scene_question)
        scope_values[scope].append(margins[key])
    family_counts = {key: len(values) for key, values in sorted(family_values.items())}
    scope_counts = {key: len(values) for key, values in sorted(scope_values.items())}
    if family_counts != v61._EXPECTED_FAMILIES or scope_counts != v61._EXPECTED_SCOPES:
        raise ValueError("V6.2 validation family or scope inventory changed")
    family_rates = {
        key: sum(value > 0.0 for value in values) / len(values)
        for key, values in sorted(family_values.items())
    }
    scope_rates = {
        key: sum(value > 0.0 for value in values) / len(values)
        for key, values in sorted(scope_values.items())
    }
    curated_margins = [margins[v61._row_key(row)] for row in curated_rows]
    correct_records = [
        {"scene_id": key[0], "question_id": key[1], "value": correct[key]}
        for key in sorted(correct)
    ]
    margin_records = [
        {"scene_id": key[0], "question_id": key[1], "value": margins[key]}
        for key in sorted(margins)
    ]
    return {
        "answer_nll_mean": sum(correct.values()) / len(correct),
        "answer_nll_count": len(correct),
        "curated_margin_mean": sum(curated_margins) / len(curated_margins),
        "curated_positive_margin_sides": sum(value > 0.0 for value in curated_margins),
        "curated_side_count": len(curated_margins),
        "curated_positive_margin_rate": sum(value > 0.0 for value in curated_margins)
        / len(curated_margins),
        "curated_complete_units": sum(
            all(value > 0.0 for value in values) for values in units.values()
        ),
        "curated_unit_count": len(units),
        "expanded_margin_mean": sum(margins.values()) / len(margins),
        "expanded_positive_margin_sides": sum(value > 0.0 for value in margins.values()),
        "expanded_side_count": len(margins),
        "expanded_positive_margin_rate": sum(value > 0.0 for value in margins.values())
        / len(margins),
        "family_counts": family_counts,
        "family_positive_margin_rates": family_rates,
        "family_macro_positive_margin_rate": sum(family_rates.values()) / len(family_rates),
        "scope_counts": scope_counts,
        "scope_positive_margin_rates": scope_rates,
        "scope_macro_positive_margin_rate": sum(scope_rates.values()) / len(scope_rates),
        "correct_nll_by_row": correct_records,
        "expanded_margin_by_row": margin_records,
        "correct_nll_sha256": v61._tuple_mapping_hash(correct),
        "expanded_margin_sha256": v61._tuple_mapping_hash(margins),
        "wrong_prefix_assignment_sha256": assignment_hash,
        "evaluation_microbatch_size": 1,
        "answer_logit_positions_only": True,
    }


def retention_kl_loss_v6_2(
    bundle: v1.ReaderBundle, row: Mapping[str, str], teacher: torch.Tensor
) -> torch.Tensor:
    loss = v1.retention_kl_loss(bundle, row, teacher)
    if loss < -1e-6:
        raise RuntimeError("V6.2 retention KL is materially negative")
    return loss.clamp_min(0.0)


@torch.inference_mode()
def evaluate_retention_v6_2(
    bundle: v1.ReaderBundle,
    corpus: Sequence[Mapping[str, str]],
    teachers: Sequence[torch.Tensor],
) -> dict[str, Any]:
    if len(corpus) != len(teachers) or len(corpus) != 16:
        raise ValueError("V6.2 retention inventory changed")
    records: list[dict[str, Any]] = []
    for index, (row, teacher) in enumerate(zip(corpus, teachers, strict=True)):
        current = v1.retention_logits(bundle, row["prompt"]).detach().cpu().float()
        teacher_float = teacher.detach().cpu().float()
        target = v1._retention_target_id(bundle, row["continuation"])
        baseline_ce = float(F.cross_entropy(teacher_float, torch.tensor([target])).item())
        current_ce = float(F.cross_entropy(current, torch.tensor([target])).item())
        teacher_probabilities = torch.softmax(teacher_float.double(), dim=-1)
        current_log_probabilities = torch.log_softmax(current.double(), dim=-1)
        teacher_log_probabilities = torch.log_softmax(teacher_float.double(), dim=-1)
        raw_kl = float(
            (teacher_probabilities * (teacher_log_probabilities - current_log_probabilities))
            .sum()
            .item()
        )
        if baseline_ce < 0.0 or current_ce < 0.0 or raw_kl < -1e-6:
            raise RuntimeError("V6.2 retention produced a negative NLL or KL")
        kl = max(0.0, raw_kl)
        baseline_top1 = int(teacher_float.argmax().item())
        current_top1 = int(current.argmax().item())
        records.append(
            {
                "index": index,
                "target_token_id": target,
                "baseline_ce_nats": baseline_ce,
                "current_ce_nats": current_ce,
                "ce_increase_nats": current_ce - baseline_ce,
                "kl_nats": kl,
                "baseline_top1_token_id": baseline_top1,
                "current_top1_token_id": current_top1,
                "top1_agreement": baseline_top1 == current_top1,
            }
        )
    increases = [float(row["ce_increase_nats"]) for row in records]
    kls = [float(row["kl_nats"]) for row in records]
    agreements = [bool(row["top1_agreement"]) for row in records]
    return {
        "example_count": len(records),
        "records": records,
        "mean_ce_increase_nats": sum(increases) / len(increases),
        "maximum_ce_increase_nats": max(increases),
        "mean_kl_nats": sum(kls) / len(kls),
        "maximum_kl_nats": max(kls),
        "next_token_top1_agreement": sum(agreements) / len(agreements),
        "metrics_sha256": _canonical_hash(records),
    }


@torch.inference_mode()
def evaluate_greedy_v6_2(
    bundle: v1.ReaderBundle, rows: Sequence[v1.ReaderRecord]
) -> dict[str, Any]:
    selected = v1._greedy_subset(rows)
    baseline = v1._baseline_prediction_index()
    records: list[dict[str, Any]] = []
    bundle.installation.eval()
    bundle.language.decoder_module.eval()
    model_dtype = next(bundle.language.model.parameters()).dtype
    for row in selected:
        prompt = v1.prompt_token_ids(
            bundle.language.tokenizer,
            str(bundle.runtime.config["language"]["system_prompt"]),
            row.question,
            bundle.language.device,
        )
        prefix = bundle.prefixes[row.scene_id].to(bundle.language.device, dtype=model_dtype)
        generated = bundle.language.generate_from_scene_prefix(
            prefix,
            prompt,
            max_new_tokens=int(bundle.runtime.config["language"]["max_answer_tokens"]),
            eos_token_ids=v1._eos_ids(bundle),
            scene_prefix_after_bos=v1.scene_prefix_after_bos_setting(bundle.runtime.config),
            scene_boundary_mode=v1.scene_boundary_mode_setting(bundle.runtime.config),
            fallback=v1.generate_from_embeddings,
        )
        decoded = bundle.language.tokenizer.decode(
            generated[0].detach().cpu().tolist(), skip_special_tokens=True
        ).strip() or "unknown"
        key = (row.scene_id, row.question_id)
        baseline_prediction = baseline[key]
        records.append(
            {
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "baseline_correct": bool(
                    v1.canonical_type_specific_match(
                        row.answer_type, baseline_prediction, row.answer
                    )
                ),
                "candidate_correct": bool(
                    v1.canonical_type_specific_match(row.answer_type, decoded, row.answer)
                ),
                "normalized_baseline_prediction": v1.normalize_answer(
                    baseline_prediction
                ),
                "normalized_candidate_prediction": v1.normalize_answer(decoded),
                "normalized_baseline_prediction_sha256": hashlib.sha256(
                    v1.normalize_answer(baseline_prediction).encode()
                ).hexdigest(),
                "normalized_candidate_prediction_sha256": hashlib.sha256(
                    v1.normalize_answer(decoded).encode()
                ).hexdigest(),
                "prefix_sha256": v1.prefix_sha256(prefix),
            }
        )
    baseline_correct = sum(bool(row["baseline_correct"]) for row in records)
    candidate_correct = sum(bool(row["candidate_correct"]) for row in records)
    return {
        "row_count": len(records),
        "records": records,
        "baseline_exact_correct": baseline_correct,
        "baseline_exact_accuracy": baseline_correct / len(records),
        "candidate_exact_correct": candidate_correct,
        "candidate_exact_accuracy": candidate_correct / len(records),
        "exact_accuracy_delta": (candidate_correct - baseline_correct) / len(records),
        "prediction_records_sha256": _canonical_hash(records),
        "question_dependent_scene_retrieval": False,
    }


def optimizer_kwargs() -> dict[str, Any]:
    return v61.optimizer_kwargs()


def _trace_item(
    *,
    update: int,
    learning_rate: float,
    contrastive: Sequence[Mapping[str, Any]],
    broad: Sequence[Mapping[str, Any]],
    retention_index: int,
    retention_kl: float,
    gradient: float,
    adapter_hash: str,
) -> dict[str, Any]:
    weighted_retention = 0.5 * retention_kl
    total = (
        sum(float(row["weighted_objective"]) for row in contrastive)
        + sum(float(row["weighted_objective"]) for row in broad)
        + weighted_retention
    )
    return {
        "update": update,
        "learning_rate": learning_rate,
        "contrastive_components": list(contrastive),
        "broad_components": list(broad),
        "retention_index": retention_index,
        "retention_kl": retention_kl,
        "weighted_retention_objective": weighted_retention,
        "total_objective": total,
        "preclip_gradient_l2": gradient,
        "adapter_state_sha256": adapter_hash,
    }


def _stage_checkpoint(
    bundle: v1.ReaderBundle,
    selection: Mapping[str, Any],
    *,
    training_release_sha256: str,
) -> StagedCheckpoint:
    publication = _lexical_path(PUBLICATION_ROOT)
    _assert_no_symlink_components(publication.parent)
    publication.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{publication.name}.checkpoint.", dir=publication.parent)
    )
    try:
        weights = temporary / "adapter.safetensors"
        state = {
            key: tensor.detach().float().cpu().contiguous()
            for key, tensor in bundle.installation.state_module.state_dict().items()
        }
        expected_keys = {
            "adapters.0.lora_a",
            "adapters.0.lora_b",
            "adapters.1.lora_a",
            "adapters.1.lora_b",
        }
        if set(state) != expected_keys:
            raise ValueError("V6.2 checkpoint acquired non-V6.2 tensors")
        save_file(state, weights)
        metadata = {
            "schema_version": 1,
            "artifact": ARTIFACT,
            "base_checkpoint_sha256": _BASE_CHECKPOINT_FINGERPRINT,
            "base_runtime_config_effective_sha256": _BASE_RUNTIME_EFFECTIVE_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "qa_forward_path": _QA_FORWARD_PATH,
            "fixed_prefix_tokens": 258,
            "scene_latents": 256,
            "scene_hidden_dimension": 1536,
            "prefix_computed_before_question": True,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "oracle_runtime_access": False,
            "adapter_type": "fresh_v6_2_full_reference_upper_decoder_lora",
            "target_modules": list(TARGET_MODULES),
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": 0.0,
            "trainable_parameter_count": LORA_PARAMETER_COUNT,
            "adapter_state_sha256": bundle.installation.state_sha256(),
            "adapter_file_sha256": _sha256_file(weights),
            "selection_summary_sha256": _canonical_hash(selection),
            "training_release_sha256": training_release_sha256,
            "parent_v6_1_terminal_failure_sha256": V6_1_FAILURE_SHA256,
        }
        metadata_path = temporary / "runtime_metadata.json"
        _write_json_file(metadata_path, metadata)
        return StagedCheckpoint(
            temporary,
            {
                "path": OUTPUT_CHECKPOINT,
                "adapter_file_sha256": metadata["adapter_file_sha256"],
                "runtime_metadata_sha256": _sha256_file(metadata_path),
                "adapter_state_sha256": metadata["adapter_state_sha256"],
                "tensor_keys": sorted(state),
            },
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _audit_payload(audit: FileAccessAudit) -> dict[str, Any]:
    loaded = audit.unique_paths
    forbidden = audit.forbidden_accesses()
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_file_audit",
        "loaded_files": loaded,
        "loaded_file_count": len(loaded),
        "loaded_file_inventory_sha256": _canonical_hash(loaded),
        "forbidden_roots": [str(path) for path in audit.forbidden_roots],
        "forbidden_component_names": sorted(audit.forbidden_component_names),
        "block_forbidden": audit.block_forbidden,
        "forbidden_accesses": forbidden,
        "passed": not forbidden,
    }


def _commit_publication(
    report: dict[str, Any],
    audit_payload: Mapping[str, Any],
    staged_checkpoint: StagedCheckpoint | None,
) -> dict[str, Any]:
    """Atomically publish result, audit, marker, and optional checkpoint."""

    destination = _lexical_path(PUBLICATION_ROOT)
    _assert_no_symlink_components(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("V6.2 atomic publication already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.publication.", dir=destination.parent)
    )
    try:
        checkpoint = None
        if staged_checkpoint is not None:
            os.rename(staged_checkpoint.directory, temporary / "checkpoint")
            checkpoint = staged_checkpoint.published
        report = dict(report)
        report["checkpoint"] = checkpoint
        report["checkpoint_published"] = checkpoint is not None
        report["file_audit_report"] = FILE_AUDIT_REPORT
        report["loaded_file_count"] = audit_payload["loaded_file_count"]
        audit_path = temporary / "file_audit.json"
        audit_sha = _write_json_file(audit_path, audit_payload)
        report["file_audit_sha256"] = audit_sha
        result_path = temporary / "terminal_result.json"
        result_sha = _write_json_file(result_path, report)
        file_hashes: dict[str, str] = {
            "file_audit.json": audit_sha,
            "terminal_result.json": result_sha,
        }
        if checkpoint is not None:
            file_hashes.update(
                {
                    "checkpoint/adapter.safetensors": _sha256_file(
                        temporary / "checkpoint/adapter.safetensors"
                    ),
                    "checkpoint/runtime_metadata.json": _sha256_file(
                        temporary / "checkpoint/runtime_metadata.json"
                    ),
                }
            )
        manifest = {
            "schema_version": 1,
            "artifact": f"{ARTIFACT}_atomic_publication",
            "status": "committed",
            "passed": report.get("passed") is True,
            "checkpoint_in_same_atomic_directory": checkpoint is not None,
            "files_sha256": file_hashes,
            "file_inventory_sha256": _canonical_hash(file_hashes),
        }
        _write_json_file(temporary / "publication_manifest.json", manifest)
        os.rename(temporary, destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if staged_checkpoint is not None:
            shutil.rmtree(staged_checkpoint.directory, ignore_errors=True)
        raise


def _execute_training(
    *, release: Mapping[str, Any], attempt_sha256: str, audit: FileAccessAudit
) -> TrainingOutcome:
    started = time.perf_counter()
    torch.manual_seed(INITIALIZATION_SEED)
    random.seed(INITIALIZATION_SEED)
    _assert_current_model_binding(release.get("local_model_binding_sha256"))
    bundle = load_base_bundle_v6_2(audit)
    _assert_current_model_binding(release.get("local_model_binding_sha256"))
    train_rows = v1.load_training_records()
    validation_rows = v1.load_validation_records()
    retention_corpus = v1.load_retention_corpus()
    schedule = build_v6_schedule(train_rows)
    wrong_assignments = answer_varying_wrong_prefixes(train_rows)
    if (
        len(schedule) != _UPDATES
        or len(wrong_assignments) != 288
        or v61._wrong_assignment_hash(wrong_assignments)
        != _TRAIN_WRONG_ASSIGNMENT_SHA256
        or len(retention_corpus) != 16
    ):
        raise ValueError("V6.2 released training inventory changed")
    teachers = v1.retention_baseline(bundle, retention_corpus)
    baseline_teacher = evaluate_teacher_forcing_v6_2(bundle, validation_rows)
    baseline_retention = evaluate_retention_v6_2(bundle, retention_corpus, teachers)
    optimizer = torch.optim.AdamW(bundle.installation.parameters(), **optimizer_kwargs())
    trace: list[dict[str, Any]] = []
    optimizer_steps = 0
    bundle.installation.train()
    for update_index, update in enumerate(schedule, start=1):
        if len(update.contrastive) != 3 or len(update.broad) != 3:
            raise ValueError("V6.2 update lost its exact 3+3 row structure")
        current_lr = learning_rate_v6(update_index)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        contrastive_records: list[dict[str, Any]] = []
        broad_records: list[dict[str, Any]] = []
        for row in update.contrastive:
            key = v61._row_key(row)
            wrong_scene = wrong_assignments[key]
            correct = answer_nll(bundle, bundle.prefixes[row.scene_id], row)
            wrong = answer_nll(bundle, bundle.prefixes[wrong_scene], row)
            loss, diagnostics = v61.contrastive_row_objective(correct, wrong)
            loss.backward()
            correct_value = float(correct.detach().cpu())
            wrong_value = float(wrong.detach().cpu())
            contrastive_records.append(
                {
                    "scene_id": row.scene_id,
                    "question_id": row.question_id,
                    "wrong_scene_id": wrong_scene,
                    "correct_nll": correct_value,
                    "wrong_nll": wrong_value,
                    "margin": float(diagnostics["margin"].detach().cpu()),
                    "hinge": float(diagnostics["hinge"].detach().cpu()),
                    "weighted_objective": float(loss.detach().cpu()),
                }
            )
        for row in update.broad:
            nll = answer_nll(bundle, bundle.prefixes[row.scene_id], row)
            loss = v61.broad_row_objective(nll)
            loss.backward()
            broad_records.append(
                {
                    "scene_id": row.scene_id,
                    "question_id": row.question_id,
                    "nll": float(nll.detach().cpu()),
                    "weighted_objective": float(loss.detach().cpu()),
                }
            )
        retention_index = (update_index - 1) % len(retention_corpus)
        retention_kl = retention_kl_loss_v6_2(
            bundle, retention_corpus[retention_index], teachers[retention_index]
        )
        v61.retention_objective(retention_kl).backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(
                bundle.installation.parameters(), _GRADIENT_CLIP
            )
            .detach()
            .cpu()
        )
        if not math.isfinite(gradient) or gradient <= 0.0:
            raise RuntimeError("V6.2 preclip gradient norm is invalid")
        optimizer.step()
        optimizer_steps += 1
        bundle.installation.validate_state()
        bundle.installation.assert_only_lora_trainable(bundle.language.model)
        item = _trace_item(
            update=update_index,
            learning_rate=current_lr,
            contrastive=contrastive_records,
            broad=broad_records,
            retention_index=retention_index,
            retention_kl=float(retention_kl.detach().cpu()),
            gradient=gradient,
            adapter_hash=bundle.installation.state_sha256(),
        )
        trace.append(item)
        print(
            json.dumps(
                {
                    "phase": "fixed_prefix_decoder_reader_v6_2_train",
                    "update": update_index,
                    "updates": _UPDATES,
                    "total_objective": item["total_objective"],
                    "gradient_l2": gradient,
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
    if optimizer_steps != _UPDATES:
        raise RuntimeError("V6.2 did not execute exactly 96 optimizer updates")
    bundle.installation.eval()
    candidate_teacher = evaluate_teacher_forcing_v6_2(bundle, validation_rows)
    candidate_retention = evaluate_retention_v6_2(bundle, retention_corpus, teachers)
    checks = v61.teacher_and_retention_checks(
        baseline_teacher, candidate_teacher, candidate_retention
    )
    greedy: dict[str, Any] | None = None
    if all(checks.values()):
        greedy = evaluate_greedy_v6_2(bundle, validation_rows)
    checks["greedy_exact_accuracy_delta"] = bool(
        greedy is not None and greedy["exact_accuracy_delta"] >= 0.02
    )
    passed = all(checks.values())
    selection = {
        "baseline_teacher": baseline_teacher,
        "candidate_teacher": candidate_teacher,
        "baseline_retention": baseline_retention,
        "candidate_retention": candidate_retention,
        "greedy": greedy,
        "checks": checks,
        "passed": passed,
    }
    memory = v1.memory_metrics()
    driver_bytes = memory.get("mps_driver_allocated_bytes")
    if driver_bytes is not None and driver_bytes > _MAX_MPS_DRIVER_BYTES:
        raise RuntimeError("V6.2 training exceeded the locked MPS driver-memory ceiling")
    training = {
        "qa_forward_path": _QA_FORWARD_PATH,
        "full_sequence_logits": True,
        "optimizer": "AdamW",
        "optimizer_kwargs": optimizer_kwargs(),
        "updates": optimizer_steps,
        "contrastive_rows_consumed_exactly_once": 288,
        "broad_rows_consumed_exactly_once": 288,
        "retention_examples": 16,
        "retention_exposures_per_example": 6,
        "trainable_parameter_count": bundle.installation.parameter_count,
        "maximum_preclip_gradient_l2": max(
            float(item["preclip_gradient_l2"]) for item in trace
        ),
        "initial_trace": trace[:3],
        "milestone_trace": [trace[index - 1] for index in (24, 48, 72, 96)],
        "final_trace": trace[-3:],
        "trace": trace,
        "trace_sha256": _canonical_hash(trace),
        "final_adapter_state_sha256": bundle.installation.state_sha256(),
        "intermediate_selection_or_checkpoint": False,
        "gradient_checkpointing": False,
    }
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_terminal_result",
        "status": "passed_checkpoint_published" if passed else "failed_no_checkpoint",
        "passed": passed,
        "promotion_eligible": passed,
        "checkpoint_published": False,
        "checkpoint": None,
        "training_release_sha256": release["sha256"],
        "parent_v6_1_terminal_failure_sha256": V6_1_FAILURE_SHA256,
        "training_attempt_sha256": attempt_sha256,
        "training": training,
        "selection": selection,
        "fixed_prefix": {
            "shape": [1, 258, 1536],
            "computed_before_question": True,
            "same_prefix_for_unchanged_scene": True,
            "question_dependent_retrieval": False,
            "all_scene_latents_present": True,
        },
        "runtime_leakage": {
            "environmental_text_inputs": [],
            "oracle_runtime_access": False,
            "training_qa_not_in_runtime_checkpoint": True,
            "validation_answers_not_in_runtime_checkpoint": True,
        },
        "deferred_and_final": {
            "deferred_scene_ids": list(v61._DEFERRED_SCENES),
            "final_scene_ids": list(v61._FINAL_SCENES),
            "accessed": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "memory": memory,
    }
    staged = None
    if passed:
        staged = _stage_checkpoint(
            bundle, selection, training_release_sha256=str(release["sha256"])
        )
    return TrainingOutcome(report, staged)


def train_and_gate() -> dict[str, Any]:
    audit = FileAccessAudit(
        training_forbidden_roots(),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    claimed = False
    release: dict[str, Any] | None = None
    attempt_sha256: str | None = None
    outcome: TrainingOutcome | None = None
    failure: BaseException | None = None
    with audit:
        try:
            release = authenticate_training_release()
            _path, attempt_sha256 = claim_training_attempt(release)
            claimed = True
            outcome = _execute_training(
                release=release, attempt_sha256=attempt_sha256, audit=audit
            )
        except BaseException as exc:  # noqa: BLE001 - persist every claimed outcome
            failure = exc
    if not claimed:
        if outcome is not None and outcome.staged_checkpoint is not None:
            shutil.rmtree(outcome.staged_checkpoint.directory, ignore_errors=True)
        if failure is not None:
            raise failure
        raise RuntimeError("V6.2 training ended before claiming its attempt")
    audit_payload = _audit_payload(audit)
    if audit_payload["passed"] is not True and failure is None:
        failure = RuntimeError("V6.2 file audit found forbidden access")
    if failure is not None:
        if outcome is not None and outcome.staged_checkpoint is not None:
            shutil.rmtree(outcome.staged_checkpoint.directory, ignore_errors=True)
        report = {
            "schema_version": 1,
            "artifact": f"{ARTIFACT}_terminal_result",
            "status": "failed_terminal_attempt_consumed_no_checkpoint",
            "passed": False,
            "promotion_eligible": False,
            "checkpoint_published": False,
            "checkpoint": None,
            "training_release_sha256": None if release is None else release["sha256"],
            "parent_v6_1_terminal_failure_sha256": V6_1_FAILURE_SHA256,
            "training_attempt_sha256": attempt_sha256,
            "error_type": type(failure).__name__,
            "error": str(failure),
            "deferred_or_final_qa_accessed": bool(audit.forbidden_accesses()),
            "oracle_accessed": any(
                "oracle" in {part.casefold() for part in Path(path).parts}
                for path in audit.forbidden_accesses()
            ),
        }
        _commit_publication(report, audit_payload, None)
        raise failure
    if outcome is None:
        raise RuntimeError("V6.2 claimed training produced no outcome")
    return _commit_publication(outcome.report, audit_payload, outcome.staged_checkpoint)


def _authenticate_retention_evidence(metrics: object) -> None:
    fields = {
        "example_count",
        "records",
        "mean_ce_increase_nats",
        "maximum_ce_increase_nats",
        "mean_kl_nats",
        "maximum_kl_nats",
        "next_token_top1_agreement",
        "metrics_sha256",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != fields:
        raise ValueError("V6.2 retention evidence schema changed")
    records = metrics.get("records")
    record_fields = {
        "index",
        "target_token_id",
        "baseline_ce_nats",
        "current_ce_nats",
        "ce_increase_nats",
        "kl_nats",
        "baseline_top1_token_id",
        "current_top1_token_id",
        "top1_agreement",
    }
    if not isinstance(records, list) or len(records) != 16:
        raise ValueError("V6.2 retention raw record count changed")
    for index, row in enumerate(records):
        if (
            not isinstance(row, Mapping)
            or set(row) != record_fields
            or row.get("index") != index
            or type(row.get("target_token_id")) is not int
            or int(row["target_token_id"]) < 0
            or type(row.get("baseline_top1_token_id")) is not int
            or type(row.get("current_top1_token_id")) is not int
            or not all(
                _finite(row.get(key))
                for key in (
                    "baseline_ce_nats",
                    "current_ce_nats",
                    "ce_increase_nats",
                    "kl_nats",
                )
            )
            or float(row["baseline_ce_nats"]) < 0.0
            or float(row["current_ce_nats"]) < 0.0
            or float(row["kl_nats"]) < 0.0
            or not _same(
                row["ce_increase_nats"],
                float(row["current_ce_nats"]) - float(row["baseline_ce_nats"]),
            )
            or row.get("top1_agreement")
            is not (row["baseline_top1_token_id"] == row["current_top1_token_id"])
        ):
            raise ValueError(f"V6.2 retention raw record changed at index {index}")
    increases = [float(row["ce_increase_nats"]) for row in records]
    kls = [float(row["kl_nats"]) for row in records]
    agreements = [bool(row["top1_agreement"]) for row in records]
    derived = {
        "example_count": 16,
        "mean_ce_increase_nats": sum(increases) / 16,
        "maximum_ce_increase_nats": max(increases),
        "mean_kl_nats": sum(kls) / 16,
        "maximum_kl_nats": max(kls),
        "next_token_top1_agreement": sum(agreements) / 16,
        "metrics_sha256": _canonical_hash(records),
    }
    for key, expected in derived.items():
        observed = metrics.get(key)
        if isinstance(expected, float):
            if not _same(observed, expected):
                raise ValueError(f"V6.2 retention derived metric changed: {key}")
        elif observed != expected:
            raise ValueError(f"V6.2 retention derived metric changed: {key}")


def _authenticate_greedy_evidence(
    metrics: object, validation_rows: Sequence[v1.ReaderRecord]
) -> bool:
    if metrics is None:
        return False
    fields = {
        "row_count",
        "records",
        "baseline_exact_correct",
        "baseline_exact_accuracy",
        "candidate_exact_correct",
        "candidate_exact_accuracy",
        "exact_accuracy_delta",
        "prediction_records_sha256",
        "question_dependent_scene_retrieval",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != fields:
        raise ValueError("V6.2 greedy evidence schema changed")
    records = metrics.get("records")
    record_fields = {
        "scene_id",
        "question_id",
        "baseline_correct",
        "candidate_correct",
        "normalized_baseline_prediction",
        "normalized_candidate_prediction",
        "normalized_baseline_prediction_sha256",
        "normalized_candidate_prediction_sha256",
        "prefix_sha256",
    }
    if not isinstance(records, list) or len(records) != 96:
        raise ValueError("V6.2 greedy raw record count changed")
    selected = v1._greedy_subset(validation_rows)
    baseline_index = v1._baseline_prediction_index()
    prefix_manifest = _read_json(_resolve(v1.PREFIX_CACHE) / "manifest.json")
    prefix_scenes = prefix_manifest.get("scenes")
    if not isinstance(prefix_scenes, Mapping):
        raise TypeError("V6.2 greedy prefix manifest changed")
    identities: set[tuple[str, str]] = set()
    for row, reference in zip(records, selected, strict=True):
        if not isinstance(row, Mapping) or set(row) != record_fields:
            raise ValueError("V6.2 greedy raw record schema changed")
        identity = (row.get("scene_id"), row.get("question_id"))
        expected_identity = (reference.scene_id, reference.question_id)
        baseline_prediction = v1.normalize_answer(baseline_index[expected_identity])
        candidate_prediction = row.get("normalized_candidate_prediction")
        prefix_entry = prefix_scenes.get(reference.scene_id)
        if (
            not all(isinstance(value, str) and value for value in identity)
            or identity in identities
            or identity != expected_identity
            or type(row.get("baseline_correct")) is not bool
            or type(row.get("candidate_correct")) is not bool
            or not isinstance(candidate_prediction, str)
            or not candidate_prediction
            or row.get("normalized_baseline_prediction") != baseline_prediction
            or row.get("baseline_correct")
            is not bool(
                v1.canonical_type_specific_match(
                    reference.answer_type, baseline_prediction, reference.answer
                )
            )
            or row.get("candidate_correct")
            is not bool(
                v1.canonical_type_specific_match(
                    reference.answer_type, candidate_prediction, reference.answer
                )
            )
            or not isinstance(prefix_entry, Mapping)
            or row.get("prefix_sha256") != prefix_entry.get("prefix_sha256")
            or not all(
                _is_sha256(row.get(key))
                for key in (
                    "normalized_baseline_prediction_sha256",
                    "normalized_candidate_prediction_sha256",
                    "prefix_sha256",
                )
            )
            or row.get("normalized_baseline_prediction_sha256")
            != hashlib.sha256(baseline_prediction.encode()).hexdigest()
            or row.get("normalized_candidate_prediction_sha256")
            != hashlib.sha256(candidate_prediction.encode()).hexdigest()
        ):
            raise ValueError("V6.2 greedy raw record value changed")
        identities.add(identity)
    baseline = sum(bool(row["baseline_correct"]) for row in records)
    candidate = sum(bool(row["candidate_correct"]) for row in records)
    expected = {
        "row_count": 96,
        "baseline_exact_correct": baseline,
        "baseline_exact_accuracy": baseline / 96,
        "candidate_exact_correct": candidate,
        "candidate_exact_accuracy": candidate / 96,
        "exact_accuracy_delta": (candidate - baseline) / 96,
        "prediction_records_sha256": _canonical_hash(records),
        "question_dependent_scene_retrieval": False,
    }
    for key, value in expected.items():
        observed = metrics.get(key)
        if isinstance(value, float):
            if not _same(observed, value):
                raise ValueError(f"V6.2 greedy derived metric changed: {key}")
        elif observed != value:
            raise ValueError(f"V6.2 greedy derived metric changed: {key}")
    return float(metrics["exact_accuracy_delta"]) >= 0.02


def _authenticate_selection_evidence(
    selection: object, validation_rows: Sequence[v1.ReaderRecord]
) -> bool:
    if not isinstance(selection, Mapping) or set(selection) != {
        "baseline_teacher",
        "candidate_teacher",
        "baseline_retention",
        "candidate_retention",
        "greedy",
        "checks",
        "passed",
    }:
        raise ValueError("V6.2 selection evidence schema changed")
    for key in ("baseline_teacher", "candidate_teacher"):
        metrics = selection[key]
        v61._authenticate_teacher_evidence(metrics, validation_rows)
        correct = v61._row_value_mapping(metrics["correct_nll_by_row"], 384)
        margins = v61._row_value_mapping(metrics["expanded_margin_by_row"], 170)
        if any(value < 0.0 for value in correct.values()):
            raise ValueError("V6.2 teacher evidence contains a negative correct NLL")
        if any(correct[key] + margin < 0.0 for key, margin in margins.items()):
            raise ValueError("V6.2 teacher evidence contains a negative wrong NLL")
    _authenticate_retention_evidence(selection["baseline_retention"])
    _authenticate_retention_evidence(selection["candidate_retention"])
    teacher_checks = v61.teacher_and_retention_checks(
        selection["baseline_teacher"],
        selection["candidate_teacher"],
        selection["candidate_retention"],
    )
    greedy_passed = _authenticate_greedy_evidence(
        selection["greedy"], validation_rows
    )
    expected_checks = {**teacher_checks, "greedy_exact_accuracy_delta": greedy_passed}
    if selection.get("checks") != expected_checks:
        raise ValueError("V6.2 stored selection checks were not independently derived")
    passed = all(expected_checks.values())
    if selection.get("passed") is not passed:
        raise ValueError("V6.2 stored selection decision changed")
    if all(teacher_checks.values()) != (selection.get("greedy") is not None):
        raise ValueError("V6.2 greedy evaluation was not correctly delayed")
    return passed


def _authenticate_training_trace(
    training: object, train_rows: Sequence[v1.ReaderRecord]
) -> None:
    fields = {
        "qa_forward_path",
        "full_sequence_logits",
        "optimizer",
        "optimizer_kwargs",
        "updates",
        "contrastive_rows_consumed_exactly_once",
        "broad_rows_consumed_exactly_once",
        "retention_examples",
        "retention_exposures_per_example",
        "trainable_parameter_count",
        "maximum_preclip_gradient_l2",
        "initial_trace",
        "milestone_trace",
        "final_trace",
        "trace",
        "trace_sha256",
        "final_adapter_state_sha256",
        "intermediate_selection_or_checkpoint",
        "gradient_checkpointing",
    }
    if not isinstance(training, Mapping) or set(training) != fields:
        raise ValueError("V6.2 training evidence schema changed")
    trace = training.get("trace")
    if (
        training.get("qa_forward_path") != _QA_FORWARD_PATH
        or training.get("full_sequence_logits") is not True
        or training.get("optimizer") != "AdamW"
        or training.get("optimizer_kwargs") != json.loads(json.dumps(optimizer_kwargs()))
        or training.get("updates") != _UPDATES
        or training.get("contrastive_rows_consumed_exactly_once") != 288
        or training.get("broad_rows_consumed_exactly_once") != 288
        or training.get("retention_examples") != 16
        or training.get("retention_exposures_per_example") != 6
        or training.get("trainable_parameter_count") != LORA_PARAMETER_COUNT
        or training.get("intermediate_selection_or_checkpoint") is not False
        or training.get("gradient_checkpointing") is not False
        or not isinstance(trace, list)
        or len(trace) != _UPDATES
        or training.get("trace_sha256") != _canonical_hash(trace)
        or not _is_sha256(training.get("final_adapter_state_sha256"))
    ):
        raise ValueError("V6.2 training evidence values changed")
    schedule = build_v6_schedule(train_rows)
    wrong_assignments = answer_varying_wrong_prefixes(train_rows)
    seen_contrastive: list[tuple[str, str]] = []
    seen_broad: list[tuple[str, str]] = []
    contrastive_fields = {
        "scene_id",
        "question_id",
        "wrong_scene_id",
        "correct_nll",
        "wrong_nll",
        "margin",
        "hinge",
        "weighted_objective",
    }
    broad_fields = {"scene_id", "question_id", "nll", "weighted_objective"}
    trace_fields = {
        "update",
        "learning_rate",
        "contrastive_components",
        "broad_components",
        "retention_index",
        "retention_kl",
        "weighted_retention_objective",
        "total_objective",
        "preclip_gradient_l2",
        "adapter_state_sha256",
    }
    for index, (item, scheduled) in enumerate(zip(trace, schedule, strict=True), start=1):
        if not isinstance(item, Mapping) or set(item) != trace_fields:
            raise ValueError(f"V6.2 training trace schema changed at update {index}")
        contrastive = item.get("contrastive_components")
        broad = item.get("broad_components")
        if not isinstance(contrastive, list) or len(contrastive) != 3:
            raise ValueError(f"V6.2 contrastive trace count changed at update {index}")
        if not isinstance(broad, list) or len(broad) != 3:
            raise ValueError(f"V6.2 broad trace count changed at update {index}")
        total = 0.0
        for observed, expected_row in zip(contrastive, scheduled.contrastive, strict=True):
            key = (expected_row.scene_id, expected_row.question_id)
            if (
                not isinstance(observed, Mapping)
                or set(observed) != contrastive_fields
                or (observed.get("scene_id"), observed.get("question_id")) != key
                or observed.get("wrong_scene_id") != wrong_assignments[key]
                or not all(
                    _finite(observed.get(name))
                    for name in (
                        "correct_nll",
                        "wrong_nll",
                        "margin",
                        "hinge",
                        "weighted_objective",
                    )
                )
                or float(observed["correct_nll"]) < 0.0
                or float(observed["wrong_nll"]) < 0.0
            ):
                raise ValueError(f"V6.2 contrastive trace value changed at update {index}")
            margin = float(observed["wrong_nll"]) - float(observed["correct_nll"])
            hinge = max(0.0, 0.5 - margin)
            objective = (0.5 / 3.0) * float(observed["correct_nll"]) + (
                4.0 / 3.0
            ) * hinge
            if not (
                _same(observed["margin"], margin, atol=1e-6)
                and _same(observed["hinge"], hinge, atol=1e-6)
                and _same(observed["weighted_objective"], objective, atol=1e-6)
            ):
                raise ValueError(f"V6.2 contrastive objective changed at update {index}")
            total += float(observed["weighted_objective"])
            seen_contrastive.append(key)
        for observed, expected_row in zip(broad, scheduled.broad, strict=True):
            key = (expected_row.scene_id, expected_row.question_id)
            if (
                not isinstance(observed, Mapping)
                or set(observed) != broad_fields
                or (observed.get("scene_id"), observed.get("question_id")) != key
                or not _finite(observed.get("nll"))
                or float(observed["nll"]) < 0.0
                or not _same(
                    observed.get("weighted_objective"),
                    (0.5 / 3.0) * float(observed["nll"]),
                    atol=1e-6,
                )
            ):
                raise ValueError(f"V6.2 broad trace value changed at update {index}")
            total += float(observed["weighted_objective"])
            seen_broad.append(key)
        expected_retention = (index - 1) % 16
        if (
            item.get("update") != index
            or not _same(item.get("learning_rate"), learning_rate_v6(index), atol=1e-15)
            or item.get("retention_index") != expected_retention
            or not _finite(item.get("retention_kl"))
            or float(item["retention_kl"]) < 0.0
            or not _same(
                item.get("weighted_retention_objective"),
                0.5 * float(item["retention_kl"]),
                atol=1e-7,
            )
            or not _finite(item.get("preclip_gradient_l2"))
            or float(item["preclip_gradient_l2"]) <= 0.0
            or not _is_sha256(item.get("adapter_state_sha256"))
        ):
            raise ValueError(f"V6.2 trace rotation or scalar changed at update {index}")
        total += float(item["weighted_retention_objective"])
        if not _same(item.get("total_objective"), total, atol=1e-6):
            raise ValueError(f"V6.2 total objective changed at update {index}")
    if not (
        len(seen_contrastive) == len(set(seen_contrastive)) == 288
        and len(seen_broad) == len(set(seen_broad)) == 288
    ):
        raise ValueError("V6.2 training row coverage changed")
    maximum = max(float(item["preclip_gradient_l2"]) for item in trace)
    if not (
        training.get("initial_trace") == trace[:3]
        and training.get("milestone_trace")
        == [trace[index - 1] for index in (24, 48, 72, 96)]
        and training.get("final_trace") == trace[-3:]
        and training.get("final_adapter_state_sha256")
        == trace[-1]["adapter_state_sha256"]
        and _same(training.get("maximum_preclip_gradient_l2"), maximum)
    ):
        raise ValueError("V6.2 training trace summaries changed")


def _is_forbidden_path(path: Path, roots: Sequence[Path]) -> bool:
    if "oracle" in {part.casefold() for part in path.parts}:
        return True
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _authenticate_audit(audit: object) -> bool:
    fields = {
        "schema_version",
        "artifact",
        "loaded_files",
        "loaded_file_count",
        "loaded_file_inventory_sha256",
        "forbidden_roots",
        "forbidden_component_names",
        "block_forbidden",
        "forbidden_accesses",
        "passed",
    }
    if not isinstance(audit, Mapping) or set(audit) != fields:
        raise ValueError("V6.2 file-audit schema changed")
    loaded = audit.get("loaded_files")
    if (
        audit.get("schema_version") != 1
        or audit.get("artifact") != f"{ARTIFACT}_file_audit"
        or not isinstance(loaded, list)
        or loaded != sorted(set(loaded))
        or not all(isinstance(path, str) and Path(path).is_absolute() for path in loaded)
        or audit.get("loaded_file_count") != len(loaded)
        or audit.get("loaded_file_inventory_sha256") != _canonical_hash(loaded)
        or audit.get("forbidden_roots")
        != [str(path) for path in training_forbidden_roots()]
        or audit.get("forbidden_component_names") != ["oracle"]
        or audit.get("block_forbidden") is not True
    ):
        raise ValueError("V6.2 file-audit inventory changed")
    roots = training_forbidden_roots()
    recomputed = [path for path in loaded if _is_forbidden_path(Path(path), roots)]
    if audit.get("forbidden_accesses") != recomputed:
        raise ValueError("V6.2 forbidden file accesses were not recomputed")
    clean = not recomputed
    if audit.get("passed") is not clean:
        raise ValueError("V6.2 file-audit pass decision changed")
    missing = _required_loaded_paths() - set(loaded)
    if missing:
        raise ValueError(f"V6.2 audit omitted required training files: {sorted(missing)}")
    return clean


def _authenticate_publication_manifest(manifest: Mapping[str, Any]) -> None:
    root = _lexical_path(PUBLICATION_ROOT)
    _assert_no_symlink_components(root)
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError("V6.2 atomic publication root is missing or unsafe")
    passed = manifest.get("passed")
    checkpoint_present = manifest.get("checkpoint_in_same_atomic_directory")
    if type(passed) is not bool or checkpoint_present is not passed:
        raise ValueError("V6.2 atomic publication pass/checkpoint marker changed")
    allowed_files = {
        "file_audit.json",
        "publication_manifest.json",
        "terminal_result.json",
    }
    allowed_directories: set[str] = set()
    if passed:
        allowed_directories.add("checkpoint")
        allowed_files.update(
            {
                "checkpoint/adapter.safetensors",
                "checkpoint/runtime_metadata.json",
            }
        )
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"V6.2 atomic publication contains a symlink: {relative}")
        if path.is_dir():
            observed_directories.add(relative)
        elif path.is_file():
            observed_files.add(relative)
        else:
            raise ValueError(f"V6.2 atomic publication contains an unsafe entry: {relative}")
    if observed_files != allowed_files or observed_directories != allowed_directories:
        raise ValueError("V6.2 atomic publication file inventory changed")
    expected_files = {
        relative: _sha256_file(root / relative)
        for relative in sorted(allowed_files - {"publication_manifest.json"})
    }
    if (
        set(manifest)
        != {
            "schema_version",
            "artifact",
            "status",
            "passed",
            "checkpoint_in_same_atomic_directory",
            "files_sha256",
            "file_inventory_sha256",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("artifact") != f"{ARTIFACT}_atomic_publication"
        or manifest.get("status") != "committed"
        or manifest.get("passed") is not passed
        or manifest.get("checkpoint_in_same_atomic_directory") is not passed
        or manifest.get("files_sha256") != expected_files
        or manifest.get("file_inventory_sha256") != _canonical_hash(expected_files)
    ):
        raise ValueError("V6.2 atomic publication manifest changed")


def _validate_common_result(
    result: Mapping[str, Any], release: Mapping[str, Any], attempt_sha: str, audit_sha: str
) -> None:
    if not (
        result.get("schema_version") == 1
        and result.get("artifact") == f"{ARTIFACT}_terminal_result"
        and result.get("training_release_sha256") == release["sha256"]
        and result.get("parent_v6_1_terminal_failure_sha256") == V6_1_FAILURE_SHA256
        and result.get("training_attempt_sha256") == attempt_sha
        and result.get("file_audit_report") == FILE_AUDIT_REPORT
        and result.get("file_audit_sha256") == audit_sha
    ):
        raise ValueError("V6.2 terminal result lineage changed")


def authenticate_result() -> dict[str, Any]:
    release = authenticate_training_release()
    publication = _lexical_path(PUBLICATION_ROOT)
    _assert_no_symlink_components(publication)
    if not publication.is_dir() or publication.is_symlink():
        raise FileNotFoundError("V6.2 atomic publication is missing or unsafe")
    result = _read_json(RESULT_REPORT)
    audit = _read_json(FILE_AUDIT_REPORT)
    attempt = _read_json(TRAINING_ATTEMPT)
    manifest = _read_json(PUBLICATION_MANIFEST)
    _authenticate_publication_manifest(manifest)
    attempt_fields = {
        "schema_version",
        "artifact",
        "status",
        "training_release_sha256",
        "parent_v6_1_terminal_failure_sha256",
        "maximum_optimizer_updates",
        "qa_forward_path",
        "checkpoint_write_authorized_before_internal_gates",
        "deferred_or_final_qa_access_authorized",
        "oracle_access_authorized",
    }
    if (
        set(attempt) != attempt_fields
        or attempt.get("schema_version") != 1
        or attempt.get("artifact") != f"{ARTIFACT}_training_attempt"
        or attempt.get("status") != "claimed_before_model_load"
        or attempt.get("training_release_sha256") != release["sha256"]
        or attempt.get("parent_v6_1_terminal_failure_sha256") != V6_1_FAILURE_SHA256
        or attempt.get("maximum_optimizer_updates") != _UPDATES
        or attempt.get("qa_forward_path") != _QA_FORWARD_PATH
        or attempt.get("checkpoint_write_authorized_before_internal_gates") is not False
        or attempt.get("deferred_or_final_qa_access_authorized") is not False
        or attempt.get("oracle_access_authorized") is not False
    ):
        raise ValueError("V6.2 training attempt evidence changed")
    attempt_sha = _sha256_file(TRAINING_ATTEMPT)
    audit_sha = _sha256_file(FILE_AUDIT_REPORT)
    audit_clean = _authenticate_audit(audit)
    _validate_common_result(result, release, attempt_sha, audit_sha)
    exceptional_fields = {
        "schema_version",
        "artifact",
        "status",
        "passed",
        "promotion_eligible",
        "checkpoint_published",
        "checkpoint",
        "training_release_sha256",
        "parent_v6_1_terminal_failure_sha256",
        "training_attempt_sha256",
        "error_type",
        "error",
        "deferred_or_final_qa_accessed",
        "oracle_accessed",
        "file_audit_report",
        "file_audit_sha256",
        "loaded_file_count",
    }
    if result.get("status") == "failed_terminal_attempt_consumed_no_checkpoint":
        if (
            set(result) != exceptional_fields
            or result.get("passed") is not False
            or result.get("promotion_eligible") is not False
            or result.get("checkpoint_published") is not False
            or result.get("checkpoint") is not None
            or not isinstance(result.get("error_type"), str)
            or not isinstance(result.get("error"), str)
            or result.get("loaded_file_count") != audit["loaded_file_count"]
            or result.get("deferred_or_final_qa_accessed")
            is not bool(audit["forbidden_accesses"])
            or result.get("oracle_accessed")
            is not any(
                "oracle" in {part.casefold() for part in Path(path).parts}
                for path in audit["forbidden_accesses"]
            )
            or _resolve(OUTPUT_CHECKPOINT).exists()
            or manifest.get("passed") is not False
            or manifest.get("checkpoint_in_same_atomic_directory") is not False
        ):
            raise ValueError("V6.2 exceptional failure evidence changed")
        return {
            "passed": False,
            "status": result["status"],
            "result_sha256": _sha256_file(RESULT_REPORT),
            "audit_clean": audit_clean,
            "checkpoint_exists": False,
        }
    ordinary_fields = {
        "schema_version",
        "artifact",
        "status",
        "passed",
        "promotion_eligible",
        "checkpoint_published",
        "checkpoint",
        "training_release_sha256",
        "parent_v6_1_terminal_failure_sha256",
        "training_attempt_sha256",
        "training",
        "selection",
        "fixed_prefix",
        "runtime_leakage",
        "deferred_and_final",
        "elapsed_seconds",
        "memory",
        "file_audit_report",
        "file_audit_sha256",
        "loaded_file_count",
    }
    if set(result) != ordinary_fields or not audit_clean:
        raise ValueError("V6.2 ordinary result schema or file audit changed")
    validation_rows = v1.load_validation_records()
    selection_passed = _authenticate_selection_evidence(
        result.get("selection"), validation_rows
    )
    train_rows = v1.load_training_records()
    _authenticate_training_trace(result.get("training"), train_rows)
    passed = result.get("passed") is True
    expected_prefix = {
        "shape": [1, 258, 1536],
        "computed_before_question": True,
        "same_prefix_for_unchanged_scene": True,
        "question_dependent_retrieval": False,
        "all_scene_latents_present": True,
    }
    expected_leakage = {
        "environmental_text_inputs": [],
        "oracle_runtime_access": False,
        "training_qa_not_in_runtime_checkpoint": True,
        "validation_answers_not_in_runtime_checkpoint": True,
    }
    expected_deferred = {
        "deferred_scene_ids": list(v61._DEFERRED_SCENES),
        "final_scene_ids": list(v61._FINAL_SCENES),
        "accessed": False,
    }
    memory = result.get("memory")
    if (
        passed is not selection_passed
        or result.get("promotion_eligible") is not passed
        or result.get("status")
        != ("passed_checkpoint_published" if passed else "failed_no_checkpoint")
        or result.get("checkpoint_published") is not passed
        or result.get("fixed_prefix") != expected_prefix
        or result.get("runtime_leakage") != expected_leakage
        or result.get("deferred_and_final") != expected_deferred
        or not _finite(result.get("elapsed_seconds"))
        or float(result["elapsed_seconds"]) < 0.0
        or not isinstance(memory, Mapping)
        or set(memory)
        != {
            "peak_process_rss_bytes",
            "mps_current_allocated_bytes",
            "mps_driver_allocated_bytes",
        }
        or any(value is not None and (type(value) is not int or value < 0) for value in memory.values())
        or result.get("loaded_file_count") != audit["loaded_file_count"]
        or manifest.get("passed") is not passed
        or manifest.get("checkpoint_in_same_atomic_directory") is not passed
    ):
        raise ValueError("V6.2 ordinary result semantics changed")
    checkpoint_path = _lexical_path(OUTPUT_CHECKPOINT)
    if not passed:
        if result.get("checkpoint") is not None or checkpoint_path.exists():
            raise ValueError("V6.2 failed gated result retained a checkpoint")
        return {
            "passed": False,
            "status": result["status"],
            "result_sha256": _sha256_file(RESULT_REPORT),
            "audit_clean": True,
            "checkpoint_exists": False,
        }
    checkpoint = result.get("checkpoint")
    weights = checkpoint_path / "adapter.safetensors"
    metadata_path = checkpoint_path / "runtime_metadata.json"
    if (
        not checkpoint_path.is_dir()
        or checkpoint_path.is_symlink()
        or weights.is_symlink()
        or metadata_path.is_symlink()
        or not weights.is_file()
        or not metadata_path.is_file()
        or not isinstance(checkpoint, Mapping)
        or set(checkpoint)
        != {
            "path",
            "adapter_file_sha256",
            "runtime_metadata_sha256",
            "adapter_state_sha256",
            "tensor_keys",
        }
        or checkpoint.get("path") != OUTPUT_CHECKPOINT
        or checkpoint.get("adapter_file_sha256") != _sha256_file(weights)
        or checkpoint.get("runtime_metadata_sha256") != _sha256_file(metadata_path)
    ):
        raise ValueError("V6.2 checkpoint publication record changed")
    tensors = load_file(str(weights), device="cpu")
    expected_keys = {
        "adapters.0.lora_a",
        "adapters.0.lora_b",
        "adapters.1.lora_a",
        "adapters.1.lora_b",
    }
    expected_shapes = {
        "adapters.0.lora_a": (LORA_RANK, 12_288),
        "adapters.0.lora_b": (1_536, LORA_RANK),
        "adapters.1.lora_a": (LORA_RANK, 12_288),
        "adapters.1.lora_b": (1_536, LORA_RANK),
    }
    state_sha = tensor_state_sha256(tensors)
    if (
        set(tensors) != expected_keys
        or any(tuple(tensors[key].shape) != shape for key, shape in expected_shapes.items())
        or any(tensors[key].dtype != torch.float32 for key in tensors)
        or any(not torch.isfinite(tensors[key]).all() for key in tensors)
        or checkpoint.get("tensor_keys") != sorted(expected_keys)
        or checkpoint.get("adapter_state_sha256") != state_sha
        or result["training"].get("final_adapter_state_sha256") != state_sha
    ):
        raise ValueError("V6.2 checkpoint tensor inventory or state changed")
    metadata = _read_json(metadata_path)
    expected_metadata = {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "base_checkpoint_sha256": _BASE_CHECKPOINT_FINGERPRINT,
        "base_runtime_config_effective_sha256": _BASE_RUNTIME_EFFECTIVE_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "qa_forward_path": _QA_FORWARD_PATH,
        "fixed_prefix_tokens": 258,
        "scene_latents": 256,
        "scene_hidden_dimension": 1536,
        "prefix_computed_before_question": True,
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "oracle_runtime_access": False,
        "adapter_type": "fresh_v6_2_full_reference_upper_decoder_lora",
        "target_modules": list(TARGET_MODULES),
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "dropout": 0.0,
        "trainable_parameter_count": LORA_PARAMETER_COUNT,
        "adapter_state_sha256": state_sha,
        "adapter_file_sha256": _sha256_file(weights),
        "selection_summary_sha256": _canonical_hash(result["selection"]),
        "training_release_sha256": release["sha256"],
        "parent_v6_1_terminal_failure_sha256": V6_1_FAILURE_SHA256,
    }
    if metadata != expected_metadata:
        raise ValueError("V6.2 promoted checkpoint metadata changed")
    return {
        "passed": True,
        "status": result["status"],
        "result_sha256": _sha256_file(RESULT_REPORT),
        "audit_clean": True,
        "checkpoint_exists": True,
    }


def structural_preflight() -> dict[str, Any]:
    lineage = authenticate_v6_1_terminal_failure()
    train = v1.load_training_records()
    validation = v1.load_validation_records()
    retention = v1.load_retention_corpus()
    schedule = build_v6_schedule(train)
    absent = {
        path: not _resolve(path).exists()
        for path in (TRAINING_RELEASE, TRAINING_ATTEMPT, PUBLICATION_ROOT)
    }
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_structural_preflight",
        "status": "passed_training_release_not_yet_created",
        "passed": (
            len(train) == 576
            and len(validation) == 384
            and len(retention) == 16
            and len(schedule) == _UPDATES
            and all(absent.values())
        ),
        "sealed_v6_1_terminal_failure": lineage,
        "qa_forward_path": _QA_FORWARD_PATH,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "retention_examples": len(retention),
        "schedule_updates": len(schedule),
        "trainable_parameter_count": LORA_PARAMETER_COUNT,
        "outputs_absent": absent,
        "optimizer_constructed": False,
        "model_loaded": False,
        "deferred_or_final_qa_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "preflight",
            "release",
            "authenticate-release",
            "attempt-status",
            "train",
            "authenticate",
        ),
    )
    mode = parser.parse_args(argv).mode
    if mode == "release":
        path, digest = write_training_release()
        result: dict[str, Any] = {"passed": True, "path": str(path), "sha256": digest}
    else:
        result = {
            "preflight": structural_preflight,
            "authenticate-release": authenticate_training_release,
            "attempt-status": authenticate_attempt_state,
            "train": train_and_gate,
            "authenticate": authenticate_result,
        }[mode]()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

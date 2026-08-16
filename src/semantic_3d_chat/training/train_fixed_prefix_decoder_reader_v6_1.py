"""Train one released V6.1 upper-decoder reader over fixed V54 prefixes.

This is a deliberately separate successor to the consumed V6 smoke attempt.
It authenticates a passing V6.1 full-model smoke and a create-once training
release before loading Gemma.  The only trainable tensors are a fresh rank-4
LoRA bank on the layer-32 and layer-33 MLP down projections.  Internal
selection uses every pinned validation answer, every fixed answer-varying
wrong-scene control, and all non-environmental retention prompts.  Deferred
and final assets remain blocked for the lifetime of the process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
import yaml
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT
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
    decoder_reader_lora_settings_v6,
    learning_rate_v6,
    validate_decoder_reader_surface_v6,
)
from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration import (
    CONFIG as V6_CONFIG,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward
from semantic_3d_chat.language.lora import (
    initialize_lora_adapter_state,
    install_lora_adapters,
    tensor_state_sha256,
)
from semantic_3d_chat.training import train_fixed_prefix_ple_v54 as v1

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_decoder_reader_v6_1"
TRAINING_RELEASE: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_1_training_release.json"
)
TRAINING_ATTEMPT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_1_training_attempt.json"
)
RESULT_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_decoder_reader_v6_1_result.json"
)
FILE_AUDIT_REPORT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_decoder_reader_v6_1_file_audit.json"
)
OUTPUT_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_decoder_reader_v6_1"
)

TRAINING_BOUND_PATHS: Final[tuple[str, ...]] = (
    V6_CONFIG,
    "src/semantic_3d_chat/evaluation/prediction_artifacts.py",
    "src/semantic_3d_chat/evaluation/fixed_prefix_decoder_reader_v6_1_release.py",
    "src/semantic_3d_chat/training/smoke_fixed_prefix_decoder_reader_v6_1.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_decoder_reader_v6_1.py",
    "scripts/run_gemma4_v54_fixed_prefix_decoder_reader_v6_1.sh",
    "tests/test_fixed_prefix_decoder_reader_v6_1_release.py",
    "tests/test_smoke_fixed_prefix_decoder_reader_v6_1.py",
    "tests/test_train_fixed_prefix_decoder_reader_v6_1.py",
)

_BASE_CHECKPOINT_FINGERPRINT: Final[str] = (
    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
)
_BASE_RUNTIME_EFFECTIVE_SHA256: Final[str] = (
    "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
)
_UPDATES: Final[int] = 96
_ROWS_PER_COMPONENT: Final[int] = 3
_PAIR_CE_WEIGHT: Final[float] = 0.5
_HINGE_WEIGHT: Final[float] = 4.0
_HINGE_MARGIN: Final[float] = 0.5
_BROAD_CE_WEIGHT: Final[float] = 0.5
_RETENTION_WEIGHT: Final[float] = 0.5
_GRADIENT_CLIP: Final[float] = 1.0
_MAX_MPS_DRIVER_BYTES: Final[int] = 25_000_000_000
_TRAIN_WRONG_ASSIGNMENT_SHA256: Final[str] = (
    "875cb3ed4893314494e90d563e1e961358a4fa34ccd6888545a20cfce903c5ff"
)
_VALIDATION_WRONG_ASSIGNMENT_SHA256: Final[str] = (
    "a2eaff713e8a51beec6779fc3d1720f179e2290ecaaf176d13ae1cc8d4362dcd"
)
_DEFERRED_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(57, 63)
)
_FINAL_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(25, 31)
)
_EXPECTED_FAMILIES: Final[dict[str, int]] = {
    "attribute": 44,
    "count": 12,
    "metric": 16,
    "orientation": 14,
    "presence": 8,
    "spatial_relation": 32,
    "support": 44,
}
_EXPECTED_SCOPES: Final[dict[str, int]] = {
    "cross_pair": 118,
    "same_counterfactual_pair": 52,
}


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


def _read_json(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"V6.1 JSON is missing or unsafe: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V6.1 JSON must contain an object: {source}")
    return value


def _atomic_create_json(
    path: str | Path, value: Mapping[str, Any]
) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V6.1 create-once artifact exists: {destination}")
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


def _source_hashes(paths: Sequence[str] = TRAINING_BOUND_PATHS) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"V6.1 bound source path is unsafe: {raw}")
        resolved = _resolve(path)
        if not resolved.is_file() or resolved.is_symlink():
            raise FileNotFoundError(f"V6.1 bound source is missing or linked: {raw}")
        result[path.as_posix()] = _sha256_file(resolved)
    return result


def _v6_1_smoke_authentication() -> dict[str, Any]:
    """Rebuild the release, then authenticate the report against that release."""

    module = importlib.import_module(
        "semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_1_release"
    )
    authenticate_release = getattr(
        module, "authenticate_v6_1_mps_smoke_release", None
    )
    authenticate_smoke = getattr(module, "authenticate_v6_1_passing_smoke", None)
    if not callable(authenticate_release) or not callable(authenticate_smoke):
        raise TypeError("V6.1 release has no complete callable authentication surface")
    release_result = authenticate_release()
    result = authenticate_smoke()
    if (
        not isinstance(release_result, tuple)
        or len(release_result) != 2
        or not isinstance(release_result[0], Mapping)
        or not isinstance(release_result[1], str)
        or
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], Mapping)
        or not isinstance(result[1], str)
    ):
        raise TypeError("V6.1 release/smoke authenticator return contract changed")
    release_payload, release_sha256 = release_result
    report, authenticated_sha256 = result
    if report.get("passed") is not True or report.get("status") != "passed":
        raise ValueError("V6.1 training requires its byte-authenticated passing MPS smoke")
    report_path = getattr(module, "MPS_SMOKE_REPORT", None)
    if report_path is None:
        report_path = getattr(module, "V6_1_MPS_SMOKE_REPORT", None)
    if report_path is None:
        raise AttributeError("V6.1 release does not expose its terminal smoke path")
    observed_sha256 = _sha256_file(report_path)
    if authenticated_sha256 != observed_sha256:
        raise ValueError("V6.1 passing-smoke authenticator returned the wrong digest")
    if (
        report.get("authorization_sha256") != release_sha256
        or release_payload.get("terminal_output") != str(report_path)
    ):
        raise ValueError("V6.1 smoke is not bound to the fully authenticated release")
    return {
        "passed": True,
        "report_path": str(report_path),
        "report_sha256": authenticated_sha256,
        "release_sha256": release_sha256,
        "authentication_sha256": _canonical_hash(dict(report)),
    }


def build_training_release() -> dict[str, Any]:
    smoke = _v6_1_smoke_authentication()
    sources = _source_hashes()
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_training_release",
        "status": "released_exactly_one_fixed_96_update_training_run",
        "parent_passing_mps_smoke": smoke,
        "bound_source_sha256": sources,
        "bound_source_inventory_sha256": _canonical_hash(sources),
        "authorized": {
            "optimizer_construction": True,
            "maximum_training_runs": 1,
            "exact_optimizer_updates": _UPDATES,
            "intermediate_checkpoint_or_selection": False,
            "checkpoint_write_only_after_all_internal_gates": True,
            "deferred_or_final_qa_access": False,
            "oracle_access": False,
        },
        "required_attempt_journal": TRAINING_ATTEMPT,
        "terminal_result": RESULT_REPORT,
        "output_checkpoint": OUTPUT_CHECKPOINT,
    }


def write_training_release(
    destination: str | Path = TRAINING_RELEASE,
) -> tuple[Path, str]:
    """Create the immutable authorization after the implementation is reviewed."""

    return _atomic_create_json(destination, build_training_release())


def authenticate_training_release(
    path: str | Path = TRAINING_RELEASE,
) -> dict[str, Any]:
    release = _read_json(path)
    expected = build_training_release()
    if release != expected:
        raise ValueError("V6.1 training release or a bound source changed")
    return {
        "passed": True,
        "path": str(Path(path)),
        "sha256": _sha256_file(path),
        "source_count": len(release["bound_source_sha256"]),
        "parent_smoke_sha256": release["parent_passing_mps_smoke"]["report_sha256"],
    }


def training_forbidden_roots() -> list[Path]:
    roots = [
        _resolve("data_diverse52/qa/validation.jsonl"),
        _resolve("data_diverse52/qa/test.jsonl"),
        _resolve("data_diverse28/qa/test.jsonl"),
        _resolve("data/qa/test.jsonl"),
        _resolve("reports/gemma4/questions/v56_fresh_development_validation.json"),
        _resolve("reports/gemma4/questions/test.json"),
        _resolve("reports/gemma4/predictions/v56_fresh_development_validation.jsonl"),
    ]
    for scene_id in (*_DEFERRED_SCENES, *_FINAL_SCENES):
        roots.extend(
            _resolve(root) / scene_id
            for root in (
                "data/oracle",
                "data/rendered",
                "data/features",
                "data/maps",
                "data_gemma4/features",
                "data_gemma4/maps",
                "data_gemma4/rendered",
                "data_gemma4/scene_tokens",
            )
        )
    return roots


def claim_training_attempt() -> tuple[Path, str]:
    release = authenticate_training_release()
    for path in (TRAINING_ATTEMPT, RESULT_REPORT, FILE_AUDIT_REPORT, OUTPUT_CHECKPOINT):
        if _resolve(path).exists() or _resolve(path).is_symlink():
            raise FileExistsError(f"V6.1 one-shot output already exists: {_resolve(path)}")
    return _atomic_create_json(
        TRAINING_ATTEMPT,
        {
            "schema_version": 1,
            "artifact": f"{ARTIFACT}_training_attempt",
            "status": "claimed_before_model_load",
            "training_release_sha256": release["sha256"],
            "parent_smoke_sha256": release["parent_smoke_sha256"],
            "maximum_optimizer_updates": _UPDATES,
            "checkpoint_write_authorized_before_internal_gates": False,
            "deferred_or_final_qa_access_authorized": False,
            "oracle_access_authorized": False,
        },
    )


def load_base_bundle_v6_1(audit: FileAccessAudit) -> v1.ReaderBundle:
    """Load frozen V54 and install only the fresh, exact V6 reader surface."""

    experiment_path = _resolve(V6_CONFIG)
    audit.record(experiment_path)
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(experiment, dict):
        raise TypeError("V6.1 base experiment config must be a mapping")
    runtime_config = load_runtime_config(
        _resolve(BASE_RUNTIME_CONFIG), record_file=audit.record
    )
    if effective_runtime_config_sha256(runtime_config) != _BASE_RUNTIME_EFFECTIVE_SHA256:
        raise ValueError("V6.1 effective runtime configuration changed")
    base_checkpoint = _resolve(BASE_CHECKPOINT)
    for source in sorted(path for path in base_checkpoint.rglob("*") if path.is_file()):
        audit.record(source)
    base_fingerprint, _ = checkpoint_fingerprint(base_checkpoint)
    if base_fingerprint != _BASE_CHECKPOINT_FINGERPRINT:
        raise ValueError("V6.1 frozen V54 checkpoint fingerprint changed")
    scene_id = v1.TRAIN_SCENES[0]
    runtime = StaticChatRuntime.load(
        runtime_config,
        scene_id,
        checkpoint=base_checkpoint,
        audit=audit,
        local_files_only=True,
    )
    if runtime.language.device.type != "mps":
        raise RuntimeError("V6.1 released training requires local MPS execution")
    prefix_root = _resolve(v1.PREFIX_CACHE)
    manifest_path = prefix_root / "manifest.json"
    audit.record(manifest_path)
    prefix_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for prefix_scene in (*v1.TRAIN_SCENES, *v1.VALIDATION_SCENES):
        filename = prefix_manifest["scenes"][prefix_scene]["filename"]
        audit.record(prefix_root / filename)
    prefixes = v1.load_prefixes()
    if runtime.scene_prefix_hash != v1.prefix_sha256(prefixes[scene_id]):
        raise ValueError("V6.1 cached prefix differs from the frozen V54 runtime")
    runtime.language.model.requires_grad_(False)
    if getattr(runtime.language.model, "is_gradient_checkpointing", False):
        raise RuntimeError("V6.1 decoder gradient checkpointing must remain disabled")
    validate_decoder_reader_surface_v6(runtime.language.model)
    installation = install_lora_adapters(
        runtime.language.model, decoder_reader_lora_settings_v6()
    )
    if installation is None:
        raise RuntimeError("V6.1 upper-decoder LoRA was not installed")
    initialize_lora_adapter_state(installation, seed=INITIALIZATION_SEED)
    installation.assert_only_lora_trainable(runtime.language.model)
    installation.validate_state()
    if (
        installation.parameter_count != LORA_PARAMETER_COUNT
        or installation.state_sha256() != INITIAL_STATE_SHA256
        or tuple(installation.target_names) != TARGET_MODULES
    ):
        raise ValueError("V6.1 adapter shape or deterministic initialization changed")
    return v1.ReaderBundle(runtime, installation, prefixes, experiment)


def answer_nll(
    bundle: v1.ReaderBundle, prefix: torch.Tensor, row: v1.ReaderRecord
) -> torch.Tensor:
    prepared = v1._prepared_batch(bundle, prefix, row)
    result = answer_tail_forward(bundle.language, prepared).mean_nll
    if result.ndim != 0 or not torch.isfinite(result):
        raise RuntimeError("V6.1 answer NLL is invalid")
    return result


def contrastive_row_objective(
    correct_nll: torch.Tensor, wrong_nll: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if correct_nll.ndim != 0 or wrong_nll.ndim != 0:
        raise ValueError("V6.1 contrastive objective requires two scalar NLLs")
    if not torch.isfinite(correct_nll) or not torch.isfinite(wrong_nll):
        raise ValueError("V6.1 contrastive objective received a nonfinite NLL")
    margin = wrong_nll - correct_nll
    hinge = F.relu(_HINGE_MARGIN - margin)
    loss = (
        (_PAIR_CE_WEIGHT / _ROWS_PER_COMPONENT) * correct_nll
        + (_HINGE_WEIGHT / _ROWS_PER_COMPONENT) * hinge
    )
    return loss, {"margin": margin, "hinge": hinge}


def broad_row_objective(nll: torch.Tensor) -> torch.Tensor:
    if nll.ndim != 0 or not torch.isfinite(nll):
        raise ValueError("V6.1 broad objective requires one finite scalar NLL")
    return (_BROAD_CE_WEIGHT / _ROWS_PER_COMPONENT) * nll


def retention_objective(kl: torch.Tensor) -> torch.Tensor:
    if kl.ndim != 0 or not torch.isfinite(kl):
        raise ValueError("V6.1 retention objective requires one finite scalar KL")
    return _RETENTION_WEIGHT * kl


def _row_key(row: v1.ReaderRecord) -> tuple[str, str]:
    return row.scene_id, row.question_id


def _wrong_assignment_hash(assignments: Mapping[tuple[str, str], str]) -> str:
    return _canonical_hash(
        [
            {"row": list(key), "wrong_scene_id": assignments[key]}
            for key in sorted(assignments)
        ]
    )


def _tuple_mapping_hash(values: Mapping[tuple[str, str], float]) -> str:
    return _canonical_hash(
        [
            {"scene_id": key[0], "question_id": key[1], "value": values[key]}
            for key in sorted(values)
        ]
    )


def _selected_scope(
    row: v1.ReaderRecord,
    wrong_scene: str,
    rows_by_scene_question: Mapping[tuple[str, str], v1.ReaderRecord],
) -> str:
    selected = rows_by_scene_question[(wrong_scene, row.question)]
    return (
        "same_counterfactual_pair"
        if row.pair_id is not None and selected.pair_id == row.pair_id
        else "cross_pair"
    )


@torch.inference_mode()
def evaluate_teacher_forcing_v6_1(
    bundle: v1.ReaderBundle,
    rows: Sequence[v1.ReaderRecord],
) -> dict[str, Any]:
    """Evaluate all 384 correct and all 170 fixed wrong-prefix examples."""

    if len(rows) != 384:
        raise ValueError("V6.1 teacher evaluation requires all 384 validation rows")
    bundle.installation.eval()
    correct: dict[tuple[str, str], float] = {}
    for row in rows:
        value = answer_nll(bundle, bundle.prefixes[row.scene_id], row)
        correct[_row_key(row)] = float(value.detach().cpu())

    assignments = answer_varying_wrong_prefixes(rows)
    assignment_hash = _wrong_assignment_hash(assignments)
    if len(assignments) != 170 or assignment_hash != _VALIDATION_WRONG_ASSIGNMENT_SHA256:
        raise ValueError("V6.1 fixed validation wrong-prefix inventory changed")
    wrong: dict[tuple[str, str], float] = {}
    for row in rows:
        key = _row_key(row)
        if key not in assignments:
            continue
        value = answer_nll(bundle, bundle.prefixes[assignments[key]], row)
        wrong[key] = float(value.detach().cpu())
    margins = {key: wrong[key] - correct[key] for key in wrong}

    curated_rows = [row for row in rows if row.changed]
    if len(curated_rows) != 52:
        raise ValueError("V6.1 curated validation inventory changed")
    units: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in curated_rows:
        if row.pair_id is None or row.pair_question_key is None:
            raise ValueError("V6.1 curated validation row lacks pair identity")
        units[(row.pair_id, row.pair_question_key)].append(margins[_row_key(row)])
    if len(units) != 26 or any(len(values) != 2 for values in units.values()):
        raise ValueError("V6.1 curated validation units changed")

    rows_by_scene_question = {(row.scene_id, row.question): row for row in rows}
    if len(rows_by_scene_question) != len(rows):
        raise ValueError("V6.1 validation scene/question keys are not unique")
    family_values: defaultdict[str, list[float]] = defaultdict(list)
    scope_values: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = _row_key(row)
        if key not in assignments:
            continue
        family_values[row.answer_type].append(margins[key])
        scope_values[
            _selected_scope(row, assignments[key], rows_by_scene_question)
        ].append(margins[key])
    family_counts = {key: len(values) for key, values in sorted(family_values.items())}
    scope_counts = {key: len(values) for key, values in sorted(scope_values.items())}
    if family_counts != _EXPECTED_FAMILIES or scope_counts != _EXPECTED_SCOPES:
        raise ValueError("V6.1 validation family or scope inventory changed")

    family_rates = {
        key: sum(value > 0.0 for value in values) / len(values)
        for key, values in sorted(family_values.items())
    }
    scope_rates = {
        key: sum(value > 0.0 for value in values) / len(values)
        for key, values in sorted(scope_values.items())
    }
    curated_margins = [margins[_row_key(row)] for row in curated_rows]
    expanded_positive = sum(value > 0.0 for value in margins.values())
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
        "curated_positive_margin_rate": (
            sum(value > 0.0 for value in curated_margins) / len(curated_margins)
        ),
        "curated_complete_units": sum(
            all(value > 0.0 for value in values) for values in units.values()
        ),
        "curated_unit_count": len(units),
        "expanded_margin_mean": sum(margins.values()) / len(margins),
        "expanded_positive_margin_sides": expanded_positive,
        "expanded_side_count": len(margins),
        "expanded_positive_margin_rate": expanded_positive / len(margins),
        "family_counts": family_counts,
        "family_positive_margin_rates": family_rates,
        "family_macro_positive_margin_rate": sum(family_rates.values()) / len(family_rates),
        "scope_counts": scope_counts,
        "scope_positive_margin_rates": scope_rates,
        "scope_macro_positive_margin_rate": sum(scope_rates.values()) / len(scope_rates),
        "correct_nll_by_row": correct_records,
        "expanded_margin_by_row": margin_records,
        "correct_nll_sha256": _tuple_mapping_hash(correct),
        "expanded_margin_sha256": _tuple_mapping_hash(margins),
        "wrong_prefix_assignment_sha256": assignment_hash,
        "evaluation_microbatch_size": 1,
        "answer_logit_positions_only": True,
    }


def teacher_and_retention_checks(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    retention: Mapping[str, Any],
) -> dict[str, bool]:
    baseline_family = baseline.get("family_positive_margin_rates")
    candidate_family = candidate.get("family_positive_margin_rates")
    baseline_scope = baseline.get("scope_positive_margin_rates")
    candidate_scope = candidate.get("scope_positive_margin_rates")
    if (
        not isinstance(baseline_family, Mapping)
        or not isinstance(candidate_family, Mapping)
        or set(baseline_family) != set(_EXPECTED_FAMILIES)
        or set(candidate_family) != set(_EXPECTED_FAMILIES)
        or not isinstance(baseline_scope, Mapping)
        or not isinstance(candidate_scope, Mapping)
        or set(baseline_scope) != set(_EXPECTED_SCOPES)
        or set(candidate_scope) != set(_EXPECTED_SCOPES)
    ):
        raise ValueError("V6.1 gate strata are incomplete or changed")
    checks = {
        "validation_answer_nll_improvement": (
            baseline["answer_nll_mean"] - candidate["answer_nll_mean"] >= 0.03
        ),
        "curated_positive_margin_rate": (
            candidate["curated_positive_margin_rate"] >= 0.65
        ),
        "curated_positive_margin_rate_delta": (
            candidate["curated_positive_margin_rate"]
            - baseline["curated_positive_margin_rate"]
            >= 0.10
        ),
        "curated_complete_unit_delta": (
            candidate["curated_complete_units"] - baseline["curated_complete_units"] >= 3
        ),
        "expanded_positive_margin_rate": (
            candidate["expanded_positive_margin_rate"] >= 0.65
        ),
        "expanded_positive_margin_rate_delta": (
            candidate["expanded_positive_margin_rate"]
            - baseline["expanded_positive_margin_rate"]
            >= 0.10
        ),
        "family_macro_positive_margin_rate": (
            candidate["family_macro_positive_margin_rate"] >= 0.65
        ),
        "family_macro_positive_margin_rate_delta": (
            candidate["family_macro_positive_margin_rate"]
            - baseline["family_macro_positive_margin_rate"]
            >= 0.10
        ),
        "every_family_positive_margin_rate": all(
            float(candidate_family[key]) >= 0.50 for key in _EXPECTED_FAMILIES
        ),
        "every_family_nonnegative_delta": all(
            float(candidate_family[key]) - float(baseline_family[key]) >= 0.0
            for key in _EXPECTED_FAMILIES
        ),
        "scope_macro_positive_margin_rate": (
            candidate["scope_macro_positive_margin_rate"] >= 0.65
        ),
        "scope_macro_positive_margin_rate_delta": (
            candidate["scope_macro_positive_margin_rate"]
            - baseline["scope_macro_positive_margin_rate"]
            >= 0.10
        ),
        "every_scope_positive_margin_rate": all(
            float(candidate_scope[key]) >= 0.55 for key in _EXPECTED_SCOPES
        ),
        "every_scope_nonnegative_delta": all(
            float(candidate_scope[key]) - float(baseline_scope[key]) >= 0.0
            for key in _EXPECTED_SCOPES
        ),
        "retention_mean_ce_increase": retention["mean_ce_increase_nats"] <= 0.03,
        "retention_mean_kl": retention["mean_kl_nats"] <= 0.02,
        "retention_next_token_top1_agreement": (
            retention["next_token_top1_agreement"] >= 0.98
        ),
    }
    return {key: bool(value) for key, value in checks.items()}


def optimizer_kwargs() -> dict[str, Any]:
    """Expose every pinned AdamW choice for construction and unit tests."""

    return {
        "lr": learning_rate_v6(1),
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
    }


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


def _stage_checkpoint(
    bundle: v1.ReaderBundle,
    selection: Mapping[str, Any],
    *,
    training_release_sha256: str,
    parent_smoke_sha256: str,
) -> StagedCheckpoint:
    destination = _resolve(OUTPUT_CHECKPOINT)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("V6.1 checkpoint target already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        weights = temporary / "adapter.safetensors"
        state = {
            key: tensor.detach().float().cpu().contiguous()
            for key, tensor in bundle.installation.state_module.state_dict().items()
        }
        if set(state) != {
            "adapters.0.lora_a",
            "adapters.0.lora_b",
            "adapters.1.lora_a",
            "adapters.1.lora_b",
        }:
            raise ValueError("V6.1 checkpoint acquired non-V6 tensors")
        save_file(state, weights)
        metadata = {
            "schema_version": 1,
            "artifact": ARTIFACT,
            "base_checkpoint_sha256": _BASE_CHECKPOINT_FINGERPRINT,
            "base_runtime_config_effective_sha256": _BASE_RUNTIME_EFFECTIVE_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "fixed_prefix_tokens": 258,
            "scene_latents": 256,
            "scene_hidden_dimension": 1536,
            "prefix_computed_before_question": True,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "oracle_runtime_access": False,
            "adapter_type": "fresh_v6_only_upper_decoder_lora",
            "target_modules": list(TARGET_MODULES),
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": 0.0,
            "trainable_parameter_count": LORA_PARAMETER_COUNT,
            "adapter_state_sha256": bundle.installation.state_sha256(),
            "adapter_file_sha256": _sha256_file(weights),
            "selection_summary_sha256": _canonical_hash(selection),
            "training_release_sha256": training_release_sha256,
            "parent_passing_smoke_sha256": parent_smoke_sha256,
        }
        metadata_path = temporary / "runtime_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        runtime_metadata_sha256 = _sha256_file(metadata_path)
        published = {
            "path": str(destination.relative_to(PROJECT_ROOT)),
            "adapter_file_sha256": metadata["adapter_file_sha256"],
            "runtime_metadata_sha256": runtime_metadata_sha256,
            "adapter_state_sha256": metadata["adapter_state_sha256"],
            "tensor_keys": sorted(state),
        }
        return StagedCheckpoint(temporary, published)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _commit_staged_checkpoint(staged: StagedCheckpoint) -> dict[str, Any]:
    destination = _resolve(OUTPUT_CHECKPOINT)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("V6.1 checkpoint target already exists at commit")
    os.rename(staged.directory, destination)
    return staged.published


def _publish_checkpoint(
    bundle: v1.ReaderBundle,
    selection: Mapping[str, Any],
    *,
    training_release_sha256: str,
    parent_smoke_sha256: str,
) -> dict[str, Any]:
    """Compatibility helper used only by model-free atomic-publication tests."""

    staged = _stage_checkpoint(
        bundle,
        selection,
        training_release_sha256=training_release_sha256,
        parent_smoke_sha256=parent_smoke_sha256,
    )
    return _commit_staged_checkpoint(staged)


def _training_trace_item(
    *,
    update: int,
    learning_rate: float,
    contrastive_losses: Sequence[float],
    margins: Sequence[float],
    broad_nlls: Sequence[float],
    retention_kl: float,
    gradient: float,
    adapter_hash: str,
) -> dict[str, Any]:
    return {
        "update": update,
        "learning_rate": learning_rate,
        "mean_contrastive_row_objective": sum(contrastive_losses)
        / len(contrastive_losses),
        "mean_preupdate_wrong_minus_correct_margin": sum(margins) / len(margins),
        "mean_broad_nll": sum(broad_nlls) / len(broad_nlls),
        "retention_kl": retention_kl,
        "preclip_gradient_l2": gradient,
        "adapter_state_sha256": adapter_hash,
    }


def _execute_training(
    *, release: Mapping[str, Any], attempt_sha256: str, audit: FileAccessAudit
) -> TrainingOutcome:
    started = time.perf_counter()
    torch.manual_seed(INITIALIZATION_SEED)
    random.seed(INITIALIZATION_SEED)
    bundle = load_base_bundle_v6_1(audit)
    train_rows = v1.load_training_records()
    validation_rows = v1.load_validation_records()
    retention_corpus = v1.load_retention_corpus()
    schedule = build_v6_schedule(train_rows)
    wrong_assignments = answer_varying_wrong_prefixes(train_rows)
    if (
        len(schedule) != _UPDATES
        or len(wrong_assignments) != 288
        or _wrong_assignment_hash(wrong_assignments) != _TRAIN_WRONG_ASSIGNMENT_SHA256
        or len(retention_corpus) != 16
    ):
        raise ValueError("V6.1 released training inventory changed")

    # The complete baseline selection population and all retention teachers are
    # evaluated before an optimizer object exists.
    teachers = v1.retention_baseline(bundle, retention_corpus)
    baseline_teacher = evaluate_teacher_forcing_v6_1(bundle, validation_rows)
    baseline_retention = v1.evaluate_retention(bundle, retention_corpus, teachers)
    optimizer = torch.optim.AdamW(bundle.installation.parameters(), **optimizer_kwargs())

    trace: list[dict[str, Any]] = []
    maximum_gradient = 0.0
    optimizer_steps = 0
    bundle.installation.train()
    for update_index, update in enumerate(schedule, start=1):
        if len(update.contrastive) != 3 or len(update.broad) != 3:
            raise ValueError("V6.1 update lost its exact 3+3 row structure")
        current_lr = learning_rate_v6(update_index)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        contrastive_losses: list[float] = []
        margins: list[float] = []
        broad_nlls: list[float] = []

        for row in update.contrastive:
            key = _row_key(row)
            correct = answer_nll(bundle, bundle.prefixes[row.scene_id], row)
            wrong = answer_nll(bundle, bundle.prefixes[wrong_assignments[key]], row)
            loss, diagnostics = contrastive_row_objective(correct, wrong)
            loss.backward()
            contrastive_losses.append(float(loss.detach().cpu()))
            margins.append(float(diagnostics["margin"].detach().cpu()))
        for row in update.broad:
            nll = answer_nll(bundle, bundle.prefixes[row.scene_id], row)
            broad_row_objective(nll).backward()
            broad_nlls.append(float(nll.detach().cpu()))
        retention_index = (update_index - 1) % len(retention_corpus)
        retention_kl = v1.retention_kl_loss(
            bundle, retention_corpus[retention_index], teachers[retention_index]
        )
        retention_objective(retention_kl).backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(
                bundle.installation.parameters(), _GRADIENT_CLIP
            )
            .detach()
            .cpu()
        )
        if not math.isfinite(gradient) or gradient <= 0.0:
            raise RuntimeError("V6.1 preclip gradient norm is invalid")
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        optimizer_steps += 1
        bundle.installation.validate_state()
        bundle.installation.assert_only_lora_trainable(bundle.language.model)
        item = _training_trace_item(
            update=update_index,
            learning_rate=current_lr,
            contrastive_losses=contrastive_losses,
            margins=margins,
            broad_nlls=broad_nlls,
            retention_kl=float(retention_kl.detach().cpu()),
            gradient=gradient,
            adapter_hash=bundle.installation.state_sha256(),
        )
        trace.append(item)
        print(
            json.dumps(
                {"phase": "fixed_prefix_decoder_reader_v6_1_train", **item},
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
    if optimizer_steps != _UPDATES:
        raise RuntimeError("V6.1 did not execute exactly 96 optimizer steps")

    bundle.installation.eval()
    candidate_teacher = evaluate_teacher_forcing_v6_1(bundle, validation_rows)
    candidate_retention = v1.evaluate_retention(bundle, retention_corpus, teachers)
    checks = teacher_and_retention_checks(
        baseline_teacher, candidate_teacher, candidate_retention
    )
    greedy: dict[str, Any] | None = None
    if all(checks.values()):
        greedy = v1.evaluate_greedy(bundle, validation_rows)
        if greedy.get("row_count") != 96:
            raise RuntimeError("V6.1 greedy gate did not use the fixed 96-row population")
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
        raise RuntimeError("V6.1 training exceeded the locked MPS driver-memory ceiling")
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_terminal_result",
        "status": "passed_checkpoint_published" if passed else "failed_no_checkpoint",
        "passed": passed,
        "promotion_eligible": passed,
        "checkpoint_published": False,
        "checkpoint": None,
        "training_release_sha256": release["sha256"],
        "parent_passing_smoke_sha256": release["parent_smoke_sha256"],
        "training_attempt_sha256": attempt_sha256,
        "training": {
            "optimizer": "AdamW",
            "optimizer_kwargs": optimizer_kwargs(),
            "updates": optimizer_steps,
            "contrastive_rows_consumed_exactly_once": 288,
            "broad_rows_consumed_exactly_once": 288,
            "retention_examples": 16,
            "retention_exposures_per_example": 6,
            "trainable_parameter_count": bundle.installation.parameter_count,
            "maximum_preclip_gradient_l2": maximum_gradient,
            "initial_trace": trace[:3],
            "milestone_trace": [trace[index - 1] for index in (24, 48, 72, 96)],
            "final_trace": trace[-3:],
            "trace": trace,
            "trace_sha256": _canonical_hash(trace),
            "final_adapter_state_sha256": bundle.installation.state_sha256(),
            "intermediate_selection_or_checkpoint": False,
            "gradient_checkpointing": False,
        },
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
            "deferred_scene_ids": list(_DEFERRED_SCENES),
            "final_scene_ids": list(_FINAL_SCENES),
            "accessed": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "memory": memory,
    }
    staged_checkpoint: StagedCheckpoint | None = None
    if passed:
        staged_checkpoint = _stage_checkpoint(
            bundle,
            selection,
            training_release_sha256=str(release["sha256"]),
            parent_smoke_sha256=str(release["parent_smoke_sha256"]),
        )
        report["checkpoint"] = staged_checkpoint.published
        report["checkpoint_published"] = True
    return TrainingOutcome(report, staged_checkpoint)


def train_and_gate() -> dict[str, Any]:
    """Consume the one-shot release and write one terminal result."""

    audit = FileAccessAudit(
        training_forbidden_roots(),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    claimed = False
    attempt_sha256: str | None = None
    release: dict[str, Any] | None = None
    outcome: TrainingOutcome | None = None
    report: dict[str, Any] | None = None
    failure: BaseException | None = None
    with audit:
        try:
            release = authenticate_training_release()
            _attempt_path, attempt_sha256 = claim_training_attempt()
            claimed = True
            outcome = _execute_training(
                release=release, attempt_sha256=attempt_sha256, audit=audit
            )
        except BaseException as exc:  # noqa: BLE001 - persist all claimed outcomes
            failure = exc

    audit_report = _audit_payload(audit)
    if audit_report["passed"] is not True and failure is None:
        failure = RuntimeError("V6.1 file audit found forbidden access")
    if claimed:
        if (
            _resolve(FILE_AUDIT_REPORT).exists()
            or _resolve(FILE_AUDIT_REPORT).is_symlink()
        ):
            raise FileExistsError("V6.1 file-audit terminal artifact already exists")
        try:
            _atomic_create_json(FILE_AUDIT_REPORT, audit_report)
        except BaseException:
            if outcome is not None and outcome.staged_checkpoint is not None:
                shutil.rmtree(outcome.staged_checkpoint.directory, ignore_errors=True)
            raise
        if failure is not None:
            if outcome is not None and outcome.staged_checkpoint is not None:
                shutil.rmtree(outcome.staged_checkpoint.directory, ignore_errors=True)
            if _resolve(OUTPUT_CHECKPOINT).exists():
                shutil.rmtree(_resolve(OUTPUT_CHECKPOINT), ignore_errors=True)
            report = {
                "schema_version": 1,
                "artifact": f"{ARTIFACT}_terminal_result",
                "status": "failed_terminal_attempt_consumed_no_checkpoint",
                "passed": False,
                "promotion_eligible": False,
                "checkpoint_published": False,
                "checkpoint": None,
                "training_release_sha256": None if release is None else release["sha256"],
                "parent_passing_smoke_sha256": (
                    None if release is None else release["parent_smoke_sha256"]
                ),
                "training_attempt_sha256": attempt_sha256,
                "error_type": type(failure).__name__,
                "error": str(failure),
                "file_audit_report": FILE_AUDIT_REPORT,
                "file_audit_sha256": _sha256_file(FILE_AUDIT_REPORT),
                "deferred_or_final_qa_accessed": bool(audit.forbidden_accesses()),
                "oracle_accessed": any(
                    "oracle" in {part.casefold() for part in Path(path).parts}
                    for path in audit.forbidden_accesses()
                ),
            }
            _atomic_create_json(RESULT_REPORT, report)
        else:
            if outcome is None:
                raise RuntimeError("V6.1 claimed training produced no outcome")
            report = outcome.report
            report["file_audit_report"] = FILE_AUDIT_REPORT
            report["file_audit_sha256"] = _sha256_file(FILE_AUDIT_REPORT)
            report["loaded_file_count"] = audit_report["loaded_file_count"]
            staged = outcome.staged_checkpoint
            checkpoint_committed = False
            try:
                if staged is not None:
                    _commit_staged_checkpoint(staged)
                    checkpoint_committed = True
                _atomic_create_json(RESULT_REPORT, report)
            except BaseException as commit_failure:
                if checkpoint_committed:
                    shutil.rmtree(_resolve(OUTPUT_CHECKPOINT), ignore_errors=True)
                elif staged is not None:
                    shutil.rmtree(staged.directory, ignore_errors=True)
                # A create-once link either succeeds completely or leaves no
                # output. Refuse to overwrite any unexpected terminal file.
                if _resolve(RESULT_REPORT).exists():
                    raise RuntimeError(
                        "V6.1 terminal result committed but checkpoint transaction failed"
                    ) from commit_failure
                failure = commit_failure
                report = {
                    "schema_version": 1,
                    "artifact": f"{ARTIFACT}_terminal_result",
                    "status": "failed_terminal_attempt_consumed_no_checkpoint",
                    "passed": False,
                    "promotion_eligible": False,
                    "checkpoint_published": False,
                    "checkpoint": None,
                    "training_release_sha256": release["sha256"],
                    "parent_passing_smoke_sha256": release[
                        "parent_smoke_sha256"
                    ],
                    "training_attempt_sha256": attempt_sha256,
                    "error_type": type(commit_failure).__name__,
                    "error": str(commit_failure),
                    "file_audit_report": FILE_AUDIT_REPORT,
                    "file_audit_sha256": _sha256_file(FILE_AUDIT_REPORT),
                    "deferred_or_final_qa_accessed": False,
                    "oracle_accessed": False,
                }
                _atomic_create_json(RESULT_REPORT, report)
    if failure is not None:
        raise failure
    if report is None:
        raise RuntimeError("V6.1 training ended without a report")
    if report["passed"] is not True and _resolve(OUTPUT_CHECKPOINT).exists():
        raise RuntimeError("V6.1 failed run unexpectedly published a checkpoint")
    return report


def structural_preflight() -> dict[str, Any]:
    smoke = _v6_1_smoke_authentication()
    train = v1.load_training_records()
    validation = v1.load_validation_records()
    retention = v1.load_retention_corpus()
    schedule = build_v6_schedule(train)
    assignments = answer_varying_wrong_prefixes(validation)
    absent = {
        path: not _resolve(path).exists()
        for path in (
            TRAINING_RELEASE,
            TRAINING_ATTEMPT,
            RESULT_REPORT,
            FILE_AUDIT_REPORT,
            OUTPUT_CHECKPOINT,
        )
    }
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_structural_preflight",
        "status": "passed_training_release_not_yet_created",
        "passed": (
            len(train) == 576
            and len(validation) == 384
            and len(retention) == 16
            and len(schedule) == 96
            and len(assignments) == 170
            and all(absent.values())
        ),
        "parent_passing_smoke": smoke,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "validation_wrong_prefix_rows": len(assignments),
        "retention_examples": len(retention),
        "schedule_updates": len(schedule),
        "trainable_parameter_count": LORA_PARAMETER_COUNT,
        "outputs_absent": absent,
        "optimizer_constructed": False,
        "model_loaded": False,
        "deferred_or_final_qa_accessed": False,
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _row_value_mapping(value: object, expected_count: int) -> dict[tuple[str, str], float]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError("V6.1 teacher row evidence count changed")
    result: dict[tuple[str, str], float] = {}
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {
            "scene_id",
            "question_id",
            "value",
        }:
            raise ValueError("V6.1 teacher row evidence schema changed")
        key = (raw["scene_id"], raw["question_id"])
        if (
            not all(isinstance(item, str) and item for item in key)
            or not _finite_number(raw["value"])
            or key in result
        ):
            raise ValueError("V6.1 teacher row evidence is invalid or duplicated")
        result[key] = float(raw["value"])
    return result


def _authenticate_teacher_evidence(
    metrics: object, rows: Sequence[v1.ReaderRecord]
) -> None:
    expected_fields = {
        "answer_nll_mean",
        "answer_nll_count",
        "curated_margin_mean",
        "curated_positive_margin_sides",
        "curated_side_count",
        "curated_positive_margin_rate",
        "curated_complete_units",
        "curated_unit_count",
        "expanded_margin_mean",
        "expanded_positive_margin_sides",
        "expanded_side_count",
        "expanded_positive_margin_rate",
        "family_counts",
        "family_positive_margin_rates",
        "family_macro_positive_margin_rate",
        "scope_counts",
        "scope_positive_margin_rates",
        "scope_macro_positive_margin_rate",
        "correct_nll_by_row",
        "expanded_margin_by_row",
        "correct_nll_sha256",
        "expanded_margin_sha256",
        "wrong_prefix_assignment_sha256",
        "evaluation_microbatch_size",
        "answer_logit_positions_only",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != expected_fields:
        raise ValueError("V6.1 teacher evidence schema changed")
    correct = _row_value_mapping(metrics["correct_nll_by_row"], 384)
    margins = _row_value_mapping(metrics["expanded_margin_by_row"], 170)
    row_index = {_row_key(row): row for row in rows}
    assignments = answer_varying_wrong_prefixes(rows)
    if set(correct) != set(row_index) or set(margins) != set(assignments):
        raise ValueError("V6.1 teacher evidence row population changed")
    curated = [row for row in rows if row.changed]
    curated_values = [margins[_row_key(row)] for row in curated]
    units: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in curated:
        if row.pair_id is None or row.pair_question_key is None:
            raise ValueError("V6.1 curated evidence lost pair identity")
        units[(row.pair_id, row.pair_question_key)].append(margins[_row_key(row)])
    by_scene_question = {(row.scene_id, row.question): row for row in rows}
    families: defaultdict[str, list[float]] = defaultdict(list)
    scopes: defaultdict[str, list[float]] = defaultdict(list)
    for key, margin in margins.items():
        row = row_index[key]
        families[row.answer_type].append(margin)
        scopes[_selected_scope(row, assignments[key], by_scene_question)].append(margin)
    family_rates = {
        key: sum(value > 0 for value in values) / len(values)
        for key, values in sorted(families.items())
    }
    scope_rates = {
        key: sum(value > 0 for value in values) / len(values)
        for key, values in sorted(scopes.items())
    }
    computed: dict[str, Any] = {
        "answer_nll_mean": sum(correct.values()) / len(correct),
        "answer_nll_count": len(correct),
        "curated_margin_mean": sum(curated_values) / len(curated_values),
        "curated_positive_margin_sides": sum(value > 0 for value in curated_values),
        "curated_side_count": len(curated_values),
        "curated_positive_margin_rate": sum(value > 0 for value in curated_values)
        / len(curated_values),
        "curated_complete_units": sum(
            all(value > 0 for value in values) for values in units.values()
        ),
        "curated_unit_count": len(units),
        "expanded_margin_mean": sum(margins.values()) / len(margins),
        "expanded_positive_margin_sides": sum(value > 0 for value in margins.values()),
        "expanded_side_count": len(margins),
        "expanded_positive_margin_rate": sum(value > 0 for value in margins.values())
        / len(margins),
        "family_counts": {key: len(values) for key, values in sorted(families.items())},
        "family_positive_margin_rates": family_rates,
        "family_macro_positive_margin_rate": sum(family_rates.values())
        / len(family_rates),
        "scope_counts": {key: len(values) for key, values in sorted(scopes.items())},
        "scope_positive_margin_rates": scope_rates,
        "scope_macro_positive_margin_rate": sum(scope_rates.values()) / len(scope_rates),
        "correct_nll_sha256": _tuple_mapping_hash(correct),
        "expanded_margin_sha256": _tuple_mapping_hash(margins),
        "wrong_prefix_assignment_sha256": _wrong_assignment_hash(assignments),
        "evaluation_microbatch_size": 1,
        "answer_logit_positions_only": True,
    }
    for key, expected in computed.items():
        observed = metrics.get(key)
        if isinstance(expected, float):
            if not _finite_number(observed) or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"V6.1 teacher evidence derived field changed: {key}")
        elif observed != expected:
            raise ValueError(f"V6.1 teacher evidence derived field changed: {key}")


def _authenticate_retention_evidence(metrics: object) -> None:
    fields = {
        "example_count",
        "mean_ce_increase_nats",
        "maximum_ce_increase_nats",
        "mean_kl_nats",
        "maximum_kl_nats",
        "next_token_top1_agreement",
        "metrics_sha256",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != fields:
        raise ValueError("V6.1 retention evidence schema changed")
    if (
        metrics.get("example_count") != 16
        or not all(
            _finite_number(metrics.get(key))
            for key in fields - {"example_count", "metrics_sha256"}
        )
        or not 0.0 <= float(metrics["next_token_top1_agreement"]) <= 1.0
        or not _is_sha256(metrics.get("metrics_sha256"))
    ):
        raise ValueError("V6.1 retention evidence values changed")


def _authenticate_greedy_evidence(metrics: object) -> bool:
    if metrics is None:
        return False
    fields = {
        "row_count",
        "baseline_exact_correct",
        "baseline_exact_accuracy",
        "candidate_exact_correct",
        "candidate_exact_accuracy",
        "exact_accuracy_delta",
        "prediction_hashes_sha256",
        "question_dependent_scene_retrieval",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != fields:
        raise ValueError("V6.1 greedy evidence schema changed")
    count = metrics.get("row_count")
    baseline = metrics.get("baseline_exact_correct")
    candidate = metrics.get("candidate_exact_correct")
    if (
        count != 96
        or type(baseline) is not int
        or type(candidate) is not int
        or not 0 <= baseline <= count
        or not 0 <= candidate <= count
        or metrics.get("question_dependent_scene_retrieval") is not False
        or not _is_sha256(metrics.get("prediction_hashes_sha256"))
    ):
        raise ValueError("V6.1 greedy evidence values changed")
    expected = {
        "baseline_exact_accuracy": baseline / count,
        "candidate_exact_accuracy": candidate / count,
        "exact_accuracy_delta": (candidate - baseline) / count,
    }
    if any(
        not _finite_number(metrics.get(key))
        or not math.isclose(
            float(metrics[key]), value, rel_tol=0.0, abs_tol=1e-12
        )
        for key, value in expected.items()
    ):
        raise ValueError("V6.1 greedy derived metrics changed")
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
        raise ValueError("V6.1 selection evidence schema changed")
    _authenticate_teacher_evidence(selection["baseline_teacher"], validation_rows)
    _authenticate_teacher_evidence(selection["candidate_teacher"], validation_rows)
    _authenticate_retention_evidence(selection["baseline_retention"])
    _authenticate_retention_evidence(selection["candidate_retention"])
    teacher_checks = teacher_and_retention_checks(
        selection["baseline_teacher"],
        selection["candidate_teacher"],
        selection["candidate_retention"],
    )
    greedy_passed = _authenticate_greedy_evidence(selection["greedy"])
    expected_checks = {
        **teacher_checks,
        "greedy_exact_accuracy_delta": greedy_passed,
    }
    if selection.get("checks") != expected_checks:
        raise ValueError("V6.1 stored selection checks were not independently derived")
    passed = all(expected_checks.values())
    if selection.get("passed") is not passed:
        raise ValueError("V6.1 stored selection decision changed")
    if all(teacher_checks.values()) != (selection.get("greedy") is not None):
        raise ValueError("V6.1 greedy evaluation was not correctly delayed")
    return passed


def _authenticate_training_trace(training: object) -> None:
    fields = {
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
        raise ValueError("V6.1 training evidence schema changed")
    expected_optimizer = json.loads(json.dumps(optimizer_kwargs()))
    trace = training.get("trace")
    if (
        training.get("optimizer") != "AdamW"
        or training.get("optimizer_kwargs") != expected_optimizer
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
        or not _finite_number(training.get("maximum_preclip_gradient_l2"))
        or float(training["maximum_preclip_gradient_l2"]) <= 0
    ):
        raise ValueError("V6.1 training evidence values changed")
    trace_fields = {
        "update",
        "learning_rate",
        "mean_contrastive_row_objective",
        "mean_preupdate_wrong_minus_correct_margin",
        "mean_broad_nll",
        "retention_kl",
        "preclip_gradient_l2",
        "adapter_state_sha256",
    }
    for update, item in enumerate(trace, start=1):
        if (
            not isinstance(item, Mapping)
            or set(item) != trace_fields
            or item.get("update") != update
            or not math.isclose(
                float(item.get("learning_rate", float("nan"))),
                learning_rate_v6(update),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not all(
                _finite_number(item.get(key))
                for key in trace_fields
                - {"update", "adapter_state_sha256"}
            )
            or float(item["preclip_gradient_l2"]) <= 0
            or not _is_sha256(item.get("adapter_state_sha256"))
        ):
            raise ValueError(f"V6.1 training trace changed at update {update}")
    if (
        training.get("initial_trace") != trace[:3]
        or training.get("milestone_trace")
        != [trace[index - 1] for index in (24, 48, 72, 96)]
        or training.get("final_trace") != trace[-3:]
        or training.get("final_adapter_state_sha256")
        != trace[-1]["adapter_state_sha256"]
        or not math.isclose(
            float(training["maximum_preclip_gradient_l2"]),
            max(float(item["preclip_gradient_l2"]) for item in trace),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("V6.1 training trace summaries changed")


def authenticate_result() -> dict[str, Any]:
    release = authenticate_training_release()
    result = _read_json(RESULT_REPORT)
    attempt = _read_json(TRAINING_ATTEMPT)
    audit = _read_json(FILE_AUDIT_REPORT)
    attempt_fields = {
        "schema_version",
        "artifact",
        "status",
        "training_release_sha256",
        "parent_smoke_sha256",
        "maximum_optimizer_updates",
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
        or attempt.get("parent_smoke_sha256") != release["parent_smoke_sha256"]
        or attempt.get("maximum_optimizer_updates") != _UPDATES
        or attempt.get("checkpoint_write_authorized_before_internal_gates") is not False
        or attempt.get("deferred_or_final_qa_access_authorized") is not False
        or attempt.get("oracle_access_authorized") is not False
    ):
        raise ValueError("V6.1 training attempt evidence changed")
    audit_fields = {
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
    loaded = audit.get("loaded_files")
    forbidden = audit.get("forbidden_accesses")
    audit_clean = isinstance(forbidden, list) and not forbidden
    if (
        set(audit) != audit_fields
        or audit.get("schema_version") != 1
        or audit.get("artifact") != f"{ARTIFACT}_file_audit"
        or not isinstance(loaded, list)
        or loaded != sorted(set(loaded))
        or not all(isinstance(path, str) and Path(path).is_absolute() for path in loaded)
        or audit.get("loaded_file_count") != len(loaded)
        or audit.get("loaded_file_inventory_sha256") != _canonical_hash(loaded)
        or not isinstance(audit.get("forbidden_roots"), list)
        or audit.get("forbidden_component_names") != ["oracle"]
        or audit.get("block_forbidden") is not True
        or not isinstance(forbidden, list)
        or any(path not in loaded for path in forbidden)
        or audit.get("passed") is not audit_clean
    ):
        raise ValueError("V6.1 file-audit evidence changed")
    if (
        result.get("schema_version") != 1
        or result.get("artifact") != f"{ARTIFACT}_terminal_result"
        or result.get("status")
        not in {
            "passed_checkpoint_published",
            "failed_no_checkpoint",
            "failed_terminal_attempt_consumed_no_checkpoint",
        }
        or result.get("training_attempt_sha256") != _sha256_file(TRAINING_ATTEMPT)
        or result.get("training_release_sha256") != release["sha256"]
        or result.get("parent_passing_smoke_sha256")
        != release["parent_smoke_sha256"]
        or result.get("file_audit_report") != FILE_AUDIT_REPORT
        or result.get("file_audit_sha256") != _sha256_file(FILE_AUDIT_REPORT)
    ):
        raise ValueError("V6.1 terminal evidence is internally inconsistent")
    passed = result.get("passed") is True
    checkpoint_exists = _resolve(OUTPUT_CHECKPOINT).is_dir()
    if passed != checkpoint_exists or result.get("checkpoint_published") is not passed:
        raise ValueError("V6.1 result/checkpoint state is inconsistent")
    if not passed:
        if checkpoint_exists or result.get("checkpoint") is not None:
            raise ValueError("V6.1 failed result retained a checkpoint")
        if result.get("status") == "failed_terminal_attempt_consumed_no_checkpoint":
            exceptional_fields = {
                "schema_version",
                "artifact",
                "status",
                "passed",
                "promotion_eligible",
                "checkpoint_published",
                "checkpoint",
                "training_release_sha256",
                "parent_passing_smoke_sha256",
                "training_attempt_sha256",
                "error_type",
                "error",
                "file_audit_report",
                "file_audit_sha256",
                "deferred_or_final_qa_accessed",
                "oracle_accessed",
            }
            if (
                set(result) != exceptional_fields
                or result.get("promotion_eligible") is not False
                or not isinstance(result.get("error_type"), str)
                or not isinstance(result.get("error"), str)
                or result.get("deferred_or_final_qa_accessed") is not bool(forbidden)
                or result.get("oracle_accessed")
                is not any(
                    "oracle" in {part.casefold() for part in Path(path).parts}
                    for path in forbidden
                )
            ):
                raise ValueError("V6.1 exceptional failure evidence changed")
        else:
            if not audit_clean:
                raise ValueError("V6.1 ordinary gated failure has a dirty audit")
            validation_rows = v1.load_validation_records()
            if _authenticate_selection_evidence(
                result.get("selection"), validation_rows
            ):
                raise ValueError("V6.1 failed result contains a passing selection")
            _authenticate_training_trace(result.get("training"))
        return {
            "passed": False,
            "status": result.get("status"),
            "result_sha256": _sha256_file(RESULT_REPORT),
            "file_audit_sha256": _sha256_file(FILE_AUDIT_REPORT),
            "checkpoint_exists": False,
            "checkpoint_published": False,
        }

    if not audit_clean or result.get("status") != "passed_checkpoint_published":
        raise ValueError("V6.1 passing result lacks a clean audit or pass status")
    validation_rows = v1.load_validation_records()
    if not _authenticate_selection_evidence(result.get("selection"), validation_rows):
        raise ValueError("V6.1 passing result failed recomputed internal gates")
    _authenticate_training_trace(result.get("training"))
    checkpoint = result.get("checkpoint")
    checkpoint_path = _resolve(OUTPUT_CHECKPOINT)
    weights = checkpoint_path / "adapter.safetensors"
    metadata_path = checkpoint_path / "runtime_metadata.json"
    if (
        not isinstance(checkpoint, Mapping)
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
        raise ValueError("V6.1 checkpoint publication record changed")
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
        raise ValueError("V6.1 checkpoint tensor inventory or state changed")
    metadata = _read_json(metadata_path)
    metadata_fields = {
        "schema_version",
        "artifact",
        "base_checkpoint_sha256",
        "base_runtime_config_effective_sha256",
        "model_id",
        "model_revision",
        "fixed_prefix_tokens",
        "scene_latents",
        "scene_hidden_dimension",
        "prefix_computed_before_question",
        "question_dependent_scene_retrieval",
        "environmental_text_inputs",
        "oracle_runtime_access",
        "adapter_type",
        "target_modules",
        "rank",
        "alpha",
        "dropout",
        "trainable_parameter_count",
        "adapter_state_sha256",
        "adapter_file_sha256",
        "selection_summary_sha256",
        "training_release_sha256",
        "parent_passing_smoke_sha256",
    }
    if (
        set(metadata) != metadata_fields
        or metadata.get("schema_version") != 1
        or metadata.get("artifact") != ARTIFACT
        or metadata.get("base_checkpoint_sha256") != _BASE_CHECKPOINT_FINGERPRINT
        or metadata.get("base_runtime_config_effective_sha256")
        != _BASE_RUNTIME_EFFECTIVE_SHA256
        or metadata.get("model_id") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("fixed_prefix_tokens") != 258
        or metadata.get("scene_latents") != 256
        or metadata.get("scene_hidden_dimension") != 1536
        or metadata.get("prefix_computed_before_question") is not True
        or metadata.get("question_dependent_scene_retrieval") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("oracle_runtime_access") is not False
        or metadata.get("adapter_type") != "fresh_v6_only_upper_decoder_lora"
        or metadata.get("target_modules") != list(TARGET_MODULES)
        or metadata.get("rank") != LORA_RANK
        or metadata.get("alpha") != LORA_ALPHA
        or metadata.get("dropout") != 0.0
        or metadata.get("trainable_parameter_count") != LORA_PARAMETER_COUNT
        or metadata.get("adapter_state_sha256") != state_sha
        or metadata.get("adapter_file_sha256") != _sha256_file(weights)
        or metadata.get("selection_summary_sha256")
        != _canonical_hash(result["selection"])
        or metadata.get("training_release_sha256") != release["sha256"]
        or metadata.get("parent_passing_smoke_sha256")
        != release["parent_smoke_sha256"]
    ):
        raise ValueError("V6.1 promoted checkpoint metadata changed")
    return {
        "passed": passed,
        "status": result.get("status"),
        "result_sha256": _sha256_file(RESULT_REPORT),
        "file_audit_sha256": _sha256_file(FILE_AUDIT_REPORT),
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_published": result.get("checkpoint_published"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("preflight", "release", "authenticate-release", "train", "authenticate"),
    )
    mode = parser.parse_args(argv).mode
    if mode == "release":
        path, digest = write_training_release()
        result: dict[str, Any] = {"passed": True, "path": str(path), "sha256": digest}
    else:
        result = {
            "preflight": structural_preflight,
            "authenticate-release": authenticate_training_release,
            "train": train_and_gate,
            "authenticate": authenticate_result,
        }[mode]()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

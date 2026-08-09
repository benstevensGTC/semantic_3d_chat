"""Report-only verifier for V20's separately executed first optimizer update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.evaluation.v19_optimizer_state import (
    V19AdamWStateViolation,
    canonical_v19_adamw_state,
    validate_v19_adamw_state_manifest,
)
from semantic_3d_chat.evaluation.v20_structural_preflight import (
    EXPECTED_SCENE_IDS,
    V20_PREFLIGHT_ROLE,
    V20StructuralPreflightViolation,
    canonical_sha256,
    evaluate_v20_structural_gate,
    validate_v20_config_contract,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.global_residual import global_scene_residual_settings
from semantic_3d_chat.scene_encoder.signed_x_dispatch import (
    construct_signed_x_scene_residual,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.scene_encoder.signed_x_local_field import (
    SignedXLocalFieldSceneResidual,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)
from semantic_3d_chat.training.train_adapter import file_sha256

UPDATE1_VERIFIER_TYPE = "v20_exact_update1_match_verifier"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHORT_SHA = re.compile(r"[0-9a-f]{12}")
_SIGNED_PREFIX = "signed_x_scene_residual."
_GLOBAL_PREFIX = "global_scene_residual."
_SCENE_PREFIXES = ("scene_model.", "composer.", "grounding.")
_LORA_BANK_PREFIX = "lora_banks."
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_EXPECTED_PAIRS = {"pair_000001", "pair_000003"}
_EXPECTED_PAIR_SCENES = {
    "pair_000001": ("scene_000003", "scene_000004"),
    "pair_000003": ("scene_000007", "scene_000008"),
}
_FORBIDDEN_INPUT_COMPONENTS = frozenset({"oracle", "rendered", "maps", "scene_tokens", "runtime"})
_IMPLEMENTATION_SOURCES = {
    "implementation_source": "src/semantic_3d_chat/evaluation/v20_structural_preflight.py",
    "signed_x_implementation_source": (
        "src/semantic_3d_chat/scene_encoder/signed_x_local_field.py"
    ),
    "signed_x_dispatch_implementation_source": (
        "src/semantic_3d_chat/scene_encoder/signed_x_dispatch.py"
    ),
}


class V20Update1Violation(ValueError):
    """A mismatch that denies V20 stage-two resume."""


def _fail(message: str) -> None:
    raise V20Update1Violation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _equal(observed: Any, expected: Any, field: str) -> None:
    if observed != expected:
        _fail(f"{field} mismatch: expected={expected!r} observed={observed!r}")


def _canonical_equal(observed: Any, expected: Any, field: str) -> None:
    try:
        observed_hash = canonical_sha256(observed)
        expected_hash = canonical_sha256(expected)
    except (TypeError, ValueError) as error:
        _fail(f"{field} is not finite canonical JSON: {error}")
    if observed_hash != expected_hash:
        _fail(f"{field} canonical mismatch")


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        _fail(f"{field} keys mismatch: expected={sorted(expected)!r} observed={sorted(observed)!r}")


def _exact_bool(observed: Any, expected: bool, field: str) -> bool:
    if type(observed) is not bool or observed is not expected:
        _fail(f"{field} mismatch: expected={expected!r} observed={observed!r}")
    return observed


def _exact_string(observed: Any, expected: str, field: str) -> str:
    if type(observed) is not str or observed != expected:
        _fail(f"{field} mismatch: expected={expected!r} observed={observed!r}")
    return observed


def _exact_int(observed: Any, expected: int, field: str) -> None:
    if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected:
        _fail(f"{field} mismatch: expected={expected} observed={observed!r}")


def _bounded_int(
    observed: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(observed, bool) or not isinstance(observed, int):
        _fail(f"{field} must be an integer")
    if minimum is not None and observed < minimum:
        _fail(f"{field} must be at least {minimum}")
    if maximum is not None and observed > maximum:
        _fail(f"{field} must be at most {maximum}")
    return observed


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _bounded_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float:
        _fail(f"{field} must be a JSON floating-point number")
    result = _finite(value, field)
    if minimum is not None and result < minimum:
        _fail(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        _fail(f"{field} must be at most {maximum}")
    return result


def _fraction(value: Any, field: str) -> float:
    return _bounded_number(value, field, minimum=0.0, maximum=1.0)


def _exact_int_list(value: Any, expected: Sequence[int], field: str) -> list[int]:
    observed = list(_sequence(value, field))
    if len(observed) != len(expected):
        _fail(f"{field} length mismatch")
    for index, (item, wanted) in enumerate(zip(observed, expected, strict=True)):
        _exact_int(item, wanted, f"{field}[{index}]")
    return observed


def _finite_tree(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{field}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(f"{field} contains NaN or infinity")


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _absolute_without_symlink_resolution(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return Path(os.path.abspath(os.fspath(path)))


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _reject_forbidden_input_path(path: Path) -> None:
    lexical = _absolute_without_symlink_resolution(path)
    candidates = {part.casefold() for part in lexical.parts}
    try:
        candidates.update(part.casefold() for part in lexical.resolve().parts)
    except (OSError, RuntimeError):
        pass
    forbidden = sorted(candidates & _FORBIDDEN_INPUT_COMPONENTS)
    if forbidden:
        _fail(f"Verifier refuses runtime/oracle artifact path components: {forbidden}")


def _safe_existing_input_path(
    value: str | Path,
    field: str,
    *,
    kind: str = "file",
) -> Path:
    """Resolve an input only after rejecting symlinks and forbidden components."""

    if kind not in {"file", "directory"}:
        raise ValueError(f"Unsupported safe-input kind: {kind}")
    lexical = _absolute_without_symlink_resolution(value)
    _reject_forbidden_input_path(lexical)
    current = Path(lexical.anchor)
    try:
        for component in lexical.parts[1:]:
            current /= component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                _fail(f"{field} may not traverse a symlink: {current}")
    except FileNotFoundError:
        _fail(f"{field} does not exist: {current}")
    except OSError as error:
        _fail(f"Cannot inspect {field} at {current}: {error}")
    resolved = lexical.resolve(strict=True)
    _reject_forbidden_input_path(resolved)
    if kind == "file" and not resolved.is_file():
        _fail(f"{field} must be a regular file: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        _fail(f"{field} must be a directory: {resolved}")
    return resolved


def _read_json(path: Path, field: str) -> dict[str, Any]:
    path = _safe_existing_input_path(path, field)
    try:
        return dict(_mapping(json.loads(path.read_text(encoding="utf-8")), field))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"Cannot read {field} JSON at {path}: {error}")


def _clean_provenance(value: Any, field: str) -> dict[str, Any]:
    provenance = dict(_mapping(value, field))
    try:
        require_clean_committed_source(provenance)
    except RuntimeError as error:
        _fail(f"{field} is not clean committed source provenance: {error}")
    _equal(provenance.get("tracked_diff_sha256"), _EMPTY_SHA256, f"{field}.tracked_diff")
    return provenance


def _validate_implementation_sources(preflight: Mapping[str, Any]) -> None:
    for path_field, expected_relative in _IMPLEMENTATION_SOURCES.items():
        hash_field = f"{path_field}_sha256"
        observed_relative = _exact_string(
            preflight.get(path_field), expected_relative, f"preflight {path_field}"
        )
        expected_path = _safe_existing_input_path(
            PROJECT_ROOT / expected_relative,
            f"canonical {path_field}",
        )
        observed_path = _safe_existing_input_path(
            observed_relative,
            f"preflight {path_field}",
        )
        _equal(observed_path, expected_path, f"preflight {path_field} canonical path")
        expected_hash = _sha256(preflight.get(hash_field), f"preflight {hash_field}")
        _equal(file_sha256(expected_path), expected_hash, f"preflight {path_field} hash")


def _validate_zero_equivalence(value: Any) -> dict[str, Any]:
    equivalence = dict(_mapping(value, "preflight zero equivalence"))
    expected_root = {
        "verified",
        "base",
        "question_dependent_scene_processing",
        "all_scene_slots_accounted",
        "scene_count",
        "scene_prefixes",
    }
    _equal(set(equivalence), expected_root, "preflight zero equivalence root keys")
    for key, expected in {
        "verified": True,
        "base": "loaded_frozen_global_scene_residual",
        "question_dependent_scene_processing": False,
        "all_scene_slots_accounted": True,
        "scene_count": 4,
    }.items():
        _equal(equivalence.get(key), expected, f"preflight zero equivalence {key}")
    prefixes = _mapping(equivalence.get("scene_prefixes"), "zero equivalence scenes")
    _equal(set(prefixes), set(EXPECTED_SCENE_IDS), "zero equivalence scene set")
    compact: dict[str, Any] = {}
    expected_row = {
        "core_scene_token_sha256",
        "v18_base_scene_token_sha256",
        "v18_base_prefix_sha256",
        "signed_x_adapted_prefix_sha256",
        "scene_tokens_exactly_equal",
        "prefixes_exactly_equal",
        "prefix_hashes_equal",
    }
    for scene_id in EXPECTED_SCENE_IDS:
        row = _mapping(prefixes[scene_id], f"zero equivalence {scene_id}")
        _equal(set(row), expected_row, f"zero equivalence {scene_id} keys")
        _sha256(row.get("core_scene_token_sha256"), f"{scene_id} core tokens")
        _sha256(row.get("v18_base_scene_token_sha256"), f"{scene_id} V18 tokens")
        base = _sha256(row.get("v18_base_prefix_sha256"), f"{scene_id} base prefix")
        adapted = _sha256(row.get("signed_x_adapted_prefix_sha256"), f"{scene_id} adapted prefix")
        _equal(adapted, base, f"zero equivalence {scene_id} identity")
        for key in ("scene_tokens_exactly_equal", "prefixes_exactly_equal", "prefix_hashes_equal"):
            _equal(row.get(key), True, f"zero equivalence {scene_id} {key}")
        compact[scene_id] = {
            "v18_base_prefix_sha256": base,
            "signed_x_adapted_prefix_sha256": adapted,
        }
    return {
        "verified": True,
        "base": "loaded_frozen_global_scene_residual",
        "question_dependent_scene_processing": False,
        "all_scene_slots_accounted": True,
        "scene_count": 4,
        "scene_prefixes": compact,
    }


def _validate_structural_row(value: Any) -> dict[str, Any]:
    row = dict(_mapping(value, "local structure"))
    _exact_keys(
        row,
        {
            "architecture_version",
            "architecture_marker",
            "scene_dim",
            "latent_count",
            "content_dim",
            "parameter_count",
            "accounted_slot_count",
            "all_slots_accounted",
            "signed_x_anchor_mean",
            "signed_x_anchor_rms",
            "spatial_centering",
            "trainable_surface",
            "spatial_statistic",
            "spatial_reduction",
        },
        "local structure",
    )
    _exact_string(row["architecture_version"], "signed_x_local_field_v2", "architecture")
    for key, expected in {
        "architecture_marker": 2,
        "scene_dim": 1536,
        "latent_count": 256,
        "content_dim": 128,
        "parameter_count": 196_608,
        "accounted_slot_count": 256,
    }.items():
        _exact_int(row[key], expected, f"local structure {key}")
    _exact_bool(row["all_slots_accounted"], True, "local structure all slots")
    _bounded_number(
        row["signed_x_anchor_mean"],
        "local structure anchor mean",
        minimum=-1.0e-6,
        maximum=1.0e-6,
    )
    _bounded_number(
        row["signed_x_anchor_rms"],
        "local structure anchor RMS",
        minimum=1.0 - 1.0e-6,
        maximum=1.0 + 1.0e-6,
    )
    for key, expected in {
        "spatial_centering": "all_slots_fp32",
        "spatial_statistic": "centered_local_content_times_unit_rms_signed_x",
        "spatial_reduction": "none",
        "trainable_surface": "bias_free_output_projection_only",
    }.items():
        _exact_string(row[key], expected, f"local structure {key}")
    return row


def _validate_local_dependence_row(value: Any) -> dict[str, Any]:
    row = dict(_mapping(value, "local dependence"))
    _exact_keys(
        row,
        {
            "schema_version",
            "probe_shape",
            "hidden_shape",
            "probe_count",
            "paired_centered_perturbations",
            "maximum_probe_spatial_mean_absolute",
            "minimum_changed_slots_per_probe",
            "maximum_changed_slots_per_probe",
            "changed_slot_union_count",
            "all_input_slots_exercised",
            "unperturbed_output_slots_exactly_unchanged",
            "exact_two_slot_local_support",
            "no_global_moment_broadcast",
            "hidden_sha256",
        },
        "local dependence",
    )
    _exact_int(row["schema_version"], 1, "local dependence schema")
    _exact_int_list(row["probe_shape"], [128, 256, 128], "local dependence probe shape")
    _exact_int_list(row["hidden_shape"], [128, 256, 128], "local dependence hidden shape")
    _exact_int(row["probe_count"], 128, "local dependence probe count")
    _bounded_number(
        row["maximum_probe_spatial_mean_absolute"],
        "local dependence maximum probe spatial mean",
        minimum=0.0,
        maximum=1.0e-7,
    )
    for key, expected in {
        "minimum_changed_slots_per_probe": 2,
        "maximum_changed_slots_per_probe": 2,
        "changed_slot_union_count": 256,
    }.items():
        _exact_int(row[key], expected, f"local dependence {key}")
    for key in (
        "paired_centered_perturbations",
        "all_input_slots_exercised",
        "unperturbed_output_slots_exactly_unchanged",
        "exact_two_slot_local_support",
        "no_global_moment_broadcast",
    ):
        _exact_bool(row[key], True, f"local dependence {key}")
    _sha256(row["hidden_sha256"], "local dependence hidden hash")
    return row


def _validate_spatial_rank_row(value: Any, scene_id: str, minimum_rank: int) -> dict[str, Any]:
    field = f"local rank {scene_id}"
    row = dict(_mapping(value, field))
    _exact_keys(
        row,
        {
            "schema_version",
            "shape",
            "relative_tolerance",
            "minimum_spatial_rank",
            "batches",
        },
        field,
    )
    _exact_int(row["schema_version"], 1, f"{field} schema")
    _exact_int_list(row["shape"], [1, 256, 128], f"{field} shape")
    tolerance = _bounded_number(
        row["relative_tolerance"], f"{field} relative tolerance", minimum=0.0
    )
    if tolerance != 1.0e-5:
        _fail(f"{field} relative tolerance must be exactly 1e-5")
    observed_minimum = _bounded_int(
        row["minimum_spatial_rank"], field, minimum=minimum_rank, maximum=128
    )
    batches = list(_sequence(row["batches"], f"{field} batches"))
    if len(batches) != 1:
        _fail(f"{field} must contain exactly one batch audit")
    batch = dict(_mapping(batches[0], f"{field} batch zero"))
    _exact_keys(
        batch,
        {
            "batch_index",
            "spatial_rank",
            "stable_rank",
            "maximum_singular_value",
            "rank_threshold",
            "top_singular_values",
        },
        f"{field} batch zero",
    )
    _exact_int(batch["batch_index"], 0, f"{field} batch index")
    spatial_rank = _bounded_int(
        batch["spatial_rank"], f"{field} spatial rank", minimum=minimum_rank, maximum=128
    )
    _exact_int(observed_minimum, spatial_rank, f"{field} minimum/batch rank")
    _bounded_number(batch["stable_rank"], f"{field} stable rank", minimum=1.0, maximum=128.0)
    maximum = _bounded_number(
        batch["maximum_singular_value"], f"{field} maximum singular value", minimum=0.0
    )
    if maximum <= 0.0:
        _fail(f"{field} maximum singular value must be positive")
    threshold = _bounded_number(batch["rank_threshold"], f"{field} rank threshold", minimum=0.0)
    if not math.isclose(threshold, maximum * tolerance, rel_tol=1.0e-9, abs_tol=1.0e-12):
        _fail(f"{field} rank threshold is inconsistent with its tolerance")
    singular_values = list(_sequence(batch["top_singular_values"], f"{field} top singular values"))
    if len(singular_values) != 8:
        _fail(f"{field} must report exactly eight top singular values")
    validated = [
        _bounded_number(item, f"{field} singular value {index}", minimum=0.0)
        for index, item in enumerate(singular_values)
    ]
    if validated != sorted(validated, reverse=True):
        _fail(f"{field} singular values must be non-increasing")
    if not math.isclose(validated[0], maximum, rel_tol=1.0e-6, abs_tol=1.0e-9):
        _fail(f"{field} leading singular value does not match the reported maximum")
    return row


def _validate_centered_content_row(
    value: Any,
    scene_id: str,
    rank: Mapping[str, Any],
) -> dict[str, Any]:
    field = f"centered content {scene_id}"
    row = dict(_mapping(value, field))
    _exact_keys(
        row,
        {
            "shape",
            "finite",
            "across_slot_mean_absolute_maximum",
            "local_hidden_rms",
            "local_hidden_sha256",
            "local_hidden_spatial_rank",
            "sha256",
        },
        field,
    )
    _exact_int_list(row["shape"], [1, 256, 128], f"{field} shape")
    _exact_bool(row["finite"], True, f"{field} finite")
    _bounded_number(
        row["across_slot_mean_absolute_maximum"],
        f"{field} spatial mean",
        minimum=0.0,
    )
    hidden_rms = _bounded_number(row["local_hidden_rms"], f"{field} hidden RMS", minimum=0.0)
    if hidden_rms <= 0.0:
        _fail(f"{field} hidden RMS must be positive")
    _sha256(row["local_hidden_sha256"], f"{field} local hidden hash")
    _sha256(row["sha256"], f"{field} hash")
    _canonical_equal(row["local_hidden_spatial_rank"], rank, f"{field} rank alias")
    return row


_SCENE_DELTA_KEYS = {
    "shape",
    "core_rms",
    "delta_rms",
    "delta_to_core_rms_ratio",
    "total_energy",
    "across_slot_mean_energy",
    "slot_varying_energy",
    "across_slot_mean_energy_fraction",
    "slot_varying_energy_fraction",
    "slot_mean_absolute_maximum",
    "delta_absolute_maximum",
    "energy_closure_absolute_error",
    "positive_finite_total_energy",
    "positive_finite_core_rms",
    "delta_sha256",
    "dtype",
}


def _validate_scene_delta_row(value: Any, scene_id: str, *, effective: bool) -> dict[str, Any]:
    phase = "effective" if effective else "raw"
    field = f"{phase} scene {scene_id}"
    row = dict(_mapping(value, field))
    _exact_keys(row, _SCENE_DELTA_KEYS, field)
    _exact_int_list(row["shape"], [1, 256, 1536], f"{field} shape")
    core_rms = _bounded_number(row["core_rms"], f"{field} core RMS", minimum=0.0)
    delta_rms = _bounded_number(row["delta_rms"], f"{field} delta RMS", minimum=0.0)
    if core_rms <= 0.0 or delta_rms <= 0.0:
        _fail(f"{field} core and delta RMS must be positive")
    ratio = _bounded_number(
        row["delta_to_core_rms_ratio"], f"{field} delta/core ratio", minimum=0.0
    )
    if not math.isclose(ratio, delta_rms / core_rms, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"{field} delta/core ratio is inconsistent")
    total = _bounded_number(row["total_energy"], f"{field} total energy", minimum=0.0)
    mean = _bounded_number(row["across_slot_mean_energy"], f"{field} mean energy", minimum=0.0)
    varying = _bounded_number(row["slot_varying_energy"], f"{field} varying energy", minimum=0.0)
    if total <= 0.0:
        _fail(f"{field} total energy must be positive")
    mean_fraction = _fraction(
        row["across_slot_mean_energy_fraction"], f"{field} mean-energy fraction"
    )
    varying_fraction = _fraction(
        row["slot_varying_energy_fraction"], f"{field} varying-energy fraction"
    )
    if not math.isclose(mean + varying, total, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"{field} energy decomposition is inconsistent")
    if not math.isclose(mean_fraction, mean / total, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"{field} mean-energy fraction is inconsistent")
    if not math.isclose(varying_fraction, varying / total, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"{field} varying-energy fraction is inconsistent")
    _bounded_number(row["slot_mean_absolute_maximum"], f"{field} slot-mean maximum", minimum=0.0)
    _bounded_number(row["delta_absolute_maximum"], f"{field} delta maximum", minimum=0.0)
    _bounded_number(row["energy_closure_absolute_error"], f"{field} closure error", minimum=0.0)
    _exact_bool(row["positive_finite_total_energy"], True, f"{field} positive energy")
    _exact_bool(row["positive_finite_core_rms"], True, f"{field} positive core RMS")
    _sha256(row["delta_sha256"], f"{field} hash")
    expected_dtype = "bfloat16_round_trip_float32_delta" if effective else "float32"
    _exact_string(row["dtype"], expected_dtype, f"{field} dtype")
    return row


def _validate_bf16_cast_row(value: Any, scene_id: str) -> dict[str, Any]:
    field = f"BF16 cast {scene_id}"
    row = dict(_mapping(value, field))
    _exact_keys(
        row,
        {
            "schema_version",
            "algorithm",
            "base_source_dtype",
            "model_dtype",
            "comparison_dtype",
            "element_count",
            "changed_element_count",
            "changed_element_fraction",
            "raw_delta_rms",
            "effective_delta_rms",
            "effective_to_raw_rms_ratio",
            "quantization_error_rms",
            "quantization_error_to_raw_rms_ratio",
            "raw_effective_cosine",
            "raw_delta_sha256",
            "effective_delta_sha256",
        },
        field,
    )
    _exact_int(row["schema_version"], 1, f"{field} schema")
    _exact_string(
        row["algorithm"], "bfloat16_cast_of_fp32_base_plus_fp32_delta", f"{field} algorithm"
    )
    _exact_string(row["base_source_dtype"], "float32", f"{field} base dtype")
    _exact_string(row["model_dtype"], "bfloat16", f"{field} model dtype")
    _exact_string(row["comparison_dtype"], "float64", f"{field} comparison dtype")
    count = 256 * 1536
    _exact_int(row["element_count"], count, f"{field} element count")
    changed = _bounded_int(
        row["changed_element_count"], f"{field} changed count", minimum=1, maximum=count
    )
    changed_fraction = _fraction(row["changed_element_fraction"], f"{field} changed fraction")
    if not math.isclose(changed_fraction, changed / count, rel_tol=1.0e-12, abs_tol=1.0e-15):
        _fail(f"{field} changed fraction is inconsistent")
    raw_rms = _bounded_number(row["raw_delta_rms"], f"{field} raw RMS", minimum=0.0)
    effective_rms = _bounded_number(
        row["effective_delta_rms"], f"{field} effective RMS", minimum=0.0
    )
    if raw_rms <= 0.0 or effective_rms <= 0.0:
        _fail(f"{field} raw and effective RMS must be positive")
    effective_ratio = _bounded_number(
        row["effective_to_raw_rms_ratio"], f"{field} effective/raw ratio", minimum=0.0
    )
    if not math.isclose(effective_ratio, effective_rms / raw_rms, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"{field} effective/raw ratio is inconsistent")
    error_rms = _bounded_number(
        row["quantization_error_rms"], f"{field} quantization error", minimum=0.0
    )
    error_ratio = _bounded_number(
        row["quantization_error_to_raw_rms_ratio"],
        f"{field} quantization-error/raw ratio",
        minimum=0.0,
    )
    if not math.isclose(error_ratio, error_rms / raw_rms, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"{field} quantization-error/raw ratio is inconsistent")
    _bounded_number(row["raw_effective_cosine"], f"{field} cosine", minimum=-1.0, maximum=1.0)
    _sha256(row["raw_delta_sha256"], f"{field} raw hash")
    _sha256(row["effective_delta_sha256"], f"{field} effective hash")
    return row


_PAIR_DELTA_KEYS = {
    "first_scene_id",
    "second_scene_id",
    "core_pair_difference_rms",
    "residual_pair_difference_rms",
    "residual_to_core_pair_difference_ratio",
    "residual_core_difference_cosine",
    "positive_finite_pair_delta",
    "positive_finite_core_difference",
}


def _validate_pair_delta_row(value: Any, pair_id: str, *, effective: bool) -> dict[str, Any]:
    phase = "effective" if effective else "raw"
    field = f"{phase} pair {pair_id}"
    row = dict(_mapping(value, field))
    _exact_keys(row, _PAIR_DELTA_KEYS, field)
    first = row["first_scene_id"]
    second = row["second_scene_id"]
    if type(first) is not str or type(second) is not str:
        _fail(f"{field} scene IDs must be strings")
    _equal((first, second), _EXPECTED_PAIR_SCENES[pair_id], f"{field} exact pair membership")
    core_rms = _bounded_number(
        row["core_pair_difference_rms"], f"{field} core pair RMS", minimum=0.0
    )
    residual_rms = _bounded_number(
        row["residual_pair_difference_rms"], f"{field} residual pair RMS", minimum=0.0
    )
    if core_rms <= 0.0:
        _fail(f"{field} core pair RMS must be positive")
    ratio = _bounded_number(
        row["residual_to_core_pair_difference_ratio"],
        f"{field} residual/core ratio",
        minimum=0.0,
    )
    if not math.isclose(ratio, residual_rms / core_rms, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"{field} residual/core ratio is inconsistent")
    _bounded_number(
        row["residual_core_difference_cosine"],
        f"{field} residual/core cosine",
        minimum=-1.0,
        maximum=1.0,
    )
    pair_delta_positive = row["positive_finite_pair_delta"]
    if type(pair_delta_positive) is not bool or pair_delta_positive != (residual_rms > 0.0):
        _fail(f"{field} positive pair-delta flag is inconsistent")
    _exact_bool(row["positive_finite_core_difference"], True, f"{field} positive core difference")
    return row


def _validate_requirements(value: Any) -> dict[str, Any]:
    row = dict(_mapping(value, "structural requirements"))
    _exact_keys(
        row,
        {
            "maximum_per_scene_raw_and_effective_delta_to_core_rms_ratio",
            "minimum_mirror_effective_residual_to_core_rms_ratio",
            "minimum_mirror_to_color_normalized_effective_selectivity",
            "minimum_local_hidden_spatial_rank",
        },
        "structural requirements",
    )
    for key in (
        "maximum_per_scene_raw_and_effective_delta_to_core_rms_ratio",
        "minimum_mirror_effective_residual_to_core_rms_ratio",
        "minimum_mirror_to_color_normalized_effective_selectivity",
    ):
        _bounded_number(row[key], f"structural requirement {key}", minimum=0.0)
    _bounded_int(
        row["minimum_local_hidden_spatial_rank"],
        "minimum local hidden rank",
        minimum=2,
        maximum=128,
    )
    return row


def _validate_selectivity_row(value: Any, field: str) -> dict[str, Any]:
    row = dict(_mapping(value, field))
    _exact_keys(
        row,
        {
            "schema_version",
            "color_residual_to_core_rms_ratio",
            "mirror_residual_to_core_rms_ratio",
            "color_ratio_exact_zero",
            "mirror_to_color_normalized_selectivity",
        },
        field,
    )
    _exact_int(row["schema_version"], 1, f"{field} schema")
    color = _bounded_number(
        row["color_residual_to_core_rms_ratio"], f"{field} color ratio", minimum=0.0
    )
    mirror = _bounded_number(
        row["mirror_residual_to_core_rms_ratio"], f"{field} mirror ratio", minimum=0.0
    )
    exact_zero = row["color_ratio_exact_zero"]
    if type(exact_zero) is not bool or exact_zero != (color == 0.0):
        _fail(f"{field} color zero flag is inconsistent")
    selectivity = row["mirror_to_color_normalized_selectivity"]
    if color == 0.0:
        if selectivity is not None:
            _fail(f"{field} selectivity must be null when color ratio is zero")
    else:
        observed = _bounded_number(selectivity, f"{field} selectivity", minimum=0.0)
        if not math.isclose(observed, mirror / color, rel_tol=1.0e-6, abs_tol=1.0e-12):
            _fail(f"{field} selectivity ratio is inconsistent")
    return row


def _validate_gate_row(
    value: Any,
    *,
    requirements: Mapping[str, Any],
    raw_scene: Mapping[str, Mapping[str, Any]],
    effective_scene: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gate = dict(_mapping(value, "structural gate"))
    _exact_keys(
        gate,
        {
            "schema_version",
            "requirements",
            "all_slots_accounted",
            "local_field_structure_verified",
            "scene_checks",
            "selectivity_checks",
            "raw_pair_selectivity",
            "bf16_effective_pair_selectivity",
            "maximum_observed_raw_delta_to_core_rms_ratio",
            "maximum_observed_bf16_effective_delta_to_core_rms_ratio",
            "passed",
        },
        "structural gate",
    )
    _exact_int(gate["schema_version"], 1, "structural gate schema")
    observed_requirements = _validate_requirements(gate["requirements"])
    _equal(observed_requirements, dict(requirements), "structural gate requirements")
    _exact_bool(gate["all_slots_accounted"], True, "structural gate all slots")
    _exact_bool(gate["local_field_structure_verified"], True, "structural gate local field")
    scene_checks = dict(_mapping(gate["scene_checks"], "structural gate scene checks"))
    _exact_keys(scene_checks, set(EXPECTED_SCENE_IDS), "structural gate scene checks")
    expected_scene_check_keys = {
        "raw_positive_finite_total_energy",
        "raw_fp32_centered",
        "raw_slot_varying",
        "raw_delta_ratio_bounded",
        "effective_finite",
        "effective_delta_ratio_bounded",
        "bf16_changed_nonzero",
        "bf16_quantization_error_finite",
        "local_hidden_spatial_rank",
    }
    for scene_id, value_row in scene_checks.items():
        checks = dict(_mapping(value_row, f"scene checks {scene_id}"))
        _exact_keys(checks, expected_scene_check_keys, f"scene checks {scene_id}")
        for key, observed in checks.items():
            _exact_bool(observed, True, f"scene check {scene_id}.{key}")
    selectivity_checks = dict(
        _mapping(gate["selectivity_checks"], "structural gate selectivity checks")
    )
    _exact_keys(
        selectivity_checks,
        {
            "raw_mirror_residual_positive_finite",
            "effective_mirror_residual_at_least_minimum",
            "effective_normalized_selectivity_at_least_minimum",
        },
        "structural gate selectivity checks",
    )
    for key, observed in selectivity_checks.items():
        _exact_bool(observed, True, f"selectivity check {key}")
    _validate_selectivity_row(gate["raw_pair_selectivity"], "raw pair selectivity")
    _validate_selectivity_row(
        gate["bf16_effective_pair_selectivity"], "BF16 effective pair selectivity"
    )
    raw_max = _bounded_number(
        gate["maximum_observed_raw_delta_to_core_rms_ratio"],
        "maximum raw scene ratio",
        minimum=0.0,
    )
    effective_max = _bounded_number(
        gate["maximum_observed_bf16_effective_delta_to_core_rms_ratio"],
        "maximum effective scene ratio",
        minimum=0.0,
    )
    _equal(
        raw_max,
        max(float(row["delta_to_core_rms_ratio"]) for row in raw_scene.values()),
        "maximum raw scene ratio",
    )
    _equal(
        effective_max,
        max(float(row["delta_to_core_rms_ratio"]) for row in effective_scene.values()),
        "maximum effective scene ratio",
    )
    _exact_bool(gate["passed"], True, "structural gate passed")
    return gate


def _validate_rich_evidence(
    preflight: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    requirements = _validate_requirements(contract.get("structural_preflight_requires"))
    structural = _validate_structural_row(preflight.get("local_field_structural_state"))
    _canonical_equal(preflight.get("signed_x_structural_state"), structural, "structural aliases")
    dependence = _validate_local_dependence_row(preflight.get("local_dependence"))

    ranks = dict(_mapping(preflight.get("local_hidden_spatial_rank"), "local ranks"))
    _equal(set(ranks), set(EXPECTED_SCENE_IDS), "local rank scene set")
    minimum_rank = requirements["minimum_local_hidden_spatial_rank"]
    ranks = {
        scene_id: _validate_spatial_rank_row(ranks[scene_id], scene_id, minimum_rank)
        for scene_id in EXPECTED_SCENE_IDS
    }
    centered = dict(_mapping(preflight.get("centered_content"), "centered content"))
    _equal(set(centered), set(EXPECTED_SCENE_IDS), "centered-content scene set")
    centered = {
        scene_id: _validate_centered_content_row(centered[scene_id], scene_id, ranks[scene_id])
        for scene_id in EXPECTED_SCENE_IDS
    }

    raw_scene = dict(_mapping(preflight.get("raw_fp32_centered_scene_delta"), "raw scene delta"))
    effective_scene = dict(
        _mapping(preflight.get("bf16_effective_scene_delta"), "BF16 scene delta")
    )
    _canonical_equal(
        preflight.get("effective_cast_scene_delta"), effective_scene, "BF16 scene alias"
    )
    casts = dict(_mapping(preflight.get("bf16_cast_audit"), "BF16 cast audit"))
    expected_scenes = set(EXPECTED_SCENE_IDS)
    for field, value in (
        ("raw scene", raw_scene),
        ("effective scene", effective_scene),
        ("cast", casts),
    ):
        _equal(set(value), expected_scenes, f"{field} scene set")
    raw_scene = {
        scene_id: _validate_scene_delta_row(raw_scene[scene_id], scene_id, effective=False)
        for scene_id in EXPECTED_SCENE_IDS
    }
    effective_scene = {
        scene_id: _validate_scene_delta_row(effective_scene[scene_id], scene_id, effective=True)
        for scene_id in EXPECTED_SCENE_IDS
    }
    casts = {
        scene_id: _validate_bf16_cast_row(casts[scene_id], scene_id)
        for scene_id in EXPECTED_SCENE_IDS
    }
    for scene_id in EXPECTED_SCENE_IDS:
        raw = raw_scene[scene_id]
        effective = effective_scene[scene_id]
        cast = casts[scene_id]
        _equal(raw.get("delta_sha256"), cast.get("raw_delta_sha256"), f"raw hash {scene_id}")
        _equal(
            effective.get("delta_sha256"),
            cast.get("effective_delta_sha256"),
            f"effective hash {scene_id}",
        )

    raw_pair = dict(_mapping(preflight.get("raw_fp32_centered_pair_delta"), "raw pair delta"))
    effective_pair = dict(_mapping(preflight.get("bf16_effective_pair_delta"), "BF16 pair delta"))
    _canonical_equal(preflight.get("effective_cast_pair_delta"), effective_pair, "BF16 pair alias")
    _equal(set(raw_pair), _EXPECTED_PAIRS, "raw pair set")
    _equal(set(effective_pair), _EXPECTED_PAIRS, "effective pair set")
    raw_pair = {
        pair_id: _validate_pair_delta_row(raw_pair[pair_id], pair_id, effective=False)
        for pair_id in sorted(_EXPECTED_PAIRS)
    }
    effective_pair = {
        pair_id: _validate_pair_delta_row(effective_pair[pair_id], pair_id, effective=True)
        for pair_id in sorted(_EXPECTED_PAIRS)
    }
    raw_pair_scenes = {
        scene_id
        for row in raw_pair.values()
        for scene_id in (row["first_scene_id"], row["second_scene_id"])
    }
    effective_pair_scenes = {
        scene_id
        for row in effective_pair.values()
        for scene_id in (row["first_scene_id"], row["second_scene_id"])
    }
    _equal(raw_pair_scenes, set(EXPECTED_SCENE_IDS), "raw pair scene coverage")
    _equal(effective_pair_scenes, set(EXPECTED_SCENE_IDS), "effective pair scene coverage")

    gate = _validate_gate_row(
        preflight.get("structural_gate"),
        requirements=requirements,
        raw_scene=raw_scene,
        effective_scene=effective_scene,
    )
    try:
        recomputed_gate = evaluate_v20_structural_gate(
            raw_scene,
            effective_scene,
            raw_pair_metrics=raw_pair,
            effective_pair_metrics=effective_pair,
            bf16_audits=casts,
            structural_state=structural,
            local_dependence=dependence,
            local_hidden_ranks=ranks,
            requirements=requirements,
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail(f"Cannot recompute the V20 structural gate: {error}")
    _canonical_equal(gate, recomputed_gate, "structural gate recomputation")
    _finite_tree(
        {
            "structural": structural,
            "dependence": dependence,
            "ranks": ranks,
            "centered": centered,
            "raw_scene": raw_scene,
            "effective_scene": effective_scene,
            "casts": casts,
            "raw_pair": raw_pair,
            "effective_pair": effective_pair,
            "gate": gate,
        },
        "rich preflight evidence",
    )
    reduction = {
        "schema_version": 1,
        "verified": True,
        "bf16_algorithm": "bfloat16_cast_of_fp32_base_plus_fp32_delta",
        "scene_ids": list(EXPECTED_SCENE_IDS),
        "pair_ids": sorted(_EXPECTED_PAIRS),
        "local_field_structural_state_sha256": canonical_sha256(structural),
        "local_dependence_sha256": canonical_sha256(dependence),
        "local_hidden_spatial_rank_sha256": canonical_sha256(ranks),
        "centered_content_sha256": canonical_sha256(centered),
        "raw_fp32_centered_scene_delta_sha256": canonical_sha256(raw_scene),
        "bf16_effective_scene_delta_sha256": canonical_sha256(effective_scene),
        "bf16_cast_audit_sha256": canonical_sha256(casts),
        "raw_fp32_centered_pair_delta_sha256": canonical_sha256(raw_pair),
        "bf16_effective_pair_delta_sha256": canonical_sha256(effective_pair),
        "structural_gate_sha256": canonical_sha256(gate),
    }
    return {**reduction, "canonical_sha256": canonical_sha256(reduction)}


def _validate_preflight(
    config: dict[str, Any],
    preflight: Mapping[str, Any],
    current_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        contract = validate_v20_config_contract(config)
    except (TypeError, ValueError, RuntimeError, V20StructuralPreflightViolation) as error:
        _fail(f"V20 config contract is invalid: {error}")
    for key, expected in {
        "schema_version": 1,
        "audit_type": V20_PREFLIGHT_ROLE,
        "authorized": True,
        "structural_authorization": True,
        "runtime_eligible": False,
        "uses_supervised_qa_metadata": True,
        "question_dependent_scene_processing": False,
        "live_optimizer_constructed": False,
        "live_optimizer_step_executed": False,
        "optimizer_steps": 0,
        "isolated_clone_optimizer_constructed": True,
        "isolated_clone_optimizer_steps": 1,
    }.items():
        _equal(preflight.get(key), expected, f"preflight.{key}")
    checks = _mapping(preflight.get("authorization_checks"), "authorization checks")
    expected_checks = {
        "source_and_config_contracts_passed",
        "exact_selection_and_order_passed",
        "step_zero_identity_all_scenes",
        "color_losses_exactly_zero",
        "color_isolated_signed_x_gradient_exactly_zero",
        "mirror_signed_x_gradient_finite_nonzero",
        "accumulated_signed_x_gradient_finite_nonzero",
        "only_signed_x_output_weight_has_gradient",
        "predicted_adamw_update_finite_nonzero",
        "local_field_rank_bf16_selectivity_gate",
        "live_source_state_unchanged",
        "live_signed_x_state_unchanged",
        "rng_state_unchanged",
    }
    _equal(set(checks), expected_checks, "authorization-check schema")
    if any(value is not True for value in checks.values()):
        _fail("Every V20 preflight authorization check must be exactly true")
    _equal(preflight.get("config_hash"), config_hash(config, length=64), "preflight.config_hash")
    _equal(preflight.get("contract"), contract, "preflight.contract")
    provenance = _clean_provenance(preflight.get("source_provenance"), "preflight provenance")
    _equal(provenance, dict(current_provenance), "current/preflight provenance")

    _validate_implementation_sources(preflight)

    training = _mapping(config.get("training"), "training")
    source = _safe_existing_input_path(
        str(training["initialize_from"]), "V18 source checkpoint", kind="directory"
    )
    _equal(preflight.get("source_checkpoint"), _display(source), "preflight source")
    _exact_int(preflight.get("source_checkpoint_epoch"), 4, "preflight source epoch")
    source_hashes = dict(_mapping(preflight.get("source_artifact_hashes"), "source artifacts"))
    for key, filename, config_key in (
        ("adapter_sha256", "adapter.safetensors", "initialize_expected_adapter_sha256"),
        ("metadata_sha256", "metadata.json", "initialize_expected_metadata_sha256"),
    ):
        expected = _sha256(training.get(config_key), f"training.{config_key}")
        source_file = _safe_existing_input_path(source / filename, f"V18 source {filename}")
        _equal(file_sha256(source_file), expected, f"source {key}")
        _equal(source_hashes.get(key), expected, f"preflight source {key}")
    source_metadata = _read_json(source / "metadata.json", "V18 source metadata")
    _exact_int(source_metadata.get("epoch"), 4, "V18 source epoch")

    expected_hashes = _mapping(contract.get("expected_hashes"), "expected hashes")
    expected_scene = _sha256(expected_hashes.get("source_scene_state_sha256"), "source scene")
    expected_global = _sha256(
        expected_hashes.get("source_global_scene_residual_state_sha256"), "source global"
    )
    expected_lora = dict(
        _mapping(expected_hashes.get("source_lora_bank_state_sha256"), "source LoRA")
    )
    _equal(
        source_metadata.get("global_scene_residual_state_sha256"),
        expected_global,
        "source global metadata",
    )
    _equal(source_metadata.get("lora_bank_state_sha256"), expected_lora, "source LoRA metadata")
    frozen = dict(_mapping(preflight.get("frozen_state_hashes"), "frozen hashes"))
    for key, expected in {
        "scene_state_sha256": expected_scene,
        "global_scene_residual_state_sha256": expected_global,
        "lora_bank_state_sha256": expected_lora,
    }.items():
        _equal(frozen.get(key), expected, f"preflight frozen {key}")
    _sha256(frozen.get("combined_source_state_sha256"), "combined frozen source")
    observed_source = _mapping(preflight.get("source_hashes"), "source hashes")
    for key, expected in {
        **source_hashes,
        "scene_state_sha256": expected_scene,
        "global_scene_residual_state_sha256": expected_global,
        "lora_bank_state_sha256": expected_lora,
    }.items():
        _equal(observed_source.get(key), expected, f"source hashes {key}")
    _equal(
        preflight.get("source_metadata_global_residual_state_sha256"),
        expected_global,
        "source metadata global",
    )
    _equal(
        preflight.get("source_metadata_lora_bank_state_sha256"),
        expected_lora,
        "source metadata LoRA",
    )

    initial = _sha256(expected_hashes.get("initial_signed_x_state_sha256"), "initial signed-X")
    for key in (
        "initial_signed_x_state_sha256",
        "live_signed_x_state_sha256_before",
        "live_signed_x_state_sha256_after",
    ):
        _equal(preflight.get(key), initial, f"preflight.{key}")
    _equal(preflight.get("live_signed_x_state_unchanged"), True, "live signed-X unchanged")
    _equal(preflight.get("live_source_state_unchanged"), True, "live source unchanged")
    _equal(preflight.get("live_parameter_state_unchanged"), True, "live parameter state")
    for key in ("live_source_state_sha256_before", "live_source_state_sha256_after"):
        _equal(preflight.get(key), frozen["combined_source_state_sha256"], f"preflight.{key}")

    for key in (
        "selection_sha256",
        "pair_membership_sha256",
        "pair_unit_selection_sha256",
        "ordered_unit_sha256",
    ):
        expected_key = key if key != "selection_sha256" else "selection_sha256"
        _equal(preflight.get(key), expected_hashes.get(expected_key), f"preflight {key}")
    microsteps = _sequence(preflight.get("microsteps"), "preflight microsteps")
    _equal(len(microsteps), 12, "preflight microstep count")
    _equal(preflight.get("microstep_losses"), list(microsteps), "microstep aliases")
    for index, row in enumerate(microsteps, start=1):
        item = _mapping(row, f"microstep {index}")
        _exact_int(item.get("microstep"), index, f"microstep {index} index")
        _finite(item.get("total_loss"), f"microstep {index} loss")
    zero_equivalence = _validate_zero_equivalence(preflight.get("zero_output_prefix_equivalence"))
    rich = _validate_rich_evidence(preflight, contract)

    optimizer_contract = dict(_mapping(training.get("optimizer"), "training.optimizer"))
    _equal(preflight.get("adamw_contract"), optimizer_contract, "preflight AdamW contract")
    pair_gradient = _mapping(preflight.get("pair_gradient_audit"), "pair gradient audit")
    for key in (
        "color_total_loss_exact_zero",
        "color_gradient_exact_zero",
        "mirror_gradient_positive_finite",
        "only_signed_x_output_weight_has_gradient",
    ):
        _equal(pair_gradient.get(key), True, f"pair gradient {key}")
    gradient = _mapping(preflight.get("gradient"), "preflight gradient")
    _equal(
        gradient.get("changed_parameter_keys"), ["output_projection.weight"], "changed parameters"
    )
    _exact_int(gradient.get("ordered_microstep_count"), 12, "gradient microsteps")
    _equal(gradient.get("accumulated_finite_nonzero"), True, "accumulated gradient")
    predicted_state = _sha256(gradient.get("predicted_signed_x_state_sha256"), "predicted state")
    predicted_output = _sha256(gradient.get("predicted_output_projection_sha256"), "predicted W")
    if predicted_state == initial:
        _fail("Predicted V20 signed-X state did not change")
    optimizer_manifest = dict(
        _mapping(gradient.get("optimizer_state_manifest"), "optimizer manifest")
    )
    try:
        calculated_optimizer_hash = validate_v19_adamw_state_manifest(
            optimizer_manifest, optimizer_contract
        )
    except V19AdamWStateViolation as error:
        _fail(f"Preflight optimizer manifest is invalid: {error}")
    optimizer_hash = _sha256(gradient.get("optimizer_state_sha256"), "optimizer hash")
    _equal(calculated_optimizer_hash, optimizer_hash, "optimizer manifest hash")
    _sha256(gradient.get("optimizer_state_tensor_sha256"), "optimizer tensor hash")
    _equal(
        preflight.get("predicted_output_weight_sha256"), predicted_output, "top-level predicted W"
    )
    _equal(
        preflight.get("predicted_signed_x_scene_residual_state_sha256"),
        predicted_state,
        "top-level predicted state",
    )
    _equal(
        preflight.get("predicted_canonical_adamw_state_manifest"),
        optimizer_manifest,
        "top-level optimizer manifest",
    )
    _equal(
        preflight.get("predicted_canonical_adamw_state_sha256"),
        optimizer_hash,
        "top-level optimizer hash",
    )
    predicted_first = _mapping(preflight.get("predicted_first_update"), "predicted first update")
    _equal(
        predicted_first.get("predicted_output_weight_sha256"), predicted_output, "first-update W"
    )
    _equal(
        predicted_first.get("predicted_signed_x_scene_residual_state_sha256"),
        predicted_state,
        "first-update state",
    )
    _equal(
        predicted_first.get("canonical_adamw_state_sha256"),
        optimizer_hash,
        "first-update optimizer",
    )
    return {
        "source": source,
        "source_metadata": source_metadata,
        "source_artifact_hashes": source_hashes,
        "source_provenance": provenance,
        "expected_scene_state_sha256": expected_scene,
        "expected_global_state_sha256": expected_global,
        "expected_lora_state_sha256": expected_lora,
        "optimizer_contract": optimizer_contract,
        "optimizer_manifest": optimizer_manifest,
        "optimizer_hash": optimizer_hash,
        "predicted_signed_x_state_sha256": predicted_state,
        "predicted_output_projection_sha256": predicted_output,
        "zero_equivalence": zero_equivalence,
        "rich_preflight_reduction": rich,
        "pair_unit_selection_sha256": expected_hashes["pair_unit_selection_sha256"],
        "pair_membership_sha256": expected_hashes["pair_membership_sha256"],
    }


def _load_tensor_evidence(
    path: Path,
    metadata: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    expected_scene: str,
    expected_global: str,
    expected_lora: Mapping[str, str],
) -> dict[str, Any]:
    path = _safe_existing_input_path(path, "checkpoint adapter")
    try:
        tensors = load_file(path, device="cpu")
    except (OSError, RuntimeError, ValueError, SafetensorError) as error:
        _fail(f"Cannot read checkpoint adapter tensors: {error}")
    nonfinite = sorted(
        key for key, value in tensors.items() if not bool(torch.isfinite(value).all())
    )
    if nonfinite:
        _fail(f"Checkpoint adapter contains nonfinite tensors: {nonfinite}")
    recognized_prefixes = (
        _SIGNED_PREFIX,
        _GLOBAL_PREFIX,
        *_SCENE_PREFIXES,
        _LORA_BANK_PREFIX,
    )
    unrecognized = sorted(
        key for key in tensors if not any(key.startswith(prefix) for prefix in recognized_prefixes)
    )
    if unrecognized:
        _fail(f"Checkpoint adapter contains unrecognized tensor keys: {unrecognized}")
    signed = {key: value for key, value in tensors.items() if key.startswith(_SIGNED_PREFIX)}
    expected_signed = {
        f"{_SIGNED_PREFIX}signed_x_anchors": ((256,), torch.float32),
        f"{_SIGNED_PREFIX}architecture_marker": ((), torch.int64),
        f"{_SIGNED_PREFIX}output_projection.weight": ((1536, 128), torch.float32),
    }
    _equal(set(signed), set(expected_signed), "checkpoint signed-X tensor keys")
    for key, (shape, dtype) in expected_signed.items():
        _equal(tuple(signed[key].shape), shape, f"checkpoint shape {key}")
        _equal(signed[key].dtype, dtype, f"checkpoint dtype {key}")
        if not bool(torch.isfinite(signed[key]).all()):
            _fail(f"Checkpoint tensor is nonfinite: {key}")
    module = construct_signed_x_scene_residual(
        config, scene_dim=1536, latent_count=256, content_dim=128
    )
    if not isinstance(module, SignedXLocalFieldSceneResidual):
        _fail("V20 config did not construct the local-field V2 module")
    try:
        module.load_state_dict(
            {key[len(_SIGNED_PREFIX) :]: value for key, value in signed.items()}, strict=True
        )
        structure = module.validate_structural_state()
    except (RuntimeError, TypeError, ValueError) as error:
        _fail(f"Checkpoint local-field structural state is invalid: {error}")
    _equal(structure.get("architecture_marker"), 2, "checkpoint architecture marker")
    _equal(structure.get("spatial_reduction"), "none", "checkpoint spatial reduction")
    signed_hash = tensor_state_sha256(signed)
    _equal(
        signed_hash,
        metadata.get("signed_x_scene_residual_state_sha256"),
        "checkpoint signed-X metadata hash",
    )
    output_hash = tensor_state_sha256(
        {
            f"{_SIGNED_PREFIX}output_projection.weight": signed[
                f"{_SIGNED_PREFIX}output_projection.weight"
            ]
        }
    )

    global_state = {key: value for key, value in tensors.items() if key.startswith(_GLOBAL_PREFIX)}
    if not global_state or any(
        not bool(torch.isfinite(value).all()) for value in global_state.values()
    ):
        _fail("Checkpoint frozen global residual is missing or nonfinite")
    global_hash = tensor_state_sha256(global_state)
    for key in ("global_scene_residual_state_sha256", "frozen_global_scene_residual_state_sha256"):
        _equal(metadata.get(key), global_hash, f"checkpoint {key}")
    _equal(global_hash, expected_global, "checkpoint/configured frozen global residual")

    scene_state = {
        key: value
        for key, value in tensors.items()
        if any(key.startswith(prefix) for prefix in _SCENE_PREFIXES)
    }
    if not scene_state or any(
        not bool(torch.isfinite(value).all()) for value in scene_state.values()
    ):
        _fail("Checkpoint frozen scene tensor subset is missing or nonfinite")
    scene_hash = tensor_state_sha256(scene_state)
    _equal(
        scene_hash, metadata.get("frozen_scene_state_sha256"), "checkpoint frozen scene metadata"
    )
    _equal(scene_hash, expected_scene, "checkpoint/configured frozen scene")

    observed_lora: dict[str, str] = {}
    consumed: set[str] = set()
    for bank_name in sorted(expected_lora):
        prefix = f"{_LORA_BANK_PREFIX}{bank_name}."
        bank = {
            key[len(prefix) :]: value for key, value in tensors.items() if key.startswith(prefix)
        }
        if not bank or any(not bool(torch.isfinite(value).all()) for value in bank.values()):
            _fail(f"Checkpoint frozen LoRA bank {bank_name!r} is missing or nonfinite")
        consumed.update(key for key in tensors if key.startswith(prefix))
        observed_lora[bank_name] = tensor_state_sha256(bank)
    _equal(
        {key for key in tensors if key.startswith(_LORA_BANK_PREFIX)}, consumed, "LoRA tensor keys"
    )
    _equal(observed_lora, metadata.get("frozen_lora_bank_state_sha256"), "frozen LoRA metadata")
    _equal(observed_lora, metadata.get("lora_bank_state_sha256"), "LoRA metadata")
    _equal(observed_lora, dict(expected_lora), "checkpoint/configured frozen LoRA")
    return {
        "signed_x_state_sha256": signed_hash,
        "output_projection_sha256": output_hash,
        "global_scene_residual_state_sha256": global_hash,
        "scene_state_sha256": scene_hash,
        "lora_bank_state_sha256": observed_lora,
    }


def _load_optimizer_evidence(
    path: Path,
    *,
    contract: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    expected_hash: str,
) -> dict[str, Any]:
    path = _safe_existing_input_path(path, "checkpoint optimizer")
    try:
        state = torch.load(path, weights_only=True, map_location="cpu")
    except (
        EOFError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        _fail(f"Cannot safely deserialize checkpoint optimizer: {error}")
    try:
        manifest, digest = canonical_v19_adamw_state(state, contract)
    except V19AdamWStateViolation as error:
        _fail(f"Checkpoint optimizer violates the V20 AdamW contract: {error}")
    _equal(manifest, dict(expected_manifest), "checkpoint/preflight optimizer manifest")
    _equal(digest, expected_hash, "checkpoint/preflight optimizer hash")
    return {"manifest": manifest, "sha256": digest}


def verify_update1(
    config: dict[str, Any], preflight_path: str | Path, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Verify V20 epoch one without loading Gemma, a map, QA, or oracle data."""

    current = _clean_provenance(capture_git_source_provenance(PROJECT_ROOT), "current provenance")
    preflight_file = _safe_existing_input_path(preflight_path, "V20 preflight")
    checkpoint = _safe_existing_input_path(
        checkpoint_path, "V20 epoch-one checkpoint", kind="directory"
    )
    metadata_file = _safe_existing_input_path(
        checkpoint / "metadata.json", "V20 epoch-one metadata"
    )
    adapter_file = _safe_existing_input_path(
        checkpoint / "adapter.safetensors", "V20 epoch-one adapter"
    )
    optimizer_file = _safe_existing_input_path(
        checkpoint / "optimizer.pt", "V20 epoch-one optimizer"
    )
    preflight = _read_json(preflight_file, "V20 preflight")
    evidence = _validate_preflight(config, preflight, current)
    metadata = _read_json(metadata_file, "V20 epoch-one metadata")
    for key, expected in {
        "schema_version": 3,
        "epoch": 1,
        "optimizer_step": 1,
        "global_step": 12,
    }.items():
        _exact_int(metadata.get(key), expected, f"checkpoint.{key}")
    history = _sequence(metadata.get("history"), "checkpoint history")
    _equal(len(history), 1, "checkpoint history length")
    history_row = _mapping(history[0], "checkpoint history epoch one")
    _exact_int(history_row.get("epoch"), 1, "checkpoint history epoch")
    _exact_int(history_row.get("pair_batch_count"), 12, "checkpoint history pair batches")
    _equal(history_row.get("pair_batch_fraction"), 1.0, "checkpoint history pair fraction")
    _finite(history_row.get("train_loss"), "checkpoint history train loss")
    _equal(
        metadata.get("train_loss"), history_row.get("train_loss"), "checkpoint/history train loss"
    )
    _equal(
        metadata.get("pair_candidate_gate"),
        history_row.get("pair_candidate_gate"),
        "checkpoint/history teacher gate",
    )
    _equal(metadata.get("config_hash"), config_hash(config), "checkpoint.config_hash")
    if (
        not isinstance(metadata.get("config_hash"), str)
        or _SHORT_SHA.fullmatch(metadata["config_hash"]) is None
    ):
        _fail("Checkpoint config hash must be a 12-character lowercase digest")
    training = _mapping(config.get("training"), "training")
    for key, expected in {
        "output_namespace": training.get("output_namespace"),
        "gradient_accumulation": 12,
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": True,
        "question_dependent_scene_processing": False,
        "scene_latents": 256,
        "language_hidden_dim": 1536,
        "counterfactual_pair_unit_count": 12,
        "counterfactual_pair_unit_selection_sha256": evidence["pair_unit_selection_sha256"],
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": evidence["pair_membership_sha256"],
        "max_questions_per_scene": 6,
    }.items():
        _equal(metadata.get(key), expected, f"checkpoint.{key}")
    _equal(
        metadata.get("source_provenance"),
        evidence["source_provenance"],
        "checkpoint/preflight provenance",
    )
    _equal(
        metadata.get("global_scene_residual"),
        global_scene_residual_settings(config).contract(),
        "checkpoint global residual contract",
    )
    _exact_int(
        metadata.get("global_scene_residual_parameter_count"),
        400_128,
        "checkpoint global parameters",
    )
    _equal(
        metadata.get("global_scene_residual_state_sha256"),
        evidence["expected_global_state_sha256"],
        "checkpoint global state",
    )
    _equal(
        metadata.get("frozen_global_scene_residual_state_sha256"),
        evidence["expected_global_state_sha256"],
        "checkpoint frozen global state",
    )
    _equal(
        metadata.get("signed_x_scene_residual"),
        signed_x_scene_residual_settings(config).contract(),
        "checkpoint V20 signed-X contract",
    )
    _exact_int(
        metadata.get("signed_x_scene_residual_parameter_count"),
        196_608,
        "checkpoint signed-X parameters",
    )
    initial = signed_x_scene_residual_settings(config).expected_initial_state_sha256
    _equal(
        metadata.get("signed_x_scene_residual_initial_state_sha256"),
        initial,
        "checkpoint initial signed-X hash",
    )
    if metadata.get("signed_x_scene_residual_state_sha256") == initial:
        _fail("Checkpoint V20 signed-X state did not change after update one")
    _equal(
        metadata.get("signed_x_scene_residual_zero_output_equivalence"),
        evidence["zero_equivalence"],
        "checkpoint/preflight zero equivalence",
    )
    _equal(
        metadata.get("frozen_scene_state_sha256"),
        evidence["expected_scene_state_sha256"],
        "checkpoint frozen scene",
    )
    _equal(
        metadata.get("frozen_lora_bank_state_sha256"),
        evidence["expected_lora_state_sha256"],
        "checkpoint frozen LoRA",
    )
    _equal(
        metadata.get("lora_bank_state_sha256"),
        evidence["expected_lora_state_sha256"],
        "checkpoint LoRA",
    )
    _exact_int(metadata.get("lora_trainable_parameter_count"), 0, "checkpoint trainable LoRA count")
    for stale in ("v18_stage_execution", "v19_stage_execution"):
        if stale in metadata:
            _fail(f"V20 checkpoint improperly carries {stale}")
    for key, expected in {
        "initialize_expected_adapter_sha256": evidence["source_artifact_hashes"]["adapter_sha256"],
        "initialize_expected_metadata_sha256": evidence["source_artifact_hashes"][
            "metadata_sha256"
        ],
        "initialize_expected_global_scene_residual_state_sha256": evidence[
            "expected_global_state_sha256"
        ],
        "initialize_source_residual_into_frozen_base": True,
    }.items():
        _equal(metadata.get(key), expected, f"checkpoint.{key}")
    initialization = _mapping(
        metadata.get("initialization_provenance"), "initialization provenance"
    )
    source_metadata = evidence["source_metadata"]
    for key, expected in {
        "schema_version": 4,
        "mode": "frozen_v18_residual_base_plus_zero_output_signed_x_residual",
        "adapter_sha256": evidence["source_artifact_hashes"]["adapter_sha256"],
        "metadata_sha256": evidence["source_artifact_hashes"]["metadata_sha256"],
        "expected_adapter_sha256": evidence["source_artifact_hashes"]["adapter_sha256"],
        "expected_metadata_sha256": evidence["source_artifact_hashes"]["metadata_sha256"],
        "checkpoint_epoch": 4,
        "checkpoint_output_namespace": source_metadata.get("output_namespace"),
        "checkpoint_config_hash": source_metadata.get("config_hash"),
        "checkpoint_source_provenance": source_metadata.get("source_provenance"),
        "source_global_scene_residual_state_sha256": evidence["expected_global_state_sha256"],
        "expected_source_global_scene_residual_state_sha256": evidence[
            "expected_global_state_sha256"
        ],
        "global_scene_residual_frozen": True,
        "signed_x_scene_residual_initial_state_sha256": initial,
        "signed_x_scene_residual_zero_output": True,
        "optimizer_state_loaded": False,
        "history_loaded": False,
    }.items():
        _equal(initialization.get(key), expected, f"initialization provenance {key}")
    _equal(
        _resolve(str(initialization.get("checkpoint"))),
        evidence["source"],
        "initialization source path",
    )

    tensors = _load_tensor_evidence(
        adapter_file,
        metadata,
        config=config,
        expected_scene=evidence["expected_scene_state_sha256"],
        expected_global=evidence["expected_global_state_sha256"],
        expected_lora=evidence["expected_lora_state_sha256"],
    )
    _equal(
        tensors["signed_x_state_sha256"],
        evidence["predicted_signed_x_state_sha256"],
        "predicted/actual signed-X state",
    )
    _equal(
        tensors["output_projection_sha256"],
        evidence["predicted_output_projection_sha256"],
        "predicted/actual output weight",
    )
    optimizer = _load_optimizer_evidence(
        optimizer_file,
        contract=evidence["optimizer_contract"],
        expected_manifest=evidence["optimizer_manifest"],
        expected_hash=evidence["optimizer_hash"],
    )
    return {
        "schema_version": 1,
        "audit_type": UPDATE1_VERIFIER_TYPE,
        "match": True,
        "stage_2_authorized": True,
        "report_only": True,
        "model_loaded": False,
        "scene_map_loaded": False,
        "oracle_loaded": False,
        "optimizer_deserialized": True,
        "optimizer_deserialization": {
            "weights_only": True,
            "map_location": "cpu",
            "canonical_state_validated": True,
        },
        "source_provenance": dict(current),
        "config_hash": config_hash(config, length=64),
        "preflight_sha256": file_sha256(preflight_file),
        "rich_preflight_reduction": evidence["rich_preflight_reduction"],
        "checkpoint": _display(checkpoint),
        "checkpoint_artifact_hashes": {
            "adapter_sha256": file_sha256(adapter_file),
            "metadata_sha256": file_sha256(metadata_file),
            "optimizer_sha256": file_sha256(optimizer_file),
        },
        "signed_x_state_sha256": tensors["signed_x_state_sha256"],
        "output_projection_sha256": tensors["output_projection_sha256"],
        "frozen_global_scene_residual_state_sha256": tensors["global_scene_residual_state_sha256"],
        "frozen_scene_state_sha256": tensors["scene_state_sha256"],
        "frozen_lora_bank_state_sha256": tensors["lora_bank_state_sha256"],
        "optimizer_state_manifest": optimizer["manifest"],
        "optimizer_state_sha256": optimizer["sha256"],
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    config_path = _safe_existing_input_path(args.config, "V20 config")
    report = verify_update1(load_config(config_path), args.preflight, args.checkpoint)
    destination = _resolve(args.report)
    _reject_forbidden_input_path(destination)
    _atomic_json(destination, report)
    print(
        json.dumps(
            {
                "phase": "v20_update1_verifier",
                "report": _display(destination),
                "match": True,
                "stage_2_authorized": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

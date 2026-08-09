"""Exact report-only verifier for V21's phase-aware first update.

The verifier intentionally never loads Gemma, QA data, maps, rendered images,
runtime artifacts, or oracle data.  It binds the authorized BF16 structural
preflight to the bytes of epoch one, safely inspects the checkpoint tensors and
one-matrix AdamW state on CPU, and authorizes the remaining three screen
updates only when the predicted and observed states match exactly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.evaluation import v20_update1_verifier as v20
from semantic_3d_chat.evaluation.v19_optimizer_state import (
    V19AdamWStateViolation,
    validate_v19_adamw_state_manifest,
)
from semantic_3d_chat.evaluation.v21_phase_aware_precision import (
    PHASE_AWARE_PRECISION_PAIR_V1,
)
from semantic_3d_chat.evaluation.v21_predicted_update_audit import (
    V21_FUNCTIONAL_AUDIT_TYPE,
    V21PredictedUpdateAuditViolation,
    evaluate_v21_predicted_update,
)
from semantic_3d_chat.evaluation.v21_structural_preflight import (
    COLOR_PAIR_ID,
    EXPECTED_SCENE_IDS,
    MIRROR_PAIR_ID,
    V21_PREFLIGHT_ROLE,
    V21StructuralPreflightViolation,
    canonical_sha256,
    evaluate_v21_structural_gate,
    validate_v21_config_contract,
)
from semantic_3d_chat.scene_encoder.global_residual import global_scene_residual_settings
from semantic_3d_chat.scene_encoder.signed_x_dispatch import signed_x_scene_residual_settings
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)
from semantic_3d_chat.training.train_adapter import file_sha256

UPDATE1_VERIFIER_TYPE = "v21_exact_update1_match_verifier"
MODEL_DTYPE = "bfloat16"
PRECISION_ALGORITHM = "bfloat16_cast_of_fp32_base_plus_fp32_delta"
PHASE_ALGORITHM = "phase_aware_bfloat16_pair_v1"
EXPECTED_V21_CONTRACT_SHA256 = "50e5522a19d4f6a3eb88884cdccfa71ab1301ebe94bf1a512d42505322799b2c"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHORT_SHA = re.compile(r"[0-9a-f]{12}")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_EXPECTED_PAIRS = {COLOR_PAIR_ID, MIRROR_PAIR_ID}
_EXPECTED_PAIR_SCENES = {
    COLOR_PAIR_ID: ("scene_000003", "scene_000004"),
    MIRROR_PAIR_ID: ("scene_000007", "scene_000008"),
}
_IMPLEMENTATION_SOURCES = {
    "implementation_source": "src/semantic_3d_chat/evaluation/v21_structural_preflight.py",
    "signed_x_implementation_source": (
        "src/semantic_3d_chat/scene_encoder/signed_x_local_field.py"
    ),
    "signed_x_dispatch_implementation_source": (
        "src/semantic_3d_chat/scene_encoder/signed_x_dispatch.py"
    ),
    "phase_audit_implementation_source": (
        "src/semantic_3d_chat/evaluation/v21_phase_aware_precision.py"
    ),
    "functional_audit_implementation_source": (
        "src/semantic_3d_chat/evaluation/v21_predicted_update_audit.py"
    ),
}
_AUTHORIZATION_CHECKS = {
    "source_and_config_contracts_passed",
    "exact_selection_and_order_passed",
    "step_zero_identity_all_scenes",
    "color_losses_exactly_zero",
    "color_isolated_signed_x_gradient_exactly_zero",
    "mirror_signed_x_gradient_finite_nonzero",
    "accumulated_signed_x_gradient_finite_nonzero",
    "only_signed_x_output_weight_has_gradient",
    "predicted_adamw_update_finite_nonzero",
    "predicted_teacher_forced_functional_gate",
    "local_field_rank_precision_phase_gate",
    "live_source_state_unchanged",
    "live_signed_x_state_unchanged",
    "rng_state_unchanged",
}
_PREFLIGHT_ROOT_KEYS = {
    "schema_version",
    "audit_type",
    "runtime_eligible",
    "uses_supervised_qa_metadata",
    "question_dependent_scene_processing",
    "model_dtype",
    "live_optimizer_constructed",
    "live_optimizer_step_executed",
    "optimizer_steps",
    "isolated_clone_optimizer_constructed",
    "isolated_clone_optimizer_steps",
    "authorized",
    "structural_authorization",
    "authorization_checks",
    "config_path",
    "config_hash",
    "contract",
    "adamw_contract",
    "source_provenance",
    "implementation_source",
    "implementation_source_sha256",
    "signed_x_implementation_source",
    "signed_x_implementation_source_sha256",
    "signed_x_dispatch_implementation_source",
    "signed_x_dispatch_implementation_source_sha256",
    "phase_audit_implementation_source",
    "phase_audit_implementation_source_sha256",
    "functional_audit_implementation_source",
    "functional_audit_implementation_source_sha256",
    "source_checkpoint",
    "source_checkpoint_epoch",
    "source_artifact_hashes",
    "frozen_state_hashes",
    "source_hashes",
    "source_metadata_global_residual_state_sha256",
    "source_metadata_scene_state_sha256",
    "source_metadata_lora_bank_state_sha256",
    "source_precision_transition",
    "initial_signed_x_state_sha256",
    "live_source_state_sha256_before",
    "live_source_state_sha256_after",
    "live_source_state_unchanged",
    "live_signed_x_state_sha256_before",
    "live_signed_x_state_sha256_after",
    "live_signed_x_state_unchanged",
    "live_parameter_state_unchanged",
    "selection_sha256",
    "pair_membership_sha256",
    "pair_unit_selection_sha256",
    "selected_pair_units",
    "ordered_unit_sha256",
    "ordered_units",
    "pair_objective_policy",
    "pair_objective_policy_coverage",
    "zero_output_prefix_equivalence",
    "signed_x_structural_state",
    "local_field_structural_state",
    "local_dependence",
    "local_hidden_spatial_rank",
    "centered_content",
    "microsteps",
    "microstep_losses",
    "pair_gradient_audit",
    "gradient",
    "predicted_first_update",
    "predicted_update_functional_audit",
    "predicted_output_weight_sha256",
    "predicted_signed_x_scene_residual_state_sha256",
    "predicted_canonical_adamw_state_sha256",
    "predicted_canonical_adamw_state_manifest",
    "raw_fp32_centered_scene_delta",
    "precision_cast_audit",
    "model_effective_scene_delta",
    "effective_cast_scene_delta",
    "raw_fp32_centered_pair_delta",
    "model_effective_pair_delta",
    "effective_cast_pair_delta",
    "phase_aware_pair_diagnostics",
    "structural_gate",
    "rng_state",
}


class V21Update1Violation(ValueError):
    """A mismatch that denies V21 stage-two execution and selection."""


def _fail(message: str) -> None:
    raise V21Update1Violation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _equal(observed: Any, expected: Any, field: str) -> None:
    if observed != expected:
        _fail(f"{field} mismatch: expected={expected!r} observed={observed!r}")


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _exact_int(value: Any, expected: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        _fail(f"{field} mismatch: expected={expected} observed={value!r}")


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _finite_tree(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"{field} contains a non-string object key")
            _finite_tree(item, f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{field}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(f"{field} contains NaN or infinity")


def _canonical_equal(observed: Any, expected: Any, field: str) -> None:
    try:
        if canonical_sha256(observed) != canonical_sha256(expected):
            _fail(f"{field} canonical mismatch")
    except (TypeError, ValueError) as error:
        _fail(f"{field} is not finite canonical JSON: {error}")


def _safe(path: str | Path, field: str, *, kind: str = "file") -> Path:
    try:
        return v20._safe_existing_input_path(path, field, kind=kind)
    except v20.V20Update1Violation as error:
        _fail(str(error))


def _read_json(path: str | Path, field: str) -> dict[str, Any]:
    safe = _safe(path, field)
    try:
        value = json.loads(safe.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"Cannot read {field} JSON at {safe}: {error}")
    result = dict(_mapping(value, field))
    _finite_tree(result, field)
    return result


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _clean_provenance(value: Any, field: str) -> dict[str, Any]:
    result = dict(_mapping(value, field))
    try:
        require_clean_committed_source(result)
    except RuntimeError as error:
        _fail(f"{field} is not clean committed source provenance: {error}")
    _equal(result.get("tracked_diff_sha256"), _EMPTY_SHA256, f"{field}.tracked_diff")
    return result


def _validate_implementation_sources(preflight: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, relative in _IMPLEMENTATION_SOURCES.items():
        observed = preflight.get(field)
        if observed != relative:
            _fail(f"preflight {field} must name the canonical V21 source")
        canonical = _safe(PROJECT_ROOT / relative, f"canonical {field}")
        observed_path = _safe(str(observed), f"preflight {field}")
        _equal(observed_path, canonical, f"preflight {field} canonical path")
        digest = _sha256(preflight.get(f"{field}_sha256"), f"preflight {field} hash")
        _equal(file_sha256(canonical), digest, f"preflight {field} byte hash")
        result[field] = relative
        result[f"{field}_sha256"] = digest
    return result


def _validate_selection_and_order(
    preflight: Mapping[str, Any], expected_hashes: Mapping[str, Any]
) -> dict[str, Any]:
    for field in (
        "selection_sha256",
        "pair_membership_sha256",
        "pair_unit_selection_sha256",
        "ordered_unit_sha256",
    ):
        _equal(preflight.get(field), expected_hashes.get(field), f"preflight.{field}")
    selected = list(_sequence(preflight.get("selected_pair_units"), "selected pair units"))
    ordered = list(_sequence(preflight.get("ordered_units"), "ordered units"))
    if len(selected) != 12 or len(ordered) != 12:
        _fail("V21 selection and epoch-one order must each contain exactly twelve units")
    _equal(
        canonical_sha256(selected),
        expected_hashes["pair_unit_selection_sha256"],
        "selected pair-unit hash",
    )
    _equal(
        canonical_sha256(ordered),
        expected_hashes["ordered_unit_sha256"],
        "ordered unit hash",
    )
    selected_identities: set[tuple[Any, ...]] = set()
    for index, raw in enumerate(selected):
        row = _mapping(raw, f"selected pair unit {index}")
        if set(row) != {"pair_id", "question_key", "scene_ids", "question_ids"}:
            _fail(f"selected pair unit {index} keys mismatch")
        scenes = list(_sequence(row["scene_ids"], f"selected pair unit {index} scenes"))
        questions = list(_sequence(row["question_ids"], f"selected pair unit {index} question IDs"))
        if len(scenes) != 2 or len(questions) != 2:
            _fail("Every selected pair unit must contain exactly two scene/question sides")
        pair_id = row["pair_id"]
        if pair_id not in _EXPECTED_PAIRS or tuple(scenes) != _EXPECTED_PAIR_SCENES[pair_id]:
            _fail(f"selected pair unit {index} has incorrect pair membership")
        identity = (pair_id, row["question_key"], *scenes, *questions)
        if identity in selected_identities:
            _fail("selected pair units contain a duplicate identity")
        selected_identities.add(identity)
    ordered_identities: set[tuple[Any, ...]] = set()
    expected_order_keys = {
        "microstep",
        "pair_id",
        "question_key",
        "reference_scene_id",
        "reference_question_id",
        "counterfactual_scene_id",
        "counterfactual_question_id",
    }
    for microstep, raw in enumerate(ordered, start=1):
        row = _mapping(raw, f"ordered unit {microstep}")
        if set(row) != expected_order_keys:
            _fail(f"ordered unit {microstep} keys mismatch")
        _exact_int(row["microstep"], microstep, f"ordered unit {microstep}.microstep")
        pair_id = row["pair_id"]
        scenes = (row["reference_scene_id"], row["counterfactual_scene_id"])
        if pair_id not in _EXPECTED_PAIRS or scenes != _EXPECTED_PAIR_SCENES[pair_id]:
            _fail(f"ordered unit {microstep} has incorrect pair membership")
        ordered_identities.add(
            (
                pair_id,
                row["question_key"],
                *scenes,
                row["reference_question_id"],
                row["counterfactual_question_id"],
            )
        )
    _equal(ordered_identities, selected_identities, "selected/ordered unit identities")
    return {
        "selection_sha256": expected_hashes["selection_sha256"],
        "pair_membership_sha256": expected_hashes["pair_membership_sha256"],
        "pair_unit_selection_sha256": expected_hashes["pair_unit_selection_sha256"],
        "ordered_unit_sha256": expected_hashes["ordered_unit_sha256"],
    }


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


def _validate_scene_delta(value: Any, scene_id: str, *, dtype: str) -> dict[str, Any]:
    row = dict(_mapping(value, f"scene delta {scene_id}"))
    if set(row) != _SCENE_DELTA_KEYS:
        _fail(f"scene delta {scene_id} keys mismatch")
    _equal(row["shape"], [1, 256, 1536], f"scene delta {scene_id}.shape")
    core = _finite(row["core_rms"], f"scene delta {scene_id}.core_rms")
    delta = _finite(row["delta_rms"], f"scene delta {scene_id}.delta_rms")
    ratio = _finite(row["delta_to_core_rms_ratio"], f"scene delta {scene_id}.ratio")
    if core <= 0.0 or delta <= 0.0 or ratio < 0.0:
        _fail(f"scene delta {scene_id} must have positive core/delta energy")
    if not math.isclose(ratio, delta / core, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"scene delta {scene_id} ratio is inconsistent")
    total = _finite(row["total_energy"], f"scene delta {scene_id}.total_energy")
    mean = _finite(row["across_slot_mean_energy"], f"scene delta {scene_id}.mean_energy")
    varying = _finite(row["slot_varying_energy"], f"scene delta {scene_id}.varying_energy")
    mean_fraction = _finite(
        row["across_slot_mean_energy_fraction"], f"scene delta {scene_id}.mean_fraction"
    )
    varying_fraction = _finite(
        row["slot_varying_energy_fraction"], f"scene delta {scene_id}.varying_fraction"
    )
    if min(total, mean, varying, mean_fraction, varying_fraction) < 0.0:
        _fail(f"scene delta {scene_id} contains negative energy")
    if not 0.0 <= mean_fraction <= 1.0 or not 0.0 <= varying_fraction <= 1.0:
        _fail(f"scene delta {scene_id} energy fraction is out of range")
    if not math.isclose(mean + varying, total, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"scene delta {scene_id} energy decomposition is inconsistent")
    if not math.isclose(mean_fraction, mean / total, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"scene delta {scene_id} mean-energy fraction is inconsistent")
    if not math.isclose(varying_fraction, varying / total, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail(f"scene delta {scene_id} varying-energy fraction is inconsistent")
    for key in (
        "slot_mean_absolute_maximum",
        "delta_absolute_maximum",
        "energy_closure_absolute_error",
    ):
        if _finite(row[key], f"scene delta {scene_id}.{key}") < 0.0:
            _fail(f"scene delta {scene_id}.{key} must be nonnegative")
    for key in ("positive_finite_total_energy", "positive_finite_core_rms"):
        if row[key] is not True:
            _fail(f"scene delta {scene_id}.{key} must be exactly true")
    _sha256(row["delta_sha256"], f"scene delta {scene_id}.delta_sha256")
    _equal(row["dtype"], dtype, f"scene delta {scene_id}.dtype")
    return row


_PRECISION_KEYS = {
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
}


def _validate_precision_row(value: Any, scene_id: str) -> dict[str, Any]:
    row = dict(_mapping(value, f"precision audit {scene_id}"))
    if set(row) != _PRECISION_KEYS:
        _fail(f"precision audit {scene_id} keys mismatch")
    _exact_int(row["schema_version"], 1, f"precision audit {scene_id}.schema")
    for key, expected in {
        "algorithm": PRECISION_ALGORITHM,
        "base_source_dtype": "float32",
        "model_dtype": MODEL_DTYPE,
        "comparison_dtype": "float64",
    }.items():
        _equal(row[key], expected, f"precision audit {scene_id}.{key}")
    count = 256 * 1536
    _exact_int(row["element_count"], count, f"precision audit {scene_id}.element_count")
    changed = row["changed_element_count"]
    if isinstance(changed, bool) or not isinstance(changed, int) or not 1 <= changed <= count:
        _fail(f"precision audit {scene_id}.changed_element_count is invalid")
    fraction = _finite(row["changed_element_fraction"], f"precision audit {scene_id}.fraction")
    if not math.isclose(fraction, changed / count, rel_tol=1.0e-12, abs_tol=1.0e-15):
        _fail(f"precision audit {scene_id}.changed_element_fraction is inconsistent")
    raw = _finite(row["raw_delta_rms"], f"precision audit {scene_id}.raw_delta_rms")
    effective = _finite(
        row["effective_delta_rms"], f"precision audit {scene_id}.effective_delta_rms"
    )
    error = _finite(
        row["quantization_error_rms"], f"precision audit {scene_id}.quantization_error_rms"
    )
    if raw <= 0.0 or effective <= 0.0 or error < 0.0:
        _fail(f"precision audit {scene_id} RMS values are invalid")
    for key, expected in {
        "effective_to_raw_rms_ratio": effective / raw,
        "quantization_error_to_raw_rms_ratio": error / raw,
    }.items():
        observed = _finite(row[key], f"precision audit {scene_id}.{key}")
        if not math.isclose(observed, expected, rel_tol=1.0e-6, abs_tol=1.0e-12):
            _fail(f"precision audit {scene_id}.{key} is inconsistent")
    cosine = _finite(row["raw_effective_cosine"], f"precision audit {scene_id}.cosine")
    if not -1.0 <= cosine <= 1.0:
        _fail(f"precision audit {scene_id}.raw_effective_cosine is out of range")
    _sha256(row["raw_delta_sha256"], f"precision audit {scene_id}.raw hash")
    _sha256(row["effective_delta_sha256"], f"precision audit {scene_id}.effective hash")
    return row


def _validate_pair_delta(value: Any, pair_id: str, *, field: str) -> dict[str, Any]:
    try:
        return v20._validate_pair_delta_row(value, pair_id, effective=field == "effective")
    except v20.V20Update1Violation as error:
        _fail(str(error))


_DECOMPOSITION_KEYS = {
    "schema_version",
    "shape",
    "element_count",
    "raw_pair_exact_zero",
    "effective_pair_exact_zero",
    "raw_pair_rms",
    "effective_pair_rms",
    "quantization_pair_error_rms",
    "effective_to_raw_rms_ratio",
    "quantization_error_to_raw_rms_ratio",
    "raw_effective_cosine",
    "aligned_gain",
    "aligned_effective_rms",
    "parallel_quantization_gain_bias",
    "parallel_quantization_rms",
    "orthogonal_quantization_rms",
    "orthogonal_quantization_to_raw_rms_ratio",
    "orthogonal_quantization_fraction_of_total_error",
    "orthogonality_absolute_dot",
    "noise_energy_closure_absolute_error",
    "decomposition_closure_absolute_maximum",
    "raw_pair_delta_sha256",
    "effective_pair_delta_sha256",
    "quantization_pair_error_sha256",
}


def _validate_decomposition(value: Any, field: str) -> dict[str, Any]:
    row = dict(_mapping(value, field))
    if set(row) != _DECOMPOSITION_KEYS:
        _fail(f"{field} keys mismatch")
    _exact_int(row["schema_version"], 1, f"{field}.schema_version")
    _equal(row["shape"], [1, 256, 1536], f"{field}.shape")
    _exact_int(row["element_count"], 256 * 1536, f"{field}.element_count")
    for key in ("raw_pair_exact_zero", "effective_pair_exact_zero"):
        if type(row[key]) is not bool:
            _fail(f"{field}.{key} must be a boolean")
    raw_rms = _finite(row["raw_pair_rms"], f"{field}.raw_pair_rms")
    effective_rms = _finite(row["effective_pair_rms"], f"{field}.effective_pair_rms")
    error_rms = _finite(row["quantization_pair_error_rms"], f"{field}.quantization_pair_error_rms")
    if raw_rms <= 0.0 or effective_rms <= 0.0 or error_rms < 0.0:
        _fail(f"{field} requires a nonzero raw/effective pair response")
    if row["raw_pair_exact_zero"] is not False or row["effective_pair_exact_zero"] is not False:
        _fail(f"{field} exact-zero flags contradict its positive RMS values")
    numeric = {
        key: _finite(row[key], f"{field}.{key}")
        for key in (
            "effective_to_raw_rms_ratio",
            "quantization_error_to_raw_rms_ratio",
            "raw_effective_cosine",
            "aligned_gain",
            "aligned_effective_rms",
            "parallel_quantization_gain_bias",
            "parallel_quantization_rms",
            "orthogonal_quantization_rms",
            "orthogonal_quantization_to_raw_rms_ratio",
            "orthogonality_absolute_dot",
            "noise_energy_closure_absolute_error",
            "decomposition_closure_absolute_maximum",
        )
    }
    orthogonal_fraction_raw = row["orthogonal_quantization_fraction_of_total_error"]
    orthogonal_fraction = (
        None
        if orthogonal_fraction_raw is None
        else _finite(
            orthogonal_fraction_raw,
            f"{field}.orthogonal_quantization_fraction_of_total_error",
        )
    )
    if not -1.0 <= numeric["raw_effective_cosine"] <= 1.0:
        _fail(f"{field}.raw_effective_cosine is out of range")
    for key in (
        "parallel_quantization_rms",
        "orthogonal_quantization_rms",
        "orthogonal_quantization_to_raw_rms_ratio",
        "orthogonality_absolute_dot",
        "noise_energy_closure_absolute_error",
        "decomposition_closure_absolute_maximum",
    ):
        if numeric[key] < 0.0:
            _fail(f"{field}.{key} must be nonnegative")
    expected_values = {
        "effective_to_raw_rms_ratio": effective_rms / raw_rms,
        "quantization_error_to_raw_rms_ratio": error_rms / raw_rms,
        "raw_effective_cosine": numeric["aligned_gain"] * raw_rms / effective_rms,
        "aligned_effective_rms": numeric["aligned_gain"] * raw_rms,
        "parallel_quantization_gain_bias": numeric["aligned_gain"] - 1.0,
        "parallel_quantization_rms": (abs(numeric["parallel_quantization_gain_bias"]) * raw_rms),
        "orthogonal_quantization_to_raw_rms_ratio": (
            numeric["orthogonal_quantization_rms"] / raw_rms
        ),
    }
    for key, expected in expected_values.items():
        if not math.isclose(numeric[key], expected, rel_tol=1.0e-6, abs_tol=1.0e-12):
            _fail(f"{field}.{key} is inconsistent")
    if error_rms == 0.0:
        if orthogonal_fraction is not None:
            _fail(
                f"{field}.orthogonal_quantization_fraction_of_total_error "
                "must be null when total quantization error is zero"
            )
    else:
        expected_fraction = numeric["orthogonal_quantization_rms"] / error_rms
        if orthogonal_fraction is None or not math.isclose(
            orthogonal_fraction,
            expected_fraction,
            rel_tol=1.0e-6,
            abs_tol=1.0e-12,
        ):
            _fail(f"{field}.orthogonal_quantization_fraction_of_total_error is inconsistent")

    aligned_rms = numeric["aligned_effective_rms"]
    parallel_rms = numeric["parallel_quantization_rms"]
    orthogonal_rms = numeric["orthogonal_quantization_rms"]
    if not math.isclose(
        effective_rms * effective_rms,
        aligned_rms * aligned_rms + orthogonal_rms * orthogonal_rms,
        rel_tol=1.0e-6,
        abs_tol=1.0e-18,
    ):
        _fail(f"{field} effective-response energy decomposition is inconsistent")
    if not math.isclose(
        error_rms * error_rms,
        parallel_rms * parallel_rms + orthogonal_rms * orthogonal_rms,
        rel_tol=1.0e-6,
        abs_tol=1.0e-18,
    ):
        _fail(f"{field} quantization-error energy decomposition is inconsistent")

    element_count = int(row["element_count"])
    orthogonality_scale = element_count * raw_rms * orthogonal_rms
    if numeric["orthogonality_absolute_dot"] > max(
        1.0e-9,
        1.0e-8 * orthogonality_scale,
    ):
        _fail(f"{field}.orthogonality_absolute_dot exceeds numerical tolerance")
    noise_energy_scale = element_count * error_rms * error_rms
    if numeric["noise_energy_closure_absolute_error"] > max(
        1.0e-9,
        1.0e-8 * noise_energy_scale,
    ):
        _fail(f"{field}.noise_energy_closure_absolute_error exceeds numerical tolerance")
    closure_scale = max(raw_rms, effective_rms, error_rms)
    if numeric["decomposition_closure_absolute_maximum"] > max(
        1.0e-12,
        1.0e-8 * closure_scale,
    ):
        _fail(f"{field}.decomposition_closure_absolute_maximum exceeds numerical tolerance")
    for key in (
        "raw_pair_delta_sha256",
        "effective_pair_delta_sha256",
        "quantization_pair_error_sha256",
    ):
        _sha256(row[key], f"{field}.{key}")
    return row


def _validate_phase_evidence(value: Any) -> dict[str, Any]:
    evidence = dict(_mapping(value, "phase-aware pair diagnostics"))
    _equal(set(evidence), _EXPECTED_PAIRS, "phase-aware pair set")
    expected_root = {
        "schema_version",
        "algorithm_family",
        "algorithm",
        "model_dtype",
        "source_dtype",
        "comparison_dtype",
        "shape",
        "element_count",
        "definitions",
        "actual_pair",
        "shared_base",
        "common_delta_null",
        "tensor_hashes",
    }
    expected_tensor_hashes = {
        "base_first_sha256",
        "base_second_sha256",
        "raw_delta_first_sha256",
        "raw_delta_second_sha256",
        "effective_first_sha256",
        "effective_second_sha256",
    }
    expected_shared = {
        "first_base",
        "second_base",
        "mean_response",
        "phase_spread_rms",
        "phase_spread_to_raw_pair_rms_ratio",
        "phase_spread_sha256",
    }
    expected_null = {
        "raw_pair_delta_exact_zero_by_construction",
        "common_delta_rms",
        "response_rms",
        "response_to_raw_pair_rms_ratio",
        "response_to_actual_effective_pair_rms_ratio",
        "response_sha256",
    }
    expected_definitions = {
        "raw_pair_delta": "raw_delta_first_minus_raw_delta_second",
        "effective_scene_delta": "model_dtype(base_plus_raw_delta)_minus_model_dtype(base)",
        "effective_pair_delta": "effective_first_minus_effective_second",
        "quantization_pair_error": "effective_pair_delta_minus_raw_pair_delta",
        "common_delta": "arithmetic_mean_of_raw_scene_deltas",
    }
    for pair_id, raw in evidence.items():
        row = dict(_mapping(raw, f"phase-aware {pair_id}"))
        if set(row) != expected_root:
            _fail(f"phase-aware {pair_id} root keys mismatch")
        for key, expected in {
            "schema_version": 1,
            "algorithm_family": PHASE_AWARE_PRECISION_PAIR_V1,
            "algorithm": PHASE_ALGORITHM,
            "model_dtype": MODEL_DTYPE,
            "source_dtype": "float32",
            "comparison_dtype": "float64_cpu",
            "shape": [1, 256, 1536],
            "element_count": 256 * 1536,
            "definitions": expected_definitions,
        }.items():
            _equal(row[key], expected, f"phase-aware {pair_id}.{key}")
        actual = _validate_decomposition(row["actual_pair"], f"phase-aware {pair_id}.actual_pair")
        shared = _mapping(row["shared_base"], f"phase-aware {pair_id}.shared_base")
        null = _mapping(row["common_delta_null"], f"phase-aware {pair_id}.common_delta_null")
        hashes = _mapping(row["tensor_hashes"], f"phase-aware {pair_id}.tensor_hashes")
        if set(shared) != expected_shared or set(null) != expected_null:
            _fail(f"phase-aware {pair_id} control schema mismatch")
        if set(hashes) != expected_tensor_hashes:
            _fail(f"phase-aware {pair_id} tensor-hash schema mismatch")
        shared_decompositions = {
            key: _validate_decomposition(shared[key], f"phase-aware {pair_id}.shared_base.{key}")
            for key in ("first_base", "second_base", "mean_response")
        }
        for key, decomposition in shared_decompositions.items():
            if decomposition["raw_pair_delta_sha256"] != actual["raw_pair_delta_sha256"]:
                _fail(f"phase-aware {pair_id}.shared_base.{key} does not use the actual raw pair")
            if not math.isclose(
                float(decomposition["raw_pair_rms"]),
                float(actual["raw_pair_rms"]),
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ):
                _fail(
                    f"phase-aware {pair_id}.shared_base.{key}.raw_pair_rms "
                    "differs from the actual raw pair"
                )
        if null["raw_pair_delta_exact_zero_by_construction"] is not True:
            _fail(f"phase-aware {pair_id} common-delta null is not exact by construction")
        for key, digest in hashes.items():
            _sha256(digest, f"phase-aware {pair_id}.tensor_hashes.{key}")
        _sha256(shared["phase_spread_sha256"], f"phase-aware {pair_id}.phase spread hash")
        _sha256(null["response_sha256"], f"phase-aware {pair_id}.null response hash")
        phase_spread_rms = _finite(
            shared["phase_spread_rms"], f"phase-aware {pair_id}.phase spread RMS"
        )
        phase_spread_ratio = _finite(
            shared["phase_spread_to_raw_pair_rms_ratio"],
            f"phase-aware {pair_id}.phase spread ratio",
        )
        common_delta_rms = _finite(
            null["common_delta_rms"], f"phase-aware {pair_id}.common delta RMS"
        )
        null_response_rms = _finite(
            null["response_rms"], f"phase-aware {pair_id}.null response RMS"
        )
        null_raw_ratio = _finite(
            null["response_to_raw_pair_rms_ratio"],
            f"phase-aware {pair_id}.null/raw ratio",
        )
        null_effective_ratio = _finite(
            null["response_to_actual_effective_pair_rms_ratio"],
            f"phase-aware {pair_id}.null/effective ratio",
        )
        if min(phase_spread_rms, common_delta_rms, null_response_rms) < 0.0:
            _fail(f"phase-aware {pair_id} control RMS values must be nonnegative")
        ratio_identities = {
            "phase_spread_to_raw_pair_rms_ratio": (
                phase_spread_ratio,
                phase_spread_rms / float(actual["raw_pair_rms"]),
            ),
            "response_to_raw_pair_rms_ratio": (
                null_raw_ratio,
                null_response_rms / float(actual["raw_pair_rms"]),
            ),
            "response_to_actual_effective_pair_rms_ratio": (
                null_effective_ratio,
                null_response_rms / float(actual["effective_pair_rms"]),
            ),
        }
        for field_name, (observed, expected) in ratio_identities.items():
            if not math.isclose(observed, expected, rel_tol=1.0e-6, abs_tol=1.0e-12):
                _fail(f"phase-aware {pair_id}.{field_name} is inconsistent")
    _finite_tree(evidence, "phase-aware pair diagnostics")
    return evidence


def _validate_functional_audit(value: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    audit = dict(_mapping(value, "predicted functional audit"))
    if audit.get("audit_type") != V21_FUNCTIONAL_AUDIT_TYPE:
        _fail("Predicted functional audit type mismatch")
    roles = _mapping(audit.get("pair_roles"), "predicted functional pair roles")
    _equal(roles, {"color": COLOR_PAIR_ID, "mirror": MIRROR_PAIR_ID}, "functional pair roles")
    requirements = _mapping(contract.get("predicted_update_requires"), "predicted requirements")
    _exact_int(
        audit.get("expected_units_per_pair"),
        int(requirements["expected_units_per_pair"]),
        "functional expected units",
    )
    _exact_int(
        audit.get("expected_sides_per_pair"),
        int(requirements["expected_sides_per_pair"]),
        "functional expected sides",
    )
    measurements = _mapping(audit.get("measurements"), "functional measurements")
    try:
        recomputed = evaluate_v21_predicted_update(
            _sequence(measurements.get("before"), "functional before measurements"),
            _sequence(measurements.get("after"), "functional after measurements"),
            policies=_mapping(audit.get("policies"), "functional policies"),
            color_pair_id=COLOR_PAIR_ID,
            mirror_pair_id=MIRROR_PAIR_ID,
            expected_units_per_pair=int(requirements["expected_units_per_pair"]),
        )
    except (KeyError, TypeError, ValueError, V21PredictedUpdateAuditViolation) as error:
        _fail(f"Cannot recompute V21 predicted functional audit: {error}")
    _canonical_equal(audit, recomputed, "predicted functional audit recomputation")
    if recomputed.get("passed") is not True:
        _fail("Predicted functional audit did not pass")
    return recomputed


def _validate_rich_evidence(
    preflight: Mapping[str, Any], contract: Mapping[str, Any], sources: Mapping[str, Any]
) -> dict[str, Any]:
    requirements = _mapping(contract.get("structural_preflight_requires"), "requirements")
    try:
        structural = v20._validate_structural_row(preflight.get("local_field_structural_state"))
        _canonical_equal(
            preflight.get("signed_x_structural_state"), structural, "structure aliases"
        )
        dependence = v20._validate_local_dependence_row(preflight.get("local_dependence"))
        ranks_raw = dict(_mapping(preflight.get("local_hidden_spatial_rank"), "local ranks"))
        _equal(set(ranks_raw), set(EXPECTED_SCENE_IDS), "local rank scene set")
        minimum_rank = int(requirements["minimum_local_hidden_spatial_rank"])
        ranks = {
            scene_id: v20._validate_spatial_rank_row(ranks_raw[scene_id], scene_id, minimum_rank)
            for scene_id in EXPECTED_SCENE_IDS
        }
        centered_raw = dict(_mapping(preflight.get("centered_content"), "centered content"))
        _equal(set(centered_raw), set(EXPECTED_SCENE_IDS), "centered-content scene set")
        centered = {
            scene_id: v20._validate_centered_content_row(
                centered_raw[scene_id], scene_id, ranks[scene_id]
            )
            for scene_id in EXPECTED_SCENE_IDS
        }
    except v20.V20Update1Violation as error:
        _fail(f"V21 local-field evidence is invalid: {error}")

    raw_scene_input = dict(
        _mapping(preflight.get("raw_fp32_centered_scene_delta"), "raw scene deltas")
    )
    effective_scene_input = dict(
        _mapping(preflight.get("model_effective_scene_delta"), "effective scene deltas")
    )
    _canonical_equal(
        preflight.get("effective_cast_scene_delta"),
        effective_scene_input,
        "effective scene alias",
    )
    precision_input = dict(_mapping(preflight.get("precision_cast_audit"), "precision audits"))
    for field, evidence in (
        ("raw scene", raw_scene_input),
        ("effective scene", effective_scene_input),
        ("precision", precision_input),
    ):
        _equal(set(evidence), set(EXPECTED_SCENE_IDS), f"{field} scene set")
    raw_scene = {
        scene_id: _validate_scene_delta(raw_scene_input[scene_id], scene_id, dtype="float32")
        for scene_id in EXPECTED_SCENE_IDS
    }
    effective_scene = {
        scene_id: _validate_scene_delta(
            effective_scene_input[scene_id],
            scene_id,
            dtype="bfloat16_round_trip_float32_delta",
        )
        for scene_id in EXPECTED_SCENE_IDS
    }
    precision = {
        scene_id: _validate_precision_row(precision_input[scene_id], scene_id)
        for scene_id in EXPECTED_SCENE_IDS
    }
    for scene_id in EXPECTED_SCENE_IDS:
        cast = precision[scene_id]
        _equal(raw_scene[scene_id]["delta_sha256"], cast["raw_delta_sha256"], "raw hash")
        _equal(
            effective_scene[scene_id]["delta_sha256"],
            cast["effective_delta_sha256"],
            "effective hash",
        )
        for metric_key, cast_key in (("delta_rms", "raw_delta_rms"),):
            if not math.isclose(
                float(raw_scene[scene_id][metric_key]),
                float(cast[cast_key]),
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ):
                _fail(f"precision/raw metric mismatch for {scene_id}")
        if not math.isclose(
            float(effective_scene[scene_id]["delta_rms"]),
            float(cast["effective_delta_rms"]),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            _fail(f"precision/effective metric mismatch for {scene_id}")

    raw_pair_input = dict(
        _mapping(preflight.get("raw_fp32_centered_pair_delta"), "raw pair deltas")
    )
    effective_pair_input = dict(
        _mapping(preflight.get("model_effective_pair_delta"), "effective pair deltas")
    )
    _canonical_equal(
        preflight.get("effective_cast_pair_delta"),
        effective_pair_input,
        "effective pair alias",
    )
    _equal(set(raw_pair_input), _EXPECTED_PAIRS, "raw pair set")
    _equal(set(effective_pair_input), _EXPECTED_PAIRS, "effective pair set")
    raw_pair = {
        pair_id: _validate_pair_delta(raw_pair_input[pair_id], pair_id, field="raw")
        for pair_id in sorted(_EXPECTED_PAIRS)
    }
    effective_pair = {
        pair_id: _validate_pair_delta(effective_pair_input[pair_id], pair_id, field="effective")
        for pair_id in sorted(_EXPECTED_PAIRS)
    }
    phase = _validate_phase_evidence(preflight.get("phase_aware_pair_diagnostics"))
    for pair_id in sorted(_EXPECTED_PAIRS):
        actual = _mapping(phase[pair_id]["actual_pair"], f"phase-aware {pair_id}.actual_pair")
        for observed, expected, field in (
            (
                actual["raw_pair_rms"],
                raw_pair[pair_id]["residual_pair_difference_rms"],
                "raw pair RMS",
            ),
            (
                actual["effective_pair_rms"],
                effective_pair[pair_id]["residual_pair_difference_rms"],
                "effective pair RMS",
            ),
        ):
            if not math.isclose(float(observed), float(expected), rel_tol=1.0e-6, abs_tol=1e-12):
                _fail(f"phase-aware {pair_id} {field} disagrees with pair metrics")
    try:
        recomputed_gate = evaluate_v21_structural_gate(
            raw_scene,
            effective_scene,
            raw_pair_metrics=raw_pair,
            effective_pair_metrics=effective_pair,
            phase_pair_diagnostics=phase,
            precision_audits=precision,
            structural_state=structural,
            local_dependence=dependence,
            local_hidden_ranks=ranks,
            requirements=requirements,
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail(f"Cannot recompute V21 structural gate: {error}")
    _canonical_equal(preflight.get("structural_gate"), recomputed_gate, "structural gate")
    if recomputed_gate.get("passed") is not True:
        _fail("V21 structural gate did not pass")
    functional = _validate_functional_audit(
        preflight.get("predicted_update_functional_audit"), contract
    )
    reduction = {
        "schema_version": 1,
        "verified": True,
        "model_dtype": MODEL_DTYPE,
        "precision_algorithm": PRECISION_ALGORITHM,
        "phase_algorithm_family": PHASE_AWARE_PRECISION_PAIR_V1,
        "phase_algorithm": PHASE_ALGORITHM,
        "legacy_effective_total_norm_selectivity_diagnostic_only": True,
        "preflight_contract_sha256": EXPECTED_V21_CONTRACT_SHA256,
        "scene_ids": list(EXPECTED_SCENE_IDS),
        "pair_ids": sorted(_EXPECTED_PAIRS),
        "implementation_sources_sha256": canonical_sha256(dict(sources)),
        "local_field_structural_state_sha256": canonical_sha256(structural),
        "local_dependence_sha256": canonical_sha256(dependence),
        "local_hidden_spatial_rank_sha256": canonical_sha256(ranks),
        "centered_content_sha256": canonical_sha256(centered),
        "raw_fp32_centered_scene_delta_sha256": canonical_sha256(raw_scene),
        "model_effective_scene_delta_sha256": canonical_sha256(effective_scene),
        "precision_cast_audit_sha256": canonical_sha256(precision),
        "raw_fp32_centered_pair_delta_sha256": canonical_sha256(raw_pair),
        "model_effective_pair_delta_sha256": canonical_sha256(effective_pair),
        "phase_aware_pair_diagnostics_sha256": canonical_sha256(phase),
        "predicted_update_functional_audit_sha256": canonical_sha256(functional),
        "structural_gate_sha256": canonical_sha256(recomputed_gate),
    }
    return {**reduction, "canonical_sha256": canonical_sha256(reduction)}


def _validate_preflight(
    config: dict[str, Any],
    preflight: Mapping[str, Any],
    current_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        contract = validate_v21_config_contract(config)
    except (TypeError, ValueError, RuntimeError, V21StructuralPreflightViolation) as error:
        _fail(f"V21 config contract is invalid: {error}")
    _equal(
        contract.get("contract_sha256"),
        EXPECTED_V21_CONTRACT_SHA256,
        "V21 normalized contract hash",
    )
    if set(preflight) != _PREFLIGHT_ROOT_KEYS:
        _fail(
            "V21 preflight root keys mismatch: "
            f"missing={sorted(_PREFLIGHT_ROOT_KEYS - set(preflight))} "
            f"unknown={sorted(set(preflight) - _PREFLIGHT_ROOT_KEYS)}"
        )
    for key, expected in {
        "schema_version": 1,
        "audit_type": V21_PREFLIGHT_ROLE,
        "runtime_eligible": False,
        "uses_supervised_qa_metadata": True,
        "question_dependent_scene_processing": False,
        "model_dtype": MODEL_DTYPE,
        "live_optimizer_constructed": False,
        "live_optimizer_step_executed": False,
        "optimizer_steps": 0,
        "isolated_clone_optimizer_constructed": True,
        "isolated_clone_optimizer_steps": 1,
        "authorized": True,
        "structural_authorization": True,
    }.items():
        _equal(preflight.get(key), expected, f"preflight.{key}")
    checks = dict(_mapping(preflight.get("authorization_checks"), "authorization checks"))
    _equal(set(checks), _AUTHORIZATION_CHECKS, "authorization-check schema")
    if any(value is not True for value in checks.values()):
        _fail("Every V21 preflight authorization check must be exactly true")
    _equal(preflight.get("config_hash"), config_hash(config, length=64), "preflight config hash")
    _canonical_equal(preflight.get("contract"), contract, "preflight config contract")
    _canonical_equal(preflight.get("adamw_contract"), contract["optimizer"], "AdamW contract")
    expected_config_path = _display(_resolve(config.get("_config_path", "")))
    _equal(preflight.get("config_path"), expected_config_path, "preflight config path")
    provenance = _clean_provenance(preflight.get("source_provenance"), "preflight provenance")
    _equal(provenance, dict(current_provenance), "current/preflight provenance")
    sources = _validate_implementation_sources(preflight)

    training = _mapping(config.get("training"), "training")
    source = _safe(str(training["initialize_from"]), "V18 source checkpoint", kind="directory")
    _equal(preflight.get("source_checkpoint"), _display(source), "preflight source checkpoint")
    _exact_int(preflight.get("source_checkpoint_epoch"), 4, "preflight source epoch")
    expected_hashes = _mapping(contract.get("expected_hashes"), "contract expected hashes")
    source_artifacts = dict(_mapping(preflight.get("source_artifact_hashes"), "source artifacts"))
    _equal(set(source_artifacts), {"adapter_sha256", "metadata_sha256"}, "source artifacts")
    for field, filename, expected_field in (
        ("adapter_sha256", "adapter.safetensors", "source_adapter_sha256"),
        ("metadata_sha256", "metadata.json", "source_metadata_sha256"),
    ):
        expected = _sha256(expected_hashes.get(expected_field), expected_field)
        _equal(source_artifacts.get(field), expected, f"preflight source {field}")
        _equal(file_sha256(_safe(source / filename, f"source {filename}")), expected, field)
    source_metadata = _read_json(source / "metadata.json", "V18 source metadata")
    _exact_int(source_metadata.get("epoch"), 4, "V18 source epoch")
    source_scene = _sha256(expected_hashes.get("source_scene_state_sha256"), "source scene")
    runtime_scene = _sha256(expected_hashes.get("runtime_scene_state_sha256"), "runtime BF16 scene")
    expected_global = _sha256(
        expected_hashes.get("source_global_scene_residual_state_sha256"), "source global"
    )
    expected_lora = dict(
        _mapping(expected_hashes.get("source_lora_bank_state_sha256"), "source LoRA")
    )
    _equal(source_metadata.get("frozen_scene_state_sha256"), source_scene, "source scene metadata")
    _equal(
        source_metadata.get("global_scene_residual_state_sha256"),
        expected_global,
        "source global metadata",
    )
    _equal(source_metadata.get("lora_bank_state_sha256"), expected_lora, "source LoRA metadata")
    frozen = dict(_mapping(preflight.get("frozen_state_hashes"), "frozen hashes"))
    for key, expected in {
        "scene_state_sha256": runtime_scene,
        "global_scene_residual_state_sha256": expected_global,
        "lora_bank_state_sha256": expected_lora,
    }.items():
        _equal(frozen.get(key), expected, f"frozen {key}")
    combined_source = _sha256(frozen.get("combined_source_state_sha256"), "combined source")
    observed_source = dict(_mapping(preflight.get("source_hashes"), "source hashes"))
    expected_source = {
        **source_artifacts,
        "scene_state_sha256": runtime_scene,
        "global_scene_residual_state_sha256": expected_global,
        "lora_bank_state_sha256": expected_lora,
    }
    _equal(observed_source, expected_source, "preflight source hashes")
    for key, expected in {
        "source_metadata_global_residual_state_sha256": expected_global,
        "source_metadata_scene_state_sha256": source_scene,
        "source_metadata_lora_bank_state_sha256": expected_lora,
    }.items():
        _equal(preflight.get(key), expected, f"preflight.{key}")
    _equal(
        preflight.get("source_precision_transition"),
        {
            "checkpoint_scene_state_sha256": source_scene,
            "runtime_scene_state_sha256": runtime_scene,
            "native_boundary_checkpoint_dtype": "bfloat16",
            "native_boundary_runtime_dtype": MODEL_DTYPE,
            "conversion_performed": False,
            "state_exactly_preserved": True,
        },
        "source precision transition",
    )

    initial = _sha256(expected_hashes.get("initial_signed_x_state_sha256"), "initial signed-X")
    for key in (
        "initial_signed_x_state_sha256",
        "live_signed_x_state_sha256_before",
        "live_signed_x_state_sha256_after",
    ):
        _equal(preflight.get(key), initial, f"preflight.{key}")
    for key in ("live_signed_x_state_unchanged", "live_source_state_unchanged"):
        _equal(preflight.get(key), True, f"preflight.{key}")
    _equal(preflight.get("live_parameter_state_unchanged"), True, "live parameter state")
    for key in ("live_source_state_sha256_before", "live_source_state_sha256_after"):
        _equal(preflight.get(key), combined_source, f"preflight.{key}")
    selection = _validate_selection_and_order(preflight, expected_hashes)
    _canonical_equal(
        preflight.get("pair_objective_policy"),
        contract["pair_objective_policy"],
        "pair objective policy",
    )
    coverage = _mapping(preflight.get("pair_objective_policy_coverage"), "policy coverage")
    if coverage.get("complete") is not True or coverage.get("unlisted_pair_ids") != []:
        _fail("V21 pair-objective policy coverage is incomplete")
    try:
        zero_equivalence = v20._validate_zero_equivalence(
            preflight.get("zero_output_prefix_equivalence")
        )
    except v20.V20Update1Violation as error:
        _fail(f"V21 zero-output equivalence is invalid: {error}")
    microsteps = list(_sequence(preflight.get("microsteps"), "preflight microsteps"))
    _equal(preflight.get("microstep_losses"), microsteps, "microstep loss alias")
    _exact_int(len(microsteps), 12, "microstep count")
    ordered_units = list(_sequence(preflight.get("ordered_units"), "ordered units"))
    for index, (raw, ordered) in enumerate(zip(microsteps, ordered_units, strict=True), start=1):
        row = _mapping(raw, f"microstep {index}")
        _exact_int(row.get("microstep"), index, f"microstep {index}.microstep")
        _equal(row.get("pair_id"), ordered["pair_id"], f"microstep {index}.pair_id")
        _equal(row.get("question_key"), ordered["question_key"], f"microstep {index}.question_key")
        _finite(row.get("total_loss"), f"microstep {index}.total_loss")

    rich = _validate_rich_evidence(preflight, contract, sources)
    pair_gradient = _mapping(preflight.get("pair_gradient_audit"), "pair gradient audit")
    for key in (
        "color_total_loss_exact_zero",
        "color_gradient_exact_zero",
        "mirror_gradient_positive_finite",
        "color_losses_exact_zero",
        "color_isolated_signed_x_gradient_exact_zero",
        "mirror_signed_x_gradient_finite_nonzero",
        "only_signed_x_output_weight_has_gradient",
    ):
        _equal(pair_gradient.get(key), True, f"pair gradient {key}")
    gradient = _mapping(preflight.get("gradient"), "gradient evidence")
    _equal(gradient.get("changed_parameter_keys"), ["output_projection.weight"], "changed keys")
    _exact_int(gradient.get("ordered_microstep_count"), 12, "gradient microsteps")
    _equal(gradient.get("accumulated_finite_nonzero"), True, "accumulated gradient")
    predicted_state = _sha256(gradient.get("predicted_signed_x_state_sha256"), "predicted state")
    predicted_output = _sha256(
        gradient.get("predicted_output_projection_sha256"), "predicted output weight"
    )
    if predicted_state == initial:
        _fail("Predicted V21 signed-X state did not change")
    optimizer_manifest = dict(
        _mapping(gradient.get("optimizer_state_manifest"), "optimizer manifest")
    )
    try:
        optimizer_hash = validate_v19_adamw_state_manifest(
            optimizer_manifest, contract["optimizer"]
        )
    except V19AdamWStateViolation as error:
        _fail(f"Preflight optimizer manifest is invalid: {error}")
    _equal(
        optimizer_hash,
        _sha256(gradient.get("optimizer_state_sha256"), "optimizer state hash"),
        "optimizer manifest hash",
    )
    _sha256(gradient.get("optimizer_state_tensor_sha256"), "optimizer tensor hash")
    for key, expected in {
        "predicted_output_weight_sha256": predicted_output,
        "predicted_signed_x_scene_residual_state_sha256": predicted_state,
        "predicted_canonical_adamw_state_sha256": optimizer_hash,
        "predicted_canonical_adamw_state_manifest": optimizer_manifest,
    }.items():
        _equal(preflight.get(key), expected, f"preflight.{key}")
    predicted_first = _mapping(preflight.get("predicted_first_update"), "predicted first update")
    for key, expected in {
        "predicted_output_weight_sha256": predicted_output,
        "predicted_signed_x_scene_residual_state_sha256": predicted_state,
        "canonical_adamw_state_sha256": optimizer_hash,
        "canonical_adamw_state_manifest": optimizer_manifest,
        "finite_update": True,
    }.items():
        _equal(predicted_first.get(key), expected, f"predicted first update.{key}")
    if int(predicted_first.get("nonzero_update_count", 0)) <= 0:
        _fail("Predicted first update contains no changed elements")
    rng = _mapping(preflight.get("rng_state"), "RNG evidence")
    _equal(rng.get("all_available_domains_unchanged"), True, "RNG unchanged")
    _equal(rng.get("restored_after_mismatch"), False, "RNG restore flag")
    return {
        "source": source,
        "source_metadata": source_metadata,
        "source_artifact_hashes": source_artifacts,
        "source_provenance": provenance,
        "expected_scene_state_sha256": runtime_scene,
        "source_checkpoint_scene_state_sha256": source_scene,
        "expected_global_state_sha256": expected_global,
        "expected_lora_state_sha256": expected_lora,
        "optimizer_contract": contract["optimizer"],
        "optimizer_manifest": optimizer_manifest,
        "optimizer_hash": optimizer_hash,
        "predicted_signed_x_state_sha256": predicted_state,
        "predicted_output_projection_sha256": predicted_output,
        "zero_equivalence": zero_equivalence,
        "rich_preflight_reduction": rich,
        "implementation_sources": sources,
        **selection,
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
    try:
        return v20._load_tensor_evidence(
            path,
            metadata,
            config=config,
            expected_scene=expected_scene,
            expected_global=expected_global,
            expected_lora=expected_lora,
        )
    except v20.V20Update1Violation as error:
        _fail(f"V21 adapter evidence is invalid: {error}")


def _load_optimizer_evidence(
    path: Path,
    *,
    contract: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    expected_hash: str,
) -> dict[str, Any]:
    try:
        return v20._load_optimizer_evidence(
            path,
            contract=contract,
            expected_manifest=expected_manifest,
            expected_hash=expected_hash,
        )
    except v20.V20Update1Violation as error:
        _fail(f"V21 optimizer evidence is invalid: {error}")


def verify_update1(
    config: dict[str, Any], preflight_path: str | Path, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Verify V21 epoch one without loading Gemma, a map, QA, or oracle data."""

    current = _clean_provenance(capture_git_source_provenance(PROJECT_ROOT), "current provenance")
    preflight_file = _safe(preflight_path, "V21 preflight")
    checkpoint = _safe(checkpoint_path, "V21 epoch-one checkpoint", kind="directory")
    metadata_file = _safe(checkpoint / "metadata.json", "V21 epoch-one metadata")
    adapter_file = _safe(checkpoint / "adapter.safetensors", "V21 epoch-one adapter")
    optimizer_file = _safe(checkpoint / "optimizer.pt", "V21 epoch-one optimizer")
    preflight = _read_json(preflight_file, "V21 preflight")
    evidence = _validate_preflight(config, preflight, current)
    metadata = _read_json(metadata_file, "V21 epoch-one metadata")
    for key, expected in {
        "schema_version": 3,
        "epoch": 1,
        "optimizer_step": 1,
        "global_step": 12,
    }.items():
        _exact_int(metadata.get(key), expected, f"checkpoint.{key}")
    history = list(_sequence(metadata.get("history"), "checkpoint history"))
    _exact_int(len(history), 1, "checkpoint history length")
    history_row = _mapping(history[0], "checkpoint epoch-one history")
    _exact_int(history_row.get("epoch"), 1, "checkpoint history epoch")
    _exact_int(history_row.get("pair_batch_count"), 12, "checkpoint pair batches")
    _equal(history_row.get("pair_batch_fraction"), 1.0, "checkpoint pair fraction")
    _finite(history_row.get("train_loss"), "checkpoint train loss")
    _equal(metadata.get("train_loss"), history_row.get("train_loss"), "checkpoint train loss")
    _equal(
        metadata.get("pair_candidate_gate"),
        history_row.get("pair_candidate_gate"),
        "checkpoint teacher gate",
    )
    short_config_hash = config_hash(config)
    _equal(metadata.get("config_hash"), short_config_hash, "checkpoint.config_hash")
    if not isinstance(short_config_hash, str) or _SHORT_SHA.fullmatch(short_config_hash) is None:
        _fail("Checkpoint config hash is not the exact 12-character digest")
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
    _equal(metadata.get("source_provenance"), evidence["source_provenance"], "provenance")
    _equal(
        metadata.get("global_scene_residual"),
        global_scene_residual_settings(config).contract(),
        "global residual contract",
    )
    _exact_int(metadata.get("global_scene_residual_parameter_count"), 400_128, "global parameters")
    for key in (
        "global_scene_residual_state_sha256",
        "frozen_global_scene_residual_state_sha256",
    ):
        _equal(metadata.get(key), evidence["expected_global_state_sha256"], f"checkpoint.{key}")
    _equal(
        metadata.get("signed_x_scene_residual"),
        signed_x_scene_residual_settings(config).contract(),
        "signed-X contract",
    )
    _exact_int(
        metadata.get("signed_x_scene_residual_parameter_count"), 196_608, "signed-X parameters"
    )
    initial = signed_x_scene_residual_settings(config).expected_initial_state_sha256
    _equal(
        metadata.get("signed_x_scene_residual_initial_state_sha256"),
        initial,
        "initial signed-X hash",
    )
    if metadata.get("signed_x_scene_residual_state_sha256") == initial:
        _fail("Checkpoint V21 signed-X state did not change after update one")
    _equal(
        metadata.get("signed_x_scene_residual_zero_output_equivalence"),
        evidence["zero_equivalence"],
        "zero-output equivalence",
    )
    _equal(
        metadata.get("frozen_scene_state_sha256"),
        evidence["expected_scene_state_sha256"],
        "runtime BF16 frozen scene",
    )
    for key in ("frozen_lora_bank_state_sha256", "lora_bank_state_sha256"):
        _equal(metadata.get(key), evidence["expected_lora_state_sha256"], f"checkpoint.{key}")
    _exact_int(metadata.get("lora_trainable_parameter_count"), 0, "trainable LoRA count")
    for stale in ("v18_stage_execution", "v19_stage_execution", "v19_screen", "v20_screen"):
        if stale in metadata:
            _fail(f"V21 checkpoint improperly carries stale controller {stale}")
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
    initialization = _mapping(metadata.get("initialization_provenance"), "initialization")
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
        _equal(initialization.get(key), expected, f"initialization.{key}")
    _equal(_resolve(str(initialization.get("checkpoint"))), evidence["source"], "source path")

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
        "model_dtype": MODEL_DTYPE,
        "optimizer_deserialized": True,
        "optimizer_deserialization": {
            "weights_only": True,
            "map_location": "cpu",
            "canonical_state_validated": True,
        },
        "source_provenance": dict(current),
        "config_hash": config_hash(config, length=64),
        "preflight_contract_sha256": EXPECTED_V21_CONTRACT_SHA256,
        "preflight_sha256": file_sha256(preflight_file),
        "preflight_implementation_sources": copy.deepcopy(evidence["implementation_sources"]),
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
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/gemma4_color_mirror_signed_x_local_field_phase_aware_v21.yaml"
        ),
    )
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_update1(load_config(args.config), args.preflight, args.checkpoint)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    _atomic_json(output, report)
    print(json.dumps({"phase": "v21_update1_verified", "report": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()

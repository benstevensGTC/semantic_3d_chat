"""Fixed train-only V47-u4 dual-margin response screen for V48.

This is a report-only, no-step diagnostic.  It authenticates the exact failed
V47 update-four checkpoint, measures three isolated side-margin gradients,
normalizes every nonzero gradient component independently within the scene and
query parameter groups, and constructs a fixed three-direction by five-alpha
fresh-Adam-sign grid.  All fifteen candidates are prehashed before candidate
evaluation.  Every candidate receives the complete 25-unit teacher-forced
pair audit, the fixed 48-row broad-NLL audit, and a candidate-relative global
scene-prefix drift measurement.  Exact V47-u4 state is restored before and
after every probe; no optimizer is constructed and no candidate is persisted.

The V47 terminal SHA is supplied explicitly at invocation time so a later
terminal can pin this module and its tests without a source/seal hash cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation import v46_v45_u4_lost_side_screen as v46
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training import train_book_continuation_v47 as v47
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    load_adapter_checkpoint,
    module_collection_state_sha256,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    build_exact_question_pair_units,
)
from semantic_3d_chat.training.train_block_cross_v35 import current_scene_tokens
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
)
from semantic_3d_chat.training.train_joint_block_cross_v36 import training_broad_nll
from semantic_3d_chat.training.train_joint_pair_v30 import require_approved_v29_source
from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
    _PARAMETER_NAMES,
    _PARAMETER_SHAPES,
    assert_v44_trainable_surface,
    freeze_for_v44,
    frozen_v44_state_sha256,
    source_prefix_trust_penalty,
    v44_contract,
)
from semantic_3d_chat.training.train_projected_gradient_v41 import (
    _prefix_replay_attestation,
    cache_v41_train_scenes,
    load_v41_bundle,
    priority_side_deficit,
    training_pair_gate_diagnostics,
    v41_loader_config,
    validate_per_unit_nll_diagnostics,
)
from semantic_3d_chat.training.train_retention_repair_v45 import (
    _BROAD_NLL_MAXIMUM,
    _PROTECTED_REPORT,
    _PROTECTED_REPORT_SHA256,
    _V41_FULL_SHA256,
    _preflight_forbidden_roots,
    _training_forbidden_roots,
    _unit_index,
    _v41_source_tensors,
    build_v45_schedule,
    load_v35_train_qa_records,
    v31_contract,
    v45_retention_diagnostics,
)
from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
    validate_v37_training_cache_boundary,
)

DEFAULT_CONFIG = v47.DEFAULT_CONFIG
DEFAULT_TERMINAL = Path("reports/gemma4/metrics/v47_book_continuation_terminal_gate.json")
DEFAULT_SOURCE = Path("data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query/update_004")
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v48_v47_u4_dual_margin_no_step_diagnostic.json")
V48_SCRIPT = Path("src/semantic_3d_chat/evaluation/v48_v47_u4_dual_margin_screen.py")
V48_TEST = Path("tests/test_v48_v47_u4_dual_margin_screen.py")

_AUTHORIZATION_ID = "v48_v47_u4_dual_margin_no_step_diagnostic"
_DIRECTION_IDS = ("dual_query_sign", "dual_both_sign", "guarded_both_sign")
_ALPHA_GRID = (0.125, 0.25, 0.5, 1.0, 2.0)
_SCENE_LR = 1.0e-5
_QUERY_LR = 8.0e-6
_PREFIX_TRUST_SCALE = 0.05
_PREFIX_TRUST_RMS_MAXIMUM = 0.002
_EXACT_CANDIDATE_COUNT = 15
_TRAIN_SCENES = tuple(
    [*(f"scene_{index:06d}" for index in range(11, 19))]
    + [*(f"scene_{index:06d}" for index in range(31, 39))]
)
_GRADIENT_SPECS = (
    ("g_book", "pair_000015", "cfq_163eb92339ad35a5", 0),
    ("g_mirror", "pair_000016", "cfq_699675ceeaf65406", 1),
    ("g5_guard", "pair_000006", "cfq_5c84a2c27d2be251", 0),
)
_DIRECTION_COMPONENTS = {
    "dual_query_sign": ("g_book", "g_mirror"),
    "dual_both_sign": ("g_book", "g_mirror"),
    "guarded_both_sign": ("g_book", "g_mirror", "g5_guard"),
}
_SOURCE_FILES = {
    "adapter.safetensors": ("8f903f5d1ba93d37ccd6204e3b58c9a5529ff9ee2b74edca0787ecb5a2c62c66"),
    TRAINING_METADATA_FILENAME: (
        "c6affe7f60c094580e2ea5f5d1330f475bf359e0a3a58bfc3bf3b3ada1de0be1"
    ),
    "optimizer.pt": ("fe66be9cae13951fbfc217e0c512e43366c347181457c9e551230a9d6001db80"),
    RUNTIME_METADATA_FILENAME: ("4e3a1af91642c9f2adb0b3e43997455a1aea31f86bf45618459d6005a68d4bbf"),
}
_READABLE_SOURCE_FILES = (
    "adapter.safetensors",
    TRAINING_METADATA_FILENAME,
    RUNTIME_METADATA_FILENAME,
)
_SOURCE_FULL_SHA256 = "adfc0400d1a3bb49b278cd3012ab571d01465f2380881f986c085a25474276e5"
_SOURCE_AUTHORIZED_SHA256 = "a23de4988774a966c0d7aac378ede5d15a3fa1d96093c5039f181a62b0bb09b0"
_SOURCE_FROZEN_SHA256 = "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"
_SOURCE_PRIORITY_DEFICIT = 30.386213302612305
_SOURCE_BROAD_NLL = 2.9172145972649255
_ORIGINAL_V41_PRIORITY_DEFICIT = 31.113729119300842
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOLERANCE = 1.0e-6


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _numeric_close(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=_TOLERANCE)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _numeric_close(left[key], right[key]) for key in left
        )
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(_numeric_close(a, b) for a, b in zip(left, right))
    return left == right


def _expected_gradient_specs() -> list[dict[str, Any]]:
    return [
        {
            "gradient_id": gradient_id,
            "pair_id": pair_id,
            "question_key": question_key,
            "side_index": side_index,
            "loss": "negative_selected_side_margin",
        }
        for gradient_id, pair_id, question_key, side_index in _GRADIENT_SPECS
    ]


def _validate_authorization(
    report: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, bool]:
    invocation = _mapping(authorization.get("invocation_contract"), "V48 invocation contract")
    integrity = _mapping(
        authorization.get("implementation_integrity"), "V48 implementation integrity"
    )
    source = _mapping(authorization.get("source"), "V48 source")
    measurements = _mapping(authorization.get("measurements"), "V48 measurements")
    grid = _mapping(authorization.get("candidate_grid"), "V48 candidate grid")
    scope = _mapping(authorization.get("scope"), "V48 scope")
    checks = {
        "artifact": report.get("artifact") == "v47_book_continuation_terminal_gate",
        "passed": report.get("passed") is True,
        "v47_failed": report.get("v47_final_train_only_gate_passed") is False,
        "successor": report.get("only_exact_successor_authorized") == _AUTHORIZATION_ID,
        "id": authorization.get("authorization_id") == _AUTHORIZATION_ID,
        "authorized": authorization.get("authorized") is True,
        "action": authorization.get("only_exact_action")
        == "one_bounded_read_only_v48_train_checkpoint_dual_margin_diagnostic",
        "script": authorization.get("authorized_script") == str(V48_SCRIPT),
        "test": authorization.get("authorized_test") == str(V48_TEST),
        "report": authorization.get("authorized_report") == str(DEFAULT_OUTPUT),
        "config": authorization.get("authorized_config") == str(DEFAULT_CONFIG),
        "explicit_cli": authorization.get("explicit_terminal_sha256_cli_required") is True,
        "terminal_path": invocation.get("terminal_path") == str(DEFAULT_TERMINAL),
        "cli_name": invocation.get("required_cli_argument") == "--expected-v47-terminal-sha256",
        "no_embedded_sha": invocation.get("v48_must_not_embed_terminal_sha256") is True,
        "script_hash": integrity.get("script_sha256") == _sha256(_resolve(V48_SCRIPT)),
        "test_hash": integrity.get("test_sha256") == _sha256(_resolve(V48_TEST)),
        "config_hash": integrity.get("config_sha256") == v47._CONFIG_FILE_SHA256,
        "source_checkpoint": source.get("checkpoint") == str(DEFAULT_SOURCE),
        "source_files": dict(_mapping(source.get("file_sha256"), "source files")) == _SOURCE_FILES,
        "source_full": source.get("full_tensor_state_sha256") == _SOURCE_FULL_SHA256,
        "source_authorized": source.get("authorized_surface_state_sha256")
        == _SOURCE_AUTHORIZED_SHA256,
        "source_frozen": source.get("frozen_state_sha256") == _SOURCE_FROZEN_SHA256,
        "optimizer_forbidden": source.get("optimizer_file_open_authorized") is False,
        "gradient_specs": list(
            _sequence(measurements.get("isolated_side_gradient_specs"), "gradient specs")
        )
        == _expected_gradient_specs(),
        "group_normalization": measurements.get("normalize_each_nonzero_component")
        == "unit_l2_within_each_scene_or_query_group_before_combination",
        "gradient_geometry": measurements.get("report_raw_norms_and_pairwise_cosines_by_group")
        is True,
        "directions": list(_sequence(grid.get("direction_ids"), "directions"))
        == list(_DIRECTION_IDS),
        "alphas": list(_sequence(grid.get("alpha_grid"), "alphas")) == list(_ALPHA_GRID),
        "formula": grid.get("candidate_formula")
        == "float32_P0-alpha*lr_group*sign(normalized_component_sum)",
        "scene_lr": grid.get("scene_readout_learning_rate") == _SCENE_LR,
        "query_lr": grid.get("query_learning_rate") == _QUERY_LR,
        "candidate_count": grid.get("exact_candidate_count") == _EXACT_CANDIDATE_COUNT,
        "full_pair": grid.get("full_25_unit_teacher_metrics_per_candidate") is True,
        "full_broad": grid.get("full_fixed_48_row_broad_nll_per_candidate") is True,
        "full_trust": grid.get("candidate_relative_prefix_trust_per_candidate") is True,
        "restore": grid.get("exact_source_restoration_before_and_after_every_probe") is True,
        "prehash": grid.get("prehash_all_candidates_before_candidate_forward") is True,
        "train_only": scope.get("train_only") is True,
        "report_only": scope.get("report_only_output") is True,
        "no_candidate": scope.get("candidate_checkpoint_write_authorized") is False,
        "no_optimizer": scope.get("optimizer_construction_or_step_authorized") is False,
        "no_selection": scope.get("candidate_selection_authorized") is False,
        "no_greedy": scope.get("greedy_generation_authorized") is False,
        "no_validation": scope.get("validation_access_authorized") is False,
        "no_oracle": scope.get("oracle_access_authorized") is False,
        "no_final": scope.get("final_test_access_authorized") is False,
        "no_selector": scope.get("selector_execution_authorized") is False,
        "no_promotion": scope.get("runtime_promotion_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"V48 V47-terminal authorization changed: {checks}")
    return checks


def require_terminal(expected_sha256: str) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V48 expected V47 terminal SHA256 must be lowercase hex")
    path = _resolve(DEFAULT_TERMINAL)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("V48 V47 terminal is unavailable or unsafe")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError("V48 V47 terminal differs from explicit invocation SHA256")
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "V47 terminal")
    authorization = _mapping(report.get("conditional_successor_authorization"), "V48 authorization")
    checks = _validate_authorization(report, authorization)
    return {
        "path": str(DEFAULT_TERMINAL),
        "sha256": observed,
        "authorization_id": _AUTHORIZATION_ID,
        "authorization": dict(authorization),
        "checks": checks,
    }


def _source_evidence() -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Read only authenticated V47-u4 files; never open optimizer state."""

    source = _resolve(DEFAULT_SOURCE)
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError("V48 exact V47 update-four source is unavailable")
    inventory = sorted(path.name for path in source.iterdir())
    if inventory != sorted(_SOURCE_FILES):
        raise ValueError("V48 source checkpoint inventory changed")
    observed_files: dict[str, str] = {}
    for name in _READABLE_SOURCE_FILES:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"V48 readable source file is unavailable: {name}")
        observed_files[name] = _sha256(path)
        if observed_files[name] != _SOURCE_FILES[name]:
            raise ValueError(f"V48 readable source file changed: {name}")
    tensors = load_file(source / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(tensors) != _SOURCE_FULL_SHA256:
        raise ValueError("V48 source full tensor state changed")
    authorized = {name: tensors[name] for name in _PARAMETER_NAMES}
    frozen = {name: value for name, value in tensors.items() if name not in authorized}
    if tensor_state_sha256(authorized) != _SOURCE_AUTHORIZED_SHA256:
        raise ValueError("V48 source authorized tensor surface changed")
    if tensor_state_sha256(frozen) != _SOURCE_FROZEN_SHA256:
        raise ValueError("V48 source frozen tensor state changed")
    metadata = _mapping(
        json.loads((source / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8")),
        "V48 source metadata",
    )
    runtime = _mapping(
        json.loads((source / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8")),
        "V48 source runtime metadata",
    )
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V48 source runtime metadata is not exact sanitization")
    stage = _mapping(metadata.get("v47_book_continuation"), "V47 source stage")
    history = _sequence(metadata.get("history"), "V47 source history")
    final = _mapping(history[-1], "V47 source final history")
    gate = _mapping(final.get("update4_final_train_only_gate"), "V47 final gate")
    if (
        metadata.get("optimizer_step") != 4
        or stage.get("optimizer_step") != 4
        or len(history) != 5
        or final.get("optimizer_update") != 4
        or final.get("saved_checkpoint") is not True
        or gate.get("passed") is not False
        or stage.get("update4_final_train_only_gate") != gate
        or stage.get("validation_qa_loaded") is not False
        or stage.get("oracle_environment_files_loaded") is not False
        or stage.get("selector_execution_authorized") is not False
        or stage.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V48 source is not exact failed V47 update four")
    return (
        dict(tensors),
        dict(metadata),
        {
            "checkpoint": str(DEFAULT_SOURCE),
            "directory_inventory": inventory,
            "readable_file_sha256": observed_files,
            "optimizer_file_sha256_provenance": _SOURCE_FILES["optimizer.pt"],
            "optimizer_file_opened": False,
            "optimizer_state_deserialized": False,
            "optimizer_state_loaded": False,
            "full_tensor_state_sha256": _SOURCE_FULL_SHA256,
            "authorized_surface_state_sha256": _SOURCE_AUTHORIZED_SHA256,
            "frozen_state_sha256": _SOURCE_FROZEN_SHA256,
            "v47_final_train_only_gate_passed": False,
        },
    )


def _validate_surface_tensors(values: Mapping[str, torch.Tensor], *, field: str) -> None:
    if tuple(values) != _PARAMETER_NAMES:
        raise ValueError(f"V48 {field} names or order changed")
    for name, shape in zip(_PARAMETER_NAMES, _PARAMETER_SHAPES):
        value = values[name]
        if value.dtype != torch.float32 or tuple(value.shape) != shape:
            raise ValueError(f"V48 {field} tensor changed: {name}")
        if value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError(f"V48 {field} must be finite float32 CPU: {name}")


def _group_names(group: str) -> tuple[str, ...]:
    if group == "scene_readout":
        return (_PARAMETER_NAMES[0],)
    if group == "query":
        return _PARAMETER_NAMES[1:]
    raise ValueError(f"Unknown V48 parameter group: {group}")


def _group_l2(values: Mapping[str, torch.Tensor], group: str) -> float:
    names = _group_names(group)
    squared = sum(float(values[name].detach().double().square().sum()) for name in names)
    result = math.sqrt(squared)
    if not math.isfinite(result):
        raise RuntimeError("V48 gradient group norm is nonfinite")
    return result


def normalize_gradient_components_by_group(
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, float]]]:
    """Unit-normalize each nonzero gradient separately in each parameter group."""

    if tuple(gradients) != tuple(value[0] for value in _GRADIENT_SPECS):
        raise ValueError("V48 gradient component IDs or order changed")
    normalized: dict[str, dict[str, torch.Tensor]] = {}
    norms: dict[str, dict[str, float]] = {}
    for gradient_id, values in gradients.items():
        _validate_surface_tensors(values, field=f"{gradient_id} gradient")
        normalized[gradient_id] = {}
        norms[gradient_id] = {}
        for group in ("scene_readout", "query"):
            norm = _group_l2(values, group)
            norms[gradient_id][group] = norm
            for name in _group_names(group):
                normalized[gradient_id][name] = (
                    values[name].clone() if norm == 0.0 else values[name] / norm
                )
        _validate_surface_tensors(normalized[gradient_id], field=f"normalized {gradient_id}")
    return normalized, norms


def build_normalized_directions(
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    normalized, norms = normalize_gradient_components_by_group(gradients)
    directions: dict[str, dict[str, torch.Tensor]] = {}
    for direction_id in _DIRECTION_IDS:
        components = _DIRECTION_COMPONENTS[direction_id]
        values = {
            name: torch.zeros_like(normalized[components[0]][name]) for name in _PARAMETER_NAMES
        }
        for gradient_id in components:
            for name in _PARAMETER_NAMES:
                values[name].add_(normalized[gradient_id][name])
        if direction_id == "dual_query_sign":
            values[_PARAMETER_NAMES[0]].zero_()
        _validate_surface_tensors(values, field=f"{direction_id} direction")
        active_groups = (
            ("query",) if direction_id == "dual_query_sign" else ("scene_readout", "query")
        )
        if any(_group_l2(values, group) == 0.0 for group in active_groups):
            raise RuntimeError(f"V48 {direction_id} has an empty active group")
        directions[direction_id] = values
    audit = {
        "normalization": ("each_nonzero_component_unit_l2_within_each_scene_or_query_group"),
        "raw_group_l2_norms": norms,
        "direction_components": {key: list(value) for key, value in _DIRECTION_COMPONENTS.items()},
        "inactive_scene_group_exact_zero_for_dual_query_sign": torch.count_nonzero(
            directions["dual_query_sign"][_PARAMETER_NAMES[0]]
        ).item()
        == 0,
    }
    return directions, audit


def candidate_from_normalized_direction(
    source: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    *,
    direction_id: str,
    alpha: float,
) -> dict[str, torch.Tensor]:
    if direction_id not in _DIRECTION_IDS:
        raise ValueError("V48 direction is outside the fixed grid")
    if alpha not in _ALPHA_GRID:
        raise ValueError("V48 alpha is outside the fixed grid")
    _validate_surface_tensors(source, field="source")
    _validate_surface_tensors(direction, field="direction")
    active = (
        set(_PARAMETER_NAMES[1:]) if direction_id == "dual_query_sign" else set(_PARAMETER_NAMES)
    )
    result = {name: value.clone() for name, value in source.items()}
    for name in _PARAMETER_NAMES:
        if name not in active:
            continue
        learning_rate = _SCENE_LR if name == _PARAMETER_NAMES[0] else _QUERY_LR
        result[name].add_(torch.sign(direction[name]), alpha=-float(alpha * learning_rate))
    _validate_surface_tensors(result, field="candidate")
    for name in set(_PARAMETER_NAMES) - active:
        if not torch.equal(result[name], source[name]):
            raise RuntimeError("V48 candidate changed an inactive tensor group")
    return result


def build_candidate_inventory(
    source_surface: Mapping[str, torch.Tensor],
    source_full: Mapping[str, torch.Tensor],
    directions: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], str]:
    if tuple(directions) != _DIRECTION_IDS:
        raise ValueError("V48 normalized direction IDs or order changed")
    rows: list[dict[str, Any]] = []
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    for direction_id in _DIRECTION_IDS:
        for alpha in _ALPHA_GRID:
            candidate_id = f"{direction_id}_alpha_{str(alpha).replace('.', 'p')}"
            candidate = candidate_from_normalized_direction(
                source_surface,
                directions[direction_id],
                direction_id=direction_id,
                alpha=alpha,
            )
            full = dict(source_full)
            full.update(candidate)
            row = {
                "candidate_id": candidate_id,
                "direction_id": direction_id,
                "alpha": alpha,
                "authorized_surface_state_sha256": tensor_state_sha256(candidate),
                "full_tensor_state_sha256": tensor_state_sha256(full),
            }
            rows.append(row)
            tensors[candidate_id] = candidate
    expected = [(direction_id, alpha) for direction_id in _DIRECTION_IDS for alpha in _ALPHA_GRID]
    if (
        len(rows) != _EXACT_CANDIDATE_COUNT
        or len(tensors) != _EXACT_CANDIDATE_COUNT
        or [(row["direction_id"], row["alpha"]) for row in rows] != expected
    ):
        raise RuntimeError("V48 candidate inventory is not the fixed 3x5 grid")
    return rows, tensors, _canonical_sha256(rows)


def _bundle_state_attestation(
    bundle: Any,
    named: Mapping[str, torch.nn.Parameter],
    *,
    expected_authorized_sha256: str,
    expected_full_sha256: str,
) -> dict[str, Any]:
    result = {
        "authorized_surface_state_sha256": tensor_state_sha256(
            {name: value.detach().cpu() for name, value in named.items()}
        ),
        "full_tensor_state_sha256": module_collection_state_sha256(bundle.checkpoint_modules),
        "frozen_state_sha256": frozen_v44_state_sha256(bundle),
        "all_parameter_gradients_absent": not any(
            parameter.grad is not None for parameter in bundle.language.model.parameters()
        )
        and not any(
            parameter.grad is not None
            for module in bundle.checkpoint_modules.values()
            for parameter in module.parameters()
        ),
    }
    result["passed"] = bool(
        result["authorized_surface_state_sha256"] == expected_authorized_sha256
        and result["full_tensor_state_sha256"] == expected_full_sha256
        and result["frozen_state_sha256"] == _SOURCE_FROZEN_SHA256
        and result["all_parameter_gradients_absent"] is True
    )
    return result


def _restore_source(
    bundle: Any,
    named: Mapping[str, torch.nn.Parameter],
    source_surface: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(source_surface[name].to(device=parameter.device, dtype=parameter.dtype))
            parameter.grad = None
            parameter.requires_grad_(False)
    result = _bundle_state_attestation(
        bundle,
        named,
        expected_authorized_sha256=_SOURCE_AUTHORIZED_SHA256,
        expected_full_sha256=_SOURCE_FULL_SHA256,
    )
    if result["passed"] is not True:
        raise RuntimeError("V48 failed to restore exact V47 update four")
    return result


def _copy_candidate(
    named: Mapping[str, torch.nn.Parameter], candidate: Mapping[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(candidate[name].to(device=parameter.device, dtype=parameter.dtype))
            parameter.grad = None
            parameter.requires_grad_(False)


def _flatten_group(values: Mapping[str, torch.Tensor], group: str) -> torch.Tensor:
    return torch.cat([values[name].reshape(-1).double() for name in _group_names(group)])


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if not torch.isfinite(denominator) or float(denominator) == 0.0:
        raise RuntimeError("V48 gradient cosine denominator is invalid")
    return float(torch.dot(left, right) / denominator)


def gradient_geometry(
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    normalized, norms = normalize_gradient_components_by_group(gradients)
    result: dict[str, Any] = {
        "raw_group_l2_norms": norms,
        "groups": {},
        "normalization": ("each_nonzero_component_unit_l2_within_each_scene_or_query_group"),
    }
    ids = tuple(gradients)
    for group in ("scene_readout", "query"):
        raw_vectors = {key: _flatten_group(gradients[key], group) for key in ids}
        normalized_vectors = {key: _flatten_group(normalized[key], group) for key in ids}
        pairwise = {}
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                pairwise[f"{left}__{right}"] = {
                    "raw_cosine": _cosine(raw_vectors[left], raw_vectors[right]),
                    "normalized_cosine": _cosine(
                        normalized_vectors[left], normalized_vectors[right]
                    ),
                }
        result["groups"][group] = {"pairwise_cosines": pairwise}
    return result


def _gradient_diagnostics(
    *,
    units_by_key: Mapping[str, CounterfactualPairUnit],
    caches: Mapping[str, Any],
    block_core: torch.nn.Module,
    bundle: Any,
    named: Mapping[str, torch.nn.Parameter],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    gradients: dict[str, dict[str, torch.Tensor]] = {}
    rows: list[dict[str, Any]] = []
    for gradient_id, pair_id, question_key, side_index in _GRADIENT_SPECS:
        gradient, row = v46._selected_side_gradient(
            unit=units_by_key[question_key],
            expected_pair_id=pair_id,
            expected_question_key=question_key,
            side_index=side_index,
            caches=caches,
            block_core=block_core,
            bundle=bundle,
            named=named,
        )
        gradients[gradient_id] = gradient
        rows.append({"gradient_id": gradient_id, **row})
    geometry = gradient_geometry(gradients)
    source_state = _bundle_state_attestation(
        bundle,
        named,
        expected_authorized_sha256=_SOURCE_AUTHORIZED_SHA256,
        expected_full_sha256=_SOURCE_FULL_SHA256,
    )
    if source_state["passed"] is not True:
        raise RuntimeError("V48 gradient measurement changed exact source state")
    return gradients, {
        "specifications": rows,
        "geometry": geometry,
        "source_state_unchanged": True,
        "source_state_after_gradient_measurement": source_state,
        "optimizer_constructed_or_loaded": False,
    }


def _focus_rows(pair_metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    keys = {
        "cfq_163eb92339ad35a5",
        "cfq_699675ceeaf65406",
        "cfq_5c84a2c27d2be251",
    }
    rows = _sequence(pair_metrics.get("units"), "V48 pair units")
    return {
        str(row["question_key"]): dict(row)
        for value in rows
        if (row := _mapping(value, "V48 pair unit")).get("question_key") in keys
    }


def candidate_threshold_diagnostic(
    pair_metrics: Mapping[str, Any], broad_nll: float, prefix_trust_rms: float
) -> dict[str, Any]:
    families = _mapping(pair_metrics.get("complete_units_by_family"), "families")
    cross_families = _mapping(
        pair_metrics.get("cross_prefix_complete_units_by_family"), "cross families"
    )
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    retention = v45_retention_diagnostics(pair_metrics)
    checks = {
        "complete_units_at_least_10": int(pair_metrics["complete_units"]) >= 10,
        "positive_sides_at_least_35": int(pair_metrics["positive_sides"]) >= 35,
        "cross_prefix_complete_units_at_least_17": int(pair_metrics["cross_prefix_complete_units"])
        >= 17,
        "complete_physical_pair_coverage_at_least_5": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= 5,
        "mirror_complete_units_at_least_2": int(families.get("mirror_lr", 0)) >= 2,
        "book_complete_units_at_least_1": int(families.get("book_support", 0)) >= 1,
        "book_cross_prefix_complete_units_at_least_1": int(cross_families.get("book_support", 0))
        >= 1,
        "priority_deficit_improvement_at_least_0_5_vs_original_v41_u0": (
            _ORIGINAL_V41_PRIORITY_DEFICIT - deficit >= 0.5
        ),
        "broad_nll_at_most_v45_maximum": broad_nll <= _BROAD_NLL_MAXIMUM,
        "both_lost_sides_strictly_positive": retention["both_lost_sides_strictly_positive"],
        "candidate_relative_prefix_trust_rms_at_most_0_002": prefix_trust_rms
        <= _PREFIX_TRUST_RMS_MAXIMUM,
    }
    return {
        "checks": checks,
        "all_numeric_thresholds_met": all(checks.values()),
        "diagnostic_only_no_candidate_authorization": True,
        "priority_side_deficit": deficit,
        "priority_deficit_improvement_vs_original_v41_u0": (
            _ORIGINAL_V41_PRIORITY_DEFICIT - deficit
        ),
        "broad_nll": broad_nll,
        "candidate_relative_prefix_trust_rms": prefix_trust_rms,
        "retention_diagnostics": retention,
    }


def _preflight(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    expected_v47_terminal_sha256: str,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    if config_file != _resolve(DEFAULT_CONFIG) or _sha256(config_file) != v47._CONFIG_FILE_SHA256:
        raise ValueError("V48 config path or bytes changed")
    terminal = require_terminal(expected_v47_terminal_sha256)
    protected = _resolve(_PROTECTED_REPORT)
    if protected.is_symlink() or _sha256(protected) != _PROTECTED_REPORT_SHA256:
        raise ValueError("V48 protected selection report changed")
    config = load_config(config_file)
    audit = FileAccessAudit(
        _preflight_forbidden_roots(config),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        tensors, _metadata, source = _source_evidence()
        loader = v41_loader_config(config)
        assert_deferred_final_scenes_absent(loader)
        records, qa_audit = load_v35_train_qa_records(loader)
        units = build_exact_question_pair_units(records)
        _schedule, schedule_audit, broad_records = build_v45_schedule(records, units, config=config)
        if len(records) != 384 or len(units) != 25 or len(broad_records) != 48:
            raise RuntimeError("V48 preflight train-only inventory changed")
    audit.assert_clean()
    return {
        "schema_version": 1,
        "artifact": "v48_v47_u4_dual_margin_screen_preflight",
        "passed": True,
        "terminal": terminal,
        "source": source,
        "source_adapter_tensor_count": len(tensors),
        "source_optimizer_opened": False,
        "train_question_count": len(records),
        "changed_pair_unit_count": len(units),
        "broad_nll_row_count": len(broad_records),
        "candidate_grid": {
            "direction_ids": list(_DIRECTION_IDS),
            "alpha_grid": list(_ALPHA_GRID),
            "candidate_count": _EXACT_CANDIDATE_COUNT,
            "adaptive_selection": False,
        },
        "qa_audit": qa_audit,
        "schedule_audit": schedule_audit,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "optimizer_constructed_or_loaded": False,
        "candidate_checkpoint_written": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_executed": False,
        "runtime_promotion_executed": False,
        "forbidden_file_accesses": audit.forbidden_accesses(),
    }


def run_screen(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    expected_v47_terminal_sha256: str,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    if config_file != _resolve(DEFAULT_CONFIG) or _sha256(config_file) != v47._CONFIG_FILE_SHA256:
        raise ValueError("V48 config path or bytes changed")
    terminal = require_terminal(expected_v47_terminal_sha256)
    protected = _resolve(_PROTECTED_REPORT)
    if protected.is_symlink() or not protected.is_file():
        raise ValueError("V48 protected selection report is unavailable or unsafe")
    protected_before = _sha256(protected)
    if protected_before != _PROTECTED_REPORT_SHA256:
        raise ValueError("V48 protected selection report changed before screen")
    config = load_config(config_file)
    audit = FileAccessAudit(
        _training_forbidden_roots(config),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        source_full, source_metadata, source_audit = _source_evidence()
        source_surface = {
            name: source_full[name].detach().float().cpu().clone() for name in _PARAMETER_NAMES
        }
        _validate_surface_tensors(source_surface, field="source surface")
        loader = v41_loader_config(config)
        assert_deferred_final_scenes_absent(loader)
        records, qa_audit = load_v35_train_qa_records(loader)
        units = build_exact_question_pair_units(records)
        units_by_key = _unit_index(units)
        _schedule, schedule_audit, broad_records = build_v45_schedule(records, units, config=config)
        if len(records) != 384 or len(units) != 25 or len(broad_records) != 48:
            raise RuntimeError("V48 train-only diagnostic inventory changed")

        construction = v44_contract(config)
        v41_tensors, v41_metadata = _v41_source_tensors(construction)
        if tensor_state_sha256(v41_tensors) != _V41_FULL_SHA256:
            raise RuntimeError("V48 V41 construction tensor state changed")
        approved = require_approved_v29_source(loader)
        bundle, block_core, loaded_v41, loader_transition = load_v41_bundle(
            config, approved, construction.source_checkpoint, v41_tensors
        )
        if loaded_v41 != v41_metadata:
            raise RuntimeError("V48 V41 construction metadata changed")
        loaded_u4 = load_adapter_checkpoint(
            _resolve(DEFAULT_SOURCE),
            bundle.checkpoint_modules,
            device="cpu",
            metadata_filename=TRAINING_METADATA_FILENAME,
        )
        if loaded_u4 != source_metadata:
            raise RuntimeError("V48 strict V47 update-four overlay metadata changed")
        named = freeze_for_v44(bundle, block_core)
        surface = assert_v44_trainable_surface(bundle, block_core)
        if (
            module_collection_state_sha256(bundle.checkpoint_modules) != _SOURCE_FULL_SHA256
            or tensor_state_sha256({name: value.detach().cpu() for name, value in named.items()})
            != _SOURCE_AUTHORIZED_SHA256
            or frozen_v44_state_sha256(bundle) != _SOURCE_FROZEN_SHA256
        ):
            raise RuntimeError("V48 live source differs from exact V47 update four")

        split = v31_contract(loader)
        if tuple(split.train_scene_ids) != _TRAIN_SCENES:
            raise RuntimeError("V48 exact train scene split changed")
        manifest_ids = (*split.train_scene_ids, *split.validation_scene_ids)
        caches, cache_audit = cache_v41_train_scenes(
            config=loader,
            bundle=bundle,
            source_metadata=source_metadata,
            scene_ids=split.train_scene_ids,
            manifest_scene_ids=manifest_ids,
        )
        cache_audit.update(
            {
                "scene_scope": "training_only",
                "authenticated_manifest_scene_count": len(manifest_ids),
                "authenticated_manifest_train_subset_count": len(split.train_scene_ids),
                "validation_scene_ids_loaded": [],
                "validation_environment_maps_loaded": False,
                "deferred_final_scene_ids_loaded": [],
            }
        )
        cache_boundary = validate_v37_training_cache_boundary(
            cache_audit=cache_audit,
            caches=caches,
            config=loader,
            train_scene_ids=split.train_scene_ids,
            validation_scene_ids=split.validation_scene_ids,
        )
        prefix_evidence = _prefix_replay_attestation(
            caches=caches,
            block_cross_residual=block_core,
            bundle=bundle,
            expected_scene_ids=split.train_scene_ids,
        )
        if len(caches) != 16 or tuple(sorted(caches)) != tuple(sorted(_TRAIN_SCENES)):
            raise RuntimeError("V48 did not cache exactly all 16 training scenes")

        settings = v47.v47_settings(config)
        source_pair, source_nll = training_pair_gate_diagnostics(
            units=units,
            caches=caches,
            block_cross_residual=block_core,
            bundle=bundle,
            settings=settings,
        )
        source_broad = training_broad_nll(
            records=broad_records,
            caches=caches,
            block_cross_residual=block_core,
            bundle=bundle,
        )
        validate_per_unit_nll_diagnostics(source_nll, source_pair)
        history = _sequence(source_metadata.get("history"), "V48 source history")
        final_history = _mapping(history[-1], "V48 source final history")
        source_replay = {
            "pair_metrics": _numeric_close(source_pair, final_history.get("pair_metrics")),
            "per_unit_nll": _numeric_close(
                source_nll, final_history.get("per_unit_nll_diagnostics")
            ),
            "broad_nll": math.isclose(
                source_broad,
                float(final_history.get("broad_diagnostic_nll")),
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            )
            and math.isclose(
                source_broad,
                _SOURCE_BROAD_NLL,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            ),
            "priority_deficit": math.isclose(
                float(priority_side_deficit(source_pair)["combined"]),
                _SOURCE_PRIORITY_DEFICIT,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            ),
        }
        source_replay["passed"] = all(source_replay.values())
        if source_replay["passed"] is not True:
            raise RuntimeError("V48 source diagnostic replay changed")

        with torch.inference_mode():
            source_scene_tokens = {
                scene_id: current_scene_tokens(
                    caches[scene_id], block_core, device=bundle.language.device
                )
                .detach()
                .cpu()
                .clone()
                for scene_id in sorted(caches)
            }
        source_prefix_hashes = {
            scene_id: tensor_state_sha256({"scene_tokens": value})
            for scene_id, value in source_scene_tokens.items()
        }
        gradients, gradient_audit = _gradient_diagnostics(
            units_by_key=units_by_key,
            caches=caches,
            block_core=block_core,
            bundle=bundle,
            named=named,
        )
        directions, direction_audit = build_normalized_directions(gradients)
        candidate_inventory, candidate_tensors, inventory_hash = build_candidate_inventory(
            source_surface, source_full, directions
        )
        if len(candidate_inventory) != _EXACT_CANDIDATE_COUNT:
            raise RuntimeError("V48 did not prehash exactly 15 candidates")

        results: list[dict[str, Any]] = []
        restorations: list[dict[str, Any]] = []
        for specification in candidate_inventory:
            candidate_id = str(specification["candidate_id"])
            before = _restore_source(bundle, named, source_surface)
            restorations.append({"candidate_id": candidate_id, "phase": "before", **before})
            try:
                _copy_candidate(named, candidate_tensors[candidate_id])
                candidate_state = _bundle_state_attestation(
                    bundle,
                    named,
                    expected_authorized_sha256=str(
                        specification["authorized_surface_state_sha256"]
                    ),
                    expected_full_sha256=str(specification["full_tensor_state_sha256"]),
                )
                if candidate_state["passed"] is not True:
                    raise RuntimeError("V48 live candidate differs from prehash")
                with torch.inference_mode():
                    _trust_penalty, trust_rms_tensor = source_prefix_trust_penalty(
                        caches=caches,
                        references=source_scene_tokens,
                        block_core=block_core,
                        device=bundle.language.device,
                        scale=_PREFIX_TRUST_SCALE,
                    )
                prefix_trust_rms = float(trust_rms_tensor.detach().cpu())
                pair_metrics, per_unit_nll = training_pair_gate_diagnostics(
                    units=units,
                    caches=caches,
                    block_cross_residual=block_core,
                    bundle=bundle,
                    settings=settings,
                )
                broad_nll = training_broad_nll(
                    records=broad_records,
                    caches=caches,
                    block_cross_residual=block_core,
                    bundle=bundle,
                )
                validate_per_unit_nll_diagnostics(per_unit_nll, pair_metrics)
                threshold = candidate_threshold_diagnostic(
                    pair_metrics, broad_nll, prefix_trust_rms
                )
                row = {
                    **specification,
                    "candidate_state_before_forward": candidate_state,
                    "pair_metrics": dict(pair_metrics),
                    "per_unit_nll_diagnostics": [dict(value) for value in per_unit_nll],
                    "broad_nll": broad_nll,
                    "candidate_relative_prefix_trust_rms": prefix_trust_rms,
                    "focus_units": _focus_rows(pair_metrics),
                    "threshold_diagnostic": threshold,
                    "candidate_checkpoint_written": False,
                    "candidate_authorized": False,
                }
                results.append(row)
                print(
                    json.dumps(
                        {
                            "phase": "v48_fixed_candidate",
                            "candidate_id": candidate_id,
                            "direction_id": specification["direction_id"],
                            "alpha": specification["alpha"],
                            "complete_units": pair_metrics["complete_units"],
                            "positive_sides": pair_metrics["positive_sides"],
                            "cross_prefix_complete_units": pair_metrics[
                                "cross_prefix_complete_units"
                            ],
                            "priority_side_deficit": threshold["priority_side_deficit"],
                            "broad_nll": broad_nll,
                            "candidate_relative_prefix_trust_rms": (prefix_trust_rms),
                            "all_numeric_thresholds_met": threshold["all_numeric_thresholds_met"],
                            "candidate_authorized": False,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                after = _restore_source(bundle, named, source_surface)
                restorations.append({"candidate_id": candidate_id, "phase": "after", **after})

        expected_order = [
            (direction_id, alpha) for direction_id in _DIRECTION_IDS for alpha in _ALPHA_GRID
        ]
        observed_order = [(str(row["direction_id"]), float(row["alpha"])) for row in results]
        if observed_order != expected_order or len(results) != _EXACT_CANDIDATE_COUNT:
            raise RuntimeError("V48 did not evaluate fixed 15-candidate grid")
        final_state = _restore_source(bundle, named, source_surface)
        final_state["all_15_before_after_restorations_passed"] = len(
            restorations
        ) == 2 * _EXACT_CANDIDATE_COUNT and all(row["passed"] is True for row in restorations)
        final_state["passed"] = bool(
            final_state["passed"] and final_state["all_15_before_after_restorations_passed"]
        )
        if final_state["passed"] is not True:
            raise RuntimeError("V48 final exact source restoration failed")

    audit.assert_clean()
    if _sha256(protected) != protected_before:
        raise RuntimeError("V48 changed protected selection report")
    optimizer_reads = [path for path in audit.unique_paths if path.endswith("/optimizer.pt")]
    if optimizer_reads:
        raise RuntimeError(f"V48 opened forbidden optimizer state: {optimizer_reads}")
    loaded_maps = sorted(path for path in audit.unique_paths if path.endswith("/voxel_map.npz"))
    expected_maps = sorted(cache_audit["loaded_environment_files"])
    if loaded_maps != expected_maps or len(loaded_maps) != 16:
        raise RuntimeError("V48 map reads differ from exactly 16 training caches")
    return {
        "schema_version": 1,
        "artifact": "v48_v47_u4_dual_margin_no_step_diagnostic",
        "screen_integrity_passed": True,
        "terminal": terminal,
        "source_audit": source_audit,
        "source_replay": {
            **source_replay,
            "pair_metrics": source_pair,
            "per_unit_nll_diagnostics": source_nll,
            "broad_nll_value": source_broad,
            "focus_units": _focus_rows(source_pair),
        },
        "source_prefix_sha256_by_train_scene": source_prefix_hashes,
        "gradient_audit": gradient_audit,
        "direction_audit": direction_audit,
        "candidate_inventory": {
            "formula": ("float32_P0-alpha*lr_group*sign(normalized_component_sum)"),
            "direction_ids": list(_DIRECTION_IDS),
            "direction_components": {
                key: list(value) for key, value in _DIRECTION_COMPONENTS.items()
            },
            "alpha_grid": list(_ALPHA_GRID),
            "scene_readout_learning_rate": _SCENE_LR,
            "query_learning_rate": _QUERY_LR,
            "candidate_relative_prefix_trust_scale": _PREFIX_TRUST_SCALE,
            "candidate_count": _EXACT_CANDIDATE_COUNT,
            "candidate_hashes_fixed_before_candidate_forward_evaluation": True,
            "candidate_inventory_sha256": inventory_hash,
            "candidates": candidate_inventory,
        },
        "candidate_results": results,
        "all_15_candidates_received_full_25_unit_metrics": all(
            int(_mapping(row["pair_metrics"], "pair metrics")["unit_count"]) == 25
            and len(_sequence(row["per_unit_nll_diagnostics"], "nll rows")) == 25
            for row in results
        ),
        "all_15_candidates_received_fixed_48_row_broad_nll": len(results) == _EXACT_CANDIDATE_COUNT,
        "all_15_candidates_received_candidate_relative_prefix_trust": all(
            math.isfinite(float(row["candidate_relative_prefix_trust_rms"])) for row in results
        ),
        "candidate_selection_performed": False,
        "adaptive_direction_or_scalar_selection": False,
        "candidate_authorization_granted": False,
        "candidate_checkpoint_written": False,
        "restoration_audit": restorations,
        "final_state": final_state,
        "trainable_surface_used_only_for_autograd": surface,
        "model_loaded_once": True,
        "optimizer_constructed_or_loaded": False,
        "optimizer_state_file_opened": False,
        "optimizer_step_executed": False,
        "parameter_state_persisted": False,
        "greedy_generation_executed": False,
        "loader_transition": loader_transition,
        "cache_boundary": cache_boundary,
        "scene_prefix_evidence": prefix_evidence,
        "qa_audit": qa_audit,
        "schedule_audit": schedule_audit,
        "all_16_training_maps_loaded": True,
        "validation_qa_loaded": False,
        "validation_environment_maps_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_executed": False,
        "runtime_promotion_executed": False,
        "chat_promotion_executed": False,
        "embodied_promotion_executed": False,
        "protected_report_sha256_before_and_after": protected_before,
        "loaded_files": audit.unique_paths,
        "forbidden_file_accesses": audit.forbidden_accesses(),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_report(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    expected_v47_terminal_sha256: str,
) -> dict[str, Any]:
    path = _resolve(output)
    if path != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V48 output path is pinned")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V48 is one-shot and will not overwrite {path}")
    report = run_screen(
        config_path=config_path,
        expected_v47_terminal_sha256=expected_v47_terminal_sha256,
    )
    _atomic_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-v47-terminal-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = (
        _preflight(
            config_path=args.config,
            expected_v47_terminal_sha256=args.expected_v47_terminal_sha256,
        )
        if args.preflight_only
        else write_report(
            args.output,
            config_path=args.config,
            expected_v47_terminal_sha256=args.expected_v47_terminal_sha256,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_candidate_inventory",
    "build_normalized_directions",
    "candidate_from_normalized_direction",
    "candidate_threshold_diagnostic",
    "gradient_geometry",
    "normalize_gradient_components_by_group",
    "require_terminal",
    "run_screen",
    "write_report",
]

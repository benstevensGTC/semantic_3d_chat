"""Fixed train-only V45 update-four lost-side response screen for V46.

This is a report-only diagnostic, not training or checkpoint selection.  It
measures three isolated side-margin gradients at the exact failed V45
update-four checkpoint.  Only the q5 gradient is used to construct a fixed
three-direction by five-alpha fresh-Adam-sign line.  Every one of the fifteen
prehashed candidates receives the complete 25-unit teacher-forced pair audit
and the same fixed 48-row broad-NLL audit.  The exact update-four state is
restored before and after every probe and no candidate state is persisted.

The V45 terminal hash is intentionally supplied at invocation time.  Embedding
that hash here would create a source/seal hash cycle because the terminal pins
this module and its tests.
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
from semantic_3d_chat.language.lora import tensor_state_sha256
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
from semantic_3d_chat.training.train_block_cross_v35 import (
    paired_cross_prefix_objective,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
)
from semantic_3d_chat.training.train_joint_block_cross_v36 import (
    training_broad_nll,
)
from semantic_3d_chat.training.train_joint_pair_v30 import (
    require_approved_v29_source,
)
from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
    _PARAMETER_NAMES,
    _PARAMETER_SHAPES,
    assert_v44_trainable_surface,
    freeze_for_v44,
    frozen_v44_state_sha256,
    v44_contract,
)
from semantic_3d_chat.training.train_projected_gradient_v41 import (
    _prefix_replay_attestation,
    cache_v41_train_scenes,
    load_v41_bundle,
    priority_side_deficit,
    training_pair_gate_diagnostics,
    v41_loader_config,
)
from semantic_3d_chat.training.train_retention_repair_v45 import (
    _CONFIG_FILE_SHA256,
    _PROTECTED_REPORT,
    _PROTECTED_REPORT_SHA256,
    _V41_FULL_SHA256,
    DEFAULT_CONFIG,
    _preflight_forbidden_roots,
    _training_forbidden_roots,
    _unit_index,
    _unit_tokens,
    _v41_source_tensors,
    build_v45_schedule,
    load_v35_train_qa_records,
    v31_contract,
    v45_retention_diagnostics,
    v45_settings,
)
from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
    validate_v37_training_cache_boundary,
)

DEFAULT_TERMINAL = Path("reports/gemma4/metrics/v45_retention_repair_terminal_gate.json")
DEFAULT_SOURCE = Path("data_gemma4/checkpoints/gemma4_v45_retention_repair_l14_query/update_004")
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v46_v45_u4_lost_side_no_step_diagnostic.json")
V46_SCRIPT = Path("src/semantic_3d_chat/evaluation/v46_v45_u4_lost_side_screen.py")
V46_TEST = Path("tests/test_v46_v45_u4_lost_side_screen.py")

_AUTHORIZATION_ID = "v46_train_only_checkpoint_gradient_diagnostic"
_DIRECTION_IDS = ("g5_scene_sign", "g5_query_sign", "g5_both_sign")
_ALPHA_GRID = (0.125, 0.25, 0.5, 1.0, 2.0)
_SCENE_LR = 1.0e-5
_QUERY_LR = 8.0e-6
_EXACT_CANDIDATE_COUNT = 15
_TRAIN_SCENES = tuple(
    [*(f"scene_{index:06d}" for index in range(11, 19))]
    + [*(f"scene_{index:06d}" for index in range(31, 39))]
)
_GRADIENT_SPECS = (
    ("g5", "pair_000006", "cfq_5c84a2c27d2be251", 0),
    ("q699", "pair_000016", "cfq_699675ceeaf65406", 1),
    ("q0a79", "pair_000006", "cfq_0a79d507273195ef", 0),
)
_SOURCE_FILES = {
    "adapter.safetensors": ("baffb29e31e1ddf0164bf4b9bcf47ab14f61160f3d46e834ceafc3c1a7c66e17"),
    TRAINING_METADATA_FILENAME: (
        "4249bcdec60dd7468e62c0687616a8a820be0bae94289636da33e4379dc7bf6c"
    ),
    "optimizer.pt": ("c409db27ccc6ef68e43c36123519810c3b65a9d579715ff64d5f3595d7da688d"),
    RUNTIME_METADATA_FILENAME: ("8beca055a77016f4ce0960b49789e750ab6b34d3edd852888cddec7a4e2980f0"),
}
_READABLE_SOURCE_FILES = (
    "adapter.safetensors",
    TRAINING_METADATA_FILENAME,
    RUNTIME_METADATA_FILENAME,
)
_SOURCE_FULL_SHA256 = "468f493a746c6125f8ebc62d57ca8ae0419160f6e13ce903dd9f40c64aa772c2"
_SOURCE_AUTHORIZED_SHA256 = "e4165bb1c2a4664eeb146a48107aead3e69bb576c1604bea39b3b7474d17c696"
_SOURCE_FROZEN_SHA256 = "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"
_ORIGINAL_V41_PRIORITY_DEFICIT = 31.113729119300842
_SOURCE_PRIORITY_DEFICIT = 29.800106167793274
_SOURCE_BROAD_NLL = 2.889571795860926
_BROAD_NLL_MAXIMUM = 2.9213306349515915
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
    """Recursively compare persisted numeric diagnostics at gate precision."""

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
            "pair_id": pair_id,
            "question_key": question_key,
            "side_index": side_index,
            "role": (
                "g5_candidate_direction_source"
                if gradient_id == "g5"
                else "diagnostic_only_never_a_candidate_direction"
            ),
        }
        for gradient_id, pair_id, question_key, side_index in _GRADIENT_SPECS
    ]


def _validate_authorization(
    report: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate every V45 authorization field consumed by this screen."""

    invocation = _mapping(authorization.get("invocation_contract"), "V46 invocation contract")
    source = _mapping(authorization.get("source"), "V46 source")
    boundary = _mapping(authorization.get("fixed_data_boundary"), "V46 data boundary")
    measurements = _mapping(authorization.get("measurements"), "V46 measurements")
    groups = _mapping(measurements.get("gradient_groups"), "V46 gradient groups")
    line = _mapping(authorization.get("fresh_adam_sign_line"), "V46 sign line")
    direction_source = _mapping(line.get("direction_source"), "V46 direction source")
    definitions = _mapping(line.get("direction_definitions"), "V46 direction definitions")
    forbidden = _mapping(authorization.get("forbidden_actions"), "V46 forbidden actions")
    scope = _mapping(authorization.get("scope"), "V46 scope")
    integrity = _mapping(
        authorization.get("implementation_integrity"),
        "V46 implementation integrity",
    )
    expected_definitions = {
        "g5_scene_sign": "scene_readout_only_sign(g5)",
        "g5_query_sign": "query_only_sign(g5)",
        "g5_both_sign": "scene_readout_and_query_sign(g5)",
    }
    checks = {
        "terminal_artifact": report.get("artifact") == "v45_retention_repair_terminal_gate",
        "terminal_passed": report.get("passed") is True,
        "only_successor": report.get("only_exact_successor_authorized") == _AUTHORIZATION_ID,
        "authorization_id": authorization.get("authorization_id") == _AUTHORIZATION_ID,
        "authorized": authorization.get("authorized") is True,
        "only_action": authorization.get("only_exact_action")
        == "one_bounded_read_only_v46_train_checkpoint_gradient_diagnostic",
        "script": authorization.get("authorized_script") == str(V46_SCRIPT),
        "test": authorization.get("authorized_test") == str(V46_TEST),
        "report": authorization.get("authorized_report") == str(DEFAULT_OUTPUT),
        "explicit_cli": authorization.get("explicit_terminal_sha256_cli_required") is True,
        "terminal_path": invocation.get("terminal_path") == str(DEFAULT_TERMINAL),
        "cli_name": invocation.get("required_cli_argument") == "--expected-v45-terminal-sha256",
        "no_embedded_sha": invocation.get("v46_must_not_embed_terminal_sha256") is True,
        "terminal_exact": invocation.get(
            "v46_must_authenticate_terminal_bytes_and_exact_authorization"
        )
        is True,
        "source_path": source.get("checkpoint") == str(DEFAULT_SOURCE),
        "source_full": source.get("full_tensor_state_sha256") == _SOURCE_FULL_SHA256,
        "source_authorized": source.get("authorized_surface_state_sha256")
        == _SOURCE_AUTHORIZED_SHA256,
        "source_frozen": source.get("frozen_state_sha256") == _SOURCE_FROZEN_SHA256,
        "source_files": dict(_mapping(source.get("file_sha256"), "source files")) == _SOURCE_FILES,
        "u2_not_source": source.get(
            "update_002_is_authenticated_by_terminal_but_not_a_v46_probe_source"
        )
        is True,
        "scene_ids": list(_sequence(boundary.get("scene_ids"), "scene ids")) == list(_TRAIN_SCENES),
        "scene_count": boundary.get("scene_count") == 16,
        "train_questions": boundary.get("train_question_count") == 384,
        "pair_units": boundary.get("changed_pair_unit_count") == 25,
        "broad_rows": boundary.get("broad_nll_row_count") == 48,
        "blocking_audit": boundary.get("blocking_file_access_audit_required") is True,
        "complete_prefix": boundary.get("complete_pre_question_scene_prefixes") is True,
        "all_blocks": boundary.get("all_occupied_blocks_processed") is True,
        "no_retrieval": boundary.get("question_dependent_retrieval") is False,
        "source_metrics": measurements.get("source_full_teacher_forced_pair_metrics") is True,
        "source_broad": measurements.get("source_fixed_full_48_row_broad_nll") is True,
        "gradient_specs": list(
            _sequence(measurements.get("isolated_side_gradient_specs"), "gradients")
        )
        == _expected_gradient_specs(),
        "gradient_loss": measurements.get("gradient_loss_formula")
        == "negative_selected_side_margin",
        "scene_group": list(_sequence(groups.get("scene_readout"), "scene group"))
        == [_PARAMETER_NAMES[0]],
        "query_group": list(_sequence(groups.get("query"), "query group"))
        == list(_PARAMETER_NAMES[1:]),
        "gradient_report": measurements.get("report_gradient_norms_and_pairwise_cosines") is True,
        "line_source": line.get("source_checkpoint") == str(DEFAULT_SOURCE),
        "direction_source_id": direction_source.get("gradient_id") == "g5",
        "direction_source_loss": direction_source.get("loss")
        == ("-side_margins[0] for pair_000006/cfq_5c84a2c27d2be251 at exact update_004"),
        "autograd_source": direction_source.get("autograd_exact_at_source") is True,
        "direction_ids": list(_sequence(line.get("direction_ids"), "directions"))
        == list(_DIRECTION_IDS),
        "direction_definitions": dict(definitions) == expected_definitions,
        "formula": line.get("candidate_formula") == "float32_P0-alpha*lr_group*sign(g5)",
        "scene_lr": line.get("scene_readout_learning_rate") == _SCENE_LR,
        "query_lr": line.get("query_learning_rate") == _QUERY_LR,
        "alpha_grid": list(_sequence(line.get("alpha_grid"), "alpha grid")) == list(_ALPHA_GRID),
        "candidate_count": line.get("exact_candidate_count") == _EXACT_CANDIDATE_COUNT,
        "full_pairs": line.get("full_25_unit_teacher_metrics_per_candidate") is True,
        "full_broad": line.get("full_fixed_48_row_broad_nll_per_candidate") is True,
        "in_memory": line.get("in_memory_only") is True,
        "restore": line.get("exact_u4_restoration_before_and_after_every_probe") is True,
        "restore_hash": line.get("full_tensor_hash_restored_after_every_probe")
        == _SOURCE_FULL_SHA256,
        "nonadaptive": line.get("adaptive_direction_or_scalar_selection") is False,
        "diagnostics_not_directions": line.get(
            "diagnostic_gradient_q699_and_q0a79_used_as_directions"
        )
        is False,
        "forbidden_all_true": all(value is True for value in forbidden.values()),
        "train_only": scope.get("train_only") is True,
        "report_only": scope.get("report_only_output") is True,
        "no_candidate_authorized": scope.get("no_candidate_is_authorized_by_this_diagnostic")
        is True,
        "new_terminal": scope.get("new_terminal_seal_required_for_any_successor") is True,
        "no_validation": scope.get("validation_access_authorized") is False,
        "no_oracle": scope.get("oracle_access_authorized") is False,
        "no_final": scope.get("final_test_access_authorized") is False,
        "no_selector": scope.get("selector_execution_authorized") is False,
        "no_promotion": scope.get("runtime_promotion_authorized") is False,
        "script_hash": integrity.get("script_sha256") == _sha256(_resolve(V46_SCRIPT)),
        "test_hash": integrity.get("test_sha256") == _sha256(_resolve(V46_TEST)),
        "config_hash": integrity.get("config_sha256") == _CONFIG_FILE_SHA256,
    }
    if not all(checks.values()):
        raise ValueError(f"V46 terminal authorization changed: {checks}")
    return checks


def require_terminal(expected_sha256: str) -> dict[str, Any]:
    """Authenticate the materialized terminal using the explicit CLI value."""

    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V46 expected terminal SHA256 must be 64 lowercase hex digits")
    path = _resolve(DEFAULT_TERMINAL)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V46 terminal is unavailable or unsafe: {path}")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError("V46 terminal SHA256 differs from the explicit invocation")
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "V45 terminal report")
    authorization = _mapping(report.get("conditional_successor_authorization"), "V46 authorization")
    checks = _validate_authorization(report, authorization)
    return {
        "path": str(DEFAULT_TERMINAL),
        "sha256": observed,
        "authorization_id": _AUTHORIZATION_ID,
        "authorization": dict(authorization),
        "checks": checks,
    }


def _source_evidence() -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Read only V46-authorized u4 files; never open its optimizer state."""

    source = _resolve(DEFAULT_SOURCE)
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError("V46 exact V45 update-four source is unavailable")
    inventory = sorted(path.name for path in source.iterdir())
    if inventory != sorted(_SOURCE_FILES):
        raise ValueError("V46 source checkpoint inventory changed")
    observed_files: dict[str, str] = {}
    for name in _READABLE_SOURCE_FILES:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"V46 readable source file is unavailable: {name}")
        observed_files[name] = _sha256(path)
        if observed_files[name] != _SOURCE_FILES[name]:
            raise ValueError(f"V46 readable source file changed: {name}")
    tensors = load_file(source / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(tensors) != _SOURCE_FULL_SHA256:
        raise ValueError("V46 source full tensor state changed")
    authorized = {name: tensors[name] for name in _PARAMETER_NAMES}
    frozen = {name: value for name, value in tensors.items() if name not in authorized}
    if tensor_state_sha256(authorized) != _SOURCE_AUTHORIZED_SHA256:
        raise ValueError("V46 source authorized tensor surface changed")
    if tensor_state_sha256(frozen) != _SOURCE_FROZEN_SHA256:
        raise ValueError("V46 source frozen tensor state changed")
    metadata = _mapping(
        json.loads((source / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8")),
        "V46 source metadata",
    )
    runtime = _mapping(
        json.loads((source / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8")),
        "V46 source runtime metadata",
    )
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V46 source runtime metadata is not the exact sanitization")
    stage = _mapping(metadata.get("v45_retention_repair"), "V45 source stage")
    history = _sequence(metadata.get("history"), "V45 source history")
    if (
        metadata.get("optimizer_step") != 4
        or stage.get("optimizer_step") != 4
        or len(history) != 5
        or _mapping(history[-1], "V45 source final history").get("optimizer_update") != 4
    ):
        raise ValueError("V46 source is not the exact failed V45 update four")
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
        },
    )


def _validate_surface_tensors(values: Mapping[str, torch.Tensor], *, field: str) -> None:
    if tuple(values) != _PARAMETER_NAMES:
        raise ValueError(f"V46 {field} names or order changed")
    for name, shape in zip(_PARAMETER_NAMES, _PARAMETER_SHAPES):
        value = values[name]
        if value.dtype != torch.float32 or tuple(value.shape) != shape:
            raise ValueError(f"V46 {field} tensor changed: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"V46 {field} tensor is nonfinite: {name}")


def candidate_from_sign_line(
    source: Mapping[str, torch.Tensor],
    g5: Mapping[str, torch.Tensor],
    *,
    direction_id: str,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Construct one exact float32 fresh-Adam-sign candidate on CPU."""

    if direction_id not in _DIRECTION_IDS:
        raise ValueError("V46 direction is outside the fixed three-direction grid")
    if alpha not in _ALPHA_GRID:
        raise ValueError("V46 alpha is outside the fixed five-value grid")
    _validate_surface_tensors(source, field="source")
    _validate_surface_tensors(g5, field="g5 gradient")
    if any(value.device.type != "cpu" for value in (*source.values(), *g5.values())):
        raise ValueError("V46 candidate construction must occur on CPU")
    active = {
        "g5_scene_sign": {_PARAMETER_NAMES[0]},
        "g5_query_sign": set(_PARAMETER_NAMES[1:]),
        "g5_both_sign": set(_PARAMETER_NAMES),
    }[direction_id]
    result = {name: value.clone() for name, value in source.items()}
    for name in _PARAMETER_NAMES:
        if name not in active:
            continue
        learning_rate = _SCENE_LR if name == _PARAMETER_NAMES[0] else _QUERY_LR
        result[name].add_(torch.sign(g5[name]), alpha=-float(alpha * learning_rate))
    _validate_surface_tensors(result, field="candidate")
    for name in set(_PARAMETER_NAMES) - active:
        if not torch.equal(result[name], source[name]):
            raise RuntimeError("V46 candidate changed an inactive tensor group")
    return result


def build_candidate_inventory(
    source_surface: Mapping[str, torch.Tensor],
    source_full: Mapping[str, torch.Tensor],
    g5: Mapping[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], str]:
    """Precompute and prehash the fixed 15 candidates before any evaluation."""

    rows: list[dict[str, Any]] = []
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    for direction_id in _DIRECTION_IDS:
        for alpha in _ALPHA_GRID:
            candidate_id = f"{direction_id}_alpha_{str(alpha).replace('.', 'p')}"
            candidate = candidate_from_sign_line(
                source_surface, g5, direction_id=direction_id, alpha=alpha
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
    if (
        len(rows) != _EXACT_CANDIDATE_COUNT
        or len(tensors) != _EXACT_CANDIDATE_COUNT
        or [(row["direction_id"], row["alpha"]) for row in rows]
        != [(direction_id, alpha) for direction_id in _DIRECTION_IDS for alpha in _ALPHA_GRID]
    ):
        raise RuntimeError("V46 candidate inventory is not the fixed 3x5 grid")
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
        raise RuntimeError("V46 failed to restore exact update four")
    return result


def _copy_candidate(
    named: Mapping[str, torch.nn.Parameter], candidate: Mapping[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(candidate[name].to(device=parameter.device, dtype=parameter.dtype))
            parameter.grad = None
            parameter.requires_grad_(False)


def _selected_side_gradient(
    *,
    unit: CounterfactualPairUnit,
    expected_pair_id: str,
    expected_question_key: str,
    side_index: int,
    caches: Mapping[str, Any],
    block_core: torch.nn.Module,
    bundle: Any,
    named: Mapping[str, torch.nn.Parameter],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if (
        unit.pair_id != expected_pair_id
        or unit.question_key != expected_question_key
        or side_index not in (0, 1)
    ):
        raise ValueError("V46 isolated-gradient unit or side changed")
    freeze_for_v44(bundle, block_core)
    if any(parameter.grad is not None for parameter in named.values()):
        raise RuntimeError("V46 isolated gradient found accumulated gradients")
    tokens = _unit_tokens(
        unit,
        caches=caches,
        block_core=block_core,
        device=bundle.language.device,
    )
    correct, side, cross, diagnostics = paired_cross_prefix_objective(
        unit=unit,
        scene_tokens=tokens,
        bundle=bundle,
        side_margin=0.5,
        cross_prefix_margin=0.1,
    )
    selected_margin = diagnostics["side_margins"].reshape(2)[side_index]
    loss = -selected_margin
    raw = torch.autograd.grad(
        loss,
        tuple(named.values()),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    gradients = {name: value.detach().float().cpu().clone() for name, value in zip(named, raw)}
    _validate_surface_tensors(gradients, field="isolated gradient")
    if any(torch.count_nonzero(value).item() == 0 for value in gradients.values()):
        raise RuntimeError("V46 isolated gradient contains a zero tensor")
    if any(parameter.grad is not None for parameter in named.values()):
        raise RuntimeError("V46 autograd.grad accumulated into parameter.grad")
    row = {
        "pair_id": unit.pair_id,
        "question_key": unit.question_key,
        "side_index": side_index,
        "selected_side_margin_in_gradient_mode": float(selected_margin.detach().float().cpu()),
        "loss_formula": "negative_selected_side_margin",
        "gradient_state_sha256": {
            name: tensor_state_sha256({"gradient": value}) for name, value in gradients.items()
        },
        "parameter_grad_accumulation": False,
    }
    del correct, side, cross, diagnostics, selected_margin, loss, raw, tokens
    if bundle.language.device.type == "mps":
        torch.mps.empty_cache()
    return gradients, row


def _flatten_group(gradients: Mapping[str, torch.Tensor], group: str) -> torch.Tensor:
    names = (_PARAMETER_NAMES[0],) if group == "scene_readout" else _PARAMETER_NAMES[1:]
    return torch.cat([gradients[name].reshape(-1).double() for name in names])


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if not torch.isfinite(denominator) or float(denominator) == 0.0:
        raise RuntimeError("V46 gradient cosine denominator is invalid")
    return float(torch.dot(left, right) / denominator)


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
        gradient, row = _selected_side_gradient(
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
    groups: dict[str, Any] = {}
    for group in ("scene_readout", "query"):
        vectors = {
            gradient_id: _flatten_group(gradient, group)
            for gradient_id, gradient in gradients.items()
        }
        groups[group] = {
            "l2_norm": {gradient_id: float(value.norm()) for gradient_id, value in vectors.items()},
            "cosine_g5_q699": _cosine(vectors["g5"], vectors["q699"]),
            "cosine_g5_q0a79": _cosine(vectors["g5"], vectors["q0a79"]),
            "cosine_q699_q0a79": _cosine(vectors["q699"], vectors["q0a79"]),
        }
    source_state = _bundle_state_attestation(
        bundle,
        named,
        expected_authorized_sha256=_SOURCE_AUTHORIZED_SHA256,
        expected_full_sha256=_SOURCE_FULL_SHA256,
    )
    if source_state["passed"] is not True:
        raise RuntimeError("V46 gradient measurement changed its source state")
    return gradients, {
        "autograd_api": "torch.autograd.grad",
        "loss_formula": "negative_selected_side_margin",
        "rows": rows,
        "groups": groups,
        "candidate_direction_gradient_ids": ["g5"],
        "diagnostic_only_gradient_ids": ["q699", "q0a79"],
        "source_state_after_gradient_measurement": source_state,
        "source_state_unchanged": True,
        "optimizer_constructed_or_loaded": False,
        "backward_called": False,
    }


def _focus_rows(pair_metrics: Mapping[str, Any]) -> dict[str, Any]:
    units = _sequence(pair_metrics.get("units"), "V46 pair metric units")
    indexed = {
        str(_mapping(value, "V46 pair metric unit")["question_key"]): _mapping(
            value, "V46 pair metric unit"
        )
        for value in units
    }
    return {
        question_key: {
            "pair_id": indexed[question_key]["pair_id"],
            "side_margins": list(indexed[question_key]["side_margins"]),
            "cross_prefix_margins": list(indexed[question_key]["cross_prefix_margins"]),
            "complete": indexed[question_key]["complete"],
            "cross_prefix_complete": indexed[question_key]["cross_prefix_complete"],
        }
        for question_key in (
            "cfq_5c84a2c27d2be251",
            "cfq_699675ceeaf65406",
            "cfq_0a79d507273195ef",
        )
    }


def _candidate_threshold_diagnostic(
    pair_metrics: Mapping[str, Any], broad_nll: float
) -> dict[str, Any]:
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    retention = v45_retention_diagnostics(pair_metrics)
    checks = {
        "complete_units_at_least_9": int(pair_metrics["complete_units"]) >= 9,
        "positive_sides_at_least_34": int(pair_metrics["positive_sides"]) >= 34,
        "cross_prefix_complete_units_at_least_17": int(pair_metrics["cross_prefix_complete_units"])
        >= 17,
        "complete_physical_pair_coverage_at_least_4": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= 4,
        "priority_deficit_improvement_at_least_0_5_vs_original_v41_u0": (
            _ORIGINAL_V41_PRIORITY_DEFICIT - deficit >= 0.5
        ),
        "broad_nll_at_most_v45_maximum": broad_nll <= _BROAD_NLL_MAXIMUM,
        "both_lost_sides_strictly_positive": retention["both_lost_sides_strictly_positive"],
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
        "retention_diagnostics": retention,
    }


def _preflight(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    expected_v45_terminal_sha256: str,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    if config_file != _resolve(DEFAULT_CONFIG) or _sha256(config_file) != _CONFIG_FILE_SHA256:
        raise ValueError("V46 config path or bytes changed")
    terminal = require_terminal(expected_v45_terminal_sha256)
    protected = _resolve(_PROTECTED_REPORT)
    if protected.is_symlink() or _sha256(protected) != _PROTECTED_REPORT_SHA256:
        raise ValueError("V46 protected selection report changed")
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
            raise RuntimeError("V46 preflight train-only inventory changed")
    audit.assert_clean()
    return {
        "schema_version": 1,
        "artifact": "v46_v45_u4_lost_side_screen_preflight",
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
    expected_v45_terminal_sha256: str,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    if config_file != _resolve(DEFAULT_CONFIG) or _sha256(config_file) != _CONFIG_FILE_SHA256:
        raise ValueError("V46 config path or bytes changed")
    terminal = require_terminal(expected_v45_terminal_sha256)
    protected = _resolve(_PROTECTED_REPORT)
    if protected.is_symlink() or not protected.is_file():
        raise ValueError("V46 protected selection report is unavailable or unsafe")
    protected_before = _sha256(protected)
    if protected_before != _PROTECTED_REPORT_SHA256:
        raise ValueError("V46 protected selection report changed before the screen")
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
            raise RuntimeError("V46 train-only diagnostic inventory changed")

        construction = v44_contract(config)
        v41_tensors, v41_metadata = _v41_source_tensors(construction)
        if tensor_state_sha256(v41_tensors) != _V41_FULL_SHA256:
            raise RuntimeError("V46 V41 construction tensor state changed")
        approved = require_approved_v29_source(loader)
        bundle, block_core, loaded_v41_metadata, loader_transition = load_v41_bundle(
            config,
            approved,
            construction.source_checkpoint,
            v41_tensors,
        )
        if loaded_v41_metadata != v41_metadata:
            raise RuntimeError("V46 V41 construction metadata changed")
        loaded_u4 = load_adapter_checkpoint(
            _resolve(DEFAULT_SOURCE),
            bundle.checkpoint_modules,
            device="cpu",
            metadata_filename=TRAINING_METADATA_FILENAME,
        )
        if loaded_u4 != source_metadata:
            raise RuntimeError("V46 strict V45 update-four overlay metadata changed")
        named = freeze_for_v44(bundle, block_core)
        surface = assert_v44_trainable_surface(bundle, block_core)
        if (
            module_collection_state_sha256(bundle.checkpoint_modules) != _SOURCE_FULL_SHA256
            or tensor_state_sha256({name: value.detach().cpu() for name, value in named.items()})
            != _SOURCE_AUTHORIZED_SHA256
            or frozen_v44_state_sha256(bundle) != _SOURCE_FROZEN_SHA256
        ):
            raise RuntimeError("V46 live source state differs from exact V45 update four")

        split = v31_contract(loader)
        if tuple(split.train_scene_ids) != _TRAIN_SCENES:
            raise RuntimeError("V46 exact train scene split changed")
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
            raise RuntimeError("V46 did not cache exactly all 16 training scenes")

        settings = v45_settings(config)
        source_pair_metrics, source_nll = training_pair_gate_diagnostics(
            units=units,
            caches=caches,
            block_cross_residual=block_core,
            bundle=bundle,
            settings=settings,
        )
        source_broad_nll = training_broad_nll(
            records=broad_records,
            caches=caches,
            block_cross_residual=block_core,
            bundle=bundle,
        )
        history = _sequence(source_metadata.get("history"), "V46 source history")
        final_history = _mapping(history[-1], "V46 source final history")
        source_replay = {
            "pair_metrics": _numeric_close(
                source_pair_metrics,
                final_history.get("pair_metrics"),
            ),
            "per_unit_nll": _numeric_close(
                source_nll,
                final_history.get("per_unit_nll_diagnostics"),
            ),
            "broad_nll": math.isclose(
                source_broad_nll,
                float(final_history.get("broad_diagnostic_nll")),
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            )
            and math.isclose(
                source_broad_nll,
                _SOURCE_BROAD_NLL,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            ),
            "priority_deficit": math.isclose(
                float(priority_side_deficit(source_pair_metrics)["combined"]),
                _SOURCE_PRIORITY_DEFICIT,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            ),
        }
        source_replay["passed"] = all(source_replay.values())
        if source_replay["passed"] is not True:
            raise RuntimeError("V46 source update-four diagnostic replay changed")

        gradients, gradient_audit = _gradient_diagnostics(
            units_by_key=units_by_key,
            caches=caches,
            block_core=block_core,
            bundle=bundle,
            named=named,
        )
        candidate_inventory, candidate_tensors, inventory_hash = build_candidate_inventory(
            source_surface, source_full, gradients["g5"]
        )
        if len(candidate_inventory) != _EXACT_CANDIDATE_COUNT:
            raise RuntimeError("V46 did not prehash exactly 15 candidates")

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
                    raise RuntimeError("V46 live candidate state differs from its prehash")
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
                threshold = _candidate_threshold_diagnostic(pair_metrics, broad_nll)
                row = {
                    **specification,
                    "candidate_state_before_forward": candidate_state,
                    "pair_metrics": dict(pair_metrics),
                    "per_unit_nll_diagnostics": [dict(value) for value in per_unit_nll],
                    "broad_nll": broad_nll,
                    "focus_units": _focus_rows(pair_metrics),
                    "threshold_diagnostic": threshold,
                    "candidate_checkpoint_written": False,
                    "candidate_authorized": False,
                }
                results.append(row)
                print(
                    json.dumps(
                        {
                            "phase": "v46_fixed_candidate",
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
            raise RuntimeError("V46 did not fully evaluate the fixed 15-candidate grid")
        final_state = _restore_source(bundle, named, source_surface)
        final_state["all_15_before_after_restorations_passed"] = (
            all(row["passed"] is True for row in restorations)
            and len(restorations) == 2 * _EXACT_CANDIDATE_COUNT
        )
        final_state["passed"] = bool(
            final_state["passed"] and final_state["all_15_before_after_restorations_passed"]
        )
        if final_state["passed"] is not True:
            raise RuntimeError("V46 final exact source restoration failed")

    audit.assert_clean()
    if _sha256(protected) != protected_before:
        raise RuntimeError("V46 changed the protected selection report")
    optimizer_reads = [path for path in audit.unique_paths if path.endswith("/optimizer.pt")]
    if optimizer_reads:
        raise RuntimeError(f"V46 opened forbidden optimizer state: {optimizer_reads}")
    loaded_maps = sorted(path for path in audit.unique_paths if path.endswith("/voxel_map.npz"))
    expected_maps = sorted(cache_audit["loaded_environment_files"])
    if loaded_maps != expected_maps or len(loaded_maps) != 16:
        raise RuntimeError("V46 map reads differ from exactly all 16 train caches")
    return {
        "schema_version": 1,
        "artifact": "v46_v45_u4_lost_side_no_step_diagnostic",
        "screen_integrity_passed": True,
        "terminal": terminal,
        "source_audit": source_audit,
        "source_replay": {
            **source_replay,
            "pair_metrics": source_pair_metrics,
            "per_unit_nll_diagnostics": source_nll,
            "broad_nll_value": source_broad_nll,
            "focus_units": _focus_rows(source_pair_metrics),
        },
        "gradient_audit": gradient_audit,
        "candidate_inventory": {
            "formula": "float32_P0-alpha*lr_group*sign(g5)",
            "direction_ids": list(_DIRECTION_IDS),
            "alpha_grid": list(_ALPHA_GRID),
            "scene_readout_learning_rate": _SCENE_LR,
            "query_learning_rate": _QUERY_LR,
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
    expected_v45_terminal_sha256: str,
) -> dict[str, Any]:
    path = _resolve(output)
    if path != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V46 output path is pinned")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V46 is one-shot and will not overwrite {path}")
    report = run_screen(
        config_path=config_path,
        expected_v45_terminal_sha256=expected_v45_terminal_sha256,
    )
    _atomic_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-v45-terminal-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = (
        _preflight(
            config_path=args.config,
            expected_v45_terminal_sha256=args.expected_v45_terminal_sha256,
        )
        if args.preflight_only
        else write_report(
            args.output,
            config_path=args.config,
            expected_v45_terminal_sha256=args.expected_v45_terminal_sha256,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_candidate_inventory",
    "candidate_from_sign_line",
    "require_terminal",
    "run_screen",
    "write_report",
]

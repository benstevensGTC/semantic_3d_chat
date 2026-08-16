"""Bounded V50 train-only scene/query alpha grid.

V50 is the exact fallback authorized after V49 missed only the original-prefix
trust bound.  It reconstructs one fixed guarded normalized gradient direction
from the immutable V47 update-004 source, keeps query alpha fixed at 2.0, and
evaluates scene alphas 1.0, 0.5, and 0.25 in declared order.  Every candidate
is reconstructed directly from the source; there is no cumulative update and
no question-dependent selection.

All three candidates receive the complete non-greedy 25-unit/48-row train
gate.  A candidate receives the complete greedy 25-unit/48-row train gate if
and only if its own non-greedy gate passes.  Selection occurs only after the
complete fixed grid has been evaluated, and chooses the first fully passing
candidate in declared order.  The sole output checkpoint is optimizer-free
and is published atomically only after exact source restoration and a clean
file-access audit.

No validation, oracle, final-test, selector, promotion, chat, or embodied
access is authorized.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import v49_guarded_candidate_greedy_gate as v49

AUTHORIZATION_ID = "v50_scene_query_alpha_grid"
V50_SCRIPT = Path("src/semantic_3d_chat/evaluation/v50_scene_query_alpha_grid.py")
V50_TEST = Path("tests/test_v50_scene_query_alpha_grid.py")
V49_TERMINAL = Path("reports/gemma4/metrics/v49_guarded_candidate_terminal_gate.json")
V49_REPORT = Path("reports/gemma4/metrics/v49_guarded_candidate_train_gate.json")
DEFAULT_REPORT = Path("reports/gemma4/metrics/v50_scene_query_alpha_grid.json")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v50_scene_query_alpha_grid"
)
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_ROOT / "update_000"
DEFAULT_CONFIG = v49.DEFAULT_CONFIG
SOURCE_CHECKPOINT = v49.SOURCE_CHECKPOINT
PREFIX_REFERENCE_CHECKPOINT = v49.PREFIX_REFERENCE_CHECKPOINT
PROTECTED_REPORT = v49.PROTECTED_REPORT

_CONFIG_SHA256 = v49._CONFIG_SHA256
_V49_REPORT_SHA256 = "7d82a503a5402dcfd80816459eea5d653849e89a49fa8ce7b585dc806ff7acc9"
_PROTECTED_REPORT_SHA256 = v49._PROTECTED_REPORT_SHA256
_SOURCE_FILES = dict(v49._SOURCE_FILES)
_SOURCE_FULL_SHA256 = v49._SOURCE_FULL_SHA256
_SOURCE_AUTHORIZED_SHA256 = v49._SOURCE_AUTHORIZED_SHA256
_FROZEN_SHA256 = v49._FROZEN_SHA256
_PREFIX_REFERENCE_FILES = dict(v49._PREFIX_REFERENCE_FILES)
_PREFIX_REFERENCE_FULL_SHA256 = v49._PREFIX_REFERENCE_FULL_SHA256
_PREFIX_REFERENCE_AUTHORIZED_SHA256 = v49._PREFIX_REFERENCE_AUTHORIZED_SHA256
_V49_CANDIDATE_FULL_SHA256 = v49._CANDIDATE_FULL_SHA256
_V49_CANDIDATE_AUTHORIZED_SHA256 = v49._CANDIDATE_AUTHORIZED_SHA256
_ORIGINAL_V41_PRIORITY_DEFICIT = v49._ORIGINAL_V41_PRIORITY_DEFICIT
_BROAD_NLL_MAXIMUM = v49._BROAD_NLL_MAXIMUM
_PREFIX_TRUST_RMS_MAXIMUM = v49._PREFIX_TRUST_RMS_MAXIMUM

_SCENE_ALPHAS = (1.0, 0.5, 0.25)
_QUERY_ALPHA = 2.0
_SCENE_LR = 1.0e-5
_QUERY_LR = 8.0e-6
_DIRECTION_ID = "guarded_both_sign"
_DIRECTION_COMPONENTS = ("g_book", "g_mirror", "g5_guard")
_GRADIENT_SPECS = (
    ("g_book", "pair_000015", "cfq_163eb92339ad35a5", 0),
    ("g_mirror", "pair_000016", "cfq_699675ceeaf65406", 1),
    ("g5_guard", "pair_000006", "cfq_5c84a2c27d2be251", 0),
)
_NON_GREEDY_CHECK_NAMES = (
    "source_v47_u4_exact_before_reconstruction",
    "reconstructed_candidate_full_tensor_state_exact",
    "reconstructed_candidate_authorized_surface_state_exact",
    "teacher_complete_units_at_least_10",
    "teacher_positive_sides_at_least_35",
    "teacher_cross_prefix_complete_units_at_least_17",
    "complete_physical_pair_coverage_at_least_5",
    "mirror_complete_units_at_least_2",
    "book_complete_units_at_least_1",
    "book_cross_prefix_complete_units_at_least_1",
    "priority_deficit_improvement_at_least_0_5_vs_original_v41_u0",
    "broad_nll_at_most_v45_maximum",
    "both_lost_sides_strictly_positive",
    "scene_readout_state_changed",
    "query_state_changed",
    "frozen_state_exact",
    "original_v46_candidate_relative_prefix_trust_rms_at_most_0_002",
)
_GREEDY_CHECK_NAMES = (
    "train_greedy_complete_units_at_least_5",
    "broad_greedy_exact_correct_at_least_23_of_48",
    "broad_greedy_row_count_exactly_48",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _alpha_id(value: float) -> str:
    return str(value).replace(".", "p")


CANDIDATE_GRID = tuple(
    {
        "candidate_id": (
            f"guarded_scene_alpha_{_alpha_id(scene_alpha)}_query_alpha_2p0"
        ),
        "declared_order": index,
        "scene_alpha": scene_alpha,
        "query_alpha": _QUERY_ALPHA,
    }
    for index, scene_alpha in enumerate(_SCENE_ALPHAS)
)


class GridBackend(Protocol):
    """Narrow execution seam shared by production and deterministic tests."""

    def authenticate_and_prepare(self) -> Mapping[str, Any]: ...

    def reconstruct_candidate(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def evaluate_non_greedy(self, candidate_id: str) -> Mapping[str, Any]: ...

    def evaluate_greedy(self, candidate_id: str) -> Mapping[str, Any]: ...

    def restore_source(self) -> Mapping[str, Any]: ...

    def stage_checkpoint(
        self,
        directory: Path,
        candidate: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def access_audit(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class GridPaths:
    terminal: Path = V49_TERMINAL
    report: Path = DEFAULT_REPORT
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    config: Path = DEFAULT_CONFIG


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    combined = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(combined))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _locked_hash(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} is unavailable or unsafe: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{field} changed: expected {expected}, observed {observed}")


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


def _validate_terminal_authorization(
    report: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, bool]:
    invocation = _mapping(authorization.get("invocation_contract"), "V50 invocation")
    integrity = _mapping(authorization.get("implementation_integrity"), "V50 integrity")
    source = _mapping(authorization.get("source"), "V50 source")
    u4 = _mapping(source.get("v47_u4"), "V50 V47-u4 source")
    prefix = _mapping(source.get("original_v46_candidate_prefix_reference"), "V50 prefix")
    v49_failed = _mapping(source.get("v49_failed_result"), "V50 V49 result")
    grid = _mapping(authorization.get("candidate_grid"), "V50 grid")
    evaluation = _mapping(
        authorization.get("evaluation_and_selection"), "V50 evaluation"
    )
    gate = _mapping(authorization.get("per_candidate_gate"), "V50 train gate")
    persistence = _mapping(authorization.get("conditional_persistence"), "V50 persistence")
    scope = _mapping(authorization.get("scope"), "V50 scope")
    expected_inventory = [dict(value) for value in CANDIDATE_GRID]
    checks = {
        "artifact": report.get("artifact") == "v49_guarded_candidate_terminal_gate",
        "passed": report.get("passed") is True,
        "terminal_ready": report.get("terminal_materialization_authorized") is True,
        "successor": report.get("only_exact_successor_authorized") == AUTHORIZATION_ID,
        "authorization_id": authorization.get("authorization_id") == AUTHORIZATION_ID,
        "authorized": authorization.get("authorized") is True,
        "action": authorization.get("only_exact_action")
        == "one_fixed_v50_train_only_scene_query_alpha_grid",
        "script": authorization.get("authorized_script") == str(V50_SCRIPT),
        "test": authorization.get("authorized_test") == str(V50_TEST),
        "report": authorization.get("authorized_report") == str(DEFAULT_REPORT),
        "checkpoint": authorization.get("conditional_checkpoint_output")
        == str(DEFAULT_CHECKPOINT),
        "config": authorization.get("authorized_config") == str(DEFAULT_CONFIG),
        "explicit_sha": authorization.get("explicit_terminal_sha256_cli_required") is True,
        "terminal_path": invocation.get("terminal_path") == str(V49_TERMINAL),
        "cli": invocation.get("required_cli_argument") == "--expected-v49-terminal-sha256",
        "no_embedded_sha": invocation.get("v50_must_not_embed_terminal_sha256") is True,
        "terminal_auth": invocation.get(
            "v50_must_authenticate_terminal_bytes_and_exact_authorization"
        )
        is True,
        "script_hash": integrity.get("script_sha256") == _sha256(_resolve(V50_SCRIPT)),
        "test_hash": integrity.get("test_sha256") == _sha256(_resolve(V50_TEST)),
        "config_hash": integrity.get("config_sha256") == _CONFIG_SHA256,
        "hashes_complete": integrity.get("hashes_complete") is True,
        "v49_report": v49_failed.get("path") == str(V49_REPORT)
        and v49_failed.get("sha256") == _V49_REPORT_SHA256,
        "v49_failed_only_trust": v49_failed.get("only_failed_check")
        == "original_v46_candidate_relative_prefix_trust_rms_at_most_0_002"
        and v49_failed.get("observed_prefix_trust_rms")
        == 0.0020444965921342373
        and v49_failed.get("prefix_trust_rms_maximum") == _PREFIX_TRUST_RMS_MAXIMUM
        and v49_failed.get("greedy_executed") is False
        and v49_failed.get("checkpoint_written") is False
        and v49_failed.get("source_restored_exact") is True,
        "source_checkpoint": u4.get("checkpoint") == str(SOURCE_CHECKPOINT),
        "source_files": u4.get("file_sha256") == _SOURCE_FILES,
        "source_full": u4.get("full_tensor_state_sha256") == _SOURCE_FULL_SHA256,
        "source_authorized": u4.get("authorized_surface_state_sha256")
        == _SOURCE_AUTHORIZED_SHA256,
        "source_frozen": u4.get("frozen_state_sha256") == _FROZEN_SHA256,
        "source_optimizer_forbidden": u4.get("optimizer_file_open_authorized") is False,
        "prefix_checkpoint": prefix.get("checkpoint") == str(PREFIX_REFERENCE_CHECKPOINT),
        "prefix_files": prefix.get("file_sha256") == _PREFIX_REFERENCE_FILES,
        "prefix_full": prefix.get("full_tensor_state_sha256")
        == _PREFIX_REFERENCE_FULL_SHA256,
        "prefix_authorized": prefix.get("authorized_surface_state_sha256")
        == _PREFIX_REFERENCE_AUTHORIZED_SHA256,
        "prefix_frozen": prefix.get("frozen_state_sha256") == _FROZEN_SHA256,
        "prefix_scope": prefix.get("scene_count") == 16
        and prefix.get("question_free_global_scene_prefix") is True,
        "grid_fixed": grid.get("fixed_before_any_candidate_forward") is True
        and grid.get("candidate_count") == 3
        and grid.get("direction_id") == _DIRECTION_ID
        and list(_sequence(grid.get("direction_components"), "V50 components"))
        == list(_DIRECTION_COMPONENTS),
        "gradient_specs": list(
            _sequence(grid.get("isolated_side_gradient_specs"), "V50 gradient specs")
        )
        == _expected_gradient_specs(),
        "normalization": grid.get("normalize_each_nonzero_component")
        == "unit_l2_within_each_scene_or_query_group_before_combination",
        "grid_inventory": list(
            _sequence(grid.get("candidates_declared_order"), "V50 candidates")
        )
        == expected_inventory,
        "grid_order": list(
            _sequence(grid.get("scene_alpha_grid_declared_order"), "V50 scene alphas")
        )
        == list(_SCENE_ALPHAS)
        and grid.get("query_alpha_fixed") == _QUERY_ALPHA,
        "grid_rates": grid.get("scene_readout_learning_rate") == _SCENE_LR
        and grid.get("query_learning_rate") == _QUERY_LR,
        "grid_formula": grid.get("candidate_formula")
        == "float32_P0-scene_or_query_alpha*lr_group*sign(normalized_component_sum)",
        "grid_integrity": grid.get(
            "candidate_hash_inventory_fixed_before_any_candidate_forward"
        )
        is True
        and grid.get("adaptive_grid_or_candidate_mutation") is False
        and grid.get("exact_source_restoration_before_and_after_each_candidate") is True,
        "evaluation": evaluation.get(
            "all_candidates_receive_full_non_greedy_gate_before_selection"
        )
        is True
        and evaluation.get("all_candidates_receive_full_25_unit_teacher_metrics_and_per_unit_nll")
        is True
        and evaluation.get("all_candidates_receive_full_fixed_48_row_broad_nll") is True
        and evaluation.get(
            "all_candidates_receive_original_v46_candidate_relative_prefix_trust"
        )
        is True
        and evaluation.get("greedy_runs_for_every_pre_gate_passing_candidate") is True
        and evaluation.get("greedy_runs_only_for_pre_gate_passing_candidates") is True
        and evaluation.get("pre_gate_failure_requires_greedy_skipped_due_pre_gate") is True
        and evaluation.get("every_pre_gate_passing_candidate_receives_full_changed_25_greedy")
        is True
        and evaluation.get("every_pre_gate_passing_candidate_receives_full_broad_48_greedy")
        is True
        and evaluation.get("winner_eligibility_requires_every_non_greedy_and_greedy_check")
        is True
        and evaluation.get("winner_selection")
        == "first_full_gate_passing_candidate_in_fixed_declared_order_after_all_evaluations"
        and evaluation.get("no_winner_if_no_candidate_passes_full_gate") is True
        and evaluation.get("no_early_stop_after_first_full_gate_pass") is True
        and evaluation.get("question_dependent_scene_processing") is False
        and evaluation.get("question_dependent_retrieval") is False,
        "check_names": list(
            _sequence(gate.get("non_greedy_check_names"), "V50 non-greedy names")
        )
        == list(_NON_GREEDY_CHECK_NAMES)
        and list(_sequence(gate.get("greedy_check_names"), "V50 greedy names"))
        == list(_GREEDY_CHECK_NAMES),
        "thresholds": gate.get("teacher_complete_units_minimum") == 10
        and gate.get("teacher_positive_sides_minimum") == 35
        and gate.get("teacher_cross_prefix_complete_units_minimum") == 17
        and gate.get("complete_physical_pair_coverage_minimum") == 5
        and gate.get("mirror_complete_units_minimum") == 2
        and gate.get("book_complete_units_minimum") == 1
        and gate.get("book_cross_prefix_complete_units_minimum") == 1
        and gate.get("priority_deficit_improvement_minimum_vs_original_v41_u0") == 0.5
        and gate.get("broad_nll_maximum") == _BROAD_NLL_MAXIMUM
        and gate.get("both_lost_side_margins_strictly_positive") is True
        and gate.get("scene_readout_state_changed") is True
        and gate.get("query_state_changed") is True
        and gate.get("frozen_state_exact") is True
        and gate.get("original_v46_candidate_relative_prefix_trust_rms_maximum")
        == _PREFIX_TRUST_RMS_MAXIMUM
        and gate.get("train_greedy_complete_units_minimum") == 5
        and gate.get("broad_greedy_exact_correct_minimum") == 23
        and gate.get("broad_greedy_row_count_exact") == 48
        and gate.get("thresholds_unchanged_from_v49") is True,
        "persistence": persistence.get("always_write_atomic_report") is True
        and persistence.get("checkpoint_write_iff_full_gate_winner_exists") is True
        and persistence.get("checkpoint_contains_declared_order_winner_only") is True
        and persistence.get("failed_grid_writes_no_checkpoint") is True
        and persistence.get("checkpoint_inventory_if_passed")
        == ["adapter.safetensors", "metadata.json", "runtime_metadata.json"]
        and persistence.get("optimizer_file_in_checkpoint") is False
        and persistence.get("runtime_metadata_exact_sanitization_required") is True
        and persistence.get(
            "checkpoint_provenance_must_bind_terminal_report_source_grid_and_winner"
        )
        is True,
        "scope": scope.get("train_only") is True
        and scope.get("fixed_three_candidate_grid") is True
        and scope.get("exact_three_autograd_grad_probes_reused_for_all_candidates") is True
        and scope.get("backward_or_parameter_gradient_accumulation_authorized") is False
        and scope.get("optimizer_construction_authorized") is False
        and scope.get("optimizer_state_file_open_authorized") is False
        and scope.get("optimizer_state_loading_authorized") is False
        and scope.get("optimizer_step_authorized") is False
        and scope.get("validation_access_authorized") is False
        and scope.get("oracle_access_authorized") is False
        and scope.get("final_test_access_authorized") is False
        and scope.get("selector_execution_authorized") is False
        and scope.get("runtime_promotion_authorized") is False
        and scope.get("chat_promotion_authorized") is False
        and scope.get("embodied_promotion_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"V50 terminal authorization changed: {checks}")
    return checks


def require_terminal(
    expected_sha256: str, path: str | Path = V49_TERMINAL
) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V50 expected V49 terminal SHA256 must be lowercase hexadecimal")
    terminal_path = _resolve(path)
    if terminal_path != _resolve(V49_TERMINAL):
        raise ValueError("V50 terminal path is pinned")
    if terminal_path.is_symlink() or not terminal_path.is_file():
        raise FileNotFoundError("V50 exact V49 terminal is unavailable or unsafe")
    payload = terminal_path.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise ValueError("V50 V49 terminal differs from explicit invocation SHA256")
    report = _mapping(json.loads(payload), "V49 terminal")
    authorization = _mapping(
        report.get("conditional_successor_authorization"), "V50 authorization"
    )
    checks = _validate_terminal_authorization(report, authorization)
    return {
        "path": str(V49_TERMINAL),
        "sha256": observed,
        "authorization_id": AUTHORIZATION_ID,
        "authorization": dict(authorization),
        "checks": checks,
    }


def candidate_from_split_alphas(
    source: Mapping[str, Any],
    direction: Mapping[str, Any],
    *,
    scene_alpha: float,
    query_alpha: float,
) -> dict[str, Any]:
    """Reconstruct one split-scaled candidate directly from immutable ``source``."""

    import torch

    from semantic_3d_chat.evaluation import v48_v47_u4_dual_margin_screen as v48
    from semantic_3d_chat.training.train_joint_scene_readout_v44 import _PARAMETER_NAMES

    if scene_alpha not in _SCENE_ALPHAS or query_alpha != _QUERY_ALPHA:
        raise ValueError("V50 candidate alpha is outside the fixed grid")
    v48._validate_surface_tensors(source, field="V50 source")
    v48._validate_surface_tensors(direction, field="V50 guarded direction")
    result = {name: value.detach().float().cpu().clone() for name, value in source.items()}
    with torch.no_grad():
        for index, name in enumerate(_PARAMETER_NAMES):
            alpha = scene_alpha if index == 0 else query_alpha
            learning_rate = _SCENE_LR if index == 0 else _QUERY_LR
            result[name].add_(
                torch.sign(direction[name]), alpha=-float(alpha * learning_rate)
            )
    v48._validate_surface_tensors(result, field="V50 candidate")
    return result


def non_greedy_pre_gate_checks(
    reconstruction: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, bool]:
    """Apply the unchanged V49 non-greedy numerical and integrity gate."""

    pair = _mapping(evidence.get("pair_metrics"), "V50 pair metrics")
    families = _mapping(pair.get("complete_units_by_family"), "V50 families")
    cross = _mapping(pair.get("cross_prefix_complete_units_by_family"), "V50 cross families")
    retention = _mapping(evidence.get("retention_diagnostics"), "V50 retention")
    deficit = _finite(evidence.get("priority_side_deficit"), "priority deficit")
    return {
        "source_v47_u4_exact_before_reconstruction": reconstruction.get(
            "source_v47_u4_exact_before_reconstruction"
        )
        is True,
        "reconstructed_candidate_full_tensor_state_exact": reconstruction.get(
            "reconstructed_candidate_full_tensor_state_exact"
        )
        is True,
        "reconstructed_candidate_authorized_surface_state_exact": reconstruction.get(
            "reconstructed_candidate_authorized_surface_state_exact"
        )
        is True,
        "teacher_complete_units_at_least_10": int(pair["complete_units"]) >= 10,
        "teacher_positive_sides_at_least_35": int(pair["positive_sides"]) >= 35,
        "teacher_cross_prefix_complete_units_at_least_17": int(
            pair["cross_prefix_complete_units"]
        )
        >= 17,
        "complete_physical_pair_coverage_at_least_5": int(
            pair["complete_physical_pair_coverage"]
        )
        >= 5,
        "mirror_complete_units_at_least_2": int(families.get("mirror_lr", 0)) >= 2,
        "book_complete_units_at_least_1": int(families.get("book_support", 0)) >= 1,
        "book_cross_prefix_complete_units_at_least_1": int(cross.get("book_support", 0))
        >= 1,
        "priority_deficit_improvement_at_least_0_5_vs_original_v41_u0": (
            _ORIGINAL_V41_PRIORITY_DEFICIT - deficit >= 0.5
        ),
        "broad_nll_at_most_v45_maximum": _finite(evidence.get("broad_nll"), "broad NLL")
        <= _BROAD_NLL_MAXIMUM,
        "both_lost_sides_strictly_positive": retention.get(
            "both_lost_sides_strictly_positive"
        )
        is True,
        "scene_readout_state_changed": reconstruction.get("scene_readout_state_changed")
        is True,
        "query_state_changed": reconstruction.get("query_state_changed") is True,
        "frozen_state_exact": reconstruction.get("frozen_state_sha256") == _FROZEN_SHA256,
        "original_v46_candidate_relative_prefix_trust_rms_at_most_0_002": _finite(
            evidence.get("original_v46_candidate_relative_prefix_trust_rms"),
            "original-candidate prefix RMS",
        )
        <= _PREFIX_TRUST_RMS_MAXIMUM,
    }


def greedy_final_gate_checks(evidence: Mapping[str, Any]) -> dict[str, bool]:
    return v49.greedy_final_gate_checks(evidence)


class RealGridBackend:
    """Production backend; construction itself performs no model or data load."""

    def __init__(self, terminal: Mapping[str, Any], paths: GridPaths) -> None:
        self.terminal = dict(terminal)
        self.paths = paths
        delegate_paths = v49.GatePaths(
            terminal=paths.terminal,
            report=paths.report,
            checkpoint=paths.checkpoint_root / "update_000",
            config=paths.config,
        )
        self._delegate = v49.RealGateBackend(terminal, delegate_paths)
        self._prepared = False
        self._direction: Mapping[str, Any] = {}
        self._candidates: dict[str, Mapping[str, Any]] = {}
        self._candidate_expected: dict[str, Mapping[str, str]] = {}
        self._candidate_states: dict[str, Mapping[str, Any]] = {}
        self._active_candidate_id: str | None = None

    def _source_attestation(self) -> dict[str, Any]:
        from semantic_3d_chat.evaluation import v48_v47_u4_dual_margin_screen as v48

        result = v48._bundle_state_attestation(
            self._delegate._bundle,
            self._delegate._named,
            expected_authorized_sha256=_SOURCE_AUTHORIZED_SHA256,
            expected_full_sha256=_SOURCE_FULL_SHA256,
        )
        if result["passed"] is not True:
            raise RuntimeError("V50 exact V47-u4 source attestation failed")
        return result

    def authenticate_and_prepare(self) -> Mapping[str, Any]:
        from semantic_3d_chat.language.lora import tensor_state_sha256
        from semantic_3d_chat.training.train_joint_scene_readout_v44 import _PARAMETER_NAMES

        if self._prepared:
            raise RuntimeError("V50 backend preparation may run only once")
        v49_reconstruction = self._delegate.authenticate_and_reconstruct()
        live_v49_candidate = {
            name: self._delegate._named[name].detach().float().cpu().clone()
            for name in _PARAMETER_NAMES
        }
        if tensor_state_sha256(live_v49_candidate) != _V49_CANDIDATE_AUTHORIZED_SHA256:
            raise RuntimeError("V50 live authenticated V49 candidate changed")
        self._direction = {
            name: self._delegate._source_surface[name] - live_v49_candidate[name]
            for name in _PARAMETER_NAMES
        }
        self.restore_source()
        self._candidates = {
            str(spec["candidate_id"]): candidate_from_split_alphas(
                self._delegate._source_surface,
                self._direction,
                scene_alpha=float(spec["scene_alpha"]),
                query_alpha=float(spec["query_alpha"]),
            )
            for spec in CANDIDATE_GRID
        }
        if tuple(self._candidates) != tuple(spec["candidate_id"] for spec in CANDIDATE_GRID):
            raise RuntimeError("V50 candidate inventory changed")
        source_full_state = {
            f"{module_name}.{name}": value.detach().cpu().clone()
            for module_name, module in self._delegate._bundle.checkpoint_modules.items()
            for name, value in module.state_dict().items()
        }
        if tensor_state_sha256(source_full_state) != _SOURCE_FULL_SHA256:
            raise RuntimeError("V50 source tensor inventory changed before candidate prehash")
        for candidate_id, values in self._candidates.items():
            candidate_full = dict(source_full_state)
            candidate_full.update(values)
            self._candidate_expected[candidate_id] = {
                "full_tensor_state_sha256": tensor_state_sha256(candidate_full),
                "authorized_surface_state_sha256": tensor_state_sha256(values),
                "frozen_state_sha256": _FROZEN_SHA256,
            }
        self.restore_source()
        self._prepared = True
        return {
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "source_full_tensor_state_sha256": _SOURCE_FULL_SHA256,
            "source_authorized_surface_state_sha256": _SOURCE_AUTHORIZED_SHA256,
            "frozen_state_sha256": _FROZEN_SHA256,
            "prefix_reference_checkpoint": str(PREFIX_REFERENCE_CHECKPOINT),
            "prefix_reference_full_tensor_state_sha256": _PREFIX_REFERENCE_FULL_SHA256,
            "prefix_reference_hash_inventory_sha256": v49_reconstruction[
                "prefix_reference_hash_inventory_sha256"
            ],
            "v49_candidate_reconstructed_before_v50_direction": (
                v49_reconstruction["full_tensor_state_sha256"]
                == _V49_CANDIDATE_FULL_SHA256
                and v49_reconstruction["authorized_surface_state_sha256"]
                == _V49_CANDIDATE_AUTHORIZED_SHA256
            ),
            "gradient_specs": _expected_gradient_specs(),
            "gradient_audit_sha256": v49_reconstruction["gradient_audit_sha256"],
            "direction_audit_sha256": v49_reconstruction["direction_audit_sha256"],
            "direction_id": _DIRECTION_ID,
            "direction_components": list(_DIRECTION_COMPONENTS),
            "candidate_grid": [dict(value) for value in CANDIDATE_GRID],
            "candidate_hash_inventory": {
                key: dict(value) for key, value in self._candidate_expected.items()
            },
            "candidate_hash_inventory_fixed_before_any_candidate_forward": True,
            "candidate_count": len(CANDIDATE_GRID),
            "exact_three_autograd_grad_probes_reused_for_all_candidates": True,
            "all_16_training_maps_cached": len(self._delegate._caches) == 16,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "optimizer_constructed_or_loaded": False,
        }

    def reconstruct_candidate(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        import torch

        from semantic_3d_chat.evaluation import v48_v47_u4_dual_margin_screen as v48
        from semantic_3d_chat.language.lora import tensor_state_sha256
        from semantic_3d_chat.training.checkpointing import module_collection_state_sha256
        from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
            _PARAMETER_NAMES,
            frozen_v44_state_sha256,
        )

        if not self._prepared:
            raise RuntimeError("V50 candidate reconstruction requires preparation")
        index = int(candidate.get("declared_order", -1))
        if index < 0 or index >= len(CANDIDATE_GRID) or dict(candidate) != dict(
            CANDIDATE_GRID[index]
        ):
            raise ValueError("V50 candidate is outside the exact declared grid")
        source_before = self.restore_source()
        candidate_id = str(candidate["candidate_id"])
        values = self._candidates[candidate_id]
        v48._copy_candidate(self._delegate._named, values)
        authorized_sha = tensor_state_sha256(
            {name: value.detach().cpu() for name, value in self._delegate._named.items()}
        )
        full_sha = module_collection_state_sha256(self._delegate._bundle.checkpoint_modules)
        frozen_sha = frozen_v44_state_sha256(self._delegate._bundle)
        expected = self._candidate_expected[candidate_id]
        state = {
            "candidate_id": candidate_id,
            "declared_order": index,
            "direction_id": _DIRECTION_ID,
            "scene_alpha": float(candidate["scene_alpha"]),
            "query_alpha": float(candidate["query_alpha"]),
            "scene_learning_rate": _SCENE_LR,
            "query_learning_rate": _QUERY_LR,
            "source_v47_u4_exact_before_reconstruction": source_before.get("passed") is True,
            "full_tensor_state_sha256": full_sha,
            "authorized_surface_state_sha256": authorized_sha,
            "frozen_state_sha256": frozen_sha,
            "reconstructed_candidate_full_tensor_state_exact": full_sha
            == expected["full_tensor_state_sha256"],
            "reconstructed_candidate_authorized_surface_state_exact": authorized_sha
            == expected["authorized_surface_state_sha256"],
            "scene_readout_state_changed": not bool(
                torch.equal(
                    values[_PARAMETER_NAMES[0]],
                    self._delegate._source_surface[_PARAMETER_NAMES[0]],
                )
            ),
            "query_state_changed": any(
                not bool(
                    torch.equal(
                        values[name], self._delegate._source_surface[name]
                    )
                )
                for name in _PARAMETER_NAMES[1:]
            ),
            "reconstructed_directly_from_v47_u4": True,
        }
        previous = self._candidate_states.get(candidate_id)
        if previous is not None and (
            previous["full_tensor_state_sha256"] != full_sha
            or previous["authorized_surface_state_sha256"] != authorized_sha
        ):
            raise RuntimeError("V50 candidate reconstruction is not deterministic")
        self._candidate_states[candidate_id] = state
        self._active_candidate_id = candidate_id
        return state

    def evaluate_non_greedy(self, candidate_id: str) -> Mapping[str, Any]:
        import torch

        from semantic_3d_chat.training.train_joint_block_cross_v36 import training_broad_nll
        from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
            source_prefix_trust_penalty,
        )
        from semantic_3d_chat.training.train_projected_gradient_v41 import (
            priority_side_deficit,
            training_pair_gate_diagnostics,
            validate_per_unit_nll_diagnostics,
        )
        from semantic_3d_chat.training.train_retention_repair_v45 import (
            v45_retention_diagnostics,
        )

        if self._active_candidate_id != candidate_id:
            raise RuntimeError("V50 non-greedy evaluation candidate is not live")
        with torch.inference_mode():
            _penalty, rms = source_prefix_trust_penalty(
                caches=self._delegate._caches,
                references=self._delegate._prefix_references,
                block_core=self._delegate._block_core,
                device=self._delegate._bundle.language.device,
                scale=0.05,
            )
        pair, per_unit = training_pair_gate_diagnostics(
            units=self._delegate._units,
            caches=self._delegate._caches,
            block_cross_residual=self._delegate._block_core,
            bundle=self._delegate._bundle,
            settings=self._delegate._settings,
        )
        validate_per_unit_nll_diagnostics(per_unit, pair)
        broad = training_broad_nll(
            records=self._delegate._broad_records,
            caches=self._delegate._caches,
            block_cross_residual=self._delegate._block_core,
            bundle=self._delegate._bundle,
        )
        return {
            "pair_metrics": pair,
            "per_unit_nll_diagnostics": per_unit,
            "broad_nll": broad,
            "broad_row_count": len(self._delegate._broad_records),
            "priority_side_deficit": float(priority_side_deficit(pair)["combined"]),
            "retention_diagnostics": v45_retention_diagnostics(pair),
            "original_v46_candidate_relative_prefix_trust_rms": float(rms.detach().cpu()),
        }

    def evaluate_greedy(self, candidate_id: str) -> Mapping[str, Any]:
        from semantic_3d_chat.training.train_joint_block_cross_v36 import training_greedy_metrics

        if self._active_candidate_id != candidate_id:
            raise RuntimeError("V50 greedy evaluation candidate is not live")
        return training_greedy_metrics(
            units=self._delegate._units,
            broad_records=self._delegate._broad_records,
            caches=self._delegate._caches,
            block_cross_residual=self._delegate._block_core,
            bundle=self._delegate._bundle,
            config=self._delegate._loader,
        )

    def restore_source(self) -> Mapping[str, Any]:
        result = self._delegate.restore_source()
        self._active_candidate_id = None
        return result

    def stage_checkpoint(
        self,
        directory: Path,
        candidate: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from safetensors.torch import load_file

        from semantic_3d_chat.config import config_hash
        from semantic_3d_chat.evaluation import v48_v47_u4_dual_margin_screen as v48
        from semantic_3d_chat.language.lora import tensor_state_sha256
        from semantic_3d_chat.training.checkpointing import (
            runtime_checkpoint_metadata,
            save_adapter_checkpoint,
            validate_runtime_checkpoint_metadata,
        )
        from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
            _PARAMETER_NAMES,
            block_source_stack_state_sha256,
        )

        candidate_id = str(candidate["candidate_id"])
        expected = self._candidate_states.get(candidate_id)
        if expected is None:
            raise RuntimeError("V50 cannot stage an unevaluated candidate")
        self.restore_source()
        v48._copy_candidate(self._delegate._named, self._candidates[candidate_id])
        try:
            metadata = copy.deepcopy(dict(self._delegate._source_metadata))
            metadata.update(
                {
                    "schema_version": 1,
                    "config_hash": config_hash(dict(self._delegate._config)),
                    "optimizer_step": 0,
                    "epoch": 0,
                    "history": [],
                    "question_dependent_scene_processing": False,
                    **self._delegate._bundle.lora_installation.checkpoint_metadata(),
                    "block_cross_residual_state_sha256": (
                        self._delegate._block_core.state_sha256()
                    ),
                    "frozen_block_cross_source_stack_state_sha256": (
                        block_source_stack_state_sha256(
                            self._delegate._bundle, self._delegate._block_core
                        )
                    ),
                }
            )
            metadata["v50_scene_query_alpha_grid"] = {
                "schema_version": 1,
                "authorization": dict(provenance),
                "winner": dict(candidate),
                "winner_full_tensor_state_sha256": expected["full_tensor_state_sha256"],
                "winner_authorized_surface_state_sha256": expected[
                    "authorized_surface_state_sha256"
                ],
                "frozen_state_sha256": _FROZEN_SHA256,
                "original_v46_candidate_prefix_reference": {
                    "checkpoint": str(PREFIX_REFERENCE_CHECKPOINT),
                    "full_tensor_state_sha256": _PREFIX_REFERENCE_FULL_SHA256,
                },
                "complete_fixed_grid_evaluated_before_selection": True,
                "selection_rule": "first_full_pass_in_declared_order",
                "training_scenes_only": True,
                "validation_qa_loaded": False,
                "oracle_environment_files_loaded": False,
                "question_dependent_scene_processing": False,
                "question_dependent_retrieval": False,
                "optimizer_constructed_or_loaded": False,
                "selector_execution_authorized": False,
                "runtime_promotion_authorized": False,
                "independent_terminal_seal_required": True,
            }
            save_adapter_checkpoint(
                directory, self._delegate._bundle.checkpoint_modules, metadata
            )
            saved = load_file(directory / "adapter.safetensors", device="cpu")
            authorized = {name: saved[name] for name in _PARAMETER_NAMES}
            frozen = {name: value for name, value in saved.items() if name not in authorized}
            state = {
                "full_tensor_state_sha256": tensor_state_sha256(saved),
                "authorized_surface_state_sha256": tensor_state_sha256(authorized),
                "frozen_state_sha256": tensor_state_sha256(frozen),
            }
            if state != {
                "full_tensor_state_sha256": expected["full_tensor_state_sha256"],
                "authorized_surface_state_sha256": expected[
                    "authorized_surface_state_sha256"
                ],
                "frozen_state_sha256": _FROZEN_SHA256,
            }:
                raise RuntimeError("V50 staged winner tensor state changed")
            runtime = _mapping(
                json.loads(
                    (directory / "runtime_metadata.json").read_text(encoding="utf-8")
                ),
                "V50 staged runtime metadata",
            )
            validate_runtime_checkpoint_metadata(runtime)
            if runtime != runtime_checkpoint_metadata(metadata):
                raise RuntimeError("V50 runtime metadata is not exact sanitization")
            return {
                **state,
                "runtime_metadata_exact_sanitization": True,
                "metadata_provenance_bound": True,
            }
        finally:
            restoration = self.restore_source()
            if restoration.get("passed") is not True:
                raise RuntimeError("V50 failed to restore source after checkpoint staging")

    def access_audit(self) -> Mapping[str, Any]:
        return self._delegate.access_audit()

    def close(self) -> None:
        self._delegate.close()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _staged_checkpoint_inventory(directory: Path) -> dict[str, Any]:
    expected = ["adapter.safetensors", "metadata.json", "runtime_metadata.json"]
    observed = sorted(path.name for path in directory.iterdir())
    if observed != expected:
        raise ValueError(f"V50 staged checkpoint inventory changed: {observed}")
    hashes = {}
    for name in expected:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"V50 staged checkpoint file is unsafe: {name}")
        hashes[name] = _sha256(path)
    return {
        "directory_inventory": expected,
        "file_sha256": hashes,
        "optimizer_file_present": False,
    }


def _concise_non_greedy(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return v49._concise_non_greedy(evidence)


def _concise_greedy(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return v49._concise_greedy(evidence)


def execute_grid_gate(
    *, terminal: Mapping[str, Any], backend: GridBackend, checkpoint_path: Path
) -> dict[str, Any]:
    """Evaluate the complete fixed grid and conditionally publish one winner."""

    preparation: Mapping[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    staged: Path | None = None
    checkpoint_stage: Mapping[str, Any] | None = None
    checkpoint_written = False
    final_restoration: Mapping[str, Any] = {"attempted": False, "passed": False}
    access: Mapping[str, Any] = {"passed": False}
    execution_errors: list[dict[str, Any]] = []
    try:
        preparation = backend.authenticate_and_prepare()
        for spec in CANDIDATE_GRID:
            reconstruction: Mapping[str, Any] = {}
            non_greedy: Mapping[str, Any] = {}
            pre_checks: dict[str, bool] = {}
            greedy: Mapping[str, Any] | None = None
            greedy_checks: dict[str, bool] = {}
            restoration: Mapping[str, Any] = {"attempted": False, "passed": False}
            error: dict[str, str] | None = None
            try:
                reconstruction = backend.reconstruct_candidate(spec)
                non_greedy = backend.evaluate_non_greedy(str(spec["candidate_id"]))
                concise_non_greedy = _concise_non_greedy(non_greedy)
                pre_checks = non_greedy_pre_gate_checks(reconstruction, non_greedy)
                pre_passed = bool(pre_checks) and all(pre_checks.values())
                if pre_passed:
                    greedy = backend.evaluate_greedy(str(spec["candidate_id"]))
                    concise_greedy = _concise_greedy(greedy)
                    greedy_checks = greedy_final_gate_checks(greedy)
                else:
                    concise_greedy = None
            except Exception as exc:  # noqa: BLE001 - grid errors are sealed in report
                concise_non_greedy = None
                concise_greedy = None
                error = {"type": type(exc).__name__, "message": str(exc)}
            finally:
                try:
                    restoration = {
                        "attempted": True,
                        **dict(backend.restore_source()),
                    }
                except Exception as exc:  # noqa: BLE001 - restoration must fail closed
                    restoration = {
                        "attempted": True,
                        "passed": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    if error is None:
                        error = {
                            "type": type(exc).__name__,
                            "message": f"candidate source restoration failed: {exc}",
                        }
            pre_passed = bool(pre_checks) and all(pre_checks.values())
            greedy_executed = greedy is not None
            greedy_passed = (
                greedy_executed and bool(greedy_checks) and all(greedy_checks.values())
            )
            row_passed = bool(
                pre_passed
                and greedy_passed
                and restoration.get("passed") is True
                and error is None
            )
            if pre_passed is not greedy_executed and error is None:
                raise RuntimeError("V50 per-candidate greedy iff-pre-gate contract violated")
            row = {
                "candidate": dict(spec),
                "candidate_reconstruction": dict(reconstruction),
                "non_greedy_pre_gate": {
                    "evaluated": bool(non_greedy),
                    "checks": pre_checks,
                    "passed": pre_passed,
                    "evidence": concise_non_greedy,
                },
                "greedy_gate": {
                    "authorized": pre_passed,
                    "executed": greedy_executed,
                    "skipped_due_pre_gate": not pre_passed,
                    "checks": greedy_checks,
                    "passed": greedy_passed,
                    "evidence": concise_greedy,
                },
                "source_restoration": dict(restoration),
                "evaluation_error": error,
                "full_gate_passed": row_passed,
            }
            candidates.append(row)
            if error is not None:
                execution_errors.append(
                    {"candidate_id": spec["candidate_id"], **error}
                )
    except Exception as exc:  # noqa: BLE001 - preparation failures still restore/audit
        execution_errors.append(
            {"candidate_id": None, "type": type(exc).__name__, "message": str(exc)}
        )

    expected_ids = [str(value["candidate_id"]) for value in CANDIDATE_GRID]
    observed_ids = [str(row["candidate"]["candidate_id"]) for row in candidates]
    grid_complete = len(candidates) == len(CANDIDATE_GRID) and observed_ids == expected_ids
    evaluation_complete = grid_complete and all(
        row["non_greedy_pre_gate"]["evaluated"]
        and row["source_restoration"].get("passed") is True
        and row["evaluation_error"] is None
        for row in candidates
    )
    passing = [row for row in candidates if row["full_gate_passed"]]
    winner_row = passing[0] if evaluation_complete and passing else None
    winner = None if winner_row is None else dict(winner_row["candidate"])

    if winner is not None:
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            staged = Path(
                tempfile.mkdtemp(
                    prefix=f".{checkpoint_path.name}.staged.", dir=checkpoint_path.parent
                )
            )
            provenance = {
                "terminal_path": terminal["path"],
                "terminal_sha256": terminal["sha256"],
                "authorization_id": AUTHORIZATION_ID,
                "source_checkpoint": str(SOURCE_CHECKPOINT),
                "source_full_tensor_state_sha256": _SOURCE_FULL_SHA256,
                "prefix_reference_checkpoint": str(PREFIX_REFERENCE_CHECKPOINT),
                "prefix_reference_full_tensor_state_sha256": (
                    _PREFIX_REFERENCE_FULL_SHA256
                ),
                "complete_grid_sha256": _canonical_sha256(candidates),
                "winner": winner,
                "selection_rule": "first_full_pass_in_declared_order",
            }
            backend_stage = dict(
                backend.stage_checkpoint(staged, winner, provenance)
            )
            checkpoint_stage = {
                **backend_stage,
                **_staged_checkpoint_inventory(staged),
            }
        except Exception as exc:  # noqa: BLE001 - staging must fail closed
            execution_errors.append(
                {
                    "candidate_id": winner["candidate_id"],
                    "type": type(exc).__name__,
                    "message": f"checkpoint staging failed: {exc}",
                }
            )

    try:
        final_restoration = {"attempted": True, **dict(backend.restore_source())}
    except Exception as exc:  # noqa: BLE001 - final restoration must fail closed
        final_restoration = {
            "attempted": True,
            "passed": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        execution_errors.append(
            {
                "candidate_id": None,
                "type": type(exc).__name__,
                "message": f"final source restoration failed: {exc}",
            }
        )
    try:
        access = dict(backend.access_audit())
    except Exception as exc:  # noqa: BLE001 - audit failures block persistence
        access = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        execution_errors.append(
            {
                "candidate_id": None,
                "type": type(exc).__name__,
                "message": f"access audit failed: {exc}",
            }
        )
    try:
        backend.close()
    except Exception as exc:  # noqa: BLE001 - close failures block persistence
        execution_errors.append(
            {
                "candidate_id": None,
                "type": type(exc).__name__,
                "message": f"backend close failed: {exc}",
            }
        )

    final_passed = bool(
        evaluation_complete
        and winner is not None
        and checkpoint_stage is not None
        and final_restoration.get("passed") is True
        and access.get("passed") is True
        and not execution_errors
        and staged is not None
    )
    if final_passed:
        if checkpoint_path.is_symlink() or checkpoint_path.exists():
            execution_errors.append(
                {
                    "candidate_id": winner["candidate_id"],
                    "type": "FileExistsError",
                    "message": "V50 checkpoint destination appeared before atomic publish",
                }
            )
            final_passed = False
        else:
            try:
                os.replace(staged, checkpoint_path)
                staged = None
                checkpoint_written = True
            except OSError as exc:
                execution_errors.append(
                    {
                        "candidate_id": winner["candidate_id"],
                        "type": type(exc).__name__,
                        "message": f"atomic checkpoint publish failed: {exc}",
                    }
                )
                final_passed = False
    if staged is not None:
        shutil.rmtree(staged)
    if checkpoint_written is not final_passed:
        raise RuntimeError("V50 checkpoint persistence violated iff-final-pass contract")

    return {
        "schema_version": 1,
        "artifact": "v50_scene_query_alpha_grid",
        "passed": final_passed,
        "authorization": {
            "terminal_path": terminal["path"],
            "terminal_sha256": terminal["sha256"],
            "authorization_id": AUTHORIZATION_ID,
            "checks": dict(_mapping(terminal.get("checks"), "terminal checks")),
        },
        "preparation": dict(preparation),
        "candidate_grid": {
            "declared": [dict(value) for value in CANDIDATE_GRID],
            "declared_count": len(CANDIDATE_GRID),
            "evaluated_count": len(candidates),
            "evaluated_ids": observed_ids,
            "complete_fixed_grid_evaluated_before_selection": evaluation_complete,
            "candidates": candidates,
        },
        "selection": {
            "performed_after_complete_grid": evaluation_complete,
            "rule": "first_full_pass_in_declared_order",
            "passing_candidate_ids": [
                row["candidate"]["candidate_id"] for row in passing
            ],
            "winner": winner,
        },
        "final_train_gate": {
            "passed": final_passed,
            "grid_complete": grid_complete,
            "evaluation_complete": evaluation_complete,
            "winner_exists": winner is not None,
            "source_restored_exact": final_restoration.get("passed") is True,
            "access_audit_passed": access.get("passed") is True,
            "execution_errors": execution_errors,
        },
        "checkpoint": {
            "root": str(DEFAULT_CHECKPOINT_ROOT),
            "path": str(checkpoint_path),
            "staged_after_complete_grid": checkpoint_stage is not None,
            "published_after_restoration_and_access_audit": checkpoint_written,
            "written": checkpoint_written,
            "write_iff_final_gate_passed": checkpoint_written is final_passed,
            "inventory": checkpoint_stage,
            "optimizer_file_written": False,
        },
        "final_source_restoration": dict(final_restoration),
        "access_audit": dict(access),
        "optimizer_constructed_or_loaded": False,
        "optimizer_state_file_opened": False,
        "optimizer_step_executed": False,
        "validation_qa_loaded": False,
        "validation_environment_maps_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_executed": False,
        "runtime_promotion_executed": False,
        "chat_promotion_executed": False,
        "embodied_promotion_executed": False,
        "question_dependent_retrieval": False,
    }


def preflight(
    *, expected_v49_terminal_sha256: str, paths: GridPaths | None = None
) -> dict[str, Any]:
    """Authenticate V50 without loading Gemma, QA records, or scene maps."""

    selected = GridPaths() if paths is None else paths
    resolved = GridPaths(
        terminal=_resolve(selected.terminal),
        report=_resolve(selected.report),
        checkpoint_root=_resolve(selected.checkpoint_root),
        config=_resolve(selected.config),
    )
    expected = GridPaths(
        terminal=_resolve(V49_TERMINAL),
        report=_resolve(DEFAULT_REPORT),
        checkpoint_root=_resolve(DEFAULT_CHECKPOINT_ROOT),
        config=_resolve(DEFAULT_CONFIG),
    )
    if resolved != expected:
        raise ValueError("V50 preflight paths are pinned")
    checkpoint = resolved.checkpoint_root / "update_000"
    if resolved.report.is_symlink() or resolved.report.exists():
        raise FileExistsError("V50 report is one-shot and already exists")
    if resolved.checkpoint_root.is_symlink() or resolved.checkpoint_root.exists():
        raise FileExistsError("V50 checkpoint root must be absent")
    _locked_hash(resolved.config, _CONFIG_SHA256, "V50 config")
    terminal = require_terminal(expected_v49_terminal_sha256, selected.terminal)
    _locked_hash(_resolve(V49_REPORT), _V49_REPORT_SHA256, "V50 V49 report")
    _locked_hash(_resolve(PROTECTED_REPORT), _PROTECTED_REPORT_SHA256, "V50 protected report")
    source = _resolve(SOURCE_CHECKPOINT)
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError("V50 source checkpoint is unavailable")
    source_inventory = sorted(path.name for path in source.iterdir())
    if source_inventory != sorted(_SOURCE_FILES):
        raise ValueError("V50 source checkpoint inventory changed")
    readable_source = {
        name: digest for name, digest in _SOURCE_FILES.items() if name != "optimizer.pt"
    }
    for name, digest in readable_source.items():
        _locked_hash(source / name, digest, f"V50 source {name}")
    prefix = _resolve(PREFIX_REFERENCE_CHECKPOINT)
    if prefix.is_symlink() or not prefix.is_dir():
        raise FileNotFoundError("V50 prefix reference checkpoint is unavailable")
    prefix_inventory = sorted(path.name for path in prefix.iterdir())
    if prefix_inventory != sorted(_PREFIX_REFERENCE_FILES):
        raise ValueError("V50 prefix reference checkpoint inventory changed")
    for name, digest in _PREFIX_REFERENCE_FILES.items():
        _locked_hash(prefix / name, digest, f"V50 prefix reference {name}")
    return {
        "schema_version": 1,
        "artifact": "v50_scene_query_alpha_grid_preflight",
        "passed": True,
        "terminal": {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
            "authorization_id": AUTHORIZATION_ID,
            "authenticated": True,
        },
        "source": {
            "checkpoint": str(SOURCE_CHECKPOINT),
            "directory_inventory": source_inventory,
            "readable_file_sha256": readable_source,
            "optimizer_file_sha256_provenance": _SOURCE_FILES["optimizer.pt"],
            "optimizer_file_opened": False,
        },
        "prefix_reference": {
            "checkpoint": str(PREFIX_REFERENCE_CHECKPOINT),
            "directory_inventory": prefix_inventory,
            "file_sha256": dict(_PREFIX_REFERENCE_FILES),
        },
        "candidate_grid": [dict(value) for value in CANDIDATE_GRID],
        "candidate_count": len(CANDIDATE_GRID),
        "checkpoint": str(checkpoint),
        "model_loaded": False,
        "qa_loaded": False,
        "maps_loaded": False,
        "optimizer_constructed_or_loaded": False,
        "optimizer_state_file_opened": False,
        "greedy_generation_executed": False,
        "checkpoint_written": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_executed": False,
        "runtime_promotion_executed": False,
    }


def run_grid(
    *,
    expected_v49_terminal_sha256: str,
    paths: GridPaths | None = None,
    backend_factory: Callable[[Mapping[str, Any], GridPaths], GridBackend] | None = None,
) -> dict[str, Any]:
    """Run the exact authenticated V50 grid and atomically emit its report."""

    selected = GridPaths() if paths is None else paths
    resolved = GridPaths(
        terminal=_resolve(selected.terminal),
        report=_resolve(selected.report),
        checkpoint_root=_resolve(selected.checkpoint_root),
        config=_resolve(selected.config),
    )
    expected = GridPaths(
        terminal=_resolve(V49_TERMINAL),
        report=_resolve(DEFAULT_REPORT),
        checkpoint_root=_resolve(DEFAULT_CHECKPOINT_ROOT),
        config=_resolve(DEFAULT_CONFIG),
    )
    if resolved != expected:
        raise ValueError("V50 terminal, report, checkpoint, and config paths are pinned")
    if resolved.report.is_symlink() or resolved.report.exists():
        raise FileExistsError("V50 report is one-shot and will not be overwritten")
    if resolved.checkpoint_root.is_symlink() or resolved.checkpoint_root.exists():
        raise FileExistsError("V50 checkpoint root must be absent")
    _locked_hash(resolved.config, _CONFIG_SHA256, "V50 config")
    terminal = require_terminal(expected_v49_terminal_sha256, selected.terminal)
    factory = RealGridBackend if backend_factory is None else backend_factory
    backend = factory(terminal, resolved)
    report = execute_grid_gate(
        terminal=terminal,
        backend=backend,
        checkpoint_path=resolved.checkpoint_root / "update_000",
    )
    _atomic_json(resolved.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-v49-terminal-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--terminal", type=Path, default=V49_TERMINAL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    paths = GridPaths(
        terminal=args.terminal,
        report=args.report,
        checkpoint_root=args.checkpoint_root,
        config=args.config,
    )
    if args.preflight:
        result = preflight(
            expected_v49_terminal_sha256=args.expected_v49_terminal_sha256,
            paths=paths,
        )
        summary = {
            "artifact": result["artifact"],
            "passed": result["passed"],
            "terminal_sha256": result["terminal"]["sha256"],
            "candidate_count": result["candidate_count"],
            "model_loaded": False,
            "qa_loaded": False,
            "maps_loaded": False,
        }
    else:
        result = run_grid(
            expected_v49_terminal_sha256=args.expected_v49_terminal_sha256,
            paths=paths,
        )
        summary = {
            "artifact": result["artifact"],
            "passed": result["passed"],
            "report": str(DEFAULT_REPORT),
            "report_sha256": _sha256(_resolve(DEFAULT_REPORT)),
            "evaluated_count": result["candidate_grid"]["evaluated_count"],
            "winner": result["selection"]["winner"],
            "checkpoint_written": result["checkpoint"]["written"],
        }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_ID",
    "CANDIDATE_GRID",
    "GridBackend",
    "GridPaths",
    "RealGridBackend",
    "candidate_from_split_alphas",
    "execute_grid_gate",
    "greedy_final_gate_checks",
    "main",
    "non_greedy_pre_gate_checks",
    "preflight",
    "require_terminal",
    "run_grid",
]

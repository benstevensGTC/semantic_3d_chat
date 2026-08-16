"""Deterministic V49 guarded-alpha2 train-only candidate gate.

V49 authenticates an explicit V48 terminal SHA, reconstructs exactly one
pre-authorized candidate from the immutable V47 update-004 source, and applies
a staged train-only gate.  Exhaustive greedy evaluation is forbidden until all
non-greedy teacher, broad, original-candidate-prefix, source, and frozen-state
checks pass.  A candidate checkpoint is written if and only if the complete
final gate passes; a concise report is always written atomically.

No optimizer, validation, oracle, final-test, selector, runtime promotion,
chat promotion, or embodied promotion access is authorized.
"""

from __future__ import annotations

import argparse
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

AUTHORIZATION_ID = "v49_guarded_alpha2_train_candidate_gate"
V49_SCRIPT = Path("src/semantic_3d_chat/evaluation/v49_guarded_candidate_greedy_gate.py")
V49_TEST = Path("tests/test_v49_guarded_candidate_greedy_gate.py")
V48_TERMINAL = Path("reports/gemma4/metrics/v48_v47_u4_dual_margin_terminal_gate.json")
DEFAULT_REPORT = Path("reports/gemma4/metrics/v49_guarded_candidate_train_gate.json")
DEFAULT_CHECKPOINT = Path("data_gemma4/checkpoints/gemma4_v49_guarded_alpha2_train_gate/update_000")
DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_book_continuation_v47.yaml")
SOURCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query/update_004"
)
PREFIX_REFERENCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query/update_000"
)
V48_REPORT = Path("reports/gemma4/metrics/v48_v47_u4_dual_margin_no_step_diagnostic.json")
PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)

_CONFIG_SHA256 = "6b15813237b217d8daad446c799127215bbb1366f2b442b61e975787efa4f6b7"
_V48_REPORT_SHA256 = "7abd2fa7741f84ea56933383199ec449d47dd99361def15d6a3874b9e154e02c"
_PROTECTED_REPORT_SHA256 = "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
_SOURCE_FILES = {
    "adapter.safetensors": "8f903f5d1ba93d37ccd6204e3b58c9a5529ff9ee2b74edca0787ecb5a2c62c66",
    "metadata.json": "c6affe7f60c094580e2ea5f5d1330f475bf359e0a3a58bfc3bf3b3ada1de0be1",
    "optimizer.pt": "fe66be9cae13951fbfc217e0c512e43366c347181457c9e551230a9d6001db80",
    "runtime_metadata.json": "4e3a1af91642c9f2adb0b3e43997455a1aea31f86bf45618459d6005a68d4bbf",
}
_SOURCE_FULL_SHA256 = "adfc0400d1a3bb49b278cd3012ab571d01465f2380881f986c085a25474276e5"
_SOURCE_AUTHORIZED_SHA256 = "a23de4988774a966c0d7aac378ede5d15a3fa1d96093c5039f181a62b0bb09b0"
_FROZEN_SHA256 = "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"
_PREFIX_REFERENCE_FILES = {
    "adapter.safetensors": "c47bbb9bacbb5bc8178e9a1797ec47b04ee4a3709042c6b30f6935eacc4686f0",
    "metadata.json": "e76e8a905af53fb082684000a6a6e16845b79e0795b6dbb047a2703245198574",
    "runtime_metadata.json": "01e645b82c5e533dd2319ef8a97171437b149c4e1ef86201f83fdb22de047987",
}
_PREFIX_REFERENCE_FULL_SHA256 = "1d5adc1fb0d7a895056b77d38c8a12aba95c9997ec8a94edf68673f9c58fb954"
_PREFIX_REFERENCE_AUTHORIZED_SHA256 = (
    "d60b665d9a970433b2ed59e6769b9114468bef608b98eae828268101d39db56c"
)
_CANDIDATE_ID = "guarded_both_sign_alpha_2p0"
_CANDIDATE_FULL_SHA256 = "69c5471d141ab56397969e2aac5f1097676f6ea328a3d5577e816c3aef6f3387"
_CANDIDATE_AUTHORIZED_SHA256 = "f43e1b2a84006c8188adeff9f206ba864ae3d51b6567cad679f6d2e5f3610cf2"
_ORIGINAL_V41_PRIORITY_DEFICIT = 31.113729119300842
_BROAD_NLL_MAXIMUM = 2.9213306349515915
_PREFIX_TRUST_RMS_MAXIMUM = 0.002
_HEX64 = re.compile(r"[0-9a-f]{64}")


class GateBackend(Protocol):
    """Narrow execution seam used by the real backend and deterministic tests."""

    def authenticate_and_reconstruct(self) -> Mapping[str, Any]: ...

    def evaluate_non_greedy(self) -> Mapping[str, Any]: ...

    def evaluate_greedy(self) -> Mapping[str, Any]: ...

    def stage_checkpoint(
        self, directory: Path, provenance: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def restore_source(self) -> Mapping[str, Any]: ...

    def access_audit(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class GatePaths:
    terminal: Path = V48_TERMINAL
    report: Path = DEFAULT_REPORT
    checkpoint: Path = DEFAULT_CHECKPOINT
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


def _validate_terminal_authorization(
    report: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, bool]:
    invocation = _mapping(authorization.get("invocation_contract"), "V49 invocation")
    integrity = _mapping(authorization.get("implementation_integrity"), "V49 integrity")
    source = _mapping(authorization.get("source"), "V49 source")
    u4 = _mapping(source.get("v47_u4"), "V49 V47-u4 source")
    v48 = _mapping(source.get("v48_report"), "V49 V48 report source")
    prefix = _mapping(
        source.get("original_v46_candidate_prefix_reference"),
        "V49 original-candidate prefix reference",
    )
    reconstruction = _mapping(authorization.get("candidate_reconstruction"), "V49 reconstruction")
    measurements = _mapping(authorization.get("measurements"), "V49 measurements")
    gate = _mapping(authorization.get("final_train_gate"), "V49 final gate")
    persistence = _mapping(authorization.get("conditional_persistence"), "V49 persistence")
    scope = _mapping(authorization.get("scope"), "V49 scope")
    checks = {
        "artifact": report.get("artifact") == "v48_v47_u4_dual_margin_terminal_gate",
        "passed": report.get("passed") is True,
        "terminal_ready": report.get("terminal_materialization_authorized") is True,
        "successor": report.get("only_exact_successor_authorized") == AUTHORIZATION_ID,
        "authorization_id": authorization.get("authorization_id") == AUTHORIZATION_ID,
        "authorized": authorization.get("authorized") is True,
        "action": authorization.get("only_exact_action")
        == "one_deterministic_v49_guarded_alpha2_train_candidate_gate",
        "script": authorization.get("authorized_script") == str(V49_SCRIPT),
        "test": authorization.get("authorized_test") == str(V49_TEST),
        "report": authorization.get("authorized_report") == str(DEFAULT_REPORT),
        "checkpoint": authorization.get("conditional_checkpoint_output") == str(DEFAULT_CHECKPOINT),
        "config": authorization.get("authorized_config") == str(DEFAULT_CONFIG),
        "explicit_sha": authorization.get("explicit_terminal_sha256_cli_required") is True,
        "terminal_path": invocation.get("terminal_path") == str(V48_TERMINAL),
        "cli": invocation.get("required_cli_argument") == "--expected-v48-terminal-sha256",
        "no_embedded_sha": invocation.get("v49_must_not_embed_terminal_sha256") is True,
        "terminal_auth": invocation.get(
            "v49_must_authenticate_terminal_bytes_and_exact_authorization"
        )
        is True,
        "script_hash": integrity.get("script_sha256") == _sha256(_resolve(V49_SCRIPT)),
        "test_hash": integrity.get("test_sha256") == _sha256(_resolve(V49_TEST)),
        "config_hash": integrity.get("config_sha256") == _CONFIG_SHA256,
        "hashes_complete": integrity.get("hashes_complete") is True,
        "source_checkpoint": u4.get("checkpoint") == str(SOURCE_CHECKPOINT),
        "source_files": u4.get("file_sha256") == _SOURCE_FILES,
        "source_full": u4.get("full_tensor_state_sha256") == _SOURCE_FULL_SHA256,
        "source_authorized": u4.get("authorized_surface_state_sha256") == _SOURCE_AUTHORIZED_SHA256,
        "source_frozen": u4.get("frozen_state_sha256") == _FROZEN_SHA256,
        "source_optimizer_forbidden": u4.get("optimizer_file_open_authorized") is False,
        "v48_report": v48.get("path") == str(V48_REPORT)
        and v48.get("sha256") == _V48_REPORT_SHA256
        and v48.get("candidate_selection_performed") is False
        and v48.get("candidate_authorization_granted") is False
        and v48.get("candidate_checkpoint_written") is False,
        "prefix_checkpoint": prefix.get("checkpoint") == str(PREFIX_REFERENCE_CHECKPOINT),
        "prefix_files": prefix.get("file_sha256") == _PREFIX_REFERENCE_FILES,
        "prefix_full": prefix.get("full_tensor_state_sha256") == _PREFIX_REFERENCE_FULL_SHA256,
        "prefix_authorized": prefix.get("authorized_surface_state_sha256")
        == _PREFIX_REFERENCE_AUTHORIZED_SHA256,
        "prefix_frozen": prefix.get("frozen_state_sha256") == _FROZEN_SHA256,
        "prefix_scope": prefix.get("scene_count") == 16
        and prefix.get("question_free_global_scene_prefix") is True,
        "candidate": reconstruction.get("candidate_id") == _CANDIDATE_ID
        and reconstruction.get("direction_id") == "guarded_both_sign"
        and reconstruction.get("alpha") == 2.0
        and reconstruction.get("expected_full_tensor_state_sha256") == _CANDIDATE_FULL_SHA256
        and reconstruction.get("expected_authorized_surface_state_sha256")
        == _CANDIDATE_AUTHORIZED_SHA256
        and reconstruction.get("expected_frozen_state_sha256") == _FROZEN_SHA256,
        "pre_gate_first": measurements.get("non_greedy_pre_gate_evaluated_first") is True,
        "greedy_iff": measurements.get("full_greedy_mandatory_iff_pre_gate_passes") is True,
        "greedy_skip": measurements.get("greedy_skipped_due_pre_gate_required_if_failed") is True,
        "pre_gate_failure": measurements.get(
            "pre_gate_failure_forces_final_failure_and_no_checkpoint"
        )
        is True,
        "pre_gate_contract": gate.get("pre_gate_passed_equals_all_non_greedy_checks") is True
        and gate.get("pre_gate_failure_forbids_any_greedy_generation") is True
        and gate.get("pre_gate_failure_forces_final_gate_failure") is True
        and gate.get("pre_gate_pass_requires_exhaustive_greedy_evaluation") is True
        and gate.get("final_gate_passed_equals_pre_gate_passed_and_all_greedy_checks") is True,
        "thresholds": gate.get("teacher_complete_units_minimum") == 10
        and gate.get("teacher_positive_sides_minimum") == 35
        and gate.get("teacher_cross_prefix_complete_units_minimum") == 17
        and gate.get("complete_physical_pair_coverage_minimum") == 5
        and gate.get("mirror_complete_units_minimum") == 2
        and gate.get("book_complete_units_minimum") == 1
        and gate.get("book_cross_prefix_complete_units_minimum") == 1
        and gate.get("priority_deficit_improvement_minimum_vs_original_v41_u0") == 0.5
        and gate.get("broad_nll_maximum") == _BROAD_NLL_MAXIMUM
        and gate.get("original_v46_candidate_relative_prefix_trust_rms_maximum")
        == _PREFIX_TRUST_RMS_MAXIMUM
        and gate.get("train_greedy_complete_units_minimum") == 5
        and gate.get("broad_greedy_exact_correct_minimum") == 23
        and gate.get("broad_greedy_row_count_exact") == 48,
        "persistence": persistence.get(
            "candidate_checkpoint_write_iff_every_final_gate_check_passes"
        )
        is True
        and persistence.get("failed_gate_writes_no_checkpoint") is True
        and persistence.get("pre_gate_failure_writes_no_checkpoint") is True
        and persistence.get("optimizer_file_in_checkpoint") is False,
        "scope": scope.get("train_only") is True
        and scope.get("deterministic_single_candidate_no_selection") is True
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
        raise ValueError(f"V49 terminal authorization changed: {checks}")
    return checks


def require_terminal(expected_sha256: str, path: str | Path = V48_TERMINAL) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V49 expected V48 terminal SHA256 must be lowercase hexadecimal")
    terminal_path = _resolve(path)
    if terminal_path != _resolve(V48_TERMINAL):
        raise ValueError("V49 terminal path is pinned")
    if terminal_path.is_symlink() or not terminal_path.is_file():
        raise FileNotFoundError("V49 exact V48 terminal is unavailable or unsafe")
    payload = terminal_path.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise ValueError("V49 V48 terminal differs from explicit invocation SHA256")
    report = _mapping(json.loads(payload), "V48 terminal")
    authorization = _mapping(report.get("conditional_successor_authorization"), "V49 authorization")
    checks = _validate_terminal_authorization(report, authorization)
    return {
        "path": str(V48_TERMINAL),
        "sha256": observed,
        "authorization_id": AUTHORIZATION_ID,
        "authorization": dict(authorization),
        "checks": checks,
    }


def non_greedy_pre_gate_checks(
    reconstruction: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, bool]:
    pair = _mapping(evidence.get("pair_metrics"), "V49 pair metrics")
    families = _mapping(pair.get("complete_units_by_family"), "V49 families")
    cross = _mapping(pair.get("cross_prefix_complete_units_by_family"), "V49 cross families")
    retention = _mapping(evidence.get("retention_diagnostics"), "V49 retention")
    deficit = _finite(evidence.get("priority_side_deficit"), "priority deficit")
    return {
        "source_v47_u4_exact_before_reconstruction": reconstruction.get(
            "source_v47_u4_exact_before_reconstruction"
        )
        is True,
        "reconstructed_candidate_full_tensor_state_exact": reconstruction.get(
            "full_tensor_state_sha256"
        )
        == _CANDIDATE_FULL_SHA256,
        "reconstructed_candidate_authorized_surface_state_exact": reconstruction.get(
            "authorized_surface_state_sha256"
        )
        == _CANDIDATE_AUTHORIZED_SHA256,
        "teacher_complete_units_at_least_10": int(pair["complete_units"]) >= 10,
        "teacher_positive_sides_at_least_35": int(pair["positive_sides"]) >= 35,
        "teacher_cross_prefix_complete_units_at_least_17": int(pair["cross_prefix_complete_units"])
        >= 17,
        "complete_physical_pair_coverage_at_least_5": int(pair["complete_physical_pair_coverage"])
        >= 5,
        "mirror_complete_units_at_least_2": int(families.get("mirror_lr", 0)) >= 2,
        "book_complete_units_at_least_1": int(families.get("book_support", 0)) >= 1,
        "book_cross_prefix_complete_units_at_least_1": int(cross.get("book_support", 0)) >= 1,
        "priority_deficit_improvement_at_least_0_5_vs_original_v41_u0": (
            _ORIGINAL_V41_PRIORITY_DEFICIT - deficit >= 0.5
        ),
        "broad_nll_at_most_v45_maximum": _finite(evidence.get("broad_nll"), "broad NLL")
        <= _BROAD_NLL_MAXIMUM,
        "both_lost_sides_strictly_positive": retention.get("both_lost_sides_strictly_positive")
        is True,
        "scene_readout_state_changed": reconstruction.get("scene_readout_state_changed") is True,
        "query_state_changed": reconstruction.get("query_state_changed") is True,
        "frozen_state_exact": reconstruction.get("frozen_state_sha256") == _FROZEN_SHA256,
        "original_v46_candidate_relative_prefix_trust_rms_at_most_0_002": _finite(
            evidence.get("original_v46_candidate_relative_prefix_trust_rms"),
            "original-candidate prefix RMS",
        )
        <= _PREFIX_TRUST_RMS_MAXIMUM,
    }


def greedy_final_gate_checks(evidence: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "train_greedy_complete_units_at_least_5": int(evidence["complete_units"]) >= 5,
        "broad_greedy_exact_correct_at_least_23_of_48": int(evidence["broad_exact_correct"]) >= 23,
        "broad_greedy_row_count_exactly_48": int(evidence["broad_row_count"]) == 48,
    }


def _authenticate_prefix_reference_checkpoint() -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    from semantic_3d_chat.language.lora import tensor_state_sha256
    from semantic_3d_chat.training.checkpointing import (
        runtime_checkpoint_metadata,
        validate_runtime_checkpoint_metadata,
    )
    from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
        _PARAMETER_NAMES,
    )

    directory = _resolve(PREFIX_REFERENCE_CHECKPOINT)
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError("V49 original candidate prefix checkpoint is unavailable")
    inventory = sorted(path.name for path in directory.iterdir())
    if inventory != sorted(_PREFIX_REFERENCE_FILES):
        raise ValueError("V49 original candidate prefix checkpoint inventory changed")
    observed = {}
    for name, expected in _PREFIX_REFERENCE_FILES.items():
        path = directory / name
        _locked_hash(path, expected, f"V49 prefix reference {name}")
        observed[name] = expected
    tensors = load_file(directory / "adapter.safetensors", device="cpu")
    authorized = {name: tensors[name] for name in _PARAMETER_NAMES}
    frozen = {name: value for name, value in tensors.items() if name not in authorized}
    if tensor_state_sha256(tensors) != _PREFIX_REFERENCE_FULL_SHA256:
        raise ValueError("V49 original candidate prefix full tensor state changed")
    if tensor_state_sha256(authorized) != _PREFIX_REFERENCE_AUTHORIZED_SHA256:
        raise ValueError("V49 original candidate prefix authorized state changed")
    if tensor_state_sha256(frozen) != _FROZEN_SHA256:
        raise ValueError("V49 original candidate prefix frozen state changed")
    metadata = _mapping(
        json.loads((directory / "metadata.json").read_text(encoding="utf-8")),
        "V49 prefix-reference metadata",
    )
    runtime = _mapping(
        json.loads((directory / "runtime_metadata.json").read_text(encoding="utf-8")),
        "V49 prefix-reference runtime metadata",
    )
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V49 prefix-reference runtime metadata is not exact sanitization")
    stage = _mapping(metadata.get("v47_book_continuation"), "V49 prefix-reference stage")
    prefix_hashes = _mapping(
        stage.get("candidate_prefix_sha256_by_train_scene"),
        "V49 original candidate prefix hashes",
    )
    if (
        metadata.get("optimizer_step") != 0
        or metadata.get("epoch") != 0
        or stage.get("optimizer_step") != 0
        or stage.get("reconstructed_candidate_id") != "g5_both_sign_alpha_1p0"
        or stage.get("reconstructed_candidate_full_tensor_state_sha256")
        != _PREFIX_REFERENCE_FULL_SHA256
        or stage.get("reconstructed_candidate_authorized_surface_state_sha256")
        != _PREFIX_REFERENCE_AUTHORIZED_SHA256
        or stage.get("frozen_excluding_authorized_state_sha256") != _FROZEN_SHA256
        or len(prefix_hashes) != 16
    ):
        raise ValueError("V49 original V46-candidate prefix identity changed")
    return dict(metadata), {
        "checkpoint": str(PREFIX_REFERENCE_CHECKPOINT),
        "directory_inventory": inventory,
        "file_sha256": observed,
        "full_tensor_state_sha256": _PREFIX_REFERENCE_FULL_SHA256,
        "authorized_surface_state_sha256": _PREFIX_REFERENCE_AUTHORIZED_SHA256,
        "frozen_state_sha256": _FROZEN_SHA256,
        "candidate_id": "g5_both_sign_alpha_1p0",
        "prefix_sha256_by_train_scene": dict(prefix_hashes),
        "scene_count": 16,
    }


class RealGateBackend:
    """Production V49 backend; construction itself performs no model load."""

    def __init__(self, terminal: Mapping[str, Any], paths: GatePaths) -> None:
        self.terminal = dict(terminal)
        self.paths = paths
        self._audit: Any | None = None
        self._audit_active = False
        self._prepared = False
        self._source_live = False
        self._bundle: Any = None
        self._block_core: Any = None
        self._named: Mapping[str, Any] = {}
        self._source_surface: Mapping[str, Any] = {}
        self._source_metadata: Mapping[str, Any] = {}
        self._config: Mapping[str, Any] = {}
        self._loader: Mapping[str, Any] = {}
        self._records: Sequence[Any] = ()
        self._units: Sequence[Any] = ()
        self._broad_records: Sequence[Any] = ()
        self._caches: Mapping[str, Any] = {}
        self._cache_audit: Mapping[str, Any] = {}
        self._settings: Any = None
        self._prefix_references: Mapping[str, Any] = {}
        self._prefix_reference_audit: Mapping[str, Any] = {}
        self._reconstruction: Mapping[str, Any] = {}
        self._non_greedy: Mapping[str, Any] = {}

    def _start_audit(self) -> None:
        if self._audit_active:
            raise RuntimeError("V49 file audit already active")
        from semantic_3d_chat.chat.file_audit import FileAccessAudit
        from semantic_3d_chat.training.train_retention_repair_v45 import (
            _training_forbidden_roots,
        )

        self._audit = FileAccessAudit(
            _training_forbidden_roots(self._config),
            forbidden_component_names={"oracle"},
            block_forbidden=True,
        )
        self._audit.__enter__()
        self._audit_active = True

    def authenticate_and_reconstruct(self) -> Mapping[str, Any]:
        import torch

        from semantic_3d_chat.config import load_config
        from semantic_3d_chat.evaluation import (
            v48_v47_u4_dual_margin_screen as v48,
        )
        from semantic_3d_chat.language.lora import tensor_state_sha256
        from semantic_3d_chat.training import train_book_continuation_v47 as v47
        from semantic_3d_chat.training.checkpointing import (
            load_adapter_checkpoint,
            module_collection_state_sha256,
        )
        from semantic_3d_chat.training.pair_curriculum import (
            build_exact_question_pair_units,
        )
        from semantic_3d_chat.training.train_block_cross_v35 import current_scene_tokens
        from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
            assert_deferred_final_scenes_absent,
        )
        from semantic_3d_chat.training.train_joint_pair_v30 import (
            require_approved_v29_source,
        )
        from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract
        from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
            _PARAMETER_NAMES,
            assert_v44_trainable_surface,
            freeze_for_v44,
            frozen_v44_state_sha256,
            v44_contract,
        )
        from semantic_3d_chat.training.train_projected_gradient_v41 import (
            cache_v41_train_scenes,
            load_v41_bundle,
            v41_loader_config,
        )
        from semantic_3d_chat.training.train_retention_repair_v45 import (
            _V41_FULL_SHA256,
            _unit_index,
            _v41_source_tensors,
            build_v45_schedule,
            load_v35_train_qa_records,
        )
        from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
            validate_v37_training_cache_boundary,
        )

        if self._prepared:
            raise RuntimeError("V49 backend reconstruction may run only once")
        self._config = load_config(self.paths.config)
        self._loader = v41_loader_config(self._config)
        self._start_audit()
        _locked_hash(_resolve(V48_REPORT), _V48_REPORT_SHA256, "V49 V48 report")
        _locked_hash(_resolve(PROTECTED_REPORT), _PROTECTED_REPORT_SHA256, "V49 protected report")
        source_full, source_metadata, source_audit = v48._source_evidence()
        self._source_metadata = source_metadata
        self._source_surface = {
            name: source_full[name].detach().float().cpu().clone() for name in _PARAMETER_NAMES
        }
        assert_deferred_final_scenes_absent(self._loader)
        records, qa_audit = load_v35_train_qa_records(self._loader)
        units = build_exact_question_pair_units(records)
        units_by_key = _unit_index(units)
        _schedule, schedule_audit, broad_records = build_v45_schedule(
            records, units, config=self._config
        )
        if len(records) != 384 or len(units) != 25 or len(broad_records) != 48:
            raise RuntimeError("V49 exact train-only inventory changed")
        construction = v44_contract(self._config)
        v41_tensors, v41_metadata = _v41_source_tensors(construction)
        if tensor_state_sha256(v41_tensors) != _V41_FULL_SHA256:
            raise RuntimeError("V49 V41 construction source changed")
        approved = require_approved_v29_source(self._loader)
        bundle, block_core, loaded_v41, loader_transition = load_v41_bundle(
            self._config,
            approved,
            construction.source_checkpoint,
            v41_tensors,
        )
        if loaded_v41 != v41_metadata:
            raise RuntimeError("V49 V41 construction metadata changed")
        loaded_u4 = load_adapter_checkpoint(
            _resolve(SOURCE_CHECKPOINT),
            bundle.checkpoint_modules,
            device="cpu",
        )
        if loaded_u4 != source_metadata:
            raise RuntimeError("V49 strict V47-u4 overlay metadata changed")
        named = freeze_for_v44(bundle, block_core)
        assert_v44_trainable_surface(bundle, block_core)
        if (
            module_collection_state_sha256(bundle.checkpoint_modules) != _SOURCE_FULL_SHA256
            or tensor_state_sha256({name: value.detach().cpu() for name, value in named.items()})
            != _SOURCE_AUTHORIZED_SHA256
            or frozen_v44_state_sha256(bundle) != _FROZEN_SHA256
        ):
            raise RuntimeError("V49 live source differs from exact V47 update four")
        self._bundle = bundle
        self._block_core = block_core
        self._named = named
        self._source_live = True

        split = v31_contract(self._loader)
        train_scenes = tuple(
            [*(f"scene_{index:06d}" for index in range(11, 19))]
            + [*(f"scene_{index:06d}" for index in range(31, 39))]
        )
        if tuple(split.train_scene_ids) != train_scenes:
            raise RuntimeError("V49 training scene split changed")
        manifest_ids = (*split.train_scene_ids, *split.validation_scene_ids)
        caches, cache_audit = cache_v41_train_scenes(
            config=self._loader,
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
        validate_v37_training_cache_boundary(
            cache_audit=cache_audit,
            caches=caches,
            config=self._loader,
            train_scene_ids=split.train_scene_ids,
            validation_scene_ids=split.validation_scene_ids,
        )
        if len(caches) != 16 or tuple(sorted(caches)) != tuple(sorted(train_scenes)):
            raise RuntimeError("V49 did not cache all and only 16 training scenes")

        prefix_metadata, prefix_audit = _authenticate_prefix_reference_checkpoint()
        loaded_prefix = load_adapter_checkpoint(
            _resolve(PREFIX_REFERENCE_CHECKPOINT),
            bundle.checkpoint_modules,
            device="cpu",
        )
        if loaded_prefix != prefix_metadata:
            raise RuntimeError("V49 strict original-candidate overlay metadata changed")
        prefix_named = freeze_for_v44(bundle, block_core)
        if (
            module_collection_state_sha256(bundle.checkpoint_modules)
            != _PREFIX_REFERENCE_FULL_SHA256
            or tensor_state_sha256(
                {name: value.detach().cpu() for name, value in prefix_named.items()}
            )
            != _PREFIX_REFERENCE_AUTHORIZED_SHA256
            or frozen_v44_state_sha256(bundle) != _FROZEN_SHA256
        ):
            raise RuntimeError("V49 live original candidate prefix reference changed")
        with torch.inference_mode():
            references = {
                scene_id: current_scene_tokens(
                    caches[scene_id], block_core, device=bundle.language.device
                )
                .detach()
                .cpu()
                .clone()
                for scene_id in sorted(caches)
            }
        prefix_hashes = {
            scene_id: tensor_state_sha256({"scene_tokens": value})
            for scene_id, value in references.items()
        }
        if prefix_hashes != prefix_audit["prefix_sha256_by_train_scene"]:
            raise RuntimeError("V49 original candidate prefix hashes changed")

        # Return to exact u4 before the three authorized gradient probes.
        named = freeze_for_v44(bundle, block_core)
        v48._restore_source(bundle, named, self._source_surface)
        named = freeze_for_v44(bundle, block_core)
        source_state = v48._bundle_state_attestation(
            bundle,
            named,
            expected_authorized_sha256=_SOURCE_AUTHORIZED_SHA256,
            expected_full_sha256=_SOURCE_FULL_SHA256,
        )
        if source_state["passed"] is not True:
            raise RuntimeError("V49 exact u4 restoration before reconstruction failed")
        gradients, gradient_audit = v48._gradient_diagnostics(
            units_by_key=units_by_key,
            caches=caches,
            block_core=block_core,
            bundle=bundle,
            named=named,
        )
        directions, direction_audit = v48.build_normalized_directions(gradients)
        candidate = v48.candidate_from_normalized_direction(
            self._source_surface,
            directions["guarded_both_sign"],
            direction_id="guarded_both_sign",
            alpha=2.0,
        )
        v48._copy_candidate(named, candidate)
        candidate_state = v48._bundle_state_attestation(
            bundle,
            named,
            expected_authorized_sha256=_CANDIDATE_AUTHORIZED_SHA256,
            expected_full_sha256=_CANDIDATE_FULL_SHA256,
        )
        if candidate_state["passed"] is not True:
            raise RuntimeError("V49 reconstructed candidate differs from exact V48 result")
        source_scene_hash = tensor_state_sha256(
            {_PARAMETER_NAMES[0]: self._source_surface[_PARAMETER_NAMES[0]]}
        )
        candidate_scene_hash = tensor_state_sha256(
            {_PARAMETER_NAMES[0]: candidate[_PARAMETER_NAMES[0]]}
        )
        source_query_hash = tensor_state_sha256(
            {name: self._source_surface[name] for name in _PARAMETER_NAMES[1:]}
        )
        candidate_query_hash = tensor_state_sha256(
            {name: candidate[name] for name in _PARAMETER_NAMES[1:]}
        )

        self._named = named
        self._records = records
        self._units = units
        self._broad_records = broad_records
        self._caches = caches
        self._cache_audit = cache_audit
        self._settings = v47.v47_settings(self._config)
        self._prefix_references = references
        self._prefix_reference_audit = prefix_audit
        self._prepared = True
        self._reconstruction = {
            "candidate_id": _CANDIDATE_ID,
            "direction_id": "guarded_both_sign",
            "alpha": 2.0,
            "source_v47_u4_exact_before_reconstruction": True,
            "full_tensor_state_sha256": candidate_state["full_tensor_state_sha256"],
            "authorized_surface_state_sha256": candidate_state["authorized_surface_state_sha256"],
            "frozen_state_sha256": candidate_state["frozen_state_sha256"],
            "all_parameter_gradients_absent_before_forward": candidate_state[
                "all_parameter_gradients_absent"
            ],
            "scene_readout_state_changed": candidate_scene_hash != source_scene_hash,
            "query_state_changed": candidate_query_hash != source_query_hash,
            "gradient_audit_sha256": _canonical_sha256(gradient_audit),
            "direction_audit_sha256": _canonical_sha256(direction_audit),
            "prefix_reference_computed_before_candidate_evaluation": True,
            "prefix_reference": {
                key: value
                for key, value in prefix_audit.items()
                if key != "prefix_sha256_by_train_scene"
            },
            "prefix_reference_hash_inventory_sha256": _canonical_sha256(prefix_hashes),
            "source_audit": source_audit,
            "loader_transition": loader_transition,
            "qa_audit": qa_audit,
            "schedule_audit_sha256": _canonical_sha256(schedule_audit),
            "all_16_training_maps_cached": True,
            "single_candidate_reconstructed": True,
            "candidate_selection_performed": False,
        }
        return self._reconstruction

    def evaluate_non_greedy(self) -> Mapping[str, Any]:
        import torch

        from semantic_3d_chat.training.train_joint_block_cross_v36 import (
            training_broad_nll,
        )
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

        if not self._prepared or self._non_greedy:
            raise RuntimeError("V49 non-greedy evaluation requires one fresh reconstruction")
        with torch.inference_mode():
            _penalty, rms = source_prefix_trust_penalty(
                caches=self._caches,
                references=self._prefix_references,
                block_core=self._block_core,
                device=self._bundle.language.device,
                scale=0.05,
            )
        pair, per_unit = training_pair_gate_diagnostics(
            units=self._units,
            caches=self._caches,
            block_cross_residual=self._block_core,
            bundle=self._bundle,
            settings=self._settings,
        )
        validate_per_unit_nll_diagnostics(per_unit, pair)
        broad = training_broad_nll(
            records=self._broad_records,
            caches=self._caches,
            block_cross_residual=self._block_core,
            bundle=self._bundle,
        )
        self._non_greedy = {
            "pair_metrics": pair,
            "per_unit_nll_diagnostics": per_unit,
            "broad_nll": broad,
            "broad_row_count": len(self._broad_records),
            "priority_side_deficit": float(priority_side_deficit(pair)["combined"]),
            "retention_diagnostics": v45_retention_diagnostics(pair),
            "original_v46_candidate_relative_prefix_trust_rms": float(rms.detach().cpu()),
        }
        return self._non_greedy

    def evaluate_greedy(self) -> Mapping[str, Any]:
        from semantic_3d_chat.training.train_joint_block_cross_v36 import (
            training_greedy_metrics,
        )

        if not self._non_greedy:
            raise RuntimeError("V49 greedy is unreachable before non-greedy evaluation")
        checks = non_greedy_pre_gate_checks(self._reconstruction, self._non_greedy)
        if not all(checks.values()):
            raise RuntimeError("V49 greedy is forbidden because the pre-gate failed")
        return training_greedy_metrics(
            units=self._units,
            broad_records=self._broad_records,
            caches=self._caches,
            block_cross_residual=self._block_core,
            bundle=self._bundle,
            config=self._loader,
        )

    def stage_checkpoint(self, directory: Path, provenance: Mapping[str, Any]) -> Mapping[str, Any]:
        import copy

        from safetensors.torch import load_file

        from semantic_3d_chat.config import config_hash
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

        if not self._prepared or not self._non_greedy:
            raise RuntimeError("V49 checkpoint staging requires evaluated candidate")
        metadata = copy.deepcopy(dict(self._source_metadata))
        metadata.update(
            {
                "schema_version": 1,
                "config_hash": config_hash(dict(self._config)),
                "optimizer_step": 0,
                "epoch": 0,
                "history": [],
                "question_dependent_scene_processing": False,
                **self._bundle.lora_installation.checkpoint_metadata(),
                "block_cross_residual_state_sha256": self._block_core.state_sha256(),
                "frozen_block_cross_source_stack_state_sha256": (
                    block_source_stack_state_sha256(self._bundle, self._block_core)
                ),
            }
        )
        metadata["v49_guarded_candidate_gate"] = {
            "schema_version": 1,
            "authorization": dict(provenance),
            "candidate_id": _CANDIDATE_ID,
            "candidate_full_tensor_state_sha256": _CANDIDATE_FULL_SHA256,
            "candidate_authorized_surface_state_sha256": (_CANDIDATE_AUTHORIZED_SHA256),
            "frozen_state_sha256": _FROZEN_SHA256,
            "original_v46_candidate_relative_prefix_reference": {
                "checkpoint": str(PREFIX_REFERENCE_CHECKPOINT),
                "full_tensor_state_sha256": _PREFIX_REFERENCE_FULL_SHA256,
                "prefix_hash_inventory_sha256": self._reconstruction[
                    "prefix_reference_hash_inventory_sha256"
                ],
            },
            "non_greedy_evidence_sha256": _canonical_sha256(_concise_non_greedy(self._non_greedy)),
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
        save_adapter_checkpoint(directory, self._bundle.checkpoint_modules, metadata)
        saved = load_file(directory / "adapter.safetensors", device="cpu")
        authorized = {name: saved[name] for name in _PARAMETER_NAMES}
        frozen = {name: value for name, value in saved.items() if name not in authorized}
        state = {
            "full_tensor_state_sha256": tensor_state_sha256(saved),
            "authorized_surface_state_sha256": tensor_state_sha256(authorized),
            "frozen_state_sha256": tensor_state_sha256(frozen),
        }
        if state != {
            "full_tensor_state_sha256": _CANDIDATE_FULL_SHA256,
            "authorized_surface_state_sha256": _CANDIDATE_AUTHORIZED_SHA256,
            "frozen_state_sha256": _FROZEN_SHA256,
        }:
            raise RuntimeError("V49 staged checkpoint tensor state changed")
        runtime = _mapping(
            json.loads((directory / "runtime_metadata.json").read_text(encoding="utf-8")),
            "V49 staged runtime metadata",
        )
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise RuntimeError("V49 staged runtime metadata is not exact sanitization")
        return {
            **state,
            "runtime_metadata_exact_sanitization": True,
            "metadata_provenance_bound": True,
        }

    def restore_source(self) -> Mapping[str, Any]:
        from semantic_3d_chat.evaluation import (
            v48_v47_u4_dual_margin_screen as v48,
        )

        if not self._source_live:
            return {
                "passed": True,
                "restoration_not_needed_before_live_source": True,
            }
        return v48._restore_source(self._bundle, self._named, self._source_surface)

    def access_audit(self) -> Mapping[str, Any]:
        if self._audit is None or not self._audit_active:
            raise RuntimeError("V49 access audit is not active")
        self._audit.assert_clean()
        optimizer_reads = [
            path for path in self._audit.unique_paths if path.endswith("/optimizer.pt")
        ]
        loaded_maps = sorted(
            path for path in self._audit.unique_paths if path.endswith("/voxel_map.npz")
        )
        expected_maps = sorted(self._cache_audit.get("loaded_environment_files", ()))
        passed = (
            not optimizer_reads
            and len(loaded_maps) == 16
            and loaded_maps == expected_maps
            and not self._audit.forbidden_accesses()
        )
        return {
            "passed": passed,
            "loaded_file_count": len(self._audit.unique_paths),
            "loaded_file_inventory_sha256": _canonical_sha256(self._audit.unique_paths),
            "training_map_count": len(loaded_maps),
            "training_map_inventory_sha256": _canonical_sha256(loaded_maps),
            "optimizer_file_reads": optimizer_reads,
            "forbidden_file_accesses": self._audit.forbidden_accesses(),
            "validation_qa_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
        }

    def close(self) -> None:
        if self._audit is not None and self._audit_active:
            self._audit.__exit__(None, None, None)
            self._audit_active = False


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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return _sha256_bytes(payload)


def _concise_non_greedy(evidence: Mapping[str, Any]) -> dict[str, Any]:
    pair = _mapping(evidence.get("pair_metrics"), "V49 pair metrics")
    units = _sequence(pair.get("units"), "V49 pair units")
    per_unit = _sequence(evidence.get("per_unit_nll_diagnostics"), "V49 per-unit NLL")
    if int(pair.get("unit_count", -1)) != 25 or len(units) != 25 or len(per_unit) != 25:
        raise ValueError("V49 non-greedy evidence must cover all 25 changed units")
    focus_keys = {
        "cfq_163eb92339ad35a5",
        "cfq_699675ceeaf65406",
        "cfq_5c84a2c27d2be251",
    }
    focus = {
        str(row["question_key"]): {
            "pair_id": row["pair_id"],
            "side_margins": list(row["side_margins"]),
            "cross_prefix_margins": list(row["cross_prefix_margins"]),
        }
        for value in units
        if (row := _mapping(value, "V49 pair unit")).get("question_key") in focus_keys
    }
    if set(focus) != focus_keys:
        raise ValueError("V49 focus-unit inventory changed")
    return {
        "unit_count": 25,
        "per_unit_nll_row_count": 25,
        "pair_metrics_sha256": _canonical_sha256(pair),
        "per_unit_nll_sha256": _canonical_sha256(list(per_unit)),
        "complete_units": int(pair["complete_units"]),
        "positive_sides": int(pair["positive_sides"]),
        "cross_prefix_complete_units": int(pair["cross_prefix_complete_units"]),
        "complete_physical_pair_coverage": int(pair["complete_physical_pair_coverage"]),
        "complete_units_by_family": dict(
            _mapping(pair.get("complete_units_by_family"), "V49 families")
        ),
        "cross_prefix_complete_units_by_family": dict(
            _mapping(
                pair.get("cross_prefix_complete_units_by_family"),
                "V49 cross families",
            )
        ),
        "priority_side_deficit": _finite(evidence.get("priority_side_deficit"), "priority deficit"),
        "priority_deficit_improvement_vs_original_v41_u0": (
            _ORIGINAL_V41_PRIORITY_DEFICIT
            - _finite(evidence.get("priority_side_deficit"), "priority deficit")
        ),
        "broad_nll": _finite(evidence.get("broad_nll"), "broad NLL"),
        "broad_row_count": int(evidence.get("broad_row_count", -1)),
        "original_v46_candidate_relative_prefix_trust_rms": _finite(
            evidence.get("original_v46_candidate_relative_prefix_trust_rms"),
            "original-candidate prefix RMS",
        ),
        "focus_units": focus,
    }


def _concise_greedy(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if int(evidence.get("changed_unit_count", -1)) != 25:
        raise ValueError("V49 greedy changed-unit inventory must be exactly 25")
    if int(evidence.get("changed_row_count", -1)) != 50:
        raise ValueError("V49 greedy changed-row inventory must be exactly 50")
    if int(evidence.get("broad_row_count", -1)) != 48:
        raise ValueError("V49 greedy broad inventory must be exactly 48")
    return {
        "changed_unit_count": 25,
        "changed_row_count": 50,
        "changed_rows_exact_correct": int(evidence["changed_rows_exact_correct"]),
        "complete_units": int(evidence["complete_units"]),
        "complete_physical_pair_coverage": int(evidence["complete_physical_pair_coverage"]),
        "complete_units_by_family": dict(
            _mapping(evidence.get("complete_units_by_family"), "greedy families")
        ),
        "broad_row_count": 48,
        "broad_exact_correct": int(evidence["broad_exact_correct"]),
        "broad_exact_accuracy": _finite(
            evidence.get("broad_exact_accuracy"), "greedy broad accuracy"
        ),
    }


def _staged_checkpoint_inventory(directory: Path) -> dict[str, Any]:
    expected = ["adapter.safetensors", "metadata.json", "runtime_metadata.json"]
    observed = sorted(path.name for path in directory.iterdir())
    if observed != expected:
        raise ValueError(f"V49 staged checkpoint inventory changed: {observed}")
    hashes = {}
    for name in expected:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"V49 staged checkpoint file is unsafe: {name}")
        hashes[name] = _sha256(path)
    return {
        "directory_inventory": expected,
        "file_sha256": hashes,
        "optimizer_file_present": False,
    }


def execute_staged_gate(
    *,
    terminal: Mapping[str, Any],
    backend: GateBackend,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Execute V49's conditional sequence against an already authorized backend."""

    reconstruction: Mapping[str, Any] = {}
    non_greedy: Mapping[str, Any] = {}
    pre_checks: dict[str, bool] = {}
    greedy: Mapping[str, Any] | None = None
    greedy_checks: dict[str, bool] = {}
    restoration: Mapping[str, Any] = {"passed": False, "attempted": False}
    access: Mapping[str, Any] = {"passed": False}
    checkpoint_stage: Mapping[str, Any] | None = None
    concise_non_greedy: Mapping[str, Any] | None = None
    concise_greedy: Mapping[str, Any] | None = None
    staged: Path | None = None
    checkpoint_written = False
    execution_error: dict[str, str] | None = None
    try:
        reconstruction = backend.authenticate_and_reconstruct()
        non_greedy = backend.evaluate_non_greedy()
        concise_non_greedy = _concise_non_greedy(non_greedy)
        pre_checks = non_greedy_pre_gate_checks(reconstruction, non_greedy)
        pre_passed = all(pre_checks.values())
        if pre_passed:
            greedy = backend.evaluate_greedy()
            concise_greedy = _concise_greedy(greedy)
            greedy_checks = greedy_final_gate_checks(greedy)
        behavioral_passed = pre_passed and all(greedy_checks.values())
        if behavioral_passed:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            staged = Path(
                tempfile.mkdtemp(
                    prefix=f".{checkpoint_path.name}.staged.",
                    dir=checkpoint_path.parent,
                )
            )
            provenance = {
                "terminal_path": terminal["path"],
                "terminal_sha256": terminal["sha256"],
                "authorization_id": AUTHORIZATION_ID,
                "candidate_id": _CANDIDATE_ID,
                "source_checkpoint": str(SOURCE_CHECKPOINT),
                "source_full_tensor_state_sha256": _SOURCE_FULL_SHA256,
                "original_prefix_reference_checkpoint": str(PREFIX_REFERENCE_CHECKPOINT),
                "original_prefix_reference_full_tensor_state_sha256": (
                    _PREFIX_REFERENCE_FULL_SHA256
                ),
                "candidate_full_tensor_state_sha256": _CANDIDATE_FULL_SHA256,
            }
            backend_stage = dict(backend.stage_checkpoint(staged, provenance))
            checkpoint_stage = {
                **backend_stage,
                **_staged_checkpoint_inventory(staged),
            }
    except Exception as exc:  # noqa: BLE001 - seal failures only after source restore
        execution_error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            restoration = {
                "attempted": True,
                **dict(backend.restore_source()),
            }
        except Exception as exc:  # noqa: BLE001 - restoration failures must fail closed
            restoration = {
                "attempted": True,
                "passed": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            if execution_error is None:
                execution_error = {
                    "type": type(exc).__name__,
                    "message": f"source restoration failed: {exc}",
                }
        try:
            access = dict(backend.access_audit())
        except Exception as exc:  # noqa: BLE001 - audit failures must block persistence
            access = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            if execution_error is None:
                execution_error = {
                    "type": type(exc).__name__,
                    "message": f"access audit failed: {exc}",
                }
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001 - close failure must block persistence
            if execution_error is None:
                execution_error = {
                    "type": type(exc).__name__,
                    "message": f"backend close failed: {exc}",
                }

    pre_passed = bool(pre_checks) and all(pre_checks.values())
    greedy_executed = greedy is not None
    greedy_passed = greedy_executed and bool(greedy_checks) and all(greedy_checks.values())
    final_passed = bool(
        pre_passed
        and greedy_passed
        and restoration.get("passed") is True
        and access.get("passed") is True
        and execution_error is None
        and staged is not None
        and checkpoint_stage is not None
    )
    if final_passed:
        if checkpoint_path.is_symlink() or checkpoint_path.exists():
            execution_error = {
                "type": "FileExistsError",
                "message": "V49 checkpoint destination appeared before atomic publish",
            }
            final_passed = False
        else:
            try:
                os.replace(staged, checkpoint_path)
                staged = None
                checkpoint_written = True
            except OSError as exc:
                execution_error = {
                    "type": type(exc).__name__,
                    "message": f"atomic checkpoint publish failed: {exc}",
                }
                final_passed = False
    if staged is not None:
        shutil.rmtree(staged)

    if pre_passed and not greedy_executed and execution_error is None:
        raise RuntimeError("V49 pre-gate passed but exhaustive greedy was not executed")
    if not pre_passed and greedy_executed:
        raise RuntimeError("V49 greedy executed despite a failed non-greedy pre-gate")
    if checkpoint_written is not final_passed:
        raise RuntimeError("V49 checkpoint persistence violated iff-final-pass contract")

    return {
        "schema_version": 1,
        "artifact": "v49_guarded_candidate_train_gate",
        "passed": final_passed,
        "authorization": {
            "terminal_path": terminal["path"],
            "terminal_sha256": terminal["sha256"],
            "authorization_id": AUTHORIZATION_ID,
            "checks": dict(_mapping(terminal.get("checks"), "terminal checks")),
        },
        "candidate_reconstruction": dict(reconstruction),
        "non_greedy_pre_gate": {
            "evaluated_first": True,
            "checks": pre_checks,
            "passed": pre_passed,
            "evidence": concise_non_greedy,
        },
        "greedy_gate": {
            "authorized": pre_passed,
            "executed": greedy_executed,
            "greedy_skipped_due_pre_gate": not pre_passed,
            "checks": greedy_checks,
            "passed": greedy_passed,
            "evidence": concise_greedy,
        },
        "final_train_gate": {
            "passed": final_passed,
            "pre_gate_passed": pre_passed,
            "greedy_gate_passed": greedy_passed,
            "source_restored_exact": restoration.get("passed") is True,
            "access_audit_passed": access.get("passed") is True,
            "execution_error": execution_error,
        },
        "checkpoint": {
            "path": str(DEFAULT_CHECKPOINT),
            "staged_after_behavioral_gate": checkpoint_stage is not None,
            "written": checkpoint_written,
            "write_iff_final_gate_passed": checkpoint_written is final_passed,
            "inventory": checkpoint_stage,
            "optimizer_file_written": False,
        },
        "source_restoration": dict(restoration),
        "access_audit": dict(access),
        "optimizer_constructed_or_loaded": False,
        "optimizer_state_file_opened": False,
        "optimizer_step_executed": False,
        "candidate_selection_performed": False,
        "validation_qa_loaded": False,
        "validation_environment_maps_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_executed": False,
        "runtime_promotion_executed": False,
        "chat_promotion_executed": False,
        "embodied_promotion_executed": False,
    }


def preflight(
    *,
    expected_v48_terminal_sha256: str,
    paths: GatePaths | None = None,
) -> dict[str, Any]:
    """Authenticate the complete launch boundary without loading model, QA, or maps."""

    selected = GatePaths() if paths is None else paths
    resolved = GatePaths(
        terminal=_resolve(selected.terminal),
        report=_resolve(selected.report),
        checkpoint=_resolve(selected.checkpoint),
        config=_resolve(selected.config),
    )
    expected = GatePaths(
        terminal=_resolve(V48_TERMINAL),
        report=_resolve(DEFAULT_REPORT),
        checkpoint=_resolve(DEFAULT_CHECKPOINT),
        config=_resolve(DEFAULT_CONFIG),
    )
    if resolved != expected:
        raise ValueError("V49 preflight paths are pinned")
    if resolved.report.is_symlink() or resolved.report.exists():
        raise FileExistsError("V49 report is one-shot and already exists")
    if resolved.checkpoint.is_symlink() or resolved.checkpoint.exists():
        raise FileExistsError("V49 checkpoint destination already exists")
    _locked_hash(resolved.config, _CONFIG_SHA256, "V49 config")
    terminal = require_terminal(expected_v48_terminal_sha256, selected.terminal)
    _locked_hash(_resolve(V48_REPORT), _V48_REPORT_SHA256, "V49 V48 report")
    _locked_hash(_resolve(PROTECTED_REPORT), _PROTECTED_REPORT_SHA256, "V49 protected report")
    source = _resolve(SOURCE_CHECKPOINT)
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError("V49 source checkpoint is unavailable")
    source_inventory = sorted(path.name for path in source.iterdir())
    if source_inventory != sorted(_SOURCE_FILES):
        raise ValueError("V49 source checkpoint inventory changed")
    readable_source = {
        name: digest for name, digest in _SOURCE_FILES.items() if name != "optimizer.pt"
    }
    for name, digest in readable_source.items():
        _locked_hash(source / name, digest, f"V49 source {name}")
    prefix = _resolve(PREFIX_REFERENCE_CHECKPOINT)
    if prefix.is_symlink() or not prefix.is_dir():
        raise FileNotFoundError("V49 prefix reference checkpoint is unavailable")
    prefix_inventory = sorted(path.name for path in prefix.iterdir())
    if prefix_inventory != sorted(_PREFIX_REFERENCE_FILES):
        raise ValueError("V49 prefix reference checkpoint inventory changed")
    for name, digest in _PREFIX_REFERENCE_FILES.items():
        _locked_hash(prefix / name, digest, f"V49 prefix reference {name}")
    return {
        "schema_version": 1,
        "artifact": "v49_guarded_candidate_train_gate_preflight",
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
        "candidate_id": _CANDIDATE_ID,
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


def run_gate(
    *,
    expected_v48_terminal_sha256: str,
    paths: GatePaths | None = None,
    backend_factory: Callable[[Mapping[str, Any], GatePaths], GateBackend] | None = None,
) -> dict[str, Any]:
    """Run the authenticated staged gate and atomically emit its report."""

    selected_paths = GatePaths() if paths is None else paths
    resolved = GatePaths(
        terminal=_resolve(selected_paths.terminal),
        report=_resolve(selected_paths.report),
        checkpoint=_resolve(selected_paths.checkpoint),
        config=_resolve(selected_paths.config),
    )
    expected_paths = GatePaths(
        terminal=_resolve(V48_TERMINAL),
        report=_resolve(DEFAULT_REPORT),
        checkpoint=_resolve(DEFAULT_CHECKPOINT),
        config=_resolve(DEFAULT_CONFIG),
    )
    if resolved != expected_paths:
        raise ValueError("V49 terminal, report, checkpoint, and config paths are pinned")
    if resolved.report.is_symlink() or resolved.report.exists():
        raise FileExistsError("V49 report is one-shot and will not be overwritten")
    if resolved.checkpoint.is_symlink() or resolved.checkpoint.exists():
        raise FileExistsError("V49 checkpoint destination must be absent")
    _locked_hash(resolved.config, _CONFIG_SHA256, "V49 config")
    terminal = require_terminal(expected_v48_terminal_sha256, path=selected_paths.terminal)
    factory = RealGateBackend if backend_factory is None else backend_factory
    backend = factory(terminal, resolved)
    report = execute_staged_gate(
        terminal=terminal,
        backend=backend,
        checkpoint_path=resolved.checkpoint,
    )
    _atomic_json(resolved.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-v48-terminal-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--terminal", type=Path, default=V48_TERMINAL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    paths = GatePaths(
        terminal=args.terminal,
        report=args.report,
        checkpoint=args.checkpoint,
        config=args.config,
    )
    if args.preflight:
        result = preflight(
            expected_v48_terminal_sha256=args.expected_v48_terminal_sha256,
            paths=paths,
        )
        summary = {
            "artifact": result["artifact"],
            "passed": result["passed"],
            "terminal_sha256": result["terminal"]["sha256"],
            "model_loaded": False,
            "qa_loaded": False,
            "maps_loaded": False,
        }
    else:
        result = run_gate(
            expected_v48_terminal_sha256=args.expected_v48_terminal_sha256,
            paths=paths,
        )
        summary = {
            "artifact": result["artifact"],
            "passed": result["passed"],
            "report": str(DEFAULT_REPORT),
            "report_sha256": _sha256(_resolve(DEFAULT_REPORT)),
            "pre_gate_passed": result["non_greedy_pre_gate"]["passed"],
            "greedy_executed": result["greedy_gate"]["executed"],
            "greedy_gate_passed": result["greedy_gate"]["passed"],
            "checkpoint_written": result["checkpoint"]["written"],
        }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_ID",
    "GateBackend",
    "GatePaths",
    "RealGateBackend",
    "execute_staged_gate",
    "greedy_final_gate_checks",
    "main",
    "non_greedy_pre_gate_checks",
    "preflight",
    "require_terminal",
    "run_gate",
]

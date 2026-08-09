"""Fail-closed authorization and final selection through V21 update 8.

Both subcommands are report-only. They safely inspect checkpoint tensors and
one-matrix AdamW state on CPU, but never load Gemma, perform model inference,
or read a map, scene token, question, rendered observation, runtime artifact,
or oracle artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config
from semantic_3d_chat.evaluation.phase_aware_local_field_profile import (
    V21_LOCAL_FIELD_PROFILE,
    PhaseAwareLocalFieldProfile,
)
from semantic_3d_chat.evaluation.v21_epoch_selector import (
    EXPECTED_EPOCHS,
    EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
    EXPECTED_FROZEN_SCENE_SHA256,
    MODEL_DTYPE,
    PINNED_CONFIG_PATH,
    V21EpochSelectorViolation,
    _color_eligible,
    _continuation_passed,
    _full_teacher_passed,
    _inspect_checkpoint_artifacts,
    _lexical_absolute,
    _load_json_strict,
    _mapping,
    _ranking_key,
    _reject_forbidden_input_path,
    _safe_checkpoint_file,
    _sequence,
    _validate_config,
    _validate_epoch_artifact,
    summarize_v21_epochs,
)
from semantic_3d_chat.evaluation.v21_epoch_selector import (
    _file_sha256 as _selector_file_sha256,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)

# Keep this component within the trainer's fail-closed 64-character namespace
# contract. The longer descriptive primary namespace is recorded separately in
# every launch/final report as ``original_output_namespace``.
OUTPUT_NAMESPACE = V21_LOCAL_FIELD_PROFILE.output_namespace
EXTENSION_NAMESPACE = "gemma4_v21_phase_aware_local_field_extension_u8"
CONTROLLER_TYPE = "strict_v21_conditional_extension_controller"
FINAL_SELECTOR_TYPE = "strict_v21_conditional_extension_final_selector"
TARGET_OPTIMIZER_UPDATE = 8
MICROSTEPS_PER_UPDATE = 12
PYTHON_EXECUTABLE = ".venv-gemma4/bin/python"
TRAINING_MODULE = "semantic_3d_chat.training.train_adapter"


class V21ExtensionViolation(ValueError):
    """A mismatch that denies a V21 conditional launch or final selection."""


def _fail(message: str) -> None:
    raise V21ExtensionViolation(message)


def _exact_int(value: Any, expected: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        _fail(f"{field} must equal {expected}")
    return value


def _file_sha256(path: Path, field: str) -> str:
    try:
        return _selector_file_sha256(path, field)
    except V21EpochSelectorViolation as error:
        _fail(str(error))


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as error:
        _fail(f"Cannot canonically hash report value: {error}")
    return hashlib.sha256(payload).hexdigest()


def _reject_path(path: Path) -> None:
    try:
        _reject_forbidden_input_path(path)
    except ValueError as error:
        _fail(str(error))


def _safe_input_file(path: str | Path, field: str) -> Path:
    """Reject missing, non-regular, forbidden, or symlinked controller inputs."""

    try:
        return _safe_checkpoint_file(Path(path), field)
    except ValueError as error:
        _fail(str(error))


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _same_path(observed: str | Path, expected: Path, field: str) -> None:
    if _resolve(observed) != expected.resolve():
        _fail(f"{field} path mismatch: observed={observed} expected={_display(expected)}")


def _checkpoint_hashes(directory: Path, field: str) -> dict[str, str]:
    return {
        "adapter_sha256": _file_sha256(directory / "adapter.safetensors", f"{field}.adapter"),
        "metadata_sha256": _file_sha256(directory / "metadata.json", f"{field}.metadata"),
        "optimizer_sha256": _file_sha256(directory / "optimizer.pt", f"{field}.optimizer"),
    }


def _load_exact_screen_report(
    config_path: str | Path,
    screen_path: str | Path,
    *,
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    resolved_config = _safe_input_file(config_path, "V21 config")
    resolved_screen = _safe_input_file(screen_path, "V21 screen report")
    try:
        config = load_config(resolved_config)
        raw_screen, screen_sha256 = _load_json_strict(resolved_screen)
        screen = dict(raw_screen)
        selection_raw = screen.get("selection_artifact_path")
        if not isinstance(selection_raw, str) or not selection_raw:
            _fail("screen.selection_artifact_path is missing")
        selection_path = Path(selection_raw)
        resolved_selection = _safe_input_file(selection_path, "V21 selection artifact")
        selection, selection_sha256 = _load_json_strict(resolved_selection)
        rows = _sequence(screen.get("epochs"), "screen.epochs")
        if len(rows) != len(EXPECTED_EPOCHS):
            _fail("screen.epochs must contain exactly updates 1,2,3,4")
        epoch_paths: dict[int, Path] = {}
        for index, value in enumerate(rows):
            row = _mapping(value, f"screen.epochs[{index}]")
            epoch = row.get("epoch")
            if (
                isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or epoch not in EXPECTED_EPOCHS
            ):
                _fail(f"screen.epochs[{index}].epoch is invalid")
            if epoch in epoch_paths:
                _fail(f"screen.epochs repeats update {epoch}")
            raw_path = row.get("checkpoint_metadata_path")
            if not isinstance(raw_path, str) or not raw_path:
                _fail(f"screen.epochs[{index}].checkpoint_metadata_path is invalid")
            epoch_paths[epoch] = Path(raw_path)
        if set(epoch_paths) != set(EXPECTED_EPOCHS):
            _fail("screen.epochs does not cover exactly updates 1,2,3,4")
        resolved_epochs = {
            epoch: _safe_input_file(path, f"V21 epoch_{epoch} metadata")
            for epoch, path in epoch_paths.items()
        }
        loaded = {epoch: _load_json_strict(path) for epoch, path in resolved_epochs.items()}
        authorization = _mapping(
            screen.get("update1_authorization"), "screen.update1_authorization"
        )
        update1_path = authorization.get("report_path")
        if not isinstance(update1_path, str) or not update1_path:
            _fail("screen.update1_authorization.report_path is missing")
        _safe_input_file(update1_path, "V21 update-one authorization")
        selector_arguments = {
            "update1_report_path": update1_path,
            "selection_path": str(selection_path),
            "selection_sha256": selection_sha256,
            "epoch_paths": {epoch: str(path) for epoch, path in epoch_paths.items()},
            "epoch_sha256": {
                epoch: digest for epoch, (_value, digest) in loaded.items()
            },
        }
        if profile is V21_LOCAL_FIELD_PROFILE:
            recomputed = summarize_v21_epochs(
                config,
                selection,
                {epoch: value for epoch, (value, _digest) in loaded.items()},
                **selector_arguments,
            )
        else:
            recomputed = summarize_v21_epochs(
                config,
                selection,
                {epoch: value for epoch, (value, _digest) in loaded.items()},
                profile=profile,
                **selector_arguments,
            )
    except V21ExtensionViolation:
        raise
    except (
        V21EpochSelectorViolation,
        ValueError,
        FileNotFoundError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        _fail(f"Cannot validate exact V21 screen report: {error}")
    if screen != recomputed:
        _fail("V21 screen differs from exact recomputation of its bound artifacts")
    return screen, screen_sha256, config, dict(selection)


def _require_extension_decision(
    screen: Mapping[str, Any],
    *,
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> int:
    for key, expected in {
        "selector_type": profile.selector_type,
        "report_only": True,
        "model_inference_executed": False,
        "gemma_model_loaded": False,
        "checkpoint_tensor_state_loaded": True,
        "checkpoint_tensor_state_safely_inspected": True,
        "optimizer_deserialized": True,
        "optimizer_deserialization_weights_only": True,
        "model_dtype": MODEL_DTYPE,
        "continuation_authorized": True,
        "continuation_gate_passed": True,
        "full_teacher_gate_passed": False,
        "greedy_audit_authorized": False,
        "greedy_audit_forbidden": True,
        "decision": "continue_selected_epoch_no_greedy_audit",
    }.items():
        if screen.get(key) != expected:
            _fail(f"screen.{key} does not authorize V21 continuation")
    _exact_int(
        screen.get("conditional_max_optimizer_updates"),
        TARGET_OPTIMIZER_UPDATE,
        "screen.conditional_max_optimizer_updates",
    )
    selected = screen.get("selected_epoch")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected not in EXPECTED_EPOCHS
    ):
        _fail("screen.selected_epoch must be one of updates 1,2,3,4")
    policy = _mapping(screen.get("selection_policy"), "screen.selection_policy")
    continuation = _mapping(
        policy.get("continuation_requires"),
        "screen.selection_policy.continuation_requires",
    )
    _exact_int(
        continuation.get("mirror_minimum_full_vocab_sides"),
        8,
        "screen mirror side threshold",
    )
    _exact_int(
        continuation.get("mirror_minimum_full_vocab_units"),
        2,
        "screen mirror unit threshold",
    )
    rows: dict[int, Mapping[str, Any]] = {}
    for index, value in enumerate(_sequence(screen.get("epochs"), "screen.epochs")):
        row = _mapping(value, f"screen.epochs[{index}]")
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch not in EXPECTED_EPOCHS:
            _fail(f"screen.epochs[{index}].epoch is invalid")
        if epoch in rows:
            _fail(f"screen.epochs repeats update {epoch}")
        rows[epoch] = row
    selected_row = rows.get(selected)
    if selected_row is None:
        _fail("screen.selected_epoch has no matching epoch row")
    if not _color_eligible(selected_row):
        _fail("screen selected epoch is not strictly color eligible")
    if not _continuation_passed(selected_row, policy):
        _fail("screen selected epoch does not meet the 8-side/2-unit continuation gate")
    if _full_teacher_passed(selected_row):
        _fail("screen selected epoch already meets the full-teacher gate")
    return selected


def _require_current_source(
    screen: Mapping[str, Any], current_provenance: Mapping[str, Any] | None
) -> dict[str, Any]:
    current = dict(
        capture_git_source_provenance(PROJECT_ROOT)
        if current_provenance is None
        else current_provenance
    )
    try:
        require_clean_committed_source(current)
    except RuntimeError as error:
        _fail(f"V21 extension requires clean committed source: {error}")
    if current != screen.get("source_provenance"):
        _fail("Current clean source provenance differs from the exact V21 screen")
    return current


def _build_launch_manifest(
    *,
    config_path: Path,
    screen_path: Path,
    screen: Mapping[str, Any],
    screen_sha256: str,
    config: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    require_namespace_absent: bool,
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> dict[str, Any]:
    if profile is V21_LOCAL_FIELD_PROFILE:
        selected_epoch = _require_extension_decision(screen)
        contract = _validate_config(config)
    else:
        selected_epoch = _require_extension_decision(screen, profile=profile)
        contract = _validate_config(config, profile=profile)
    selected_metadata = screen.get("selected_checkpoint_metadata_path")
    if not isinstance(selected_metadata, str) or not selected_metadata:
        _fail("screen.selected_checkpoint_metadata_path is missing")
    checkpoint_root = artifact_root(dict(config), "checkpoints").resolve()
    original_root = (checkpoint_root / profile.output_namespace).resolve()
    extension_root = (checkpoint_root / profile.extension_namespace).resolve()
    selected_checkpoint = (original_root / f"epoch_{selected_epoch:03d}").resolve()
    _same_path(selected_metadata, selected_checkpoint / "metadata.json", "selected checkpoint")
    selected_checkpoint_from_screen = screen.get("selected_checkpoint")
    if not isinstance(selected_checkpoint_from_screen, str):
        _fail("screen.selected_checkpoint is missing")
    _same_path(
        selected_checkpoint_from_screen,
        selected_checkpoint,
        "screen.selected_checkpoint",
    )
    if extension_root == original_root or extension_root.is_relative_to(original_root):
        _fail("V21 extension namespace is not isolated")
    if require_namespace_absent and extension_root.exists():
        _fail(f"Refusing to reuse or overwrite existing V21 extension namespace: {extension_root}")
    selected_hashes = _checkpoint_hashes(selected_checkpoint, "selected_checkpoint")
    if selected_hashes != screen.get("selected_checkpoint_artifact_hashes"):
        _fail("Selected checkpoint adapter/metadata/optimizer no longer match the V21 screen")
    if selected_hashes["metadata_sha256"] != screen.get("selected_checkpoint_metadata_sha256"):
        _fail("Selected checkpoint metadata alias no longer matches the V21 screen")
    update1_authorization = deepcopy(
        dict(_mapping(screen.get("update1_authorization"), "screen.update1_authorization"))
    )
    python_path = _resolve(PYTHON_EXECUTABLE)
    if not python_path.is_file():
        _fail(f"Pinned Gemma Python executable is unavailable: {python_path}")
    argv = [
        PYTHON_EXECUTABLE,
        "-m",
        TRAINING_MODULE,
        "--config",
        _display(config_path),
        "--resume",
        _display(selected_checkpoint),
        "--output-namespace",
        profile.extension_namespace,
        "--epochs",
        str(TARGET_OPTIMIZER_UPDATE),
    ]
    expected_epochs = list(range(selected_epoch + 1, TARGET_OPTIMIZER_UPDATE + 1))
    return {
        "schema_version": 1,
        "controller_type": profile.extension_controller_type,
        "authorized": True,
        "report_only": True,
        "model_inference_executed": False,
        "gemma_model_loaded": False,
        "checkpoint_tensor_state_loaded": True,
        "checkpoint_tensor_state_safely_inspected": True,
        "optimizer_deserialized": True,
        "optimizer_deserialization_weights_only": True,
        "question_dependent_scene_processing": False,
        "config_path": _display(config_path),
        "config_hash": contract["config_hash"],
        "config_hash_full": contract["config_hash_full"],
        "preflight_contract_sha256": contract["preflight_contract_sha256"],
        "model_dtype": MODEL_DTYPE,
        "frozen_scene_state_sha256": EXPECTED_FROZEN_SCENE_SHA256,
        "frozen_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
        "screen_report_path": _display(screen_path),
        "screen_report_sha256": screen_sha256,
        "screen_report_canonical_sha256": _canonical_sha256(screen),
        "update1_authorization": update1_authorization,
        "source_provenance": deepcopy(dict(source_provenance)),
        "screen_decision": "continue_selected_epoch_no_greedy_audit",
        "continuation_authorized": True,
        "greedy_audit_authorized": False,
        "greedy_audit_forbidden_during_extension": True,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": _display(selected_checkpoint),
        "selected_checkpoint_artifact_hashes": selected_hashes,
        "selected_signed_x_state_sha256": screen["selected_signed_x_state_sha256"],
        "selected_optimizer_state_sha256": screen["selected_optimizer_state_sha256"],
        "original_output_namespace": profile.output_namespace,
        "extension_output_namespace": profile.extension_namespace,
        "extension_checkpoint_root": _display(extension_root),
        "extension_namespace_absent_at_authorization": True,
        "start_optimizer_update": selected_epoch + 1,
        "target_optimizer_update": TARGET_OPTIMIZER_UPDATE,
        "microsteps_per_optimizer_update": MICROSTEPS_PER_UPDATE,
        "expected_extension_epochs": expected_epochs,
        "expected_final_global_step": TARGET_OPTIMIZER_UPDATE * MICROSTEPS_PER_UPDATE,
        "overwrite_original_namespace": False,
        "trainer": {
            "working_directory": str(PROJECT_ROOT.resolve()),
            "environment": {"PYTHONPATH": "src"},
            "argv": argv,
            "shell_command": "PYTHONPATH=src " + shlex.join(argv),
            "executes_on_prepare": False,
        },
    }


def prepare_extension_launch(
    config_path: str | Path,
    screen_path: str | Path,
    *,
    current_provenance: Mapping[str, Any] | None = None,
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> dict[str, Any]:
    """Authorize, but never execute, the isolated V21 update-8 continuation."""

    if profile is V21_LOCAL_FIELD_PROFILE:
        screen, screen_sha256, config, _selection = _load_exact_screen_report(
            config_path, screen_path
        )
    else:
        screen, screen_sha256, config, _selection = _load_exact_screen_report(
            config_path,
            screen_path,
            profile=profile,
        )
    source = _require_current_source(screen, current_provenance)
    arguments = {
        "config_path": _resolve(config_path),
        "screen_path": _resolve(screen_path),
        "screen": screen,
        "screen_sha256": screen_sha256,
        "config": config,
        "source_provenance": source,
        "require_namespace_absent": True,
    }
    if profile is V21_LOCAL_FIELD_PROFILE:
        return _build_launch_manifest(**arguments)
    return _build_launch_manifest(**arguments, profile=profile)


def _validate_launch_manifest(
    manifest_path: str | Path,
    *,
    current_provenance: Mapping[str, Any] | None = None,
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved = _safe_input_file(manifest_path, "V21 extension launch manifest")
    try:
        raw_manifest, _digest = _load_json_strict(resolved)
    except (ValueError, FileNotFoundError, OSError, UnicodeError) as error:
        _fail(f"Cannot load V21 extension launch manifest: {error}")
    manifest = dict(raw_manifest)
    if (
        manifest.get("controller_type") != profile.extension_controller_type
        or manifest.get("authorized") is not True
    ):
        _fail("V21 launch manifest is not an authorization from this controller")
    config_path = manifest.get("config_path")
    screen_path = manifest.get("screen_report_path")
    if not isinstance(config_path, str) or not isinstance(screen_path, str):
        _fail("V21 launch config/screen path is invalid")
    if profile is V21_LOCAL_FIELD_PROFILE:
        screen, screen_sha256, config, _selection = _load_exact_screen_report(
            config_path, screen_path
        )
    else:
        screen, screen_sha256, config, _selection = _load_exact_screen_report(
            config_path,
            screen_path,
            profile=profile,
        )
    source = _require_current_source(screen, current_provenance)
    arguments = {
        "config_path": _resolve(config_path),
        "screen_path": _resolve(screen_path),
        "screen": screen,
        "screen_sha256": screen_sha256,
        "config": config,
        "source_provenance": source,
        "require_namespace_absent": False,
    }
    if profile is V21_LOCAL_FIELD_PROFILE:
        expected = _build_launch_manifest(**arguments)
    else:
        expected = _build_launch_manifest(**arguments, profile=profile)
    if manifest != expected:
        _fail("V21 launch manifest differs from exact current authorization")
    return manifest, screen, config


def _validate_extension_epoch(
    epoch: int,
    metadata_path: Path,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    expected_source: Mapping[str, Any],
    *,
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> dict[str, Any]:
    metadata_path = _safe_input_file(metadata_path, f"V21 extension update {epoch} metadata")
    try:
        raw, metadata_sha256 = _load_json_strict(metadata_path)
    except (ValueError, FileNotFoundError, OSError, UnicodeError) as error:
        _fail(f"Cannot load V21 extension update {epoch}: {error}")
    artifact = dict(raw)
    if artifact.get("output_namespace") != profile.extension_namespace:
        _fail(f"Extension update {epoch} is not in the isolated namespace")
    if artifact.get("source_provenance") != expected_source:
        _fail(f"Extension update {epoch} source provenance differs from authorization")
    normalized = deepcopy(artifact)
    normalized["output_namespace"] = profile.output_namespace
    try:
        arguments = {
            "path": _display(metadata_path),
            "artifact_sha256": metadata_sha256,
        }
        if profile is V21_LOCAL_FIELD_PROFILE:
            validated = _validate_epoch_artifact(
                epoch,
                normalized,
                contract,
                **arguments,
            )
        else:
            validated = _validate_epoch_artifact(
                epoch,
                normalized,
                contract,
                profile=profile,
                **arguments,
            )
    except ValueError as error:
        _fail(f"Extension update {epoch} violates the strict V21 contract: {error}")
    try:
        inspection = _inspect_checkpoint_artifacts(
            epoch,
            artifact,
            config=config,
            metadata_path=metadata_path,
        )
    except ValueError as error:
        _fail(f"Extension update {epoch} artifact state violates V21: {error}")
    if inspection["checkpoint_artifact_hashes"]["metadata_sha256"] != metadata_sha256:
        _fail(f"Extension update {epoch} metadata hash changed during inspection")
    if inspection["tensor_evidence"]["signed_x_state_sha256"] != validated["signed_x_state_sha256"]:
        _fail(f"Extension update {epoch} signed-X tensor state differs from metadata")
    validated["checkpoint_inspection"] = inspection
    validated["checkpoint_artifact_hashes"] = inspection["checkpoint_artifact_hashes"]
    validated["optimizer_state_sha256"] = inspection["optimizer_state_sha256"]
    return validated


def _candidate(
    epoch: int,
    metrics: Mapping[str, Any],
    *,
    metadata_path: str,
    metadata_sha256: str,
    artifact_hashes: Mapping[str, str],
    state_sha256: str,
    optimizer_state_sha256: str,
    screen: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = {
        "epoch": epoch,
        "optimizer_step": epoch,
        "cumulative_microsteps": epoch * MICROSTEPS_PER_UPDATE,
        "checkpoint_metadata_path": metadata_path,
        "checkpoint_metadata_sha256": metadata_sha256,
        "checkpoint_artifact_hashes": deepcopy(dict(artifact_hashes)),
        "signed_x_state_sha256": state_sha256,
        "optimizer_state_sha256": optimizer_state_sha256,
        "frozen_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
        "frozen_scene_state_sha256": EXPECTED_FROZEN_SCENE_SHA256,
        "model_dtype": MODEL_DTYPE,
        "color": deepcopy(dict(_mapping(metrics.get("color"), "metrics.color"))),
        "mirror": deepcopy(dict(_mapping(metrics.get("mirror"), "metrics.mirror"))),
    }
    candidate["color_eligible"] = _color_eligible(candidate)
    candidate["continuation_gate_passed"] = _continuation_passed(
        candidate, _mapping(screen.get("selection_policy"), "screen.selection_policy")
    )
    candidate["full_teacher_gate_passed"] = _full_teacher_passed(candidate)
    return candidate


def select_final_extension(
    manifest_path: str | Path,
    *,
    current_provenance: Mapping[str, Any] | None = None,
    profile: PhaseAwareLocalFieldProfile = V21_LOCAL_FIELD_PROFILE,
) -> dict[str, Any]:
    """Validate and rank the exact selected-prefix plus update-8 trajectory."""

    if profile is V21_LOCAL_FIELD_PROFILE:
        manifest, screen, config = _validate_launch_manifest(
            manifest_path,
            current_provenance=current_provenance,
        )
        contract = _validate_config(config)
    else:
        manifest, screen, config = _validate_launch_manifest(
            manifest_path,
            current_provenance=current_provenance,
            profile=profile,
        )
        contract = _validate_config(config, profile=profile)
    selected_epoch = int(manifest["selected_epoch"])
    extension_root = _lexical_absolute(str(manifest["extension_checkpoint_root"]))
    original_root = artifact_root(config, "checkpoints").resolve() / profile.output_namespace
    if (
        extension_root.resolve() == original_root.resolve()
        or extension_root.name != profile.extension_namespace
    ):
        _fail("Launch extension root is not the exact isolated V21 namespace")
    expected_epochs = list(range(selected_epoch + 1, TARGET_OPTIMIZER_UPDATE + 1))
    if manifest.get("expected_extension_epochs") != expected_epochs:
        _fail("Launch extension epoch sequence is inconsistent")
    selected_metadata_path = _safe_input_file(
        _lexical_absolute(str(manifest["selected_checkpoint"])) / "metadata.json",
        "selected V21 checkpoint metadata",
    )
    try:
        selected_metadata, selected_metadata_sha256 = _load_json_strict(selected_metadata_path)
    except (ValueError, FileNotFoundError, OSError, UnicodeError) as error:
        _fail(f"Cannot reload selected V21 checkpoint metadata: {error}")
    selected_hashes = _checkpoint_hashes(selected_metadata_path.parent, "selected_checkpoint")
    if selected_hashes != manifest.get("selected_checkpoint_artifact_hashes"):
        _fail("Selected V21 checkpoint changed after extension authorization")
    if selected_metadata_sha256 != selected_hashes["metadata_sha256"]:
        raise AssertionError("Selected metadata changed while validating")

    extension_rows = []
    for epoch in expected_epochs:
        arguments = (
            epoch,
            extension_root / f"epoch_{epoch:03d}" / "metadata.json",
            config,
            contract,
            _mapping(manifest.get("source_provenance"), "launch.source_provenance"),
        )
        if profile is V21_LOCAL_FIELD_PROFILE:
            extension_rows.append(_validate_extension_epoch(*arguments))
        else:
            extension_rows.append(_validate_extension_epoch(*arguments, profile=profile))
    selected_history = list(_sequence(selected_metadata.get("history"), "selected history"))
    if len(selected_history) != selected_epoch:
        _fail("Selected checkpoint history length differs from selected epoch")
    previous_history = selected_history
    previous_initialization = selected_metadata.get("initialization_provenance")
    previous_equivalence = selected_metadata.get("signed_x_scene_residual_zero_output_equivalence")
    for row in extension_rows:
        history = list(row["history"])
        if history[: len(previous_history)] != previous_history:
            _fail(f"Extension update {row['epoch']} forks cumulative history")
        if row["initialization_provenance"] != previous_initialization:
            _fail(f"Extension update {row['epoch']} changed V18 initialization provenance")
        if row["zero_output_equivalence"] != previous_equivalence:
            _fail(f"Extension update {row['epoch']} changed update-0 equivalence")
        previous_history = history

    screen_epochs = {
        int(_mapping(value, "screen.epochs[]")["epoch"]): dict(_mapping(value, "screen.epochs[]"))
        for value in _sequence(screen.get("epochs"), "screen.epochs")
    }
    candidates: list[dict[str, Any]] = []
    state_hashes: list[str] = []
    for epoch in range(1, selected_epoch + 1):
        row = screen_epochs[epoch]
        candidate = _candidate(
            epoch,
            row,
            metadata_path=str(row["checkpoint_metadata_path"]),
            metadata_sha256=str(row["checkpoint_metadata_sha256"]),
            artifact_hashes=_mapping(
                row.get("checkpoint_artifact_hashes"),
                f"screen epoch {epoch} artifact hashes",
            ),
            state_sha256=str(row["signed_x_state_sha256"]),
            optimizer_state_sha256=str(row["optimizer_state_sha256"]),
            screen=screen,
        )
        candidates.append(candidate)
        state_hashes.append(candidate["signed_x_state_sha256"])
    extension_artifacts: list[dict[str, Any]] = []
    for row in extension_rows:
        epoch = int(row["epoch"])
        candidate = _candidate(
            epoch,
            row["metrics"],
            metadata_path=str(row["path"]),
            metadata_sha256=str(row["artifact_sha256"]),
            artifact_hashes=row["checkpoint_artifact_hashes"],
            state_sha256=str(row["signed_x_state_sha256"]),
            optimizer_state_sha256=str(row["optimizer_state_sha256"]),
            screen=screen,
        )
        candidates.append(candidate)
        state_hashes.append(candidate["signed_x_state_sha256"])
        extension_artifacts.append(
            {
                "epoch": epoch,
                "checkpoint": _display(extension_root / f"epoch_{epoch:03d}"),
                "artifact_hashes": row["checkpoint_artifact_hashes"],
                "optimizer_state_sha256": row["optimizer_state_sha256"],
            }
        )
    if [candidate["epoch"] for candidate in candidates] != list(
        range(1, TARGET_OPTIMIZER_UPDATE + 1)
    ):
        _fail("Final V21 trajectory does not cover optimizer updates 1 through 8")
    first_seen: dict[str, dict[str, Any]] = {}
    repeated_plateau: list[int] = []
    for candidate, state_hash in zip(candidates, state_hashes, strict=True):
        epoch = int(candidate["epoch"])
        prior = first_seen.get(state_hash)
        if prior is None:
            first_seen[state_hash] = candidate
            continue
        exact_metrics_match = all(candidate[key] == prior[key] for key in ("color", "mirror"))
        both_full_teacher = all(
            row["full_teacher_gate_passed"] is True for row in (prior, candidate)
        )
        if int(prior["epoch"]) != epoch - 1 or not both_full_teacher or not exact_metrics_match:
            _fail("V21 trajectory repeats or rolls back state before a full-teacher plateau")
        repeated_plateau.append(epoch)
        first_seen[state_hash] = candidate

    ranking = sorted(
        (deepcopy(candidate) for candidate in candidates if candidate["color_eligible"]),
        key=_ranking_key,
    )
    if not ranking:
        _fail("Final V21 trajectory has no color-eligible checkpoint")
    for rank, candidate in enumerate(ranking, start=1):
        candidate["rank"] = rank
    selected = ranking[0]
    full_teacher = bool(selected["full_teacher_gate_passed"])
    return {
        "schema_version": 1,
        "selector_type": profile.extension_final_selector_type,
        "report_only": True,
        "model_inference_executed": False,
        "gemma_model_loaded": False,
        "checkpoint_tensor_state_loaded": True,
        "checkpoint_tensor_state_safely_inspected": True,
        "optimizer_deserialized": True,
        "optimizer_deserialization_weights_only": True,
        "question_dependent_scene_processing": False,
        "model_dtype": MODEL_DTYPE,
        "config_path": manifest["config_path"],
        "config_hash": manifest["config_hash"],
        "config_hash_full": manifest["config_hash_full"],
        "preflight_contract_sha256": manifest["preflight_contract_sha256"],
        "launch_manifest_path": _display(_resolve(manifest_path)),
        "launch_manifest_sha256": _file_sha256(_resolve(manifest_path), "launch_manifest"),
        "launch_manifest_canonical_sha256": _canonical_sha256(manifest),
        "screen_report_path": manifest["screen_report_path"],
        "screen_report_sha256": manifest["screen_report_sha256"],
        "update1_authorization": deepcopy(manifest["update1_authorization"]),
        "source_provenance": deepcopy(manifest["source_provenance"]),
        "original_selected_epoch": selected_epoch,
        "original_selected_checkpoint_artifact_hashes": selected_hashes,
        "extension_output_namespace": profile.extension_namespace,
        "target_optimizer_update": TARGET_OPTIMIZER_UPDATE,
        "conditional_limit_reached": True,
        "cumulative_update_evidence": {
            "optimizer_steps": list(range(1, 9)),
            "cumulative_microsteps": [epoch * 12 for epoch in range(1, 9)],
            "selected_screen_history_prefix_exact": True,
            "extension_history_prefixes_exact": True,
            "update1_authorization_transitively_bound": True,
            "all_checkpoint_artifact_hashes_bound": True,
            "all_optimizer_steps_safely_validated": True,
            "signed_x_states_unique_before_full_teacher_plateau": True,
            "repeated_full_teacher_plateau_epochs": repeated_plateau,
            "frozen_global_state_sha256": EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
            "frozen_scene_state_sha256": EXPECTED_FROZEN_SCENE_SHA256,
            "model_dtype": MODEL_DTYPE,
        },
        "extension_checkpoint_artifacts": extension_artifacts,
        "epoch_count": len(candidates),
        "epochs": candidates,
        "eligible_epoch_count": len(ranking),
        "ranking": ranking,
        "selected_epoch": selected["epoch"],
        "selected_checkpoint_metadata_path": selected["checkpoint_metadata_path"],
        "selected_checkpoint_metadata_sha256": selected["checkpoint_metadata_sha256"],
        "selected_checkpoint_artifact_hashes": deepcopy(selected["checkpoint_artifact_hashes"]),
        "selected_optimizer_state_sha256": selected["optimizer_state_sha256"],
        "selected_signed_x_state_sha256": selected["signed_x_state_sha256"],
        "continuation_authorized": False,
        "full_teacher_gate_passed": full_teacher,
        "greedy_audit_authorized": full_teacher,
        "greedy_audit_forbidden": not full_teacher,
        "decision": (
            "full_teacher_gate_passed_greedy_audit_allowed"
            if full_teacher
            else "conditional_limit_reached_no_greedy_audit"
        ),
    }


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    destination = _resolve(path)
    _reject_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=PINNED_CONFIG_PATH)
    prepare.add_argument("--screen", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("select-final")
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        report = prepare_extension_launch(args.config, args.screen)
        destination = write_report(report, args.output)
        print(
            json.dumps(
                {
                    "output": str(destination),
                    "authorized": True,
                    "selected_epoch": report["selected_epoch"],
                    "extension_output_namespace": EXTENSION_NAMESPACE,
                    "trainer_argv": report["trainer"]["argv"],
                },
                sort_keys=True,
            )
        )
        return 0
    report = select_final_extension(args.manifest)
    destination = write_report(report, args.output)
    print(
        json.dumps(
            {
                "output": str(destination),
                "selected_epoch": report["selected_epoch"],
                "full_teacher_gate_passed": report["full_teacher_gate_passed"],
                "greedy_audit_authorized": report["greedy_audit_authorized"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

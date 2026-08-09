"""Fail-closed authorization and final selection for V19 updates 5--12.

The four-update V19 screen deliberately has a conditional continuation gate.
This module turns that gate into a narrow, reproducible continuation without
weakening the original selector or reusing its checkpoint namespace:

``prepare``
    Recompute and byte-bind the exact V19 screen report, require continuation
    without greedy authorization, bind the selected checkpoint's three files,
    require the repository to be at the same clean committed source, refuse an
    existing extension namespace, and emit the exact trainer argv.

``select-final``
    Revalidate the launch authorization and every cumulative extension
    checkpoint through optimizer update 12, then apply the original V19
    ranking and full-teacher gate to the actual resumed trajectory.

Both operations are report-only.  They read JSON and hash checkpoint files;
they never deserialize tensor state or load a model, map, scene token, question
file, rendered observation, runtime artifact, or oracle artifact.
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
from semantic_3d_chat.evaluation.v19_epoch_selector import (
    EXPECTED_EPOCHS,
    EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
    OUTPUT_NAMESPACE,
    PINNED_CONFIG_PATH,
    V19EpochSelectorViolation,
    _color_eligible,
    _continuation_passed,
    _full_teacher_passed,
    _load_json_strict,
    _ranking_key,
    _reject_forbidden_input_path,
    _validate_config,
    _validate_epoch_artifact,
    summarize_v19_epochs,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)

EXTENSION_NAMESPACE = "gemma4_color_mirror_signed_x_moment_v19_extension_u12"
CONTROLLER_TYPE = "strict_v19_conditional_extension_controller"
FINAL_SELECTOR_TYPE = "strict_v19_conditional_extension_final_selector"
TARGET_OPTIMIZER_UPDATE = 12
MICROSTEPS_PER_UPDATE = 12
PYTHON_EXECUTABLE = ".venv-gemma4/bin/python"
TRAINING_MODULE = "semantic_3d_chat.training.train_adapter"


class V19ExtensionViolation(ValueError):
    """A mismatch that denies a V19 conditional launch or final selection."""


def _fail(message: str) -> None:
    raise V19ExtensionViolation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _exact_int(value: Any, expected: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        _fail(f"{field} must equal {expected}")
    return value


def _file_sha256(path: Path, field: str) -> str:
    if not path.is_file():
        _fail(f"{field} is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail(f"Cannot canonically hash report value: {error}")
    return hashlib.sha256(payload).hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


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
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Recompute the complete four-update report from its bound artifacts."""

    resolved_config = _resolve(config_path)
    resolved_screen = _resolve(screen_path)
    _reject_forbidden_input_path(resolved_config)
    try:
        config = load_config(resolved_config)
        raw_screen, screen_sha256 = _load_json_strict(resolved_screen)
        screen = dict(raw_screen)
        selection_path_raw = screen.get("selection_artifact_path")
        if not isinstance(selection_path_raw, str) or not selection_path_raw:
            _fail("screen.selection_artifact_path is missing")
        selection_path = Path(selection_path_raw)
        selection, selection_sha256 = _load_json_strict(_resolve(selection_path))
        rows = _sequence(screen.get("epochs"), "screen.epochs")
        if len(rows) != len(EXPECTED_EPOCHS):
            _fail("screen.epochs must contain exactly updates 1,2,3,4")
        epoch_paths: dict[int, Path] = {}
        for index, value in enumerate(rows, start=1):
            row = _mapping(value, f"screen.epochs[{index - 1}]")
            epoch = row.get("epoch")
            if (
                isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or epoch not in EXPECTED_EPOCHS
            ):
                _fail(f"screen.epochs[{index - 1}].epoch is invalid")
            if epoch in epoch_paths:
                _fail(f"screen.epochs repeats update {epoch}")
            raw_path = row.get("checkpoint_metadata_path")
            if not isinstance(raw_path, str) or not raw_path:
                _fail(f"screen.epochs[{index - 1}].checkpoint_metadata_path is invalid")
            epoch_paths[epoch] = Path(raw_path)
        if set(epoch_paths) != set(EXPECTED_EPOCHS):
            _fail("screen.epochs does not cover exactly updates 1,2,3,4")
        loaded = {epoch: _load_json_strict(_resolve(path)) for epoch, path in epoch_paths.items()}
        recomputed = summarize_v19_epochs(
            config,
            selection,
            {epoch: value for epoch, (value, _digest) in loaded.items()},
            selection_path=str(selection_path),
            selection_sha256=selection_sha256,
            epoch_paths={epoch: str(path) for epoch, path in epoch_paths.items()},
            epoch_sha256={epoch: digest for epoch, (_value, digest) in loaded.items()},
        )
    except V19ExtensionViolation:
        raise
    except (
        V19EpochSelectorViolation,
        FileNotFoundError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        _fail(f"Cannot validate exact V19 screen report: {error}")
    if screen != recomputed:
        _fail("V19 screen report differs from exact recomputation of its bound artifacts")
    return screen, screen_sha256, config, dict(selection)


def _require_extension_decision(screen: Mapping[str, Any]) -> int:
    if screen.get("selector_type") != "strict_v19_signed_x_moment_epoch_selector":
        _fail("screen.selector_type is not the strict V19 selector")
    if screen.get("report_only") is not True or screen.get("model_inference_executed") is not False:
        _fail("screen is not report-only evidence")
    if screen.get("checkpoint_tensor_state_loaded") is not False:
        _fail("screen unexpectedly deserialized checkpoint tensor state")
    if screen.get("continuation_authorized") is not True:
        _fail("V19 conditional continuation was not authorized")
    if screen.get("continuation_gate_passed") is not True:
        _fail("V19 continuation gate did not pass")
    if screen.get("full_teacher_gate_passed") is not False:
        _fail("V19 extension is forbidden after the full teacher gate already passed")
    if screen.get("greedy_audit_authorized") is not False:
        _fail("V19 extension is forbidden when greedy audit is already authorized")
    if screen.get("greedy_audit_forbidden") is not True:
        _fail("V19 screen does not explicitly forbid greedy audit during continuation")
    if screen.get("decision") != "continue_selected_epoch_no_greedy_audit":
        _fail("V19 screen decision is not the predeclared continuation-only decision")
    _exact_int(
        screen.get("conditional_max_optimizer_updates"),
        TARGET_OPTIMIZER_UPDATE,
        "screen.conditional_max_optimizer_updates",
    )
    selected_epoch = screen.get("selected_epoch")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch not in EXPECTED_EPOCHS
    ):
        _fail("screen.selected_epoch must be one of updates 1,2,3,4")
    return selected_epoch


def _require_current_source(
    screen: Mapping[str, Any],
    current_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current = dict(
        capture_git_source_provenance(PROJECT_ROOT)
        if current_provenance is None
        else current_provenance
    )
    try:
        require_clean_committed_source(current)
    except RuntimeError as error:
        _fail(f"V19 extension requires clean committed source: {error}")
    if current != screen.get("source_provenance"):
        _fail("Current clean source provenance differs from the exact V19 screen")
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
) -> dict[str, Any]:
    selected_epoch = _require_extension_decision(screen)
    contract = _validate_config(config)
    selected_metadata_raw = screen.get("selected_checkpoint_metadata_path")
    if not isinstance(selected_metadata_raw, str) or not selected_metadata_raw:
        _fail("screen.selected_checkpoint_metadata_path is missing")
    checkpoint_root = artifact_root(dict(config), "checkpoints").resolve()
    original_root = (checkpoint_root / OUTPUT_NAMESPACE).resolve()
    extension_root = (checkpoint_root / EXTENSION_NAMESPACE).resolve()
    selected_checkpoint = (original_root / f"epoch_{selected_epoch:03d}").resolve()
    _same_path(selected_metadata_raw, selected_checkpoint / "metadata.json", "selected checkpoint")
    if extension_root == original_root or extension_root.is_relative_to(original_root):
        _fail("Extension checkpoint namespace is not isolated from the original V19 screen")
    if require_namespace_absent and extension_root.exists():
        _fail(f"Refusing to reuse or overwrite existing V19 extension namespace: {extension_root}")
    selected_hashes = _checkpoint_hashes(selected_checkpoint, "selected_checkpoint")
    if selected_hashes["metadata_sha256"] != screen.get("selected_checkpoint_metadata_sha256"):
        _fail("Selected checkpoint metadata no longer matches the V19 screen hash")
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
        EXTENSION_NAMESPACE,
        "--epochs",
        str(TARGET_OPTIMIZER_UPDATE),
    ]
    expected_epochs = list(range(selected_epoch + 1, TARGET_OPTIMIZER_UPDATE + 1))
    return {
        "schema_version": 1,
        "controller_type": CONTROLLER_TYPE,
        "authorized": True,
        "report_only": True,
        "model_inference_executed": False,
        "checkpoint_tensor_state_loaded": False,
        "question_dependent_scene_processing": False,
        "config_path": _display(config_path),
        "config_hash": contract["config_hash"],
        "screen_report_path": _display(screen_path),
        "screen_report_sha256": screen_sha256,
        "screen_report_canonical_sha256": _canonical_sha256(screen),
        "source_provenance": deepcopy(dict(source_provenance)),
        "screen_decision": "continue_selected_epoch_no_greedy_audit",
        "continuation_authorized": True,
        "greedy_audit_authorized": False,
        "greedy_audit_forbidden_during_extension": True,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": _display(selected_checkpoint),
        "selected_checkpoint_artifact_hashes": selected_hashes,
        "original_output_namespace": OUTPUT_NAMESPACE,
        "extension_output_namespace": EXTENSION_NAMESPACE,
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
) -> dict[str, Any]:
    """Authorize, but never execute, the isolated V19 update-12 continuation."""

    screen, screen_sha256, config, _selection = _load_exact_screen_report(config_path, screen_path)
    source = _require_current_source(screen, current_provenance)
    return _build_launch_manifest(
        config_path=_resolve(config_path),
        screen_path=_resolve(screen_path),
        screen=screen,
        screen_sha256=screen_sha256,
        config=config,
        source_provenance=source,
        require_namespace_absent=True,
    )


def _validate_launch_manifest(
    manifest_path: str | Path,
    *,
    current_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved_manifest = _resolve(manifest_path)
    try:
        raw_manifest, _manifest_sha256 = _load_json_strict(resolved_manifest)
    except (V19EpochSelectorViolation, FileNotFoundError, OSError, UnicodeError) as error:
        _fail(f"Cannot load V19 extension launch manifest: {error}")
    manifest = dict(raw_manifest)
    if manifest.get("controller_type") != CONTROLLER_TYPE or manifest.get("authorized") is not True:
        _fail("V19 extension launch manifest is not an authorization from this controller")
    config_path = manifest.get("config_path")
    screen_path = manifest.get("screen_report_path")
    if not isinstance(config_path, str) or not config_path:
        _fail("launch.config_path is invalid")
    if not isinstance(screen_path, str) or not screen_path:
        _fail("launch.screen_report_path is invalid")
    screen, screen_sha256, config, _selection = _load_exact_screen_report(config_path, screen_path)
    source = _require_current_source(screen, current_provenance)
    expected = _build_launch_manifest(
        config_path=_resolve(config_path),
        screen_path=_resolve(screen_path),
        screen=screen,
        screen_sha256=screen_sha256,
        config=config,
        source_provenance=source,
        require_namespace_absent=False,
    )
    if manifest != expected:
        _fail("V19 extension launch manifest differs from exact current authorization")
    return manifest, screen, config


def _validate_extension_epoch(
    epoch: int,
    metadata_path: Path,
    contract: Mapping[str, Any],
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        raw, metadata_sha256 = _load_json_strict(metadata_path)
    except (V19EpochSelectorViolation, FileNotFoundError, OSError, UnicodeError) as error:
        _fail(f"Cannot load V19 extension update {epoch}: {error}")
    artifact = dict(raw)
    if artifact.get("output_namespace") != EXTENSION_NAMESPACE:
        _fail(f"Extension update {epoch} is not in the isolated output namespace")
    if artifact.get("source_provenance") != expected_source:
        _fail(f"Extension update {epoch} source provenance differs from authorization")
    # The original validator pins every scientific field and cumulative-history
    # invariant.  Only the deliberately isolated artifact namespace differs.
    normalized = deepcopy(artifact)
    normalized["output_namespace"] = OUTPUT_NAMESPACE
    try:
        validated = _validate_epoch_artifact(
            epoch,
            normalized,
            contract,
            path=_display(metadata_path),
            artifact_sha256=metadata_sha256,
        )
    except V19EpochSelectorViolation as error:
        _fail(f"Extension update {epoch} violates the strict V19 contract: {error}")
    checkpoint = metadata_path.parent
    validated["checkpoint_artifact_hashes"] = _checkpoint_hashes(
        checkpoint, f"extension_update_{epoch}"
    )
    if validated["checkpoint_artifact_hashes"]["metadata_sha256"] != metadata_sha256:
        raise AssertionError("metadata hash changed while validating extension checkpoint")
    return validated


def _candidate_from_metrics(
    epoch: int,
    metrics: Mapping[str, Any],
    *,
    metadata_path: str,
    metadata_sha256: str,
    signed_x_state_sha256: str,
    screen: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = {
        "epoch": epoch,
        "optimizer_step": epoch,
        "cumulative_microsteps": epoch * MICROSTEPS_PER_UPDATE,
        "checkpoint_metadata_path": metadata_path,
        "checkpoint_metadata_sha256": metadata_sha256,
        "signed_x_state_sha256": signed_x_state_sha256,
        "frozen_global_scene_residual_state_sha256": (EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
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
) -> dict[str, Any]:
    """Validate and rank the exact selected-prefix plus resumed update-12 branch."""

    manifest, screen, config = _validate_launch_manifest(
        manifest_path, current_provenance=current_provenance
    )
    contract = _validate_config(config)
    selected_epoch = int(manifest["selected_epoch"])
    extension_root = _resolve(str(manifest["extension_checkpoint_root"]))
    original_root = artifact_root(config, "checkpoints").resolve() / OUTPUT_NAMESPACE
    if extension_root == original_root.resolve() or extension_root.name != EXTENSION_NAMESPACE:
        _fail("Launch manifest extension root is not the exact isolated namespace")
    expected_epochs = list(range(selected_epoch + 1, TARGET_OPTIMIZER_UPDATE + 1))
    if manifest.get("expected_extension_epochs") != expected_epochs:
        _fail("Launch manifest extension epoch sequence is inconsistent")
    selected_metadata_path = _resolve(str(manifest["selected_checkpoint"])) / "metadata.json"
    try:
        selected_metadata, selected_metadata_sha256 = _load_json_strict(selected_metadata_path)
    except (V19EpochSelectorViolation, FileNotFoundError, OSError, UnicodeError) as error:
        _fail(f"Cannot reload selected V19 checkpoint metadata: {error}")
    selected_hashes = _checkpoint_hashes(selected_metadata_path.parent, "selected_checkpoint")
    if selected_hashes != manifest.get("selected_checkpoint_artifact_hashes"):
        _fail("Selected V19 checkpoint changed after extension authorization")
    if selected_metadata_sha256 != selected_hashes["metadata_sha256"]:
        raise AssertionError("selected metadata hash changed while validating")

    extension_rows: list[dict[str, Any]] = []
    for epoch in expected_epochs:
        metadata_path = extension_root / f"epoch_{epoch:03d}" / "metadata.json"
        extension_rows.append(
            _validate_extension_epoch(
                epoch,
                metadata_path,
                contract,
                _mapping(manifest.get("source_provenance"), "launch.source_provenance"),
            )
        )

    selected_history = list(
        _sequence(selected_metadata.get("history"), "selected_checkpoint.history")
    )
    if len(selected_history) != selected_epoch:
        _fail("Selected checkpoint cumulative history length differs from selected epoch")
    previous_history = selected_history
    previous_initialization = selected_metadata.get("initialization_provenance")
    previous_equivalence = selected_metadata.get("signed_x_scene_residual_zero_output_equivalence")
    for row in extension_rows:
        history = list(row["history"])
        if history[: len(previous_history)] != previous_history:
            _fail(f"Extension update {row['epoch']} forks or rewrites cumulative history")
        if row["initialization_provenance"] != previous_initialization:
            _fail(f"Extension update {row['epoch']} changed V18 initialization provenance")
        if row["zero_output_equivalence"] != previous_equivalence:
            _fail(f"Extension update {row['epoch']} changed update-0 prefix equivalence")
        previous_history = history

    screen_epochs = {
        int(_mapping(value, "screen.epochs[]")["epoch"]): dict(_mapping(value, "screen.epochs[]"))
        for value in _sequence(screen.get("epochs"), "screen.epochs")
    }
    candidates: list[dict[str, Any]] = []
    state_hashes: list[str] = []
    for epoch in range(1, selected_epoch + 1):
        row = screen_epochs[epoch]
        candidate = _candidate_from_metrics(
            epoch,
            row,
            metadata_path=str(row["checkpoint_metadata_path"]),
            metadata_sha256=str(row["checkpoint_metadata_sha256"]),
            signed_x_state_sha256=str(row["signed_x_state_sha256"]),
            screen=screen,
        )
        candidates.append(candidate)
        state_hashes.append(candidate["signed_x_state_sha256"])
    extension_artifacts: list[dict[str, Any]] = []
    for row in extension_rows:
        epoch = int(row["epoch"])
        metrics = row["metrics"]
        candidate = _candidate_from_metrics(
            epoch,
            metrics,
            metadata_path=str(row["path"]),
            metadata_sha256=str(row["artifact_sha256"]),
            signed_x_state_sha256=str(row["signed_x_state_sha256"]),
            screen=screen,
        )
        candidates.append(candidate)
        state_hashes.append(candidate["signed_x_state_sha256"])
        extension_artifacts.append(
            {
                "epoch": epoch,
                "checkpoint": _display(extension_root / f"epoch_{epoch:03d}"),
                "artifact_hashes": row["checkpoint_artifact_hashes"],
            }
        )
    if [candidate["epoch"] for candidate in candidates] != list(
        range(1, TARGET_OPTIMIZER_UPDATE + 1)
    ):
        _fail("Final V19 trajectory does not cover optimizer updates 1 through 12")
    first_seen_state: dict[str, int] = {}
    repeated_full_gate_epochs: list[int] = []
    for candidate, state_hash in zip(candidates, state_hashes, strict=True):
        epoch = int(candidate["epoch"])
        prior_epoch = first_seen_state.get(state_hash)
        if prior_epoch is None:
            first_seen_state[state_hash] = epoch
            continue
        if prior_epoch != epoch - 1 or candidate["full_teacher_gate_passed"] is not True:
            _fail(
                "Final V19 trajectory repeats or rolls back a signed-X state before a "
                "full-teacher plateau"
            )
        repeated_full_gate_epochs.append(epoch)
        first_seen_state[state_hash] = epoch

    ranking = sorted(
        (deepcopy(candidate) for candidate in candidates if candidate["color_eligible"]),
        key=_ranking_key,
    )
    if not ranking:
        _fail("Final V19 trajectory unexpectedly has no color-eligible checkpoint")
    for rank, candidate in enumerate(ranking, start=1):
        candidate["rank"] = rank
    selected = ranking[0]
    full_teacher_gate_passed = bool(selected["full_teacher_gate_passed"])
    greedy_authorized = bool(full_teacher_gate_passed)
    return {
        "schema_version": 1,
        "selector_type": FINAL_SELECTOR_TYPE,
        "report_only": True,
        "model_inference_executed": False,
        "checkpoint_tensor_state_loaded": False,
        "question_dependent_scene_processing": False,
        "config_path": manifest["config_path"],
        "config_hash": manifest["config_hash"],
        "launch_manifest_path": _display(_resolve(manifest_path)),
        "launch_manifest_sha256": _file_sha256(_resolve(manifest_path), "launch_manifest"),
        "launch_manifest_canonical_sha256": _canonical_sha256(manifest),
        "screen_report_path": manifest["screen_report_path"],
        "screen_report_sha256": manifest["screen_report_sha256"],
        "source_provenance": deepcopy(manifest["source_provenance"]),
        "original_selected_epoch": selected_epoch,
        "original_selected_checkpoint_artifact_hashes": selected_hashes,
        "extension_output_namespace": EXTENSION_NAMESPACE,
        "target_optimizer_update": TARGET_OPTIMIZER_UPDATE,
        "conditional_limit_reached": True,
        "cumulative_update_evidence": {
            "optimizer_steps": list(range(1, TARGET_OPTIMIZER_UPDATE + 1)),
            "cumulative_microsteps": [
                epoch * MICROSTEPS_PER_UPDATE for epoch in range(1, TARGET_OPTIMIZER_UPDATE + 1)
            ],
            "selected_screen_history_prefix_exact": True,
            "extension_history_prefixes_exact": True,
            "signed_x_states_unique_before_full_teacher_plateau": True,
            "repeated_full_teacher_plateau_epochs": repeated_full_gate_epochs,
            "frozen_global_state_sha256": EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
        },
        "extension_checkpoint_artifacts": extension_artifacts,
        "epoch_count": len(candidates),
        "epochs": candidates,
        "eligible_epoch_count": len(ranking),
        "ranking": ranking,
        "selected_epoch": selected["epoch"],
        "selected_checkpoint_metadata_path": selected["checkpoint_metadata_path"],
        "selected_checkpoint_metadata_sha256": selected["checkpoint_metadata_sha256"],
        "selected_signed_x_state_sha256": selected["signed_x_state_sha256"],
        "continuation_authorized": False,
        "full_teacher_gate_passed": full_teacher_gate_passed,
        "greedy_audit_authorized": greedy_authorized,
        "greedy_audit_forbidden": not greedy_authorized,
        "decision": (
            "full_teacher_gate_passed_greedy_audit_allowed"
            if greedy_authorized
            else "conditional_limit_reached_no_greedy_audit"
        ),
    }


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    destination = _resolve(path)
    _reject_forbidden_input_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
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
    prepare_parser = subparsers.add_parser("prepare", help="authorize and emit trainer argv")
    prepare_parser.add_argument("--config", type=Path, default=PINNED_CONFIG_PATH)
    prepare_parser.add_argument("--screen", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    final_parser = subparsers.add_parser(
        "select-final", help="validate and rank the cumulative update-12 trajectory"
    )
    final_parser.add_argument("--manifest", type=Path, required=True)
    final_parser.add_argument("--output", type=Path, required=True)
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
                    "trainer_shell_command": report["trainer"]["shell_command"],
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


if __name__ == "__main__":  # pragma: no cover - exercised through the public CLI
    raise SystemExit(main())

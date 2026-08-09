"""Fail-closed V24 replay gate and bounded update-eight selector.

The sealed four-update V24 screen selected update one.  A continuation must
therefore resume that exact checkpoint, reproduce updates two through four in
an isolated namespace, and prove that the adapter tensors, decoded optimizer
state, cumulative history, metrics, and opaque margin detail are identical
before novel updates five through eight are allowed.  Final selection ranks
the complete unique trajectory: primary update one plus isolated updates two
through eight, with replay and novel rows labeled separately.

This module is report-only.  It safely inspects checkpoint tensors and
optimizer state on CPU, but never loads Gemma, performs inference, or reads
questions, maps, scene tokens, rendered observations, or oracle artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, config_hash, load_config
from semantic_3d_chat.evaluation import v24_shared_query_controller as shared
from semantic_3d_chat.language.lora import (
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
)
from semantic_3d_chat.training.source_provenance import capture_git_source_provenance
from semantic_3d_chat.training.train_adapter import file_sha256

CONFIG_PATH = shared.CONFIG_PATH
SCREEN_PATH = Path("reports/gemma4/metrics/v24_epoch_screen.json")
UPDATE1_PATH = Path("reports/gemma4/metrics/v24_update1_match.json")
MANIFEST_PATH = Path("reports/gemma4/metrics/v24_extension_launch.json")
REPLAY_REPORT_PATH = Path("reports/gemma4/metrics/v24_extension_replay.json")
FINAL_REPORT_PATH = Path("reports/gemma4/metrics/v24_extension_final.json")
PRIMARY_NAMESPACE = shared.PRIMARY_NAMESPACE
EXTENSION_NAMESPACE = shared.EXTENSION_NAMESPACE
SELECTED_EPOCH = 1
REPLAY_EPOCHS = (2, 3, 4)
NOVEL_EPOCHS = (5, 6, 7, 8)
TARGET_OPTIMIZER_UPDATE = 8
MICROSTEPS_PER_UPDATE = 12

# These values seal the exact evidence that authorized the conditional branch.
EXPECTED_SCREEN_SHA256 = "146890f2106d473ee0f6ea72facf43c00243000900a5eef1fd4b68270113b002"
EXPECTED_UPDATE1_SHA256 = "4906d7f2b27b5a770a944c75c97ed0e8adf9b7de615ac77dce8f3816939a277c"
EXPECTED_TRAINING_SOURCE_PROVENANCE = {
    "schema_version": 1,
    "scope": "repository_excluding_generated_artifacts_v1",
    "available": True,
    "is_clean": True,
    "head_commit": "622dd3687756c0d6cebd332de860a7d01899bb8b",
    "head_tree": "45c2126beedd0dc34c34fb1b876cb56a0ae50af3",
    "tracked_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
EXPECTED_SELECTED_ARTIFACTS = {
    "adapter_sha256": "45e0c5affa9cf556e29bab5de418dffb867817b703c848bc6828255347748d31",
    "metadata_sha256": "216c501f5b248aa8f44e86198be3902d5f45d87774ad932056bafc95c4637e7b",
    "optimizer_sha256": "f1121353fc2c6b9239b8163390a0593832825abd9ff9f8dce4cce7f1cff99669",
}
EXPECTED_SELECTED_BANK_SHA256 = "6db2807476506b947bbaf01837490e97c12e57b1906bab671ef7c82ed36d6399"
EXPECTED_SELECTED_OPTIMIZER_STATE_SHA256 = (
    "6ff36bb2913d186fdecc1193bdc9834716092c9a40c312996e3cafcdad8be099"
)
EXPECTED_CONTROL_PLANE_TRANSITION = {
    "Makefile": "M",
    "src/semantic_3d_chat/evaluation/v24_extension_controller.py": "A",
    "tests/test_v24_extension_controller.py": "A",
}


class V24ExtensionViolation(ValueError):
    """An extension authorization, replay, or final artifact violated its contract."""


def _fail(message: str) -> None:
    raise V24ExtensionViolation(message)


def _equal(observed: Any, expected: Any, field: str) -> None:
    try:
        shared._equal(observed, expected, field)
    except shared.V24ControlViolation as error:
        _fail(str(error))


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    try:
        return shared._mapping(value, field)
    except shared.V24ControlViolation as error:
        _fail(str(error))


def _sequence(value: Any, field: str) -> Sequence[Any]:
    try:
        return shared._sequence(value, field)
    except shared.V24ControlViolation as error:
        _fail(str(error))


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()


def _display(path: str | Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _regular_file(path: str | Path, field: str) -> Path:
    try:
        return shared._regular_file(Path(path), field)
    except shared.V24ControlViolation as error:
        _fail(str(error))


def _load_json(path: str | Path, field: str) -> dict[str, Any]:
    try:
        return shared._load_json(Path(path), field)
    except shared.V24ControlViolation as error:
        _fail(str(error))


def _clean_provenance(value: Any, field: str) -> dict[str, Any]:
    try:
        return shared._clean_provenance(value, field)
    except shared.V24ControlViolation as error:
        _fail(str(error))


def _reject_symlink_chain(path: Path, field: str) -> None:
    """Reject symlinks at the target or any in-repository parent component."""

    absolute = path if path.is_absolute() else PROJECT_ROOT / path
    current = absolute
    root = PROJECT_ROOT.resolve()
    while True:
        if current.is_symlink():
            _fail(f"{field} contains a symlink component: {current}")
        if current == root or current.parent == current:
            break
        current = current.parent


def _checkpoint_hashes(checkpoint: Path, field: str) -> dict[str, str]:
    return {
        "adapter_sha256": file_sha256(_regular_file(checkpoint / "adapter.safetensors", field)),
        "metadata_sha256": file_sha256(_regular_file(checkpoint / "metadata.json", field)),
        "optimizer_sha256": file_sha256(_regular_file(checkpoint / "optimizer.pt", field)),
    }


def _selection_policy() -> dict[str, Any]:
    return {
        "eligibility": "complete color pair with positive candidate/full-vocabulary minima",
        "ranking_descending": [
            "mirror_full_vocab_units",
            "mirror_full_vocab_sides",
            "mirror_mean_full_vocab_margin",
            "mirror_minimum_full_vocab_margin",
            "mirror_mean_candidate_margin",
            "mirror_minimum_candidate_margin",
        ],
        "tie_breaker": "earlier_epoch",
        "extension_requires": "selected mirror >=8/12 sides and >=2/6 units",
        "greedy_requires": "both pairs 12/12 sides, 6/6 units, all minima positive",
        "model_inference_during_selection": False,
    }


def _ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return shared._ranking_key(row)
    except shared.V24ControlViolation as error:
        _fail(str(error))


def _validate_update1(update1: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "audit_type",
        "match",
        "stage_2_authorized",
        "report_only",
        "model_loaded",
        "oracle_loaded",
        "preflight_sha256",
        "config_sha256",
        "contract_sha256",
        "checkpoint",
        "checkpoint_artifact_hashes",
        "new_bank_state_sha256",
        "ordered_parameter_shapes",
        "a_tensors_unchanged",
        "b_tensors_all_changed",
        "all_prior_tensors_frozen",
        "optimizer_manifest",
        "recomputed_payload_hashes",
        "color",
        "mirror",
        "opaque_unit_margin_detail",
        "source_provenance",
    }
    _equal(set(update1), expected_keys, "update-1 report keys")
    for field, expected in {
        "schema_version": 1,
        "audit_type": "v24_shared_query_update1_verifier",
        "match": True,
        "stage_2_authorized": True,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "config_sha256": shared.EXPECTED_CONFIG_SHA256,
        "contract_sha256": shared.EXPECTED_CONTRACT_SHA256,
        "checkpoint": f"data_gemma4/checkpoints/{PRIMARY_NAMESPACE}/epoch_001",
        "ordered_parameter_shapes": [list(shape) for shape in shared.EXPECTED_PARAMETER_SHAPES],
        "a_tensors_unchanged": True,
        "b_tensors_all_changed": True,
        "all_prior_tensors_frozen": True,
    }.items():
        _equal(update1.get(field), expected, f"update-1 {field}")
    return _clean_provenance(update1.get("source_provenance"), "training source provenance")


def _load_exact_screen(
    config_path: str | Path = CONFIG_PATH,
    screen_path: str | Path = SCREEN_PATH,
) -> dict[str, Any]:
    """Recompute the sealed screen from its bound checkpoint files."""

    _equal(_resolve(config_path), _resolve(CONFIG_PATH), "V24 config path")
    _equal(_resolve(screen_path), _resolve(SCREEN_PATH), "V24 screen path")
    config = load_config(config_path)
    try:
        shared._validate_contract(config)
    except shared.V24ControlViolation as error:
        _fail(str(error))
    screen_file = _regular_file(screen_path, "V24 screen report")
    _equal(file_sha256(screen_file), EXPECTED_SCREEN_SHA256, "externally pinned screen hash")
    screen = _load_json(screen_file, "V24 screen report")
    expected_screen_keys = {
        "schema_version",
        "audit_type",
        "decision",
        "selected_epoch",
        "selected_checkpoint",
        "continuation_authorized",
        "full_teacher_gate_passed",
        "greedy_audit_authorized",
        "static_chat_authorized",
        "embodied_phase_authorized",
        "report_only",
        "model_loaded",
        "oracle_loaded",
        "question_dependent_scene_processing",
        "config_sha256",
        "contract_sha256",
        "update1_report_sha256",
        "epochs",
        "ranking",
        "selection_policy",
    }
    _equal(set(screen), expected_screen_keys, "screen report keys")
    update1_file = _regular_file(UPDATE1_PATH, "V24 update-1 report")
    _equal(file_sha256(update1_file), EXPECTED_UPDATE1_SHA256, "externally pinned update-1 hash")
    _equal(file_sha256(update1_file), screen.get("update1_report_sha256"), "screen/update-1")
    update1 = _load_json(update1_file, "V24 update-1 report")
    training_source = _validate_update1(update1)
    _equal(
        training_source,
        EXPECTED_TRAINING_SOURCE_PROVENANCE,
        "externally pinned training source provenance",
    )
    rows: list[dict[str, Any]] = []
    for epoch in range(1, 5):
        metadata_path = (
            PROJECT_ROOT
            / "data_gemma4/checkpoints"
            / PRIMARY_NAMESPACE
            / f"epoch_{epoch:03d}/metadata.json"
        )
        try:
            rows.append(shared._epoch_record(config, epoch, metadata_path, training_source))
        except shared.V24ControlViolation as error:
            _fail(str(error))
    epoch1 = rows[0]
    _equal(
        update1.get("checkpoint_artifact_hashes"),
        {
            "adapter_sha256": epoch1["adapter_sha256"],
            "metadata_sha256": epoch1["metadata_sha256"],
            "optimizer_sha256": epoch1["optimizer_sha256"],
        },
        "update-1 artifact binding",
    )
    for key in (
        "new_bank_state_sha256",
        "optimizer_manifest",
        "recomputed_payload_hashes",
        "color",
        "mirror",
        "opaque_unit_margin_detail",
    ):
        _equal(update1.get(key), epoch1[key], f"update-1 epoch-1 {key}")
    eligible = [row for row in rows if shared._color_eligible(row["color"])]
    ranking = sorted(eligible, key=_ranking_key, reverse=True)
    selected = ranking[0] if ranking else None
    full_teacher = bool(
        selected is not None
        and shared._full_pair(selected["color"])
        and shared._full_pair(selected["mirror"])
    )
    continuation = bool(
        selected is not None
        and not full_teacher
        and shared._mirror_continuation(selected["mirror"])
    )
    expected_screen = {
        "schema_version": 1,
        "audit_type": "v24_shared_query_epoch_selector",
        "decision": "screen_passed_extension_authorized_no_greedy_audit",
        "selected_epoch": SELECTED_EPOCH,
        "selected_checkpoint": f"data_gemma4/checkpoints/{PRIMARY_NAMESPACE}/epoch_001",
        "continuation_authorized": True,
        "full_teacher_gate_passed": False,
        "greedy_audit_authorized": False,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "config_sha256": shared.EXPECTED_CONFIG_SHA256,
        "contract_sha256": shared.EXPECTED_CONTRACT_SHA256,
        "update1_report_sha256": file_sha256(update1_file),
        "epochs": rows,
        "ranking": [
            {
                "rank": rank,
                "epoch": row["epoch"],
                "color": row["color"],
                "mirror": row["mirror"],
            }
            for rank, row in enumerate(ranking, start=1)
        ],
        "selection_policy": _selection_policy(),
    }
    _equal(selected["epoch"] if selected else None, SELECTED_EPOCH, "selected epoch")
    _equal(continuation, True, "recomputed continuation gate")
    _equal(full_teacher, False, "recomputed full-teacher gate")
    _equal(screen, expected_screen, "exact recomputed V24 screen")
    selected = rows[SELECTED_EPOCH - 1]
    _equal(
        {
            "adapter_sha256": selected["adapter_sha256"],
            "metadata_sha256": selected["metadata_sha256"],
            "optimizer_sha256": selected["optimizer_sha256"],
        },
        EXPECTED_SELECTED_ARTIFACTS,
        "externally pinned selected artifacts",
    )
    _equal(
        selected["new_bank_state_sha256"],
        EXPECTED_SELECTED_BANK_SHA256,
        "externally pinned selected bank",
    )
    _equal(
        selected["optimizer_manifest"]["all_state_tensors_sha256"],
        EXPECTED_SELECTED_OPTIMIZER_STATE_SHA256,
        "externally pinned selected optimizer state",
    )
    return {
        "config": config,
        "screen": screen,
        "screen_sha256": file_sha256(screen_file),
        "training_source_provenance": training_source,
        "epochs": rows,
        "selected": selected,
    }


def _git(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _validate_control_plane_transition(
    training_source: Mapping[str, Any],
    *,
    current_provenance: Mapping[str, Any] | None = None,
    transition: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Require one exact, clean, committed report-only source transition."""

    current = _clean_provenance(
        capture_git_source_provenance(PROJECT_ROOT)
        if current_provenance is None
        else current_provenance,
        "controller source provenance",
    )
    base_commit = str(training_source["head_commit"])
    if current["head_commit"] == base_commit:
        _fail("extension controller must be a committed control-plane transition")
    if transition is None:
        tree = _git(["rev-parse", f"{base_commit}^{{tree}}"])
        if tree.returncode != 0 or tree.stdout.strip() != training_source.get("head_tree"):
            _fail("training source commit/tree is unavailable or changed")
        ancestor = _git(["merge-base", "--is-ancestor", base_commit, str(current["head_commit"])])
        if ancestor.returncode != 0:
            _fail("training source is not an ancestor of controller source")
        diff = _git(
            [
                "diff",
                "--name-status",
                "--no-renames",
                base_commit,
                str(current["head_commit"]),
                "--",
            ]
        )
        if diff.returncode != 0:
            _fail("cannot inspect training-to-controller source transition")
        observed: dict[str, str] = {}
        for line in diff.stdout.splitlines():
            status, separator, path = line.partition("\t")
            if not separator or not path or path in observed:
                _fail("malformed or duplicate control-plane transition record")
            observed[path] = status
    else:
        observed = dict(transition)
    _equal(observed, EXPECTED_CONTROL_PLANE_TRANSITION, "control-plane transition")
    return current, observed


def _extension_root(config: Mapping[str, Any]) -> Path:
    checkpoint_root = artifact_root(dict(config), "checkpoints")
    extension_root = checkpoint_root / EXTENSION_NAMESPACE
    primary_root = checkpoint_root / PRIMARY_NAMESPACE
    _reject_symlink_chain(checkpoint_root, "checkpoint root")
    _reject_symlink_chain(extension_root, "extension root")
    _equal(extension_root.parent, checkpoint_root, "extension namespace parent")
    if extension_root == primary_root or extension_root.is_relative_to(primary_root):
        _fail("extension namespace is not isolated from the primary namespace")
    return extension_root


def _inspect_extension_layout(config: Mapping[str, Any], *, field: str) -> tuple[Path, set[int]]:
    root = _extension_root(config)
    if not root.is_dir() or root.is_symlink():
        _fail(f"{field} is not a regular extension directory: {root}")
    observed: set[int] = set()
    for entry in root.iterdir():
        if entry.name == "best":
            if entry.is_symlink() or not entry.is_dir():
                _fail(f"{field} best entry is not a regular directory: {entry}")
            continue
        if not entry.name.startswith("epoch_"):
            _fail(f"{field} contains an unexpected entry: {entry.name}")
        suffix = entry.name.removeprefix("epoch_")
        if len(suffix) != 3 or not suffix.isdigit() or entry.is_symlink() or not entry.is_dir():
            _fail(f"{field} contains malformed or symlinked epoch entry: {entry}")
        observed.add(int(suffix))
        _reject_symlink_chain(entry, f"{field} epoch {suffix}")
    return root, observed


def _require_extension_layout(config: Mapping[str, Any], expected: set[int], *, field: str) -> Path:
    root, observed = _inspect_extension_layout(config, field=field)
    _equal(observed, expected, f"{field} epoch directory set")
    return root


def _require_replay_or_final_layout(config: Mapping[str, Any], *, field: str) -> Path:
    root, observed = _inspect_extension_layout(config, field=field)
    allowed = (set(REPLAY_EPOCHS), set(range(2, TARGET_OPTIMIZER_UPDATE + 1)))
    if not any(observed == candidate for candidate in allowed):
        _fail(
            f"{field} epoch directory set must be exactly {sorted(allowed[0])} "
            f"or {sorted(allowed[1])}; observed={sorted(observed)}"
        )
    return root


def _selected_checkpoint(evidence: Mapping[str, Any]) -> Path:
    checkpoint = (
        PROJECT_ROOT / "data_gemma4/checkpoints" / PRIMARY_NAMESPACE / f"epoch_{SELECTED_EPOCH:03d}"
    )
    _reject_symlink_chain(checkpoint, "selected checkpoint")
    if not checkpoint.is_dir():
        _fail("selected checkpoint is not a regular directory")
    selected = _mapping(evidence.get("selected"), "selected epoch")
    _equal(
        _checkpoint_hashes(checkpoint, "selected checkpoint"),
        {
            "adapter_sha256": selected["adapter_sha256"],
            "metadata_sha256": selected["metadata_sha256"],
            "optimizer_sha256": selected["optimizer_sha256"],
        },
        "selected checkpoint artifacts",
    )
    return checkpoint


def _validate_history_contract(metadata: Mapping[str, Any], *, epoch: int) -> list[Any]:
    history = list(_sequence(metadata.get("history"), f"extension epoch {epoch} history"))
    _equal(len(history), epoch, f"extension epoch {epoch} history length")
    labels = [
        _mapping(value, f"extension epoch {epoch} history[{index}]").get("epoch")
        for index, value in enumerate(history)
    ]
    _equal(labels, list(range(1, epoch + 1)), f"extension epoch {epoch} history labels")
    final = _mapping(history[-1], f"extension epoch {epoch} final history")
    _equal(
        metadata.get("pair_candidate_gate"),
        final.get("pair_candidate_gate"),
        f"extension epoch {epoch} top-level/history pair gate",
    )
    return history


def _manifest_body(
    evidence: Mapping[str, Any],
    controller_source: Mapping[str, Any],
    transition: Mapping[str, str],
) -> dict[str, Any]:
    selected = _mapping(evidence.get("selected"), "selected epoch")
    checkpoint = _selected_checkpoint(evidence)
    metadata = _load_json(checkpoint / "metadata.json", "selected metadata")
    history = _validate_history_contract(metadata, epoch=SELECTED_EPOCH)
    extension_root = _extension_root(evidence["config"])
    return {
        "schema_version": 1,
        "audit_type": "v24_shared_query_extension_launch",
        "authorized": True,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "config_path": str(CONFIG_PATH),
        "config_sha256": shared.EXPECTED_CONFIG_SHA256,
        "contract_sha256": shared.EXPECTED_CONTRACT_SHA256,
        "screen_path": str(SCREEN_PATH),
        "screen_sha256": evidence["screen_sha256"],
        "screen_decision": "screen_passed_extension_authorized_no_greedy_audit",
        "training_source_provenance": copy.deepcopy(dict(evidence["training_source_provenance"])),
        "controller_source_provenance": copy.deepcopy(dict(controller_source)),
        "control_plane_transition": dict(transition),
        "selected_epoch": SELECTED_EPOCH,
        "selected_checkpoint": _display(checkpoint),
        "selected_checkpoint_artifact_hashes": {
            "adapter_sha256": selected["adapter_sha256"],
            "metadata_sha256": selected["metadata_sha256"],
            "optimizer_sha256": selected["optimizer_sha256"],
        },
        "selected_new_bank_state_sha256": selected["new_bank_state_sha256"],
        "selected_optimizer_manifest": copy.deepcopy(selected["optimizer_manifest"]),
        "selected_history_sha256": shared._canonical_sha256(history),
        "selected_initialization_provenance_sha256": shared._canonical_sha256(
            metadata.get("initialization_provenance")
        ),
        "original_output_namespace": PRIMARY_NAMESPACE,
        "extension_output_namespace": EXTENSION_NAMESPACE,
        "extension_checkpoint_root": _display(extension_root),
        "extension_namespace_absent_at_authorization": True,
        "replay_resume_epoch": SELECTED_EPOCH,
        "replay_target_epoch": REPLAY_EPOCHS[-1],
        "replay_epochs": list(REPLAY_EPOCHS),
        "novel_epochs": list(NOVEL_EPOCHS),
        "final_selection_epochs": list(range(1, TARGET_OPTIMIZER_UPDATE + 1)),
        "target_optimizer_update": TARGET_OPTIMIZER_UPDATE,
        "microsteps_per_optimizer_update": MICROSTEPS_PER_UPDATE,
        "expected_branch_epochs": list(range(2, 9)),
        "greedy_audit_authorized": False,
        "stage_b_authorized": False,
        "trainer": {
            "module": "semantic_3d_chat.training.train_adapter",
            "python_executable": ".venv-gemma4/bin/python",
            "working_directory": "temporary_detached_training_source_worktree",
            "environment": {"PYTHONPATH": "src"},
            "requires_detached_training_source_worktree": True,
            "requires_exact_provenance_preflight": True,
            "training_source_commit": evidence["training_source_provenance"]["head_commit"],
            "replay_resume": _display(checkpoint),
            "replay_epochs": REPLAY_EPOCHS[-1],
            "replay_argv": [
                "-m",
                "semantic_3d_chat.training.train_adapter",
                "--config",
                str(CONFIG_PATH),
                "--resume",
                _display(checkpoint),
                "--output-namespace",
                EXTENSION_NAMESPACE,
                "--epochs",
                str(REPLAY_EPOCHS[-1]),
            ],
            "novel_resume": f"data_gemma4/checkpoints/{EXTENSION_NAMESPACE}/epoch_004",
            "novel_target_epochs": TARGET_OPTIMIZER_UPDATE,
            "novel_argv": [
                "-m",
                "semantic_3d_chat.training.train_adapter",
                "--config",
                str(CONFIG_PATH),
                "--resume",
                f"data_gemma4/checkpoints/{EXTENSION_NAMESPACE}/epoch_004",
                "--output-namespace",
                EXTENSION_NAMESPACE,
                "--epochs",
                str(TARGET_OPTIMIZER_UPDATE),
            ],
            "linked_ignored_inputs": [
                "data/qa",
                "data_gemma4/checkpoints",
                "data_gemma4/features",
                "data_gemma4/maps",
            ],
        },
    }


def prepare_extension_launch(
    config_path: str | Path = CONFIG_PATH,
    screen_path: str | Path = SCREEN_PATH,
    *,
    current_provenance: Mapping[str, Any] | None = None,
    transition: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    evidence = _load_exact_screen(config_path, screen_path)
    controller_source, observed_transition = _validate_control_plane_transition(
        evidence["training_source_provenance"],
        current_provenance=current_provenance,
        transition=transition,
    )
    extension_root = _extension_root(evidence["config"])
    if extension_root.exists() or extension_root.is_symlink():
        _fail(f"refusing to overwrite existing extension namespace: {extension_root}")
    return _manifest_body(evidence, controller_source, observed_transition)


def _validate_manifest(
    manifest_path: str | Path,
    *,
    current_provenance: Mapping[str, Any] | None = None,
    transition: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _equal(_resolve(manifest_path), _resolve(MANIFEST_PATH), "extension manifest path")
    manifest = _load_json(manifest_path, "V24 extension manifest")
    evidence = _load_exact_screen(CONFIG_PATH, SCREEN_PATH)
    controller_source, observed_transition = _validate_control_plane_transition(
        evidence["training_source_provenance"],
        current_provenance=current_provenance,
        transition=transition,
    )
    expected = _manifest_body(evidence, controller_source, observed_transition)
    _equal(manifest, expected, "exact extension manifest recomputation")
    _equal(
        manifest["controller_source_provenance"],
        controller_source,
        "manifest/current controller provenance",
    )
    return manifest, evidence


def _expected_lora_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    collection = shared._install_shape_only(config)
    settings = lora_banks_settings(config)
    optimizer = lora_banks_optimizer_settings(config, settings)
    if optimizer is None:
        _fail("V24 extension requires its exact LoRA optimizer")
    return lora_banks_checkpoint_contract(settings, optimizer, collection.parameter_counts)


def _require_nonreset_a_moments(manifest: Mapping[str, Any], *, epoch: int) -> None:
    states = _sequence(manifest.get("parameter_states"), f"epoch {epoch} optimizer states")
    for state_value in states:
        state = _mapping(state_value, f"epoch {epoch} optimizer state")
        if state["role"] == "A" and (
            state["exp_avg_nonzero"] == 0 or state["exp_avg_sq_nonzero"] == 0
        ):
            _fail(f"extension epoch {epoch} reset a LoRA-A optimizer moment")


def _branch_epoch_record(
    config: Mapping[str, Any], source: Mapping[str, Any], epoch: int
) -> dict[str, Any]:
    checkpoint = (
        artifact_root(dict(config), "checkpoints") / EXTENSION_NAMESPACE / f"epoch_{epoch:03d}"
    )
    metadata_path = _regular_file(checkpoint / "metadata.json", f"extension epoch {epoch}")
    metadata = _load_json(metadata_path, f"extension epoch {epoch} metadata")
    for field, expected in {
        "epoch": epoch,
        "global_step": epoch * MICROSTEPS_PER_UPDATE,
        "optimizer_step": epoch,
        "output_namespace": EXTENSION_NAMESPACE,
        "config_hash": config_hash(dict(config)),
        "source_provenance": dict(source),
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": False,
        "train_lora_with_frozen_scene_residual_stack": True,
        "frozen_scene_state_sha256": shared.EXPECTED_FROZEN_HASHES["scene"],
        "frozen_global_scene_residual_state_sha256": shared.EXPECTED_FROZEN_HASHES["global"],
        "frozen_signed_x_scene_residual_state_sha256": shared.EXPECTED_FROZEN_HASHES["signed_x"],
        "global_scene_residual_state_sha256": shared.EXPECTED_FROZEN_HASHES["global"],
        "signed_x_scene_residual_state_sha256": shared.EXPECTED_FROZEN_HASHES["signed_x"],
        "lora": _expected_lora_contract(config),
    }.items():
        _equal(metadata.get(field), expected, f"extension epoch {epoch} {field}")
    frozen_banks = {name: shared.EXPECTED_FROZEN_HASHES[name] for name in shared.FROZEN_BANKS}
    _equal(
        metadata.get("frozen_lora_bank_state_sha256"),
        frozen_banks,
        f"extension epoch {epoch} frozen banks",
    )
    bank_hashes = dict(
        _mapping(metadata.get("lora_bank_state_sha256"), f"extension epoch {epoch} banks")
    )
    _equal(
        set(bank_hashes),
        {*shared.FROZEN_BANKS, shared.NEW_BANK},
        f"extension epoch {epoch} bank keys",
    )
    try:
        shared._require_frozen_bank_pins(
            bank_hashes, field=f"extension epoch {epoch} metadata frozen bank"
        )
    except shared.V24ControlViolation as error:
        _fail(str(error))
    adapter = _regular_file(checkpoint / "adapter.safetensors", f"extension epoch {epoch}")
    optimizer = _regular_file(checkpoint / "optimizer.pt", f"extension epoch {epoch}")
    try:
        payload = shared._adapter_payload(adapter)
        shared._require_frozen_bank_pins(
            payload["lora_bank_state_sha256"],
            field=f"extension epoch {epoch} recomputed frozen bank",
        )
        shared._require_new_bank_tensor_contract(
            payload["new_bank_state"], field=f"extension epoch {epoch} new bank"
        )
        optimizer_manifest = shared._optimizer_manifest(optimizer, expected_step=epoch)
        detail = shared._opaque_unit_margin_detail(metadata)
    except shared.V24ControlViolation as error:
        _fail(str(error))
    for field, expected in {
        "scene_state_sha256": shared.EXPECTED_FROZEN_HASHES["scene"],
        "global_scene_residual_state_sha256": shared.EXPECTED_FROZEN_HASHES["global"],
        "signed_x_scene_residual_state_sha256": shared.EXPECTED_FROZEN_HASHES["signed_x"],
        "lora_bank_state_sha256": bank_hashes,
    }.items():
        _equal(payload[field], expected, f"extension epoch {epoch} recomputed {field}")
    _require_nonreset_a_moments(optimizer_manifest, epoch=epoch)
    history = _validate_history_contract(metadata, epoch=epoch)
    return {
        "epoch": epoch,
        "optimizer_step": epoch,
        "cumulative_microsteps": epoch * MICROSTEPS_PER_UPDATE,
        "checkpoint": _display(checkpoint),
        "metadata_path": _display(metadata_path),
        "adapter_sha256": file_sha256(adapter),
        "metadata_sha256": file_sha256(metadata_path),
        "optimizer_sha256": file_sha256(optimizer),
        "new_bank_state_sha256": bank_hashes[shared.NEW_BANK],
        "recomputed_payload_hashes": {
            key: value for key, value in payload.items() if key != "new_bank_state"
        },
        "optimizer_manifest": optimizer_manifest,
        "color": shared._pair_metrics(metadata, "pair_000001"),
        "mirror": shared._pair_metrics(metadata, "pair_000003"),
        "opaque_unit_margin_detail": detail,
        "history": history,
        "initialization_provenance": copy.deepcopy(metadata.get("initialization_provenance")),
        "raw_metadata": metadata,
    }


def _normalized_replay_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(metadata))
    normalized.pop("output_namespace", None)
    return normalized


def _require_replay_semantic_equivalence(
    replay: Mapping[str, Any], primary: Mapping[str, Any], *, epoch: int
) -> dict[str, Any]:
    """Bind optimizer tensors, not non-canonical ``torch.save`` container bytes."""

    for key in (
        "adapter_sha256",
        "new_bank_state_sha256",
        "recomputed_payload_hashes",
        "optimizer_manifest",
        "color",
        "mirror",
        "opaque_unit_margin_detail",
    ):
        _equal(replay[key], primary[key], f"replay epoch {epoch} primary {key}")
    replay_raw = str(replay["optimizer_sha256"])
    primary_raw = str(primary["optimizer_sha256"])
    bytes_equal = replay_raw == primary_raw
    optimizer_manifest = _mapping(replay["optimizer_manifest"], "replay optimizer manifest")
    return {
        "replay_optimizer_sha256": replay_raw,
        "primary_optimizer_sha256": primary_raw,
        "container_bytes_equal": bytes_equal,
        "container_byte_difference_present": not bytes_equal,
        "container_byte_difference_classification": (
            "none" if bytes_equal else "expected_non_semantic_torch_save_reserialization"
        ),
        "decoded_optimizer_manifest_exact": True,
        "decoded_all_state_tensors_sha256": optimizer_manifest["all_state_tensors_sha256"],
    }


def _public_branch_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in {"raw_metadata", "history", "initialization_provenance"}
    }


def _build_replay_report(
    manifest_path: str | Path,
    *,
    current_provenance: Mapping[str, Any] | None = None,
    transition: Mapping[str, str] | None = None,
    expected_layout_epochs: set[int] | None = None,
) -> dict[str, Any]:
    manifest, evidence = _validate_manifest(
        manifest_path,
        current_provenance=current_provenance,
        transition=transition,
    )
    if expected_layout_epochs is None:
        _require_replay_or_final_layout(evidence["config"], field="V24 replay namespace")
    else:
        _require_extension_layout(
            evidence["config"], expected_layout_epochs, field="V24 replay namespace"
        )
    selected_metadata = _load_json(
        Path(manifest["selected_checkpoint"]) / "metadata.json", "selected epoch-1 metadata"
    )
    selected_history = _validate_history_contract(selected_metadata, epoch=SELECTED_EPOCH)
    _equal(
        shared._canonical_sha256(selected_history),
        manifest["selected_history_sha256"],
        "selected history hash",
    )
    initialization = selected_metadata.get("initialization_provenance")
    rows: list[dict[str, Any]] = []
    previous_history = selected_history
    for epoch in REPLAY_EPOCHS:
        row = _branch_epoch_record(
            evidence["config"], evidence["training_source_provenance"], epoch
        )
        _equal(
            row["history"][: len(previous_history)],
            previous_history,
            f"replay epoch {epoch} history prefix",
        )
        _equal(
            row["initialization_provenance"],
            initialization,
            f"replay epoch {epoch} initialization provenance",
        )
        primary = evidence["epochs"][epoch - 1]
        optimizer_audit = _require_replay_semantic_equivalence(row, primary, epoch=epoch)
        primary_metadata = _load_json(
            Path(primary["metadata_path"]), f"primary epoch {epoch} metadata"
        )
        _equal(
            _normalized_replay_metadata(row["raw_metadata"]),
            _normalized_replay_metadata(primary_metadata),
            f"replay epoch {epoch} normalized metadata",
        )
        previous_history = row["history"]
        public = _public_branch_row(row)
        public["optimizer_container_audit"] = optimizer_audit
        rows.append(public)
    return {
        "schema_version": 1,
        "audit_type": "v24_shared_query_extension_replay_verifier",
        "match": True,
        "stage_b_authorized": True,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": file_sha256(_regular_file(manifest_path, "extension manifest")),
        "screen_sha256": manifest["screen_sha256"],
        "training_source_provenance": manifest["training_source_provenance"],
        "controller_source_provenance": manifest["controller_source_provenance"],
        "selected_epoch": SELECTED_EPOCH,
        "replay_epochs": list(REPLAY_EPOCHS),
        "normalized_metadata_difference_allowlist": ["output_namespace"],
        "adapter_and_decoded_optimizer_exact_replay": True,
        "optimizer_container_byte_identity_required": False,
        "history_prefix_exact": True,
        "initialization_provenance_exact": True,
        "epochs": rows,
    }


def verify_replay(
    manifest_path: str | Path = MANIFEST_PATH,
    *,
    current_provenance: Mapping[str, Any] | None = None,
    transition: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return _build_replay_report(
        manifest_path,
        current_provenance=current_provenance,
        transition=transition,
    )


def _validate_replay_report(
    replay_path: str | Path,
    *,
    current_provenance: Mapping[str, Any] | None = None,
    transition: Mapping[str, str] | None = None,
    expected_layout_epochs: set[int] | None = None,
) -> dict[str, Any]:
    _equal(_resolve(replay_path), _resolve(REPLAY_REPORT_PATH), "replay report path")
    observed = _load_json(replay_path, "V24 replay report")
    expected = _build_replay_report(
        MANIFEST_PATH,
        current_provenance=current_provenance,
        transition=transition,
        expected_layout_epochs=expected_layout_epochs,
    )
    _equal(observed, expected, "exact replay report recomputation")
    return observed


def authorize_stage_b(
    replay_path: str | Path = REPLAY_REPORT_PATH,
    *,
    current_provenance: Mapping[str, Any] | None = None,
    transition: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    report = _validate_replay_report(
        replay_path,
        current_provenance=current_provenance,
        transition=transition,
    )
    return {
        "authorized": True,
        "stage_b_authorized": True,
        "resume_checkpoint": f"data_gemma4/checkpoints/{EXTENSION_NAMESPACE}/epoch_004",
        "target_optimizer_update": TARGET_OPTIMIZER_UPDATE,
        "training_source_commit": report["training_source_provenance"]["head_commit"],
    }


def _candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "epoch",
        "optimizer_step",
        "cumulative_microsteps",
        "metadata_path",
        "metadata_sha256",
        "adapter_sha256",
        "optimizer_sha256",
        "new_bank_state_sha256",
        "recomputed_payload_hashes",
        "optimizer_manifest",
        "color",
        "mirror",
        "opaque_unit_margin_detail",
    )
    candidate = {key: copy.deepcopy(row[key]) for key in keys}
    candidate["color_eligible"] = shared._color_eligible(candidate["color"])
    candidate["continuation_gate_passed"] = shared._mirror_continuation(candidate["mirror"])
    candidate["full_teacher_gate_passed"] = shared._full_pair(
        candidate["color"]
    ) and shared._full_pair(candidate["mirror"])
    return candidate


def _select_final_candidates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank the complete unique update-one-through-eight trajectory."""

    _equal(
        [int(row["epoch"]) for row in rows],
        list(range(1, TARGET_OPTIMIZER_UPDATE + 1)),
        "final selection epoch sequence",
    )
    candidates = [_candidate(row) for row in rows]
    eligible = [row for row in candidates if row["color_eligible"]]
    ranking = sorted((copy.deepcopy(row) for row in eligible), key=_ranking_key, reverse=True)
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    selected = ranking[0] if ranking else None
    full_teacher = bool(selected is not None and selected["full_teacher_gate_passed"])
    return {
        "candidates": candidates,
        "ranking": ranking,
        "selected": selected,
        "full_teacher_gate_passed": full_teacher,
        "greedy_audit_authorized": full_teacher,
    }


def _require_unique_trajectory(candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    seen_bank: dict[str, Mapping[str, Any]] = {}
    seen_optimizer: dict[str, Mapping[str, Any]] = {}
    plateaus: list[int] = []
    for row in candidates:
        state = str(row["new_bank_state_sha256"])
        prior = seen_bank.get(state)
        if prior is not None:
            allowed = (
                int(prior["epoch"]) == int(row["epoch"]) - 1
                and prior["full_teacher_gate_passed"] is True
                and row["full_teacher_gate_passed"] is True
                and prior["color"] == row["color"]
                and prior["mirror"] == row["mirror"]
            )
            if not allowed:
                _fail("extension trajectory repeats or rolls back a trainable bank state")
            plateaus.append(int(row["epoch"]))
        seen_bank[state] = row
        optimizer_state = str(row["optimizer_manifest"]["all_state_tensors_sha256"])
        prior_optimizer = seen_optimizer.get(optimizer_state)
        if prior_optimizer is not None:
            allowed = (
                int(prior_optimizer["epoch"]) == int(row["epoch"]) - 1
                and prior_optimizer["full_teacher_gate_passed"] is True
                and row["full_teacher_gate_passed"] is True
                and prior_optimizer["color"] == row["color"]
                and prior_optimizer["mirror"] == row["mirror"]
            )
            if not allowed:
                _fail("extension trajectory repeats or rolls back an optimizer state")
        seen_optimizer[optimizer_state] = row
    return plateaus


def select_final_extension(
    manifest_path: str | Path = MANIFEST_PATH,
    replay_path: str | Path = REPLAY_REPORT_PATH,
    *,
    current_provenance: Mapping[str, Any] | None = None,
    transition: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest, evidence = _validate_manifest(
        manifest_path,
        current_provenance=current_provenance,
        transition=transition,
    )
    final_layout = set(range(2, TARGET_OPTIMIZER_UPDATE + 1))
    _require_extension_layout(
        evidence["config"], final_layout, field="V24 final extension namespace"
    )
    replay = _validate_replay_report(
        replay_path,
        current_provenance=current_provenance,
        transition=transition,
        expected_layout_epochs=final_layout,
    )
    selected_metadata = _load_json(
        Path(manifest["selected_checkpoint"]) / "metadata.json", "selected metadata"
    )
    initialization = selected_metadata.get("initialization_provenance")
    previous_history = _validate_history_contract(selected_metadata, epoch=SELECTED_EPOCH)
    branch_rows: list[dict[str, Any]] = []
    for epoch in range(2, TARGET_OPTIMIZER_UPDATE + 1):
        row = _branch_epoch_record(
            evidence["config"], evidence["training_source_provenance"], epoch
        )
        _equal(
            row["history"][: len(previous_history)],
            previous_history,
            f"extension epoch {epoch} cumulative history prefix",
        )
        _equal(
            row["initialization_provenance"],
            initialization,
            f"extension epoch {epoch} initialization provenance",
        )
        previous_history = row["history"]
        public = _public_branch_row(row)
        if epoch in REPLAY_EPOCHS:
            replay_row = replay["epochs"][epoch - REPLAY_EPOCHS[0]]
            for key, value in public.items():
                _equal(value, replay_row[key], f"final/replay epoch {epoch} {key}")
        branch_rows.append(public)
    trajectory = [_candidate(evidence["selected"]), *[_candidate(row) for row in branch_rows]]
    _equal([row["epoch"] for row in trajectory], list(range(1, 9)), "final trajectory")
    plateaus = _require_unique_trajectory(trajectory)
    final_selection = _select_final_candidates(trajectory)
    selected = final_selection["selected"]
    full_teacher = bool(final_selection["full_teacher_gate_passed"])
    first_full_teacher_epoch = next(
        (row["epoch"] for row in trajectory if row["full_teacher_gate_passed"]), None
    )
    if first_full_teacher_epoch is not None and first_full_teacher_epoch not in NOVEL_EPOCHS:
        _fail("full-teacher gate unexpectedly first passed in the sealed/replay trajectory")
    selected_checkpoint = None if selected is None else str(Path(selected["metadata_path"]).parent)
    selected_hashes = (
        None
        if selected is None
        else {
            "adapter_sha256": selected["adapter_sha256"],
            "metadata_sha256": selected["metadata_sha256"],
            "optimizer_sha256": selected["optimizer_sha256"],
        }
    )
    return {
        "schema_version": 1,
        "audit_type": "v24_shared_query_extension_final_selector",
        "decision": (
            "full_teacher_gate_passed_greedy_audit_authorized"
            if full_teacher
            else "conditional_limit_reached_no_greedy_audit"
        ),
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "config_sha256": shared.EXPECTED_CONFIG_SHA256,
        "contract_sha256": shared.EXPECTED_CONTRACT_SHA256,
        "manifest_sha256": file_sha256(_regular_file(manifest_path, "extension manifest")),
        "replay_report_sha256": file_sha256(_regular_file(replay_path, "replay report")),
        "screen_sha256": manifest["screen_sha256"],
        "training_source_provenance": manifest["training_source_provenance"],
        "controller_source_provenance": manifest["controller_source_provenance"],
        "original_selected_epoch": SELECTED_EPOCH,
        "replay_epochs": list(REPLAY_EPOCHS),
        "novel_epochs": list(NOVEL_EPOCHS),
        "selection_scope": "complete_unique_trajectory_epochs_1_through_8",
        "target_optimizer_update": TARGET_OPTIMIZER_UPDATE,
        "conditional_limit_reached": True,
        "continuation_authorized": False,
        "full_teacher_gate_passed": full_teacher,
        "greedy_audit_authorized": full_teacher,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "selected_epoch": None if selected is None else selected["epoch"],
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_artifact_hashes": selected_hashes,
        "full_teacher_first_pass_epoch": first_full_teacher_epoch,
        "repeated_full_teacher_plateau_epochs": plateaus,
        "epochs": trajectory,
        "replay_candidates": trajectory[1:4],
        "novel_candidates": trajectory[4:],
        "ranking": final_selection["ranking"],
        "selection_policy": {
            **_selection_policy(),
            "selection_scope": "complete unique trajectory epochs 1 through 8",
            "new_full_teacher_evidence_scope": "novel epochs 5 through 8",
        },
    }


def _write(report: Mapping[str, Any], output: str | Path, expected: Path) -> Path:
    destination = _resolve(output)
    _equal(destination, _resolve(expected), "controller output path")
    _reject_symlink_chain(destination, "controller output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=CONFIG_PATH)
    prepare.add_argument("--screen", type=Path, default=SCREEN_PATH)
    prepare.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-launch")
    validate.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    replay = commands.add_parser("verify-replay")
    replay.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    replay.add_argument("--output", type=Path, required=True)
    stage_b = commands.add_parser("authorize-stage-b")
    stage_b.add_argument("--replay", type=Path, default=REPLAY_REPORT_PATH)
    final = commands.add_parser("select-final")
    final.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    final.add_argument("--replay", type=Path, default=REPLAY_REPORT_PATH)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_extension_launch(args.config, args.screen)
        destination = _write(result, args.output, MANIFEST_PATH)
        summary = {
            "authorized": True,
            "output": str(destination),
            "selected_epoch": SELECTED_EPOCH,
        }
    elif args.command == "validate-launch":
        result, _evidence = _validate_manifest(args.manifest)
        summary = {
            "authorized": result["authorized"],
            "training_source_commit": result["training_source_provenance"]["head_commit"],
        }
    elif args.command == "verify-replay":
        result = verify_replay(args.manifest)
        destination = _write(result, args.output, REPLAY_REPORT_PATH)
        summary = {"stage_b_authorized": True, "output": str(destination)}
    elif args.command == "authorize-stage-b":
        summary = authorize_stage_b(args.replay)
    else:
        result = select_final_extension(args.manifest, args.replay)
        destination = _write(result, args.output, FINAL_REPORT_PATH)
        summary = {
            "decision": result["decision"],
            "greedy_audit_authorized": result["greedy_audit_authorized"],
            "output": str(destination),
            "selected_epoch": result["selected_epoch"],
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - local checkpoint command
    raise SystemExit(main())


__all__ = [
    "EXTENSION_NAMESPACE",
    "V24ExtensionViolation",
    "authorize_stage_b",
    "prepare_extension_launch",
    "select_final_extension",
    "verify_replay",
]

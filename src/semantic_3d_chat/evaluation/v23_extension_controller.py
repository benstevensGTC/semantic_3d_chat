"""Fail-closed V23 replay gate and bounded update-eight extension selector.

The four-update screen was produced by a clean, immutable training source.  The
control plane may be newer, but it may differ from that source only in an exact
allowlist of V23 controller, Makefile, and test files.  Training therefore runs
from a detached worktree at the original source commit.  Updates three and four
are replayed first and must exactly reproduce the primary branch before novel
updates five through eight are authorized.

This module is report-only: it never loads Gemma, scene maps, questions, or
oracle artifacts.
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
from semantic_3d_chat.evaluation import v23_shared_kv_controller as shared
from semantic_3d_chat.language.lora import (
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
)
from semantic_3d_chat.training.train_adapter import file_sha256

CONFIG_PATH = shared.CONFIG_PATH
SCREEN_PATH = Path("reports/gemma4/metrics/v23_epoch_screen.json")
UPDATE1_PATH = Path("reports/gemma4/metrics/v23_update1_match.json")
MANIFEST_PATH = Path("reports/gemma4/metrics/v23_extension_launch.json")
REPLAY_REPORT_PATH = Path("reports/gemma4/metrics/v23_extension_replay.json")
FINAL_REPORT_PATH = Path("reports/gemma4/metrics/v23_extension_final.json")
PRIMARY_NAMESPACE = shared.PRIMARY_NAMESPACE
EXTENSION_NAMESPACE = shared.EXTENSION_NAMESPACE
SELECTED_EPOCH = 2
REPLAY_EPOCHS = (3, 4)
NOVEL_EPOCHS = (5, 6, 7, 8)
TARGET_OPTIMIZER_UPDATE = 8
MICROSTEPS_PER_UPDATE = 12
EXPECTED_SCREEN_SHA256 = "edc57280257a5610a81ea9e2e54f2adaa77f8d2bebf41002aaa7c2ce7383b4d4"
EXPECTED_UPDATE1_SHA256 = "149b7f14ada0a2a9b0a9101355b3419a3820f863755e259809a1f20daad160a5"
EXPECTED_TRAINING_SOURCE_PROVENANCE = {
    "schema_version": 1,
    "scope": "repository_excluding_generated_artifacts_v1",
    "available": True,
    "is_clean": True,
    "head_commit": "2a8cd075f7961092d756efaa1e619f62a14c1262",
    "head_tree": "2018b80f79736809637bde1d20ade56cc18e40d5",
    "tracked_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
EXPECTED_SELECTED_ARTIFACTS = {
    "adapter_sha256": "dba2511db49fa46af905b293fc999642286f8533fa1d4cca2c872ffda2980ea8",
    "metadata_sha256": "1c0436549e832c2ac9723e2556ad8bf09862020c6cda47db8358b2232b391ba0",
    "optimizer_sha256": "08c3618d765346a018e78e4a608361d29f1d0c88cf01d5c2af26e0b20c9a3daa",
}
EXPECTED_SELECTED_BANK_SHA256 = "91a9eea577cab5a37e840cdf4007722a398415846af91280713bcb2cda0f045c"
EXPECTED_SELECTED_OPTIMIZER_STATE_SHA256 = (
    "87b09f0bd1951531ced90b2ea6a5df392385a63185a90a974c940b2b1ec867bb"
)

# This exact source transition is deliberately narrower than a generic
# "controller-only" rule.  Any config, trainer, model, or data-pipeline change
# requires a new screen rather than inheriting V23 evidence.
EXPECTED_CONTROL_PLANE_TRANSITION = {
    "Makefile": "M",
    "src/semantic_3d_chat/evaluation/v23_extension_controller.py": "A",
    "src/semantic_3d_chat/evaluation/v23_shared_kv_controller.py": "M",
    "tests/test_v23_extension_controller.py": "A",
    "tests/test_v23_shared_kv_controller.py": "M",
}


class V23ExtensionViolation(ValueError):
    """An extension authorization, replay, or final artifact violated its contract."""


def _fail(message: str) -> None:
    raise V23ExtensionViolation(message)


def _equal(observed: Any, expected: Any, field: str) -> None:
    try:
        shared._equal(observed, expected, field)
    except shared.V23ControlViolation as error:
        _fail(str(error))


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    try:
        return shared._mapping(value, field)
    except shared.V23ControlViolation as error:
        _fail(str(error))


def _sequence(value: Any, field: str) -> Sequence[Any]:
    try:
        return shared._sequence(value, field)
    except shared.V23ControlViolation as error:
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
    except shared.V23ControlViolation as error:
        _fail(str(error))


def _load_json(path: str | Path, field: str) -> dict[str, Any]:
    try:
        return shared._load_json(Path(path), field)
    except shared.V23ControlViolation as error:
        _fail(str(error))


def _clean_provenance(value: Any, field: str) -> dict[str, Any]:
    try:
        return shared._clean_provenance(value, field)
    except shared.V23ControlViolation as error:
        _fail(str(error))


def _checkpoint_hashes(checkpoint: Path, field: str) -> dict[str, str]:
    return {
        "adapter_sha256": file_sha256(_regular_file(checkpoint / "adapter.safetensors", field)),
        "metadata_sha256": file_sha256(_regular_file(checkpoint / "metadata.json", field)),
        "optimizer_sha256": file_sha256(_regular_file(checkpoint / "optimizer.pt", field)),
    }


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


def _require_extension_layout(
    config: Mapping[str, Any],
    expected_epochs: set[int],
    *,
    field: str,
) -> Path:
    root, observed_epochs = _inspect_extension_layout(config, field=field)
    _equal(observed_epochs, expected_epochs, f"{field} epoch directory set")
    return root


def _inspect_extension_layout(
    config: Mapping[str, Any],
    *,
    field: str,
) -> tuple[Path, set[int]]:
    root = _extension_root(config)
    if not root.is_dir() or root.is_symlink():
        _fail(f"{field} is not a regular extension directory: {root}")
    observed_epochs: set[int] = set()
    allowed_names = {"best"}
    for entry in root.iterdir():
        if entry.name.startswith("epoch_"):
            suffix = entry.name.removeprefix("epoch_")
            if len(suffix) != 3 or not suffix.isdigit() or entry.is_symlink() or not entry.is_dir():
                _fail(f"{field} contains malformed or symlinked epoch entry: {entry}")
            observed_epochs.add(int(suffix))
            continue
        if entry.name not in allowed_names:
            _fail(f"{field} contains an unexpected entry: {entry.name}")
        if entry.is_symlink() or not entry.is_dir():
            _fail(f"{field} auxiliary entry is not a regular directory: {entry}")
    for epoch in observed_epochs:
        _reject_symlink_chain(root / f"epoch_{epoch:03d}", f"{field} epoch {epoch}")
    return root, observed_epochs


def _require_replay_or_final_layout(config: Mapping[str, Any], *, field: str) -> Path:
    """Accept only a complete replay or a complete bounded final branch."""

    root, observed = _inspect_extension_layout(config, field=field)
    allowed = (set(REPLAY_EPOCHS), set(range(3, TARGET_OPTIMIZER_UPDATE + 1)))
    if not any(observed == candidate for candidate in allowed):
        _fail(
            f"{field} epoch directory set must be exactly {sorted(allowed[0])} "
            f"or {sorted(allowed[1])}; observed={sorted(observed)}"
        )
    return root


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
    }


def _ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    mirror = _mapping(row.get("mirror"), "candidate mirror")
    return (
        mirror["full_vocab_units"],
        mirror["full_vocab_sides"],
        mirror["mean_full_vocab_margin"],
        mirror["minimum_full_vocab_margin"],
        mirror["mean_candidate_margin"],
        mirror["minimum_candidate_margin"],
        -int(row["epoch"]),
    )


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
        "optimizer_manifest",
        "recomputed_payload_hashes",
        "color",
        "mirror",
        "source_provenance",
    }
    _equal(set(update1), expected_keys, "update-1 report keys")
    for field, expected in {
        "schema_version": 1,
        "audit_type": "v23_shared_kv_update1_verifier",
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
    }.items():
        _equal(update1.get(field), expected, f"update-1 {field}")
    return _clean_provenance(update1.get("source_provenance"), "training source provenance")


def _load_exact_screen(
    config_path: str | Path,
    screen_path: str | Path,
) -> dict[str, Any]:
    """Recompute the complete four-update screen from bound checkpoint files."""

    _equal(_resolve(config_path), _resolve(CONFIG_PATH), "V23 config path")
    _equal(_resolve(screen_path), _resolve(SCREEN_PATH), "V23 screen path")
    config = load_config(config_path)
    try:
        shared._validate_contract(config)
    except shared.V23ControlViolation as error:
        _fail(str(error))
    screen_file = _regular_file(screen_path, "V23 screen report")
    _equal(file_sha256(screen_file), EXPECTED_SCREEN_SHA256, "externally pinned screen hash")
    screen = _load_json(screen_file, "V23 screen report")
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
        "question_dependent_scene_processing",
        "config_sha256",
        "contract_sha256",
        "update1_report_sha256",
        "epochs",
        "selection_policy",
    }
    _equal(set(screen), expected_screen_keys, "screen report keys")
    update1_file = _regular_file(UPDATE1_PATH, "V23 update-1 report")
    _equal(file_sha256(update1_file), EXPECTED_UPDATE1_SHA256, "externally pinned update-1 hash")
    _equal(file_sha256(update1_file), screen.get("update1_report_sha256"), "screen/update-1")
    update1 = _load_json(update1_file, "V23 update-1 report")
    training_source = _validate_update1(update1)
    _equal(
        training_source,
        EXPECTED_TRAINING_SOURCE_PROVENANCE,
        "externally pinned training source provenance",
    )
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, 5):
        metadata_path = (
            PROJECT_ROOT
            / "data_gemma4/checkpoints"
            / PRIMARY_NAMESPACE
            / f"epoch_{epoch:03d}/metadata.json"
        )
        try:
            epoch_rows.append(shared._epoch_record(config, epoch, metadata_path, training_source))
        except shared.V23ControlViolation as error:
            _fail(str(error))
    epoch1 = epoch_rows[0]
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
    ):
        _equal(update1.get(key), epoch1[key], f"update-1 epoch-1 {key}")
    eligible = [row for row in epoch_rows if shared._color_eligible(row["color"])]
    selected = None if not eligible else max(eligible, key=_ranking_key)
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
        "audit_type": "v23_shared_kv_epoch_selector",
        "decision": "screen_passed_extension_authorized_no_greedy_audit",
        "selected_epoch": SELECTED_EPOCH,
        "selected_checkpoint": (
            f"data_gemma4/checkpoints/{PRIMARY_NAMESPACE}/epoch_{SELECTED_EPOCH:03d}"
        ),
        "continuation_authorized": True,
        "full_teacher_gate_passed": False,
        "greedy_audit_authorized": False,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "report_only": True,
        "question_dependent_scene_processing": False,
        "config_sha256": shared.EXPECTED_CONFIG_SHA256,
        "contract_sha256": shared.EXPECTED_CONTRACT_SHA256,
        "update1_report_sha256": file_sha256(update1_file),
        "epochs": epoch_rows,
        "selection_policy": _selection_policy(),
    }
    _equal(selected["epoch"] if selected else None, SELECTED_EPOCH, "recomputed selected epoch")
    _equal(continuation, True, "recomputed continuation gate")
    _equal(full_teacher, False, "recomputed full-teacher gate")
    _equal(screen, expected_screen, "exact recomputed V23 screen")
    selected = epoch_rows[SELECTED_EPOCH - 1]
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
        "epochs": epoch_rows,
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


def _selected_checkpoint(evidence: Mapping[str, Any]) -> Path:
    checkpoint = (
        PROJECT_ROOT
        / "data_gemma4/checkpoints"
        / PRIMARY_NAMESPACE
        / f"epoch_{SELECTED_EPOCH:03d}"
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


def _manifest_body(
    evidence: Mapping[str, Any],
    controller_source: Mapping[str, Any],
    transition: Mapping[str, str],
) -> dict[str, Any]:
    selected = _mapping(evidence.get("selected"), "selected epoch")
    checkpoint = _selected_checkpoint(evidence)
    extension_root = _extension_root(evidence["config"])
    return {
        "schema_version": 1,
        "audit_type": "v23_shared_kv_extension_launch",
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
        "original_output_namespace": PRIMARY_NAMESPACE,
        "extension_output_namespace": EXTENSION_NAMESPACE,
        "extension_checkpoint_root": _display(extension_root),
        "extension_namespace_absent_at_authorization": True,
        "replay_resume_epoch": SELECTED_EPOCH,
        "replay_target_epoch": 4,
        "replay_epochs": list(REPLAY_EPOCHS),
        "novel_epochs": list(NOVEL_EPOCHS),
        "target_optimizer_update": TARGET_OPTIMIZER_UPDATE,
        "microsteps_per_optimizer_update": MICROSTEPS_PER_UPDATE,
        "expected_branch_epochs": list(range(3, 9)),
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
            "replay_epochs": 4,
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
                "4",
            ],
            "novel_resume": (
                f"data_gemma4/checkpoints/{EXTENSION_NAMESPACE}/epoch_004"
            ),
            "novel_target_epochs": 8,
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
                "8",
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
    manifest = _load_json(manifest_path, "V23 extension manifest")
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
        _fail("V23 extension requires its exact LoRA optimizer")
    return lora_banks_checkpoint_contract(settings, optimizer, collection.parameter_counts)


def _branch_epoch_record(
    config: Mapping[str, Any],
    training_source: Mapping[str, Any],
    epoch: int,
) -> dict[str, Any]:
    checkpoint = (
        artifact_root(dict(config), "checkpoints")
        / EXTENSION_NAMESPACE
        / f"epoch_{epoch:03d}"
    )
    metadata_path = _regular_file(checkpoint / "metadata.json", f"extension epoch {epoch}")
    metadata = _load_json(metadata_path, f"extension epoch {epoch} metadata")
    for field, expected in {
        "epoch": epoch,
        "global_step": epoch * MICROSTEPS_PER_UPDATE,
        "optimizer_step": epoch,
        "output_namespace": EXTENSION_NAMESPACE,
        "config_hash": config_hash(dict(config)),
        "source_provenance": dict(training_source),
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
    frozen_banks = {
        "inherited_v12": shared.EXPECTED_FROZEN_HASHES["inherited_v12"],
        "extension_v13": shared.EXPECTED_FROZEN_HASHES["extension_v13"],
    }
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
        {"inherited_v12", "extension_v13", shared.NEW_BANK},
        f"extension epoch {epoch} bank keys",
    )
    try:
        shared._require_frozen_bank_pins(
            bank_hashes,
            field=f"extension epoch {epoch} metadata frozen bank",
        )
    except shared.V23ControlViolation as error:
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
            payload["new_bank_state"],
            field=f"extension epoch {epoch} new bank",
        )
        optimizer_manifest = shared._optimizer_manifest(optimizer, expected_step=epoch)
    except shared.V23ControlViolation as error:
        _fail(str(error))
    _equal(
        payload["scene_state_sha256"],
        shared.EXPECTED_FROZEN_HASHES["scene"],
        f"extension epoch {epoch} recomputed scene",
    )
    _equal(
        payload["global_scene_residual_state_sha256"],
        shared.EXPECTED_FROZEN_HASHES["global"],
        f"extension epoch {epoch} recomputed global residual",
    )
    _equal(
        payload["signed_x_scene_residual_state_sha256"],
        shared.EXPECTED_FROZEN_HASHES["signed_x"],
        f"extension epoch {epoch} recomputed signed-X residual",
    )
    _equal(
        payload["lora_bank_state_sha256"],
        bank_hashes,
        f"extension epoch {epoch} recomputed banks",
    )
    _require_nonreset_a_moments(optimizer_manifest, epoch=epoch)
    history = _validate_history_contract(metadata, epoch=epoch)
    initialization = dict(
        _mapping(metadata.get("initialization_provenance"), "initialization provenance")
    )
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
        "history": history,
        "initialization_provenance": initialization,
        "raw_metadata": metadata,
    }


def _require_nonreset_a_moments(manifest: Mapping[str, Any], *, epoch: int) -> None:
    states = _sequence(manifest.get("parameter_states"), f"epoch {epoch} optimizer states")
    for state_value in states:
        state = _mapping(state_value, f"epoch {epoch} optimizer state")
        if state["role"] == "A" and (
            state["exp_avg_nonzero"] == 0 or state["exp_avg_sq_nonzero"] == 0
        ):
            _fail(f"extension epoch {epoch} reset a LoRA-A optimizer moment")


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


def _normalized_replay_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(metadata))
    normalized.pop("output_namespace", None)
    return normalized


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
        _require_replay_or_final_layout(
            evidence["config"],
            field="V23 replay namespace",
        )
    else:
        _require_extension_layout(
            evidence["config"],
            expected_layout_epochs,
            field="V23 replay namespace",
        )
    selected_metadata = _load_json(
        Path(manifest["selected_checkpoint"]) / "metadata.json",
        "selected epoch-2 metadata",
    )
    selected_history = _validate_history_contract(selected_metadata, epoch=SELECTED_EPOCH)
    initialization = selected_metadata.get("initialization_provenance")
    rows: list[dict[str, Any]] = []
    previous_history = selected_history
    for epoch in REPLAY_EPOCHS:
        row = _branch_epoch_record(evidence["config"], evidence["training_source_provenance"], epoch)
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
        primary_row = evidence["epochs"][epoch - 1]
        for key in (
            "adapter_sha256",
            "optimizer_sha256",
            "new_bank_state_sha256",
            "recomputed_payload_hashes",
            "optimizer_manifest",
            "color",
            "mirror",
        ):
            _equal(row[key], primary_row[key], f"replay epoch {epoch} primary {key}")
        primary_metadata = _load_json(
            Path(primary_row["metadata_path"]),
            f"primary epoch {epoch} metadata",
        )
        _equal(
            _normalized_replay_metadata(row["raw_metadata"]),
            _normalized_replay_metadata(primary_metadata),
            f"replay epoch {epoch} normalized metadata",
        )
        previous_history = row["history"]
        row.pop("raw_metadata")
        row.pop("history")
        row.pop("initialization_provenance")
        rows.append(row)
    return {
        "schema_version": 1,
        "audit_type": "v23_shared_kv_extension_replay_verifier",
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
        "adapter_optimizer_exact_replay": True,
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
    observed = _load_json(replay_path, "V23 replay report")
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
    candidate = {
        key: copy.deepcopy(row[key])
        for key in (
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
        )
    }
    candidate["color_eligible"] = shared._color_eligible(candidate["color"])
    candidate["continuation_gate_passed"] = shared._mirror_continuation(candidate["mirror"])
    candidate["full_teacher_gate_passed"] = (
        shared._full_pair(candidate["color"]) and shared._full_pair(candidate["mirror"])
    )
    return candidate


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
    _require_extension_layout(
        evidence["config"],
        set(range(3, 9)),
        field="V23 final extension namespace",
    )
    replay = _validate_replay_report(
        replay_path,
        current_provenance=current_provenance,
        transition=transition,
        expected_layout_epochs=set(range(3, 9)),
    )
    selected_metadata = _load_json(
        Path(manifest["selected_checkpoint"]) / "metadata.json",
        "selected metadata",
    )
    initialization = selected_metadata.get("initialization_provenance")
    previous_history = _validate_history_contract(selected_metadata, epoch=SELECTED_EPOCH)
    branch_rows: list[dict[str, Any]] = []
    for epoch in range(3, 9):
        row = _branch_epoch_record(evidence["config"], evidence["training_source_provenance"], epoch)
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
        if epoch in REPLAY_EPOCHS:
            replay_row = replay["epochs"][epoch - REPLAY_EPOCHS[0]]
            for key in replay_row:
                if key not in {"checkpoint", "metadata_path", "metadata_sha256"}:
                    _equal(row[key], replay_row[key], f"final/replay epoch {epoch} {key}")
        row.pop("raw_metadata")
        row.pop("history")
        row.pop("initialization_provenance")
        branch_rows.append(row)
    candidates = [
        _candidate(evidence["epochs"][0]),
        _candidate(evidence["epochs"][1]),
        *[_candidate(row) for row in branch_rows],
    ]
    _equal([row["epoch"] for row in candidates], list(range(1, 9)), "final trajectory")
    seen: dict[str, dict[str, Any]] = {}
    seen_optimizer: dict[str, dict[str, Any]] = {}
    repeated_full_plateau: list[int] = []
    for row in candidates:
        state = str(row["new_bank_state_sha256"])
        prior = seen.get(state)
        if prior is not None:
            allowed_plateau = (
                int(prior["epoch"]) == int(row["epoch"]) - 1
                and prior["full_teacher_gate_passed"] is True
                and row["full_teacher_gate_passed"] is True
                and prior["color"] == row["color"]
                and prior["mirror"] == row["mirror"]
            )
            if not allowed_plateau:
                _fail("extension trajectory repeats or rolls back a trainable bank state")
            repeated_full_plateau.append(int(row["epoch"]))
        seen[state] = row
        optimizer_state = str(row["optimizer_manifest"]["all_state_tensors_sha256"])
        prior_optimizer = seen_optimizer.get(optimizer_state)
        if prior_optimizer is not None:
            allowed_optimizer_plateau = (
                int(prior_optimizer["epoch"]) == int(row["epoch"]) - 1
                and prior_optimizer["full_teacher_gate_passed"] is True
                and row["full_teacher_gate_passed"] is True
                and prior_optimizer["color"] == row["color"]
                and prior_optimizer["mirror"] == row["mirror"]
            )
            if not allowed_optimizer_plateau:
                _fail("extension trajectory repeats or rolls back an optimizer state")
        seen_optimizer[optimizer_state] = row
    eligible = [row for row in candidates if row["color_eligible"]]
    if not eligible:
        _fail("final extension trajectory contains no color-eligible checkpoint")
    ranking = sorted(
        (copy.deepcopy(row) for row in eligible),
        key=_ranking_key,
        reverse=True,
    )
    # _ranking_key already encodes earlier epoch as a larger final component.
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    selected = ranking[0]
    full_teacher = bool(selected["full_teacher_gate_passed"])
    return {
        "schema_version": 1,
        "audit_type": "v23_shared_kv_extension_final_selector",
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
        "target_optimizer_update": TARGET_OPTIMIZER_UPDATE,
        "conditional_limit_reached": True,
        "continuation_authorized": False,
        "full_teacher_gate_passed": full_teacher,
        "greedy_audit_authorized": full_teacher,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": str(Path(selected["metadata_path"]).parent),
        "selected_checkpoint_artifact_hashes": {
            "adapter_sha256": selected["adapter_sha256"],
            "metadata_sha256": selected["metadata_sha256"],
            "optimizer_sha256": selected["optimizer_sha256"],
        },
        "repeated_full_teacher_plateau_epochs": repeated_full_plateau,
        "epochs": candidates,
        "ranking": ranking,
        "selection_policy": _selection_policy(),
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
        summary = {"authorized": True, "output": str(destination), "selected_epoch": 2}
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
    "EXPECTED_CONTROL_PLANE_TRANSITION",
    "EXTENSION_NAMESPACE",
    "V23ExtensionViolation",
    "authorize_stage_b",
    "prepare_extension_launch",
    "select_final_extension",
    "verify_replay",
]

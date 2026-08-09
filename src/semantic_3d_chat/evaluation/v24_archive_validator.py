"""Validate the sealed V24 shared-query screen and bounded extension archive.

V24 trained from one immutable source commit and ran its extension from a
separately pinned, clean controller commit.  Historical validation therefore
must not depend on the current checkout.  This module uses only the Python
standard library and validates either the tracked summary seal alone or all
locally retained reports and checkpoint artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_RELATIVE_PATH = Path("reports/gemma4/metrics/v24_final_summary.json")
EXPECTED_SUMMARY_SHA256 = "287328190cc2f9e3ff771fa9d9f08f7186c75c9cb22b3eadad3bd002e55b9eb3"
EXPECTED_TRAINING_COMMIT = "622dd3687756c0d6cebd332de860a7d01899bb8b"
EXPECTED_TRAINING_TREE = "45c2126beedd0dc34c34fb1b876cb56a0ae50af3"
EXPECTED_CONTROLLER_COMMIT = "b52cd8750a3f3bcecfd0e9c9073e2e214da65d06"
EXPECTED_CONTROLLER_TREE = "0c1a21c9a5ff3374badc6c75889f85dbae191b7a"
EXPECTED_CONFIG_SHA256 = "82d5fee205842fb86133498eb4ac7765e61c22e7e7bc2745cfa6a2e36b9447f1"
EXPECTED_CONTRACT_SHA256 = "3922eaed356dffa9a46ee601135cceb3e5a68e81e459805c8ddb8664a4c8a996"
EXPECTED_DECISION = "conditional_limit_reached_no_greedy_audit"
EXPECTED_SELECTED_EPOCH = 1
EXPECTED_SCREEN_RANKING = (1, 3, 2, 4)
EXPECTED_FINAL_RANKING = (1, 3, 6, 2, 4, 5, 8, 7)
# epoch, train loss, color full-vocab sides/units, color candidate/full means
# and minima, mirror full-vocab sides/units, mirror candidate/full means and minima
EXPECTED_TRAJECTORY = (
    (
        1,
        2.9427083333333335,
        12,
        6,
        4.254557132720947,
        0.875,
        2.0859375,
        0.8125,
        10,
        4,
        0.59375,
        -0.875,
        0.578125,
        -0.875,
    ),
    (
        2,
        2.8645833333333335,
        12,
        6,
        4.252604007720947,
        0.96875,
        2.1067707538604736,
        0.75,
        9,
        3,
        0.5833333134651184,
        -0.9375,
        0.5572916865348816,
        -0.9375,
    ),
    (
        3,
        2.9166666666666665,
        12,
        6,
        4.223795413970947,
        0.90625,
        2.0729167461395264,
        0.75,
        9,
        3,
        0.5885416865348816,
        -0.8125,
        0.5677083134651184,
        -0.8125,
    ),
    (
        4,
        2.7864583333333335,
        12,
        6,
        4.260091304779053,
        0.9375,
        2.1067707538604736,
        0.75,
        9,
        3,
        0.5833333134651184,
        -0.8125,
        0.5520833134651184,
        -0.8125,
    ),
    (
        5,
        2.7604166666666665,
        12,
        6,
        4.249674320220947,
        0.9375,
        2.1067707538604736,
        0.8125,
        9,
        3,
        0.5885416865348816,
        -0.8125,
        0.546875,
        -0.8125,
    ),
    (
        6,
        2.7604166666666665,
        12,
        6,
        4.255208492279053,
        0.9375,
        2.1223957538604736,
        0.8125,
        9,
        3,
        0.59375,
        -0.75,
        0.5625,
        -0.75,
    ),
    (
        7,
        2.8125,
        12,
        6,
        4.244791507720947,
        0.96875,
        2.0989582538604736,
        0.875,
        9,
        3,
        0.5729166865348816,
        -0.75,
        0.53125,
        -0.75,
    ),
    (
        8,
        2.8385416666666665,
        12,
        6,
        4.26708984375,
        0.96875,
        2.1067707538604736,
        0.875,
        9,
        3,
        0.609375,
        -0.8125,
        0.5416666865348816,
        -0.8125,
    ),
)


class V24ArchiveViolation(ValueError):
    """The V24 seal, evidence, checkpoint, or denied-output state changed."""


def _fail(message: str) -> None:
    raise V24ArchiveViolation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _equal(observed: Any, expected: Any, field: str) -> None:
    if observed != expected:
        _fail(f"{field} mismatch: expected={expected!r} observed={observed!r}")


def _file_sha256(path: Path, field: str) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            _fail(f"{field} is not a regular non-symlink file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except V24ArchiveViolation:
        raise
    except OSError as error:
        _fail(f"Cannot hash {field} at {path}: {error}")


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"Cannot load {field} at {path}: {error}")
    return dict(_mapping(value, field))


def _candidate_path(repo_root: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(f"{field} must be a non-empty relative path")
    lexical = Path(relative)
    if lexical.is_absolute() or ".." in lexical.parts:
        _fail(f"{field} must remain inside the repository: {relative}")
    root = repo_root.resolve()
    try:
        resolved = (root / lexical).resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        _fail(f"Cannot resolve {field} inside {root}: {error}")
    return resolved


def _require_clean_provenance(value: Any, *, commit: str, tree: str, field: str) -> None:
    provenance = _mapping(value, field)
    _equal(provenance.get("head_commit"), commit, f"{field}.head_commit")
    _equal(provenance.get("head_tree"), tree, f"{field}.head_tree")
    _equal(provenance.get("is_clean"), True, f"{field}.is_clean")
    _equal(provenance.get("scope"), "repository_excluding_generated_artifacts_v1", f"{field}.scope")
    _equal(
        provenance.get("tracked_diff_sha256"),
        hashlib.sha256(b"").hexdigest(),
        f"{field}.tracked_diff_sha256",
    )


def _artifact_index(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = _sequence(summary.get("authoritative_artifacts"), "authoritative_artifacts")
    indexed: dict[str, Mapping[str, Any]] = {}
    paths: set[str] = set()
    for index, value in enumerate(artifacts):
        artifact = _mapping(value, f"authoritative_artifacts[{index}]")
        role = artifact.get("role")
        path = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(role, str) or not role or role in indexed:
            _fail(f"authoritative_artifacts[{index}].role is empty or duplicated")
        if not isinstance(path, str) or not path or path in paths:
            _fail(f"authoritative_artifacts[{index}].path is empty or duplicated")
        if not isinstance(digest, str) or len(digest) != 64:
            _fail(f"authoritative_artifacts[{index}].sha256 is invalid")
        indexed[role] = artifact
        paths.add(path)
    _equal(
        set(indexed),
        {
            "config",
            "structural_preflight",
            "update_1_exact_match",
            "primary_training",
            "training_selection",
            "epoch_screen",
            "extension_launch",
            "exact_replay",
            "extension_training",
            "extension_selection",
            "final_selector",
        },
        "authoritative artifact roles",
    )
    return indexed


def _metric_tuple(row: Mapping[str, Any], train_loss: Any) -> tuple[Any, ...]:
    color = _mapping(row.get("color"), "trajectory color")
    mirror = _mapping(row.get("mirror"), "trajectory mirror")
    return (
        row.get("epoch"),
        train_loss,
        color.get("full_vocab_sides"),
        color.get("full_vocab_units"),
        color.get("mean_candidate_margin"),
        color.get("minimum_candidate_margin"),
        color.get("mean_full_vocab_margin"),
        color.get("minimum_full_vocab_margin"),
        mirror.get("full_vocab_sides"),
        mirror.get("full_vocab_units"),
        mirror.get("mean_candidate_margin"),
        mirror.get("minimum_candidate_margin"),
        mirror.get("mean_full_vocab_margin"),
        mirror.get("minimum_full_vocab_margin"),
    )


def validate_summary_contract(summary: Mapping[str, Any]) -> None:
    """Validate immutable facts defining the completed V24 experiment."""

    _equal(summary.get("schema_version"), 1, "schema_version")
    _equal(summary.get("archive_type"), "immutable_v24_final_summary", "archive_type")
    source = _mapping(summary.get("sealed_from_source"), "sealed_from_source")
    _require_clean_provenance(
        source.get("training"),
        commit=EXPECTED_TRAINING_COMMIT,
        tree=EXPECTED_TRAINING_TREE,
        field="sealed_from_source.training",
    )
    _require_clean_provenance(
        source.get("extension_controller"),
        commit=EXPECTED_CONTROLLER_COMMIT,
        tree=EXPECTED_CONTROLLER_TREE,
        field="sealed_from_source.extension_controller",
    )

    policy = _mapping(summary.get("integrity_policy"), "integrity_policy")
    _equal(policy.get("validator_is_source_head_independent"), True, "validator policy")
    _equal(policy.get("current_source_head_required"), False, "current source policy")
    _equal(policy.get("question_dependent_scene_processing"), False, "scene policy")

    config = _mapping(summary.get("config"), "config")
    _equal(config.get("resolved_config_sha256"), EXPECTED_CONFIG_SHA256, "config SHA-256")
    _equal(config.get("preflight_contract_sha256"), EXPECTED_CONTRACT_SHA256, "contract SHA-256")
    architecture = _mapping(summary.get("architecture"), "architecture")
    for field, expected in {
        "trainable_bank": "extension_v24_shared_query",
        "trainable_parameter_count": 36864,
        "rank": 4,
        "alpha": 8.0,
        "learning_rate": 0.0003,
        "scene_latent_count": 256,
        "language_hidden_size": 1536,
        "full_scene_prefix_tokens": 258,
    }.items():
        _equal(architecture.get(field), expected, f"architecture.{field}")
    _equal(
        architecture.get("target_modules"),
        [
            "model.language_model.layers.28.self_attn.q_proj",
            "model.language_model.layers.29.self_attn.q_proj",
        ],
        "architecture.target_modules",
    )

    preflight = _mapping(summary.get("preflight"), "preflight")
    for field, expected in {
        "authorized": True,
        "stage_1_authorized": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "new_bank_exact_zero_output": True,
        "all_final_prompt_queries_reach_entire_scene_prefix": True,
    }.items():
        _equal(preflight.get(field), expected, f"preflight.{field}")
    update1 = _mapping(summary.get("update_1_verification"), "update_1_verification")
    for field, expected in {
        "match": True,
        "stage_2_authorized": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "report_only": True,
        "all_prior_tensors_frozen": True,
        "a_tensors_unchanged": True,
        "b_tensors_all_changed": True,
    }.items():
        _equal(update1.get(field), expected, f"update_1_verification.{field}")
    replay = _mapping(summary.get("extension_replay"), "extension_replay")
    for field, expected in {
        "match": True,
        "stage_b_authorized": True,
        "history_prefix_exact": True,
        "initialization_provenance_exact": True,
        "adapter_and_decoded_optimizer_exact_replay": True,
        "optimizer_container_byte_identity_required": False,
    }.items():
        _equal(replay.get(field), expected, f"extension_replay.{field}")
    _equal(replay.get("replayed_epochs"), [2, 3, 4], "replayed epochs")

    outcome = _mapping(summary.get("outcome"), "outcome")
    for field, expected in {
        "decision": EXPECTED_DECISION,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "screen_optimizer_updates": 4,
        "conditional_max_optimizer_updates": 8,
        "extension_executed": True,
        "conditional_limit_reached": True,
        "full_teacher_gate_passed": False,
        "greedy_audit_authorized": False,
        "promotion_authorized": False,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "question_dependent_scene_processing": False,
        "report_only_final_selection": True,
        "selected_color_full_vocab_sides": 12,
        "selected_color_full_vocab_units": 6,
        "selected_mirror_full_vocab_sides": 10,
        "selected_mirror_full_vocab_units": 4,
        "training_question_count": 24,
        "validation_question_count": 0,
    }.items():
        _equal(outcome.get(field), expected, f"outcome.{field}")

    trajectory = _sequence(summary.get("trajectory"), "trajectory")
    _equal(len(trajectory), 8, "trajectory length")
    for value, expected in zip(trajectory, EXPECTED_TRAJECTORY, strict=True):
        row = _mapping(value, "trajectory row")
        _equal(
            _metric_tuple(row, row.get("train_loss")), expected, f"trajectory epoch {expected[0]}"
        )
        _equal(row.get("optimizer_step"), expected[0], "trajectory optimizer step")
        _equal(row.get("cumulative_microsteps"), expected[0] * 12, "trajectory microsteps")
        _equal(row.get("full_teacher_gate_passed"), False, "trajectory full gate")
    _equal(summary.get("ranking"), list(EXPECTED_FINAL_RANKING), "final ranking")

    selected = _mapping(summary.get("selected_checkpoint"), "selected_checkpoint")
    _equal(selected.get("epoch"), EXPECTED_SELECTED_EPOCH, "selected checkpoint epoch")
    _equal(
        selected.get("artifact_hashes"),
        _mapping(trajectory[0], "trajectory epoch 1").get("checkpoint_artifact_hashes"),
        "selected checkpoint hashes",
    )
    _equal(
        selected.get("new_bank_state_sha256"),
        _mapping(trajectory[0], "trajectory epoch 1").get("new_bank_state_sha256"),
        "selected bank hash",
    )
    _artifact_index(summary)

    limitations = _mapping(summary.get("limitations"), "limitations")
    for field, expected in {
        "teacher_forced_training_distribution_only": True,
        "held_out_scene_qa_evaluated": False,
        "greedy_generation_evaluated": False,
        "static_chat_validated": False,
        "leakage_suite_run_for_v24": False,
        "robot_or_mcp_validated": False,
        "research_goal_completed": False,
    }.items():
        _equal(limitations.get(field), expected, f"limitations.{field}")


def _validate_report_contracts(
    summary: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]], repo_root: Path
) -> None:
    outcome = _mapping(summary.get("outcome"), "outcome")
    preflight = _load_json(
        _candidate_path(repo_root, artifacts["structural_preflight"].get("path"), "preflight.path"),
        "structural preflight",
    )
    _require_clean_provenance(
        preflight.get("source_provenance"),
        commit=EXPECTED_TRAINING_COMMIT,
        tree=EXPECTED_TRAINING_TREE,
        field="preflight.source_provenance",
    )
    _equal(preflight.get("config_sha256"), EXPECTED_CONFIG_SHA256, "preflight config")
    _equal(preflight.get("contract_sha256"), EXPECTED_CONTRACT_SHA256, "preflight contract")
    _equal(preflight.get("stage_1_authorized"), True, "preflight authorization")

    update1 = _load_json(
        _candidate_path(repo_root, artifacts["update_1_exact_match"].get("path"), "update1.path"),
        "update 1",
    )
    _require_clean_provenance(
        update1.get("source_provenance"),
        commit=EXPECTED_TRAINING_COMMIT,
        tree=EXPECTED_TRAINING_TREE,
        field="update1.source_provenance",
    )
    _equal(update1.get("match"), True, "update 1 match")
    _equal(update1.get("stage_2_authorized"), True, "update 1 authorization")

    primary = _load_json(
        _candidate_path(repo_root, artifacts["primary_training"].get("path"), "primary.path"),
        "primary training",
    )
    extension = _load_json(
        _candidate_path(repo_root, artifacts["extension_training"].get("path"), "extension.path"),
        "extension training",
    )
    for name, report, epochs in (("primary", primary, 4), ("extension", extension, 8)):
        _require_clean_provenance(
            report.get("source_provenance"),
            commit=EXPECTED_TRAINING_COMMIT,
            tree=EXPECTED_TRAINING_TREE,
            field=f"{name}.source_provenance",
        )
        _equal(report.get("epochs"), epochs, f"{name} epochs")
        _equal(report.get("optimizer_steps"), epochs, f"{name} optimizer steps")
        _equal(report.get("question_count"), 24, f"{name} question count")
        _equal(report.get("validation_question_count"), 0, f"{name} validation count")
        _equal(report.get("question_dependent_scene_processing"), False, f"{name} scene policy")

    screen = _load_json(
        _candidate_path(repo_root, artifacts["epoch_screen"].get("path"), "screen.path"),
        "epoch screen",
    )
    _equal(
        screen.get("decision"),
        "screen_passed_extension_authorized_no_greedy_audit",
        "screen decision",
    )
    _equal(screen.get("selected_epoch"), 1, "screen selected epoch")
    _equal(
        tuple(
            _mapping(row, "screen ranking row").get("epoch")
            for row in _sequence(screen.get("ranking"), "screen ranking")
        ),
        EXPECTED_SCREEN_RANKING,
        "screen ranking",
    )
    _equal(
        screen.get("update1_report_sha256"),
        artifacts["update_1_exact_match"].get("sha256"),
        "screen update 1 binding",
    )

    launch = _load_json(
        _candidate_path(repo_root, artifacts["extension_launch"].get("path"), "launch.path"),
        "extension launch",
    )
    _require_clean_provenance(
        launch.get("training_source_provenance"),
        commit=EXPECTED_TRAINING_COMMIT,
        tree=EXPECTED_TRAINING_TREE,
        field="launch.training_source_provenance",
    )
    _require_clean_provenance(
        launch.get("controller_source_provenance"),
        commit=EXPECTED_CONTROLLER_COMMIT,
        tree=EXPECTED_CONTROLLER_TREE,
        field="launch.controller_source_provenance",
    )
    _equal(
        launch.get("screen_sha256"),
        artifacts["epoch_screen"].get("sha256"),
        "launch screen binding",
    )
    _equal(
        launch.get("selected_checkpoint_artifact_hashes"),
        _mapping(summary.get("selected_checkpoint"), "selected").get("artifact_hashes"),
        "launch selected hashes",
    )
    _equal(launch.get("replay_epochs"), [2, 3, 4], "launch replay epochs")
    _equal(launch.get("novel_epochs"), [5, 6, 7, 8], "launch novel epochs")

    replay = _load_json(
        _candidate_path(repo_root, artifacts["exact_replay"].get("path"), "replay.path"),
        "extension replay",
    )
    _equal(
        replay.get("manifest_sha256"),
        artifacts["extension_launch"].get("sha256"),
        "replay manifest binding",
    )
    _equal(replay.get("match"), True, "replay match")
    _equal(replay.get("stage_b_authorized"), True, "replay authorization")
    _equal(replay.get("adapter_and_decoded_optimizer_exact_replay"), True, "decoded replay")

    final = _load_json(
        _candidate_path(repo_root, artifacts["final_selector"].get("path"), "final.path"),
        "final selector",
    )
    for field in (
        "decision",
        "conditional_limit_reached",
        "full_teacher_gate_passed",
        "greedy_audit_authorized",
        "static_chat_authorized",
        "embodied_phase_authorized",
        "question_dependent_scene_processing",
    ):
        _equal(final.get(field), outcome.get(field), f"final selector {field}")
    _equal(final.get("selected_epoch"), EXPECTED_SELECTED_EPOCH, "final selected epoch")
    _equal(final.get("model_loaded"), False, "final selector model_loaded")
    _equal(final.get("oracle_loaded"), False, "final selector oracle_loaded")
    _equal(final.get("report_only"), True, "final selector report_only")
    _equal(
        final.get("screen_sha256"), artifacts["epoch_screen"].get("sha256"), "final screen binding"
    )
    _equal(
        final.get("manifest_sha256"),
        artifacts["extension_launch"].get("sha256"),
        "final launch binding",
    )
    _equal(
        final.get("replay_report_sha256"),
        artifacts["exact_replay"].get("sha256"),
        "final replay binding",
    )
    _equal(
        tuple(
            _mapping(row, "ranking row").get("epoch")
            for row in _sequence(final.get("ranking"), "ranking")
        ),
        EXPECTED_FINAL_RANKING,
        "final ranking",
    )
    final_epochs = _sequence(final.get("epochs"), "final epochs")
    trajectory = _sequence(summary.get("trajectory"), "summary trajectory")
    _equal(len(final_epochs), len(trajectory), "final epoch count")
    for final_row, summary_row in zip(final_epochs, trajectory, strict=True):
        observed = _mapping(final_row, "final epoch")
        archived = _mapping(summary_row, "summary epoch")
        _equal(
            _metric_tuple(observed, archived.get("train_loss")),
            _metric_tuple(archived, archived.get("train_loss")),
            f"final epoch {archived.get('epoch')}",
        )
        _equal(
            observed.get("adapter_sha256"),
            _mapping(archived.get("checkpoint_artifact_hashes"), "checkpoint hashes").get(
                "adapter_sha256"
            ),
            "final adapter hash",
        )
        _equal(
            observed.get("metadata_sha256"),
            _mapping(archived.get("checkpoint_artifact_hashes"), "checkpoint hashes").get(
                "metadata_sha256"
            ),
            "final metadata hash",
        )
        _equal(
            observed.get("optimizer_sha256"),
            _mapping(archived.get("checkpoint_artifact_hashes"), "checkpoint hashes").get(
                "optimizer_sha256"
            ),
            "final optimizer hash",
        )


def _validate_checkpoints(summary: Mapping[str, Any], repo_root: Path) -> int:
    trajectory = _sequence(summary.get("trajectory"), "trajectory")
    for index, value in enumerate(trajectory):
        row = _mapping(value, f"trajectory[{index}]")
        root = _candidate_path(repo_root, row.get("checkpoint_path"), "checkpoint_path")
        hashes = _mapping(row.get("checkpoint_artifact_hashes"), "checkpoint hashes")
        for filename, field in (
            ("adapter.safetensors", "adapter_sha256"),
            ("metadata.json", "metadata_sha256"),
            ("optimizer.pt", "optimizer_sha256"),
        ):
            _equal(
                _file_sha256(root / filename, f"epoch {row.get('epoch')} {filename}"),
                hashes.get(field),
                f"epoch {row.get('epoch')} {field}",
            )
        metadata = _load_json(root / "metadata.json", f"epoch {row.get('epoch')} metadata")
        _equal(metadata.get("epoch"), row.get("epoch"), "checkpoint metadata epoch")
        _equal(metadata.get("optimizer_step"), row.get("optimizer_step"), "optimizer step")
        _equal(
            metadata.get("lora_bank_state_sha256", {}).get("extension_v24_shared_query"),
            row.get("new_bank_state_sha256"),
            "new bank state SHA-256",
        )
    return len(trajectory)


def _validate_denial_absence(repo_root: Path, forbidden: Mapping[str, Any]) -> None:
    filename = forbidden.get("forbidden_checkpoint_filename")
    if not isinstance(filename, str) or not filename:
        _fail("forbidden checkpoint filename is invalid")
    for relative in _sequence(forbidden.get("checkpoint_roots"), "forbidden roots"):
        root = _candidate_path(repo_root, relative, "forbidden checkpoint root")
        if root.exists() and any(root.rglob(filename)):
            _fail(f"V24 denied promotion output exists below {root}")
    metrics = repo_root / "reports/gemma4/metrics"
    for pattern in _sequence(forbidden.get("forbidden_metric_globs"), "forbidden globs"):
        if not isinstance(pattern, str) or not pattern:
            _fail("forbidden metric glob is invalid")
        matches = sorted(path for path in metrics.glob(pattern) if path.is_file())
        if matches:
            _fail(f"V24 denied downstream metric output exists: {matches[0]}")


def validate_archive(
    archive_path: Path | None = None,
    *,
    repo_root: Path = PROJECT_ROOT,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    """Validate the immutable summary and optionally all retained evidence."""

    path = repo_root / ARCHIVE_RELATIVE_PATH if archive_path is None else archive_path
    observed_sha = _file_sha256(path, "V24 archive summary")
    _equal(observed_sha, EXPECTED_SUMMARY_SHA256, "V24 archive summary SHA-256")
    summary = _load_json(path, "V24 archive summary")
    validate_summary_contract(summary)
    artifacts = _artifact_index(summary)
    checkpoint_count = 0
    denial_verified = False
    if verify_bound_files:
        for role, artifact in artifacts.items():
            target = _candidate_path(repo_root, artifact.get("path"), f"{role}.path")
            _equal(_file_sha256(target, role), artifact.get("sha256"), f"{role} SHA-256")
        _validate_report_contracts(summary, artifacts, repo_root)
        checkpoint_count = _validate_checkpoints(summary, repo_root)
        _validate_denial_absence(
            repo_root, _mapping(summary.get("forbidden_outputs"), "forbidden_outputs")
        )
        denial_verified = True
    return {
        "valid": True,
        "summary_sha256": observed_sha,
        "training_source_commit": EXPECTED_TRAINING_COMMIT,
        "controller_source_commit": EXPECTED_CONTROLLER_COMMIT,
        "current_source_head_checked": False,
        "bound_files_verified": verify_bound_files,
        "denial_absence_verified": denial_verified,
        "authoritative_artifact_count": len(artifacts),
        "checkpoint_epoch_count": checkpoint_count,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "decision": EXPECTED_DECISION,
        "greedy_audit_authorized": False,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=PROJECT_ROOT / ARCHIVE_RELATIVE_PATH)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Validate the tracked summary seal without requiring ignored checkpoints.",
    )
    args = parser.parse_args(argv)
    result = validate_archive(
        args.archive, repo_root=PROJECT_ROOT, verify_bound_files=not args.summary_only
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

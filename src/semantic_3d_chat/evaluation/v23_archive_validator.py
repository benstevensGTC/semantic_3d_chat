"""Validate the sealed V23 screen and bounded extension archive.

The live V23 run used one clean training source and a separately pinned,
clean extension controller.  This validator deliberately does not consult the
current Git checkout: later experiments may change the source tree without
changing the historical result.  It uses only the Python standard library and
can validate either the tracked summary seal alone or every locally retained
report and checkpoint artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_RELATIVE_PATH = Path("reports/gemma4/metrics/v23_final_summary.json")
EXPECTED_SUMMARY_SHA256 = "cdb7cabc2b6a9a8420e682a20067dd501cc58a69addb2fb828d2b99fc94df208"
EXPECTED_TRAINING_COMMIT = "2a8cd075f7961092d756efaa1e619f62a14c1262"
EXPECTED_TRAINING_TREE = "2018b80f79736809637bde1d20ade56cc18e40d5"
EXPECTED_CONTROLLER_COMMIT = "39c12a1422aa01949efe32c0fa6f28221c6aa348"
EXPECTED_CONTROLLER_TREE = "820ed21bc294f1999f34bc75873bfc236de331f0"
EXPECTED_CONFIG_SHA256 = "5416ac62c7670cea067a92e6edfaadda450f9f78c3412b18209e1c63c578053e"
EXPECTED_CONTRACT_SHA256 = "a26ebc16efff574e15c61a541f8d0f68700da6bbb54654cfa90855e48c2f9fe4"
EXPECTED_DECISION = "conditional_limit_reached_no_greedy_audit"
EXPECTED_SELECTED_EPOCH = 2
EXPECTED_TRAJECTORY = (
    (1, 12, 6, 8, 2, 0.8125, -0.8125),
    (2, 12, 6, 10, 4, 0.8125, -1.0),
    (3, 12, 6, 9, 3, 0.875, -0.8125),
    (4, 12, 6, 9, 3, 0.8125, -0.9375),
    (5, 12, 6, 9, 3, 0.8125, -0.8125),
    (6, 12, 6, 9, 3, 0.75, -0.75),
    (7, 12, 6, 8, 2, 0.84375, -0.8125),
    (8, 12, 6, 8, 2, 0.8125, -0.6875),
)
EXPECTED_RANKING = (2, 3, 4, 6, 5, 1, 8, 7)


class V23ArchiveViolation(ValueError):
    """The V23 seal, evidence, checkpoint, or denied-output state changed."""


def _fail(message: str) -> None:
    raise V23ArchiveViolation(message)


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
    except V23ArchiveViolation:
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
    _equal(
        provenance.get("scope"),
        "repository_excluding_generated_artifacts_v1",
        f"{field}.scope",
    )
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


def validate_summary_contract(summary: Mapping[str, Any]) -> None:
    """Validate the immutable facts defining the completed V23 experiment."""

    _equal(summary.get("schema_version"), 1, "schema_version")
    _equal(summary.get("archive_type"), "immutable_v23_final_summary", "archive_type")
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
    _equal(architecture.get("trainable_parameter_count"), 30720, "trainable parameters")
    _equal(architecture.get("rank"), 4, "LoRA rank")
    _equal(architecture.get("alpha"), 8.0, "LoRA alpha")
    _equal(architecture.get("learning_rate"), 0.0003, "learning rate")

    preflight = _mapping(summary.get("preflight"), "preflight")
    for field, expected in {
        "authorized": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "new_bank_exact_zero_output": True,
    }.items():
        _equal(preflight.get(field), expected, f"preflight.{field}")
    update1 = _mapping(summary.get("update_1_verification"), "update_1_verification")
    for field, expected in {
        "match": True,
        "stage_2_authorized": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "report_only": True,
    }.items():
        _equal(update1.get(field), expected, f"update_1_verification.{field}")
    replay = _mapping(summary.get("extension_replay"), "extension_replay")
    for field, expected in {
        "match": True,
        "stage_b_authorized": True,
        "adapter_and_decoded_optimizer_exact_replay": True,
        "optimizer_container_byte_identity_required": False,
    }.items():
        _equal(replay.get(field), expected, f"extension_replay.{field}")
    _equal(replay.get("replayed_epochs"), [3, 4], "replayed epochs")

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
        color = _mapping(row.get("color"), "trajectory color")
        mirror = _mapping(row.get("mirror"), "trajectory mirror")
        observed = (
            row.get("epoch"),
            color.get("full_vocab_sides"),
            color.get("full_vocab_units"),
            mirror.get("full_vocab_sides"),
            mirror.get("full_vocab_units"),
            color.get("minimum_full_vocab_margin"),
            mirror.get("minimum_full_vocab_margin"),
        )
        _equal(observed, expected, f"trajectory epoch {expected[0]}")
        _equal(row.get("optimizer_step"), expected[0], "trajectory optimizer step")
        _equal(row.get("cumulative_microsteps"), expected[0] * 12, "trajectory microsteps")
        _equal(row.get("full_teacher_gate_passed"), False, "trajectory full gate")

    selected = _mapping(summary.get("selected_checkpoint"), "selected_checkpoint")
    _equal(selected.get("epoch"), EXPECTED_SELECTED_EPOCH, "selected checkpoint epoch")
    _equal(
        selected.get("artifact_hashes"),
        _mapping(trajectory[1], "trajectory epoch 2").get("checkpoint_artifact_hashes"),
        "selected checkpoint hashes",
    )
    _artifact_index(summary)
    superseded = _sequence(summary.get("superseded_lineage"), "superseded_lineage")
    _equal(len(superseded), 1, "superseded lineage count")
    _equal(
        _mapping(superseded[0], "superseded lineage").get("controller_commit"),
        "1697fcc3d547b9619d73352a93f4359cf1a8dc7e",
        "superseded controller commit",
    )


def _validate_final_selector(
    summary: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]], repo_root: Path
) -> None:
    artifact = artifacts["final_selector"]
    final = _load_json(
        _candidate_path(repo_root, artifact.get("path"), "final_selector.path"),
        "final selector",
    )
    outcome = _mapping(summary.get("outcome"), "outcome")
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
        EXPECTED_RANKING,
        "final ranking",
    )
    final_epochs = _sequence(final.get("epochs"), "final epochs")
    trajectory = _sequence(summary.get("trajectory"), "summary trajectory")
    _equal(len(final_epochs), len(trajectory), "final epoch count")
    for final_row, summary_row in zip(final_epochs, trajectory, strict=True):
        final_metrics = _mapping(final_row, "final epoch")
        archived = _mapping(summary_row, "summary epoch")
        _equal(final_metrics.get("epoch"), archived.get("epoch"), "final epoch number")
        for pair in ("color", "mirror"):
            observed = _mapping(final_metrics.get(pair), f"final {pair}")
            expected = _mapping(archived.get(pair), f"summary {pair}")
            for field in (
                "full_vocab_sides",
                "full_vocab_units",
                "mean_full_vocab_margin",
                "minimum_full_vocab_margin",
            ):
                _equal(observed.get(field), expected.get(field), f"final {pair}.{field}")


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
            metadata.get("lora_bank_state_sha256", {}).get("extension_v23_shared_kv"),
            row.get("new_bank_state_sha256"),
            "new bank state SHA-256",
        )
    return len(trajectory)


def _validate_superseded_lineage(summary: Mapping[str, Any], repo_root: Path) -> None:
    record = _mapping(
        _sequence(summary.get("superseded_lineage"), "superseded_lineage")[0],
        "superseded lineage",
    )
    launch = _candidate_path(repo_root, record.get("launch_path"), "superseded launch")
    _equal(
        _file_sha256(launch, "superseded launch"),
        record.get("launch_sha256"),
        "superseded launch SHA-256",
    )
    root = _candidate_path(repo_root, record.get("checkpoint_root"), "superseded root")
    checkpoints = _mapping(record.get("checkpoint_artifact_hashes"), "superseded hashes")
    for name, value in checkpoints.items():
        hashes = _mapping(value, f"superseded {name}")
        checkpoint = root / name
        for filename, field in (
            ("adapter.safetensors", "adapter_sha256"),
            ("metadata.json", "metadata_sha256"),
            ("optimizer.pt", "optimizer_sha256"),
        ):
            _equal(
                _file_sha256(checkpoint / filename, f"superseded {name} {filename}"),
                hashes.get(field),
                f"superseded {name} {field}",
            )


def _validate_denial_absence(repo_root: Path, forbidden: Mapping[str, Any]) -> None:
    filename = forbidden.get("forbidden_checkpoint_filename")
    if not isinstance(filename, str) or not filename:
        _fail("forbidden checkpoint filename is invalid")
    for relative in _sequence(forbidden.get("checkpoint_roots"), "forbidden roots"):
        root = _candidate_path(repo_root, relative, "forbidden checkpoint root")
        if root.exists() and any(root.rglob(filename)):
            _fail(f"V23 denied promotion output exists below {root}")
    metrics = repo_root / "reports/gemma4/metrics"
    for pattern in _sequence(forbidden.get("forbidden_metric_globs"), "forbidden globs"):
        if not isinstance(pattern, str) or not pattern:
            _fail("forbidden metric glob is invalid")
        matches = sorted(path for path in metrics.glob(pattern) if path.is_file())
        if matches:
            _fail(f"V23 denied downstream metric output exists: {matches[0]}")


def validate_archive(
    archive_path: Path | None = None,
    *,
    repo_root: Path = PROJECT_ROOT,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    """Validate the immutable summary and optionally all retained local evidence."""

    path = repo_root / ARCHIVE_RELATIVE_PATH if archive_path is None else archive_path
    observed_sha = _file_sha256(path, "V23 archive summary")
    _equal(observed_sha, EXPECTED_SUMMARY_SHA256, "V23 archive summary SHA-256")
    summary = _load_json(path, "V23 archive summary")
    validate_summary_contract(summary)
    artifacts = _artifact_index(summary)
    checkpoint_count = 0
    denial_verified = False
    superseded_verified = False
    if verify_bound_files:
        for role, artifact in artifacts.items():
            target = _candidate_path(repo_root, artifact.get("path"), f"{role}.path")
            _equal(
                _file_sha256(target, role),
                artifact.get("sha256"),
                f"{role} SHA-256",
            )
        _validate_final_selector(summary, artifacts, repo_root)
        checkpoint_count = _validate_checkpoints(summary, repo_root)
        _validate_superseded_lineage(summary, repo_root)
        superseded_verified = True
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
        "superseded_attempt_verified": superseded_verified,
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
        args.archive,
        repo_root=PROJECT_ROOT,
        verify_bound_files=not args.summary_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

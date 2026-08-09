"""Validate the sealed V22 screen without consulting the current Git HEAD.

V22's live evidence was correctly produced from one clean, exact source commit.
This archival validator keeps that completed result verifiable after later
experiments change the checkout. It uses only the Python standard library,
validates the pinned summary bytes and every bound artifact/checkpoint, and
fails if any forbidden extension, greedy-audit, or promotion output appears.
It deliberately performs no Git query and has no current-source dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARCHIVE_RELATIVE_PATH = Path("reports/gemma4/metrics/v22_final_summary.json")
EXPECTED_SUMMARY_SHA256 = "d13946cf5811b17eaee9573a0a0d245ba147fd793ad7961ccacac243e884cf8d"
EXPECTED_SOURCE_COMMIT = "cffeb3686996cfbac5406ebc107c895ea06cceac"
EXPECTED_SOURCE_TREE = "b90e68a82b5276a7560d42442a27606067de888f"
EXPECTED_CONFIG_SHA256 = "b336be25fd68127191e86c2337d9b66baf0f5972cc6dade27bfcecfd5368c999"
EXPECTED_CONTRACT_SHA256 = "a8994abafc02720a96f47fbdab222f487e2ea6310c690dedd4c8b2f5232c3c4b"
EXPECTED_DECISION = "screen_failed_no_extension_no_greedy_audit"
EXPECTED_SELECTED_EPOCH = 3


class V22ArchiveViolation(ValueError):
    """The sealed V22 summary, evidence, denial state, or checkpoint changed."""


def _fail(message: str) -> None:
    raise V22ArchiveViolation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _require_equal(observed: Any, expected: Any, field: str) -> None:
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
    except V22ArchiveViolation:
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


def _bound_path(repo_root: Path, relative: Any, field: str) -> Path:
    resolved = _candidate_path(repo_root, relative, field)
    if not resolved.exists():
        _fail(f"Required {field} does not exist: {resolved}")
    return resolved


def _artifact_index(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = _sequence(summary.get("authoritative_artifacts"), "authoritative_artifacts")
    indexed: dict[str, Mapping[str, Any]] = {}
    paths: set[str] = set()
    for index, value in enumerate(artifacts):
        artifact = _mapping(value, f"authoritative_artifacts[{index}]")
        role = artifact.get("role")
        path = artifact.get("path")
        if not isinstance(role, str) or not role or role in indexed:
            _fail(f"authoritative_artifacts[{index}].role is empty or duplicated")
        if not isinstance(path, str) or not path or path in paths:
            _fail(f"authoritative_artifacts[{index}].path is empty or duplicated")
        indexed[role] = artifact
        paths.add(path)
    _require_equal(
        set(indexed),
        {
            "config",
            "structural_preflight",
            "update_1_exact_match",
            "training_report",
            "training_selection",
            "epoch_screen",
        },
        "authoritative artifact roles",
    )
    return indexed


def validate_summary_contract(summary: Mapping[str, Any]) -> None:
    """Validate the immutable values that define the V22 historical result."""

    _require_equal(summary.get("schema_version"), 1, "schema_version")
    _require_equal(summary.get("archive_type"), "immutable_v22_final_summary", "archive_type")
    source = _mapping(summary.get("sealed_from_source"), "sealed_from_source")
    _require_equal(source.get("head_commit"), EXPECTED_SOURCE_COMMIT, "source commit")
    _require_equal(source.get("head_tree"), EXPECTED_SOURCE_TREE, "source tree")
    _require_equal(source.get("is_clean"), True, "original source clean flag")
    _require_equal(
        source.get("tracked_diff_sha256"),
        hashlib.sha256(b"").hexdigest(),
        "original tracked diff hash",
    )
    integrity = _mapping(summary.get("integrity_policy"), "integrity_policy")
    _require_equal(
        integrity.get("validator_is_source_head_independent"),
        True,
        "source-HEAD-independent policy",
    )
    _require_equal(
        integrity.get("current_source_head_required"), False, "current source HEAD policy"
    )

    config = _mapping(summary.get("config"), "config")
    _require_equal(
        config.get("resolved_config_sha256"), EXPECTED_CONFIG_SHA256, "resolved config hash"
    )
    _require_equal(
        config.get("preflight_contract_sha256"),
        EXPECTED_CONTRACT_SHA256,
        "preflight contract hash",
    )
    _artifact_index(summary)

    preflight = _mapping(summary.get("preflight"), "preflight")
    for field, expected in {
        "authorized": True,
        "structural_authorization": True,
        "structural_gate_passed": True,
        "predicted_update_functional_audit_passed": True,
        "live_optimizer_constructed": False,
        "live_optimizer_step_executed": False,
        "live_optimizer_steps": 0,
        "isolated_clone_optimizer_constructed": True,
        "isolated_clone_optimizer_steps": 1,
        "question_dependent_scene_processing": False,
        "runtime_eligible": False,
    }.items():
        _require_equal(preflight.get(field), expected, f"preflight.{field}")

    update1 = _mapping(summary.get("update_1_verification"), "update_1_verification")
    for field, expected in {
        "match": True,
        "stage_2_authorized": True,
        "model_loaded": False,
        "report_only": True,
        "optimizer_deserialized": True,
        "optimizer_deserialization_weights_only": True,
        "scene_map_loaded": False,
        "oracle_loaded": False,
    }.items():
        _require_equal(update1.get(field), expected, f"update_1_verification.{field}")

    outcome = _mapping(summary.get("outcome"), "outcome")
    for field, expected in {
        "decision": EXPECTED_DECISION,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "screen_optimizer_updates": 4,
        "selected_optimizer_step": 3,
        "selected_cumulative_microsteps": 36,
        "continuation_gate_passed": False,
        "continuation_authorized": False,
        "extension_authorized": False,
        "extension_executed": False,
        "full_teacher_gate_passed": False,
        "greedy_audit_authorized": False,
        "greedy_audit_forbidden": True,
        "promotion_authorized": False,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "question_dependent_scene_processing": False,
        "report_only_final_selection": True,
        "selected_color_full_vocab_sides": 12,
        "selected_color_full_vocab_units": 6,
        "selected_mirror_full_vocab_sides": 7,
        "selected_mirror_full_vocab_units": 1,
    }.items():
        _require_equal(outcome.get(field), expected, f"outcome.{field}")

    trajectory = _sequence(summary.get("trajectory"), "trajectory")
    _require_equal(len(trajectory), 4, "trajectory length")
    _require_equal(
        [_mapping(row, "trajectory row").get("epoch") for row in trajectory],
        [1, 2, 3, 4],
        "trajectory epoch order",
    )
    for row in trajectory:
        metrics = _mapping(row, "trajectory row")
        _require_equal(metrics.get("continuation_gate_passed"), False, "continuation row")
        _require_equal(metrics.get("full_teacher_gate_passed"), False, "full-teacher row")
        mirror = _mapping(metrics.get("mirror"), "trajectory mirror")
        _require_equal(mirror.get("full_vocab_sides"), 7, "mirror sides")
        _require_equal(mirror.get("full_vocab_units"), 1, "mirror units")

    selected = _mapping(summary.get("selected_checkpoint"), "selected_checkpoint")
    _require_equal(selected.get("epoch"), EXPECTED_SELECTED_EPOCH, "selected checkpoint epoch")
    _require_equal(
        selected.get("artifact_hashes"),
        _mapping(trajectory[2], "trajectory epoch 3").get("checkpoint_artifact_hashes"),
        "selected checkpoint hashes",
    )
    _require_equal(summary.get("superseded_lineage"), [], "superseded lineage")


def _metric_projection(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "full_vocab_sides": metrics.get("full_vocab_sides"),
        "full_vocab_units": metrics.get("full_vocab_units"),
        "mean_full_vocab_margin": metrics.get("mean_full_vocab_margin"),
        "minimum_full_vocab_margin": metrics.get("minimum_full_vocab_margin"),
    }


def _trajectory_from_reports(
    screen: Mapping[str, Any], training: Mapping[str, Any]
) -> list[dict[str, Any]]:
    epochs = _sequence(screen.get("epochs"), "screen.epochs")
    history = _sequence(training.get("history"), "training.history")
    if len(epochs) != 4 or len(history) != 4:
        _fail("V22 screen epochs and training history must both contain four rows")
    losses: dict[int, Any] = {}
    for index, value in enumerate(history):
        row = _mapping(value, f"training.history[{index}]")
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch in losses:
            _fail("Training history has an invalid or duplicate epoch")
        losses[epoch] = row.get("train_loss")

    result: list[dict[str, Any]] = []
    for index, value in enumerate(epochs):
        row = _mapping(value, f"screen.epochs[{index}]")
        epoch = row.get("epoch")
        result.append(
            {
                "epoch": epoch,
                "optimizer_step": row.get("optimizer_step"),
                "cumulative_microsteps": row.get("cumulative_microsteps"),
                "train_loss": losses.get(epoch),
                "continuation_gate_passed": row.get("continuation_gate_passed"),
                "full_teacher_gate_passed": row.get("full_teacher_gate_passed"),
                "color": _metric_projection(_mapping(row.get("color"), "epoch color")),
                "mirror": _metric_projection(_mapping(row.get("mirror"), "epoch mirror")),
                "checkpoint_artifact_hashes": row.get("checkpoint_artifact_hashes"),
                "optimizer_state_sha256": row.get("optimizer_state_sha256"),
                "signed_x_state_sha256": row.get("signed_x_state_sha256"),
            }
        )
    return result


def _validate_denial_absence(repo_root: Path, forbidden: Mapping[str, Any]) -> None:
    """Fail if any output forbidden by V22's closed selector now exists."""

    extension_root = _candidate_path(
        repo_root, forbidden.get("extension_checkpoint_root"), "extension checkpoint root"
    )
    if extension_root.exists() or extension_root.is_symlink():
        _fail(f"V22 extension was denied, but its checkpoint root exists: {extension_root}")

    for index, relative in enumerate(
        _sequence(forbidden.get("exact_report_paths"), "forbidden exact report paths")
    ):
        path = _candidate_path(repo_root, relative, f"forbidden exact report path {index}")
        if path.exists() or path.is_symlink():
            _fail(f"V22 extension/greedy output was denied, but report exists: {path}")

    primary_root = _bound_path(
        repo_root, forbidden.get("primary_checkpoint_root"), "primary checkpoint root"
    )
    if not primary_root.is_dir():
        _fail(f"V22 primary checkpoint root is not a directory: {primary_root}")
    promotion_name = forbidden.get("promotion_filename")
    if not isinstance(promotion_name, str) or not promotion_name:
        _fail("forbidden_outputs.promotion_filename is invalid")
    promotions = sorted(primary_root.rglob(promotion_name))
    if promotions:
        _fail(f"V22 was denied promotion, but promotion record exists: {promotions[0]}")

    reports_root = _bound_path(repo_root, "reports/gemma4/metrics", "metrics report root")
    for pattern in _sequence(forbidden.get("report_globs"), "forbidden report globs"):
        if not isinstance(pattern, str) or not pattern or "/" in pattern or ".." in pattern:
            _fail(f"Invalid forbidden report glob: {pattern!r}")
        matches = sorted(reports_root.glob(pattern))
        if matches:
            _fail(f"V22 greedy/promotion output was denied, but report exists: {matches[0]}")


def validate_archive(
    summary_path: str | Path,
    *,
    repo_root: str | Path,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    """Validate V22's sealed screen and denial state without inspecting Git."""

    root = Path(repo_root).resolve()
    summary_file = Path(summary_path).resolve()
    summary_sha256 = _file_sha256(summary_file, "V22 final summary")
    _require_equal(summary_sha256, EXPECTED_SUMMARY_SHA256, "archive summary SHA-256")
    summary = _load_json(summary_file, "V22 final summary")
    validate_summary_contract(summary)
    if not verify_bound_files:
        return {
            "valid": True,
            "archive_type": summary["archive_type"],
            "summary_sha256": summary_sha256,
            "bound_files_verified": False,
            "denial_absence_verified": False,
            "current_source_head_checked": False,
            "sealed_source_commit": EXPECTED_SOURCE_COMMIT,
            "decision": EXPECTED_DECISION,
        }

    artifacts = _artifact_index(summary)
    loaded: dict[str, dict[str, Any]] = {}
    for role, artifact in artifacts.items():
        path = _bound_path(root, artifact.get("path"), f"{role}.path")
        _require_equal(_file_sha256(path, role), artifact.get("sha256"), f"{role} SHA-256")
        if path.suffix == ".json":
            loaded[role] = _load_json(path, role)

    source = summary["sealed_from_source"]
    config = _mapping(summary["config"], "config")
    _require_equal(artifacts["config"].get("path"), config.get("path"), "config path")
    _require_equal(artifacts["config"].get("sha256"), config.get("file_sha256"), "config hash")

    preflight_report = loaded["structural_preflight"]
    preflight = _mapping(summary["preflight"], "preflight")
    for report_field, summary_field in (
        ("audit_type", "audit_type"),
        ("authorized", "authorized"),
        ("structural_authorization", "structural_authorization"),
        ("live_optimizer_constructed", "live_optimizer_constructed"),
        ("live_optimizer_step_executed", "live_optimizer_step_executed"),
        ("optimizer_steps", "live_optimizer_steps"),
        ("isolated_clone_optimizer_constructed", "isolated_clone_optimizer_constructed"),
        ("isolated_clone_optimizer_steps", "isolated_clone_optimizer_steps"),
        ("model_dtype", "model_dtype"),
        ("question_dependent_scene_processing", "question_dependent_scene_processing"),
        ("uses_supervised_qa_metadata", "uses_supervised_qa_metadata"),
        ("runtime_eligible", "runtime_eligible"),
    ):
        _require_equal(
            preflight_report.get(report_field),
            preflight.get(summary_field),
            f"preflight report {report_field}",
        )
    _require_equal(preflight_report.get("source_provenance"), source, "preflight source")
    _require_equal(preflight_report.get("config_hash"), EXPECTED_CONFIG_SHA256, "preflight config")
    _require_equal(
        _mapping(preflight_report.get("contract"), "preflight contract").get("contract_sha256"),
        EXPECTED_CONTRACT_SHA256,
        "preflight contract",
    )
    structural_gate = _mapping(preflight_report.get("structural_gate"), "structural gate")
    _require_equal(
        structural_gate.get("passed"), preflight["structural_gate_passed"], "structural gate"
    )
    _require_equal(
        _mapping(structural_gate.get("raw_pair_selectivity"), "raw selectivity").get(
            "mirror_to_color_normalized_selectivity"
        ),
        preflight["raw_mirror_to_color_normalized_selectivity"],
        "raw mirror/color selectivity",
    )
    _require_equal(
        structural_gate.get("mirror_signal_to_orthogonal_noise_ratio"),
        preflight["mirror_signal_to_orthogonal_noise_ratio"],
        "mirror signal/noise ratio",
    )
    predicted = _mapping(
        preflight_report.get("predicted_update_functional_audit"), "predicted audit"
    )
    _require_equal(predicted.get("passed"), True, "predicted audit pass")
    objective = _mapping(predicted.get("mirror_objective_change"), "predicted mirror change")
    for report_field, summary_field in (
        ("before", "predicted_mirror_objective_before"),
        ("after", "predicted_mirror_objective_after"),
        ("absolute_improvement", "predicted_mirror_objective_absolute_improvement"),
    ):
        _require_equal(
            objective.get(report_field), preflight.get(summary_field), f"predicted {report_field}"
        )

    update1_report = loaded["update_1_exact_match"]
    update1 = _mapping(summary["update_1_verification"], "update_1_verification")
    for field in (
        "audit_type",
        "match",
        "stage_2_authorized",
        "model_loaded",
        "report_only",
        "optimizer_deserialized",
        "scene_map_loaded",
        "oracle_loaded",
        "preflight_sha256",
        "checkpoint_artifact_hashes",
        "optimizer_state_sha256",
        "signed_x_state_sha256",
    ):
        _require_equal(update1_report.get(field), update1.get(field), f"update-1 {field}")
    _require_equal(
        _mapping(update1_report.get("optimizer_deserialization"), "optimizer deserialization").get(
            "weights_only"
        ),
        update1["optimizer_deserialization_weights_only"],
        "update-1 weights-only deserialization",
    )
    _require_equal(update1_report.get("source_provenance"), source, "update-1 source")
    _require_equal(update1_report.get("config_hash"), EXPECTED_CONFIG_SHA256, "update-1 config")
    _require_equal(
        update1_report.get("preflight_contract_sha256"),
        EXPECTED_CONTRACT_SHA256,
        "update-1 contract",
    )

    screen = loaded["epoch_screen"]
    outcome = _mapping(summary["outcome"], "outcome")
    for field in (
        "decision",
        "selected_epoch",
        "continuation_gate_passed",
        "continuation_authorized",
        "full_teacher_gate_passed",
        "greedy_audit_authorized",
        "greedy_audit_forbidden",
        "question_dependent_scene_processing",
    ):
        _require_equal(screen.get(field), outcome.get(field), f"screen {field}")
    _require_equal(screen.get("report_only"), True, "screen report-only flag")
    _require_equal(screen.get("model_inference_executed"), False, "screen inference flag")
    _require_equal(screen.get("gemma_model_loaded"), False, "screen Gemma load flag")
    _require_equal(screen.get("source_provenance"), source, "screen source")
    _require_equal(screen.get("config_hash_full"), EXPECTED_CONFIG_SHA256, "screen config")
    _require_equal(
        screen.get("preflight_contract_sha256"), EXPECTED_CONTRACT_SHA256, "screen contract"
    )

    training = loaded["training_report"]
    _require_equal(training.get("source_provenance"), source, "training source")
    _require_equal(training.get("optimizer_steps"), 4, "training optimizer steps")
    _require_equal(training.get("epochs"), 4, "training epochs")
    _require_equal(training.get("stopped_early"), False, "training early-stop flag")
    _require_equal(
        training.get("elapsed_seconds"),
        _mapping(summary["timing"], "timing").get("updates_1_to_4_training_elapsed_seconds"),
        "training elapsed seconds",
    )
    _require_equal(
        _trajectory_from_reports(screen, training), summary.get("trajectory"), "V22 trajectory"
    )

    primary_root = _mapping(summary["forbidden_outputs"], "forbidden outputs").get(
        "primary_checkpoint_root"
    )
    for value in _sequence(summary["trajectory"], "trajectory"):
        row = _mapping(value, "trajectory row")
        epoch = row.get("epoch")
        checkpoint = f"{primary_root}/epoch_{epoch:03d}"
        hashes = _mapping(row.get("checkpoint_artifact_hashes"), "epoch artifact hashes")
        for filename, hash_field in (
            ("adapter.safetensors", "adapter_sha256"),
            ("metadata.json", "metadata_sha256"),
            ("optimizer.pt", "optimizer_sha256"),
        ):
            path = _bound_path(root, f"{checkpoint}/{filename}", f"epoch {epoch} {filename}")
            _require_equal(
                _file_sha256(path, f"epoch {epoch} {filename}"),
                hashes.get(hash_field),
                f"epoch {epoch} {hash_field}",
            )

    selected = _mapping(summary["selected_checkpoint"], "selected checkpoint")
    _require_equal(
        screen.get("selected_checkpoint"), selected.get("checkpoint_path"), "selected path"
    )
    _require_equal(
        screen.get("selected_checkpoint_artifact_hashes"),
        selected.get("artifact_hashes"),
        "selected artifact hashes",
    )
    internal = _mapping(selected["internal_state_hashes"], "selected internal hashes")
    _require_equal(
        screen.get("selected_optimizer_state_sha256"),
        internal.get("optimizer_state_sha256"),
        "selected optimizer-state hash",
    )
    _require_equal(
        screen.get("selected_signed_x_state_sha256"),
        internal.get("signed_x_state_sha256"),
        "selected signed-X hash",
    )
    selected_metadata_path = _bound_path(root, selected.get("metadata_path"), "selected metadata")
    selected_metadata = _load_json(selected_metadata_path, "selected metadata")
    for metadata_field, summary_field in (
        ("signed_x_scene_residual_state_sha256", "signed_x_state_sha256"),
        ("frozen_global_scene_residual_state_sha256", "frozen_global_scene_residual_state_sha256"),
        ("frozen_scene_state_sha256", "frozen_scene_state_sha256"),
    ):
        _require_equal(
            selected_metadata.get(metadata_field),
            internal.get(summary_field),
            f"selected metadata {metadata_field}",
        )

    _validate_denial_absence(root, _mapping(summary["forbidden_outputs"], "forbidden outputs"))
    return {
        "valid": True,
        "archive_type": summary["archive_type"],
        "summary_sha256": summary_sha256,
        "bound_files_verified": True,
        "authoritative_artifact_count": len(artifacts),
        "checkpoint_epoch_count": 4,
        "denial_absence_verified": True,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "current_source_head_checked": False,
        "sealed_source_commit": EXPECTED_SOURCE_COMMIT,
        "decision": EXPECTED_DECISION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ARCHIVE_RELATIVE_PATH)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Validate the tracked summary seal without requiring generated evidence files.",
    )
    args = parser.parse_args()
    result = validate_archive(
        args.summary,
        repo_root=args.repo_root,
        verify_bound_files=not args.summary_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHIVE_RELATIVE_PATH",
    "EXPECTED_SUMMARY_SHA256",
    "V22ArchiveViolation",
    "validate_archive",
    "validate_summary_contract",
]

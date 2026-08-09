"""Validate the immutable V21 evidence archive without consulting Git HEAD.

The V21 trainer and selectors correctly required a clean, exact source commit
while producing evidence.  This archival validator has a different job: keep
that completed evidence verifiable after later experiments change the source
tree.  It therefore validates the pinned summary bytes, every named artifact,
the recorded original source provenance, the final decision, the metric
trajectory, and the selected checkpoint, but deliberately performs no Git
query and makes no assertion about the current checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARCHIVE_RELATIVE_PATH = Path("reports/gemma4/metrics/v21_final_summary.json")
EXPECTED_SUMMARY_SHA256 = "c7d0e7e7a40d99f64775e770bce10341ad33e09d7af7fb42700a8390f55b95e4"
EXPECTED_SOURCE_COMMIT = "806309b71127d4efa7f2e3a5ded7f3dafd853c2e"
EXPECTED_SOURCE_TREE = "dd4c60fdfa386f7a34d2c22c9e33db6bee68fdd3"
EXPECTED_DECISION = "conditional_limit_reached_no_greedy_audit"
EXPECTED_SELECTED_EPOCH = 8


class V21ArchiveViolation(ValueError):
    """The sealed V21 summary or one of its bound artifacts was altered."""


def _fail(message: str) -> None:
    raise V21ArchiveViolation(message)


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
    except V21ArchiveViolation:
        raise
    except OSError as error:
        _fail(f"Cannot hash {field} at {path}: {error}")


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"Cannot load {field} at {path}: {error}")
    return dict(_mapping(value, field))


def _bound_path(repo_root: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(f"{field} must be a non-empty relative path")
    lexical = Path(relative)
    if lexical.is_absolute() or ".." in lexical.parts:
        _fail(f"{field} must remain inside the repository: {relative}")
    root = repo_root.resolve()
    try:
        resolved = (root / lexical).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        _fail(f"Cannot resolve {field} inside {root}: {error}")
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
    required = {
        "config",
        "structural_preflight",
        "update_1_exact_match",
        "base_training_report",
        "base_selection_report",
        "epoch_4_screen",
        "extension_launch",
        "extension_training_report",
        "extension_selection_report",
        "final_selector_report",
    }
    _require_equal(set(indexed), required, "authoritative artifact roles")
    return indexed


def validate_summary_contract(summary: Mapping[str, Any]) -> None:
    """Validate constants that make the tracked summary a fail-closed seal."""

    _require_equal(summary.get("schema_version"), 1, "schema_version")
    _require_equal(summary.get("archive_type"), "immutable_v21_final_summary", "archive_type")
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
        config.get("resolved_config_sha256"),
        "ae17da8b9a712e9be89cc7d0f04d6db54bce0c239adf69c3236d848b64d9b04b",
        "resolved config hash",
    )
    _require_equal(
        config.get("preflight_contract_sha256"),
        "50e5522a19d4f6a3eb88884cdccfa71ab1301ebe94bf1a512d42505322799b2c",
        "preflight contract hash",
    )
    _artifact_index(summary)

    selected = _mapping(summary.get("selected_checkpoint"), "selected_checkpoint")
    _require_equal(selected.get("epoch"), EXPECTED_SELECTED_EPOCH, "selected epoch")
    hashes = _mapping(selected.get("artifact_hashes"), "selected checkpoint hashes")
    _require_equal(
        dict(hashes),
        {
            "adapter_sha256": "ce9e97061389a7eae5703593d0a8869f87bd12544f56f5976570965056b65f44",
            "metadata_sha256": "bbc8309d25db86e40fa01ec744e19b3c0fc1c61953ebfc5072f11c84bbd2e997",
            "optimizer_sha256": "465a9075c7d890bc87caa94ecf2fe316750e714716fa39bb48cda72d80c9bf93",
        },
        "selected checkpoint hashes",
    )

    outcome = _mapping(summary.get("outcome"), "outcome")
    exact_outcome = {
        "decision": EXPECTED_DECISION,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "optimizer_updates": 8,
        "cumulative_microsteps": 96,
        "conditional_limit_reached": True,
        "full_teacher_gate_passed": False,
        "continuation_authorized": False,
        "greedy_audit_authorized": False,
        "greedy_audit_forbidden": True,
        "promotion_authorized": False,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "question_dependent_scene_processing": False,
        "report_only_final_selection": True,
        "final_color_full_vocab_sides": 12,
        "final_color_full_vocab_units": 6,
        "final_mirror_full_vocab_sides": 8,
        "final_mirror_full_vocab_units": 2,
    }
    for field, expected in exact_outcome.items():
        _require_equal(outcome.get(field), expected, f"outcome.{field}")

    trajectory = _sequence(summary.get("trajectory"), "trajectory")
    _require_equal(len(trajectory), 8, "trajectory length")
    _require_equal(
        [dict(_mapping(row, "trajectory row")).get("epoch") for row in trajectory],
        list(range(1, 9)),
        "trajectory epoch order",
    )
    if any(_mapping(row, "trajectory row").get("full_teacher_gate_passed") for row in trajectory):
        _fail("No V21 trajectory row may claim the full teacher gate passed")

    lineage = _sequence(summary.get("superseded_lineage"), "superseded_lineage")
    _require_equal(len(lineage), 2, "superseded lineage length")
    _require_equal(
        [_mapping(row, "superseded lineage row").get("source_commit") for row in lineage],
        [
            "16bc57fe744c87fd4e2cddf5980efb97ee423766",
            "0a21a78213258459f43d6e0660d5bda368a546c9",
        ],
        "superseded source lineage",
    )


def _trajectory_from_reports(
    final_report: Mapping[str, Any], training_report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    epochs = _sequence(final_report.get("epochs"), "final report epochs")
    history = _sequence(training_report.get("history"), "extension training history")
    if len(epochs) != 8 or len(history) != 8:
        _fail("Final selector epochs and extension training history must both contain 8 rows")
    losses: dict[int, Any] = {}
    for index, value in enumerate(history):
        row = _mapping(value, f"extension training history[{index}]")
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch in losses:
            _fail("Extension training history has an invalid or duplicate epoch")
        losses[epoch] = row.get("train_loss")

    result: list[dict[str, Any]] = []
    for index, value in enumerate(epochs):
        row = _mapping(value, f"final report epochs[{index}]")
        epoch = row.get("epoch")
        color = _mapping(row.get("color"), f"final epoch {epoch} color")
        mirror = _mapping(row.get("mirror"), f"final epoch {epoch} mirror")
        result.append(
            {
                "epoch": epoch,
                "optimizer_step": row.get("optimizer_step"),
                "cumulative_microsteps": row.get("cumulative_microsteps"),
                "train_loss": losses.get(epoch),
                "full_teacher_gate_passed": row.get("full_teacher_gate_passed"),
                "color": {
                    "full_vocab_sides": color.get("full_vocab_sides"),
                    "full_vocab_units": color.get("full_vocab_units"),
                    "mean_full_vocab_margin": color.get("mean_full_vocab_margin"),
                    "minimum_full_vocab_margin": color.get("minimum_full_vocab_margin"),
                },
                "mirror": {
                    "full_vocab_sides": mirror.get("full_vocab_sides"),
                    "full_vocab_units": mirror.get("full_vocab_units"),
                    "mean_full_vocab_margin": mirror.get("mean_full_vocab_margin"),
                    "minimum_full_vocab_margin": mirror.get("minimum_full_vocab_margin"),
                },
            }
        )
    return result


def validate_archive(
    summary_path: str | Path,
    *,
    repo_root: str | Path,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    """Validate the sealed V21 archive without inspecting current Git state."""

    root = Path(repo_root).resolve()
    summary_file = Path(summary_path).resolve()
    summary_sha256 = _file_sha256(summary_file, "V21 final summary")
    _require_equal(summary_sha256, EXPECTED_SUMMARY_SHA256, "archive summary SHA-256")
    summary = _load_json(summary_file, "V21 final summary")
    validate_summary_contract(summary)

    if not verify_bound_files:
        return {
            "valid": True,
            "archive_type": summary["archive_type"],
            "summary_sha256": summary_sha256,
            "bound_files_verified": False,
            "current_source_head_checked": False,
            "sealed_source_commit": EXPECTED_SOURCE_COMMIT,
            "decision": EXPECTED_DECISION,
        }

    artifacts = _artifact_index(summary)
    loaded: dict[str, dict[str, Any]] = {}
    for role, artifact in artifacts.items():
        path = _bound_path(root, artifact.get("path"), f"{role}.path")
        digest = _file_sha256(path, role)
        _require_equal(digest, artifact.get("sha256"), f"{role} SHA-256")
        if path.suffix == ".json":
            loaded[role] = _load_json(path, role)

    config = _mapping(summary["config"], "config")
    _require_equal(artifacts["config"].get("path"), config.get("path"), "config artifact path")
    _require_equal(
        artifacts["config"].get("sha256"),
        config.get("file_sha256"),
        "config artifact hash",
    )

    final_report = loaded["final_selector_report"]
    outcome = _mapping(summary["outcome"], "outcome")
    for field in (
        "decision",
        "selected_epoch",
        "conditional_limit_reached",
        "full_teacher_gate_passed",
        "continuation_authorized",
        "greedy_audit_authorized",
        "greedy_audit_forbidden",
        "question_dependent_scene_processing",
    ):
        _require_equal(final_report.get(field), outcome.get(field), f"final report {field}")
    _require_equal(final_report.get("report_only"), True, "final report-only flag")
    _require_equal(final_report.get("model_inference_executed"), False, "model inference flag")
    _require_equal(final_report.get("gemma_model_loaded"), False, "Gemma load flag")
    _require_equal(
        final_report.get("source_provenance"),
        summary.get("sealed_from_source"),
        "final report original source provenance",
    )
    _require_equal(
        final_report.get("config_hash_full"),
        config.get("resolved_config_sha256"),
        "final report resolved config hash",
    )
    _require_equal(
        final_report.get("preflight_contract_sha256"),
        config.get("preflight_contract_sha256"),
        "final report preflight contract hash",
    )

    selected = _mapping(summary["selected_checkpoint"], "selected_checkpoint")
    selected_hashes = _mapping(selected["artifact_hashes"], "selected artifact hashes")
    _require_equal(
        final_report.get("selected_checkpoint_artifact_hashes"),
        selected_hashes,
        "final report selected checkpoint hashes",
    )
    _require_equal(
        final_report.get("selected_checkpoint_metadata_path"),
        selected.get("metadata_path"),
        "final report selected metadata path",
    )
    internal = _mapping(selected["internal_state_hashes"], "selected internal state hashes")
    _require_equal(
        final_report.get("selected_optimizer_state_sha256"),
        internal.get("optimizer_state_sha256"),
        "selected optimizer-state hash",
    )
    _require_equal(
        final_report.get("selected_signed_x_state_sha256"),
        internal.get("signed_x_state_sha256"),
        "selected signed-X state hash",
    )

    for field, hash_field in (
        ("metadata_path", "metadata_sha256"),
        ("adapter_path", "adapter_sha256"),
        ("optimizer_path", "optimizer_sha256"),
    ):
        path = _bound_path(root, selected.get(field), f"selected_checkpoint.{field}")
        _require_equal(
            _file_sha256(path, f"selected checkpoint {field}"),
            selected_hashes.get(hash_field),
            f"selected checkpoint {hash_field}",
        )

    checkpoint_path = _bound_path(root, selected.get("checkpoint_path"), "selected checkpoint path")
    if not checkpoint_path.is_dir():
        _fail(f"Selected checkpoint is not a directory: {checkpoint_path}")
    promotion_path = checkpoint_path / "promotion.json"
    if promotion_path.exists() or promotion_path.is_symlink():
        _fail("V21 was denied promotion, but selected checkpoint/promotion.json now exists")

    selected_metadata_path = _bound_path(
        root, selected.get("metadata_path"), "selected checkpoint metadata"
    )
    selected_metadata = _load_json(selected_metadata_path, "selected checkpoint metadata")
    # The optimizer-state digest is produced by the selector's safe tensor
    # inspector and is intentionally not duplicated in trainer metadata.
    # The remaining state digests are written by both components and must agree.
    for field in (
        "signed_x_scene_residual_state_sha256",
        "frozen_global_scene_residual_state_sha256",
        "frozen_scene_state_sha256",
    ):
        summary_field = (
            "signed_x_state_sha256" if field == "signed_x_scene_residual_state_sha256" else field
        )
        _require_equal(
            selected_metadata.get(field),
            internal.get(summary_field),
            f"selected metadata {field}",
        )

    extension_training = loaded["extension_training_report"]
    _require_equal(
        _trajectory_from_reports(final_report, extension_training),
        summary.get("trajectory"),
        "archived V21 trajectory",
    )
    timings = _mapping(summary["timings"], "timings")
    base_elapsed = loaded["base_training_report"].get("elapsed_seconds")
    extension_elapsed = extension_training.get("elapsed_seconds")
    _require_equal(
        base_elapsed,
        timings.get("base_updates_1_to_4_elapsed_seconds"),
        "base training elapsed seconds",
    )
    _require_equal(
        extension_elapsed,
        timings.get("extension_updates_5_to_8_elapsed_seconds"),
        "extension training elapsed seconds",
    )
    _require_equal(
        base_elapsed + extension_elapsed,
        timings.get("combined_training_process_elapsed_seconds"),
        "combined training elapsed seconds",
    )

    superseded_count = 0
    for lineage_index, value in enumerate(
        _sequence(summary["superseded_lineage"], "superseded_lineage")
    ):
        lineage = _mapping(value, f"superseded_lineage[{lineage_index}]")
        _require_equal(
            lineage.get("disposition"),
            "not_authoritative",
            f"superseded_lineage[{lineage_index}].disposition",
        )
        for artifact_index, artifact_value in enumerate(
            _sequence(lineage.get("artifacts"), "superseded artifacts")
        ):
            artifact = _mapping(
                artifact_value,
                f"superseded_lineage[{lineage_index}].artifacts[{artifact_index}]",
            )
            path = _bound_path(root, artifact.get("path"), "superseded artifact path")
            _require_equal(
                _file_sha256(path, "superseded artifact"),
                artifact.get("sha256"),
                "superseded artifact SHA-256",
            )
            superseded_count += 1

    return {
        "valid": True,
        "archive_type": summary["archive_type"],
        "summary_sha256": summary_sha256,
        "bound_files_verified": True,
        "authoritative_artifact_count": len(artifacts),
        "superseded_artifact_count": superseded_count,
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
    "V21ArchiveViolation",
    "validate_archive",
    "validate_summary_contract",
]

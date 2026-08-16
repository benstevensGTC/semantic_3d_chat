"""One bounded train-only refinement of the V51 query-alpha boundary.

V52 does not train, retrieve, or inspect held-out data.  It authenticates the
exact failed V51 report and reuses V51's already-tested production backend to
reconstruct four predeclared query-only candidates directly from immutable
V47 update 004.  The scene update is identical for every candidate, so all
sixteen complete-scene prefixes must remain bit-identical.

The fixed query-alpha grid brackets the measured boundary between the V50
alpha-2.0 candidate (broad retention passed; one positive side missing) and
the V51 alpha-2.25 candidate (all semantic checks passed; broad retention
missed).  All V49/V51 thresholds remain unchanged.  Every candidate receives
the complete non-greedy train gate, greedy generation is allowed if and only
if that candidate passes its pre-gate, and selection happens only after the
whole grid.  A single optimizer-free checkpoint is atomically published only
for a fully passing candidate after exact source restoration and a clean
file-access audit.

No validation, oracle, deferred-final, selector, promotion, chat, or embodied
access is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import v51_query_alpha_grid as v51

AUTHORIZATION_ID = "v52_query_alpha_refinement"
V51_REPORT = Path("reports/gemma4/metrics/v51_query_alpha_grid.json")
DEFAULT_REPORT = Path("reports/gemma4/metrics/v52_query_alpha_refinement.json")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v52_query_alpha_refinement"
)
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_ROOT / "update_000"
DEFAULT_CONFIG = v51.DEFAULT_CONFIG
SOURCE_CHECKPOINT = v51.SOURCE_CHECKPOINT
PREFIX_REFERENCE_CHECKPOINT = v51.PREFIX_REFERENCE_CHECKPOINT
PROTECTED_REPORT = v51.PROTECTED_REPORT

V51_REPORT_SHA256 = "f99cb6e262247a21708f71632ab58a4ebda42ad750deef69961ff81f534a4e14"
QUERY_ALPHAS = (2.03125, 2.0625, 2.125, 2.1875)
SCENE_ALPHA = 1.0
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SCOPE_LOCK = threading.Lock()


def _alpha_id(value: float) -> str:
    return str(value).replace(".", "p")


CANDIDATE_GRID = tuple(
    {
        "candidate_id": f"guarded_scene_alpha_1p0_query_alpha_{_alpha_id(alpha)}",
        "declared_order": index,
        "scene_alpha": SCENE_ALPHA,
        "query_alpha": alpha,
    }
    for index, alpha in enumerate(QUERY_ALPHAS)
)


@dataclass(frozen=True)
class RefinementPaths:
    predecessor: Path = V51_REPORT
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


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _locked_hash(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} is unavailable or unsafe: {path}")
    if _sha256(path) != expected:
        raise ValueError(f"{field} SHA256 changed")


def _failed_checks(candidate: Mapping[str, Any]) -> set[str]:
    gate = _mapping(candidate.get("non_greedy_pre_gate"), "V51 candidate pre-gate")
    checks = _mapping(gate.get("checks"), "V51 candidate checks")
    return {str(name) for name, passed in checks.items() if passed is not True}


def authenticate_predecessor(
    expected_sha256: str, path: str | Path = V51_REPORT
) -> dict[str, Any]:
    """Authenticate the exact V51 failure that predeclares this refinement."""

    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V52 expected V51 report SHA256 must be lowercase hexadecimal")
    if expected_sha256 != V51_REPORT_SHA256:
        raise ValueError("V52 invocation did not name the pinned V51 report SHA256")
    predecessor = _resolve(path)
    if predecessor != _resolve(V51_REPORT):
        raise ValueError("V52 predecessor path is pinned")
    if predecessor.is_symlink() or not predecessor.is_file():
        raise FileNotFoundError("V52 exact V51 report is unavailable or unsafe")
    payload = predecessor.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise ValueError("V52 V51 report differs from the explicit invocation SHA256")
    report = _mapping(json.loads(payload), "V51 report")
    selection = _mapping(report.get("selection"), "V51 selection")
    checkpoint = _mapping(report.get("checkpoint"), "V51 checkpoint")
    restoration = _mapping(
        report.get("final_source_restoration"), "V51 source restoration"
    )
    access = _mapping(report.get("access_audit"), "V51 access audit")
    invariance = _mapping(
        report.get("query_only_scene_prefix_invariance"), "V51 prefix invariance"
    )
    grid = _mapping(report.get("candidate_grid"), "V51 candidate grid")
    final_gate = _mapping(report.get("final_train_gate"), "V51 final train gate")
    rows = grid.get("candidates")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("V52 requires the complete four-candidate V51 grid")
    by_alpha = {
        float(_mapping(row.get("candidate"), "V51 candidate")["query_alpha"]): row
        for row in rows
    }
    candidate_restorations_exact = all(
        row.get("evaluation_error") is None
        and _mapping(row.get("source_restoration"), "V51 candidate restoration").get(
            "passed"
        )
        is True
        and _mapping(row.get("source_restoration"), "V51 candidate restoration").get(
            "full_tensor_state_sha256"
        )
        == v51._SOURCE_FULL_SHA256
        and _mapping(row.get("source_restoration"), "V51 candidate restoration").get(
            "authorized_surface_state_sha256"
        )
        == v51._SOURCE_AUTHORIZED_SHA256
        and _mapping(row.get("source_restoration"), "V51 candidate restoration").get(
            "frozen_state_sha256"
        )
        == v51._FROZEN_SHA256
        for row in rows
    )
    checks = {
        "artifact": report.get("artifact") == "v51_query_alpha_grid",
        "v51_failed_without_winner": report.get("passed") is False
        and selection.get("winner") is None
        and selection.get("passing_candidate_ids") == [],
        "complete_grid_before_selection": grid.get("evaluated_count") == 4
        and grid.get("complete_fixed_grid_evaluated_before_selection") is True
        and selection.get("performed_after_complete_grid") is True,
        "alpha_2p25_only_failed_broad": 2.25 in by_alpha
        and _failed_checks(by_alpha[2.25]) == {"broad_nll_at_most_v45_maximum"},
        "alpha_1p5_only_failed_positive_side": 1.5 in by_alpha
        and _failed_checks(by_alpha[1.5]) == {"teacher_positive_sides_at_least_35"},
        "query_only_prefix_invariance": invariance.get("passed") is True
        and invariance.get("all_candidate_scene_tensors_bit_identical") is True
        and invariance.get("all_candidate_scene_prefixes_bit_identical") is True
        and invariance.get("all_candidate_prefix_rms_matches_v50_anchor") is True,
        "source_restored": restoration.get("passed") is True,
        "source_restoration_hashes_exact": restoration.get(
            "full_tensor_state_sha256"
        )
        == v51._SOURCE_FULL_SHA256
        and restoration.get("authorized_surface_state_sha256")
        == v51._SOURCE_AUTHORIZED_SHA256
        and restoration.get("frozen_state_sha256") == v51._FROZEN_SHA256,
        "every_candidate_restored_without_error": candidate_restorations_exact,
        "final_gate_complete_without_winner": final_gate.get("passed") is False
        and final_gate.get("grid_complete") is True
        and final_gate.get("candidate_evaluation_complete") is True
        and final_gate.get("evaluation_complete") is True
        and final_gate.get("winner_exists") is False
        and final_gate.get("source_restored_exact") is True
        and final_gate.get("access_audit_passed") is True
        and final_gate.get("execution_errors") == [],
        "access_clean": access.get("passed") is True
        and access.get("training_map_count") == 16
        and access.get("optimizer_file_reads") == []
        and access.get("forbidden_file_accesses") == []
        and access.get("validation_qa_loaded") is False
        and access.get("oracle_loaded") is False
        and access.get("final_test_loaded") is False,
        "no_checkpoint": checkpoint.get("written") is False
        and checkpoint.get("inventory") is None,
        "restricted_actions_absent": report.get("optimizer_constructed_or_loaded")
        is False
        and report.get("optimizer_state_file_opened") is False
        and report.get("optimizer_step_executed") is False
        and report.get("validation_qa_loaded") is False
        and report.get("validation_environment_maps_loaded") is False
        and report.get("oracle_loaded") is False
        and report.get("final_test_scenes_touched") is False
        and report.get("selector_executed") is False
        and report.get("runtime_promotion_executed") is False
        and report.get("chat_promotion_executed") is False
        and report.get("embodied_promotion_executed") is False,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"V51 report does not authorize V52: {failed}")
    return {
        "path": str(V51_REPORT),
        "sha256": observed,
        "authorization_id": AUTHORIZATION_ID,
        "checks": checks,
    }


@contextmanager
def scoped_v51_refinement() -> Iterator[None]:
    """Temporarily parameterize the immutable V51 engine for the V52 grid."""

    if not _SCOPE_LOCK.acquire(blocking=False):
        raise RuntimeError("V52 V51-engine scope is already active")
    overrides: dict[str, object] = {
        "AUTHORIZATION_ID": AUTHORIZATION_ID,
        "_QUERY_ALPHAS": QUERY_ALPHAS,
        "CANDIDATE_GRID": CANDIDATE_GRID,
        "DEFAULT_REPORT": DEFAULT_REPORT,
        "DEFAULT_CHECKPOINT_ROOT": DEFAULT_CHECKPOINT_ROOT,
        "DEFAULT_CHECKPOINT": DEFAULT_CHECKPOINT,
    }
    originals = {name: getattr(v51, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(v51, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(v51, name, value)
        _SCOPE_LOCK.release()


class RealRefinementBackend(v51.RealGridBackend):
    """V51 production backend with V52-only checkpoint provenance."""

    def stage_checkpoint(
        self,
        directory: Path,
        candidate: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from semantic_3d_chat.training.checkpointing import (
            runtime_checkpoint_metadata,
            validate_runtime_checkpoint_metadata,
        )

        state = dict(super().stage_checkpoint(directory, candidate, provenance))
        metadata_path = directory / "metadata.json"
        metadata = dict(
            _mapping(json.loads(metadata_path.read_text(encoding="utf-8")), "metadata")
        )
        stage = dict(
            _mapping(metadata.pop("v51_query_alpha_grid", None), "V51 engine stage")
        )
        stage.update(
            {
                "artifact": AUTHORIZATION_ID,
                "engine_reuse": "v51_query_alpha_grid",
                "authenticated_v51_report": {
                    "path": str(V51_REPORT),
                    "sha256": V51_REPORT_SHA256,
                },
                "fixed_query_alpha_grid": list(QUERY_ALPHAS),
                "scene_alpha_fixed": SCENE_ALPHA,
            }
        )
        metadata[AUTHORIZATION_ID] = stage
        v51._atomic_json(metadata_path, metadata)
        runtime = runtime_checkpoint_metadata(metadata)
        validate_runtime_checkpoint_metadata(runtime)
        v51._atomic_json(directory / "runtime_metadata.json", runtime)
        final_metadata = _mapping(
            json.loads(metadata_path.read_text(encoding="utf-8")),
            "V52 final metadata",
        )
        final_stage = _mapping(
            final_metadata.get(AUTHORIZATION_ID), "V52 final metadata stage"
        )
        final_authorization = _mapping(
            final_stage.get("authorization"), "V52 final authorization"
        )
        provenance_exact = bool(
            "v51_query_alpha_grid" not in final_metadata
            and final_stage.get("artifact") == AUTHORIZATION_ID
            and final_stage.get("winner") == dict(candidate)
            and final_stage.get("fixed_query_alpha_grid") == list(QUERY_ALPHAS)
            and final_stage.get("scene_alpha_fixed") == SCENE_ALPHA
            and _mapping(
                final_stage.get("authenticated_v51_report"),
                "V52 authenticated predecessor",
            ).get("sha256")
            == V51_REPORT_SHA256
            and final_authorization.get("authorization_id") == AUTHORIZATION_ID
            and final_authorization.get("terminal_sha256") == V51_REPORT_SHA256
            and final_authorization.get("source_checkpoint")
            == str(SOURCE_CHECKPOINT)
            and final_authorization.get("source_full_tensor_state_sha256")
            == v51._SOURCE_FULL_SHA256
            and final_stage.get("winner_full_tensor_state_sha256")
            == state["full_tensor_state_sha256"]
            and final_stage.get("winner_authorized_surface_state_sha256")
            == state["authorized_surface_state_sha256"]
            and final_stage.get("frozen_state_sha256") == v51._FROZEN_SHA256
        )
        if not provenance_exact:
            raise RuntimeError("V52 rewritten checkpoint provenance is not exact")
        if json.loads((directory / "runtime_metadata.json").read_text()) != runtime:
            raise RuntimeError("V52 runtime metadata rewrite changed")
        return {
            **state,
            "v52_training_metadata_provenance": provenance_exact,
            "runtime_metadata_exact_sanitization": True,
        }


def _v51_paths(paths: RefinementPaths) -> v51.GridPaths:
    return v51.GridPaths(
        terminal=paths.predecessor,
        report=paths.report,
        checkpoint_root=paths.checkpoint_root,
        config=paths.config,
    )


def execute_refinement_gate(
    *,
    predecessor: Mapping[str, Any],
    backend: v51.GridBackend,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Run the complete fixed V52 grid through the unchanged V51 gate engine."""

    with scoped_v51_refinement():
        report = v51.execute_grid_gate(
            terminal=predecessor,
            backend=backend,
            checkpoint_path=checkpoint_path,
        )
    report["artifact"] = AUTHORIZATION_ID
    report["authorization"] = {
        "predecessor_path": predecessor["path"],
        "predecessor_sha256": predecessor["sha256"],
        "authorization_id": AUTHORIZATION_ID,
        "checks": dict(_mapping(predecessor.get("checks"), "predecessor checks")),
    }
    report["refinement"] = {
        "schema_version": 1,
        "source": "immutable_v47_update_004",
        "direction": "guarded_both_sign",
        "gradient_probe_count": 3,
        "new_gradient_probes": 0,
        "optimizer_constructed_or_loaded": False,
        "scene_alpha_fixed": SCENE_ALPHA,
        "query_alpha_grid_declared_order": list(QUERY_ALPHAS),
        "complete_grid_required_before_selection": True,
        "v51_engine_reused_without_persistent_mutation": True,
    }
    report["checkpoint"]["root"] = str(DEFAULT_CHECKPOINT_ROOT)
    return report


def _resolved_paths(paths: RefinementPaths | None) -> RefinementPaths:
    selected = RefinementPaths() if paths is None else paths
    resolved = RefinementPaths(
        predecessor=_resolve(selected.predecessor),
        report=_resolve(selected.report),
        checkpoint_root=_resolve(selected.checkpoint_root),
        config=_resolve(selected.config),
    )
    expected = RefinementPaths(
        predecessor=_resolve(V51_REPORT),
        report=_resolve(DEFAULT_REPORT),
        checkpoint_root=_resolve(DEFAULT_CHECKPOINT_ROOT),
        config=_resolve(DEFAULT_CONFIG),
    )
    if resolved != expected:
        raise ValueError("V52 predecessor, report, checkpoint, and config paths are pinned")
    return resolved


def _publish_report_or_rollback(
    *, report_path: Path, checkpoint_root: Path, report: Mapping[str, Any]
) -> None:
    """Publish a completed report or remove its newly-created checkpoint.

    The checkpoint root is required to be absent before V52 starts.  It is
    therefore safe and necessary to remove that exact root if serialization or
    atomic report publication fails after the inherited engine publishes a
    winner.  This prevents an unaudited orphan from looking selectable.
    """

    checkpoint = _mapping(report.get("checkpoint"), "V52 checkpoint report")
    written = checkpoint.get("written") is True
    if report.get("artifact") != AUTHORIZATION_ID:
        raise ValueError("V52 completed report artifact changed")
    if _mapping(report.get("authorization"), "V52 report authorization").get(
        "predecessor_sha256"
    ) != V51_REPORT_SHA256:
        raise ValueError("V52 completed report predecessor changed")
    if _mapping(report.get("refinement"), "V52 refinement").get(
        "query_alpha_grid_declared_order"
    ) != list(QUERY_ALPHAS):
        raise ValueError("V52 completed report grid changed")
    if written and not (checkpoint_root / "update_000").is_dir():
        raise RuntimeError("V52 report claims a checkpoint that is absent")
    if not written and checkpoint_root.exists():
        raise RuntimeError("V52 failed report left an unexpected checkpoint root")
    try:
        # Force strict JSON validation before the report filesystem mutation.
        json.dumps(report, sort_keys=True, allow_nan=False)
        v51._atomic_json(report_path, report)
    except BaseException:
        if written and checkpoint_root.is_dir() and not report_path.exists():
            shutil.rmtree(checkpoint_root)
        raise


def preflight(
    *, expected_v51_report_sha256: str, paths: RefinementPaths | None = None
) -> dict[str, Any]:
    """Authenticate V52 inputs without loading Gemma, maps, or QA records."""

    resolved = _resolved_paths(paths)
    if resolved.report.is_symlink() or resolved.report.exists():
        raise FileExistsError("V52 report is one-shot and already exists")
    if resolved.checkpoint_root.is_symlink() or resolved.checkpoint_root.exists():
        raise FileExistsError("V52 checkpoint root must be absent")
    predecessor = authenticate_predecessor(
        expected_v51_report_sha256, resolved.predecessor
    )
    _locked_hash(resolved.config, v51._CONFIG_SHA256, "V52 config")
    _locked_hash(_resolve(PROTECTED_REPORT), v51._PROTECTED_REPORT_SHA256, "protected report")
    source = _resolve(SOURCE_CHECKPOINT)
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError("V52 source checkpoint is unavailable")
    if sorted(path.name for path in source.iterdir()) != sorted(v51._SOURCE_FILES):
        raise ValueError("V52 source checkpoint inventory changed")
    for name, digest in v51._SOURCE_FILES.items():
        if name != "optimizer.pt":
            _locked_hash(source / name, digest, f"V52 source {name}")
    prefix = _resolve(PREFIX_REFERENCE_CHECKPOINT)
    if prefix.is_symlink() or not prefix.is_dir():
        raise FileNotFoundError("V52 prefix reference checkpoint is unavailable")
    if sorted(path.name for path in prefix.iterdir()) != sorted(v51._PREFIX_REFERENCE_FILES):
        raise ValueError("V52 prefix reference checkpoint inventory changed")
    for name, digest in v51._PREFIX_REFERENCE_FILES.items():
        _locked_hash(prefix / name, digest, f"V52 prefix reference {name}")
    return {
        "schema_version": 1,
        "artifact": f"{AUTHORIZATION_ID}_preflight",
        "passed": True,
        "predecessor": predecessor,
        "candidate_grid": [dict(value) for value in CANDIDATE_GRID],
        "candidate_count": len(CANDIDATE_GRID),
        "checkpoint": str(DEFAULT_CHECKPOINT),
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
    }


def run_grid(
    *,
    expected_v51_report_sha256: str,
    paths: RefinementPaths | None = None,
    backend_factory: Callable[[Mapping[str, Any], v51.GridPaths], v51.GridBackend]
    | None = None,
) -> dict[str, Any]:
    """Run the exact one-shot V52 boundary refinement."""

    resolved = _resolved_paths(paths)
    if resolved.report.is_symlink() or resolved.report.exists():
        raise FileExistsError("V52 report is one-shot and will not be overwritten")
    if resolved.checkpoint_root.is_symlink() or resolved.checkpoint_root.exists():
        raise FileExistsError("V52 checkpoint root must be absent")
    _locked_hash(resolved.config, v51._CONFIG_SHA256, "V52 config")
    predecessor = authenticate_predecessor(
        expected_v51_report_sha256, resolved.predecessor
    )
    engine_paths = _v51_paths(resolved)
    factory = RealRefinementBackend if backend_factory is None else backend_factory
    with scoped_v51_refinement():
        backend = factory(predecessor, engine_paths)
    report = execute_refinement_gate(
        predecessor=predecessor,
        backend=backend,
        checkpoint_path=resolved.checkpoint_root / "update_000",
    )
    _publish_report_or_rollback(
        report_path=resolved.report,
        checkpoint_root=resolved.checkpoint_root,
        report=report,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-v51-report-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--predecessor", type=Path, default=V51_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    paths = RefinementPaths(
        predecessor=args.predecessor,
        report=args.report,
        checkpoint_root=args.checkpoint_root,
        config=args.config,
    )
    if args.preflight:
        result = preflight(
            expected_v51_report_sha256=args.expected_v51_report_sha256, paths=paths
        )
        summary = {
            "artifact": result["artifact"],
            "passed": result["passed"],
            "predecessor_sha256": result["predecessor"]["sha256"],
            "candidate_count": result["candidate_count"],
            "model_loaded": False,
            "qa_loaded": False,
            "maps_loaded": False,
        }
    else:
        result = run_grid(
            expected_v51_report_sha256=args.expected_v51_report_sha256, paths=paths
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
    "QUERY_ALPHAS",
    "RealRefinementBackend",
    "RefinementPaths",
    "authenticate_predecessor",
    "execute_refinement_gate",
    "main",
    "preflight",
    "run_grid",
    "scoped_v51_refinement",
]

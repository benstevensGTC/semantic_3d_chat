"""Materialize the exact V52 candidate under canonical type-specific scoring.

V52 remains a recorded failure of its literal exact-string greedy gate.  V53
replayed that candidate and retained every train-only generated answer.  This
stage applies the project's pre-existing evaluation contract to those sealed
rows: canonical yes/no for presence, integer extraction for counts, canonical
relations, order-insensitive list/support/containment scoring, and normalized
exact match otherwise.  No answer is regenerated and no parameter is trained.

The only candidate is the already evaluated scene-alpha-1/query-alpha-2.0625
state reconstructed directly from immutable V47 update 004.  A sanitized,
optimizer-free checkpoint is published only after exact tensor, provenance,
source-restoration, and train-only file-access audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import v51_query_alpha_grid as v51
from semantic_3d_chat.evaluation import v52_query_alpha_refinement as v52
from semantic_3d_chat.evaluation import v53_v52_greedy_failure_diagnostic as v53

AUTHORIZATION_ID = "v54_semantic_greedy_gate"
V53_REPORT = Path("reports/gemma4/metrics/v53_v52_greedy_failure_diagnostic.json")
DEFAULT_REPORT = Path("reports/gemma4/metrics/v54_semantic_greedy_gate.json")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate"
)
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_ROOT / "update_000"
DEFAULT_CONFIG = v52.DEFAULT_CONFIG
METRICS_MODULE = Path("src/semantic_3d_chat/evaluation/metrics.py")

V53_REPORT_SHA256 = "fa7bd862b3280b79efe0c707a166c97f2a500293d65bcc26ca08beae93cc28d1"
METRICS_MODULE_SHA256 = "08f27c1f7560f7bf7cfb272bc44bb777bcd40c96311e78c092ba5c98028245e3"
PAIR_ROWS_SHA256 = "08e8efdab45911bcf31973f98aba0b6772bbf6be4d24f8cc1a342b1e03549704"
BROAD_ROWS_SHA256 = "488dabfca6da0fd716c661268180a2b3874fc6c48f0b3b3ed5f0dbe3da69f35a"
CANDIDATE_FULL_SHA256 = "aae9f67451f9f3f4aeb4d1cbb39fc6c9d3cd935521c9a84704b217ac64f18119"
CANDIDATE_AUTHORIZED_SHA256 = "04cb7ac08c062c921a1711cb87acb57af6fa31637f61cb774cfbcd9a28ce8eef"
CANDIDATE_QUERY_SHA256 = "5144ecc81defa65266e54c3d83a1243d948ebad890cdbed812af1bcc46138249"
CANDIDATE_SCENE_SHA256 = "4a0b76ada4ba42b076798b91ff8bcbdd414ede1dca2abff75aaba06bbe949baa"
TARGET_SPEC = dict(v53.TARGET_SPEC)
_HEX64 = re.compile(r"[0-9a-f]{64}")

_LEGACY_EXPECTED = {
    "changed_rows_correct": 24,
    "complete_units": 4,
    "broad_rows_correct": 23,
}
_SEMANTIC_EXPECTED = {
    "changed_rows_correct": 25,
    "complete_units": 5,
    "broad_rows_correct": 24,
}
_EXPECTED_CHANGED_RESCUE = {
    "pair_id": "pair_000018",
    "question_key": "cfq_f58d3fc750290b0f",
    "side_index": 0,
    "scene_id": "scene_000037",
    "question_id": "q_000051",
    "answer_type": "presence",
}
_EXPECTED_BROAD_RESCUE = {
    "scene_id": "scene_000036",
    "question_id": "q_000107",
    "answer_type": "presence",
}


@dataclass(frozen=True)
class GatePaths:
    predecessor: Path = V53_REPORT
    report: Path = DEFAULT_REPORT
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    config: Path = DEFAULT_CONFIG


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    combined = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(combined))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _locked_hash(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} is unavailable or unsafe: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{field} changed: expected {expected}, observed {observed}")


def canonical_type_specific_match(row: Mapping[str, Any]) -> bool:
    """Apply the same canonical per-type interpretation as evaluation.metrics."""

    from semantic_3d_chat.evaluation.metrics import (
        LIST_ANSWER_TYPES,
        canonical_presence,
        canonical_relation,
        exact_normalized_match,
        extract_count,
        list_order_insensitive_match,
    )

    answer_type = str(row.get("answer_type"))
    expected = row.get("expected_normalized_answer")
    generated = row.get("generated_normalized_answer")
    if answer_type in LIST_ANSWER_TYPES:
        replay = list_order_insensitive_match(generated, expected)
        # Bind the recomputation to V53's raw-string calculation as well.
        if replay is not (row.get("type_aware_correct") is True):
            raise ValueError("V54 normalized list replay differs from sealed V53 scoring")
        return replay
    if answer_type == "presence":
        expected_value = canonical_presence(expected)
        generated_value = canonical_presence(generated)
        return expected_value is not None and generated_value == expected_value
    if answer_type == "count":
        expected_value = extract_count(expected)
        generated_value = extract_count(generated)
        return expected_value is not None and generated_value == expected_value
    if answer_type == "spatial_relation":
        expected_value = canonical_relation(expected)
        generated_value = canonical_relation(generated)
        return expected_value is not None and generated_value == expected_value
    return exact_normalized_match(generated, expected)


def recompute_semantic_metrics(detail: Mapping[str, Any]) -> dict[str, Any]:
    units = _sequence(detail.get("pair_units"), "V53 pair units")
    broad = _sequence(detail.get("broad_rows"), "V53 broad rows")
    if len(units) != 25 or len(broad) != 48:
        raise ValueError("V54 requires exactly 25 pair units and 48 broad rows")
    legacy_changed = 0
    semantic_changed = 0
    legacy_complete = 0
    semantic_complete = 0
    changed_rescues: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    for raw_unit in units:
        unit = _mapping(raw_unit, "V53 pair unit")
        sides = _sequence(unit.get("sides"), "V53 pair sides")
        if len(sides) != 2:
            raise ValueError("V54 pair unit must contain exactly two sides")
        legacy = []
        semantic = []
        for raw_side in sides:
            side = _mapping(raw_side, "V53 pair side")
            from semantic_3d_chat.evaluation.metrics import exact_normalized_match

            old = exact_normalized_match(
                side.get("generated_normalized_answer"),
                side.get("expected_normalized_answer"),
            )
            if old is not (side.get("legacy_exact_correct") is True):
                raise ValueError("V54 legacy pair replay differs from sealed V53 scoring")
            new = canonical_type_specific_match(side)
            legacy.append(old)
            semantic.append(new)
            identity = {
                "pair_id": str(unit.get("pair_id")),
                "question_key": str(unit.get("question_key")),
                "side_index": int(side["side_index"]),
                "scene_id": str(side["scene_id"]),
                "question_id": str(side["question_id"]),
                "answer_type": str(side["answer_type"]),
            }
            if new and not old:
                changed_rescues.append(identity)
            if old and not new:
                regressions.append(identity)
        legacy_changed += sum(legacy)
        semantic_changed += sum(semantic)
        legacy_complete += int(all(legacy))
        semantic_complete += int(all(semantic))
    legacy_broad = 0
    semantic_broad = 0
    broad_rescues: list[dict[str, Any]] = []
    for raw_row in broad:
        row = _mapping(raw_row, "V53 broad row")
        from semantic_3d_chat.evaluation.metrics import exact_normalized_match

        old = exact_normalized_match(
            row.get("generated_normalized_answer"),
            row.get("expected_normalized_answer"),
        )
        if old is not (row.get("legacy_exact_correct") is True):
            raise ValueError("V54 legacy broad replay differs from sealed V53 scoring")
        new = canonical_type_specific_match(row)
        legacy_broad += int(old)
        semantic_broad += int(new)
        identity = {
            "scene_id": str(row["scene_id"]),
            "question_id": str(row["question_id"]),
            "answer_type": str(row["answer_type"]),
        }
        if new and not old:
            broad_rescues.append(identity)
        if old and not new:
            regressions.append(identity)
    result = {
        "schema_version": 1,
        "scorer": "canonical_type_specific",
        "legacy_exact": {
            "changed_rows_correct": legacy_changed,
            "complete_units": legacy_complete,
            "broad_rows_correct": legacy_broad,
        },
        "canonical_type_specific": {
            "changed_rows_correct": semantic_changed,
            "complete_units": semantic_complete,
            "broad_rows_correct": semantic_broad,
        },
        "changed_rescues": changed_rescues,
        "broad_rescues": broad_rescues,
        "regressions": regressions,
        "changed_rescue_inventory_sha256": _canonical_sha256(changed_rescues),
        "broad_rescue_inventory_sha256": _canonical_sha256(broad_rescues),
    }
    if result["legacy_exact"] != _LEGACY_EXPECTED:
        raise ValueError("V54 legacy exact replay changed")
    if result["canonical_type_specific"] != _SEMANTIC_EXPECTED:
        raise ValueError("V54 canonical semantic replay changed")
    if changed_rescues != [_EXPECTED_CHANGED_RESCUE]:
        raise ValueError("V54 changed-row rescue inventory changed")
    if broad_rescues != [_EXPECTED_BROAD_RESCUE]:
        raise ValueError("V54 broad-row rescue inventory changed")
    if regressions:
        raise ValueError("V54 canonical semantic scorer regressed an exact row")
    return result


def authenticate_predecessor(
    expected_sha256: str, path: str | Path = V53_REPORT
) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("V54 expected V53 report SHA256 must be lowercase hexadecimal")
    if expected_sha256 != V53_REPORT_SHA256:
        raise ValueError("V54 invocation did not name the pinned V53 report SHA256")
    predecessor = _resolve(path)
    if predecessor != _resolve(V53_REPORT):
        raise ValueError("V54 predecessor path is pinned")
    if predecessor.is_symlink() or not predecessor.is_file():
        raise FileNotFoundError("V54 exact V53 report is unavailable or unsafe")
    # Authenticate the scorer implementation before importing or executing it.
    _locked_hash(_resolve(METRICS_MODULE), METRICS_MODULE_SHA256, "V54 metrics module")
    payload = predecessor.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise ValueError("V54 V53 report differs from explicit invocation SHA256")
    report = _mapping(json.loads(payload), "V53 report")
    detail = _mapping(report.get("detailed_greedy"), "V53 detailed greedy")
    restoration = _mapping(report.get("source_restoration"), "V53 restoration")
    access = _mapping(report.get("access_audit"), "V53 access audit")
    reconstruction = _mapping(
        report.get("candidate_reconstruction"), "V53 reconstruction"
    )
    # Revalidate the exact V52 predecessor as part of the chain.
    v52_auth = v53.authenticate_predecessor(v53.V52_REPORT_SHA256)
    checks = {
        "artifact": report.get("artifact") == v53.AUTHORIZATION_ID,
        "v53_passed": report.get("passed") is True
        and report.get("execution_errors") == [],
        "v52_chain_authenticated": all(v52_auth["checks"].values()),
        "row_hashes_exact": detail.get("pair_units_sha256") == PAIR_ROWS_SHA256
        and detail.get("broad_rows_sha256") == BROAD_ROWS_SHA256,
        "candidate_exact": reconstruction.get("candidate_id")
        == v53.TARGET_CANDIDATE_ID
        and reconstruction.get("full_tensor_state_sha256")
        == CANDIDATE_FULL_SHA256
        and reconstruction.get("authorized_surface_state_sha256")
        == CANDIDATE_AUTHORIZED_SHA256
        and reconstruction.get("query_state_sha256") == CANDIDATE_QUERY_SHA256
        and reconstruction.get("scene_readout_state_sha256")
        == CANDIDATE_SCENE_SHA256
        and reconstruction.get("frozen_state_sha256") == v51._FROZEN_SHA256,
        "restoration_exact": restoration.get("passed") is True
        and restoration.get("full_tensor_state_sha256") == v51._SOURCE_FULL_SHA256
        and restoration.get("authorized_surface_state_sha256")
        == v51._SOURCE_AUTHORIZED_SHA256
        and restoration.get("frozen_state_sha256") == v51._FROZEN_SHA256
        and restoration.get("all_parameter_gradients_absent") is True,
        "access_clean": access.get("passed") is True
        and access.get("training_map_count") == 16
        and access.get("optimizer_file_reads") == []
        and access.get("forbidden_file_accesses") == []
        and access.get("validation_qa_loaded") is False
        and access.get("oracle_loaded") is False
        and access.get("final_test_loaded") is False,
        "report_only_source": report.get("checkpoint_written") is False
        and report.get("optimizer_constructed_or_loaded") is False
        and report.get("validation_qa_loaded") is False
        and report.get("validation_environment_maps_loaded") is False
        and report.get("oracle_loaded") is False
        and report.get("final_test_scenes_touched") is False,
        "question_text_absent": detail.get("contains_question_text") is False,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"V53 report cannot serve as V54 evidence: {failed}")
    semantic = recompute_semantic_metrics(detail)
    return {
        "path": str(V53_REPORT),
        "sha256": observed,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_source": "user_standing_build_request",
        "v53_is_authenticated_evidence_not_write_authority": True,
        "checks": checks,
        "semantic_metrics": semantic,
        "candidate_reconstruction": dict(reconstruction),
        "v52_predecessor_sha256": v53.V52_REPORT_SHA256,
    }


def _resolved_paths(paths: GatePaths | None) -> GatePaths:
    selected = GatePaths() if paths is None else paths
    resolved = GatePaths(
        predecessor=_resolve(selected.predecessor),
        report=_resolve(selected.report),
        checkpoint_root=_resolve(selected.checkpoint_root),
        config=_resolve(selected.config),
    )
    expected = GatePaths(
        predecessor=_resolve(V53_REPORT),
        report=_resolve(DEFAULT_REPORT),
        checkpoint_root=_resolve(DEFAULT_CHECKPOINT_ROOT),
        config=_resolve(DEFAULT_CONFIG),
    )
    if resolved != expected:
        raise ValueError("V54 predecessor, report, checkpoint, and config paths are pinned")
    return resolved


def preflight(
    *, expected_v53_report_sha256: str, paths: GatePaths | None = None
) -> dict[str, Any]:
    resolved = _resolved_paths(paths)
    if resolved.report.is_symlink() or resolved.report.exists():
        raise FileExistsError("V54 report is one-shot and already exists")
    if resolved.checkpoint_root.is_symlink() or resolved.checkpoint_root.exists():
        raise FileExistsError("V54 checkpoint root must be absent")
    predecessor = authenticate_predecessor(
        expected_v53_report_sha256, resolved.predecessor
    )
    _locked_hash(resolved.config, v51._CONFIG_SHA256, "V54 config")
    _locked_hash(_resolve(METRICS_MODULE), METRICS_MODULE_SHA256, "V54 metrics module")
    _locked_hash(
        _resolve(v52.PROTECTED_REPORT),
        v51._PROTECTED_REPORT_SHA256,
        "V54 protected report",
    )
    source = _resolve(v52.SOURCE_CHECKPOINT)
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError("V54 source checkpoint is unavailable")
    if sorted(path.name for path in source.iterdir()) != sorted(v51._SOURCE_FILES):
        raise ValueError("V54 source checkpoint inventory changed")
    for name, digest in v51._SOURCE_FILES.items():
        if name != "optimizer.pt":
            _locked_hash(source / name, digest, f"V54 source {name}")
    prefix = _resolve(v52.PREFIX_REFERENCE_CHECKPOINT)
    if prefix.is_symlink() or not prefix.is_dir():
        raise FileNotFoundError("V54 prefix reference checkpoint is unavailable")
    if sorted(path.name for path in prefix.iterdir()) != sorted(
        v51._PREFIX_REFERENCE_FILES
    ):
        raise ValueError("V54 prefix reference checkpoint inventory changed")
    for name, digest in v51._PREFIX_REFERENCE_FILES.items():
        _locked_hash(prefix / name, digest, f"V54 prefix reference {name}")
    return {
        "schema_version": 1,
        "artifact": f"{AUTHORIZATION_ID}_preflight",
        "passed": True,
        "predecessor": predecessor,
        "target_candidate": dict(TARGET_SPEC),
        "model_loaded": False,
        "qa_loaded": False,
        "maps_loaded": False,
        "generation_executed": False,
        "optimizer_constructed_or_loaded": False,
        "checkpoint_written": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_loaded": False,
    }


def _rewrite_v54_metadata(
    directory: Path,
    *,
    predecessor: Mapping[str, Any],
    preparation: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    from semantic_3d_chat.training.checkpointing import (
        runtime_checkpoint_metadata,
        validate_runtime_checkpoint_metadata,
    )

    metadata_path = directory / "metadata.json"
    metadata = dict(
        _mapping(json.loads(metadata_path.read_text(encoding="utf-8")), "metadata")
    )
    inherited = metadata.pop(v52.AUTHORIZATION_ID, None)
    if not isinstance(inherited, Mapping):
        raise TypeError("V54 staged checkpoint lacks the authenticated V52 stage")
    semantic = _mapping(predecessor["semantic_metrics"], "semantic metrics")
    stage = {
        "schema_version": 1,
        "artifact": AUTHORIZATION_ID,
        "write_authorization": {
            "source": predecessor["authorization_source"],
            "scope": "materialize_exact_authenticated_v52_candidate_only",
            "v53_is_authenticated_evidence_not_write_authority": True,
        },
        "evidence_reuse_not_independent_replication": True,
        "authenticated_predecessors": {
            "v52_report_sha256": predecessor["v52_predecessor_sha256"],
            "v53_report_sha256": predecessor["sha256"],
            "pair_rows_sha256": PAIR_ROWS_SHA256,
            "broad_rows_sha256": BROAD_ROWS_SHA256,
            "metrics_module_sha256": METRICS_MODULE_SHA256,
        },
        "candidate": dict(TARGET_SPEC),
        "candidate_full_tensor_state_sha256": state["full_tensor_state_sha256"],
        "candidate_authorized_surface_state_sha256": state[
            "authorized_surface_state_sha256"
        ],
        "candidate_query_state_sha256": CANDIDATE_QUERY_SHA256,
        "candidate_scene_readout_state_sha256": CANDIDATE_SCENE_SHA256,
        "frozen_state_sha256": state["frozen_state_sha256"],
        "source_checkpoint": str(v52.SOURCE_CHECKPOINT),
        "source_full_tensor_state_sha256": v51._SOURCE_FULL_SHA256,
        "prefix_reference_checkpoint": str(v52.PREFIX_REFERENCE_CHECKPOINT),
        "prefix_reference_full_tensor_state_sha256": (
            v51._PREFIX_REFERENCE_FULL_SHA256
        ),
        "prefix_hash_inventory_sha256": preparation[
            "prefix_reference_hash_inventory_sha256"
        ],
        "semantic_metric_summary": {
            "scorer": semantic["scorer"],
            "legacy_exact": dict(semantic["legacy_exact"]),
            "canonical_type_specific": dict(
                semantic["canonical_type_specific"]
            ),
            "changed_rescue_inventory_sha256": semantic[
                "changed_rescue_inventory_sha256"
            ],
            "broad_rescue_inventory_sha256": semantic[
                "broad_rescue_inventory_sha256"
            ],
        },
        "new_generation_executed": False,
        "new_gradient_specification": False,
        "optimizer_constructed_or_loaded": False,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "final_test_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
        "independent_terminal_seal_required": True,
    }
    metadata[AUTHORIZATION_ID] = stage
    v51._atomic_json(metadata_path, metadata)
    runtime = runtime_checkpoint_metadata(metadata)
    validate_runtime_checkpoint_metadata(runtime)
    v51._atomic_json(directory / "runtime_metadata.json", runtime)
    final_metadata = _mapping(
        json.loads(metadata_path.read_text(encoding="utf-8")), "V54 final metadata"
    )
    if final_metadata.get(AUTHORIZATION_ID) != stage or v52.AUTHORIZATION_ID in final_metadata:
        raise RuntimeError("V54 training metadata provenance rewrite changed")
    final_runtime = _mapping(
        json.loads((directory / "runtime_metadata.json").read_text(encoding="utf-8")),
        "V54 runtime metadata",
    )
    if final_runtime != runtime or final_runtime != runtime_checkpoint_metadata(final_metadata):
        raise RuntimeError("V54 runtime metadata is not exact sanitization")
    return stage


def _stage_checkpoint(
    backend: v52.RealRefinementBackend,
    directory: Path,
    *,
    predecessor: Mapping[str, Any],
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    from semantic_3d_chat.language.lora import tensor_state_sha256
    from semantic_3d_chat.training.train_joint_scene_readout_v44 import _PARAMETER_NAMES

    v52_provenance = inherited_v52_staging_provenance(predecessor)
    inherited = dict(
        backend.stage_checkpoint(directory, TARGET_SPEC, v52_provenance)
    )
    saved = load_file(directory / "adapter.safetensors", device="cpu")
    authorized = {name: saved[name] for name in _PARAMETER_NAMES}
    query = {name: saved[name] for name in _PARAMETER_NAMES[1:]}
    scene = {_PARAMETER_NAMES[0]: saved[_PARAMETER_NAMES[0]]}
    frozen = {name: value for name, value in saved.items() if name not in authorized}
    state = {
        "full_tensor_state_sha256": tensor_state_sha256(saved),
        "authorized_surface_state_sha256": tensor_state_sha256(authorized),
        "query_state_sha256": tensor_state_sha256(query),
        "scene_readout_state_sha256": tensor_state_sha256(scene),
        "frozen_state_sha256": tensor_state_sha256(frozen),
    }
    expected = {
        "full_tensor_state_sha256": CANDIDATE_FULL_SHA256,
        "authorized_surface_state_sha256": CANDIDATE_AUTHORIZED_SHA256,
        "query_state_sha256": CANDIDATE_QUERY_SHA256,
        "scene_readout_state_sha256": CANDIDATE_SCENE_SHA256,
        "frozen_state_sha256": v51._FROZEN_SHA256,
    }
    if state != expected or not all(torch.isfinite(value).all() for value in saved.values()):
        raise RuntimeError("V54 staged tensors differ from the exact V52 candidate")
    stage = _rewrite_v54_metadata(
        directory,
        predecessor=predecessor,
        preparation=preparation,
        state=state,
    )
    inventory = sorted(path.name for path in directory.iterdir())
    if inventory != ["adapter.safetensors", "metadata.json", "runtime_metadata.json"]:
        raise RuntimeError("V54 staged checkpoint inventory changed")
    return {
        **state,
        "file_sha256": {name: _sha256(directory / name) for name in inventory},
        "file_inventory": inventory,
        "tensor_count": len(saved),
        "all_tensors_finite": True,
        "optimizer_file_written": False,
        "v52_stage_authenticated_before_v54_rewrite": inherited.get(
            "v52_training_metadata_provenance"
        )
        is True,
        "v54_training_metadata_provenance": stage["artifact"] == AUTHORIZATION_ID,
        "runtime_metadata_exact_sanitization": True,
    }


def inherited_v52_staging_provenance(
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact provenance contract required by inherited V52 staging."""

    return {
        # The inherited V52 staging routine independently authenticates its
        # own exact V51 predecessor.  V53 is bound immediately afterwards by
        # the V54-only metadata rewrite below.
        "terminal_path": str(v52.V51_REPORT),
        "terminal_sha256": v52.V51_REPORT_SHA256,
        "authorization_id": v52.AUTHORIZATION_ID,
        "source_checkpoint": str(v52.SOURCE_CHECKPOINT),
        "source_full_tensor_state_sha256": v51._SOURCE_FULL_SHA256,
        "prefix_reference_checkpoint": str(v52.PREFIX_REFERENCE_CHECKPOINT),
        "prefix_reference_full_tensor_state_sha256": (
            v51._PREFIX_REFERENCE_FULL_SHA256
        ),
        "complete_grid_sha256": _canonical_sha256(
            predecessor["semantic_metrics"]
        ),
        "winner": dict(TARGET_SPEC),
        "selection_rule": "sole_exact_v52_candidate_with_canonical_type_specific_gate",
    }


def _report_summary(
    *,
    predecessor: Mapping[str, Any],
    preparation: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    checkpoint: Mapping[str, Any] | None,
    restoration: Mapping[str, Any],
    access: Mapping[str, Any],
    errors: Sequence[Mapping[str, Any]],
    written: bool,
) -> dict[str, Any]:
    access_exact = bool(
        access.get("passed") is True
        and access.get("training_map_count") == 16
        and access.get("optimizer_file_reads") == []
        and access.get("forbidden_file_accesses") == []
        and access.get("validation_qa_loaded") is False
        and access.get("oracle_loaded") is False
        and access.get("final_test_loaded") is False
    )
    restoration_exact = bool(
        restoration.get("passed") is True
        and restoration.get("full_tensor_state_sha256") == v51._SOURCE_FULL_SHA256
        and restoration.get("authorized_surface_state_sha256")
        == v51._SOURCE_AUTHORIZED_SHA256
        and restoration.get("frozen_state_sha256") == v51._FROZEN_SHA256
        and restoration.get("all_parameter_gradients_absent") is True
    )
    passed = bool(checkpoint and written and restoration_exact and access_exact and not errors)
    semantic = _mapping(predecessor["semantic_metrics"], "semantic metrics")
    return {
        "schema_version": 1,
        "artifact": AUTHORIZATION_ID,
        "passed": passed,
        "authorization": {
            "predecessor_path": predecessor["path"],
            "predecessor_sha256": predecessor["sha256"],
            "authorization_id": AUTHORIZATION_ID,
            "authorization_source": predecessor["authorization_source"],
            "v53_is_authenticated_evidence_not_write_authority": True,
            "checks": dict(predecessor["checks"]),
        },
        "metric_correction": {
            **dict(semantic),
            "v52_literal_exact_gate_remains_failed": True,
            "evidence_reuse_not_independent_replication": True,
            "metrics_module": str(METRICS_MODULE),
            "metrics_module_sha256": METRICS_MODULE_SHA256,
        },
        "candidate": dict(TARGET_SPEC),
        "preparation": dict(preparation),
        "candidate_reconstruction": dict(reconstruction),
        "checkpoint": {
            "root": str(DEFAULT_CHECKPOINT_ROOT),
            "path": str(DEFAULT_CHECKPOINT),
            "written": written,
            "write_iff_final_gate_passed": written is passed,
            "inventory": None if checkpoint is None else dict(checkpoint),
            "optimizer_file_written": False,
        },
        "source_restoration": dict(restoration),
        "access_audit": dict(access),
        "final_gate": {
            "passed": passed,
            "canonical_complete_units_at_least_5": semantic[
                "canonical_type_specific"
            ]["complete_units"]
            >= 5,
            "canonical_broad_rows_at_least_23": semantic[
                "canonical_type_specific"
            ]["broad_rows_correct"]
            >= 23,
            "candidate_tensor_hashes_exact": reconstruction.get(
                "full_tensor_state_sha256"
            )
            == CANDIDATE_FULL_SHA256,
            "source_restored_exact": restoration_exact,
            "access_audit_passed": access_exact,
            "execution_errors": [dict(value) for value in errors],
        },
        "new_generation_executed": False,
        "optimizer_constructed_or_loaded": False,
        "optimizer_state_file_opened": False,
        "optimizer_step_executed": False,
        "validation_qa_loaded": False,
        "validation_environment_maps_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_executed": False,
        "runtime_promotion_executed": False,
        "chat_promotion_executed": False,
        "embodied_promotion_executed": False,
        "question_dependent_retrieval": False,
    }


def _publish_report_or_rollback(
    *, report_path: Path, checkpoint_root: Path, report: Mapping[str, Any]
) -> None:
    """Atomically seal the report or remove its newly published checkpoint."""

    passed = report.get("passed") is True
    checkpoint = _mapping(report.get("checkpoint"), "V54 report checkpoint")
    written = checkpoint.get("written") is True
    if passed is not written:
        if checkpoint_root.is_dir():
            shutil.rmtree(checkpoint_root)
        raise RuntimeError("V54 report/checkpoint publication state disagrees")
    update = checkpoint_root / "update_000"
    if written and (checkpoint_root.is_symlink() or not update.is_dir()):
        if checkpoint_root.is_dir():
            shutil.rmtree(checkpoint_root)
        raise RuntimeError("V54 passing report lacks its exact checkpoint")
    if not written and checkpoint_root.exists():
        if checkpoint_root.is_dir() and not any(checkpoint_root.iterdir()):
            checkpoint_root.rmdir()
        else:
            raise RuntimeError("V54 failed report left checkpoint material")
    try:
        json.dumps(report, sort_keys=True, allow_nan=False)
        if report_path.is_symlink() or report_path.exists():
            raise FileExistsError("V54 report destination appeared before publication")
        v51._atomic_json(report_path, dict(report))
    except BaseException:
        # The checkpoint root was proven absent by preflight and created by
        # this invocation, so it must never outlive a failed report seal.
        if written and checkpoint_root.is_dir() and not checkpoint_root.is_symlink():
            shutil.rmtree(checkpoint_root)
        raise


def run_gate(
    *, expected_v53_report_sha256: str, paths: GatePaths | None = None
) -> dict[str, Any]:
    resolved = _resolved_paths(paths)
    pre = preflight(
        expected_v53_report_sha256=expected_v53_report_sha256, paths=paths
    )
    predecessor = _mapping(pre["predecessor"], "V54 predecessor")
    engine_paths = v51.GridPaths(
        terminal=resolved.predecessor,
        report=resolved.report,
        checkpoint_root=resolved.checkpoint_root,
        config=resolved.config,
    )
    backend: v52.RealRefinementBackend | None = None
    staged: Path | None = None
    preparation: Mapping[str, Any] = {}
    reconstruction: Mapping[str, Any] = {}
    checkpoint: Mapping[str, Any] | None = None
    restoration: Mapping[str, Any] = {"passed": False}
    access: Mapping[str, Any] = {"passed": False}
    errors: list[dict[str, str]] = []
    written = False
    try:
        with v52.scoped_v51_refinement():
            backend = v52.RealRefinementBackend(predecessor, engine_paths)
            preparation = backend.authenticate_and_prepare()
            reconstruction = backend.reconstruct_candidate(TARGET_SPEC)
            if (
                reconstruction.get("full_tensor_state_sha256")
                != CANDIDATE_FULL_SHA256
                or reconstruction.get("authorized_surface_state_sha256")
                != CANDIDATE_AUTHORIZED_SHA256
                or reconstruction.get("query_state_sha256") != CANDIDATE_QUERY_SHA256
                or reconstruction.get("scene_readout_state_sha256")
                != CANDIDATE_SCENE_SHA256
                or reconstruction.get("frozen_state_sha256") != v51._FROZEN_SHA256
            ):
                raise RuntimeError("V54 live candidate reconstruction changed")
            resolved.checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
            staged = Path(
                tempfile.mkdtemp(
                    prefix=f".{resolved.checkpoint_root.name}.staged.",
                    dir=resolved.checkpoint_root.parent,
                )
            )
            checkpoint = _stage_checkpoint(
                backend,
                staged,
                predecessor=predecessor,
                preparation=preparation,
            )
            restoration = backend.restore_source()
            access = backend.access_audit()
            backend.close()
            backend = None
        restoration_exact = bool(
            restoration.get("passed") is True
            and restoration.get("full_tensor_state_sha256")
            == v51._SOURCE_FULL_SHA256
            and restoration.get("authorized_surface_state_sha256")
            == v51._SOURCE_AUTHORIZED_SHA256
            and restoration.get("frozen_state_sha256") == v51._FROZEN_SHA256
            and restoration.get("all_parameter_gradients_absent") is True
        )
        access_exact = bool(
            access.get("passed") is True
            and access.get("training_map_count") == 16
            and access.get("optimizer_file_reads") == []
            and access.get("forbidden_file_accesses") == []
            and access.get("validation_qa_loaded") is False
            and access.get("oracle_loaded") is False
            and access.get("final_test_loaded") is False
        )
        if not (
            restoration_exact
            and access_exact
            and checkpoint.get("v54_training_metadata_provenance") is True
            and checkpoint.get("v52_stage_authenticated_before_v54_rewrite") is True
            and checkpoint.get("runtime_metadata_exact_sanitization") is True
            and checkpoint.get("optimizer_file_written") is False
        ):
            raise RuntimeError("V54 final restoration/access/provenance gate failed")
        resolved.checkpoint_root.mkdir(parents=False, exist_ok=False)
        os.replace(staged, resolved.checkpoint_root / "update_000")
        staged = None
        written = True
    except Exception as exc:  # noqa: BLE001 - failures are sealed and fail closed
        errors.append({"type": type(exc).__name__, "message": str(exc)})
        if backend is not None:
            try:
                with v52.scoped_v51_refinement():
                    restoration = backend.restore_source()
                    access = backend.access_audit()
                    backend.close()
            except Exception as cleanup:  # noqa: BLE001
                errors.append(
                    {"type": type(cleanup).__name__, "message": f"cleanup failed: {cleanup}"}
                )
        if written and resolved.checkpoint_root.is_dir():
            shutil.rmtree(resolved.checkpoint_root)
            written = False
        elif (
            not written
            and resolved.checkpoint_root.is_dir()
            and not any(resolved.checkpoint_root.iterdir())
        ):
            resolved.checkpoint_root.rmdir()
    finally:
        if staged is not None and staged.is_dir():
            shutil.rmtree(staged)
    try:
        report = _report_summary(
            predecessor=predecessor,
            preparation=preparation,
            reconstruction=reconstruction,
            checkpoint=checkpoint,
            restoration=restoration,
            access=access,
            errors=errors,
            written=written,
        )
        _publish_report_or_rollback(
            report_path=resolved.report,
            checkpoint_root=resolved.checkpoint_root,
            report=report,
        )
    except BaseException:
        if (
            written
            and resolved.checkpoint_root.is_dir()
            and not resolved.checkpoint_root.is_symlink()
        ):
            shutil.rmtree(resolved.checkpoint_root)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-v53-report-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--predecessor", type=Path, default=V53_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    paths = GatePaths(
        predecessor=args.predecessor,
        report=args.report,
        checkpoint_root=args.checkpoint_root,
        config=args.config,
    )
    if args.preflight:
        result = preflight(
            expected_v53_report_sha256=args.expected_v53_report_sha256,
            paths=paths,
        )
        summary = {
            "artifact": result["artifact"],
            "passed": result["passed"],
            "predecessor_sha256": result["predecessor"]["sha256"],
            "canonical_complete_units": result["predecessor"]["semantic_metrics"][
                "canonical_type_specific"
            ]["complete_units"],
            "canonical_broad_rows_correct": result["predecessor"][
                "semantic_metrics"
            ]["canonical_type_specific"]["broad_rows_correct"],
            "model_loaded": False,
            "qa_loaded": False,
            "maps_loaded": False,
        }
    else:
        result = run_gate(
            expected_v53_report_sha256=args.expected_v53_report_sha256,
            paths=paths,
        )
        summary = {
            "artifact": result["artifact"],
            "passed": result["passed"],
            "report": str(DEFAULT_REPORT),
            "report_sha256": _sha256(_resolve(DEFAULT_REPORT)),
            "checkpoint_written": result["checkpoint"]["written"],
            "checkpoint": str(DEFAULT_CHECKPOINT),
        }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_ID",
    "GatePaths",
    "authenticate_predecessor",
    "canonical_type_specific_match",
    "inherited_v52_staging_provenance",
    "main",
    "preflight",
    "recompute_semantic_metrics",
    "run_gate",
]

"""Pre-materialization seal and fail-closed guard for V96 final evaluation.

The only mutable decision made here is whether an already-fixed V96 candidate
may enter the physically deferred evaluation.  Thresholds are copied exactly
from V95's sealed final contract and source-bound before any deferred label can
exist.  Later model-bearing processes authenticate this small seal, the V96
unlock, and only the receipts they are allowed to inspect; they never parse the
semantic scene recipe or an answer-bearing file.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, ParamSpec, TypeVar

import yaml

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.seal_v96_known_development_v2 import (
    authenticate_final_evidence_v96,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    MEMORY_ROOT,
    QUESTION_MANIFEST,
    RECEIPT_ROOT,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    PREREGISTRATION as MATERIALIZATION_PREREGISTRATION,
)
from semantic_3d_chat.evaluation.v95_deferred_final_qa import (
    FINAL_QA,
    PAIR_SCENES,
    SCENE_IDS,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    assert_deferred_final_absent_v96,
    load_config_v96,
)
from semantic_3d_chat.evaluation.v96_deferred_final import (
    DEFAULT_UNLOCK,
    MATERIALIZATION_STAGE_ORDER,
    PINNED_V95_MATERIALIZATION_FILE_SHA256,
    PINNED_V95_MATERIALIZATION_IDENTITY_SHA256,
    UNLOCK_ARTIFACT,
    UNLOCK_STATUS,
)
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
    authenticate_fixed_final_candidate_v96,
    canonical_sha256_v96,
    read_json_strict_v96,
    resolve_v96,
    write_json_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    exclusive_evaluation_lock_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation_v2 import (
    authenticate_evaluation_implementation_v96_v2,
)

SCHEMA_VERSION: Final[int] = 96
ARTIFACT: Final[str] = "gemma4_v96_deferred_final_evaluation_preregistration_v1"
STATUS: Final[str] = "sealed_before_deferred_materialization_or_labels"
PREREGISTRATION: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/"
    "gemma4_v96_deferred_final_evaluation_preregistration.json"
)
V95_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/gemma4_v95_strict_causal_successor.yaml"
)
V94_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/gemma4_v94_strict_multiscene_full40.yaml"
)
PINNED_V95_CONFIG_SHA256: Final[str] = (
    "9115c36b417d03bec935257b42e30597170d5acbf6c4683b5c021a8e4d9bbea2"
)

QUESTION_COUNT: Final[int] = 216
CHANGED_UNIT_COUNT: Final[int] = 12
CHANGED_SIDE_COUNT: Final[int] = 24
ROWS_PER_SCENE: Final[int] = 36
PAIR_SCENE: Final[dict[str, str]] = {
    scene: paired
    for pair in PAIR_SCENES.values()
    for scene, paired in (pair, tuple(reversed(pair)))
}
ANSWER_TYPE_TOTALS: Final[dict[str, int]] = {
    "attribute": 48,
    "count": 42,
    "metric": 6,
    "orientation": 6,
    "presence": 42,
    "spatial_relation": 48,
    "support": 24,
}

# Exact copy of V95's already-sealed, pre-deferred final gates.  This mapping
# must match the V95 YAML byte-for-byte at the semantic mapping level.
FINAL_GATE_CONTRACT: Final[dict[str, Any]] = {
    "canonical_accuracy_minimum": 0.65,
    "canonical_accuracy_margin_over_fixed_v94_same_rows": 0.03,
    "attribute_correct_minimum": 24,
    "attribute_total": 48,
    "count_correct_minimum": 38,
    "count_total": 42,
    "metric_correct_minimum": 4,
    "metric_total": 6,
    "orientation_correct_minimum": 4,
    "orientation_total": 6,
    "presence_correct_minimum": 30,
    "presence_total": 42,
    "spatial_relation_correct_minimum": 29,
    "spatial_relation_total": 48,
    "support_correct_minimum": 16,
    "support_total": 24,
    "changed_side_correct_minimum": 14,
    "changed_side_total": 24,
    "complete_changed_units_minimum": 6,
    "changed_unit_total": 12,
    "canonical_prediction_changing_units_minimum": 8,
    "mean_changed_side_wrong_minus_correct_nll_minimum": 0.2,
    "zero_payload_mean_nll_gap_minimum": 0.5,
    "permutation_mean_nll_gap_minimum": 0.35,
    "correct_scene_nll_below_zero_payload_required": True,
    "correct_scene_nll_below_permuted_payload_required": True,
    "exact_prefix_hash_invariance_required": True,
    "question_label_isolation_required": True,
    "protected_read_count_maximum": 0,
    "runtime_packaging_requires_separate_leakage_gate": True,
    "automatic_runtime_promotion": False,
}

_IMPLEMENTATION_SOURCES: Final[tuple[str, ...]] = (
    "Makefile",
    "src/semantic_3d_chat/chat/file_audit.py",
    "src/semantic_3d_chat/chat/runtime_config.py",
    "src/semantic_3d_chat/evaluation/baseline_io.py",
    "src/semantic_3d_chat/evaluation/prediction_artifacts.py",
    "src/semantic_3d_chat/evaluation/question_manifest.py",
    "src/semantic_3d_chat/evaluation/predict_v96_known_development.py",
    "src/semantic_3d_chat/evaluation/seal_v96_known_development.py",
    "src/semantic_3d_chat/evaluation/authenticate_v96_known_development_v2.py",
    "src/semantic_3d_chat/evaluation/nll_v96_known_development_v2.py",
    "src/semantic_3d_chat/evaluation/predict_v96_known_development_v2.py",
    "src/semantic_3d_chat/evaluation/score_v96_known_development_v2.py",
    "src/semantic_3d_chat/evaluation/seal_v96_known_development_v2.py",
    "src/semantic_3d_chat/evaluation/v56_fresh_development_score.py",
    "src/semantic_3d_chat/evaluation/v85_strict_multiscene_preflight.py",
    "src/semantic_3d_chat/evaluation/v94_strict_multiscene_preflight.py",
    "src/semantic_3d_chat/evaluation/v95_strict_causal_successor_preflight.py",
    "src/semantic_3d_chat/evaluation/v96_atomic_pair_repair_preflight.py",
    "src/semantic_3d_chat/evaluation/v96_known_development_common.py",
    "src/semantic_3d_chat/evaluation/v96_known_development_common_v2.py",
    "src/semantic_3d_chat/evaluation/v96_known_development_candidate_attestation.py",
    "src/semantic_3d_chat/evaluation/v96_evaluation_io_v2.py",
    "src/semantic_3d_chat/evaluation/v96_known_development_implementation.py",
    "src/semantic_3d_chat/evaluation/v96_known_development_implementation_v2.py",
    "src/semantic_3d_chat/evaluation/v96_deferred_final.py",
    "src/semantic_3d_chat/evaluation/v96_deferred_final_materialization.py",
    "src/semantic_3d_chat/evaluation/v96_deferred_final_qa.py",
    "src/semantic_3d_chat/evaluation/v96_deferred_final_evaluation.py",
    "src/semantic_3d_chat/evaluation/v96_deferred_final_common.py",
    "src/semantic_3d_chat/evaluation/predict_v96_deferred_final.py",
    "src/semantic_3d_chat/evaluation/score_v96_deferred_final.py",
    "src/semantic_3d_chat/evaluation/nll_v96_deferred_final.py",
    "src/semantic_3d_chat/evaluation/seal_v96_deferred_final.py",
    "src/semantic_3d_chat/evaluation/authenticate_v96_deferred_final.py",
    "src/semantic_3d_chat/evaluation/evaluate_v94_strict_multiscene_full40.py",
    "src/semantic_3d_chat/language/prefix_injection.py",
    "src/semantic_3d_chat/scene_encoder/v81_scene_memory_artifact.py",
    "src/semantic_3d_chat/training/train_v84_strict_bridge.py",
    "src/semantic_3d_chat/training/train_v94_strict_multiscene_full40.py",
    "configs/experiments/gemma4_v94_strict_multiscene_full40.yaml",
    "configs/runtime/gemma4_v85_strict_multiscene.yaml",
)

_OUTPUTS: Final[dict[str, str]] = {
    "v96_predictions": ("reports/gemma4/predictions/gemma4_v96_deferred_final_question_only.jsonl"),
    "v96_prediction_provenance": (
        "reports/gemma4/predictions/gemma4_v96_deferred_final_question_only.jsonl.provenance.json"
    ),
    "v96_prediction_access": (
        "reports/gemma4/predictions/gemma4_v96_deferred_final_question_only.jsonl.access.json"
    ),
    "v96_prediction_completion": (
        "reports/gemma4/predictions/gemma4_v96_deferred_final_question_only.jsonl.completion.json"
    ),
    "v94_predictions": (
        "reports/gemma4/predictions/gemma4_v94_deferred_final_same_rows_question_only.jsonl"
    ),
    "v94_prediction_provenance": (
        "reports/gemma4/predictions/"
        "gemma4_v94_deferred_final_same_rows_question_only.jsonl.provenance.json"
    ),
    "v94_prediction_access": (
        "reports/gemma4/predictions/"
        "gemma4_v94_deferred_final_same_rows_question_only.jsonl.access.json"
    ),
    "v94_prediction_completion": (
        "reports/gemma4/predictions/"
        "gemma4_v94_deferred_final_same_rows_question_only.jsonl.completion.json"
    ),
    "structured_score": ("reports/gemma4/metrics/gemma4_v96_deferred_final_structured.json"),
    "structured_access": (
        "reports/gemma4/metrics/gemma4_v96_deferred_final_structured_access.json"
    ),
    "nll": "reports/gemma4/metrics/gemma4_v96_deferred_final_nll.json",
    "nll_access": ("reports/gemma4/metrics/gemma4_v96_deferred_final_nll_access.json"),
    "nll_completion": ("reports/gemma4/metrics/gemma4_v96_deferred_final_nll_completion.json"),
    "final_score": "reports/gemma4/metrics/gemma4_v96_deferred_final.json",
    "evidence": ("reports/gemma4/metrics/gemma4_v96_deferred_final_evidence.json"),
}

_P = ParamSpec("_P")
_R = TypeVar("_R")


def output_paths_v96_final() -> dict[str, Path]:
    return {name: resolve_v96(path) for name, path in _OUTPUTS.items()}


def _read_v95_contract() -> dict[str, Any]:
    if V95_CONFIG.is_symlink() or not V95_CONFIG.is_file():
        raise FileNotFoundError(V95_CONFIG)
    if sha256_file_v85(V95_CONFIG) != PINNED_V95_CONFIG_SHA256:
        raise ValueError("Pinned V95 deferred-final gate source bytes changed")
    raw = yaml.safe_load(V95_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != {"v95"}:
        raise ValueError("V96 final requires the sealed V95 config")
    value = raw["v95"]
    if not isinstance(value, Mapping):
        raise TypeError("V95 sealed final contract is malformed")
    gates = value.get("gates")
    deferred = value.get("deferred_evaluation")
    sources = value.get("sources")
    if (
        gates != FINAL_GATE_CONTRACT
        or not isinstance(deferred, Mapping)
        or deferred.get("scene_ids") != list(SCENE_IDS)
        or deferred.get("scene_count") != 6
        or deferred.get("pair_count") != 3
        or deferred.get("expected_row_count_after_unlock") != QUESTION_COUNT
        or deferred.get("expected_changed_unit_count_after_unlock") != CHANGED_UNIT_COUNT
        or deferred.get("expected_changed_side_count_after_unlock") != CHANGED_SIDE_COUNT
        or deferred.get("labels_opened_only_by_separate_final_scorer") is not True
        or deferred.get("frozen_v94_comparator_scored_on_same_rows") is not True
        or not isinstance(sources, Mapping)
    ):
        raise ValueError("V95 sealed deferred-final contract changed")
    return dict(value)


def _source_inventory() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _IMPLEMENTATION_SOURCES:
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = sha256_file_v85(path)
    return result


def _known_gate_binding() -> tuple[dict[str, Any], dict[str, Any]]:
    implementation = authenticate_evaluation_implementation_v96_v2(
        config_path=CONFIG
    )
    gate = authenticate_final_evidence_v96(CONFIG)
    config = load_config_v96(CONFIG, allow_draft=False)
    candidate = authenticate_fixed_final_candidate_v96(
        config,
        config_path=CONFIG,
        implementation=implementation,
    )
    if (
        gate.get("authenticated") is not True
        or gate.get("known_development_gate_passed") is not True
        or gate.get("deferred_final_unlock_eligible") is not True
        or gate.get("fixed_final_checkpoint_immutable") is not True
        or gate.get("frozen_v95_parent_immutable") is not True
        or gate.get("scene_prefix_question_independent") is not True
        or gate.get("protected_read_count") != 0
        or not isinstance(gate.get("gate_results"), Mapping)
        or not gate["gate_results"]
        or not all(value is True for value in gate["gate_results"].values())
        or gate.get("candidate_fingerprint_sha256") != candidate["fingerprint_sha256"]
        or gate.get("candidate_attestation_file_sha256")
        != candidate["attestation_file_sha256"]
        or gate.get("candidate_attestation_identity_sha256")
        != candidate["attestation_identity_sha256"]
        or gate.get("candidate_attestation_immutable") is not True
        or gate.get("implementation_seal_sha256") != implementation["seal_sha256"]
        or gate.get("implementation_source_inventory_sha256")
        != implementation["source_inventory_sha256"]
        or gate.get("v1_implementation_seal_sha256")
        != implementation["v1_implementation_seal_sha256"]
        or candidate.get("v2_implementation_seal_sha256")
        != implementation["seal_sha256"]
    ):
        raise RuntimeError("V96 final evaluation requires a complete known-development PASS")
    return gate, candidate


def build_preregistration_v96_final() -> dict[str, Any]:
    """Build the fixed final-evaluation contract without deferred data reads."""

    gate, candidate = _known_gate_binding()
    v95 = _read_v95_contract()
    sources = v95["sources"]
    source_inventory = _source_inventory()
    receipts = {
        stage: (RECEIPT_ROOT / f"{stage}.json").relative_to(PROJECT_ROOT).as_posix()
        for stage in MATERIALIZATION_STAGE_ORDER
    }
    memory_paths = {
        scene: (MEMORY_ROOT / scene).relative_to(PROJECT_ROOT).as_posix() for scene in SCENE_IDS
    }
    payload = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "candidate": {
            key: candidate[key]
            for key in (
                "fingerprint_sha256",
                "state_sha256",
                "weights_sha256",
                "metadata_file_sha256",
                "training_report_sha256",
                "config_sha256",
                "preregistration_sha256",
                "cpu_preflight_sha256",
                "frozen_v95_state_sha256",
                "attestation_file_sha256",
                "attestation_identity_sha256",
                "v2_implementation_seal_sha256",
            )
        },
        "known_development": {
            "final_score_sha256": gate["final_score_sha256"],
            "evidence_sha256": gate["evidence_sha256"],
            "gate_results_sha256": canonical_sha256_v96(gate["gate_results"]),
            "candidate_attestation_file_sha256": gate[
                "candidate_attestation_file_sha256"
            ],
            "candidate_attestation_identity_sha256": gate[
                "candidate_attestation_identity_sha256"
            ],
            "implementation_seal_sha256": gate["implementation_seal_sha256"],
            "implementation_source_inventory_sha256": gate[
                "implementation_source_inventory_sha256"
            ],
            "v1_implementation_seal_sha256": gate[
                "v1_implementation_seal_sha256"
            ],
            "passed": True,
        },
        "v95_gate_source": {
            "path": V95_CONFIG.relative_to(PROJECT_ROOT).as_posix(),
            "file_sha256": sha256_file_v85(V95_CONFIG),
            "contract": dict(FINAL_GATE_CONTRACT),
            "contract_sha256": canonical_sha256_v96(FINAL_GATE_CONTRACT),
        },
        "scope": {
            "scene_ids": list(SCENE_IDS),
            "pair_scenes": {key: list(value) for key, value in PAIR_SCENES.items()},
            "pair_scene_lookup": dict(PAIR_SCENE),
            "scene_count": 6,
            "pair_count": 3,
            "row_count": QUESTION_COUNT,
            "rows_per_scene": ROWS_PER_SCENE,
            "changed_unit_count": CHANGED_UNIT_COUNT,
            "changed_side_count": CHANGED_SIDE_COUNT,
            "answer_type_totals": dict(ANSWER_TYPE_TOTALS),
        },
        "materialization": {
            "preregistration_path": MATERIALIZATION_PREREGISTRATION.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "preregistration_file_sha256": PINNED_V95_MATERIALIZATION_FILE_SHA256,
            "preregistration_identity_sha256": (PINNED_V95_MATERIALIZATION_IDENTITY_SHA256),
            "unlock_path": DEFAULT_UNLOCK.relative_to(PROJECT_ROOT).as_posix(),
            "receipt_paths": receipts,
            "memory_paths": memory_paths,
            "question_manifest": QUESTION_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
            "labels_path": FINAL_QA.relative_to(PROJECT_ROOT).as_posix(),
        },
        "v94_same_row_comparator": {
            "experiment_config": V94_CONFIG.relative_to(PROJECT_ROOT).as_posix(),
            "experiment_config_sha256": sha256_file_v85(V94_CONFIG),
            "fixed_final": str(sources["frozen_v94_fixed_final"]),
            "bridge_sha256": str(sources["frozen_v94_bridge_sha256"]),
            "bridge_metadata_sha256": str(sources["frozen_v94_bridge_metadata_sha256"]),
            "bridge_state_sha256": v95["frozen_stack"]["v94_bank_state_sha256"],
            "same_question_manifest_and_memory_required": True,
            "historical_v94_predictions_permitted": False,
        },
        "outputs": dict(_OUTPUTS),
        "implementation_source_sha256": source_inventory,
        "implementation_source_inventory_sha256": canonical_sha256_v96(source_inventory),
        "all_thresholds_sealed_before_deferred_labels": True,
        "all_memories_bound_before_questions": True,
        "question_dependent_retrieval": False,
        "predictors_are_label_blind": True,
        "structured_and_nll_scorers_are_separate_label_processes": True,
        "row_level_labels_forbidden_from_aggregate_outputs": True,
        "runtime_packaging_requires_separate_leakage_gate": True,
        "automatic_runtime_promotion": False,
    }
    payload["preregistration_identity_sha256"] = canonical_sha256_v96(payload)
    return payload


def seal_preregistration_v96_final() -> dict[str, Any]:
    """Create the one preregistration while every deferred output is absent."""

    if PREREGISTRATION.exists() or PREREGISTRATION.is_symlink():
        raise FileExistsError(PREREGISTRATION)
    config = load_config_v96(CONFIG, allow_draft=False)
    assert_deferred_final_absent_v96(config)
    for path in (*output_paths_v96_final().values(), DEFAULT_UNLOCK, RECEIPT_ROOT):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"V96 final preregistration requires absence: {path}")
    payload = build_preregistration_v96_final()
    write_json_create_once_v96(PREREGISTRATION, payload)
    return authenticate_preregistration_v96_final()


def _authenticate_static_bindings(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reauthenticate without opening prior predictions or semantic recipes."""

    config = load_config_v96(CONFIG, allow_draft=False)
    implementation = authenticate_evaluation_implementation_v96_v2(
        config_path=CONFIG
    )
    candidate = authenticate_fixed_final_candidate_v96(
        config,
        config_path=CONFIG,
        implementation=implementation,
    )
    known_gate = authenticate_final_evidence_v96(CONFIG)
    stored_candidate = payload.get("candidate")
    stored_known = payload.get("known_development")
    v95_source = payload.get("v95_gate_source")
    if not all(
        isinstance(value, Mapping) for value in (stored_candidate, stored_known, v95_source)
    ):
        raise TypeError("V96 final preregistration bindings are missing")
    current_sources = _source_inventory()
    v95 = _read_v95_contract()
    v95_sources = v95["sources"]
    expected_comparator = {
        "experiment_config": V94_CONFIG.relative_to(PROJECT_ROOT).as_posix(),
        "experiment_config_sha256": sha256_file_v85(V94_CONFIG),
        "fixed_final": str(v95_sources["frozen_v94_fixed_final"]),
        "bridge_sha256": str(v95_sources["frozen_v94_bridge_sha256"]),
        "bridge_metadata_sha256": str(v95_sources["frozen_v94_bridge_metadata_sha256"]),
        "bridge_state_sha256": v95["frozen_stack"]["v94_bank_state_sha256"],
        "same_question_manifest_and_memory_required": True,
        "historical_v94_predictions_permitted": False,
    }
    expected_scope = {
        "scene_ids": list(SCENE_IDS),
        "pair_scenes": {key: list(value) for key, value in PAIR_SCENES.items()},
        "pair_scene_lookup": dict(PAIR_SCENE),
        "scene_count": 6,
        "pair_count": 3,
        "row_count": QUESTION_COUNT,
        "rows_per_scene": ROWS_PER_SCENE,
        "changed_unit_count": CHANGED_UNIT_COUNT,
        "changed_side_count": CHANGED_SIDE_COUNT,
        "answer_type_totals": dict(ANSWER_TYPE_TOTALS),
    }
    expected_materialization = {
        "preregistration_path": MATERIALIZATION_PREREGISTRATION.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "preregistration_file_sha256": PINNED_V95_MATERIALIZATION_FILE_SHA256,
        "preregistration_identity_sha256": (PINNED_V95_MATERIALIZATION_IDENTITY_SHA256),
        "unlock_path": DEFAULT_UNLOCK.relative_to(PROJECT_ROOT).as_posix(),
        "receipt_paths": {
            stage: (RECEIPT_ROOT / f"{stage}.json").relative_to(PROJECT_ROOT).as_posix()
            for stage in MATERIALIZATION_STAGE_ORDER
        },
        "memory_paths": {
            scene: (MEMORY_ROOT / scene).relative_to(PROJECT_ROOT).as_posix() for scene in SCENE_IDS
        },
        "question_manifest": QUESTION_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "labels_path": FINAL_QA.relative_to(PROJECT_ROOT).as_posix(),
    }
    expected_candidate = {
        key: candidate[key]
        for key in (
            "fingerprint_sha256",
            "state_sha256",
            "weights_sha256",
            "metadata_file_sha256",
            "training_report_sha256",
            "config_sha256",
            "preregistration_sha256",
            "cpu_preflight_sha256",
            "frozen_v95_state_sha256",
            "attestation_file_sha256",
            "attestation_identity_sha256",
            "v2_implementation_seal_sha256",
        )
    }
    if (
        payload.get("artifact") != ARTIFACT
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != STATUS
        or payload.get("preregistration_identity_sha256")
        != canonical_sha256_v96(
            {
                key: value
                for key, value in payload.items()
                if key != "preregistration_identity_sha256"
            }
        )
        or dict(stored_candidate) != expected_candidate
        or v95_source.get("file_sha256") != sha256_file_v85(V95_CONFIG)
        or v95_source.get("contract") != FINAL_GATE_CONTRACT
        or v95_source.get("contract_sha256") != canonical_sha256_v96(FINAL_GATE_CONTRACT)
        or payload.get("implementation_source_sha256") != current_sources
        or payload.get("implementation_source_inventory_sha256")
        != canonical_sha256_v96(current_sources)
        or payload.get("v94_same_row_comparator") != expected_comparator
        or payload.get("scope") != expected_scope
        or payload.get("materialization") != expected_materialization
        or payload.get("outputs") != _OUTPUTS
        or payload.get("all_thresholds_sealed_before_deferred_labels") is not True
        or payload.get("all_memories_bound_before_questions") is not True
        or payload.get("question_dependent_retrieval") is not False
        or payload.get("predictors_are_label_blind") is not True
        or payload.get("automatic_runtime_promotion") is not False
    ):
        raise ValueError("V96 final preregistration static binding changed")
    known_paths = (
        resolve_v96("reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development.json"),
        resolve_v96(
            "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development_evidence.json"
        ),
    )
    if (
        any(path.is_symlink() or not path.is_file() for path in known_paths)
        or sha256_file_v85(known_paths[0]) != stored_known.get("final_score_sha256")
        or sha256_file_v85(known_paths[1]) != stored_known.get("evidence_sha256")
        or stored_known.get("passed") is not True
        or known_gate.get("authenticated") is not True
        or known_gate.get("known_development_gate_passed") is not True
        or known_gate.get("final_score_sha256") != stored_known.get("final_score_sha256")
        or known_gate.get("evidence_sha256") != stored_known.get("evidence_sha256")
        or canonical_sha256_v96(known_gate.get("gate_results"))
        != stored_known.get("gate_results_sha256")
        or known_gate.get("candidate_attestation_file_sha256")
        != stored_known.get("candidate_attestation_file_sha256")
        or known_gate.get("candidate_attestation_identity_sha256")
        != stored_known.get("candidate_attestation_identity_sha256")
        or known_gate.get("implementation_seal_sha256")
        != stored_known.get("implementation_seal_sha256")
        or known_gate.get("implementation_source_inventory_sha256")
        != stored_known.get("implementation_source_inventory_sha256")
        or known_gate.get("v1_implementation_seal_sha256")
        != stored_known.get("v1_implementation_seal_sha256")
    ):
        raise ValueError("V96 known-development PASS bytes changed after final seal")
    return candidate


def authenticate_preregistration_v96_final() -> dict[str, Any]:
    source = PREREGISTRATION
    payload = read_json_strict_v96(source)
    candidate = _authenticate_static_bindings(payload)
    return {
        **payload,
        "preregistration_file_sha256": sha256_file_v85(source),
        "current_candidate_fingerprint_sha256": candidate["fingerprint_sha256"],
        "authenticated": True,
    }


def authenticate_unlock_blind_v96_final(
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the aggregate unlock without parsing the scene recipe."""

    path = DEFAULT_UNLOCK
    unlock = read_json_strict_v96(path)
    candidate = preregistration["candidate"]
    known = preregistration["known_development"]
    materialization = preregistration["materialization"]
    identity = unlock.get("unlock_identity_sha256")
    if (
        unlock.get("artifact") != UNLOCK_ARTIFACT
        or unlock.get("schema_version") != SCHEMA_VERSION
        or unlock.get("status") != UNLOCK_STATUS
        or unlock.get("candidate_fingerprint_sha256") != candidate["fingerprint_sha256"]
        or unlock.get("candidate_state_sha256") != candidate["state_sha256"]
        or unlock.get("candidate_attestation_file_sha256")
        != candidate["attestation_file_sha256"]
        or unlock.get("candidate_attestation_identity_sha256")
        != candidate["attestation_identity_sha256"]
        or unlock.get("known_development_final_score_sha256") != known["final_score_sha256"]
        or unlock.get("known_development_evidence_sha256") != known["evidence_sha256"]
        or unlock.get("known_development_implementation_seal_sha256")
        != known["implementation_seal_sha256"]
        or unlock.get("known_development_implementation_source_inventory_sha256")
        != known["implementation_source_inventory_sha256"]
        or unlock.get("known_development_v1_implementation_seal_sha256")
        != known["v1_implementation_seal_sha256"]
        or unlock.get("materialization_preregistration_file_sha256")
        != materialization["preregistration_file_sha256"]
        or unlock.get("materialization_preregistration_identity_sha256")
        != materialization["preregistration_identity_sha256"]
        or unlock.get("final_evaluation_preregistration_file_sha256")
        != preregistration["preregistration_file_sha256"]
        or unlock.get("final_evaluation_preregistration_identity_sha256")
        != preregistration["preregistration_identity_sha256"]
        or unlock.get("final_evaluation_gate_contract_sha256")
        != preregistration["v95_gate_source"]["contract_sha256"]
        or unlock.get("final_evaluation_implementation_source_inventory_sha256")
        != preregistration["implementation_source_inventory_sha256"]
        or unlock.get("final_evaluation_preregistered_before_unlock") is not True
        or unlock.get("explicit_unlock_created") is not True
        or unlock.get("materialization_stage_execution_authorized") is not True
        or unlock.get("protected_read_count") != 0
        or unlock.get("runtime_promotion_authorized") is not False
        or identity
        != canonical_sha256_v96(
            {key: value for key, value in unlock.items() if key != "unlock_identity_sha256"}
        )
    ):
        raise ValueError("V96 final blind unlock authentication failed")
    return {
        **unlock,
        "unlock_file_sha256": sha256_file_v85(path),
        "authenticated": True,
    }


def authenticate_stage_receipt_v96_final(
    preregistration: Mapping[str, Any],
    unlock: Mapping[str, Any],
    stage: str,
    *,
    hash_outputs: bool,
) -> dict[str, Any]:
    """Authenticate one fixed receipt, optionally hashing its output files."""

    if stage not in MATERIALIZATION_STAGE_ORDER:
        raise ValueError(f"Unknown V96 final materialization stage: {stage}")
    raw_path = preregistration["materialization"]["receipt_paths"][stage]
    path = resolve_v96(raw_path)
    receipt = read_json_strict_v96(path)
    output_hashes = receipt.get("output_sha256")
    execution = unlock.get("execution_contract")
    if not isinstance(output_hashes, Mapping) or not output_hashes:
        raise TypeError("V96 final receipt output hashes are missing")
    if not isinstance(execution, Mapping):
        raise TypeError("V96 final unlock execution contract is missing")
    if hash_outputs:
        for raw, expected in output_hashes.items():
            output = resolve_v96(str(raw))
            if output.is_symlink() or not output.is_file():
                raise FileNotFoundError(output)
            if sha256_file_v85(output) != expected:
                raise ValueError(f"V96 final materialized output changed: {raw}")
    identity = receipt.get("receipt_identity_sha256")
    if (
        receipt.get("artifact") != "gemma4_v96_deferred_final_stage_receipt_v1"
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("stage") != stage
        or receipt.get("status") != "completed_after_authenticated_v96_unlock"
        or receipt.get("unlock_file_sha256") != unlock["unlock_file_sha256"]
        or receipt.get("unlock_identity_sha256") != unlock["unlock_identity_sha256"]
        or receipt.get("candidate_fingerprint_sha256")
        != preregistration["candidate"]["fingerprint_sha256"]
        or receipt.get("preregistration_file_sha256")
        != preregistration["materialization"]["preregistration_file_sha256"]
        or receipt.get("preregistration_identity_sha256")
        != preregistration["materialization"]["preregistration_identity_sha256"]
        or receipt.get("implementation_source_inventory_sha256")
        != unlock["implementation_source_inventory_sha256"]
        or receipt.get("stage_execution_performed") is not True
        or receipt.get("v96_authorization_override") != (stage == "qa_select")
        or canonical_sha256_v96(receipt.get("child_argv"))
        != execution["materialization_stage_child_argv_sha256"][stage]
        or canonical_sha256_v96(receipt.get("expected_outputs"))
        != execution["materialization_stage_output_contract_sha256"][stage]
        or set(output_hashes) != set(receipt.get("expected_outputs", []))
        or receipt.get("automatic_runtime_promotion") is not False
        or receipt.get("output_inventory_sha256") != canonical_sha256_v96(output_hashes)
        or identity
        != canonical_sha256_v96(
            {key: value for key, value in receipt.items() if key != "receipt_identity_sha256"}
        )
    ):
        raise ValueError(f"V96 final stage receipt changed: {stage}")
    return {
        **receipt,
        "receipt_file_sha256": sha256_file_v85(path),
        "outputs_rehashed": hash_outputs,
        "authenticated": True,
    }


def authenticate_materialized_inputs_v96_final(
    *,
    label_process: bool,
) -> dict[str, Any]:
    preregistration = authenticate_preregistration_v96_final()
    unlock = authenticate_unlock_blind_v96_final(preregistration)
    stages = MATERIALIZATION_STAGE_ORDER if label_process else ("memory", "questions")
    receipts = {
        stage: authenticate_stage_receipt_v96_final(
            preregistration,
            unlock,
            stage,
            # A predictor authenticates the questions receipt but deliberately
            # does not open/hash the manifest until all memories are bound.
            hash_outputs=(stage == "memory" or (label_process and stage == "questions")),
        )
        for stage in stages
    }
    return {
        "preregistration": preregistration,
        "unlock": unlock,
        "receipts": receipts,
        "label_process": label_process,
        "semantic_recipe_parsed": False,
        "authenticated": True,
    }


@contextmanager
def deferred_evaluation_guard_v96(
    *,
    label_process: bool,
) -> Iterator[Mapping[str, Any]]:
    with exclusive_evaluation_lock_v96():
        yield authenticate_materialized_inputs_v96_final(label_process=label_process)


def hardened_deferred_evaluation_stage_v96(
    *,
    label_process: bool,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(function)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with deferred_evaluation_guard_v96(label_process=label_process):
                return function(*args, **kwargs)

        return guarded

    return decorate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "authenticate"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        seal_preregistration_v96_final()
        if args.command == "seal"
        else authenticate_preregistration_v96_final()
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANSWER_TYPE_TOTALS",
    "ARTIFACT",
    "CHANGED_SIDE_COUNT",
    "CHANGED_UNIT_COUNT",
    "FINAL_GATE_CONTRACT",
    "PAIR_SCENE",
    "PREREGISTRATION",
    "QUESTION_COUNT",
    "ROWS_PER_SCENE",
    "SCENE_IDS",
    "authenticate_materialized_inputs_v96_final",
    "authenticate_preregistration_v96_final",
    "authenticate_stage_receipt_v96_final",
    "authenticate_unlock_blind_v96_final",
    "build_preregistration_v96_final",
    "deferred_evaluation_guard_v96",
    "hardened_deferred_evaluation_stage_v96",
    "main",
    "output_paths_v96_final",
    "seal_preregistration_v96_final",
]

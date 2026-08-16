"""Fail-closed V94 strict multiscene runtime packaging and promotion.

This module is deliberately a *post-evaluation* release wrapper.  It refuses
to create even a candidate package unless the independent V94 evidence
verifier authenticates both the sealed aggregate score and every
preregistered behavior gate.  The chat subprocess never imports this module.

The standalone runtime contains only:

* the exact seven-bank V85 adapter archive plus the two trained V94 tensors;
* a sanitized runtime configuration with all eight final bank-state hashes;
* six V81-format continuous scene-memory artifacts for opaque scenes 57--62.

No question, answer, label, caption, oracle record, or evaluation report is
serialized in those runtime artifacts.  Promotion is a separate operation
after an oracle-physically-unavailable child-process smoke.  Adapter bytes and
all six memory tensor files are copied byte-for-byte from the smoked candidate
to the promoted release; only runtime metadata is rebound to the promoted
checkpoint identity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT, config_hash
from semantic_3d_chat.evaluation import v94_strict_multiscene_evidence
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.strict_direct_release_core import (
    BridgeSourceContract,
    base_bank_order,
    compose_exact_bank_archive,
    extend_runtime_lora_config,
    extend_runtime_metadata,
    load_bridge_source,
    sha256_file,
    validate_runtime_bank_inventory,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    MEMORY_FILENAME,
    METADATA_FILENAME,
    load_v81_scene_memory,
    save_v81_scene_memory,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    validate_runtime_checkpoint_metadata,
)

SCHEMA_VERSION: Final[int] = 94
ARTIFACT: Final[str] = "gemma4_v94_strict_runtime_release_v1"
PROMOTION_DECISION: Final[str] = "strict_multiscene_experimental_primary"
PENDING_DECISION: Final[str] = "pending_isolated_runtime_smoke"
SCENE_IDS: Final[tuple[str, ...]] = tuple(f"scene_{index:06d}" for index in range(57, 63))
MEMORY_SHAPE: Final[tuple[int, int, int]] = (1, 738, 1536)

V94_BANK: Final[str] = "v94_strict_multiscene_full40_bridge"
V94_TARGET: Final[str] = "model.language_model.layers.34.mlp.gate_proj"
V94_TRAINED_STATE_SHA256: Final[str] = (
    "9f503f0b2c605238a6f32c15740c0600702d46da08a527d867fbc19e6b639452"
)
V94_PARAMETER_COUNT: Final[int] = 110_592
EXPECTED_ADAPTER_PARAMETER_COUNT: Final[int] = 675_840

PARENT_BANKS: Final[tuple[str, ...]] = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
    "extension_v28_stage_b_query",
    "extension_v30_joint_pair_query",
    "v85_strict_multiscene_bridge",
)
EXPECTED_BANKS: Final[tuple[str, ...]] = (*PARENT_BANKS, V94_BANK)

PARENT_RUNTIME_CONFIG_SHA256: Final[str] = (
    "49d9595e6167fe352dc49f1ca363af396fe3d3be91d50cb430721bc74b130575"
)
PARENT_ADAPTER_SHA256: Final[str] = (
    "163ceb462bafe4b1d38099c07ca32bcc22944ad93917222b019e75634883fd8d"
)
PARENT_METADATA_SHA256: Final[str] = (
    "093d43c80722dd00cd3f7347d55451eca1cc45e416f6bda220157f18024ce0bd"
)
SOURCE_CONTROLLER_WEIGHTS_SHA256: Final[str] = (
    "bb112f42ca5df71b88b4cd7721b9107f9be9b0dc01b612a4ace6212548da669c"
)
SOURCE_PROBE_TENSOR_SHA256: Final[str] = (
    "fb32c687dd787f108fab03e9745eefb2273891c2be990d0acf50ca111eb637e8"
)

EXPERIMENT_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/gemma4_v94_strict_multiscene_full40.yaml"
)
PARENT_RUNTIME_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/gemma4_v85_strict_multiscene.yaml"
)
PARENT_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v85_strict_runtime_candidate"
)
V94_BRIDGE_CANDIDATE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v94_strict_multiscene_full40_final"
)
EVALUATION_CACHE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v94_strict_multiscene_full40/evaluation_cache"
)
CONTROLLER_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
)

RUNTIME_CONFIG: Final[Path] = PROJECT_ROOT / "configs/runtime/gemma4_v94_strict_multiscene.yaml"
CANDIDATE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v94_strict_runtime_candidate"
)
CANDIDATE_MEMORY_ROOT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v94_strict_runtime_candidate_memories"
)
SMOKE_ROOT: Final[Path] = PROJECT_ROOT / "reports/gemma4/artifacts/v94_strict_runtime_smoke"
SMOKE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v94_strict_runtime_smoke.json"
)
RELEASE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/checkpoints/gemma4_v94_strict_multiscene_release_v1"
)
RELEASE_MEMORY_ROOT: Final[Path] = PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v94"
RELEASE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v94_strict_runtime_release.json"
)

_SMOKE_QUESTIONS: Final[tuple[str, ...]] = (
    "What is in the room?",
    "What is closest to the camera?",
)
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_CACHE_TENSOR_METADATA: Final[dict[str, str]] = {
    "artifact": "v94_question_independent_evaluation_memory_cache_v1",
    "environmental_text_serialized": "false",
    "question_inputs_used": "false",
}
_REQUIRED_SMOKE_GATES: Final[frozenset[str]] = frozenset(
    {
        "hardened_score_and_behavior_evidence_passed",
        "all_six_runtime_processes_exit_zero",
        "all_oracle_directories_physically_unavailable",
        "all_oracle_directories_restored",
        "all_six_children_report_oracle_unavailable",
        "all_six_children_use_exact_eight_frozen_banks",
        "all_six_children_report_candidate_mode",
        "all_twelve_questions_return_nonempty_answers",
        "every_scene_prefix_is_invariant",
        "every_scene_prefix_matches_attested_cache_memory",
        "direct_memory_layout_retained_for_every_answer",
        "file_audit_forbidden_read_count_zero",
        "file_audit_protected_read_count_zero",
        "candidate_adapter_bytes_unchanged",
        "candidate_memory_tensor_bytes_unchanged",
        "no_expectation_channel_in_child_protocol",
    }
)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"V94 release requires a lowercase SHA-256 for {label}")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _score_evidence(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    score = evidence.get("score")
    if not isinstance(score, Mapping):
        raise TypeError("V94 release requires the authenticated aggregate score")
    if score.get("behavior_gate_passed") is not True:
        raise ValueError("V94 release requires every preregistered behavior gate")
    _require_sha256(score.get("score_sha256"), "aggregate score")
    return score


def _contract_from_evidence(evidence: Mapping[str, Any]) -> BridgeSourceContract:
    state = _require_sha256(evidence.get("candidate_state_sha256"), "V94 bridge state")
    if state != V94_TRAINED_STATE_SHA256:
        raise ValueError("V94 trained bridge state differs from the runtime contract")
    return BridgeSourceContract(
        root=V94_BRIDGE_CANDIDATE,
        artifact="gemma4_v94_strict_multiscene_full40_fixed_final_v1",
        bank_name=V94_BANK,
        target_module=V94_TARGET,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        parameter_count=V94_PARAMETER_COUNT,
        state_sha256=state,
        weights_sha256=_require_sha256(
            evidence.get("candidate_weights_sha256"), "V94 bridge weights"
        ),
        metadata_sha256=_require_sha256(
            evidence.get("candidate_metadata_sha256"), "V94 bridge metadata"
        ),
    )


def authenticate_v94_model_gate() -> dict[str, Any]:
    """Require the hardened V94 score *and* behavior-pass evidence."""

    evidence = v94_strict_multiscene_evidence.authenticate_v94_evidence(
        EXPERIMENT_CONFIG,
        root=PROJECT_ROOT,
        require_score=True,
        require_behavior_pass=True,
    )
    score = _score_evidence(evidence)
    memory_hashes = evidence.get("memory_sha256")
    if (
        evidence.get("artifact") != v94_strict_multiscene_evidence.ARTIFACT
        or evidence.get("passed") is not True
        or evidence.get("behavior_score_present") is not True
        or evidence.get("behavior_gate_passed") is not True
        or not isinstance(memory_hashes, Mapping)
        or tuple(sorted(memory_hashes)) != SCENE_IDS
        or any(_SHA256.fullmatch(str(value)) is None for value in memory_hashes.values())
        or _require_sha256(evidence.get("bundle_sha256"), "evidence bundle")
        != _canonical_sha256(
            {key: value for key, value in evidence.items() if key != "bundle_sha256"}
        )
        or score.get("status") != "passed_awaiting_separate_leakage_packaging"
    ):
        raise ValueError("V94 hardened model-gate evidence is incomplete")
    pinned = {
        PARENT_RUNTIME_CONFIG: PARENT_RUNTIME_CONFIG_SHA256,
        PARENT_CHECKPOINT / "adapter.safetensors": PARENT_ADAPTER_SHA256,
        PARENT_CHECKPOINT / RUNTIME_METADATA_FILENAME: PARENT_METADATA_SHA256,
    }
    if any(not path.is_file() or sha256_file(path) != digest for path, digest in pinned.items()):
        raise ValueError("V94 release parent source bytes changed")
    load_bridge_source(_contract_from_evidence(evidence))
    return evidence


def _authenticated_parent_state() -> tuple[dict[str, Any], dict[str, str]]:
    metadata = _read_json(PARENT_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    states = metadata.get("lora_bank_state_sha256")
    if (
        base_bank_order(metadata) != PARENT_BANKS
        or not isinstance(states, dict)
        or set(states) != set(PARENT_BANKS)
    ):
        raise ValueError("V94 V85 parent bank inventory changed")
    rebound = {
        name: _require_sha256(states.get(name), f"V85 final bank {name}") for name in PARENT_BANKS
    }
    # This is the historical field that was null in V85's YAML.  A standalone
    # runtime must bind it to the authenticated final checkpoint state.
    if rebound["extension_v28_stage_b_query"] == "":  # pragma: no cover
        raise RuntimeError("V94 extension_v28 final state was not rebound")
    return metadata, rebound


def build_runtime_config_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Build exact V85-seven-plus-V94-one frozen runtime configuration."""

    if sha256_file(PARENT_RUNTIME_CONFIG) != PARENT_RUNTIME_CONFIG_SHA256:
        raise ValueError("V94 parent runtime configuration changed")
    parent = load_runtime_config(PARENT_RUNTIME_CONFIG)
    parent.pop("_config_path", None)
    _metadata, states = _authenticated_parent_state()
    configured = parent.get("language", {}).get("lora_banks")
    if not isinstance(configured, dict) or tuple(configured) != PARENT_BANKS:
        raise ValueError("V94 parent runtime bank order changed")
    for name in PARENT_BANKS:
        row = configured[name]
        if not isinstance(row, dict) or row.get("trainable") is not False:
            raise ValueError(f"V94 parent bank is not frozen: {name}")
        row["expected_initial_state_sha256"] = states[name]
    payload = extend_runtime_lora_config(
        parent_runtime_config=parent,
        added_bridges=(_contract_from_evidence(evidence),),
        expected_final_banks=EXPECTED_BANKS,
    )
    banks = payload["language"]["lora_banks"]
    if (
        tuple(banks) != EXPECTED_BANKS
        or len(banks) != 8
        or any(row.get("trainable") is not False for row in banks.values())
        or any(
            _SHA256.fullmatch(str(row.get("expected_initial_state_sha256"))) is None
            for row in banks.values()
        )
    ):
        raise RuntimeError("V94 runtime config lost the exact frozen final-state inventory")
    return payload


def materialize_runtime_config(evidence: Mapping[str, Any]) -> dict[str, Any]:
    authenticated = authenticate_v94_model_gate()
    if dict(evidence) != authenticated:
        raise ValueError("V94 runtime-config evidence is not current")
    payload = build_runtime_config_payload(authenticated)
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    if RUNTIME_CONFIG.exists():
        if RUNTIME_CONFIG.is_symlink() or RUNTIME_CONFIG.read_text(encoding="utf-8") != encoded:
            raise ValueError("Existing V94 runtime config differs from authenticated evidence")
    else:
        RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with RUNTIME_CONFIG.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    return load_runtime_config(RUNTIME_CONFIG)


def _composed_adapter(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    return compose_exact_bank_archive(
        base_checkpoint=PARENT_CHECKPOINT,
        expected_base_banks=PARENT_BANKS,
        added_bridges=(_contract_from_evidence(evidence),),
        expected_final_banks=EXPECTED_BANKS,
    )


def _source_stack_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    source = {
        name: value
        for name, value in tensors.items()
        if not name.startswith("block_cross_residual.")
    }
    if not source or len(source) >= len(tensors):
        raise RuntimeError("V94 frozen source-stack inventory is invalid")
    return tensor_state_sha256(source)


def build_runtime_metadata(
    evidence: Mapping[str, Any],
    *,
    promotion: str,
    smoke_report_sha256: str | None,
) -> dict[str, Any]:
    if promotion not in {PENDING_DECISION, PROMOTION_DECISION}:
        raise ValueError("Unknown V94 runtime promotion state")
    promoted = promotion == PROMOTION_DECISION
    if promoted != (smoke_report_sha256 is not None):
        raise ValueError("V94 promotion and smoke binding disagree")
    if smoke_report_sha256 is not None:
        _require_sha256(smoke_report_sha256, "runtime smoke report")
    score = _score_evidence(evidence)
    config = build_runtime_config_payload(evidence)
    parent, parent_states = _authenticated_parent_state()
    metadata = extend_runtime_metadata(
        parent_metadata=parent,
        added_bridges=(_contract_from_evidence(evidence),),
        expected_final_banks=EXPECTED_BANKS,
    )
    states = metadata["lora_bank_state_sha256"]
    if {name: states[name] for name in PARENT_BANKS} != parent_states:
        raise RuntimeError("V94 extended metadata changed a V85 final bank state")
    # Rebind every architecture row, including V85's legacy extension_v28 null.
    for row in metadata["lora"]["banks"]:
        row["expected_initial_state_sha256"] = states[str(row["name"])]
    metadata["config_hash"] = config_hash(config)
    tensors, _composition = _composed_adapter(evidence)
    metadata["frozen_block_cross_source_stack_state_sha256"] = _source_stack_sha256(tensors)
    provenance = copy.deepcopy(dict(metadata.get("initialization_provenance", {})))
    provenance["v94_strict_runtime_release"] = {
        "schema_version": SCHEMA_VERSION,
        "source_v94_evidence_sha256": evidence["bundle_sha256"],
        "source_v94_score_sha256": score["score_sha256"],
        "v94_bridge_state_sha256": evidence["candidate_state_sha256"],
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "promotion_decision": promotion,
        "runtime_promotion_authorized": promoted,
        "smoke_report_sha256": smoke_report_sha256,
        "held_out_generalization_claim": True,
    }
    metadata["initialization_provenance"] = provenance
    validate_runtime_checkpoint_metadata(metadata)
    validate_runtime_bank_inventory(
        runtime_config=config,
        checkpoint_metadata=metadata,
        expected_bank_order=EXPECTED_BANKS,
        expected_states={name: str(states[name]) for name in EXPECTED_BANKS},
    )
    if (
        metadata.get("lora_parameter_count") != EXPECTED_ADAPTER_PARAMETER_COUNT
        or metadata.get("lora_trainable_parameter_count") != 0
        or any(
            row.get("expected_initial_state_sha256") != states[row["name"]]
            for row in metadata["lora"]["banks"]
        )
    ):
        raise RuntimeError("V94 runtime metadata parameter or final-state inventory changed")
    return metadata


def _atomic_checkpoint(
    destination: Path,
    *,
    metadata: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_adapter: Path | None = None,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        if source_adapter is None:
            tensors, inheritance = _composed_adapter(evidence)
            save_file(tensors, str(temporary / "adapter.safetensors"))
        else:
            if source_adapter.is_symlink() or not source_adapter.is_file():
                raise FileNotFoundError(source_adapter)
            shutil.copyfile(source_adapter, temporary / "adapter.safetensors")
            inheritance = {"candidate_adapter_bytes_reused_exactly": True}
        _write_json(temporary / RUNTIME_METADATA_FILENAME, metadata)
        if {item.name for item in temporary.iterdir()} != {
            "adapter.safetensors",
            RUNTIME_METADATA_FILENAME,
        }:
            raise RuntimeError("V94 checkpoint is not an exact two-file package")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    fingerprint, files = checkpoint_fingerprint(destination)
    return {
        **inheritance,
        "checkpoint_sha256": fingerprint,
        "checkpoint_files": files,
        "adapter_sha256": sha256_file(destination / "adapter.safetensors"),
        "runtime_metadata_sha256": sha256_file(destination / RUNTIME_METADATA_FILENAME),
        "exact_two_file_checkpoint": True,
    }


def _cache_manifest(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if EVALUATION_CACHE.is_symlink() or not EVALUATION_CACHE.is_dir():
        raise FileNotFoundError("V94 attested evaluation cache is unavailable")
    expected_files = {"manifest.json", *(f"{scene}.safetensors" for scene in SCENE_IDS)}
    if {item.name for item in EVALUATION_CACHE.iterdir()} != expected_files:
        raise ValueError("V94 evaluation cache file inventory changed")
    manifest_path = EVALUATION_CACHE / "manifest.json"
    if sha256_file(manifest_path) != evidence.get("cache_manifest_sha256"):
        raise ValueError("V94 cache manifest differs from authenticated evidence")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("artifact") != "v94_question_independent_evaluation_memory_cache_v1"
        or manifest.get("scene_ids") != list(SCENE_IDS)
        or manifest.get("scene_count") != 6
        or manifest.get("shape_each") != list(MEMORY_SHAPE)
        or manifest.get("dtype") != "bfloat16"
        or manifest.get("compiled_before_questions") is not True
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_retrieval") is not False
        or manifest.get("all_memory_slots_retained") is not True
        or manifest.get("environmental_text_inputs") != []
        or manifest.get("source_controller_weights_sha256") != SOURCE_CONTROLLER_WEIGHTS_SHA256
        or manifest.get("source_probe_weights_sha256") != SOURCE_PROBE_TENSOR_SHA256
        or set(manifest.get("scenes", {})) != set(SCENE_IDS)
    ):
        raise ValueError("V94 cache manifest contract changed")
    return manifest


def _load_attested_cache_memory(
    scene_id: str, evidence: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    if scene_id not in SCENE_IDS:
        raise ValueError(f"V94 release scene is outside the attested set: {scene_id}")
    entry = manifest["scenes"][scene_id]
    if not isinstance(entry, Mapping) or set(entry) != {
        "filename",
        "file_sha256",
        "file_size_bytes",
        "memory_sha256",
    }:
        raise ValueError(f"V94 cache entry changed for {scene_id}")
    path = EVALUATION_CACHE / f"{scene_id}.safetensors"
    if (
        entry.get("filename") != path.name
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != entry.get("file_size_bytes")
        or sha256_file(path) != entry.get("file_sha256")
        or entry.get("memory_sha256") != evidence["memory_sha256"][scene_id]
    ):
        raise ValueError(f"V94 cache bytes changed for {scene_id}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"scene_memory"} or handle.metadata() != _CACHE_TENSOR_METADATA:
            raise ValueError(f"V94 cache tensor contract changed for {scene_id}")
    memory = load_file(str(path), device="cpu")["scene_memory"].contiguous()
    if (
        tuple(memory.shape) != MEMORY_SHAPE
        or memory.dtype != torch.bfloat16
        or not bool(torch.isfinite(memory).all())
        or prefix_sha256(memory) != entry["memory_sha256"]
    ):
        raise ValueError(f"V94 cached scene memory changed for {scene_id}")
    return memory, dict(entry)


def _package_candidate_memories(
    destination: Path,
    *,
    evidence: Mapping[str, Any],
    checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> dict[str, dict[str, Any]]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    manifest = _cache_manifest(evidence)
    controller_sha, _controller_files = checkpoint_fingerprint(CONTROLLER_CHECKPOINT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    summaries: dict[str, dict[str, Any]] = {}
    try:
        for scene_id in SCENE_IDS:
            memory, source = _load_attested_cache_memory(scene_id, evidence, manifest)
            metadata = save_v81_scene_memory(
                temporary / scene_id,
                memory,
                scene_id=scene_id,
                source_base_checkpoint_sha256=checkpoint_sha256,
                runtime_config_sha256=runtime_config_sha256,
                source_control_checkpoint_sha256=controller_sha,
                source_probe_tensor_sha256=SOURCE_PROBE_TENSOR_SHA256,
            )
            if metadata["canonical_prefix_sha256"] != evidence["memory_sha256"][scene_id]:
                raise RuntimeError(f"V94 V81 packaging changed memory for {scene_id}")
            summaries[scene_id] = {
                "source_cache_file_sha256": source["file_sha256"],
                "packaged_tensor_file_sha256": metadata["tensor_file_sha256"],
                "canonical_prefix_sha256": metadata["canonical_prefix_sha256"],
                "metadata_only_environmental_text_inputs": [],
            }
        if tuple(sorted(item.name for item in temporary.iterdir())) != SCENE_IDS:
            raise RuntimeError("V94 candidate memory bundle is not exactly six scenes")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summaries


def _copy_rebound_v81_memory(
    source: Path,
    destination: Path,
    *,
    scene_id: str,
    source_checkpoint_sha256: str,
    destination_checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> dict[str, Any]:
    """Copy one tensor byte-for-byte while rebinding runtime-only metadata."""

    loaded = load_v81_scene_memory(
        source,
        expected_scene_id=scene_id,
        expected_base_checkpoint_sha256=source_checkpoint_sha256,
        expected_runtime_config_sha256=runtime_config_sha256,
        expected_model_device="cpu",
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    metadata = dict(loaded.metadata)
    metadata["source_base_checkpoint_sha256"] = destination_checkpoint_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        shutil.copyfile(source / MEMORY_FILENAME, temporary / MEMORY_FILENAME)
        _write_json(temporary / METADATA_FILENAME, metadata)
        if sha256_file(temporary / MEMORY_FILENAME) != sha256_file(source / MEMORY_FILENAME):
            raise RuntimeError(f"V94 promoted memory tensor bytes changed for {scene_id}")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    rebound = load_v81_scene_memory(
        destination,
        expected_scene_id=scene_id,
        expected_base_checkpoint_sha256=destination_checkpoint_sha256,
        expected_runtime_config_sha256=runtime_config_sha256,
        expected_model_device="cpu",
    )
    if rebound.metadata["canonical_prefix_sha256"] != loaded.metadata["canonical_prefix_sha256"]:
        raise RuntimeError(f"V94 promoted memory semantic hash changed for {scene_id}")
    return {
        "candidate_tensor_file_sha256": sha256_file(source / MEMORY_FILENAME),
        "release_tensor_file_sha256": sha256_file(destination / MEMORY_FILENAME),
        "canonical_prefix_sha256": rebound.metadata["canonical_prefix_sha256"],
        "tensor_bytes_reused_exactly": True,
        "metadata_only_rebinding": True,
    }


def _promote_memory_bundle(
    *,
    candidate_checkpoint_sha256: str,
    release_checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> dict[str, dict[str, Any]]:
    if RELEASE_MEMORY_ROOT.exists() or RELEASE_MEMORY_ROOT.is_symlink():
        raise FileExistsError(RELEASE_MEMORY_ROOT)
    RELEASE_MEMORY_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{RELEASE_MEMORY_ROOT.name}.", dir=RELEASE_MEMORY_ROOT.parent)
    )
    summaries: dict[str, dict[str, Any]] = {}
    try:
        for scene_id in SCENE_IDS:
            summaries[scene_id] = _copy_rebound_v81_memory(
                CANDIDATE_MEMORY_ROOT / scene_id,
                temporary / scene_id,
                scene_id=scene_id,
                source_checkpoint_sha256=candidate_checkpoint_sha256,
                destination_checkpoint_sha256=release_checkpoint_sha256,
                runtime_config_sha256=runtime_config_sha256,
            )
        os.rename(temporary, RELEASE_MEMORY_ROOT)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summaries


def prepare_candidate() -> dict[str, Any]:
    """Create a candidate only after hardened held-out V94 gates pass."""

    evidence = authenticate_v94_model_gate()
    if any(
        path.exists() or path.is_symlink() for path in (CANDIDATE_CHECKPOINT, CANDIDATE_MEMORY_ROOT)
    ):
        raise FileExistsError("V94 strict runtime candidate destination already exists")
    config = materialize_runtime_config(evidence)
    runtime_config_sha = effective_runtime_config_sha256(config)
    metadata = build_runtime_metadata(
        evidence, promotion=PENDING_DECISION, smoke_report_sha256=None
    )
    checkpoint = _atomic_checkpoint(CANDIDATE_CHECKPOINT, metadata=metadata, evidence=evidence)
    memories = _package_candidate_memories(
        CANDIDATE_MEMORY_ROOT,
        evidence=evidence,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=runtime_config_sha,
    )
    return {
        "phase": "v94_strict_runtime_candidate_prepared",
        "checkpoint": checkpoint,
        "scene_memories": memories,
        "scene_count": 6,
        "runtime_config_sha256": runtime_config_sha,
        "passed": True,
    }


def verify_candidate() -> dict[str, Any]:
    evidence = authenticate_v94_model_gate()
    if not CANDIDATE_CHECKPOINT.is_dir() or not CANDIDATE_MEMORY_ROOT.is_dir():
        raise FileNotFoundError("V94 strict runtime candidate is incomplete")
    if CANDIDATE_MEMORY_ROOT.is_symlink() or {
        item.name for item in CANDIDATE_MEMORY_ROOT.iterdir()
    } != set(SCENE_IDS):
        raise ValueError("V94 candidate memory root is not exactly six opaque scenes")
    config = load_runtime_config(RUNTIME_CONFIG)
    runtime_config_sha = effective_runtime_config_sha256(config)
    metadata = _read_json(CANDIDATE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    expected_metadata = build_runtime_metadata(
        evidence, promotion=PENDING_DECISION, smoke_report_sha256=None
    )
    if metadata != expected_metadata:
        raise ValueError("V94 candidate runtime metadata changed")
    fingerprint, files = checkpoint_fingerprint(CANDIDATE_CHECKPOINT)
    candidate = load_file(str(CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    expected, composition = _composed_adapter(evidence)
    memory: dict[str, dict[str, Any]] = {}
    for scene_id in SCENE_IDS:
        loaded = load_v81_scene_memory(
            CANDIDATE_MEMORY_ROOT / scene_id,
            expected_scene_id=scene_id,
            expected_base_checkpoint_sha256=fingerprint,
            expected_runtime_config_sha256=runtime_config_sha,
            expected_model_device="cpu",
        )
        if loaded.metadata["canonical_prefix_sha256"] != evidence["memory_sha256"][scene_id]:
            raise ValueError(f"V94 candidate memory differs from evidence for {scene_id}")
        memory[scene_id] = {
            "tensor_file_sha256": sha256_file(CANDIDATE_MEMORY_ROOT / scene_id / MEMORY_FILENAME),
            "metadata_sha256": sha256_file(CANDIDATE_MEMORY_ROOT / scene_id / METADATA_FILENAME),
            "canonical_prefix_sha256": loaded.metadata["canonical_prefix_sha256"],
        }
    checks = {
        "exact_two_file_checkpoint": {row["path"] for row in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "exact_eight_bank_order": composition["final_bank_order"] == list(EXPECTED_BANKS),
        "exact_two_v94_tensors_added": composition["added_tensor_count"] == 2,
        "v85_parent_tensors_byte_identical": composition["base_tensors_byte_identical"] is True,
        "exact_tensor_inventory": set(candidate) == set(expected),
        "all_tensor_values_equal": set(candidate) == set(expected)
        and all(torch.equal(candidate[name], expected[name]) for name in candidate),
        "all_eight_final_state_hashes_bound": set(metadata["lora_bank_state_sha256"])
        == set(EXPECTED_BANKS)
        and all(
            row["expected_initial_state_sha256"] == metadata["lora_bank_state_sha256"][row["name"]]
            for row in metadata["lora"]["banks"]
        ),
        "legacy_extension_v28_null_rebound": metadata["lora_bank_state_sha256"][
            "extension_v28_stage_b_query"
        ]
        == metadata["lora"]["banks"][4]["expected_initial_state_sha256"],
        "exact_six_attested_v81_memories": tuple(sorted(memory)) == SCENE_IDS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V94 strict candidate verification failed: {checks}")
    return {
        "phase": "v94_strict_runtime_candidate_verified",
        "checkpoint_sha256": fingerprint,
        "adapter_sha256": sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors"),
        "runtime_metadata_sha256": sha256_file(CANDIDATE_CHECKPOINT / RUNTIME_METADATA_FILENAME),
        "runtime_config_sha256": runtime_config_sha,
        "v94_bridge_state_sha256": evidence["candidate_state_sha256"],
        "scene_memories": memory,
        "checks": checks,
        "passed": True,
    }


def _smoke_command(scene_id: str, *, audit_path: Path, chat_path: Path) -> list[str]:
    python = PROJECT_ROOT / ".venv-gemma4/bin/python"
    command = [
        str(python),
        "-m",
        "semantic_3d_chat.chat.v94_strict_multiscene_cli",
        "--config",
        str(RUNTIME_CONFIG),
        "--scene",
        scene_id,
        "--base-checkpoint",
        str(CANDIDATE_CHECKPOINT),
        "--scene-memory",
        str(CANDIDATE_MEMORY_ROOT / scene_id),
        "--audit-log",
        str(audit_path),
        "--chat-log",
        str(chat_path),
        "--allow-candidate",
    ]
    for question in _SMOKE_QUESTIONS:
        command.extend(("--question", question))
    return command


def _protected_smoke_reads(audit: Mapping[str, Any]) -> list[str]:
    loaded = audit.get("loaded_files")
    if not isinstance(loaded, list):
        return ["<missing-loaded-file-inventory>"]
    explicit = {
        EXPERIMENT_CONFIG.resolve(),
        V94_BRIDGE_CANDIDATE.resolve(),
        EVALUATION_CACHE.resolve(),
        (EVALUATION_CACHE.parent / "evaluation_cache_compilation_attestation").resolve(),
        (PROJECT_ROOT / "reports/gemma4/predictions").resolve(),
        (PROJECT_ROOT / "reports/gemma4/questions").resolve(),
        (PROJECT_ROOT / "data_diverse52/qa").resolve(),
    }
    violations: list[str] = []
    for raw in loaded:
        if not isinstance(raw, str):
            violations.append(str(raw))
            continue
        path = Path(raw).resolve()
        components = {part.casefold() for part in path.parts}
        protected = bool(components & {"oracle", "qa", "scorer"}) or any(
            part.casefold().startswith(".oracle-unavailable-") for part in path.parts
        )
        protected = protected or any(path == root or root in path.parents for root in explicit)
        if protected:
            violations.append(str(path))
    return sorted(set(violations))


def _oracle_directories() -> tuple[Path, ...]:
    candidates = tuple(PROJECT_ROOT.glob("data*/oracle"))
    unsafe = [
        path for path in candidates if path.is_symlink() or (path.exists() and not path.is_dir())
    ]
    if unsafe:
        raise ValueError(f"V94 oracle roots must be physical directories: {unsafe}")
    roots = {path.resolve() for path in candidates if path.is_dir()}
    return tuple(sorted(roots))


def _parse_json_objects(stdout: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def validate_runtime_smoke_report_v94(
    smoke: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    score = _score_evidence(evidence)
    gates = smoke.get("gates")
    records = smoke.get("scenes")
    candidate = verify_candidate()
    if (
        smoke.get("artifact") != "gemma4_v94_strict_runtime_smoke_v1"
        or smoke.get("schema_version") != SCHEMA_VERSION
        or smoke.get("source_v94_evidence_sha256") != evidence.get("bundle_sha256")
        or smoke.get("source_v94_score_sha256") != score.get("score_sha256")
        or smoke.get("v94_bridge_state_sha256") != V94_TRAINED_STATE_SHA256
        or smoke.get("candidate_checkpoint_sha256") != candidate["checkpoint_sha256"]
        or smoke.get("candidate_adapter_sha256") != candidate["adapter_sha256"]
        or smoke.get("scene_ids") != list(SCENE_IDS)
        or smoke.get("questions") != list(_SMOKE_QUESTIONS)
        or smoke.get("expected_answers_supplied_to_children") is not False
        or smoke.get("behavior_assertions_in_children") is not False
        or not isinstance(records, Mapping)
        or tuple(sorted(records)) != SCENE_IDS
        or any(
            not isinstance(records[scene], Mapping)
            or records[scene].get("prefix_hashes")
            != [evidence["memory_sha256"][scene]] * len(_SMOKE_QUESTIONS)
            or records[scene].get("environment_conditioned_input_hashes")
            != [evidence["memory_sha256"][scene]] * len(_SMOKE_QUESTIONS)
            or records[scene].get("chat_sha256")
            != sha256_file(SMOKE_ROOT / "chat" / f"{scene}.jsonl")
            or records[scene].get("audit_sha256")
            != sha256_file(SMOKE_ROOT / "audit" / f"{scene}.json")
            for scene in SCENE_IDS
        )
        or not isinstance(gates, Mapping)
        or set(gates) != _REQUIRED_SMOKE_GATES
        or any(value is not True for value in gates.values())
        or smoke.get("passed") is not True
        or smoke.get("promotion_authorized") is not True
        or smoke.get("held_out_generalization_claim") is not True
    ):
        raise ValueError("V94 isolated runtime smoke did not pass exactly")


def run_smoke() -> dict[str, Any]:
    """Run each candidate scene while every local oracle root is renamed."""

    if SMOKE_REPORT.is_file():
        evidence = authenticate_v94_model_gate()
        existing = _read_json(SMOKE_REPORT)
        validate_runtime_smoke_report_v94(existing, evidence)
        return existing
    if SMOKE_ROOT.exists() or SMOKE_ROOT.is_symlink():
        raise FileExistsError("V94 smoke work root already exists")
    evidence = authenticate_v94_model_gate()
    candidate = verify_candidate()
    python = PROJECT_ROOT / ".venv-gemma4/bin/python"
    if not python.is_file():
        raise FileNotFoundError("V94 local Gemma Python environment is unavailable")
    oracle_roots = _oracle_directories()
    if not oracle_roots:
        raise FileNotFoundError("V94 smoke requires at least one physical oracle directory")
    moves: list[tuple[Path, Path]] = []
    for index, source in enumerate(oracle_roots):
        hidden = source.parent / f".oracle-unavailable-v94-{os.getpid()}-{index}"
        if hidden.exists() or hidden.is_symlink():
            raise FileExistsError(hidden)
        moves.append((source, hidden))
    SMOKE_ROOT.mkdir(parents=True)
    (SMOKE_ROOT / "chat").mkdir()
    (SMOKE_ROOT / "audit").mkdir()
    before_adapter = sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
    before_memories = {
        scene: sha256_file(CANDIDATE_MEMORY_ROOT / scene / MEMORY_FILENAME) for scene in SCENE_IDS
    }
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    completed: dict[str, subprocess.CompletedProcess[str]] = {}
    physically_unavailable = False
    try:
        for source, hidden in moves:
            os.rename(source, hidden)
        physically_unavailable = all(
            not source.exists() and hidden.is_dir() for source, hidden in moves
        )
        for scene_id in SCENE_IDS:
            audit_path = SMOKE_ROOT / "audit" / f"{scene_id}.json"
            chat_path = SMOKE_ROOT / "chat" / f"{scene_id}.jsonl"
            command = _smoke_command(scene_id, audit_path=audit_path, chat_path=chat_path)
            if any(flag in command for flag in ("--expected", "--answer", "--reference")):
                raise RuntimeError("V94 child protocol contains an expectation channel")
            completed[scene_id] = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
    finally:
        for source, hidden in reversed(moves):
            if hidden.exists():
                os.rename(hidden, source)
    failures = {
        scene: {
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        for scene, process in completed.items()
        if process.returncode != 0
    }
    if failures or tuple(sorted(completed)) != SCENE_IDS:
        raise RuntimeError(f"V94 strict runtime child failed: {failures}")

    scenes: dict[str, dict[str, Any]] = {}
    child_oracle_unavailable = True
    exact_banks = True
    candidate_mode = True
    nonempty = True
    prefixes_invariant = True
    prefixes_match = True
    direct_layout = True
    forbidden_clean = True
    protected_clean = True
    for scene_id in SCENE_IDS:
        process = completed[scene_id]
        objects = _parse_json_objects(process.stdout)
        startups = [row for row in objects if row.get("phase") == "v94_strict_multiscene_ready"]
        completions = [row for row in objects if row.get("phase") == "v94_chat_audit_complete"]
        startup = startups[0] if len(startups) == 1 else {}
        completion = completions[0] if len(completions) == 1 else {}
        chat_path = SMOKE_ROOT / "chat" / f"{scene_id}.jsonl"
        audit_path = SMOKE_ROOT / "audit" / f"{scene_id}.json"
        rows = [
            json.loads(line)
            for line in chat_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        audit = _read_json(audit_path)
        expected_hash = evidence["memory_sha256"][scene_id]
        prefix_hashes = [row.get("prefix_hash") for row in rows]
        input_hashes = [row.get("environment_conditioned_input_sha256") for row in rows]
        child_oracle_unavailable &= (
            startup.get("oracle_directory_available_at_runtime_start") is False
        )
        exact_banks &= (
            startup.get("frozen_lora_bank_count") == 8
            and startup.get("trainable_runtime_parameter_count") == 0
            and startup.get("v94_bridge_state_sha256") == V94_TRAINED_STATE_SHA256
        )
        candidate_mode &= (
            startup.get("runtime_package_mode") == "candidate"
            and completion.get("runtime_package_mode") == "candidate"
            and startup.get("runtime_promotion_authorized") is False
        )
        nonempty &= len(rows) == len(_SMOKE_QUESTIONS) and all(
            isinstance(row.get("answer"), str) and bool(row["answer"].strip()) for row in rows
        )
        prefixes_invariant &= (
            prefix_hashes == input_hashes
            and len(set(prefix_hashes)) == 1
            and completion.get("prefix_hash_invariant") is True
        )
        prefixes_match &= set(prefix_hashes) == {expected_hash}
        direct_layout &= (
            all(
                isinstance(row.get("prepared_layout_audit"), Mapping)
                and row["prepared_layout_audit"].get("fixed_scene_memory_tokens_supplied_to_gemma")
                == 738
                and row["prepared_layout_audit"].get("question_derived_environmental_tokens") == 0
                for row in rows
            )
            and completion.get("exact_738_token_memory_supplied_directly_to_gemma") is True
        )
        forbidden_clean &= (
            audit.get("passed") is True
            and audit.get("forbidden_accesses") == []
            and completion.get("forbidden_access_count") == 0
        )
        protected = _protected_smoke_reads(audit)
        protected_clean &= protected == []
        scenes[scene_id] = {
            "returncode": process.returncode,
            "chat_sha256": sha256_file(chat_path),
            "audit_sha256": sha256_file(audit_path),
            "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
            "prefix_hashes": prefix_hashes,
            "environment_conditioned_input_hashes": input_hashes,
            "protected_reads": protected,
        }
    after_memories = {
        scene: sha256_file(CANDIDATE_MEMORY_ROOT / scene / MEMORY_FILENAME) for scene in SCENE_IDS
    }
    gates = {
        "hardened_score_and_behavior_evidence_passed": evidence["behavior_gate_passed"] is True,
        "all_six_runtime_processes_exit_zero": all(
            row.returncode == 0 for row in completed.values()
        ),
        "all_oracle_directories_physically_unavailable": physically_unavailable,
        "all_oracle_directories_restored": all(source.is_dir() for source, _ in moves),
        "all_six_children_report_oracle_unavailable": child_oracle_unavailable,
        "all_six_children_use_exact_eight_frozen_banks": exact_banks,
        "all_six_children_report_candidate_mode": candidate_mode,
        "all_twelve_questions_return_nonempty_answers": nonempty,
        "every_scene_prefix_is_invariant": prefixes_invariant,
        "every_scene_prefix_matches_attested_cache_memory": prefixes_match,
        "direct_memory_layout_retained_for_every_answer": direct_layout,
        "file_audit_forbidden_read_count_zero": forbidden_clean,
        "file_audit_protected_read_count_zero": protected_clean,
        "candidate_adapter_bytes_unchanged": before_adapter
        == sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
        == candidate["adapter_sha256"],
        "candidate_memory_tensor_bytes_unchanged": before_memories == after_memories,
        "no_expectation_channel_in_child_protocol": all(
            all(
                flag
                not in _smoke_command(
                    scene,
                    audit_path=SMOKE_ROOT / "audit" / f"{scene}.json",
                    chat_path=SMOKE_ROOT / "chat" / f"{scene}.jsonl",
                )
                for flag in ("--expected", "--answer", "--reference")
            )
            for scene in SCENE_IDS
        ),
    }
    score = _score_evidence(evidence)
    report = {
        "artifact": "gemma4_v94_strict_runtime_smoke_v1",
        "schema_version": SCHEMA_VERSION,
        "source_v94_evidence_sha256": evidence["bundle_sha256"],
        "source_v94_score_sha256": score["score_sha256"],
        "v94_bridge_state_sha256": V94_TRAINED_STATE_SHA256,
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_adapter_sha256": candidate["adapter_sha256"],
        "scene_ids": list(SCENE_IDS),
        "questions": list(_SMOKE_QUESTIONS),
        "expected_answers_supplied_to_children": False,
        "behavior_assertions_in_children": False,
        "scenes": scenes,
        "gates": gates,
        "passed": all(gates.values()),
        "promotion_authorized": all(gates.values()),
        "held_out_generalization_claim": True,
    }
    _write_json(SMOKE_REPORT, report)
    return report


def promote_release() -> dict[str, Any]:
    evidence = authenticate_v94_model_gate()
    if any(
        path.exists() or path.is_symlink()
        for path in (RELEASE_CHECKPOINT, RELEASE_MEMORY_ROOT, RELEASE_REPORT)
    ):
        raise FileExistsError("V94 strict runtime release destination already exists")
    smoke = _read_json(SMOKE_REPORT)
    validate_runtime_smoke_report_v94(smoke, evidence)
    candidate = verify_candidate()
    if (
        smoke.get("candidate_checkpoint_sha256") != candidate["checkpoint_sha256"]
        or smoke.get("candidate_adapter_sha256") != candidate["adapter_sha256"]
    ):
        raise ValueError("V94 smoked candidate bytes changed before promotion")
    smoke_sha = sha256_file(SMOKE_REPORT)
    metadata = build_runtime_metadata(
        evidence,
        promotion=PROMOTION_DECISION,
        smoke_report_sha256=smoke_sha,
    )
    checkpoint = _atomic_checkpoint(
        RELEASE_CHECKPOINT,
        metadata=metadata,
        evidence=evidence,
        source_adapter=CANDIDATE_CHECKPOINT / "adapter.safetensors",
    )
    if checkpoint["adapter_sha256"] != candidate["adapter_sha256"]:
        raise RuntimeError("Promoted V94 adapter differs from smoked candidate")
    runtime_config_sha = effective_runtime_config_sha256(load_runtime_config(RUNTIME_CONFIG))
    memories = _promote_memory_bundle(
        candidate_checkpoint_sha256=str(candidate["checkpoint_sha256"]),
        release_checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=runtime_config_sha,
    )
    score = _score_evidence(evidence)
    release = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "promotion_decision": PROMOTION_DECISION,
        "promotion_scope": "strict_direct_continuous_scene_memory_scenes_57_through_62",
        "scene_ids": list(SCENE_IDS),
        "scene_count": 6,
        "held_out_generalization_claim": True,
        "strict_input_contract": {
            "shape_each": list(MEMORY_SHAPE),
            "continuous_environment_payload_tokens": 736,
            "native_boi_eoi_retained": True,
            "compiled_before_question": True,
            "same_exact_memory_reused_for_every_question_per_scene": True,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "environmental_text_inputs": [],
        },
        "runtime_config": _relative(RUNTIME_CONFIG),
        "runtime_config_sha256": runtime_config_sha,
        "checkpoint": checkpoint,
        "scene_memories": memories,
        "bindings": {
            "source_v94_evidence_sha256": evidence["bundle_sha256"],
            "source_v94_score_sha256": score["score_sha256"],
            "runtime_smoke_sha256": smoke_sha,
            "v94_bridge_state_sha256": V94_TRAINED_STATE_SHA256,
        },
        "chat_runtime_loads_training_or_evaluation_reports": False,
        "runtime_checkpoint_contains_environmental_text": False,
        "runtime_checkpoint_contains_supervision": False,
        "scene_memory_metadata_only_rebinding": True,
        "scene_memory_tensor_bytes_unchanged_from_smoked_candidate": True,
        "adapter_bytes_unchanged_from_smoked_candidate": True,
        "all_release_gates_passed": True,
    }
    _write_json(RELEASE_REPORT, release)
    return release


def verify_release() -> dict[str, Any]:
    evidence = authenticate_v94_model_gate()
    smoke = _read_json(SMOKE_REPORT)
    validate_runtime_smoke_report_v94(smoke, evidence)
    release = _read_json(RELEASE_REPORT)
    smoke_sha = sha256_file(SMOKE_REPORT)
    metadata = _read_json(RELEASE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    expected_metadata = build_runtime_metadata(
        evidence,
        promotion=PROMOTION_DECISION,
        smoke_report_sha256=smoke_sha,
    )
    if metadata != expected_metadata:
        raise ValueError("V94 promoted runtime metadata changed")
    fingerprint, files = checkpoint_fingerprint(RELEASE_CHECKPOINT)
    candidate_fingerprint, _candidate_files = checkpoint_fingerprint(CANDIDATE_CHECKPOINT)
    config = load_runtime_config(RUNTIME_CONFIG)
    runtime_config_sha = effective_runtime_config_sha256(config)
    if RELEASE_MEMORY_ROOT.is_symlink() or {
        item.name for item in RELEASE_MEMORY_ROOT.iterdir()
    } != set(SCENE_IDS):
        raise ValueError("V94 release memory root is not exactly six opaque scenes")
    parent = load_file(str(PARENT_CHECKPOINT / "adapter.safetensors"), device="cpu")
    promoted = load_file(str(RELEASE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    memory_equal = True
    memory_bound = True
    for scene_id in SCENE_IDS:
        loaded = load_v81_scene_memory(
            RELEASE_MEMORY_ROOT / scene_id,
            expected_scene_id=scene_id,
            expected_base_checkpoint_sha256=fingerprint,
            expected_runtime_config_sha256=runtime_config_sha,
            expected_model_device="cpu",
        )
        memory_equal &= sha256_file(
            RELEASE_MEMORY_ROOT / scene_id / MEMORY_FILENAME
        ) == sha256_file(CANDIDATE_MEMORY_ROOT / scene_id / MEMORY_FILENAME)
        memory_bound &= (
            loaded.metadata["canonical_prefix_sha256"] == evidence["memory_sha256"][scene_id]
        )
    provenance = metadata["initialization_provenance"]["v94_strict_runtime_release"]
    checks = {
        "release_report_identity": release.get("artifact") == ARTIFACT
        and release.get("schema_version") == SCHEMA_VERSION
        and release.get("all_release_gates_passed") is True,
        "release_report_promoted": release.get("promotion_decision") == PROMOTION_DECISION,
        "exact_two_file_checkpoint": {row["path"] for row in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "checkpoint_fingerprint_matches_release": fingerprint
        == release.get("checkpoint", {}).get("checkpoint_sha256"),
        "adapter_byte_identical_to_smoked_candidate": sha256_file(
            RELEASE_CHECKPOINT / "adapter.safetensors"
        )
        == sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
        == smoke["candidate_adapter_sha256"],
        "v85_parent_tensors_byte_identical": set(parent).issubset(promoted)
        and all(torch.equal(promoted[name], value) for name, value in parent.items()),
        "exact_two_v94_tensors_added": len(set(promoted) - set(parent)) == 2,
        "all_six_memory_tensor_files_byte_identical_to_candidate": memory_equal,
        "all_six_memories_bound_to_attested_prefixes": memory_bound,
        "exact_eight_frozen_final_state_banks": tuple(
            row["name"] for row in metadata["lora"]["banks"]
        )
        == EXPECTED_BANKS
        and metadata["lora"]["adapter_parameter_count"] == EXPECTED_ADAPTER_PARAMETER_COUNT
        and metadata["lora"]["trainable_adapter_parameter_count"] == 0
        and all(
            row["expected_initial_state_sha256"] == metadata["lora_bank_state_sha256"][row["name"]]
            for row in metadata["lora"]["banks"]
        ),
        "hardened_evidence_binding_exact": provenance["source_v94_evidence_sha256"]
        == evidence["bundle_sha256"],
        "score_binding_exact": provenance["source_v94_score_sha256"]
        == evidence["score"]["score_sha256"],
        "runtime_smoke_binding_exact": provenance["smoke_report_sha256"]
        == smoke_sha
        == release["bindings"]["runtime_smoke_sha256"],
        "v94_state_binding_exact": provenance["v94_bridge_state_sha256"]
        == V94_TRAINED_STATE_SHA256,
        "runtime_promotion_authorized": provenance["runtime_promotion_authorized"] is True,
        "candidate_checkpoint_identity_retained_in_smoke": candidate_fingerprint
        == smoke["candidate_checkpoint_sha256"],
        "held_out_generalization_claim": release.get("held_out_generalization_claim") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V94 strict runtime release verification failed: {checks}")
    return {
        "phase": "v94_strict_runtime_release_verified",
        "checks": checks,
        "passed": True,
    }


def cleanup_failed_candidate() -> None:
    """Remove only an un-smoked, unpromoted partial V94 candidate."""

    if any(
        path.exists() or path.is_symlink()
        for path in (SMOKE_REPORT, RELEASE_REPORT, RELEASE_CHECKPOINT, RELEASE_MEMORY_ROOT)
    ):
        raise RuntimeError("Refusing V94 cleanup after smoke or release evidence exists")
    for root in (CANDIDATE_CHECKPOINT, CANDIDATE_MEMORY_ROOT, SMOKE_ROOT):
        if root.is_symlink():
            raise ValueError(f"Refusing to clean symbolic-link V94 candidate: {root}")
        if root.exists():
            shutil.rmtree(root)
    if RUNTIME_CONFIG.exists():
        if RUNTIME_CONFIG.is_symlink():
            raise ValueError("Refusing to clean symbolic-link V94 runtime config")
        RUNTIME_CONFIG.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "authenticate",
            "prepare",
            "verify-candidate",
            "smoke",
            "promote",
            "verify",
            "cleanup-failed-candidate",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    functions = {
        "authenticate": authenticate_v94_model_gate,
        "prepare": prepare_candidate,
        "verify-candidate": verify_candidate,
        "smoke": run_smoke,
        "promote": promote_release,
        "verify": verify_release,
        "cleanup-failed-candidate": lambda: (
            cleanup_failed_candidate() or {"phase": "v94_failed_candidate_cleaned", "passed": True}
        ),
    }
    try:
        result = functions[args.command]()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V94 strict runtime {args.command} refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if args.command == "smoke" and result.get("passed") is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "CANDIDATE_CHECKPOINT",
    "CANDIDATE_MEMORY_ROOT",
    "EXPECTED_BANKS",
    "PARENT_BANKS",
    "RELEASE_CHECKPOINT",
    "RELEASE_MEMORY_ROOT",
    "RELEASE_REPORT",
    "RUNTIME_CONFIG",
    "SCENE_IDS",
    "SMOKE_REPORT",
    "V94_BANK",
    "V94_TARGET",
    "V94_TRAINED_STATE_SHA256",
    "authenticate_v94_model_gate",
    "build_runtime_config_payload",
    "build_runtime_metadata",
    "cleanup_failed_candidate",
    "main",
    "materialize_runtime_config",
    "prepare_candidate",
    "promote_release",
    "run_smoke",
    "validate_runtime_smoke_report_v94",
    "verify_candidate",
    "verify_release",
]

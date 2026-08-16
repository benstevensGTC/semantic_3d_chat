"""Promotion-gated V96 chat over one immutable continuous scene memory.

This module is deliberately inference-only.  It never imports the V96
experiment, trainer, evaluator, question manifest, labels, predictions, or
oracle data.  A separate model-free process must first authenticate the sealed
V96 fixed-final candidate and its known-development PASS evidence.  Only the
resulting hash-only authorization is accepted here.

The runtime starts from the exact frozen seven-bank V85 package, adds the
frozen V94, V95, and V96 banks, and supplies one complete numeric
``[1, 738, 1536]`` V81 memory directly to Gemma for every question.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    V83DirectSceneMemoryChatRuntime,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    LoadedV81SceneMemory,
    load_v81_scene_memory,
)

RUNTIME_KIND: Final[str] = "v96_explicit_candidate_direct_scene_memory"
AUTHORIZATION_ARTIFACT: Final[str] = "gemma4_v96_explicit_candidate_authorization_v1"
AUTHORIZATION_STATUS: Final[str] = "authenticated_pass_unpromoted_explicit_use_only"
FINAL_GATE_ARTIFACT: Final[str] = "gemma4_v96_known_development_gate_v1"
EVIDENCE_ARTIFACT: Final[str] = "gemma4_v96_known_development_evidence_v1"
FINAL_GATE_PASS_STATUS: Final[str] = (
    "passed_deferred_final_explicit_unlock_eligible"
)

V94_BANK: Final[str] = "v94_strict_multiscene_full40_bridge"
V94_TARGETS: Final[tuple[str, ...]] = (
    "model.language_model.layers.34.mlp.gate_proj",
)
V94_STATE_SHA256: Final[str] = (
    "9f503f0b2c605238a6f32c15740c0600702d46da08a527d867fbc19e6b639452"
)
V94_PARAMETER_COUNT: Final[int] = 110_592

V95_BANK: Final[str] = "v95_strict_causal_successor_bridge"
V95_TARGETS: Final[tuple[str, ...]] = (
    "model.language_model.layers.9.self_attn.k_proj",
    "model.language_model.layers.9.self_attn.v_proj",
    "model.language_model.layers.34.mlp.up_proj",
)
V95_PARAMETER_COUNT: Final[int] = 143_360

V96_BANK: Final[str] = "v96_atomic_pair_repair_bridge"
V96_TARGETS: Final[tuple[str, ...]] = (
    "model.language_model.layers.9.self_attn.q_proj",
)
V96_PARAMETER_COUNT: Final[int] = 45_056

BASE_PARAMETER_COUNT: Final[int] = 565_248
EXTENSION_PARAMETER_COUNT: Final[int] = (
    V94_PARAMETER_COUNT + V95_PARAMETER_COUNT + V96_PARAMETER_COUNT
)
TOTAL_PARAMETER_COUNT: Final[int] = BASE_PARAMETER_COUNT + EXTENSION_PARAMETER_COUNT

_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_RUNTIME_COMPONENTS: Final[frozenset[str]] = frozenset(
    {
        "oracle",
        "qa",
        "rendered",
        "features",
        "training",
        "scorer",
        "predictions",
    }
)


@dataclass(frozen=True)
class _BaseBankSpec:
    name: str
    targets: tuple[str, ...]
    rank: int
    alpha: float
    parameter_count: int
    state_sha256: str


_BASE_BANK_SPECS: Final[tuple[_BaseBankSpec, ...]] = (
    _BaseBankSpec(
        "inherited_v12",
        (
            "model.language_model.layers.34.self_attn.q_proj",
            "model.language_model.layers.34.self_attn.o_proj",
        ),
        4,
        8.0,
        45_056,
        "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594",
    ),
    _BaseBankSpec(
        "extension_v13",
        tuple(
            f"model.language_model.layers.{layer}.self_attn.{projection}_proj"
            for layer in range(30, 34)
            for projection in ("q", "o")
        ),
        8,
        16.0,
        229_376,
        "4eb90fb9b0bea579d14cfcb0f61ebd5b6d566fd600bd3d5e1bfe5177a39e1b34",
    ),
    _BaseBankSpec(
        "extension_v23_shared_kv",
        tuple(
            f"model.language_model.layers.{layer}.self_attn.{projection}_proj"
            for layer in (13, 14)
            for projection in ("k", "v")
        ),
        4,
        8.0,
        30_720,
        "5c9233bd96b381e2f63443f8a739a868a21997b28061d90fb407a46d9de2d4cb",
    ),
    _BaseBankSpec(
        "extension_v24_shared_query",
        (
            "model.language_model.layers.28.self_attn.q_proj",
            "model.language_model.layers.29.self_attn.q_proj",
        ),
        4,
        8.0,
        36_864,
        "6db2807476506b947bbaf01837490e97c12e57b1906bab671ef7c82ed36d6399",
    ),
    _BaseBankSpec(
        "extension_v28_stage_b_query",
        (
            "model.language_model.layers.13.self_attn.q_proj",
            "model.language_model.layers.14.self_attn.q_proj",
        ),
        4,
        8.0,
        36_864,
        "ac90fc60e944b792d41fc18a21daca3ed87a7ec634a7a5c8594339371b0631e9",
    ),
    _BaseBankSpec(
        "extension_v30_joint_pair_query",
        tuple(
            f"model.language_model.layers.{layer}.self_attn.q_proj"
            for layer in range(18, 22)
        ),
        8,
        16.0,
        131_072,
        "050706c300e6fb0ac8e4cc02e26c565b54a9a89505104302d4ffcedc02124c64",
    ),
    _BaseBankSpec(
        "v85_strict_multiscene_bridge",
        ("model.language_model.layers.34.mlp.down_proj",),
        4,
        8.0,
        55_296,
        "f31b8f99f77f1b7b92dafd74220e5e12ccfa35cbc8630a6d7640f2fe1f93c581",
    ),
)

BASE_BANKS: Final[tuple[str, ...]] = tuple(spec.name for spec in _BASE_BANK_SPECS)
EXPECTED_BANKS: Final[tuple[str, ...]] = BASE_BANKS + (V94_BANK, V95_BANK, V96_BANK)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"V96 authorization {label} must be lowercase SHA-256")
    return str(value)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V96 {label} must be a mapping")
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


def _sha256_file(path: Path, audit: FileAccessAudit | None = None) -> str:
    if audit is not None:
        audit.record(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V96 {label} path contains a symbolic link: {current}")


def _runtime_path(path: str | Path, label: str) -> Path:
    raw = Path(path).expanduser()
    rooted = raw if raw.is_absolute() else PROJECT_ROOT / raw
    source = Path(os.path.abspath(rooted))
    _reject_symlink_components(source, label)
    forbidden = _FORBIDDEN_RUNTIME_COMPONENTS.intersection(
        component.casefold() for component in source.parts
    )
    if forbidden:
        raise ValueError(
            f"V96 refused {label} under forbidden runtime components: {sorted(forbidden)}"
        )
    return source


def _checkpoint_fingerprint(
    checkpoint: Path, audit: FileAccessAudit | None = None
) -> str:
    entries: list[dict[str, Any]] = []
    for name in ("adapter.safetensors", "metadata.json", "runtime_metadata.json"):
        path = checkpoint / name
        if not path.is_file():
            if name in {"adapter.safetensors", "runtime_metadata.json"}:
                raise FileNotFoundError(path)
            continue
        entries.append(
            {
                "path": name,
                "sha256": _sha256_file(path, audit),
                "size_bytes": path.stat().st_size,
            }
        )
    return _canonical_sha256(entries)


def validate_v96_pass_evidence(
    *, candidate: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the official fixed candidate and every preregistered V96 gate."""

    gates = _mapping(evidence.get("gate_results"), "known-development gate results")
    if not gates or any(value is not True for value in gates.values()):
        raise ValueError("V96 explicit runtime requires every known-development gate to PASS")
    candidate_hashes = {
        key: _require_sha256(candidate.get(key), f"candidate {key}")
        for key in (
            "fingerprint_sha256",
            "state_sha256",
            "weights_sha256",
            "metadata_file_sha256",
            "metadata_canonical_sha256",
            "frozen_v95_state_sha256",
            "config_sha256",
            "preregistration_sha256",
            "cpu_preflight_sha256",
            "training_report_sha256",
            "attestation_file_sha256",
            "attestation_identity_sha256",
            "v2_implementation_seal_sha256",
        )
    }
    for key in (
        "final_score_sha256",
        "evidence_sha256",
        "implementation_seal_sha256",
        "implementation_source_inventory_sha256",
        "v1_implementation_seal_sha256",
        "candidate_attestation_file_sha256",
        "candidate_attestation_identity_sha256",
    ):
        _require_sha256(evidence.get(key), f"evidence {key}")
    if (
        candidate.get("artifact") != "gemma4_v96_fixed_final_fingerprint_v1"
        or candidate.get("directory_inventory")
        != ["bridge.safetensors", "runtime_metadata.json"]
        or candidate.get("fixed_final_optimizer_updates") != 285
        or candidate.get("known_development_scored") is not False
        or candidate.get("deferred_final_generated") is not False
        or candidate.get("runtime_promotion_authorized") is not False
        or evidence.get("artifact") != FINAL_GATE_ARTIFACT
        or evidence.get("schema_version") != 96
        or evidence.get("status") != FINAL_GATE_PASS_STATUS
        or evidence.get("authenticated") is not True
        or evidence.get("known_development_gate_passed") is not True
        or evidence.get("deferred_final_unlock_eligible") is not True
        or evidence.get("deferred_final_unlock_requires_explicit_separate_command")
        is not True
        or evidence.get("scene_prefix_question_independent") is not True
        or evidence.get("fixed_final_checkpoint_immutable") is not True
        or evidence.get("candidate_attestation_immutable") is not True
        or evidence.get("frozen_v95_parent_immutable") is not True
        or evidence.get("protected_read_count") != 0
        or evidence.get("row_level_content_serialized") is not False
        or evidence.get("automatic_runtime_promotion") is not False
        or evidence.get("runtime_promotion_authorized") is not False
        or evidence.get("candidate_fingerprint_sha256")
        != candidate_hashes["fingerprint_sha256"]
        or evidence.get("candidate_state_sha256") != candidate_hashes["state_sha256"]
        or evidence.get("frozen_v95_state_sha256")
        != candidate_hashes["frozen_v95_state_sha256"]
        or evidence.get("config_sha256") != candidate_hashes["config_sha256"]
        or evidence.get("preregistration_sha256")
        != candidate_hashes["preregistration_sha256"]
        or evidence.get("cpu_preflight_sha256")
        != candidate_hashes["cpu_preflight_sha256"]
        or evidence.get("training_report_sha256")
        != candidate_hashes["training_report_sha256"]
        or evidence.get("candidate_attestation_file_sha256")
        != candidate_hashes["attestation_file_sha256"]
        or evidence.get("candidate_attestation_identity_sha256")
        != candidate_hashes["attestation_identity_sha256"]
        or evidence.get("implementation_seal_sha256")
        != candidate_hashes["v2_implementation_seal_sha256"]
    ):
        raise ValueError("V96 candidate and authenticated PASS evidence are not mutually bound")
    return {
        "candidate_fingerprint_sha256": candidate_hashes["fingerprint_sha256"],
        "candidate_state_sha256": candidate_hashes["state_sha256"],
        "frozen_v95_state_sha256": candidate_hashes["frozen_v95_state_sha256"],
        "candidate_attestation_file_sha256": candidate_hashes[
            "attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": candidate_hashes[
            "attestation_identity_sha256"
        ],
        "v1_implementation_seal_sha256": str(
            evidence["v1_implementation_seal_sha256"]
        ),
        "v2_implementation_seal_sha256": candidate_hashes[
            "v2_implementation_seal_sha256"
        ],
        "gate_results_sha256": _canonical_sha256(dict(gates)),
        "gate_count": len(gates),
        "all_gate_results_passed": True,
    }


@dataclass(frozen=True)
class V96CandidateAuthorization:
    """Hash-only result of isolated official V96 authentication."""

    artifact: str
    schema_version: int
    status: str
    authorization_config_path: str
    authorization_config_sha256: str
    runtime_config_path: str
    runtime_config_file_sha256: str
    runtime_config_effective_sha256: str
    v85_checkpoint_path: str
    v85_adapter_sha256: str
    v85_metadata_sha256: str
    v94_bridge_path: str
    v94_weights_sha256: str
    v94_metadata_sha256: str
    v94_state_sha256: str
    v95_bridge_path: str
    v95_weights_sha256: str
    v95_metadata_sha256: str
    v95_state_sha256: str
    v96_candidate_path: str
    v96_weights_sha256: str
    v96_metadata_file_sha256: str
    v96_metadata_canonical_sha256: str
    v96_state_sha256: str
    candidate_fingerprint_sha256: str
    config_sha256: str
    preregistration_sha256: str
    cpu_preflight_sha256: str
    training_report_sha256: str
    final_score_path: str
    final_score_sha256: str
    evidence_path: str
    evidence_sha256: str
    implementation_seal_sha256: str
    implementation_source_inventory_sha256: str
    v1_implementation_seal_sha256: str
    v2_implementation_seal_sha256: str
    candidate_attestation_file_sha256: str
    candidate_attestation_identity_sha256: str
    candidate_attestation_immutable: bool
    gate_results_sha256: str
    gate_count: int
    all_gate_results_passed: bool
    candidate_authenticated: bool
    pass_evidence_authenticated: bool
    known_development_gate_passed: bool
    scene_prefix_question_independent: bool
    row_level_content_serialized: bool
    environmental_text_inputs: tuple[str, ...]
    deferred_final_unlock_eligible: bool
    automatic_runtime_promotion: bool
    runtime_promotion_authorized: bool
    explicit_candidate_flag_required: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> V96CandidateAuthorization:
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                "V96 authorization payload fields changed: "
                f"missing={sorted(expected - set(payload))} "
                f"extra={sorted(set(payload) - expected)}"
            )
        values = dict(payload)
        raw_environment = values.get("environmental_text_inputs")
        if not isinstance(raw_environment, (list, tuple)):
            raise TypeError("V96 authorization environmental_text_inputs must be a sequence")
        values["environmental_text_inputs"] = tuple(raw_environment)
        result = cls(**values)
        result.validate()
        return result

    def to_payload(self) -> dict[str, Any]:
        return {
            field.name: (
                list(value)
                if field.name == "environmental_text_inputs"
                else value
            )
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }

    def validate(self) -> None:
        for field in fields(self):
            if field.name.endswith("sha256"):
                _require_sha256(getattr(self, field.name), field.name)
        if (
            self.artifact != AUTHORIZATION_ARTIFACT
            or self.schema_version != 96
            or self.status != AUTHORIZATION_STATUS
            or self.v94_state_sha256 != V94_STATE_SHA256
            or isinstance(self.gate_count, bool)
            or not isinstance(self.gate_count, int)
            or self.gate_count < 1
            or self.all_gate_results_passed is not True
            or self.candidate_authenticated is not True
            or self.pass_evidence_authenticated is not True
            or self.known_development_gate_passed is not True
            or self.scene_prefix_question_independent is not True
            or self.candidate_attestation_immutable is not True
            or self.implementation_seal_sha256
            != self.v2_implementation_seal_sha256
            or self.row_level_content_serialized is not False
            or self.environmental_text_inputs != ()
            or self.deferred_final_unlock_eligible is not True
            or self.automatic_runtime_promotion is not False
            or self.runtime_promotion_authorized is not False
            or self.explicit_candidate_flag_required is not True
        ):
            raise ValueError("V96 authorization does not represent an authenticated PASS candidate")
        for name in (
            "runtime_config_path",
            "v85_checkpoint_path",
            "v94_bridge_path",
            "v95_bridge_path",
            "v96_candidate_path",
        ):
            _runtime_path(getattr(self, name), name)


def frozen_v96_extension_settings(
    authorization: V96CandidateAuthorization,
) -> LoRABanksSettings:
    """Construct only V94/V95/V96; V85's seven banks are already installed."""

    authorization.validate()
    specs = (
        (V94_BANK, V94_TARGETS, V94_PARAMETER_COUNT, V94_STATE_SHA256),
        (V95_BANK, V95_TARGETS, V95_PARAMETER_COUNT, authorization.v95_state_sha256),
        (V96_BANK, V96_TARGETS, V96_PARAMETER_COUNT, authorization.v96_state_sha256),
    )
    banks = tuple(
        LoRABankSettings(
            name=name,
            trainable=False,
            adapter=LoRASettings(
                enabled=True,
                rank=8,
                alpha=16.0,
                dropout=0.0,
                target_modules=targets,
            ),
            initialization_algorithm="checkpoint_overwrite",
            expected_initial_state_sha256=state_sha256,
        )
        for name, targets, _count, state_sha256 in specs
    )
    settings = LoRABanksSettings(banks)
    dimensions = {
        V94_BANK: ((1536, 12288),),
        V95_BANK: ((1536, 512), (1536, 512), (1536, 12288)),
        V96_BANK: ((1536, 4096),),
    }
    calculated = sum(
        8 * sum(input_dim + output_dim for input_dim, output_dim in dimensions[name])
        for name, _targets, _count, _state_sha256 in specs
    )
    # The explicit formula above is intentionally checked once here; runtime
    # installation below validates the actual Gemma module dimensions.
    if calculated != EXTENSION_PARAMETER_COUNT:
        raise RuntimeError("V96 extension parameter-count contract drifted")
    return settings


def validate_v96_v85_base_checkpoint_contract(metadata: Mapping[str, Any]) -> None:
    """Require the exact seven frozen banks before installing successors."""

    lora = _mapping(metadata.get("lora"), "V85 LoRA metadata")
    rows_raw = lora.get("banks")
    if not isinstance(rows_raw, list) or not all(
        isinstance(row, Mapping) for row in rows_raw
    ):
        raise TypeError("V96 V85 bank inventory is malformed")
    rows: list[Mapping[str, Any]] = rows_raw
    states = _mapping(metadata.get("lora_bank_state_sha256"), "V85 state inventory")
    modules = _mapping(metadata.get("lora_bank_wrapped_modules"), "V85 module inventory")
    counts = _mapping(metadata.get("lora_bank_parameter_counts"), "V85 count inventory")
    if (
        tuple(str(row.get("name")) for row in rows) != BASE_BANKS
        or set(states) != set(BASE_BANKS)
        or set(modules) != set(BASE_BANKS)
        or set(counts) != set(BASE_BANKS)
        or lora.get("schema_version") != 2
        or lora.get("enabled") is not True
        or lora.get("adapter_parameter_count") != BASE_PARAMETER_COUNT
        or lora.get("trainable_adapter_parameter_count") != 0
        or metadata.get("lora_parameter_count") != BASE_PARAMETER_COUNT
        or metadata.get("lora_trainable_parameter_count") != 0
    ):
        raise ValueError("V96 requires the exact ordered frozen seven-bank V85 base")
    by_name = {str(row["name"]): row for row in rows}
    for spec in _BASE_BANK_SPECS:
        row = by_name[spec.name]
        count_row = _mapping(counts[spec.name], f"V85 counts {spec.name}")
        if (
            row.get("trainable") is not False
            or row.get("rank") != spec.rank
            or float(row.get("alpha", -1.0)) != spec.alpha
            or float(row.get("dropout", -1.0)) != 0.0
            or tuple(row.get("target_modules", ())) != spec.targets
            or row.get("adapter_parameter_count") != spec.parameter_count
            or states.get(spec.name) != spec.state_sha256
            or tuple(modules.get(spec.name, ())) != spec.targets
            or set(count_row) != set(spec.targets)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in count_row.values()
            )
            or sum(count_row.values()) != spec.parameter_count
        ):
            raise ValueError(f"V96 frozen V85 bank changed: {spec.name}")


def _validate_bridge_metadata(
    metadata: Mapping[str, Any],
    *,
    schema_version: int,
    artifact: str,
    status: str,
    bank_name: str,
    targets: tuple[str, ...],
    parameter_count: int,
    state_sha256: str,
    weights_sha256: str,
) -> None:
    target_matches = (
        metadata.get("target_module") == targets[0]
        if schema_version == 94
        else metadata.get("target_modules") == list(targets)
    )
    score_field = "evaluation_scored" if schema_version == 94 else "known_development_scored"
    if (
        metadata.get("schema_version") != schema_version
        or metadata.get("artifact") != artifact
        or metadata.get("status") != status
        or metadata.get("bank_name") != bank_name
        or not target_matches
        or metadata.get("rank") != 8
        or float(metadata.get("alpha", -1.0)) != 16.0
        or float(metadata.get("dropout", -1.0)) != 0.0
        or metadata.get("parameter_count") != parameter_count
        or metadata.get("state_sha256") != state_sha256
        or metadata.get("weights_sha256") != weights_sha256
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get(score_field) is not False
        or metadata.get("runtime_promotion_authorized") is not False
        or (schema_version >= 95 and metadata.get("deferred_final_generated") is not False)
    ):
        raise ValueError(f"V96 rejected exact frozen bridge metadata: {bank_name}")


def _load_exact_bridge(
    collection: LoRABankCollection,
    *,
    root: str | Path,
    bank_name: str,
    schema_version: int,
    artifact: str,
    status: str,
    targets: tuple[str, ...],
    parameter_count: int,
    state_sha256: str,
    expected_weights_sha256: str,
    expected_metadata_sha256: str,
    audit: FileAccessAudit | None,
) -> dict[str, str]:
    source = _runtime_path(root, f"{bank_name} bridge")
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError(source)
    if {path.name for path in source.iterdir()} != {
        "bridge.safetensors",
        "runtime_metadata.json",
    }:
        raise ValueError(f"V96 {bank_name} bridge file inventory changed")
    weights = source / "bridge.safetensors"
    metadata_path = source / "runtime_metadata.json"
    if any(path.is_symlink() or not path.is_file() for path in (weights, metadata_path)):
        raise ValueError(f"V96 {bank_name} bridge entries must be regular files")
    weights_hash = _sha256_file(weights, audit)
    metadata_hash = _sha256_file(metadata_path, audit)
    if (
        weights_hash != expected_weights_sha256
        or metadata_hash != expected_metadata_sha256
    ):
        raise ValueError(f"V96 {bank_name} authorized source bytes changed")
    if audit is not None:
        audit.record(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise TypeError(f"V96 {bank_name} metadata must be an object")
    _validate_bridge_metadata(
        metadata,
        schema_version=schema_version,
        artifact=artifact,
        status=status,
        bank_name=bank_name,
        targets=targets,
        parameter_count=parameter_count,
        state_sha256=state_sha256,
        weights_sha256=weights_hash,
    )
    installation = collection.bank(bank_name).installation
    expected_state = installation.state_module.state_dict()
    if audit is not None:
        audit.record(weights)
    with safe_open(str(weights), framework="pt", device="cpu") as archive:
        raw_keys = sorted(archive.keys())
        raw = {key: archive.get_tensor(key) for key in raw_keys}
    state = (
        {f"adapters.0.{key}": value for key, value in raw.items()}
        if schema_version == 94
        else raw
    )
    if (
        set(state) != set(expected_state)
        or any(tuple(state[key].shape) != tuple(expected_state[key].shape) for key in state)
        or any(tensor.dtype != torch.float32 for tensor in state.values())
        or any(not bool(torch.isfinite(tensor).all()) for tensor in state.values())
        or tensor_state_sha256(state) != state_sha256
    ):
        raise ValueError(f"V96 {bank_name} tensor inventory, shape, dtype, or state changed")
    installation.state_module.load_state_dict(state, strict=True)
    installation.eval()
    if installation.state_sha256() != state_sha256:
        raise RuntimeError(f"V96 {bank_name} loaded state does not match authorization")
    return {
        "weights_sha256": weights_hash,
        "metadata_sha256": metadata_hash,
        "state_sha256": installation.state_sha256(),
    }


def install_v96_extension_banks(
    model: torch.nn.Module,
    *,
    authorization: V96CandidateAuthorization,
    audit: FileAccessAudit | None = None,
) -> LoRABankCollection:
    """Install and authenticate V94/V95/V96, then freeze the whole model."""

    authorization.validate()
    collection = install_lora_banks(model, frozen_v96_extension_settings(authorization))
    if collection is None:
        raise RuntimeError("V96 extension banks were not installed")
    expected = (
        (
            V94_BANK,
            authorization.v94_bridge_path,
            94,
            "gemma4_v94_strict_multiscene_full40_fixed_final_v1",
            "fixed_final_awaiting_preregistered_acceptance_gates",
            V94_TARGETS,
            V94_PARAMETER_COUNT,
            V94_STATE_SHA256,
            authorization.v94_weights_sha256,
            authorization.v94_metadata_sha256,
        ),
        (
            V95_BANK,
            authorization.v95_bridge_path,
            95,
            "gemma4_v95_strict_causal_successor_fixed_final_v1",
            "fixed_final_awaiting_known_development_gate",
            V95_TARGETS,
            V95_PARAMETER_COUNT,
            authorization.v95_state_sha256,
            authorization.v95_weights_sha256,
            authorization.v95_metadata_sha256,
        ),
        (
            V96_BANK,
            authorization.v96_candidate_path,
            96,
            "gemma4_v96_atomic_pair_repair_fixed_final_v1",
            "fixed_final_awaiting_known_development_gate",
            V96_TARGETS,
            V96_PARAMETER_COUNT,
            authorization.v96_state_sha256,
            authorization.v96_weights_sha256,
            authorization.v96_metadata_file_sha256,
        ),
    )
    for (
        name,
        root,
        schema,
        artifact,
        status,
        targets,
        count,
        state,
        weights,
        metadata,
    ) in expected:
        _load_exact_bridge(
            collection,
            root=root,
            bank_name=name,
            schema_version=schema,
            artifact=artifact,
            status=status,
            targets=targets,
            parameter_count=count,
            state_sha256=state,
            expected_weights_sha256=weights,
            expected_metadata_sha256=metadata,
            audit=audit,
        )
    if (
        collection.bank_names != (V94_BANK, V95_BANK, V96_BANK)
        or collection.parameter_count != EXTENSION_PARAMETER_COUNT
        or collection.trainable_parameter_count != 0
        or collection.state_sha256()
        != {
            V94_BANK: V94_STATE_SHA256,
            V95_BANK: authorization.v95_state_sha256,
            V96_BANK: authorization.v96_state_sha256,
        }
    ):
        raise RuntimeError("V96 exact frozen three-bank extension changed")
    model.requires_grad_(False)
    model.eval()
    collection.eval()
    collection.validate_state()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("V96 chat model retained trainable parameters")
    return collection


def validate_v96_scene_memory_contract(
    *, scene_id: str, loaded: LoadedV81SceneMemory
) -> None:
    """Accept only one complete, question-independent, oracle-free tensor."""

    metadata = loaded.metadata
    if (
        _SCENE_ID.fullmatch(scene_id) is None
        or metadata.get("scene_id") != scene_id
        or tuple(loaded.memory.shape) != (1, 738, 1536)
        or loaded.memory.dtype != torch.bfloat16
        or not bool(torch.isfinite(loaded.memory).all())
        or metadata.get("shape") != [1, 738, 1536]
        or metadata.get("fixed_memory_tokens") != 738
        or metadata.get("hidden_size") != 1536
        or metadata.get("compiled_before_user_question") is not True
        or metadata.get("question_inputs_used_for_compilation") is not False
        or metadata.get("question_dependent_scene_processing") is not False
        or metadata.get("question_dependent_retrieval") is not False
        or metadata.get("semantic_or_spatial_top_k_selection") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_loaded") is not False
    ):
        raise ValueError(
            "V96 requires one complete bfloat16 oracle-free [1,738,1536] scene memory"
        )


class V96ExplicitCandidateChatRuntime(V83DirectSceneMemoryChatRuntime):
    """Direct V81 memory chat with the authenticated frozen V96 stack."""

    def __init__(
        self,
        base: StaticChatRuntime,
        loaded: LoadedV81SceneMemory,
        *,
        authorization: V96CandidateAuthorization,
        extension_banks: LoRABankCollection,
    ) -> None:
        authorization.validate()
        validate_v96_scene_memory_contract(scene_id=base.scene_id, loaded=loaded)
        validate_v96_v85_base_checkpoint_contract(base.checkpoint_metadata)
        if (
            extension_banks.bank_names != (V94_BANK, V95_BANK, V96_BANK)
            or extension_banks.parameter_count != EXTENSION_PARAMETER_COUNT
            or extension_banks.trainable_parameter_count != 0
        ):
            raise ValueError("V96 runtime did not receive its exact frozen extension banks")
        super().__init__(base, loaded)
        self.authorization = authorization
        self.extension_banks = extension_banks
        self.environment_conditioned_input_hashes: list[str] = []

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        scene_id: str,
        *,
        authorization: V96CandidateAuthorization,
        scene_memory: str | Path,
        audit: FileAccessAudit | None = None,
        local_files_only: bool = True,
    ) -> V96ExplicitCandidateChatRuntime:
        """Validate authorization before the first model or scene load."""

        authorization.validate()
        if config.get("_runtime_safe_config") is not True:
            raise ValueError("V96 chat requires a standalone validated runtime config")
        configured_path = Path(str(config.get("_config_path", ""))).resolve()
        authorized_config = Path(authorization.runtime_config_path).resolve()
        if configured_path != authorized_config:
            raise ValueError("V96 runtime config path differs from the authenticated source")
        if (
            _sha256_file(configured_path, audit)
            != authorization.runtime_config_file_sha256
            or effective_runtime_config_sha256(config)
            != authorization.runtime_config_effective_sha256
        ):
            raise ValueError("V96 runtime configuration changed after authorization")

        checkpoint = _runtime_path(
            authorization.v85_checkpoint_path, "authenticated V85 checkpoint"
        )
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        adapter = checkpoint / "adapter.safetensors"
        metadata_path = checkpoint / "runtime_metadata.json"
        if (
            _sha256_file(adapter, audit) != authorization.v85_adapter_sha256
            or _sha256_file(metadata_path, audit) != authorization.v85_metadata_sha256
        ):
            raise ValueError("V96 authenticated V85 checkpoint bytes changed")

        # StaticChatRuntime authenticates and loads the seven-bank V85 package.
        # Authorization and all source hashes have already been checked above.
        base = StaticChatRuntime.load(
            config,
            scene_id,
            checkpoint=checkpoint,
            audit=audit,
            local_files_only=local_files_only,
        )
        validate_v96_v85_base_checkpoint_contract(base.checkpoint_metadata)
        extension = install_v96_extension_banks(
            base.language.model,
            authorization=authorization,
            audit=audit,
        )
        checkpoint_sha256 = _checkpoint_fingerprint(checkpoint, audit)
        loaded = load_v81_scene_memory(
            scene_memory,
            expected_scene_id=scene_id,
            expected_base_checkpoint_sha256=checkpoint_sha256,
            expected_runtime_config_sha256=effective_runtime_config_sha256(config),
            expected_model_device=base.language.device,
            record_file=None if audit is None else audit.record,
        )
        validate_v96_scene_memory_contract(scene_id=scene_id, loaded=loaded)
        return cls(
            base,
            loaded,
            authorization=authorization,
            extension_banks=extension,
        )

    def startup_summary(self) -> dict[str, Any]:
        summary = super().startup_summary()
        return {
            **summary,
            "phase": "v96_explicit_candidate_ready",
            "runtime_kind": RUNTIME_KIND,
            "authorization_status": self.authorization.status,
            "candidate_fingerprint_sha256": (
                self.authorization.candidate_fingerprint_sha256
            ),
            "v96_candidate_state_sha256": self.authorization.v96_state_sha256,
            "v96_final_score_sha256": self.authorization.final_score_sha256,
            "v96_evidence_sha256": self.authorization.evidence_sha256,
            "v96_gate_results_sha256": self.authorization.gate_results_sha256,
            "known_development_gate_passed": True,
            "pass_evidence_authenticated": True,
            "frozen_lora_bank_count": 10,
            "frozen_lora_parameter_count": TOTAL_PARAMETER_COUNT,
            "trainable_runtime_parameter_count": 0,
            "lora_bank_order": list(EXPECTED_BANKS),
            "runtime_package_mode": "explicit_candidate",
            "explicit_candidate_flag_required": True,
            "automatic_runtime_promotion": False,
            "runtime_promotion_authorized": False,
            "exact_total_environment_conditioned_input_sha256": self.scene_prefix_hash,
            "environment_conditioned_input_hashes_observed": list(
                self.environment_conditioned_input_hashes
            ),
            "runtime_loaded_oracle_or_text_metadata": False,
            "runtime_loaded_training_evaluation_or_scorer_files": False,
        }

    def answer(self, question: str) -> ChatAnswer:
        before = prefix_sha256(self.fixed_scene_memory)
        if before != self.scene_prefix_hash:
            raise RuntimeError("V96 fixed scene memory changed before the question")
        result = super().answer(question)
        after = prefix_sha256(self.fixed_scene_memory)
        if result.prefix_hash != before or after != before:
            raise RuntimeError("V96 environment-conditioned input changed across generation")
        self.environment_conditioned_input_hashes.append(after)
        if len(set(self.environment_conditioned_input_hashes)) != 1:
            raise RuntimeError("V96 scene prefix became question-dependent")
        return result


__all__ = [
    "AUTHORIZATION_ARTIFACT",
    "AUTHORIZATION_STATUS",
    "BASE_BANKS",
    "BASE_PARAMETER_COUNT",
    "EXPECTED_BANKS",
    "EXTENSION_PARAMETER_COUNT",
    "RUNTIME_KIND",
    "TOTAL_PARAMETER_COUNT",
    "V94_BANK",
    "V94_STATE_SHA256",
    "V95_BANK",
    "V96_BANK",
    "V96CandidateAuthorization",
    "V96ExplicitCandidateChatRuntime",
    "frozen_v96_extension_settings",
    "install_v96_extension_banks",
    "validate_v96_pass_evidence",
    "validate_v96_scene_memory_contract",
    "validate_v96_v85_base_checkpoint_contract",
]

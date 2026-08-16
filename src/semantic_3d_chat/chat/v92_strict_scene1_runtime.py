"""Strict V92 chat over one immutable pre-question continuous scene memory.

This runtime-only module accepts an already promoted V92 checkpoint.  It
imports no trainer, evaluator, prediction, question, answer, or oracle
surface.  Environmental information reaches local Gemma only through the
complete numeric ``[1, 738, 1536]`` scene memory built before user input.

V90 and V91 were measured development candidates, not independently promoted
runtimes.  Their exact two-tensor bridges are frozen lineage banks beneath the
dynamic V92 bridge.  A release packager must bind V92 identically across the
standalone config, sanitized checkpoint, bank-state table, and authenticated
release provenance before this class can start.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    V83DirectSceneMemoryChatRuntime,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    LoadedV81SceneMemory,
)

RUNTIME_KIND: Final[str] = "v92_strict_scene1_direct_scene_memory"
SCENE_ID: Final[str] = "scene_000001"
V90_BANK: Final[str] = "v90_scene1_conversational_bridge"
V90_TARGET: Final[str] = "model.language_model.layers.28.self_attn.o_proj"
V90_STATE_SHA256: Final[str] = (
    "70e236711d8ac1fe7cf808f6f4e939b29db476016c8ef49db143707df0f3bde7"
)
V90_PARAMETER_COUNT: Final[int] = 28_672
V91_BANK: Final[str] = "v91_scene1_conversational_repair"
V91_TARGET: Final[str] = "model.language_model.layers.33.mlp.down_proj"
V91_STATE_SHA256: Final[str] = (
    "53022311c3bc5e249a6d262fbb19b6e893a6af085be542e4d6941f7a13ea72cd"
)
V91_PARAMETER_COUNT: Final[int] = 221_184
V92_BANK: Final[str] = "v92_scene1_retention_conversation_repair"
V92_TARGET: Final[str] = "model.language_model.layers.29.self_attn.o_proj"
V92_RANK: Final[int] = 8
V92_ALPHA: Final[float] = 16.0
V92_PARAMETER_COUNT: Final[int] = 45_056
EXPECTED_ADAPTER_PARAMETER_COUNT: Final[int] = 1_167_360
PROMOTION_DECISION: Final[str] = (
    "strict_scene1_retention_conversation_repair_primary"
)
EXPECTED_BANKS: Final[tuple[str, ...]] = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
    "extension_v28_stage_b_query",
    "extension_v30_joint_pair_query",
    "v85_strict_multiscene_bridge",
    "v86_scene1_demo_bridge",
    "v87_scene1_balanced_bridge",
    "v88_scene1_augmented_bridge",
    "v89_scene1_retention_bridge",
    V90_BANK,
    V91_BANK,
    V92_BANK,
)
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _BankSpec:
    name: str
    targets: tuple[str, ...]
    rank: int
    alpha: float
    parameter_count: int
    fixed_state_sha256: str | None


_BANK_SPECS: Final[tuple[_BankSpec, ...]] = (
    _BankSpec(
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
    _BankSpec(
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
    _BankSpec(
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
    _BankSpec(
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
    _BankSpec(
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
    _BankSpec(
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
    _BankSpec(
        "v85_strict_multiscene_bridge",
        ("model.language_model.layers.34.mlp.down_proj",),
        4,
        8.0,
        55_296,
        "f31b8f99f77f1b7b92dafd74220e5e12ccfa35cbc8630a6d7640f2fe1f93c581",
    ),
    _BankSpec(
        "v86_scene1_demo_bridge",
        ("model.language_model.layers.34.mlp.up_proj",),
        8,
        16.0,
        110_592,
        "8b6bd801716132c8aac50c6288b9ba588417dc5e6a7c2c15dd9515892f714260",
    ),
    _BankSpec(
        "v87_scene1_balanced_bridge",
        ("model.language_model.layers.34.mlp.gate_proj",),
        8,
        16.0,
        110_592,
        "618c03e102d9a9eb98d405d5a040cd8285194539b5a4043d34f16356ac08769e",
    ),
    _BankSpec(
        "v88_scene1_augmented_bridge",
        ("model.language_model.layers.27.self_attn.q_proj",),
        16,
        32.0,
        57_344,
        "ff311624150056c67ad1c0a06752a77af2de89878778049ae886aa59db3376aa",
    ),
    _BankSpec(
        "v89_scene1_retention_bridge",
        ("model.language_model.layers.27.self_attn.o_proj",),
        8,
        16.0,
        28_672,
        "de2388828b4a95770e6e55639baa4538a8360ab1323b68b05ab915aaaba68bd8",
    ),
    _BankSpec(
        V90_BANK,
        (V90_TARGET,),
        8,
        16.0,
        V90_PARAMETER_COUNT,
        V90_STATE_SHA256,
    ),
    _BankSpec(
        V91_BANK,
        (V91_TARGET,),
        16,
        32.0,
        V91_PARAMETER_COUNT,
        V91_STATE_SHA256,
    ),
    _BankSpec(
        V92_BANK,
        (V92_TARGET,),
        V92_RANK,
        V92_ALPHA,
        V92_PARAMETER_COUNT,
        None,
    ),
)
_REQUIRED_RELEASE_HASHES: Final[tuple[str, ...]] = (
    "experiment_config_sha256",
    "preregistration_sha256",
    "cpu_preflight_sha256",
    "training_report_sha256",
    "model_gate_report_sha256",
    "evaluation_predictions_sha256",
    "v90_bridge_state_sha256",
    "v91_bridge_state_sha256",
    "v92_bridge_state_sha256",
    "smoke_report_sha256",
)


def _lower_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V92 runtime {label} is malformed")
    return value


def validate_v92_runtime_contract(
    *,
    scene_id: str,
    runtime_config: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one sanitized frozen runtime surface without loading a model."""

    if scene_id != SCENE_ID:
        raise ValueError("V92 strict scene-one runtime accepts only scene_000001")
    language = _mapping(runtime_config.get("language"), "language config")
    configured = _mapping(language.get("lora_banks"), "config bank inventory")
    lora = _mapping(checkpoint_metadata.get("lora"), "checkpoint LoRA contract")
    banks_raw = lora.get("banks")
    if not isinstance(banks_raw, list) or not all(
        isinstance(row, Mapping) for row in banks_raw
    ):
        raise TypeError("V92 runtime checkpoint bank inventory is malformed")
    banks: list[Mapping[str, Any]] = banks_raw
    states = _mapping(
        checkpoint_metadata.get("lora_bank_state_sha256"), "bank-state bindings"
    )
    modules = _mapping(
        checkpoint_metadata.get("lora_bank_wrapped_modules"), "bank-module bindings"
    )
    counts = _mapping(
        checkpoint_metadata.get("lora_bank_parameter_counts"),
        "bank parameter bindings",
    )
    bank_names = tuple(str(row.get("name")) for row in banks)
    expected_set = set(EXPECTED_BANKS)
    if (
        lora.get("schema_version") != 2
        or lora.get("enabled") is not True
        or tuple(configured) != EXPECTED_BANKS
        or bank_names != EXPECTED_BANKS
        or set(states) != expected_set
        or set(modules) != expected_set
        or set(counts) != expected_set
        or lora.get("adapter_parameter_count") != EXPECTED_ADAPTER_PARAMETER_COUNT
        or lora.get("trainable_adapter_parameter_count") != 0
        or checkpoint_metadata.get("lora_parameter_count")
        != EXPECTED_ADAPTER_PARAMETER_COUNT
        or checkpoint_metadata.get("lora_trainable_parameter_count") != 0
        or checkpoint_metadata.get("question_dependent_scene_processing") is not False
    ):
        raise ValueError("V92 runtime requires the exact ordered frozen 14-bank stack")

    metadata_by_name = {str(row["name"]): row for row in banks}
    v92_state = states.get(V92_BANK)
    if not _lower_sha256(v92_state):
        raise ValueError("V92 bridge state must be a lowercase SHA-256 digest")
    if states.get(V90_BANK) != V90_STATE_SHA256:
        raise ValueError("V92 runtime requires the exact failed V90 bridge")
    if states.get(V91_BANK) != V91_STATE_SHA256:
        raise ValueError("V92 runtime requires the exact failed V91 bridge")
    for spec in _BANK_SPECS:
        config_row = _mapping(configured.get(spec.name), f"config bank {spec.name}")
        metadata_row = metadata_by_name[spec.name]
        count_row = _mapping(counts.get(spec.name), f"parameter count {spec.name}")
        expected_state = spec.fixed_state_sha256 or str(v92_state)
        if (
            config_row.get("trainable") is not False
            or config_row.get("rank") != spec.rank
            or float(config_row.get("alpha", -1.0)) != spec.alpha
            or float(config_row.get("dropout", -1.0)) != 0.0
            or tuple(config_row.get("target_modules", ())) != spec.targets
            or config_row.get("expected_initial_state_sha256") != expected_state
            or metadata_row.get("trainable") is not False
            or metadata_row.get("rank") != spec.rank
            or float(metadata_row.get("alpha", -1.0)) != spec.alpha
            or float(metadata_row.get("dropout", -1.0)) != 0.0
            or tuple(metadata_row.get("target_modules", ())) != spec.targets
            or metadata_row.get("adapter_parameter_count") != spec.parameter_count
            or metadata_row.get("expected_initial_state_sha256") != expected_state
            or states.get(spec.name) != expected_state
            or tuple(modules.get(spec.name, ())) != spec.targets
            or sum(int(value) for value in count_row.values()) != spec.parameter_count
        ):
            raise ValueError(f"V92 runtime frozen bank changed: {spec.name}")

    provenance_root = _mapping(
        checkpoint_metadata.get("initialization_provenance"),
        "initialization provenance",
    )
    release = _mapping(
        provenance_root.get("v92_strict_runtime_release"), "release provenance"
    )
    if any(not _lower_sha256(release.get(key)) for key in _REQUIRED_RELEASE_HASHES):
        raise ValueError("V92 runtime release hash bindings are incomplete")
    if (
        release.get("schema_version") != 92
        or release.get("v90_bridge_state_sha256") != V90_STATE_SHA256
        or release.get("v91_bridge_state_sha256") != V91_STATE_SHA256
        or release.get("v92_bridge_state_sha256") != v92_state
        or release.get("promotion_decision") != PROMOTION_DECISION
        or release.get("runtime_promotion_authorized") is not True
        or release.get("model_acceptance_gate_passed") is not True
        or release.get("model_gate_report_authenticated") is not True
        or release.get("held_out_generalization_claim") is not False
    ):
        raise ValueError("V92 runtime is not bound to the promoted repair gate")
    return {
        "v92_bridge_state_sha256": str(v92_state),
        "release_provenance": dict(release),
        "frozen_lora_bank_count": len(banks),
        "adapter_parameter_count": EXPECTED_ADAPTER_PARAMETER_COUNT,
        "runtime_promotion_authorized": True,
    }


class V92StrictScene1ChatRuntime(V83DirectSceneMemoryChatRuntime):
    """Direct 738-token memory with the exact frozen fourteen-bank V92 stack."""

    def __init__(self, base: StaticChatRuntime, loaded: LoadedV81SceneMemory) -> None:
        super().__init__(base, loaded)
        if (
            self.scene_id != SCENE_ID
            or loaded.metadata.get("scene_id") != SCENE_ID
            or tuple(self.fixed_scene_memory.shape) != (1, 738, 1536)
            or loaded.metadata.get("shape") != [1, 738, 1536]
            or loaded.metadata.get("fixed_memory_tokens") != 738
            or loaded.metadata.get("hidden_size") != 1536
            or loaded.metadata.get("compiled_before_user_question") is not True
            or loaded.metadata.get("question_inputs_used_for_compilation") is not False
            or loaded.metadata.get("questions_or_answers_serialized") is not False
            or loaded.metadata.get("oracle_loaded") is not False
        ):
            raise ValueError("V92 requires the exact oracle-free scene-one 738-token memory")
        contract = validate_v92_runtime_contract(
            scene_id=self.scene_id,
            runtime_config=self.config,
            checkpoint_metadata=self.base.checkpoint_metadata,
        )
        self.release_provenance = contract["release_provenance"]
        self.runtime_promotion_authorized = contract["runtime_promotion_authorized"]
        self.v92_bridge_state_sha256 = contract["v92_bridge_state_sha256"]
        self.environment_conditioned_input_hashes: list[str] = []

    def startup_summary(self) -> dict[str, Any]:
        summary = super().startup_summary()
        return {
            **summary,
            "phase": "v92_strict_scene1_ready",
            "runtime_kind": RUNTIME_KIND,
            "frozen_lora_bank_count": 14,
            "frozen_lora_parameter_count": EXPECTED_ADAPTER_PARAMETER_COUNT,
            "trainable_runtime_parameter_count": 0,
            "v92_bridge_bank": V92_BANK,
            "v92_bridge_target": V92_TARGET,
            "v92_bridge_state_sha256": self.v92_bridge_state_sha256,
            "v91_parent_bridge_state_sha256": V91_STATE_SHA256,
            "v90_parent_bridge_state_sha256": V90_STATE_SHA256,
            "exact_total_environment_conditioned_input_sha256": self.scene_prefix_hash,
            "environment_conditioned_input_hashes_observed": list(
                self.environment_conditioned_input_hashes
            ),
            "promotion_decision": self.release_provenance["promotion_decision"],
            "runtime_promotion_authorized": self.runtime_promotion_authorized,
            "model_gate_report_authenticated": True,
            "held_out_generalization_claim": False,
            "runtime_loaded_training_or_evaluation_reports": False,
        }

    def answer(self, question: str) -> ChatAnswer:
        result = super().answer(question)
        if self.last_prepared_layout_audit is None:
            raise RuntimeError("V92 generation did not produce a direct-memory layout audit")
        if result.prefix_hash != self.scene_prefix_hash:
            raise RuntimeError("V92 total environment-conditioned input changed")
        self.environment_conditioned_input_hashes.append(result.prefix_hash)
        if len(set(self.environment_conditioned_input_hashes)) != 1:
            raise RuntimeError("V92 environment-conditioned input is question-dependent")
        return result


__all__ = [
    "EXPECTED_ADAPTER_PARAMETER_COUNT",
    "EXPECTED_BANKS",
    "PROMOTION_DECISION",
    "RUNTIME_KIND",
    "SCENE_ID",
    "V90_BANK",
    "V90_PARAMETER_COUNT",
    "V90_STATE_SHA256",
    "V90_TARGET",
    "V91_BANK",
    "V91_PARAMETER_COUNT",
    "V91_STATE_SHA256",
    "V91_TARGET",
    "V92_ALPHA",
    "V92_BANK",
    "V92_PARAMETER_COUNT",
    "V92_RANK",
    "V92_TARGET",
    "V92StrictScene1ChatRuntime",
    "validate_v92_runtime_contract",
]

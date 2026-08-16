"""Standalone V94 chat over one immutable numeric 3D scene memory.

This module is intentionally limited to the inference surface.  It accepts
only the authenticated eight-bank V94 package and a sanitized V81 numeric
memory.  It does not import the V94 experiment, scorer, packager, question
manifest, or any source of environmental text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch

from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    V83DirectSceneMemoryChatRuntime,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    LoadedV81SceneMemory,
)

RUNTIME_KIND: Final[str] = "v94_strict_multiscene_direct_scene_memory"
V94_BANK: Final[str] = "v94_strict_multiscene_full40_bridge"
V94_TARGET: Final[str] = "model.language_model.layers.34.mlp.gate_proj"
V94_STATE_SHA256: Final[str] = (
    "9f503f0b2c605238a6f32c15740c0600702d46da08a527d867fbc19e6b639452"
)
EXPECTED_ADAPTER_PARAMETER_COUNT: Final[int] = 675_840
CANDIDATE_DECISION: Final[str] = "pending_isolated_runtime_smoke"
PROMOTED_DECISION: Final[str] = "strict_multiscene_experimental_primary"
EXPECTED_BANKS: Final[tuple[str, ...]] = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
    "extension_v28_stage_b_query",
    "extension_v30_joint_pair_query",
    "v85_strict_multiscene_bridge",
    V94_BANK,
)
_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _BankSpec:
    name: str
    targets: tuple[str, ...]
    rank: int
    alpha: float
    parameter_count: int
    state_sha256: str


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
        V94_BANK,
        (V94_TARGET,),
        8,
        16.0,
        110_592,
        V94_STATE_SHA256,
    ),
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V94 runtime {label} is malformed")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def validate_v94_scene_memory_contract(
    *, scene_id: str, loaded: LoadedV81SceneMemory
) -> None:
    """Fail closed unless ``loaded`` is the complete text-free V81 memory."""

    metadata = loaded.metadata
    if _SCENE_ID.fullmatch(scene_id) is None:
        raise ValueError("V94 scene ID must be opaque scene_NNNNNN")
    if (
        metadata.get("scene_id") != scene_id
        or tuple(loaded.memory.shape) != (1, 738, 1536)
        or not loaded.memory.is_floating_point()
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
        raise ValueError("V94 requires one complete oracle-free 738-token scene memory")


def validate_v94_runtime_contract(
    *, runtime_config: Mapping[str, Any], checkpoint_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate V94's exact frozen stack and release provenance."""

    language = _mapping(runtime_config.get("language"), "language config")
    configured = _mapping(language.get("lora_banks"), "config bank inventory")
    lora = _mapping(checkpoint_metadata.get("lora"), "checkpoint LoRA contract")
    banks_raw = lora.get("banks")
    if not isinstance(banks_raw, list) or not all(
        isinstance(row, Mapping) for row in banks_raw
    ):
        raise TypeError("V94 runtime checkpoint bank inventory is malformed")
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
    expected_set = set(EXPECTED_BANKS)
    if (
        lora.get("schema_version") != 2
        or lora.get("enabled") is not True
        or tuple(configured) != EXPECTED_BANKS
        or tuple(str(row.get("name")) for row in banks) != EXPECTED_BANKS
        or set(states) != expected_set
        or set(modules) != expected_set
        or set(counts) != expected_set
        or lora.get("adapter_parameter_count")
        != EXPECTED_ADAPTER_PARAMETER_COUNT
        or lora.get("trainable_adapter_parameter_count") != 0
        or checkpoint_metadata.get("lora_parameter_count")
        != EXPECTED_ADAPTER_PARAMETER_COUNT
        or checkpoint_metadata.get("lora_trainable_parameter_count") != 0
        or checkpoint_metadata.get("question_dependent_scene_processing") is not False
    ):
        raise ValueError("V94 runtime requires the exact ordered frozen eight-bank stack")

    metadata_by_name = {str(row["name"]): row for row in banks}
    for spec in _BANK_SPECS:
        config_row = _mapping(configured.get(spec.name), f"config bank {spec.name}")
        metadata_row = metadata_by_name[spec.name]
        count_row = _mapping(counts.get(spec.name), f"parameter count {spec.name}")
        count_values = tuple(count_row.values())
        if (
            config_row.get("trainable") is not False
            or config_row.get("rank") != spec.rank
            or float(config_row.get("alpha", -1.0)) != spec.alpha
            or float(config_row.get("dropout", -1.0)) != 0.0
            or tuple(config_row.get("target_modules", ())) != spec.targets
            or config_row.get("expected_initial_state_sha256")
            != spec.state_sha256
            or metadata_row.get("trainable") is not False
            or metadata_row.get("rank") != spec.rank
            or float(metadata_row.get("alpha", -1.0)) != spec.alpha
            or float(metadata_row.get("dropout", -1.0)) != 0.0
            or tuple(metadata_row.get("target_modules", ())) != spec.targets
            or metadata_row.get("adapter_parameter_count") != spec.parameter_count
            or metadata_row.get("expected_initial_state_sha256")
            != spec.state_sha256
            or states.get(spec.name) != spec.state_sha256
            or tuple(modules.get(spec.name, ())) != spec.targets
            or set(count_row) != set(spec.targets)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in count_values
            )
            or sum(count_values) != spec.parameter_count
        ):
            raise ValueError(f"V94 runtime frozen bank changed: {spec.name}")

    provenance = _mapping(
        checkpoint_metadata.get("initialization_provenance"),
        "initialization provenance",
    )
    release = _mapping(
        provenance.get("v94_strict_runtime_release"), "release provenance"
    )
    for key in (
        "source_v94_evidence_sha256",
        "source_v94_score_sha256",
        "v94_bridge_state_sha256",
    ):
        if not _is_sha256(release.get(key)):
            raise ValueError("V94 runtime release hash bindings are incomplete")
    decision = release.get("promotion_decision")
    authorized = release.get("runtime_promotion_authorized")
    smoke_hash = release.get("smoke_report_sha256")
    candidate = (
        decision == CANDIDATE_DECISION
        and authorized is False
        and smoke_hash is None
    )
    promoted = (
        decision == PROMOTED_DECISION
        and authorized is True
        and _is_sha256(smoke_hash)
    )
    if (
        release.get("schema_version") != 94
        or release.get("v94_bridge_state_sha256") != V94_STATE_SHA256
        or release.get("model_acceptance_gate_passed") is not True
        or release.get("model_gate_report_authenticated") is not True
        or release.get("held_out_generalization_claim") is not True
        or not (candidate or promoted)
    ):
        raise ValueError("V94 runtime is not bound to an authenticated release gate")
    return {
        "v94_bridge_state_sha256": V94_STATE_SHA256,
        "release_provenance": dict(release),
        "frozen_lora_bank_count": len(banks),
        "adapter_parameter_count": EXPECTED_ADAPTER_PARAMETER_COUNT,
        "runtime_package_mode": "promoted" if promoted else "candidate",
        "runtime_promotion_authorized": bool(promoted),
    }


class V94StrictMultisceneChatRuntime(V83DirectSceneMemoryChatRuntime):
    """V83 direct memory with the exact frozen eight-bank V94 stack."""

    def __init__(self, base: StaticChatRuntime, loaded: LoadedV81SceneMemory) -> None:
        validate_v94_scene_memory_contract(scene_id=base.scene_id, loaded=loaded)
        super().__init__(base, loaded)
        contract = validate_v94_runtime_contract(
            runtime_config=self.config,
            checkpoint_metadata=self.base.checkpoint_metadata,
        )
        self.release_provenance = contract["release_provenance"]
        self.runtime_package_mode = contract["runtime_package_mode"]
        self.runtime_promotion_authorized = contract["runtime_promotion_authorized"]
        self.v94_bridge_state_sha256 = contract["v94_bridge_state_sha256"]
        self.environment_conditioned_input_hashes: list[str] = []

    def startup_summary(self) -> dict[str, Any]:
        summary = super().startup_summary()
        return {
            **summary,
            "phase": "v94_strict_multiscene_ready",
            "runtime_kind": RUNTIME_KIND,
            "frozen_lora_bank_count": 8,
            "frozen_lora_parameter_count": EXPECTED_ADAPTER_PARAMETER_COUNT,
            "trainable_runtime_parameter_count": 0,
            "v94_bridge_bank": V94_BANK,
            "v94_bridge_target": V94_TARGET,
            "v94_bridge_state_sha256": self.v94_bridge_state_sha256,
            "exact_total_environment_conditioned_input_sha256": (
                self.scene_prefix_hash
            ),
            "environment_conditioned_input_hashes_observed": list(
                self.environment_conditioned_input_hashes
            ),
            "runtime_package_mode": self.runtime_package_mode,
            "promotion_decision": self.release_provenance["promotion_decision"],
            "runtime_promotion_authorized": self.runtime_promotion_authorized,
            "model_gate_report_authenticated": True,
            "held_out_generalization_claim": True,
            "runtime_loaded_training_evaluation_or_scorer_files": False,
        }

    def answer(self, question: str) -> ChatAnswer:
        result = super().answer(question)
        if self.last_prepared_layout_audit is None:
            raise RuntimeError("V94 generation did not produce a direct-memory audit")
        if result.prefix_hash != self.scene_prefix_hash:
            raise RuntimeError("V94 total environment-conditioned input changed")
        self.environment_conditioned_input_hashes.append(result.prefix_hash)
        if len(set(self.environment_conditioned_input_hashes)) != 1:
            raise RuntimeError("V94 environment-conditioned input is question-dependent")
        return result


__all__ = [
    "CANDIDATE_DECISION",
    "EXPECTED_ADAPTER_PARAMETER_COUNT",
    "EXPECTED_BANKS",
    "PROMOTED_DECISION",
    "RUNTIME_KIND",
    "V94_BANK",
    "V94_STATE_SHA256",
    "V94_TARGET",
    "V94StrictMultisceneChatRuntime",
    "validate_v94_runtime_contract",
    "validate_v94_scene_memory_contract",
]

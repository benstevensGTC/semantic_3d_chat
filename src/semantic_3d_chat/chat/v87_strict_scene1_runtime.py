"""Strict V87 chat over one immutable, pre-question 738-token scene memory.

This module is deliberately runtime-only.  It authenticates the nine frozen
LoRA banks embedded in the two-file runtime checkpoint, but never opens V87's
training configuration, QA rows, predictions, evaluation report, or oracle.
"""

from __future__ import annotations

import re
from typing import Any, Final

from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    V83DirectSceneMemoryChatRuntime,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import LoadedV81SceneMemory

RUNTIME_KIND: Final[str] = "v87_strict_scene1_direct_scene_memory"
V86_BANK: Final[str] = "v86_scene1_demo_bridge"
V86_TARGET: Final[str] = "model.language_model.layers.34.mlp.up_proj"
V86_STATE_SHA256: Final[str] = (
    "8b6bd801716132c8aac50c6288b9ba588417dc5e6a7c2c15dd9515892f714260"
)
V87_BANK: Final[str] = "v87_scene1_balanced_bridge"
V87_TARGET: Final[str] = "model.language_model.layers.34.mlp.gate_proj"
V85_BANK: Final[str] = "v85_strict_multiscene_bridge"
V85_STATE_SHA256: Final[str] = (
    "f31b8f99f77f1b7b92dafd74220e5e12ccfa35cbc8630a6d7640f2fe1f93c581"
)
_EXPECTED_BANKS: Final[tuple[str, ...]] = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
    "extension_v28_stage_b_query",
    "extension_v30_joint_pair_query",
    V85_BANK,
    V86_BANK,
    V87_BANK,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
SCENE_ID: Final[str] = "scene_000001"


def _one_bank(banks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    selected = [row for row in banks if row.get("name") == name]
    if len(selected) != 1:
        raise ValueError(f"V87 runtime requires exactly one {name!r} bank")
    return selected[0]


class V87StrictScene1ChatRuntime(V83DirectSceneMemoryChatRuntime):
    """V83 direct memory with V85, V86, and V87's frozen learned bridges."""

    def __init__(self, base: StaticChatRuntime, loaded: LoadedV81SceneMemory) -> None:
        super().__init__(base, loaded)
        if self.scene_id != SCENE_ID or loaded.metadata.get("scene_id") != SCENE_ID:
            raise ValueError("V87 strict scene-one adapter accepts only scene_000001")
        metadata = self.base.checkpoint_metadata
        lora = metadata.get("lora")
        if not isinstance(lora, dict) or lora.get("schema_version") != 2:
            raise ValueError("V87 runtime requires the named frozen LoRA-bank contract")
        banks_raw = lora.get("banks")
        if not isinstance(banks_raw, list) or not all(
            isinstance(row, dict) for row in banks_raw
        ):
            raise TypeError("V87 runtime LoRA-bank inventory is malformed")
        banks: list[dict[str, Any]] = banks_raw
        names = tuple(str(row.get("name")) for row in banks)
        if names != _EXPECTED_BANKS:
            raise ValueError("V87 runtime requires the exact ordered nine-bank stack")
        if any(row.get("trainable") is not False for row in banks):
            raise ValueError("Every V87 runtime LoRA bank must be frozen")

        v86 = _one_bank(banks, V86_BANK)
        v87 = _one_bank(banks, V87_BANK)
        if (
            v86.get("target_modules") != [V86_TARGET]
            or v86.get("rank") != 8
            or float(v86.get("alpha", -1.0)) != 16.0
            or v86.get("adapter_parameter_count") != 110_592
            or v87.get("target_modules") != [V87_TARGET]
            or v87.get("rank") != 8
            or float(v87.get("alpha", -1.0)) != 16.0
            or v87.get("adapter_parameter_count") != 110_592
            or lora.get("adapter_parameter_count") != 786_432
            or lora.get("trainable_adapter_parameter_count") != 0
            or metadata.get("lora_parameter_count") != 786_432
            or metadata.get("lora_trainable_parameter_count") != 0
        ):
            raise ValueError("V87 runtime bridge topology changed")

        states = metadata.get("lora_bank_state_sha256")
        if not isinstance(states, dict):
            raise TypeError("V87 runtime bank-state bindings are missing")
        v87_state = states.get(V87_BANK)
        if (
            states.get(V85_BANK) != V85_STATE_SHA256
            or states.get(V86_BANK) != V86_STATE_SHA256
            or not isinstance(v87_state, str)
            or _SHA256.fullmatch(v87_state) is None
        ):
            raise ValueError("V87 runtime bridge state identity changed")

        configured_banks = self.config.get("language", {}).get("lora_banks")
        configured_v87 = (
            configured_banks.get(V87_BANK)
            if isinstance(configured_banks, dict)
            else None
        )
        if (
            not isinstance(configured_v87, dict)
            or configured_v87.get("expected_initial_state_sha256") != v87_state
        ):
            raise ValueError("V87 standalone runtime config is not bound to its bank state")

        provenance_root = metadata.get("initialization_provenance")
        release = (
            provenance_root.get("v87_strict_runtime_release")
            if isinstance(provenance_root, dict)
            else None
        )
        required_hashes = (
            "experiment_config_sha256",
            "preregistration_sha256",
            "cpu_preflight_sha256",
            "training_report_sha256",
            "model_gate_report_sha256",
            "v86_bridge_state_sha256",
            "v87_bridge_state_sha256",
        )
        if not isinstance(release, dict) or any(
            not isinstance(release.get(key), str)
            or _SHA256.fullmatch(str(release[key])) is None
            for key in required_hashes
        ):
            raise ValueError("V87 runtime release bindings are incomplete")
        if (
            release["v86_bridge_state_sha256"] != V86_STATE_SHA256
            or release["v87_bridge_state_sha256"] != v87_state
            or release.get("model_acceptance_gate_passed") is not True
            or release.get("model_gate_report_authenticated") is not True
        ):
            raise ValueError("V87 runtime is not bound to an authenticated passing model gate")
        decision = release.get("promotion_decision")
        if decision not in {
            "pending_oracle_isolated_runtime_smoke",
            "strict_scene1_experimental_primary",
        }:
            raise ValueError("V87 runtime has no recognized promotion decision")
        self.release_provenance = dict(release)
        self.runtime_promotion_authorized = (
            decision == "strict_scene1_experimental_primary"
            and release.get("runtime_promotion_authorized") is True
        )
        self.v87_bridge_state_sha256 = v87_state
        self.environment_conditioned_input_hashes: list[str] = []

    def startup_summary(self) -> dict[str, Any]:
        summary = super().startup_summary()
        return {
            **summary,
            "phase": "v87_strict_scene1_ready",
            "runtime_kind": RUNTIME_KIND,
            "frozen_lora_bank_count": 9,
            "trainable_runtime_parameter_count": 0,
            "v86_bridge_bank": V86_BANK,
            "v86_bridge_target": V86_TARGET,
            "v86_bridge_state_sha256": V86_STATE_SHA256,
            "v87_bridge_bank": V87_BANK,
            "v87_bridge_target": V87_TARGET,
            "v87_bridge_state_sha256": self.v87_bridge_state_sha256,
            "exact_total_environment_conditioned_input_sha256": self.scene_prefix_hash,
            "environment_conditioned_input_hashes_observed": list(
                self.environment_conditioned_input_hashes
            ),
            "promotion_decision": self.release_provenance["promotion_decision"],
            "runtime_promotion_authorized": self.runtime_promotion_authorized,
            "model_gate_report_authenticated": True,
            "runtime_loaded_training_or_evaluation_reports": False,
        }

    def answer(self, question: str) -> ChatAnswer:
        result = super().answer(question)
        if self.last_prepared_layout_audit is None:
            raise RuntimeError("V87 generation did not produce a direct-memory layout audit")
        if result.prefix_hash != self.scene_prefix_hash:
            raise RuntimeError("V87 total environment-conditioned input changed")
        self.environment_conditioned_input_hashes.append(result.prefix_hash)
        if len(set(self.environment_conditioned_input_hashes)) != 1:
            raise RuntimeError("V87 environment-conditioned input is question-dependent")
        return result


__all__ = [
    "RUNTIME_KIND",
    "SCENE_ID",
    "V85_BANK",
    "V85_STATE_SHA256",
    "V86_BANK",
    "V86_STATE_SHA256",
    "V86_TARGET",
    "V87_BANK",
    "V87_TARGET",
    "V87StrictScene1ChatRuntime",
]

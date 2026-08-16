"""Strict V89 chat over one immutable, pre-question continuous scene memory.

This is deliberately a runtime-only module.  It authenticates the exact
eleven-bank checkpoint and its release provenance, but imports no V89
preflight, trainer, evaluator, predictions, question rows, oracle metadata, or
release code.  Environmental information reaches Gemma only through the
numeric 738-token scene memory loaded by the direct-memory parent runtime.
"""

from __future__ import annotations

import re
from typing import Any, Final

from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    V83DirectSceneMemoryChatRuntime,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    LoadedV81SceneMemory,
)

RUNTIME_KIND: Final[str] = "v89_strict_scene1_direct_scene_memory"
SCENE_ID: Final[str] = "scene_000001"
V85_BANK: Final[str] = "v85_strict_multiscene_bridge"
V85_STATE_SHA256: Final[str] = (
    "f31b8f99f77f1b7b92dafd74220e5e12ccfa35cbc8630a6d7640f2fe1f93c581"
)
V86_BANK: Final[str] = "v86_scene1_demo_bridge"
V86_TARGET: Final[str] = "model.language_model.layers.34.mlp.up_proj"
V86_STATE_SHA256: Final[str] = (
    "8b6bd801716132c8aac50c6288b9ba588417dc5e6a7c2c15dd9515892f714260"
)
V87_BANK: Final[str] = "v87_scene1_balanced_bridge"
V87_TARGET: Final[str] = "model.language_model.layers.34.mlp.gate_proj"
V87_STATE_SHA256: Final[str] = (
    "618c03e102d9a9eb98d405d5a040cd8285194539b5a4043d34f16356ac08769e"
)
V88_BANK: Final[str] = "v88_scene1_augmented_bridge"
V88_TARGET: Final[str] = "model.language_model.layers.27.self_attn.q_proj"
V88_STATE_SHA256: Final[str] = (
    "ff311624150056c67ad1c0a06752a77af2de89878778049ae886aa59db3376aa"
)
V89_BANK: Final[str] = "v89_scene1_retention_bridge"
V89_TARGET: Final[str] = "model.language_model.layers.27.self_attn.o_proj"
EXPECTED_BANKS: Final[tuple[str, ...]] = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
    "extension_v28_stage_b_query",
    "extension_v30_joint_pair_query",
    V85_BANK,
    V86_BANK,
    V87_BANK,
    V88_BANK,
    V89_BANK,
)
EXPECTED_ADAPTER_PARAMETER_COUNT: Final[int] = 872_448
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _one_bank(banks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    selected = [row for row in banks if row.get("name") == name]
    if len(selected) != 1:
        raise ValueError(f"V89 runtime requires exactly one {name!r} bank")
    return selected[0]


class V89StrictScene1ChatRuntime(V83DirectSceneMemoryChatRuntime):
    """Direct 738-token memory with the exact frozen eleven-bank V89 stack."""

    def __init__(self, base: StaticChatRuntime, loaded: LoadedV81SceneMemory) -> None:
        super().__init__(base, loaded)
        if self.scene_id != SCENE_ID or loaded.metadata.get("scene_id") != SCENE_ID:
            raise ValueError("V89 strict scene-one adapter accepts only scene_000001")

        metadata = self.base.checkpoint_metadata
        lora = metadata.get("lora")
        if not isinstance(lora, dict) or lora.get("schema_version") != 2:
            raise ValueError("V89 runtime requires the named frozen LoRA-bank contract")
        banks_raw = lora.get("banks")
        if not isinstance(banks_raw, list) or not all(
            isinstance(row, dict) for row in banks_raw
        ):
            raise TypeError("V89 runtime LoRA-bank inventory is malformed")
        banks: list[dict[str, Any]] = banks_raw
        if tuple(str(row.get("name")) for row in banks) != EXPECTED_BANKS:
            raise ValueError("V89 runtime requires the exact ordered eleven-bank stack")
        if any(row.get("trainable") is not False for row in banks):
            raise ValueError("Every V89 runtime LoRA bank must be frozen")

        expected_topology = (
            (V86_BANK, V86_TARGET, 8, 16.0, 110_592),
            (V87_BANK, V87_TARGET, 8, 16.0, 110_592),
            (V88_BANK, V88_TARGET, 16, 32.0, 57_344),
            (V89_BANK, V89_TARGET, 8, 16.0, 28_672),
        )
        for name, target, rank, alpha, count in expected_topology:
            row = _one_bank(banks, name)
            if (
                row.get("target_modules") != [target]
                or row.get("rank") != rank
                or float(row.get("alpha", -1.0)) != alpha
                or float(row.get("dropout", -1.0)) != 0.0
                or row.get("adapter_parameter_count") != count
            ):
                raise ValueError(f"V89 runtime bridge topology changed: {name}")
        if (
            lora.get("adapter_parameter_count") != EXPECTED_ADAPTER_PARAMETER_COUNT
            or lora.get("trainable_adapter_parameter_count") != 0
            or metadata.get("lora_parameter_count")
            != EXPECTED_ADAPTER_PARAMETER_COUNT
            or metadata.get("lora_trainable_parameter_count") != 0
        ):
            raise ValueError("V89 runtime adapter parameter inventory changed")

        states = metadata.get("lora_bank_state_sha256")
        if not isinstance(states, dict) or set(states) != set(EXPECTED_BANKS):
            raise TypeError("V89 runtime bank-state bindings are missing or inexact")
        v89_state = states.get(V89_BANK)
        if (
            states.get(V85_BANK) != V85_STATE_SHA256
            or states.get(V86_BANK) != V86_STATE_SHA256
            or states.get(V87_BANK) != V87_STATE_SHA256
            or states.get(V88_BANK) != V88_STATE_SHA256
            or not isinstance(v89_state, str)
            or _SHA256.fullmatch(v89_state) is None
        ):
            raise ValueError("V89 runtime bridge state identity changed")

        configured = self.config.get("language", {}).get("lora_banks")
        if not isinstance(configured, dict) or tuple(configured) != EXPECTED_BANKS:
            raise ValueError("V89 standalone runtime config bank inventory changed")
        metadata_by_name = {str(row["name"]): row for row in banks}
        if any(
            not isinstance(configured[name], dict)
            or configured[name].get("trainable") is not False
            or configured[name].get("expected_initial_state_sha256")
            != metadata_by_name[name].get("expected_initial_state_sha256")
            or configured[name].get("expected_initial_state_sha256")
            != states[name]
            for name in EXPECTED_BANKS
        ):
            raise ValueError("V89 standalone runtime config bank contracts changed")

        provenance_root = metadata.get("initialization_provenance")
        release = (
            provenance_root.get("v89_strict_runtime_release")
            if isinstance(provenance_root, dict)
            else None
        )
        required_hashes = (
            "experiment_config_sha256",
            "preregistration_sha256",
            "cpu_preflight_sha256",
            "training_report_sha256",
            "model_gate_report_sha256",
            "evaluation_predictions_sha256",
            "v86_bridge_state_sha256",
            "v87_bridge_state_sha256",
            "v88_bridge_state_sha256",
            "v89_bridge_state_sha256",
        )
        if not isinstance(release, dict) or any(
            not isinstance(release.get(key), str)
            or _SHA256.fullmatch(str(release[key])) is None
            for key in required_hashes
        ):
            raise ValueError("V89 runtime release bindings are incomplete")
        if (
            release["v86_bridge_state_sha256"] != V86_STATE_SHA256
            or release["v87_bridge_state_sha256"] != V87_STATE_SHA256
            or release["v88_bridge_state_sha256"] != V88_STATE_SHA256
            or release["v89_bridge_state_sha256"] != v89_state
            or release.get("model_acceptance_gate_passed") is not True
            or release.get("model_gate_report_authenticated") is not True
            or release.get("development_known_smoke_trained") is not True
            or release.get("held_out_smoke_claim") is not False
            or release.get("held_out_generalization_claim") is not False
        ):
            raise ValueError("V89 runtime is not bound to an authenticated model gate")
        decision = release.get("promotion_decision")
        if decision not in {
            "pending_isolated_runtime_smoke",
            "strict_scene1_experimental_primary",
        }:
            raise ValueError("V89 runtime has no recognized promotion decision")

        self.release_provenance = dict(release)
        self.runtime_promotion_authorized = (
            decision == "strict_scene1_experimental_primary"
            and release.get("runtime_promotion_authorized") is True
        )
        self.v89_bridge_state_sha256 = v89_state
        self.environment_conditioned_input_hashes: list[str] = []

    def startup_summary(self) -> dict[str, Any]:
        summary = super().startup_summary()
        return {
            **summary,
            "phase": "v89_strict_scene1_ready",
            "runtime_kind": RUNTIME_KIND,
            "frozen_lora_bank_count": 11,
            "trainable_runtime_parameter_count": 0,
            "v89_bridge_bank": V89_BANK,
            "v89_bridge_target": V89_TARGET,
            "v89_bridge_state_sha256": self.v89_bridge_state_sha256,
            "exact_total_environment_conditioned_input_sha256": (
                self.scene_prefix_hash
            ),
            "environment_conditioned_input_hashes_observed": list(
                self.environment_conditioned_input_hashes
            ),
            "promotion_decision": self.release_provenance["promotion_decision"],
            "runtime_promotion_authorized": self.runtime_promotion_authorized,
            "model_gate_report_authenticated": True,
            "development_known_smoke_trained": True,
            "held_out_smoke_claim": False,
            "runtime_loaded_training_or_evaluation_reports": False,
        }

    def answer(self, question: str) -> ChatAnswer:
        result = super().answer(question)
        if self.last_prepared_layout_audit is None:
            raise RuntimeError(
                "V89 generation did not produce a direct-memory layout audit"
            )
        if result.prefix_hash != self.scene_prefix_hash:
            raise RuntimeError("V89 total environment-conditioned input changed")
        self.environment_conditioned_input_hashes.append(result.prefix_hash)
        if len(set(self.environment_conditioned_input_hashes)) != 1:
            raise RuntimeError(
                "V89 environment-conditioned input is question-dependent"
            )
        return result


__all__ = [
    "EXPECTED_ADAPTER_PARAMETER_COUNT",
    "EXPECTED_BANKS",
    "RUNTIME_KIND",
    "SCENE_ID",
    "V85_BANK",
    "V85_STATE_SHA256",
    "V86_BANK",
    "V86_STATE_SHA256",
    "V86_TARGET",
    "V87_BANK",
    "V87_STATE_SHA256",
    "V87_TARGET",
    "V88_BANK",
    "V88_STATE_SHA256",
    "V88_TARGET",
    "V89_BANK",
    "V89_TARGET",
    "V89StrictScene1ChatRuntime",
]

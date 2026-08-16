"""Strict V85 local chat over one exact pre-question 738-token scene memory."""

from __future__ import annotations

from typing import Any, Final

from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    V83DirectSceneMemoryChatRuntime,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import LoadedV81SceneMemory

RUNTIME_KIND: Final[str] = "v85_strict_multiscene_direct_scene_memory"
BRIDGE_BANK: Final[str] = "v85_strict_multiscene_bridge"
BRIDGE_TARGET: Final[str] = "model.language_model.layers.34.mlp.down_proj"
BRIDGE_STATE_SHA256: Final[str] = (
    "f31b8f99f77f1b7b92dafd74220e5e12ccfa35cbc8630a6d7640f2fe1f93c581"
)
_BINDINGS: Final[dict[str, str]] = {
    "experiment_config_sha256": (
        "d4f653dc20a7ad129eb9fa92b586c8ca472a49fdb72675cbddb4f03007b4c36d"
    ),
    "preregistration_sha256": (
        "4af534bc37cd09fe7431042ff6fb75bd734a267380e1fe425c6c87b2cb42afff"
    ),
    "training_report_sha256": (
        "d7c352fd0d6c6dec23f80de61f49efe00635aac30988e3a783a7483c97f79e96"
    ),
    "development_score_sha256": (
        "202134d8900e105d63f23d1cc1d19d68a882c4464382b7a63b7aa007f2714828"
    ),
    "fixed_bridge_state_sha256": BRIDGE_STATE_SHA256,
}


class V85StrictMultisceneChatRuntime(V83DirectSceneMemoryChatRuntime):
    """V83's exact direct memory with the sole learned V85 bridge attached."""

    def __init__(self, base: StaticChatRuntime, loaded: LoadedV81SceneMemory) -> None:
        super().__init__(base, loaded)
        metadata = self.base.checkpoint_metadata
        lora = metadata.get("lora")
        if not isinstance(lora, dict) or lora.get("schema_version") != 2:
            raise ValueError("V85 runtime requires the named frozen LoRA-bank contract")
        banks = lora.get("banks")
        if not isinstance(banks, list) or len(banks) != 7:
            raise ValueError("V85 runtime requires exactly six V54 banks plus one V85 bank")
        bridge = [row for row in banks if isinstance(row, dict) and row.get("name") == BRIDGE_BANK]
        if len(bridge) != 1 or bridge[0].get("target_modules") != [BRIDGE_TARGET]:
            raise ValueError("V85 runtime bridge topology changed")
        if any(row.get("trainable") is not False for row in banks):
            raise ValueError("Every V85 runtime LoRA bank must be frozen")
        states = metadata.get("lora_bank_state_sha256")
        if not isinstance(states, dict) or states.get(BRIDGE_BANK) != BRIDGE_STATE_SHA256:
            raise ValueError("V85 runtime bridge state identity changed")
        provenance = metadata.get("initialization_provenance")
        release = (
            provenance.get("v85_strict_runtime_release")
            if isinstance(provenance, dict)
            else None
        )
        if not isinstance(release, dict) or any(
            release.get(key) != value for key, value in _BINDINGS.items()
        ):
            raise ValueError("V85 runtime release bindings changed")
        decision = release.get("promotion_decision")
        if decision not in {
            "pending_strict_runtime_leakage",
            "strict_experimental_primary",
        }:
            raise ValueError("V85 runtime has no recognized promotion decision")
        self.release_provenance = dict(release)
        self.runtime_promotion_authorized = (
            decision == "strict_experimental_primary"
            and release.get("runtime_promotion_authorized") is True
        )
        self.environment_conditioned_input_hashes: list[str] = []

    def startup_summary(self) -> dict[str, Any]:
        summary = super().startup_summary()
        return {
            **summary,
            "phase": "v85_strict_multiscene_ready",
            "runtime_kind": RUNTIME_KIND,
            "v85_bridge_bank": BRIDGE_BANK,
            "v85_bridge_target": BRIDGE_TARGET,
            "v85_bridge_state_sha256": BRIDGE_STATE_SHA256,
            "v85_bridge_parameter_count": 55_296,
            "frozen_lora_bank_count": 7,
            "trainable_runtime_parameter_count": 0,
            "exact_total_environment_conditioned_input_sha256": self.scene_prefix_hash,
            "environment_conditioned_input_hashes_observed": list(
                self.environment_conditioned_input_hashes
            ),
            "promotion_decision": self.release_provenance["promotion_decision"],
            "runtime_promotion_authorized": self.runtime_promotion_authorized,
            "v75_comparator_retained": self.release_provenance[
                "v75_comparator_retained"
            ],
            "runtime_loaded_training_or_development_reports": False,
        }

    def answer(self, question: str) -> ChatAnswer:
        result = super().answer(question)
        # V83's generation-time layout audit has already proved byte equality
        # between this memory and the complete environmental segment supplied to
        # Gemma.  Retain that exact identity after every independent question.
        if self.last_prepared_layout_audit is None:
            raise RuntimeError("V85 generation did not produce a direct-memory layout audit")
        observed = result.prefix_hash
        if observed != self.scene_prefix_hash:
            raise RuntimeError("V85 total environment-conditioned input changed")
        self.environment_conditioned_input_hashes.append(observed)
        if len(set(self.environment_conditioned_input_hashes)) != 1:
            raise RuntimeError("V85 environment-conditioned input is question-dependent")
        return result


__all__ = [
    "BRIDGE_BANK",
    "BRIDGE_STATE_SHA256",
    "BRIDGE_TARGET",
    "RUNTIME_KIND",
    "V85StrictMultisceneChatRuntime",
]

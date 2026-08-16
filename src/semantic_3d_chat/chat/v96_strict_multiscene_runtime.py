"""Standalone V96 chat over one immutable continuous scene memory.

This module is deliberately inference-only.  It accepts an exact frozen
ten-bank V96 package and one sanitized V81 numeric scene memory.  It never
imports experiment, trainer, evaluator, question, answer, prediction, or
oracle modules.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    V83DirectSceneMemoryChatRuntime,
)
from semantic_3d_chat.chat.v96_explicit_candidate_runtime import (
    EXPECTED_BANKS,
    TOTAL_PARAMETER_COUNT,
    V94_BANK,
    V94_STATE_SHA256,
    V95_BANK,
    V96_BANK,
    validate_v96_scene_memory_contract,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    LoadedV81SceneMemory,
    load_v81_scene_memory,
)

RUNTIME_KIND: Final[str] = "v96_strict_multiscene_direct_scene_memory"
PENDING_DECISION: Final[str] = "pending_isolated_runtime_smoke"
PROMOTED_DECISION: Final[str] = "strict_v96_deferred_final_primary"
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

# This is the inference implementation that is sealed by the isolated smoke.
# It is deliberately explicit: runtime verification must not walk data,
# evaluation-output, or oracle directories merely to discover source files.
RUNTIME_IMPLEMENTATION_FILES: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/chat/file_audit.py",
    "src/semantic_3d_chat/chat/grounding_sidecar_v78_runtime.py",
    "src/semantic_3d_chat/chat/question_control_runtime.py",
    "src/semantic_3d_chat/chat/runtime.py",
    "src/semantic_3d_chat/chat/runtime_config.py",
    "src/semantic_3d_chat/chat/v81_scene_memory_runtime.py",
    "src/semantic_3d_chat/chat/v83_direct_scene_memory_runtime.py",
    "src/semantic_3d_chat/chat/v96_explicit_candidate_runtime.py",
    "src/semantic_3d_chat/chat/v96_strict_multiscene_cli.py",
    "src/semantic_3d_chat/chat/v96_strict_multiscene_runtime.py",
    "src/semantic_3d_chat/config.py",
    "src/semantic_3d_chat/device.py",
    "src/semantic_3d_chat/evaluation/prediction_artifacts.py",
    "src/semantic_3d_chat/language/generation.py",
    "src/semantic_3d_chat/language/local_lm.py",
    "src/semantic_3d_chat/language/lora.py",
    "src/semantic_3d_chat/language/prefix_injection.py",
    "src/semantic_3d_chat/language/v81_structured_dense_atlas_sidecar.py",
    "src/semantic_3d_chat/scene_encoder/block_cross_residual.py",
    "src/semantic_3d_chat/scene_encoder/dense_alignment.py",
    "src/semantic_3d_chat/scene_encoder/dense_sidecar_adapter.py",
    "src/semantic_3d_chat/scene_encoder/fixed_prefix_atlas.py",
    "src/semantic_3d_chat/scene_encoder/global_residual.py",
    "src/semantic_3d_chat/scene_encoder/map_io.py",
    "src/semantic_3d_chat/scene_encoder/perceiver.py",
    "src/semantic_3d_chat/scene_encoder/point_tokens.py",
    "src/semantic_3d_chat/scene_encoder/projector.py",
    "src/semantic_3d_chat/scene_encoder/signed_x_dispatch.py",
    "src/semantic_3d_chat/scene_encoder/signed_x_local_field.py",
    "src/semantic_3d_chat/scene_encoder/signed_x_residual.py",
    "src/semantic_3d_chat/scene_encoder/spatial_blocks.py",
    "src/semantic_3d_chat/scene_encoder/v81_scene_memory_artifact.py",
    "src/semantic_3d_chat/training/checkpointing.py",
    "src/semantic_3d_chat/training/losses.py",
)


def runtime_implementation_inventory_v96() -> dict[str, Any]:
    """Hash the exact local inference sources without discovering data files."""

    files: list[dict[str, Any]] = []
    for relative in RUNTIME_IMPLEMENTATION_FILES:
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"V96 runtime source must be physical: {path}")
        payload = path.read_bytes()
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    encoded = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "files": files,
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
    }


@dataclass(frozen=True)
class _BankSpec:
    name: str
    targets: tuple[str, ...]
    rank: int
    alpha: float
    parameter_count: int
    state_sha256: str | None


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
        ("model.language_model.layers.34.mlp.gate_proj",),
        8,
        16.0,
        110_592,
        V94_STATE_SHA256,
    ),
    _BankSpec(
        V95_BANK,
        (
            "model.language_model.layers.9.self_attn.k_proj",
            "model.language_model.layers.9.self_attn.v_proj",
            "model.language_model.layers.34.mlp.up_proj",
        ),
        8,
        16.0,
        143_360,
        None,
    ),
    _BankSpec(
        V96_BANK,
        ("model.language_model.layers.9.self_attn.q_proj",),
        8,
        16.0,
        45_056,
        None,
    ),
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V96 runtime {label} is malformed")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _checkpoint_fingerprint(checkpoint: Path) -> str:
    entries: list[dict[str, Any]] = []
    for name in ("adapter.safetensors", "metadata.json", "runtime_metadata.json"):
        path = checkpoint / name
        if not path.is_file():
            if name in {"adapter.safetensors", "runtime_metadata.json"}:
                raise FileNotFoundError(path)
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append(
            {"path": name, "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}
        )
    encoded = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_v96_release_runtime_contract(
    *, runtime_config: Mapping[str, Any], checkpoint_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate the exact ten frozen banks and held-out release binding."""

    language = _mapping(runtime_config.get("language"), "language config")
    configured = _mapping(language.get("lora_banks"), "configured banks")
    lora = _mapping(checkpoint_metadata.get("lora"), "checkpoint LoRA contract")
    banks_raw = lora.get("banks")
    if not isinstance(banks_raw, list) or not all(
        isinstance(row, Mapping) for row in banks_raw
    ):
        raise TypeError("V96 runtime checkpoint bank inventory is malformed")
    banks: list[Mapping[str, Any]] = banks_raw
    states = _mapping(checkpoint_metadata.get("lora_bank_state_sha256"), "bank states")
    modules = _mapping(checkpoint_metadata.get("lora_bank_wrapped_modules"), "bank modules")
    counts = _mapping(checkpoint_metadata.get("lora_bank_parameter_counts"), "bank counts")
    expected_set = set(EXPECTED_BANKS)
    if (
        tuple(configured) != EXPECTED_BANKS
        or tuple(str(row.get("name")) for row in banks) != EXPECTED_BANKS
        or set(states) != expected_set
        or set(modules) != expected_set
        or set(counts) != expected_set
        or lora.get("schema_version") != 2
        or lora.get("enabled") is not True
        or lora.get("adapter_parameter_count") != TOTAL_PARAMETER_COUNT
        or lora.get("trainable_adapter_parameter_count") != 0
        or checkpoint_metadata.get("lora_parameter_count") != TOTAL_PARAMETER_COUNT
        or checkpoint_metadata.get("lora_trainable_parameter_count") != 0
        or checkpoint_metadata.get("question_dependent_scene_processing") is not False
    ):
        raise ValueError("V96 runtime requires the exact ordered frozen ten-bank stack")

    by_name = {str(row["name"]): row for row in banks}
    for spec in _BANK_SPECS:
        config_row = _mapping(configured.get(spec.name), f"config bank {spec.name}")
        metadata_row = by_name[spec.name]
        count_row = _mapping(counts.get(spec.name), f"count bank {spec.name}")
        state = states.get(spec.name)
        if (
            not _is_sha256(state)
            or (spec.state_sha256 is not None and state != spec.state_sha256)
            or config_row.get("trainable") is not False
            or config_row.get("rank") != spec.rank
            or float(config_row.get("alpha", -1.0)) != spec.alpha
            or float(config_row.get("dropout", -1.0)) != 0.0
            or tuple(config_row.get("target_modules", ())) != spec.targets
            or config_row.get("expected_initial_state_sha256") != state
            or metadata_row.get("trainable") is not False
            or metadata_row.get("rank") != spec.rank
            or float(metadata_row.get("alpha", -1.0)) != spec.alpha
            or float(metadata_row.get("dropout", -1.0)) != 0.0
            or tuple(metadata_row.get("target_modules", ())) != spec.targets
            or metadata_row.get("adapter_parameter_count") != spec.parameter_count
            or metadata_row.get("expected_initial_state_sha256") != state
            or tuple(modules.get(spec.name, ())) != spec.targets
            or set(count_row) != set(spec.targets)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in count_row.values()
            )
            or sum(count_row.values()) != spec.parameter_count
        ):
            raise ValueError(f"V96 runtime frozen bank changed: {spec.name}")

    provenance = _mapping(
        checkpoint_metadata.get("initialization_provenance"), "initialization provenance"
    )
    release = _mapping(provenance.get("v96_strict_runtime_release"), "release provenance")
    for key in (
        "candidate_fingerprint_sha256",
        "candidate_attestation_file_sha256",
        "candidate_attestation_identity_sha256",
        "v1_implementation_seal_sha256",
        "v2_implementation_seal_sha256",
        "deferred_final_evidence_sha256",
        "deferred_final_score_sha256",
        "runtime_implementation_inventory_sha256",
        "v94_state_sha256",
        "v95_state_sha256",
        "v96_state_sha256",
    ):
        if not _is_sha256(release.get(key)):
            raise ValueError("V96 runtime release hash bindings are incomplete")
    current_implementation = runtime_implementation_inventory_v96()
    if (
        release.get("runtime_implementation_inventory_sha256")
        != current_implementation["inventory_sha256"]
    ):
        raise ValueError("V96 runtime implementation changed after isolated smoke")
    candidate = (
        release.get("promotion_decision") == PENDING_DECISION
        and release.get("runtime_promotion_authorized") is False
        and release.get("smoke_report_sha256") is None
    )
    promoted = (
        release.get("promotion_decision") == PROMOTED_DECISION
        and release.get("runtime_promotion_authorized") is True
        and _is_sha256(release.get("smoke_report_sha256"))
    )
    if (
        release.get("schema_version") != 96
        or release.get("v94_state_sha256") != V94_STATE_SHA256
        or release.get("v95_state_sha256") != states[V95_BANK]
        or release.get("v96_state_sha256") != states[V96_BANK]
        or release.get("known_development_gate_passed") is not True
        or release.get("deferred_final_gate_passed") is not True
        or release.get("deferred_final_evidence_authenticated") is not True
        or release.get("supervision_isolation_proven") is not True
        or release.get("prefix_hash_invariant_in_evaluation") is not True
        or release.get("held_out_generalization_claim") is not True
        or release.get("environmental_text_inputs") != []
        or not (candidate or promoted)
    ):
        raise ValueError("V96 runtime is not bound to an authenticated held-out gate")
    return {
        "release_provenance": dict(release),
        "runtime_package_mode": "promoted" if promoted else "candidate",
        "runtime_promotion_authorized": promoted,
        "v95_state_sha256": str(states[V95_BANK]),
        "v96_state_sha256": str(states[V96_BANK]),
    }


class V96StrictMultisceneChatRuntime(V83DirectSceneMemoryChatRuntime):
    """V83 direct memory with the exact frozen ten-bank V96 release stack."""

    def __init__(self, base: StaticChatRuntime, loaded: LoadedV81SceneMemory) -> None:
        validate_v96_scene_memory_contract(scene_id=base.scene_id, loaded=loaded)
        super().__init__(base, loaded)
        contract = validate_v96_release_runtime_contract(
            runtime_config=self.config,
            checkpoint_metadata=self.base.checkpoint_metadata,
        )
        self.release_provenance = contract["release_provenance"]
        self.runtime_package_mode = str(contract["runtime_package_mode"])
        self.runtime_promotion_authorized = bool(
            contract["runtime_promotion_authorized"]
        )
        self.v95_state_sha256 = str(contract["v95_state_sha256"])
        self.v96_state_sha256 = str(contract["v96_state_sha256"])
        self.environment_conditioned_input_hashes: list[str] = []

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        scene_id: str,
        *,
        base_checkpoint: str | Path,
        scene_memory: str | Path,
        audit: FileAccessAudit | None = None,
        local_files_only: bool = True,
    ) -> V96StrictMultisceneChatRuntime:
        if config.get("_runtime_safe_config") is not True:
            raise ValueError("V96 chat requires a standalone validated runtime config")
        checkpoint = Path(base_checkpoint).expanduser().resolve()
        base = StaticChatRuntime.load(
            config,
            scene_id,
            checkpoint=checkpoint,
            audit=audit,
            local_files_only=local_files_only,
        )
        validate_v96_release_runtime_contract(
            runtime_config=config,
            checkpoint_metadata=base.checkpoint_metadata,
        )
        loaded = load_v81_scene_memory(
            scene_memory,
            expected_scene_id=scene_id,
            expected_base_checkpoint_sha256=_checkpoint_fingerprint(checkpoint),
            expected_runtime_config_sha256=effective_runtime_config_sha256(config),
            expected_model_device=base.language.device,
            record_file=None if audit is None else audit.record,
        )
        validate_v96_scene_memory_contract(scene_id=scene_id, loaded=loaded)
        return cls(base, loaded)

    def startup_summary(self) -> dict[str, Any]:
        summary = super().startup_summary()
        return {
            **summary,
            "phase": "v96_strict_multiscene_ready",
            "runtime_kind": RUNTIME_KIND,
            "frozen_lora_bank_count": 10,
            "frozen_lora_parameter_count": TOTAL_PARAMETER_COUNT,
            "trainable_runtime_parameter_count": 0,
            "lora_bank_order": list(EXPECTED_BANKS),
            "v95_state_sha256": self.v95_state_sha256,
            "v96_state_sha256": self.v96_state_sha256,
            "exact_total_environment_conditioned_input_sha256": self.scene_prefix_hash,
            "environment_conditioned_input_hashes_observed": list(
                self.environment_conditioned_input_hashes
            ),
            "runtime_package_mode": self.runtime_package_mode,
            "promotion_decision": self.release_provenance["promotion_decision"],
            "runtime_promotion_authorized": self.runtime_promotion_authorized,
            "known_development_gate_passed": True,
            "deferred_final_gate_passed": True,
            "held_out_generalization_claim": True,
            "runtime_loaded_oracle_or_text_metadata": False,
            "runtime_loaded_training_evaluation_or_scorer_files": False,
        }

    def answer(self, question: str) -> ChatAnswer:
        before = prefix_sha256(self.fixed_scene_memory)
        if before != self.scene_prefix_hash:
            raise RuntimeError("V96 fixed scene memory changed before the question")
        result = super().answer(question)
        after = prefix_sha256(self.fixed_scene_memory)
        if (
            self.last_prepared_layout_audit is None
            or result.prefix_hash != before
            or after != before
        ):
            raise RuntimeError("V96 environment-conditioned input changed")
        self.environment_conditioned_input_hashes.append(after)
        if len(set(self.environment_conditioned_input_hashes)) != 1:
            raise RuntimeError("V96 scene prefix became question-dependent")
        return result


__all__ = [
    "PENDING_DECISION",
    "PROMOTED_DECISION",
    "RUNTIME_IMPLEMENTATION_FILES",
    "RUNTIME_KIND",
    "V96StrictMultisceneChatRuntime",
    "runtime_implementation_inventory_v96",
    "validate_v96_release_runtime_contract",
]

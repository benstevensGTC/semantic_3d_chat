"""Strict local Gemma runtime over the exact immutable V81 scene memory.

V83 deliberately has no learned or deterministic question-conditioned reader.
The complete 738-token memory is compiled and authenticated before a question
can be accepted, then inserted byte-for-byte (modulo model dtype conversion) in
Gemma's native image-prefix slot for every question.  Gemma itself is the only
question-dependent consumer of the environmental tensor.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import sanitize_generated_answer
from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.chat.v81_scene_memory_runtime import V81SceneMemoryChatRuntime
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    FIXED_MEMORY_TOKENS,
    HIDDEN_SIZE,
    LoadedV81SceneMemory,
    load_v81_scene_memory,
)

RUNTIME_KIND: Final[str] = "v83_exact_738_token_direct_scene_memory"
DIRECT_PAYLOAD_TOKENS: Final[int] = FIXED_MEMORY_TOKENS - 2


def audit_v83_direct_prepared_layout(
    *,
    backend: Any,
    fixed_memory: torch.Tensor,
    prompt_ids: torch.Tensor,
    prepared: Any,
) -> dict[str, Any]:
    """Prove exact BOI/interior/EOI placement and Gemma PAD-PLE semantics."""

    if tuple(fixed_memory.shape) != (1, FIXED_MEMORY_TOKENS, HIDDEN_SIZE):
        raise ValueError(
            f"V83 requires exact scene memory [1,{FIXED_MEMORY_TOKENS},{HIDDEN_SIZE}]"
        )
    if prompt_ids.ndim != 2 or tuple(prompt_ids.shape[:1]) != (1,):
        raise ValueError("V83 prompt IDs must have shape [1,T]")
    contract = backend.native_image_contract()
    if int(prompt_ids[0, 0].item()) != int(contract["bos_token_id"]):
        raise ValueError("V83 prompt must begin with the checkpoint-native BOS token")

    expected_total = FIXED_MEMORY_TOKENS + int(prompt_ids.shape[1])
    if (
        prepared.scene_prefix_length != FIXED_MEMORY_TOKENS
        or tuple(prepared.inputs_embeds.shape)
        != (1, expected_total, HIDDEN_SIZE)
        or tuple(prepared.attention_mask.shape) != (1, expected_total)
        or tuple(prepared.mm_token_type_ids.shape) != (1, expected_total)
    ):
        raise RuntimeError("V83 direct Gemma sequence shape changed")

    # Layout is BOS, then the complete BOI..EOI memory, then prompt remainder.
    scene_start = 1
    scene_stop = scene_start + FIXED_MEMORY_TOKENS
    expected_memory = fixed_memory.to(prepared.inputs_embeds)
    if not torch.equal(prepared.inputs_embeds[:, scene_start:scene_stop], expected_memory):
        raise RuntimeError("V83 exact 738-token memory was not supplied directly to Gemma")

    boi_position = scene_start
    payload_start = boi_position + 1
    payload_stop = scene_stop - 1
    eoi_position = payload_stop
    modality = prepared.mm_token_type_ids
    if (
        bool(torch.any(modality[:, :payload_start] != 0))
        or bool(torch.any(modality[:, payload_start:payload_stop] != 1))
        or bool(torch.any(modality[:, payload_stop:] != 0))
    ):
        raise RuntimeError("V83 native Gemma image-modality IDs changed")

    _pad_embeddings, pad_ple, _pad_types = backend.padding_values(
        1,
        DIRECT_PAYLOAD_TOKENS,
        device=prepared.inputs_embeds.device,
    )
    if not torch.equal(
        prepared.per_layer_inputs[:, payload_start:payload_stop], pad_ple
    ):
        raise RuntimeError("V83 direct payload lost exact checkpoint-native PAD PLE")

    boundary_ids = torch.tensor(
        [[int(contract["boi_token_id"]), int(contract["eoi_token_id"])]],
        dtype=torch.long,
        device=prepared.inputs_embeds.device,
    )
    _boundary_embeddings, boundary_ple = backend._token_embeddings_and_ple(boundary_ids)
    if not torch.equal(
        prepared.per_layer_inputs[:, boi_position : boi_position + 1],
        boundary_ple[:, :1],
    ) or not torch.equal(
        prepared.per_layer_inputs[:, eoi_position : eoi_position + 1],
        boundary_ple[:, 1:],
    ):
        raise RuntimeError("V83 BOI/EOI lost exact checkpoint-native PLE")
    if not bool(torch.all(prepared.attention_mask == 1)):
        raise RuntimeError("V83 direct scene memory is not fully visible to Gemma")

    return {
        "sequence": [
            "prompt_bos",
            "exact_fixed_scene_memory_738",
            "remaining_system_and_latest_user_prompt",
        ],
        "fixed_scene_memory_tokens_supplied_to_gemma": FIXED_MEMORY_TOKENS,
        "continuous_environment_payload_tokens": DIRECT_PAYLOAD_TOKENS,
        "native_boi_tokens": 1,
        "native_eoi_tokens": 1,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "prepared_tokens": expected_total,
        "payload_pad_ple_exact": True,
        "payload_image_modality_exact": True,
        "boi_eoi_native_ple_exact": True,
        "all_payload_tokens_unmasked": True,
        "control_activation_tokens": 0,
        "question_derived_environmental_tokens": 0,
    }


class V83DirectSceneMemoryChatRuntime(V81SceneMemoryChatRuntime):
    """Direct full-memory baseline with no environmental reader or retrieval."""

    def __init__(self, base: StaticChatRuntime, loaded: LoadedV81SceneMemory) -> None:
        super().__init__(base, loaded, grounding_sidecar=None)
        if scene_boundary_mode_setting(self.config) != SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE:
            raise ValueError("V83 requires checkpoint-native Gemma BOI/EOI boundaries")
        if scene_prefix_after_bos_setting(self.config) is not True:
            raise ValueError("V83 requires BOS-first native image-prefix placement")
        backend = self.base.language.prefix_backend
        if backend is None or self.base.language.backend_name != "gemma4":
            raise RuntimeError("V83 requires the local Gemma 4 continuous-prefix backend")

        # This question-free startup preflight proves the 738-token memory is a
        # valid native Gemma prefix before answer() becomes callable.
        contract = backend.native_image_contract()
        bos_only = torch.tensor(
            [[int(contract["bos_token_id"])]],
            dtype=torch.long,
            device=self.base.language.device,
        )
        prepared = backend.prepare(
            self.fixed_scene_memory,
            bos_only,
            scene_prefix_after_bos=True,
            scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
            control_tokens=None,
        )
        self._startup_layout_audit = audit_v83_direct_prepared_layout(
            backend=backend,
            fixed_memory=self.fixed_scene_memory,
            prompt_ids=bos_only,
            prepared=prepared,
        )
        self.last_reader_audit = None
        self.last_grounding_audit = None
        self.last_control_tokens_sha256 = None
        self.last_prepared_layout_audit: dict[str, Any] | None = None

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
    ) -> V83DirectSceneMemoryChatRuntime:
        """Authenticate and bind the complete environmental tensor at startup."""

        if config.get("_runtime_safe_config") is not True:
            raise ValueError("V83 chat requires a standalone validated runtime config")
        base = StaticChatRuntime.load(
            config,
            scene_id,
            checkpoint=base_checkpoint,
            audit=audit,
            local_files_only=local_files_only,
        )
        base_sha256, _files = checkpoint_fingerprint(base_checkpoint)
        loaded = load_v81_scene_memory(
            scene_memory,
            expected_scene_id=scene_id,
            expected_base_checkpoint_sha256=base_sha256,
            expected_runtime_config_sha256=effective_runtime_config_sha256(config),
            expected_model_device=base.language.device,
            record_file=None if audit is None else audit.record,
        )
        return cls(base, loaded)

    def startup_summary(self) -> dict[str, Any]:
        self.assert_prefix_unchanged()
        base = self.base.startup_summary()
        return {
            **base,
            "phase": "v83_direct_fixed_scene_memory_ready",
            "runtime_kind": RUNTIME_KIND,
            "scene_id": self.scene_id,
            "fixed_scene_memory_path": str(self.scene_memory_path),
            "fixed_scene_memory_shape": list(self.fixed_scene_memory.shape),
            "fixed_scene_memory_sha256": self.scene_prefix_hash,
            "fixed_scene_memory_tensor_sha256": self.scene_memory_tensor_sha256,
            # Override the inherited 258-token diagnostic identity: V83's
            # actual generation-time environmental input is the 738-token hash.
            "prefix_hash": self.scene_prefix_hash,
            "prefix_shape": list(self.fixed_scene_memory.shape),
            "environment_conditioned_input_sha256": self.scene_prefix_hash,
            "base_reconstructed_prefix_sha256": self.base_scene_prefix_hash,
            "base_reconstructed_prefix_shape": list(self.scene_prefix.shape),
            "scene_prefix_computed_before_question": self._questions_answered == 0,
            "fixed_memory_compiled_before_user_question": True,
            "same_fixed_memory_reused_for_every_question": True,
            "strict_fixed_environment_embedding_input": True,
            "exact_738_token_memory_supplied_directly_to_gemma": True,
            "continuous_environment_payload_tokens": DIRECT_PAYLOAD_TOKENS,
            "native_boundary_tokens": 2,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_scene_processing": False,
            "question_dependent_scene_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "reader_enabled": False,
            "control_activation_tokens": 0,
            "environmental_text_inputs": [],
            "questions_or_answers_serialized_in_memory": False,
            "compiler_or_probe_bank_loaded_by_chat": False,
            "grounding_disabled_to_avoid_a_separate_question_conditioned_readout": True,
            "startup_layout_audit": dict(self._startup_layout_audit),
            "runtime_promotion_authorized": False,
        }

    @torch.inference_mode()
    def answer(self, question: str) -> ChatAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        self.assert_prefix_unchanged()
        started = time.perf_counter()
        prompt_ids = prompt_token_ids(
            self.base.language.tokenizer,
            str(self.config["language"]["system_prompt"]),
            question,
            self.base.language.device,
        )
        backend = self.base.language.prefix_backend
        if backend is None or self.base.language.backend_name != "gemma4":
            raise RuntimeError("V83 requires the local Gemma 4 continuous-prefix backend")
        prepared = backend.prepare(
            self.fixed_scene_memory,
            prompt_ids,
            scene_prefix_after_bos=True,
            scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
            control_tokens=None,
        )
        self.last_prepared_layout_audit = audit_v83_direct_prepared_layout(
            backend=backend,
            fixed_memory=self.fixed_scene_memory,
            prompt_ids=prompt_ids,
            prepared=prepared,
        )
        generated = backend.generate(
            prepared,
            max_new_tokens=int(self.config["language"]["max_answer_tokens"]),
            eos_token_ids=self.base._eos_token_ids(),
        )
        decoded = self.base.language.tokenizer.decode(
            generated[0].detach().cpu().tolist(), skip_special_tokens=True
        ).strip()
        answer = sanitize_generated_answer(decoded)
        self.assert_prefix_unchanged()
        self._questions_answered += 1
        return ChatAnswer(
            question=question,
            answer=answer,
            # V83 intentionally performs no separate question-conditioned
            # scene readout. Grounding is evaluated by the independent V78 path.
            grounding_xyz_m=(0.0, 0.0, 0.0),
            grounding_confidence=0.0,
            grounding_support_distance_m=0.0,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=int(generated.shape[-1]),
            elapsed_seconds=time.perf_counter() - started,
        )


__all__ = [
    "DIRECT_PAYLOAD_TOKENS",
    "RUNTIME_KIND",
    "V83DirectSceneMemoryChatRuntime",
    "audit_v83_direct_prepared_layout",
]

"""Local Gemma chat over one sealed 738-token continuous scene memory.

The complete environmental memory is loaded and hash-bound before this object
can accept a user question.  At answer time a dense, positive-floor read over
all 96 numeric atlas groups reconstructs four continuous activations.  The
user question never selects or removes scene regions, and no environment is
decoded to text before Gemma receives it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.grounding_sidecar_v78_runtime import (
    V78GroundingSidecarRuntime,
)
from semantic_3d_chat.chat.question_control_runtime import sanitize_generated_answer
from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.prefix_injection import (
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    ATLAS_UNIFORM_FLOOR_MASS,
    INPUT_EMBEDDING_TENSOR_NAME,
    MAXIMUM_CONTROL_RMS,
    MINIMUM_ATLAS_WEIGHT,
    MODEL_BLOB_SHA256_IDENTITY,
    RAW_ATLAS_LOGIT_SCALE,
    V81LatestUserQuery,
    V81PrefixBinding,
    assert_prefix_binding_v81,
    audit_v75_v2_prefix_v81,
    bind_fixed_prefix_before_question_v81,
    deterministic_atlas_read_v81,
    latest_user_question_query_v81,
    reconstruct_base_v54_prefix_v81,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    BASE_PREFIX_TOKENS,
    FIXED_MEMORY_TOKENS,
    HIDDEN_SIZE,
    LoadedV81SceneMemory,
    load_v81_scene_memory,
)

RUNTIME_KIND: Final[str] = "v81_sealed_fixed_scene_memory_dense_reader"


class V81SceneMemoryChatRuntime:
    """Primary research chat runtime with an immutable pre-question memory."""

    def __init__(
        self,
        base: StaticChatRuntime,
        loaded: LoadedV81SceneMemory,
        *,
        grounding_sidecar: V78GroundingSidecarRuntime | None = None,
    ) -> None:
        self.base = base
        self.config = base.config
        self.scene_id = base.scene_id
        self.scene_memory_path = loaded.root
        self.scene_memory_metadata = dict(loaded.metadata)
        self.fixed_scene_memory = loaded.memory.detach()
        self.scene_prefix = reconstruct_base_v54_prefix_v81(self.fixed_scene_memory).detach()
        self.binding: V81PrefixBinding = bind_fixed_prefix_before_question_v81(
            self.fixed_scene_memory
        )
        self.scene_prefix_hash = self.binding.fixed_prefix_sha256
        self.base_scene_prefix_hash = self.binding.base_prefix_sha256
        self.scene_memory_tensor_sha256 = tensor_sha256(self.fixed_scene_memory)
        self._questions_answered = 0
        self.last_reader_audit: dict[str, Any] | None = None
        self.grounding_sidecar = grounding_sidecar
        self._grounding_scene_prefix = (
            None if grounding_sidecar is None else self.scene_prefix.detach().clone()
        )
        self.last_grounding_audit: dict[str, Any] | None = None
        self.last_control_tokens_sha256: str | None = None
        self.last_environment_conditioned_input_sha256 = self.scene_prefix_hash
        self.last_prepared_layout_audit: dict[str, Any] | None = None

        if tuple(self.fixed_scene_memory.shape) != (
            1,
            FIXED_MEMORY_TOKENS,
            HIDDEN_SIZE,
        ):
            raise RuntimeError("V81 loaded scene memory changed shape")
        if tuple(self.scene_prefix.shape) != (1, BASE_PREFIX_TOKENS, HIDDEN_SIZE):
            raise RuntimeError("V81 reconstructed base prefix changed shape")
        if self.scene_memory_tensor_sha256 != self.scene_memory_metadata["tensor_sha256"]:
            raise RuntimeError("V81 dtype-aware scene-memory identity changed after load")
        if self.scene_prefix_hash != self.scene_memory_metadata["canonical_prefix_sha256"]:
            raise RuntimeError("V81 canonical scene-memory identity changed after load")
        if self.base_scene_prefix_hash != self.scene_memory_metadata["base_prefix_sha256"]:
            raise RuntimeError("V81 reconstructed base-prefix identity changed after load")
        if (
            tensor_sha256(self.scene_prefix)
            != self.scene_memory_metadata["base_prefix_tensor_sha256"]
        ):
            raise RuntimeError("V81 dtype-aware base-prefix identity changed after load")
        # The sealed memory and the map-derived prefix must identify the same
        # deterministic scene.  Chat uses the sealed copy, never a recomputed
        # question-conditioned replacement.
        if tuple(base.scene_prefix.shape) != tuple(self.scene_prefix.shape) or not torch.equal(
            base.scene_prefix.to(self.scene_prefix), self.scene_prefix
        ):
            raise ValueError(
                "V81 sealed memory does not match the deterministic map-derived base prefix"
            )
        self.assert_prefix_unchanged()

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        scene_id: str,
        *,
        base_checkpoint: str | Path,
        scene_memory: str | Path,
        grounding_checkpoint: str | Path | None = None,
        audit: FileAccessAudit | None = None,
        local_files_only: bool = True,
    ) -> V81SceneMemoryChatRuntime:
        """Load and bind all environmental bytes before returning a chat object."""

        if config.get("_runtime_safe_config") is not True:
            raise ValueError("V81 chat requires a standalone validated runtime config")
        base = StaticChatRuntime.load(
            config,
            scene_id,
            checkpoint=base_checkpoint,
            audit=audit,
            local_files_only=local_files_only,
        )
        base_sha256, _files = checkpoint_fingerprint(base_checkpoint)
        runtime_sha256 = effective_runtime_config_sha256(config)
        loaded = load_v81_scene_memory(
            scene_memory,
            expected_scene_id=scene_id,
            expected_base_checkpoint_sha256=base_sha256,
            expected_runtime_config_sha256=runtime_sha256,
            expected_model_device=base.language.device,
            record_file=None if audit is None else audit.record,
        )
        grounding_sidecar = None
        if grounding_checkpoint is not None:
            grounding_scene_prefix = reconstruct_base_v54_prefix_v81(loaded.memory).detach()
            if tuple(base.scene_prefix.shape) != tuple(
                grounding_scene_prefix.shape
            ) or not torch.equal(
                base.scene_prefix.to(grounding_scene_prefix),
                grounding_scene_prefix,
            ):
                raise ValueError(
                    "V81 sealed memory does not match the deterministic map-derived base prefix"
                )
            grounding_sidecar = V78GroundingSidecarRuntime.load(
                grounding_checkpoint,
                # Bind V78 to the exact 258-token reconstruction from the
                # authenticated V81 memory, not to an independently selected
                # or question-time scene tensor.
                scene_prefix=grounding_scene_prefix,
                room_min=base.map_data.room_min,
                room_max=base.map_data.room_max,
                base_checkpoint_sha256=base_sha256,
                base_runtime_config_sha256=runtime_sha256,
                model_id=str(config["language"]["model_id"]),
                model_revision=str(config["language"]["revision"]),
                device=base.language.device,
                audit=audit,
            )
        return cls(base, loaded, grounding_sidecar=grounding_sidecar)

    @property
    def questions_answered(self) -> int:
        return self._questions_answered

    def current_prefix_hash(self) -> str:
        return prefix_sha256(self.fixed_scene_memory)

    def assert_prefix_unchanged(self) -> None:
        self.base.assert_prefix_unchanged()
        assert_prefix_binding_v81(self.fixed_scene_memory, binding=self.binding)
        if self.current_prefix_hash() != self.scene_prefix_hash:
            raise RuntimeError("V81 fixed scene memory changed after startup")
        if tensor_sha256(self.fixed_scene_memory) != self.scene_memory_tensor_sha256:
            raise RuntimeError("V81 fixed scene memory dtype or bytes changed after startup")
        if prefix_sha256(self.scene_prefix) != self.base_scene_prefix_hash:
            raise RuntimeError("V81 reconstructed base prefix changed after startup")
        if self.grounding_sidecar is not None:
            if self._grounding_scene_prefix is None:
                raise RuntimeError("V81/V78 grounding lost its bound scene prefix")
            self.grounding_sidecar.assert_prefix_unchanged(self._grounding_scene_prefix)
            if not torch.equal(self._grounding_scene_prefix, self.scene_prefix):
                raise RuntimeError("V81/V78 grounding prefix differs from generation")

    def startup_summary(self) -> dict[str, Any]:
        self.assert_prefix_unchanged()
        prefix_audit = audit_v75_v2_prefix_v81(self.fixed_scene_memory)
        base = self.base.startup_summary()
        return {
            **base,
            "phase": "v81_fixed_scene_memory_ready",
            "runtime_kind": RUNTIME_KIND,
            "scene_id": self.scene_id,
            "fixed_scene_memory_path": str(self.scene_memory_path),
            "fixed_scene_memory_shape": list(self.fixed_scene_memory.shape),
            "fixed_scene_memory_sha256": self.scene_prefix_hash,
            "fixed_scene_memory_tensor_sha256": self.scene_memory_tensor_sha256,
            "base_scene_prefix_shape": list(self.scene_prefix.shape),
            "base_scene_prefix_sha256": self.base_scene_prefix_hash,
            "scene_prefix_computed_before_question": self._questions_answered == 0,
            "fixed_memory_compiled_before_user_question": True,
            "same_fixed_memory_reused_for_every_question": True,
            "strict_fixed_environment_embedding_input": True,
            "complete_base_scene_prefix_preserved": True,
            "atlas_probe_group_count": prefix_audit.probe_count,
            "atlas_value_token_count": prefix_audit.stage_a_positive_floor_value_count,
            "all_atlas_groups_receive_positive_dense_weight": True,
            "all_384_atlas_values_receive_positive_floor_weight": True,
            "all_256_base_latents_use_native_gemma_path": True,
            "all_738_tokens_claimed_strict_positive_payload_influence": False,
            "reader_logit_scale": RAW_ATLAS_LOGIT_SCALE,
            "reader_uniform_floor_mass": ATLAS_UNIFORM_FLOOR_MASS,
            "minimum_atlas_group_weight": MINIMUM_ATLAS_WEIGHT,
            "question_dependent_scene_processing": False,
            "question_dependent_scene_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "question_conditioned_dense_reader_activations": True,
            "environmental_text_inputs": [],
            "questions_or_answers_serialized_in_memory": False,
            "compiler_or_probe_bank_loaded_by_chat": False,
            "optional_v78_grounding_enabled": self.grounding_sidecar is not None,
            "v78_grounding_uses_exact_reconstructed_base_prefix": (
                self.grounding_sidecar is not None
            ),
            "v78_grounding_numeric_map_inputs": (
                [] if self.grounding_sidecar is None else ["xyz", "confidence"]
            ),
            "optional_v78_grounding": (
                None if self.grounding_sidecar is None else self.grounding_sidecar.startup_audit()
            ),
            "prefix_audit": prefix_audit.as_dict(),
        }

    def _reader_control_tokens(
        self, query: V81LatestUserQuery
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return V81 controls; V82 overrides only this numeric reader hook."""

        reader = deterministic_atlas_read_v81(
            self.fixed_scene_memory,
            query.query,
            binding=self.binding,
        )
        if (
            not reader.finite
            or not reader.all_96_groups_positive
            or not reader.all_384_values_receive_positive_floor_weight
            or float(reader.control_rms.max().detach().cpu())
            > MAXIMUM_CONTROL_RMS + 1e-5
        ):
            raise RuntimeError("V81 dense reader failed its runtime numeric audit")
        control = reader.reconstructed_controls.to(self.scene_prefix)
        audit = {
            "architecture": "normalized_query_probe_cosine_dense_read_v81",
            "fixed_scene_memory_sha256": reader.fixed_prefix_sha256,
            "atlas_memory_sha256": reader.atlas_memory_sha256,
            "base_prefix_sha256": reader.base_prefix_sha256,
            "latest_user_token_count": query.token_count,
            "latest_user_only": True,
            "add_special_tokens": query.add_special_tokens,
            "system_prompt_in_reader_query": query.included_system_prompt,
            "history_in_reader_query": query.included_history,
            "answer_in_reader_query": query.included_answer,
            "query_detached": query.detached,
            "reader_logit_scale": RAW_ATLAS_LOGIT_SCALE,
            "uniform_floor_mass": ATLAS_UNIFORM_FLOOR_MASS,
            "minimum_attention_weight": float(
                reader.atlas_weights.min().detach().cpu()
            ),
            "attention_sum": float(reader.attention_sums.max().detach().cpu()),
            "maximum_control_rms": float(reader.control_rms.max().detach().cpu()),
            "all_96_groups_positive": reader.all_96_groups_positive,
            "all_384_values_receive_positive_floor_weight": (
                reader.all_384_values_receive_positive_floor_weight
            ),
            "all_256_base_latents_receive_positive_floor_weight": False,
            "all_256_base_latents_use_native_gemma_path": True,
            "question_dependent_scene_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "environmental_text_inputs": [],
        }
        return control, audit

    @torch.inference_mode()
    def answer(self, question: str) -> ChatAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        # Rebind before *tokenizing* the live question.  The memory was already
        # loaded, parsed, and bound in __init__ before this method was callable.
        self.assert_prefix_unchanged()
        started = time.perf_counter()
        embedding_layer = self.base.language.model.get_input_embeddings()
        query = latest_user_question_query_v81(
            tokenizer=self.base.language.tokenizer,
            embedding_layer=embedding_layer,
            latest_user_question=question,
            device=self.base.language.device,
            maximum_question_tokens=int(self.config["language"]["max_question_tokens"]),
            model_blob_sha256_identity=MODEL_BLOB_SHA256_IDENTITY,
            embedding_tensor_name=INPUT_EMBEDDING_TENSOR_NAME,
        )
        control, reader_audit = self._reader_control_tokens(query)
        prompt_ids = prompt_token_ids(
            self.base.language.tokenizer,
            str(self.config["language"]["system_prompt"]),
            question,
            self.base.language.device,
        )
        backend = self.base.language.prefix_backend
        if backend is None:
            raise RuntimeError("V81 primary runtime requires the local Gemma 4 backend")
        prepared = backend.prepare(
            self.scene_prefix,
            prompt_ids,
            scene_prefix_after_bos=scene_prefix_after_bos_setting(self.config),
            scene_boundary_mode=scene_boundary_mode_setting(self.config),
            control_tokens=control,
        )
        expected_length = BASE_PREFIX_TOKENS + int(prompt_ids.shape[1]) + int(control.shape[1])
        control_start = expected_length - int(control.shape[1])
        expected_control = control.to(prepared.inputs_embeds)
        if (
            prepared.scene_prefix_length != BASE_PREFIX_TOKENS
            or prepared.inputs_embeds.shape[1] != expected_length
            or not torch.equal(prepared.inputs_embeds[:, control_start:], expected_control)
            or bool(torch.any(prepared.mm_token_type_ids[:, control_start:] != 0))
        ):
            raise RuntimeError("V81 Gemma sequence layout changed")
        _pad_embeddings, expected_control_ple, _pad_types = backend.padding_values(
            1,
            int(control.shape[1]),
            device=self.scene_prefix.device,
        )
        if not torch.equal(prepared.per_layer_inputs[:, control_start:], expected_control_ple):
            raise RuntimeError("V81 controls lost their exact PAD-PLE identity")
        generated = backend.generate(
            prepared,
            max_new_tokens=int(self.config["language"]["max_answer_tokens"]),
            eos_token_ids=self.base._eos_token_ids(),
        )
        decoded = self.base.language.tokenizer.decode(
            generated[0].detach().cpu().tolist(), skip_special_tokens=True
        ).strip()
        answer = sanitize_generated_answer(decoded)
        question_embeddings = embedding_layer(query.token_ids)
        if self.grounding_sidecar is None:
            grounding_xyz, confidence, support_distance = self.base._predict_grounding(
                question_embeddings
            )
            self.last_grounding_audit = None
        else:
            if self._grounding_scene_prefix is None:
                raise RuntimeError("V81/V78 grounding lost its bound scene prefix")
            map_xyz = self.base.map_data.xyz
            map_confidence = self.base.map_data.confidence
            grounded = self.grounding_sidecar.predict(
                question_embeddings,
                scene_prefix=self._grounding_scene_prefix,
                map_xyz=map_xyz,
                map_confidence=map_confidence,
            )
            grounding_xyz = grounded.xyz_m
            confidence = grounded.confidence
            support_distance = grounded.support_distance_m
            self.last_grounding_audit = {
                **grounded.audit,
                "exact_reconstructed_base_scene_prefix": True,
                "full_base_prefix_sha256": self.base_scene_prefix_hash,
                "numeric_map_inputs": ["xyz", "confidence"],
                "map_xyz_shape": list(map_xyz.shape),
                "map_confidence_shape": list(map_confidence.shape),
            }
        self.last_control_tokens_sha256 = prefix_sha256(control)
        self.last_environment_conditioned_input_sha256 = self.scene_prefix_hash
        self.last_reader_audit = reader_audit
        self.last_prepared_layout_audit = {
            "sequence": [
                "prompt_bos",
                "base_scene_prefix_258",
                "remaining_system_and_latest_user_prompt",
                "four_dense_reader_activations",
            ],
            "base_scene_prefix_tokens": BASE_PREFIX_TOKENS,
            "prompt_tokens": int(prompt_ids.shape[1]),
            "control_activation_tokens": int(control.shape[1]),
            "prepared_tokens": int(prepared.inputs_embeds.shape[1]),
            "control_pad_ple": True,
            "control_text_modality_zero": True,
        }
        self.assert_prefix_unchanged()
        self._questions_answered += 1
        return ChatAnswer(
            question=question,
            answer=answer,
            grounding_xyz_m=grounding_xyz,
            grounding_confidence=confidence,
            grounding_support_distance_m=support_distance,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=int(generated.shape[-1]),
            elapsed_seconds=time.perf_counter() - started,
        )


__all__ = ["RUNTIME_KIND", "V81SceneMemoryChatRuntime"]

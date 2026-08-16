"""Static chat whose complete environment-conditioned input is immutable.

Unlike the experimental question-control runtime, this runtime never sends the
live user question through a scene adapter.  It compiles a continuous
scene-only key/value atlas during construction, destroys the compiler objects,
and only then accepts ordinary text questions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.language.local_lm import prompt_token_ids, question_token_ids
from semantic_3d_chat.language.prefix_injection import (
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import (
    FixedPrefixAtlasAudit,
    compile_fixed_scene_atlas,
)
from semantic_3d_chat.training.fixed_prefix_atlas_checkpoint import (
    load_fixed_prefix_atlas_checkpoint,
)


class FixedPrefixAtlasChatRuntime:
    """Chat over one pre-question, globally complete continuous scene prefix."""

    def __init__(
        self,
        base: StaticChatRuntime,
        *,
        fixed_scene_prefix: torch.Tensor,
        atlas_audit: FixedPrefixAtlasAudit,
        atlas_metadata: dict[str, Any],
        atlas_checkpoint_path: Path,
    ) -> None:
        self.base = base
        self.config = base.config
        self.scene_id = base.scene_id
        self.scene_prefix = fixed_scene_prefix.detach()
        self.scene_prefix_hash = prefix_sha256(self.scene_prefix)
        self.base_scene_prefix_hash = base.scene_prefix_hash
        self.atlas_audit = atlas_audit
        self.atlas_metadata = dict(atlas_metadata)
        self.atlas_checkpoint_path = atlas_checkpoint_path
        self._questions_answered = 0
        if self.scene_prefix_hash != atlas_audit.fixed_scene_prefix_sha256:
            raise RuntimeError("Fixed-prefix atlas hash changed after compilation")
        if atlas_audit.user_question_inputs_used_for_compilation:
            raise ValueError("Fixed-prefix runtime refuses a question-conditioned atlas")
        if atlas_audit.question_dependent_scene_processing:
            raise ValueError("Fixed-prefix runtime refuses question-dependent scene processing")

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        scene_id: str,
        *,
        base_checkpoint: str | Path,
        atlas_checkpoint: str | Path,
        audit: FileAccessAudit | None = None,
        local_files_only: bool = True,
    ) -> FixedPrefixAtlasChatRuntime:
        """Compile the final prefix before returning an object that can answer."""

        base = StaticChatRuntime.load(
            config,
            scene_id,
            checkpoint=base_checkpoint,
            audit=audit,
            local_files_only=local_files_only,
        )
        base_sha256, _files = checkpoint_fingerprint(base.checkpoint_path)
        runtime_sha256 = effective_runtime_config_sha256(config)
        loaded = load_fixed_prefix_atlas_checkpoint(
            atlas_checkpoint,
            device=base.language.device,
            expected_hidden_size=base.language.hidden_size,
            expected_base_checkpoint_sha256=base_sha256,
            expected_base_runtime_config_sha256=runtime_sha256,
            record_file=None if audit is None else audit.record,
        )
        compiled = compile_fixed_scene_atlas(
            base.scene_prefix,
            loaded.controller,
            loaded.probe_embeddings,
        )
        if (
            compiled.audit.probe_count != loaded.metadata["probe_count"]
            or compiled.audit.values_per_probe != loaded.metadata["values_per_probe"]
            or compiled.audit.fixed_prefix_token_count
            != loaded.metadata["fixed_prefix_tokens"]
            or compiled.audit.probe_bank_sha256 != loaded.metadata["probe_bank_sha256"]
        ):
            raise RuntimeError("Compiled scene atlas differs from its checkpoint contract")
        # The learned compiler and fixed probes are startup-only.  They are not
        # retained on the chat object, so answer() has no callable scene adapter.
        metadata = dict(loaded.metadata)
        checkpoint_path = loaded.checkpoint_path
        del loaded
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return cls(
            base,
            fixed_scene_prefix=compiled.scene_prefix,
            atlas_audit=compiled.audit,
            atlas_metadata=metadata,
            atlas_checkpoint_path=checkpoint_path,
        )

    @property
    def questions_answered(self) -> int:
        return self._questions_answered

    def current_prefix_hash(self) -> str:
        return prefix_sha256(self.scene_prefix)

    def assert_prefix_unchanged(self) -> None:
        self.base.assert_prefix_unchanged()
        observed = self.current_prefix_hash()
        if observed != self.scene_prefix_hash:
            raise RuntimeError(
                "Strict fixed scene prefix changed after startup: "
                f"{self.scene_prefix_hash} != {observed}"
            )

    def startup_summary(self) -> dict[str, Any]:
        self.assert_prefix_unchanged()
        base = self.base.startup_summary()
        return {
            **base,
            "phase": "fixed_prefix_atlas_scene_ready",
            "base_scene_prefix_hash": self.base_scene_prefix_hash,
            "prefix_hash": self.scene_prefix_hash,
            "prefix_shape": list(self.scene_prefix.shape),
            "base_scene_latents": self.atlas_audit.environment_latent_count,
            "atlas_probe_count": self.atlas_audit.probe_count,
            "atlas_values_per_probe": self.atlas_audit.values_per_probe,
            "atlas_memory_tokens": self.atlas_audit.atlas_memory_token_count,
            "atlas_checkpoint": str(self.atlas_checkpoint_path),
            "prefix_compiled_before_user_question": True,
            "scene_prefix_computed_before_question": self._questions_answered == 0,
            "strict_fixed_environment_embedding_input": True,
            "environment_conditioned_input_sha256": self.scene_prefix_hash,
            "question_conditioned_scene_readout_tokens": False,
            "user_question_inputs_used_for_compilation": False,
            "question_dependent_scene_processing": False,
            "language_model_environment_conditioning_question_dependent": False,
            "question_dependent_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "auxiliary_grounding_question_conditioned": True,
            "auxiliary_grounding_affects_language_model": False,
            "complete_base_scene_prefix_preserved": True,
            "compiler_retained_after_startup": False,
            "environmental_text_inputs": [],
            "atlas_audit": self.atlas_audit.as_dict(),
        }

    def answer(self, question: str) -> ChatAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        # The immutable environmental bytes are checked before tokenizing the
        # live question.  No scene/compiler method is called below this point.
        self.assert_prefix_unchanged()
        question_count = self.base._question_token_count(question)
        maximum = int(self.config["language"]["max_question_tokens"])
        if question_count > maximum:
            raise ValueError(f"Question has {question_count} tokens; maximum is {maximum}")
        started = time.perf_counter()
        prompt_ids = prompt_token_ids(
            self.base.language.tokenizer,
            str(self.config["language"]["system_prompt"]),
            question,
            self.base.language.device,
        )
        embedding_layer = self.base.language.model.get_input_embeddings()
        with torch.inference_mode():
            generated = self.base.language.generate_from_scene_prefix(
                self.scene_prefix,
                prompt_ids,
                max_new_tokens=int(self.config["language"]["max_answer_tokens"]),
                eos_token_ids=self.base._eos_token_ids(),
                scene_prefix_after_bos=scene_prefix_after_bos_setting(self.config),
                scene_boundary_mode=scene_boundary_mode_setting(self.config),
                fallback=self.base._generation_function,
            )
            # Grounding is an auxiliary diagnostic and does not add or alter
            # any language-model scene token.
            grounding_ids = question_token_ids(
                self.base.language.tokenizer,
                question,
                self.base.language.device,
            )
            grounding_embeddings = embedding_layer(grounding_ids)
            grounding_xyz, grounding_confidence, support_distance = (
                self.base._predict_grounding(grounding_embeddings)
            )
        decoded = self.base.language.tokenizer.decode(
            generated[0].detach().cpu().tolist(), skip_special_tokens=True
        ).strip()
        self.assert_prefix_unchanged()
        self._questions_answered += 1
        return ChatAnswer(
            question=question,
            answer=decoded or "unknown",
            grounding_xyz_m=grounding_xyz,
            grounding_confidence=grounding_confidence,
            grounding_support_distance_m=support_distance,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=int(generated.shape[-1]),
            elapsed_seconds=time.perf_counter() - started,
        )


__all__ = ["FixedPrefixAtlasChatRuntime"]

"""Local Gemma chat using the learned V82 reader over sealed V81 memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.grounding_sidecar_v78_runtime import (
    V78GroundingSidecarRuntime,
)
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.chat.v81_scene_memory_runtime import (
    V81SceneMemoryChatRuntime,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    MAXIMUM_CONTROL_RMS,
    V81LatestUserQuery,
    reconstruct_base_v54_prefix_v81,
)
from semantic_3d_chat.language.v82_dense_learned_reader import (
    ARCHITECTURE,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    LoadedV81SceneMemory,
    load_v81_scene_memory,
)
from semantic_3d_chat.training.v82_reader_artifacts import (
    LoadedV82Candidate,
    load_v82_candidate,
)

RUNTIME_KIND: Final[str] = "v82_sealed_fixed_scene_memory_learned_dense_reader"


class V82SceneMemoryChatRuntime(V81SceneMemoryChatRuntime):
    """V81 runtime with only the dense numeric reader replaced by V82."""

    def __init__(
        self,
        base: StaticChatRuntime,
        loaded_memory: LoadedV81SceneMemory,
        learned_reader: LoadedV82Candidate,
        *,
        grounding_sidecar: V78GroundingSidecarRuntime | None = None,
    ) -> None:
        self.learned_reader = learned_reader.model
        self.learned_reader_metadata = dict(learned_reader.metadata)
        self.learned_reader_path = learned_reader.root
        self._learned_reader_state = {
            name: value.detach().cpu().clone()
            for name, value in self.learned_reader.state_dict().items()
        }
        super().__init__(
            base,
            loaded_memory,
            grounding_sidecar=grounding_sidecar,
        )

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        scene_id: str,
        *,
        base_checkpoint: str | Path,
        scene_memory: str | Path,
        reader_checkpoint: str | Path,
        grounding_checkpoint: str | Path | None = None,
        audit: FileAccessAudit | None = None,
        local_files_only: bool = True,
    ) -> V82SceneMemoryChatRuntime:
        if config.get("_runtime_safe_config") is not True:
            raise ValueError("V82 chat requires a standalone validated runtime config")
        base = StaticChatRuntime.load(
            config,
            scene_id,
            checkpoint=base_checkpoint,
            audit=audit,
            local_files_only=local_files_only,
        )
        base_sha256, _files = checkpoint_fingerprint(base_checkpoint)
        runtime_sha256 = effective_runtime_config_sha256(config)
        loaded_memory = load_v81_scene_memory(
            scene_memory,
            expected_scene_id=scene_id,
            expected_base_checkpoint_sha256=base_sha256,
            expected_runtime_config_sha256=runtime_sha256,
            expected_model_device=base.language.device,
            record_file=None if audit is None else audit.record,
        )
        learned_reader = load_v82_candidate(
            reader_checkpoint,
            device=base.language.device,
            record_file=None if audit is None else audit.record,
        )
        grounding_sidecar = None
        if grounding_checkpoint is not None:
            grounding_scene_prefix = reconstruct_base_v54_prefix_v81(
                loaded_memory.memory
            ).detach()
            if tuple(base.scene_prefix.shape) != tuple(
                grounding_scene_prefix.shape
            ) or not torch.equal(
                base.scene_prefix.to(grounding_scene_prefix),
                grounding_scene_prefix,
            ):
                raise ValueError(
                    "V82 sealed memory does not match the map-derived base prefix"
                )
            grounding_sidecar = V78GroundingSidecarRuntime.load(
                grounding_checkpoint,
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
        return cls(
            base,
            loaded_memory,
            learned_reader,
            grounding_sidecar=grounding_sidecar,
        )

    def assert_prefix_unchanged(self) -> None:
        super().assert_prefix_unchanged()
        observed = self.learned_reader.state_dict()
        if set(observed) != set(self._learned_reader_state):
            raise RuntimeError("V82 learned-reader tensor inventory changed")
        for name, value in observed.items():
            if value.requires_grad or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"V82 learned-reader parameter is mutable: {name}")
            if not torch.equal(value.detach().cpu(), self._learned_reader_state[name]):
                raise RuntimeError(f"V82 learned-reader parameter changed: {name}")

    def startup_summary(self) -> dict[str, Any]:
        result = super().startup_summary()
        result.update(
            {
                "phase": "v82_fixed_scene_memory_learned_reader_ready",
                "runtime_kind": RUNTIME_KIND,
                "reader_architecture": ARCHITECTURE,
                "reader_checkpoint_path": str(self.learned_reader_path),
                "reader_weights_sha256": self.learned_reader_metadata[
                    "weights_sha256"
                ],
                "reader_trainable_parameter_count_during_fit": (
                    self.learned_reader_metadata["trainable_parameter_count"]
                ),
                "reader_parameters_frozen_at_runtime": True,
                "all_384_atlas_values_receive_positive_floor_weight": True,
                "all_256_base_latents_receive_positive_floor_weight": True,
                "boi_eoi_and_96_probe_keys_are_not_payload": True,
                "all_738_tokens_claimed_strict_positive_payload_influence": False,
                "training_cache_loaded_by_chat": False,
                "development_cache_loaded_by_chat": False,
                "questions_or_answers_serialized_in_reader": False,
                "runtime_promotion_authorized": False,
            }
        )
        return result

    def _reader_control_tokens(
        self, query: V81LatestUserQuery
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        output = self.learned_reader(
            self.fixed_scene_memory,
            query.query,
            binding=self.binding,
        )
        if (
            not output.all_96_groups_positive
            or not output.all_384_atlas_values_positive
            or not output.all_256_base_latents_positive
            or output.zero_environmental_payload
            or float(output.control_rms.max().detach().cpu())
            > MAXIMUM_CONTROL_RMS + 1e-5
        ):
            raise RuntimeError("V82 learned reader failed its runtime numeric audit")
        controls = output.controls.to(self.scene_prefix)
        return controls, {
            "architecture": ARCHITECTURE,
            "reader_weights_sha256": self.learned_reader_metadata["weights_sha256"],
            "fixed_scene_memory_sha256": self.scene_prefix_hash,
            "base_prefix_sha256": self.base_scene_prefix_hash,
            "latest_user_token_count": query.token_count,
            "latest_user_only": True,
            "add_special_tokens": query.add_special_tokens,
            "system_prompt_in_reader_query": query.included_system_prompt,
            "history_in_reader_query": query.included_history,
            "answer_in_reader_query": query.included_answer,
            "query_detached": query.detached,
            "minimum_atlas_attention_weight": float(
                output.atlas_weights.min().detach().cpu()
            ),
            "minimum_base_attention_weight": float(
                output.base_weights.min().detach().cpu()
            ),
            "atlas_attention_sum": float(
                output.atlas_attention_sums.max().detach().cpu()
            ),
            "base_attention_sum": float(
                output.base_attention_sums.max().detach().cpu()
            ),
            "maximum_control_rms": float(
                output.control_rms.max().detach().cpu()
            ),
            "all_96_groups_positive": output.all_96_groups_positive,
            "all_384_values_receive_positive_floor_weight": (
                output.all_384_atlas_values_positive
            ),
            "all_256_base_latents_receive_positive_floor_weight": (
                output.all_256_base_latents_positive
            ),
            "boi_eoi_and_96_probe_keys_are_not_payload": True,
            "strict_positive_payload_claim_for_all_738_tokens": False,
            "question_dependent_scene_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "environmental_text_inputs": [],
        }


__all__ = ["RUNTIME_KIND", "V82SceneMemoryChatRuntime"]

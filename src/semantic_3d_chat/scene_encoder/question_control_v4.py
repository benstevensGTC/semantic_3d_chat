"""V4 scene-conditioned routing over the frozen V60 teacher-basis controller.

V60 learned excellent continuous control values but used a question-only gate,
which cannot assign different routes to identical wording in different rooms.
V4 copies every V60 tensor exactly, freezes that complete inherited state, and
learns only a small gate over V60's normalized full-scene-by-question trunk.
No scene token is retrieved, ranked, or omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisControlOutput,
    TeacherBasisFullSceneQuestionControlV3,
)


@dataclass(frozen=True)
class SceneConditionedGateAuditV4:
    scene_token_count: int
    environment_latent_count: int
    control_token_count: int
    scene_moment_count: int
    output_basis_rank: int
    every_environment_latent_influenced_signature: bool
    control_values_scene_question_bilinear: bool
    gate_scene_question_conditioned: bool
    inherited_v60_state_frozen: bool
    question_dependent_scene_retrieval: bool
    softmax_scene_attention_used: bool
    gate_probability: float
    control_used: bool
    maximum_control_rms: float


class SceneConditionedGateTeacherBasisControlV4(TeacherBasisFullSceneQuestionControlV3):
    """A frozen V60 value branch with a compact scene-conditioned route gate."""

    def __init__(
        self,
        hidden_size: int,
        output_basis: torch.Tensor,
        *,
        control_tokens: int = 4,
        expected_environment_latents: int = 256,
        moment_count: int = 8,
        interaction_dim: int = 24,
        trunk_dim: int = 128,
        maximum_control_rms: float = 0.2,
        initial_control_rms: float = 0.075,
        gate_threshold: float = 0.5,
        gate_hidden_dim: int = 32,
    ) -> None:
        if (
            isinstance(gate_hidden_dim, bool)
            or not isinstance(gate_hidden_dim, int)
            or gate_hidden_dim < 1
        ):
            raise ValueError("V4 gate_hidden_dim must be a positive integer")
        super().__init__(
            hidden_size,
            output_basis,
            control_tokens=control_tokens,
            expected_environment_latents=expected_environment_latents,
            moment_count=moment_count,
            interaction_dim=interaction_dim,
            trunk_dim=trunk_dim,
            maximum_control_rms=maximum_control_rms,
            initial_control_rms=initial_control_rms,
            gate_threshold=gate_threshold,
        )
        self.gate_hidden_dim = int(gate_hidden_dim)
        gate_width = self.control_token_count * self.trunk_dim
        self.scene_question_gate = nn.Sequential(
            nn.LayerNorm(gate_width),
            nn.Linear(gate_width, self.gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.gate_hidden_dim, 1),
        )
        self._last_v4_audit: SceneConditionedGateAuditV4 | None = None

    @property
    def inherited_state_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.state_dict() if not name.startswith("scene_question_gate.")
        )

    @property
    def gate_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("scene_question_gate.")
        )

    def freeze_inherited_v60_state(self) -> None:
        """Freeze every inherited V3/V60 parameter, including its unused route."""

        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("scene_question_gate."))

    @property
    def inherited_v60_state_frozen(self) -> bool:
        return all(
            not parameter.requires_grad
            for name, parameter in self.named_parameters()
            if not name.startswith("scene_question_gate.")
        )

    @classmethod
    def from_v60(
        cls,
        source: TeacherBasisFullSceneQuestionControlV3,
        *,
        gate_hidden_dim: int = 32,
    ) -> SceneConditionedGateTeacherBasisControlV4:
        """Construct V4 and prove its inherited state is an exact V60 copy."""

        if type(source) is not TeacherBasisFullSceneQuestionControlV3:
            raise TypeError("V4 source must be an exact V3/V60 controller")
        module = cls(
            source.hidden_size,
            source.output_basis.detach().clone(),
            control_tokens=source.control_token_count,
            expected_environment_latents=source.expected_environment_latents,
            moment_count=source.moment_count,
            interaction_dim=source.interaction_dim,
            trunk_dim=source.trunk_dim,
            maximum_control_rms=source.maximum_control_rms,
            initial_control_rms=source.initial_control_rms,
            gate_threshold=source.gate_threshold,
            gate_hidden_dim=gate_hidden_dim,
        )
        source_state = source.state_dict()
        missing, unexpected = module.load_state_dict(source_state, strict=False)
        expected_missing = {
            name for name in module.state_dict() if name.startswith("scene_question_gate.")
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"V4 inherited state contract changed: missing={missing} unexpected={unexpected}"
            )
        if set(source_state) != set(module.inherited_state_names) or any(
            not torch.equal(source_state[name].cpu(), module.state_dict()[name].cpu())
            for name in source_state
        ):
            raise RuntimeError("V4 failed to copy V60 state exactly")
        module.freeze_inherited_v60_state()
        return module

    def _value_trunk(
        self,
        scene_signature: torch.Tensor,
        normalized_question: torch.Tensor,
    ) -> torch.Tensor:
        scene_factors = self.scene_projection(scene_signature.float())
        question_factors = self.question_projection(normalized_question).reshape(
            -1, self.control_token_count, self.interaction_dim
        )
        interaction = (scene_factors[:, None, :, :] * question_factors[:, :, None, :]).reshape(
            -1,
            self.control_token_count,
            self.moment_count * self.interaction_dim,
        )
        return self.control_trunk(interaction)

    def gate_logits_from_trunk(self, value_trunk: torch.Tensor) -> torch.Tensor:
        if value_trunk.ndim != 3 or value_trunk.shape[1:] != (
            self.control_token_count,
            self.trunk_dim,
        ):
            raise ValueError("V4 value trunk shape changed")
        return self.scene_question_gate(value_trunk.float().flatten(1)).squeeze(-1)

    def forward_from_signature(
        self,
        scene_signature: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> TeacherBasisControlOutput:
        if scene_signature.ndim != 3 or scene_signature.shape[1:] != (
            self.moment_count,
            self.hidden_size,
        ):
            raise ValueError("V4 scene signature shape changed")
        if question_embeddings.ndim != 3 or question_embeddings.shape[-1] != self.hidden_size:
            raise ValueError("V4 question embeddings must have shape [B,Q,H]")
        if scene_signature.shape[0] != question_embeddings.shape[0]:
            raise ValueError("V4 scene and question batch sizes must match")
        # Delegate the entire continuous value path to the inherited V60
        # implementation.  V4 only replaces routing, so these returned tensors
        # are bit-identical to V60 for identical state and inputs.
        inherited = super().forward_from_signature(
            scene_signature,
            question_embeddings,
            question_attention_mask,
        )
        normalized_question = self.normalized_question(question_embeddings, question_attention_mask)
        value_trunk = self._value_trunk(scene_signature, normalized_question)
        gate_logits = self.gate_logits_from_trunk(value_trunk)
        gate_probabilities = torch.sigmoid(gate_logits)
        if not all(
            torch.isfinite(value).all()
            for value in (
                inherited.control_tokens,
                inherited.coefficient_directions,
                inherited.control_rms,
                gate_probabilities,
            )
        ):
            raise RuntimeError("V4 controller produced NaN or infinity")
        if scene_signature.shape[0] == 1:
            probability = float(gate_probabilities[0].detach().cpu())
            self._last_v4_audit = SceneConditionedGateAuditV4(
                scene_token_count=self.expected_environment_latents + 2,
                environment_latent_count=self.expected_environment_latents,
                control_token_count=self.control_token_count,
                scene_moment_count=self.moment_count,
                output_basis_rank=self.output_basis_rank,
                every_environment_latent_influenced_signature=True,
                control_values_scene_question_bilinear=True,
                gate_scene_question_conditioned=True,
                inherited_v60_state_frozen=self.inherited_v60_state_frozen,
                question_dependent_scene_retrieval=False,
                softmax_scene_attention_used=False,
                gate_probability=probability,
                control_used=probability >= self.gate_threshold,
                maximum_control_rms=float(inherited.control_rms.max().detach().cpu()),
            )
        else:
            self._last_v4_audit = None
        return TeacherBasisControlOutput(
            control_tokens=inherited.control_tokens,
            coefficient_directions=inherited.coefficient_directions,
            control_rms=inherited.control_rms,
            gate_logits=gate_logits,
            gate_probabilities=gate_probabilities,
        )

    def audit(self) -> SceneConditionedGateAuditV4:
        if self._last_v4_audit is None:
            raise RuntimeError("V4 audit requires a completed batch-one forward pass")
        return self._last_v4_audit


__all__ = [
    "SceneConditionedGateAuditV4",
    "SceneConditionedGateTeacherBasisControlV4",
]

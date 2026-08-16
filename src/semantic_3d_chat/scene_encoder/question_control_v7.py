"""Always-on continuous full-scene question control.

V7 deliberately removes the learned route decision that failed to generalize
in V65.  It retains V3's bounded, scene-by-question bilinear value function
and fixed global DCT moments over every environment latent, but every valid
question receives the resulting continuous tokens.  The class has no
question-dependent scene retrieval and does not turn its numeric output into
text before Gemma consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisControlOutput,
    TeacherBasisFullSceneQuestionControlV3,
)


@dataclass(frozen=True)
class AlwaysOnControlAuditV7:
    scene_token_count: int
    environment_latent_count: int
    control_token_count: int
    scene_moment_count: int
    output_basis_rank: int
    every_environment_latent_influenced_signature: bool
    control_values_scene_question_bilinear: bool
    question_dependent_scene_retrieval: bool
    softmax_scene_attention_used: bool
    gate_probability: float
    control_used: bool
    maximum_control_rms: float
    always_on_continuous_control: bool
    gate_scene_question_conditioned: bool
    exact_no_control_route: bool
    legacy_route_parameters_ignored: bool


class AlwaysOnTeacherBasisFullSceneQuestionControlV7(
    TeacherBasisFullSceneQuestionControlV3
):
    """Use every bounded continuous value prediction without a route gate."""

    _ALWAYS_ON_LOGIT = 20.0

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
    ) -> None:
        # The inherited threshold is retained only for state compatibility.
        # V7 never consults inherited route logits or this threshold.
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
            gate_threshold=0.5,
        )
        for name, parameter in self.named_parameters():
            if name.startswith("route_"):
                parameter.requires_grad_(False)
        self._last_v7_audit: AlwaysOnControlAuditV7 | None = None

    @classmethod
    def from_v3(
        cls,
        source: TeacherBasisFullSceneQuestionControlV3,
    ) -> AlwaysOnTeacherBasisFullSceneQuestionControlV7:
        """Copy a fitted V3 value function without changing any value tensor."""

        if type(source) is not TeacherBasisFullSceneQuestionControlV3:
            raise TypeError("V7 source must be an exact fitted V3 controller")
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
        )
        module.load_state_dict(source.state_dict(), strict=True)
        source_state = source.state_dict()
        copied_state = module.state_dict()
        if set(source_state) != set(copied_state) or any(
            not torch.equal(source_state[name].cpu(), copied_state[name].cpu())
            for name in source_state
        ):
            raise RuntimeError("V7 failed to copy the fitted V3 state exactly")
        return module

    def forward_from_signature(
        self,
        scene_signature: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> TeacherBasisControlOutput:
        inherited = super().forward_from_signature(
            scene_signature,
            question_embeddings,
            question_attention_mask,
        )
        logits = torch.full_like(inherited.gate_logits, self._ALWAYS_ON_LOGIT)
        probabilities = torch.sigmoid(logits)
        if scene_signature.shape[0] == 1:
            self._last_v7_audit = AlwaysOnControlAuditV7(
                scene_token_count=self.expected_environment_latents + 2,
                environment_latent_count=self.expected_environment_latents,
                control_token_count=self.control_token_count,
                scene_moment_count=self.moment_count,
                output_basis_rank=self.output_basis_rank,
                every_environment_latent_influenced_signature=True,
                control_values_scene_question_bilinear=True,
                question_dependent_scene_retrieval=False,
                softmax_scene_attention_used=False,
                gate_probability=float(probabilities[0].detach().cpu()),
                control_used=True,
                maximum_control_rms=float(
                    inherited.control_rms.max().detach().cpu()
                ),
                always_on_continuous_control=True,
                gate_scene_question_conditioned=False,
                exact_no_control_route=False,
                legacy_route_parameters_ignored=True,
            )
        else:
            self._last_v7_audit = None
        return TeacherBasisControlOutput(
            control_tokens=inherited.control_tokens,
            coefficient_directions=inherited.coefficient_directions,
            control_rms=inherited.control_rms,
            gate_logits=logits,
            gate_probabilities=probabilities,
        )

    def audit(self) -> AlwaysOnControlAuditV7:
        if self._last_v7_audit is None:
            raise RuntimeError("V7 audit requires a completed batch-one forward pass")
        return self._last_v7_audit


__all__ = [
    "AlwaysOnControlAuditV7",
    "AlwaysOnTeacherBasisFullSceneQuestionControlV7",
]

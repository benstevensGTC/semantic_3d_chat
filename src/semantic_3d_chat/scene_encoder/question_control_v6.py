"""Unified scene-conditioned magnitude gating over V3 continuous values."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisControlOutput,
    TeacherBasisFullSceneQuestionControlV3,
)


@dataclass(frozen=True)
class MagnitudeGatedControlAuditV6:
    scene_token_count: int
    environment_latent_count: int
    control_token_count: int
    scene_moment_count: int
    output_basis_rank: int
    every_environment_latent_influenced_signature: bool
    control_values_scene_question_bilinear: bool
    gate_scene_question_conditioned: bool
    question_dependent_scene_retrieval: bool
    softmax_scene_attention_used: bool
    gate_probability: float
    control_used: bool
    maximum_control_rms: float
    activation_rms: float
    activation_rms_threshold: float
    exact_no_control_below_threshold: bool


class MagnitudeGatedTeacherBasisFullSceneQuestionControlV6(
    TeacherBasisFullSceneQuestionControlV3
):
    """Route from the magnitude of the same scene-question value function.

    V6 introduces no independent classifier and no retrieval.  The inherited
    V3 value trunk predicts four bounded continuous tokens from fixed DCT
    moments of every scene latent and the current question.  The maximum token
    RMS is the activation score.  Production inserts all tokens iff that score
    is at least the fixed threshold; otherwise it inserts no control token at
    all, preserving the base model's exact no-token execution path.
    """

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
        activation_rms_threshold: float = 0.01,
    ) -> None:
        if (
            isinstance(activation_rms_threshold, bool)
            or not isinstance(activation_rms_threshold, (int, float))
            or not math.isfinite(float(activation_rms_threshold))
            or not 0.0 < float(activation_rms_threshold) < float(maximum_control_rms)
        ):
            raise ValueError(
                "V6 activation_rms_threshold must be finite and in "
                "(0, maximum_control_rms)"
            )
        # V3's legacy threshold remains valid metadata/state but V6 never uses
        # its question-prototype route.  Supplying 0.5 keeps construction
        # stable while the explicit RMS threshold owns all routing semantics.
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
        self.activation_rms_threshold = float(activation_rms_threshold)
        self._last_v6_audit: MagnitudeGatedControlAuditV6 | None = None

    @classmethod
    def from_v65(
        cls,
        source: TeacherBasisFullSceneQuestionControlV3,
        *,
        activation_rms_threshold: float = 0.01,
    ) -> MagnitudeGatedTeacherBasisFullSceneQuestionControlV6:
        if type(source) is not TeacherBasisFullSceneQuestionControlV3:
            raise TypeError("V6 source must be an exact V3/V65 controller")
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
            activation_rms_threshold=activation_rms_threshold,
        )
        module.load_state_dict(source.state_dict(), strict=True)
        if set(source.state_dict()) != set(module.state_dict()) or any(
            not torch.equal(
                source.state_dict()[name].cpu(), module.state_dict()[name].cpu()
            )
            for name in source.state_dict()
        ):
            raise RuntimeError("V6 failed to copy all V65 value tensors exactly")
        return module

    @staticmethod
    def activation_rms(control_rms: torch.Tensor) -> torch.Tensor:
        if (
            control_rms.ndim != 2
            or control_rms.shape[1] < 1
            or not torch.isfinite(control_rms).all()
            or torch.any(control_rms < 0.0)
        ):
            raise ValueError("V6 control RMS must be finite nonnegative [B,C]")
        return control_rms.max(dim=-1).values

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
        activation = self.activation_rms(inherited.control_rms)
        # A monotone signed margin is exposed as a diagnostic logit.  Routing
        # itself compares native RMS directly and never relies on sigmoid
        # roundoff or a learned classifier.
        gate_logits = (activation - self.activation_rms_threshold) / max(
            self.activation_rms_threshold,
            1e-8,
        )
        # Keep the inherited output contract internally coherent: callers that
        # inspect the diagnostic probability must see sigmoid(gate_logits),
        # with exactly 0.5 at the RMS decision boundary.  Production routing
        # still compares native RMS directly below.
        probability = torch.sigmoid(gate_logits)
        if scene_signature.shape[0] == 1:
            score = float(activation[0].detach().cpu())
            self._last_v6_audit = MagnitudeGatedControlAuditV6(
                scene_token_count=self.expected_environment_latents + 2,
                environment_latent_count=self.expected_environment_latents,
                control_token_count=self.control_token_count,
                scene_moment_count=self.moment_count,
                output_basis_rank=self.output_basis_rank,
                every_environment_latent_influenced_signature=True,
                control_values_scene_question_bilinear=True,
                gate_scene_question_conditioned=True,
                question_dependent_scene_retrieval=False,
                softmax_scene_attention_used=False,
                gate_probability=float(probability[0].detach().cpu()),
                control_used=score >= self.activation_rms_threshold,
                maximum_control_rms=float(inherited.control_rms.max().detach().cpu()),
                activation_rms=score,
                activation_rms_threshold=self.activation_rms_threshold,
                exact_no_control_below_threshold=True,
            )
        else:
            self._last_v6_audit = None
        return TeacherBasisControlOutput(
            control_tokens=inherited.control_tokens,
            coefficient_directions=inherited.coefficient_directions,
            control_rms=inherited.control_rms,
            gate_logits=gate_logits,
            gate_probabilities=probability,
        )

    def audit(self) -> MagnitudeGatedControlAuditV6:
        if self._last_v6_audit is None:
            raise RuntimeError("V6 audit requires a completed batch-one forward pass")
        return self._last_v6_audit


__all__ = [
    "MagnitudeGatedControlAuditV6",
    "MagnitudeGatedTeacherBasisFullSceneQuestionControlV6",
]

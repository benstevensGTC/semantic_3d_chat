"""Normalized factorized scene-question routing over frozen V60 values.

V5 deliberately separates route learning from V60's prompt-regression trunk.
The route head has one normalized projection for the frozen pooled question and
another normalized projection over *all* fixed DCT scene moments.  A low-rank
diagonal bilinear compatibility plus small linear calibration terms produces a
single gate logit.  It performs no retrieval, ranking, or softmax over scene
content.

The inherited V60 controller remains the sole producer of continuous control
values.  Every inherited tensor is copied exactly and frozen; V5 can therefore
change only whether those byte-identical values are injected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisControlOutput,
    TeacherBasisFullSceneQuestionControlV3,
)


@dataclass(frozen=True)
class FactorizedRouteFeaturesV5:
    """Separately normalized factors suitable for pair-disjoint caching."""

    question: torch.Tensor
    scene: torch.Tensor


@dataclass(frozen=True)
class NormalizedFactorizedRouteAuditV5:
    scene_token_count: int
    environment_latent_count: int
    control_token_count: int
    scene_moment_count: int
    output_basis_rank: int
    route_factor_rank: int
    every_environment_latent_influenced_signature: bool
    control_values_scene_question_bilinear: bool
    gate_scene_question_conditioned: bool
    separate_question_scene_route_projections: bool
    all_scene_moments_consumed_by_route: bool
    normalized_route_factors: bool
    low_rank_bilinear_route: bool
    route_uses_inherited_value_trunk: bool
    inherited_v60_state_frozen: bool
    question_dependent_scene_retrieval: bool
    softmax_scene_attention_used: bool
    gate_probability: float
    control_used: bool
    maximum_control_rms: float


class NormalizedFactorizedRouteHeadV5(nn.Module):
    """Two normalized linear towers and a calibrated low-rank compatibility."""

    def __init__(self, hidden_size: int, moment_count: int, factor_rank: int) -> None:
        super().__init__()
        dimensions = {
            "hidden_size": hidden_size,
            "moment_count": moment_count,
            "factor_rank": factor_rank,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in dimensions.values()
        ):
            raise ValueError("V5 route dimensions must be positive integers")
        self.hidden_size = int(hidden_size)
        self.moment_count = int(moment_count)
        self.factor_rank = int(factor_rank)

        # Ws consumes the complete ordered DCT signature.  Flattening is
        # intentional: no moment is selected or pooled away before projection.
        self.question_projection = nn.Linear(self.hidden_size, self.factor_rank, bias=False)
        self.scene_projection = nn.Linear(
            self.moment_count * self.hidden_size,
            self.factor_rank,
            bias=False,
        )
        self.bilinear_diagonal = nn.Parameter(torch.ones(self.factor_rank))
        self.question_calibration = nn.Linear(self.factor_rank, 1, bias=False)
        self.scene_calibration = nn.Linear(self.factor_rank, 1, bias=False)
        self.log_bilinear_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.route_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.question_calibration.weight)
        nn.init.zeros_(self.scene_calibration.weight)

    def encode_question(self, normalized_question: torch.Tensor) -> torch.Tensor:
        if (
            normalized_question.ndim != 2
            or normalized_question.shape[-1] != self.hidden_size
            or not torch.isfinite(normalized_question).all()
        ):
            raise ValueError("V5 normalized question must have shape [B,H]")
        projected = self.question_projection(normalized_question.float())
        if torch.any(projected.square().sum(dim=-1) <= 1e-12):
            raise RuntimeError("V5 question projection produced a zero-norm factor")
        return F.normalize(projected, dim=-1, eps=1e-8)

    def encode_scene(self, scene_signature: torch.Tensor) -> torch.Tensor:
        if (
            scene_signature.ndim != 3
            or scene_signature.shape[1:] != (self.moment_count, self.hidden_size)
            or not torch.isfinite(scene_signature).all()
        ):
            raise ValueError("V5 scene signature must have shape [B,M,H]")
        projected = self.scene_projection(scene_signature.float().flatten(1))
        if torch.any(projected.square().sum(dim=-1) <= 1e-12):
            raise RuntimeError("V5 scene projection produced a zero-norm factor")
        return F.normalize(projected, dim=-1, eps=1e-8)

    def logits(self, question_factor: torch.Tensor, scene_factor: torch.Tensor) -> torch.Tensor:
        expected = self.factor_rank
        if (
            question_factor.ndim != 2
            or scene_factor.ndim != 2
            or question_factor.shape != scene_factor.shape
            or question_factor.shape[-1] != expected
            or not torch.isfinite(question_factor).all()
            or not torch.isfinite(scene_factor).all()
        ):
            raise ValueError("V5 route factors must share finite shape [B,R]")
        # Re-normalization makes the public cached-feature API robust to storage
        # roundoff and guarantees that scale is calibration, not vector norm.
        question = F.normalize(question_factor.float(), dim=-1, eps=1e-8)
        scene = F.normalize(scene_factor.float(), dim=-1, eps=1e-8)
        diagonal = torch.tanh(self.bilinear_diagonal)
        compatibility = torch.sum(question * diagonal * scene, dim=-1)
        scale = self.log_bilinear_scale.exp().clamp(max=100.0)
        question_term = self.question_calibration(question).squeeze(-1)
        scene_term = self.scene_calibration(scene).squeeze(-1)
        result = scale * compatibility + question_term + scene_term + self.route_bias
        if not torch.isfinite(result).all():
            raise RuntimeError("V5 route head produced NaN or infinity")
        return result


class NormalizedFactorizedSceneQuestionControlV5(TeacherBasisFullSceneQuestionControlV3):
    """Frozen V60 continuous values with a separately factorized route head."""

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
        route_factor_rank: int = 32,
    ) -> None:
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
        self.factorized_route = NormalizedFactorizedRouteHeadV5(
            self.hidden_size,
            self.moment_count,
            route_factor_rank,
        )
        self.route_factor_rank = self.factorized_route.factor_rank
        self._last_v5_audit: NormalizedFactorizedRouteAuditV5 | None = None

    @property
    def inherited_state_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.state_dict() if not name.startswith("factorized_route."))

    @property
    def gate_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.factorized_route.parameters())

    def freeze_inherited_v60_state(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("factorized_route."))

    @property
    def inherited_v60_state_frozen(self) -> bool:
        return all(
            not parameter.requires_grad
            for name, parameter in self.named_parameters()
            if not name.startswith("factorized_route.")
        )

    @classmethod
    def from_v60(
        cls,
        source: TeacherBasisFullSceneQuestionControlV3,
        *,
        route_factor_rank: int = 32,
    ) -> NormalizedFactorizedSceneQuestionControlV5:
        if type(source) is not TeacherBasisFullSceneQuestionControlV3:
            raise TypeError("V5 source must be an exact V3/V60 controller")
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
            route_factor_rank=route_factor_rank,
        )
        source_state = source.state_dict()
        missing, unexpected = module.load_state_dict(source_state, strict=False)
        expected_missing = {
            name for name in module.state_dict() if name.startswith("factorized_route.")
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"V5 inherited state contract changed: missing={missing} unexpected={unexpected}"
            )
        if set(source_state) != set(module.inherited_state_names) or any(
            not torch.equal(source_state[name].cpu(), module.state_dict()[name].cpu())
            for name in source_state
        ):
            raise RuntimeError("V5 failed to copy V60 state exactly")
        module.freeze_inherited_v60_state()
        return module

    def encode_route_question(
        self,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.normalized_question(question_embeddings, question_attention_mask)
        return self.factorized_route.encode_question(normalized)

    def encode_route_scene(self, scene_signature: torch.Tensor) -> torch.Tensor:
        return self.factorized_route.encode_scene(scene_signature)

    def route_features_from_signature(
        self,
        scene_signature: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> FactorizedRouteFeaturesV5:
        question = self.encode_route_question(question_embeddings, question_attention_mask)
        scene = self.encode_route_scene(scene_signature)
        if question.shape[0] != scene.shape[0]:
            raise ValueError("V5 scene and question route batches must match")
        return FactorizedRouteFeaturesV5(question=question, scene=scene)

    def route_logits_from_features(self, features: FactorizedRouteFeaturesV5) -> torch.Tensor:
        if not isinstance(features, FactorizedRouteFeaturesV5):
            raise TypeError("V5 route features have the wrong type")
        return self.factorized_route.logits(features.question, features.scene)

    def route_logits_from_signature(
        self,
        scene_signature: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.route_logits_from_features(
            self.route_features_from_signature(
                scene_signature,
                question_embeddings,
                question_attention_mask,
            )
        )

    def forward_from_signature(
        self,
        scene_signature: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> TeacherBasisControlOutput:
        # V60 remains the one and only continuous-value implementation.
        inherited = super().forward_from_signature(
            scene_signature,
            question_embeddings,
            question_attention_mask,
        )
        gate_logits = self.route_logits_from_signature(
            scene_signature,
            question_embeddings,
            question_attention_mask,
        )
        gate_probabilities = torch.sigmoid(gate_logits)
        if scene_signature.shape[0] == 1:
            probability = float(gate_probabilities[0].detach().cpu())
            self._last_v5_audit = NormalizedFactorizedRouteAuditV5(
                scene_token_count=self.expected_environment_latents + 2,
                environment_latent_count=self.expected_environment_latents,
                control_token_count=self.control_token_count,
                scene_moment_count=self.moment_count,
                output_basis_rank=self.output_basis_rank,
                route_factor_rank=self.route_factor_rank,
                every_environment_latent_influenced_signature=True,
                control_values_scene_question_bilinear=True,
                gate_scene_question_conditioned=True,
                separate_question_scene_route_projections=True,
                all_scene_moments_consumed_by_route=True,
                normalized_route_factors=True,
                low_rank_bilinear_route=True,
                route_uses_inherited_value_trunk=False,
                inherited_v60_state_frozen=self.inherited_v60_state_frozen,
                question_dependent_scene_retrieval=False,
                softmax_scene_attention_used=False,
                gate_probability=probability,
                control_used=probability >= self.gate_threshold,
                maximum_control_rms=float(inherited.control_rms.max().detach().cpu()),
            )
        else:
            self._last_v5_audit = None
        return TeacherBasisControlOutput(
            control_tokens=inherited.control_tokens,
            coefficient_directions=inherited.coefficient_directions,
            control_rms=inherited.control_rms,
            gate_logits=gate_logits,
            gate_probabilities=gate_probabilities,
        )

    def audit(self) -> NormalizedFactorizedRouteAuditV5:
        if self._last_v5_audit is None:
            raise RuntimeError("V5 audit requires a completed batch-one forward pass")
        return self._last_v5_audit


__all__ = [
    "FactorizedRouteFeaturesV5",
    "NormalizedFactorizedRouteAuditV5",
    "NormalizedFactorizedRouteHeadV5",
    "NormalizedFactorizedSceneQuestionControlV5",
]

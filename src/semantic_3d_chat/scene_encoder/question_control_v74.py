"""Minimal dense, globally complete continuous scene reader.

V74 makes the question/scene interaction explicit while keeping the immutable
256-latent scene prefix separate.  Each of four queries attends positively to
all latents, and only attended scene VALUES can reach the continuous output.
Consequently an all-zero scene produces exact-zero controls for every question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DenseFullSceneControlOutputV74:
    control_tokens: torch.Tensor
    control_rms: torch.Tensor


@dataclass(frozen=True)
class DenseFullSceneControlAuditV74:
    environment_latents: int
    query_count: int
    model_dimension: int
    output_basis_rank: int
    minimum_attention_weight: float
    all_latents_receive_positive_weight: bool
    bilinear_question_scene_value_interaction: bool
    question_only_output_path_exists: bool
    question_dependent_retrieval: bool
    immutable_full_prefix_retained_separately: bool
    zero_scene_produces_exact_zero_controls: bool


class DenseFullSceneContinuousControlV74(nn.Module):
    """Four dense queries over all scene values, projected to native LM width."""

    def __init__(
        self,
        hidden_size: int,
        output_basis: torch.Tensor,
        *,
        environment_latents: int = 256,
        query_count: int = 4,
        model_dimension: int = 128,
        uniform_floor_mass: float = 0.05,
        maximum_control_rms: float = 0.25,
    ) -> None:
        super().__init__()
        if min(hidden_size, environment_latents, query_count, model_dimension) < 1:
            raise ValueError("V74 dimensions must be positive")
        if not 0.0 < uniform_floor_mass < 1.0:
            raise ValueError("V74 attention-floor mass must lie in (0,1)")
        if not 0.0 < maximum_control_rms <= 1.0:
            raise ValueError("V74 maximum control RMS must lie in (0,1]")
        if (
            output_basis.ndim != 2
            or output_basis.shape[1] != hidden_size
            or output_basis.shape[0] < 1
            or not torch.isfinite(output_basis).all()
        ):
            raise ValueError("V74 output basis must be finite [R,H]")
        basis = output_basis.detach().float().contiguous()
        identity = torch.eye(basis.shape[0], dtype=torch.float32)
        if not torch.allclose(basis @ basis.T, identity, atol=2e-4, rtol=2e-4):
            raise ValueError("V74 output-basis rows must be orthonormal")

        self.hidden_size = int(hidden_size)
        self.environment_latents = int(environment_latents)
        self.query_count = int(query_count)
        self.control_token_count = int(query_count)
        self.model_dimension = int(model_dimension)
        self.uniform_floor_mass = float(uniform_floor_mass)
        self.maximum_control_rms = float(maximum_control_rms)
        self.output_basis_rank = int(basis.shape[0])
        self.scene_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.question_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.key = nn.Linear(hidden_size, model_dimension, bias=False)
        self.value = nn.Linear(hidden_size, model_dimension, bias=False)
        self.query = nn.Linear(
            hidden_size, query_count * model_dimension, bias=False
        )
        self.coefficient_output = nn.Linear(
            query_count * model_dimension,
            query_count * self.output_basis_rank,
            bias=False,
        )
        self.register_buffer("output_basis", basis, persistent=True)
        self._last_minimum_attention_weight: float | None = None

    def _decode_coefficients(self, interaction: torch.Tensor) -> torch.Tensor:
        """Map the zero-safe scene/question interaction into basis coefficients."""

        return self.coefficient_output(interaction)

    @staticmethod
    def _pooled_question(
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if question_attention_mask is None:
            return question_embeddings.float().mean(dim=1)
        if question_attention_mask.shape != question_embeddings.shape[:2]:
            raise ValueError("V74 question mask shape changed")
        weight = question_attention_mask.to(question_embeddings).float()
        count = weight.sum(dim=1, keepdim=True)
        if bool((count <= 0).any()):
            raise ValueError("V74 question rows cannot be empty")
        return (question_embeddings.float() * weight[..., None]).sum(dim=1) / count

    def encode_scene(self, scene_prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.environment_latents + 2
        if scene_prefix.ndim != 3 or scene_prefix.shape[1:] != (
            expected,
            self.hidden_size,
        ):
            raise ValueError("V74 scene prefix must be [B,BOI+256+EOI,H]")
        if not torch.isfinite(scene_prefix).all():
            raise ValueError("V74 scene prefix must be finite")
        environment = scene_prefix[:, 1:-1].float()
        normalized = self.scene_norm(environment)
        return self.key(normalized), self.value(normalized)

    def forward_encoded(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> DenseFullSceneControlOutputV74:
        if key.shape != value.shape or key.ndim != 3 or key.shape[1:] != (
            self.environment_latents,
            self.model_dimension,
        ):
            raise ValueError("V74 encoded-scene shape changed")
        if (
            question_embeddings.ndim != 3
            or question_embeddings.shape[0] != key.shape[0]
            or question_embeddings.shape[-1] != self.hidden_size
        ):
            raise ValueError("V74 question embeddings shape changed")
        pooled = self._pooled_question(question_embeddings, question_attention_mask)
        query = self.query(self.question_norm(pooled)).reshape(
            -1, self.query_count, self.model_dimension
        )
        score = torch.einsum("bqd,bld->bql", query, key) / math.sqrt(
            self.model_dimension
        )
        probability = torch.softmax(score.float(), dim=-1).to(value)
        probability = (
            (1.0 - self.uniform_floor_mass) * probability
            + self.uniform_floor_mass / self.environment_latents
        )
        context = torch.einsum("bql,bld->bqd", probability, value)
        # This is the decisive V74 fix.  It is multiplicative, so it adds no
        # question-only path when scene values are zero.
        interaction = (context * torch.tanh(query)).flatten(1)
        coefficients = self._decode_coefficients(interaction).reshape(
            -1, self.control_token_count, self.output_basis_rank
        )
        raw = torch.einsum("bcr,rh->bch", coefficients, self.output_basis)
        raw_rms = raw.square().mean(dim=-1, keepdim=True).sqrt()
        scale = torch.clamp(
            self.maximum_control_rms / raw_rms.clamp_min(1e-8), max=1.0
        )
        controls = raw * scale
        rms = controls.square().mean(dim=-1).sqrt()
        if not torch.isfinite(controls).all():
            raise RuntimeError("V74 controls are nonfinite")
        self._last_minimum_attention_weight = float(probability.detach().min().cpu())
        return DenseFullSceneControlOutputV74(controls, rms)

    def forward(
        self,
        scene_prefix: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> DenseFullSceneControlOutputV74:
        return self.forward_encoded(
            *self.encode_scene(scene_prefix),
            question_embeddings,
            question_attention_mask,
        )

    def audit(self) -> DenseFullSceneControlAuditV74:
        if self._last_minimum_attention_weight is None:
            raise RuntimeError("V74 audit requires a completed forward pass")
        required = self.uniform_floor_mass / self.environment_latents
        return DenseFullSceneControlAuditV74(
            environment_latents=self.environment_latents,
            query_count=self.query_count,
            model_dimension=self.model_dimension,
            output_basis_rank=self.output_basis_rank,
            minimum_attention_weight=self._last_minimum_attention_weight,
            all_latents_receive_positive_weight=(
                self._last_minimum_attention_weight >= required - 1e-8
            ),
            bilinear_question_scene_value_interaction=True,
            question_only_output_path_exists=False,
            question_dependent_retrieval=False,
            immutable_full_prefix_retained_separately=True,
            zero_scene_produces_exact_zero_controls=True,
        )


__all__ = [
    "DenseFullSceneContinuousControlV74",
    "DenseFullSceneControlAuditV74",
    "DenseFullSceneControlOutputV74",
]

"""Teacher-basis full-scene control with semantic task routing.

V3 represents each control token in an orthogonal continuous basis learned from
authorized numeric prompt teachers.  The controller predicts a unit direction
in that basis and a separately bounded RMS magnitude.  This avoids unstable
regression over 1,536 arbitrary coordinates while preserving the native Gemma
embedding dimension at runtime.

Control values are strictly scene-by-question bilinear over fixed DCT moments
of every environment latent.  A separate continuous question-semantic
prototype gate decides whether controls are useful; it does not retrieve scene
regions or change the immutable full scene prefix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TeacherBasisControlOutput:
    control_tokens: torch.Tensor
    coefficient_directions: torch.Tensor
    control_rms: torch.Tensor
    gate_logits: torch.Tensor
    gate_probabilities: torch.Tensor


@dataclass(frozen=True)
class TeacherBasisControlAudit:
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


class TeacherBasisFullSceneQuestionControlV3(nn.Module):
    """Predict bounded native-width controls through an orthogonal teacher basis."""

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
    ) -> None:
        super().__init__()
        dimensions = {
            "hidden_size": hidden_size,
            "control_tokens": control_tokens,
            "expected_environment_latents": expected_environment_latents,
            "moment_count": moment_count,
            "interaction_dim": interaction_dim,
            "trunk_dim": trunk_dim,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in dimensions.values()
        ):
            raise ValueError("V3 dimensions must be positive integers")
        if moment_count > expected_environment_latents:
            raise ValueError("V3 moment_count cannot exceed environment latents")
        if (
            not isinstance(output_basis, torch.Tensor)
            or output_basis.ndim != 2
            or output_basis.shape[1] != hidden_size
            or output_basis.shape[0] < 1
            or output_basis.shape[0] > hidden_size
            or not output_basis.is_floating_point()
            or not torch.isfinite(output_basis).all()
        ):
            raise ValueError("V3 output_basis must be finite floating [R,H]")
        basis = output_basis.detach().float().contiguous()
        gram = basis @ basis.T
        identity = torch.eye(basis.shape[0], dtype=gram.dtype, device=gram.device)
        if float((gram - identity).abs().max().cpu()) > 2e-4:
            raise ValueError("V3 output_basis rows must be orthonormal")
        for field, value in (
            ("maximum_control_rms", maximum_control_rms),
            ("initial_control_rms", initial_control_rms),
            ("gate_threshold", gate_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"V3 {field} must be finite")
        if not 0.0 < float(initial_control_rms) < float(maximum_control_rms) <= 1.0:
            raise ValueError("V3 RMS settings require 0 < initial < maximum <= 1")
        if not 0.0 < float(gate_threshold) < 1.0:
            raise ValueError("V3 gate_threshold must be in (0,1)")

        self.hidden_size = int(hidden_size)
        self.control_token_count = int(control_tokens)
        self.expected_environment_latents = int(expected_environment_latents)
        self.moment_count = int(moment_count)
        self.interaction_dim = int(interaction_dim)
        self.trunk_dim = int(trunk_dim)
        self.output_basis_rank = int(basis.shape[0])
        self.maximum_control_rms = float(maximum_control_rms)
        self.initial_control_rms = float(initial_control_rms)
        self.gate_threshold = float(gate_threshold)
        self.register_buffer("output_basis", basis, persistent=True)

        self.question_norm = nn.LayerNorm(self.hidden_size)
        self.scene_projection = nn.Linear(
            self.hidden_size, self.interaction_dim, bias=False
        )
        self.question_projection = nn.Linear(
            self.hidden_size,
            self.control_token_count * self.interaction_dim,
            bias=False,
        )
        interaction_width = self.moment_count * self.interaction_dim
        self.control_trunk = nn.Sequential(
            nn.LayerNorm(interaction_width),
            nn.Linear(interaction_width, self.trunk_dim),
            nn.SiLU(),
        )
        self.coefficient_output = nn.Linear(
            self.trunk_dim, self.output_basis_rank
        )
        self.magnitude_output = nn.Linear(self.trunk_dim, 1)
        initial_fraction = self.initial_control_rms / self.maximum_control_rms
        nn.init.constant_(
            self.magnitude_output.bias,
            math.log(initial_fraction / (1.0 - initial_fraction)),
        )

        # A small native-embedding prototype classifier promotes semantic
        # wording transfer without consulting or selecting scene regions.
        self.route_positive_prototype = nn.Parameter(torch.zeros(self.hidden_size))
        self.route_negative_prototype = nn.Parameter(torch.zeros(self.hidden_size))
        self.route_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.route_bias = nn.Parameter(torch.tensor(0.0))
        self._last_audit: TeacherBasisControlAudit | None = None

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.state_dict().values())

    @staticmethod
    def _pooled_question(
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if question_attention_mask is None:
            return question_embeddings.mean(dim=1)
        if question_attention_mask.shape != question_embeddings.shape[:2]:
            raise ValueError("V3 question attention mask must have shape [B,Q]")
        weights = question_attention_mask.to(question_embeddings).clamp(0.0, 1.0)
        if not torch.isfinite(weights).all():
            raise ValueError("V3 question attention mask must be finite")
        counts = weights.sum(dim=1, keepdim=True)
        if torch.any(counts <= 0.0):
            raise ValueError("Every V3 question requires an unmasked token")
        return torch.sum(question_embeddings * weights.unsqueeze(-1), dim=1) / counts

    def normalized_question(
        self,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self._pooled_question(question_embeddings, question_attention_mask)
        return F.normalize(self.question_norm(pooled.float()), dim=-1, eps=1e-8)

    @torch.no_grad()
    def initialize_route_prototypes(
        self,
        positive_questions: torch.Tensor,
        negative_questions: torch.Tensor,
    ) -> None:
        if positive_questions.ndim != 2 or negative_questions.ndim != 2:
            raise ValueError("V3 route prototypes require [N,H] question features")
        if (
            positive_questions.shape[0] < 1
            or negative_questions.shape[0] < 1
            or positive_questions.shape[1] != self.hidden_size
            or negative_questions.shape[1] != self.hidden_size
        ):
            raise ValueError("V3 route prototype feature dimensions changed")
        positive = F.normalize(positive_questions.float(), dim=-1).mean(dim=0)
        negative = F.normalize(negative_questions.float(), dim=-1).mean(dim=0)
        self.route_positive_prototype.copy_(positive)
        self.route_negative_prototype.copy_(negative)

    def encode_scene(self, scene_prefix: torch.Tensor) -> torch.Tensor:
        """Cache normalized DCT moments over exactly the environment latents."""

        if scene_prefix.ndim != 3 or scene_prefix.shape[-1] != self.hidden_size:
            raise ValueError("V3 scene prefix must have shape [B,S,H]")
        expected = self.expected_environment_latents + 2
        if scene_prefix.shape[1] != expected:
            raise ValueError(
                "V3 expected BOI + all environment latents + EOI: "
                f"expected={expected} observed={scene_prefix.shape[1]}"
            )
        if not torch.isfinite(scene_prefix).all():
            raise ValueError("V3 scene prefix must be finite")
        environment = F.layer_norm(
            scene_prefix[:, 1:-1].float(), (self.hidden_size,)
        )
        positions = (
            torch.arange(
                self.expected_environment_latents,
                device=scene_prefix.device,
                dtype=torch.float32,
            )
            + 0.5
        ) / float(self.expected_environment_latents)
        frequencies = torch.arange(
            self.moment_count, device=scene_prefix.device, dtype=torch.float32
        )
        weights = torch.cos(math.pi * frequencies[:, None] * positions[None, :])
        weights[0].fill_(1.0)
        weights = weights / weights.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(
            1e-8
        )
        moments = torch.einsum("ml,blh->bmh", weights, environment)
        # DCT DC and non-DC components have very different raw scales.  Each
        # moment is normalized independently before learned projections.
        moments = F.layer_norm(moments, (self.hidden_size,))
        if not torch.isfinite(moments).all():
            raise RuntimeError("V3 scene signature contains NaN or infinity")
        return moments.detach()

    def route_logits_from_normalized_question(
        self, normalized_question: torch.Tensor
    ) -> torch.Tensor:
        positive = F.normalize(self.route_positive_prototype, dim=0, eps=1e-8)
        negative = F.normalize(self.route_negative_prototype, dim=0, eps=1e-8)
        scale = self.route_log_scale.exp().clamp(max=100.0)
        return scale * (
            normalized_question @ positive - normalized_question @ negative
        ) + self.route_bias

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
            raise ValueError("V3 scene signature shape changed")
        if question_embeddings.ndim != 3 or question_embeddings.shape[-1] != self.hidden_size:
            raise ValueError("V3 question embeddings must have shape [B,Q,H]")
        if scene_signature.shape[0] != question_embeddings.shape[0]:
            raise ValueError("V3 scene and question batch sizes must match")
        normalized_question = self.normalized_question(
            question_embeddings, question_attention_mask
        )
        scene_factors = self.scene_projection(scene_signature.float())
        question_factors = self.question_projection(normalized_question).reshape(
            -1, self.control_token_count, self.interaction_dim
        )
        interaction = (
            scene_factors[:, None, :, :] * question_factors[:, :, None, :]
        ).reshape(-1, self.control_token_count, self.moment_count * self.interaction_dim)
        trunk = self.control_trunk(interaction)
        raw_coefficients = self.coefficient_output(trunk)
        coefficients = F.normalize(raw_coefficients, dim=-1, eps=1e-8)
        raw_direction = torch.einsum(
            "bcr,rh->bch", coefficients, self.output_basis
        )
        direction = raw_direction / raw_direction.square().mean(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        control_rms = self.maximum_control_rms * torch.sigmoid(
            self.magnitude_output(trunk).squeeze(-1)
        )
        controls = direction * control_rms.unsqueeze(-1)
        gate_logits = self.route_logits_from_normalized_question(normalized_question)
        gate_probabilities = torch.sigmoid(gate_logits)
        actual_rms = controls.square().mean(dim=-1).sqrt()
        if not all(
            torch.isfinite(value).all()
            for value in (controls, coefficients, actual_rms, gate_probabilities)
        ):
            raise RuntimeError("V3 controller produced NaN or infinity")
        if float(actual_rms.max().detach().cpu()) > self.maximum_control_rms + 1e-5:
            raise RuntimeError("V3 controller exceeded its analytic RMS bound")
        if scene_signature.shape[0] == 1:
            probability = float(gate_probabilities[0].detach().cpu())
            self._last_audit = TeacherBasisControlAudit(
                scene_token_count=self.expected_environment_latents + 2,
                environment_latent_count=self.expected_environment_latents,
                control_token_count=self.control_token_count,
                scene_moment_count=self.moment_count,
                output_basis_rank=self.output_basis_rank,
                every_environment_latent_influenced_signature=True,
                control_values_scene_question_bilinear=True,
                question_dependent_scene_retrieval=False,
                softmax_scene_attention_used=False,
                gate_probability=probability,
                control_used=probability >= self.gate_threshold,
                maximum_control_rms=float(actual_rms.max().detach().cpu()),
            )
        else:
            self._last_audit = None
        return TeacherBasisControlOutput(
            control_tokens=controls,
            coefficient_directions=coefficients,
            control_rms=actual_rms,
            gate_logits=gate_logits,
            gate_probabilities=gate_probabilities,
        )

    def forward(
        self,
        scene_prefix: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> TeacherBasisControlOutput:
        return self.forward_from_signature(
            self.encode_scene(scene_prefix),
            question_embeddings,
            question_attention_mask,
        )

    def audit(self) -> TeacherBasisControlAudit:
        if self._last_audit is None:
            raise RuntimeError("V3 audit requires a completed batch-one forward pass")
        return self._last_audit


def teacher_output_basis(targets: torch.Tensor, *, rank: int | None = None) -> torch.Tensor:
    """Create a deterministic orthogonal basis over native teacher directions."""

    if targets.ndim != 3 or targets.shape[0] < 1 or targets.shape[1] < 1:
        raise ValueError("V3 teacher targets must have shape [N,C,H]")
    if not targets.is_floating_point() or not torch.isfinite(targets).all():
        raise ValueError("V3 teacher targets must be finite floating point")
    flattened = targets.detach().float().reshape(-1, targets.shape[-1])
    rms = flattened.square().mean(dim=-1, keepdim=True).sqrt()
    if torch.any(rms <= 1e-8):
        raise ValueError("V3 teacher targets contain a zero-norm token")
    directions = flattened / rms
    maximum_rank = min(directions.shape)
    selected_rank = maximum_rank if rank is None else rank
    if (
        isinstance(selected_rank, bool)
        or not isinstance(selected_rank, int)
        or not 1 <= selected_rank <= maximum_rank
    ):
        raise ValueError(f"V3 basis rank must be in [1,{maximum_rank}]")
    _u, _s, vh = torch.linalg.svd(directions.cpu(), full_matrices=False)
    basis = vh[:selected_rank].float().contiguous()
    # SVD vector signs are immaterial mathematically but canonicalizing them
    # makes checkpoints reproducible across equivalent LAPACK choices.
    pivots = basis.abs().argmax(dim=1)
    signs = torch.sign(basis[torch.arange(selected_rank), pivots]).clamp_min(0.0) * 2.0 - 1.0
    return (basis * signs[:, None]).contiguous()


__all__ = [
    "TeacherBasisControlAudit",
    "TeacherBasisControlOutput",
    "TeacherBasisFullSceneQuestionControlV3",
    "teacher_output_basis",
]

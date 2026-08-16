"""Independent 8- and 32-moment full-scene continuous controller.

V71 keeps two complete value branches.  Each branch receives fixed DCT
moments computed over all 256 environment latents, has its own question and
scene projections, trunk, coefficient head, and magnitude head, and emits
native-width continuous control tokens.  A single global scalar, bounded
strictly away from either endpoint, fuses the two outputs.  The scalar is a
model parameter: in pair-disjoint experiments it is optimized only from the
training folds and never selected or tuned on held rows.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisControlOutput,
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)


@dataclass(frozen=True)
class MultiscaleControlAuditV71:
    scene_token_count: int
    environment_latent_count: int
    control_token_count: int
    branch_moment_counts: tuple[int, int]
    signature_moment_count: int
    output_basis_rank: int
    every_environment_latent_influences_both_branches: bool
    independent_scene_projections: bool
    independent_question_projections: bool
    independent_trunks_and_heads: bool
    fusion_weight_branch_8: float
    fusion_weight_branch_32: float
    fusion_bounded_away_from_endpoints: bool
    question_dependent_scene_retrieval: bool
    softmax_scene_attention_used: bool
    control_used: bool
    maximum_control_rms: float


class _DualBranchCoefficientOutput(nn.Module):
    """Keep both coefficient heads and the fusion scalar in one trainable scope."""

    def __init__(self, branch_8: nn.Module, branch_32: nn.Module) -> None:
        super().__init__()
        self.branch_8 = copy.deepcopy(branch_8)
        self.branch_32 = copy.deepcopy(branch_32)
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))


class MultiscaleAlwaysOnTeacherBasisControlV71(nn.Module):
    """Fuse independent first-8 and first-32 DCT full-scene value branches."""

    branch_moment_counts = (8, 32)
    moment_count = 40
    fusion_floor = 0.10
    fusion_span = 0.80

    def __init__(
        self,
        branch_8: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
        branch_32: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
    ) -> None:
        super().__init__()
        if (
            type(branch_8) is not AlwaysOnTeacherBasisFullSceneQuestionControlV7
            or type(branch_32) is not AlwaysOnTeacherBasisFullSceneQuestionControlV7
        ):
            raise TypeError("V71 branches must be exact fitted V7 controllers")
        if branch_8.moment_count != 8 or branch_32.moment_count != 32:
            raise ValueError("V71 requires independent 8- and 32-moment branches")
        scalar_attributes = (
            "hidden_size",
            "control_token_count",
            "expected_environment_latents",
            "interaction_dim",
            "trunk_dim",
            "output_basis_rank",
            "maximum_control_rms",
            "initial_control_rms",
        )
        if any(
            getattr(branch_8, field) != getattr(branch_32, field)
            for field in scalar_attributes
        ):
            raise ValueError("V71 branch dimensions or RMS bounds differ")
        if not torch.equal(branch_8.output_basis, branch_32.output_basis):
            raise ValueError("V71 branches must share an exact output basis")

        self.hidden_size = int(branch_8.hidden_size)
        self.control_token_count = int(branch_8.control_token_count)
        self.expected_environment_latents = int(
            branch_8.expected_environment_latents
        )
        self.interaction_dim = int(branch_8.interaction_dim)
        self.trunk_dim = int(branch_8.trunk_dim)
        self.output_basis_rank = int(branch_8.output_basis_rank)
        self.maximum_control_rms = float(branch_8.maximum_control_rms)
        self.initial_control_rms = float(branch_8.initial_control_rms)
        self.register_buffer(
            "output_basis", branch_8.output_basis.detach().clone(), persistent=True
        )

        # These names deliberately preserve V68/V69's audited value-parameter
        # scope while every contained module remains branch-independent.
        self.question_norm = nn.ModuleDict(
            {
                "branch_8": copy.deepcopy(branch_8.question_norm),
                "branch_32": copy.deepcopy(branch_32.question_norm),
            }
        )
        self.scene_projection = nn.ModuleDict(
            {
                "branch_8": copy.deepcopy(branch_8.scene_projection),
                "branch_32": copy.deepcopy(branch_32.scene_projection),
            }
        )
        self.question_projection = nn.ModuleDict(
            {
                "branch_8": copy.deepcopy(branch_8.question_projection),
                "branch_32": copy.deepcopy(branch_32.question_projection),
            }
        )
        self.control_trunk = nn.ModuleDict(
            {
                "branch_8": copy.deepcopy(branch_8.control_trunk),
                "branch_32": copy.deepcopy(branch_32.control_trunk),
            }
        )
        self.coefficient_output = _DualBranchCoefficientOutput(
            branch_8.coefficient_output,
            branch_32.coefficient_output,
        )
        self.magnitude_output = nn.ModuleDict(
            {
                "branch_8": copy.deepcopy(branch_8.magnitude_output),
                "branch_32": copy.deepcopy(branch_32.magnitude_output),
            }
        )
        self.question_norm.requires_grad_(False)
        self._last_audit: MultiscaleControlAuditV71 | None = None

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.state_dict().values())

    @staticmethod
    def _dct_moments(environment: torch.Tensor, count: int) -> torch.Tensor:
        latent_count = environment.shape[1]
        positions = (
            torch.arange(latent_count, device=environment.device, dtype=torch.float32)
            + 0.5
        ) / float(latent_count)
        frequencies = torch.arange(
            count, device=environment.device, dtype=torch.float32
        )
        weights = torch.cos(math.pi * frequencies[:, None] * positions[None, :])
        weights[0].fill_(1.0)
        weights = weights / weights.square().sum(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-8)
        moments = torch.einsum("ml,blh->bmh", weights, environment)
        return F.layer_norm(moments, (environment.shape[-1],))

    def encode_scene(self, scene_prefix: torch.Tensor) -> torch.Tensor:
        if scene_prefix.ndim != 3 or scene_prefix.shape[-1] != self.hidden_size:
            raise ValueError("V71 scene prefix must have shape [B,S,H]")
        expected = self.expected_environment_latents + 2
        if scene_prefix.shape[1] != expected:
            raise ValueError(
                "V71 expected BOI + every environment latent + EOI: "
                f"expected={expected} observed={scene_prefix.shape[1]}"
            )
        if not torch.isfinite(scene_prefix).all():
            raise ValueError("V71 scene prefix must be finite")
        environment = F.layer_norm(
            scene_prefix[:, 1:-1].float(), (self.hidden_size,)
        )
        signature = torch.cat(
            [
                self._dct_moments(environment, self.branch_moment_counts[0]),
                self._dct_moments(environment, self.branch_moment_counts[1]),
            ],
            dim=1,
        )
        if not torch.isfinite(signature).all():
            raise RuntimeError("V71 scene signature contains NaN or infinity")
        return signature.detach()

    def _branch_forward(
        self,
        branch: str,
        signature: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        pooled = TeacherBasisFullSceneQuestionControlV3._pooled_question(
            question_embeddings, question_attention_mask
        )
        normalized_question = F.normalize(
            self.question_norm[branch](pooled.float()), dim=-1, eps=1e-8
        )
        scene_factors = self.scene_projection[branch](signature.float())
        question_factors = self.question_projection[branch](
            normalized_question
        ).reshape(-1, self.control_token_count, self.interaction_dim)
        interaction = (
            scene_factors[:, None, :, :] * question_factors[:, :, None, :]
        ).reshape(
            -1,
            self.control_token_count,
            signature.shape[1] * self.interaction_dim,
        )
        trunk = self.control_trunk[branch](interaction)
        coefficient_head = getattr(self.coefficient_output, branch)
        coefficients = F.normalize(coefficient_head(trunk), dim=-1, eps=1e-8)
        raw_direction = torch.einsum(
            "bcr,rh->bch", coefficients, self.output_basis
        )
        direction = raw_direction / raw_direction.square().mean(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        rms = self.maximum_control_rms * torch.sigmoid(
            self.magnitude_output[branch](trunk).squeeze(-1)
        )
        return direction * rms.unsqueeze(-1)

    def fusion_weight(self) -> torch.Tensor:
        return self.fusion_floor + self.fusion_span * torch.sigmoid(
            self.coefficient_output.fusion_logit
        )

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
            raise ValueError("V71 scene signature shape changed")
        if (
            question_embeddings.ndim != 3
            or question_embeddings.shape[-1] != self.hidden_size
            or question_embeddings.shape[0] != scene_signature.shape[0]
        ):
            raise ValueError("V71 questions must have matching shape [B,Q,H]")
        split = self.branch_moment_counts[0]
        branch_8 = self._branch_forward(
            "branch_8",
            scene_signature[:, :split],
            question_embeddings,
            question_attention_mask,
        )
        branch_32 = self._branch_forward(
            "branch_32",
            scene_signature[:, split:],
            question_embeddings,
            question_attention_mask,
        )
        weight_8 = self.fusion_weight()
        controls = weight_8 * branch_8 + (1.0 - weight_8) * branch_32
        actual_rms = controls.square().mean(dim=-1).sqrt()
        raw_coefficients = torch.einsum(
            "bch,rh->bcr", controls, self.output_basis
        )
        coefficients = F.normalize(raw_coefficients, dim=-1, eps=1e-8)
        gate_logits = torch.full(
            (scene_signature.shape[0],),
            20.0,
            device=controls.device,
            dtype=controls.dtype,
        )
        probabilities = torch.sigmoid(gate_logits)
        if not all(
            torch.isfinite(value).all()
            for value in (controls, coefficients, actual_rms, weight_8)
        ):
            raise RuntimeError("V71 controller produced NaN or infinity")
        if float(actual_rms.max().detach().cpu()) > self.maximum_control_rms + 1e-5:
            raise RuntimeError("V71 controller exceeded its analytic RMS bound")
        if scene_signature.shape[0] == 1:
            observed_weight = float(weight_8.detach().cpu())
            self._last_audit = MultiscaleControlAuditV71(
                scene_token_count=self.expected_environment_latents + 2,
                environment_latent_count=self.expected_environment_latents,
                control_token_count=self.control_token_count,
                branch_moment_counts=self.branch_moment_counts,
                signature_moment_count=self.moment_count,
                output_basis_rank=self.output_basis_rank,
                every_environment_latent_influences_both_branches=True,
                independent_scene_projections=True,
                independent_question_projections=True,
                independent_trunks_and_heads=True,
                fusion_weight_branch_8=observed_weight,
                fusion_weight_branch_32=1.0 - observed_weight,
                fusion_bounded_away_from_endpoints=(
                    self.fusion_floor <= observed_weight <= 1.0 - self.fusion_floor
                ),
                question_dependent_scene_retrieval=False,
                softmax_scene_attention_used=False,
                control_used=True,
                maximum_control_rms=float(actual_rms.max().detach().cpu()),
            )
        else:
            self._last_audit = None
        return TeacherBasisControlOutput(
            control_tokens=controls,
            coefficient_directions=coefficients,
            control_rms=actual_rms,
            gate_logits=gate_logits,
            gate_probabilities=probabilities,
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

    def audit(self) -> MultiscaleControlAuditV71:
        if self._last_audit is None:
            raise RuntimeError("V71 audit requires a completed batch-one forward pass")
        return self._last_audit


__all__ = [
    "MultiscaleAlwaysOnTeacherBasisControlV71",
    "MultiscaleControlAuditV71",
]

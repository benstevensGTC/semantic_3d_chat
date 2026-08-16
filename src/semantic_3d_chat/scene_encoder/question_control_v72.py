"""Question-adaptive fusion of two complete-scene continuous controllers.

V71 demonstrated that independent 8- and 32-moment DCT branches make
complementary held-pair errors, but its single global fusion scalar remained
numerically fixed at one half.  V72 keeps both branches unchanged and replaces
only that scalar with a bounded, low-rank continuous gate.  The gate is a
function of the frozen numeric question embedding; it never selects, retrieves,
or drops scene latents.  Both branches still receive DCT moments in which every
one of the 256 environment latents has non-zero influence.
"""

from __future__ import annotations

import copy
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
from semantic_3d_chat.scene_encoder.question_control_v71 import (
    MultiscaleAlwaysOnTeacherBasisControlV71,
)


@dataclass(frozen=True)
class AdaptiveFusionAuditV72:
    scene_token_count: int
    environment_latent_count: int
    control_token_count: int
    branch_moment_counts: tuple[int, int]
    signature_moment_count: int
    output_basis_rank: int
    every_environment_latent_influences_both_branches: bool
    both_complete_scene_branches_executed: bool
    question_dependent_scene_retrieval: bool
    latent_selection_or_top_k_used: bool
    fusion_conditioned_on_question_embedding: bool
    fusion_conditioned_on_scene_latent_identity: bool
    fusion_weight_minimum: float
    fusion_weight_maximum: float
    fusion_weight_mean: float
    fusion_weight_standard_deviation: float
    fusion_bounded_away_from_endpoints: bool
    maximum_control_rms: float


class _AdaptiveDualBranchCoefficientOutputV72(nn.Module):
    """Own both branch heads and a zero-initialized low-rank fusion gate."""

    def __init__(
        self,
        branch_8: nn.Module,
        branch_32: nn.Module,
        *,
        hidden_size: int,
        control_token_count: int,
        output_basis_rank: int,
        gate_hidden_size: int,
    ) -> None:
        super().__init__()
        self.branch_8 = copy.deepcopy(branch_8)
        self.branch_32 = copy.deepcopy(branch_32)
        self.fusion_input = nn.Linear(hidden_size, gate_hidden_size, bias=False)
        self.fusion_output = nn.Linear(
            gate_hidden_size,
            control_token_count * output_basis_rank,
            bias=True,
        )
        # V72 begins as the exact arithmetic 50/50 V71 mixture.  The first
        # calibration gradient updates only this last layer; subsequent steps
        # can learn the low-rank semantic question projection as well.
        nn.init.zeros_(self.fusion_output.weight)
        nn.init.zeros_(self.fusion_output.bias)


class AdaptiveMultiscaleTeacherBasisControlV72(
    MultiscaleAlwaysOnTeacherBasisControlV71
):
    """Mix complete 8/32-moment branches per question, token, and basis axis."""

    fusion_floor = 0.05
    fusion_span = 0.90

    def __init__(
        self,
        branch_8: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
        branch_32: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
        *,
        gate_hidden_size: int = 64,
    ) -> None:
        if (
            isinstance(gate_hidden_size, bool)
            or not isinstance(gate_hidden_size, int)
            or gate_hidden_size < 1
        ):
            raise ValueError("V72 gate_hidden_size must be a positive integer")
        super().__init__(branch_8, branch_32)
        inherited = self.coefficient_output
        self.coefficient_output = _AdaptiveDualBranchCoefficientOutputV72(
            inherited.branch_8,
            inherited.branch_32,
            hidden_size=self.hidden_size,
            control_token_count=self.control_token_count,
            output_basis_rank=self.output_basis_rank,
            gate_hidden_size=gate_hidden_size,
        )
        self.gate_hidden_size = int(gate_hidden_size)
        self._last_v72_audit: AdaptiveFusionAuditV72 | None = None

    def normalized_fusion_question(
        self,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = TeacherBasisFullSceneQuestionControlV3._pooled_question(
            question_embeddings, question_attention_mask
        )
        # Both fitted branches carry the same authenticated frozen question
        # normalization.  Reusing branch_8 does not expose scene information.
        return F.normalize(
            self.question_norm["branch_8"](pooled.float()), dim=-1, eps=1e-8
        )

    def fusion_weights(
        self,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.normalized_fusion_question(
            question_embeddings, question_attention_mask
        )
        hidden = F.silu(self.coefficient_output.fusion_input(normalized))
        logits = self.coefficient_output.fusion_output(hidden).reshape(
            -1, self.control_token_count, self.output_basis_rank
        )
        weights = self.fusion_floor + self.fusion_span * torch.sigmoid(logits)
        if not torch.isfinite(weights).all():
            raise RuntimeError("V72 fusion weights contain NaN or infinity")
        return weights

    def branch_outputs_from_signature(
        self,
        scene_signature: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scene_signature.ndim != 3 or scene_signature.shape[1:] != (
            self.moment_count,
            self.hidden_size,
        ):
            raise ValueError("V72 scene signature shape changed")
        if (
            question_embeddings.ndim != 3
            or question_embeddings.shape[-1] != self.hidden_size
            or question_embeddings.shape[0] != scene_signature.shape[0]
        ):
            raise ValueError("V72 questions must have matching shape [B,Q,H]")
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
        return branch_8, branch_32

    def fuse_branch_outputs(
        self,
        branch_8: torch.Tensor,
        branch_32: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse already-computed branches without inspecting scene latents."""

        expected = (question_embeddings.shape[0], self.control_token_count, self.hidden_size)
        if branch_8.shape != expected or branch_32.shape != expected:
            raise ValueError("V72 branch output shape changed")
        weights = self.fusion_weights(
            question_embeddings, question_attention_mask
        ).to(branch_8)
        coefficients_8 = torch.einsum("bch,rh->bcr", branch_8, self.output_basis)
        coefficients_32 = torch.einsum(
            "bch,rh->bcr", branch_32, self.output_basis
        )
        mixed_coefficients = weights * coefficients_8 + (1.0 - weights) * coefficients_32
        controls = torch.einsum(
            "bcr,rh->bch", mixed_coefficients, self.output_basis
        )
        uncapped_rms = controls.square().mean(dim=-1, keepdim=True).sqrt()
        scale = torch.clamp(
            self.maximum_control_rms / uncapped_rms.clamp_min(1e-8), max=1.0
        )
        controls = controls * scale
        actual_rms = controls.square().mean(dim=-1).sqrt()
        directions = F.normalize(mixed_coefficients, dim=-1, eps=1e-8)
        return controls, directions, actual_rms, weights

    def forward_from_signature(
        self,
        scene_signature: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> TeacherBasisControlOutput:
        branch_8, branch_32 = self.branch_outputs_from_signature(
            scene_signature, question_embeddings, question_attention_mask
        )
        # Every branch output lies in the shared orthonormal teacher basis.
        # Coordinate-wise mixing is continuous and directly inspectable; the
        # helper applies the analytic RMS cap required for coordinate mixing.
        controls, directions, actual_rms, weights = self.fuse_branch_outputs(
            branch_8,
            branch_32,
            question_embeddings,
            question_attention_mask,
        )
        gate_logits = torch.full(
            (scene_signature.shape[0],),
            20.0,
            device=controls.device,
            dtype=controls.dtype,
        )
        probabilities = torch.sigmoid(gate_logits)
        if not all(
            torch.isfinite(value).all()
            for value in (controls, directions, actual_rms, weights)
        ):
            raise RuntimeError("V72 controller produced NaN or infinity")
        if float(actual_rms.max().detach().cpu()) > self.maximum_control_rms + 1e-5:
            raise RuntimeError("V72 controller exceeded its analytic RMS bound")

        if scene_signature.shape[0] == 1:
            flat_weights = weights.detach().float().cpu().flatten()
            self._last_v72_audit = AdaptiveFusionAuditV72(
                scene_token_count=self.expected_environment_latents + 2,
                environment_latent_count=self.expected_environment_latents,
                control_token_count=self.control_token_count,
                branch_moment_counts=self.branch_moment_counts,
                signature_moment_count=self.moment_count,
                output_basis_rank=self.output_basis_rank,
                every_environment_latent_influences_both_branches=True,
                both_complete_scene_branches_executed=True,
                question_dependent_scene_retrieval=False,
                latent_selection_or_top_k_used=False,
                fusion_conditioned_on_question_embedding=True,
                fusion_conditioned_on_scene_latent_identity=False,
                fusion_weight_minimum=float(flat_weights.min()),
                fusion_weight_maximum=float(flat_weights.max()),
                fusion_weight_mean=float(flat_weights.mean()),
                fusion_weight_standard_deviation=float(flat_weights.std(unbiased=False)),
                fusion_bounded_away_from_endpoints=bool(
                    float(flat_weights.min()) >= self.fusion_floor - 1e-6
                    and float(flat_weights.max()) <= 1.0 - self.fusion_floor + 1e-6
                ),
                maximum_control_rms=float(actual_rms.max().detach().cpu()),
            )
        else:
            self._last_v72_audit = None
        return TeacherBasisControlOutput(
            control_tokens=controls,
            coefficient_directions=directions,
            control_rms=actual_rms,
            gate_logits=gate_logits,
            gate_probabilities=probabilities,
        )

    def audit(self) -> AdaptiveFusionAuditV72:
        if self._last_v72_audit is None:
            raise RuntimeError("V72 audit requires a completed batch-one forward pass")
        return self._last_v72_audit


__all__ = [
    "AdaptiveFusionAuditV72",
    "AdaptiveMultiscaleTeacherBasisControlV72",
]

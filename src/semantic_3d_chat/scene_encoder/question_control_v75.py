"""Nonlinear successor to the globally complete V74 scene reader.

V75 changes only the coefficient decoder.  A bias-free Linear/GELU/Linear
mapping gives the complete scene/question interaction enough capacity to match
verified continuous Gemma prompts more precisely.  GELU maps zero to zero and
both linear layers omit biases, so the inherited exact-zero-scene guarantee is
preserved without a question-only output path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
    DenseFullSceneControlAuditV74,
)


@dataclass(frozen=True)
class DenseFullSceneControlAuditV75(DenseFullSceneControlAuditV74):
    coefficient_decoder_hidden_dimension: int
    bias_free_nonlinear_coefficient_decoder: bool
    zero_preserving_coefficient_activation: bool


class DenseFullSceneContinuousControlV75(DenseFullSceneContinuousControlV74):
    """V74 dense attention with a zero-safe nonlinear coefficient decoder."""

    def __init__(
        self,
        hidden_size: int,
        output_basis: torch.Tensor,
        *,
        environment_latents: int = 256,
        query_count: int = 4,
        model_dimension: int = 128,
        coefficient_decoder_hidden_dimension: int = 768,
        uniform_floor_mass: float = 0.05,
        maximum_control_rms: float = 0.25,
    ) -> None:
        if coefficient_decoder_hidden_dimension < 1:
            raise ValueError("V75 coefficient-decoder dimension must be positive")
        super().__init__(
            hidden_size,
            output_basis,
            environment_latents=environment_latents,
            query_count=query_count,
            model_dimension=model_dimension,
            uniform_floor_mass=uniform_floor_mass,
            maximum_control_rms=maximum_control_rms,
        )
        input_dimension = self.query_count * self.model_dimension
        output_dimension = self.query_count * self.output_basis_rank
        self.coefficient_decoder_hidden_dimension = int(
            coefficient_decoder_hidden_dimension
        )
        del self.coefficient_output
        self.coefficient_hidden = nn.Linear(
            input_dimension,
            self.coefficient_decoder_hidden_dimension,
            bias=False,
        )
        self.coefficient_activation = nn.GELU()
        self.coefficient_output = nn.Linear(
            self.coefficient_decoder_hidden_dimension,
            output_dimension,
            bias=False,
        )

    def _decode_coefficients(self, interaction: torch.Tensor) -> torch.Tensor:
        return self.coefficient_output(
            self.coefficient_activation(self.coefficient_hidden(interaction))
        )

    def audit(self) -> DenseFullSceneControlAuditV75:
        base = super().audit()
        return DenseFullSceneControlAuditV75(
            **base.__dict__,
            coefficient_decoder_hidden_dimension=(
                self.coefficient_decoder_hidden_dimension
            ),
            bias_free_nonlinear_coefficient_decoder=True,
            zero_preserving_coefficient_activation=True,
        )


__all__ = [
    "DenseFullSceneContinuousControlV75",
    "DenseFullSceneControlAuditV75",
]

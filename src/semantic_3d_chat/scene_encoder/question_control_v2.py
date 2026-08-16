"""Compact bounded scene/question control with an exact no-control route.

V1 proved that Gemma can consume continuous scene controls, but its learned
softmax scene attention saturated and hurt ordinary answers.  V2 removes that
attention entirely.  Before any question, it computes fixed DCT moments over
all 256 environment latents (excluding the protocol BOI/EOI embeddings).  A
small bilinear scene-by-question network then produces bounded control tokens
and a scene-by-question route decision.  The off route supplies no controls at
all, reproducing the base decoder path exactly.

This is global full-scene conditioning, not retrieval: every environment latent
contributes to the DC moment and no question selects a spatial subset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BoundedQuestionControlOutput:
    control_tokens: torch.Tensor
    gate_logits: torch.Tensor
    gate_probabilities: torch.Tensor
    control_rms: torch.Tensor


@dataclass(frozen=True)
class BoundedQuestionControlAudit:
    scene_token_count: int
    environment_latent_count: int
    control_token_count: int
    scene_moment_count: int
    every_environment_latent_influenced_signature: bool
    question_dependent_scene_retrieval: bool
    softmax_scene_attention_used: bool
    gate_probability: float
    control_used: bool
    maximum_control_rms: float


class BoundedFullSceneQuestionControlV2(nn.Module):
    """Fixed global scene signature plus compact bilinear continuous controls."""

    def __init__(
        self,
        hidden_size: int,
        *,
        control_tokens: int = 4,
        expected_environment_latents: int = 256,
        moment_count: int = 8,
        interaction_dim: int = 24,
        output_rank: int = 64,
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
            "output_rank": output_rank,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in dimensions.values()
        ):
            raise ValueError("V2 dimensions must be positive integers")
        if moment_count > expected_environment_latents:
            raise ValueError("V2 moment_count cannot exceed environment latents")
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
                raise TypeError(f"V2 {field} must be a finite number")
        if not 0.0 < float(maximum_control_rms) <= 1.0:
            raise ValueError("V2 maximum_control_rms must be in (0, 1]")
        if not 0.0 < float(initial_control_rms) < float(maximum_control_rms):
            raise ValueError(
                "V2 initial_control_rms must be in (0, maximum_control_rms)"
            )
        if not 0.0 < float(gate_threshold) < 1.0:
            raise ValueError("V2 gate_threshold must be in (0, 1)")

        self.hidden_size = int(hidden_size)
        self.control_token_count = int(control_tokens)
        self.expected_environment_latents = int(expected_environment_latents)
        self.moment_count = int(moment_count)
        self.interaction_dim = int(interaction_dim)
        self.output_rank = int(output_rank)
        self.maximum_control_rms = float(maximum_control_rms)
        self.initial_control_rms = float(initial_control_rms)
        self.gate_threshold = float(gate_threshold)

        self.question_norm = nn.LayerNorm(self.hidden_size)
        self.scene_projection = nn.Linear(self.hidden_size, self.interaction_dim)
        self.question_control_projection = nn.Linear(
            self.hidden_size, self.control_token_count * self.interaction_dim
        )
        interaction_width = self.moment_count * self.interaction_dim
        self.control_output = nn.Sequential(
            nn.LayerNorm(interaction_width),
            nn.Linear(interaction_width, self.output_rank),
            nn.SiLU(),
            nn.Linear(self.output_rank, self.hidden_size),
        )
        self.control_gain = nn.Linear(interaction_width, 1)
        self.question_gate_projection = nn.Linear(
            self.hidden_size, self.interaction_dim
        )
        self.route_gate = nn.Linear(self.interaction_dim, 1)
        nn.init.zeros_(self.route_gate.weight)
        nn.init.constant_(self.route_gate.bias, -2.0)
        # Stable near-zero initial prompts without creating a pure-question or
        # constant bypass around the scene/question bilinear interaction.
        output = self.control_output[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("V2 control output layout changed")
        nn.init.normal_(output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(output.bias)
        nn.init.zeros_(self.control_gain.weight)
        initial_fraction = self.initial_control_rms / self.maximum_control_rms
        nn.init.constant_(
            self.control_gain.bias,
            math.log(initial_fraction / (1.0 - initial_fraction)),
        )
        self._last_audit: BoundedQuestionControlAudit | None = None

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _pooled_question(
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if question_attention_mask is None:
            return question_embeddings.mean(dim=1)
        if question_attention_mask.shape != question_embeddings.shape[:2]:
            raise ValueError("V2 question attention mask must have shape [B,Q]")
        weights = question_attention_mask.to(question_embeddings).clamp(0.0, 1.0)
        if not torch.isfinite(weights).all():
            raise ValueError("V2 question attention mask must be finite")
        counts = weights.sum(dim=1, keepdim=True)
        if torch.any(counts <= 0):
            raise ValueError("Every V2 question requires an unmasked token")
        return torch.sum(question_embeddings * weights.unsqueeze(-1), dim=1) / counts

    def encode_scene(self, scene_prefix: torch.Tensor) -> torch.Tensor:
        """Compute the fixed full-environment signature before any question."""

        if scene_prefix.ndim != 3 or scene_prefix.shape[-1] != self.hidden_size:
            raise ValueError("V2 scene prefix must have shape [B,S,H]")
        expected_tokens = self.expected_environment_latents + 2
        if scene_prefix.shape[1] != expected_tokens:
            raise ValueError(
                "V2 expected BOI + complete environment latents + EOI: "
                f"expected={expected_tokens} observed={scene_prefix.shape[1]}"
            )
        if not torch.isfinite(scene_prefix).all():
            raise ValueError("V2 scene prefix must be finite")
        environment = scene_prefix[:, 1:-1].float()
        environment = environment / environment.square().mean(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)
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
        signature = torch.einsum("ml,blh->bmh", weights, environment)
        if not torch.isfinite(signature).all():
            raise RuntimeError("V2 scene signature contains NaN or infinity")
        return signature.detach()

    def forward_from_signature(
        self,
        scene_signature: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> BoundedQuestionControlOutput:
        if scene_signature.ndim != 3 or scene_signature.shape[1:] != (
            self.moment_count,
            self.hidden_size,
        ):
            raise ValueError("V2 scene signature shape changed")
        if question_embeddings.ndim != 3 or question_embeddings.shape[-1] != self.hidden_size:
            raise ValueError("V2 question embeddings must have shape [B,Q,H]")
        if scene_signature.shape[0] != question_embeddings.shape[0]:
            raise ValueError("V2 signature and question batch sizes must match")
        if not torch.isfinite(scene_signature).all() or not torch.isfinite(
            question_embeddings
        ).all():
            raise ValueError("V2 signature and question embeddings must be finite")

        pooled = self._pooled_question(question_embeddings, question_attention_mask)
        normalized_question = self.question_norm(pooled.float())
        scene_factors = self.scene_projection(scene_signature.float())
        question_factors = self.question_control_projection(
            normalized_question
        ).reshape(-1, self.control_token_count, self.interaction_dim)
        interaction = (
            scene_factors[:, None, :, :] * question_factors[:, :, None, :]
        ).reshape(-1, self.control_token_count, self.moment_count * self.interaction_dim)
        raw_control = self.control_output(interaction)
        # RMS-normalize then bound the learned gain with sigmoid.  This retains
        # direction while guaranteeing a declared finite magnitude.
        raw_rms = raw_control.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        bounded_gain = torch.sigmoid(self.control_gain(interaction))
        control = raw_control / raw_rms * self.maximum_control_rms * bounded_gain

        scene_gate = scene_factors[:, 0, :]
        question_gate = self.question_gate_projection(normalized_question)
        gate_interaction = scene_gate * question_gate
        gate_logits = self.route_gate(gate_interaction).squeeze(-1)
        gate_probabilities = torch.sigmoid(gate_logits)
        control_rms = control.square().mean(dim=-1).sqrt()
        if not all(
            torch.isfinite(value).all()
            for value in (control, gate_logits, gate_probabilities, control_rms)
        ):
            raise RuntimeError("V2 controller produced NaN or infinity")
        if float(control_rms.max().detach().cpu()) > self.maximum_control_rms + 1e-5:
            raise RuntimeError("V2 control exceeded its declared RMS bound")

        if scene_signature.shape[0] == 1:
            probability = float(gate_probabilities[0].detach().cpu())
            self._last_audit = BoundedQuestionControlAudit(
                scene_token_count=self.expected_environment_latents + 2,
                environment_latent_count=self.expected_environment_latents,
                control_token_count=self.control_token_count,
                scene_moment_count=self.moment_count,
                every_environment_latent_influenced_signature=True,
                question_dependent_scene_retrieval=False,
                softmax_scene_attention_used=False,
                gate_probability=probability,
                control_used=probability >= self.gate_threshold,
                maximum_control_rms=float(control_rms.max().detach().cpu()),
            )
        else:
            self._last_audit = None
        return BoundedQuestionControlOutput(
            control_tokens=control,
            gate_logits=gate_logits,
            gate_probabilities=gate_probabilities,
            control_rms=control_rms,
        )

    def forward(
        self,
        scene_prefix: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> BoundedQuestionControlOutput:
        signature = self.encode_scene(scene_prefix)
        return self.forward_from_signature(
            signature, question_embeddings, question_attention_mask
        )

    def audit(self) -> BoundedQuestionControlAudit:
        if self._last_audit is None:
            raise RuntimeError("V2 audit requires a completed batch-one forward pass")
        return self._last_audit


__all__ = [
    "BoundedFullSceneQuestionControlV2",
    "BoundedQuestionControlAudit",
    "BoundedQuestionControlOutput",
]

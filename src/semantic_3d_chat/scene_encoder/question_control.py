"""Question-conditioned continuous control over a complete static scene prefix.

The module is deliberately *not* a retriever.  Every question attends to every
global scene latent, and a uniform probability floor guarantees that no latent
can be omitted.  Its outputs are continuous decoder-side tokens; it has no
object vocabulary, label lookup, caption, scene graph, or textual intermediate.
The immutable scene prefix remains unchanged and can therefore be hashed once
before any question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class QuestionControlAudit:
    scene_token_count: int
    control_token_count: int
    minimum_attention_weight: float
    maximum_attention_weight: float
    every_scene_token_influenced_output: bool


class FullSceneQuestionControl(nn.Module):
    """Map a fixed full-scene prefix and user question to continuous tokens."""

    def __init__(
        self,
        hidden_size: int,
        *,
        attention_dim: int = 256,
        control_tokens: int = 4,
        uniform_floor: float = 0.05,
        output_scale: float = 0.25,
    ) -> None:
        super().__init__()
        dimensions = {
            "hidden_size": hidden_size,
            "attention_dim": attention_dim,
            "control_tokens": control_tokens,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in dimensions.values()
        ):
            raise ValueError("Question-control dimensions must be positive integers")
        if (
            isinstance(uniform_floor, bool)
            or not isinstance(uniform_floor, (int, float))
            or not math.isfinite(float(uniform_floor))
            or not 0.0 < float(uniform_floor) <= 1.0
        ):
            raise ValueError("uniform_floor must be in (0, 1]")
        if (
            isinstance(output_scale, bool)
            or not isinstance(output_scale, (int, float))
            or not math.isfinite(float(output_scale))
            or float(output_scale) <= 0.0
        ):
            raise ValueError("output_scale must be finite and positive")
        self.hidden_size = int(hidden_size)
        self.attention_dim = int(attention_dim)
        self.control_token_count = int(control_tokens)
        self.uniform_floor = float(uniform_floor)
        self.output_scale = float(output_scale)

        self.scene_norm = nn.LayerNorm(hidden_size)
        self.question_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, control_tokens * attention_dim)
        self.key = nn.Linear(hidden_size, attention_dim, bias=False)
        self.value = nn.Linear(hidden_size, attention_dim, bias=False)
        self.output = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, attention_dim * 2),
            nn.SiLU(),
            nn.Linear(attention_dim * 2, hidden_size),
        )
        self.control_identity = nn.Parameter(
            torch.zeros(1, control_tokens, attention_dim)
        )
        self._last_attention: torch.Tensor | None = None

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        scene_prefix: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scene_prefix.ndim != 3 or question_embeddings.ndim != 3:
            raise ValueError("Scene prefix and question embeddings must have shape [B,L,H]")
        if scene_prefix.shape[0] != question_embeddings.shape[0]:
            raise ValueError("Scene and question batch sizes must match")
        if (
            scene_prefix.shape[-1] != self.hidden_size
            or question_embeddings.shape[-1] != self.hidden_size
        ):
            raise ValueError("Question-control hidden dimension mismatch")
        if scene_prefix.shape[1] < 1 or question_embeddings.shape[1] < 1:
            raise ValueError("Scene and question sequences must be nonempty")
        if not torch.isfinite(scene_prefix).all() or not torch.isfinite(
            question_embeddings
        ).all():
            raise ValueError("Question-control inputs must be finite")

        if question_attention_mask is None:
            pooled_question = question_embeddings.mean(dim=1)
        else:
            if question_attention_mask.shape != question_embeddings.shape[:2]:
                raise ValueError("Question attention mask must have shape [B,Q]")
            weights = question_attention_mask.to(question_embeddings)
            if not torch.isfinite(weights).all():
                raise ValueError("Question attention mask must be finite")
            weights = weights.clamp(0.0, 1.0)
            counts = weights.sum(dim=1, keepdim=True)
            if torch.any(counts <= 0):
                raise ValueError("Every question requires at least one unmasked token")
            pooled_question = torch.sum(
                question_embeddings * weights.unsqueeze(-1), dim=1
            ) / counts

        normalized_scene = self.scene_norm(scene_prefix)
        normalized_question = self.question_norm(pooled_question)
        query = self.query(normalized_question).reshape(
            scene_prefix.shape[0], self.control_token_count, self.attention_dim
        )
        query = query + self.control_identity
        keys = self.key(normalized_scene)
        values = self.value(normalized_scene)
        logits = torch.einsum("bcd,bld->bcl", query, keys) / math.sqrt(
            self.attention_dim
        )
        learned = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        token_count = scene_prefix.shape[1]
        attention = (
            (1.0 - self.uniform_floor) * learned
            + self.uniform_floor / float(token_count)
        )
        context = torch.einsum("bcl,bld->bcd", attention, values)
        control = self.output(context + query) * self.output_scale
        if not torch.isfinite(control).all():
            raise RuntimeError("Question-control output contains NaN or infinity")
        self._last_attention = attention.detach()
        return control

    def audit(self) -> QuestionControlAudit:
        if self._last_attention is None:
            raise RuntimeError("Question-control audit requires a completed forward pass")
        attention = self._last_attention.float()
        minimum = float(attention.min().cpu())
        maximum = float(attention.max().cpu())
        return QuestionControlAudit(
            scene_token_count=int(attention.shape[-1]),
            control_token_count=int(attention.shape[-2]),
            minimum_attention_weight=minimum,
            maximum_attention_weight=maximum,
            every_scene_token_influenced_output=minimum > 0.0,
        )


__all__ = ["FullSceneQuestionControl", "QuestionControlAudit"]

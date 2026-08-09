from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from semantic_3d_chat.scene_encoder.perceiver import spatial_anchors
from semantic_3d_chat.scene_encoder.point_tokens import FourierXYZ


@dataclass(frozen=True)
class DenseRelationTokenOutput:
    """Continuous relation evidence derived from a complete scene-token set."""

    relation_token: torch.Tensor
    target_xyz_normalized: torch.Tensor
    reference_xyz_normalized: torch.Tensor
    delta_xyz_normalized: torch.Tensor
    target_attention: torch.Tensor
    reference_attention: torch.Tensor


class DenseRelationTokenAdapter(nn.Module):
    """Bind two question roles to geometry without retrieving scene subsets.

    Two learned role queries first attend to the complete question embedding
    sequence. Both conditioned roles then attend densely to every fixed scene
    token. Scene keys combine continuous scene content with deterministic Halton
    anchors and Fourier XYZ features. Attention is floored and renormalized in
    float32, so no scene token is removed by top-k, radius, or underflow.

    The target-minus-reference scene feature and coordinate delta are projected
    into one continuous LM-dimensional token. The projection is deliberately
    odd (bias-free linear layers with ``tanh``), so exchanging the two roles
    exchanges their attention/coordinates and negates the relation token.
    """

    def __init__(
        self,
        language_hidden_dim: int,
        *,
        num_scene_tokens: int = 256,
        adapter_dim: int = 128,
        heads: int = 4,
        fourier_bands: int = 4,
        initialization_seed: int = 15008,
    ) -> None:
        super().__init__()
        if language_hidden_dim < 1:
            raise ValueError("language_hidden_dim must be positive")
        if num_scene_tokens < 1:
            raise ValueError("num_scene_tokens must be positive")
        if adapter_dim < 1:
            raise ValueError("adapter_dim must be positive")
        if heads < 1 or adapter_dim % heads != 0:
            raise ValueError("heads must be positive and divide adapter_dim")
        if fourier_bands < 1:
            raise ValueError("fourier_bands must be positive")
        if not isinstance(initialization_seed, int):
            raise TypeError("initialization_seed must be an integer")

        self.language_hidden_dim = int(language_hidden_dim)
        self.num_scene_tokens = int(num_scene_tokens)
        self.adapter_dim = int(adapter_dim)
        self.initialization_seed = int(initialization_seed)

        # Construction is reproducible without consuming or replacing the
        # caller's global RNG state. All trainable tensors are initialized here.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.question_projection = nn.Linear(language_hidden_dim, adapter_dim)
            self.question_norm = nn.LayerNorm(adapter_dim)
            self.question_attention = nn.MultiheadAttention(
                adapter_dim,
                heads,
                dropout=0.0,
                batch_first=True,
            )
            self.role_queries = nn.Parameter(torch.empty(2, adapter_dim))
            nn.init.normal_(self.role_queries, mean=0.0, std=0.02)

            self.scene_projection = nn.Linear(language_hidden_dim, adapter_dim)
            self.scene_position_projection = nn.Linear(
                3 + 3 * fourier_bands * 2,
                adapter_dim,
            )
            self.scene_norm = nn.LayerNorm(adapter_dim)
            self.scene_key = nn.Linear(adapter_dim, adapter_dim, bias=False)
            self.scene_value = nn.Linear(adapter_dim, adapter_dim, bias=False)

            # With no biases and an odd nonlinearity, f(-x) == -f(x).
            self.relation_projection = nn.Sequential(
                nn.Linear(adapter_dim + 3, adapter_dim, bias=False),
                nn.Tanh(),
                nn.Linear(adapter_dim, language_hidden_dim, bias=False),
            )

        self.fourier = FourierXYZ(fourier_bands)
        self.register_buffer(
            "scene_anchors",
            spatial_anchors(num_scene_tokens),
            persistent=False,
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_inputs(
        self,
        scene_tokens: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        expected_scene_suffix = (self.num_scene_tokens, self.language_hidden_dim)
        if scene_tokens.ndim != 3 or tuple(scene_tokens.shape[1:]) != expected_scene_suffix:
            raise ValueError(
                "scene_tokens must have shape "
                f"[B,{self.num_scene_tokens},{self.language_hidden_dim}]"
            )
        if (
            question_embeddings.ndim != 3
            or question_embeddings.shape[0] != scene_tokens.shape[0]
            or question_embeddings.shape[1] < 1
            or question_embeddings.shape[2] != self.language_hidden_dim
        ):
            raise ValueError(
                "question_embeddings must have shape "
                f"[B,T,{self.language_hidden_dim}] with T >= 1 and matching batch size"
            )
        if not scene_tokens.is_floating_point() or not question_embeddings.is_floating_point():
            raise TypeError("scene_tokens and question_embeddings must be floating tensors")
        if not bool(torch.isfinite(scene_tokens).all()):
            raise ValueError("scene_tokens contains NaN or infinity")
        if not bool(torch.isfinite(question_embeddings).all()):
            raise ValueError("question_embeddings contains NaN or infinity")

        if question_mask is None:
            return None
        if question_mask.shape != question_embeddings.shape[:2]:
            raise ValueError("question_mask must have shape [B,T]")
        if question_mask.dtype == torch.bool:
            valid = question_mask
        elif question_mask.dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            if not bool(((question_mask == 0) | (question_mask == 1)).all()):
                raise ValueError("integer question_mask values must be zero or one")
            valid = question_mask.bool()
        else:
            raise TypeError("question_mask must be boolean or an integer zero/one tensor")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("question_mask must retain at least one token per batch item")
        return valid

    def _positive_dense_attention(self, logits: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(logits.float(), dim=-1)
        weights = weights.clamp_min(torch.finfo(torch.float32).tiny)
        return weights / weights.sum(dim=-1, keepdim=True)

    def _question_positions(
        self,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return deterministic sinusoidal positions without learned text metadata."""

        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        even_dimensions = torch.arange(
            0,
            self.adapter_dim,
            2,
            device=device,
            dtype=torch.float32,
        )
        frequencies = torch.exp(even_dimensions * (-math.log(10_000.0) / float(self.adapter_dim)))
        angles = positions * frequencies.unsqueeze(0)
        encoding = torch.zeros(length, self.adapter_dim, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=dtype)

    def forward(
        self,
        scene_tokens: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_mask: torch.Tensor | None = None,
        *,
        swap_roles: bool = False,
    ) -> DenseRelationTokenOutput:
        valid_question_mask = self._validate_inputs(
            scene_tokens,
            question_embeddings,
            question_mask,
        )
        if not isinstance(swap_roles, bool):
            raise TypeError("swap_roles must be a boolean")

        parameter_dtype = self.question_projection.weight.dtype
        parameter_device = self.question_projection.weight.device
        if (
            scene_tokens.device != parameter_device
            or question_embeddings.device != parameter_device
        ):
            raise ValueError("Adapter and input tensors must be on the same device")

        question = self.question_projection(question_embeddings.to(dtype=parameter_dtype))
        question = question + self._question_positions(
            question.shape[1],
            device=parameter_device,
            dtype=parameter_dtype,
        ).unsqueeze(0)
        question = self.question_norm(question)
        roles = self.role_queries.unsqueeze(0).expand(scene_tokens.shape[0], -1, -1)
        conditioned_roles, _ = self.question_attention(
            roles,
            question,
            question,
            key_padding_mask=(
                None if valid_question_mask is None else ~valid_question_mask.to(parameter_device)
            ),
            need_weights=False,
        )
        conditioned_roles = conditioned_roles + roles
        if swap_roles:
            conditioned_roles = conditioned_roles.flip(dims=(1,))

        anchors = self.scene_anchors.to(device=parameter_device, dtype=torch.float32)
        position_features = torch.cat((anchors, self.fourier(anchors)), dim=-1)
        scene = self.scene_projection(scene_tokens.to(dtype=parameter_dtype))
        scene = scene + self.scene_position_projection(
            position_features.to(dtype=parameter_dtype)
        ).unsqueeze(0)
        scene = self.scene_norm(scene)
        keys = self.scene_key(scene)
        values = self.scene_value(scene)

        attention_logits = torch.matmul(
            conditioned_roles.float(),
            keys.float().transpose(1, 2),
        ) / math.sqrt(self.adapter_dim)
        attention = self._positive_dense_attention(attention_logits)
        pooled = torch.matmul(attention, values.float())
        role_xyz = torch.matmul(attention, anchors)

        pooled_delta = pooled[:, 0] - pooled[:, 1]
        xyz_delta = role_xyz[:, 0] - role_xyz[:, 1]
        relation_features = torch.cat((pooled_delta, xyz_delta), dim=-1)
        relation_token = self.relation_projection(
            relation_features.to(dtype=parameter_dtype)
        ).unsqueeze(1)

        outputs = (
            relation_token,
            role_xyz,
            xyz_delta,
            attention,
        )
        if not all(bool(torch.isfinite(value).all()) for value in outputs):
            raise RuntimeError("Dense relation adapter produced NaN or infinity")
        return DenseRelationTokenOutput(
            relation_token=relation_token,
            target_xyz_normalized=role_xyz[:, 0],
            reference_xyz_normalized=role_xyz[:, 1],
            delta_xyz_normalized=xyz_delta,
            target_attention=attention[:, 0],
            reference_attention=attention[:, 1],
        )

"""V73 full-scene, positive-floor attention controller.

The immutable scene prefix remains the primary LM prefix.  This module reads
the 256 environment latents without replacing, retrieving, or ranking them and
emits four additional native-width continuous control tokens.  Every scene
attention distribution is mixed with a uniform distribution, so each latent
receives a strictly positive, auditable minimum weight.

``DCT40QuestionControlBaselineV73`` is an explicit same-stack ablation of the
V71 scene bottleneck: it replaces the 256 latent memory with the concatenated
first-8 and first-32 DCT moments while leaving the question reader and output
head unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn


_DEFAULT_ENVIRONMENT_LATENTS: Final[int] = 256
_DEFAULT_CONTROL_TOKENS: Final[int] = 4


@dataclass(frozen=True)
class PositiveFloorAttentionTraceV73:
    query_tokens: int
    memory_tokens: int
    head_count: int
    uniform_floor_mass: float
    required_minimum_weight: float
    observed_minimum_weight: float
    all_memory_tokens_receive_positive_weight: bool


@dataclass(frozen=True)
class FullSceneControlAuditV73:
    scene_prefix_tokens: int
    environment_latent_count: int
    scene_memory_tokens: int
    control_token_count: int
    hidden_size: int
    model_dimension: int
    scene_encoder_layers: int
    scene_cross_attention_layers: int
    internal_reader_slots: int
    uniform_floor_mass: float
    minimum_cross_attention_weight: float
    required_cross_attention_weight: float
    every_environment_latent_processed: bool
    full_prefix_retained_separately_for_language_model: bool
    question_conditioned_continuous_attention: bool
    question_dependent_retrieval: bool
    latent_selection_or_top_k_used: bool
    environmental_text_inputs: int
    dct_scene_bottleneck_used: bool
    question_only_output_path_exists: bool
    output_computed_only_from_scene_value_contexts: bool
    zero_scene_produces_exact_zero_controls: bool
    maximum_control_rms: float


@dataclass(frozen=True)
class FullSceneControlOutputV73:
    control_tokens: torch.Tensor
    coefficient_directions: torch.Tensor
    control_rms: torch.Tensor


def _sinusoidal_positions(
    length: int, dimension: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if length < 1 or dimension < 2:
        raise ValueError("V73 sinusoidal positions require positive length and dim >= 2")
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    even_count = (dimension + 1) // 2
    scale = torch.exp(
        torch.arange(even_count, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(even_count - 1, 1))
    )[None, :]
    angles = position * scale
    value = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    value[:, 0::2] = torch.sin(angles[:, : value[:, 0::2].shape[1]])
    value[:, 1::2] = torch.cos(angles[:, : value[:, 1::2].shape[1]])
    return value.to(dtype=dtype)


class PositiveFloorMultiheadAttentionV73(nn.Module):
    """Multihead attention with an analytic positive floor over valid keys."""

    def __init__(
        self,
        dimension: int,
        head_count: int,
        *,
        uniform_floor_mass: float,
    ) -> None:
        super().__init__()
        if dimension < 1 or head_count < 1 or dimension % head_count:
            raise ValueError("V73 attention dimension must be divisible by heads")
        if not 0.0 < float(uniform_floor_mass) < 1.0:
            raise ValueError("V73 uniform floor mass must lie in (0,1)")
        self.dimension = int(dimension)
        self.head_count = int(head_count)
        self.head_dimension = self.dimension // self.head_count
        self.uniform_floor_mass = float(uniform_floor_mass)
        # Bias-free values and output prevent a query-only path.  Query/key
        # biases could only change weights, but are omitted as well to keep the
        # mathematical dependency especially easy to audit.
        self.query_projection = nn.Linear(dimension, dimension, bias=False)
        self.key_projection = nn.Linear(dimension, dimension, bias=False)
        self.value_projection = nn.Linear(dimension, dimension, bias=False)
        self.output_projection = nn.Linear(dimension, dimension, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, PositiveFloorAttentionTraceV73 | None]:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError("V73 attention inputs must have shape [B,T,D]")
        if (
            query.shape[0] != memory.shape[0]
            or query.shape[-1] != self.dimension
            or memory.shape[-1] != self.dimension
            or query.shape[1] < 1
            or memory.shape[1] < 1
        ):
            raise ValueError("V73 attention batch, token, or feature dimensions changed")
        batch, query_tokens, _ = query.shape
        memory_tokens = memory.shape[1]

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch, value.shape[1], self.head_count, self.head_dimension
            ).transpose(1, 2)

        q_source = query
        k_source = memory
        if query_positions is not None:
            if query_positions.shape not in {
                query.shape,
                (1, query.shape[1], query.shape[2]),
            }:
                raise ValueError("V73 query positions shape changed")
            q_source = q_source + query_positions.to(q_source)
        if memory_positions is not None:
            if memory_positions.shape not in {
                memory.shape,
                (1, memory.shape[1], memory.shape[2]),
            }:
                raise ValueError("V73 memory positions shape changed")
            # Positions influence keys, never values.
            k_source = k_source + memory_positions.to(k_source)
        q = heads(self.query_projection(q_source))
        k = heads(self.key_projection(k_source))
        v = heads(self.value_projection(memory))
        logits = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float())
        logits = logits / math.sqrt(float(self.head_dimension))

        if memory_mask is None:
            valid = torch.ones(
                batch, memory_tokens, device=memory.device, dtype=torch.bool
            )
        else:
            if memory_mask.shape != (batch, memory_tokens):
                raise ValueError("V73 memory mask must have shape [B,K]")
            valid = memory_mask.to(device=memory.device, dtype=torch.bool)
            if not bool(valid.any(dim=1).all()):
                raise ValueError("V73 attention cannot receive an empty memory row")
        expanded_valid = valid[:, None, None, :]
        probabilities = torch.softmax(
            logits.masked_fill(~expanded_valid, -torch.inf), dim=-1
        )
        valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1).float()
        uniform = valid.float() / valid_count
        weights = (
            (1.0 - self.uniform_floor_mass) * probabilities
            + self.uniform_floor_mass * uniform[:, None, None, :]
        )
        weights = weights.masked_fill(~expanded_valid, 0.0)
        attended = torch.einsum("bhqk,bhkd->bhqd", weights.to(v), v)
        attended = attended.transpose(1, 2).reshape(batch, query_tokens, self.dimension)
        output = self.output_projection(attended)
        if not torch.isfinite(output).all():
            raise RuntimeError("V73 attention produced NaN or infinity")

        trace: PositiveFloorAttentionTraceV73 | None = None
        if return_trace:
            valid_weights = weights.masked_select(expanded_valid.expand_as(weights))
            observed = float(valid_weights.min().detach().cpu())
            maximum_valid = int(valid.sum(dim=-1).max().detach().cpu())
            required = self.uniform_floor_mass / float(maximum_valid)
            trace = PositiveFloorAttentionTraceV73(
                query_tokens=query_tokens,
                memory_tokens=memory_tokens,
                head_count=self.head_count,
                uniform_floor_mass=self.uniform_floor_mass,
                required_minimum_weight=required,
                observed_minimum_weight=observed,
                all_memory_tokens_receive_positive_weight=bool(observed > 0.0),
            )
        return output, trace


class _SelfAttentionBlockV73(nn.Module):
    def __init__(
        self,
        dimension: int,
        head_count: int,
        feedforward_dimension: int,
        uniform_floor_mass: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.attention = PositiveFloorMultiheadAttentionV73(
            dimension, head_count, uniform_floor_mass=uniform_floor_mass
        )
        self.feedforward_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward_dimension, bias=False),
            nn.GELU(),
            nn.Linear(feedforward_dimension, dimension, bias=False),
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, PositiveFloorAttentionTraceV73 | None]:
        normalized = self.attention_norm(value)
        attended, trace = self.attention(
            normalized,
            normalized,
            query_positions=positions,
            memory_positions=positions,
            return_trace=return_trace,
        )
        value = value + attended
        return value + self.feedforward(self.feedforward_norm(value)), trace


class _SceneValueReadHopV73(nn.Module):
    """Return scene VALUE context only; never residual-add the query."""

    def __init__(
        self,
        dimension: int,
        head_count: int,
        feedforward_dimension: int,
        uniform_floor_mass: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.attention = PositiveFloorMultiheadAttentionV73(
            dimension, head_count, uniform_floor_mass=uniform_floor_mass
        )
        self.feedforward_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward_dimension, bias=False),
            nn.GELU(),
            nn.Linear(feedforward_dimension, dimension, bias=False),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, PositiveFloorAttentionTraceV73 | None]:
        attended, trace = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            memory_mask=memory_mask,
            memory_positions=memory_positions,
            return_trace=return_trace,
        )
        # Crucially, the result contains no additive query path.  If every
        # scene value is zero, attended and the bias-free feed-forward result
        # are both exactly zero regardless of the question.
        return attended + self.feedforward(self.feedforward_norm(attended)), trace


class FullSceneSetAttentionQuestionControlV73(nn.Module):
    """Question-conditioned continuous reader over all 256 scene latents."""

    dct_scene_bottleneck_used = False

    def __init__(
        self,
        hidden_size: int,
        output_basis: torch.Tensor,
        *,
        expected_environment_latents: int = _DEFAULT_ENVIRONMENT_LATENTS,
        control_token_count: int = _DEFAULT_CONTROL_TOKENS,
        model_dimension: int = 192,
        head_count: int = 6,
        feedforward_dimension: int = 512,
        scene_encoder_layers: int = 2,
        scene_cross_attention_layers: int = 2,
        internal_reader_slots: int = 8,
        uniform_floor_mass: float = 0.10,
        maximum_control_rms: float = 0.25,
        initial_control_rms: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_size < 2 or expected_environment_latents < 1:
            raise ValueError("V73 hidden size and latent count must be positive")
        if control_token_count < 1 or scene_encoder_layers < 0:
            raise ValueError("V73 requires control tokens and nonnegative scene layers")
        if scene_cross_attention_layers < 1:
            raise ValueError("V73 requires at least one scene cross-attention layer")
        if internal_reader_slots < control_token_count or (
            internal_reader_slots % control_token_count
        ):
            raise ValueError(
                "V73 internal reader slots must be a multiple of control tokens"
            )
        if model_dimension % head_count:
            raise ValueError("V73 model dimension must be divisible by head count")
        if not 0.0 < initial_control_rms < maximum_control_rms <= 1.0:
            raise ValueError("V73 RMS bounds require 0 < initial < maximum <= 1")
        if (
            output_basis.ndim != 2
            or output_basis.shape[1] != hidden_size
            or output_basis.shape[0] < 1
            or not torch.isfinite(output_basis).all()
        ):
            raise ValueError("V73 output basis must be finite [R,H]")
        gram = output_basis.float() @ output_basis.float().T
        identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
            raise ValueError("V73 output basis must be row-orthonormal")

        self.hidden_size = int(hidden_size)
        self.expected_environment_latents = int(expected_environment_latents)
        self.control_token_count = int(control_token_count)
        self.model_dimension = int(model_dimension)
        self.head_count = int(head_count)
        self.feedforward_dimension = int(feedforward_dimension)
        self.scene_encoder_layer_count = int(scene_encoder_layers)
        self.scene_cross_attention_layer_count = int(scene_cross_attention_layers)
        self.internal_reader_slots = int(internal_reader_slots)
        self.uniform_floor_mass = float(uniform_floor_mass)
        self.maximum_control_rms = float(maximum_control_rms)
        self.initial_control_rms = float(initial_control_rms)
        self.output_basis_rank = int(output_basis.shape[0])

        self.scene_input_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.scene_projection = nn.Linear(hidden_size, model_dimension, bias=False)
        self.question_input_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.question_projection = nn.Linear(hidden_size, model_dimension, bias=False)
        self.scene_positions = nn.Parameter(
            torch.empty(self.scene_memory_token_count, model_dimension)
        )
        self.control_queries = nn.Parameter(
            torch.empty(internal_reader_slots, model_dimension)
        )
        self.scene_blocks = nn.ModuleList(
            _SelfAttentionBlockV73(
                model_dimension,
                head_count,
                feedforward_dimension,
                uniform_floor_mass,
            )
            for _ in range(scene_encoder_layers)
        )
        self.question_block = _SceneValueReadHopV73(
            model_dimension,
            head_count,
            feedforward_dimension,
            uniform_floor_mass,
        )
        self.scene_cross_blocks = nn.ModuleList(
            _SceneValueReadHopV73(
                model_dimension,
                head_count,
                feedforward_dimension,
                uniform_floor_mass,
            )
            for _ in range(scene_cross_attention_layers)
        )
        self.hop_context_projection = nn.ModuleList(
            nn.Linear(model_dimension, model_dimension, bias=False)
            for _ in range(max(scene_cross_attention_layers - 1, 0))
        )
        self.output_norm = nn.LayerNorm(model_dimension, elementwise_affine=False)
        self.coefficient_output = nn.Linear(
            model_dimension, self.output_basis_rank, bias=False
        )
        self.register_buffer(
            "output_basis", output_basis.detach().float().clone(), persistent=True
        )
        self._last_audit: FullSceneControlAuditV73 | None = None
        self.reset_parameters()

    @property
    def scene_memory_token_count(self) -> int:
        return self.expected_environment_latents

    @property
    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.state_dict().values())

    def reset_parameters(self) -> None:
        nn.init.normal_(self.scene_positions, std=0.02)
        nn.init.normal_(self.control_queries, std=0.02)
        # Exact-zero output initialization is a scientific gate.  Upstream
        # scene/question reader weights remain random and receive gradients
        # after the first optimizer update through this linear head.
        nn.init.zeros_(self.coefficient_output.weight)

    def _scene_source(self, environment: torch.Tensor) -> torch.Tensor:
        return environment

    def encode_scene(
        self, scene_prefix: torch.Tensor, *, return_traces: bool = False
    ) -> tuple[torch.Tensor, tuple[PositiveFloorAttentionTraceV73, ...]]:
        expected_tokens = self.expected_environment_latents + 2
        if (
            scene_prefix.ndim != 3
            or scene_prefix.shape[1:] != (expected_tokens, self.hidden_size)
        ):
            raise ValueError(
                "V73 scene prefix must be BOI + all 256 latents + EOI with shape [B,258,H]"
            )
        if not torch.isfinite(scene_prefix).all():
            raise ValueError("V73 scene prefix must be finite")
        # BOI and EOI remain in the separately supplied immutable LM prefix.
        environment = scene_prefix[:, 1:-1].float()
        source = self._scene_source(environment)
        if source.shape[1] != self.scene_memory_token_count:
            raise RuntimeError("V73 scene source token count changed")
        memory = self.scene_projection(self.scene_input_norm(source))
        traces: list[PositiveFloorAttentionTraceV73] = []
        for block in self.scene_blocks:
            memory, trace = block(
                memory,
                positions=self.scene_positions[None].to(memory),
                return_trace=return_traces,
            )
            if trace is not None:
                traces.append(trace)
        if not torch.isfinite(memory).all():
            raise RuntimeError("V73 encoded scene memory contains NaN or infinity")
        return memory, tuple(traces)

    def forward_from_scene_memory(
        self,
        scene_memory: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
        *,
        return_traces: bool = False,
    ) -> FullSceneControlOutputV73:
        if (
            scene_memory.ndim != 3
            or scene_memory.shape[1:] != (
                self.scene_memory_token_count,
                self.model_dimension,
            )
        ):
            raise ValueError("V73 scene memory shape changed")
        if (
            question_embeddings.ndim != 3
            or question_embeddings.shape[0] != scene_memory.shape[0]
            or question_embeddings.shape[-1] != self.hidden_size
            or question_embeddings.shape[1] < 1
        ):
            raise ValueError("V73 questions must have matching shape [B,Q,H]")
        if question_attention_mask is None:
            question_attention_mask = torch.ones(
                question_embeddings.shape[:2],
                device=question_embeddings.device,
                dtype=torch.bool,
            )
        else:
            if question_attention_mask.shape != question_embeddings.shape[:2]:
                raise ValueError("V73 question mask shape changed")
            question_attention_mask = question_attention_mask.to(
                device=question_embeddings.device, dtype=torch.bool
            )
        question = self.question_projection(
            self.question_input_norm(question_embeddings.float())
        )
        question = question + _sinusoidal_positions(
            question.shape[1],
            self.model_dimension,
            device=question.device,
            dtype=question.dtype,
        )[None]
        question_queries = self.control_queries[None].expand(
            question.shape[0], -1, -1
        )
        # This produces a question context used only as a query.  It is never
        # added to the scene value context returned below.
        question_queries, _question_trace = self.question_block(
            question_queries,
            question,
            memory_mask=question_attention_mask,
            return_trace=False,
        )
        scene_traces: list[PositiveFloorAttentionTraceV73] = []
        scene_context: torch.Tensor | None = None
        for index, block in enumerate(self.scene_cross_blocks):
            query = question_queries
            if scene_context is not None:
                query = query + self.hop_context_projection[index - 1](scene_context)
            scene_context, trace = block(
                query,
                scene_memory,
                memory_positions=self.scene_positions[None].to(scene_memory),
                return_trace=return_traces,
            )
            if trace is not None:
                scene_traces.append(trace)
        if scene_context is None:
            raise RuntimeError("V73 scene reader executed no dense hop")
        slots_per_output = self.internal_reader_slots // self.control_token_count
        controls = scene_context.reshape(
            scene_context.shape[0],
            self.control_token_count,
            slots_per_output,
            self.model_dimension,
        ).mean(dim=2)
        raw_coefficients = self.coefficient_output(self.output_norm(controls))
        coefficient_norm = (
            raw_coefficients.square().sum(dim=-1, keepdim=True) + 1e-16
        ).sqrt()
        coefficients = raw_coefficients / coefficient_norm.clamp_min(1e-8)
        raw = torch.einsum("bcr,rh->bch", raw_coefficients, self.output_basis)
        # The epsilon makes the exact-zero initialization differentiable while
        # the output itself remains exactly zero.
        raw_rms = (raw.square().mean(dim=-1, keepdim=True) + 1e-16).sqrt()
        scale = torch.clamp(
            self.maximum_control_rms / raw_rms.clamp_min(1e-8), max=1.0
        )
        output = raw * scale
        rms = output.square().mean(dim=-1).sqrt()
        if not all(torch.isfinite(value).all() for value in (output, coefficients, rms)):
            raise RuntimeError("V73 control output contains NaN or infinity")
        if float(rms.detach().max().cpu()) > self.maximum_control_rms + 1e-6:
            raise RuntimeError("V73 control output exceeded its RMS bound")

        if return_traces:
            if not scene_traces:
                raise RuntimeError("V73 scene cross-attention trace is missing")
            minimum = min(trace.observed_minimum_weight for trace in scene_traces)
            required = self.uniform_floor_mass / float(self.scene_memory_token_count)
            self._last_audit = FullSceneControlAuditV73(
                scene_prefix_tokens=self.expected_environment_latents + 2,
                environment_latent_count=self.expected_environment_latents,
                scene_memory_tokens=self.scene_memory_token_count,
                control_token_count=self.control_token_count,
                hidden_size=self.hidden_size,
                model_dimension=self.model_dimension,
                scene_encoder_layers=self.scene_encoder_layer_count,
                scene_cross_attention_layers=self.scene_cross_attention_layer_count,
                internal_reader_slots=self.internal_reader_slots,
                uniform_floor_mass=self.uniform_floor_mass,
                minimum_cross_attention_weight=minimum,
                required_cross_attention_weight=required,
                every_environment_latent_processed=(
                    not self.dct_scene_bottleneck_used
                    and all(trace.all_memory_tokens_receive_positive_weight for trace in scene_traces)
                ),
                full_prefix_retained_separately_for_language_model=True,
                question_conditioned_continuous_attention=True,
                question_dependent_retrieval=False,
                latent_selection_or_top_k_used=False,
                environmental_text_inputs=0,
                dct_scene_bottleneck_used=self.dct_scene_bottleneck_used,
                question_only_output_path_exists=False,
                output_computed_only_from_scene_value_contexts=True,
                zero_scene_produces_exact_zero_controls=True,
                maximum_control_rms=float(rms.detach().max().cpu()),
            )
        else:
            self._last_audit = None
        return FullSceneControlOutputV73(output, coefficients, rms)

    def forward(
        self,
        scene_prefix: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
        *,
        return_traces: bool = False,
    ) -> FullSceneControlOutputV73:
        memory, _scene_traces = self.encode_scene(
            scene_prefix, return_traces=return_traces
        )
        return self.forward_from_scene_memory(
            memory,
            question_embeddings,
            question_attention_mask,
            return_traces=return_traces,
        )

    def audit(self) -> FullSceneControlAuditV73:
        if self._last_audit is None:
            raise RuntimeError("V73 audit requires a traced forward pass")
        return self._last_audit


class DCT40QuestionControlBaselineV73(FullSceneSetAttentionQuestionControlV73):
    """Same downstream reader after V71's duplicated 8+32 DCT bottleneck."""

    dct_scene_bottleneck_used = True
    dct_moment_counts = (8, 32)

    @property
    def scene_memory_token_count(self) -> int:
        return sum(self.dct_moment_counts)

    @staticmethod
    def _dct_moments(environment: torch.Tensor, count: int) -> torch.Tensor:
        latent_count = environment.shape[1]
        positions = (
            torch.arange(latent_count, device=environment.device, dtype=torch.float32)
            + 0.5
        ) / float(latent_count)
        frequencies = torch.arange(count, device=environment.device, dtype=torch.float32)
        weights = torch.cos(math.pi * frequencies[:, None] * positions[None, :])
        weights[0].fill_(1.0)
        weights = weights / weights.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(
            1e-8
        )
        return F.layer_norm(
            torch.einsum("ml,blh->bmh", weights, environment),
            (environment.shape[-1],),
        )

    def _scene_source(self, environment: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self._dct_moments(environment, count) for count in self.dct_moment_counts],
            dim=1,
        )


__all__ = [
    "DCT40QuestionControlBaselineV73",
    "FullSceneControlAuditV73",
    "FullSceneControlOutputV73",
    "FullSceneSetAttentionQuestionControlV73",
    "PositiveFloorAttentionTraceV73",
    "PositiveFloorMultiheadAttentionV73",
]

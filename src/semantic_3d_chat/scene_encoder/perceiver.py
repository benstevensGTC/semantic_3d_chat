from __future__ import annotations

import math

import torch
from torch import nn

SCENE_ENCODER_ARCHITECTURE_VERSION = "signal_preserving_resampler_v3"


def _halton_value(index: int, base: int) -> float:
    """Return one deterministic Halton coordinate in the open unit interval."""

    result = 0.0
    fraction = 1.0 / base
    value = index
    while value:
        result += fraction * (value % base)
        value //= base
        fraction /= base
    return result


def spatial_anchors(count: int) -> torch.Tensor:
    """Deterministic low-discrepancy latent anchors in normalized room XYZ."""

    if count < 1:
        raise ValueError("count must be positive")
    values = [
        [_halton_value(index, base) * 2.0 - 1.0 for base in (2, 3, 5)]
        for index in range(1, count + 1)
    ]
    return torch.tensor(values, dtype=torch.float32)


def spatial_coverage_weights(
    positions: torch.Tensor,
    anchors: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Globally normalize distance weights without top-k or radius pruning.

    Both the hierarchical learned resampler and the native language-aligned
    bypass use this exact function. Positions and anchors are normalized room
    coordinates in ``[-1, 1]``. Float32 keeps even distant room contributions
    representable at the validated temperature.
    """

    if temperature <= 0:
        raise ValueError("coverage temperature must be positive")
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[1] == 0:
        raise ValueError("positions must be nonempty [T,3] or [B,T,3]")
    if anchors.ndim != 2 or anchors.shape[-1] != 3 or anchors.shape[0] == 0:
        raise ValueError("anchors must be nonempty [L,3]")
    anchor_values = anchors.to(device=positions.device, dtype=torch.float32)
    position_values = positions.to(dtype=torch.float32)
    squared_distance = (
        (anchor_values.view(1, anchors.shape[0], 1, 3) - position_values.unsqueeze(1))
        .square()
        .sum(dim=-1)
    )
    return torch.softmax(-squared_distance / float(temperature), dim=-1)


def _fixed_query_identities(count: int, model_dim: int) -> torch.Tensor:
    """Build stable, well-separated identities without changing the global RNG."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x5CE3E)
    if count <= model_dim:
        # QR gives exactly orthogonal row identities in the common 256 x 384 case.
        sample = torch.randn(model_dim, count, generator=generator)
        identities = torch.linalg.qr(sample, mode="reduced").Q.transpose(0, 1)
    else:
        identities = torch.randn(count, model_dim, generator=generator)
        identities = nn.functional.normalize(identities, dim=-1)
    return identities.contiguous()


def _fixed_isometric_projection(input_dim: int, output_dim: int) -> torch.Tensor:
    """Return a deterministic map that preserves norms when output_dim >= input_dim."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x51A1)
    if output_dim >= input_dim:
        sample = torch.randn(output_dim, input_dim, generator=generator)
        return torch.linalg.qr(sample, mode="reduced").Q.transpose(0, 1).contiguous()
    sample = torch.randn(input_dim, output_dim, generator=generator)
    return torch.linalg.qr(sample, mode="reduced").Q.contiguous()


class PerceiverLayer(nn.Module):
    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.latent_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(model_dim)
        self.ff_norm = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        query_identity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Reintroducing a fixed identity in the query path prevents a shared scene
        # mean from overwhelming otherwise distinct learned latent parameters.
        query = latents if query_identity is None else latents + query_identity
        normalized_context = self.context_norm(context)
        cross, _ = self.cross_attention(
            self.latent_norm(query),
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        latents = latents + cross
        normalized = self.self_norm(latents)
        self_attended, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        latents = latents + self_attended
        return latents + self.ff(self.ff_norm(latents))


class GlobalSceneResampler(nn.Module):
    """Question-independent, spatially covered latents attending to every block.

    Learned cross-attention remains global.  A second deterministic soft spatial
    coverage path gives every latent a different room anchor.  Its softmax has no
    top-k or hard radius, so every occupied block contributes to every latent while
    nearby blocks have more influence.  Fixed query identities and an isometric
    output bypass make collapse into a scene-independent prompt structurally
    difficult. Fixed identities are used only as slot/query positions; they are
    deliberately excluded from the final payload so diversity metrics and pair
    losses measure scene content rather than a constant positional scaffold.
    """

    def __init__(
        self,
        model_dim: int,
        num_latents: int,
        heads: int = 8,
        layers: int = 2,
        *,
        coverage_temperature: float = 0.20,
        coverage_scale: float = 4.0,
        query_identity_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if coverage_temperature <= 0:
            raise ValueError("coverage_temperature must be positive")
        if coverage_scale <= 0:
            raise ValueError("coverage_scale must be positive")
        if query_identity_scale < 0:
            raise ValueError("query_identity_scale cannot be negative")
        self.num_latents = int(num_latents)
        self.model_dim = int(model_dim)
        self.coverage_temperature = float(coverage_temperature)
        self.coverage_scale = float(coverage_scale)
        self.query_identity_scale = float(query_identity_scale)
        self.learned_queries = nn.Parameter(torch.randn(num_latents, model_dim) * 0.02)
        self.layers = nn.ModuleList(PerceiverLayer(model_dim, heads) for _ in range(layers))
        self.final_norm = nn.LayerNorm(model_dim)
        self.coverage_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.register_buffer("latent_anchors", spatial_anchors(num_latents), persistent=False)
        self.register_buffer(
            "query_identities",
            _fixed_query_identities(num_latents, model_dim),
            persistent=False,
        )

    def _coverage_context(self, context: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
        if positions.shape != (*context.shape[:2], 3):
            raise ValueError(
                "block_positions must have shape [T,3] or [B,T,3] matching block_tokens"
            )
        weights = spatial_coverage_weights(
            positions.to(device=context.device),
            self.latent_anchors,
            self.coverage_temperature,
        )
        normalized_context = self.coverage_norm(context.float())
        return torch.matmul(weights, normalized_context).to(context.dtype)

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if block_tokens.ndim == 2:
            block_tokens = block_tokens.unsqueeze(0)
        if block_tokens.ndim != 3 or block_tokens.shape[1] == 0:
            raise ValueError("Expected nonempty block tokens [B,T,D]")
        if block_positions is None:
            raise ValueError(
                f"{SCENE_ENCODER_ARCHITECTURE_VERSION} requires normalized block positions"
            )
        batch = block_tokens.shape[0]
        identity = self.query_identities.to(block_tokens).unsqueeze(0).expand(batch, -1, -1)
        identity = identity * (math.sqrt(self.model_dim) * self.query_identity_scale)
        coverage = self._coverage_context(block_tokens, block_positions)
        latents = self.learned_queries.unsqueeze(0).expand(batch, -1, -1) + coverage
        for layer in self.layers:
            latents = layer(latents, block_tokens, identity)
        # Coverage is added again after globally mixing layers so localized scene
        # changes cannot be averaged away by the shared attention context.
        return self.final_norm(latents) + self.coverage_scale * coverage


class SignalPreservingProjection(nn.Module):
    """Trainable LM projection with a small, fixed injective residual path."""

    def __init__(self, model_dim: int, language_hidden_dim: int, skip_scale: float = 1.0) -> None:
        super().__init__()
        if skip_scale <= 0:
            raise ValueError("skip_scale must be positive")
        self.skip_scale = float(skip_scale)
        self.trainable = nn.Sequential(
            nn.Linear(model_dim, language_hidden_dim),
            nn.GELU(),
            nn.Linear(language_hidden_dim, language_hidden_dim),
            nn.LayerNorm(language_hidden_dim),
        )
        self.register_buffer(
            "fixed_projection",
            _fixed_isometric_projection(model_dim, language_hidden_dim),
            persistent=False,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        learned = self.trainable(values)
        bypass = torch.matmul(values.float(), self.fixed_projection.float()).to(learned.dtype)
        return learned + self.skip_scale * bypass

from __future__ import annotations

import torch
from torch import nn

from .point_tokens import FourierXYZ


class SpatialBlockEncoder(nn.Module):
    """Attention-pool every occupied spatial block into multiple tokens.

    The implementation deliberately has no question input and no top-k path. Every
    voxel appears as a key/value in exactly one occupied block.
    """

    def __init__(
        self,
        model_dim: int,
        block_size_m: float,
        tokens_per_block: int = 2,
        heads: int = 8,
        fourier_bands: int = 8,
        block_chunk_size: int = 256,
        content_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if tokens_per_block < 2:
            raise ValueError(
                "Use at least two tokens per block to avoid a single-average bottleneck"
            )
        self.block_size_m = float(block_size_m)
        self.tokens_per_block = int(tokens_per_block)
        self.block_chunk_size = int(block_chunk_size)
        self.content_residual_scale = float(content_residual_scale)
        if self.block_chunk_size < 1:
            raise ValueError("block_chunk_size must be positive")
        if self.content_residual_scale < 0:
            raise ValueError("content_residual_scale cannot be negative")
        self.queries = nn.Parameter(torch.randn(tokens_per_block, model_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.query_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.fourier = FourierXYZ(fourier_bands)
        self.position_projection = nn.Sequential(
            nn.Linear(3 + self.fourier.output_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        point_tokens: torch.Tensor,
        xyz_m: torch.Tensor,
        room_min: torch.Tensor,
        room_max: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if point_tokens.ndim != 2 or xyz_m.shape != (point_tokens.shape[0], 3):
            raise ValueError("Expected point_tokens [N,D] and xyz_m [N,3]")
        if point_tokens.shape[0] == 0:
            raise ValueError("Cannot encode an empty scene")
        # Block membership is fixed geometry, so grouping on CPU avoids backend gaps
        # in MPS `unique(dim=...)` without disconnecting semantic gradients.
        integer_blocks_cpu = torch.floor(
            (xyz_m.detach().cpu() - room_min.detach().cpu()) / self.block_size_m
        ).to(torch.int64)
        unique_blocks_cpu, inverse_cpu, counts_cpu = torch.unique(
            integer_blocks_cpu, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        unique_blocks = unique_blocks_cpu.to(xyz_m.device)
        inverse = inverse_cpu.to(xyz_m.device)
        counts = counts_cpu.to(xyz_m.device)
        extent = (room_max - room_min).clamp_min(1e-6)
        block_outputs: list[torch.Tensor] = []
        block_position_outputs: list[torch.Tensor] = []
        for chunk_start in range(0, unique_blocks.shape[0], self.block_chunk_size):
            chunk_stop = min(chunk_start + self.block_chunk_size, unique_blocks.shape[0])
            chunk_counts = counts_cpu[chunk_start:chunk_stop]
            block_count = int(chunk_stop - chunk_start)
            max_points = int(chunk_counts.max().item())
            members = point_tokens.new_zeros(block_count, max_points, point_tokens.shape[-1])
            padding_mask = torch.ones(
                block_count, max_points, dtype=torch.bool, device=point_tokens.device
            )
            for local_index, block_index in enumerate(range(chunk_start, chunk_stop)):
                point_indices = torch.nonzero(inverse_cpu == block_index, as_tuple=False).flatten()
                point_indices = point_indices.to(point_tokens.device)
                member_count = point_indices.numel()
                members[local_index, :member_count] = point_tokens.index_select(0, point_indices)
                padding_mask[local_index, :member_count] = False
            queries = self.queries.unsqueeze(0).expand(block_count, -1, -1)
            attended, _ = self.cross_attention(
                self.query_norm(queries),
                self.context_norm(members),
                self.context_norm(members),
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            # An untrained attention module can converge to a nearly constant
            # block prompt.  The deterministic mean is therefore retained as a
            # residual: every voxel in the occupied block affects both block
            # tokens even if learned attention becomes insensitive.
            member_counts = (~padding_mask).sum(dim=1).clamp_min(1).to(members.dtype)
            content_residual = members.sum(dim=1) / member_counts.unsqueeze(-1)
            center_m = (
                room_min + (unique_blocks[chunk_start:chunk_stop].float() + 0.5) * self.block_size_m
            )
            center_norm = ((center_m - room_min) / extent).mul(2).sub(1)
            position = torch.cat((center_norm, self.fourier(center_norm)), dim=-1)
            attended = (
                attended
                + queries
                + self.position_projection(position).unsqueeze(1)
                + self.content_residual_scale * content_residual.unsqueeze(1)
            )
            block_outputs.append(self.output_norm(attended).flatten(0, 1))
            block_position_outputs.append(
                center_norm.repeat_interleave(self.tokens_per_block, dim=0)
            )
        tokens = torch.cat(block_outputs, dim=0)
        token_positions = torch.cat(block_position_outputs, dim=0)
        audit = {
            "block_indices": unique_blocks,
            "voxel_counts": counts,
            "voxel_to_block": inverse,
            "processed_voxels": counts.sum(),
            "block_token_positions_normalized": token_positions,
        }
        return tokens, audit

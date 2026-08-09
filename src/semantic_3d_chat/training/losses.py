from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from semantic_3d_chat.scene_encoder.perceiver import spatial_anchors


class QuestionGroundingHead(nn.Module):
    """Ground a user question by attending to spatially anchored scene latents.

    The earlier proof-of-concept averaged all scene latents and the entire
    system+user prompt.  That erased both spatial slots and most question
    variation.  This head instead uses the user-question embedding as a query,
    predicts a distribution over the fixed scene anchors, and adds a bounded
    learned residual to the anchor-weighted coordinate.
    """

    def __init__(
        self,
        scene_dim: int,
        language_dim: int,
        latent_count: int,
        hidden_dim: int = 384,
    ) -> None:
        super().__init__()
        if latent_count < 1:
            raise ValueError("latent_count must be positive")
        self.scene_norm = nn.LayerNorm(scene_dim)
        self.question_norm = nn.LayerNorm(language_dim)
        self.question_projection = nn.Linear(language_dim, scene_dim)
        self.key_projection = nn.Linear(scene_dim, scene_dim)
        self.value_projection = nn.Linear(scene_dim, hidden_dim)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim + scene_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.register_buffer(
            "anchors_normalized",
            spatial_anchors(latent_count),
            persistent=False,
        )

    def forward_with_attention(
        self, scene_latents: torch.Tensor, question_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if scene_latents.ndim != 3:
            raise ValueError("scene_latents must have shape [B,L,D]")
        if scene_latents.shape[1] != self.anchors_normalized.shape[0]:
            raise ValueError(
                "scene latent count does not match grounding anchors: "
                f"{scene_latents.shape[1]} != {self.anchors_normalized.shape[0]}"
            )
        question_summary = (
            question_embeddings.mean(dim=1)
            if question_embeddings.ndim == 3
            else question_embeddings
        )
        if question_summary.ndim != 2:
            raise ValueError("question_embeddings must be [B,H] or [B,T,H]")
        if question_summary.shape[0] != scene_latents.shape[0]:
            raise ValueError("question and scene batches must have the same size")
        query = self.question_projection(self.question_norm(question_summary.float()))
        keys = self.key_projection(self.scene_norm(scene_latents.float()))
        normalized_query = F.normalize(query, dim=-1, eps=1e-6)
        normalized_keys = F.normalize(keys, dim=-1, eps=1e-6)
        scale = self.logit_scale.float().exp().clamp(max=100.0)
        logits = torch.einsum("bd,bld->bl", normalized_query, normalized_keys) * scale
        weights = torch.softmax(logits, dim=-1)
        values = self.value_projection(self.scene_norm(scene_latents.float()))
        attended = torch.einsum("bl,blh->bh", weights, values)
        anchors = self.anchors_normalized.to(device=weights.device, dtype=weights.dtype)
        anchor_xyz = torch.matmul(weights, anchors)
        offset = self.residual(torch.cat((attended, query), dim=-1)) * 0.25
        predicted = (anchor_xyz + offset).clamp(-1.0, 1.0)
        return predicted, logits, weights

    def forward(
        self, scene_latents: torch.Tensor, question_embeddings: torch.Tensor
    ) -> torch.Tensor:
        predicted, _, _ = self.forward_with_attention(scene_latents, question_embeddings)
        return predicted

    def nearest_anchor_targets(self, normalized_xyz: torch.Tensor) -> torch.Tensor:
        """Return the closest fixed anchor index for coordinate supervision."""

        if normalized_xyz.ndim != 2 or normalized_xyz.shape[-1] != 3:
            raise ValueError("normalized_xyz must have shape [B,3]")
        anchors = self.anchors_normalized.to(
            device=normalized_xyz.device, dtype=normalized_xyz.dtype
        )
        distances = (normalized_xyz.unsqueeze(1) - anchors.unsqueeze(0)).square().sum(dim=-1)
        return distances.argmin(dim=-1)


def normalize_xyz(
    target_xyz: torch.Tensor, room_min: torch.Tensor, room_max: torch.Tensor
) -> torch.Tensor:
    return ((target_xyz - room_min) / (room_max - room_min).clamp_min(1e-6)).mul(2).sub(1)


def nearest_spatial_anchor_indices(
    target_xyz: torch.Tensor,
    room_min: torch.Tensor,
    room_max: torch.Tensor,
    latent_count: int,
) -> torch.Tensor:
    """Map metric targets to their nearest fixed full-scene latent anchors.

    This helper is deliberately independent of the learned grounding head.  It
    uses the same deterministic Halton anchors as the global scene resampler,
    allowing training-only oracle coordinates to supervise a stable *final*
    scene-token slot without introducing any inference-time metadata path.
    """

    if target_xyz.ndim != 2 or target_xyz.shape[-1] != 3:
        raise ValueError("target_xyz must have shape [B,3]")
    if room_min.shape not in {(3,), target_xyz.shape}:
        raise ValueError("room_min must have shape [3] or [B,3]")
    if room_max.shape not in {(3,), target_xyz.shape}:
        raise ValueError("room_max must have shape [3] or [B,3]")
    normalized = normalize_xyz(target_xyz.float(), room_min.float(), room_max.float())
    anchors = spatial_anchors(latent_count).to(device=normalized.device, dtype=normalized.dtype)
    distances = (normalized.unsqueeze(1) - anchors.unsqueeze(0)).square().sum(dim=-1)
    return distances.argmin(dim=-1)


def spatial_scene_answer_contrastive_loss(
    scene_tokens: torch.Tensor,
    anchor_indices: torch.Tensor,
    own_answer_embeddings: torch.Tensor,
    alternate_answer_embeddings: torch.Tensor,
    *,
    margin: float = 0.2,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Align a target-localized final scene token with its paired answer.

    ``scene_tokens`` are the final LM-dimensional tokens, after all learned and
    native-aligned scene-token mixing.  Answer embeddings are treated as frozen
    targets.  For each eligible side of a counterfactual unit, the hinge asks
    the selected scene token to prefer its own answer embedding over the other
    side's answer embedding by ``margin``.
    """

    if scene_tokens.ndim != 3:
        raise ValueError("scene_tokens must have shape [B,L,H]")
    if scene_tokens.shape[0] < 1 or scene_tokens.shape[1] < 1 or scene_tokens.shape[2] < 1:
        raise ValueError("scene_tokens cannot be empty")
    if anchor_indices.shape != (scene_tokens.shape[0],):
        raise ValueError("anchor_indices must have shape [B]")
    expected_embedding_shape = (scene_tokens.shape[0], scene_tokens.shape[2])
    if own_answer_embeddings.shape != expected_embedding_shape:
        raise ValueError("own_answer_embeddings must match scene batch and hidden dimension")
    if alternate_answer_embeddings.shape != expected_embedding_shape:
        raise ValueError("alternate_answer_embeddings must match scene batch and hidden dimension")
    if anchor_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("anchor_indices must use an integer dtype")
    if torch.any(anchor_indices < 0) or torch.any(anchor_indices >= scene_tokens.shape[1]):
        raise ValueError("anchor index is outside the scene-token range")
    if not 0.0 <= margin <= 2.0:
        raise ValueError("margin must be in [0, 2]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    rows = torch.arange(scene_tokens.shape[0], device=scene_tokens.device)
    selected = scene_tokens[rows, anchor_indices.to(device=scene_tokens.device)].float()
    selected = F.normalize(selected, dim=-1, eps=epsilon)
    # Frozen targets must never update the LM embedding table, even if a caller
    # accidentally passes tensors whose source parameters require gradients.
    own = F.normalize(own_answer_embeddings.detach().float(), dim=-1, eps=epsilon)
    alternate = F.normalize(alternate_answer_embeddings.detach().float(), dim=-1, eps=epsilon)
    own_similarity = (selected * own).sum(dim=-1)
    alternate_similarity = (selected * alternate).sum(dim=-1)
    achieved_margin = own_similarity - alternate_similarity
    loss = F.relu(float(margin) - achieved_margin).mean()
    return loss, {
        "eligible_side_count": int(scene_tokens.shape[0]),
        "own_similarity": own_similarity.detach(),
        "alternate_similarity": alternate_similarity.detach(),
        "achieved_margin": achieved_margin.detach(),
        "configured_margin": torch.tensor(
            float(margin), device=scene_tokens.device, dtype=torch.float32
        ),
        "anchor_indices": anchor_indices.detach(),
    }


def ordered_spatial_relation_contrastive_loss(
    scene_tokens: torch.Tensor,
    normalized_target_xyz: torch.Tensor,
    normalized_reference_xyz: torch.Tensor,
    own_answer_embeddings: torch.Tensor,
    alternate_answer_embeddings: torch.Tensor,
    *,
    temperature: float = 0.2,
    margin: float = 0.2,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Align an ordered, spatially pooled scene relation with its answer direction.

    Every final LM-dimensional scene token participates in two deterministic soft
    pools: one centered on the normalized target coordinate and one on the
    normalized reference coordinate.  The target-minus-reference pooled scene
    vector is aligned with the frozen own-minus-alternate answer direction using
    a cosine hinge.  Fixed Halton anchors provide the question-independent spatial
    identity of each final scene-token slot.

    The answer embeddings are detached deliberately.  This loss therefore updates
    the continuous scene path, not the language model's answer embedding table.
    """

    if scene_tokens.ndim != 3:
        raise ValueError("scene_tokens must have shape [B,L,H]")
    batch_size, latent_count, hidden_size = scene_tokens.shape
    if batch_size < 1 or latent_count < 1 or hidden_size < 1:
        raise ValueError("scene_tokens cannot be empty")
    expected_xyz_shape = (batch_size, 3)
    if normalized_target_xyz.shape != expected_xyz_shape:
        raise ValueError("normalized_target_xyz must have shape [B,3]")
    if normalized_reference_xyz.shape != expected_xyz_shape:
        raise ValueError("normalized_reference_xyz must have shape [B,3]")
    expected_embedding_shape = (batch_size, hidden_size)
    if own_answer_embeddings.shape != expected_embedding_shape:
        raise ValueError("own_answer_embeddings must have shape [B,H]")
    if alternate_answer_embeddings.shape != expected_embedding_shape:
        raise ValueError("alternate_answer_embeddings must have shape [B,H]")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(margin)) or not 0.0 <= margin <= 1.0:
        raise ValueError("margin must be finite and in [0, 1]")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    device = scene_tokens.device
    target_xyz = normalized_target_xyz.to(device=device, dtype=torch.float32)
    reference_xyz = normalized_reference_xyz.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(target_xyz).all()):
        raise ValueError("normalized_target_xyz must contain only finite values")
    if not bool(torch.isfinite(reference_xyz).all()):
        raise ValueError("normalized_reference_xyz must contain only finite values")

    anchors = spatial_anchors(latent_count).to(device=device, dtype=torch.float32)

    def soft_pool_weights(coordinates: torch.Tensor) -> torch.Tensor:
        squared_distance = (anchors.unsqueeze(0) - coordinates.unsqueeze(1)).square().sum(-1)
        weights = torch.softmax(-squared_distance / float(temperature), dim=-1)
        # Extremely small temperatures can underflow distant softmax entries to
        # exactly zero in float32.  A positive floor preserves the stated dense
        # all-token contract without assigning a practically meaningful hard
        # retrieval cutoff, then renormalization keeps each row a distribution.
        weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
        return weights / weights.sum(dim=-1, keepdim=True)

    target_weights = soft_pool_weights(target_xyz)
    reference_weights = soft_pool_weights(reference_xyz)
    tokens = scene_tokens.float()
    target_pool = torch.einsum("bl,blh->bh", target_weights, tokens)
    reference_pool = torch.einsum("bl,blh->bh", reference_weights, tokens)
    relation = target_pool - reference_pool

    # Normalize each frozen answer first so embedding-table norm cannot dominate
    # the semantic own-minus-alternate direction.
    own = F.normalize(
        own_answer_embeddings.detach().to(device=device, dtype=torch.float32),
        dim=-1,
        eps=epsilon,
    )
    alternate = F.normalize(
        alternate_answer_embeddings.detach().to(device=device, dtype=torch.float32),
        dim=-1,
        eps=epsilon,
    )
    raw_answer_direction = own - alternate
    raw_answer_direction_norm = raw_answer_direction.norm(dim=-1)
    if bool((raw_answer_direction_norm <= epsilon).any()):
        raise ValueError("own and alternate answer embeddings must define a nonzero direction")
    answer_direction = F.normalize(raw_answer_direction, dim=-1, eps=epsilon)

    relation_norm = relation.norm(dim=-1)
    cosine = F.cosine_similarity(relation, answer_direction, dim=-1, eps=epsilon)
    hinge_shortfall = F.relu(float(margin) - cosine)
    loss = hinge_shortfall.mean()

    return loss, {
        "eligible_side_count": int(batch_size),
        "latent_count": int(latent_count),
        "target_weights": target_weights.detach(),
        "reference_weights": reference_weights.detach(),
        "minimum_target_weight": target_weights.min().detach(),
        "minimum_reference_weight": reference_weights.min().detach(),
        "achieved_margin": cosine.detach(),
        "relation_answer_cosine": cosine.detach(),
        "hinge_shortfall": hinge_shortfall.detach(),
        "relation_norm": relation_norm.detach(),
        "answer_direction_norm_before_normalization": raw_answer_direction_norm.detach(),
        "configured_temperature": torch.tensor(
            float(temperature), device=device, dtype=torch.float32
        ),
        "configured_margin": torch.tensor(float(margin), device=device, dtype=torch.float32),
    }


def denormalize_xyz(
    target_normalized: torch.Tensor, room_min: torch.Tensor, room_max: torch.Tensor
) -> torch.Tensor:
    return target_normalized.add(1).div(2).mul(room_max - room_min).add(room_min)


def _deterministic_latent_indices(
    latent_count: int,
    max_latents: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Select a deterministic, spatially even subset of latent indices."""

    if latent_count < 1:
        raise ValueError("latent_count must be positive")
    if max_latents is None or max_latents >= latent_count:
        return torch.arange(latent_count, device=device)
    if max_latents < 2:
        raise ValueError("max_latents must be at least 2")
    # Integer arithmetic includes both ends and avoids device-dependent random
    # sampling. With max_latents <= latent_count these indices are unique.
    numerator = torch.arange(max_latents, device=device) * (latent_count - 1)
    return torch.div(numerator, max_latents - 1, rounding_mode="floor")


def latent_diversity_loss(
    latents: torch.Tensor,
    *,
    cosine_margin: float = 0.2,
    max_latents: int | None = 128,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Penalize excessive within-scene latent cosine similarity.

    The loss is computed independently within each batch item, never between
    scenes. It is zero for off-diagonal cosine values at or below the margin and
    grows quadratically above it. Computation stays in float32 for stable
    normalization even when the scene encoder uses float16.
    """

    if latents.ndim != 3:
        raise ValueError("latents must have shape [batch, latent_count, dimension]")
    if latents.shape[-1] < 1:
        raise ValueError("latent dimension must be positive")
    if not -1.0 <= cosine_margin < 1.0:
        raise ValueError("cosine_margin must be in [-1, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    indices = _deterministic_latent_indices(latents.shape[1], max_latents, latents.device)
    selected = latents.index_select(1, indices).float()
    if selected.shape[1] < 2:
        zero = selected.sum() * 0.0
        return zero, {
            "sampled_latent_count": int(selected.shape[1]),
            "mean_off_diagonal_cosine": zero.detach(),
            "mean_absolute_off_diagonal_cosine": zero.detach(),
            "max_off_diagonal_cosine": zero.detach(),
            "fraction_above_margin": zero.detach(),
        }
    normalized = F.normalize(selected, dim=-1, eps=epsilon)
    cosine = normalized @ normalized.transpose(-2, -1)
    diagonal = torch.eye(selected.shape[1], dtype=torch.bool, device=selected.device).unsqueeze(0)
    off_diagonal = cosine.masked_select(~diagonal.expand_as(cosine))
    excess = F.relu(off_diagonal - cosine_margin)
    loss = excess.square().mean()
    diagnostics: dict[str, torch.Tensor | int] = {
        "sampled_latent_count": int(selected.shape[1]),
        "mean_off_diagonal_cosine": off_diagonal.mean().detach(),
        "mean_absolute_off_diagonal_cosine": off_diagonal.abs().mean().detach(),
        "max_off_diagonal_cosine": off_diagonal.max().detach(),
        "fraction_above_margin": (off_diagonal > cosine_margin).float().mean().detach(),
    }
    return loss, diagnostics


def paired_scene_separation_loss(
    first_latents: torch.Tensor,
    second_latents: torch.Tensor,
    *,
    cosine_distance_margin: float = 0.05,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Require paired training scenes to produce measurably different latents.

    Global latent slots have consistent learned-query identities, so the
    objective averages their aligned cosine similarities. It uses no question,
    answer, changed attribute, or oracle label: only the fact that two *training*
    scenes form a counterfactual pair. A hinge avoids pushing already-separated
    representations farther apart without evidence.
    """

    if first_latents.ndim != 3 or second_latents.ndim != 3:
        raise ValueError("paired latents must have shape [batch, latent_count, dimension]")
    if first_latents.shape != second_latents.shape:
        raise ValueError("paired latent tensors must have identical shapes")
    if first_latents.shape[1] < 1 or first_latents.shape[2] < 1:
        raise ValueError("paired latent tensors cannot be empty")
    if not 0.0 <= cosine_distance_margin <= 2.0:
        raise ValueError("cosine_distance_margin must be in [0, 2]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    first = F.normalize(first_latents.float(), dim=-1, eps=epsilon)
    second = F.normalize(second_latents.float(), dim=-1, eps=epsilon)
    aligned_cosine = (first * second).sum(dim=-1)
    mean_cosine = aligned_cosine.mean()
    cosine_distance = 1.0 - mean_cosine
    shortfall = F.relu(cosine_distance_margin - cosine_distance)
    loss = shortfall.square()
    return loss, {
        "mean_aligned_cosine": mean_cosine.detach(),
        "cosine_distance": cosine_distance.detach(),
        "fraction_slots_within_margin": ((1.0 - aligned_cosine) < cosine_distance_margin)
        .float()
        .mean()
        .detach(),
    }

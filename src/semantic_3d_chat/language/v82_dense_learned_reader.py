"""Small learned reader over one immutable V81 738-token scene memory.

V82 never selects or retrieves environmental tokens.  The atlas branch gives
all 96 groups a strict positive attention floor and the base branch gives all
256 base latents a strict positive floor.  Four direct atlas value reads and a
small positive base mixture form the control tokens before a bias-free learned
residual.  Consequently all 384 atlas values and all 256 base latents
participate in every read, while an all-zero environmental payload produces
exactly zero controls.

The module owns no tokenizer, scene compiler, oracle loader, or language model.
Question inputs are detached 1536D numeric embeddings and scene inputs are the
already sealed, question-independent V81 memories.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import nn

from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    ATLAS_UNIFORM_FLOOR_MASS,
    BASE_ENVIRONMENT_LATENTS,
    BASE_UNIFORM_FLOOR_MASS,
    HIDDEN_SIZE,
    MAXIMUM_CONTROL_RMS,
    PROBE_COUNT,
    RAW_ATLAS_LOGIT_SCALE,
    VALUES_PER_PROBE,
    V81PrefixBinding,
    assert_prefix_binding_v81,
    dense_floor_attention,
    split_v75_v2_prefix_v81,
)

ARTIFACT: Final[str] = "gemma4_v82_strict_dense_learned_reader_v1"
ARCHITECTURE: Final[str] = "positive_floor_dual_bank_reader_v82"
INTERNAL_DIMENSION: Final[int] = 64
BASE_DIRECT_MIX: Final[float] = 0.01
ATLAS_DIRECT_MIX: Final[float] = 1.0 - BASE_DIRECT_MIX

CANDIDATE_TENSOR_NAMES: Final[frozenset[str]] = frozenset(
    {
        "atlas_residual_scale",
        "base_logit_scale",
        "question_projection.weight",
        "atlas_key_projection.weight",
        "base_key_projection.weight",
        "control_value_projection.weight",
        "base_value_projection.weight",
        "residual_output.weight",
    }
)
TRAINABLE_PARAMETER_COUNT: Final[int] = (
    5 * HIDDEN_SIZE * INTERNAL_DIMENSION
    + HIDDEN_SIZE * (2 * INTERNAL_DIMENSION)
    + 2
)


@dataclass(frozen=True)
class V82ReaderOutput:
    """Controls and structural evidence from one complete dense read."""

    controls: torch.Tensor
    atlas_weights: torch.Tensor
    base_weights: torch.Tensor
    atlas_contexts: torch.Tensor
    base_context: torch.Tensor
    residual: torch.Tensor
    control_rms: torch.Tensor
    atlas_attention_sums: torch.Tensor
    base_attention_sums: torch.Tensor
    zero_environmental_payload: bool
    all_96_groups_positive: bool
    all_384_atlas_values_positive: bool
    all_256_base_latents_positive: bool


@dataclass(frozen=True)
class V82ReaderAudit:
    schema_version: int
    artifact: str
    architecture: str
    fixed_memory_tokens: int
    hidden_size: int
    probe_count: int
    values_per_probe: int
    base_environment_latents: int
    internal_dimension: int
    trainable_parameter_count: int
    atlas_uniform_floor_mass: float
    base_uniform_floor_mass: float
    atlas_minimum_weight: float
    base_minimum_weight: float
    atlas_direct_mix: float
    base_direct_mix: float
    every_atlas_value_participates: bool
    every_base_latent_participates: bool
    question_dependent_retrieval: bool
    semantic_or_spatial_top_k_selection: bool
    environmental_text_inputs: tuple[str, ...]
    zero_environment_produces_exact_zero_controls: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["environmental_text_inputs"] = list(self.environmental_text_inputs)
        return value


def _validate_query(query: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if (
        not isinstance(query, torch.Tensor)
        or query.ndim != 2
        or tuple(query.shape) != (batch_size, HIDDEN_SIZE)
        or not bool(torch.isfinite(query).all())
    ):
        raise ValueError("V82 detached question query must be finite [B,1536]")
    if bool(torch.any(query.detach().float().norm(dim=-1) <= 1e-8)):
        raise ValueError("V82 detached question query cannot contain a zero row")
    return query.detach().float()


def wrong_scene_contrast_loss_v82(
    prediction: torch.Tensor,
    own_target: torch.Tensor,
    paired_wrong_target: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hinge requiring a prediction to resemble its own scene more than its pair."""

    if (
        prediction.ndim != 3
        or own_target.shape != prediction.shape
        or paired_wrong_target.shape != prediction.shape
        or prediction.shape[1:] != (VALUES_PER_PROBE, HIDDEN_SIZE)
    ):
        raise ValueError("V82 contrast tensors must match [B,4,1536]")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("V82 contrast margin must be finite and nonnegative")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (prediction, own_target, paired_wrong_target)
    ):
        raise ValueError("V82 contrast tensors contain NaN or infinity")
    predicted = F.normalize(prediction.float().flatten(1), dim=-1, eps=1e-8)
    own = F.normalize(own_target.detach().float().flatten(1), dim=-1, eps=1e-8)
    wrong = F.normalize(
        paired_wrong_target.detach().float().flatten(1), dim=-1, eps=1e-8
    )
    preference = (predicted * own).sum(dim=-1) - (predicted * wrong).sum(dim=-1)
    return F.relu(float(margin) - preference).mean(), preference


class DenseLearnedSceneReaderV82(nn.Module):
    """Bias-free 688k-parameter learned dual-bank reader."""

    def __init__(self, *, initialization_seed: int = 820082) -> None:
        super().__init__()
        self.question_projection = nn.Linear(
            HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False
        )
        self.atlas_key_projection = nn.Linear(
            HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False
        )
        self.base_key_projection = nn.Linear(
            HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False
        )
        self.control_value_projection = nn.Linear(
            HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False
        )
        self.base_value_projection = nn.Linear(
            HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False
        )
        self.residual_output = nn.Linear(
            2 * INTERNAL_DIMENSION, HIDDEN_SIZE, bias=False
        )
        self.atlas_residual_scale = nn.Parameter(torch.tensor(0.0))
        self.base_logit_scale = nn.Parameter(torch.tensor(0.0))
        self._initialize(initialization_seed)
        self.assert_parameter_contract()

    def _initialize(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("V82 initialization seed must be nonnegative")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for module in (
                self.question_projection,
                self.atlas_key_projection,
                self.base_key_projection,
                self.control_value_projection,
                self.base_value_projection,
            ):
                source = torch.empty(module.weight.shape, dtype=torch.float32)
                nn.init.kaiming_uniform_(
                    source, a=math.sqrt(5), generator=generator
                )
                module.weight.copy_(source)
            self.residual_output.weight.zero_()
            self.atlas_residual_scale.zero_()
            self.base_logit_scale.zero_()

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def assert_parameter_contract(self) -> None:
        if set(self.state_dict()) != CANDIDATE_TENSOR_NAMES:
            raise RuntimeError("V82 candidate tensor inventory changed")
        if self.trainable_parameter_count != TRAINABLE_PARAMETER_COUNT:
            raise RuntimeError("V82 trainable parameter count changed")
        for name, parameter in self.named_parameters():
            if (
                parameter.dtype != torch.float32
                or not parameter.requires_grad
                or not bool(torch.isfinite(parameter.detach()).all())
            ):
                raise RuntimeError(f"V82 invalid trainable parameter: {name}")
        biased = [
            name
            for name, module in self.named_modules()
            if isinstance(module, nn.Linear) and module.bias is not None
        ]
        if biased:
            raise RuntimeError(f"V82 payload path gained bias: {biased}")

    def audit(self) -> V82ReaderAudit:
        return V82ReaderAudit(
            schema_version=1,
            artifact=ARTIFACT,
            architecture=ARCHITECTURE,
            fixed_memory_tokens=738,
            hidden_size=HIDDEN_SIZE,
            probe_count=PROBE_COUNT,
            values_per_probe=VALUES_PER_PROBE,
            base_environment_latents=BASE_ENVIRONMENT_LATENTS,
            internal_dimension=INTERNAL_DIMENSION,
            trainable_parameter_count=TRAINABLE_PARAMETER_COUNT,
            atlas_uniform_floor_mass=ATLAS_UNIFORM_FLOOR_MASS,
            base_uniform_floor_mass=BASE_UNIFORM_FLOOR_MASS,
            atlas_minimum_weight=ATLAS_UNIFORM_FLOOR_MASS / PROBE_COUNT,
            base_minimum_weight=(
                BASE_UNIFORM_FLOOR_MASS / BASE_ENVIRONMENT_LATENTS
            ),
            atlas_direct_mix=ATLAS_DIRECT_MIX,
            base_direct_mix=BASE_DIRECT_MIX,
            every_atlas_value_participates=True,
            every_base_latent_participates=True,
            question_dependent_retrieval=False,
            semantic_or_spatial_top_k_selection=False,
            environmental_text_inputs=(),
            zero_environment_produces_exact_zero_controls=True,
        )

    def candidate_state_dict(self) -> dict[str, torch.Tensor]:
        self.assert_parameter_contract()
        return {
            name: value.detach().cpu().float().contiguous()
            for name, value in self.state_dict().items()
        }

    def forward(
        self,
        fixed_memory: torch.Tensor,
        detached_question_query: torch.Tensor,
        *,
        binding: V81PrefixBinding,
    ) -> V82ReaderOutput:
        assert_prefix_binding_v81(fixed_memory, binding=binding)
        banks = split_v75_v2_prefix_v81(fixed_memory.detach())
        query = _validate_query(
            detached_question_query, batch_size=fixed_memory.shape[0]
        )
        keys = banks.probe_keys.detach().float()
        atlas_values = banks.atlas_values.detach().float()
        base_values = banks.base_latents.detach().float()
        zero_payload = (
            int(torch.count_nonzero(atlas_values)) == 0
            and int(torch.count_nonzero(base_values)) == 0
        )

        if zero_payload:
            atlas_weights = torch.full(
                (fixed_memory.shape[0], PROBE_COUNT),
                1.0 / PROBE_COUNT,
                device=fixed_memory.device,
                dtype=torch.float32,
            )
            base_weights = torch.full(
                (fixed_memory.shape[0], BASE_ENVIRONMENT_LATENTS),
                1.0 / BASE_ENVIRONMENT_LATENTS,
                device=fixed_memory.device,
                dtype=torch.float32,
            )
            atlas_contexts = torch.zeros(
                fixed_memory.shape[0],
                VALUES_PER_PROBE,
                HIDDEN_SIZE,
                device=fixed_memory.device,
                dtype=torch.float32,
            )
            base_context = torch.zeros(
                fixed_memory.shape[0],
                HIDDEN_SIZE,
                device=fixed_memory.device,
                dtype=torch.float32,
            )
            residual = torch.zeros_like(atlas_contexts)
            controls = torch.zeros_like(atlas_contexts)
        else:
            if bool(torch.any(keys.norm(dim=-1) <= 1e-8)):
                raise ValueError("V82 probe keys contain a zero row")
            normalized_query = F.normalize(query, dim=-1)
            deterministic_logits = torch.einsum(
                "bd,bpd->bp", normalized_query, F.normalize(keys, dim=-1)
            ) * RAW_ATLAS_LOGIT_SCALE
            projected_query = F.normalize(
                self.question_projection(query), dim=-1, eps=1e-8
            )
            projected_atlas_keys = F.normalize(
                self.atlas_key_projection(keys), dim=-1, eps=1e-8
            )
            learned_atlas_logits = torch.einsum(
                "bd,bpd->bp", projected_query, projected_atlas_keys
            )
            atlas_scale = torch.tanh(self.atlas_residual_scale.float()) * 32.0
            atlas_weights = dense_floor_attention(
                deterministic_logits + atlas_scale * learned_atlas_logits,
                uniform_floor_mass=ATLAS_UNIFORM_FLOOR_MASS,
            )
            atlas_contexts = torch.einsum(
                "bp,bpvh->bvh", atlas_weights, atlas_values
            )

            projected_base_keys = F.normalize(
                self.base_key_projection(base_values), dim=-1, eps=1e-8
            )
            base_scale = F.softplus(self.base_logit_scale.float()).clamp(max=32.0)
            base_logits = torch.einsum(
                "bd,bld->bl", projected_query, projected_base_keys
            ) * base_scale
            base_weights = dense_floor_attention(
                base_logits, uniform_floor_mass=BASE_UNIFORM_FLOOR_MASS
            )
            base_context = torch.einsum("bl,blh->bh", base_weights, base_values)

            direct = (
                ATLAS_DIRECT_MIX * atlas_contexts
                + BASE_DIRECT_MIX * base_context.unsqueeze(1)
            )
            control_features = self.control_value_projection(direct)
            base_features = self.base_value_projection(base_context).unsqueeze(1)
            base_features = base_features.expand(-1, VALUES_PER_PROBE, -1)
            residual = self.residual_output(
                torch.cat((control_features, base_features), dim=-1)
            )
            controls = direct + residual
            raw_rms = controls.square().mean(dim=-1, keepdim=True).sqrt()
            controls = controls * torch.clamp(
                MAXIMUM_CONTROL_RMS / raw_rms.clamp_min(1e-8), max=1.0
            )

        atlas_sums = atlas_weights.sum(dim=-1)
        base_sums = base_weights.sum(dim=-1)
        control_rms = controls.square().mean(dim=-1).sqrt()
        minimum_atlas = ATLAS_UNIFORM_FLOOR_MASS / PROBE_COUNT
        minimum_base = BASE_UNIFORM_FLOOR_MASS / BASE_ENVIRONMENT_LATENTS
        atlas_positive = bool(torch.all(atlas_weights >= minimum_atlas - 1e-8))
        base_positive = bool(torch.all(base_weights >= minimum_base - 1e-8))
        if (
            tuple(controls.shape)
            != (fixed_memory.shape[0], VALUES_PER_PROBE, HIDDEN_SIZE)
            or not all(
                bool(torch.isfinite(value).all())
                for value in (
                    controls,
                    atlas_weights,
                    base_weights,
                    atlas_contexts,
                    base_context,
                    residual,
                )
            )
            or not torch.allclose(
                atlas_sums, torch.ones_like(atlas_sums), atol=1e-6, rtol=0.0
            )
            or not torch.allclose(
                base_sums, torch.ones_like(base_sums), atol=1e-6, rtol=0.0
            )
            or not atlas_positive
            or not base_positive
            or float(control_rms.detach().max()) > MAXIMUM_CONTROL_RMS + 1e-5
        ):
            raise RuntimeError("V82 dense reader failed its numeric contract")
        if zero_payload and float(controls.abs().max()) != 0.0:
            raise RuntimeError("V82 zero environmental payload was not exactly zero")
        return V82ReaderOutput(
            controls=controls,
            atlas_weights=atlas_weights,
            base_weights=base_weights,
            atlas_contexts=atlas_contexts,
            base_context=base_context,
            residual=residual,
            control_rms=control_rms,
            atlas_attention_sums=atlas_sums,
            base_attention_sums=base_sums,
            zero_environmental_payload=zero_payload,
            all_96_groups_positive=atlas_positive,
            all_384_atlas_values_positive=atlas_positive,
            all_256_base_latents_positive=base_positive,
        )


__all__ = [
    "ARCHITECTURE",
    "ARTIFACT",
    "ATLAS_DIRECT_MIX",
    "BASE_DIRECT_MIX",
    "CANDIDATE_TENSOR_NAMES",
    "INTERNAL_DIMENSION",
    "TRAINABLE_PARAMETER_COUNT",
    "DenseLearnedSceneReaderV82",
    "V82ReaderAudit",
    "V82ReaderOutput",
    "wrong_scene_contrast_loss_v82",
]

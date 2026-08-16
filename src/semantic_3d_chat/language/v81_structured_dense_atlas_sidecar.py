"""Dense readout over one immutable V75-V2 738-token scene memory.

Stage A is deterministic: a detached frozen-question embedding addresses all
96 probe keys with normalized cosine scores at train-selected scale 160 and a positive
floor.  The same weights directly reconstruct the four aligned V75 scene-value
banks.  Keys, queries, BOI, and EOI never enter the payload.

The module also defines a quarantined Stage B dual-bank residual.  It reads all
96 atlas groups and all 256 base latents, then adds a bias-free residual after
a detached decoder state and before a frozen LM head.  Its output projection is
zero initialized, so disabled or zero-payload operation is exactly inert.  No
function here loads Gemma, retrieves tokens, performs top-k selection, or
serializes scene memory.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import nn

from semantic_3d_chat.language.local_lm import question_token_ids
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256

HIDDEN_SIZE: Final[int] = 1536
PROBE_COUNT: Final[int] = 96
VALUES_PER_PROBE: Final[int] = 4
ATLAS_GROUP_TOKENS: Final[int] = 1 + VALUES_PER_PROBE
ATLAS_MEMORY_TOKENS: Final[int] = PROBE_COUNT * ATLAS_GROUP_TOKENS
BASE_ENVIRONMENT_LATENTS: Final[int] = 256
FIXED_PREFIX_TOKENS: Final[int] = 2 + ATLAS_MEMORY_TOKENS + BASE_ENVIRONMENT_LATENTS
BASE_PREFIX_TOKENS: Final[int] = 2 + BASE_ENVIRONMENT_LATENTS
INTERNAL_DIMENSION: Final[int] = 128
RAW_ATLAS_LOGIT_SCALE: Final[float] = 160.0
MAXIMUM_CONTROL_RMS: Final[float] = 0.25
FINAL_LOGIT_SOFTCAPPING: Final[float] = 30.0
MODEL_BLOB_SHA256_IDENTITY: Final[str] = (
    "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
)
INPUT_EMBEDDING_TENSOR_NAME: Final[str] = "model.language_model.embed_tokens.weight"
ATLAS_UNIFORM_FLOOR_MASS: Final[float] = 0.05
BASE_UNIFORM_FLOOR_MASS: Final[float] = 0.10
MINIMUM_ATLAS_WEIGHT: Final[float] = ATLAS_UNIFORM_FLOOR_MASS / PROBE_COUNT
MINIMUM_BASE_WEIGHT: Final[float] = BASE_UNIFORM_FLOOR_MASS / BASE_ENVIRONMENT_LATENTS
ARCHITECTURE: Final[str] = "structured_dual_bank_dense_atlas_sidecar_v81"
STAGE_A_ARCHITECTURE: Final[str] = "normalized_query_probe_cosine_dense_read_v81"
ARTIFACT: Final[str] = "gemma4_v81_strict_fixed_prefix_reader_diagnostic_v1"

CANDIDATE_TENSOR_NAMES: Final[frozenset[str]] = frozenset(
    {
        "atlas_logit_scale",
        "base_logit_scale",
        "atlas_query.weight",
        "atlas_key.weight",
        "atlas_value_projections.0.weight",
        "atlas_value_projections.1.weight",
        "atlas_value_projections.2.weight",
        "atlas_value_projections.3.weight",
        "base_query.weight",
        "base_key.weight",
        "base_value.weight",
        "residual_output.weight",
    }
)
TRAINABLE_PARAMETER_COUNT: Final[int] = (
    (2 + VALUES_PER_PROBE + 3) * HIDDEN_SIZE * INTERNAL_DIMENSION
    + HIDDEN_SIZE * ((VALUES_PER_PROBE + 1) * INTERNAL_DIMENSION)
    + 2
)


@dataclass(frozen=True)
class V81AtlasBanks:
    """Exact role-preserving views into one unmodified V75-V2 prefix."""

    boi: torch.Tensor
    probe_keys: torch.Tensor
    atlas_values: torch.Tensor
    base_latents: torch.Tensor
    eoi: torch.Tensor


@dataclass(frozen=True)
class V81PrefixAudit:
    schema_version: int
    architecture: str
    layout: tuple[str, ...]
    fixed_prefix_sha256: str
    base_prefix_sha256: str
    boi_sha256: str
    atlas_memory_sha256: str
    probe_keys_sha256: str
    atlas_values_sha256: str
    base_latents_sha256: str
    eoi_sha256: str
    fixed_prefix_tokens: int
    atlas_memory_tokens: int
    probe_count: int
    values_per_probe: int
    base_environment_latents: int
    hidden_size: int
    exact_reconstruction: bool
    boundary_tokens_retained: bool
    all_atlas_tokens_retained: bool
    all_base_latents_retained: bool
    boi_eoi_are_payload: bool
    probe_keys_are_payload: bool
    question_queries_are_payload: bool
    only_scene_values_and_base_latents_are_payload: bool
    stage_a_positive_floor_value_count: int
    stage_a_dense_score_key_count: int
    base_latents_use_native_frozen_gemma_path_in_stage_a: bool
    all_738_tokens_claimed_strict_positive_payload_influence: bool
    question_inputs_used_to_parse: bool
    question_dependent_retrieval: bool
    top_k_selection: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["layout"] = list(self.layout)
        return value


@dataclass(frozen=True)
class V81StageAOutput:
    """Direct four-control reconstruction before any optional fusion."""

    reconstructed_controls: torch.Tensor
    atlas_weights: torch.Tensor
    atlas_logits: torch.Tensor
    fixed_prefix_sha256: str
    atlas_memory_sha256: str
    base_prefix_sha256: str
    control_rms: torch.Tensor
    attention_sums: torch.Tensor
    finite: bool
    all_96_groups_positive: bool
    all_384_values_receive_positive_floor_weight: bool


@dataclass(frozen=True)
class V81PrefixBinding:
    """Hashes captured once before any latest-user question is tokenized."""

    fixed_prefix_sha256: str
    atlas_memory_sha256: str
    base_prefix_sha256: str
    compiled_before_question: bool


@dataclass(frozen=True)
class V81LatestUserQuery:
    """Detached mean of complete latest-user-only token embeddings."""

    query: torch.Tensor
    token_ids: torch.Tensor
    token_count: int
    add_special_tokens: bool
    included_system_prompt: bool
    included_history: bool
    included_answer: bool
    detached: bool


@dataclass(frozen=True)
class V81SidecarOutput:
    """Stage A controls plus an optional post-decoder Stage B residual."""

    fused_hidden: torch.Tensor
    residual: torch.Tensor
    reconstructed_controls: torch.Tensor
    atlas_weights: torch.Tensor
    learned_atlas_weights: torch.Tensor
    base_weights: torch.Tensor
    atlas_value_contexts: torch.Tensor
    base_context: torch.Tensor
    stage_b_enabled: bool


def split_v75_v2_prefix_v81(prefix: torch.Tensor) -> V81AtlasBanks:
    """Return exact views for ``[BOI, 96*(key+4 values), 256 base, EOI]``."""

    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 3:
        raise ValueError("V81 fixed prefix must have shape [B,738,1536]")
    if tuple(prefix.shape[1:]) != (FIXED_PREFIX_TOKENS, HIDDEN_SIZE):
        raise ValueError(
            f"V81 fixed prefix must have shape [B,738,1536]; observed={tuple(prefix.shape)}"
        )
    if prefix.shape[0] < 1 or not prefix.is_floating_point():
        raise ValueError("V81 fixed prefix must be a nonempty floating tensor")
    if not bool(torch.isfinite(prefix).all()):
        raise ValueError("V81 fixed prefix contains NaN or infinity")

    boi = prefix[:, :1]
    atlas_memory = prefix[:, 1 : 1 + ATLAS_MEMORY_TOKENS]
    groups = atlas_memory.reshape(prefix.shape[0], PROBE_COUNT, ATLAS_GROUP_TOKENS, HIDDEN_SIZE)
    base_start = 1 + ATLAS_MEMORY_TOKENS
    return V81AtlasBanks(
        boi=boi,
        probe_keys=groups[:, :, 0],
        atlas_values=groups[:, :, 1:],
        base_latents=prefix[:, base_start : base_start + BASE_ENVIRONMENT_LATENTS],
        eoi=prefix[:, -1:],
    )


def reconstruct_base_v54_prefix_v81(prefix: torch.Tensor) -> torch.Tensor:
    """Recover the exact 258-token BOI/base-latent/EOI scene prefix."""

    banks = split_v75_v2_prefix_v81(prefix)
    return torch.cat((banks.boi, banks.base_latents, banks.eoi), dim=1)


def bind_fixed_prefix_before_question_v81(prefix: torch.Tensor) -> V81PrefixBinding:
    """Bind fixed, atlas, and base identities before user tokenization begins."""

    audit = audit_v75_v2_prefix_v81(prefix)
    return V81PrefixBinding(
        fixed_prefix_sha256=audit.fixed_prefix_sha256,
        atlas_memory_sha256=audit.atlas_memory_sha256,
        base_prefix_sha256=audit.base_prefix_sha256,
        compiled_before_question=True,
    )


def assert_prefix_binding_v81(prefix: torch.Tensor, *, binding: V81PrefixBinding) -> None:
    """Reassert all prequestion scene-memory hashes before every dense read."""

    if not isinstance(binding, V81PrefixBinding) or not binding.compiled_before_question:
        raise ValueError("V81 requires a valid prequestion prefix binding")
    audit = audit_v75_v2_prefix_v81(prefix)
    observed = (
        audit.fixed_prefix_sha256,
        audit.atlas_memory_sha256,
        audit.base_prefix_sha256,
    )
    expected = (
        binding.fixed_prefix_sha256,
        binding.atlas_memory_sha256,
        binding.base_prefix_sha256,
    )
    if observed != expected:
        raise ValueError(
            "V81 fixed scene memory changed after prequestion binding: "
            f"expected={expected} observed={observed}"
        )


def assert_fixed_prefix_identity_v81(prefix: torch.Tensor, *, expected_sha256: str) -> None:
    """Fail closed if a per-question caller changes the bound 738-token memory."""

    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("V81 expected fixed-prefix SHA-256 is invalid")
    observed = prefix_sha256(prefix)
    if observed != expected_sha256:
        raise ValueError(
            "V81 fixed V75-V2 memory changed after prequestion binding: "
            f"expected={expected_sha256} observed={observed}"
        )


def audit_v75_v2_prefix_v81(prefix: torch.Tensor) -> V81PrefixAudit:
    """Hash every fixed-memory role and prove a lossless structured parse."""

    banks = split_v75_v2_prefix_v81(prefix)
    atlas_memory = torch.cat((banks.probe_keys.unsqueeze(2), banks.atlas_values), dim=2).reshape(
        prefix.shape[0], ATLAS_MEMORY_TOKENS, HIDDEN_SIZE
    )
    reconstructed = torch.cat((banks.boi, atlas_memory, banks.base_latents, banks.eoi), dim=1)
    exact = torch.equal(reconstructed, prefix)
    if not exact:
        raise RuntimeError("V81 fixed-prefix role parsing was not lossless")
    base = reconstruct_base_v54_prefix_v81(prefix)
    return V81PrefixAudit(
        schema_version=1,
        architecture="exact_v75_v2_structured_memory_parse_v81",
        layout=(
            "boi",
            "96_interleaved_probe_key_plus_four_scene_value_groups",
            "all_256_v54_environment_latents",
            "eoi",
        ),
        fixed_prefix_sha256=prefix_sha256(prefix),
        base_prefix_sha256=prefix_sha256(base),
        boi_sha256=tensor_sha256(banks.boi),
        atlas_memory_sha256=tensor_sha256(atlas_memory),
        probe_keys_sha256=tensor_sha256(banks.probe_keys),
        atlas_values_sha256=tensor_sha256(banks.atlas_values),
        base_latents_sha256=tensor_sha256(banks.base_latents),
        eoi_sha256=tensor_sha256(banks.eoi),
        fixed_prefix_tokens=FIXED_PREFIX_TOKENS,
        atlas_memory_tokens=ATLAS_MEMORY_TOKENS,
        probe_count=PROBE_COUNT,
        values_per_probe=VALUES_PER_PROBE,
        base_environment_latents=BASE_ENVIRONMENT_LATENTS,
        hidden_size=HIDDEN_SIZE,
        exact_reconstruction=exact,
        boundary_tokens_retained=torch.equal(banks.boi, prefix[:, :1])
        and torch.equal(banks.eoi, prefix[:, -1:]),
        all_atlas_tokens_retained=atlas_memory.shape[1] == ATLAS_MEMORY_TOKENS,
        all_base_latents_retained=(banks.base_latents.shape[1] == BASE_ENVIRONMENT_LATENTS),
        boi_eoi_are_payload=False,
        probe_keys_are_payload=False,
        question_queries_are_payload=False,
        only_scene_values_and_base_latents_are_payload=True,
        stage_a_positive_floor_value_count=PROBE_COUNT * VALUES_PER_PROBE,
        stage_a_dense_score_key_count=PROBE_COUNT,
        base_latents_use_native_frozen_gemma_path_in_stage_a=True,
        all_738_tokens_claimed_strict_positive_payload_influence=False,
        question_inputs_used_to_parse=False,
        question_dependent_retrieval=False,
        top_k_selection=False,
    )


def dense_floor_attention(logits: torch.Tensor, *, uniform_floor_mass: float) -> torch.Tensor:
    """Return dense softmax weights with a strict positive floor per entry."""

    if logits.ndim < 2 or logits.shape[-1] < 1:
        raise ValueError("V81 attention logits require a nonempty memory axis")
    if not math.isfinite(float(uniform_floor_mass)) or not 0.0 < float(uniform_floor_mass) < 1.0:
        raise ValueError("V81 uniform floor mass must lie strictly between zero and one")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("V81 attention logits contain NaN or infinity")
    count = logits.shape[-1]
    return float(uniform_floor_mass) / count + (1.0 - float(uniform_floor_mass)) * torch.softmax(
        logits.float(), dim=-1
    )


def latest_user_question_query_v81(
    *,
    tokenizer: Any,
    embedding_layer: nn.Module,
    latest_user_question: str,
    device: torch.device,
    maximum_question_tokens: int,
    model_blob_sha256_identity: str,
    embedding_tensor_name: str,
) -> V81LatestUserQuery:
    """Build the exact latest-user-only V81 address query.

    This deliberately uses the same ``add_special_tokens=False`` tokenization
    as ``question_token_ids`` while accepting no system prompt, history, or
    answer arguments.  The frozen embedding lookup is evaluated under
    inference mode and the complete nonempty sequence is mean pooled in FP32.
    """

    if model_blob_sha256_identity != MODEL_BLOB_SHA256_IDENTITY:
        raise ValueError("V81 frozen Gemma model blob identity changed")
    if embedding_tensor_name != INPUT_EMBEDDING_TENSOR_NAME:
        raise ValueError("V81 frozen Gemma input-embedding tensor identity changed")
    if any(parameter.requires_grad for parameter in embedding_layer.parameters()):
        raise ValueError("V81 latest-user query requires a frozen embedding layer")
    if not isinstance(latest_user_question, str) or not latest_user_question:
        raise ValueError("V81 latest user question must be a nonempty string")
    if (
        isinstance(maximum_question_tokens, bool)
        or not isinstance(maximum_question_tokens, int)
        or maximum_question_tokens < 1
    ):
        raise ValueError("V81 maximum question tokens must be a positive integer")
    token_ids = question_token_ids(tokenizer, latest_user_question, device)
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] < 1:
        raise ValueError("V81 latest-user tokenizer returned no usable token sequence")
    if token_ids.shape[1] > maximum_question_tokens:
        raise ValueError(
            "V81 latest user question exceeds configured token limit: "
            f"{token_ids.shape[1]} > {maximum_question_tokens}"
        )
    with torch.inference_mode():
        embeddings = embedding_layer(token_ids).detach().float()
        query = embeddings.mean(dim=1).detach().contiguous()
    if tuple(query.shape) != (1, HIDDEN_SIZE) or not bool(torch.isfinite(query).all()):
        raise RuntimeError("V81 frozen latest-user query has invalid shape or values")
    return V81LatestUserQuery(
        query=query,
        token_ids=token_ids.detach(),
        token_count=int(token_ids.shape[1]),
        add_special_tokens=False,
        included_system_prompt=False,
        included_history=False,
        included_answer=False,
        detached=not query.requires_grad,
    )


def deterministic_atlas_read_v81(
    fixed_prefix: torch.Tensor,
    frozen_question_query: torch.Tensor,
    *,
    binding: V81PrefixBinding,
    logit_scale: float = RAW_ATLAS_LOGIT_SCALE,
    uniform_floor_mass: float = ATLAS_UNIFORM_FLOOR_MASS,
) -> V81StageAOutput:
    """Densely reconstruct four controls from all atlas groups without weights."""

    assert_prefix_binding_v81(fixed_prefix, binding=binding)
    banks = split_v75_v2_prefix_v81(fixed_prefix.detach())
    if (
        not isinstance(frozen_question_query, torch.Tensor)
        or frozen_question_query.ndim != 2
        or tuple(frozen_question_query.shape) != (fixed_prefix.shape[0], HIDDEN_SIZE)
    ):
        raise ValueError("V81 frozen question query must have shape [B,1536]")
    if not math.isfinite(float(logit_scale)) or float(logit_scale) <= 0.0:
        raise ValueError("V81 deterministic atlas logit scale must be positive")
    question = frozen_question_query.detach().float()
    keys = banks.probe_keys.detach().float()
    values = banks.atlas_values.detach().float()
    if not bool(torch.isfinite(question).all()):
        raise ValueError("V81 frozen question query contains NaN or infinity")
    zero_payload = int(torch.count_nonzero(values)) == 0
    if zero_payload:
        # Environmental zero is a required algebraic identity.  It must not be
        # rejected merely because a zeroed synthetic prefix also has zero keys.
        logits = torch.zeros(
            fixed_prefix.shape[0],
            PROBE_COUNT,
            dtype=torch.float32,
            device=fixed_prefix.device,
        )
        weights = torch.full_like(logits, 1.0 / PROBE_COUNT)
        controls = torch.zeros(
            fixed_prefix.shape[0],
            VALUES_PER_PROBE,
            HIDDEN_SIZE,
            dtype=torch.float32,
            device=fixed_prefix.device,
        )
    else:
        if bool(torch.any(question.norm(dim=-1) <= 1e-8)) or bool(
            torch.any(keys.norm(dim=-1) <= 1e-8)
        ):
            raise ValueError("V81 cosine addressing requires nonzero query and probe keys")
        logits = torch.einsum(
            "bd,bpd->bp",
            F.normalize(question, dim=-1),
            F.normalize(keys, dim=-1),
        ) * float(logit_scale)
        weights = dense_floor_attention(logits, uniform_floor_mass=uniform_floor_mass)
        controls = torch.einsum("bp,bpvh->bvh", weights, values)
    expected_shape = (fixed_prefix.shape[0], VALUES_PER_PROBE, HIDDEN_SIZE)
    attention_sums = weights.sum(dim=-1)
    control_rms = controls.square().mean(dim=-1).sqrt()
    finite = bool(
        torch.isfinite(logits).all()
        and torch.isfinite(weights).all()
        and torch.isfinite(controls).all()
    )
    all_positive = bool(torch.all(weights >= float(uniform_floor_mass) / PROBE_COUNT))
    if (
        tuple(controls.shape) != expected_shape
        or not finite
        or not torch.allclose(
            attention_sums,
            torch.ones_like(attention_sums),
            atol=1e-6,
            rtol=0.0,
        )
        or not all_positive
        or float(control_rms.max()) > MAXIMUM_CONTROL_RMS + 1e-5
    ):
        raise RuntimeError("V81 deterministic atlas read failed its numeric audit")
    return V81StageAOutput(
        reconstructed_controls=controls,
        atlas_weights=weights,
        atlas_logits=logits,
        fixed_prefix_sha256=binding.fixed_prefix_sha256,
        atlas_memory_sha256=binding.atlas_memory_sha256,
        base_prefix_sha256=binding.base_prefix_sha256,
        control_rms=control_rms,
        attention_sums=attention_sums,
        finite=finite,
        all_96_groups_positive=all_positive,
        all_384_values_receive_positive_floor_weight=all_positive,
    )


def frozen_lm_head_logits_v81(
    fused_hidden: torch.Tensor, *, frozen_lm_head: nn.Module
) -> torch.Tensor:
    """Apply Gemma's frozen tied LM head and exact final-logit softcap.

    The head remains in the autograd graph with immutable parameters, allowing
    a future loss to update only the sidecar while preventing Gemma gradients.
    """

    if (
        not isinstance(fused_hidden, torch.Tensor)
        or fused_hidden.ndim != 2
        or fused_hidden.shape[1] != HIDDEN_SIZE
        or not bool(torch.isfinite(fused_hidden).all())
    ):
        raise ValueError("V81 fused hidden state must be finite [B,1536]")
    parameters = tuple(frozen_lm_head.parameters())
    if not parameters or any(parameter.requires_grad for parameter in parameters):
        raise ValueError("V81 LM head must own frozen parameters")
    raw_logits = frozen_lm_head(fused_hidden)
    if (
        not isinstance(raw_logits, torch.Tensor)
        or raw_logits.ndim != 2
        or raw_logits.shape[0] != fused_hidden.shape[0]
        or not bool(torch.isfinite(raw_logits).all())
    ):
        raise RuntimeError("V81 frozen LM head returned invalid logits")
    logits = (
        torch.tanh(raw_logits / FINAL_LOGIT_SOFTCAPPING)
        * FINAL_LOGIT_SOFTCAPPING
    )
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("V81 Gemma final-logit softcap produced invalid logits")
    return logits


class StructuredDenseAtlasSidecarV81(nn.Module):
    """Bias-free optional Stage B over detached frozen-model activations."""

    def __init__(self, *, allow_stage_b: bool = False) -> None:
        super().__init__()
        self.allow_stage_b = bool(allow_stage_b)
        self.atlas_query = nn.Linear(HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False)
        self.atlas_key = nn.Linear(HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False)
        self.atlas_value_projections = nn.ModuleList(
            [
                nn.Linear(HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False)
                for _ in range(VALUES_PER_PROBE)
            ]
        )
        self.base_query = nn.Linear(HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False)
        self.base_key = nn.Linear(HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False)
        self.base_value = nn.Linear(HIDDEN_SIZE, INTERNAL_DIMENSION, bias=False)
        self.residual_output = nn.Linear(
            (VALUES_PER_PROBE + 1) * INTERNAL_DIMENSION,
            HIDDEN_SIZE,
            bias=False,
        )
        self.atlas_logit_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.base_logit_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        nn.init.zeros_(self.residual_output.weight)
        self.assert_parameter_contract()

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def assert_parameter_contract(self) -> None:
        state = self.state_dict()
        if set(state) != CANDIDATE_TENSOR_NAMES:
            raise RuntimeError("V81 candidate tensor inventory changed")
        if self.trainable_parameter_count != TRAINABLE_PARAMETER_COUNT:
            raise RuntimeError("V81 trainable parameter count changed")
        for name, parameter in self.named_parameters():
            if (
                parameter.dtype != torch.float32
                or not parameter.requires_grad
                or not bool(torch.isfinite(parameter.detach()).all())
            ):
                raise RuntimeError(f"V81 parameter is not finite trainable FP32: {name}")
        biased = [
            name
            for name, module in self.named_modules()
            if isinstance(module, nn.Linear) and module.bias is not None
        ]
        if biased:
            raise RuntimeError(f"V81 linear payload path gained bias: {biased}")

    def candidate_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the only numeric tensor inventory allowed in a future candidate."""

        self.assert_parameter_contract()
        return {
            name: value.detach().cpu().float().contiguous()
            for name, value in self.state_dict().items()
        }

    def forward(
        self,
        fixed_prefix: torch.Tensor,
        frozen_question_query: torch.Tensor,
        detached_decoder_hidden: torch.Tensor,
        *,
        binding: V81PrefixBinding,
        enable_stage_b: bool = False,
    ) -> V81SidecarOutput:
        if (
            not isinstance(detached_decoder_hidden, torch.Tensor)
            or detached_decoder_hidden.ndim != 2
            or tuple(detached_decoder_hidden.shape) != (fixed_prefix.shape[0], HIDDEN_SIZE)
        ):
            raise ValueError("V81 detached decoder hidden must have shape [B,1536]")
        if not bool(torch.isfinite(detached_decoder_hidden).all()):
            raise ValueError("V81 detached decoder hidden contains NaN or infinity")
        if enable_stage_b and not self.allow_stage_b:
            raise PermissionError("V81 Stage B is quarantined pending explicit approval")

        stage_a = deterministic_atlas_read_v81(
            fixed_prefix,
            frozen_question_query,
            binding=binding,
        )
        banks = split_v75_v2_prefix_v81(fixed_prefix.detach())
        decoder = detached_decoder_hidden.detach().float()
        batch_size = fixed_prefix.shape[0]

        if not enable_stage_b:
            zero_residual = torch.zeros_like(decoder)
            zero_atlas_contexts = torch.zeros(
                batch_size,
                VALUES_PER_PROBE,
                INTERNAL_DIMENSION,
                dtype=decoder.dtype,
                device=decoder.device,
            )
            zero_base_context = torch.zeros(
                batch_size,
                INTERNAL_DIMENSION,
                dtype=decoder.dtype,
                device=decoder.device,
            )
            base_weights = torch.full(
                (batch_size, BASE_ENVIRONMENT_LATENTS),
                1.0 / BASE_ENVIRONMENT_LATENTS,
                dtype=decoder.dtype,
                device=decoder.device,
            )
            return V81SidecarOutput(
                fused_hidden=decoder,
                residual=zero_residual,
                reconstructed_controls=stage_a.reconstructed_controls,
                atlas_weights=stage_a.atlas_weights,
                learned_atlas_weights=stage_a.atlas_weights,
                base_weights=base_weights,
                atlas_value_contexts=zero_atlas_contexts,
                base_context=zero_base_context,
                stage_b_enabled=False,
            )

        question = frozen_question_query.detach().float()
        probe_keys = banks.probe_keys.detach().float()
        atlas_values = banks.atlas_values.detach().float()
        base_latents = banks.base_latents.detach().float()
        learned_atlas_logits = torch.einsum(
            "bd,bpd->bp",
            self.atlas_query(question),
            self.atlas_key(probe_keys),
        )
        atlas_scale = self.atlas_logit_scale.float().exp().clamp(max=100.0)
        learned_atlas_logits = learned_atlas_logits * (atlas_scale / math.sqrt(INTERNAL_DIMENSION))
        learned_atlas_weights = dense_floor_attention(
            learned_atlas_logits,
            uniform_floor_mass=ATLAS_UNIFORM_FLOOR_MASS,
        )
        atlas_contexts = torch.stack(
            [
                torch.einsum(
                    "bp,bpd->bd",
                    learned_atlas_weights,
                    projection(atlas_values[:, :, bank]),
                )
                for bank, projection in enumerate(self.atlas_value_projections)
            ],
            dim=1,
        )

        base_logits = torch.einsum(
            "bd,bld->bl",
            self.base_query(decoder),
            self.base_key(base_latents),
        )
        base_scale = self.base_logit_scale.float().exp().clamp(max=100.0)
        base_logits = base_logits * (base_scale / math.sqrt(INTERNAL_DIMENSION))
        base_weights = dense_floor_attention(
            base_logits,
            uniform_floor_mass=BASE_UNIFORM_FLOOR_MASS,
        )
        base_context = torch.einsum("bl,bld->bd", base_weights, self.base_value(base_latents))
        payload = torch.cat((atlas_contexts.reshape(batch_size, -1), base_context), dim=-1)
        residual = self.residual_output(payload)
        return V81SidecarOutput(
            fused_hidden=decoder + residual,
            residual=residual,
            reconstructed_controls=stage_a.reconstructed_controls,
            atlas_weights=stage_a.atlas_weights,
            learned_atlas_weights=learned_atlas_weights,
            base_weights=base_weights,
            atlas_value_contexts=atlas_contexts,
            base_context=base_context,
            stage_b_enabled=True,
        )


def sanitized_candidate_metadata_v81(*, weights_sha256: str) -> dict[str, Any]:
    """Return answer-free metadata allowed beside future Stage B weights."""

    if len(weights_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in weights_sha256
    ):
        raise ValueError("V81 candidate weights SHA-256 is invalid")
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "architecture": ARCHITECTURE,
        "stage_a_architecture": STAGE_A_ARCHITECTURE,
        "weights_sha256": weights_sha256,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "fixed_prefix_tokens": FIXED_PREFIX_TOKENS,
        "probe_count": PROBE_COUNT,
        "values_per_probe": VALUES_PER_PROBE,
        "base_environment_latents": BASE_ENVIRONMENT_LATENTS,
        "raw_atlas_logit_scale": RAW_ATLAS_LOGIT_SCALE,
        "maximum_control_rms": MAXIMUM_CONTROL_RMS,
        "atlas_uniform_floor_mass": ATLAS_UNIFORM_FLOOR_MASS,
        "base_uniform_floor_mass": BASE_UNIFORM_FLOOR_MASS,
        "bias_free_payload": True,
        "decoder_and_question_inputs_detached": True,
        "residual_after_decoder_before_frozen_lm_head": True,
        "zero_environmental_payload_produces_zero_residual": True,
        "question_dependent_retrieval": False,
        "top_k_selection": False,
        "probe_bank_serialized": False,
        "atlas_values_serialized": False,
        "base_latents_serialized": False,
        "environmental_prefix_cache_serialized": False,
        "questions_serialized": False,
        "answers_serialized": False,
        "prototypes_serialized": False,
        "class_ids_serialized": False,
        "teacher_cache_serialized": False,
        "prediction_cache_serialized": False,
        "environmental_text_serialized": False,
        "runtime_publication_authorized": False,
    }


__all__ = [
    "ARCHITECTURE",
    "ARTIFACT",
    "ATLAS_MEMORY_TOKENS",
    "ATLAS_UNIFORM_FLOOR_MASS",
    "BASE_ENVIRONMENT_LATENTS",
    "BASE_PREFIX_TOKENS",
    "BASE_UNIFORM_FLOOR_MASS",
    "CANDIDATE_TENSOR_NAMES",
    "FINAL_LOGIT_SOFTCAPPING",
    "FIXED_PREFIX_TOKENS",
    "HIDDEN_SIZE",
    "INPUT_EMBEDDING_TENSOR_NAME",
    "INTERNAL_DIMENSION",
    "MAXIMUM_CONTROL_RMS",
    "MINIMUM_ATLAS_WEIGHT",
    "MINIMUM_BASE_WEIGHT",
    "MODEL_BLOB_SHA256_IDENTITY",
    "PROBE_COUNT",
    "RAW_ATLAS_LOGIT_SCALE",
    "STAGE_A_ARCHITECTURE",
    "TRAINABLE_PARAMETER_COUNT",
    "VALUES_PER_PROBE",
    "StructuredDenseAtlasSidecarV81",
    "V81AtlasBanks",
    "V81PrefixAudit",
    "V81PrefixBinding",
    "V81SidecarOutput",
    "V81StageAOutput",
    "assert_fixed_prefix_identity_v81",
    "assert_prefix_binding_v81",
    "audit_v75_v2_prefix_v81",
    "bind_fixed_prefix_before_question_v81",
    "dense_floor_attention",
    "deterministic_atlas_read_v81",
    "frozen_lm_head_logits_v81",
    "latest_user_question_query_v81",
    "reconstruct_base_v54_prefix_v81",
    "sanitized_candidate_metadata_v81",
    "split_v75_v2_prefix_v81",
]

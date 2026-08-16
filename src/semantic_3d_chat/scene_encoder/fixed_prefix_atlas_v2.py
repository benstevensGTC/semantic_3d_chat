"""Versioned fixed-prefix atlas layout with base scene latents nearest text.

V1 places the complete base scene before its compiled key/value atlas.  This
module leaves that hash-pinned implementation untouched and performs one exact,
question-independent reordering after V1 compilation::

    BOI, every atlas key/value token, every base scene latent, EOI

No tensor is averaged, selected, dropped, or recomputed during the reordering.
The function deliberately has no user-question argument.  This module defines
the layout contract only; it does not assert that a suitable sealed controller
checkpoint exists or that the layout improves behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import (
    FixedPrefixAtlasOutput,
    compile_fixed_scene_atlas,
    tensor_sha256,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)


@dataclass(frozen=True)
class FixedPrefixAtlasV2Audit:
    """Exact-construction evidence for the V2 scene-prefix layout."""

    schema_version: int
    architecture: str
    layout: tuple[str, ...]
    source_compiler_architecture: str
    base_scene_prefix_sha256: str
    source_compiled_prefix_sha256: str
    scene_signature_sha256: str
    probe_bank_sha256: str
    atlas_key_sha256: str
    atlas_value_sha256: str
    boi_tensor_sha256: str
    atlas_memory_tensor_sha256: str
    base_environment_tensor_sha256: str
    eoi_tensor_sha256: str
    fixed_scene_prefix_sha256: str
    environment_latent_count: int
    probe_count: int
    values_per_probe: int
    atlas_memory_token_count: int
    fixed_prefix_token_count: int
    hidden_size: int
    atlas_start_index_in_scene_prefix: int
    atlas_end_index_exclusive_in_scene_prefix: int
    base_start_index_in_scene_prefix: int
    base_end_index_exclusive_in_scene_prefix: int
    base_environment_tokens_preserved_exactly: bool
    atlas_key_value_tokens_preserved_exactly: bool
    boundary_tokens_preserved_exactly: bool
    every_environment_latent_influenced_signature: bool
    every_probe_processed: bool
    complete_atlas_included: bool
    compiled_before_user_question: bool
    user_question_inputs_used_for_compilation: bool
    question_dependent_scene_processing: bool
    question_dependent_retrieval: bool
    semantic_or_spatial_top_k_selection: bool
    environmental_text_inputs: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layout"] = list(self.layout)
        payload["environmental_text_inputs"] = list(self.environmental_text_inputs)
        return payload


@dataclass(frozen=True)
class FixedPrefixAtlasV2Output:
    """One immutable V2 prefix and the tensors used to audit it."""

    scene_prefix: torch.Tensor
    scene_signature: torch.Tensor
    atlas_keys: torch.Tensor
    atlas_values: torch.Tensor
    audit: FixedPrefixAtlasV2Audit


def compile_fixed_scene_atlas_v2(
    base_scene_prefix: torch.Tensor,
    controller: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
    probe_embeddings: torch.Tensor,
) -> FixedPrefixAtlasV2Output:
    """Compile and exactly reorder one scene-only atlas before any question.

    V1 remains the single implementation of controller evaluation.  V2 takes
    its complete output and moves the already-materialized atlas block ahead of
    the already-materialized base-latent block.  Calling V1 here is intentional:
    it prevents the two versions from silently diverging in controller math.
    """

    return reorder_compiled_scene_atlas_v2(
        compile_fixed_scene_atlas(
            base_scene_prefix,
            controller,
            probe_embeddings,
        )
    )


def reorder_compiled_scene_atlas_v2(
    source: FixedPrefixAtlasOutput,
) -> FixedPrefixAtlasV2Output:
    """Losslessly reorder one already-compiled, complete numeric atlas.

    This helper deliberately performs no controller evaluation and accepts no
    question.  A future versioned compiler may produce a complete V1-style
    :class:`FixedPrefixAtlasOutput` with a different sealed controller type and
    then reuse this layout transform.  The exact-V7 wrapper above still calls
    the hash-pinned V1 compiler, so its controller type check is not weakened.
    """

    if not isinstance(source, FixedPrefixAtlasOutput):
        raise TypeError("V2 layout requires a complete FixedPrefixAtlasOutput")
    if (
        source.audit.user_question_inputs_used_for_compilation
        or source.audit.question_dependent_scene_processing
        or source.audit.question_dependent_retrieval
        or source.audit.semantic_or_spatial_top_k_selection
        or source.audit.environmental_text_inputs
    ):
        raise ValueError("V2 layout refuses a conditioned, retrieved, or textual source atlas")
    if (
        not source.audit.base_environment_tokens_preserved_exactly
        or not source.audit.every_environment_latent_influenced_signature
        or not source.audit.every_probe_processed
        or not source.audit.complete_atlas_appended
        or not source.audit.compiled_before_user_question
    ):
        raise ValueError("V2 layout requires a complete pre-question source atlas")
    if prefix_sha256(source.scene_prefix) != source.audit.fixed_scene_prefix_sha256:
        raise ValueError("V2 source scene prefix does not match its audited hash")
    if tensor_sha256(source.scene_signature) != source.audit.scene_signature_sha256:
        raise ValueError("V2 source scene signature does not match its audited hash")
    if tensor_sha256(source.atlas_keys) != source.audit.atlas_key_sha256:
        raise ValueError("V2 source atlas keys do not match their audited hash")
    if tensor_sha256(source.atlas_values) != source.audit.atlas_value_sha256:
        raise ValueError("V2 source atlas values do not match their audited hash")

    base_count = source.audit.environment_latent_count
    atlas_count = source.audit.atlas_memory_token_count
    expected_count = base_count + atlas_count + 2
    if source.scene_prefix.shape[1] != expected_count:
        raise RuntimeError("V1 source prefix does not match its audited token counts")

    boi = source.scene_prefix[:, :1]
    base_latents = source.scene_prefix[:, 1 : 1 + base_count]
    atlas_memory = source.scene_prefix[:, 1 + base_count : -1]
    eoi = source.scene_prefix[:, -1:]
    if atlas_memory.shape[1] != atlas_count:
        raise RuntimeError("V1 source prefix does not contain the complete atlas memory")

    fixed = torch.cat((boi, atlas_memory, base_latents, eoi), dim=1).detach()
    atlas_start = 1
    atlas_end = atlas_start + atlas_count
    base_start = atlas_end
    base_end = base_start + base_count

    atlas_preserved = torch.equal(fixed[:, atlas_start:atlas_end], atlas_memory)
    base_preserved = torch.equal(fixed[:, base_start:base_end], base_latents)
    boundaries_preserved = torch.equal(fixed[:, :1], boi) and torch.equal(
        fixed[:, -1:], eoi
    )
    if (
        fixed.shape != source.scene_prefix.shape
        or not atlas_preserved
        or not base_preserved
        or not boundaries_preserved
        or not bool(torch.isfinite(fixed).all().item())
    ):
        raise RuntimeError("V2 fixed-prefix reordering failed its lossless layout contract")

    audit = FixedPrefixAtlasV2Audit(
        schema_version=2,
        architecture="fixed_scene_key_value_atlas_v2",
        layout=("boi", "all_atlas_key_value_tokens", "all_base_scene_latents", "eoi"),
        source_compiler_architecture=source.audit.architecture,
        base_scene_prefix_sha256=source.audit.base_scene_prefix_sha256,
        source_compiled_prefix_sha256=source.audit.fixed_scene_prefix_sha256,
        scene_signature_sha256=source.audit.scene_signature_sha256,
        probe_bank_sha256=source.audit.probe_bank_sha256,
        atlas_key_sha256=source.audit.atlas_key_sha256,
        atlas_value_sha256=source.audit.atlas_value_sha256,
        boi_tensor_sha256=tensor_sha256(boi),
        atlas_memory_tensor_sha256=tensor_sha256(atlas_memory),
        base_environment_tensor_sha256=tensor_sha256(base_latents),
        eoi_tensor_sha256=tensor_sha256(eoi),
        fixed_scene_prefix_sha256=prefix_sha256(fixed),
        environment_latent_count=base_count,
        probe_count=source.audit.probe_count,
        values_per_probe=source.audit.values_per_probe,
        atlas_memory_token_count=atlas_count,
        fixed_prefix_token_count=fixed.shape[1],
        hidden_size=source.audit.hidden_size,
        atlas_start_index_in_scene_prefix=atlas_start,
        atlas_end_index_exclusive_in_scene_prefix=atlas_end,
        base_start_index_in_scene_prefix=base_start,
        base_end_index_exclusive_in_scene_prefix=base_end,
        base_environment_tokens_preserved_exactly=base_preserved,
        atlas_key_value_tokens_preserved_exactly=atlas_preserved,
        boundary_tokens_preserved_exactly=boundaries_preserved,
        every_environment_latent_influenced_signature=(
            source.audit.every_environment_latent_influenced_signature
        ),
        every_probe_processed=source.audit.every_probe_processed,
        complete_atlas_included=source.audit.complete_atlas_appended,
        compiled_before_user_question=True,
        user_question_inputs_used_for_compilation=False,
        question_dependent_scene_processing=False,
        question_dependent_retrieval=False,
        semantic_or_spatial_top_k_selection=False,
        environmental_text_inputs=(),
    )
    return FixedPrefixAtlasV2Output(
        scene_prefix=fixed,
        scene_signature=source.scene_signature,
        atlas_keys=source.atlas_keys,
        atlas_values=source.atlas_values,
        audit=audit,
    )


__all__ = [
    "FixedPrefixAtlasV2Audit",
    "FixedPrefixAtlasV2Output",
    "compile_fixed_scene_atlas_v2",
    "reorder_compiled_scene_atlas_v2",
]

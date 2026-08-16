"""Compile the sealed V75 reader into one question-independent scene atlas.

V75 normally applies its dense all-latent reader after the user question is
known.  This module evaluates that same numeric value function over a fixed
bank of continuous probe embeddings during scene startup.  Every probe and
every resulting value token is placed in the environmental prefix before chat
begins.  The compiler has no user-question argument, performs no retrieval,
and preserves all 256 base scene latents byte-for-byte.

This module defines the numeric compilation mechanism only.  It does not
authorize a checkpoint, behavioral claim, or protected-split evaluation.
"""

from __future__ import annotations

import torch

from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import (
    FixedPrefixAtlasAudit,
    FixedPrefixAtlasOutput,
    tensor_sha256,
    validate_probe_bank,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v2 import (
    FixedPrefixAtlasV2Output,
    reorder_compiled_scene_atlas_v2,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)


def compile_fixed_scene_atlas_v75(
    base_scene_prefix: torch.Tensor,
    controller: DenseFullSceneContinuousControlV75,
    probe_embeddings: torch.Tensor,
) -> FixedPrefixAtlasOutput:
    """Compile all V75 probe/value groups into one immutable V1-layout prefix.

    Layout::

        BOI, every base scene latent,
        probe_0, value_0_0, ..., value_0_C,
        ...,
        probe_P, value_P_0, ..., value_P_C,
        EOI

    ``probe_embeddings`` is a checkpointed numeric tensor bank produced
    offline.  It is not selected or changed by a live question.
    """

    if type(controller) is not DenseFullSceneContinuousControlV75:
        raise TypeError("V75 atlas compilation requires the exact V75 controller")
    expected_tokens = controller.environment_latents + 2
    expected_shape = (1, expected_tokens, controller.hidden_size)
    if tuple(base_scene_prefix.shape) != expected_shape:
        raise ValueError(
            "V75 atlas base prefix must contain BOI, every scene latent, and EOI: "
            f"expected={expected_shape} observed={tuple(base_scene_prefix.shape)}"
        )
    if not base_scene_prefix.is_floating_point() or not bool(
        torch.isfinite(base_scene_prefix).all().item()
    ):
        raise ValueError("V75 atlas base prefix must be finite floating point")

    probes_cpu = validate_probe_bank(
        probe_embeddings,
        hidden_size=controller.hidden_size,
    )
    try:
        controller_device = next(controller.parameters()).device
    except StopIteration:  # pragma: no cover - V75 always owns parameters
        controller_device = base_scene_prefix.device

    controller.eval()
    with torch.inference_mode():
        base_fp32 = base_scene_prefix.detach().to(controller_device).float()
        key, value = controller.encode_scene(base_fp32)
        probes = probes_cpu.to(controller_device)
        probe_count = probes.shape[0]
        output = controller.forward_encoded(
            key.expand(probe_count, -1, -1),
            value.expand(probe_count, -1, -1),
            probes[:, None, :],
        )
        values = output.control_tokens.detach().float()
        controller_audit = controller.audit()

    expected_encoded = (
        1,
        controller.environment_latents,
        controller.model_dimension,
    )
    expected_values = (
        probes_cpu.shape[0],
        controller.control_token_count,
        controller.hidden_size,
    )
    if (
        tuple(key.shape) != expected_encoded
        or tuple(value.shape) != expected_encoded
        or not bool(torch.isfinite(key).all().item())
        or not bool(torch.isfinite(value).all().item())
    ):
        raise RuntimeError("V75 atlas scene K/V cache has an invalid shape or value")
    if tuple(values.shape) != expected_values or not bool(
        torch.isfinite(values).all().item()
    ):
        raise RuntimeError("V75 atlas values have an invalid shape or value")
    if (
        not controller_audit.all_latents_receive_positive_weight
        or controller_audit.question_dependent_retrieval
        or controller_audit.question_only_output_path_exists
    ):
        raise RuntimeError("V75 controller does not satisfy complete-scene atlas use")
    if float(output.control_rms.max().detach().cpu()) > (
        controller.maximum_control_rms + 1e-5
    ):
        raise RuntimeError("V75 atlas values exceeded the controller RMS bound")

    target = base_scene_prefix.detach()
    keys_for_prefix = probes.to(device=target.device, dtype=target.dtype)
    values_for_prefix = values.to(device=target.device, dtype=target.dtype)
    groups = torch.cat((keys_for_prefix[:, None, :], values_for_prefix), dim=1)
    memory = groups.reshape(1, -1, controller.hidden_size)
    fixed = torch.cat((target[:, :-1], memory, target[:, -1:]), dim=1).detach()

    expected_memory_tokens = probes_cpu.shape[0] * (
        1 + controller.control_token_count
    )
    base_preserved = bool(
        torch.equal(fixed[:, : expected_tokens - 1], target[:, :-1])
        and torch.equal(fixed[:, -1:], target[:, -1:])
    )
    complete = fixed.shape[1] == expected_tokens + expected_memory_tokens
    if (
        not base_preserved
        or not complete
        or not bool(torch.isfinite(fixed).all().item())
    ):
        raise RuntimeError("V75 fixed-prefix atlas failed its exact layout contract")

    # The cached K/V tensors are scene-only and collectively retain every
    # environment position.  Their concatenation is the audited V75 scene
    # signature; it is computed before and independently of every live question.
    scene_signature = torch.cat((key, value), dim=-1).detach().cpu().float().contiguous()
    keys_cpu = probes.detach().cpu().float().contiguous()
    values_cpu = values.detach().cpu().float().contiguous()
    audit = FixedPrefixAtlasAudit(
        schema_version=75,
        architecture="fixed_scene_key_value_atlas_v75_v1",
        base_scene_prefix_sha256=prefix_sha256(target),
        scene_signature_sha256=tensor_sha256(scene_signature),
        probe_bank_sha256=tensor_sha256(probes_cpu),
        atlas_key_sha256=tensor_sha256(keys_cpu),
        atlas_value_sha256=tensor_sha256(values_cpu),
        fixed_scene_prefix_sha256=prefix_sha256(fixed),
        environment_latent_count=controller.environment_latents,
        probe_count=probes_cpu.shape[0],
        values_per_probe=controller.control_token_count,
        atlas_memory_token_count=expected_memory_tokens,
        fixed_prefix_token_count=fixed.shape[1],
        hidden_size=controller.hidden_size,
        base_environment_tokens_preserved_exactly=base_preserved,
        every_environment_latent_influenced_signature=True,
        every_probe_processed=True,
        complete_atlas_appended=complete,
        compiled_before_user_question=True,
        user_question_inputs_used_for_compilation=False,
        question_dependent_scene_processing=False,
        question_dependent_retrieval=False,
        semantic_or_spatial_top_k_selection=False,
        environmental_text_inputs=(),
    )
    return FixedPrefixAtlasOutput(
        scene_prefix=fixed,
        scene_signature=scene_signature,
        atlas_keys=keys_cpu,
        atlas_values=values_cpu,
        audit=audit,
    )


def compile_fixed_scene_atlas_v75_v2(
    base_scene_prefix: torch.Tensor,
    controller: DenseFullSceneContinuousControlV75,
    probe_embeddings: torch.Tensor,
) -> FixedPrefixAtlasV2Output:
    """Compile V75 once, then losslessly move all base latents nearest text."""

    return reorder_compiled_scene_atlas_v2(
        compile_fixed_scene_atlas_v75(
            base_scene_prefix,
            controller,
            probe_embeddings,
        )
    )


__all__ = [
    "compile_fixed_scene_atlas_v75",
    "compile_fixed_scene_atlas_v75_v2",
]

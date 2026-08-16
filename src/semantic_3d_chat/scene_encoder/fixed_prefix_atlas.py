"""Compile a learned scene controller into one immutable full-scene prefix.

The controller used by the experimental V66 path normally receives the live
user question.  This module provides a stricter use of the learned value
function: evaluate it exactly once over a checkpointed bank of continuous
probe vectors, before any user text is accepted, and place *every* resulting
key/value group inside the scene prefix.

The final language-model input is therefore conventional causal attention over
one fixed environmental memory followed by ordinary text.  There is no
question-conditioned scene computation, retrieval, selection, or appended
readout token in this path.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import torch

from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash exact tensor identity, shape, dtype, and bytes."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FixedPrefixAtlasAudit:
    schema_version: int
    architecture: str
    base_scene_prefix_sha256: str
    scene_signature_sha256: str
    probe_bank_sha256: str
    atlas_key_sha256: str
    atlas_value_sha256: str
    fixed_scene_prefix_sha256: str
    environment_latent_count: int
    probe_count: int
    values_per_probe: int
    atlas_memory_token_count: int
    fixed_prefix_token_count: int
    hidden_size: int
    base_environment_tokens_preserved_exactly: bool
    every_environment_latent_influenced_signature: bool
    every_probe_processed: bool
    complete_atlas_appended: bool
    compiled_before_user_question: bool
    user_question_inputs_used_for_compilation: bool
    question_dependent_scene_processing: bool
    question_dependent_retrieval: bool
    semantic_or_spatial_top_k_selection: bool
    environmental_text_inputs: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["environmental_text_inputs"] = list(self.environmental_text_inputs)
        return payload


@dataclass(frozen=True)
class FixedPrefixAtlasOutput:
    """One immutable prefix plus the evidence needed to audit its construction."""

    scene_prefix: torch.Tensor
    scene_signature: torch.Tensor
    atlas_keys: torch.Tensor
    atlas_values: torch.Tensor
    audit: FixedPrefixAtlasAudit


def validate_probe_bank(
    probes: torch.Tensor,
    *,
    hidden_size: int,
    minimum_probes: int = 1,
    maximum_probes: int = 1024,
) -> torch.Tensor:
    """Return a detached FP32 probe bank after a fail-closed shape check."""

    if not isinstance(probes, torch.Tensor) or probes.ndim != 2:
        raise ValueError("Fixed-prefix probes must have shape [P,H]")
    if probes.shape[1] != hidden_size:
        raise ValueError(
            f"Fixed-prefix probe hidden size must be {hidden_size}; got {probes.shape[1]}"
        )
    if not minimum_probes <= probes.shape[0] <= maximum_probes:
        raise ValueError(
            "Fixed-prefix probe count must be in "
            f"[{minimum_probes}, {maximum_probes}]; got {probes.shape[0]}"
        )
    if not probes.is_floating_point() or not bool(torch.isfinite(probes).all().item()):
        raise ValueError("Fixed-prefix probes must be finite floating-point values")
    norms = probes.detach().float().norm(dim=-1)
    if bool(torch.any(norms <= 1e-8).item()):
        raise ValueError("Fixed-prefix probes must all be nonzero")
    return probes.detach().cpu().float().contiguous()


def compile_fixed_scene_atlas(
    base_scene_prefix: torch.Tensor,
    controller: AlwaysOnTeacherBasisFullSceneQuestionControlV7,
    probe_embeddings: torch.Tensor,
) -> FixedPrefixAtlasOutput:
    """Build one scene-only key/value atlas with no live-question argument.

    Layout::

        BOI, 256 base scene latents,
        probe_0, value_0_0, ..., value_0_C,
        ...,
        probe_P, value_P_0, ..., value_P_C,
        EOI

    The base scene latents remain byte-identical and every probe/value group is
    included.  The function deliberately has no user-question parameter.
    """

    if type(controller) is not AlwaysOnTeacherBasisFullSceneQuestionControlV7:
        raise TypeError("Fixed-prefix compilation requires the exact sealed V7 controller")
    if base_scene_prefix.ndim != 3 or base_scene_prefix.shape[0] != 1:
        raise ValueError("Base scene prefix must have shape [1,S,H]")
    expected_tokens = controller.expected_environment_latents + 2
    if tuple(base_scene_prefix.shape[1:]) != (
        expected_tokens,
        controller.hidden_size,
    ):
        raise ValueError(
            "Base scene prefix must contain BOI, every expected environment latent, "
            f"and EOI: expected=(1,{expected_tokens},{controller.hidden_size}) "
            f"observed={tuple(base_scene_prefix.shape)}"
        )
    if not base_scene_prefix.is_floating_point() or not bool(
        torch.isfinite(base_scene_prefix).all().item()
    ):
        raise ValueError("Base scene prefix must contain finite floating-point values")

    probes_cpu = validate_probe_bank(
        probe_embeddings,
        hidden_size=controller.hidden_size,
    )
    try:
        controller_device = next(controller.parameters()).device
    except StopIteration:  # pragma: no cover - V7 always owns parameters
        controller_device = base_scene_prefix.device
    controller.eval()
    with torch.inference_mode():
        base_fp32 = base_scene_prefix.detach().to(controller_device).float()
        signature = controller.encode_scene(base_fp32)
        probes = probes_cpu.to(controller_device)
        expanded_signature = signature.expand(probes.shape[0], -1, -1)
        result = controller.forward_from_signature(
            expanded_signature,
            probes.unsqueeze(1),
        )
        values = result.control_tokens.detach().float()
        keys = probes.detach().float()

    expected_values = (
        probes_cpu.shape[0],
        controller.control_token_count,
        controller.hidden_size,
    )
    if tuple(values.shape) != expected_values or not bool(torch.isfinite(values).all().item()):
        raise RuntimeError(
            f"Fixed-prefix atlas values must have shape {expected_values} and be finite"
        )
    actual_rms = values.square().mean(dim=-1).sqrt()
    if float(actual_rms.max().cpu()) > controller.maximum_control_rms + 1e-5:
        raise RuntimeError("Fixed-prefix atlas exceeded the controller RMS bound")

    # Keep the exact base prefix dtype.  The controller is intentionally FP32,
    # while the persistent Gemma prefix normally lives in BF16.
    target = base_scene_prefix.detach()
    keys_for_prefix = keys.to(device=target.device, dtype=target.dtype)
    values_for_prefix = values.to(device=target.device, dtype=target.dtype)
    groups = torch.cat((keys_for_prefix[:, None, :], values_for_prefix), dim=1)
    memory = groups.reshape(1, -1, controller.hidden_size)
    fixed = torch.cat((target[:, :-1], memory, target[:, -1:]), dim=1).detach()
    if not bool(torch.isfinite(fixed).all().item()):
        raise RuntimeError("Compiled fixed scene prefix contains NaN or infinity")

    base_preserved = bool(
        torch.equal(fixed[:, : expected_tokens - 1], target[:, :-1])
        and torch.equal(fixed[:, -1:], target[:, -1:])
    )
    expected_memory_tokens = probes_cpu.shape[0] * (1 + controller.control_token_count)
    complete = fixed.shape[1] == expected_tokens + expected_memory_tokens
    if not base_preserved or not complete:
        raise RuntimeError("Fixed-prefix compilation failed its exact layout contract")

    signature_cpu = signature.detach().cpu().float().contiguous()
    keys_cpu = keys.detach().cpu().float().contiguous()
    values_cpu = values.detach().cpu().float().contiguous()
    audit = FixedPrefixAtlasAudit(
        schema_version=1,
        architecture="fixed_scene_key_value_atlas_v1",
        base_scene_prefix_sha256=prefix_sha256(target),
        scene_signature_sha256=tensor_sha256(signature_cpu),
        probe_bank_sha256=tensor_sha256(probes_cpu),
        atlas_key_sha256=tensor_sha256(keys_cpu),
        atlas_value_sha256=tensor_sha256(values_cpu),
        fixed_scene_prefix_sha256=prefix_sha256(fixed),
        environment_latent_count=controller.expected_environment_latents,
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
        scene_signature=signature_cpu,
        atlas_keys=keys_cpu,
        atlas_values=values_cpu,
        audit=audit,
    )


__all__ = [
    "FixedPrefixAtlasAudit",
    "FixedPrefixAtlasOutput",
    "compile_fixed_scene_atlas",
    "tensor_sha256",
    "validate_probe_bank",
]

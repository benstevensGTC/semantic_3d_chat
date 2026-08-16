"""Source-only V89 strict-runtime topology skeleton.

V89 has no sealed final state or passing gate yet.  This module therefore has
no experiment hashes, artifact paths, runtime YAML, packaging, smoke, or
promotion function.  It only validates that a future post-gate wrapper extends
the exact frozen ten-bank V88 topology with one disjoint layer-27 ``o_proj``
bank and remains bound to scene one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from semantic_3d_chat.chat.strict_scene1_stack_contract import (
    SCENE_ID,
    FrozenRuntimeBankContract,
    StrictScene1StackContract,
    extend_stack_contract,
    validate_strict_scene1_stack,
)

V89_BANK: Final[str] = "v89_scene1_retention_bridge"
V89_TARGET: Final[str] = "model.language_model.layers.27.self_attn.o_proj"
V89_RANK: Final[int] = 8
V89_ALPHA: Final[float] = 16.0
V89_PARAMETER_COUNT: Final[int] = 28_672
V88_PARENT_PARAMETER_COUNT: Final[int] = 843_776
V89_FINAL_PARAMETER_COUNT: Final[int] = 872_448
V88_PARENT_BANKS: Final[tuple[str, ...]] = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
    "extension_v28_stage_b_query",
    "extension_v30_joint_pair_query",
    "v85_strict_multiscene_bridge",
    "v86_scene1_demo_bridge",
    "v87_scene1_balanced_bridge",
    "v88_scene1_augmented_bridge",
)
V89_FINAL_BANKS: Final[tuple[str, ...]] = V88_PARENT_BANKS + (V89_BANK,)


def build_v89_stack_contract(
    *,
    parent: StrictScene1StackContract,
    final_state_sha256: str,
) -> StrictScene1StackContract:
    """Construct only after a versioned wrapper authenticates a final state."""

    if (
        parent.bank_order != V88_PARENT_BANKS
        or len(parent.banks) != 10
        or parent.expected_total_parameter_count != V88_PARENT_PARAMETER_COUNT
    ):
        raise ValueError("V89 requires the exact frozen ten-bank V88 parent")
    parent_targets = {
        target for bank in parent.banks for target in bank.target_modules
    }
    if V89_TARGET in parent_targets:
        raise ValueError("V89 fresh target overlaps a frozen parent target")
    fresh = FrozenRuntimeBankContract(
        name=V89_BANK,
        target_modules=(V89_TARGET,),
        rank=V89_RANK,
        alpha=V89_ALPHA,
        parameter_count=V89_PARAMETER_COUNT,
        state_sha256=final_state_sha256,
    )
    result = extend_stack_contract(parent, (fresh,))
    if (
        result.bank_order != V89_FINAL_BANKS
        or len(result.banks) != 11
        or result.expected_total_parameter_count != V89_FINAL_PARAMETER_COUNT
    ):
        raise RuntimeError("V89 final frozen stack identity changed")
    return result


def validate_v89_runtime_stack(
    *,
    scene_id: str,
    runtime_config: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    parent: StrictScene1StackContract,
    final_state_sha256: str,
) -> None:
    """Validate one in-memory future candidate; never writes or loads a model."""

    final = build_v89_stack_contract(
        parent=parent,
        final_state_sha256=final_state_sha256,
    )
    validate_strict_scene1_stack(
        scene_id=scene_id,
        runtime_config=runtime_config,
        checkpoint_metadata=checkpoint_metadata,
        contract=final,
    )


__all__ = [
    "SCENE_ID",
    "V88_PARENT_BANKS",
    "V88_PARENT_PARAMETER_COUNT",
    "V89_ALPHA",
    "V89_BANK",
    "V89_FINAL_BANKS",
    "V89_FINAL_PARAMETER_COUNT",
    "V89_PARAMETER_COUNT",
    "V89_RANK",
    "V89_TARGET",
    "build_v89_stack_contract",
    "validate_v89_runtime_stack",
]

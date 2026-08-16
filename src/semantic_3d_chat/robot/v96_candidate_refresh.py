"""Question-free V96 scene-memory refresh for the embodied/MCP action path.

This module is an explicit-candidate bridge, not a promotion.  The caller must
first obtain :class:`V96CandidateAuthorization` from the isolated V96
authorizer.  The same frozen ten-bank Gemma instance is then retained while a
sanitized numeric map update rebuilds the 258-token base scene prefix and
compiles a new complete 738-token V81 memory *before* the map transaction is
committed.  No user question is an argument to any compiler method.

The existing MCP server remains numeric-only.  Its
``scene_control_signature_sha256`` receipt field binds the current 738-token
memory, while ``scene_prefix_sha256`` binds the reconstructed 258-token base.
This module does not add a text-returning MCP tool and does not read oracle,
QA, render, feature, training, prediction, or scorer trees.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.v96_explicit_candidate_runtime import (
    EXPECTED_BANKS,
    EXTENSION_PARAMETER_COUNT,
    V96CandidateAuthorization,
    V96ExplicitCandidateChatRuntime,
    validate_v96_scene_memory_contract,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.runtime_refresh import (
    RefreshingEmbodiedChatRuntime,
    build_refreshing_embodied_runtime,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    FIXED_MEMORY_TOKENS,
    HIDDEN_SIZE,
    TENSOR_NAME,
    LoadedV81SceneMemory,
    build_v81_scene_memory_metadata,
)

_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_OPAQUE_SCENE: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {
        "oracle",
        "qa",
        "rendered",
        "features",
        "training",
        "scorer",
        "scorer_only",
        "scorer-only",
        "predictions",
        "questions",
    }
)
_PROBE_SAFE_METADATA: Final[dict[str, str]] = {
    "artifact": "v75_fixed_atlas_numeric_probe_bank_v1",
    "schema_version": "1",
    "tensor_name": "probe_embeddings",
    "questions_or_answers_serialized": "false",
    "answer_codebook_serialized": "false",
    "environmental_text_serialized": "false",
    "runtime_promotion_authorized": "false",
}
_MEMORY_SAFE_METADATA: Final[dict[str, str]] = {
    "artifact": "v81_fixed_continuous_scene_memory_v1",
    "schema_version": "81",
    "tensor_name": TENSOR_NAME,
    "environmental_text_inputs": "false",
    "questions_or_answers_serialized": "false",
}
BRIDGE_HOOK_ARTIFACT: Final[str] = "gemma4_v96_explicit_candidate_mcp_bridge_v1"
BRIDGE_HOOK_MODE: Final[str] = "explicit_candidate_only_not_default"
_RELEASE_VERIFY_PHASE: Final[str] = "v96_strict_runtime_release_verified"
_RELEASE_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "phase",
        "passed",
        "candidate_fingerprint_sha256",
        "candidate_checkpoint_sha256",
        "candidate_adapter_sha256",
        "deferred_final_evidence_sha256",
        "runtime_smoke_sha256",
        "release_report_sha256",
        "release_checkpoint_sha256",
        "release_adapter_sha256",
        "v95_state_sha256",
        "v96_state_sha256",
        "runtime_implementation_inventory_sha256",
        "scene_ids",
        "checks",
    }
)
_RELEASE_RECEIPT_HASH_FIELDS: Final[frozenset[str]] = frozenset(
    _RELEASE_RECEIPT_FIELDS
    - {"phase", "passed", "scene_ids", "checks"}
)
_REQUIRED_RELEASE_CHECKS: Final[frozenset[str]] = frozenset(
    {
        "release_report_identity",
        "release_report_promoted",
        "exact_two_file_checkpoint",
        "checkpoint_fingerprint_matches_release",
        "adapter_byte_identical_to_smoked_candidate",
        "all_six_memory_tensor_files_byte_identical_to_candidate",
        "all_six_memories_bound_to_attested_prefixes",
        "all_six_runtime_maps_bound_to_smoked_bytes",
        "exact_ten_frozen_final_state_banks",
        "deferred_final_binding_exact",
        "runtime_smoke_binding_exact",
        "runtime_implementation_binding_exact",
        "release_implementation_binding_exact",
        "runtime_promotion_authorized",
        "candidate_checkpoint_identity_retained_in_smoke",
        "default_runtime_pointer_unchanged",
    }
)


@dataclass(frozen=True)
class V96CandidateMCPHook:
    """Immutable sanitized paths for the optional embodied candidate bridge."""

    candidate_hook: Path
    atlas_control_checkpoint: Path
    atlas_probe_bank: Path


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def load_v96_candidate_mcp_hook(path: str | Path) -> V96CandidateMCPHook:
    """Load a declaration-only hook; this never authenticates or loads a model."""

    source = _safe_runtime_path(path, purpose="V96 MCP candidate hook")
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "artifact",
        "mode",
        "candidate_hook",
        "atlas_control_checkpoint",
        "atlas_probe_bank",
        "action_transport",
        "numeric_tool_outputs_only",
        "full_memory_recompiled_before_map_commit",
        "require_authenticated_pass_evidence",
        "require_authenticated_deferred_final_pass",
        "require_authenticated_runtime_leakage_smoke",
        "require_promoted_runtime_release_verification",
        "require_explicit_candidate_flag",
        "default_runtime_pointer_modified",
        "runtime_promotion_authorized",
        "direct_v96_answer_robot_tokens_authenticated",
        "environmental_text_inputs",
    }
    if not isinstance(raw, Mapping) or set(raw) != {"v96_candidate_mcp_bridge"}:
        raise ValueError("V96 MCP hook must contain one bridge mapping")
    hook = raw["v96_candidate_mcp_bridge"]
    if not isinstance(hook, Mapping) or set(hook) != expected_fields:
        raise ValueError("V96 MCP bridge hook fields changed")
    if (
        hook.get("schema_version") != 96
        or hook.get("artifact") != BRIDGE_HOOK_ARTIFACT
        or hook.get("mode") != BRIDGE_HOOK_MODE
        or hook.get("action_transport") != "official_python_mcp_sdk_stdio"
        or hook.get("numeric_tool_outputs_only") is not True
        or hook.get("full_memory_recompiled_before_map_commit") is not True
        or hook.get("require_authenticated_pass_evidence") is not True
        or hook.get("require_authenticated_deferred_final_pass") is not True
        or hook.get("require_authenticated_runtime_leakage_smoke") is not True
        or hook.get("require_promoted_runtime_release_verification") is not True
        or hook.get("require_explicit_candidate_flag") is not True
        or hook.get("default_runtime_pointer_modified") is not False
        or hook.get("runtime_promotion_authorized") is not False
        or hook.get("direct_v96_answer_robot_tokens_authenticated") is not False
        or hook.get("environmental_text_inputs") != []
    ):
        raise ValueError("V96 MCP bridge hook is not a safe unpromoted contract")
    paths = tuple(
        _safe_runtime_path(str(hook[field]), purpose=f"V96 MCP hook {field}")
        for field in (
            "candidate_hook",
            "atlas_control_checkpoint",
            "atlas_probe_bank",
        )
    )
    return V96CandidateMCPHook(*paths)


def run_isolated_v96_release_verification(
    *, timeout_seconds: float = 600.0
) -> dict[str, Any]:
    """Require the promoted deferred-final release in a model-free child.

    The child owns all evaluation/report reads.  This MCP process receives only
    a strict boolean verification receipt and refuses to inspect protected
    evidence itself.  Requiring the exact check inventory makes a weaker or
    older known-development-only result insufficient.
    """

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0.0 < float(timeout_seconds) <= 3600.0
    ):
        raise ValueError("V96 release verification timeout is invalid")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_3d_chat.evaluation.v96_strict_runtime_release",
            "verify",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout_seconds),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "strict V96 release verification failed"
        raise RuntimeError(detail)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V96 release receipt field: {key}")
            result[key] = value
        return result

    try:
        receipt = json.loads(
            completed.stdout,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as error:
        raise ValueError("V96 release verifier returned invalid JSON") from error
    if not isinstance(receipt, dict) or set(receipt) != _RELEASE_RECEIPT_FIELDS:
        raise ValueError("V96 release verifier returned an unexpected receipt")
    checks = receipt.get("checks")
    scene_ids = tuple(f"scene_{index:06d}" for index in range(25, 31))
    if (
        receipt.get("phase") != _RELEASE_VERIFY_PHASE
        or receipt.get("passed") is not True
        or any(not _is_sha256(receipt.get(field)) for field in _RELEASE_RECEIPT_HASH_FIELDS)
        or receipt.get("scene_ids") != list(scene_ids)
        or not isinstance(checks, Mapping)
        or set(checks) != _REQUIRED_RELEASE_CHECKS
        or any(value is not True for value in checks.values())
    ):
        raise ValueError(
            "V96 MCP requires the exact deferred-final and runtime-leakage release PASS"
        )
    return {
        "phase": _RELEASE_VERIFY_PHASE,
        "passed": True,
        "check_count": len(_REQUIRED_RELEASE_CHECKS),
        "deferred_final_binding_exact": True,
        "runtime_smoke_binding_exact": True,
        "promoted_runtime_release_verified": True,
        "candidate_fingerprint_sha256": receipt["candidate_fingerprint_sha256"],
        "deferred_final_evidence_sha256": receipt[
            "deferred_final_evidence_sha256"
        ],
        "runtime_smoke_sha256": receipt["runtime_smoke_sha256"],
        "release_checkpoint_sha256": receipt["release_checkpoint_sha256"],
        "release_adapter_sha256": receipt["release_adapter_sha256"],
        "v95_state_sha256": receipt["v95_state_sha256"],
        "v96_state_sha256": receipt["v96_state_sha256"],
        "runtime_implementation_inventory_sha256": receipt[
            "runtime_implementation_inventory_sha256"
        ],
        "scene_ids": list(scene_ids),
    }


class QuestionFreeV96MemoryCompiler(Protocol):
    """Scene-only compiler seam used transactionally by the map refresher."""

    source_control_checkpoint_sha256: str
    authenticated_control_sha256s: frozenset[str]
    source_probe_tensor_sha256: str

    def compile(
        self,
        base: StaticChatRuntime,
        *,
        prior_metadata: Mapping[str, Any],
    ) -> LoadedV81SceneMemory: ...


def _safe_runtime_path(path: str | Path, *, purpose: str) -> Path:
    raw = Path(path).expanduser()
    rooted = raw if raw.is_absolute() else PROJECT_ROOT / raw
    source = Path(os.path.abspath(rooted))
    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} path contains a symbolic link: {current}")
    forbidden = _FORBIDDEN_COMPONENTS.intersection(
        component.casefold() for component in source.parts
    )
    if forbidden:
        raise ValueError(f"{purpose} entered forbidden runtime data: {sorted(forbidden)}")
    return source


def _sha256_file(path: Path, audit: FileAccessAudit | None = None) -> str:
    if audit is not None:
        audit.record(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path, audit: FileAccessAudit | None = None) -> dict[str, Any]:
    if audit is not None:
        audit.record(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V96 compiler metadata field: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict):
        raise TypeError("V96 compiler metadata must be a JSON object")
    return value


def _checkpoint_fingerprint(
    root: Path, audit: FileAccessAudit | None = None
) -> str:
    expected = ("control.safetensors", "runtime_metadata.json")
    if root.is_symlink() or not root.is_dir() or sorted(
        item.name for item in root.iterdir()
    ) != sorted(expected):
        raise ValueError("V96 compiler control checkpoint is not runtime-minimal")
    entries: list[dict[str, Any]] = []
    for name in expected:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("V96 compiler checkpoint entries must be regular files")
        entries.append(
            {
                "name": name,
                "sha256": _sha256_file(path, audit),
                "size_bytes": path.stat().st_size,
            }
        )
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_numeric_probe_bank(
    root: str | Path,
    *,
    audit: FileAccessAudit | None,
) -> tuple[torch.Tensor, str]:
    source = _safe_runtime_path(root, purpose="V96 numeric probe bank")
    if source.is_symlink() or not source.is_dir() or {
        item.name for item in source.iterdir()
    } != {"probes.safetensors", "runtime_metadata.json"}:
        raise ValueError("V96 numeric probe bank inventory changed")
    tensor_path = source / "probes.safetensors"
    metadata_path = source / "runtime_metadata.json"
    if any(path.is_symlink() or not path.is_file() for path in (tensor_path, metadata_path)):
        raise ValueError("V96 numeric probe bank entries must be regular files")
    metadata = _strict_json(metadata_path, audit)
    if (
        metadata.get("schema_version") != 1
        or metadata.get("artifact") != "v75_fixed_atlas_numeric_probe_bank_v1"
        or metadata.get("probe_count") != 96
        or metadata.get("hidden_size") != HIDDEN_SIZE
        or metadata.get("dtype") != "torch.float32"
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("answer_codebook_serialized") is not False
        or metadata.get("environmental_text_serialized") is not False
        or metadata.get("oracle_loaded") is not False
        or metadata.get("official_validation_loaded") is not False
        or metadata.get("official_test_loaded") is not False
        or metadata.get("deferred_final_loaded") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V96 numeric probe bank contains an unsafe runtime contract")
    for field in ("probe_file_sha256", "probe_tensor_sha256"):
        if not isinstance(metadata.get(field), str) or _SHA256.fullmatch(
            str(metadata[field])
        ) is None:
            raise ValueError(f"V96 numeric probe {field} is invalid")
    if _sha256_file(tensor_path, audit) != metadata["probe_file_sha256"]:
        raise ValueError("V96 numeric probe file bytes changed")
    if audit is not None:
        audit.record(tensor_path)
    with safe_open(str(tensor_path), framework="pt", device="cpu") as archive:
        if set(archive.keys()) != {"probe_embeddings"} or archive.metadata() != (
            _PROBE_SAFE_METADATA
        ):
            raise ValueError("V96 numeric probe tensor contract changed")
    probes = load_file(str(tensor_path), device="cpu")["probe_embeddings"]
    probes = probes.detach().float().contiguous()
    if (
        tuple(probes.shape) != (96, HIDDEN_SIZE)
        or not bool(torch.isfinite(probes).all())
        or bool((probes.norm(dim=-1) <= 1e-8).any())
        or tensor_sha256(probes) != metadata["probe_tensor_sha256"]
    ):
        raise ValueError("V96 numeric probe tensor identity changed")
    return probes, str(metadata["probe_tensor_sha256"])


class V75QuestionFreeV96MemoryCompiler:
    """Exact V75 all-probe compiler retained for transactional robot scans."""

    def __init__(
        self,
        controller: DenseFullSceneContinuousControlV75,
        probes: torch.Tensor,
        *,
        source_control_checkpoint_sha256: str,
        source_probe_tensor_sha256: str,
        additional_control_sha256s: frozenset[str] = frozenset(),
    ) -> None:
        if type(controller) is not DenseFullSceneContinuousControlV75:
            raise TypeError("V96 embodied compilation requires the exact V75 controller")
        for value, label in (
            (source_control_checkpoint_sha256, "control checkpoint"),
            (source_probe_tensor_sha256, "probe tensor"),
            *((value, "additional control identity") for value in additional_control_sha256s),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"V96 {label} hash must be lowercase SHA-256")
        if (
            tuple(probes.shape) != (96, HIDDEN_SIZE)
            or probes.dtype != torch.float32
            or not bool(torch.isfinite(probes).all())
            or tensor_sha256(probes) != source_probe_tensor_sha256
        ):
            raise ValueError("V96 compiler probe tensor differs from its binding")
        self.controller = controller.eval()
        for parameter in self.controller.parameters():
            parameter.requires_grad_(False)
        self.probes = probes.detach().cpu().float().contiguous()
        self.source_control_checkpoint_sha256 = source_control_checkpoint_sha256
        self.authenticated_control_sha256s = frozenset(
            {source_control_checkpoint_sha256, *additional_control_sha256s}
        )
        self.source_probe_tensor_sha256 = source_probe_tensor_sha256

    @classmethod
    def load(
        cls,
        control_checkpoint: str | Path,
        probe_bank: str | Path,
        *,
        device: torch.device | str,
        audit: FileAccessAudit | None = None,
    ) -> V75QuestionFreeV96MemoryCompiler:
        """Load only sanitized learned compiler parameters and numeric probes."""

        # This chat-owned loader validates the sealed checkpoint and does not
        # open its adjacent training artifacts.  The import is delayed so
        # merely importing the V96 robot bridge performs no checkpoint work.
        from semantic_3d_chat.chat.question_control_runtime import _load_control_head

        checkpoint = _safe_runtime_path(
            control_checkpoint, purpose="V96 question-free control checkpoint"
        )
        control_hash = _checkpoint_fingerprint(checkpoint, audit)
        controller, metadata = _load_control_head(
            checkpoint,
            hidden_size=HIDDEN_SIZE,
            device=torch.device(device),
            audit=audit,
        )
        if (
            type(controller) is not DenseFullSceneContinuousControlV75
            or metadata.get("environmental_text_inputs") != []
            or metadata.get("training_answers_runtime_loaded") is not False
            or metadata.get("answer_text_runtime_loaded") is not False
            or metadata.get("answer_class_codebook_runtime_loaded") is not False
            or metadata.get("teacher_cache_runtime_loaded") is not False
            or metadata.get("oracle_runtime_loaded") is not False
            or metadata.get("question_or_answer_text_serialized") is not False
            or metadata.get("question_dependent_scene_retrieval") is not False
            or metadata.get("latent_selection_or_top_k_used") is not False
        ):
            raise ValueError("V96 V75 compiler checkpoint is not runtime-sanitized")
        probes, probe_hash = _load_numeric_probe_bank(probe_bank, audit=audit)
        return cls(
            controller,
            probes,
            # Some sealed V81 exports bind the complete two-file checkpoint;
            # later runtime releases bind its exact weights digest.  Both are
            # authenticated here and the refreshed metadata preserves the
            # convention used by the initial memory.
            source_control_checkpoint_sha256=str(metadata["weights_sha256"]),
            source_probe_tensor_sha256=probe_hash,
            additional_control_sha256s=frozenset({control_hash}),
        )

    @torch.inference_mode()
    def compile(
        self,
        base: StaticChatRuntime,
        *,
        prior_metadata: Mapping[str, Any],
    ) -> LoadedV81SceneMemory:
        """Compile all scene latents and all probes without accepting user text."""

        if _OPAQUE_SCENE.fullmatch(base.scene_id) is None:
            raise ValueError("V96 compiler scene identity is not opaque")
        if (
            prior_metadata.get("scene_id") != base.scene_id
            or prior_metadata.get("source_control_checkpoint_sha256")
            not in self.authenticated_control_sha256s
            or prior_metadata.get("source_probe_tensor_sha256")
            != self.source_probe_tensor_sha256
            or prior_metadata.get("environmental_text_inputs") != []
            or prior_metadata.get("questions_or_answers_serialized") is not False
            or prior_metadata.get("oracle_loaded") is not False
        ):
            raise ValueError("V96 compiler sources differ from the authorized initial memory")
        compiled = compile_fixed_scene_atlas_v75_v2(
            base.scene_prefix,
            self.controller,
            self.probes,
        )
        memory = compiled.scene_prefix.detach().to(
            device=base.scene_prefix.device,
            dtype=torch.bfloat16,
        ).contiguous()
        audit = compiled.audit
        if (
            tuple(memory.shape) != (1, FIXED_MEMORY_TOKENS, HIDDEN_SIZE)
            or audit.base_scene_prefix_sha256 != prefix_sha256(base.scene_prefix)
            or not audit.base_environment_tokens_preserved_exactly
            or not audit.atlas_key_value_tokens_preserved_exactly
            or not audit.every_probe_processed
            or not audit.complete_atlas_included
            or not audit.compiled_before_user_question
            or audit.user_question_inputs_used_for_compilation
            or audit.question_dependent_scene_processing
            or audit.question_dependent_retrieval
            or audit.semantic_or_spatial_top_k_selection
            or audit.environmental_text_inputs
        ):
            raise RuntimeError("V96 question-free full-memory compilation failed")

        serialized = save(
            {TENSOR_NAME: memory.detach().cpu()}, metadata=_MEMORY_SAFE_METADATA
        )
        metadata = build_v81_scene_memory_metadata(
            memory,
            scene_id=base.scene_id,
            tensor_file_sha256=hashlib.sha256(serialized).hexdigest(),
            source_base_checkpoint_sha256=str(
                prior_metadata["source_base_checkpoint_sha256"]
            ),
            runtime_config_sha256=str(prior_metadata["runtime_config_sha256"]),
            source_control_checkpoint_sha256=(
                str(prior_metadata["source_control_checkpoint_sha256"])
            ),
            source_probe_tensor_sha256=self.source_probe_tensor_sha256,
        )
        loaded = LoadedV81SceneMemory(
            root=(
                PROJECT_ROOT
                / "data_gemma4/runtime/ephemeral_scene_memories/v96_embodied"
                / base.scene_id
                / metadata["canonical_prefix_sha256"]
            ),
            memory=memory,
            metadata=metadata,
        )
        validate_v96_scene_memory_contract(scene_id=base.scene_id, loaded=loaded)
        return loaded


def _rebuild_base(previous: StaticChatRuntime, map_data: MapTensorData) -> StaticChatRuntime:
    """Re-run the unchanged full-map tokenizer while retaining all model weights."""

    return StaticChatRuntime(
        config=previous.config,
        scene_id=previous.scene_id,
        checkpoint_path=previous.checkpoint_path,
        checkpoint_metadata=previous.checkpoint_metadata,
        language=previous.language,
        map_data=map_data,
        scene_model=previous.scene_model,
        dense_aligner=previous.dense_aligner,
        dense_sidecar_adapter=previous.dense_sidecar_adapter,
        block_cross_residual=previous.block_cross_residual,
        global_scene_residual=previous.global_scene_residual,
        signed_x_scene_residual=previous.signed_x_scene_residual,
        composer=previous.composer,
        grounding=previous.grounding,
        warnings=previous.warnings,
        generation_function=previous._generation_function,
    )


class V96CandidateRuntimeBuilder:
    """Callable map-refresher adapter pinned to one authenticated V96 candidate."""

    def __init__(
        self,
        authorization: V96CandidateAuthorization,
        compiler: QuestionFreeV96MemoryCompiler,
        initial_runtime: V96ExplicitCandidateChatRuntime,
        *,
        allow_explicit_candidate: bool,
    ) -> None:
        if allow_explicit_candidate is not True:
            raise ValueError(
                "V96 embodied refresh requires explicit acknowledgement of the "
                "unpromoted candidate"
            )
        authorization.validate()
        self.authorization = authorization
        self.compiler = compiler
        self._validate_runtime(initial_runtime, require_unanswered=True)
        metadata = initial_runtime.scene_memory_metadata
        if (
            metadata.get("source_control_checkpoint_sha256")
            not in compiler.authenticated_control_sha256s
            or metadata.get("source_probe_tensor_sha256")
            != compiler.source_probe_tensor_sha256
        ):
            raise ValueError("V96 embodied compiler differs from initial scene memory")

    def _validate_runtime(
        self,
        runtime: V96ExplicitCandidateChatRuntime,
        *,
        require_unanswered: bool,
    ) -> None:
        if not isinstance(runtime, V96ExplicitCandidateChatRuntime):
            raise TypeError("V96 embodied refresh requires the explicit candidate runtime")
        runtime.authorization.validate()
        if runtime.authorization.to_payload() != self.authorization.to_payload():
            raise ValueError("V96 embodied runtime authorization changed")
        banks = runtime.extension_banks
        if (
            banks.bank_names != EXPECTED_BANKS[-3:]
            or banks.parameter_count != EXTENSION_PARAMETER_COUNT
            or banks.trainable_parameter_count != 0
        ):
            raise ValueError("V96 embodied runtime is not the exact frozen ten-bank stack")
        if require_unanswered and runtime.questions_answered != 0:
            raise ValueError("V96 refreshed memory must be bound before any question")
        runtime.assert_prefix_unchanged()
        if runtime.current_prefix_hash() != runtime.scene_prefix_hash:
            raise RuntimeError("V96 embodied scene memory changed")

    def __call__(
        self,
        previous: Any,
        map_data: MapTensorData,
    ) -> V96ExplicitCandidateChatRuntime:
        self._validate_runtime(previous, require_unanswered=False)
        base = _rebuild_base(previous.base, map_data)
        loaded = self.compiler.compile(
            base,
            prior_metadata=previous.scene_memory_metadata,
        )
        refreshed = V96ExplicitCandidateChatRuntime(
            base,
            loaded,
            authorization=self.authorization,
            extension_banks=previous.extension_banks,
        )
        self._validate_runtime(refreshed, require_unanswered=True)
        return refreshed


def build_v96_candidate_refreshing_embodied_runtime(
    config: dict[str, Any],
    scene_id: str,
    *,
    checkpoint: str | Path,
    authorization: V96CandidateAuthorization,
    chat_runtime: V96ExplicitCandidateChatRuntime,
    memory_compiler: QuestionFreeV96MemoryCompiler,
    allow_explicit_candidate: bool = False,
    **kwargs: Any,
) -> RefreshingEmbodiedChatRuntime:
    """Use V96 in the existing direct robot/MCP runtime without promotion."""

    authorization.validate()
    builder = V96CandidateRuntimeBuilder(
        authorization,
        memory_compiler,
        chat_runtime,
        allow_explicit_candidate=allow_explicit_candidate,
    )
    return build_refreshing_embodied_runtime(
        config,
        scene_id,
        checkpoint=checkpoint,
        chat_runtime=chat_runtime,
        runtime_builder=builder,
        **kwargs,
    )


__all__ = [
    "BRIDGE_HOOK_ARTIFACT",
    "BRIDGE_HOOK_MODE",
    "QuestionFreeV96MemoryCompiler",
    "V75QuestionFreeV96MemoryCompiler",
    "V96CandidateMCPHook",
    "V96CandidateRuntimeBuilder",
    "build_v96_candidate_refreshing_embodied_runtime",
    "load_v96_candidate_mcp_hook",
    "run_isolated_v96_release_verification",
]

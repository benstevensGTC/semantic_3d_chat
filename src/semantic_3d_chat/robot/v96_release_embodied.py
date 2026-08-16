"""Promoted-V96-only embodied runtime with question-free map refresh.

This is a downstream runtime surface.  It deliberately does not participate in
V96 known-development selection, deferred-final scoring, source sealing, or
promotion.  A model-free child must first verify the already-promoted release.
Only then may this module open the standalone runtime config, frozen checkpoint,
numeric scene memory, numeric map, vision weights, or Blender asset.

Each accepted RGB-D update is compiled transactionally into the complete
738-token V96 memory before the staged semantic map becomes visible.  No user
question is accepted by the compiler and no environmental text is introduced.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.chat.v96_strict_multiscene_runtime import (
    PROMOTED_DECISION,
    V96StrictMultisceneChatRuntime,
)
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.robot.runtime_refresh import (
    RefreshingEmbodiedChatRuntime,
    build_refreshing_embodied_runtime,
)
from semantic_3d_chat.robot.v96_candidate_refresh import (
    QuestionFreeV96MemoryCompiler,
    V75QuestionFreeV96MemoryCompiler,
    run_isolated_v96_release_verification,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    FIXED_MEMORY_TOKENS,
    HIDDEN_SIZE,
)

RELEASE_SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(25, 31)
)
RELEASE_RUNTIME_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/gemma4_v96_strict_multiscene.yaml"
)
RELEASE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT
    / "data_gemma4/runtime/checkpoints/gemma4_v96_strict_multiscene_release_v1"
)
RELEASE_MEMORY_ROOT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v96"
)
RELEASE_MAP_ROOT: Final[Path] = PROJECT_ROOT / "data_gemma4/runtime/maps/v96"
DEFAULT_EMBODIED_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/embodied_live.yaml"
)
DEFAULT_COMPILER_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT
    / "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
)
DEFAULT_PROBE_BANK: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/probe_bank"
)
DEFAULT_ROBOT_STATE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/checkpoints/robot_state_numeric_v1"
)
V75_CONTROL_FILE_SHA256S: Final[dict[str, str]] = {
    "control.safetensors": "bb112f42ca5df71b88b4cd7721b9107f9be9b0dc01b612a4ace6212548da669c",
    "runtime_metadata.json": "a45a192d27336329580612524d43f71f08e3f472e5fe833747ffc1395e2aa2be",
}
V75_PROBE_FILE_SHA256S: Final[dict[str, str]] = {
    "probes.safetensors": "fb32c687dd787f108fab03e9745eefb2273891c2be990d0acf50ca111eb637e8",
    "runtime_metadata.json": "3e736940f4c83b55e96aa5e36f6774fd007454508722f5b25ddc44f298c2518d",
}
ROBOT_STATE_FILE_SHA256S: Final[dict[str, str]] = {
    "state.safetensors": "5d6aa13208264e0a99755d84e8f68b7727249b274c460e9d4e26541cd8e46938",
    "runtime_metadata.json": "c48b8748dbde04f2c9294321974b1b13be2d77083970f051ba1c11a9b42d1985",
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RELEASE_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "phase",
        "passed",
        "check_count",
        "deferred_final_binding_exact",
        "runtime_smoke_binding_exact",
        "promoted_runtime_release_verified",
        "candidate_fingerprint_sha256",
        "deferred_final_evidence_sha256",
        "runtime_smoke_sha256",
        "release_checkpoint_sha256",
        "release_adapter_sha256",
        "v95_state_sha256",
        "v96_state_sha256",
        "runtime_implementation_inventory_sha256",
        "scene_ids",
    }
)
_RELEASE_HASH_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "candidate_fingerprint_sha256",
        "deferred_final_evidence_sha256",
        "runtime_smoke_sha256",
        "release_checkpoint_sha256",
        "release_adapter_sha256",
        "v95_state_sha256",
        "v96_state_sha256",
        "runtime_implementation_inventory_sha256",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_nonsymlink(path: str | Path, *, purpose: str) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    source = Path(os.path.abspath(rooted))
    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} cannot contain symbolic links")
    return source


def _require_exact_artifact(
    root: str | Path,
    expected: Mapping[str, str],
    *,
    canonical_root: Path,
    purpose: str,
) -> Path:
    source = _absolute_nonsymlink(root, purpose=purpose)
    canonical = _absolute_nonsymlink(canonical_root, purpose=purpose)
    if source != canonical or not source.is_dir():
        raise ValueError(f"{purpose} must use the fixed local release path")
    entries = {entry.name for entry in source.iterdir()}
    if entries != set(expected):
        raise ValueError(f"{purpose} file inventory changed")
    for name, expected_sha256 in expected.items():
        path = source / name
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected_sha256:
            raise ValueError(f"{purpose} bytes changed: {name}")
    return source


def validate_promoted_v96_release_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the small hash-only receipt returned by the isolated child."""

    value = dict(receipt)
    if set(value) != _RELEASE_RECEIPT_FIELDS:
        raise ValueError("Promoted V96 release receipt fields changed")
    if (
        value.get("phase") != "v96_strict_runtime_release_verified"
        or value.get("passed") is not True
        or value.get("promoted_runtime_release_verified") is not True
        or value.get("deferred_final_binding_exact") is not True
        or value.get("runtime_smoke_binding_exact") is not True
        or isinstance(value.get("check_count"), bool)
        or not isinstance(value.get("check_count"), int)
        or int(value["check_count"]) < 1
        or value.get("scene_ids") != list(RELEASE_SCENE_IDS)
        or any(
            not isinstance(value.get(field), str)
            or _SHA256.fullmatch(str(value[field])) is None
            for field in _RELEASE_HASH_FIELDS
        )
    ):
        raise ValueError("Embodied V96 requires the exact promoted release PASS")
    return value


def _rebuild_static_base(
    previous: StaticChatRuntime,
    map_data: MapTensorData,
) -> StaticChatRuntime:
    """Run the unchanged all-voxel tokenizer with already-loaded weights."""

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


class PromotedV96RuntimeBuilder:
    """Recompile a complete promoted V96 memory before each map commit."""

    def __init__(
        self,
        release_receipt: Mapping[str, Any],
        compiler: QuestionFreeV96MemoryCompiler,
        initial_runtime: V96StrictMultisceneChatRuntime,
    ) -> None:
        self.release_receipt = validate_promoted_v96_release_receipt(release_receipt)
        self.compiler = compiler
        self._validate_runtime(initial_runtime, require_unanswered=True)
        metadata = initial_runtime.scene_memory_metadata
        if (
            metadata.get("source_control_checkpoint_sha256")
            not in compiler.authenticated_control_sha256s
            or metadata.get("source_probe_tensor_sha256")
            != compiler.source_probe_tensor_sha256
        ):
            raise ValueError("Promoted V96 memory and question-free compiler differ")

    def _validate_runtime(
        self,
        runtime: V96StrictMultisceneChatRuntime,
        *,
        require_unanswered: bool,
    ) -> None:
        if not isinstance(runtime, V96StrictMultisceneChatRuntime):
            raise TypeError("Embodied V96 requires the strict promoted runtime")
        runtime.assert_prefix_unchanged()
        provenance = runtime.release_provenance
        if (
            runtime.scene_id not in RELEASE_SCENE_IDS
            or runtime.runtime_package_mode != "promoted"
            or runtime.runtime_promotion_authorized is not True
            or provenance.get("promotion_decision") != PROMOTED_DECISION
            or provenance.get("candidate_fingerprint_sha256")
            != self.release_receipt["candidate_fingerprint_sha256"]
            or runtime.v95_state_sha256 != self.release_receipt["v95_state_sha256"]
            or runtime.v96_state_sha256 != self.release_receipt["v96_state_sha256"]
            or tuple(runtime.fixed_scene_memory.shape)
            != (1, FIXED_MEMORY_TOKENS, HIDDEN_SIZE)
            or runtime.current_prefix_hash() != runtime.scene_prefix_hash
            or (require_unanswered and runtime.questions_answered != 0)
        ):
            raise ValueError("Promoted V96 embodied runtime contract changed")

    def __call__(
        self,
        previous: Any,
        map_data: MapTensorData,
    ) -> V96StrictMultisceneChatRuntime:
        self._validate_runtime(previous, require_unanswered=False)
        base = _rebuild_static_base(previous.base, map_data)
        loaded = self.compiler.compile(
            base,
            prior_metadata=previous.scene_memory_metadata,
        )
        refreshed = V96StrictMultisceneChatRuntime(base, loaded)
        self._validate_runtime(refreshed, require_unanswered=True)
        return refreshed


def _embodied_config_with_release_maps(
    embodied_config: str | Path,
    *,
    release_map_root: str | Path,
    audit: FileAccessAudit | None,
) -> dict[str, Any]:
    if audit is not None:
        source = Path(embodied_config).expanduser()
        audit.record(source if source.is_absolute() else PROJECT_ROOT / source)
    config = copy.deepcopy(load_config(embodied_config))
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise TypeError("Embodied V96 config has no path mapping")
    paths["maps_root"] = str(Path(release_map_root).expanduser().resolve())
    return config


def build_promoted_v96_embodied_runtime(
    scene_id: str,
    *,
    runtime_asset: str | Path,
    embodied_config: str | Path = DEFAULT_EMBODIED_CONFIG,
    release_runtime_config: str | Path = RELEASE_RUNTIME_CONFIG,
    release_checkpoint: str | Path = RELEASE_CHECKPOINT,
    release_memory_root: str | Path = RELEASE_MEMORY_ROOT,
    release_map_root: str | Path = RELEASE_MAP_ROOT,
    compiler_checkpoint: str | Path = DEFAULT_COMPILER_CHECKPOINT,
    probe_bank: str | Path = DEFAULT_PROBE_BANK,
    robot_state_checkpoint: str | Path | None = None,
    persistent_map_path: str | Path | None = None,
    scan_output_directory: str | Path | None = None,
    audit: FileAccessAudit | None = None,
    release_verifier: Callable[[], Mapping[str, Any]] = (
        run_isolated_v96_release_verification
    ),
    local_files_only: bool = True,
) -> RefreshingEmbodiedChatRuntime:
    """Load the promoted static release and attach deterministic live sensing.

    The verifier call intentionally occurs before any path is opened or model
    object is constructed.  This function therefore fails closed while the
    static V96 release is absent or unpromoted.
    """

    if scene_id not in RELEASE_SCENE_IDS:
        raise ValueError("Promoted V96 embodied runtime is limited to release scenes")
    # Security boundary: do not reorder file/config/model work above this call.
    receipt = validate_promoted_v96_release_receipt(release_verifier())

    # The promoted embodied path is intentionally not a caller-selectable model
    # composition.  Exact paths plus bytes are fixed before any local model is
    # loaded; copied or replacement checkpoints are rejected even if their
    # metadata is superficially compatible.
    checkpoint = _absolute_nonsymlink(release_checkpoint, purpose="V96 release checkpoint")
    if checkpoint != _absolute_nonsymlink(
        RELEASE_CHECKPOINT, purpose="V96 release checkpoint"
    ):
        raise ValueError("V96 embodied runtime requires the canonical promoted checkpoint")
    checkpoint_sha256, checkpoint_files = checkpoint_fingerprint(checkpoint)
    if (
        {row["path"] for row in checkpoint_files}
        != {"adapter.safetensors", "runtime_metadata.json"}
        or checkpoint_sha256 != receipt["release_checkpoint_sha256"]
        or _sha256_file(checkpoint / "adapter.safetensors")
        != receipt["release_adapter_sha256"]
    ):
        raise ValueError("V96 embodied runtime checkpoint differs from its release receipt")
    compiler_checkpoint = _require_exact_artifact(
        compiler_checkpoint,
        V75_CONTROL_FILE_SHA256S,
        canonical_root=DEFAULT_COMPILER_CHECKPOINT,
        purpose="V96 question-free compiler checkpoint",
    )
    probe_bank = _require_exact_artifact(
        probe_bank,
        V75_PROBE_FILE_SHA256S,
        canonical_root=DEFAULT_PROBE_BANK,
        purpose="V96 numeric probe bank",
    )
    if robot_state_checkpoint is None:
        robot_state_checkpoint = DEFAULT_ROBOT_STATE_CHECKPOINT
    robot_state_checkpoint = _require_exact_artifact(
        robot_state_checkpoint,
        ROBOT_STATE_FILE_SHA256S,
        canonical_root=DEFAULT_ROBOT_STATE_CHECKPOINT,
        purpose="V96 numeric robot-state checkpoint",
    )
    for supplied, canonical, purpose in (
        (release_runtime_config, RELEASE_RUNTIME_CONFIG, "V96 release runtime config"),
        (release_memory_root, RELEASE_MEMORY_ROOT, "V96 release memory root"),
        (release_map_root, RELEASE_MAP_ROOT, "V96 release map root"),
        (embodied_config, DEFAULT_EMBODIED_CONFIG, "V96 embodied runtime config"),
    ):
        if _absolute_nonsymlink(supplied, purpose=purpose) != _absolute_nonsymlink(
            canonical, purpose=purpose
        ):
            raise ValueError(f"{purpose} is not caller-selectable")

    static_config = load_runtime_config(
        release_runtime_config,
        record_file=None if audit is None else audit.record,
    )
    config = _embodied_config_with_release_maps(
        embodied_config,
        release_map_root=release_map_root,
        audit=audit,
    )
    if (
        config.get("scene", {}).get("room_size_m")
        != static_config.get("scene", {}).get("room_size_m")
        or config.get("language", {}).get("model_id")
        != static_config.get("language", {}).get("model_id")
        or config.get("language", {}).get("revision")
        != static_config.get("language", {}).get("revision")
    ):
        raise ValueError("Embodied and promoted V96 static configs are incompatible")

    memory = Path(release_memory_root).expanduser().resolve() / scene_id
    chat_runtime = V96StrictMultisceneChatRuntime.load(
        static_config,
        scene_id,
        base_checkpoint=checkpoint,
        scene_memory=memory,
        audit=audit,
        local_files_only=local_files_only,
    )
    compiler = V75QuestionFreeV96MemoryCompiler.load(
        compiler_checkpoint,
        probe_bank,
        device=chat_runtime.base.language.device,
        audit=audit,
    )
    runtime_builder = PromotedV96RuntimeBuilder(receipt, compiler, chat_runtime)

    from semantic_3d_chat.robot.blender_scanner import SanitizedBlenderScanner

    resolution = tuple(int(value) for value in config["render"]["resolution"])
    output = (
        Path(scan_output_directory).expanduser().resolve()
        if scan_output_directory is not None
        else PROJECT_ROOT / "data_gemma4/runtime/v96_embodied/scans" / scene_id
    )
    persistent = (
        Path(persistent_map_path).expanduser().resolve()
        if persistent_map_path is not None
        else PROJECT_ROOT
        / "data_gemma4/runtime/v96_embodied/robot"
        / scene_id
        / "semantic_map.npz"
    )
    scanner = SanitizedBlenderScanner(
        scene_id,
        runtime_asset,
        resolution=resolution,
        horizontal_fov_degrees=float(config["render"]["horizontal_fov_degrees"]),
        engine=str(config["render"]["engine"]),
        samples=int(config["render"]["samples"]),
        max_depth_m=float(config["mapping"]["depth_max_m"]),
        output_directory=output,
    )
    return build_refreshing_embodied_runtime(
        config,
        scene_id,
        checkpoint=checkpoint,
        chat_runtime=chat_runtime,
        runtime_builder=runtime_builder,
        observation_scanner=scanner,
        robot_state_checkpoint=robot_state_checkpoint,
        persistent_map_path=persistent,
        audit=audit,
        local_files_only=local_files_only,
    )


__all__ = [
    "DEFAULT_COMPILER_CHECKPOINT",
    "DEFAULT_EMBODIED_CONFIG",
    "DEFAULT_PROBE_BANK",
    "DEFAULT_ROBOT_STATE_CHECKPOINT",
    "RELEASE_CHECKPOINT",
    "RELEASE_MAP_ROOT",
    "RELEASE_MEMORY_ROOT",
    "RELEASE_RUNTIME_CONFIG",
    "RELEASE_SCENE_IDS",
    "ROBOT_STATE_FILE_SHA256S",
    "V75_CONTROL_FILE_SHA256S",
    "V75_PROBE_FILE_SHA256S",
    "PromotedV96RuntimeBuilder",
    "build_promoted_v96_embodied_runtime",
    "validate_promoted_v96_release_receipt",
]

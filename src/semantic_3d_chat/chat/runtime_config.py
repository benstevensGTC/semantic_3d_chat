"""Strict, standalone configuration surface for production Gemma chat.

Training configurations intentionally contain supervision, split, and ablation
metadata.  A chat process must never parse those files merely to recover the
small numerical architecture surface needed for inference.  Runtime configs
therefore live in ``configs/runtime``, may not inherit another YAML file, and
are validated against an explicit allowlist before use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.scene_encoder.block_cross_residual import (
    block_cross_residual_settings,
)

RUNTIME_CONFIG_SCHEMA_VERSION: Final[int] = 1
RUNTIME_CONFIG_ROOT: Final[Path] = (PROJECT_ROOT / "configs" / "runtime").resolve()
APPROVED_SYSTEM_PROMPT: Final[str] = (
    "You answer using only the continuous 3D scene memory supplied before this "
    "conversation. Do not invent objects or relationships unsupported by the scene. "
    "If there is not enough evidence, answer unknown."
)

_ALLOWED_TOP_LEVEL = frozenset(
    {"runtime", "paths", "scene", "vision", "scene_encoder", "language", "training"}
)
_ALLOWED_PATH_KEYS = frozenset(
    {"data_root", "reports_root", "maps_root", "checkpoints_root"}
)
_REFERENCE_VIEWPOINT_KEYS = frozenset(
    {"position_m", "yaw_degrees", "pitch_degrees", "scan_view_count"}
)
_ALLOWED_SCENE_KEYS = frozenset({"room_size_m"})
_ALLOWED_VISION_KEYS = frozenset({"backend", "model_id", "revision"})
_ALLOWED_TRAINING_KEYS = frozenset(
    {"lora_learning_rate", "lora_weight_decay", "freeze_scene_adapter"}
)
_FORBIDDEN_ENVIRONMENT_FRAGMENTS = frozenset(
    {
        "oracle",
        "experiment",
        "category",
        "change_type",
        "chair",
        "bowl",
        "book",
        "picture",
        "frame",
        "cube",
        "table",
        "lamp",
        "door",
        "plant",
        "cabinet",
    }
)
_QA_TOKEN = re.compile(r"(?<![a-z0-9])qa(?![a-z0-9])")
_ALLOWED_LANGUAGE_KEYS = frozenset(
    {
        "backend",
        "model_id",
        "revision",
        "dtype",
        "scene_prefix_after_bos",
        "max_question_tokens",
        "max_answer_tokens",
        "system_prompt",
        "scene_boundary_mode",
        "gemma4_native_image_contract",
        "lora_banks",
    }
)
_REQUIRED_SCENE_ENCODER_KEYS = frozenset(
    {
        "architecture_version",
        "input_voxel_size_m",
        "block_size_m",
        "tokens_per_block",
        "model_dim",
        "global_latents",
        "heads",
        "block_layers",
        "global_layers",
        "fourier_bands",
        "coverage_temperature",
        "coverage_scale",
        "query_identity_scale",
        "projection_skip_scale",
        "semantic_skip_scale",
        "geometry_skip_scale",
        "block_content_residual_scale",
        "language_aligned_tail_dim",
        "native_aligned_coverage_scale",
        "learned_scene_token_scale",
        "learned_scene_token_rms_target",
        "global_scene_residual",
        "signed_x_scene_residual",
        "dense_alignment",
        "dense_sidecar_adapter",
    }
)
_OPTIONAL_SCENE_ENCODER_KEYS = frozenset({"block_cross_residual"})
_GLOBAL_RESIDUAL_KEYS = frozenset(
    {
        "enabled",
        "width",
        "fourier_bands",
        "initialization_seed",
        "expected_initial_state_sha256",
        "architecture_version",
        "gate_temperature",
    }
)
_SIGNED_X_RESIDUAL_KEYS = frozenset(
    {"enabled", "architecture_version", "expected_initial_state_sha256"}
)
_DENSE_ALIGNMENT_KEYS = frozenset(
    {
        "enabled",
        "dense_dim",
        "aligned_dim",
        "rank",
        "alpha",
        "initialization_seed",
        "expected_initial_state_sha256",
        "application_mode",
        "sidecar_scale",
    }
)
_DENSE_SIDECAR_KEYS = frozenset(
    {
        "enabled",
        "width",
        "fourier_bands",
        "max_direct_scale",
        "initialization_seed",
        "expected_initial_state_sha256",
    }
)
_BLOCK_CROSS_RESIDUAL_KEYS = frozenset(
    {
        "enabled",
        "attention_dim",
        "heads",
        "spatial_temperature",
        "residual_scale",
        "uniform_floor",
        "initialization_seed",
        "expected_initial_state_sha256",
    }
)
_NATIVE_IMAGE_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "model_revision",
        "bos_token_id",
        "pad_token_id",
        "boi_token_id",
        "image_token_id",
        "eoi_token_id",
        "use_bidirectional_attention",
    }
)
_LORA_BANK_KEYS = frozenset(
    {
        "trainable",
        "rank",
        "alpha",
        "dropout",
        "initialization_algorithm",
        "initialization_seed",
        "expected_initial_state_sha256",
        "target_modules",
    }
)


def _plain_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Runtime config {field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"Runtime config {field} keys must be strings")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], allowed: frozenset[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Runtime config {field} contains forbidden keys: {unknown}")


def _reject_environmental_vocabulary(value: object, location: str = "root") -> None:
    """Reject supervision/environment strings even when hidden in nested values."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_environmental_vocabulary(str(key), f"{location}.<key>")
            _reject_environmental_vocabulary(nested, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_environmental_vocabulary(nested, f"{location}[{index}]")
        return
    if not isinstance(value, str):
        return
    lowered = value.casefold().replace("-", "_")
    forbidden = sorted(
        fragment for fragment in _FORBIDDEN_ENVIRONMENT_FRAGMENTS if fragment in lowered
    )
    if forbidden or _QA_TOKEN.search(lowered):
        raise ValueError(
            f"Runtime config contains forbidden environmental vocabulary at {location}"
        )


def _reject_nonfinite_numbers(value: object, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite_numbers(nested, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_nonfinite_numbers(nested, f"{location}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Runtime config contains a nonfinite number at {location}")


def _require_exact_nested_mapping(
    value: object,
    required: frozenset[str],
    field: str,
) -> dict[str, Any]:
    mapping = _plain_mapping(value, field)
    _exact_keys(mapping, required, field)
    if missing := sorted(required - set(mapping)):
        raise ValueError(f"Runtime config {field} is missing fields: {missing}")
    return mapping


def _require_positive_integer(mapping: Mapping[str, Any], key: str, field: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Runtime config {field}.{key} must be an integer")
    if value < 1:
        raise ValueError(f"Runtime config {field}.{key} must be positive")
    return value


def _require_finite_number(
    mapping: Mapping[str, Any],
    key: str,
    field: str,
    *,
    minimum: float = 0.0,
    strictly_greater: bool = False,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Runtime config {field}.{key} must be numeric")
    parsed = float(value)
    invalid = parsed <= minimum if strictly_greater else parsed < minimum
    if not math.isfinite(parsed) or invalid:
        qualifier = "greater than" if strictly_greater else "at least"
        raise ValueError(f"Runtime config {field}.{key} must be {qualifier} {minimum}")
    return parsed


def _resolve_runtime_config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    unresolved = Path(os.path.abspath(rooted))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                "Runtime config must not use symbolic-link path components: "
                f"{current}"
            )
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(RUNTIME_CONFIG_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Production Gemma chat accepts only standalone configs below configs/runtime"
        ) from exc
    if resolved.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("Runtime config must be a YAML file")
    if not resolved.is_file():
        raise FileNotFoundError(f"Runtime config does not exist: {resolved}")
    return resolved


def validate_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated copy of the inference-only configuration."""

    result = _plain_mapping(config, "root")
    result.pop("_runtime_safe_config", None)
    result.pop("_config_path", None)
    _reject_environmental_vocabulary(result)
    _reject_nonfinite_numbers(result)
    if "_base_" in result:
        raise ValueError("Runtime config inheritance is forbidden")
    _exact_keys(result, _ALLOWED_TOP_LEVEL, "root")
    missing = sorted(_ALLOWED_TOP_LEVEL - set(result))
    if missing:
        raise ValueError(f"Runtime config is missing required sections: {missing}")

    runtime = _plain_mapping(result["runtime"], "runtime")
    _exact_keys(
        runtime,
        frozenset({"schema_version", "production", "reference_viewpoint"}),
        "runtime",
    )
    if (
        runtime.get("schema_version") != RUNTIME_CONFIG_SCHEMA_VERSION
        or runtime.get("production") is not True
    ):
        raise ValueError("Runtime config marker is invalid")
    viewpoint = _require_exact_nested_mapping(
        runtime.get("reference_viewpoint"),
        _REFERENCE_VIEWPOINT_KEYS,
        "runtime.reference_viewpoint",
    )
    position = viewpoint.get("position_m")
    if (
        not isinstance(position, list)
        or len(position) != 3
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in position)
    ):
        raise ValueError("Runtime reference position must contain three finite numbers")
    for field in ("yaw_degrees", "pitch_degrees"):
        value = viewpoint.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Runtime reference {field} must be numeric")
    view_count = viewpoint.get("scan_view_count")
    if isinstance(view_count, bool) or not isinstance(view_count, int) or view_count < 1:
        raise ValueError("Runtime reference scan_view_count must be a positive integer")

    paths = _require_exact_nested_mapping(result["paths"], _ALLOWED_PATH_KEYS, "paths")
    for key, value in paths.items():
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"Runtime config paths.{key} must be a nonempty path string")

    scene = _require_exact_nested_mapping(result["scene"], _ALLOWED_SCENE_KEYS, "scene")
    room_size = scene.get("room_size_m")
    if (
        not isinstance(room_size, list)
        or len(room_size) != 3
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in room_size)
        or any(float(value) <= 0.0 for value in room_size)
    ):
        raise ValueError("Runtime config scene.room_size_m must contain three positive numbers")

    vision = _require_exact_nested_mapping(result["vision"], _ALLOWED_VISION_KEYS, "vision")
    if vision.get("backend") != "gemma4":
        raise ValueError("Runtime config vision backend must be gemma4")
    if not isinstance(vision.get("model_id"), str) or not vision["model_id"].strip():
        raise TypeError("Runtime config vision.model_id must be a nonempty string")
    if not isinstance(vision.get("revision"), str) or re.fullmatch(
        r"[0-9a-f]{40}", vision["revision"]
    ) is None:
        raise ValueError("Runtime config vision.revision must be a pinned Git digest")

    scene_encoder = _plain_mapping(result["scene_encoder"], "scene_encoder")
    if missing_encoder := sorted(_REQUIRED_SCENE_ENCODER_KEYS - set(scene_encoder)):
        raise ValueError(f"Runtime scene encoder is incomplete: {missing_encoder}")
    if unknown_encoder := sorted(
        set(scene_encoder) - _REQUIRED_SCENE_ENCODER_KEYS - _OPTIONAL_SCENE_ENCODER_KEYS
    ):
        raise ValueError(
            f"Runtime scene encoder contains non-inference keys: {unknown_encoder}"
        )
    if not isinstance(scene_encoder.get("architecture_version"), str) or not scene_encoder[
        "architecture_version"
    ]:
        raise TypeError("Runtime scene encoder architecture_version must be nonempty")
    for key in (
        "tokens_per_block",
        "model_dim",
        "global_latents",
        "heads",
        "block_layers",
        "global_layers",
        "fourier_bands",
        "language_aligned_tail_dim",
    ):
        _require_positive_integer(scene_encoder, key, "scene_encoder")
    if scene_encoder["model_dim"] % scene_encoder["heads"]:
        raise ValueError("Runtime scene encoder model_dim must be divisible by heads")
    for key in ("input_voxel_size_m", "block_size_m", "coverage_temperature"):
        _require_finite_number(
            scene_encoder,
            key,
            "scene_encoder",
            strictly_greater=True,
        )
    for key in (
        "coverage_scale",
        "query_identity_scale",
        "projection_skip_scale",
        "semantic_skip_scale",
        "geometry_skip_scale",
        "block_content_residual_scale",
        "native_aligned_coverage_scale",
        "learned_scene_token_scale",
        "learned_scene_token_rms_target",
    ):
        _require_finite_number(scene_encoder, key, "scene_encoder")
    _require_exact_nested_mapping(
        scene_encoder["global_scene_residual"],
        _GLOBAL_RESIDUAL_KEYS,
        "scene_encoder.global_scene_residual",
    )
    _require_exact_nested_mapping(
        scene_encoder["signed_x_scene_residual"],
        _SIGNED_X_RESIDUAL_KEYS,
        "scene_encoder.signed_x_scene_residual",
    )
    _require_exact_nested_mapping(
        scene_encoder["dense_alignment"],
        _DENSE_ALIGNMENT_KEYS,
        "scene_encoder.dense_alignment",
    )
    _require_exact_nested_mapping(
        scene_encoder["dense_sidecar_adapter"],
        _DENSE_SIDECAR_KEYS,
        "scene_encoder.dense_sidecar_adapter",
    )
    if "block_cross_residual" in scene_encoder:
        block_cross = _require_exact_nested_mapping(
            scene_encoder["block_cross_residual"],
            _BLOCK_CROSS_RESIDUAL_KEYS,
            "scene_encoder.block_cross_residual",
        )
        if not isinstance(block_cross.get("enabled"), bool):
            raise TypeError("Runtime block-cross-residual enabled must be boolean")
        # The architecture parser additionally owns numerical ranges and the
        # deterministic-initial-state provenance digest. Keeping this key
        # optional preserves compatibility with pre-V35 runtime configs.
        block_cross_residual_settings(
            {"scene_encoder": {"block_cross_residual": block_cross}}
        )

    language = _require_exact_nested_mapping(
        result["language"], _ALLOWED_LANGUAGE_KEYS, "language"
    )
    if language.get("backend") != "gemma4":
        raise ValueError("Runtime config language backend must be gemma4")
    if language.get("system_prompt") != APPROVED_SYSTEM_PROMPT:
        raise ValueError("Runtime config must use the exact approved generic system prompt")
    if vision.get("model_id") != language.get("model_id"):
        raise ValueError("Runtime vision and language model IDs must match")
    if vision.get("revision") != language.get("revision"):
        raise ValueError("Runtime vision and language revisions must match")
    if language.get("dtype") not in {"float16", "bfloat16", "float32"}:
        raise ValueError("Runtime language dtype is unsupported")
    if language.get("scene_prefix_after_bos") is not True:
        raise ValueError("Runtime Gemma scene prefix must follow the native BOS")
    if language.get("scene_boundary_mode") != "gemma4_native_image":
        raise ValueError("Runtime Gemma scene boundary mode is unsupported")
    for key in ("max_question_tokens", "max_answer_tokens"):
        _require_positive_integer(language, key, "language")
    native_contract = _require_exact_nested_mapping(
        language["gemma4_native_image_contract"],
        _NATIVE_IMAGE_CONTRACT_KEYS,
        "language.gemma4_native_image_contract",
    )
    if (
        native_contract.get("schema_version") != 1
        or native_contract.get("model_revision") != language["revision"]
    ):
        raise ValueError("Runtime native-image contract identity is invalid")
    for key in ("bos_token_id", "pad_token_id", "boi_token_id", "image_token_id", "eoi_token_id"):
        value = native_contract.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Runtime native-image contract {key} must be nonnegative")
    attention_mode = native_contract.get("use_bidirectional_attention")
    if attention_mode is not None and not isinstance(attention_mode, str):
        raise TypeError("Runtime native-image attention mode must be null or a string")
    lora_banks = _plain_mapping(language["lora_banks"], "language.lora_banks")
    if not lora_banks:
        raise ValueError("Runtime config language.lora_banks must not be empty")
    for bank_name, bank in lora_banks.items():
        parsed_bank = _require_exact_nested_mapping(
            bank,
            _LORA_BANK_KEYS,
            f"language.lora_banks.{bank_name}",
        )
        if not isinstance(parsed_bank.get("trainable"), bool):
            raise TypeError(f"Runtime LoRA bank {bank_name} trainable must be boolean")
        _require_positive_integer(parsed_bank, "rank", f"language.lora_banks.{bank_name}")
        _require_finite_number(
            parsed_bank,
            "alpha",
            f"language.lora_banks.{bank_name}",
            strictly_greater=True,
        )
        dropout = _require_finite_number(
            parsed_bank,
            "dropout",
            f"language.lora_banks.{bank_name}",
        )
        if dropout >= 1.0:
            raise ValueError(f"Runtime LoRA bank {bank_name} dropout must be below one")
        targets = parsed_bank.get("target_modules")
        if (
            not isinstance(targets, list)
            or not targets
            or not all(isinstance(target, str) and target for target in targets)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError(f"Runtime LoRA bank {bank_name} targets must be unique paths")
        initialization_algorithm = parsed_bank.get("initialization_algorithm")
        if not isinstance(initialization_algorithm, str):
            raise TypeError(f"Runtime LoRA bank {bank_name} initialization must be a string")
        initial_hash = parsed_bank.get("expected_initial_state_sha256")
        checkpoint_overwrite_without_constructor_digest = (
            initialization_algorithm == "checkpoint_overwrite" and initial_hash is None
        )
        if not checkpoint_overwrite_without_constructor_digest and (
            not isinstance(initial_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", initial_hash) is None
        ):
            raise ValueError(f"Runtime LoRA bank {bank_name} initial digest is invalid")
        seed = parsed_bank.get("initialization_seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError(f"Runtime LoRA bank {bank_name} seed must be an integer or null")

    training = _require_exact_nested_mapping(
        result["training"], _ALLOWED_TRAINING_KEYS, "training"
    )
    if training.get("freeze_scene_adapter") is not True:
        raise ValueError("Runtime config must attest a frozen scene adapter")
    _require_finite_number(
        training,
        "lora_learning_rate",
        "training",
        strictly_greater=True,
    )
    _require_finite_number(training, "lora_weight_decay", "training")

    # The marker is internal and excluded by the existing effective-config hash.
    result["_runtime_safe_config"] = True
    return result


def load_runtime_config(
    path: str | Path,
    *,
    record_file: Callable[[str | Path], None] | None = None,
) -> dict[str, Any]:
    """Load one physically isolated runtime YAML without recursive inheritance."""

    resolved = _resolve_runtime_config_path(path)
    if record_file is not None:
        record_file(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    config = validate_runtime_config(_plain_mapping(loaded, "root"))
    config["_config_path"] = str(resolved)
    return config


def runtime_config_file_sha256(path: str | Path) -> str:
    resolved = _resolve_runtime_config_path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_runtime_config_sha256(config: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in config.items() if not str(key).startswith("_")}
    encoded = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_runtime_config_path(path: str | Path) -> bool:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(RUNTIME_CONFIG_ROOT)
    except ValueError:
        return False
    return True


__all__ = [
    "APPROVED_SYSTEM_PROMPT",
    "RUNTIME_CONFIG_ROOT",
    "effective_runtime_config_sha256",
    "is_runtime_config_path",
    "load_runtime_config",
    "runtime_config_file_sha256",
    "validate_runtime_config",
]

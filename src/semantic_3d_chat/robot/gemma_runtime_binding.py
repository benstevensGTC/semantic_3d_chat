"""Content identity for the exact Gemma decoder used by waypoint control.

Waypoint-head training caches Gemma hidden states.  Those states are valid only
for the exact decoder stack that produced them: base model revision, runtime
configuration, and every loaded LoRA tensor.  This module gives training and
production one small, shared, path-free contract for that identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256

BINDING_SCHEMA: Final[str] = "semantic_3d_chat.gemma_runtime_binding.v1"
_KINDS: Final[frozenset[str]] = frozenset(
    {"raw_hf_gemma", "question_controlled_production"}
)
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_FILES: Final[tuple[str, ...]] = (
    "adapter.safetensors",
    "metadata.json",
    "runtime_metadata.json",
)
_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "runtime_kind",
        "model_id",
        "model_revision",
        "language_backend",
        "language_dtype",
        "effective_runtime_config_sha256",
        "base_checkpoint_sha256",
        "control_checkpoint_sha256",
        "lora_bank_state_sha256",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint_sha256(path: str | Path) -> str:
    """Hash exactly the inference-bearing files of an adapter checkpoint."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Runtime checkpoint does not exist: {root}")
    adapter = root / "adapter.safetensors"
    metadata = root / "runtime_metadata.json"
    if not adapter.is_file() or not metadata.is_file():
        raise FileNotFoundError(
            "Runtime checkpoint requires adapter.safetensors and runtime_metadata.json"
        )
    entries = []
    for name in _CHECKPOINT_FILES:
        candidate = root / name
        if not candidate.is_file():
            continue
        if candidate.is_symlink():
            raise ValueError("Runtime checkpoint identity cannot follow symbolic links")
        entries.append(
            {
                "path": name,
                "sha256": _sha256_file(candidate),
                "size_bytes": candidate.stat().st_size,
            }
        )
    return hashlib.sha256(_canonical_json(entries)).hexdigest()


def control_checkpoint_fingerprint_sha256(path: str | Path) -> str:
    """Hash the runtime-minimal continuous question-control checkpoint."""

    root = Path(path).expanduser().resolve()
    expected = ("control.safetensors", "runtime_metadata.json")
    if not root.is_dir() or sorted(item.name for item in root.iterdir()) != sorted(
        expected
    ):
        raise ValueError("Control checkpoint inventory is not runtime-minimal")
    entries = []
    for name in expected:
        candidate = root / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("Control checkpoint entries must be regular files")
        entries.append(
            {
                "name": name,
                "sha256": _sha256_file(candidate),
                "size_bytes": candidate.stat().st_size,
            }
        )
    return hashlib.sha256(_canonical_json(entries)).hexdigest()


def validate_gemma_runtime_binding(value: object) -> dict[str, Any]:
    """Return a canonical validated copy or fail closed on any missing field."""

    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("Gemma runtime binding fields changed")
    result = dict(value)
    if result.get("schema") != BINDING_SCHEMA or result.get("runtime_kind") not in _KINDS:
        raise ValueError("Gemma runtime binding schema or kind differs")
    for name in ("model_id", "model_revision", "language_backend", "language_dtype"):
        item = result.get(name)
        if not isinstance(item, str) or not item:
            raise ValueError(f"Gemma runtime binding {name} must be nonempty")
    lora = result.get("lora_bank_state_sha256")
    if not isinstance(lora, Mapping) or not all(
        isinstance(name, str)
        and bool(name)
        and isinstance(digest, str)
        and _SHA256.fullmatch(digest) is not None
        for name, digest in lora.items()
    ):
        raise ValueError("Gemma runtime binding LoRA-state hashes are invalid")
    result["lora_bank_state_sha256"] = {
        name: str(lora[name]) for name in sorted(lora)
    }
    hash_fields = (
        "effective_runtime_config_sha256",
        "base_checkpoint_sha256",
        "control_checkpoint_sha256",
    )
    production = result["runtime_kind"] == "question_controlled_production"
    for name in hash_fields:
        item = result.get(name)
        if production:
            if not isinstance(item, str) or _SHA256.fullmatch(item) is None:
                raise ValueError(f"Production Gemma binding {name} is invalid")
        elif item is not None:
            raise ValueError(f"Raw-HF Gemma binding cannot declare {name}")
    if production and not result["lora_bank_state_sha256"]:
        raise ValueError("Production Gemma binding requires loaded LoRA state")
    if not production and result["lora_bank_state_sha256"]:
        raise ValueError("Raw-HF Gemma binding cannot declare LoRA state")
    return result


def gemma_runtime_binding_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(validate_gemma_runtime_binding(value))).hexdigest()


def raw_hf_gemma_runtime_binding(
    *, model_id: str, model_revision: str, language_dtype: str
) -> dict[str, Any]:
    return validate_gemma_runtime_binding(
        {
            "schema": BINDING_SCHEMA,
            "runtime_kind": "raw_hf_gemma",
            "model_id": model_id,
            "model_revision": model_revision,
            "language_backend": "gemma4",
            "language_dtype": language_dtype,
            "effective_runtime_config_sha256": None,
            "base_checkpoint_sha256": None,
            "control_checkpoint_sha256": None,
            "lora_bank_state_sha256": {},
        }
    )


def question_controlled_gemma_runtime_binding(
    runtime: Any,
    config: Mapping[str, Any],
    *,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
) -> dict[str, Any]:
    """Bind the loaded production runtime, including base/control artifacts."""

    base = getattr(runtime, "base", None)
    language = getattr(base, "language", None)
    metadata = getattr(base, "checkpoint_metadata", None)
    control_metadata = getattr(runtime, "control_metadata", None)
    if language is None or not isinstance(metadata, Mapping) or not isinstance(
        control_metadata, Mapping
    ):
        raise TypeError("Gemma binding requires a loaded QuestionControlledChatRuntime")
    language_config = config.get("language")
    if not isinstance(language_config, Mapping):
        raise TypeError("Gemma binding requires runtime language settings")
    base_digest = checkpoint_fingerprint_sha256(base_checkpoint)
    control_digest = control_checkpoint_fingerprint_sha256(control_checkpoint)
    runtime_digest = effective_runtime_config_sha256(config)
    if control_metadata.get("base_checkpoint_sha256") != base_digest:
        raise ValueError("Loaded control runtime is bound to a different base checkpoint")
    if control_metadata.get("base_runtime_config_sha256") != runtime_digest:
        raise ValueError("Loaded control runtime is bound to a different runtime config")
    loaded_base = getattr(base, "checkpoint_path", None)
    if loaded_base is None or checkpoint_fingerprint_sha256(loaded_base) != base_digest:
        raise ValueError("Loaded Gemma base differs from the requested production checkpoint")
    lora_hashes = metadata.get("lora_bank_state_sha256")
    binding = {
        "schema": BINDING_SCHEMA,
        "runtime_kind": "question_controlled_production",
        "model_id": language_config.get("model_id"),
        "model_revision": language_config.get("revision"),
        "language_backend": getattr(language, "backend_name", None),
        "language_dtype": language_config.get("dtype"),
        "effective_runtime_config_sha256": runtime_digest,
        "base_checkpoint_sha256": base_digest,
        "control_checkpoint_sha256": control_digest,
        "lora_bank_state_sha256": lora_hashes,
    }
    return validate_gemma_runtime_binding(binding)


def attach_gemma_runtime_binding(language: Any, value: object) -> dict[str, Any]:
    binding = validate_gemma_runtime_binding(value)
    language._waypoint_gemma_runtime_binding = binding
    return binding


def language_gemma_runtime_binding(language: Any) -> dict[str, Any]:
    value = getattr(language, "_waypoint_gemma_runtime_binding", None)
    if value is None:
        raise ValueError("Local Gemma has no authenticated waypoint runtime binding")
    return validate_gemma_runtime_binding(value)


__all__ = [
    "BINDING_SCHEMA",
    "attach_gemma_runtime_binding",
    "checkpoint_fingerprint_sha256",
    "control_checkpoint_fingerprint_sha256",
    "gemma_runtime_binding_sha256",
    "language_gemma_runtime_binding",
    "question_controlled_gemma_runtime_binding",
    "raw_hf_gemma_runtime_binding",
    "validate_gemma_runtime_binding",
]

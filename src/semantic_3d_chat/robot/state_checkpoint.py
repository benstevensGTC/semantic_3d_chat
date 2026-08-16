"""Strict, sanitized checkpoints for continuous numeric robot-state tokens."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot.state_encoder import RobotStateEncoder

_ARCHITECTURE = "numeric_robot_state_mlp_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_OUTPUT_DIM = 8192
_MAX_HIDDEN_DIM = 4096
_MAX_TOKEN_COUNT = 64
_FILES = frozenset({"state.safetensors", "runtime_metadata.json"})
_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "architecture",
        "output_dim",
        "hidden_dim",
        "token_count",
        "initialization_seed",
        "output_scale",
        "task_trained",
        "numeric_inputs_only",
        "environmental_text_inputs",
        "weights_sha256",
    }
)


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _reject_unsafe_path(path: Path) -> None:
    if {"oracle", "qa"} & {part.casefold() for part in path.parts}:
        raise ValueError("Robot-state checkpoints cannot use oracle or QA paths")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Robot-state checkpoint paths cannot contain symlinks")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Robot-state {field} must be a positive integer")
    return value


def _bounded_dimension(value: object, field: str, maximum: int) -> int:
    result = _positive_int(value, field)
    if result > maximum:
        raise ValueError(f"Robot-state {field} exceeds the safety limit {maximum}")
    return result


def _strict_metadata(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate robot-state metadata field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Robot-state runtime metadata is invalid JSON") from error
    if not isinstance(value, dict) or set(value) != _METADATA_FIELDS:
        raise ValueError("Robot-state runtime metadata fields changed")
    return value


def build_deterministic_robot_state_encoder(
    output_dim: int,
    *,
    hidden_dim: int = 256,
    token_count: int = 4,
    seed: int = 20260812,
    output_scale: float = 0.02,
) -> RobotStateEncoder:
    """Create a reproducible low-amplitude numeric projection for local research.

    This initialization is deliberately not described as task trained.  It
    establishes the continuous interface and keeps the contribution bounded so
    embodied QA can later replace it with a supervised state adapter without an
    architecture change.
    """

    output_dim = _bounded_dimension(output_dim, "output_dim", _MAX_OUTPUT_DIM)
    hidden_dim = _bounded_dimension(hidden_dim, "hidden_dim", _MAX_HIDDEN_DIM)
    token_count = _bounded_dimension(token_count, "token_count", _MAX_TOKEN_COUNT)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Robot-state initialization_seed must be a nonnegative integer")
    if not isinstance(output_scale, (int, float)) or not math.isfinite(float(output_scale)):
        raise ValueError("Robot-state output_scale must be finite")
    if not 0.0 < float(output_scale) <= 0.1:
        raise ValueError("Robot-state output_scale must be in (0, 0.1]")
    encoder = RobotStateEncoder(
        output_dim,
        hidden_dim=hidden_dim,
        token_count=token_count,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = encoder.network[0]
    normalization = encoder.network[2]
    last = encoder.network[3]
    assert isinstance(first, torch.nn.Linear)
    assert isinstance(normalization, torch.nn.LayerNorm)
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(first.weight, generator=generator)
        first.bias.zero_()
        normalization.weight.fill_(1.0)
        normalization.bias.zero_()
        torch.nn.init.xavier_uniform_(last.weight, generator=generator)
        last.weight.mul_(float(output_scale))
        last.bias.zero_()
    return encoder.eval()


def create_robot_state_checkpoint(
    destination: str | Path,
    *,
    output_dim: int,
    hidden_dim: int = 256,
    token_count: int = 4,
    seed: int = 20260812,
    output_scale: float = 0.02,
) -> dict[str, Any]:
    """Write exactly two inference-safe files; refuse to overwrite anything."""

    root = _rooted(destination)
    _reject_unsafe_path(root)
    if root.exists():
        raise FileExistsError(f"Robot-state checkpoint already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        encoder = build_deterministic_robot_state_encoder(
            output_dim,
            hidden_dim=hidden_dim,
            token_count=token_count,
            seed=seed,
            output_scale=output_scale,
        )
        weights = temporary / "state.safetensors"
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in encoder.state_dict().items()},
            str(weights),
        )
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "architecture": _ARCHITECTURE,
            "output_dim": int(output_dim),
            "hidden_dim": int(hidden_dim),
            "token_count": int(token_count),
            "initialization_seed": int(seed),
            "output_scale": float(output_scale),
            "task_trained": False,
            "numeric_inputs_only": True,
            "environmental_text_inputs": [],
            "weights_sha256": _sha256(weights),
        }
        (temporary / "runtime_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root)
        return {**metadata, "checkpoint": str(root)}
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()


def load_robot_state_checkpoint(
    checkpoint: str | Path,
    *,
    expected_output_dim: int,
    device: torch.device | str = "cpu",
    audit: FileAccessAudit | None = None,
) -> tuple[RobotStateEncoder, str, dict[str, Any]]:
    """Strictly load and hash-bind a sanitized numeric state encoder."""

    root = _rooted(checkpoint)
    _reject_unsafe_path(root)
    if not root.is_dir() or {item.name for item in root.iterdir()} != _FILES:
        raise ValueError("Robot-state checkpoint must contain exactly two sanitized files")
    weights = root / "state.safetensors"
    metadata_path = root / "runtime_metadata.json"
    if any(item.is_symlink() or not item.is_file() for item in (weights, metadata_path)):
        raise ValueError("Robot-state checkpoint entries must be regular files")
    if audit is not None:
        audit.record(metadata_path)
    metadata = _strict_metadata(metadata_path)
    output_dim = _bounded_dimension(
        metadata["output_dim"], "output_dim", _MAX_OUTPUT_DIM
    )
    hidden_dim = _bounded_dimension(
        metadata["hidden_dim"], "hidden_dim", _MAX_HIDDEN_DIM
    )
    token_count = _bounded_dimension(
        metadata["token_count"], "token_count", _MAX_TOKEN_COUNT
    )
    digest = metadata.get("weights_sha256")
    initialization_seed = metadata.get("initialization_seed")
    output_scale = metadata.get("output_scale")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("architecture") != _ARCHITECTURE
        or output_dim
        != _bounded_dimension(
            expected_output_dim, "expected_output_dim", _MAX_OUTPUT_DIM
        )
        or metadata.get("task_trained") is not False
        or metadata.get("numeric_inputs_only") is not True
        or metadata.get("environmental_text_inputs") != []
        or isinstance(initialization_seed, bool)
        or not isinstance(initialization_seed, int)
        or initialization_seed < 0
        or isinstance(output_scale, bool)
        or not isinstance(output_scale, (int, float))
        or not math.isfinite(float(output_scale))
        or not 0.0 < float(output_scale) <= 0.1
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or _sha256(weights) != digest
    ):
        raise ValueError("Robot-state runtime checkpoint contract mismatch")
    if audit is not None:
        audit.record(weights)
    state = load_file(str(weights), device="cpu")
    encoder = build_deterministic_robot_state_encoder(
        output_dim,
        hidden_dim=hidden_dim,
        token_count=token_count,
        seed=initialization_seed,
        output_scale=float(output_scale),
    )
    expected_state = encoder.state_dict()
    if set(state) != set(expected_state) or any(
        not torch.equal(state[name], expected_state[name]) for name in expected_state
    ):
        raise ValueError(
            "Robot-state weights do not match their deterministic metadata provenance"
        )
    encoder = encoder.to(device=device, dtype=torch.float32).eval()
    if any(not torch.isfinite(value).all() for value in encoder.state_dict().values()):
        raise ValueError("Robot-state checkpoint contains NaN or infinity")
    # Imported lazily to keep checkpoint parsing independent of the embodied
    # runtime while returning the exact in-memory state identity that the
    # runtime binds and re-verifies.
    from semantic_3d_chat.robot.runtime_refresh import robot_state_encoder_sha256

    return encoder, robot_state_encoder_sha256(encoder), metadata


__all__ = [
    "build_deterministic_robot_state_encoder",
    "create_robot_state_checkpoint",
    "load_robot_state_checkpoint",
]

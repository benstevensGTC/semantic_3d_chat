"""Shared fail-closed launch resolution for CLI and web chat entry points."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.promotion import (
    resolve_primary_pointer,
    sha256_file,
    validate_chat_promotion,
)
from semantic_3d_chat.chat.runtime_config import (
    is_runtime_config_path,
    load_runtime_config,
)
from semantic_3d_chat.config import (
    PROJECT_ROOT,
    default_checkpoint_path,
    load_config,
    project_path,
)

_LEGACY_CHAT_CONFIGS = frozenset(
    (PROJECT_ROOT / "configs" / name).resolve()
    for name in ("default.yaml", "small_mac.yaml", "large_mac.yaml")
)


@dataclass(frozen=True)
class ChatLaunch:
    config_path: Path
    checkpoint_path: Path
    config: dict[str, Any]
    promotion: dict[str, Any] | None

    @property
    def is_production_gemma(self) -> bool:
        return self.promotion is not None

    def _promoted_scene_entry(self, scene_id: str) -> Mapping[str, Any] | None:
        if self.promotion is None:
            return None
        manifest = self.promotion.get("scene_runtime_manifest")
        if not isinstance(manifest, Mapping):
            raise TypeError("Promotion has no valid scene runtime manifest")
        if scene_id not in manifest:
            raise ValueError(
                f"Requested scene is not attested by the primary promotion: {scene_id}"
            )
        entry = manifest.get(scene_id)
        if not isinstance(entry, Mapping):
            raise TypeError(f"Promoted scene attestation is invalid: {scene_id}")
        return entry

    def verify_scene_map(
        self, scene_id: str, *, audit: FileAccessAudit | None = None
    ) -> Path | None:
        """Verify exact promoted map bytes before constructing a production runtime."""

        entry = self._promoted_scene_entry(scene_id)
        if entry is None:
            return None
        unresolved_map = _unresolved_rooted(
            project_path(self.config, "maps", scene_id, "voxel_map.npz")
        )
        _reject_symlink_components(unresolved_map, "Promoted scene voxel map")
        map_path = unresolved_map.resolve()
        if not map_path.is_file():
            raise FileNotFoundError(f"Promoted scene voxel map is missing: {map_path}")
        if audit is not None:
            audit.record(map_path)
        expected_size = entry.get("voxel_map_size_bytes")
        expected_sha256 = entry.get("voxel_map_sha256")
        observed_size = map_path.stat().st_size
        observed_sha256 = sha256_file(map_path)
        if observed_size != expected_size or observed_sha256 != expected_sha256:
            raise ValueError(
                "Promoted scene voxel-map bytes do not match the attestation: "
                f"scene={scene_id} expected_size={expected_size} "
                f"observed_size={observed_size} expected_sha256={expected_sha256} "
                f"observed_sha256={observed_sha256}"
            )
        return map_path

    def verify_scene_prefix(
        self,
        requested_scene_id: str,
        *,
        loaded_scene_id: str,
        prefix_sha256: str,
    ) -> None:
        """Bind the loaded continuous prefix to the promoted scene identity."""

        entry = self._promoted_scene_entry(requested_scene_id)
        if entry is None:
            return
        if loaded_scene_id != requested_scene_id:
            raise ValueError(
                "Loaded runtime scene does not match the requested promoted scene: "
                f"requested={requested_scene_id} loaded={loaded_scene_id}"
            )
        expected_prefix = entry.get("scene_prefix_sha256")
        if prefix_sha256 != expected_prefix:
            raise ValueError(
                "Computed scene prefix does not match the primary promotion: "
                f"scene={requested_scene_id} expected={expected_prefix} "
                f"observed={prefix_sha256}"
            )


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _unresolved_rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _reject_symlink_components(path: Path, purpose: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} must not use symbolic-link path components: {current}")


def _reject_experiment_config_before_open(path: Path) -> None:
    experiment_root = (PROJECT_ROOT / "configs/experiments").resolve()
    try:
        path.relative_to(experiment_root)
    except ValueError:
        return
    raise ValueError(
        "Interactive chat refuses training/experiment configs; use a standalone "
        "configs/runtime configuration"
    )


def _reject_unapproved_legacy_config_before_open(path: Path) -> None:
    if is_runtime_config_path(path) or path in _LEGACY_CHAT_CONFIGS:
        return
    raise ValueError(
        "Interactive chat accepts only standalone configs/runtime YAML or the "
        "explicit legacy default/small_mac/large_mac allowlist"
    )


def resolve_chat_launch(
    *,
    config_path: str | Path | None,
    checkpoint: str | Path | None,
    primary_pointer: str | Path | None,
    audit: FileAccessAudit | None = None,
) -> ChatLaunch:
    """Resolve a legacy launch or a promotion-bound production Gemma launch."""

    recorder = None if audit is None else audit.record
    if primary_pointer is not None:
        if config_path is not None or checkpoint is not None:
            raise ValueError(
                "--primary-pointer cannot be combined with --config or --checkpoint"
            )
        resolved_config, resolved_checkpoint = resolve_primary_pointer(
            primary_pointer,
            record_file=recorder,
        )
        config = load_runtime_config(resolved_config, record_file=recorder)
        promotion = validate_chat_promotion(
            resolved_checkpoint,
            resolved_config,
            config,
            record_file=recorder,
        )
        return ChatLaunch(
            config_path=resolved_config,
            checkpoint_path=resolved_checkpoint,
            config=config,
            promotion=promotion,
        )

    unresolved_config = _unresolved_rooted(config_path or "configs/default.yaml")
    _reject_symlink_components(unresolved_config, "Interactive chat config")
    resolved_config = unresolved_config.resolve()
    _reject_experiment_config_before_open(resolved_config)
    _reject_unapproved_legacy_config_before_open(resolved_config)
    if is_runtime_config_path(resolved_config):
        if checkpoint is None:
            raise ValueError(
                "Production Gemma chat requires an explicit promoted checkpoint or "
                "--primary-pointer; training best/ is never selected implicitly"
            )
        config = load_runtime_config(unresolved_config, record_file=recorder)
        resolved_config = Path(str(config["_config_path"]))
        promotion = validate_chat_promotion(
            checkpoint,
            resolved_config,
            config,
            record_file=recorder,
        )
        resolved_checkpoint = _rooted(checkpoint)
        return ChatLaunch(
            config_path=resolved_config,
            checkpoint_path=resolved_checkpoint,
            config=config,
            promotion=promotion,
        )

    if audit is not None:
        audit.record(resolved_config)
    config = load_config(unresolved_config)
    resolved_config = Path(str(config["_config_path"]))
    if str(config.get("language", {}).get("backend", "auto")).casefold() == "gemma4":
        raise ValueError(
            "Gemma chat requires a standalone configs/runtime configuration and a "
            "behaviorally promoted checkpoint"
        )
    resolved_checkpoint = _rooted(checkpoint or default_checkpoint_path(config))
    return ChatLaunch(
        config_path=resolved_config,
        checkpoint_path=resolved_checkpoint,
        config=config,
        promotion=None,
    )


__all__ = ["ChatLaunch", "resolve_chat_launch"]

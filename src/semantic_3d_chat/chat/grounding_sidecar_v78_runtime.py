"""Fail-closed optional runtime for the V78 numeric grounding diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.grounding_sidecar_v78 import (
    ARCHITECTURE,
    ARTIFACT,
    EXPECTED_CHECKPOINT_FILES,
    METADATA_FILENAME,
    WEIGHTS_FILENAME,
    GroundingSidecarV78,
    denormalize_xyz,
)
from semantic_3d_chat.training.grounding_sidecar_v78_release import RUNTIME_ARTIFACT

_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "training", "validation", "test", "deferred", "final_once"}
)
_EXPECTED_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "all_scene_tokens_scored",
        "answer_text_serialized",
        "architecture",
        "artifact",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "coordinate_hidden_dim",
        "embedding_tensor_key",
        "environmental_text_inputs",
        "initialization_seed",
        "maximum_residual",
        "model_id",
        "model_revision",
        "object_ids_serialized",
        "official_test_loaded",
        "official_validation_evidence",
        "official_validation_loaded",
        "optional_runtime_demo_authorized",
        "oracle_runtime_loaded",
        "positive_softmax_attention",
        "question_adapter_rank",
        "question_dependent_scene_retrieval",
        "question_only_coordinate_path_exists",
        "question_text_serialized",
        "room_max_m",
        "room_min_m",
        "runtime_promotion_authorized",
        "scene_dim",
        "scene_latent_count",
        "schema_version",
        "source_candidate_artifact",
        "source_candidate_metadata_sha256",
        "source_candidate_weights_sha256",
        "source_prefix_base_checkpoint_sha256",
        "source_prefix_manifest_sha256",
        "target_coordinates_serialized",
        "training_metadata_runtime_loaded",
        "weights_sha256",
        "zero_scene_produces_exact_room_center",
    }
)


@dataclass(frozen=True)
class V78GroundingOutput:
    xyz_m: tuple[float, float, float]
    confidence: float
    support_distance_m: float
    audit: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_without_symlinks(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    candidate = Path(os.path.abspath(rooted))
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V78 grounding checkpoint path contains a symlink: {current}")
    return candidate


def _checkpoint_files(checkpoint: str | Path) -> tuple[Path, Path, Path]:
    root = _resolve_without_symlinks(checkpoint)
    lowered = {component.casefold() for component in root.parts}
    if lowered & _FORBIDDEN_COMPONENTS:
        raise ValueError("V78 grounding checkpoint is inside a prohibited data tree")
    if not root.is_dir():
        raise FileNotFoundError(f"V78 grounding checkpoint is unavailable: {root}")
    inventory = {item.name for item in root.iterdir()}
    if inventory != EXPECTED_CHECKPOINT_FILES:
        raise ValueError(
            "V78 grounding checkpoint must contain exactly its two sanitized files: "
            f"expected={sorted(EXPECTED_CHECKPOINT_FILES)} observed={sorted(inventory)}"
        )
    weights = root / WEIGHTS_FILENAME
    metadata = root / METADATA_FILENAME
    if any(path.is_symlink() or not path.is_file() for path in (weights, metadata)):
        raise ValueError("V78 grounding checkpoint inventory is not regular files")
    return root, weights, metadata


def _finite_triplet(value: object, *, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"V78 {label} must contain three numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"V78 {label} contains NaN or infinity")
    return result


class V78GroundingSidecarRuntime:
    """Authenticated numeric reader bound to one immutable full-scene prefix."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        model: GroundingSidecarV78,
        metadata: dict[str, Any],
        scene_prefix: torch.Tensor,
        room_min: torch.Tensor,
        room_max: torch.Tensor,
    ) -> None:
        self.checkpoint = checkpoint
        self.model = model.eval()
        self.metadata = dict(metadata)
        expected = (1, model.latent_count + 2, model.scene_dim)
        if tuple(scene_prefix.shape) != expected:
            raise ValueError(
                f"V78 scene prefix shape mismatch: {tuple(scene_prefix.shape)} != {expected}"
            )
        if not torch.isfinite(scene_prefix.float()).all():
            raise ValueError("V78 scene prefix contains NaN or infinity")
        self._full_prefix = scene_prefix.detach()
        self._full_prefix_sha256 = prefix_sha256(self._full_prefix)
        self._scene_tokens = self._full_prefix[:, 1:-1].detach().float().clone()
        self._scene_tokens_sha256 = prefix_sha256(self._scene_tokens)
        self.room_min = room_min.detach().float().clone()
        self.room_max = room_max.detach().float().clone()
        if self.room_min.shape != (3,) or self.room_max.shape != (3,):
            raise ValueError("V78 runtime room bounds must have shape [3]")
        if not torch.all(self.room_max > self.room_min):
            raise ValueError("V78 runtime room bounds are invalid")
        self.last_audit: dict[str, Any] | None = None

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        scene_prefix: torch.Tensor,
        room_min: torch.Tensor,
        room_max: torch.Tensor,
        base_checkpoint_sha256: str,
        base_runtime_config_sha256: str,
        model_id: str,
        model_revision: str,
        device: torch.device | str,
        audit: FileAccessAudit | None = None,
    ) -> V78GroundingSidecarRuntime:
        root, weights_path, metadata_path = _checkpoint_files(checkpoint)
        if audit is not None:
            # Native safetensors reads do not always emit CPython open events.
            audit.record(metadata_path)
            audit.record(weights_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if set(metadata) != _EXPECTED_METADATA_FIELDS:
            missing = sorted(_EXPECTED_METADATA_FIELDS - set(metadata))
            unexpected = sorted(set(metadata) - _EXPECTED_METADATA_FIELDS)
            raise ValueError(
                f"V78 grounding metadata schema mismatch: missing={missing} "
                f"unexpected={unexpected}"
            )
        if metadata.get("schema_version") != 1:
            raise ValueError("V78 grounding metadata schema version changed")
        if (
            metadata.get("artifact") != RUNTIME_ARTIFACT
            or metadata.get("source_candidate_artifact") != ARTIFACT
            or metadata.get("architecture") != ARCHITECTURE
        ):
            raise ValueError("V78 grounding architecture identity changed")
        for field in (
            "weights_sha256",
            "source_prefix_manifest_sha256",
            "source_prefix_base_checkpoint_sha256",
            "source_candidate_metadata_sha256",
            "source_candidate_weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
        ):
            value = metadata.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"V78 grounding {field} is not a SHA-256 digest")
        if _sha256_file(weights_path) != metadata["weights_sha256"]:
            raise ValueError("V78 grounding weights changed")
        if metadata["source_candidate_weights_sha256"] != metadata["weights_sha256"]:
            raise ValueError("V78 grounding runtime weights differ from its source candidate")
        if metadata["base_checkpoint_sha256"] != base_checkpoint_sha256:
            raise ValueError("V78 grounding sidecar is bound to a different base checkpoint")
        if metadata["base_runtime_config_sha256"] != base_runtime_config_sha256:
            raise ValueError("V78 grounding sidecar is bound to a different runtime config")
        if metadata.get("model_id") != model_id or metadata.get("model_revision") != model_revision:
            raise ValueError("V78 grounding sidecar is bound to a different Gemma identity")
        required_true = {
            "all_scene_tokens_scored",
            "positive_softmax_attention",
            "zero_scene_produces_exact_room_center",
            "optional_runtime_demo_authorized",
        }
        if any(metadata.get(field) is not True for field in required_true):
            raise ValueError("V78 grounding sidecar lacks a required numeric-runtime attestation")
        required_false = {
            "answer_text_serialized",
            "object_ids_serialized",
            "official_test_loaded",
            "official_validation_evidence",
            "official_validation_loaded",
            "oracle_runtime_loaded",
            "question_dependent_scene_retrieval",
            "question_only_coordinate_path_exists",
            "question_text_serialized",
            "runtime_promotion_authorized",
            "target_coordinates_serialized",
            "training_metadata_runtime_loaded",
        }
        if any(metadata.get(field) is not False for field in required_false):
            raise ValueError("V78 grounding sidecar permits a prohibited input or evidence claim")
        if metadata.get("environmental_text_inputs") != []:
            raise ValueError("V78 grounding sidecar declares environmental text inputs")
        if metadata.get("embedding_tensor_key") != "model.language_model.embed_tokens.weight":
            raise ValueError("V78 grounding question-embedding contract changed")
        expected_numeric = {
            "scene_dim": 1536,
            "scene_latent_count": 256,
            "question_adapter_rank": 64,
            "coordinate_hidden_dim": 256,
            "maximum_residual": 0.5,
        }
        for field, expected_value in expected_numeric.items():
            if metadata.get(field) != expected_value:
                raise ValueError(
                    f"V78 grounding {field} changed: "
                    f"{metadata.get(field)!r} != {expected_value!r}"
                )
        model = GroundingSidecarV78(
            scene_dim=int(metadata["scene_dim"]),
            latent_count=int(metadata["scene_latent_count"]),
            rank=int(metadata["question_adapter_rank"]),
            hidden_dim=int(metadata["coordinate_hidden_dim"]),
            maximum_residual=float(metadata["maximum_residual"]),
        )
        state = load_file(str(weights_path), device="cpu")
        if set(state) != set(model.state_dict()):
            raise ValueError("V78 grounding tensor schema changed")
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("V78 grounding weights contain NaN or infinity")
        model.load_state_dict(state, strict=True)
        model = model.to(device=device, dtype=torch.float32).eval()
        configured_min = torch.tensor(_finite_triplet(metadata["room_min_m"], label="room_min_m"))
        configured_max = torch.tensor(_finite_triplet(metadata["room_max_m"], label="room_max_m"))
        if not torch.equal(room_min.detach().float().cpu(), configured_min):
            raise ValueError("V78 grounding room minimum differs from the runtime map")
        if not torch.equal(room_max.detach().float().cpu(), configured_max):
            raise ValueError("V78 grounding room maximum differs from the runtime map")
        with torch.inference_mode():
            zero, _, _ = model(
                torch.ones((1, model.scene_dim), device=device),
                torch.zeros((1, model.latent_count, model.scene_dim), device=device),
            )
        if not torch.equal(zero, torch.zeros_like(zero)):
            raise ValueError("V78 grounding model failed its zero-scene runtime probe")
        return cls(
            checkpoint=root,
            model=model,
            metadata=metadata,
            scene_prefix=scene_prefix,
            room_min=room_min,
            room_max=room_max,
        )

    @property
    def full_prefix_sha256(self) -> str:
        return self._full_prefix_sha256

    @property
    def scene_tokens_sha256(self) -> str:
        return self._scene_tokens_sha256

    def assert_prefix_unchanged(self, scene_prefix: torch.Tensor) -> None:
        if prefix_sha256(scene_prefix) != self._full_prefix_sha256:
            raise RuntimeError("V78 grounding full-scene prefix changed")
        if prefix_sha256(scene_prefix[:, 1:-1].float()) != self._scene_tokens_sha256:
            raise RuntimeError("V78 grounding scene-token inventory changed")

    @torch.inference_mode()
    def predict(
        self,
        question_token_embeddings: torch.Tensor,
        *,
        scene_prefix: torch.Tensor,
        map_xyz: torch.Tensor,
        map_confidence: torch.Tensor,
    ) -> V78GroundingOutput:
        self.assert_prefix_unchanged(scene_prefix)
        if question_token_embeddings.ndim != 3:
            raise ValueError("V78 grounding question embeddings must have shape [B,T,H]")
        if question_token_embeddings.shape[0] != 1 or question_token_embeddings.shape[1] < 1:
            raise ValueError("V78 grounding requires one non-empty user question")
        question = question_token_embeddings.float().mean(dim=1)
        scene = self._scene_tokens.to(device=question.device)
        normalized, _, weights = self.model(question, scene)
        if not torch.all(weights > 0) or not torch.allclose(
            weights.sum(dim=-1), torch.ones(1, device=weights.device)
        ):
            raise RuntimeError("V78 grounding did not score every scene token positively")
        xyz = denormalize_xyz(
            normalized,
            self.room_min.to(normalized.device),
            self.room_max.to(normalized.device),
        )[0]
        if map_xyz.ndim != 2 or map_xyz.shape[-1] != 3 or map_xyz.shape[0] < 1:
            raise ValueError("V78 grounding requires a non-empty numeric map")
        if map_confidence.shape != (map_xyz.shape[0],):
            raise ValueError("V78 grounding map confidence shape mismatch")
        distances = torch.linalg.vector_norm(map_xyz.to(xyz.device).float() - xyz, dim=-1)
        nearest = int(torch.argmin(distances))
        support_distance = float(distances[nearest].detach().cpu())
        nearest_confidence = float(
            map_confidence[nearest].detach().float().cpu().clamp(0.0, 1.0)
        )
        safe_weights = weights.float().clamp_min(torch.finfo(torch.float32).tiny)
        entropy = -(safe_weights * safe_weights.log()).sum(dim=-1)[0]
        normalized_entropy = float(
            (entropy / math.log(self.model.latent_count)).detach().cpu().clamp(0.0, 1.0)
        )
        confidence = max(0.0, min(1.0, 1.0 - normalized_entropy))
        audit = {
            "schema_version": 1,
            "scene_latent_count": self.model.latent_count,
            "all_scene_tokens_scored": True,
            "minimum_attention_weight": float(weights.min().detach().cpu()),
            "maximum_attention_weight": float(weights.max().detach().cpu()),
            "normalized_attention_entropy": normalized_entropy,
            "effective_attention_token_count": float(entropy.exp().detach().cpu()),
            "confidence": confidence,
            "nearest_map_confidence": nearest_confidence,
            "support_distance_m": support_distance,
            "question_dependent_scene_retrieval": False,
            "top_k_selection_used": False,
            "scene_tokens_sha256": self._scene_tokens_sha256,
        }
        self.last_audit = audit
        coordinates = tuple(float(value) for value in xyz.detach().float().cpu().tolist())
        return V78GroundingOutput(
            xyz_m=coordinates,
            confidence=confidence,
            support_distance_m=support_distance,
            audit=audit,
        )

    def startup_audit(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "artifact": RUNTIME_ARTIFACT,
            "source_candidate_artifact": ARTIFACT,
            "architecture": ARCHITECTURE,
            "checkpoint": str(self.checkpoint),
            "weights_sha256": self.metadata["weights_sha256"],
            "base_checkpoint_sha256": self.metadata["base_checkpoint_sha256"],
            "base_runtime_config_sha256": self.metadata["base_runtime_config_sha256"],
            "full_prefix_sha256": self._full_prefix_sha256,
            "scene_tokens_sha256": self._scene_tokens_sha256,
            "scene_latent_count": self.model.latent_count,
            "scene_dim": self.model.scene_dim,
            "all_scene_tokens_scored": True,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "runtime_promotion_authorized": False,
            "official_validation_evidence": False,
        }


def authenticate_v78_grounding_checkpoint(
    checkpoint: str | Path,
    *,
    base_checkpoint_sha256: str,
    base_runtime_config_sha256: str,
    model_id: str,
    model_revision: str,
    audit: FileAccessAudit | None = None,
) -> dict[str, Any]:
    """Authenticate the optional diagnostic without loading Gemma or scene data."""

    root, _, metadata_path = _checkpoint_files(checkpoint)
    if audit is not None:
        audit.record(metadata_path)
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if raw.get("scene_dim") != 1536 or raw.get("scene_latent_count") != 256:
        raise ValueError("V78 grounding preflight dimensions changed")
    room_min = torch.tensor(_finite_triplet(raw.get("room_min_m"), label="room_min_m"))
    room_max = torch.tensor(_finite_triplet(raw.get("room_max_m"), label="room_max_m"))
    dummy_prefix = torch.zeros((1, 258, 1536), dtype=torch.float32)
    runtime = V78GroundingSidecarRuntime.load(
        root,
        scene_prefix=dummy_prefix,
        room_min=room_min,
        room_max=room_max,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_runtime_config_sha256=base_runtime_config_sha256,
        model_id=model_id,
        model_revision=model_revision,
        device="cpu",
        audit=audit,
    )
    result = runtime.startup_audit()
    result.update(
        {
            "checkpoint_inventory": sorted(EXPECTED_CHECKPOINT_FILES),
            "metadata_sha256": _sha256_file(metadata_path),
            "gemma_model_loaded": False,
            "scene_data_loaded": False,
            "passed": True,
        }
    )
    return result


__all__ = [
    "V78GroundingOutput",
    "V78GroundingSidecarRuntime",
    "authenticate_v78_grounding_checkpoint",
]

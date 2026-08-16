"""Label-free continuous-semantic grounding and bounded robot navigation.

The only environmental input to this policy is the sanitized numerical voxel
map.  User target text is embedded with the local model's matching text
embedding table and compared directly with the language-aligned visual tail of
every voxel.  No category inventory, simulator instance ID, caption, scene
graph, segmentation, or oracle file is accepted by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_PROJECTED_DIM,
    GEMMA4_PROJECTED_START,
    load_category_embeddings_selective,
    resolve_local_snapshot,
)
from semantic_3d_chat.robot.planner import NumericPathPlan, NumericWaypointPlanner
from semantic_3d_chat.robot.semantic_mapping import semantic_map_content_hash
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator
from semantic_3d_chat.scene_encoder.map_io import RUNTIME_MAP_FIELDS


def _safe_runtime_path(path: str | Path, *, purpose: str) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else Path.cwd() / candidate
    unresolved = Path(os.path.abspath(rooted))
    current = Path(unresolved.anchor)
    for part in unresolved.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{purpose} cannot use a symbolic-link path")
    if {"oracle", "qa"} & {part.casefold() for part in unresolved.parts}:
        raise ValueError(f"{purpose} cannot use an oracle or QA path")
    return unresolved


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


class ContinuousTextEncoder(Protocol):
    output_dim: int

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray: ...


class BoundedActionSurface(Protocol):
    """Numerical action seam shared by direct and refreshed/MCP runtimes."""

    def turn(self, angle_degrees: float) -> Mapping[str, Any]: ...

    def move_to(self, x: float, y: float) -> Mapping[str, Any]: ...

    def scan(self) -> Mapping[str, Any]: ...


class GemmaProjectedTextEncoder:
    """Selectively read local Gemma token rows; never construct the full LM."""

    output_dim = GEMMA4_PROJECTED_DIM

    def __init__(self, snapshot: str | Path) -> None:
        self.snapshot = _safe_runtime_path(snapshot, purpose="local model snapshot")
        if not self.snapshot.is_dir():
            raise FileNotFoundError(f"Local model snapshot is unavailable: {self.snapshot}")
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GemmaProjectedTextEncoder:
        vision = config.get("vision")
        if not isinstance(vision, Mapping) or vision.get("backend") != "gemma4":
            raise ValueError("Gemma projected grounding requires vision.backend=gemma4")
        model_id = vision.get("model_id")
        revision = vision.get("revision")
        if not isinstance(model_id, str) or not isinstance(revision, str):
            raise TypeError("Gemma model ID and revision must be pinned strings")
        return cls(resolve_local_snapshot(model_id, revision))

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("Text queries must contain non-empty strings")
        normalized = [query.strip() for query in queries]
        missing = list(dict.fromkeys(query for query in normalized if query not in self._cache))
        if missing:
            embeddings, _audit = load_category_embeddings_selective(
                self.snapshot,
                missing,
                expected_dim=self.output_dim,
            )
            for query, embedding in zip(missing, embeddings, strict=True):
                self._cache[query] = np.asarray(embedding, dtype=np.float32).copy()
        result = np.stack([self._cache[query] for query in normalized])
        if result.shape != (len(normalized), self.output_dim) or not np.isfinite(result).all():
            raise RuntimeError("Local text encoder returned invalid continuous embeddings")
        return result


@dataclass(frozen=True)
class ContinuousSemanticGrounding:
    """A numeric target estimate safe to pass to the geometry-only planner."""

    target_xyz_m: tuple[float, float, float]
    seed_xyz_m: tuple[float, float, float]
    cosine_similarity: float
    similarity_q99: float
    peak_margin_over_q99: float
    local_support_voxels: int
    scored_voxels: int
    eligible_voxels: int
    prompt_variant_index: int
    query_embedding_sha256: str
    map_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "target_xyz_m": list(self.target_xyz_m),
            "seed_xyz_m": list(self.seed_xyz_m),
            "cosine_similarity": self.cosine_similarity,
            "similarity_q99": self.similarity_q99,
            "peak_margin_over_q99": self.peak_margin_over_q99,
            "local_support_voxels": self.local_support_voxels,
            "scored_voxels": self.scored_voxels,
            "eligible_voxels": self.eligible_voxels,
            "prompt_variant_index": self.prompt_variant_index,
            "query_embedding_sha256": self.query_embedding_sha256,
            "map_sha256": self.map_sha256,
        }


class ContinuousSemanticTargetGrounder:
    """Ground user target text against every continuous voxel feature."""

    def __init__(
        self,
        map_path: str | Path,
        text_encoder: ContinuousTextEncoder,
        *,
        room_size_m: Sequence[float],
        feature_start: int = GEMMA4_PROJECTED_START,
        feature_dim: int = GEMMA4_PROJECTED_DIM,
        floor_clearance_m: float = 0.02,
        ceiling_clearance_m: float = 0.70,
        article_peak_gain: float = 0.01,
        local_radius_m: float = 0.18,
        local_similarity_margin: float = 0.015,
        local_temperature: float = 0.0075,
    ) -> None:
        self.map_path = _safe_runtime_path(map_path, purpose="semantic target map")
        if not self.map_path.is_file():
            raise FileNotFoundError(f"Semantic target map is unavailable: {self.map_path}")
        self.text_encoder = text_encoder
        if feature_start < 0 or feature_dim < 1 or text_encoder.output_dim != feature_dim:
            raise ValueError("Grounding feature slice and text embedding dimensions differ")
        room = np.asarray(room_size_m, dtype=np.float32)
        numeric = (
            floor_clearance_m,
            ceiling_clearance_m,
            article_peak_gain,
            local_radius_m,
            local_similarity_margin,
            local_temperature,
        )
        if room.shape != (3,) or not np.isfinite(room).all() or np.any(room <= 0):
            raise ValueError("room_size_m must contain three finite positive values")
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Grounding parameters must be finite")
        if (
            floor_clearance_m < 0
            or ceiling_clearance_m < 0
            or floor_clearance_m + ceiling_clearance_m >= room[2]
            or article_peak_gain < 0
            or local_radius_m <= 0
            or local_similarity_margin < 0
            or local_temperature <= 0
        ):
            raise ValueError("Grounding parameter ranges are invalid")

        with np.load(self.map_path, allow_pickle=False) as archive:
            if set(archive.files) != set(RUNTIME_MAP_FIELDS):
                raise ValueError("Semantic target map differs from the runtime numeric allowlist")
            xyz = archive["centers_world"].astype(np.float32)
            confidence = archive["confidence"].astype(np.float32)
            raw_features = archive["semantic_features"]
            if raw_features.ndim != 2 or feature_start + feature_dim > raw_features.shape[1]:
                raise ValueError("Semantic target feature slice is unavailable")
            features = raw_features[:, feature_start : feature_start + feature_dim].astype(
                np.float32
            )
            raw_header = archive["metadata_json"].item()
        if (
            xyz.ndim != 2
            or xyz.shape[1] != 3
            or confidence.shape != (len(xyz),)
            or features.shape != (len(xyz), feature_dim)
            or not len(xyz)
            or not np.isfinite(xyz).all()
            or not np.isfinite(confidence).all()
            or not np.isfinite(features).all()
        ):
            raise ValueError("Semantic target map contains invalid numeric arrays")
        if not isinstance(raw_header, str):
            raise TypeError("Semantic target map metadata must be a JSON string")
        header = json.loads(raw_header)
        metadata = header.get("metadata") if isinstance(header, dict) else None
        if not isinstance(metadata, dict) or not isinstance(metadata.get("scene_id"), str):
            raise TypeError("Semantic target map lacks an opaque scene identity")
        scene_id = metadata["scene_id"]
        if not (len(scene_id) == 12 and scene_id.startswith("scene_") and scene_id[6:].isdigit()):
            raise ValueError("Semantic target map scene identity is not opaque")

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if np.any(norms <= 1e-8):
            raise ValueError("Semantic target map contains a zero-norm aligned feature")
        self.xyz = xyz
        self.features = np.ascontiguousarray(features / norms)
        self.confidence = np.clip(confidence, 0.05, None)
        self.scene_id = scene_id
        self.map_sha256 = semantic_map_content_hash(self.map_path)
        self.article_peak_gain = float(article_peak_gain)
        self.local_radius_m = float(local_radius_m)
        self.local_similarity_margin = float(local_similarity_margin)
        self.local_temperature = float(local_temperature)
        self.eligible = (xyz[:, 2] >= floor_clearance_m) & (
            xyz[:, 2] <= room[2] - ceiling_clearance_m
        )
        if not np.any(self.eligible):
            raise ValueError("Grounding height filter removed every voxel")

    @staticmethod
    def _query_variants(target_text: str) -> tuple[str, ...]:
        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError("Target text must be a non-empty user instruction")
        target = " ".join(target_text.strip().split())
        first = target.casefold().split(maxsplit=1)[0]
        return (target,) if first in {"a", "an", "the"} else (target, f"a {target}")

    def ground(self, target_text: str) -> ContinuousSemanticGrounding:
        variants = self._query_variants(target_text)
        embeddings = np.asarray(self.text_encoder.encode_queries(variants), dtype=np.float32)
        if embeddings.shape != (len(variants), self.features.shape[1]):
            raise ValueError("Text encoder output shape differs from semantic map")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms <= 1e-8) or not np.isfinite(norms).all():
            raise ValueError("Text encoder returned a zero or nonfinite embedding")
        embeddings = embeddings / norms
        scores = self.features @ embeddings.T
        eligible_indices = np.flatnonzero(self.eligible)
        eligible_scores = scores[eligible_indices]
        peak_by_variant = eligible_scores.max(axis=0)
        selected_variant = 0
        if (
            len(variants) > 1
            and float(peak_by_variant[1] - peak_by_variant[0]) >= self.article_peak_gain
        ):
            selected_variant = 1
        selected_scores = scores[:, selected_variant]
        seed_index = int(eligible_indices[np.argmax(selected_scores[eligible_indices])])
        peak = float(selected_scores[seed_index])
        distances = np.linalg.norm(self.xyz - self.xyz[seed_index], axis=1)
        support = (
            self.eligible
            & (distances <= self.local_radius_m)
            & (selected_scores >= peak - self.local_similarity_margin)
        )
        support_indices = np.flatnonzero(support)
        if not len(support_indices):
            raise RuntimeError("Semantic target local mode unexpectedly has no support")
        weights = (
            np.exp(
                np.clip(
                    (selected_scores[support_indices] - peak) / self.local_temperature,
                    -40.0,
                    0.0,
                )
            )
            * self.confidence[support_indices]
        )
        target = np.average(self.xyz[support_indices], axis=0, weights=weights)
        q99 = float(np.quantile(selected_scores[eligible_indices], 0.99))
        return ContinuousSemanticGrounding(
            target_xyz_m=tuple(float(value) for value in target),
            seed_xyz_m=tuple(float(value) for value in self.xyz[seed_index]),
            cosine_similarity=peak,
            similarity_q99=q99,
            peak_margin_over_q99=peak - q99,
            local_support_voxels=len(support_indices),
            scored_voxels=len(self.xyz),
            eligible_voxels=len(eligible_indices),
            prompt_variant_index=selected_variant,
            query_embedding_sha256=_sha256_array(embeddings[selected_variant]),
            map_sha256=self.map_sha256,
        )


@dataclass(frozen=True)
class SemanticNavigationResult:
    success: bool
    grounding: ContinuousSemanticGrounding
    plan: NumericPathPlan
    final_position_m: tuple[float, float, float]
    movement_actions: int
    turn_actions: int
    collision_count: int
    distance_moved: float
    final_target_distance_m: float
    scan_success: bool | None
    scene_version: int

    def as_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "grounding": self.grounding.as_dict(),
            "plan": self.plan.as_dict(),
            "final_position_m": list(self.final_position_m),
            "movement_actions": self.movement_actions,
            "turn_actions": self.turn_actions,
            "collision_count": self.collision_count,
            "distance_moved": self.distance_moved,
            "final_target_distance_m": self.final_target_distance_m,
            "scan_success": self.scan_success,
            "scene_version": self.scene_version,
        }


class LabelFreeSemanticNavigator:
    """Compose continuous grounding, geometry planning, and bounded actions."""

    def __init__(
        self,
        simulator: EmbodiedCameraSimulator,
        grounder: ContinuousSemanticTargetGrounder,
        *,
        planner: NumericWaypointPlanner | None = None,
        action_surface: BoundedActionSurface | None = None,
    ) -> None:
        if simulator.state.scene_id != grounder.scene_id:
            raise ValueError("Simulator and semantic map opaque scene IDs differ")
        self.simulator = simulator
        self.grounder = grounder
        self.action_surface = action_surface or simulator
        self.planner = planner or NumericWaypointPlanner(
            simulator.collision_map,
            max_waypoint_step_m=min(
                0.50,
                float(simulator.settings.get("max_move_to_m", 1.0)),
            ),
        )

    def _face(self, target_xy: np.ndarray) -> tuple[int, bool]:
        delta = target_xy - self.simulator.state.position_xy_m
        if float(np.linalg.norm(delta)) <= 1e-8:
            return 0, True
        desired = math.degrees(math.atan2(-float(delta[0]), float(delta[1])))
        remaining = (desired - self.simulator.state.body_yaw_degrees + 180.0) % 360.0 - 180.0
        maximum = float(self.simulator.settings["max_turn_degrees"])
        actions = 0
        while abs(remaining) > 1e-7:
            step = max(-maximum, min(maximum, remaining))
            result = self.action_surface.turn(step)
            actions += 1
            if not result["success"]:
                return actions, False
            remaining -= step
        return actions, True

    def navigate(
        self,
        target_text: str,
        *,
        scan_on_arrival: bool = True,
    ) -> SemanticNavigationResult:
        grounding = self.grounder.ground(target_text)
        start = self.simulator.state.position_xy_m.copy()
        target_xy = np.asarray(grounding.target_xyz_m[:2], dtype=np.float64)
        plan = self.planner.plan(start, target_xy)
        distance = 0.0
        collisions = 0
        movement_actions = 0
        movement_ok = True
        for waypoint in plan.waypoints_xy_m:
            result = self.action_surface.move_to(*waypoint)
            movement_actions += 1
            distance += float(result["distance_moved"])
            collisions += int(bool(result["collision"]))
            if not result["success"]:
                movement_ok = False
                break
        turn_actions, facing_ok = self._face(target_xy) if movement_ok else (0, False)
        scan_success: bool | None = None
        if movement_ok and facing_ok and scan_on_arrival:
            scan_success = bool(self.action_surface.scan()["success"])
        final = self.simulator.get_robot_state()
        final_position = tuple(float(value) for value in final["position_m"])
        return SemanticNavigationResult(
            success=movement_ok and facing_ok and collisions == 0 and (scan_success is not False),
            grounding=grounding,
            plan=plan,
            final_position_m=final_position,
            movement_actions=movement_actions,
            turn_actions=turn_actions,
            collision_count=collisions,
            distance_moved=distance,
            final_target_distance_m=float(
                np.linalg.norm(np.asarray(final_position[:2]) - target_xy)
            ),
            scan_success=scan_success,
            scene_version=int(final["scene_version"]),
        )


__all__ = [
    "BoundedActionSurface",
    "ContinuousSemanticGrounding",
    "ContinuousSemanticTargetGrounder",
    "ContinuousTextEncoder",
    "GemmaProjectedTextEncoder",
    "LabelFreeSemanticNavigator",
    "SemanticNavigationResult",
]

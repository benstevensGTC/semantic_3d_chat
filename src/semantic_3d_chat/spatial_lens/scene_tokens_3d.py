"""Turn the semantic point cloud into tokens Gemma can read as a scene.

The 1536-D payload fused into every voxel is not an arbitrary embedding: it is
the output of Gemma's *own* vision projector, so it already lives in the space
the decoder consumes image tokens from.  That means a 3D scene can be handed to
the language model through the pathway it already understands, with nothing
trained.

The room's floor is divided into a grid of columns.  Each column is pooled into
one token from the voxels standing above it -- weighted by height and
observation count, so furniture dominates over floor -- and the grid is emitted
in raster order between Gemma's native image-boundary tokens.  Position is
carried by token order exactly as it is for image patches.

This is not a rendered picture: no camera sees the whole room, and every token
is fused from many views through exact depth and pose.  It is also not a text
summary: nothing here is named, described or discretized into words.  It is the
3D semantic field itself, in the model's native input format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_3d_chat.spatial_lens.perceive import SemanticCloud

TOKENS_SCHEMA = "semantic_3d_chat.spatial_lens.scene_tokens_3d.v1"


@dataclass(frozen=True)
class SceneTokens3D:
    """A bird's-eye grid of fused semantic tokens, plus its geometry."""

    tokens: np.ndarray  # [grid*grid, 1536] float32, raster order
    occupancy: np.ndarray  # [grid, grid] bool: did any voxel land here
    grid: int
    room_size_m: tuple[float, float, float]
    # Where each token actually is, in metres. Raster order already implies a
    # position, but only as an index the reader has to decode. Carrying the
    # centroid lets the decoder receive position as a rotation instead -- see
    # semantic_3d_chat.language.rope3d_patch.
    centroids_m: np.ndarray | None = None

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def occupied_fraction(self) -> float:
        """How much of the grid carries any observation at all."""

        return float(self.occupancy.mean())

    def cell_center_m(self, index: int) -> tuple[float, float]:
        row, column = divmod(index, self.grid)
        width, depth, _ = self.room_size_m
        return (
            (column + 0.5) * width / self.grid - width / 2.0,
            (row + 0.5) * depth / self.grid - depth / 2.0,
        )


def build_scene_tokens_3d(
    cloud: SemanticCloud,
    *,
    grid: int = 16,
    floor_margin_m: float = 0.09,
    height_weighting: float = 1.0,
) -> SceneTokens3D:
    """Pool the semantic point cloud into a raster grid of scene tokens.

    Floor and ceiling voxels are dropped so a column reports what *stands* in
    it.  Remaining voxels are averaged with a weight that grows with height, so
    a chair back is not washed out by the patch of floor beneath it.
    """

    if grid < 2:
        raise ValueError("grid must be at least 2")
    width, depth, height = cloud.room_size_m
    centers = np.asarray(cloud.centers_m, dtype=np.float64)
    features = np.asarray(cloud.features, dtype=np.float32)

    standing = (centers[:, 2] > floor_margin_m) & (
        centers[:, 2] < height - floor_margin_m
    )
    centers, features = centers[standing], features[standing]
    counts = np.asarray(cloud.counts, dtype=np.float32)[standing]

    columns = np.clip(
        ((centers[:, 0] + width / 2.0) / width * grid).astype(int), 0, grid - 1
    )
    rows = np.clip(
        ((centers[:, 1] + depth / 2.0) / depth * grid).astype(int), 0, grid - 1
    )
    cell = rows * grid + columns

    weights = (1.0 + height_weighting * centers[:, 2]).astype(np.float32) * np.sqrt(
        np.maximum(counts, 1.0)
    )
    dimension = features.shape[1]
    totals = np.zeros((grid * grid, dimension), dtype=np.float64)
    mass = np.zeros(grid * grid, dtype=np.float64)
    np.add.at(totals, cell, features * weights[:, None])
    np.add.at(mass, cell, weights)
    places = np.zeros((grid * grid, 3), dtype=np.float64)
    np.add.at(places, cell, centers * weights[:, None])

    occupied = mass > 0
    tokens = np.zeros_like(totals, dtype=np.float32)
    tokens[occupied] = (totals[occupied] / mass[occupied, None]).astype(np.float32)

    # Empty columns stay zero. An earlier comment here claimed they were given
    # the mean of the occupied ones, but the expression multiplied that mean by
    # zero, so they never were. Zero is the defensible choice -- a column with
    # nothing standing in it has nothing to report -- but it has a consequence
    # worth stating rather than hiding: a typical room fills well under half its
    # columns, so most of a real scene is already identical to the zeroed
    # control, and the gap between them is correspondingly smaller than the word
    # "zeroed" suggests. Callers report occupied_fraction so that is visible.

    # An empty column still has a place: its cell centre on the floor.
    centroids = np.zeros((grid * grid, 3), dtype=np.float32)
    centroids[occupied] = (places[occupied] / mass[occupied, None]).astype(np.float32)
    empty = np.flatnonzero(~occupied)
    if empty.size:
        rows_e, columns_e = np.divmod(empty, grid)
        centroids[empty, 0] = (columns_e + 0.5) * width / grid - width / 2.0
        centroids[empty, 1] = (rows_e + 0.5) * depth / grid - depth / 2.0

    return SceneTokens3D(
        tokens=tokens,
        occupancy=occupied.reshape(grid, grid),
        grid=grid,
        room_size_m=(float(width), float(depth), float(height)),
        centroids_m=centroids,
    )


def shuffled(tokens: SceneTokens3D, *, seed: int = 20260816) -> SceneTokens3D:
    """Same tokens, scrambled layout: the control for spatial grounding."""

    generator = np.random.default_rng(seed)
    order = generator.permutation(tokens.token_count)
    return SceneTokens3D(
        tokens=tokens.tokens[order].copy(),
        occupancy=tokens.occupancy,
        grid=tokens.grid,
        room_size_m=tokens.room_size_m,
        centroids_m=tokens.centroids_m,
    )


def zeroed(tokens: SceneTokens3D) -> SceneTokens3D:
    """No scene at all: the control for whether the tokens matter."""

    return SceneTokens3D(
        tokens=np.zeros_like(tokens.tokens),
        occupancy=tokens.occupancy,
        grid=tokens.grid,
        room_size_m=tokens.room_size_m,
        centroids_m=tokens.centroids_m,
    )


__all__ = [
    "TOKENS_SCHEMA",
    "SceneTokens3D",
    "build_scene_tokens_3d",
    "shuffled",
    "zeroed",
]

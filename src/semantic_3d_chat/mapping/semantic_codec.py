"""Semantic feature storage codecs.

The first experiment deliberately uses an identity codec: no dimensions are
removed and no learned or lossy semantic compression is applied.  A float16
storage cast is allowed because the uncompressed encoder outputs are separately
cached and the map needs a practical on-disk representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class SemanticCodec(ABC):
    """Interface for future rate-versus-understanding codec experiments."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable codec identifier written to map metadata."""

    @abstractmethod
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode a ``[..., feature_dim]`` semantic tensor for storage."""

    @abstractmethod
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode stored codes without changing their feature dimension."""


def _validate_semantic_array(features: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(features)
    if array.ndim < 1 or array.shape[-1] < 1:
        raise ValueError(f"{label} must have a non-empty final feature dimension")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{label} must be numeric, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    if array.size and array.ndim >= 2:
        flat = array.reshape(-1, array.shape[-1]).astype(np.float32, copy=False)
        if np.any(np.linalg.norm(flat, axis=1) == 0):
            raise ValueError(f"{label} contains a zero-norm semantic vector")
    return array


@dataclass(frozen=True)
class IdentitySemanticCodec(SemanticCodec):
    """Dimension-preserving storage codec used by the initial baseline.

    ``storage_dtype`` defaults to float16.  Decoding defaults to float32 for
    numerically stable downstream pooling, while preserving every dimension.
    Set ``decoded_dtype=None`` to retain the storage dtype on decode.
    """

    storage_dtype: np.dtype | type | str = np.float16
    decoded_dtype: np.dtype | type | str | None = np.float32

    def __post_init__(self) -> None:
        storage_dtype = np.dtype(self.storage_dtype)
        if storage_dtype.kind != "f":
            raise TypeError("IdentitySemanticCodec storage_dtype must be floating point")
        if self.decoded_dtype is not None and np.dtype(self.decoded_dtype).kind != "f":
            raise TypeError("IdentitySemanticCodec decoded_dtype must be floating point")
        object.__setattr__(self, "storage_dtype", storage_dtype)
        if self.decoded_dtype is not None:
            object.__setattr__(self, "decoded_dtype", np.dtype(self.decoded_dtype))

    @property
    def name(self) -> str:
        return f"identity-{self.storage_dtype.name}"

    def encode(self, features: np.ndarray) -> np.ndarray:
        array = _validate_semantic_array(features, label="features")
        return array.astype(self.storage_dtype, copy=True)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        array = _validate_semantic_array(codes, label="codes")
        if self.decoded_dtype is None:
            return array.copy()
        return array.astype(self.decoded_dtype, copy=True)

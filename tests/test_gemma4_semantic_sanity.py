from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_PROJECTED_DIM,
    GEMMA4_PROJECTED_SLICE,
    GEMMA4_PROJECTED_START,
    GEMMA4_TOKEN_EMBEDDING_KEY,
    GEMMA4_TOTAL_SEMANTIC_DIM,
    category_token_ids,
    load_category_embeddings_selective,
    mean_token_embeddings,
)
from semantic_3d_chat.evaluation.semantic_sanity import (
    OracleTarget,
    SemanticQuery,
    extract_feature_slice,
    score_semantic_queries,
)


class FakeTokenizer:
    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        self.calls.append((text, add_special_tokens))
        return {"input_ids": self.mapping[text]}


def _unit(index: int) -> np.ndarray:
    vector = np.zeros(GEMMA4_PROJECTED_DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _fused(vector: np.ndarray) -> np.ndarray:
    feature = np.zeros(GEMMA4_TOTAL_SEMANTIC_DIM, dtype=np.float32)
    feature[GEMMA4_PROJECTED_START:] = vector
    return feature


def test_mean_token_embeddings_uses_every_token_and_normalizes() -> None:
    weight = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    values = mean_token_embeddings([[0, 1], [2]], weight)
    assert values.shape == (2, 3)
    assert np.allclose(values[0], np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))
    assert np.allclose(values[1], [0.0, 0.0, 1.0])


def test_selective_loader_reads_only_requested_gemma_key_and_rows(tmp_path: Path) -> None:
    weight = torch.zeros(5, GEMMA4_PROJECTED_DIM, dtype=torch.float32)
    weight[0, 0] = 2.0
    weight[1, 0] = 4.0
    weight[2, 1] = 3.0
    save_file(
        {
            GEMMA4_TOKEN_EMBEDDING_KEY: weight,
            "unrelated.parameter": torch.ones(2, 2),
        },
        tmp_path / "model.safetensors",
    )
    tokenizer = FakeTokenizer({"alpha": [0, 1], "beta": [2]})
    values, metadata = load_category_embeddings_selective(
        tmp_path,
        ["alpha", "beta"],
        tokenizer=tokenizer,
    )
    assert np.allclose(values[0], _unit(0))
    assert np.allclose(values[1], _unit(1))
    assert tokenizer.calls == [("alpha", False), ("beta", False)]
    assert metadata["loaded_parameter_keys"] == [GEMMA4_TOKEN_EMBEDDING_KEY]
    assert metadata["unique_token_rows_read"] == 3
    assert metadata["weight_shape"] == [5, GEMMA4_PROJECTED_DIM]
    assert metadata["selective_row_read"] is True


def test_gemma_slice_reuses_dimension_agnostic_localization_scorer() -> None:
    centers = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=np.float32,
    )
    features = np.stack(
        [_fused(_unit(0)), _fused(_unit(0)), _fused(_unit(1)), _fused(_unit(1))]
    )
    queries = [
        SemanticQuery("query_000", "alpha", "alpha", "alpha"),
        SemanticQuery("query_001", "beta", "beta", "beta"),
    ]
    targets = [
        OracleTarget("i_000", "alpha", (-0.05, -0.05, -0.05), (0.15, 0.05, 0.05)),
        OracleTarget("i_001", "beta", (0.95, -0.05, -0.05), (1.15, 0.05, 0.05)),
    ]
    selected = extract_feature_slice(features, GEMMA4_PROJECTED_SLICE)
    assert selected.shape == (4, GEMMA4_PROJECTED_DIM)
    assert np.allclose(selected[:2], _unit(0))
    metrics, similarities = score_semantic_queries(
        centers,
        features,
        queries,
        np.stack([_unit(0), _unit(1)]),
        targets,
        top_k=1,
        feature_slice=GEMMA4_PROJECTED_SLICE,
    )
    assert similarities.shape == (4, 2)
    assert metrics["aggregate"]["top1_localization_accuracy"] == 1.0
    assert metrics["aggregate"]["top_k_localization_accuracy"] == 1.0
    assert metrics["aggregate"]["mean_precision_at_k"] == 1.0


def test_category_tokenization_and_dimension_failures_are_explicit(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer({"alpha": [1, 2]})
    assert category_token_ids(["alpha"], tokenizer) == [[1, 2]]
    with pytest.raises(ValueError, match="at least one token"):
        mean_token_embeddings([[]], np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="3072"):
        extract_feature_slice(
            np.ones((2, GEMMA4_TOTAL_SEMANTIC_DIM - 1), dtype=np.float32),
            GEMMA4_PROJECTED_SLICE,
        )

    bad_weight = torch.ones(4, 12)
    save_file({GEMMA4_TOKEN_EMBEDDING_KEY: bad_weight}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="1536"):
        load_category_embeddings_selective(tmp_path, ["alpha"], tokenizer=tokenizer)

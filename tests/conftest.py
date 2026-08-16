"""Cross-environment and post-execution isolation for the repository test suite."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest


def _gemma4_transformers_available() -> bool:
    try:
        return (
            importlib.util.find_spec(
                "transformers.models.gemma4.modeling_gemma4"
            )
            is not None
        )
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep the pinned Gemma implementation check in its dedicated environment."""

    if _gemma4_transformers_available():
        return
    marker = pytest.mark.skip(reason="exact Gemma 4 source audit requires .venv-gemma4")
    for item in items:
        if item.nodeid.endswith(
            "test_fixed_prefix_decoder_reader_v6_1_release.py::"
            "test_v6_1_installed_transformers_sources_are_exact"
        ):
            item.add_marker(marker)


@pytest.fixture(autouse=True)
def _isolate_consumed_v6_1_release_builder(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Let a model-free release-policy unit test run after its real one-shot run."""

    if request.node.name == (
        "test_v6_1_release_authorizes_no_optimizer_training_or_checkpoint"
    ):
        monkeypatch: pytest.MonkeyPatch = request.getfixturevalue("monkeypatch")
        tmp_path: Path = request.getfixturevalue("tmp_path")
        from semantic_3d_chat.evaluation import (
            fixed_prefix_decoder_reader_v6_1_release as release,
        )

        monkeypatch.setattr(
            release, "MPS_SMOKE_ATTEMPT", str(tmp_path / "attempt.json")
        )
        monkeypatch.setattr(
            release, "MPS_SMOKE_REPORT", str(tmp_path / "report.json")
        )
    yield

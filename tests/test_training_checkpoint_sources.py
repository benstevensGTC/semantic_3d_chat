from pathlib import Path

import pytest

from semantic_3d_chat.training.train_adapter import resolve_checkpoint_sources


def test_explicit_resume_overrides_configured_initialization() -> None:
    resume = Path("data/checkpoints/run/epoch_012")

    resolved_resume, resolved_initialize = resolve_checkpoint_sources(
        cli_resume=resume,
        cli_initialize_from=None,
        training_config={"initialize_from": "data/checkpoints/seed/epoch_036"},
    )

    assert resolved_resume == resume
    assert resolved_initialize is None


def test_explicit_resume_and_explicit_initialization_conflict() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_checkpoint_sources(
            cli_resume=Path("data/checkpoints/run/epoch_012"),
            cli_initialize_from=Path("data/checkpoints/seed/epoch_036"),
            training_config={},
        )


def test_configured_resume_and_initialization_conflict() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_checkpoint_sources(
            cli_resume=None,
            cli_initialize_from=None,
            training_config={
                "resume_from": "data/checkpoints/run/epoch_012",
                "initialize_from": "data/checkpoints/seed/epoch_036",
            },
        )


def test_explicit_initialization_does_not_override_configured_resume() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_checkpoint_sources(
            cli_resume=None,
            cli_initialize_from=Path("data/checkpoints/seed/epoch_036"),
            training_config={"resume_from": "data/checkpoints/run/epoch_012"},
        )

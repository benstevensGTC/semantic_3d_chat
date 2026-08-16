from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.training.soft_prompt_teacher_v59 import (
    ExpansionTeacherTarget,
    load_expansion_teachers,
    save_expansion_teachers,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


def test_v59_teacher_cache_is_numeric_opaque_training_only(tmp_path: Path) -> None:
    destination = tmp_path / "teacher_cache"
    targets = [
        ExpansionTeacherTarget("scene_000033", "q_000023", torch.randn(1, 4, 8)),
        ExpansionTeacherTarget("scene_000034", "q_000084", torch.randn(1, 4, 8)),
    ]
    hashes = save_expansion_teachers(
        destination,
        targets=targets,
        base_checkpoint_sha256=_A,
        base_runtime_config_sha256=_B,
        source_control_checkpoint_sha256=_C,
        selection_sha256=_D,
    )
    assert set(hashes) == {"weights_sha256", "metadata_sha256"}
    loaded, metadata = load_expansion_teachers(destination)
    assert set(loaded) == {target.key for target in targets}
    assert metadata["environmental_text_inputs"] == []
    assert metadata["runtime_load_permitted"] is False
    serialized = json.dumps(metadata).casefold()
    assert "answer" not in serialized
    assert "caption" not in serialized
    assert "oracle" not in serialized
    with pytest.raises(FileExistsError):
        save_expansion_teachers(
            destination,
            targets=targets,
            base_checkpoint_sha256=_A,
            base_runtime_config_sha256=_B,
            source_control_checkpoint_sha256=_C,
            selection_sha256=_D,
        )

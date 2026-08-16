from __future__ import annotations

from pathlib import Path

import yaml

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)


def test_v56_runtime_config_is_sanitized_alias_of_v54() -> None:
    path = Path("configs/runtime/gemma4_v56_question_control.yaml")
    source = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(source)
    v56 = load_runtime_config(path)
    v54 = load_runtime_config("configs/runtime/gemma4_v54.yaml")

    assert "_base_" not in raw
    assert {
        key: value for key, value in v56.items() if not key.startswith("_")
    } == {key: value for key, value in v54.items() if not key.startswith("_")}
    assert effective_runtime_config_sha256(v56) == effective_runtime_config_sha256(v54)
    assert "batch" not in v56
    assert set(v56["paths"]) == {
        "data_root",
        "reports_root",
        "maps_root",
        "checkpoints_root",
    }
    assert not ({"oracle_root", "qa_root", "rendered_root", "features_root"} & set(v56["paths"]))
    comments = "\n".join(
        line.casefold() for line in source.splitlines() if line.lstrip().startswith("#")
    )
    assert all(
        term not in comments
        for term in (
            "qa",
            "oracle",
            "object",
            "relationship",
            "counterfactual",
            "scene-generation",
        )
    )

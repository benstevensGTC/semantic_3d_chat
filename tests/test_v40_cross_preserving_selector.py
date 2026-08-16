from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation import v40_cross_preserving_selector as selector
from semantic_3d_chat.training.train_cross_preserving_v40 import v40_contract

CONFIG_PATH = PROJECT_ROOT / selector.DEFAULT_CONFIG


def _config() -> dict:
    return load_config(CONFIG_PATH)


def test_selector_surface_is_exact_existing_v28_layer14_lora_b() -> None:
    assert selector._QUERY_BANK == "extension_v28_stage_b_query"
    assert selector._QUERY_PARAMETER_NAMES == (
        "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b",
    )
    assert selector._QUERY_SHAPES == ((4096, 4),)
    assert selector._QUERY_MODULES == (
        "model.language_model.layers.14.self_attn.q_proj",
    )


def test_query_state_selects_only_b_and_v30_mutation_remains_frozen() -> None:
    target = selector._QUERY_PARAMETER_NAMES[0]
    v30 = "lora_banks.extension_v30_joint_pair_query.adapters.0.lora_b"
    tensors = {
        target: torch.zeros(4096, 4),
        "lora_banks.extension_v28_stage_b_query.adapters.1.lora_a": torch.zeros(4, 1536),
        v30: torch.ones(2048, 8),
    }
    assert tuple(selector._query_state(tensors)) == ("lora_b",)
    frozen = selector._frozen_excluding_query(tensors)
    assert target not in frozen
    assert v30 in frozen
    assert len(frozen) == 2


def test_target_installer_updates_b_and_keeps_a_bit_exact() -> None:
    class Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_a = torch.nn.Parameter(torch.randn(4, 1536))
            self.lora_b = torch.nn.Parameter(torch.zeros(4096, 4))

    adapter = Adapter()
    a_before = adapter.lora_a.detach().clone()
    target = torch.full((4096, 4), 0.25)
    audit = selector.install_v40_target_b(adapter, {"lora_b": target})
    assert torch.equal(adapter.lora_a, a_before)
    assert torch.equal(adapter.lora_b, target)
    assert audit["lora_a_before_sha256"] == audit["lora_a_after_sha256"]


def test_checkpoint_root_alias_is_rejected_before_any_file_read(tmp_path: Path) -> None:
    contract = v40_contract(_config())
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="differs from terminal authorization"):
        selector._checkpoint_paths_or_raise(tmp_path, contract)


def test_incomplete_envelope_refuses_before_evaluator_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v40_contract(_config())
    monkeypatch.setattr(selector, "_AUTHORIZED_CHECKPOINT_ROOT", tmp_path.resolve())
    (tmp_path / "update_000").mkdir()
    constructed = False

    def forbidden_factory(*_args, **_kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("validation evaluator must not be constructed")

    with pytest.raises(FileNotFoundError, match="exact completed"):
        selector.select_v40(CONFIG_PATH, tmp_path, evaluator_factory=forbidden_factory)
    assert constructed is False


def test_exact_envelope_requires_every_authorized_step(tmp_path: Path, monkeypatch) -> None:
    contract = v40_contract(_config())
    monkeypatch.setattr(selector, "_AUTHORIZED_CHECKPOINT_ROOT", tmp_path.resolve())
    for step in contract.saved_optimizer_steps[:-1]:
        (tmp_path / f"update_{step:03d}").mkdir()
    with pytest.raises(FileNotFoundError, match="exact completed"):
        selector._checkpoint_paths_or_raise(tmp_path, contract)


def test_selector_module_has_unique_definitions_and_no_stale_v30_surface() -> None:
    source = Path(selector.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert {name for name, count in Counter(names).items() if count > 1} == set()
    assert "131_072" not in source
    assert "131072" not in source
    assert "fresh Adam" not in source
    assert '"extension_v30_joint_pair_query"' not in source
    assert "module_collection_state_sha256(self.bundle.checkpoint_modules)" in source
    assert "!= tensor_state_sha256(tensors)" in source

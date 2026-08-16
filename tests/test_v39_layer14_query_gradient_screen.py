from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semantic_3d_chat.evaluation import v39_layer14_query_gradient_screen as v39


@pytest.fixture(scope="module")
def preflight() -> dict:
    return v39.preflight_v39()


def test_v39_preflight_authenticates_exact_terminal_and_source(preflight: dict) -> None:
    assert preflight["passed"] is True
    assert preflight["terminal"] == {
        "path": "reports/gemma4/metrics/v38_update8_terminal_gate.json",
        "sha256": "1015949e802abccd562f7762cc01111818646527f3366aeaf01de3854bbe164a",
        "exact_revision_2_authorization_verified": True,
    }
    source = preflight["source"]
    assert source["source_full_tensor_state_sha256"] == (
        "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
    )
    assert source["target_source_state_sha256"] == (
        "9ff9d535a094f96328483c46ff8c8ea5fca30edc35878492976c35f8674a9f87"
    )
    assert source["frozen_excluding_target_state_sha256"] == (
        "7f33e541d36de33b10ceeac25e5f40374bffd1cf4b234af7a6b6341198b85360"
    )
    assert source["source_optimizer_file_opened"] is False
    assert source["update8_checkpoint_opened"] is False


def test_v39_preflight_locks_exact_layer14_surface(preflight: dict) -> None:
    surface = preflight["target_surface"]
    assert surface["existing_bank"] == "extension_v28_stage_b_query"
    assert surface["existing_adapter_index"] == 1
    assert surface["module_path"] == (
        "model.language_model.layers.14.self_attn.q_proj"
    )
    assert surface["parameter_names"] == [
        "lora_banks.extension_v28_stage_b_query.adapters.1.lora_a",
        "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b",
    ]
    assert surface["parameter_shapes"] == [[4, 1536], [4096, 4]]
    assert surface["tensor_count"] == 2
    assert surface["parameter_count"] == 22_528
    assert surface["rank"] == 4
    assert surface["alpha"] == 8.0
    assert surface["dropout"] == 0.0


def test_v39_preflight_uses_exact_train_only_inventory(preflight: dict) -> None:
    inventory = preflight["screen_inventory"]
    assert inventory["priority_unit_count"] == 8
    assert inventory["broad_row_count"] == 8
    assert inventory["priority_families"] == [
        "book_support",
        "picture_support",
    ] * 4
    assert len(set(inventory["priority_question_keys"])) == 8
    assert len(
        {(row["scene_id"], row["question_id"]) for row in inventory["broad_rows"]}
    ) == 8
    assert preflight["qa_audit"]["train_question_count"] == 384
    assert len(preflight["qa_audit"]["train_scene_ids"]) == 16
    assert preflight["qa_audit"]["validation_qa_loaded"] is False


def test_v39_preflight_loads_no_gemma_map_optimizer_or_forbidden_data(
    preflight: dict,
) -> None:
    assert preflight["gemma_loaded"] is False
    assert preflight["scene_maps_loaded"] is False
    assert preflight["optimizer_constructed"] is False
    assert preflight["optimizer_file_opened"] is False
    assert preflight["validation_qa_loaded"] is False
    assert preflight["oracle_loaded"] is False
    assert preflight["final_test_scenes_touched"] is False
    assert preflight["forbidden_file_access_count"] == 0
    assert not any("/maps/" in path for path in preflight["loaded_file_inventory"])
    assert not any(
        Path(path).name == "optimizer.pt"
        or Path(path).name in {
            "v34_update32_terminal_gate.json",
            "v33_update64_terminal_gate.json",
        }
        for path in preflight["loaded_file_inventory"]
    )
    assert preflight["source_cache_evidence"]["external_terminal_report_opened"] is False
    assert preflight["source_cache_evidence"]["optimizer_file_opened"] is False


def test_v39_source_cache_evidence_never_replays_terminal_or_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import semantic_3d_chat.evaluation.v33_terminal_gate as v33_terminal
    import semantic_3d_chat.evaluation.v34_terminal_gate as v34_terminal
    import semantic_3d_chat.training.train_block_cross_v35 as v35_training
    from semantic_3d_chat.config import load_config
    from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("recursive terminal/optimizer path must not be called")

    monkeypatch.setattr(v35_training, "require_v34_terminal_gate", forbidden)
    monkeypatch.setattr(v34_terminal, "audit_v34_update32", forbidden)
    monkeypatch.setattr(v33_terminal, "audit_v33_update64", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    config = load_config(v39.DEFAULT_CONFIG)
    source = v39._authenticate_source(config)
    loader = v39.v38_loader_config(config)
    split = v31_contract(loader)
    evidence = v39.v39_source_cache_evidence(
        loader,
        source.metadata,
        scene_ids=split.train_scene_ids,
        manifest_scene_ids=(*split.train_scene_ids, *split.validation_scene_ids),
    )
    assert evidence["scene_count"] == 16
    assert evidence["external_terminal_report_opened"] is False
    assert evidence["v34_recursive_audit_called"] is False
    assert evidence["v33_recursive_audit_called"] is False
    assert evidence["optimizer_file_opened"] is False
    source_text = Path(v39.__file__).read_text(encoding="utf-8")
    assert "require_v34_terminal_gate" not in source_text
    assert "audit_v34_update32" not in source_text
    assert "audit_v33_update64" not in source_text


def test_v39_cache_evidence_accepts_unsorted_train_then_validation_manifest_and_boundary() -> None:
    from semantic_3d_chat.config import artifact_root, load_config
    from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract
    from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
        validate_v37_training_cache_boundary,
    )

    config = load_config(v39.DEFAULT_CONFIG)
    source = v39._authenticate_source(config)
    loader = v39.v38_loader_config(config)
    split = v31_contract(loader)
    manifest = (*split.train_scene_ids, *split.validation_scene_ids)
    assert tuple(manifest) != tuple(sorted(manifest))
    evidence = v39.v39_source_cache_evidence(
        loader,
        source.metadata,
        scene_ids=split.train_scene_ids,
        manifest_scene_ids=manifest,
    )
    fake_caches = {scene_id: object() for scene_id in split.train_scene_ids}
    boundary = validate_v37_training_cache_boundary(
        cache_audit=evidence,
        caches=fake_caches,  # type: ignore[arg-type]
        config=loader,
        train_scene_ids=split.train_scene_ids,
        validation_scene_ids=split.validation_scene_ids,
    )
    expected_maps = [
        str(
            (
                artifact_root(loader, "maps") / scene_id / "voxel_map.npz"
            ).resolve()
        )
        for scene_id in split.train_scene_ids
    ]
    assert evidence["loaded_environment_files"] == expected_maps
    assert boundary["loaded_environment_files"] == expected_maps
    assert boundary["exact_train_scene_count"] == 16
    assert evidence["authenticated_manifest_scene_count"] == 22
    assert evidence["validation_scene_ids_loaded"] == []
    assert evidence["validation_environment_maps_loaded"] is False


def test_v39_cache_evidence_rejects_authenticated_source_tamper() -> None:
    from semantic_3d_chat.config import load_config
    from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract

    config = load_config(v39.DEFAULT_CONFIG)
    source = v39._authenticate_source(config)
    loader = v39.v38_loader_config(config)
    split = v31_contract(loader)
    tampered = json.loads(json.dumps(source.metadata))
    tampered["v38_query_recovery"]["scene_cache"][
        "source_prefix_sha256_by_scene"
    ][split.train_scene_ids[0]] = "0" * 64
    with pytest.raises(ValueError, match="differs from inherited V35 evidence"):
        v39.v39_source_cache_evidence(
            loader,
            tampered,
            scene_ids=split.train_scene_ids,
            manifest_scene_ids=(*split.train_scene_ids, *split.validation_scene_ids),
        )


def test_v39_file_boundary_blocks_any_optimizer_open() -> None:
    from semantic_3d_chat.config import load_config

    config = load_config(v39.DEFAULT_CONFIG)
    optimizer = Path(
        "data_gemma4/checkpoints/"
        "gemma4_v38_diverse28_query_recovery/update_008/optimizer.pt"
    ).resolve()
    assert optimizer.is_file()
    audit = v39.FileAccessAudit(
        v39._forbidden_roots(config),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit, pytest.raises(PermissionError, match="Blocked forbidden"):
        optimizer.open("rb")
    assert str(optimizer) in audit.forbidden_accesses()


def test_v39_terminal_revision2_authorizes_measurement_not_training() -> None:
    report, authorization = v39._terminal_authorization()
    assert report["conditional_v39_v28_layer14_gradient_cosine_screen_authorized"]
    assert report["v39_training_authorized"] is False
    assert authorization["gradient_computation_authorized"] is True
    assert authorization["temporary_requires_grad_toggle_authorized"] is True
    assert authorization["gradient_accumulation_across_objectives_authorized"] is False
    assert authorization["optimizer_construction_authorized"] is False
    assert authorization["optimizer_step_authorized"] is False
    assert authorization["parameter_or_buffer_write_authorized"] is False
    assert authorization["training_authorized"] is False
    assert authorization["validation_access_authorized"] is False
    assert authorization["final_test_access_authorized"] is False
    assert authorization["oracle_access_authorized"] is False


def test_v39_rejects_any_terminal_byte_change(tmp_path: Path) -> None:
    source = Path("reports/gemma4/metrics/v38_update8_terminal_gate.json")
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="exact V38 terminal revision 2"):
        v39._terminal_authorization(changed)


def test_v39_objective_separates_scene_discrimination_from_answer_nll() -> None:
    objective = v39.objective_contract()
    assert objective["answer_nll_weight"] == 0.5
    assert objective["side_hinge_weight"] == 8.0
    assert objective["cross_prefix_hinge_weight"] == 4.0
    assert objective["scene_discriminative_components_reported_separately"] is True
    assert objective["answer_nll_component_reported_separately"] is True
    assert objective["gradient_accumulation_across_objectives"] is False


def test_v39_cosine_and_conflict_matrix_are_signed_and_deterministic() -> None:
    vectors = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([0.5, 0.5]),
        "c": torch.tensor([-1.0, 0.0]),
    }
    assert v39.gradient_cosine(vectors["a"], vectors["b"]) == pytest.approx(
        2**-0.5
    )
    assert v39.gradient_dot(vectors["a"], vectors["c"]) == -1.0
    matrix = v39.cosine_conflict_matrix(vectors)
    assert matrix["names"] == ["a", "b", "c"]
    assert matrix["cosine"][0][0] == 1.0
    assert matrix["conflict"][0][2] is True
    assert matrix["negative_cosine_pair_count"] == 2
    assert {(row["first"], row["second"]) for row in matrix["negative_cosine_pairs"]} == {
        ("a", "c"),
        ("b", "c"),
    }


def _priority_rows() -> list[dict]:
    return [
        {
            "gradients": {
                "full_priority": {
                    "all_finite": True,
                    "all_target_tensors_nonzero": True,
                }
            }
        }
        for _ in range(8)
    ]


def test_v39_predeclared_pass_contract_requires_directional_compatibility() -> None:
    aligned = {
        "priority_aggregate": torch.ones(22_528),
        "book_support_aggregate": torch.ones(22_528),
        "picture_support_aggregate": torch.ones(22_528) * 2,
        "book_scene_discriminative_aggregate": torch.ones(22_528) * 0.8,
        "picture_scene_discriminative_aggregate": torch.ones(22_528) * 0.7,
        "broad_retention_aggregate": torch.ones(22_528) * 0.5,
        "scene_discriminative_aggregate": torch.ones(22_528) * 0.75,
        "cross_prefix_maintenance_aggregate": torch.ones(22_528) * 0.25,
        "proposed_training_aggregate": torch.ones(22_528) * 1.5,
    }
    passed = v39.evaluate_pass_contract(
        priority_rows=_priority_rows(),
        aggregates=aligned,
        state_exact=True,
        model_versions_exact=True,
        surface_exact=True,
        surface_restored=True,
        frozen_has_no_gradients=True,
    )
    assert passed["passed"] is True
    assert passed["passing_this_screen_authorizes_training"] is False
    conflicted = dict(aligned)
    conflicted["broad_retention_aggregate"] = -torch.ones(22_528)
    conflicted["proposed_training_aggregate"] = torch.ones(22_528) * 0.5
    failed = v39.evaluate_pass_contract(
        priority_rows=_priority_rows(),
        aggregates=conflicted,
        state_exact=True,
        model_versions_exact=True,
        surface_exact=True,
        surface_restored=True,
        frozen_has_no_gradients=True,
    )
    assert failed["passed"] is False
    directional = failed["directional_compatibility"]
    assert directional[
        "proposed_training_aggregate__broad_retention_aggregate"
    ]["passed"] is False


def test_v39_pass_contract_rejects_zero_scene_discriminative_gradient() -> None:
    aggregates = {
        "priority_aggregate": torch.ones(22_528),
        "book_support_aggregate": torch.ones(22_528),
        "picture_support_aggregate": torch.ones(22_528),
        "book_scene_discriminative_aggregate": torch.zeros(22_528),
        "picture_scene_discriminative_aggregate": torch.zeros(22_528),
        "broad_retention_aggregate": torch.ones(22_528),
        "scene_discriminative_aggregate": torch.zeros(22_528),
        "cross_prefix_maintenance_aggregate": torch.zeros(22_528),
        "proposed_training_aggregate": torch.ones(22_528) * 2,
    }
    result = v39.evaluate_pass_contract(
        priority_rows=_priority_rows(),
        aggregates=aggregates,
        state_exact=True,
        model_versions_exact=True,
        surface_exact=True,
        surface_restored=True,
        frozen_has_no_gradients=True,
    )
    assert result["passed"] is False
    assert result["checks"][
        "scene_discriminative_aggregate_gradient_target_tensors_nonzero"
    ] is False
    assert result["directional_compatibility"][
        "proposed_training_aggregate__scene_discriminative_aggregate"
    ]["passed"] is False


def test_v39_pass_contract_rejects_lexically_masked_family_scene_conflict() -> None:
    ones = torch.ones(22_528)
    aggregates = {
        "priority_aggregate": ones,
        "book_support_aggregate": ones,
        "picture_support_aggregate": ones,
        "book_scene_discriminative_aggregate": -ones,
        "picture_scene_discriminative_aggregate": ones,
        "broad_retention_aggregate": ones,
        "scene_discriminative_aggregate": ones,
        "cross_prefix_maintenance_aggregate": ones,
        "proposed_training_aggregate": ones * 2,
    }
    result = v39.evaluate_pass_contract(
        priority_rows=_priority_rows(),
        aggregates=aggregates,
        state_exact=True,
        model_versions_exact=True,
        surface_exact=True,
        surface_restored=True,
        frozen_has_no_gradients=True,
    )
    assert result["passed"] is False
    assert result["directional_compatibility"][
        "proposed_training_aggregate__book_support_aggregate"
    ]["passed"] is True
    assert result["directional_compatibility"][
        "proposed_training_aggregate__book_scene_discriminative_aggregate"
    ]["passed"] is False


class _FakeAdapter(nn.Module):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.lora_a = nn.Parameter(torch.ones(4, 1536))
        self.lora_b = nn.Parameter(torch.ones(out_features, 4))


class _FakeBank:
    def __init__(self) -> None:
        self.adapters = (_FakeAdapter(2048), _FakeAdapter(4096))

    def eval(self) -> _FakeBank:
        for adapter in self.adapters:
            adapter.eval()
        return self


def test_v39_freeze_surface_enables_only_layer14_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    bank = _FakeBank()
    model = nn.Module()
    model.layer13 = bank.adapters[0]
    model.layer14 = bank.adapters[1]
    model.unrelated = nn.Linear(2, 2)
    bundle = SimpleNamespace(
        language=SimpleNamespace(model=model), checkpoint_modules={"model": model}
    )
    monkeypatch.setattr(v39, "_v28_bank", lambda _bundle: bank)
    target = v39.freeze_for_v39(bundle)
    surface = v39.assert_v39_surface(bundle, target)
    assert surface["trainable_tensor_count"] == 2
    assert surface["trainable_parameter_count"] == 22_528
    assert bank.adapters[0].lora_a.requires_grad is False
    assert bank.adapters[0].lora_b.requires_grad is False
    assert bank.adapters[1].lora_a.requires_grad is True
    assert bank.adapters[1].lora_b.requires_grad is True
    assert model.unrelated.weight.requires_grad is False


def test_v39_live_module_has_no_optimizer_step_or_backward_call() -> None:
    path = Path(v39.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr in {"backward", "step"}
        for node in calls
    )
    assert not any(
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "optim"
        for node in calls
    )


def test_v39_expected_gemma_architecture_is_causally_pinned(preflight: dict) -> None:
    architecture = preflight["expected_gemma_architecture"]
    assert architecture == {
        "language_layer_count": 35,
        "num_kv_shared_layers": 20,
        "first_shared_kv_layer": 15,
        "layer_13_attention_type": "sliding_attention",
        "layer_14_attention_type": "full_attention",
        "layer_13_role": "last_nonshared_sliding_kv_producer",
        "layer_14_role": "last_nonshared_full_kv_producer",
        "layers_15_through_34_reuse_shared_kv_states": True,
    }


def test_v39_preflight_json_contains_no_question_or_answer_text(preflight: dict) -> None:
    serialized = json.dumps(preflight, sort_keys=True)
    assert '"question"' not in serialized
    assert '"answer"' not in serialized

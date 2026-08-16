from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation import v75_fixed_atlas_behavior as v75_behavior
from semantic_3d_chat.evaluation import (
    v80_atlas_attention_reader_preregistration as prereg,
)
from semantic_3d_chat.evaluation import v80_cpu_preflight_correction as correction
from semantic_3d_chat.language.v80_atlas_attention_reader import (
    PARAMETER_COUNT,
    TARGET_MODULES,
    OuterAdditiveFP32LoRA,
    causal_prefix_visibility,
    install_v80,
)
from semantic_3d_chat.training import (
    train_v80_v75_atlas_attention_reader as training,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    changed_units_v73,
    load_training_rows_v73,
    split_rows_v73,
)

PREREGISTRATION = Path(
    "reports/gemma4/metrics/"
    "gemma4_v80_v75_atlas_attention_reader_preregistration.json"
)
CPU_PREFLIGHT = Path(
    "reports/gemma4/metrics/"
    "gemma4_v80_v75_atlas_attention_reader_cpu_preflight.json"
)


def test_v80_config_and_create_once_seals_are_exact() -> None:
    config = prereg.load_v80_config()
    assert prereg.sha256_file(prereg.CONFIG) == prereg.EXPECTED_CONFIG_SHA256
    assert prereg.sha256_file(PREREGISTRATION) == training.PREREGISTRATION_SHA256
    assert prereg.sha256_file(CPU_PREFLIGHT) == training.CPU_PREFLIGHT_SHA256
    assert config["atlas_contract"]["fixed_prefix_tokens"] == 738
    assert config["reader"]["trainable_parameter_count"] == PARAMETER_COUNT
    assert tuple(config["reader"]["target_modules"]) == TARGET_MODULES

    authenticated = training._authenticate_live_inputs(config)
    assert authenticated["live:gemma_model_blob_sha256_identity"] == config["inputs"][
        "model_file_sha256"
    ]
    assert len(authenticated) == 7


def test_v80_sealed_cpu_preflight_did_not_touch_the_real_model() -> None:
    sealed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    cpu = json.loads(CPU_PREFLIGHT.read_text(encoding="utf-8"))

    assert sealed["training_executed"] is False
    assert sealed["optimizer_constructed"] is False
    assert sealed["checkpoint_published"] is False
    assert cpu["passed"] is True
    assert cpu["real_model"] == {
        "gradient_smoke_run": False,
        "loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
    }
    # The historical builder wrote this ambiguous literal True.  A separate
    # immutable correction below marks the factual interpretation false; the
    # explicit real_model record above is the authoritative zero-update value.
    assert cpu["checks"]["optimizer_updates_on_real_model"] is True


def test_v80_create_once_correction_authenticates_and_disambiguates_cpu_fact() -> None:
    payload = correction.validate_correction(
        expected_sha256=training.CPU_PREFLIGHT_CORRECTION_SHA256
    )
    assert correction.sha256_file(correction.CORRECTION) == (
        "0f0d0183d4e6deed942465116305f5698d09717c5f233ef351445c828729c2cb"
    )
    assert correction.sha256_file(correction.SUPERSEDED_CORRECTION) == (
        correction.SUPERSEDED_CORRECTION_SHA256
    )
    assert payload["correction"] == {
        "field": "checks.optimizer_updates_on_real_model",
        "original_value": True,
        "corrected_value": False,
        "classification": "misnamed_unconditional_pass_boolean_not_an_update_fact",
        "authoritative_field": "real_model.optimizer_updates",
        "authoritative_value": 0,
        "original_artifact_overwritten": False,
    }
    assert payload["authoritative_real_model_state"]["optimizer_updates"] == 0
    assert len(payload["prefix_authentication"]["entries"]) == 40
    assert payload["checks"]["v75_held_subset_loader_compatibility_preserved"] is True
    assert len(payload["source_dependency_sha256"]) == len(
        correction.SOURCE_DEPENDENCIES
    )
    assert payload["runtime_authentication"]["v54"]["metadata_sha256"] == (
        correction.V54_METADATA_SHA256
    )
    assert payload["runtime_authentication"]["v75"]["metadata_sha256"] == (
        correction.V75_METADATA_SHA256
    )
    assert payload["probe_authentication"]["metadata_sha256"] == (
        correction.PROBE_METADATA_SHA256
    )


def test_v80_refuses_even_byte_identical_noncanonical_config(tmp_path: Path) -> None:
    alternate = tmp_path / "v80.yaml"
    alternate.write_bytes(Path(prereg.CONFIG).read_bytes())
    with pytest.raises(ValueError, match="noncanonical"):
        prereg.load_v80_config(alternate)


def test_v80_prefix_hardening_preserves_v75_16_scene_subset_loader() -> None:
    root = Path("data_gemma4/scene_tokens/v56_question_control_full_prefixes")
    prefixes, manifest = v75_behavior._load_base_prefixes(
        root,
        v75_behavior.SCENE_IDS,
        FileAccessAudit(),
    )
    assert set(prefixes) == set(v75_behavior.SCENE_IDS)
    assert len(prefixes) == 16
    assert manifest["scene_count"] == 40
    for scene_id, prefix in prefixes.items():
        entry = manifest["scenes"][scene_id]
        assert prefix.dtype == torch.bfloat16
        assert tuple(prefix.shape) == (1, 258, 1536)
        assert torch.equal(
            prefix,
            load_file(str(root / entry["filename"]), device="cpu")["scene_prefix"],
        )


def test_v80_contract_is_one_bounded_nonpublishing_arm() -> None:
    config = prereg.load_v80_config()
    atlas = config["atlas_contract"]
    forward = config["forward_contract"]
    optimization = config["optimization"]
    memory = config["memory_safety"]
    publication = config["publication"]

    assert atlas["layout"] == [
        "boi",
        "all_480_atlas_key_value_tokens",
        "all_256_base_scene_latents",
        "eoi",
    ]
    assert atlas["compile_every_scene_before_any_question_preparation"] is True
    assert atlas["question_dependent_scene_processing"] is False
    assert atlas["question_dependent_retrieval"] is False
    assert atlas["semantic_or_spatial_top_k_selection"] is False
    assert atlas["every_prefix_token_retained"] is True
    assert forward["full_huggingface_model_forward_for_every_branch"] is True
    assert forward["prepared_hidden_state_shortcut"] is False
    assert forward["answer_tail_only_lm_head_via_native_logits_to_keep"] is True
    assert optimization["updates"] == 16
    assert optimization["changed_units_per_update"] == [3] * 8 + [2] * 8
    assert sum(optimization["changed_units_per_update"]) == 40
    assert memory["sequential_microbranches_only"] is True
    assert memory["maximum_live_teacher_forced_batch_size"] == 1
    assert memory["duplicate_model_instances"] is False
    assert memory["checkpoint_writer_present"] is False
    assert publication["gradient_smoke_sufficient_for_training"] is True
    assert all(
        publication[key] is False
        for key in (
            "held_smoke_sufficient_for_runtime_promotion",
            "checkpoint_publication_authorized",
            "runtime_publication_authorized",
            "official_validation_authorized",
            "official_test_authorized",
            "deferred_final_authorized",
            "oracle_authorized",
        )
    )


def test_v80_v73_train_and_pair_scene_disjoint_held_inventories_are_exact() -> None:
    config = prereg.load_v80_config()
    rows = load_training_rows_v73(config["inputs"]["historical_training_qa"])
    train, held = split_rows_v73(rows)
    schedule = prereg.build_schedule_v80(changed_units_v73(train))
    held_smoke = prereg.select_held_smoke_v80(held)
    broad_train = prereg.select_broad_train_v80(train)
    broad_held = prereg.select_broad_held_v80(held)

    flattened = [unit for update in schedule for unit in update]
    keys = [(unit.pair_id, unit.question_key) for unit in flattened]
    assert len(schedule) == 16
    assert [len(update) for update in schedule] == [3] * 8 + [2] * 8
    assert len(keys) == len(set(keys)) == 40
    assert len(held_smoke) == 8
    assert len({unit.change_type for unit in held_smoke}) == 8
    assert len({row.scene_id for unit in held_smoke for row in (unit.left, unit.right)}) == 16
    assert len(broad_train) == 16
    assert len(broad_held) == 16
    assert {row.pair_id for row in train}.isdisjoint({row.pair_id for row in held})
    assert {row.scene_id for row in train}.isdisjoint({row.scene_id for row in held})


def test_v80_every_atlas_token_is_visible_without_selection() -> None:
    proof = causal_prefix_visibility(
        prefix_tokens=738,
        prompt_tokens=23,
        answer_tokens=4,
    )
    assert proof["target_layers"] == [14, 34]
    assert proof["attention_type"] == "full_attention"
    assert proof["visible_prefix_token_count_per_text_query"] == 738
    assert proof["all_prefix_tokens_visible"] is True
    assert proof["selection_or_top_k"] is False


def test_v80_zero_output_outer_lora_has_only_output_factor_gradient() -> None:
    torch.manual_seed(80)
    base = nn.Linear(5, 3, bias=False)
    wrapped = OuterAdditiveFP32LoRA(base, rank=2, alpha=4.0)
    with torch.no_grad():
        wrapped.residual_a.normal_()
    inputs = torch.randn(7, 5)
    expected = base(inputs).detach()
    observed = wrapped(inputs)
    assert torch.equal(observed, expected)

    F.mse_loss(observed, torch.randn_like(observed)).backward()
    assert wrapped.residual_a.grad is not None
    assert float(wrapped.residual_a.grad.norm()) == 0.0
    assert wrapped.residual_b.grad is not None
    assert math.isfinite(float(wrapped.residual_b.grad.norm()))
    assert float(wrapped.residual_b.grad.norm()) > 0.0


def test_v80_shape_faithful_install_targets_exactly_four_modules() -> None:
    model = prereg._SyntheticV80Model()
    installation = install_v80(model)
    assert installation.target_names == TARGET_MODULES
    assert installation.parameter_count == 122_880
    installation.assert_only_adapters_trainable(model)
    assert len(installation.parameters()) == 8
    assert all(parameter.dtype == torch.float32 for parameter in installation.parameters())
    installation.assert_fp32_finite()
    with torch.no_grad():
        installation.adapters[0].residual_a[0, 0] = torch.inf
    with pytest.raises(RuntimeError, match="finite FP32"):
        installation.assert_fp32_finite()


@pytest.mark.parametrize(
    ("correct", "wrong", "target", "correct_weight", "pair_weight", "scale"),
    [
        (1.2, 1.7, 0.5, 0.2, 1.0, 0.125),
        (11.0, 0.2, 0.5, 0.2, 1.0, 0.25),
        (0.1, 20.0, 0.5, 0.2, 1.0, 1.0),
    ],
)
def test_v80_sequential_causal_gradient_coefficients_equal_joint_autograd(
    correct: float,
    wrong: float,
    target: float,
    correct_weight: float,
    pair_weight: float,
    scale: float,
) -> None:
    correct_tensor = torch.tensor(correct, dtype=torch.float64, requires_grad=True)
    wrong_tensor = torch.tensor(wrong, dtype=torch.float64, requires_grad=True)
    loss = scale * (
        correct_weight * correct_tensor
        + pair_weight * F.softplus(target - wrong_tensor + correct_tensor)
    )
    expected = torch.autograd.grad(loss, (correct_tensor, wrong_tensor))
    observed = training.softplus_nll_gradient_coefficients(
        correct_nll=correct,
        wrong_nll=wrong,
        target=target,
        correct_weight=correct_weight,
        pair_weight=pair_weight,
        scale=scale,
    )
    assert observed == pytest.approx(tuple(float(value) for value in expected), abs=1e-15)


def test_v80_real_smoke_has_no_optimizer_and_screen_authenticates_before_one() -> None:
    smoke_source = inspect.getsource(training.run_gradient_smoke)
    screen_source = inspect.getsource(training.run_bounded_screen)
    module_source = inspect.getsource(training)

    assert "torch.optim" not in smoke_source
    assert '"optimizer_constructed": False' in smoke_source
    assert '"optimizer_updates": 0' in smoke_source
    assert screen_source.index("smoke.get(\"passed\")") < screen_source.index(
        "torch.optim.AdamW"
    )
    assert "for update, units in enumerate(bundle.schedule, 1)" in screen_source
    assert "save_file" not in module_source
    assert "save_pretrained" not in module_source
    assert '"runtime_promotion_authorized": False' in module_source


def test_v80_native_answer_tail_path_is_full_forward_and_not_hidden_shortcut() -> None:
    tail_source = inspect.getsource(training.answer_tail_forward)
    kwargs_source = inspect.getsource(training.answer_tail_forward.__globals__["answer_tail_model_kwargs"])
    trainer_source = inspect.getsource(training._tail)
    assert "language.model(**kwargs)" in tail_source
    assert '"logits_to_keep": causal_positions' in kwargs_source
    assert '"use_cache": False' in kwargs_source
    assert '"labels": None' in kwargs_source
    assert "prepared_hidden" not in trainer_source

from __future__ import annotations

from pathlib import Path

from semantic_3d_chat.evaluation import fixed_prefix_attention_reader_v6_3_evidence as v63e
from semantic_3d_chat.training import train_fixed_prefix_attention_reader_v6_3 as v63
from semantic_3d_chat.training import train_fixed_prefix_attention_reader_v6_4 as v64


def test_v6_4_parent_v6_3_evidence_authenticates() -> None:
    result = v63e.authenticate_terminal_marker()
    assert result["passed"] is True
    assert result["continuation"] == "v6_4_pair_disjoint_train_only_confirmation"
    assert result["runtime_checkpoint_promotion_authorized"] is False


def test_v6_4_split_is_hard_physical_pair_and_scene_disjoint() -> None:
    units = v63.build_pair_units(v63.v1.load_training_records())
    train, held = v64.split_pair_units(units)
    train_scenes = {
        scene for unit in train for scene in (unit.first.scene_id, unit.second.scene_id)
    }
    held_scenes = {
        scene for unit in held for scene in (unit.first.scene_id, unit.second.scene_id)
    }
    assert len(train) == 28
    assert len(held) == 12
    assert len(train_scenes) == 18
    assert held_scenes == set(v64.HELD_SCENE_IDS)
    assert train_scenes.isdisjoint(held_scenes)
    assert {unit.pair_id for unit in held} == set(v64.HELD_PAIR_IDS)
    assert {unit.key for unit in train}.isdisjoint(unit.key for unit in held)


def test_v6_4_schedule_is_three_exact_train_epochs_and_never_held() -> None:
    units = v63.build_pair_units(v63.v1.load_training_records())
    train, _held = v64.split_pair_units(units)
    schedule = v64.build_schedule(train)
    diagnostics = v64.schedule_diagnostics(schedule)
    assert len(schedule) == 12
    assert {len(update) for update in schedule} == {7}
    assert diagnostics["total_unit_exposures"] == 84
    assert diagnostics["unique_training_units"] == 28
    assert diagnostics["exposures_per_unit_distribution"] == {3: 28}
    assert diagnostics["held_pair_ids_in_schedule"] == []


def test_v6_4_config_has_exact_surface_gates_and_hard_stop() -> None:
    import yaml

    config = yaml.safe_load(v64._resolve(v64.CONFIG).read_text(encoding="utf-8"))
    assert config["attention_reader"]["exact_targets"] == list(v63.TARGET_MODULES)
    assert config["attention_reader"]["trainable_parameter_count"] == 30_720
    assert config["optimization"]["updates"] == 12
    assert config["optimization"]["hard_runtime_seconds"] == 480
    assert config["optimization"]["checkpoint_publication"] is False
    assert config["gates"]["promotion_authorized_if_all_pass"] is False


def test_v6_4_source_has_no_validation_loader_or_checkpoint_writer() -> None:
    source = Path(v64.__file__).read_text(encoding="utf-8")
    assert "load_validation_records(" not in source
    assert "save_file(" not in source
    assert "torch.save(" not in source
    assert "OUTPUT_CHECKPOINT" not in source
    assert "internal_validation_inputs_loaded" in source
    assert "oracle_inputs_loaded" in source


def test_v6_4_forbidden_roots_inherit_validation_deferred_and_final_blocks() -> None:
    roots = set(v63.training_forbidden_roots())
    prefix_root = v63._resolve(v63.PREFIX_CACHE)
    assert all(
        (prefix_root / f"{scene_id}.safetensors").resolve() in roots
        for scene_id in v63.VALIDATION_SCENES
    )
    assert v63._resolve(v63.v1.VALIDATION_QUESTIONS) in roots
    assert v63._resolve(v63.v1.VALIDATION_REFERENCES) in roots

"""Outcome-independent V95 deferred-final materialization preregistration.

This module seals one new six-scene recipe before any deferred artifact or
answer-bearing file exists.  Recipe validation reads only the new recipe, the
public default configuration, and semantics-free implementation/model
contracts.  It never consults the legacy diverse20/diverse28/diverse52 scene
plans.

Execution is deliberately separate.  ``run-stage`` first authenticates both
this preregistration and the post-known-development V95 unlock, then runs one
fixed stage.  Importing, validating, preregistering, or authenticating this
module never starts Blender, loads Gemma weights, generates a scene, or opens
QA/oracle data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import yaml

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.data.scene_variants import (
    ScenePlan,
    batch_scene_plans,
    batch_scene_splits,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_deferred_final_qa import (
    ANSWER_TYPE_TOTALS,
    CHANGED_UNITS_PER_PAIR,
    FINAL_QA,
    PAIR_SCENES,
    PAIR_UNIT_QUOTAS,
    ROWS_PER_SCENE,
    SCENE_IDS,
    SELECTION_MANIFEST,
)

SCHEMA_VERSION: Final[int] = 95
ARTIFACT: Final[str] = "gemma4_v95_deferred_final_materialization_preregistration_v1"
STATUS: Final[str] = "sealed_before_known_development_labels_or_deferred_materialization"
RECIPE: Final[Path] = (
    PROJECT_ROOT
    / "configs/experiments/gemma4_v95_deferred_final_materialization.yaml"
)
DEFAULT_CONFIG: Final[Path] = PROJECT_ROOT / "configs/default.yaml"
PREREGISTRATION: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/"
    "gemma4_v95_deferred_final_materialization_preregistration.json"
)
WORK_ROOT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v95_deferred_final"
)
QUESTION_MANIFEST: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/questions/gemma4_v95_deferred_final.json"
)
MEMORY_ROOT: Final[Path] = WORK_ROOT / "memory_cache"
RECEIPT_ROOT: Final[Path] = WORK_ROOT / "receipts"
V95_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/gemma4_v95_strict_causal_successor.yaml"
)
UNLOCK_PATH: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/"
    "gemma4_v95_strict_causal_successor_deferred_final_unlock.json"
)

_EXPECTED_PLANS: Final[dict[str, dict[str, Any]]] = {
    "scene_000025": {
        "seed": 20285024,
        "color_variant": "base",
        "layout_variant": "base",
        "pair_id": "pair_000013",
        "paired_scene_id": "scene_000026",
        "change_type": "color_swap",
        "pair_role": "reference",
    },
    "scene_000026": {
        "seed": 20285024,
        "color_variant": "swap_red_blue",
        "layout_variant": "base",
        "pair_id": "pair_000013",
        "paired_scene_id": "scene_000025",
        "change_type": "color_swap",
        "pair_role": "counterfactual",
    },
    "scene_000027": {
        "seed": 20287042,
        "color_variant": "base",
        "layout_variant": "cube_on",
        "pair_id": "pair_000014",
        "paired_scene_id": "scene_000028",
        "change_type": "cube_support",
        "pair_role": "reference",
    },
    "scene_000028": {
        "seed": 20287042,
        "color_variant": "base",
        "layout_variant": "cube_under",
        "pair_id": "pair_000014",
        "paired_scene_id": "scene_000027",
        "change_type": "cube_support",
        "pair_role": "counterfactual",
    },
    "scene_000029": {
        "seed": 20289060,
        "color_variant": "base",
        "layout_variant": "base",
        "pair_id": "pair_000015",
        "paired_scene_id": "scene_000030",
        "change_type": "mirror_lr",
        "pair_role": "reference",
    },
    "scene_000030": {
        "seed": 20289060,
        "color_variant": "base",
        "layout_variant": "mirror_lr",
        "pair_id": "pair_000015",
        "paired_scene_id": "scene_000029",
        "change_type": "mirror_lr",
        "pair_role": "counterfactual",
    },
}

_SOURCE_FILES: Final[tuple[str, ...]] = (
    "configs/default.yaml",
    "configs/experiments/gemma4_v95_deferred_final_materialization.yaml",
    "scripts/generate_scene_batch.py",
    "scripts/build_map.py",
    "scripts/prepare_v81_scene_memory.py",
    "blender/generate_scene.py",
    "blender/render_scan.py",
    "blender/scene_utils.py",
    "src/semantic_3d_chat/config.py",
    "src/semantic_3d_chat/scan_plan.py",
    "src/semantic_3d_chat/rendering_io.py",
    "src/semantic_3d_chat/data/scene_variants.py",
    "src/semantic_3d_chat/data/qa_generator.py",
    "src/semantic_3d_chat/vision/model_registry.py",
    "src/semantic_3d_chat/vision/gemma4_encoder.py",
    "src/semantic_3d_chat/vision/encoder.py",
    "src/semantic_3d_chat/vision/batch_encoder.py",
    "src/semantic_3d_chat/vision/patch_features.py",
    "src/semantic_3d_chat/mapping/depth_projection.py",
    "src/semantic_3d_chat/mapping/fusion.py",
    "src/semantic_3d_chat/mapping/voxel_map.py",
    "src/semantic_3d_chat/evaluation/prepare_questions.py",
    "src/semantic_3d_chat/evaluation/v95_deferred_final_qa.py",
    "src/semantic_3d_chat/evaluation/v95_deferred_final_materialization.py",
)

_NUMERIC_COMPILER_FILES: Final[tuple[str, ...]] = (
    "configs/runtime/gemma4_v85_strict_multiscene.yaml",
    "reports/gemma4/artifacts/v85_strict_runtime_candidate/adapter.safetensors",
    "reports/gemma4/artifacts/v85_strict_runtime_candidate/runtime_metadata.json",
    "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1/control.safetensors",
    "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1/runtime_metadata.json",
    "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/probe_bank/probes.safetensors",
    "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/probe_bank/runtime_metadata.json",
)

_STAGE_ORDER: Final[tuple[str, ...]] = (
    "generate",
    "render",
    "features",
    "maps",
    "memory",
    "qa_raw",
    "qa_select",
    "questions",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _strict_destination(value: str | Path | None, expected: Path, label: str) -> Path:
    result = expected.resolve() if value is None else Path(value).expanduser().resolve()
    if result != expected.resolve():
        raise ValueError(f"V95 {label} has one fixed path")
    return result


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V95 JSON must contain one object: {path}")
    return value


def _plan_identity(plan: ScenePlan) -> dict[str, Any]:
    result = {
        "scene_id": plan.scene_id,
        "seed": plan.seed,
        "plan_version": plan.plan_version,
        "color_variant": plan.color_variant,
        "layout_variant": plan.layout_variant,
        "remove_instance_ids": list(plan.remove_instance_ids),
        "chair_count": plan.chair_count,
        "chair_orientation": plan.chair_orientation,
        "picture_placement": plan.picture_placement,
        "bowl_placement": plan.bowl_placement,
        "book_placement": plan.book_placement,
        "pair_id": plan.pair_id,
        "paired_scene_id": plan.paired_scene_id,
        "change_type": plan.change_type,
        "pair_role": plan.pair_role,
    }
    return result


def validate_recipe_v95(path: str | Path = RECIPE) -> dict[str, Any]:
    """Validate only public deterministic controls; never inspect artifacts."""

    source = Path(path).expanduser().resolve()
    if source != RECIPE.resolve() or source.is_symlink() or not source.is_file():
        raise ValueError("V95 deferred final has one regular fixed recipe")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("_base_") != "../default.yaml":
        raise ValueError("V95 recipe must inherit only the public default config")
    config = load_config(source)
    default = load_config(DEFAULT_CONFIG)
    if config.get("seed") != default.get("seed") or config.get("render") != default.get(
        "render"
    ):
        raise ValueError("V95 recipe changed the public default seed or scan geometry")
    plans = batch_scene_plans(config)
    splits = batch_scene_splits(config, plans)
    if (
        tuple(plan.scene_id for plan in plans) != SCENE_IDS
        or splits
        != {"train": [], "validation": [], "test": list(SCENE_IDS)}
        or config["batch"].get("deferred_splits") != ["test"]
        or config["batch"].get("base_config") != "configs/default.yaml"
        or config["batch"].get("require_visibility_evidence") is not True
        or config["batch"].get("expected_scene_count") != 6
    ):
        raise ValueError("V95 six-scene split/lock contract changed")
    identities = [_plan_identity(plan) for plan in plans]
    for identity in identities:
        expected = _EXPECTED_PLANS[identity["scene_id"]]
        if (
            identity["plan_version"] != 2
            or identity["remove_instance_ids"] != []
            or identity["chair_count"] != 1
            or identity["chair_orientation"] != "upright"
            or identity["picture_placement"] != "wall"
            or identity["bowl_placement"] != "floor_left"
            or identity["book_placement"] != "table"
            or any(identity.get(field) != value for field, value in expected.items())
        ):
            raise ValueError(f"V95 scene recipe changed: {identity['scene_id']}")
    qa = config.get("qa", {}).get("v95_exact_final_selection")
    expected_qa = {
        "schema_version": 1,
        "seed": 950095,
        "rows_per_scene": ROWS_PER_SCENE,
        "changed_units_per_pair": CHANGED_UNITS_PER_PAIR,
        "pair_unit_quotas": PAIR_UNIT_QUOTAS,
        "answer_type_totals": ANSWER_TYPE_TOTALS,
    }
    if qa != expected_qa or config["qa"]["balanced_selection"].get("enabled") is not False:
        raise ValueError("V95 exact QA recipe changed")
    vision = config.get("vision")
    expected_vision = {
        "backend": "gemma4",
        "model_id": "google/gemma-4-E2B-it",
        "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "input_size": 224,
        "middle_layer": 8,
        "late_layer": 16,
        "feature_mode": "middle_late_projected",
        "aligned_method": "pooled_native_projector_broadcast",
        "dtype": "bfloat16",
        "storage_dtype": "float16",
        "batch_size": 1,
    }
    if vision != expected_vision:
        raise ValueError("V95 semantics-free dense model contract changed")
    return {
        "recipe_path": _relative(source),
        "recipe_sha256": sha256_file_v85(source),
        "default_config_sha256": sha256_file_v85(DEFAULT_CONFIG),
        "default_seed": int(default["seed"]),
        "default_render_contract_sha256": _canonical_sha256(default["render"]),
        "scene_plans": identities,
        "scene_plan_inventory_sha256": _canonical_sha256(identities),
        "pair_scenes": {key: list(value) for key, value in PAIR_SCENES.items()},
        "qa_contract": expected_qa,
        "vision_contract": expected_vision,
        "legacy_plan_files_opened": [],
    }


def _resolve_executable(candidates: Sequence[Path | str], label: str) -> Path:
    for candidate in candidates:
        value = Path(candidate).expanduser()
        resolved = shutil.which(str(value)) if value.name == str(value) else None
        path = Path(resolved) if resolved else value
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        # Preserve virtual-environment launcher symlinks. Invoking the resolved
        # base interpreter would silently drop that environment's packages.
        path = Path(os.path.abspath(path))
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(f"V95 required {label} executable is unavailable")


def _python_identity(executable: Path, distributions: Sequence[str]) -> dict[str, Any]:
    script = (
        "import importlib.metadata,json,platform,sys;"
        "names=" + repr(list(distributions)) + ";"
        "print(json.dumps({'implementation':platform.python_implementation(),"
        "'version':platform.python_version(),'packages':{name:"
        "importlib.metadata.version(name) for name in names}},sort_keys=True))"
    )
    completed = subprocess.run(
        [str(executable), "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    value = json.loads(completed.stdout)
    return {
        "path": str(executable),
        "resolved_path": str(executable.resolve()),
        "executable_sha256": sha256_file_v85(executable),
        **value,
    }


def _toolchain_identity() -> dict[str, Any]:
    support = _resolve_executable((PROJECT_ROOT / ".venv/bin/python", sys.executable), "support Python")
    gemma = _resolve_executable((PROJECT_ROOT / ".venv-gemma4/bin/python",), "Gemma Python")
    blender = _resolve_executable(
        (
            "blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ),
        "Blender",
    )
    blender_version = subprocess.run(
        [str(blender), "--version"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    if not blender_version:
        raise ValueError("Blender returned no version identity")
    return {
        "support_python": _python_identity(support, ("PyYAML", "numpy")),
        "gemma_python": _python_identity(
            gemma, ("torch", "transformers", "safetensors", "numpy", "Pillow")
        ),
        "blender": {
            "path": str(blender),
            "resolved_path": str(blender.resolve()),
            "executable_sha256": sha256_file_v85(blender),
            "version_output": blender_version,
        },
    }


def _file_identities(relative_paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = (PROJECT_ROOT / relative).resolve()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"V95 fixed source is unavailable: {path}")
        result[relative] = sha256_file_v85(path)
    return result


def _frame_outputs(scene_id: str, suffix: str, directory: str) -> list[str]:
    return [
        f"data/rendered/{scene_id}/{directory}/f_{index:06d}.{suffix}"
        for index in range(24)
    ]


def _stage_outputs() -> dict[str, list[str]]:
    generated: list[str] = []
    rendered: list[str] = []
    features: list[str] = []
    maps: list[str] = []
    memories: list[str] = []
    for scene_id in SCENE_IDS:
        generated.extend(
            (
                f"data/oracle/{scene_id}/oracle.json",
                f"data/oracle/{scene_id}/scene.blend",
                f"data/rendered/{scene_id}/p_000000.png",
            )
        )
        rendered.extend(
            (
                f"data/oracle/{scene_id}/visibility.json",
                f"data/rendered/{scene_id}/manifest.json",
                *_frame_outputs(scene_id, "png", "rgb"),
                *_frame_outputs(scene_id, "npy", "depth"),
            )
        )
        features.extend(
            (
                f"data_gemma4/features/{scene_id}/manifest.json",
                *(
                    f"data_gemma4/features/{scene_id}/f_{index:06d}.npz"
                    for index in range(24)
                ),
            )
        )
        maps.extend(
            (
                f"data_gemma4/maps/{scene_id}/voxel_map.npz",
                f"reports/gemma4/metrics/map_{scene_id}.json",
            )
        )
        memories.extend(
            (
                (
                    f"reports/gemma4/artifacts/v95_deferred_final/memory_cache/"
                    f"{scene_id}/memory.safetensors"
                ),
                (
                    f"reports/gemma4/artifacts/v95_deferred_final/memory_cache/"
                    f"{scene_id}/runtime_metadata.json"
                ),
                f"reports/gemma4/metrics/v95_deferred_final_memory_access_{scene_id}.json",
            )
        )
    return {
        "generate": generated,
        "render": rendered,
        "features": features,
        "maps": maps,
        "qa_raw": [
            "reports/gemma4/artifacts/v95_deferred_final/qa_raw/train.jsonl",
            "reports/gemma4/artifacts/v95_deferred_final/qa_raw/validation.jsonl",
            "reports/gemma4/artifacts/v95_deferred_final/qa_raw/test.jsonl",
            "reports/gemma4/artifacts/v95_deferred_final/qa_raw/splits.json",
        ],
        "qa_select": [_relative(FINAL_QA), _relative(SELECTION_MANIFEST)],
        "memory": memories,
        "questions": [_relative(QUESTION_MANIFEST)],
    }


def _commands(toolchain: Mapping[str, Any]) -> dict[str, list[list[str]]]:
    support = str(toolchain["support_python"]["path"])
    gemma = str(toolchain["gemma_python"]["path"])
    blender = str(toolchain["blender"]["path"])
    recipe = _relative(RECIPE)
    commands: dict[str, list[list[str]]] = {
        "generate": [
            [
                support,
                "scripts/generate_scene_batch.py",
                "--config",
                recipe,
                "--stage",
                "generate",
                "--split",
                "test",
                "--include-deferred-test",
                "--blender",
                blender,
            ]
        ],
        "render": [
            [
                support,
                "scripts/generate_scene_batch.py",
                "--config",
                recipe,
                "--stage",
                "render",
                "--split",
                "test",
                "--include-deferred-test",
                "--blender",
                blender,
            ]
        ],
        "features": [
            [
                gemma,
                "-m",
                "semantic_3d_chat.vision.batch_encoder",
                "--config",
                recipe,
                "--split",
                "test",
                "--include-deferred-test",
                "--offline",
            ]
        ],
        "maps": [
            [gemma, "scripts/build_map.py", "--config", recipe, "--scene", scene_id]
            for scene_id in SCENE_IDS
        ],
        "qa_raw": [
            [
                support,
                "-m",
                "semantic_3d_chat.data.qa_generator",
                "--config",
                recipe,
                "--include-deferred-test",
            ]
        ],
        "qa_select": [
            [
                support,
                "-m",
                "semantic_3d_chat.evaluation.v95_deferred_final_qa",
                "select",
                "--config",
                _relative(V95_CONFIG),
            ]
        ],
        "memory": [],
        "questions": [
            [
                support,
                "-m",
                "semantic_3d_chat.evaluation.prepare_questions",
                "--config",
                recipe,
                "--split",
                "test",
                "--qa",
                _relative(FINAL_QA),
                "--output",
                _relative(QUESTION_MANIFEST),
            ]
        ],
    }
    for scene_id in SCENE_IDS:
        commands["memory"].append(
            [
                gemma,
                "scripts/prepare_v81_scene_memory.py",
                "--config",
                "configs/runtime/gemma4_v85_strict_multiscene.yaml",
                "--scene",
                scene_id,
                "--base-checkpoint",
                "reports/gemma4/artifacts/v85_strict_runtime_candidate",
                "--control-checkpoint",
                "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1",
                "--probe-bank",
                "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/probe_bank",
                "--output",
                f"reports/gemma4/artifacts/v95_deferred_final/memory_cache/{scene_id}",
                "--audit-report",
                f"reports/gemma4/metrics/v95_deferred_final_memory_access_{scene_id}.json",
            ]
        )
    return commands


def _forbidden_roots() -> list[Path]:
    roots = [path.resolve() for path in PROJECT_ROOT.glob("data*/oracle")]
    roots.extend(path.resolve() for path in PROJECT_ROOT.glob("data*/qa"))
    roots.extend(
        (PROJECT_ROOT / f"configs/experiments/diverse{size}.yaml").resolve()
        for size in (20, 28, 52)
    )
    return list(dict.fromkeys(roots))


def build_materialization_preregistration_v95(
    *,
    toolchain: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = FileAccessAudit(
        _forbidden_roots(),
        forbidden_component_names={"oracle", "qa"},
        block_forbidden=True,
    )
    with audit:
        recipe = validate_recipe_v95()
        resolved_tools = dict(toolchain or _toolchain_identity())
        source_hashes = _file_identities(_SOURCE_FILES)
        compiler_hashes = _file_identities(_NUMERIC_COMPILER_FILES)
        outputs = _stage_outputs()
        commands = _commands(resolved_tools)
    audit.assert_clean()
    stage_contracts = {
        stage: {
            "authorized_entrypoint": [
                resolved_tools["support_python"]["path"],
                "-m",
                "semantic_3d_chat.evaluation.v95_deferred_final_materialization",
                "run-stage",
                "--stage",
                stage,
            ],
            "child_argv": commands[stage],
            "expected_outputs": outputs[stage],
            "receipt": _relative(RECEIPT_ROOT / f"{stage}.json"),
        }
        for stage in _STAGE_ORDER
    }
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "recipe": recipe,
        "source_sha256": source_hashes,
        "numeric_compiler_source_sha256": compiler_hashes,
        "toolchain": resolved_tools,
        "stage_order": list(_STAGE_ORDER),
        "stages": stage_contracts,
        "scene_count": len(SCENE_IDS),
        "pair_count": len(PAIR_SCENES),
        "intended_row_count": len(SCENE_IDS) * ROWS_PER_SCENE,
        "intended_changed_unit_count": len(PAIR_SCENES) * CHANGED_UNITS_PER_PAIR,
        "intended_changed_side_count": len(PAIR_SCENES)
        * CHANGED_UNITS_PER_PAIR
        * 2,
        "answer_type_totals": dict(ANSWER_TYPE_TOTALS),
        "generation_requires_authenticated_unlock": True,
        "every_execution_stage_requires_authenticated_unlock": True,
        "legacy_plan_files_opened": [],
        "known_development_labels_opened": False,
        "deferred_labels_opened": False,
        "deferred_oracle_opened": False,
        "deferred_artifacts_generated": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "protected_read_count": len(audit.forbidden_accesses()),
        "loaded_file_inventory_sha256": _canonical_sha256(audit.unique_paths),
        "automatic_runtime_promotion": False,
    }
    payload["preregistration_identity_sha256"] = _canonical_sha256(payload)
    return payload


def preregister_materialization_v95(
    output: str | Path | None = None,
) -> dict[str, Any]:
    destination = _strict_destination(output, PREREGISTRATION, "preregistration")
    if destination.exists() or destination.is_symlink():
        result = authenticate_materialization_preregistration_v95(destination)
        return {**result, "reused_authenticated_preregistration": True}
    payload = build_materialization_preregistration_v95()
    _atomic_create_json(destination, payload)
    result = authenticate_materialization_preregistration_v95(destination)
    return {**result, "reused_authenticated_preregistration": False}


def authenticate_materialization_preregistration_v95(
    output: str | Path | None = None,
) -> dict[str, Any]:
    source = _strict_destination(output, PREREGISTRATION, "preregistration")
    actual = _strict_json(source)
    expected = build_materialization_preregistration_v95()
    if actual != expected:
        raise ValueError("V95 materialization preregistration no longer matches its contract")
    return {
        **actual,
        "preregistration_file_sha256": sha256_file_v85(source),
        "authenticated": True,
    }


def _existing_output_identity(paths: Sequence[str]) -> dict[str, str]:
    identities: dict[str, str] = {}
    missing: list[str] = []
    for relative in paths:
        path = (PROJECT_ROOT / relative).resolve()
        if path.is_symlink() or not path.is_file():
            missing.append(relative)
        else:
            identities[relative] = sha256_file_v85(path)
    if missing:
        raise FileNotFoundError(f"V95 stage outputs are incomplete: {missing}")
    return identities


def _authenticate_predecessor_receipts(
    preregistration: Mapping[str, Any], stage: str
) -> None:
    index = _STAGE_ORDER.index(stage)
    for predecessor in _STAGE_ORDER[:index]:
        receipt_path = RECEIPT_ROOT / f"{predecessor}.json"
        receipt = _strict_json(receipt_path)
        expected_outputs = preregistration["stages"][predecessor]["expected_outputs"]
        current = _existing_output_identity(expected_outputs)
        if (
            receipt.get("artifact") != "gemma4_v95_deferred_final_stage_receipt_v1"
            or receipt.get("stage") != predecessor
            or receipt.get("output_sha256") != current
            or receipt.get("output_inventory_sha256") != _canonical_sha256(current)
            or receipt.get("preregistration_file_sha256")
            != preregistration["preregistration_file_sha256"]
        ):
            raise ValueError(f"V95 predecessor receipt changed: {predecessor}")


def run_materialization_stage_v95(stage: str) -> dict[str, Any]:
    """Run one fixed stage only after both immutable authorizations pass."""

    if stage not in _STAGE_ORDER:
        raise ValueError(f"Unknown V95 materialization stage: {stage}")
    preregistration = authenticate_materialization_preregistration_v95()

    # Delayed to avoid a module cycle: the unlock authenticator itself binds
    # this preregistration.  Crucially, both checks precede every child process.
    from semantic_3d_chat.evaluation.v95_deferred_final import (
        authenticate_deferred_final_unlock_v95,
    )

    unlock = authenticate_deferred_final_unlock_v95(V95_CONFIG, UNLOCK_PATH)
    _authenticate_predecessor_receipts(preregistration, stage)
    receipt_path = RECEIPT_ROOT / f"{stage}.json"
    expected_outputs = preregistration["stages"][stage]["expected_outputs"]
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _strict_json(receipt_path)
        current = _existing_output_identity(expected_outputs)
        if receipt.get("output_sha256") != current:
            raise ValueError(f"V95 existing stage receipt/output changed: {stage}")
        return {**receipt, "reused_authenticated_receipt": True}
    environment = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    for argv in preregistration["stages"][stage]["child_argv"]:
        subprocess.run(argv, cwd=PROJECT_ROOT, env=environment, check=True)
    output_hashes = _existing_output_identity(expected_outputs)
    receipt = {
        "artifact": "gemma4_v95_deferred_final_stage_receipt_v1",
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "status": "completed_after_authenticated_unlock",
        "preregistration_file_sha256": preregistration[
            "preregistration_file_sha256"
        ],
        "unlock_file_sha256": unlock["unlock_file_sha256"],
        "unlock_identity_sha256": unlock["unlock_identity_sha256"],
        "child_argv": preregistration["stages"][stage]["child_argv"],
        "output_sha256": output_hashes,
        "output_inventory_sha256": _canonical_sha256(output_hashes),
        "automatic_runtime_promotion": False,
    }
    _atomic_create_json(receipt_path, receipt)
    return {**receipt, "reused_authenticated_receipt": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preregister")
    subparsers.add_parser("authenticate")
    run = subparsers.add_parser("run-stage")
    run.add_argument("--stage", choices=_STAGE_ORDER, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preregister":
        result = preregister_materialization_v95()
    elif args.command == "authenticate":
        result = authenticate_materialization_preregistration_v95()
    else:
        result = run_materialization_stage_v95(args.stage)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "PREREGISTRATION",
    "RECIPE",
    "STATUS",
    "authenticate_materialization_preregistration_v95",
    "build_materialization_preregistration_v95",
    "main",
    "preregister_materialization_v95",
    "run_materialization_stage_v95",
    "validate_recipe_v95",
]

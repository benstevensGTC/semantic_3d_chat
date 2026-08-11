"""Fail-closed controller for the V25 all-voxel dense-alignment screen.

V25 starts from the immutable V24 epoch-1 checkpoint, freezes the complete
scene/decoder stack, and adds one zero-output rank-eight residual to every
3,072D voxel feature.  A bounded training-only semantic calibration stage may
use oracle boxes and Gemma token rows, but those payloads are discarded before
the question-independent runtime checkpoint is written.  Paired QA training
is authorized only when calibration reaches its preregistered accuracy,
margin, and residual-size gates.

This controller never loads Gemma weights, scene maps, questions, or oracle
metadata.  Preflight is structural and report-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.language.lora import lora_banks_settings
from semantic_3d_chat.scene_encoder.dense_alignment import (
    DENSE_ALIGNMENT_ARCHITECTURE_VERSION,
    DenseAlignmentResidual,
    construct_dense_alignment,
    dense_alignment_settings,
    validate_dense_alignment_state,
)
from semantic_3d_chat.training.dense_alignment_supervision import (
    dense_alignment_supervision_settings,
    dense_alignment_warmup_settings,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)

CONFIG_PATH = Path("configs/experiments/gemma4_color_mirror_dense_alignment_v25.yaml")
SOURCE_ARCHIVE = Path("reports/gemma4/metrics/v24_final_summary.json")
SOURCE_CHECKPOINT = Path("data_gemma4/checkpoints/gemma4_v24_shared_query/epoch_001")
FAILURE_REPORTS = {
    "scene_000007": Path("reports/gemma4/metrics/v24_failure_semantic_scene_000007.json"),
    "scene_000008": Path("reports/gemma4/metrics/v24_failure_semantic_scene_000008.json"),
}
PRIMARY_NAMESPACE = "gemma4_v25_dense_alignment"
FROZEN_BANKS = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
)
EXPECTED_SOURCE_ARTIFACTS = {
    "adapter_sha256": "45e0c5affa9cf556e29bab5de418dffb867817b703c848bc6828255347748d31",
    "metadata_sha256": "216c501f5b248aa8f44e86198be3902d5f45d87774ad932056bafc95c4637e7b",
    "optimizer_sha256": "f1121353fc2c6b9239b8163390a0593832825abd9ff9f8dce4cce7f1cff99669",
}
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "287328190cc2f9e3ff771fa9d9f08f7186c75c9cb22b3eadad3bd002e55b9eb3"
)
EXPECTED_FAILURE_REPORT_SHA256 = {
    "scene_000007": "cdaaba41b37f6dbe9df0083e35a5b8c03a106aa9a83b1a549840e4baa3fe96f8",
    "scene_000008": "2e68ce27e5f50983b57e0519d04e50c51d050efcb81f27767ebd8d3e941eeb8a",
}
EXPECTED_FROZEN_HASHES = {
    "scene": "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b",
    "global": "ce3bf864eed6dd4a50f1b67296981e144d6a79e9cf192ad9a9230f2ae18208dc",
    "signed_x": "e8dabc69627f60723b89520b02dfee985e49b7b7e35fdd1213cc79f7b8164f58",
    "inherited_v12": "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594",
    "extension_v13": "4eb90fb9b0bea579d14cfcb0f61ebd5b6d566fd600bd3d5e1bfe5177a39e1b34",
    "extension_v23_shared_kv": (
        "91a9eea577cab5a37e840cdf4007722a398415846af91280713bcb2cda0f045c"
    ),
    "extension_v24_shared_query": (
        "6db2807476506b947bbaf01837490e97c12e57b1906bab671ef7c82ed36d6399"
    ),
}
EXPECTED_DENSE_INITIAL_SHA256 = (
    "bdff604538cfc82ba16ce5a701a573a4a64de486791c6e4ae7c4d5faba094874"
)
EXPECTED_DENSE_PARAMETER_COUNT = 24_576
EXPECTED_CALIBRATION_REPORT_SHA256 = (
    "642f16599892a8d7b9a2f21a7d74c1dba6d5f2dbb64c9d37333ffac43dae8637"
)
EXPECTED_CALIBRATION_FINAL_SHA256 = (
    "9a3a71fe4d7894cae694c00fa5eec5adcaff75fc27751067df7d1bf795c3566e"
)
EXPECTED_CALIBRATION_HISTORY_SHA256 = (
    "44250ee3cc93c82b5d0947030b0b0a9f500aea564919593356c12783dcd17b6d"
)
EXPECTED_HELD_OUT_LOCALIZATION_SHA256 = (
    "5b5c8f43b721c01fac453963eaedee1b4bee72a04b57ebd14bb20e6f2aa0f1a4"
)

# Filled only after the complete reviewed YAML and normalized contract are
# stable.  Explicit pins make later config drift fail before any model load.
EXPECTED_CONFIG_SHA256 = "34e1473397e6fb53485477ccb8160aadb83bd7a4c8e757ebf2125f43449fb6d4"
EXPECTED_CONTRACT_SHA256 = "30f6b2c9614288b275a5d452d1f858911426df11efb5e9afe7a3a3c87ce4c4b6"

_PROHIBITED_RUNTIME_KEY_PARTS = (
    "oracle",
    "category",
    "caption",
    "prototype",
    "text_embedding",
    "instance_id",
    "segmentation",
)
_OPAQUE_SCENE = re.compile(r"scene_[0-9]{6}")
_OPAQUE_QUESTION = re.compile(r"q_[0-9]{6}")
COLOR_PAIR_ID = "pair_000001"
MIRROR_PAIR_ID = "pair_000003"


class V25ControlViolation(ValueError):
    """A V25 authorization input or stage outcome violated the contract."""


def _fail(message: str) -> None:
    raise V25ControlViolation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _equal(observed: Any, expected: Any, field: str) -> None:
    scalar = (bool, int, float, str)
    if (
        isinstance(expected, scalar)
        and type(observed) is not type(expected)
        or observed != expected
    ):
        _fail(f"{field} mismatch: expected={expected!r} observed={observed!r}")


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail(f"{field} must be a finite number")
    return parsed


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _regular_file(path: Path, field: str) -> Path:
    resolved = _resolve(path)
    if resolved.is_symlink() or not resolved.is_file():
        _fail(f"{field} is not a regular non-symlink file: {resolved}")
    return resolved


def _load_json(path: Path, field: str) -> dict[str, Any]:
    resolved = _regular_file(path, field)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot load {field}: {error}")
    return dict(_mapping(payload, field))


def _write_json(value: Mapping[str, Any], path: Path) -> Path:
    destination = _resolve(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _clean_provenance(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    provenance = dict(value)
    try:
        require_clean_committed_source(provenance)
    except RuntimeError as error:
        _fail(f"{field} is not a clean committed source: {error}")
    return provenance


def _runtime_key_audit(keys: Sequence[str], *, field: str) -> dict[str, Any]:
    normalized = [str(key) for key in keys]
    prohibited = sorted(
        key
        for key in normalized
        if any(part in key.casefold() for part in _PROHIBITED_RUNTIME_KEY_PARTS)
    )
    if prohibited:
        _fail(f"{field} contains prohibited environmental payload keys: {prohibited}")
    return {
        "tensor_key_count": len(normalized),
        "prohibited_key_count": 0,
        "category_strings_serialized": False,
        "text_prototypes_serialized": False,
        "oracle_payload_serialized": False,
    }


def _require_numeric_hash_only(value: Any, *, field: str = "calibration audit") -> None:
    """Reject any serialized training payload other than scalars and hashes."""

    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{field} contains NaN or infinity")
        return
    if isinstance(value, str):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            _fail(f"{field} contains a non-hash string")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                _fail(f"{field} contains a non-string key")
            _require_numeric_hash_only(nested, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _require_numeric_hash_only(nested, field=f"{field}[{index}]")
        return
    _fail(f"{field} contains forbidden type {type(value).__name__}")


def v25_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize every preregistered V25 training and isolation decision."""

    training = _mapping(config.get("training"), "training")
    experiment = _mapping(config.get("experiment"), "experiment")
    screen = _mapping(config.get("v25_screen"), "v25_screen")
    dense = dense_alignment_settings(config)
    supervision = dense_alignment_supervision_settings(config)
    warmup = dense_alignment_warmup_settings(config)
    lora = lora_banks_settings(config).contract()
    banks = _sequence(lora.get("banks"), "language.lora_banks")
    frozen_bank_hashes: dict[str, Any] = {}
    for value in banks:
        bank = _mapping(value, "language.lora_banks entry")
        name = str(bank.get("name"))
        if name in FROZEN_BANKS:
            frozen_bank_hashes[name] = bank.get("expected_initial_state_sha256")
            _equal(bank.get("trainable"), False, f"LoRA bank {name} trainable")
    _equal(set(frozen_bank_hashes), set(FROZEN_BANKS), "frozen LoRA bank set")
    optimizer = _mapping(training.get("optimizer"), "training.optimizer")
    return {
        "schema_version": 1,
        "role": "frozen_v24_epoch1_all_voxel_dense_alignment_falsifier",
        "config_sha256": config_hash(dict(config), length=64),
        "source": {
            "archive_path": str(SOURCE_ARCHIVE),
            "archive_sha256": screen.get("source_archive_summary_sha256"),
            "checkpoint": training.get("initialize_from"),
            "checkpoint_epoch": screen.get("source_checkpoint_epoch"),
            "adapter_sha256": training.get("initialize_expected_adapter_sha256"),
            "metadata_sha256": training.get("initialize_expected_metadata_sha256"),
            "frozen_hashes": {
                "scene": experiment.get("source_scene_state_sha256"),
                "global": experiment.get("source_global_scene_residual_state_sha256"),
                "signed_x": experiment.get("source_signed_x_scene_residual_state_sha256"),
                **frozen_bank_hashes,
            },
            "failure_localization_sha256": screen.get(
                "source_failure_localization_sha256"
            ),
        },
        "dense_alignment": dense.contract(),
        "dense_alignment_trainable_parameter_count": experiment.get(
            "dense_alignment_trainable_parameter_count"
        ),
        "supervision": supervision.contract(),
        "warmup": warmup.contract(),
        "paired_qa": {
            "pair_steps_per_epoch": training.get("pair_steps_per_epoch"),
            "gradient_accumulation": training.get("gradient_accumulation"),
            "learning_rate": training.get("dense_alignment_learning_rate"),
            "weight_decay": training.get("dense_alignment_weight_decay"),
            "optimizer": dict(optimizer),
            "screen_optimizer_updates": screen.get("screen_optimizer_updates"),
            "conditional_max_optimizer_updates": screen.get(
                "conditional_max_optimizer_updates"
            ),
        },
        "training_surface": {
            "initialize_named_lora_freeze_for_dense_alignment_transition": training.get(
                "initialize_named_lora_freeze_for_dense_alignment_transition"
            ),
            "train_dense_alignment_only": training.get("train_dense_alignment_only"),
            "freeze_scene_adapter": training.get("freeze_scene_adapter"),
            "train_signed_x_scene_residual_only": training.get(
                "train_signed_x_scene_residual_only"
            ),
            "train_global_scene_residual_only": training.get(
                "train_global_scene_residual_only"
            ),
            "train_lora_with_frozen_scene_residual_stack": training.get(
                "train_lora_with_frozen_scene_residual_stack"
            ),
            "frozen_lora_bank_hashes": frozen_bank_hashes,
        },
        "runtime_isolation": {
            "question_dependent_scene_processing": experiment.get(
                "question_dependent_scene_processing"
            ),
            "question_dependent_retrieval": experiment.get(
                "question_dependent_retrieval"
            ),
            "runtime_oracle_access": experiment.get("runtime_oracle_access"),
            "runtime_category_strings": experiment.get("runtime_category_strings"),
            "runtime_text_prototypes": experiment.get("runtime_text_prototypes"),
            "all_voxels_transformed": experiment.get("all_voxels_transformed"),
        },
        "screen": {
            "stage_1_optimizer_updates": screen.get("stage_1_optimizer_updates"),
            "stage_1_stop_required": screen.get("stage_1_stop_required"),
            "held_out_localization_requires": screen.get(
                "held_out_localization_requires"
            ),
            "stage_2_requires": screen.get("stage_2_requires"),
            "eligibility_requires": screen.get("eligibility_requires"),
            "continuation_requires": screen.get("continuation_requires"),
            "full_teacher_gate_requires": screen.get("full_teacher_gate_requires"),
            "greedy_audit_only_after_full_teacher_gate": screen.get(
                "greedy_audit_only_after_full_teacher_gate"
            ),
        },
    }


def _validate_contract(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    resolved_sha = config_hash(dict(config), length=64)
    _equal(resolved_sha, EXPECTED_CONFIG_SHA256, "resolved V25 config SHA-256")
    contract = v25_contract(config)
    contract_sha = _canonical_sha256(contract)
    _equal(contract_sha, EXPECTED_CONTRACT_SHA256, "normalized V25 contract SHA-256")
    return contract, contract_sha


def _source_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    training = _mapping(config.get("training"), "training")
    archive_path = _regular_file(SOURCE_ARCHIVE, "V24 immutable summary")
    _equal(
        _file_sha256(archive_path),
        EXPECTED_SOURCE_ARCHIVE_SHA256,
        "V24 immutable summary SHA-256",
    )
    archive = _load_json(SOURCE_ARCHIVE, "V24 immutable summary")
    _equal(archive.get("archive_type"), "immutable_v24_final_summary", "V24 archive type")
    outcome = _mapping(archive.get("outcome"), "V24 outcome")
    _equal(outcome.get("selected_epoch"), 1, "V24 selected epoch")
    selected = _mapping(archive.get("selected_checkpoint"), "V24 selected checkpoint")
    _equal(selected.get("epoch"), 1, "V24 selected checkpoint epoch")
    _equal(selected.get("checkpoint_path"), str(SOURCE_CHECKPOINT), "V24 checkpoint path")
    artifact_hashes = _mapping(selected.get("artifact_hashes"), "V24 selected hashes")
    _equal(dict(artifact_hashes), EXPECTED_SOURCE_ARTIFACTS, "V24 selected artifact hashes")

    observed_hashes: dict[str, str] = {}
    for name in ("adapter.safetensors", "metadata.json", "optimizer.pt"):
        path = _regular_file(SOURCE_CHECKPOINT / name, f"V24 source {name}")
        observed_hashes[name] = _file_sha256(path)
    expected_by_name = {
        "adapter.safetensors": EXPECTED_SOURCE_ARTIFACTS["adapter_sha256"],
        "metadata.json": EXPECTED_SOURCE_ARTIFACTS["metadata_sha256"],
        "optimizer.pt": EXPECTED_SOURCE_ARTIFACTS["optimizer_sha256"],
    }
    _equal(observed_hashes, expected_by_name, "V24 source artifact hashes")
    for key, expected in {
        "initialize_expected_adapter_sha256": EXPECTED_SOURCE_ARTIFACTS["adapter_sha256"],
        "initialize_expected_metadata_sha256": EXPECTED_SOURCE_ARTIFACTS["metadata_sha256"],
    }.items():
        _equal(training.get(key), expected, f"training.{key}")

    metadata = _load_json(SOURCE_CHECKPOINT / "metadata.json", "V24 source metadata")
    _equal(metadata.get("epoch"), 1, "V24 metadata epoch")
    _equal(metadata.get("optimizer_step"), 1, "V24 metadata optimizer step")
    _equal(metadata.get("output_namespace"), "gemma4_v24_shared_query", "V24 namespace")
    _equal(
        metadata.get("frozen_scene_state_sha256"),
        EXPECTED_FROZEN_HASHES["scene"],
        "V24 scene state",
    )
    _equal(
        metadata.get("global_scene_residual_state_sha256"),
        EXPECTED_FROZEN_HASHES["global"],
        "V24 global residual state",
    )
    _equal(
        metadata.get("signed_x_scene_residual_state_sha256"),
        EXPECTED_FROZEN_HASHES["signed_x"],
        "V24 signed-X residual state",
    )
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "V24 LoRA hashes")
    _equal(
        dict(bank_hashes),
        {name: EXPECTED_FROZEN_HASHES[name] for name in FROZEN_BANKS},
        "V24 LoRA hashes",
    )
    source_adapter = _regular_file(
        SOURCE_CHECKPOINT / "adapter.safetensors", "V24 source adapter"
    )
    with safe_open(source_adapter, framework="pt", device="cpu") as handle:
        payload_audit = _runtime_key_audit(list(handle.keys()), field="V24 source adapter")

    failure_hashes: dict[str, str] = {}
    for scene_id, path in FAILURE_REPORTS.items():
        if _OPAQUE_SCENE.fullmatch(scene_id) is None:
            _fail("V25 failure report key is not an opaque scene ID")
        failure_hashes[scene_id] = _file_sha256(_regular_file(path, "V24 failure report"))
    _equal(failure_hashes, EXPECTED_FAILURE_REPORT_SHA256, "failure report hashes")
    return {
        "archive_path": str(SOURCE_ARCHIVE),
        "archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "checkpoint": str(SOURCE_CHECKPOINT),
        "artifact_hashes": EXPECTED_SOURCE_ARTIFACTS,
        "frozen_state_sha256": dict(EXPECTED_FROZEN_HASHES),
        "failure_localization_sha256": failure_hashes,
        "runtime_payload": payload_audit,
        "selected_v24_outcome": {
            "color_full_vocab_sides": outcome.get("selected_color_full_vocab_sides"),
            "mirror_full_vocab_sides": outcome.get("selected_mirror_full_vocab_sides"),
            "mirror_minimum_full_vocab_margin": outcome.get(
                "selected_mirror_minimum_full_vocab_margin"
            ),
        },
    }


def _dense_alignment_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    module = construct_dense_alignment(config, semantic_dim=3072)
    if not isinstance(module, DenseAlignmentResidual):
        _fail("V25 dense alignment was not constructed")
    audit = validate_dense_alignment_state(
        module,
        expected_parameter_count=EXPECTED_DENSE_PARAMETER_COUNT,
        context="V25 preflight",
    )
    _equal(audit.get("state_sha256"), EXPECTED_DENSE_INITIAL_SHA256, "dense initial hash")
    _equal(audit.get("b_exact_zero"), True, "dense B exact-zero state")
    _equal(
        tuple(inspect.signature(module.forward).parameters),
        ("semantic",),
        "dense runtime input signature",
    )
    state_keys = list(module.state_dict())
    _equal(
        set(state_keys),
        {"architecture_marker", "scaling", "alignment_a", "alignment_b"},
        "dense runtime tensor keys",
    )
    payload = _runtime_key_audit(state_keys, field="dense-alignment state")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(25025)
    semantic = torch.randn(17, 3072, generator=generator, dtype=torch.float32)
    with torch.inference_mode():
        delta = module.residual_delta(semantic)
        output = module(semantic)
    if torch.count_nonzero(delta).item() != 0 or not torch.equal(output, semantic):
        _fail("Dense alignment is not exact zero-output before calibration")

    locality_probe = copy.deepcopy(module)
    with torch.no_grad():
        locality_probe.alignment_b.copy_(
            torch.linspace(
                -0.002,
                0.002,
                locality_probe.alignment_b.numel(),
                dtype=torch.float32,
            ).reshape_as(locality_probe.alignment_b)
        )
    changed_input = semantic.clone()
    changed_input[8, :1536].add_(0.25)
    with torch.inference_mode():
        before = locality_probe(semantic)
        after = locality_probe(changed_input)
    unchanged_rows = torch.cat((after[:8] == before[:8], after[9:] == before[9:]), dim=0)
    if not bool(unchanged_rows.all()) or torch.equal(after[8], before[8]):
        _fail("Dense alignment is not voxel-local and order-preserving")
    if not torch.equal(before[:, :1536], semantic[:, :1536]):
        _fail("Dense alignment changed the native dense feature slice")
    return {
        **audit,
        "architecture_version": DENSE_ALIGNMENT_ARCHITECTURE_VERSION,
        "exact_zero_delta_nonzero_count": 0,
        "exact_zero_forward_bitwise_identity": True,
        "input_voxel_count": 17,
        "output_voxel_count": 17,
        "voxel_order_preserved": True,
        "voxel_locality_probe_passed": True,
        "dense_slice_bitwise_preserved": True,
        "runtime_payload": payload,
    }


def _training_surface_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = v25_contract(config)
    surface = _mapping(contract.get("training_surface"), "training surface")
    expected_surface = {
        "initialize_named_lora_freeze_for_dense_alignment_transition": True,
        "train_dense_alignment_only": True,
        "freeze_scene_adapter": True,
        "train_signed_x_scene_residual_only": False,
        "train_global_scene_residual_only": False,
        "train_lora_with_frozen_scene_residual_stack": False,
        "frozen_lora_bank_hashes": {
            name: EXPECTED_FROZEN_HASHES[name] for name in FROZEN_BANKS
        },
    }
    _equal(dict(surface), expected_surface, "V25 isolated trainable surface")
    _equal(
        contract.get("dense_alignment_trainable_parameter_count"),
        EXPECTED_DENSE_PARAMETER_COUNT,
        "V25 dense trainable parameter count",
    )
    paired = _mapping(contract.get("paired_qa"), "paired QA contract")
    _equal(paired.get("pair_steps_per_epoch"), 12, "paired QA microsteps")
    _equal(paired.get("gradient_accumulation"), 12, "paired QA accumulation")
    _equal(paired.get("learning_rate"), 0.0003, "paired QA learning rate")
    _equal(paired.get("weight_decay"), 0.0, "paired QA weight decay")
    return {
        "only_trainable_module": "dense_alignment",
        "trainable_parameter_names": ["alignment_a", "alignment_b"],
        "trainable_parameter_count": EXPECTED_DENSE_PARAMETER_COUNT,
        "all_scene_decoder_and_lora_surfaces_frozen": True,
        "paired_qa_optimizer_constructed_after_warmup": True,
        "paired_qa_optimizer_state_starts_empty": True,
        "pair_steps_per_optimizer_update": 12,
    }


def _runtime_isolation_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment = _mapping(config.get("experiment"), "experiment")
    expected = {
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "runtime_oracle_access": False,
        "runtime_category_strings": False,
        "runtime_text_prototypes": False,
        "all_voxels_transformed": True,
    }
    observed = {key: experiment.get(key) for key in expected}
    _equal(observed, expected, "V25 runtime isolation contract")
    supervision = dense_alignment_supervision_settings(config)
    warmup = dense_alignment_warmup_settings(config)
    _equal(
        supervision.calibration_scene_ids,
        tuple(
            f"scene_{index:06d}" for index in (*range(1, 7), 9, 10)
        ),
        "semantic calibration scene IDs",
    )
    _equal(
        supervision.held_out_scene_ids,
        ("scene_000007", "scene_000008"),
        "held-out scene IDs",
    )
    return {
        **expected,
        "semantic_calibration_scene_count": len(supervision.calibration_scene_ids),
        "held_out_scene_count": len(supervision.held_out_scene_ids),
        "calibration_and_held_out_disjoint": True,
        "warmup_training_only": warmup.training_only,
        "held_out_scene_gradient_access": warmup.held_out_scene_gradient_access,
        "runtime_checkpoint_tensor_payload_only": True,
        "environmental_text_serialized": False,
    }


def run_preflight(
    config_path: Path = CONFIG_PATH,
    output: Path | None = None,
    *,
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authorize only the bounded semantic-calibration stage."""

    config = load_config(config_path)
    contract, contract_sha = _validate_contract(config)
    provenance = (
        capture_git_source_provenance(PROJECT_ROOT)
        if source_provenance is None
        else dict(source_provenance)
    )
    clean_provenance = _clean_provenance(provenance, "V25 controller provenance")
    report = {
        "schema_version": 1,
        "audit_type": "v25_dense_alignment_structural_preflight",
        "authorized": True,
        "calibration_stage_authorized": True,
        "paired_qa_stage_authorized": False,
        "paired_qa_authorization_condition": "successful_bounded_semantic_calibration",
        "runtime_eligible": False,
        "report_only": True,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "optimizer_steps": 0,
        "oracle_loaded": False,
        "config_path": str(CONFIG_PATH),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract": contract,
        "contract_sha256": contract_sha,
        "source_provenance": clean_provenance,
        "source": _source_audit(config),
        "dense_alignment": _dense_alignment_audit(config),
        "training_surface": _training_surface_audit(config),
        "runtime_isolation": _runtime_isolation_audit(config),
        "warmup": dense_alignment_warmup_settings(config).contract(),
        "supervision": dense_alignment_supervision_settings(config).contract(),
    }
    if output is not None:
        destination = _write_json(report, output)
        report["output"] = str(destination)
    return report


def validate_preflight(path: Path, config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Validate a persisted report without trusting its authorization booleans."""

    report = _load_json(path, "V25 preflight")
    config = load_config(config_path)
    contract, contract_sha = _validate_contract(config)
    for field, expected in {
        "schema_version": 1,
        "audit_type": "v25_dense_alignment_structural_preflight",
        "authorized": True,
        "calibration_stage_authorized": True,
        "paired_qa_stage_authorized": False,
        "paired_qa_authorization_condition": "successful_bounded_semantic_calibration",
        "runtime_eligible": False,
        "report_only": True,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "optimizer_steps": 0,
        "oracle_loaded": False,
        "config_path": str(CONFIG_PATH),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract_sha256": contract_sha,
        "contract": contract,
    }.items():
        _equal(report.get(field), expected, f"V25 preflight {field}")
    _clean_provenance(
        _mapping(report.get("source_provenance"), "V25 preflight source provenance"),
        "V25 preflight source provenance",
    )
    dense = _mapping(report.get("dense_alignment"), "V25 preflight dense alignment")
    _equal(dense.get("state_sha256"), EXPECTED_DENSE_INITIAL_SHA256, "preflight dense hash")
    _equal(dense.get("exact_zero_forward_bitwise_identity"), True, "preflight zero output")
    runtime = _mapping(report.get("runtime_isolation"), "V25 preflight runtime isolation")
    _equal(runtime.get("runtime_oracle_access"), False, "preflight runtime oracle access")
    _equal(runtime.get("all_voxels_transformed"), True, "preflight all-voxel path")
    return report


def verify_calibration_report(
    *,
    config_path: Path,
    calibration_path: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Verify the deterministic terminal calibration denial without relaxing it."""

    config = load_config(config_path)
    _validate_contract(config)
    settings = dense_alignment_warmup_settings(config)
    supervision = dense_alignment_supervision_settings(config)
    report_path = _regular_file(calibration_path, "V25 calibration report")
    report_sha = _file_sha256(report_path)
    _equal(
        report_sha,
        EXPECTED_CALIBRATION_REPORT_SHA256,
        "V25 deterministic calibration report SHA-256",
    )
    audit = _load_json(report_path, "V25 calibration report")
    _require_numeric_hash_only(audit)
    for field, expected in {
        "schema_version": 1,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "calibration_split_sha256": _canonical_sha256(
            list(supervision.calibration_scene_ids)
        ),
        "held_out_split_sha256": _canonical_sha256(list(supervision.held_out_scene_ids)),
        "initial_state_sha256": EXPECTED_DENSE_INITIAL_SHA256,
        "final_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
        "calibration_scene_count": 8,
        "held_out_scene_count": 2,
        "category_count": 9,
        "region_count": 68,
        "skipped_underfilled_region_count": 3,
        "selective_token_row_count": 12,
        "loaded_parameter_count": 1,
        "cpu_only": True,
        "local_files_only": True,
        "raw_map_write_count": 0,
        "raw_maps_preserved": True,
        "question_dependent_selection": False,
        "qa_update_authorized": False,
        "bridge_written": False,
        "bridge_sha256": None,
    }.items():
        _equal(audit.get(field), expected, f"V25 calibration {field}")
    training = _mapping(audit.get("training"), "V25 calibration training")
    for field, expected in {
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "delta_mse_regularization_weight": settings.delta_rms_regularization_weight,
        "maximum_optimizer_steps": settings.max_optimizer_steps,
        "optimizer_steps": settings.max_optimizer_steps,
        "stopped_at_first_pass": False,
        "calibration_passed": False,
        "final_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
    }.items():
        _equal(training.get(field), expected, f"V25 calibration training.{field}")
    history = _sequence(training.get("history"), "V25 calibration history")
    _equal(len(history), settings.max_optimizer_steps, "V25 calibration history length")
    _equal(
        _canonical_sha256(history),
        EXPECTED_CALIBRATION_HISTORY_SHA256,
        "V25 calibration history SHA-256",
    )
    for expected_step, value in enumerate(history, start=1):
        row = _mapping(value, f"V25 calibration history {expected_step}")
        _equal(row.get("optimizer_step"), expected_step, "V25 calibration history step")
        _equal(row.get("passed"), False, "V25 calibration history pass")
    held_out = _mapping(audit.get("held_out_localization"), "V25 held-out localization")
    _equal(held_out.get("passed"), True, "V25 held-out localization pass")
    _equal(held_out.get("target_region_count"), 4, "V25 held-out target count")
    _equal(held_out.get("all_target_hit_at_k"), True, "V25 held-out hit-at-k")
    _equal(
        _canonical_sha256(held_out),
        EXPECTED_HELD_OUT_LOCALIZATION_SHA256,
        "V25 held-out localization SHA-256",
    )
    final = _mapping(history[-1], "V25 final calibration history row")
    decision = {
        "schema_version": 1,
        "audit_type": "v25_dense_alignment_calibration_terminal_decision",
        "decision": "bounded_calibration_failed_stop_before_paired_qa",
        "calibration_authorized": False,
        "paired_qa_stage_authorized": False,
        "terminal_stop": True,
        "thresholds_preserved": True,
        "threshold_relaxation_permitted": False,
        "optimizer_steps": settings.max_optimizer_steps,
        "maximum_optimizer_steps": settings.max_optimizer_steps,
        "final_top1_accuracy": _finite_float(
            final.get("top1_accuracy"), "V25 final calibration accuracy"
        ),
        "final_minimum_cosine_margin": _finite_float(
            final.get("minimum_cosine_margin"), "V25 final calibration margin"
        ),
        "final_delta_rms": _finite_float(final.get("delta_rms"), "V25 final delta RMS"),
        "final_delta_abs_max": _finite_float(
            final.get("delta_abs_max"), "V25 final delta absolute maximum"
        ),
        "held_out_localization_passed": True,
        "qa_update_authorized": False,
        "greedy_audit_authorized": False,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "model_loaded": False,
        "oracle_loaded_by_controller": False,
        "report_only": True,
        "source_report": str(calibration_path),
        "source_report_sha256": report_sha,
        "final_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
        "history_sha256": EXPECTED_CALIBRATION_HISTORY_SHA256,
        "held_out_localization_sha256": EXPECTED_HELD_OUT_LOCALIZATION_SHA256,
    }
    if output is not None:
        destination = _write_json(decision, output)
        decision["output"] = str(destination)
    return decision


def _warmup_gate(audit: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    _require_numeric_hash_only(audit)
    settings = dense_alignment_warmup_settings(config)
    supervision = dense_alignment_supervision_settings(config)
    training = _mapping(audit.get("training"), "dense-alignment calibration training")
    history = _sequence(training.get("history"), "dense-alignment calibration history")
    if not history:
        _fail("dense-alignment calibration history cannot be empty")
    final = _mapping(history[-1], "dense-alignment calibration final history row")
    steps = training.get("optimizer_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= (
        settings.max_optimizer_steps
    ):
        _fail("dense-alignment warm-up optimizer steps are outside the bounded screen")
    _equal(len(history), steps, "dense-alignment warm-up history length")
    for expected_step, value in enumerate(history, start=1):
        row = _mapping(value, f"dense-alignment calibration history {expected_step}")
        _equal(row.get("optimizer_step"), expected_step, "warm-up history optimizer step")
        if expected_step < steps:
            _equal(row.get("passed"), False, "pre-final warm-up history pass")
    accuracy = _finite_float(final.get("top1_accuracy"), "warm-up top1 accuracy")
    margin = _finite_float(final.get("minimum_cosine_margin"), "warm-up minimum margin")
    delta_rms = _finite_float(final.get("delta_rms"), "warm-up delta RMS")
    delta_abs_max = _finite_float(final.get("delta_abs_max"), "warm-up delta abs max")
    final_state = audit.get("final_state_sha256")
    checks = {
        "schema_version": audit.get("schema_version") == 1,
        "config_sha256": audit.get("config_sha256") == EXPECTED_CONFIG_SHA256,
        "calibration_split_sha256": audit.get("calibration_split_sha256")
        == _canonical_sha256(list(supervision.calibration_scene_ids)),
        "held_out_split_sha256": audit.get("held_out_split_sha256")
        == _canonical_sha256(list(supervision.held_out_scene_ids)),
        "calibration_scene_count": audit.get("calibration_scene_count") == 8,
        "held_out_scene_count": audit.get("held_out_scene_count") == 2,
        "category_count": audit.get("category_count") == 9,
        "skipped_underfilled_region_count": (
            audit.get("skipped_underfilled_region_count") == 3
        ),
        "loaded_parameter_count": audit.get("loaded_parameter_count") == 1,
        "cpu_only": audit.get("cpu_only") is True,
        "local_files_only": audit.get("local_files_only") is True,
        "raw_map_write_count": audit.get("raw_map_write_count") == 0,
        "raw_maps_preserved": audit.get("raw_maps_preserved") is True,
        "question_dependent_selection": audit.get("question_dependent_selection") is False,
        "training_learning_rate": training.get("learning_rate") == settings.learning_rate,
        "training_weight_decay": training.get("weight_decay") == settings.weight_decay,
        "training_regularizer": training.get("delta_mse_regularization_weight")
        == settings.delta_rms_regularization_weight,
        "training_maximum_steps": training.get("maximum_optimizer_steps")
        == settings.max_optimizer_steps,
        "stopped_at_first_passing_evaluation": (
            training.get("stopped_at_first_pass") is True
        ),
        "calibration_passed": training.get("calibration_passed") is True,
        "final_history_passed": final.get("passed") is True,
        "top1_accuracy": accuracy >= settings.early_stop_top1_accuracy,
        "minimum_margin": margin >= settings.early_stop_minimum_margin,
        "delta_rms": delta_rms <= settings.delta_rms_cap,
        "delta_abs_max": delta_abs_max <= settings.delta_abs_max_cap,
        "qa_update_authorized": audit.get("qa_update_authorized") is True,
        "pair_optimizer_state_empty_before_warmup": (
            audit.get("pair_optimizer_state_empty_before_warmup") is True
        ),
        "pair_optimizer_rebuilt_after_warmup": (
            audit.get("pair_optimizer_rebuilt_after_warmup") is True
        ),
        "pair_optimizer_state_empty_after_warmup": (
            audit.get("pair_optimizer_state_empty_after_warmup") is True
        ),
        "pair_optimizer_steps_before_qa": audit.get("pair_optimizer_steps_before_qa") == 0,
        "held_out_scene_gradient_access": (
            audit.get("held_out_scene_gradient_access") is False
        ),
        "category_text_prototypes_serialized": (
            audit.get("category_text_prototypes_serialized") is False
        ),
        "oracle_payload_retained": audit.get("oracle_payload_retained") is False,
        "nested_final_state": training.get("final_state_sha256") == final_state,
    }
    held_out = _mapping(
        audit.get("held_out_localization"), "dense-alignment held-out localization"
    )
    checks.update(
        {
            "held_out_localization_passed": held_out.get("passed") is True,
            "held_out_target_region_count": held_out.get("target_region_count") == 4,
            "held_out_all_hit": held_out.get("all_target_hit_at_k") is True,
            "held_out_precision": _finite_float(
                held_out.get("minimum_precision_at_k"), "held-out minimum precision"
            )
            >= 0.10,
            "held_out_region_margin": _finite_float(
                held_out.get("minimum_region_margin"), "held-out minimum region margin"
            )
            > 0.0,
            "held_out_query_margin": _finite_float(
                held_out.get("minimum_correct_vs_distractor_margin"),
                "held-out minimum correct-vs-distractor margin",
            )
            > 0.0,
            "held_out_mirror_error": _finite_float(
                held_out.get("maximum_mirror_centroid_error_m"),
                "held-out maximum mirror centroid error",
            )
            <= 0.15,
        }
    )
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        _fail(f"dense-alignment warm-up gate failed: {failed}")
    initial_hash = audit.get("initial_state_sha256")
    final_hash = final_state
    _equal(initial_hash, EXPECTED_DENSE_INITIAL_SHA256, "warm-up initial dense hash")
    if not isinstance(final_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", final_hash):
        _fail("warm-up final state hash is not a lowercase SHA-256")
    if final_hash == initial_hash:
        _fail("warm-up did not change the dense-alignment state")
    return {
        "passed": True,
        "optimizer_steps": steps,
        "top1_accuracy": accuracy,
        "minimum_margin": margin,
        "delta_rms": delta_rms,
        "delta_abs_max": delta_abs_max,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "checks": checks,
    }


def _empirical_count(value: Any, denominator: int, field: str) -> int:
    accuracy = _finite_float(value, field)
    if not 0.0 <= accuracy <= 1.0:
        _fail(f"{field} must be a probability")
    scaled = accuracy * denominator
    count = round(scaled)
    if not math.isclose(scaled, count, rel_tol=0.0, abs_tol=1e-5):
        _fail(f"{field} is not an exact {denominator}-way empirical fraction")
    return count


def _pair_metrics(metadata: Mapping[str, Any], pair_id: str) -> dict[str, Any]:
    gate = _mapping(metadata.get("pair_candidate_gate"), "pair candidate gate")
    by_pair = _mapping(gate.get("by_pair"), "pair candidate gate by_pair")
    pair = _mapping(by_pair.get(pair_id), f"pair candidate gate {pair_id}")
    return {
        "full_vocab_sides": _empirical_count(
            pair.get("first_answer_token_top1_accuracy"), 12, f"{pair_id} side accuracy"
        ),
        "full_vocab_units": _empirical_count(
            pair.get("first_answer_token_top1_unit_accuracy"),
            6,
            f"{pair_id} unit accuracy",
        ),
        "mean_candidate_margin": _finite_float(
            pair.get("mean_own_vs_alternate_candidate_logit_margin"),
            f"{pair_id} mean candidate margin",
        ),
        "minimum_candidate_margin": _finite_float(
            pair.get("minimum_own_vs_alternate_candidate_logit_margin"),
            f"{pair_id} minimum candidate margin",
        ),
        "mean_full_vocab_margin": _finite_float(
            pair.get("mean_first_answer_token_target_vs_best_other_logit_margin"),
            f"{pair_id} mean full-vocabulary margin",
        ),
        "minimum_full_vocab_margin": _finite_float(
            pair.get("minimum_first_answer_token_target_vs_best_other_logit_margin"),
            f"{pair_id} minimum full-vocabulary margin",
        ),
    }


def _mirror_side_margins(metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    gate = _mapping(metadata.get("pair_candidate_gate"), "pair candidate gate")
    detail = _mapping(gate.get("detail"), "pair candidate gate detail")
    for field, expected in {
        "schema_version": 1,
        "artifact": "training_candidate_gate_detail",
        "training_only": True,
        "free_generation_evaluated": False,
        "contains_question_text": False,
        "contains_oracle_geometry": False,
        "contains_canonical_training_targets": False,
        "full_vocab_first_token_evaluated": True,
        "unit_count": 12,
        "side_count": 24,
    }.items():
        _equal(detail.get(field), expected, f"candidate detail {field}")
    units = _sequence(detail.get("units"), "candidate detail units")
    _equal(len(units), 12, "candidate detail unit count")
    sides_by_key: dict[str, dict[str, Any]] = {}
    for expected_unit_index, value in enumerate(units[6:], start=6):
        unit = _mapping(value, f"mirror candidate unit {expected_unit_index}")
        _equal(unit.get("unit_index"), expected_unit_index, "mirror unit index")
        sides = _sequence(unit.get("sides"), "mirror candidate sides")
        _equal(len(sides), 2, "mirror unit side count")
        for expected_side_index, side_value in enumerate(sides):
            side = _mapping(side_value, "mirror candidate side")
            _equal(side.get("side_index"), expected_side_index, "mirror side index")
            scene_id = side.get("scene_id")
            question_id = side.get("question_id")
            if not isinstance(scene_id, str) or _OPAQUE_SCENE.fullmatch(scene_id) is None:
                _fail("mirror candidate detail contains a non-opaque scene ID")
            if not isinstance(question_id, str) or _OPAQUE_QUESTION.fullmatch(question_id) is None:
                _fail("mirror candidate detail contains a non-opaque question ID")
            candidate = _finite_float(
                side.get("own_vs_alternate_candidate_logit_margin"),
                "mirror side candidate margin",
            )
            full = _finite_float(
                side.get("first_token_target_vs_best_other_logit_margin"),
                "mirror side full-vocabulary margin",
            )
            _equal(side.get("own_preference_passed"), candidate > 0.0, "mirror side preference")
            _equal(side.get("full_vocab_top1_passed"), full > 0.0, "mirror side full pass")
            key = f"{scene_id}:{question_id}"
            if key in sides_by_key:
                _fail("mirror candidate detail contains a duplicate opaque side")
            sides_by_key[key] = {
                "scene_id": scene_id,
                "question_id": question_id,
                "candidate_margin": candidate,
                "full_vocab_margin": full,
            }
    _equal(len(sides_by_key), 12, "mirror side count")
    metrics = _pair_metrics(metadata, MIRROR_PAIR_ID)
    candidate_values = [float(value["candidate_margin"]) for value in sides_by_key.values()]
    full_values = [float(value["full_vocab_margin"]) for value in sides_by_key.values()]
    _equal(
        sum(value > 0.0 for value in full_values),
        metrics["full_vocab_sides"],
        "mirror detail full-vocabulary sides",
    )
    if not math.isclose(
        min(full_values),
        float(metrics["minimum_full_vocab_margin"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        _fail("mirror detail minimum full-vocabulary margin differs from aggregate")
    if not math.isclose(
        min(candidate_values),
        float(metrics["minimum_candidate_margin"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        _fail("mirror detail minimum candidate margin differs from aggregate")
    return sides_by_key


def _teacher_forced_gate(metadata: Mapping[str, Any]) -> dict[str, Any]:
    color = _pair_metrics(metadata, COLOR_PAIR_ID)
    mirror = _pair_metrics(metadata, MIRROR_PAIR_ID)
    source_metadata = _load_json(SOURCE_CHECKPOINT / "metadata.json", "V24 source metadata")
    source_sides = _mirror_side_margins(source_metadata)
    current_sides = _mirror_side_margins(metadata)
    _equal(set(current_sides), set(source_sides), "V25/source mirror side identities")
    source_negative = sorted(
        key for key, value in source_sides.items() if float(value["full_vocab_margin"]) <= 0.0
    )
    _equal(len(source_negative), 2, "V24 source negative mirror side count")
    strictly_improved = all(
        float(current_sides[key]["full_vocab_margin"])
        > float(source_sides[key]["full_vocab_margin"])
        for key in source_negative
    )
    new_negative = sorted(
        key
        for key, source in source_sides.items()
        if float(source["full_vocab_margin"]) > 0.0
        and float(current_sides[key]["full_vocab_margin"]) <= 0.0
    )
    color_eligible = bool(
        color["full_vocab_sides"] == 12
        and color["full_vocab_units"] == 6
        and color["minimum_candidate_margin"] > 0.0
        and color["minimum_full_vocab_margin"] > 0.0
    )
    continuation_eligible = bool(
        mirror["full_vocab_sides"] >= 10 and mirror["full_vocab_units"] >= 4
    )
    stage_2_passed = bool(
        color_eligible
        and continuation_eligible
        and strictly_improved
        and not new_negative
    )
    full_teacher_gate = bool(
        color_eligible
        and mirror["full_vocab_sides"] == 12
        and mirror["full_vocab_units"] == 6
        and mirror["minimum_candidate_margin"] > 0.0
        and mirror["minimum_full_vocab_margin"] > 0.0
    )
    return {
        "color": color,
        "mirror": mirror,
        "source_negative_side_count": 2,
        "source_negative_side_ids_sha256": _canonical_sha256(source_negative),
        "both_source_negative_margins_strictly_improved": strictly_improved,
        "new_negative_mirror_side_count": len(new_negative),
        "new_negative_mirror_side_ids_sha256": _canonical_sha256(new_negative),
        "color_eligible": color_eligible,
        "continuation_eligible": continuation_eligible,
        "stage_2_passed": stage_2_passed,
        "full_teacher_gate_passed": full_teacher_gate,
    }


def verify_update1_metadata(
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the logical warm-up boundary and first paired QA update."""

    _equal(metadata.get("epoch"), 1, "V25 update-1 epoch")
    _equal(metadata.get("optimizer_step"), 1, "V25 paired QA optimizer step")
    _equal(metadata.get("output_namespace"), PRIMARY_NAMESPACE, "V25 output namespace")
    _equal(metadata.get("train_dense_alignment_only"), True, "V25 dense-only mode")
    _equal(metadata.get("freeze_scene_adapter"), True, "V25 frozen scene adapter")
    _equal(
        metadata.get("dense_alignment_parameter_count"),
        EXPECTED_DENSE_PARAMETER_COUNT,
        "V25 dense parameter count",
    )
    warmup = _mapping(metadata.get("dense_alignment_calibration"), "V25 warm-up audit")
    warmup_gate = _warmup_gate(warmup, config)
    _equal(
        metadata.get("frozen_scene_state_sha256"),
        EXPECTED_FROZEN_HASHES["scene"],
        "V25 frozen scene state",
    )
    for field, key in (
        ("global_scene_residual_state_sha256", "global"),
        ("signed_x_scene_residual_state_sha256", "signed_x"),
    ):
        _equal(metadata.get(field), EXPECTED_FROZEN_HASHES[key], f"V25 frozen {key} state")
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "V25 LoRA hashes")
    _equal(
        dict(bank_hashes),
        {name: EXPECTED_FROZEN_HASHES[name] for name in FROZEN_BANKS},
        "V25 frozen LoRA hashes",
    )
    teacher_gate = _teacher_forced_gate(metadata)
    if not bool(teacher_gate["stage_2_passed"]):
        _fail("V25 update 1 did not pass the preregistered stage-2 teacher-forced gate")
    return {
        "match": True,
        "calibration_gate": warmup_gate,
        "teacher_forced_gate": teacher_gate,
        "paired_qa_optimizer_step": 1,
        "all_source_surfaces_frozen": True,
        "stage_2_authorized": True,
        "greedy_audit_authorized": False,
    }


def verify_update1(
    *,
    config_path: Path,
    preflight_path: Path,
    checkpoint: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Bind a stage-1 checkpoint to its structural preflight and warm-up gate."""

    config = load_config(config_path)
    preflight = validate_preflight(preflight_path, config_path)
    checkpoint_path = _resolve(checkpoint)
    metadata_path = _regular_file(checkpoint_path / "metadata.json", "V25 update-1 metadata")
    adapter_path = _regular_file(
        checkpoint_path / "adapter.safetensors", "V25 update-1 adapter"
    )
    metadata = _load_json(metadata_path, "V25 update-1 metadata")
    verification = verify_update1_metadata(metadata, config)
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        payload_audit = _runtime_key_audit(keys, field="V25 update-1 adapter")
        dense_state = {
            key.removeprefix("dense_aligner."): handle.get_tensor(key)
            for key in keys
            if key.startswith("dense_aligner.")
        }
    _equal(
        set(dense_state),
        {"architecture_marker", "scaling", "alignment_a", "alignment_b"},
        "V25 checkpoint dense tensor keys",
    )
    module = construct_dense_alignment(config, semantic_dim=3072)
    if not isinstance(module, DenseAlignmentResidual):
        _fail("V25 checkpoint dense module could not be constructed")
    module.load_state_dict(dense_state, strict=True)
    dense_audit = validate_dense_alignment_state(
        module,
        expected_parameter_count=EXPECTED_DENSE_PARAMETER_COUNT,
        context="V25 update-1 checkpoint",
    )
    _equal(
        dense_audit.get("state_sha256"),
        verification["calibration_gate"]["final_state_sha256"],
        "V25 checkpoint warm-up final dense hash",
    )
    _equal(dense_audit.get("b_exact_zero"), False, "V25 checkpoint dense B state")
    report = {
        "schema_version": 1,
        "audit_type": "v25_dense_alignment_update1_verification",
        "match": True,
        "stage_2_authorized": True,
        "greedy_audit_authorized": False,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "preflight_sha256": _file_sha256(_regular_file(preflight_path, "V25 preflight")),
        "preflight_contract_sha256": preflight["contract_sha256"],
        "checkpoint": str(checkpoint),
        "artifact_hashes": {
            "adapter_sha256": _file_sha256(adapter_path),
            "metadata_sha256": _file_sha256(metadata_path),
        },
        "verification": verification,
        "dense_alignment": dense_audit,
        "runtime_payload": payload_audit,
    }
    if output is not None:
        destination = _write_json(report, output)
        report["output"] = str(destination)
    return report


def _epoch_audit(
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    expected_epoch: int,
) -> dict[str, Any]:
    _equal(metadata.get("epoch"), expected_epoch, f"V25 epoch {expected_epoch}")
    _equal(
        metadata.get("optimizer_step"),
        expected_epoch,
        f"V25 epoch {expected_epoch} optimizer step",
    )
    namespace = metadata.get("output_namespace")
    screen = _mapping(config.get("v25_screen"), "v25_screen")
    allowed_namespaces = {
        screen.get("primary_output_namespace"),
        screen.get("extension_output_namespace"),
    }
    if namespace not in allowed_namespaces:
        _fail(f"V25 epoch {expected_epoch} has an unauthorized output namespace")
    _equal(metadata.get("train_dense_alignment_only"), True, "V25 dense-only mode")
    _equal(metadata.get("freeze_scene_adapter"), True, "V25 frozen scene adapter")
    _equal(
        metadata.get("dense_alignment_parameter_count"),
        EXPECTED_DENSE_PARAMETER_COUNT,
        "V25 dense parameter count",
    )
    calibration = _mapping(
        metadata.get("dense_alignment_calibration"), "V25 calibration audit"
    )
    calibration_gate = _warmup_gate(calibration, config)
    _equal(
        metadata.get("frozen_scene_state_sha256"),
        EXPECTED_FROZEN_HASHES["scene"],
        "V25 frozen scene state",
    )
    _equal(
        metadata.get("global_scene_residual_state_sha256"),
        EXPECTED_FROZEN_HASHES["global"],
        "V25 frozen global state",
    )
    _equal(
        metadata.get("signed_x_scene_residual_state_sha256"),
        EXPECTED_FROZEN_HASHES["signed_x"],
        "V25 frozen signed-X state",
    )
    _equal(
        dict(_mapping(metadata.get("lora_bank_state_sha256"), "V25 LoRA hashes")),
        {name: EXPECTED_FROZEN_HASHES[name] for name in FROZEN_BANKS},
        "V25 frozen LoRA hashes",
    )
    dense_state_sha = metadata.get("dense_alignment_state_sha256")
    if not isinstance(dense_state_sha, str) or re.fullmatch(r"[0-9a-f]{64}", dense_state_sha) is None:
        _fail("V25 epoch dense-alignment state hash is invalid")
    if dense_state_sha == EXPECTED_DENSE_INITIAL_SHA256:
        _fail("V25 epoch retained the exact-zero initial dense-alignment state")
    teacher = _teacher_forced_gate(metadata)
    return {
        "epoch": expected_epoch,
        "optimizer_step": expected_epoch,
        "output_namespace": namespace,
        "dense_alignment_state_sha256": dense_state_sha,
        "calibration_final_state_sha256": calibration_gate["final_state_sha256"],
        "calibration_optimizer_steps": calibration_gate["optimizer_steps"],
        "teacher_forced_gate": teacher,
    }


def select_epoch_metadata(
    epochs: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the exact four-update screen or its conditional eight-update cap."""

    observed_epochs = sorted(epochs)
    if observed_epochs not in (list(range(1, 5)), list(range(1, 9))):
        _fail("V25 selection requires exactly epochs 1--4 or exactly epochs 1--8")
    audits = [
        _epoch_audit(epochs[epoch], config, expected_epoch=epoch)
        for epoch in observed_epochs
    ]
    calibration_hashes = {
        str(audit["calibration_final_state_sha256"]) for audit in audits
    }
    _equal(len(calibration_hashes), 1, "V25 calibration state across epochs")
    eligible = [
        audit
        for audit in audits
        if bool(audit["teacher_forced_gate"]["stage_2_passed"])
    ]
    full = [
        audit
        for audit in audits
        if bool(audit["teacher_forced_gate"]["full_teacher_gate_passed"])
    ]
    if full:
        selected = min(full, key=lambda audit: int(audit["epoch"]))
        decision = "full_teacher_gate_passed_greedy_audit_authorized"
        greedy_authorized = True
        conditional_extension = False
    elif eligible:
        selected = max(
            eligible,
            key=lambda audit: (
                int(audit["teacher_forced_gate"]["mirror"]["full_vocab_sides"]),
                int(audit["teacher_forced_gate"]["mirror"]["full_vocab_units"]),
                float(
                    audit["teacher_forced_gate"]["mirror"][
                        "minimum_full_vocab_margin"
                    ]
                ),
                float(
                    audit["teacher_forced_gate"]["mirror"]["minimum_candidate_margin"]
                ),
                -int(audit["epoch"]),
            ),
        )
        greedy_authorized = False
        conditional_extension = len(audits) == 4
        decision = (
            "conditional_extension_to_update_8_authorized"
            if conditional_extension
            else "conditional_update_8_limit_reached_no_greedy_audit"
        )
    else:
        selected = None
        decision = "no_eligible_epoch_stop"
        greedy_authorized = False
        conditional_extension = False
    ranking = sorted(
        audits,
        key=lambda audit: (
            bool(audit["teacher_forced_gate"]["full_teacher_gate_passed"]),
            bool(audit["teacher_forced_gate"]["stage_2_passed"]),
            int(audit["teacher_forced_gate"]["mirror"]["full_vocab_sides"]),
            int(audit["teacher_forced_gate"]["mirror"]["full_vocab_units"]),
            float(audit["teacher_forced_gate"]["mirror"]["minimum_full_vocab_margin"]),
            -int(audit["epoch"]),
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "audit_type": "v25_dense_alignment_bounded_epoch_selector",
        "decision": decision,
        "evaluated_optimizer_updates": len(audits),
        "hard_optimizer_update_limit": 8,
        "primary_screen_complete": len(audits) >= 4,
        "conditional_limit_reached": len(audits) == 8 and not full,
        "conditional_extension_authorized": conditional_extension,
        "full_teacher_gate_passed": bool(full),
        "full_teacher_first_pass_epoch": (
            None if not full else min(int(audit["epoch"]) for audit in full)
        ),
        "greedy_audit_authorized": greedy_authorized,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "runtime_eligible": False,
        "selected_epoch": None if selected is None else selected["epoch"],
        "ranking": [audit["epoch"] for audit in ranking],
        "epochs": audits,
    }


def _parse_epoch_binding(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("epoch binding must be EPOCH=PATH")
    epoch_text, path_text = value.split("=", 1)
    try:
        epoch = int(epoch_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("epoch binding must be EPOCH=PATH") from error
    if epoch < 1 or not path_text:
        raise argparse.ArgumentTypeError("epoch binding must be EPOCH=PATH")
    return epoch, Path(path_text)


def select_epochs(
    *,
    config_path: Path,
    update1_report_path: Path,
    epoch_bindings: Sequence[tuple[int, Path]],
    output: Path | None = None,
) -> dict[str, Any]:
    """Load bounded checkpoint metadata and persist its report-only decision."""

    config = load_config(config_path)
    _validate_contract(config)
    update1 = _load_json(update1_report_path, "V25 update-1 verification")
    for field, expected in {
        "schema_version": 1,
        "audit_type": "v25_dense_alignment_update1_verification",
        "match": True,
        "stage_2_authorized": True,
        "greedy_audit_authorized": False,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
    }.items():
        _equal(update1.get(field), expected, f"V25 update-1 report {field}")
    if len({epoch for epoch, _path in epoch_bindings}) != len(epoch_bindings):
        _fail("V25 epoch bindings contain duplicate epochs")
    metadata: dict[int, Mapping[str, Any]] = {}
    artifact_hashes: dict[str, str] = {}
    paths: dict[int, str] = {}
    for epoch, path in epoch_bindings:
        metadata_path = _regular_file(path, f"V25 epoch {epoch} metadata")
        metadata[epoch] = _load_json(metadata_path, f"V25 epoch {epoch} metadata")
        artifact_hashes[str(epoch)] = _file_sha256(metadata_path)
        paths[epoch] = str(path)
    report = select_epoch_metadata(metadata, config)
    report.update(
        {
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "update1_report_sha256": _file_sha256(
                _regular_file(update1_report_path, "V25 update-1 verification")
            ),
            "epoch_metadata_sha256": artifact_hashes,
            "selected_checkpoint": (
                None
                if report["selected_epoch"] is None
                else str(Path(paths[int(report["selected_epoch"])]).parent)
            ),
            "model_loaded": False,
            "oracle_loaded": False,
            "report_only": True,
            "question_dependent_scene_processing": False,
        }
    )
    if output is not None:
        destination = _write_json(report, output)
        report["output"] = str(destination)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, default=CONFIG_PATH)
    preflight.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate-preflight")
    validate.add_argument("--config", type=Path, default=CONFIG_PATH)
    validate.add_argument("--preflight", type=Path, required=True)
    calibration = subparsers.add_parser("verify-calibration")
    calibration.add_argument("--config", type=Path, default=CONFIG_PATH)
    calibration.add_argument("--calibration", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    update1 = subparsers.add_parser("verify-update1")
    update1.add_argument("--config", type=Path, default=CONFIG_PATH)
    update1.add_argument("--preflight", type=Path, required=True)
    update1.add_argument("--checkpoint", type=Path, required=True)
    update1.add_argument("--output", type=Path, required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--config", type=Path, default=CONFIG_PATH)
    select.add_argument("--update1-report", type=Path, required=True)
    select.add_argument(
        "--epoch",
        type=_parse_epoch_binding,
        action="append",
        required=True,
        dest="epoch_bindings",
    )
    select.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "preflight":
        report = run_preflight(args.config, args.output)
    elif args.command == "validate-preflight":
        report = validate_preflight(args.preflight, args.config)
    elif args.command == "verify-calibration":
        report = verify_calibration_report(
            config_path=args.config,
            calibration_path=args.calibration,
            output=args.output,
        )
    elif args.command == "verify-update1":
        report = verify_update1(
            config_path=args.config,
            preflight_path=args.preflight,
            checkpoint=args.checkpoint,
            output=args.output,
        )
    elif args.command == "select":
        report = select_epochs(
            config_path=args.config,
            update1_report_path=args.update1_report,
            epoch_bindings=args.epoch_bindings,
            output=args.output,
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_CONTRACT_SHA256",
    "V25ControlViolation",
    "run_preflight",
    "select_epoch_metadata",
    "select_epochs",
    "v25_contract",
    "validate_preflight",
    "verify_calibration_report",
    "verify_update1",
    "verify_update1_metadata",
]

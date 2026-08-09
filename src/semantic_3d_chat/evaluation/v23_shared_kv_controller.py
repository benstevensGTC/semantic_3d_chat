"""Fail-closed controller for the V23 frozen-scene shared-K/V screen.

V23 is intentionally not a continuation of V21's optimizer. It loads the
sealed V21 epoch-8 weights, freezes the complete learned scene stack and two
existing decoder banks, and adds one deterministic zero-output shared-K/V LoRA
bank. This controller binds that transition, stops after one real optimizer
update for structural inspection, and selects the bounded four-update screen.
It never reads oracle data and never authorizes greedy generation before the
complete teacher-forced gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.evaluation.v21_archive_validator import (
    EXPECTED_SUMMARY_SHA256 as V21_ARCHIVE_SHA256,
)
from semantic_3d_chat.evaluation.v21_archive_validator import validate_archive
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    install_lora_banks,
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
    tensor_state_sha256,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)
from semantic_3d_chat.training.train_adapter import (
    file_sha256,
    named_lora_extension_transition_mismatch,
)

CONFIG_PATH = Path("configs/experiments/gemma4_color_mirror_shared_kv_v23.yaml")
SOURCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v21_phase_aware_local_field_extension_u8/epoch_008"
)
PRIMARY_NAMESPACE = "gemma4_v23_shared_kv"
EXTENSION_NAMESPACE = "gemma4_v23_shared_kv_extension_u8"
NEW_BANK = "extension_v23_shared_kv"
EXPECTED_CONFIG_SHA256 = "5416ac62c7670cea067a92e6edfaadda450f9f78c3412b18209e1c63c578053e"
EXPECTED_CONTRACT_SHA256 = "a26ebc16efff574e15c61a541f8d0f68700da6bbb54654cfa90855e48c2f9fe4"
EXPECTED_SOURCE_ARTIFACTS = {
    "adapter_sha256": "ce9e97061389a7eae5703593d0a8869f87bd12544f56f5976570965056b65f44",
    "metadata_sha256": "bbc8309d25db86e40fa01ec744e19b3c0fc1c61953ebfc5072f11c84bbd2e997",
    "optimizer_sha256": "465a9075c7d890bc87caa94ecf2fe316750e714716fa39bb48cda72d80c9bf93",
}
EXPECTED_FROZEN_HASHES = {
    "scene": "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b",
    "global": "ce3bf864eed6dd4a50f1b67296981e144d6a79e9cf192ad9a9230f2ae18208dc",
    "signed_x": "e8dabc69627f60723b89520b02dfee985e49b7b7e35fdd1213cc79f7b8164f58",
    "inherited_v12": "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594",
    "extension_v13": "4eb90fb9b0bea579d14cfcb0f61ebd5b6d566fd600bd3d5e1bfe5177a39e1b34",
}
EXPECTED_NEW_BANK_INITIAL_SHA256 = (
    "707defddb599baf670ab3fec6594d8f8ccccd6b31689393c1c7ca30abaeed59d"
)
EXPECTED_TARGETS = (
    "model.language_model.layers.13.self_attn.k_proj",
    "model.language_model.layers.13.self_attn.v_proj",
    "model.language_model.layers.14.self_attn.k_proj",
    "model.language_model.layers.14.self_attn.v_proj",
)
EXPECTED_PARAMETER_SHAPES = (
    (4, 1536),
    (256, 4),
    (4, 1536),
    (256, 4),
    (4, 1536),
    (512, 4),
    (4, 1536),
    (512, 4),
)


class V23ControlViolation(ValueError):
    """A V23 authorization input, checkpoint, or outcome violated the contract."""


def _fail(message: str) -> None:
    raise V23ControlViolation(message)


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


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    if value != value.casefold():
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _load_json(path: Path, field: str) -> dict[str, Any]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if resolved.is_symlink() or not resolved.is_file():
        _fail(f"{field} is not a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot load {field}: {error}")
    return dict(_mapping(value, field))


def _regular_file(path: Path, field: str) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if resolved.is_symlink() or not resolved.is_file():
        _fail(f"{field} is not a regular non-symlink file: {resolved}")
    return resolved


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be a finite number")
    return result


def _clean_provenance(value: Any, field: str) -> dict[str, Any]:
    provenance = dict(_mapping(value, field))
    try:
        require_clean_committed_source(provenance)
    except RuntimeError as error:
        _fail(f"{field} is not a clean committed source: {error}")
    return provenance


def _write_json(value: Mapping[str, Any], path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


class _ShapeOnlyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(1, dtype=torch.float32).expand(out_features, in_features),
            requires_grad=False,
        )
        self.bias = None


class _Attention(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        if layer == 13:
            self.k_proj = _ShapeOnlyLinear(1536, 256)
            self.v_proj = _ShapeOnlyLinear(1536, 256)
        if layer == 14:
            self.k_proj = _ShapeOnlyLinear(1536, 512)
            self.v_proj = _ShapeOnlyLinear(1536, 512)
        if layer in (30, 31, 32, 33):
            self.q_proj = _ShapeOnlyLinear(1536, 2048)
            self.o_proj = _ShapeOnlyLinear(2048, 1536)
        if layer == 34:
            self.q_proj = _ShapeOnlyLinear(1536, 4096)
            self.o_proj = _ShapeOnlyLinear(4096, 1536)


class _Layer(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.self_attn = _Attention(layer)


class _ShapeOnlyGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(_Layer(layer) for layer in range(35))


def _install_shape_only(config: Mapping[str, Any]) -> LoRABankCollection:
    settings = lora_banks_settings(config)
    collection = install_lora_banks(_ShapeOnlyGemma().requires_grad_(False), settings)
    if not isinstance(collection, LoRABankCollection):
        _fail("V23 did not install a named LoRA bank collection")
    return collection


def v23_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    training = _mapping(config.get("training"), "training")
    screen = _mapping(config.get("v23_screen"), "v23_screen")
    experiment = _mapping(config.get("experiment"), "experiment")
    settings = lora_banks_settings(config)
    optimizer = lora_banks_optimizer_settings(config, settings)
    if optimizer is None:
        _fail("V23 requires an explicit LoRA optimizer")
    bank_contracts = {str(record["name"]): record for record in settings.contract()["banks"]}
    return {
        "schema_version": 1,
        "role": "frozen_v21_scene_stack_shared_kv_screen",
        "config_sha256": config_hash(dict(config), length=64),
        "source_archive_sha256": screen.get("source_archive_summary_sha256"),
        "source_checkpoint": str(training.get("initialize_from")),
        "source_artifact_hashes": {
            "adapter_sha256": training.get("initialize_expected_adapter_sha256"),
            "metadata_sha256": training.get("initialize_expected_metadata_sha256"),
        },
        "source_frozen_hashes": {
            "scene": experiment.get("source_scene_state_sha256"),
            "global": training.get("initialize_expected_global_scene_residual_state_sha256"),
            "signed_x": training.get("initialize_expected_signed_x_scene_residual_state_sha256"),
            "inherited_v12": experiment.get("source_inherited_bank_sha256"),
            "extension_v13": experiment.get("source_extension_bank_sha256"),
        },
        "new_bank": bank_contracts[NEW_BANK],
        "new_bank_parameter_count": experiment.get("decoder_trainable_parameter_count"),
        "optimizer": {
            **optimizer.contract(),
            "adamw": training.get("optimizer"),
            "gradient_accumulation": training.get("gradient_accumulation"),
        },
        "pair_objectives": training.get("pair_objectives"),
        "primary_namespace": screen.get("primary_output_namespace"),
        "extension_namespace": screen.get("extension_output_namespace"),
        "screen_optimizer_updates": screen.get("screen_optimizer_updates"),
        "conditional_max_optimizer_updates": screen.get("conditional_max_optimizer_updates"),
        "stage_1_optimizer_updates": screen.get("stage_1_optimizer_updates"),
        "stage_1_stop_required": screen.get("stage_1_stop_required"),
        "stage_2_requires": screen.get("stage_2_requires"),
        "eligibility_requires": screen.get("eligibility_requires"),
        "continuation_requires": screen.get("continuation_requires"),
        "full_teacher_gate_requires": screen.get("full_teacher_gate_requires"),
        "greedy_audit_only_after_full_teacher_gate": screen.get(
            "greedy_audit_only_after_full_teacher_gate"
        ),
        "question_dependent_scene_processing": experiment.get(
            "question_dependent_scene_processing"
        ),
        "question_dependent_retrieval": experiment.get("question_dependent_retrieval"),
        "runtime_oracle_access": experiment.get("runtime_oracle_access"),
    }


def _validate_contract(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    full_config_hash = config_hash(dict(config), length=64)
    _equal(full_config_hash, EXPECTED_CONFIG_SHA256, "resolved config SHA-256")
    contract = v23_contract(config)
    digest = _canonical_sha256(contract)
    _equal(digest, EXPECTED_CONTRACT_SHA256, "V23 normalized contract SHA-256")
    return contract, digest


def run_preflight(config_path: Path, output: Path) -> dict[str, Any]:
    config = load_config(config_path)
    contract, contract_digest = _validate_contract(config)
    archive = validate_archive(
        PROJECT_ROOT / "reports/gemma4/metrics/v21_final_summary.json",
        repo_root=PROJECT_ROOT,
        verify_bound_files=True,
    )
    _equal(archive.get("summary_sha256"), V21_ARCHIVE_SHA256, "V21 archive SHA-256")
    _equal(archive.get("selected_epoch"), 8, "V21 selected epoch")
    _equal(archive.get("decision"), "conditional_limit_reached_no_greedy_audit", "V21 outcome")

    source = PROJECT_ROOT / SOURCE_CHECKPOINT
    artifact_hashes = {
        "adapter_sha256": file_sha256(source / "adapter.safetensors"),
        "metadata_sha256": file_sha256(source / "metadata.json"),
        "optimizer_sha256": file_sha256(source / "optimizer.pt"),
    }
    _equal(artifact_hashes, EXPECTED_SOURCE_ARTIFACTS, "source checkpoint artifacts")
    metadata = _load_json(source / "metadata.json", "V21 source metadata")
    for field, expected in {
        "epoch": 8,
        "output_namespace": "gemma4_v21_phase_aware_local_field_extension_u8",
        "frozen_scene_state_sha256": EXPECTED_FROZEN_HASHES["scene"],
        "global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["signed_x"],
    }.items():
        _equal(metadata.get(field), expected, f"source metadata {field}")
    _equal(
        metadata.get("lora_bank_state_sha256"),
        {
            "inherited_v12": EXPECTED_FROZEN_HASHES["inherited_v12"],
            "extension_v13": EXPECTED_FROZEN_HASHES["extension_v13"],
        },
        "source frozen LoRA hashes",
    )

    collection = _install_shape_only(config)
    transition_mismatch = named_lora_extension_transition_mismatch(metadata, collection)
    if transition_mismatch is not None:
        _fail(f"source-to-V23 transition mismatch: {transition_mismatch}")
    bank = collection.bank(NEW_BANK)
    _equal(bank.settings.adapter.target_modules, EXPECTED_TARGETS, "new bank targets")
    _equal(bank.installation.parameter_count, 30_720, "new bank parameter count")
    _equal(
        bank.installation.state_sha256(),
        EXPECTED_NEW_BANK_INITIAL_SHA256,
        "new bank initial state",
    )
    parameters = list(bank.installation.parameters())
    _equal(
        [tuple(parameter.shape) for parameter in parameters],
        list(EXPECTED_PARAMETER_SHAPES),
        "parameter order",
    )
    if any(not parameter.requires_grad for parameter in parameters):
        _fail("one or more V23 bank parameters are not trainable")
    if any(torch.count_nonzero(adapter.lora_b).item() for adapter in bank.installation.adapters):
        _fail("V23 bank is not exact zero-output")

    provenance = capture_git_source_provenance(PROJECT_ROOT)
    provenance = _clean_provenance(provenance, "current source provenance")
    report = {
        "schema_version": 1,
        "audit_type": "v23_shared_kv_structural_preflight",
        "authorized": True,
        "stage_1_authorized": True,
        "runtime_eligible": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "optimizer_steps": 0,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "config_path": str(config_path),
        "config_sha256": config_hash(config, length=64),
        "contract": contract,
        "contract_sha256": contract_digest,
        "source_archive": archive,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_artifact_hashes": artifact_hashes,
        "source_metadata_sha256": artifact_hashes["metadata_sha256"],
        "source_provenance": provenance,
        "new_bank": {
            "name": NEW_BANK,
            "state_sha256": bank.installation.state_sha256(),
            "parameter_count": bank.installation.parameter_count,
            "target_modules": list(bank.installation.target_names),
            "ordered_parameter_shapes": [list(shape) for shape in EXPECTED_PARAMETER_SHAPES],
            "exact_zero_output": True,
        },
    }
    destination = _write_json(report, output)
    report["output"] = str(destination)
    return report


def _load_preflight(path: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    report = _load_json(path, "V23 preflight")
    preflight_file = _regular_file(path, "V23 preflight")
    digest = file_sha256(preflight_file)
    contract, contract_digest = _validate_contract(config)
    expected_keys = {
        "schema_version",
        "audit_type",
        "authorized",
        "stage_1_authorized",
        "runtime_eligible",
        "model_loaded",
        "optimizer_constructed",
        "optimizer_step_executed",
        "optimizer_steps",
        "oracle_loaded",
        "question_dependent_scene_processing",
        "config_path",
        "config_sha256",
        "contract",
        "contract_sha256",
        "source_archive",
        "source_checkpoint",
        "source_artifact_hashes",
        "source_metadata_sha256",
        "source_provenance",
        "new_bank",
    }
    _equal(set(report), expected_keys, "preflight root keys")
    for field, expected in {
        "schema_version": 1,
        "audit_type": "v23_shared_kv_structural_preflight",
        "authorized": True,
        "stage_1_authorized": True,
        "runtime_eligible": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "optimizer_steps": 0,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "config_path": str(CONFIG_PATH),
    }.items():
        _equal(report.get(field), expected, f"preflight {field}")
    _equal(report.get("config_sha256"), EXPECTED_CONFIG_SHA256, "preflight config hash")
    _equal(report.get("contract"), contract, "preflight contract")
    _equal(report.get("contract_sha256"), contract_digest, "preflight contract hash")
    _equal(report.get("source_checkpoint"), str(SOURCE_CHECKPOINT), "preflight source checkpoint")
    _equal(
        report.get("source_artifact_hashes"),
        EXPECTED_SOURCE_ARTIFACTS,
        "preflight source artifact hashes",
    )
    _equal(
        report.get("source_metadata_sha256"),
        EXPECTED_SOURCE_ARTIFACTS["metadata_sha256"],
        "preflight source metadata hash",
    )
    archive = _mapping(report.get("source_archive"), "preflight source archive")
    _equal(archive.get("summary_sha256"), V21_ARCHIVE_SHA256, "preflight V21 archive")
    _equal(archive.get("selected_epoch"), 8, "preflight V21 selected epoch")
    _equal(
        archive.get("decision"),
        "conditional_limit_reached_no_greedy_audit",
        "preflight V21 decision",
    )
    _equal(
        report.get("new_bank"),
        {
            "name": NEW_BANK,
            "state_sha256": EXPECTED_NEW_BANK_INITIAL_SHA256,
            "parameter_count": 30_720,
            "target_modules": list(EXPECTED_TARGETS),
            "ordered_parameter_shapes": [list(shape) for shape in EXPECTED_PARAMETER_SHAPES],
            "exact_zero_output": True,
        },
        "preflight new bank",
    )
    _clean_provenance(report.get("source_provenance"), "preflight source provenance")
    return report, digest


def _adapter_payload(adapter_path: Path) -> dict[str, Any]:
    safe_adapter = _regular_file(adapter_path, "V23 adapter safetensors")
    scene_prefixes = ("scene_model.", "composer.", "grounding.")
    global_prefix = "global_scene_residual."
    signed_x_prefix = "signed_x_scene_residual."
    bank_prefixes = {
        "inherited_v12": "lora_banks.inherited_v12.",
        "extension_v13": "lora_banks.extension_v13.",
        NEW_BANK: f"lora_banks.{NEW_BANK}.",
    }
    scene_state: dict[str, torch.Tensor] = {}
    global_state: dict[str, torch.Tensor] = {}
    signed_x_state: dict[str, torch.Tensor] = {}
    bank_states: dict[str, dict[str, torch.Tensor]] = {name: {} for name in bank_prefixes}
    unknown: list[str] = []
    with safe_open(safe_adapter, framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118 - safe_open itself is not iterable
            tensor = handle.get_tensor(key)
            if key.startswith(scene_prefixes):
                scene_state[key] = tensor
            elif key.startswith(global_prefix):
                global_state[key] = tensor
            elif key.startswith(signed_x_prefix):
                signed_x_state[key] = tensor
            else:
                for bank_name, prefix in bank_prefixes.items():
                    if key.startswith(prefix):
                        bank_states[bank_name][key.removeprefix(prefix)] = tensor
                        break
                else:
                    unknown.append(key)
    if unknown:
        _fail(f"V23 adapter contains unknown tensor keys: {unknown}")
    if not scene_state or not global_state or not signed_x_state:
        _fail("V23 adapter is missing a frozen scene/residual tensor group")
    if any(not state for state in bank_states.values()):
        _fail("V23 adapter is missing one or more LoRA bank tensor groups")
    state = bank_states[NEW_BANK]
    expected_keys = {
        f"adapters.{index}.{suffix}" for index in range(4) for suffix in ("lora_a", "lora_b")
    }
    _equal(set(state), expected_keys, "V23 safetensors bank keys")
    return {
        "scene_state_sha256": tensor_state_sha256(scene_state),
        "global_scene_residual_state_sha256": tensor_state_sha256(global_state),
        "signed_x_scene_residual_state_sha256": tensor_state_sha256(signed_x_state),
        "lora_bank_state_sha256": {
            name: tensor_state_sha256(bank_state) for name, bank_state in bank_states.items()
        },
        "new_bank_state": state,
        "tensor_count": (
            len(scene_state)
            + len(global_state)
            + len(signed_x_state)
            + sum(len(bank_state) for bank_state in bank_states.values())
        ),
    }


def _require_frozen_bank_pins(
    bank_hashes: Mapping[str, Any],
    *,
    field: str,
) -> None:
    """Pin both inherited banks independently of checkpoint metadata aliases."""

    _equal(
        bank_hashes.get("inherited_v12"),
        EXPECTED_FROZEN_HASHES["inherited_v12"],
        f"{field} inherited_v12",
    )
    _equal(
        bank_hashes.get("extension_v13"),
        EXPECTED_FROZEN_HASHES["extension_v13"],
        f"{field} extension_v13",
    )


def _require_new_bank_tensor_contract(state: Mapping[str, torch.Tensor], *, field: str) -> None:
    """Validate every trainable tensor's exact key, shape, dtype, and finiteness."""

    expected_keys = {
        f"adapters.{index}.{suffix}" for index in range(4) for suffix in ("lora_a", "lora_b")
    }
    _equal(set(state), expected_keys, f"{field} tensor keys")
    ordered_keys = [
        f"adapters.{index}.{suffix}"
        for index in range(4)
        for suffix in ("lora_a", "lora_b")
    ]
    for key, expected_shape in zip(ordered_keys, EXPECTED_PARAMETER_SHAPES, strict=True):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            _fail(f"{field} {key} is not a tensor")
        _equal(tuple(tensor.shape), expected_shape, f"{field} {key} shape")
        _equal(tensor.dtype, torch.float32, f"{field} {key} dtype")
        if not bool(torch.isfinite(tensor).all()):
            _fail(f"{field} {key} is non-finite")


def _optimizer_manifest(path: Path, *, expected_step: int = 1) -> dict[str, Any]:
    safe_optimizer = _regular_file(path, "V23 optimizer state")
    try:
        state_dict = torch.load(safe_optimizer, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        _fail(f"cannot safely load V23 optimizer state: {error}")
    root = _mapping(state_dict, "optimizer root")
    _equal(set(root), {"state", "param_groups"}, "optimizer root keys")
    groups = _sequence(root.get("param_groups"), "optimizer param_groups")
    _equal(len(groups), 1, "optimizer group count")
    group = dict(_mapping(groups[0], "optimizer group"))
    expected_group = {
        "name": "language_lora",
        "lr": 0.0003,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "decoupled_weight_decay": True,
        "params": list(range(8)),
    }
    _equal(group, expected_group, "optimizer group contract")
    states = _mapping(root.get("state"), "optimizer state")
    _equal(set(states), set(range(8)), "optimizer state IDs")
    tensor_manifest: list[dict[str, Any]] = []
    aggregate: dict[str, torch.Tensor] = {}
    for parameter_id, expected_shape in enumerate(EXPECTED_PARAMETER_SHAPES):
        state = _mapping(states[parameter_id], f"optimizer state {parameter_id}")
        _equal(
            set(state), {"step", "exp_avg", "exp_avg_sq"}, f"optimizer state {parameter_id} keys"
        )
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        for name, tensor in (("step", step), ("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
                _fail(f"optimizer state {parameter_id}.{name} is not an FP32 tensor")
            if not bool(torch.isfinite(tensor).all()):
                _fail(f"optimizer state {parameter_id}.{name} is non-finite")
            aggregate[f"{parameter_id}.{name}"] = tensor
        _equal(tuple(step.shape), (), f"optimizer state {parameter_id}.step shape")
        _equal(float(step), float(expected_step), f"optimizer state {parameter_id}.step")
        _equal(
            tuple(exp_avg.shape), expected_shape, f"optimizer state {parameter_id}.exp_avg shape"
        )
        _equal(
            tuple(exp_avg_sq.shape),
            expected_shape,
            f"optimizer state {parameter_id}.exp_avg_sq shape",
        )
        is_a = parameter_id % 2 == 0
        if bool(torch.lt(exp_avg_sq, 0).any()):
            _fail(f"optimizer state {parameter_id}.exp_avg_sq contains a negative value")
        if is_a and expected_step == 1:
            if torch.count_nonzero(exp_avg).item() or torch.count_nonzero(exp_avg_sq).item():
                _fail(f"LoRA-A optimizer moments are nonzero at parameter {parameter_id}")
        elif not is_a and (
            not torch.count_nonzero(exp_avg).item() or not torch.count_nonzero(exp_avg_sq).item()
        ):
            _fail(f"LoRA-B optimizer moments are zero at parameter {parameter_id}")
        tensor_manifest.append(
            {
                "parameter_id": parameter_id,
                "role": "A" if is_a else "B",
                "shape": list(expected_shape),
                "step": float(step),
                "exp_avg_nonzero": int(torch.count_nonzero(exp_avg)),
                "exp_avg_sq_nonzero": int(torch.count_nonzero(exp_avg_sq)),
                "state_sha256": tensor_state_sha256(
                    {"step": step, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
                ),
            }
        )
    serialized_group = dict(group)
    serialized_group["betas"] = list(group["betas"])
    return {
        "optimizer": "AdamW",
        "expected_step": expected_step,
        "group": serialized_group,
        "parameter_states": tensor_manifest,
        "all_state_tensors_sha256": tensor_state_sha256(aggregate),
    }


def _exact_accuracy_count(value: Any, denominator: int, field: str) -> int:
    accuracy = _finite_float(value, field)
    if not 0.0 <= accuracy <= 1.0:
        _fail(f"{field} must be a finite probability")
    scaled = accuracy * denominator
    count = round(scaled)
    # The training audit persists an FP32 mean, so allow only its expected
    # representation error around an integer count. This is still far tighter
    # than a near-perfect but non-empirical value such as 0.999.
    if not math.isclose(scaled, count, rel_tol=0.0, abs_tol=1e-5):
        _fail(f"{field} is not an exact {denominator}-way empirical fraction: {accuracy}")
    return count


def _pair_metrics(metadata: Mapping[str, Any], pair_id: str) -> dict[str, Any]:
    gate = _mapping(metadata.get("pair_candidate_gate"), "pair_candidate_gate")
    by_pair = _mapping(gate.get("by_pair"), "pair_candidate_gate.by_pair")
    pair = _mapping(by_pair.get(pair_id), f"pair gate {pair_id}")
    return {
        "full_vocab_sides": _exact_accuracy_count(
            pair.get("first_answer_token_top1_accuracy"),
            12,
            f"{pair_id} side accuracy",
        ),
        "full_vocab_units": _exact_accuracy_count(
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


def _color_eligible(metrics: Mapping[str, Any]) -> bool:
    return (
        metrics.get("full_vocab_sides") == 12
        and metrics.get("full_vocab_units") == 6
        and float(metrics.get("minimum_candidate_margin")) > 0.0
        and float(metrics.get("minimum_full_vocab_margin")) > 0.0
    )


def _mirror_continuation(metrics: Mapping[str, Any]) -> bool:
    return metrics.get("full_vocab_sides", 0) >= 8 and metrics.get("full_vocab_units", 0) >= 2


def _full_pair(metrics: Mapping[str, Any]) -> bool:
    return (
        metrics.get("full_vocab_sides") == 12
        and metrics.get("full_vocab_units") == 6
        and float(metrics.get("minimum_candidate_margin")) > 0.0
        and float(metrics.get("minimum_full_vocab_margin")) > 0.0
    )


def verify_update1(
    config_path: Path,
    preflight_path: Path,
    checkpoint_path: Path,
    output: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    preflight, preflight_sha256 = _load_preflight(preflight_path, config)
    current_provenance = _clean_provenance(
        capture_git_source_provenance(PROJECT_ROOT),
        "current source provenance",
    )
    _equal(
        current_provenance,
        preflight.get("source_provenance"),
        "current/preflight source provenance",
    )
    checkpoint = (
        checkpoint_path if checkpoint_path.is_absolute() else PROJECT_ROOT / checkpoint_path
    )
    expected_checkpoint = PROJECT_ROOT / "data_gemma4/checkpoints" / PRIMARY_NAMESPACE / "epoch_001"
    _equal(checkpoint.resolve(), expected_checkpoint.resolve(), "update-1 checkpoint path")
    adapter_path = _regular_file(checkpoint / "adapter.safetensors", "update-1 adapter")
    metadata_path = _regular_file(checkpoint / "metadata.json", "update-1 metadata")
    optimizer_path = _regular_file(checkpoint / "optimizer.pt", "update-1 optimizer")
    artifacts = {
        "adapter_sha256": file_sha256(adapter_path),
        "metadata_sha256": file_sha256(metadata_path),
        "optimizer_sha256": file_sha256(optimizer_path),
    }
    metadata = _load_json(metadata_path, "V23 update-1 metadata")
    for field, expected in {
        "epoch": 1,
        "global_step": 12,
        "optimizer_step": 1,
        "output_namespace": PRIMARY_NAMESPACE,
        "config_hash": config_hash(config),
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": False,
        "train_lora_with_frozen_scene_residual_stack": True,
        "frozen_scene_state_sha256": EXPECTED_FROZEN_HASHES["scene"],
        "frozen_global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "frozen_signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["signed_x"],
        "global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["signed_x"],
    }.items():
        _equal(metadata.get(field), expected, f"update-1 metadata {field}")
    _equal(
        metadata.get("source_provenance"), preflight.get("source_provenance"), "source provenance"
    )
    _equal(
        metadata.get("frozen_lora_bank_state_sha256"),
        {
            "inherited_v12": EXPECTED_FROZEN_HASHES["inherited_v12"],
            "extension_v13": EXPECTED_FROZEN_HASHES["extension_v13"],
        },
        "frozen LoRA hashes",
    )
    initialization = _mapping(
        metadata.get("initialization_provenance"), "initialization provenance"
    )
    for field, expected in {
        "schema_version": 5,
        "mode": "frozen_scene_residual_stack_plus_zero_output_named_lora_extension",
        "checkpoint": str(SOURCE_CHECKPOINT),
        "adapter_sha256": EXPECTED_SOURCE_ARTIFACTS["adapter_sha256"],
        "metadata_sha256": EXPECTED_SOURCE_ARTIFACTS["metadata_sha256"],
        "optimizer_state_loaded": False,
        "history_loaded": False,
        "all_source_scene_residuals_frozen": True,
        "new_trainable_lora_banks_zero_output": True,
        "source_scene_state_sha256": EXPECTED_FROZEN_HASHES["scene"],
        "expected_source_scene_state_sha256": EXPECTED_FROZEN_HASHES["scene"],
        "source_global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "expected_source_global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "source_signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["signed_x"],
        "expected_source_signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["signed_x"],
    }.items():
        _equal(initialization.get(field), expected, f"initialization provenance {field}")
    _equal(
        initialization.get("source_lora_bank_state_sha256"),
        {
            "inherited_v12": EXPECTED_FROZEN_HASHES["inherited_v12"],
            "extension_v13": EXPECTED_FROZEN_HASHES["extension_v13"],
        },
        "initialization frozen LoRA hashes",
    )

    collection = _install_shape_only(config)
    settings = lora_banks_settings(config)
    optimizer_settings = lora_banks_optimizer_settings(config, settings)
    assert optimizer_settings is not None
    expected_lora = lora_banks_checkpoint_contract(
        settings,
        optimizer_settings,
        collection.parameter_counts,
    )
    _equal(metadata.get("lora"), expected_lora, "checkpoint LoRA contract")
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "LoRA bank hashes")
    _equal(set(bank_hashes), {"inherited_v12", "extension_v13", NEW_BANK}, "LoRA bank hash keys")
    _equal(
        bank_hashes.get("inherited_v12"), EXPECTED_FROZEN_HASHES["inherited_v12"], "inherited bank"
    )
    _equal(
        bank_hashes.get("extension_v13"), EXPECTED_FROZEN_HASHES["extension_v13"], "extension bank"
    )
    if bank_hashes.get(NEW_BANK) == EXPECTED_NEW_BANK_INITIAL_SHA256:
        _fail("V23 update-1 bank remained at its initial state")

    payload = _adapter_payload(adapter_path)
    _equal(
        payload["scene_state_sha256"],
        EXPECTED_FROZEN_HASHES["scene"],
        "recomputed frozen scene state",
    )
    _equal(
        payload["global_scene_residual_state_sha256"],
        EXPECTED_FROZEN_HASHES["global"],
        "recomputed frozen global residual state",
    )
    _equal(
        payload["signed_x_scene_residual_state_sha256"],
        EXPECTED_FROZEN_HASHES["signed_x"],
        "recomputed frozen signed-X state",
    )
    _equal(payload["lora_bank_state_sha256"], dict(bank_hashes), "recomputed LoRA bank states")
    _require_frozen_bank_pins(
        payload["lora_bank_state_sha256"],
        field="update-1 recomputed frozen bank",
    )
    checkpoint_state = payload["new_bank_state"]
    _require_new_bank_tensor_contract(checkpoint_state, field="update-1 new bank")
    _equal(tensor_state_sha256(checkpoint_state), bank_hashes.get(NEW_BANK), "new bank state hash")
    initial_state = collection.bank(NEW_BANK).installation.state_module.state_dict()
    for index in range(4):
        a_key = f"adapters.{index}.lora_a"
        b_key = f"adapters.{index}.lora_b"
        if not torch.equal(checkpoint_state[a_key], initial_state[a_key]):
            _fail(f"LoRA-A tensor changed at target index {index}")
        if not torch.count_nonzero(checkpoint_state[b_key]).item():
            _fail(f"LoRA-B tensor remained zero at target index {index}")
        if not bool(torch.isfinite(checkpoint_state[b_key]).all()):
            _fail(f"LoRA-B tensor is non-finite at target index {index}")
    optimizer_manifest = _optimizer_manifest(optimizer_path, expected_step=1)

    color = _pair_metrics(metadata, "pair_000001")
    mirror = _pair_metrics(metadata, "pair_000003")
    stage_2_authorized = _color_eligible(color) and _mirror_continuation(mirror)
    report = {
        "schema_version": 1,
        "audit_type": "v23_shared_kv_update1_verifier",
        "match": True,
        "stage_2_authorized": stage_2_authorized,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "preflight_sha256": preflight_sha256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_artifact_hashes": artifacts,
        "new_bank_state_sha256": bank_hashes[NEW_BANK],
        "ordered_parameter_shapes": [list(shape) for shape in EXPECTED_PARAMETER_SHAPES],
        "a_tensors_unchanged": True,
        "b_tensors_all_changed": True,
        "optimizer_manifest": optimizer_manifest,
        "recomputed_payload_hashes": {
            key: value for key, value in payload.items() if key != "new_bank_state"
        },
        "color": color,
        "mirror": mirror,
        "source_provenance": metadata["source_provenance"],
    }
    destination = _write_json(report, output)
    report["output"] = str(destination)
    return report


def _epoch_record(
    config: Mapping[str, Any],
    epoch: int,
    path: Path,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_path = path if path.is_absolute() else PROJECT_ROOT / path
    expected_metadata = (
        PROJECT_ROOT
        / "data_gemma4/checkpoints"
        / PRIMARY_NAMESPACE
        / f"epoch_{epoch:03d}"
        / "metadata.json"
    )
    _equal(
        metadata_path.resolve(),
        expected_metadata.resolve(),
        f"epoch {epoch} canonical metadata path",
    )
    metadata_path = _regular_file(metadata_path, f"V23 epoch {epoch} metadata")
    metadata = _load_json(metadata_path, f"V23 epoch {epoch} metadata")
    for field, expected in {
        "epoch": epoch,
        "global_step": epoch * 12,
        "optimizer_step": epoch,
        "output_namespace": PRIMARY_NAMESPACE,
        "config_hash": config_hash(dict(config)),
        "frozen_scene_state_sha256": EXPECTED_FROZEN_HASHES["scene"],
        "frozen_global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "frozen_signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["signed_x"],
        "global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["signed_x"],
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": False,
        "train_lora_with_frozen_scene_residual_stack": True,
    }.items():
        _equal(metadata.get(field), expected, f"epoch {epoch} {field}")
    _equal(metadata.get("source_provenance"), source_provenance, f"epoch {epoch} source provenance")
    frozen_banks = metadata.get("frozen_lora_bank_state_sha256")
    _equal(
        frozen_banks,
        {
            "inherited_v12": EXPECTED_FROZEN_HASHES["inherited_v12"],
            "extension_v13": EXPECTED_FROZEN_HASHES["extension_v13"],
        },
        f"epoch {epoch} frozen banks",
    )
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), f"epoch {epoch} bank hashes")
    _equal(
        set(bank_hashes), {"inherited_v12", "extension_v13", NEW_BANK}, f"epoch {epoch} bank keys"
    )
    _require_frozen_bank_pins(bank_hashes, field=f"epoch {epoch} metadata frozen bank")
    collection = _install_shape_only(config)
    settings = lora_banks_settings(config)
    optimizer_settings = lora_banks_optimizer_settings(config, settings)
    assert optimizer_settings is not None
    _equal(
        metadata.get("lora"),
        lora_banks_checkpoint_contract(settings, optimizer_settings, collection.parameter_counts),
        f"epoch {epoch} LoRA contract",
    )
    checkpoint = metadata_path.parent
    adapter_path = _regular_file(checkpoint / "adapter.safetensors", f"epoch {epoch} adapter")
    optimizer_path = _regular_file(checkpoint / "optimizer.pt", f"epoch {epoch} optimizer")
    payload = _adapter_payload(adapter_path)
    _equal(
        payload["scene_state_sha256"],
        EXPECTED_FROZEN_HASHES["scene"],
        f"epoch {epoch} recomputed frozen scene",
    )
    _equal(
        payload["global_scene_residual_state_sha256"],
        EXPECTED_FROZEN_HASHES["global"],
        f"epoch {epoch} recomputed frozen global residual",
    )
    _equal(
        payload["signed_x_scene_residual_state_sha256"],
        EXPECTED_FROZEN_HASHES["signed_x"],
        f"epoch {epoch} recomputed frozen signed-X residual",
    )
    _equal(
        payload["lora_bank_state_sha256"],
        dict(bank_hashes),
        f"epoch {epoch} recomputed LoRA banks",
    )
    _require_frozen_bank_pins(
        payload["lora_bank_state_sha256"],
        field=f"epoch {epoch} recomputed frozen bank",
    )
    _require_new_bank_tensor_contract(
        payload["new_bank_state"],
        field=f"epoch {epoch} new bank",
    )
    optimizer_manifest = _optimizer_manifest(optimizer_path, expected_step=epoch)
    return {
        "epoch": epoch,
        "optimizer_step": epoch,
        "cumulative_microsteps": epoch * 12,
        "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
        "metadata_sha256": file_sha256(metadata_path),
        "adapter_sha256": file_sha256(adapter_path),
        "optimizer_sha256": file_sha256(optimizer_path),
        "new_bank_state_sha256": bank_hashes[NEW_BANK],
        "recomputed_payload_hashes": {
            key: value for key, value in payload.items() if key != "new_bank_state"
        },
        "optimizer_manifest": optimizer_manifest,
        "color": _pair_metrics(metadata, "pair_000001"),
        "mirror": _pair_metrics(metadata, "pair_000003"),
    }


def select_epochs(
    config_path: Path,
    update1_path: Path,
    epoch_bindings: Mapping[int, Path],
    output: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_contract(config)
    update1 = _load_json(update1_path, "V23 update-1 verifier")
    expected_update1_keys = {
        "schema_version",
        "audit_type",
        "match",
        "stage_2_authorized",
        "report_only",
        "model_loaded",
        "oracle_loaded",
        "preflight_sha256",
        "config_sha256",
        "contract_sha256",
        "checkpoint",
        "checkpoint_artifact_hashes",
        "new_bank_state_sha256",
        "ordered_parameter_shapes",
        "a_tensors_unchanged",
        "b_tensors_all_changed",
        "optimizer_manifest",
        "recomputed_payload_hashes",
        "color",
        "mirror",
        "source_provenance",
    }
    _equal(set(update1), expected_update1_keys, "update-1 report root keys")
    for field, expected in {
        "schema_version": 1,
        "audit_type": "v23_shared_kv_update1_verifier",
        "match": True,
        "stage_2_authorized": True,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "checkpoint": f"data_gemma4/checkpoints/{PRIMARY_NAMESPACE}/epoch_001",
        "ordered_parameter_shapes": [list(shape) for shape in EXPECTED_PARAMETER_SHAPES],
        "a_tensors_unchanged": True,
        "b_tensors_all_changed": True,
    }.items():
        _equal(update1.get(field), expected, f"update-1 {field}")
    _sha256(update1.get("preflight_sha256"), "update-1 preflight hash")
    _sha256(update1.get("new_bank_state_sha256"), "update-1 bank hash")
    source_provenance = _clean_provenance(
        update1.get("source_provenance"), "update-1 source provenance"
    )
    current_provenance = _clean_provenance(
        capture_git_source_provenance(PROJECT_ROOT), "current source provenance"
    )
    _equal(current_provenance, source_provenance, "current/update-1 source provenance")
    update1_color = dict(_mapping(update1.get("color"), "update-1 color metrics"))
    update1_mirror = dict(_mapping(update1.get("mirror"), "update-1 mirror metrics"))
    _equal(
        _color_eligible(update1_color) and _mirror_continuation(update1_mirror),
        True,
        "recomputed update-1 stage-2 gate",
    )
    _equal(update1.get("contract_sha256"), EXPECTED_CONTRACT_SHA256, "update-1 contract")
    expected_epochs = set(range(1, 5))
    _equal(set(epoch_bindings), expected_epochs, "screen epoch bindings")
    epochs = [
        _epoch_record(config, epoch, epoch_bindings[epoch], source_provenance)
        for epoch in sorted(epoch_bindings)
    ]
    epoch1 = epochs[0]
    _equal(
        update1.get("checkpoint_artifact_hashes"),
        {
            "adapter_sha256": epoch1["adapter_sha256"],
            "metadata_sha256": epoch1["metadata_sha256"],
            "optimizer_sha256": epoch1["optimizer_sha256"],
        },
        "update-1 checkpoint artifact binding",
    )
    _equal(
        update1.get("new_bank_state_sha256"),
        epoch1["new_bank_state_sha256"],
        "update-1 bank/epoch-1 binding",
    )
    _equal(
        update1.get("optimizer_manifest"),
        epoch1["optimizer_manifest"],
        "update-1 optimizer/epoch-1 binding",
    )
    _equal(
        update1.get("recomputed_payload_hashes"),
        epoch1["recomputed_payload_hashes"],
        "update-1 tensor/epoch-1 binding",
    )
    _equal(update1_color, epoch1["color"], "update-1 color/epoch-1 binding")
    _equal(update1_mirror, epoch1["mirror"], "update-1 mirror/epoch-1 binding")
    eligible = [row for row in epochs if _color_eligible(row["color"])]
    selected = (
        None
        if not eligible
        else max(
            eligible,
            key=lambda row: (
                row["mirror"]["full_vocab_units"],
                row["mirror"]["full_vocab_sides"],
                row["mirror"]["mean_full_vocab_margin"],
                row["mirror"]["minimum_full_vocab_margin"],
                row["mirror"]["mean_candidate_margin"],
                row["mirror"]["minimum_candidate_margin"],
                -row["epoch"],
            ),
        )
    )
    full_teacher = bool(
        selected is not None and _full_pair(selected["color"]) and _full_pair(selected["mirror"])
    )
    continuation = bool(
        selected is not None and not full_teacher and _mirror_continuation(selected["mirror"])
    )
    greedy = full_teacher
    if full_teacher:
        decision = "full_teacher_gate_passed_greedy_audit_authorized"
    elif continuation:
        decision = "screen_passed_extension_authorized_no_greedy_audit"
    else:
        decision = "screen_failed_no_extension_no_greedy_audit"
    report = {
        "schema_version": 1,
        "audit_type": "v23_shared_kv_epoch_selector",
        "decision": decision,
        "selected_epoch": None if selected is None else selected["epoch"],
        "selected_checkpoint": (
            None if selected is None else str(Path(selected["metadata_path"]).parent)
        ),
        "continuation_authorized": continuation,
        "full_teacher_gate_passed": full_teacher,
        "greedy_audit_authorized": greedy,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "report_only": True,
        "question_dependent_scene_processing": False,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "update1_report_sha256": file_sha256(
            update1_path if update1_path.is_absolute() else PROJECT_ROOT / update1_path
        ),
        "epochs": epochs,
        "selection_policy": {
            "eligibility": "complete color pair with positive candidate/full-vocabulary minima",
            "ranking_descending": [
                "mirror_full_vocab_units",
                "mirror_full_vocab_sides",
                "mirror_mean_full_vocab_margin",
                "mirror_minimum_full_vocab_margin",
                "mirror_mean_candidate_margin",
                "mirror_minimum_candidate_margin",
            ],
            "tie_breaker": "earlier_epoch",
            "extension_requires": "selected mirror >=8/12 sides and >=2/6 units",
            "greedy_requires": "both pairs 12/12 sides, 6/6 units, all minima positive",
        },
    }
    destination = _write_json(report, output)
    report["output"] = str(destination)
    return report


def _parse_epoch(value: str) -> tuple[int, Path]:
    raw_epoch, separator, raw_path = value.partition("=")
    if not separator or not raw_path:
        raise argparse.ArgumentTypeError("epoch binding must be EPOCH=PATH")
    try:
        epoch = int(raw_epoch)
    except ValueError as error:
        raise argparse.ArgumentTypeError("epoch binding must be EPOCH=PATH") from error
    return epoch, Path(raw_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, default=CONFIG_PATH)
    preflight.add_argument("--output", type=Path, required=True)
    update1 = subparsers.add_parser("verify-update1")
    update1.add_argument("--config", type=Path, default=CONFIG_PATH)
    update1.add_argument("--preflight", type=Path, required=True)
    update1.add_argument("--checkpoint", type=Path, required=True)
    update1.add_argument("--output", type=Path, required=True)
    selector = subparsers.add_parser("select")
    selector.add_argument("--config", type=Path, default=CONFIG_PATH)
    selector.add_argument("--update1-report", type=Path, required=True)
    selector.add_argument("--epoch", action="append", type=_parse_epoch, required=True)
    selector.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args.config, args.output)
    elif args.command == "verify-update1":
        result = verify_update1(args.config, args.preflight, args.checkpoint, args.output)
    else:
        bindings = dict(args.epoch)
        if len(bindings) != len(args.epoch):
            parser.error("duplicate epoch binding")
        result = select_epochs(args.config, args.update1_report, bindings, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_CONTRACT_SHA256",
    "V23ControlViolation",
    "run_preflight",
    "select_epochs",
    "v23_contract",
    "verify_update1",
]

"""Fail-closed controller for the V24 frozen-V23 shared-query screen.

V24 starts from the sealed V23 epoch-2 checkpoint, freezes the complete scene
stack and every inherited decoder bank, and adds one deterministic zero-output
LoRA bank to the real query projections at physical layers 28 and 29.  This
module performs only structural/report inspection: it never loads Gemma, runs
inference, reads oracle data, or authorizes downstream generation before the
complete preregistered teacher-forced gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import transformers
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.data.dataset import SceneQADataset
from semantic_3d_chat.evaluation.gemma4_semantic_sanity import resolve_local_snapshot
from semantic_3d_chat.evaluation.v23_archive_validator import (
    EXPECTED_SUMMARY_SHA256 as V23_ARCHIVE_SHA256,
)
from semantic_3d_chat.evaluation.v23_archive_validator import validate_archive
from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    install_lora_banks,
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
    tensor_state_sha256,
)
from semantic_3d_chat.training.pair_curriculum import (
    build_exact_question_pair_units,
    cap_pair_units_per_pair,
    pair_curriculum_settings,
    select_pair_only_records,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)
from semantic_3d_chat.training.train_adapter import (
    file_sha256,
    named_lora_freeze_and_extend_transition_mismatch,
    select_training_records,
    tokenize_answer,
)

CONFIG_PATH = Path("configs/experiments/gemma4_color_mirror_shared_query_v24.yaml")
SOURCE_ARCHIVE = Path("reports/gemma4/metrics/v23_final_summary.json")
SOURCE_CHECKPOINT = Path("data_gemma4/checkpoints/gemma4_v23_shared_kv/epoch_002")
SOURCE_SELECTION = Path("reports/gemma4/metrics/training_selection_gemma4_v23_shared_kv.json")
PRIMARY_NAMESPACE = "gemma4_v24_shared_query"
EXTENSION_NAMESPACE = "gemma4_v24_shared_query_extension_u8"
NEW_BANK = "extension_v24_shared_query"

# These are replaced only when the resolved, reviewed V24 configuration is
# finalized.  Keeping them explicit makes any config drift fail before model
# loading or optimizer construction.
EXPECTED_CONFIG_SHA256 = "82d5fee205842fb86133498eb4ac7765e61c22e7e7bc2745cfa6a2e36b9447f1"
EXPECTED_CONTRACT_SHA256 = "3922eaed356dffa9a46ee601135cceb3e5a68e81e459805c8ddb8664a4c8a996"
EXPECTED_NEW_BANK_INITIAL_SHA256 = (
    "e8734db171db6bd47a9a4f8c9d4a540903cc214a88abaab74820d566ee245f6b"
)

EXPECTED_SOURCE_ARTIFACTS = {
    "adapter_sha256": "dba2511db49fa46af905b293fc999642286f8533fa1d4cca2c872ffda2980ea8",
    "metadata_sha256": "1c0436549e832c2ac9723e2556ad8bf09862020c6cda47db8358b2232b391ba0",
    "optimizer_sha256": "08c3618d765346a018e78e4a608361d29f1d0c88cf01d5c2af26e0b20c9a3daa",
}
EXPECTED_FROZEN_HASHES = {
    "scene": "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b",
    "global": "ce3bf864eed6dd4a50f1b67296981e144d6a79e9cf192ad9a9230f2ae18208dc",
    "signed_x": "e8dabc69627f60723b89520b02dfee985e49b7b7e35fdd1213cc79f7b8164f58",
    "inherited_v12": "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594",
    "extension_v13": "4eb90fb9b0bea579d14cfcb0f61ebd5b6d566fd600bd3d5e1bfe5177a39e1b34",
    "extension_v23_shared_kv": ("91a9eea577cab5a37e840cdf4007722a398415846af91280713bcb2cda0f045c"),
}
EXPECTED_TARGETS = (
    "model.language_model.layers.28.self_attn.q_proj",
    "model.language_model.layers.29.self_attn.q_proj",
)
EXPECTED_PARAMETER_SHAPES = (
    (4, 1536),
    (2048, 4),
    (4, 1536),
    (4096, 4),
)
FROZEN_BANKS = ("inherited_v12", "extension_v13", "extension_v23_shared_kv")
EXPECTED_SOURCE_SELECTION_SHA256 = (
    "64c6a17eec8b17a7a46aedc70ee15e59825d9c1f168e255f92f7bc8d03aee6b5"
)
EXPECTED_PAIR_UNIT_SELECTION_SHA256 = (
    "d5928cb783339ef62fff5c14a8c7f85f90d3a7a6cb8edad0a784998082740d3e"
)
EXPECTED_TRAIN_DATA_SHA256 = "ffa721d57849ade8fdd0811e3e1e62fe807200f710aec780dc4d3dcecd4fb0e0"
MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
MODEL_ARTIFACT_SHA256 = {
    "config.json": "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633",
    "chat_template.jinja": "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
    "model.safetensors": "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550",
}


class V24ControlViolation(ValueError):
    """A V24 authorization input, checkpoint, or outcome violated the contract."""


def _fail(message: str) -> None:
    raise V24ControlViolation(message)


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
    if not isinstance(value, str) or len(value) != 64 or value != value.casefold():
        _fail(f"{field} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


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
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot load {field}: {error}")
    return dict(_mapping(value, field))


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
    resolved = _resolve(path)
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
        if layer == 28:
            self.q_proj = _ShapeOnlyLinear(1536, 2048)
        if layer == 29:
            self.q_proj = _ShapeOnlyLinear(1536, 4096)
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
        _fail("V24 did not install a named LoRA bank collection")
    return collection


def v24_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    training = _mapping(config.get("training"), "training")
    screen = _mapping(config.get("v24_screen"), "v24_screen")
    experiment = _mapping(config.get("experiment"), "experiment")
    settings = lora_banks_settings(config)
    optimizer = lora_banks_optimizer_settings(config, settings)
    if optimizer is None:
        _fail("V24 requires an explicit LoRA optimizer")
    bank_contracts = {str(record["name"]): record for record in settings.contract()["banks"]}
    if NEW_BANK not in bank_contracts:
        _fail(f"V24 config is missing bank {NEW_BANK}")
    return {
        "schema_version": 1,
        "role": "frozen_v23_scene_stack_shared_query_screen",
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
            **{
                bank: bank_contracts[bank].get("expected_initial_state_sha256")
                for bank in FROZEN_BANKS
            },
        },
        "new_bank": bank_contracts[NEW_BANK],
        "new_bank_parameter_count": experiment.get("decoder_trainable_parameter_count"),
        "optimizer": {
            **optimizer.contract(),
            "adamw": training.get("optimizer"),
            "gradient_accumulation": training.get("gradient_accumulation"),
        },
        "initialize_named_lora_freeze_and_extend_transition": training.get(
            "initialize_named_lora_freeze_and_extend_transition"
        ),
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
    contract = v24_contract(config)
    digest = _canonical_sha256(contract)
    _equal(digest, EXPECTED_CONTRACT_SHA256, "V24 normalized contract SHA-256")
    return contract, digest


def _source_to_v24_transition_mismatch(
    metadata: Mapping[str, Any], collection: LoRABankCollection
) -> dict[str, Any] | None:
    """Validate freezing every V23 bank while adding exactly one new bank.

    The generic extension helper intentionally rejects a source bank that was
    trainable.  V24 is the composed transition: V23's selected trainable bank
    becomes a frozen, hash-pinned source bank while V24's bank is newly
    trainable and absent from the source checkpoint.
    """

    source_lora = metadata.get("lora")
    if not isinstance(source_lora, Mapping) or source_lora.get("schema_version") != 2:
        return {"checkpoint_lora": source_lora}
    source_records = source_lora.get("banks")
    if not isinstance(source_records, list):
        return {"checkpoint_banks": source_records}
    frozen = tuple(bank for bank in collection.banks if not bank.settings.trainable)
    trainable = tuple(bank for bank in collection.banks if bank.settings.trainable)
    mismatches: dict[str, Any] = {}
    if [bank.settings.name for bank in trainable] != [NEW_BANK]:
        mismatches["trainable_banks"] = [bank.settings.name for bank in trainable]
    source_by_name = {
        str(record.get("name")): record
        for record in source_records
        if isinstance(record, Mapping) and isinstance(record.get("name"), str)
    }
    expected_source_names = {bank.settings.name for bank in frozen}
    if set(source_by_name) != expected_source_names:
        mismatches["source_bank_names"] = {
            "checkpoint": sorted(source_by_name),
            "runtime": sorted(expected_source_names),
        }
    source_hashes = metadata.get("lora_bank_state_sha256")
    source_wrapped = metadata.get("lora_bank_wrapped_modules")
    source_counts = metadata.get("lora_bank_parameter_counts")
    for field, observed in (
        ("lora_bank_state_sha256", source_hashes),
        ("lora_bank_wrapped_modules", source_wrapped),
        ("lora_bank_parameter_counts", source_counts),
    ):
        if not isinstance(observed, Mapping) or set(observed) != expected_source_names:
            mismatches[field] = observed
    for bank in frozen:
        name = bank.settings.name
        source = source_by_name.get(name)
        if source is None:
            continue
        architecture = {
            "rank": bank.settings.adapter.rank,
            "alpha": bank.settings.adapter.alpha,
            "dropout": bank.settings.adapter.dropout,
            "target_modules": list(bank.settings.adapter.target_modules),
            "adapter_parameter_count": bank.installation.parameter_count,
        }
        observed = {key: source.get(key) for key in architecture}
        if observed != architecture:
            mismatches[f"{name}.architecture"] = {
                "checkpoint": observed,
                "runtime": architecture,
            }
        expected_hash = EXPECTED_FROZEN_HASHES.get(name)
        if (
            expected_hash is None
            or bank.settings.expected_initial_state_sha256 != expected_hash
            or not isinstance(source_hashes, Mapping)
            or source_hashes.get(name) != expected_hash
        ):
            mismatches[f"{name}.state"] = {
                "checkpoint": source_hashes.get(name)
                if isinstance(source_hashes, Mapping)
                else None,
                "runtime": bank.settings.expected_initial_state_sha256,
                "pinned": expected_hash,
            }
        if isinstance(source_wrapped, Mapping) and source_wrapped.get(name) != list(
            bank.installation.target_names
        ):
            mismatches[f"{name}.wrapped_modules"] = source_wrapped.get(name)
        if isinstance(source_counts, Mapping) and source_counts.get(name) != (
            bank.installation.parameter_counts
        ):
            mismatches[f"{name}.parameter_counts"] = source_counts.get(name)
    return mismatches or None


def _sequence_length_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute exact token reachability without loading Gemma weights."""

    language = _mapping(config.get("language"), "language")
    _equal(language.get("model_id"), MODEL_ID, "V24 model ID")
    _equal(language.get("revision"), MODEL_REVISION, "V24 model revision")
    snapshot = resolve_local_snapshot(MODEL_ID, MODEL_REVISION)
    _equal(snapshot.name, MODEL_REVISION, "local snapshot revision directory")
    artifact_paths: dict[str, Path] = {}
    snapshot_root = snapshot.resolve()
    cache_root = snapshot_root.parents[1].resolve()
    for name, expected_sha in MODEL_ARTIFACT_SHA256.items():
        entry = snapshot / name
        if not entry.is_symlink():
            _fail(f"pinned local model entry is not the expected cache symlink: {entry}")
        try:
            resolved = entry.resolve(strict=True)
            resolved.relative_to(cache_root)
        except (OSError, ValueError) as error:
            _fail(f"local model entry escapes its pinned cache: {name}: {error}")
        if resolved.is_symlink() or not resolved.is_file():
            _fail(f"local model blob is not a regular file: {resolved}")
        _equal(file_sha256(resolved), expected_sha, f"local model {name} SHA-256")
        artifact_paths[name] = resolved

    model_config = _load_json(artifact_paths["config.json"], "local Gemma config")
    text_config = _mapping(model_config.get("text_config"), "local Gemma text_config")
    layer_types = _sequence(text_config.get("layer_types"), "Gemma layer types")
    for field, expected in {
        "hidden_size": 1536,
        "num_hidden_layers": 35,
        "num_kv_shared_layers": 20,
        "num_attention_heads": 8,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 512,
        "bos_token_id": 2,
        "pad_token_id": 0,
    }.items():
        _equal(text_config.get(field), expected, f"Gemma text_config.{field}")
    _equal(len(layer_types), 35, "Gemma layer type count")
    _equal(layer_types[28], "sliding_attention", "Gemma layer 28 type")
    _equal(layer_types[29], "full_attention", "Gemma layer 29 type")
    q_shapes: dict[str, Any] = {}
    with safe_open(artifact_paths["model.safetensors"], framework="pt", device="cpu") as handle:
        for layer, shape in ((28, [2048, 1536]), (29, [4096, 1536])):
            key = f"model.language_model.layers.{layer}.self_attn.q_proj.weight"
            tensor_slice = handle.get_slice(key)
            _equal(tensor_slice.get_shape(), shape, f"Gemma layer {layer} q_proj weight shape")
            _equal(tensor_slice.get_dtype(), "BF16", f"Gemma layer {layer} q_proj weight dtype")
            q_shapes[str(layer)] = {"key": key, "shape": shape, "dtype": "BF16"}

    modeling_source = PROJECT_ROOT / (
        ".venv-gemma4/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py"
    )
    modeling_source = _regular_file(modeling_source, "Gemma runtime modeling source")
    _equal(
        file_sha256(modeling_source),
        "ccab8e2dd80b71e9ca34e2c87291e17c40a27c755006e554da2ebf70d6616916",
        "Gemma runtime modeling source SHA-256",
    )
    _equal(platform.python_version(), "3.12.13", "Gemma runtime Python version")
    _equal(transformers.__version__, "5.14.1", "Gemma runtime Transformers version")
    _equal(str(torch.__version__), "2.13.0", "Gemma runtime Torch version")

    selection_path = _regular_file(SOURCE_SELECTION, "source training selection")
    _equal(file_sha256(selection_path), EXPECTED_SOURCE_SELECTION_SHA256, "selection SHA-256")
    selection = _load_json(selection_path, "source training selection")
    _equal(selection.get("counterfactual_pair_unit_count"), 12, "source pair unit count")
    _equal(
        selection.get("counterfactual_pair_unit_selection_sha256"),
        EXPECTED_PAIR_UNIT_SELECTION_SHA256,
        "source pair unit selection SHA-256",
    )
    _equal(selection.get("scene_prefix_after_bos"), True, "scene prefix placement")
    _equal(selection.get("scene_boundary_mode"), "gemma4_native_image", "scene boundary mode")

    train_path = _regular_file(Path("data/qa/train.jsonl"), "training QA split")
    _equal(file_sha256(train_path), EXPECTED_TRAIN_DATA_SHA256, "training QA SHA-256")
    dataset = SceneQADataset(train_path)
    curriculum = pair_curriculum_settings(dict(config))
    records = select_pair_only_records(dataset.records, curriculum.pair_only_scene_ids)
    records = cap_pair_units_per_pair(
        records,
        curriculum.max_units_per_pair,
        seed=int(config["seed"]),
    )
    records = select_training_records(
        records,
        max_questions_per_scene=int(
            _mapping(config["training"], "training")["max_questions_per_scene"]
        ),
    )
    units = build_exact_question_pair_units(records)
    unit_selection = [
        {
            "pair_id": unit.pair_id,
            "question_key": unit.question_key,
            "scene_ids": list(unit.scene_ids),
            "question_ids": [record.question_id for record in unit.records],
        }
        for unit in sorted(
            units,
            key=lambda item: (
                item.pair_id,
                item.question_key,
                item.reference.question_id,
                item.counterfactual.question_id,
            ),
        )
    ]
    recomputed_selection_sha = _canonical_sha256(unit_selection)
    _equal(len(records), 24, "recomputed selected record count")
    _equal(len(units), 12, "recomputed selected pair unit count")
    _equal(recomputed_selection_sha, EXPECTED_PAIR_UNIT_SELECTION_SHA256, "recomputed selection")

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    native = _mapping(language.get("gemma4_native_image_contract"), "native image contract")
    _equal(tokenizer.bos_token_id, native.get("bos_token_id"), "tokenizer/native BOS")
    _equal(tokenizer.pad_token_id, native.get("pad_token_id"), "tokenizer/native PAD")
    prompt_lengths = [
        int(
            prompt_token_ids(
                tokenizer,
                str(language["system_prompt"]),
                record.question,
                torch.device("cpu"),
            ).shape[1]
        )
        for record in records
    ]
    answer_lengths = [
        int(tokenize_answer(tokenizer, record.answer, torch.device("cpu")).shape[1])
        for record in records
    ]
    _equal([min(prompt_lengths), max(prompt_lengths)], [57, 62], "prompt token range")
    _equal(sorted(set(answer_lengths)), [2], "answer-plus-EOS token counts")
    scene = _mapping(config.get("scene_encoder"), "scene_encoder")
    _equal(scene.get("global_latents"), 256, "V24 global scene latent count")
    continuous_prefix = 256 + 2
    total_lengths = [
        prompt + continuous_prefix + answer
        for prompt, answer in zip(prompt_lengths, answer_lengths, strict=True)
    ]
    prefix_prompt_lengths = [prompt + continuous_prefix for prompt in prompt_lengths]
    _equal([min(total_lengths), max(total_lengths)], [317, 322], "total token range")
    _equal(
        [min(prefix_prompt_lengths), max(prefix_prompt_lengths)],
        [315, 320],
        "prefix-plus-prompt token range",
    )
    # BOS is position 0, BOI is 1, and the first continuous scene latent is 2.
    # The final prompt query for the longest prefix+prompt is position 319.
    max_to_boi = max(prefix_prompt_lengths) - 2
    max_to_first_latent = max(prefix_prompt_lengths) - 3
    _equal(max_to_boi, 318, "maximum final-query-to-BOI distance")
    _equal(max_to_first_latent, 317, "maximum final-query-to-first-latent distance")
    window = int(text_config["sliding_window"])
    if max_to_boi >= window or max_to_first_latent >= window:
        _fail("V24 final prompt queries cannot reach the complete continuous scene prefix")
    return {
        "schema_version": 1,
        "audit_type": "v24_sliding_attention_sequence_reachability",
        "source": "recomputed_local_tokenizer_and_safetensors_headers_no_model_inference",
        "model_loaded": False,
        "tokenizer_loaded": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_artifact_sha256": dict(MODEL_ARTIFACT_SHA256),
        "model_config": {
            "hidden_size": 1536,
            "num_hidden_layers": 35,
            "num_kv_shared_layers": 20,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "sliding_window": window,
            "target_layer_types": {"28": layer_types[28], "29": layer_types[29]},
        },
        "runtime_modeling_source": {
            "transformers_version": "5.14.1",
            "python_version": "3.12.13",
            "torch_version": "2.13.0",
            "path": str(modeling_source.relative_to(PROJECT_ROOT)),
            "sha256": "ccab8e2dd80b71e9ca34e2c87291e17c40a27c755006e554da2ebf70d6616916",
        },
        "target_q_proj_weights": q_shapes,
        "record_count": len(records),
        "pair_unit_count": len(units),
        "pair_unit_selection_sha256": recomputed_selection_sha,
        "training_qa_path": "data/qa/train.jsonl",
        "training_qa_sha256": EXPECTED_TRAIN_DATA_SHA256,
        "source_selection_path": str(SOURCE_SELECTION),
        "source_selection_sha256": EXPECTED_SOURCE_SELECTION_SHA256,
        "prompt_token_range_inclusive": [min(prompt_lengths), max(prompt_lengths)],
        "answer_plus_eos_tokens": answer_lengths[0],
        "scene_latents": 256,
        "scene_boundary_tokens": 2,
        "continuous_scene_prefix_tokens": continuous_prefix,
        "total_sequence_token_range_inclusive": [min(total_lengths), max(total_lengths)],
        "prefix_plus_prompt_token_range_inclusive": [
            min(prefix_prompt_lengths),
            max(prefix_prompt_lengths),
        ],
        "sliding_window_tokens": window,
        "maximum_final_query_to_boi_distance": max_to_boi,
        "maximum_final_query_to_first_scene_latent_distance": max_to_first_latent,
        "maximum_final_query_to_boi_inclusive_span": max_to_boi + 1,
        "maximum_final_query_to_first_scene_latent_inclusive_span": max_to_first_latent + 1,
        "all_final_prompt_queries_reach_entire_scene_prefix": True,
        "target_interpretation": (
            "highest-depth previously untouched sliding/full pair immediately before adapted tail"
        ),
    }


def run_preflight(config_path: Path, output: Path) -> dict[str, Any]:
    config = load_config(config_path)
    contract, contract_digest = _validate_contract(config)
    archive = validate_archive(
        PROJECT_ROOT / SOURCE_ARCHIVE,
        repo_root=PROJECT_ROOT,
        verify_bound_files=True,
    )
    _equal(archive.get("summary_sha256"), V23_ARCHIVE_SHA256, "V23 archive SHA-256")
    _equal(archive.get("selected_epoch"), 2, "V23 selected epoch")
    _equal(
        archive.get("decision"),
        "conditional_limit_reached_no_greedy_audit",
        "V23 outcome",
    )
    _equal(archive.get("bound_files_verified"), True, "V23 bound files")

    source = PROJECT_ROOT / SOURCE_CHECKPOINT
    artifacts = {
        "adapter_sha256": file_sha256(source / "adapter.safetensors"),
        "metadata_sha256": file_sha256(source / "metadata.json"),
        "optimizer_sha256": file_sha256(source / "optimizer.pt"),
    }
    _equal(artifacts, EXPECTED_SOURCE_ARTIFACTS, "source checkpoint artifacts")
    metadata = _load_json(source / "metadata.json", "V23 source metadata")
    for field, expected in {
        "epoch": 2,
        "output_namespace": "gemma4_v23_shared_kv",
        "frozen_scene_state_sha256": EXPECTED_FROZEN_HASHES["scene"],
        "global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["signed_x"],
    }.items():
        _equal(metadata.get(field), expected, f"source metadata {field}")
    _equal(
        metadata.get("lora_bank_state_sha256"),
        {bank: EXPECTED_FROZEN_HASHES[bank] for bank in FROZEN_BANKS},
        "source LoRA bank hashes",
    )

    collection = _install_shape_only(config)
    trainer_transition_mismatch = named_lora_freeze_and_extend_transition_mismatch(
        metadata, collection
    )
    if trainer_transition_mismatch is not None:
        _fail(f"trainer source-to-V24 transition mismatch: {trainer_transition_mismatch}")
    transition_mismatch = _source_to_v24_transition_mismatch(metadata, collection)
    if transition_mismatch is not None:
        _fail(f"source-to-V24 transition mismatch: {transition_mismatch}")
    bank = collection.bank(NEW_BANK)
    _equal(bank.settings.adapter.target_modules, EXPECTED_TARGETS, "new bank targets")
    _equal(bank.installation.parameter_count, 36_864, "new bank parameter count")
    _equal(
        bank.installation.state_sha256(),
        EXPECTED_NEW_BANK_INITIAL_SHA256,
        "new bank initial state",
    )
    parameters = list(bank.installation.parameters())
    _equal(
        [tuple(parameter.shape) for parameter in parameters],
        list(EXPECTED_PARAMETER_SHAPES),
        "new bank parameter order",
    )
    if any(not parameter.requires_grad for parameter in parameters):
        _fail("one or more V24 bank parameters are not trainable")
    if any(
        parameter.requires_grad
        for frozen_bank in collection.banks
        if frozen_bank.settings.name in FROZEN_BANKS
        for parameter in frozen_bank.installation.parameters()
    ):
        _fail("one or more inherited V24 bank parameters remained trainable")
    if any(torch.count_nonzero(adapter.lora_b).item() for adapter in bank.installation.adapters):
        _fail("V24 bank is not exact zero-output")
    sequence_audit = _sequence_length_audit(config)

    provenance = _clean_provenance(
        capture_git_source_provenance(PROJECT_ROOT), "current source provenance"
    )
    report = {
        "schema_version": 1,
        "audit_type": "v24_shared_query_structural_preflight",
        "authorized": True,
        "stage_1_authorized": True,
        "runtime_eligible": False,
        "report_only": True,
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
        "source_artifact_hashes": artifacts,
        "source_metadata_sha256": artifacts["metadata_sha256"],
        "source_provenance": provenance,
        "sequence_length_audit": sequence_audit,
        "frozen_bank_state_sha256": {
            bank_name: EXPECTED_FROZEN_HASHES[bank_name] for bank_name in FROZEN_BANKS
        },
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
    report = _load_json(path, "V24 preflight")
    digest = file_sha256(_regular_file(path, "V24 preflight"))
    contract, _contract_digest = _validate_contract(config)
    expected_keys = {
        "schema_version",
        "audit_type",
        "authorized",
        "stage_1_authorized",
        "runtime_eligible",
        "report_only",
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
        "sequence_length_audit",
        "frozen_bank_state_sha256",
        "new_bank",
    }
    _equal(set(report), expected_keys, "preflight root keys")
    for field, expected in {
        "schema_version": 1,
        "audit_type": "v24_shared_query_structural_preflight",
        "authorized": True,
        "stage_1_authorized": True,
        "runtime_eligible": False,
        "report_only": True,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "optimizer_steps": 0,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "config_path": str(CONFIG_PATH),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_artifact_hashes": EXPECTED_SOURCE_ARTIFACTS,
        "source_metadata_sha256": EXPECTED_SOURCE_ARTIFACTS["metadata_sha256"],
        "frozen_bank_state_sha256": {bank: EXPECTED_FROZEN_HASHES[bank] for bank in FROZEN_BANKS},
        "new_bank": {
            "name": NEW_BANK,
            "state_sha256": EXPECTED_NEW_BANK_INITIAL_SHA256,
            "parameter_count": 36_864,
            "target_modules": list(EXPECTED_TARGETS),
            "ordered_parameter_shapes": [list(shape) for shape in EXPECTED_PARAMETER_SHAPES],
            "exact_zero_output": True,
        },
    }.items():
        _equal(report.get(field), expected, f"preflight {field}")
    _equal(report.get("contract"), contract, "preflight contract")
    _equal(report.get("sequence_length_audit"), _sequence_length_audit(config), "sequence audit")
    archive = _mapping(report.get("source_archive"), "preflight source archive")
    _equal(archive.get("summary_sha256"), V23_ARCHIVE_SHA256, "preflight V23 archive")
    _equal(archive.get("selected_epoch"), 2, "preflight V23 selected epoch")
    _equal(archive.get("bound_files_verified"), True, "preflight V23 bound files")
    _clean_provenance(report.get("source_provenance"), "preflight source provenance")
    return report, digest


def _adapter_payload(adapter_path: Path) -> dict[str, Any]:
    safe_adapter = _regular_file(adapter_path, "V24 adapter safetensors")
    scene_prefixes = ("scene_model.", "composer.", "grounding.")
    global_prefix = "global_scene_residual."
    signed_x_prefix = "signed_x_scene_residual."
    bank_prefixes = {name: f"lora_banks.{name}." for name in (*FROZEN_BANKS, NEW_BANK)}
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
                for name, prefix in bank_prefixes.items():
                    if key.startswith(prefix):
                        bank_states[name][key.removeprefix(prefix)] = tensor
                        break
                else:
                    unknown.append(key)
    if unknown:
        _fail(f"V24 adapter contains unknown tensor keys: {unknown}")
    if not scene_state or not global_state or not signed_x_state:
        _fail("V24 adapter is missing a frozen scene/residual tensor group")
    if any(not state for state in bank_states.values()):
        _fail("V24 adapter is missing one or more LoRA bank tensor groups")
    new_state = bank_states[NEW_BANK]
    expected_keys = {
        f"adapters.{index}.{suffix}" for index in range(2) for suffix in ("lora_a", "lora_b")
    }
    _equal(set(new_state), expected_keys, "V24 safetensors bank keys")
    return {
        "scene_state_sha256": tensor_state_sha256(scene_state),
        "global_scene_residual_state_sha256": tensor_state_sha256(global_state),
        "signed_x_scene_residual_state_sha256": tensor_state_sha256(signed_x_state),
        "lora_bank_state_sha256": {
            name: tensor_state_sha256(state) for name, state in bank_states.items()
        },
        "new_bank_state": new_state,
        "tensor_count": (
            len(scene_state)
            + len(global_state)
            + len(signed_x_state)
            + sum(len(state) for state in bank_states.values())
        ),
    }


def _require_frozen_bank_pins(bank_hashes: Mapping[str, Any], *, field: str) -> None:
    _equal(set(bank_hashes) & set(FROZEN_BANKS), set(FROZEN_BANKS), f"{field} bank keys")
    for name in FROZEN_BANKS:
        _equal(bank_hashes.get(name), EXPECTED_FROZEN_HASHES[name], f"{field} {name}")


def _require_new_bank_tensor_contract(state: Mapping[str, torch.Tensor], *, field: str) -> None:
    expected_keys = {
        f"adapters.{index}.{suffix}" for index in range(2) for suffix in ("lora_a", "lora_b")
    }
    _equal(set(state), expected_keys, f"{field} tensor keys")
    ordered_keys = [
        f"adapters.{index}.{suffix}" for index in range(2) for suffix in ("lora_a", "lora_b")
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
    safe_optimizer = _regular_file(path, "V24 optimizer state")
    try:
        state_dict = torch.load(safe_optimizer, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        _fail(f"cannot safely load V24 optimizer state: {error}")
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
        "params": list(range(4)),
    }
    _equal(group, expected_group, "optimizer group contract")
    states = _mapping(root.get("state"), "optimizer state")
    _equal(set(states), set(range(4)), "optimizer state IDs")
    manifest: list[dict[str, Any]] = []
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
        _equal(tuple(exp_avg.shape), expected_shape, f"optimizer state {parameter_id}.exp_avg")
        _equal(
            tuple(exp_avg_sq.shape), expected_shape, f"optimizer state {parameter_id}.exp_avg_sq"
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
        manifest.append(
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
        "parameter_states": manifest,
        "all_state_tensors_sha256": tensor_state_sha256(aggregate),
    }


def _exact_accuracy_count(value: Any, denominator: int, field: str) -> int:
    accuracy = _finite_float(value, field)
    if not 0.0 <= accuracy <= 1.0:
        _fail(f"{field} must be a finite probability")
    scaled = accuracy * denominator
    count = round(scaled)
    if not math.isclose(scaled, count, rel_tol=0.0, abs_tol=1e-5):
        _fail(f"{field} is not an exact {denominator}-way empirical fraction: {accuracy}")
    return count


def _pair_metrics(metadata: Mapping[str, Any], pair_id: str) -> dict[str, Any]:
    gate = _mapping(metadata.get("pair_candidate_gate"), "pair_candidate_gate")
    by_pair = _mapping(gate.get("by_pair"), "pair_candidate_gate.by_pair")
    pair = _mapping(by_pair.get(pair_id), f"pair gate {pair_id}")
    return {
        "full_vocab_sides": _exact_accuracy_count(
            pair.get("first_answer_token_top1_accuracy"), 12, f"{pair_id} side accuracy"
        ),
        "full_vocab_units": _exact_accuracy_count(
            pair.get("first_answer_token_top1_unit_accuracy"), 6, f"{pair_id} unit accuracy"
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


_OPAQUE_SCENE = re.compile(r"^scene_[0-9a-f]{6,64}$")
_OPAQUE_QUESTION = re.compile(r"^q_[0-9a-f]{6,64}$")


def _opaque_unit_margin_detail(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and reduce trainer-emitted margins without running inference."""

    gate = _mapping(metadata.get("pair_candidate_gate"), "pair_candidate_gate")
    detail = _mapping(gate.get("detail"), "pair_candidate_gate.detail")
    for field, expected in {
        "schema_version": 1,
        "artifact": "training_candidate_gate_detail",
        "training_only": True,
        "free_generation_evaluated": False,
        "candidate_representation": "candidate_token_ids",
        "contains_question_text": False,
        "contains_oracle_geometry": False,
        "contains_canonical_training_targets": False,
        "full_vocab_first_token_evaluated": True,
        "unit_count": 12,
        "side_count": 24,
    }.items():
        _equal(detail.get(field), expected, f"candidate detail {field}")

    def nested_keys(value: Any) -> set[str]:
        if isinstance(value, Mapping):
            return {str(key) for key in value} | {
                nested for item in value.values() for nested in nested_keys(item)
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return {nested for item in value for nested in nested_keys(item)}
        return set()

    observed_keys = nested_keys(detail)
    for prohibited in (
        "own_canonical_target",
        "alternate_canonical_target",
        "predicted_canonical_target",
        "question_text",
        "oracle_geometry",
    ):
        if prohibited in observed_keys:
            _fail(f"candidate detail contains prohibited field {prohibited}")
    units = _sequence(detail.get("units"), "candidate detail units")
    _equal(len(units), 12, "candidate detail unit count")
    pair_order = ("pair_000001", "pair_000003")
    normalized_by_pair: dict[str, Any] = {}
    for pair_offset, pair_id in enumerate(pair_order):
        normalized_units: list[dict[str, Any]] = []
        candidate_values: list[float] = []
        full_values: list[float] = []
        for local_index, value in enumerate(units[pair_offset * 6 : (pair_offset + 1) * 6]):
            unit = _mapping(value, f"candidate detail {pair_id} unit {local_index}")
            _equal(unit.get("unit_index"), pair_offset * 6 + local_index, "detail unit index")
            scene_ids = list(_sequence(unit.get("scene_ids"), "detail scene IDs"))
            question_ids = list(_sequence(unit.get("question_ids"), "detail question IDs"))
            if len(scene_ids) != 2 or any(
                not isinstance(item, str) or _OPAQUE_SCENE.fullmatch(item) is None
                for item in scene_ids
            ):
                _fail("candidate detail scene IDs are not opaque")
            if len(question_ids) != 2 or any(
                not isinstance(item, str) or _OPAQUE_QUESTION.fullmatch(item) is None
                for item in question_ids
            ):
                _fail("candidate detail question IDs are not opaque")
            sides = _sequence(unit.get("sides"), "candidate detail sides")
            _equal(len(sides), 2, "candidate detail side count")
            normalized_sides: list[dict[str, Any]] = []
            for side_index, side_value in enumerate(sides):
                side = _mapping(side_value, "candidate detail side")
                _equal(side.get("side_index"), side_index, "candidate detail side index")
                _equal(side.get("scene_id"), scene_ids[side_index], "detail side scene")
                _equal(side.get("question_id"), question_ids[side_index], "detail side question")
                candidate = _finite_float(
                    side.get("own_vs_alternate_candidate_logit_margin"),
                    "candidate detail candidate margin",
                )
                full = _finite_float(
                    side.get("first_token_target_vs_best_other_logit_margin"),
                    "candidate detail full-vocabulary margin",
                )
                _equal(side.get("own_preference_passed"), candidate > 0.0, "detail candidate pass")
                _equal(side.get("full_vocab_top1_passed"), full > 0.0, "detail full pass")
                candidate_values.append(candidate)
                full_values.append(full)
                normalized_sides.append(
                    {
                        "side_index": side_index,
                        "scene_id": scene_ids[side_index],
                        "question_id": question_ids[side_index],
                        "candidate_margin": candidate,
                        "full_vocab_margin": full,
                    }
                )
            normalized_units.append(
                {
                    "unit_index": local_index,
                    "candidate_both_positive": all(
                        side["candidate_margin"] > 0.0 for side in normalized_sides
                    ),
                    "full_vocab_both_positive": all(
                        side["full_vocab_margin"] > 0.0 for side in normalized_sides
                    ),
                    "sides": normalized_sides,
                }
            )
        aggregate = _pair_metrics(metadata, pair_id)
        recomputed = {
            "full_vocab_sides": sum(value > 0.0 for value in full_values),
            "full_vocab_units": sum(
                all(value > 0.0 for value in full_values[index : index + 2])
                for index in range(0, 12, 2)
            ),
            "mean_candidate_margin": sum(candidate_values) / 12,
            "minimum_candidate_margin": min(candidate_values),
            "mean_full_vocab_margin": sum(full_values) / 12,
            "minimum_full_vocab_margin": min(full_values),
        }
        for field, observed in recomputed.items():
            expected = aggregate[field]
            if isinstance(observed, float):
                if not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1e-6):
                    _fail(
                        f"candidate detail {pair_id} {field} mismatch: "
                        f"expected={expected!r} observed={observed!r}"
                    )
            else:
                _equal(observed, expected, f"candidate detail {pair_id} {field}")
        normalized_by_pair[pair_id] = {"unit_count": 6, "units": normalized_units}
    return {
        "schema_version": 1,
        "source": "checkpoint_metadata_only_no_model_inference",
        "contains_environment_text": False,
        "by_pair": normalized_by_pair,
        "source_detail_sha256": _canonical_sha256(detail),
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
    return _color_eligible(metrics)


def verify_update1(
    config_path: Path,
    preflight_path: Path,
    checkpoint_path: Path,
    output: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    preflight, preflight_sha256 = _load_preflight(preflight_path, config)
    current_provenance = _clean_provenance(
        capture_git_source_provenance(PROJECT_ROOT), "current source provenance"
    )
    _equal(current_provenance, preflight.get("source_provenance"), "current/preflight source")
    checkpoint = _resolve(checkpoint_path)
    expected = PROJECT_ROOT / "data_gemma4/checkpoints" / PRIMARY_NAMESPACE / "epoch_001"
    _equal(checkpoint.resolve(), expected.resolve(), "update-1 checkpoint path")
    adapter_path = _regular_file(checkpoint / "adapter.safetensors", "update-1 adapter")
    metadata_path = _regular_file(checkpoint / "metadata.json", "update-1 metadata")
    optimizer_path = _regular_file(checkpoint / "optimizer.pt", "update-1 optimizer")
    artifacts = {
        "adapter_sha256": file_sha256(adapter_path),
        "metadata_sha256": file_sha256(metadata_path),
        "optimizer_sha256": file_sha256(optimizer_path),
    }
    metadata = _load_json(metadata_path, "V24 update-1 metadata")
    for field, expected_value in {
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
        _equal(metadata.get(field), expected_value, f"update-1 metadata {field}")
    _equal(
        metadata.get("source_provenance"), preflight.get("source_provenance"), "source provenance"
    )
    expected_frozen_banks = {name: EXPECTED_FROZEN_HASHES[name] for name in FROZEN_BANKS}
    _equal(
        metadata.get("frozen_lora_bank_state_sha256"),
        expected_frozen_banks,
        "frozen LoRA hashes",
    )
    initialization = _mapping(
        metadata.get("initialization_provenance"), "initialization provenance"
    )
    for field, expected_value in {
        "schema_version": 6,
        "mode": "existing_named_lora_banks_frozen_plus_zero_output_named_lora_extension",
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
        _equal(initialization.get(field), expected_value, f"initialization provenance {field}")
    _equal(
        initialization.get("source_lora_bank_state_sha256"),
        expected_frozen_banks,
        "initialization frozen LoRA hashes",
    )

    collection = _install_shape_only(config)
    settings = lora_banks_settings(config)
    optimizer_settings = lora_banks_optimizer_settings(config, settings)
    assert optimizer_settings is not None
    _equal(
        metadata.get("lora"),
        lora_banks_checkpoint_contract(settings, optimizer_settings, collection.parameter_counts),
        "checkpoint LoRA contract",
    )
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "LoRA bank hashes")
    _equal(set(bank_hashes), {*FROZEN_BANKS, NEW_BANK}, "LoRA bank hash keys")
    _require_frozen_bank_pins(bank_hashes, field="update-1 metadata frozen bank")
    if bank_hashes.get(NEW_BANK) == EXPECTED_NEW_BANK_INITIAL_SHA256:
        _fail("V24 update-1 bank remained at its initial state")

    payload = _adapter_payload(adapter_path)
    for field, expected_value in {
        "scene_state_sha256": EXPECTED_FROZEN_HASHES["scene"],
        "global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["signed_x"],
        "lora_bank_state_sha256": dict(bank_hashes),
    }.items():
        _equal(payload[field], expected_value, f"recomputed {field}")
    _require_frozen_bank_pins(payload["lora_bank_state_sha256"], field="recomputed frozen bank")
    checkpoint_state = payload["new_bank_state"]
    _require_new_bank_tensor_contract(checkpoint_state, field="update-1 new bank")
    _equal(tensor_state_sha256(checkpoint_state), bank_hashes.get(NEW_BANK), "new bank state hash")
    initial_state = collection.bank(NEW_BANK).installation.state_module.state_dict()
    for index in range(2):
        a_key = f"adapters.{index}.lora_a"
        b_key = f"adapters.{index}.lora_b"
        if not torch.equal(checkpoint_state[a_key], initial_state[a_key]):
            _fail(f"LoRA-A tensor changed at target index {index}")
        if not torch.count_nonzero(checkpoint_state[b_key]).item():
            _fail(f"LoRA-B tensor remained zero at target index {index}")
    optimizer_manifest = _optimizer_manifest(optimizer_path, expected_step=1)
    color = _pair_metrics(metadata, "pair_000001")
    mirror = _pair_metrics(metadata, "pair_000003")
    detail = _opaque_unit_margin_detail(metadata)
    stage_2 = _color_eligible(color) and _mirror_continuation(mirror)
    report = {
        "schema_version": 1,
        "audit_type": "v24_shared_query_update1_verifier",
        "match": True,
        "stage_2_authorized": stage_2,
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
        "all_prior_tensors_frozen": True,
        "optimizer_manifest": optimizer_manifest,
        "recomputed_payload_hashes": {
            key: value for key, value in payload.items() if key != "new_bank_state"
        },
        "color": color,
        "mirror": mirror,
        "opaque_unit_margin_detail": detail,
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
    metadata_path = _resolve(path)
    expected_path = (
        PROJECT_ROOT
        / "data_gemma4/checkpoints"
        / PRIMARY_NAMESPACE
        / f"epoch_{epoch:03d}"
        / "metadata.json"
    )
    _equal(metadata_path.resolve(), expected_path.resolve(), f"epoch {epoch} metadata path")
    metadata_path = _regular_file(metadata_path, f"V24 epoch {epoch} metadata")
    metadata = _load_json(metadata_path, f"V24 epoch {epoch} metadata")
    for field, expected_value in {
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
        _equal(metadata.get(field), expected_value, f"epoch {epoch} {field}")
    _equal(metadata.get("source_provenance"), source_provenance, f"epoch {epoch} provenance")
    expected_frozen_banks = {name: EXPECTED_FROZEN_HASHES[name] for name in FROZEN_BANKS}
    _equal(
        metadata.get("frozen_lora_bank_state_sha256"),
        expected_frozen_banks,
        f"epoch {epoch} frozen banks",
    )
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), f"epoch {epoch} banks")
    _equal(set(bank_hashes), {*FROZEN_BANKS, NEW_BANK}, f"epoch {epoch} bank keys")
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
    history = _sequence(metadata.get("history"), f"epoch {epoch} history")
    _equal(len(history), epoch, f"epoch {epoch} history length")
    for index, row in enumerate(history, start=1):
        _equal(_mapping(row, "history row").get("epoch"), index, "history epoch sequence")
    _equal(
        metadata.get("pair_candidate_gate"),
        _mapping(history[-1], "final history row").get("pair_candidate_gate"),
        f"epoch {epoch} final gate/history binding",
    )
    checkpoint = metadata_path.parent
    adapter_path = _regular_file(checkpoint / "adapter.safetensors", f"epoch {epoch} adapter")
    optimizer_path = _regular_file(checkpoint / "optimizer.pt", f"epoch {epoch} optimizer")
    payload = _adapter_payload(adapter_path)
    for field, expected_value in {
        "scene_state_sha256": EXPECTED_FROZEN_HASHES["scene"],
        "global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["signed_x"],
        "lora_bank_state_sha256": dict(bank_hashes),
    }.items():
        _equal(payload[field], expected_value, f"epoch {epoch} recomputed {field}")
    _require_frozen_bank_pins(payload["lora_bank_state_sha256"], field="recomputed frozen bank")
    _require_new_bank_tensor_contract(payload["new_bank_state"], field=f"epoch {epoch} new bank")
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
        "optimizer_manifest": _optimizer_manifest(optimizer_path, expected_step=epoch),
        "color": _pair_metrics(metadata, "pair_000001"),
        "mirror": _pair_metrics(metadata, "pair_000003"),
        "opaque_unit_margin_detail": _opaque_unit_margin_detail(metadata),
    }


def _ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    mirror = _mapping(row.get("mirror"), "ranking mirror metrics")
    return (
        int(mirror["full_vocab_units"]),
        int(mirror["full_vocab_sides"]),
        float(mirror["mean_full_vocab_margin"]),
        float(mirror["minimum_full_vocab_margin"]),
        float(mirror["mean_candidate_margin"]),
        float(mirror["minimum_candidate_margin"]),
        -int(row["epoch"]),
    )


def select_epochs(
    config_path: Path,
    update1_path: Path,
    epoch_bindings: Mapping[int, Path],
    output: Path,
) -> dict[str, Any]:
    """Select the bounded V24 screen using checkpoint files only."""

    config = load_config(config_path)
    _validate_contract(config)
    update1_path = _regular_file(update1_path, "V24 update-1 verifier")
    update1 = _load_json(update1_path, "V24 update-1 verifier")
    expected_keys = {
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
        "all_prior_tensors_frozen",
        "optimizer_manifest",
        "recomputed_payload_hashes",
        "color",
        "mirror",
        "opaque_unit_margin_detail",
        "source_provenance",
    }
    _equal(set(update1), expected_keys, "update-1 report root keys")
    for field, expected_value in {
        "schema_version": 1,
        "audit_type": "v24_shared_query_update1_verifier",
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
        "all_prior_tensors_frozen": True,
    }.items():
        _equal(update1.get(field), expected_value, f"update-1 {field}")
    _sha256(update1.get("preflight_sha256"), "update-1 preflight SHA-256")
    _sha256(update1.get("new_bank_state_sha256"), "update-1 bank SHA-256")
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
    detail = _mapping(update1.get("opaque_unit_margin_detail"), "update-1 opaque detail")
    _equal(detail.get("contains_environment_text"), False, "update-1 detail leakage")
    _equal(
        detail.get("source"),
        "checkpoint_metadata_only_no_model_inference",
        "update-1 detail source",
    )
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
    for update_field, epoch_field, label in (
        ("new_bank_state_sha256", "new_bank_state_sha256", "bank"),
        ("optimizer_manifest", "optimizer_manifest", "optimizer"),
        ("recomputed_payload_hashes", "recomputed_payload_hashes", "tensor payload"),
        ("color", "color", "color metrics"),
        ("mirror", "mirror", "mirror metrics"),
        ("opaque_unit_margin_detail", "opaque_unit_margin_detail", "opaque detail"),
    ):
        _equal(update1.get(update_field), epoch1[epoch_field], f"update-1/epoch-1 {label}")

    eligible = [row for row in epochs if _color_eligible(row["color"])]
    ranking = sorted(eligible, key=_ranking_key, reverse=True)
    selected = ranking[0] if ranking else None
    full_teacher = bool(
        selected is not None and _full_pair(selected["color"]) and _full_pair(selected["mirror"])
    )
    continuation = bool(
        selected is not None and not full_teacher and _mirror_continuation(selected["mirror"])
    )
    if full_teacher:
        decision = "full_teacher_gate_passed_greedy_audit_authorized"
    elif continuation:
        decision = "screen_passed_extension_authorized_no_greedy_audit"
    else:
        decision = "screen_failed_no_extension_no_greedy_audit"
    report = {
        "schema_version": 1,
        "audit_type": "v24_shared_query_epoch_selector",
        "decision": decision,
        "selected_epoch": None if selected is None else selected["epoch"],
        "selected_checkpoint": (
            None if selected is None else str(Path(selected["metadata_path"]).parent)
        ),
        "continuation_authorized": continuation,
        "full_teacher_gate_passed": full_teacher,
        "greedy_audit_authorized": full_teacher,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "update1_report_sha256": file_sha256(update1_path),
        "epochs": epochs,
        "ranking": [
            {
                "rank": rank,
                "epoch": row["epoch"],
                "color": row["color"],
                "mirror": row["mirror"],
            }
            for rank, row in enumerate(ranking, start=1)
        ],
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
            "model_inference_during_selection": False,
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
    "V24ControlViolation",
    "run_preflight",
    "select_epochs",
    "v24_contract",
    "verify_update1",
]

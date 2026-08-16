"""Scene-disjoint training data assembly for Gemma-4 tool-decoder V2."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    HIDDEN_SIZE,
    NumericToolContextProjectorV2,
    canonical_answer_token_ids,
    canonical_tool_json_from_trace,
    prepare_tool_decoder_inputs,
    tool_decoder_system_prompt,
)
from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.robot.navigation_policy import ACTION_NAMES
from semantic_3d_chat.robot.navigation_policy_v3 import grounded_target_state
from semantic_3d_chat.robot.state_checkpoint import load_robot_state_checkpoint
from semantic_3d_chat.robot.state_encoder import insert_robot_state_tokens
from semantic_3d_chat.training.gemma4_tool_decoder_v2_clearance import (
    load_clearance_cache_v2,
)

TRAIN_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(11, 25)
)
VALIDATION_SCENES: Final[tuple[str, ...]] = (
    "scene_000031",
    "scene_000032",
    "scene_000033",
    "scene_000034",
    "scene_000035",
    "scene_000036",
    "scene_000037",
    "scene_000039",
)
CAUSAL_ROWS_PER_SCENE_FAMILY: Final[int] = 8
PRIMARY_ROWS_PER_SCENE_FAMILY: Final[int] = 16
CAUSAL_VALIDATION_SAMPLE_COUNT: Final[int] = 448
PRIMARY_VALIDATION_SAMPLE_COUNT: Final[int] = 832
CAUSAL_VALIDATION_SAMPLE_IDS_SHA256: Final[str] = (
    "a411ffacbbcf0ba348a528e605884f1309ddc837b1bcd0c32ac1af2446e4a622"
)
PRIMARY_VALIDATION_SAMPLE_IDS_SHA256: Final[str] = (
    "801c421f24b6e8d3d902450d56e55fed5a6c45e85d56cdf80e21ce5e17b28115"
)
GREEDY_CONTROL_SAMPLE_COUNT_V2_1: Final[int] = 56
GREEDY_CONTROL_SAMPLE_IDS_SHA256_V2_1: Final[str] = (
    "1f83bd0479016fb12cca0cc835af01f395f124ce6ade1e0afffcd3a83c867a5b"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ToolDecoderSampleV2:
    sample_id: str
    scene_id: str
    split: str
    family: str
    instruction: str
    action_index: int
    action_name: str
    normalized_argument: float
    state_features: torch.Tensor
    robot_tokens: torch.Tensor
    target_state: torch.Tensor
    clearance_state: torch.Tensor
    collision_targets: torch.Tensor
    canonical_answer: str


@dataclass(frozen=True)
class ToolDecoderDatasetV2:
    prefixes: Mapping[str, torch.Tensor]
    samples: tuple[ToolDecoderSampleV2, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    prefix_inventory_sha256: str
    clearance_cache_sha256: str
    trace_rows_sha256: str

    def sample(self, index: int) -> ToolDecoderSampleV2:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("V2 dataset index must be an integer")
        return self.samples[index]


def action_balanced_schedule_v2(
    dataset: ToolDecoderDatasetV2,
    *,
    microbatch_count: int,
    seed: int,
) -> tuple[int, ...]:
    """Return a deterministic near-uniform schedule over the five actions."""

    if isinstance(microbatch_count, bool) or not isinstance(microbatch_count, int):
        raise TypeError("V2 microbatch_count must be an integer")
    if microbatch_count < 1:
        raise ValueError("V2 microbatch_count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("V2 sampler seed must be nonnegative")
    by_action: dict[str, list[int]] = defaultdict(list)
    for index in dataset.train_indices:
        by_action[dataset.samples[index].action_name].append(index)
    if set(by_action) != set(ACTION_NAMES) or any(not by_action[name] for name in ACTION_NAMES):
        raise ValueError("V2 sampler requires every fixed action in the training split")
    randomizer = random.Random(seed)
    for values in by_action.values():
        randomizer.shuffle(values)
    cursors = {name: 0 for name in ACTION_NAMES}
    order = list(ACTION_NAMES)
    randomizer.shuffle(order)
    result: list[int] = []
    while len(result) < microbatch_count:
        for name in order:
            values = by_action[name]
            cursor = cursors[name]
            if cursor == len(values):
                randomizer.shuffle(values)
                cursor = 0
            result.append(values[cursor])
            cursors[name] = cursor + 1
            if len(result) == microbatch_count:
                break
        order = order[1:] + order[:1]
    counts = {name: 0 for name in ACTION_NAMES}
    for index in result:
        counts[dataset.samples[index].action_name] += 1
    if max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError("V2 action-balanced schedule drifted by more than one sample")
    return tuple(result)


def _sample_id_digest(dataset: ToolDecoderDatasetV2, indices: Sequence[int]) -> str:
    sample_ids = [dataset.samples[index].sample_id for index in indices]
    return hashlib.sha256(
        json.dumps(sample_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stratified_validation_indices_v2(
    dataset: ToolDecoderDatasetV2,
    *,
    rows_per_scene_family: int,
) -> tuple[int, ...]:
    """Select fixed scene/family strata while round-robining available actions."""

    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in dataset.validation_indices:
        sample = dataset.samples[index]
        grouped[(sample.scene_id, sample.family)][sample.action_name].append(index)
    result: list[int] = []
    for key in sorted(grouped):
        by_action = grouped[key]
        target_count = min(
            rows_per_scene_family, sum(len(values) for values in by_action.values())
        )
        for values in by_action.values():
            values.sort(key=lambda index: dataset.samples[index].sample_id)
        names = sorted(by_action)
        cursors = {name: 0 for name in names}
        while sum(cursors.values()) < target_count:
            progressed = False
            for name in names:
                if (
                    cursors[name] < len(by_action[name])
                    and sum(cursors.values()) < target_count
                ):
                    result.append(by_action[name][cursors[name]])
                    cursors[name] += 1
                    progressed = True
            if not progressed:
                break
        if sum(cursors.values()) != target_count:
            raise ValueError(f"V2 validation stratum is too small: {key}")
    return tuple(sorted(result))


def causal_validation_indices_v2(
    dataset: ToolDecoderDatasetV2,
) -> tuple[int, ...]:
    """Return the immutable 448-row causal-generation evaluation subset."""

    indices = _stratified_validation_indices_v2(
        dataset, rows_per_scene_family=CAUSAL_ROWS_PER_SCENE_FAMILY
    )
    if (
        len(indices) != CAUSAL_VALIDATION_SAMPLE_COUNT
        or _sample_id_digest(dataset, indices) != CAUSAL_VALIDATION_SAMPLE_IDS_SHA256
    ):
        raise ValueError("V2 causal validation sample IDs changed")
    return indices


def primary_validation_indices_v2(
    dataset: ToolDecoderDatasetV2,
) -> tuple[int, ...]:
    """Return the immutable 832-row primary greedy-generation subset."""

    indices = _stratified_validation_indices_v2(
        dataset, rows_per_scene_family=PRIMARY_ROWS_PER_SCENE_FAMILY
    )
    if (
        len(indices) != PRIMARY_VALIDATION_SAMPLE_COUNT
        or _sample_id_digest(dataset, indices) != PRIMARY_VALIDATION_SAMPLE_IDS_SHA256
    ):
        raise ValueError("V2 primary validation sample IDs changed")
    return indices


def greedy_control_validation_indices_v2_1(
    dataset: ToolDecoderDatasetV2,
) -> tuple[int, ...]:
    """Return one deterministic row from every held-out scene/family stratum."""

    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in dataset.validation_indices:
        sample = dataset.samples[index]
        grouped[(sample.scene_id, sample.family)][sample.action_name].append(index)
    result: list[int] = []
    for ordinal, key in enumerate(sorted(grouped)):
        by_action = grouped[key]
        names = sorted(by_action)
        action = names[ordinal % len(names)]
        candidates = sorted(
            by_action[action], key=lambda index: dataset.samples[index].sample_id
        )
        result.append(candidates[0])
    indices = tuple(sorted(result))
    if (
        len(indices) != GREEDY_CONTROL_SAMPLE_COUNT_V2_1
        or _sample_id_digest(dataset, indices)
        != GREEDY_CONTROL_SAMPLE_IDS_SHA256_V2_1
    ):
        raise ValueError("V2.1 greedy-control sample IDs changed")
    return indices


def _load_prefixes(prefix_root: Path) -> tuple[dict[str, torch.Tensor], str]:
    manifest = json.loads((prefix_root / "manifest.json").read_text(encoding="utf-8"))
    scenes = manifest.get("scenes")
    if (
        not isinstance(scenes, Mapping)
        or manifest.get("complete_scene_prefixes") is not True
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_scene_retrieval") is not False
        or manifest.get("environmental_text_inputs") != []
    ):
        raise ValueError("V2 prefix-cache contract changed")
    prefixes: dict[str, torch.Tensor] = {}
    inventory: dict[str, str] = {}
    for scene_id in (*TRAIN_SCENES, *VALIDATION_SCENES):
        entry = scenes.get(scene_id)
        if not isinstance(entry, Mapping):
            raise FileNotFoundError(f"V2 has no prefix-cache entry for {scene_id}")
        path = prefix_root / str(entry.get("filename"))
        digest = _sha256(path)
        if digest != entry.get("file_sha256"):
            raise ValueError(f"V2 prefix file changed for {scene_id}")
        state = load_file(str(path), device="cpu")
        if set(state) != {"scene_prefix"}:
            raise ValueError("V2 prefix tensor inventory changed")
        prefix = state["scene_prefix"]
        if prefix.shape != (1, 258, HIDDEN_SIZE) or prefix.dtype != torch.bfloat16:
            raise ValueError("V2 prefix shape or dtype changed")
        if not torch.isfinite(prefix.float()).all():
            raise ValueError("V2 prefix contains NaN or infinity")
        prefixes[scene_id] = prefix.contiguous()
        inventory[scene_id] = digest
    inventory_digest = hashlib.sha256(
        json.dumps(
            inventory, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if inventory_digest != (
        "c477fd12bc4104f147f73c2f6d46904e0b83b3c584206cb227fd70e9371d0d63"
    ):
        raise ValueError("V2 prefix inventory digest changed")
    return prefixes, inventory_digest


@torch.inference_mode()
def load_tool_decoder_dataset_v2(
    config: Mapping[str, Any],
    *,
    trace_root: str | Path = "data_gemma4/training/navigation_policy_v3",
    prefix_root: str | Path = "data_gemma4/scene_tokens/v56_question_control_full_prefixes",
    clearance_root: str | Path = (
        "data_gemma4/training/gemma4_embodied_tool_decoder_v2"
    ),
) -> ToolDecoderDatasetV2:
    """Load authenticated continuous inputs and training-only numeric targets."""

    experiment = config.get("gemma4_embodied_tool_decoder_v2")
    robot = config.get("robot")
    scene = config.get("scene")
    language = config.get("language")
    if not all(isinstance(value, Mapping) for value in (experiment, robot, scene, language)):
        raise TypeError("V2 dataset config mappings are incomplete")
    trace_directory = PROJECT_ROOT / Path(trace_root)
    trace_path = trace_directory / "traces.jsonl"
    trace_manifest_path = trace_directory / "manifest.json"
    trace_manifest = json.loads(trace_manifest_path.read_text(encoding="utf-8"))
    trace_digest = _sha256(trace_path)
    if (
        trace_digest
        != "72434178ff1cf23c2dfeb98d52cb7b4c443fcc8715c1dd4ee883d87ae127e7ad"
        or trace_manifest.get("train_scene_ids") != list(TRAIN_SCENES)
        or trace_manifest.get("validation_scene_ids") != list(VALIDATION_SCENES)
        or set(TRAIN_SCENES) & set(VALIDATION_SCENES)
        or trace_manifest.get("scene_splits_disjoint") is not True
    ):
        raise ValueError("V2 trace identity or scene split changed")
    rows: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row.get("sample_id") != f"g_{index:08d}":
                raise ValueError("V2 trace row ordering changed")
            rows.append(row)
    if len(rows) != 6468:
        raise ValueError("V2 trace sample count changed")

    cache_sha = "658822707389e67481fa59b035a7e7f19c360487b19d3157b80bc23ede1db048"
    clearance, collision, _cache_manifest = load_clearance_cache_v2(
        clearance_root,
        expected_cache_sha256=cache_sha,
        expected_manifest_sha256=(
            "51cf6c0b155e149627f300c17d39369f91f14e415099fe10d9de1682ef8c7e24"
        ),
        expected_trace_rows_sha256=trace_digest,
    )
    prefixes, inventory_sha = _load_prefixes(PROJECT_ROOT / Path(prefix_root))
    encoder, _state_hash, state_metadata = load_robot_state_checkpoint(
        PROJECT_ROOT / str(experiment["robot_state_checkpoint"]),
        expected_output_dim=HIDDEN_SIZE,
        device="cpu",
    )
    if (
        state_metadata.get("numeric_inputs_only") is not True
        or state_metadata.get("token_count") != 4
    ):
        raise ValueError("V2 robot-state checkpoint contract changed")
    room = scene.get("room_size_m")
    states = torch.tensor([row["state_features"] for row in rows], dtype=torch.float32)
    robot_tokens = encoder(states).float().contiguous()
    target_xyz = torch.tensor(
        [row["oracle_target_xyz_m"] for row in rows], dtype=torch.float32
    )
    available = torch.tensor(
        [bool(row["target_state_available"]) for row in rows], dtype=torch.float32
    )
    target_states = grounded_target_state(
        target_xyz, states, available, room_size_m=room
    ).contiguous()
    max_turn = float(robot["max_turn_degrees"])
    max_move = float(robot["max_move_m"])
    samples: list[ToolDecoderSampleV2] = []
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for index, row in enumerate(rows):
        split = str(row["split"])
        scene_id = str(row["scene_id"])
        if (split == "train" and scene_id not in TRAIN_SCENES) or (
            split == "validation" and scene_id not in VALIDATION_SCENES
        ):
            raise ValueError("V2 trace row crossed its sealed scene split")
        if split not in {"train", "validation"}:
            raise ValueError("V2 trace row has an invalid split")
        sample = ToolDecoderSampleV2(
            sample_id=str(row["sample_id"]),
            scene_id=scene_id,
            split=split,
            family=str(row["family"]),
            instruction=str(row["instruction"]),
            action_index=int(row["action_index"]),
            action_name=str(row["action_name"]),
            normalized_argument=float(row["argument_target_normalized"]),
            state_features=states[index].contiguous(),
            robot_tokens=robot_tokens[index].contiguous(),
            target_state=target_states[index].contiguous(),
            clearance_state=clearance[index].contiguous(),
            collision_targets=collision[index].contiguous(),
            canonical_answer=canonical_tool_json_from_trace(
                row, max_turn_degrees=max_turn, max_move_m=max_move
            ),
        )
        samples.append(sample)
        (train_indices if split == "train" else validation_indices).append(index)
    if len(train_indices) != 4200 or len(validation_indices) != 2268:
        raise ValueError("V2 trace split sample counts changed")
    return ToolDecoderDatasetV2(
        prefixes=prefixes,
        samples=tuple(samples),
        train_indices=tuple(train_indices),
        validation_indices=tuple(validation_indices),
        prefix_inventory_sha256=inventory_sha,
        clearance_cache_sha256=cache_sha,
        trace_rows_sha256=trace_digest,
    )


def controlled_sample_inputs_v2(
    dataset: ToolDecoderDatasetV2,
    index: int,
    *,
    control: str = "primary",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ToolDecoderSampleV2]:
    """Return scene, robot, target, and clearance under a sealed causal control."""

    modes = {
        "primary",
        "wrong_scene",
        "zero_scene",
        "wrong_robot",
        "zero_robot",
        "wrong_target",
        "zero_target",
        "wrong_clearance",
        "zero_clearance",
    }
    if control not in modes:
        raise ValueError(f"Unknown V2 causal control: {control}")
    sample = dataset.sample(index)
    validation = dataset.validation_indices
    if index not in validation:
        pool = dataset.train_indices
    else:
        pool = validation
    position = pool.index(index)

    def alternate_for(mode: str) -> ToolDecoderSampleV2:
        """Choose a deterministic, genuinely different control value.

        Adjacent trace rows can belong to the same scene or repeat a numeric
        context.  A nominal ``wrong_*`` control must therefore search the
        sealed split instead of assuming the next row differs.
        """

        for offset in range(1, len(pool)):
            candidate = dataset.samples[pool[(position + offset) % len(pool)]]
            if mode == "wrong_scene" and candidate.scene_id != sample.scene_id:
                return candidate
            if mode == "wrong_robot" and not torch.equal(
                candidate.robot_tokens, sample.robot_tokens
            ):
                return candidate
            if mode == "wrong_target" and not torch.equal(
                candidate.target_state, sample.target_state
            ):
                return candidate
            if mode == "wrong_clearance" and not torch.equal(
                candidate.clearance_state, sample.clearance_state
            ):
                return candidate
        raise RuntimeError(f"V2 cannot construct a genuinely different {mode} control")

    scene = dataset.prefixes[sample.scene_id].clone()
    robot = sample.robot_tokens.unsqueeze(0).clone()
    target = sample.target_state.unsqueeze(0).clone()
    clearance = sample.clearance_state.unsqueeze(0).clone()
    if control == "wrong_scene":
        alternate = alternate_for(control)
        scene = dataset.prefixes[alternate.scene_id].clone()
    elif control == "zero_scene":
        scene[:, 1:-1].zero_()
    elif control == "wrong_robot":
        alternate = alternate_for(control)
        robot = alternate.robot_tokens.unsqueeze(0).clone()
    elif control == "zero_robot":
        robot.zero_()
    elif control == "wrong_target":
        alternate = alternate_for(control)
        target = alternate.target_state.unsqueeze(0).clone()
    elif control == "zero_target":
        target.zero_()
    elif control == "wrong_clearance":
        alternate = alternate_for(control)
        clearance = alternate.clearance_state.unsqueeze(0).clone()
    elif control == "zero_clearance":
        clearance.zero_()
    active = insert_robot_state_tokens(scene, robot)
    return active, target, clearance, sample


def prepare_microbatch_v2(
    dataset: ToolDecoderDatasetV2,
    index: int,
    *,
    language: Any,
    projector: NumericToolContextProjectorV2,
    max_turn_degrees: float,
    max_move_m: float,
    control: str = "primary",
    include_answer: bool = True,
) -> tuple[Any, ToolDecoderSampleV2]:
    """Build PLE-aware ``inputs_embeds`` with answer-only JSON labels."""

    if getattr(language, "backend_name", None) != "gemma4":
        raise ValueError("V2 microbatches require the local Gemma-4 backend")
    backend = getattr(language, "prefix_backend", None)
    if backend is None:
        raise TypeError("V2 language backend has no Gemma PLE interface")
    active, target, clearance, sample = controlled_sample_inputs_v2(
        dataset, index, control=control
    )
    device = language.device
    # The complete scene prefix is selected before tokenizing user text.
    active = active.to(device)
    system = tool_decoder_system_prompt(
        max_turn_degrees=max_turn_degrees, max_move_m=max_move_m
    )
    prompt_ids = prompt_token_ids(
        language.tokenizer, system, sample.instruction, device
    )
    answer_ids = (
        canonical_answer_token_ids(
            language.tokenizer, sample.canonical_answer, device=device
        )
        if include_answer
        else None
    )
    prepared = prepare_tool_decoder_inputs(
        backend,
        active,
        prompt_ids,
        projector,
        target.to(device),
        clearance.to(device),
        answer_ids=answer_ids,
    )
    return prepared, sample


__all__ = [
    "CAUSAL_VALIDATION_SAMPLE_COUNT",
    "CAUSAL_VALIDATION_SAMPLE_IDS_SHA256",
    "GREEDY_CONTROL_SAMPLE_COUNT_V2_1",
    "GREEDY_CONTROL_SAMPLE_IDS_SHA256_V2_1",
    "PRIMARY_VALIDATION_SAMPLE_COUNT",
    "PRIMARY_VALIDATION_SAMPLE_IDS_SHA256",
    "TRAIN_SCENES",
    "VALIDATION_SCENES",
    "ToolDecoderDatasetV2",
    "ToolDecoderSampleV2",
    "action_balanced_schedule_v2",
    "causal_validation_indices_v2",
    "controlled_sample_inputs_v2",
    "greedy_control_validation_indices_v2_1",
    "load_tool_decoder_dataset_v2",
    "prepare_microbatch_v2",
    "primary_validation_indices_v2",
]

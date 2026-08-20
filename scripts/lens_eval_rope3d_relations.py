#!/usr/bin/env python3
"""Ask Gemma relational questions that only real 3D geometry can answer.

Grid readout ("which cell?") is entangled with a convention this repo invented,
so a failure there is ambiguous: it could be missing geometry or an unlearned
addressing scheme. These two questions avoid that entirely.

  higher   Which of two objects is further off the floor?
  nearer   Which of two objects is closer to a third?

Neither needs a frame of reference, a grid, or any coordinate written as text,
and both are decided by the room rather than by word co-occurrence. ``higher``
matters most: the bird's-eye pooling averages a whole floor column into one
token, so token order carries no height at all. If the answer improves when the
same tokens are rotated by their 3D positions, height reached the decoder
through the rotation, which is the only channel carrying it.

Ground truth is perception's own -- discovered voxels and the names Gemma gave
them -- and every question is repeated with the positions scrambled, which
leaves the semantics untouched and the geometry destroyed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.discover import discover_objects
from semantic_3d_chat.spatial_lens.grounding_data import available_rooms
from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.reason_3d import ask_3d
from semantic_3d_chat.spatial_lens.scene_tokens_3d import build_scene_tokens_3d, zeroed

# raster           position implied by the order the tokens arrive in
# rope3d_xyz        all three axes drive the decoder's rotary channel
# rope3d_z_only     raster order kept, height added on a third of the slots
# rope3d_scrambled  same as xyz, with the positions permuted among the tokens
# zeroed            no scene at all
CONDITIONS = (
    "raster",
    "rope3d_xyz",
    "rope3d_z_only",
    "rope3d_scrambled",
    "zeroed",
)

SYSTEM = (
    "You are looking at a three-dimensional scan of a room, supplied as visual "
    "tokens. Answer the question about the objects in it.\n"
    "Reply with ONE json object and nothing else: "
    '{"answer": "<exactly one of the two names offered>"}.'
)


def scrambled_positions(tokens, seed: int = 20260818):
    from dataclasses import replace

    order = np.random.default_rng(seed).permutation(tokens.centroids_m.shape[0])
    return replace(tokens, centroids_m=tokens.centroids_m[order].copy())


def parse(reply: str, options: tuple[str, str]) -> str | None:
    import re

    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match is not None:
        try:
            said = str(json.loads(match.group(0)).get("answer", "")).casefold()
        except ValueError:
            said = ""
        # Longest match, not first: "shelf" is inside "bookshelf", so taking
        # whichever option happens to be presented first hands the point to the
        # shorter name every time the two are nested.
        hits = [o for o in options if o.casefold() in said]
        if hits:
            return max(hits, key=len)
    lowered = reply.casefold()
    mentioned = [o for o in options if o.casefold() in lowered]
    if len(mentioned) == 2:
        longer, shorter = sorted(mentioned, key=len, reverse=True)
        nested = shorter.casefold() in longer.casefold()
        if nested and lowered.count(shorter.casefold()) == lowered.count(longer.casefold()):
            return longer
    return mentioned[0] if len(mentioned) == 1 else None


def build_questions(cloud: SemanticCloud, named: dict[str, str], rng: random.Random):
    """Every question this room supports, with its answer from the geometry."""

    objects = []
    for proposal in discover_objects(cloud):
        name = named.get(proposal.proposal_id)
        if not name or name == "unidentified object":
            continue
        points = np.asarray(cloud.centers_m, dtype=np.float64)[proposal.voxel_indices]
        objects.append((name, points.mean(axis=0), points))
    # Repeated names would make the answer ambiguous rather than wrong.
    counts: dict[str, int] = {}
    for name, _, _ in objects:
        counts[name] = counts.get(name, 0) + 1
    objects = [o for o in objects if counts[o[0]] == 1]

    questions = []
    for (a_name, a_mid, _), (b_name, b_mid, _) in itertools.combinations(objects, 2):
        # A near-tie is not a fact about the room, so do not score it.
        if abs(a_mid[2] - b_mid[2]) >= 0.25:
            first, second = (a_name, b_name) if rng.random() < 0.5 else (b_name, a_name)
            questions.append({
                "kind": "higher",
                "text": f"Which is higher off the floor, the {first} or the {second}?",
                "options": (first, second),
                "answer": a_name if a_mid[2] > b_mid[2] else b_name,
                "margin_m": round(float(abs(a_mid[2] - b_mid[2])), 3),
            })
    for anchor, anchor_mid, _ in objects:
        others = [o for o in objects if o[0] != anchor]
        for (b_name, b_mid, _), (c_name, c_mid, _) in itertools.combinations(others, 2):
            gap_b = float(np.linalg.norm(b_mid[:2] - anchor_mid[:2]))
            gap_c = float(np.linalg.norm(c_mid[:2] - anchor_mid[:2]))
            if abs(gap_b - gap_c) < 0.6:
                continue
            first, second = (b_name, c_name) if rng.random() < 0.5 else (c_name, b_name)
            questions.append({
                "kind": "nearer",
                "text": f"Which is closer to the {anchor}, the {first} or the {second}?",
                "options": (first, second),
                "answer": b_name if gap_b < gap_c else c_name,
                "margin_m": round(abs(gap_b - gap_c), 3),
            })
    return questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rooms", type=int, default=0,
                        help="cap on rooms measured; 0 uses every scanned room")
    parser.add_argument("--room-prefix", default=None,
                        help="only measure rooms whose name starts with this")
    parser.add_argument("--per-room", type=int, default=6,
                        help="questions sampled per room per kind")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--report",
                        default="reports/gemma4/metrics/rope3d_relations.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shuffled = list(available_rooms())
    if args.room_prefix:
        shuffled = [r for r in shuffled if r.startswith(args.room_prefix)]
    random.Random(20260818).shuffle(shuffled)
    rooms = sorted(shuffled[: args.rooms] if args.rooms else shuffled)
    print(f"measuring {len(rooms)} rooms")

    from semantic_3d_chat.language.local_lm import load_local_language_model

    language = load_local_language_model(
        "google/gemma-4-E2B-it",
        revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        requested_dtype="bfloat16",
        local_files_only=True,
    )

    occupancy: list[float] = []
    tally: dict[tuple[str, str], list[float]] = {}
    answered: dict[tuple[str, str], list[float]] = {}
    unparsed = dict.fromkeys(CONDITIONS, 0)
    trials = []
    for room in rooms:
        root = PROJECT_ROOT / "data" / "spatial_lens" / room
        cloud = SemanticCloud.load(root / "point_cloud.npz")
        tokens = build_scene_tokens_3d(cloud)
        occupancy.append(tokens.occupied_fraction)
        named = {
            item["object_id"]: item["name"]
            for item in json.loads((root / "scene_graph.json").read_text())["objects"]
        }
        variants = {
            "raster": (tokens, False, "xyz"),
            "rope3d_xyz": (tokens, True, "xyz"),
            "rope3d_z_only": (tokens, True, "z_only"),
            "rope3d_scrambled": (scrambled_positions(tokens), True, "xyz"),
            "zeroed": (zeroed(tokens), False, "xyz"),
        }
        questions = build_questions(cloud, named, rng)
        for kind in ("higher", "nearer"):
            pool = [q for q in questions if q["kind"] == kind]
            rng.shuffle(pool)
            for question in pool[: args.per_room]:
                marks = {}
                for condition, (field, use_rope, axes) in variants.items():
                    reply = ask_3d(
                        language, field, question["text"],
                        system=SYSTEM, max_new_tokens=40,
                        rope3d=use_rope, axes=axes,
                    )
                    said = parse(reply, question["options"])
                    if said is None:
                        unparsed[condition] += 1
                    # An unreadable answer is a wrong answer, not a skipped one
                    # -- but the two are tracked apart, because a condition that
                    # simply refuses more often would otherwise look worse at
                    # spatial reasoning than one that guesses.
                    correct = float(said == question["answer"])
                    tally.setdefault((kind, condition), []).append(correct)
                    answered.setdefault((kind, condition), []).append(
                        float(said is not None)
                    )
                    marks[condition] = correct
                trials.append({"room": room, **{
                    k: v for k, v in question.items() if k != "options"}, "marks": marks})
                print(f"  {room:8s} {kind:7s} " + "  ".join(
                    f"{c}={marks[c]:.0f}" for c in CONDITIONS))

    from semantic_3d_chat.evaluation.proportions import (
        holm_adjust,
        mcnemar_exact,
        wilson_interval,
    )

    def cell(kind: str, condition: str) -> dict[str, object]:
        marks_list = tally[(kind, condition)]
        replies = answered[(kind, condition)]
        total = len(marks_list)
        hits = int(sum(marks_list))
        answered_count = int(sum(replies))
        answered_hits = sum(
            m for m, a in zip(marks_list, replies, strict=True) if a
        )
        return {
            "questions": total,
            "correct": hits,
            "accuracy": round(hits / total, 3) if total else None,
            "interval_95": wilson_interval(hits, total),
            "answered": answered_count,
            # A coin-flipper that refuses as often as this arm did would score
            # this, which is the floor the raw accuracy should be read against.
            "chance_given_this_refusal_rate": round(
                0.5 * answered_count / total, 3
            ) if total else None,
            "accuracy_when_answered": (
                round(answered_hits / answered_count, 3) if answered_count else None
            ),
        }

    results = {
        kind: {c: cell(kind, c) for c in CONDITIONS if (kind, c) in tally}
        for kind in ("higher", "nearer")
    }
    # Every condition answers the same questions, so the comparison against the
    # raster layout is paired and does not need to treat the arms as independent.
    against_raster = {
        kind: {
            c: mcnemar_exact(tally[(kind, c)], tally[(kind, "raster")])
            for c in CONDITIONS
            if c != "raster" and (kind, c) in tally and (kind, "raster") in tally
        }
        for kind in ("higher", "nearer")
    }
    summary = {
        "rooms": rooms,
        "room_prefix": args.room_prefix,
        "chance_when_answered": 0.5,
        # Empty columns are zero vectors, so this much of a "real" scene differs
        # from the zeroed control at all. Read that control accordingly.
        "occupied_fraction": round(float(np.mean(occupancy)), 3),
        "unparsed_replies": unparsed,
        "results": results,
        "mcnemar_vs_raster": against_raster,
        "holm_adjusted_p": {
            kind: holm_adjust({k: v["p_value"] for k, v in arms.items()})
            for kind, arms in against_raster.items()
        },
        "trials": trials,
    }
    destination = PROJECT_ROOT / args.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train the spatial grounding head, and measure it on rooms it never saw.

The split is by ROOM, not by example: every phrase from a held-out room is
held out with it. A head that merely memorised which cell a lamp lives in would
score at chance here, which is the point.

Supervision comes from perception alone -- discovered footprints and the names
Gemma gave them -- so no oracle or human annotation enters training.
"""

from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np
import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.grounding import (
    SpatialGroundingHead,
    dihedral,
    footprint_distance_m,
    locate_error_m,
    save_head,
    soft_cross_entropy,
)
from semantic_3d_chat.spatial_lens.grounding_data import (
    available_rooms,
    collect,
    embed_phrases,
)


def _batches(items, size, rng):
    order = list(range(len(items)))
    rng.shuffle(order)
    for start in range(0, len(order), size):
        yield [items[index] for index in order[start : start + size]]


def _stack(examples, embeddings, index_of, device, rng=None):
    scenes, targets = [], []
    for example in examples:
        if rng is None:
            scenes.append(example.scene)
            targets.append(example.target)
            continue
        field, mass = dihedral(
            example.scene, example.target, example.grid, rng.randrange(8)
        )
        scenes.append(field)
        targets.append(mass)
    scene = torch.from_numpy(np.stack(scenes)).to(device).float()
    query = torch.from_numpy(
        np.stack([embeddings[index_of[id(e)]] for e in examples])
    ).to(device).float()
    target = torch.from_numpy(np.stack(targets)).to(device).float()
    return scene, query, target


@torch.no_grad()
def evaluate(head, examples, embeddings, index_of, device, tolerance_m):
    if not examples:
        return {"examples": 0}
    head.eval()
    scene, query, target = _stack(examples, embeddings, index_of, device)
    logits = head(scene, query)
    errors = []
    for i, example in enumerate(examples):
        errors.extend(
            locate_error_m(
                logits[i : i + 1].cpu(), target[i : i + 1].cpu(),
                example.grid, example.room_size_m,
            )
        )
    # A prediction also counts as a hit when it lands on a cell the object
    # actually occupies, which is the honest test for a multi-cell footprint.
    on_object = sum(
        float(target[i, int(logits[i].argmax())] > 0) for i in range(len(examples))
    )
    # Distance to the object's footprint, which is zero on a hit -- the metric
    # that matches "could the rover drive to this answer".
    gaps, soft_gaps = [], []
    soft = head.expected_cell(logits).cpu()
    for i, example in enumerate(examples):
        gaps.extend(footprint_distance_m(
            logits[i : i + 1].cpu(), target[i : i + 1].cpu(),
            example.grid, example.room_size_m))
        soft_gaps.extend(footprint_distance_m(
            logits[i : i + 1].cpu(), target[i : i + 1].cpu(),
            example.grid, example.room_size_m, soft=soft[i : i + 1]))
    return {
        "examples": len(examples),
        "median_error_m": round(float(np.median(errors)), 3),
        "mean_error_m": round(float(np.mean(errors)), 3),
        "within_tolerance": round(float(np.mean([e <= tolerance_m for e in errors])), 4),
        "hits_object_cell": round(on_object / len(examples), 4),
        "median_footprint_gap_m": round(float(np.median(gaps)), 3),
        "mean_footprint_gap_m": round(float(np.mean(gaps)), 3),
        "footprint_gap_under_0p5m": round(float(np.mean([g <= 0.5 for g in gaps])), 4),
        "soft_median_footprint_gap_m": round(float(np.median(soft_gaps)), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, default=4, help="rooms reserved for testing")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tolerance-m", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", default="data_gemma4/checkpoints/spatial_grounding_v1")
    parser.add_argument("--metrics", default="reports/gemma4/metrics/spatial_lens_grounding.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    rooms = available_rooms()
    if len(rooms) < args.holdout + 2:
        raise SystemExit(f"need more perceived rooms; have {len(rooms)}: {rooms}")
    shuffled_rooms = list(rooms)
    rng.shuffle(shuffled_rooms)
    test_rooms = sorted(shuffled_rooms[: args.holdout])
    train_rooms = sorted(shuffled_rooms[args.holdout :])
    print(f"train rooms ({len(train_rooms)}): {', '.join(train_rooms)}")
    print(f"held-out rooms ({len(test_rooms)}): {', '.join(test_rooms)}\n")

    train = collect(train_rooms, grid=args.grid)
    test = collect(test_rooms, grid=args.grid)
    if not train or not test:
        raise SystemExit("no grounding examples were produced")
    print(f"examples: {len(train)} train, {len(test)} held out")

    from semantic_3d_chat.language.local_lm import load_local_language_model

    language = load_local_language_model(
        "google/gemma-4-E2B-it",
        revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        requested_dtype="bfloat16",
        local_files_only=True,
    )
    every = train + test
    index_of = {id(example): i for i, example in enumerate(every)}
    embeddings = embed_phrases(language, [example.phrase for example in every])
    del language

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    head = SpatialGroundingHead(
        feature_dim=train[0].scene.shape[1],
        model_dim=args.model_dim,
        layers=args.layers,
        grid=args.grid,
        dropout=args.dropout,
    ).to(device)
    parameters = sum(p.numel() for p in head.parameters())
    print(f"grounding head: {parameters/1e6:.2f}M parameters (the only thing trained)\n")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # A pre-norm transformer on a few hundred examples is easy to blow up: an
    # earlier run spiked at epoch 200 and a later one diverged to NaN outright.
    # Warm up, then decay, and refuse to apply a step that is not finite.
    warmup = max(1, args.epochs // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (
            (epoch + 1) / warmup
            if epoch < warmup
            else 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, args.epochs - warmup)))
        ),
    )
    best = None
    history = []
    skipped = 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for batch in _batches(train, args.batch_size, rng):
            scene, query, target = _stack(batch, embeddings, index_of, device, rng)
            loss = soft_cross_entropy(head(scene, query), target)
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            if not torch.isfinite(norm):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        if not losses:
            raise SystemExit("every batch was non-finite; training cannot proceed")
        if epoch % 25 == 0 or epoch == args.epochs:
            held = evaluate(head, test, embeddings, index_of, device, args.tolerance_m)
            fit = evaluate(head, train, embeddings, index_of, device, args.tolerance_m)
            history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4),
                            "train": fit, "heldout": held})
            print(f"  epoch {epoch:4d} loss {np.mean(losses):.4f}  "
                  f"train hit {fit['hits_object_cell']:.2f}   "
                  f"HELD-OUT hit {held['hits_object_cell']:.2f} "
                  f"gap {held['median_footprint_gap_m']:.2f} m "
                  f"(<0.5 m {held['footprint_gap_under_0p5m']:.0%})", flush=True)
            if best is None or held["hits_object_cell"] > best[0]:
                best = (held["hits_object_cell"], epoch,
                        {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})

    assert best is not None
    head.load_state_dict(best[2])
    final_held = evaluate(head, test, embeddings, index_of, device, args.tolerance_m)
    final_train = evaluate(head, train, embeddings, index_of, device, args.tolerance_m)

    # Chance: pick a cell uniformly; probability it lands on the object.
    chance = float(np.mean([(e.target > 0).sum() / e.target.size for e in test]))

    save_head(PROJECT_ROOT / args.output, head, {
        "feature_dim": int(train[0].scene.shape[1]), "model_dim": args.model_dim,
        "heads": 8, "layers": args.layers, "grid": args.grid,
        "dihedral_augmentation": True, "dropout": args.dropout,
        "train_rooms": train_rooms, "heldout_rooms": test_rooms,
        "parameters": int(parameters), "best_epoch": int(best[1]),
        "supervision": "self_supervised_from_perception",
        "oracle_used": False,
    })
    report = {
        "schema": "semantic_3d_chat.spatial_lens.grounding_eval.v1",
        "train_rooms": train_rooms, "heldout_rooms": test_rooms,
        "train_examples": len(train), "heldout_examples": len(test),
        "parameters": int(parameters), "best_epoch": int(best[1]),
        "grid": args.grid, "tolerance_m": args.tolerance_m,
        "skipped_nonfinite_steps": skipped,
        "random_baseline_hit_rate": round(chance, 4),
        "train": final_train, "heldout": final_held,
        "oracle_used": False,
        "supervision": "self_supervised_from_perception",
        "history": history,
    }
    destination = PROJECT_ROOT / args.metrics
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nHELD-OUT ROOMS ({len(test_rooms)} never trained on):")
    print(f"  lands on the object   : {final_held['hits_object_cell']:.1%}  "
          f"(random {chance:.1%})")
    print(f"  median error          : {final_held['median_error_m']:.2f} m")
    print(f"  within {args.tolerance_m:.1f} m         : {final_held['within_tolerance']:.1%}")
    print(f"  gap to footprint      : {final_held['median_footprint_gap_m']:.2f} m median, "
          f"{final_held['footprint_gap_under_0p5m']:.1%} under 0.5 m")
    print(f"  lift over chance      : {final_held['hits_object_cell']/max(chance,1e-9):.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

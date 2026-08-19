#!/usr/bin/env python3
"""Train and measure the 3D-rotary point grounding model.

Rooms are split, not examples: every phrase from a held-out room is held out
with it, so a model that memorised where lamps live in the training rooms scores
at chance here.

Three position schemes share everything else, which is what makes the comparison
mean something:

  rope3d            position enters as a rotation of the embedding, so attention
                    depends on the displacement between two points
  learned_absolute  position enters as a learned vector added to the embedding
  none              no position at all -- a bag of semantic points

Augmentation is a rigid motion of the room in metres: any yaw, an optional
mirror, and a translation. The grid model could only manage the eight dihedral
symmetries, because a raster has no other ones.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.grounding_data import available_rooms
from semantic_3d_chat.spatial_lens.point_grounding import PointGroundingModel, save_model
from semantic_3d_chat.spatial_lens.point_grounding_data import collect, collect_relational

CACHE = PROJECT_ROOT / "data" / "spatial_lens" / "phrase_embeddings.npz"


def load_phrase_vectors() -> dict[str, np.ndarray]:
    if not CACHE.is_file():
        raise SystemExit("run scripts/lens_cache_phrases.py first")
    with np.load(CACHE, allow_pickle=False) as data:
        return dict(zip(data["phrases"].tolist(), data["vectors"], strict=True))


def rigid(points: np.ndarray, rng: random.Random) -> np.ndarray:
    """A random rigid motion of the room, in metres."""

    angle = rng.uniform(0.0, 2.0 * math.pi)
    cos, sin = math.cos(angle), math.sin(angle)
    mirror = -1.0 if rng.random() < 0.5 else 1.0
    x = points[:, 0] * mirror
    moved = np.stack(
        [
            x * cos - points[:, 1] * sin + rng.uniform(-2.0, 2.0),
            x * sin + points[:, 1] * cos + rng.uniform(-2.0, 2.0),
            points[:, 2],
        ],
        axis=1,
    )
    return moved.astype(np.float32)


def point_features(example, mode: str) -> np.ndarray:
    """What each point is allowed to say about itself."""

    if mode == "gemma":
        return example.features
    if mode == "rgb":
        # Colour alone, in a form that survives the LayerNorm the projection
        # starts with. Writing RGB into three slots of 1536 and zeroing the rest
        # does not: LayerNorm removes the mean, which for a mostly-zero vector is
        # dominated by the padding, and brightness -- the part that separates a
        # tan cabinet from a brown one -- is the first thing lost. A Fourier
        # expansion spreads each channel across the full width instead, so the
        # control genuinely carries the colour it claims to.
        colour = np.asarray(example.rgb, dtype=np.float32)
        width = example.features.shape[1]
        bands = max(width // 6, 1)
        frequencies = (2.0 ** np.arange(bands, dtype=np.float32)) * np.pi
        angles = colour[:, :, None] * frequencies[None, None, :]
        expanded = np.concatenate(
            [np.sin(angles).reshape(len(colour), -1),
             np.cos(angles).reshape(len(colour), -1)], axis=1
        )
        out = np.zeros_like(example.features)
        out[:, : expanded.shape[1]] = expanded[:, :width]
        return out
    raise ValueError(f"unknown feature mode: {mode}")


def stack(examples, vectors, device, rng=None, feature_mode="gemma"):
    points = np.stack(
        [rigid(e.points, rng) if rng is not None else e.points for e in examples]
    )
    features = np.stack([point_features(e, feature_mode) for e in examples])
    query = np.stack([vectors[e.phrase] for e in examples])
    target = np.stack([e.target for e in examples])
    def to(array):
        return torch.from_numpy(array).to(device).float()

    return to(points), to(features), to(query), to(target)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


@torch.no_grad()
def evaluate(model, examples, vectors, device, batch=8, feature_mode="gemma"):
    if not examples:
        return {"examples": 0}
    model.eval()
    hits, gaps, chance, object_chance = [], [], [], []
    for start in range(0, len(examples), batch):
        chunk = examples[start : start + batch]
        points, features, query, target = stack(
            chunk, vectors, device, feature_mode=feature_mode
        )
        logits = model(features, points, query)
        predicted = model.predict_position(features, points, query)
        best = logits.argmax(dim=-1)
        for index, example in enumerate(chunk):
            hits.append(float(target[index, best[index]] > 0))
            # Two nulls, because they answer different questions. The first is a
            # uniformly random point; the second is a guesser that already knows
            # the answer is an object rather than floor or wall, and picks among
            # them at random. For the relational task especially, the second is
            # the one a claim should be measured against -- against the first,
            # simply learning "objects are not floor" looks like spatial skill.
            chance.append(float((example.target > 0).mean()))
            if example.candidate_count:
                # The answer is one of this room's objects, so a guesser that
                # knows only that much is right one time in however many there
                # are -- regardless of how few points each one occupies.
                object_chance.append(1.0 / float(example.candidate_count))
            # Against the object's full-resolution voxels, so the number is
            # comparable to the grid head's footprint gap.
            occupied = (
                example.footprint
                if example.footprint is not None
                else example.points[example.target > 0]
            )
            gap = np.linalg.norm(
                occupied - predicted[index].cpu().numpy()[None, :], axis=1
            ).min()
            gaps.append(float(gap))
    informed = float(np.mean(object_chance)) if object_chance else float("nan")
    return {
        "examples": len(examples),
        "hits_object": round(float(np.mean(hits)), 4),
        "chance_uniform_point": round(float(np.mean(chance)), 4),
        "chance_random_object": round(informed, 4),
        "lift_over_uniform_point": round(
            float(np.mean(hits) / max(np.mean(chance), 1e-9)), 2
        ),
        "lift_over_random_object": (
            round(float(np.mean(hits) / max(informed, 1e-9)), 2)
            if object_chance else None
        ),
        "median_gap_m": round(float(np.median(gaps)), 3),
        "mean_gap_m": round(float(np.mean(gaps)), 3),
        "gap_under_0p5m": round(float(np.mean([g <= 0.5 for g in gaps])), 4),
        "gap_under_1m": round(float(np.mean([g <= 1.0 for g in gaps])), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="object",
                        choices=["object", "relational", "both"],
                        help="'relational' phrases name an object by where it is "
                             "relative to another, which semantics alone cannot resolve")
    parser.add_argument("--feature-mode", default="gemma", choices=["gemma", "rgb"],
                        help="'rgb' replaces Gemma's embedding with the point's "
                             "colour, to separate semantics from paint")
    parser.add_argument("--position-mode", default="rope3d",
                        choices=["rope3d", "learned_absolute", "none"])
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--train-rooms", type=int, default=0,
                        help="cap on training rooms; 0 uses every remaining room")
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--metres-per-cycle", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    rooms = available_rooms()
    shuffled = list(rooms)
    # The split is fixed across every run so the scaling curve and the ablation
    # are measured on exactly the same held-out rooms.
    random.Random(20260818).shuffle(shuffled)
    held_out = sorted(shuffled[: args.holdout])
    train_rooms = shuffled[args.holdout :]
    if args.train_rooms:
        train_rooms = train_rooms[: args.train_rooms]
    train_rooms = sorted(train_rooms)

    gather = {
        "object": lambda rooms: collect(rooms, token_budget=args.token_budget),
        "relational": lambda rooms: collect_relational(
            rooms, token_budget=args.token_budget
        ),
        "both": lambda rooms: collect(rooms, token_budget=args.token_budget)
        + collect_relational(rooms, token_budget=args.token_budget),
    }[args.task]
    train = gather(train_rooms)
    test = gather(held_out)
    if not train or not test:
        raise SystemExit("no grounding examples were produced")
    print(f"rooms: {len(train_rooms)} train, {len(held_out)} held out")
    print(f"examples: {len(train)} train, {len(test)} held out")

    vectors = load_phrase_vectors()
    missing = {e.phrase for e in train + test} - set(vectors)
    if missing:
        raise SystemExit(f"phrase cache is stale, missing {len(missing)}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PointGroundingModel(
        feature_dim=train[0].features.shape[1],
        model_dim=args.model_dim,
        heads=args.heads,
        layers=args.layers,
        metres_per_cycle=args.metres_per_cycle,
        dropout=args.dropout,
        position_mode=args.position_mode,
    ).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{args.position_mode}: {parameters/1e6:.2f}M parameters")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=0.01)
    steps = args.epochs * max(1, math.ceil(len(train) / args.batch_size))
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: (step + 1) / warmup if step < warmup
        else 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))),
    )

    started = time.time()
    skipped = 0
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            chunk = [train[i] for i in order[start : start + args.batch_size]]
            points, features, query, target = stack(
                chunk, vectors, device, None if args.no_augment else rng,
                feature_mode=args.feature_mode,
            )
            loss = soft_cross_entropy(model(features, points, query), target)
            if not torch.isfinite(loss):
                skipped += 1
                optimiser.zero_grad(set_to_none=True)
                continue
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                skipped += 1
                optimiser.zero_grad(set_to_none=True)
                continue
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:4d}  loss {total/max(1,len(order)/args.batch_size):.4f}")

    report = {
        "task": args.task,
        "feature_mode": args.feature_mode,
        "position_mode": args.position_mode,
        "train_rooms": train_rooms,
        "held_out_rooms": held_out,
        "train_room_count": len(train_rooms),
        "train_examples": len(train),
        "token_budget": args.token_budget,
        "parameters": parameters,
        "augmented": not args.no_augment,
        "epochs": args.epochs,
        "seed": args.seed,
        "skipped_nonfinite_steps": skipped,
        "train_minutes": round((time.time() - started) / 60.0, 2),
        "train_fit": evaluate(model, train, vectors, device,
                              feature_mode=args.feature_mode),
        "held_out": evaluate(model, test, vectors, device,
                             feature_mode=args.feature_mode),
    }
    print(json.dumps(report["held_out"], indent=2))

    if args.out:
        save_model(args.out, model, {
            "feature_dim": train[0].features.shape[1],
            "model_dim": args.model_dim,
            "heads": args.heads,
            "layers": args.layers,
            "metres_per_cycle": args.metres_per_cycle,
            "position_mode": args.position_mode,
            "token_budget": args.token_budget,
            "held_out_rooms": held_out,
            "held_out": report["held_out"],
        })
    if args.report:
        destination = PROJECT_ROOT / args.report
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

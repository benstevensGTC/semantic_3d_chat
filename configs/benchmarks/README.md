# Local-Gemma navigation benchmark

This benchmark supports two deliberately distinct policy paths:

- the original, explicitly **untrained** Gemma JSON-generation seam, retained as
  a control;
- the gated, supervised continuous navigation controller, which consumes the
  same question-independent full-scene prefix, numeric robot-state tokens, and
  the user's instruction.

The learned controller does not receive captions, object labels, oracle target
IDs, or scorer metadata. Its sanitized checkpoint contains exactly
`policy.safetensors` and `runtime_metadata.json`, and the loader rejects training
paths, oracle paths, extra files, symlinks, hash mismatches, and non-finite
tensors.

The two commands are intentionally separate processes. The inference command
blocks all oracle, QA, training, and scorer-only paths before opening the local
Gemma model. It receives one user instruction, the continuous full-scene plus
numeric robot prefix, and bounded numeric action receipts. It never receives a
target instance ID, object box, expected direction, category inventory, scene
caption, or simulator relationship.

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv-gemma4/bin/python scripts/run_llm_navigation_inference.py
```

To reproduce a learned-policy run and then score it in the separate oracle
process, use a fresh opaque run ID:

```bash
NAVIGATION_POLICY_RUN_ID=learned_v1_reproduction \
  make navigation-policy-benchmark
```

The launcher refuses to overwrite an existing journal, audit, score, or map.
The policy's offline lifecycle is independently rerunnable with:

```bash
make navigation-policy-generate-traces
make navigation-policy-train
make navigation-policy-evaluate
make navigation-policy-controls
make navigation-policy-audit
```

If the process is interrupted after an episode is sealed, resume it explicitly:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv-gemma4/bin/python scripts/run_llm_navigation_inference.py --resume
```

Only after inference exits with a complete, hash-validated prediction journal,
run the oracle scorer:

```bash
PYTHONPATH=src .venv-gemma4/bin/python scripts/score_llm_navigation.py
```

The runtime task file is
`configs/benchmarks/llm_navigation_scene_000001.json`. It contains only opaque
task IDs, task-family protocol values, and literal user instructions. The
physically separate sidecar at
`configs/benchmarks/oracle/llm_navigation_scene_000001.json` contains target IDs
and thresholds and is hash-bound to the runtime task file. The inference audit
blocks the entire `oracle` component. The scorer also opens the generated scene
oracle to obtain bounding boxes.

The benchmark covers:

- facing a target;
- approaching and stopping near a target;
- explicit displacement and stop behavior;
- collision-free navigation around an obstacle;
- initial left/right turn choice and final heading;
- scanning, refreshing the persistent map/prefix, and acting from that update.

Every model output remains a proposal until strict JSON, static bounds, dynamic
pose bounds, and context hashes validate it. Raw model output is not retained.
Each step contributes to a transcript hash chain, each episode has a canonical
hash, and the complete predictions journal has a canonical root hash. The
scorer refuses incomplete or tampered journals before it opens any oracle file.
It also refuses a nominally complete journal until the inference process has
attached a clean blocking file-audit receipt. The robot-state token projector
is deterministic and frozen. The legacy Gemma JSON seam remains task-untrained;
the learned path is identified by a hash-attested supervised-policy status in
the sealed journal and scorer output.

The first accepted controller (`navigation_policy_v1`) reached 87.87% held-out
offline action accuracy, 87.80% stop recall, 87.62% turn-sign accuracy, and
0.250 m argument MAE over eight scene-disjoint validation scenes. In the sealed
six-task live benchmark it completed 3/6 tasks, up from 0/6 for the untrained
control, with zero collisions, zero action failures, and zero policy
rejections. This is measurable progress, not a finished navigation result: the
approach, explicit-displacement, and update-after-scan tasks remain documented
failures.

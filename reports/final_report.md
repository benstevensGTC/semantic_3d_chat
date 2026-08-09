# Semantic 3D Chat — First Proof-of-Concept Report

> **PRIMARY-RESULT WARNING — LEGACY REPORT BODY.** The CLIP/Qwen tables and chat
> evidence preserved below are superseded as the project's primary model path. They
> remain useful failure/control evidence, but they are not Gemma 4 results and do
> not establish scene-conditioned understanding. The current primary Gemma 4 E2B
> path has a working dense 3D map and semantic-localization result. V9 passed its
> strict teacher-forced full-vocabulary gate and both epoch 30 and epoch 36
> free-generate all 12 trained color sides exactly, while scoring 0/70 exact mirror
> sides and 0/8 exact held-out cube-support sides. V10 completed its deterministic
> weights-only color-plus-mirror continuation but failed: both audited checkpoints
> regressed to 9/12 exact color sides, learned 0/6 selected mirror units, and scored
> 0/8 on held-out support. V11 restored 12/12 color sides with a full-vocabulary
> margin, but reached only 3/12 selected mirror sides, 0/6 complete mirror units,
> 7/70 all-mirror sides, and 1/8 held-out support sides. V12 preserved 12/12 color
> sides but reached 0/12 selected mirror sides, 0/70 all-mirror sides, and 0/8
> held-out support sides even after its ordered spatial auxiliary loss passed its
> own margin target. V13 passed its decoder-bank integrity, training, checkpoint-
> reload, and runtime-prefix checks and retained 12/12 trained color sides, but it
> still reached 0/12 selected mirror sides, 0/70 all-mirror sides, and 0/8 held-out
> support sides. All 70 mirror outputs were literal `unknown` responses. V13 is not
> promoted. There is no accepted Gemma static-chat, Gemma leakage, semantic
> embodied-agent, or one-command-demo result yet.

Updated from local artifacts on `2026-08-09T10:40:47Z`. This report does not run models and does not infer missing measurements.

## 1. Research question

Can a local language model answer questions about a synthetic room when the environment reaches it only as continuous, spatially fused visual embeddings and geometry—without a caption, object list, textual scene graph, simulator labels, or question-dependent retrieval?

### Current primary Gemma 4 E2B outcome

The primary representation is 24 complete-image Gemma observations, each retaining
a 48×48 field of middle 768D, late 768D, and native projected 1536D features
(3072D float16 total), fused into a 5 cm persistent map and deterministically
aggregated to 15 cm tokenizer input. Every occupied block feeds the
question-independent `signal_preserving_resampler_v3`, producing 256 scene
latents at model dimension 384 before projection to Gemma's 1536D hidden space.

#### v9 hardened color gate: exact trained generation, zero control transfer

V9 trained the same 45,056 layer-34 LoRA parameters as v8, but required the
canonical first answer token to be the unique full-vocabulary top-1 token on every
side in addition to passing pairwise candidate ranking. The fresh MPS run completed
36 epochs, 216 decoder microsteps, and 36 optimizer updates in 1,539.387 seconds.
The pairwise gate was already 12/12 at epoch 22, when the full-vocabulary check was
only 1/12. The composite gate first passed at epoch 30: 12/12 top-1 sides, 6/6
complete units, mean target-versus-best-other logit margin 1.500651, and minimum
margin 0.03125. Epoch 36 remained 12/12 and 6/6 while strengthening those margins
to mean 2.903809 and minimum 1.0. The exact trace is
`gemma4/metrics/training_gemma4_color_wiring_v9.json`.

Model-validated greedy decoding produced identical normalized-exact results for
the stored epoch-30 `best` checkpoint and epoch 36:

| Intervention | Split status | Exact sides | Exact complete units | Changed predictions |
| --- | --- | ---: | ---: | ---: |
| Color swap | trained | 12/12 | 6/6 | 6/6 |
| Mirrored left/right room | not trained by v9 | 0/70 | 0/35 | 0/35 |
| Cube on/under support | held-out test | 0/8 | 0/4 | 0/4 |

The hardened checkpoint audits are
`gemma4/metrics/scene_signal_audit_gemma4_color_wiring_v9_best_epoch30.json` and
`gemma4/metrics/scene_signal_audit_gemma4_color_wiring_v9_epoch036.json`.
Prediction change is deliberately separate from exact correctness. The structural
metrics do not support repeating the historical Perceiver-collapse diagnosis for
v9: native-latent/block-token relative-L2 retention is 0.622951–1.002638 for epoch
30, and its native-latent mean off-diagonal cosine is 0.917063–0.940826. The
measured failure is behavioral non-transfer beyond the trained color intervention,
not disappearance of all scene signal in the resampler.

No `promotion.json` was created. V9 therefore establishes a successful trained
color-wiring overfit only. It does not establish held-out static QA, interactive
chat, Gemma prefix invariance, Gemma oracle-deletion/leakage isolation, or
language-conditioned semantic robot navigation.

#### v10 deterministic color-plus-mirror continuation: failed gate and forgetting

V10 started from v9 epoch 36 as `weights_only_new_curriculum`: it restored
compatible adapter/scene-prefix weights but reset optimizer state, epoch history,
and curriculum. The initialization record binds the source v9 checkpoint and
adapter hash
`8ecbf84fc8f544d67fe3e65a313023c3808870c5648b913de2839ec525630c90`, and confirms
that neither optimizer state nor history was loaded. Its audited selection
contained six complete color units and six complete mirror units, or 24 records
across four opaque scene IDs. The cap was a deterministic seed hash over opaque
pair/question keys; it did not inspect question text or answers, and both scene
sides remained indivisible. Held-out cube support was not trained.

V9 exposed a checkpoint-selection defect: the original full-vocabulary hinge
reached zero at the first pass, so epoch 30 remained `best` even though epoch 36
had a much stronger minimum margin. The corrected selector still ranks every
passing checkpoint above every failure, then ranks passes by negative minimum
target-versus-best-other margin. Later, more robust passes can now replace a barely
positive first pass. The run inputs are
`configs/experiments/gemma4_color_mirror_wiring_v10.yaml` and
`gemma4/metrics/training_selection_gemma4_color_mirror_wiring_v10.json`.

The MPS run completed 12 epochs, 144 decoder microsteps, and 12 optimizer updates
in 933.685 seconds. Its strict composite gate never passed. The monitor selected
epoch 8 as `best`, where the full-vocabulary first-answer check reached only 9/24
sides and 3/12 complete units; final epoch 12 had the same counts. The final
minimum target-versus-best-other margin was still -9.75.

Model-validated greedy generation measured:

| Checkpoint and intervention | Training status | Exact sides | Exact complete units | Changed predictions |
| --- | --- | ---: | ---: | ---: |
| Best epoch 8 — color swap | selected | 9/12 | 3/6 | 6/6 |
| Best epoch 8 — mirror subset | selected | 0/12 | 0/6 | 0/6 |
| Best epoch 8 — all mirror units | selected + unselected | 1/70 | 0/35 | 3/35 |
| Best epoch 8 — cube support | held-out test | 0/8 | 0/4 | 1/4 |
| Final epoch 12 — color swap | selected | 9/12 | 3/6 | 6/6 |
| Final epoch 12 — mirror subset | selected | 0/12 | 0/6 | 0/6 |
| Final epoch 12 — all mirror units | selected + unselected | 0/70 | 0/35 | 1/35 |
| Final epoch 12 — cube support | held-out test | 0/8 | 0/4 | 2/4 |

For the 29 unselected mirror units, best epoch 8 scored 1/58 exact sides and
changed 3/29 predictions; final epoch 12 scored 0/58 and changed 1/29. Prediction
change is reported separately because changing between two wrong answers is not a
success. V10 therefore partially forgot v9's previously exact 12/12 color behavior
without learning any selected mirror unit or any held-out support side. Exact
artifacts are `gemma4/metrics/training_gemma4_color_mirror_wiring_v10.json`,
`gemma4/metrics/scene_signal_audit_gemma4_color_mirror_wiring_v10_best_epoch8.json`,
and `gemma4/metrics/scene_signal_audit_gemma4_color_mirror_wiring_v10_epoch012.json`.
No `promotion.json` was created.

#### v11 full-vocabulary retry: color restored, mirror still failed

The controlled v11 retry restarted from the same v9 epoch-36 checkpoint with the
same six color and six mirror units. It retains the candidate-pair hinge and adds
a differentiable first-answer objective
`relu(1 - (target_logit - max_non_target_logit))` at weight 2. This directly
targets v10's observed candidate-versus-full-vocabulary gap using the same decoder
forward and existing answer supervision. The run completed 12 epochs, 144 decoder
microsteps, and 12 optimizer updates in 947.428 seconds. Best and final are the
same epoch-12 adapter, SHA-256
`eee7b3aa8ce2e7584cfe1fc80d8852d4d645b24c156ccd43369cb4ba7e047e22`.

The strict teacher gate failed: color reached 12/12 candidate sides and 12/12
full-vocabulary top-1 sides (6/6 complete units), but mirror stayed at 6/12
candidate sides and 0/6 units, with only 3/12 full-vocabulary sides and 0/6 units.
Greedy generation confirmed the failure:

| Intervention | Training status | Exact sides | Exact complete units | Changed predictions |
| --- | --- | ---: | ---: | ---: |
| Color swap | selected | 12/12 | 6/6 | 6/6 |
| Mirror subset | selected | 3/12 | 0/6 | 2/6 |
| Mirror, all units | selected + unselected | 7/70 | 0/35 | 10/35 |
| Cube support | held-out test | 1/8 | 0/4 | 4/4 |

Strict normalized exact is the promotion metric. Secondary canonical relation
parsing counts unambiguous verbose answers and reaches 28/70 mirror sides and 5/35
complete units, but all five complete units are unselected; the selected subset
remains 0/6. Output decisiveness did improve: V11 substantially reduced the prior
`unknown` mode. That did not route mirror-prefix differences into the correct
left/right contrast. The mirror prefix itself remains measurably different across
scenes (0.358191 relative L2; 98.55% changed elements), so this is not evidence of
global scene-token collapse. Exact artifacts are
`gemma4/metrics/training_gemma4_color_mirror_full_vocab_v11.json` and
`gemma4/metrics/scene_signal_audit_gemma4_color_mirror_full_vocab_v11_epoch012.json`.
No `promotion.json` was created.

#### v12 ordered-relation retry: auxiliary margin passed, decoder still failed

V12 used
`configs/experiments/gemma4_color_mirror_spatial_relation_v12.yaml`. The old
balanced hinge has an exact shared-preference saddle: margins `[d, -d]` yield zero
gradient while both sides violate the margin. V12 kept the v9 initialization,
selection, decoder objective, and schedule, and added an ordered
target-minus-reference objective over dense soft pools of all 256 scene latents.
The ordered coordinates are training/evaluation-only QA fields. They never enter
the chat runtime, which retains the same global question-independent prefix.

The auxiliary objective succeeded on its own terms. Its scene-only warmup stopped
after 21 forward passes and 20 optimizer steps when all 12 eligible mirror sides
exceeded the configured 0.1 margin (mean 0.246168; minimum 0.116128). The complete
run then finished 12 epochs, 144 decoder microsteps, and 12 main optimizer updates
in 1,073.493 seconds (17m 53.493s). The recorded source scope was clean at commit
`6837426d2f8c943ae08646f17ff521d7df3d29c4`.

The teacher-forced decoder result must be distinguished from the auxiliary loss
and from actual generation. At both epochs 8 and 12, color reached 12/12 candidate
sides and 12/12 full-vocabulary first-token sides (6/6 complete units). Mirror
remained at 6/12 candidate sides and 0/6 units, while its full-vocabulary score was
0/12 sides and 0/6 units. Thus the spatial objective separated its dense pooled
representations, but Gemma's next-token distribution did not consume that signal
as the required relation.

Both model-validated greedy audits measured:

| Checkpoint and intervention | Training status | Exact sides | Exact complete units | Changed predictions |
| --- | --- | ---: | ---: | ---: |
| Epoch 8 — color swap | selected | 12/12 | 6/6 | 6/6 |
| Epoch 8 — mirror subset | selected | 0/12 | 0/6 | 0/6 |
| Epoch 8 — all mirror units | selected + unselected | 0/70 | 0/35 | 0/35 |
| Epoch 8 — cube support | held-out test | 0/8 | 0/4 | 0/4 |
| Final/best epoch 12 — color swap | selected | 12/12 | 6/6 | 6/6 |
| Final/best epoch 12 — mirror subset | selected | 0/12 | 0/6 | 0/6 |
| Final/best epoch 12 — all mirror units | selected + unselected | 0/70 | 0/35 | 0/35 |
| Final/best epoch 12 — cube support | held-out test | 0/8 | 0/4 | 0/4 |

Every one of the 70 mirror outputs was the literal model response `unknown`.
Neither audit used an answer fallback, observed an empty decode, nor exhausted its
token budget. Both have zero checkpoint-contract warnings and validate native
boundary embeddings, BF16 runtime dtype, and model-runtime prefix parity. These
checks establish a clean execution contract, not behavioral correctness or a
Gemma oracle-deletion result. Epoch 8's adapter SHA-256 is
`a4c85c14a214e4e594992e489a784cb4bacb64d3dfda519ad3da18b1595d9f22`;
final/best epoch 12 is
`1d46e754873431b11e8dc58066f08f06c17e3bcaa4c47b139358bb0f28ceabb1`.

The exact local artifacts are
`gemma4/metrics/training_gemma4_color_mirror_spatial_relation_v12.json`,
`gemma4/metrics/scene_signal_audit_gemma4_color_mirror_spatial_relation_v12_epoch008.json`,
and
`gemma4/metrics/scene_signal_audit_gemma4_color_mirror_spatial_relation_v12_best.json`.
No `promotion.json` was created. The honest diagnosis is narrower than “the scene
encoder collapsed”: the ordered auxiliary head can discriminate the selected
regions, while the current shallow shared decoder adaptation fails to turn that
discrimination into left/right tokens and retreats to `unknown`.

#### v13 frozen-scene decoder banks: execution contract passed, behavior failed

V13 is the bounded decoder-capacity falsifier defined in
[`../configs/experiments/gemma4_color_mirror_decoder_banks_v13.yaml`](../configs/experiments/gemma4_color_mirror_decoder_banks_v13.yaml).
Its pre-run probe used the pinned V12 epoch-8 adapter SHA-256
`a4c85c14a214e4e594992e489a784cb4bacb64d3dfda519ad3da18b1595d9f22`,
froze the complete scene adapter and inherited rank-4 layer-34 q/o LoRA, and
installed a disjoint exact-zero-output rank-8 q/o bank in layers 30-33. The new
bank contains 229,376 parameters. All 12 probed first-answer vocabulary
distributions were bitwise identical to the bank-free baseline; no optimizer was
constructed, zero updates occurred, and no protected state changed. The weighted
candidate-hinge gradients across all six mirror units had cancellation ratio
0.987037 and cosine 0.948397; the complete decoder objective measured 0.987413 and
0.949781. This passed the pre-run non-cancellation falsifier but did not predict
the eventual behavioral outcome. The exact no-update artifact is
[`gemma4/metrics/mirrored_gradient_probe_v13_epoch008.json`](gemma4/metrics/mirrored_gradient_probe_v13_epoch008.json),
SHA-256 `59638470edf63a8c8b4a450f3a833a7084c171a4147334366fa5016e709533e6`.

Source commit `990589363b42b2cd3451ec24f7a912ffac8411f6` implements a
schema-2 named-bank contract while preserving legacy schema-1 checkpoint loading.
It rejects overlapping bank targets, binds deterministic initialization and
per-bank compact-state hashes, places only trainable banks in the optimizer, and
checks frozen state before checkpoint writes and after reload. V13 initializes
from V12 epoch 8, whose metadata SHA-256 is
`f097c6477546460440e77a3d225afb55818cb13abf9cbb4a90500f75a879b0f5`.
The immutable scene state is
`690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b`;
the frozen inherited bank is
`dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594`.
The extension bank uses seed 13008 and begins at
`b4ec0518e4759dda33fc93c9c1d4c76f52f1024fd5b8b1667ad1b4ef5da198af`;
after training it is
`caaf9b13c13b2371463a2cf9d450453f925846b0202bdb0610103b6aa85e435b`.
The checkpoint contains 274,432 LoRA parameters in total, of which exactly
229,376 are trainable. Gemma base weights and all scene-adapter weights remain
outside the optimizer.

A separate one-update smoke run completed 12 microsteps in 44.822 seconds and
preserved both frozen hashes while changing the extension hash. This smoke is
training-wiring evidence only; it is not included in behavioral accuracy. Its
machine-readable training trace is
[`gemma4/metrics/training_gemma4_color_mirror_decoder_banks_v13_smoke.json`](gemma4/metrics/training_gemma4_color_mirror_decoder_banks_v13_smoke.json).
Its deterministic selection trace is
[`gemma4/metrics/training_selection_gemma4_color_mirror_decoder_banks_v13_smoke.json`](gemma4/metrics/training_selection_gemma4_color_mirror_decoder_banks_v13_smoke.json).

The full MPS run completed 12 epochs, 144 decoder microsteps, and 12 optimizer
updates in 467.248 seconds. Its source provenance is clean at commit `9905893`,
and it uses the same deterministic 24-record, 12-unit color-plus-mirror selection
as V10-V12. The discrete teacher-gate counts were unchanged at every measured
gate (epochs 2, 4, 6, 8, 10, and 12):

| Intervention | Candidate sides | Candidate units | Full-vocabulary sides | Full-vocabulary units |
| --- | ---: | ---: | ---: | ---: |
| Color swap | 12/12 | 6/6 | 12/12 | 6/6 |
| Mirror subset | 6/12 | 0/6 | 0/12 | 0/6 |

The aggregate gate therefore never passed, although mirror margins improved
measurably: candidate minimum moved from -4.625 at epoch 2 to -4.125 at epoch 12,
and full-vocabulary mean moved from -19.880209 to -19.231771. Epoch 12 is both
`best` and final;
those directories contain byte-identical adapter and metadata files. Adapter
SHA-256 is
`9b59d15ba9e4d3be8d8a64ea6d9d3071d1e8650333ee8c21c5504e7900353c7c`,
and metadata SHA-256 is
`83ba42f6fc5b8ef2025588f35a3a2bba9a9d7e4074487d85c43ef5b25fc13a7b`.
Exact selection and training evidence is
[`gemma4/metrics/training_selection_gemma4_color_mirror_decoder_banks_v13.json`](gemma4/metrics/training_selection_gemma4_color_mirror_decoder_banks_v13.json)
and
[`gemma4/metrics/training_gemma4_color_mirror_decoder_banks_v13.json`](gemma4/metrics/training_gemma4_color_mirror_decoder_banks_v13.json)
(training file SHA-256
`f86659cbc3ab4407c82d583f2e846c9acf6713c82d89c5dfaded4d91bed6a79c`).

Saved-and-reloaded, model-validated BF16 greedy generation measured:

| Intervention | Training status | Exact sides | Exact complete units | Changed predictions |
| --- | --- | ---: | ---: | ---: |
| Color swap | selected | 12/12 | 6/6 | 6/6 |
| Mirror subset | selected | 0/12 | 0/6 | 0/6 |
| Mirror, all units | selected + unselected | 0/70 | 0/35 | 0/35 |
| Cube support | held-out test | 0/8 | 0/4 | 0/4 |

All 70 mirror outputs were the literal model response `unknown`. None used an
audit answer fallback, decoded to an empty string, or exhausted the 32-token
budget. The eight support outputs were six `unknown` responses and two copies of
`The floor is light gray.`; none was exact. Thus neither fluency nor a changed
wrong answer inflates the strict scores.

The first audit attempt correctly stopped on a constructor defect in the audit
harness: it created FP32 native-boundary placeholders before loading the
checkpoint, silently casting persisted BF16 BOI/EOI buffers and causing the frozen
scene-state hash check to fail. It did not mutate the saved checkpoint or affect
training. Fix commit `a10579d6dadecf8082cc179a201bb1db517656aa`
constructs native boundaries from the loaded model, or directly in configured BF16
for skip-generation audits. The complete rerun has zero contract warnings,
validates the model's BF16 runtime dtype and native boundaries, and confirms exact
runtime-prefix parity. The audit is
[`gemma4/metrics/scene_signal_audit_gemma4_color_mirror_decoder_banks_v13_best.json`](gemma4/metrics/scene_signal_audit_gemma4_color_mirror_decoder_banks_v13_best.json),
file SHA-256 `042fd7c5b085858ac334aaf40f08533e8f2db1ab0ef258917fa901be224256dc`.

After the boundary-dtype repair, the V13 tree passed the standard full suite with
310 tests passed and 15 skipped; a broader Gemma-focused suite passed 92 tests
with two benign SWIG warnings. Focused standard tests passed 17/17 and Gemma
audit/boundary tests passed 26/26. These results validate the implementation and
checkpoint contract, not scene understanding. No `promotion.json` was created.
V13 therefore falsifies this added-decoder-bank configuration: it preserves the
trained color overfit but learns no complete mirror unit and transfers to no
held-out support side. Held-out static QA, interactive chat, Gemma-specific
leakage/prefix-invariance tests, and semantic robot navigation remain gated.

The earlier v7 and v8 results remain below as historical adapter lineage.

Gemma v7 completed 12 epochs, 72 decoder microsteps, and 12 optimizer updates in
567.844 seconds. The best checkpoint was epoch 7: 3/6 changed units, 9/12 correct
sides, 3/6 prediction flips, 3/6 wrong-prefix flips, mean ranking margin 0.439616,
minimum margin -0.71875, and hinge 0.314453. The final checkpoint again reached
3/6 changed units and 9/12 sides, with hinge 0.386068 and minimum margin -1.484375.
The required teacher-forced counterfactual gate therefore failed. The higher
4/6 changed-unit count at epoch 3 also failed because its hinge was 0.716146 and its
minimum margin was -3.59375. Runs v1-v6 and the v6 resumes through epochs 18 and 24
also failed; none is promoted.

The exact source hash loaded by the already-running v7 process was not captured.
The implementation hashes in
`gemma4/metrics/gemma4_color_wiring_v7_failure.json` are post-run audited snapshots
and do not prove that later padding or audit/resume fixes executed in v7.

#### v8 controlled fallback: teacher-forced overfit passed, free generation failed

The controlled v8 fallback leaves the native boundaries and complete continuous
scene prefix unchanged. It adapts only layer 34
`model.language_model.layers.34.self_attn.q_proj` and `o_proj`, using rank 4,
alpha 8, dropout 0, LoRA learning rate `1e-4`, and weight decay 0. This is 45,056
FP32 A/B-only parameters (180,224 bytes, approximately 176 KiB). Strict config,
optimizer, checkpoint, SHA/tamper, resume, chat-load, and scene-signal-audit paths
have test coverage.

The fresh MPS run completed epoch 12 in 574.56 seconds and failed the gate: 4/6
changed units, 10/12 correct sides, and candidate-ranking hinge 0.272786. Its exact
result is `gemma4/metrics/training_gemma4_color_wiring_v8.json`. A controlled resume
from that checkpoint stopped early at epoch 22 after 22 total optimizer updates.
The resumed portion took 388.31 seconds and passed the teacher-forced gate with 6/6
changed units, 12/12 correct sides, prediction-flip rate 1.0, wrong-prefix-flip
rate 1.0, and minimum candidate margin 0.0390625. Its exact result is
`gemma4/metrics/training_gemma4_color_wiring_v8_resume24.json`.

This pass is a same-distribution teacher-forced wiring/overfit result only. The
model-validated free-generation audit changed outputs for 5/6 training color-swap
questions, but canonical correctness remained poor: responses were mostly
`orange` or `unknown`, with only isolated correct colors. It changed 0/35 answers
for the mirrored-room pair and 0/4 for the held-out cube-support pair. The exact
decoded outputs and expected answers are in
`gemma4/metrics/scene_signal_audit_gemma4_color_wiring_v8_resume24.json`. Because
the learned behavior did not transfer to those controls, no `promotion.json` was
created and static held-out QA, interactive chat, prefix/oracle-deletion inference,
and language-conditioned robot navigation remain gated.

Reproduce selection, fresh training, and the bounded resume with:

```bash
PYTHONPATH=src .venv-gemma4/bin/python -m semantic_3d_chat.training.train_adapter \
  --config configs/experiments/gemma4_color_wiring_v8.yaml --selection-only
PYTHONPATH=src .venv-gemma4/bin/python -m semantic_3d_chat.training.train_adapter \
  --config configs/experiments/gemma4_color_wiring_v8.yaml
PYTHONPATH=src .venv-gemma4/bin/python -m semantic_3d_chat.training.train_adapter \
  --config configs/experiments/gemma4_color_wiring_v8.yaml \
  --resume data_gemma4/checkpoints/gemma4_color_wiring_v8/epoch_012 \
  --epochs 24 --output-namespace gemma4_color_wiring_v8_resume24
```

The current Gemma semantic prerequisite did pass on `scene_000001`: 61.54% top-1,
84.62% hit@100, and 45.23% precision@100 versus 8.07% random precision; same-voxel
cosine was 0.5889 versus 0.4009 for different voxels. This validates useful map
signal, not language-model scene understanding.

### Preserved legacy CLIP/Qwen evidence

The preserved legacy data path satisfies the representation constraint: the runtime consumes continuous scene features and geometry, not environmental text. Held-out QA is measured on 274 test records at 74.8% exact accuracy. The v1 controls invalidate that raw score as evidence of scene-specific understanding: wrong-scene and shuffled-content prefixes matched or exceeded the primary path, and changed-fact consistency was zero. The v2 resampler has a structural, no-training signal-preservation diagnostic, but no v2 held-out behavioral result is yet associated with the report artifacts.

## 2. Preserved legacy CLIP/Qwen architecture

![Continuous scene-memory architecture](figures/architecture.png)

The scan is rendered with exact metric camera-Z depth, intrinsics, and camera-to-world poses. Each complete RGB image is encoded once. Middle 768-D, late 768-D, and MaskCLIP-value-aligned 512-D patch streams form a 2,048-D feature. Weighted voxel fusion builds the persistent map before any question. Every occupied block contributes to the question-independent global scene-token set. The selected encoder architecture is `spatial_coverage_resampler_v2`, projected directly into the local LM embedding space.

## 3. Hardware used

| Item | Measured value |
| --- | --- |
| Architecture | arm64 |
| Processor identifier | arm |
| Logical / physical CPUs | 12 / 12 |
| Unified memory | 24.0 GiB |
| Free disk at inspection | 136.0 GiB |
| PyTorch MPS built / available / smoke | True / True / True |

The exact Apple chip model is not present in the current machine-report JSON; the report therefore does not guess it.

## 4. Software versions

| Component | Version / revision |
| --- | --- |
| macOS | 26.5 |
| Python | 3.12.13 |
| Blender | Blender 5.2.0 LTS |
| uv | uv 0.10.11 (Homebrew 2026-03-16) |
| PyTorch | 2.13.0 |
| Vision weights | openai/clip-vit-base-patch16 @ 57c216476eefef5ab752ec549e440a49ae4ae5f3 |
| Language weights | Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775 |

## 5. Preserved legacy vision encoder

`openai/clip-vit-base-patch16` at pinned revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`. One complete 224×224 image produces a localized 14×14 patch grid; no manual patch crops are independently encoded. The preserved legacy aligned slice uses `maskclip_value`.

## 6. Preserved legacy language model

`Qwen/Qwen2.5-0.5B-Instruct` at pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`. Scene latents are passed through `inputs_embeds`; no scene caption or decoded object list is interposed. CLIP is MIT-licensed and Qwen2.5 is Apache-2.0 according to the project records.

## 7–11. Preserved legacy representation dimensions and scan scale

| Parameter | Value |
| --- | --- |
| Scan images | 24 |
| Render resolution | 224 × 224 |
| Feature layout | middle 768 + late 768 + aligned 512 = 2,048 |
| Aligned method | maskclip_value |
| Stored semantic dtype | float16 |
| Voxel size | 0.050 m |
| Occupied voxels | 74,699 |
| Raw observations | 301,056 |
| Tokenizer input voxels | 8,422 |
| Occupied spatial blocks | 3,019 |
| Global scene latents | 256 |
| Scene encoder dimension | 384 |
| LM hidden dimension | 896 |
| Continuous prefix shape | [1, 258, 896] |

![Camera scan montage](figures/scan_montage.png)

## 12–13. Training dataset and split

QA records: `train=823, validation=260, test=274`. Scene split metadata: `{"test": ["scene_000005", "scene_000006"], "train": ["scene_000001", "scene_000002", "scene_000003", "scene_000004", "scene_000007", "scene_000008"], "validation": ["scene_000009", "scene_000010"]}`.


## 14. Preserved legacy training

| Measurement | Value |
| --- | --- |
| Selected run namespace | multiscene_anticollapse |
| Scene-encoder architecture | spatial_coverage_resampler_v2 |
| Completed / target epochs | 2 / Not measured |
| Latest checkpoint epoch | 2 |
| Latest checkpoint train loss | 0.962692 |
| Best checkpoint epoch | 2 |
| Best checkpoint loss | 0.962692 |
| Best validation loss | 0.752224 |
| Scenes in checkpoint | 6 |
| Selected-run training time | Not measured |
| Peak memory | Not measured |

![Training loss curve](figures/training_loss.png)

### Adapter-generation lineage

- **v1 multi-scene:** 6 training scenes, 5 epochs, best validation loss `0.385679`, elapsed `903.9 s`.
- **v2 anti-collapse:** architecture `spatial_coverage_resampler_v2`; completed/target epochs `2 / Not measured`.
- The v1 held-out score is retained as a failure result, not promoted as evidence of scene-conditioned language behavior.
- The CPU-only, no-training v2 diagnostic increased projected pairwise scene-change magnitude by `255.9×–607.7×`. This is structural evidence only, not a QA result.

## 15. Preserved legacy static QA failure result

| Metric | Result |
| --- | --- |
| Normalized exact accuracy | 74.8% |
| Order-insensitive list accuracy | 0.0% |
| Count accuracy | 100.0% |
| Spatial-relation accuracy | 78.8% |
| Presence precision | 100.0% |
| Presence recall | 66.7% |

![Accuracy by question type](figures/accuracy_by_question_type.png)

Artifact lineage: **v1** (`/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/predictions/multiscene_test.jsonl`).

**Interpretation:** these are genuine held-out structured scores, but they do not demonstrate use of scene-specific content. The v1 wrong-scene, semantic-shuffle, position-shuffle, and geometry-only controls matched or slightly exceeded the primary score.

### Preserved legacy CLIP semantic-map prerequisite

| Metric | Observed | Random control | Lift |
| --- | --- | --- | --- |
| Top-1 localization | 61.5% | 8.1% | 53.5% |
| Hit@100 | 92.3% | 55.7% | 36.6% |
| Precision@100 | 54.9% | 8.1% | 46.9% |

Cross-view consistency: same-voxel cosine `0.7903` versus different-voxel `0.7223`; margin `0.0680` across `174,405` same-voxel view pairs.

![Semantic localization by category](figures/semantic_localization_by_category.png)

## 16. Preserved legacy counterfactual failure result

| Metric | Result |
| --- | --- |
| Eligible pairs | 134 |
| Pair accuracy | 73.9% |
| Changed when expected | 0.0% |
| Invariant when expected | 99.2% |

![Counterfactual consistency](figures/counterfactual_consistency.png)

**Failure:** none of the expected-change pairs changed answer. High aggregate pair accuracy is dominated by invariant pairs and must not be read as counterfactual success.

## 17. Preserved legacy grounding result

| Metric | Result |
| --- | --- |
| Grounding coverage | 100.0% |
| Mean coordinate error | 1.7023 m |
| Median coordinate error | 1.7134 m |
| Within 0.5 m | 0.0% |

## 18. Preserved legacy ablation results

| Condition | Exact | Spatial relation | Changed when expected | Exact Δ vs primary |
| --- | --- | --- | --- | --- |
| primary | 74.8% | 78.8% | 0.0% | 0.0000 |
| wrong_scene_prefix | 75.2% | 79.4% | 0.0% | 0.0036 |
| semantic_shuffle | 75.5% | 79.4% | 0.0% | 0.0073 |
| position_shuffle | 75.2% | 79.4% | 0.0% | 0.0036 |
| geometry_only | 75.2% | 79.4% | 25.0% | 0.0036 |
| semantics_without_xyz | 75.2% | 79.4% | 0.0% | 0.0036 |
| remove_rgb | 75.2% | 79.4% | 0.0% | 0.0036 |
| remove_normals | 74.8% | 78.8% | 0.0% | 0.0000 |
| empty_scene_prefix | 0.0% | 0.0% | 0.0% | -0.7482 |

![Ablation accuracy](figures/ablation_accuracy.png)

**Provenance limitation:** this aggregate does not record its checkpoint or architecture. The numeric rows are not automatically attributed to the selected v2 checkpoint.

Artifact-supplied interpretation: **A nonzero prefix is necessary, but the first multiscene checkpoint is insensitive to its scene-specific content.**

### v1 collapse diagnosis

The counterfactual signal is present in raw and aggregated maps and remains visible in spatial block tokens. The dominant loss occurs in the global Perceiver resampler: its 256 native latents are almost duplicate vectors, and scene-pair relative L2 falls by roughly two to three further orders of magnitude before LM projection. Distinct SHA-256 hashes therefore certify only bitwise inequality, not a behaviorally useful separation.
Measured raw-to-projected attenuation: `28549.6×–56551.4×`.

### v2 structural diagnostic

The v2 artifact compares the legacy checkpoint under the old and new resamplers without retraining. It tests signal preservation only; it must not be cited as held-out language behavior.
| Pair change | Native signal gain | Projected signal gain | v2 native latent cosine | v2 projected cosine |
| --- | --- | --- | --- | --- |
| color_swap | 30.3 | 312.5 | 0.8734 | 0.9328 |
| mirror_lr | 63.8 | 607.7 | 0.8734 | 0.9328 |
| cube_support | 25.9 | 255.9 | 0.8734 | 0.9328 |

## 19. Direct multi-view image baseline

Not measured.


## 20. Oracle-text upper bound

Not measured.


## 21. Preserved legacy leakage-test result

| Control | Result |
| --- | --- |
| Overall leakage test | PASS |
| Oracle unavailable during inference | True |
| Oracle restored | True |
| Forbidden accesses | 0 |
| Prefix built before first question | True |
| Prefix invariant | True |
| Prefix hash | f4ed6bc9cbf75bd878bcebab04e432ad3eb04ef236aadd6659931c8f214b7c9b |
| Audited loaded files | 4347 |

This PASS belongs only to the legacy Qwen checkpoint lineage. It has not been run
for Gemma v9 and cannot be transferred to any Gemma checkpoint. V9's exact trained
color generation is not a Gemma leakage-test or oracle-deletion result.

## 22. Robot-navigation results

| Measurement | Result |
| --- | --- |
| Benchmark scope | bounded_numeric_actions_collision_scan_and_reset |
| Checks passed | 11 / 11 |
| Pass rate | 100.0% |
| Trajectory steps | 15 |
| MCP tools registered | 9 |
| MCP SDK | 2.0.0 |
| Semantic target navigation evaluated | False |

Artifact snapshot:

```json
{
  "benchmark_scope": "bounded_numeric_actions_collision_scan_and_reset",
  "checks": {
    "bounded_look_succeeded": true,
    "bounded_turn_succeeded": true,
    "collision_rejected_atomically": true,
    "free_space_move_succeeded": true,
    "mcp_structured_call_succeeded": true,
    "mcp_tools_registered": true,
    "numeric_start_state": true,
    "reset_restores_episode": true,
    "scan_updated_scene": true,
    "stop_blocks_motion": true,
    "turn_limit_rejected": true
  },
  "final_state": {
    "action_count": 0,
    "angular_velocity_degrees": 0.0,
    "body_yaw_degrees": 0.0,
    "camera_position_m": [
      0.0,
      0.0,
      1.2
    ],
    "camera_yaw_degrees": 0.0,
    "clearance_m": null,
    "collision": false,
    "distance_moved": 0.0,
    "error_code": null,
    "last_movement_delta_m": [
      0.0,
      0.0,
      0.0
    ],
    "linear_velocity_xy_m": [
      0.0,
      0.0
    ],
    "observation_id": null,
    "pitch_degrees": 0.0,
    "position_m": [
      0.0,
      0.0,
      0.0
    ],
    "scan_count": 0,
    "scan_coverage": 0.0,
    "scene_id": "scene_000001",
    "scene_version": 0,
    "seed": 20260808,
    "stopped": false,
    "success": true,
    "turn_degrees": 0.0,
    "visible_voxels": 0
  },
  "map_source": "numeric_voxel_map",
  "mcp_sdk_version": "2.0.0",
  "mcp_tool_count": 9,
  "metadata_or_labels_loaded": false,
  "pass_rate": 1.0,
  "passed": 11,
  "scene_id": "scene_000001",
  "schema_version": 1,
  "semantic_target_navigation_evaluated": false,
  "total": 11,
  "trajectory_steps": 15
}
```

This is a bounded numeric action, collision, scan-update, reset, and MCP wiring benchmark. It does **not** demonstrate that the chatbot can navigate to a named object or follow language-conditioned semantic directions.
The measured mechanics do not change the central limitation: language-conditioned semantic target navigation remains unmeasured.

## 23. Preserved legacy representative conversation

| Question | Answer | Grounding XYZ (m) | Prefix hash |
| --- | --- | --- | --- |
| Is there a chair? | yes | [-0.08731317520141602, 0.26656126976013184, 0.699421226978302] | 4f45ffbd6edd4ee8… |

These examples demonstrate runnable local inference only. Their correctness is not inferred from fluency; structured held-out metrics are reported separately.

## 24. Representative failures

Semantic localization missed hit@k for: `book`.
The initial tokenwise CLIP patch projection failed the semantic sanity gate and was replaced by MaskCLIP-style final-block value features. The obsolete numeric run is not promoted as a current result.
The v1 adapter reached a high raw held-out score but failed scene-content controls; it learned a near-constant nonzero soft-prompt/prior solution.
Gemma v8-resume24 passed its teacher-forced color-ranking gate but generated mostly
`orange`/`unknown`, stayed unchanged across all 35 mirror questions, and stayed
unchanged across all four held-out cube-support questions. It is an overfit/wiring
milestone, not a scene-understanding result.
Gemma v9 corrected the trained-color decoding failure at both epoch 30 and epoch
36, reaching 12/12 normalized-exact color sides and 6/6 complete units. It still
produced 0/70 exact mirror sides and 0/8 exact held-out cube-support sides, with no
prediction changes on either control. This is a stronger overfit milestone, not a
held-out scene-understanding result.
Gemma v10 then initialized from v9 epoch 36 and trained color plus a selected mirror
subset. Its best and final audits both fell to 9/12 exact color sides and 3/6 color
units. Both scored 0/12 exact selected mirror sides and 0/6 selected mirror units;
the best checkpoint scored only 1/70 across all mirror sides and the final scored
0/70. Both scored 0/8 on held-out support. This is failed continuation with partial
forgetting, not a scene-understanding result.
Gemma v11 restored all trained color answers and made mirror outputs more decisive,
but it still completed none of the six trained mirror pairs and none of the four
held-out support pairs. Its improved 7/70 strict mirror sides and 28/70 secondary
canonical sides do not meet promotion criteria.
Gemma v12 made all 12 selected relation sides pass its auxiliary spatial margin,
but both epoch 8 and epoch 12 then generated 0/12 exact selected mirror sides,
0/70 across all mirror sides, and 0/8 held-out support sides. All 70 mirror
responses were literal `unknown` outputs, not audit fallbacks. This falsifies the
claim that satisfying the current spatial auxiliary objective is sufficient for
the existing decoder adapter to express the relation.
Gemma v13 kept the V12 epoch-8 scene state and inherited decoder bank immutable and
trained only a disjoint 229,376-parameter bank across layers 30-33. It retained
12/12 exact color sides but again produced 0/12 selected mirror sides, 0/70 across
all mirror sides, and 0/8 held-out support sides. Every mirror response was still
the literal `unknown`; passing decoder-gradient, checkpoint-integrity, and runtime-
reload checks was not sufficient to produce the desired relation behavior.
Fluent chat samples must not be treated as evidence of scene understanding; only the structured held-out and control measurements support behavioral claims.

## 25. Preserved legacy prefix-invariance evidence

PASS for checkpoint `data/checkpoints/best`. Prefix `f4ed6bc9cbf75bd878bcebab04e432ad3eb04ef236aadd6659931c8f214b7c9b` was constructed before the first question and remained identical across 3 questions.

## 26. Preserved legacy oracle-deletion evidence

PASS for checkpoint `data/checkpoints/best`. The oracle directory was atomically renamed away during local inference, no forbidden path was opened, answers completed, and the directory was restored. This result is not automatically transferred to a different checkpoint without rerunning the test.

## 27. Exact remaining limitations

- Gemma v9 passes its strict six-unit teacher-forced full-vocabulary gate and
  free-generates all trained color answers exactly, but scores 0/70 exact mirror
  sides and 0/8 exact held-out support sides.
- Gemma v10 completed a weights-only continuation from v9 epoch 36 but never passed
  its gate. Both audited checkpoints score 9/12 color sides and 3/6 color units,
  0/12 selected mirror sides and 0/6 selected mirror units, and 0/8 held-out support
  sides. V10 partially forgets v9's trained-color behavior and is not promoted.
- Gemma v11 restores 12/12 exact color sides but reaches only 3/12 selected mirror
  sides, 0/6 selected mirror units, 7/70 all-mirror sides, and 1/8 held-out support
  sides. It is not promoted.
- Gemma v12 preserves 12/12 exact color sides and its spatial-relation warmup
  reaches 12/12 auxiliary-margin sides, but greedy generation reaches 0/12
  selected mirror sides, 0/70 all-mirror sides, and 0/8 held-out support sides.
  It is not promoted.
- Gemma v13 passes its no-update parity/gradient probe and its training,
  checkpoint-integrity, BF16 runtime-reload, and prefix-parity checks. It preserves
  12/12 exact color sides but reaches 0/12 selected mirror sides, 0/70 all-mirror
  sides, and 0/8 held-out support sides. It is not promoted.
- No Gemma held-out static-QA, interactive-chat, prefix-invariance, or
  oracle-deletion/leakage inference result exists for v9-v13.
- The exact source hash loaded by the v7 process was not captured; current source
  hashes are post-run audited snapshots.
- The v1 multi-scene adapter is scene-content-insensitive despite its raw held-out accuracy; wrong-scene and content-shuffle controls invalidate a scene-understanding claim for that checkpoint.
- The v2 structural diagnostic preserves more scene signal, but no explicitly v2-tagged held-out QA artifact is available yet.
- v8, v8-resume24, and v9-v13 wall-clock times are recorded, but peak training
  memory is not.
- Preserved legacy expected-change counterfactual consistency is zero.
- The direct multi-view image baseline is not scored.
- The prohibited oracle-text upper bound is not scored.
- The robot benchmark covers numeric mechanics and MCP wiring only; language-conditioned semantic target navigation is unmeasured.
- The deterministic robot scan is a pose-dependent numerical map reobservation, not an arbitrary-pose Blender render plus CLIP remapping.
- A center scan reconstructs visible surfaces but cannot reveal occluded rear surfaces.
- Legacy CLIP patch semantics missed its top-k query set for: book.

## 28. Recommended next experiments

1. Preserve V13 as a failed controlled ablation; do not extend its schedule blindly.
   Localize why its non-cancelling decoder gradients fail to move the mirror
   full-vocabulary decision before selecting another adapter architecture or loss.
2. Require the next saved-and-reloaded candidate to preserve 12/12 strict color
   sides and reach 12/12 strict selected-mirror sides with 6/6 changed units. Only
   then rerun all-mirror and held-out support controls before static QA, chat,
   leakage, robot, or promotion work.
3. Run the direct multi-view VLM and isolated oracle-text upper-bound baselines.
4. Train and evaluate language-conditioned target-facing and approach behavior
   only after the static semantic gate passes, without returning semantic labels
   through tools.
5. Preserve the v1/v2 CLIP/Qwen runs as historical anti-collapse evidence, not as
   primary-model results.

## Geometry validation detail

| Metric | Value |
| --- | --- |
| Validation status | PASS |
| Sampled points | 75,264 |
| Inside-room fraction | 100.000% |
| Reprojection RMSE | 0.00000627 px |
| Depth round-trip RMSE | 0.0000001034 m |
| Cube median surface error | 0.0000000298 m |

### Artifact-version warning

The mapping summary content hash differs from the semantic-sanity map hash, consistent with the later MaskCLIP map rebuild. Semantic results refer to the map hash recorded in `semantic_sanity_scene_000001.json`; regenerate the mapping summary for a fully synchronized manifest.

### Runtime warnings

- Full config hash differs from training, but all inference architecture fields match: checkpoint=ad07d037490e runtime=6c6d150bca7a

## Preserved legacy artifact inventory

Present sources: `{"ablations": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/ablations.json", "best_checkpoint": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/data/checkpoints/multiscene_anticollapse/best/metadata.json", "geometry": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/geometry.json", "leakage": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/leakage.json", "machine": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/machine_report.json", "map": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/map_scene_000001.json", "models": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/model_revisions.json", "qa": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/metrics.json", "render_manifest": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/data/rendered/scene_000001/manifest.json", "resampler_diagnostic": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/resampler_fix_diagnostic.json", "robot": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/robot_navigation.json", "semantic": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/semantic_sanity_scene_000001.json", "signal_audit": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/scene_signal_audit.json", "training_v1": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/training_multiscene.json", "validation": "/Users/stevens/Desktop/GTC/GTCAgenticOps/semantic_3d_chat/reports/metrics/validation_metrics.json"}`

Missing metric groups: `baselines, training`

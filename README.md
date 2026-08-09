# Semantic 3D Chat

`semantic_3d_chat` is a local, deterministic research prototype that turns complete
RGB-D room scans into a persistent continuous 3D semantic memory and injects that
memory into a frozen local causal language model. The primary inference path never
receives an object list, caption, textual scene graph, simulator labels, or oracle
metadata.

The current primary experiment uses `google/gemma-4-E2B-it`. Real-weight
full-image feature extraction, 3D map fusion, and zero-shot semantic localization
run locally on this Mac. The completed v9 LoRA run first passed its stricter
teacher-forced full-vocabulary gate at epoch 30 and remained passed through epoch
36. Both checkpoints then free-generated the trained color-swap answers at 12/12
normalized-exact sides and 6/6 complete units, but neither transferred to mirror or
held-out support controls. V10 subsequently initialized from the v9 epoch-36
weights and trained a deterministic color-plus-mirror curriculum. It did not pass
its gate: the best and final checkpoints both regressed to 9/12 exact color sides
and 3/6 complete color units, learned 0/6 selected mirror units, and scored 0/8
held-out cube-support sides. V11 restarted from the same v9 weights and added a
full-vocabulary first-token margin. It restored color to 12/12 exact sides and
6/6 units, but learned only 3/12 exact selected mirror sides and 0/6 complete
mirror units; across all mirror questions it reached 7/70 exact sides and 0/35
units, with 1/8 held-out support sides. V12 then kept the same initialization and
selection while adding an ordered spatial-relation objective. Its scene-only
warmup separated all 12 selected relation sides past the configured margin, but
the decoder gate remained at 6/12 mirror candidate sides and 0/12 mirror
full-vocabulary sides. Greedy generation preserved color at 12/12 sides and 6/6
units, yet scored 0/12 on the selected mirror subset, 0/70 over all mirror sides,
and 0/8 on held-out support. V13 then froze that V12 epoch-8 scene adapter and
inherited layer-34 LoRA, trained a disjoint 229,376-parameter decoder bank in
layers 30-33, and preserved the 12/12 color result. It nevertheless generated
0/12 exact selected-mirror sides, 0/70 across all mirror sides, and 0/8 held-out
support sides; all 70 mirror outputs were the literal model response `unknown`.
V14 then screened four exact-restart learning rates for that frozen-scene decoder
bank. All four arms retained 12/12 full-vocabulary color sides after four updates;
the predeclared ranker selected 2e-3 because it reached 5/12 mirror sides. An
exact optimizer/history resume to 12 updates peaked discretely at epoch 7 with
7/12 mirror sides and 1/6 complete mirror units while retaining 12/12 color, but
the scalar-selected epoch 10 checkpoint had only 5/12 mirror sides and final
epoch 12 had 6/12 and 0/6 units. The complete gate never passed, so no greedy
audit was run. V15's fixed-prefix shared-K/V screen preserved color through epoch
3 but peaked at only 6/12 mirror sides and 0/6 units, below its continuation gate;
epoch 4 also regressed color, so it was stopped. V16 then trained a zero-output,
question-independent global residual over all 256 scene tokens from the exact V14
epoch-7 checkpoint. Its best screen epoch reached 11/12 color sides and 5/6 color
units but only 5/12 mirror sides and 0/6 mirror units, so it also failed without
extension or generation. V17 then repeated the residual experiment as exact
four-update restarts at `1e-4` and `3e-4`. The strict selector chose `1e-4`
epoch 3, which preserved color at 12/12 sides and 6/6 units but reached only
6/12 mirror sides and 0/6 complete mirror units. Neither arm met the declared
continuation gate, so learning-rate tuning stopped without an extension or
generation audit. V9 remains a color-wiring overfit milestone; v10-v17 are
failed continuations, not promoted scene chatbots. V18's centered-content bridge
also failed: it retained color at 12/12 sides and 6/6 units but reached only 5/12
mirror sides and 0/6 units. Gemma held-out static QA,
interactive chat, Gemma
leakage claims, and language-conditioned robot navigation remain gated.

## Current primary stack

- Blender 5.2 for deterministic RGB rendering and exact metric ray-cast depth.
- Python 3.12 managed by `uv`.
- PyTorch with MPS preferred and CPU fallback.
- `google/gemma-4-E2B-it`: one complete image produces a native 48×48 spatial
  field. Middle 768D, late 768D, and native projected 1536D streams are retained as
  a 3072D float16 feature per spatial location.
- A 5 cm persistent semantic voxel map, followed by the globally complete
  `signal_preserving_resampler_v3`: every occupied block contributes to 256
  question-independent 384D scene latents, which are projected into Gemma's 1536D
  decoder space.
- Gemma decoder base weights remain frozen. V9-v12 adapt only the explicitly
  listed 45,056-parameter layer-34 LoRA state. V13-v14 use a versioned,
  disjoint-bank checkpoint contract: that 45,056-parameter V12 bank is frozen and
  a 229,376-parameter rank-8 bank in layers 30-33 is trainable. The scene adapter
  is also frozen. The decoder receives continuous scene tokens, numeric geometry,
  and the user's question—never an environmental caption, label list, or oracle
  metadata.

Gemma 4 is therefore used on both sides of the bridge: its native multimodal
vision tower supplies the full-image patch field, and its causal language decoder
receives the learned continuous 3D prefix. Directly chatting over rendered images
is a useful baseline, but it bypasses the persistent fused 3D memory and does not
satisfy the primary experiment.

Gemma choices are configurable in `configs/gemma4_e2b.yaml` and its experiment
overlays. They target the detected 24 GB Apple Silicon machine. `configs/default.yaml`,
`small_mac.yaml`, and `large_mac.yaml` remain legacy CLIP/Qwen configurations.

### Preserved legacy/control stack

The former `openai/clip-vit-base-patch16` +
`Qwen/Qwen2.5-0.5B-Instruct` path remains runnable as legacy infrastructure and as
a documented soft-prompt-collapse control. Its 74.8% raw held-out score is not
scene-understanding evidence because wrong-scene and shuffled-content prefixes
matched or exceeded it. `HuggingFaceTB/SmolVLM-256M-Instruct` remains reserved for
the unmeasured direct multi-view baseline. None of these is the current primary
model path.

### Gemma 4 E2B primary path (not behaviorally validated)

`google/gemma-4-E2B-it` is the selected integrated vision/language model. Its public
checkpoint is 10.25 GB: Google describes it as 2.3B effective parameters and
5.1B total with embeddings. The installed implementation needs Transformers 5,
while the preserved legacy stack uses Transformers 4. Gemma therefore runs in an
isolated `.venv-gemma4` and leaves the legacy environment unchanged:

```bash
make setup-gemma4-probe
make download-gemma4-config  # pinned config + hub metadata only; no weights
make gemma4-probe             # offline tiny-model CPU capability checks
make gemma4-probe-test
```

The probe verifies two non-obvious contracts directly against Transformers
5.14.1. First, one complete-image vision call exposes localized 768D pre-pooling
layer states indexed by explicit `(x, y)` patch positions, followed by 3×3 spatial
pooling. For a square 224×224 input, the default aspect-preserving processor uses
the whole image, resizes it to 768×768, produces a 48×48 pre-pool grid, and emits a
16×16 pooled grid. The production extractor retains 48×48 middle and late native
grids. It sends only the native 16×16 post-pool output through Gemma's trained
vision-to-language projector, then broadcasts each projected token to its exact
3×3 owner cells. It does not mislabel a pre-pool projection as native. The fused
field is therefore 768 + 768 + 1536 = 3072 dimensions at 48×48. Compute uses
BF16 on MPS, while the high-dimensional cached feature field is retained as
portable FP16; these are separate configurable dtypes.

The extractor instantiates and strictly loads only the checkpoint's 167.4M vision
parameters plus 1.18M projector parameters from safetensors; it does not
materialize the 4.6B-parameter decoder/audio stack during mapping. Second,
arbitrary continuous decoder prefixes work only when
Gemma 4's per-layer-embedding side input is supplied explicitly: scene latents use
the same non-semantic PAD-token PLE identity as native visual soft tokens, normal
prompt tokens keep their learned PLE, and both gain the model's context projection.
The probe also validates prefix KV-cache reuse.

The primary experiment is Gemma vision patches → depth-projected 3D map →
question-independent full-scene resampler → continuous Gemma decoder prefix. It is
not direct-image chat. The metadata/probe targets above do not download or load
checkpoint weights. The production path is selected separately with
`configs/gemma4_e2b.yaml`; it shares existing sanitized renders and QA supervision
but isolates derived features, maps, and checkpoints under `data_gemma4`. Passing
the teacher-forced adapter gate is necessary but not sufficient: free generation
and held-out scene controls must also succeed before any Gemma checkpoint can be
promoted. The generated capability record is
`reports/metrics/gemma4_e2b_capability_probe.json`.

The real-weight path is an explicit, pinned download followed by offline
extraction. Expect approximately 9.54 GiB of checkpoint storage:

```bash
make download-gemma4-weights
make extract-gemma4-scene SCENE=scene_000001
make build-gemma4-map SCENE=scene_000001
make gemma4-semantic-sanity SCENE=scene_000001
```

`extract-gemma4-scene` uses the isolated Gemma Python environment and selectively
loads only vision/projector tensors. `build-gemma4-map` reads the resulting 3072D
48×48 fields and writes the persistent map below `data_gemma4/maps`.
`gemma4-semantic-sanity` is explicitly evaluation-only: it scores the final
1536D native projected stream against mean bare-category token embeddings, reads
only `model.language_model.embed_tokens.weight` (and only the 15 rows needed for
this scene), and uses oracle boxes solely for metrics and heatmap overlays. It
does not instantiate Gemma or load any decoder layer.

The measured `scene_000001` result over 74,699 voxels and 13 category queries is
61.54% top-1 localization, 84.62% hit@100, and 45.23% precision@100 versus an
8.07% random precision baseline. Across 24 views, 53,292 multiply observed voxels
had mean cosine consistency 0.5889 versus 0.4009 for the different-voxel control
(+0.1880). The machine-readable result is
`reports/gemma4/metrics/gemma4_semantic_sanity_scene_000001.json`; its 13 opaque
query heatmaps and consistency histogram are under
`reports/gemma4/figures/gemma4_semantic_sanity/scene_000001`.

After building every scene in the persisted training split, the isolated
environment runs adapter training:

```bash
make train-gemma4
```

`make chat-gemma4` deliberately has no default checkpoint. It remains locked until
an adapter passes the behavioral gate and receives an explicit, hash-bound
`promotion.json`; none of the current Gemma checkpoints qualifies.

Variable-length teacher-forcing batches retain Gemma's explicit PLE and
multimodal-type streams, use native PAD PLE at masked positions, and supervise
answer tokens only. The frozen 10.25 GB decoder is streamed directly from the
safetensors checkpoint to MPS, avoiding a transient full CPU-plus-MPS duplicate.
Gemma training also enables non-reentrant activation checkpointing on the text
decoder only, recomputing frozen decoder activations during scene-prefix
backpropagation. Vision/audio towers remain untouched, and every inference loader
keeps checkpointing disabled. The switch is recorded as
`training.language_decoder_gradient_checkpointing` in config, selection audits,
adapter checkpoint metadata, and final training metrics; resume rejects a mode
mismatch.

The default and all runs through `gemma4_color_wiring_v5` retain the original
`[scene prefix][text prompt]` ordering. The isolated v6 experiment enables
`language.scene_prefix_after_bos` and uses exactly `[native BOS embedding]
[learned scene_start][all scene latents][learned scene_end][remaining prompt]
[answer]`. Gemma's BOS keeps its native token PLE, while continuous scene entries
keep PAD-token PLE; no BOI/EOI identities or textual environmental markers are
inserted. This layout choice is checkpoint-contracted, appears in selection and
training reports, and treats a missing legacy metadata field as `false` only.

The isolated v7 experiment keeps v6's BOS-first layout but switches the strict
`language.scene_boundary_mode` enum to `gemma4_native_image`. Its effective
prefix is `[BOS][BOI][256 continuous scene latents][EOI][remaining prompt]`.
BOI/EOI are exact frozen scaled embeddings from the pinned Gemma checkpoint,
not learned adapter parameters. They receive their exact native token PLE;
scene latents receive PAD-token PLE and modality type 1, while BOS, boundaries,
prompt, and answer receive type 0. The runtime derives these identities from the
loaded model and tokenizer and compares them with the explicit config and
checkpoint contract (IDs, revision, and bidirectional-attention setting) before
training, resume, chat, or generation. The markers are identical for every
scene and carry no environmental text or labels. Run selection or training with
`configs/experiments/gemma4_color_wiring_v7.yaml`; its artifacts are isolated
under `gemma4_color_wiring_v7` and do not overwrite v6.

The completed v7 run took 567.844 seconds for 72 decoder microsteps and 12 optimizer
updates. Its best checkpoint was epoch 7: 3/6 changed units, 9/12 correct sides,
3/6 prediction flips, 3/6 wrong-prefix flips, mean ranking margin 0.439616,
minimum margin -0.71875, and hinge 0.314453. The final epoch again reached only 3/6
changed units and 9/12 sides, with hinge 0.386068 and minimum margin -1.484375.
It therefore failed the required gate. Free generation, held-out QA, Gemma chat,
prefix invariance, and oracle-deletion inference were not run. The exact source
hash loaded by the already-running v7 process was not captured; hashes in the v7
failure report identify post-run audited source snapshots and must not be used to
attribute later padding or audit fixes to that execution.

#### Gemma v8 controlled fallback — teacher-forced gate passed, promotion failed

Because v7 failed with a fully frozen decoder, v8 keeps the native
`[BOS][BOI][256 continuous scene latents][EOI][prompt]` layout and complete,
question-independent scene prefix unchanged, while adding LoRA only to layer 34
`q_proj` and `o_proj`:

- exact targets:
  `model.language_model.layers.34.self_attn.q_proj` and
  `model.language_model.layers.34.self_attn.o_proj`;
- rank 4, alpha 8, dropout 0;
- 45,056 FP32 A/B-only trainable parameters (180,224 bytes, approximately 176 KiB);
- LoRA learning rate `1e-4` and weight decay `0`.

The implementation has strict target/optimizer/checkpoint contracts, compact-state
SHA validation and tensor-tamper rejection, resume validation, chat restoration,
and scene-signal-audit restoration coverage. The fresh MPS run completed epoch 12
in 574.56 seconds but failed the teacher-forced gate at 4/6 changed units, 10/12
correct sides, and hinge 0.272786. Its exact metrics are in
`reports/gemma4/metrics/training_gemma4_color_wiring_v8.json`.

A resume from epoch 12 stopped early at epoch 22 after 22 total optimizer updates.
It passed the same teacher-forced gate with 6/6 changed units, 12/12 correct sides,
prediction-flip rate 1.0, wrong-prefix-flip rate 1.0, and minimum candidate margin
0.0390625; the resumed portion took 388.31 seconds. The exact metrics are in
`reports/gemma4/metrics/training_gemma4_color_wiring_v8_resume24.json`.

The passed gate did not survive free decoding well enough for promotion. On the
training color-swap pair, outputs changed for 5/6 questions, but the canonical
answers were still mostly wrong (`orange` or `unknown`; only isolated answers were
correct). Outputs changed for 0/35 mirrored-room questions and 0/4 held-out
cube-support questions. The model-validated BF16 generation audit is
`reports/gemma4/metrics/scene_signal_audit_gemma4_color_wiring_v8_resume24.json`.
Consequently this checkpoint establishes only a small teacher-forced wiring/overfit
milestone. It is not promoted for held-out static QA, chat, or embodied-agent use.

The exact selection, fresh-training, and resume command shapes are:

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

#### Gemma v9 hardened gate — exact trained-color generation, no transfer

V9 adds a strict teacher-forced full-vocabulary check to the earlier pairwise
candidate gate: for every side, the canonical first answer token must be the unique
top-1 token across the entire vocabulary. The fresh 36-epoch MPS run completed 216
decoder microsteps and 36 optimizer updates in 1,539.387 seconds. The old pairwise
gate was already 12/12 at epoch 22, but only 1/12 expected first tokens were
full-vocabulary top-1 then. The composite gate first passed at epoch 30 with 12/12
top-1 sides, 6/6 complete units, mean target-versus-best-other margin 1.500651,
and minimum margin 0.03125. Epoch 36 was stronger at mean 2.903809 and minimum
1.0. Exact training history is in
`reports/gemma4/metrics/training_gemma4_color_wiring_v9.json`.

The model-validated greedy generation audit gives the same decisive result for
both the stored epoch-30 `best` checkpoint and epoch 36:

| Counterfactual intervention | Normalized-exact sides | Complete units | Predictions changed |
| --- | ---: | ---: | ---: |
| Trained color swap | 12/12 | 6/6 | 6/6 |
| Mirrored left/right room | 0/70 | 0/35 | 0/35 |
| Held-out cube support | 0/8 | 0/4 | 0/4 |

The checkpoint-specific audits are
`reports/gemma4/metrics/scene_signal_audit_gemma4_color_wiring_v9_best_epoch30.json`
and
`reports/gemma4/metrics/scene_signal_audit_gemma4_color_wiring_v9_epoch036.json`.
Prediction-change rate is reported separately from correctness; changing between
two wrong answers never counts as an exact success. The calculated tensor metrics
also do not support the old blanket Perceiver-collapse diagnosis for this
checkpoint: scene signal remains measurably distinct through the current
resampler. The behavioral failure is lack of transfer beyond the trained color
intervention.

V9's `best` alias points to epoch 30 because its original selection monitor was a
hinge that became zero at the first pass and could not distinguish stronger later
passes. The corrected checkpoint selector preserves the fail-before-pass ordering,
then ranks passing full-vocabulary checkpoints by their minimum
target-versus-best-other margin. A later robust pass can therefore replace a
barely positive one.

#### Gemma v10 color-plus-mirror continuation — gate failed

V10 initialized from v9 epoch 36 in `weights_only_new_curriculum` mode. It restored
compatible scene-encoder, projector, composer, and LoRA weights while deliberately
starting a new optimizer, history, and curriculum. The recorded initialization
contract identifies the v9 epoch-36 checkpoint and adapter hash
`8ecbf84fc8f544d67fe3e65a313023c3808870c5648b913de2839ec525630c90`, and confirms
that optimizer state and history were not loaded. Its deterministic selection
contained six complete color units and six complete mirror units—24 records across
four opaque scenes. A seed hash over opaque pair/question keys applied the per-pair
cap without inspecting question text or answers, and both physical scene sides
remained indivisible. Held-out cube support remained evaluation-only.

The MPS run completed 12 epochs, 144 decoder microsteps, and 12 optimizer updates
in 933.685 seconds. The strict teacher-forced composite gate never passed. Epoch 8
was retained as `best`, but its full-vocabulary first-answer check was only 9/24
sides and 3/12 complete units; the final epoch had the same counts. The final gate
also retained a negative minimum target-versus-best-other margin of -9.75.

Model-validated greedy generation confirmed the failure:

| Checkpoint and intervention | Exact sides | Complete units | Predictions changed |
| --- | ---: | ---: | ---: |
| Best epoch 8 — trained color | 9/12 | 3/6 | 6/6 |
| Best epoch 8 — trained mirror, selected subset | 0/12 | 0/6 | 0/6 |
| Best epoch 8 — mirror, all units | 1/70 | 0/35 | 3/35 |
| Best epoch 8 — held-out cube support | 0/8 | 0/4 | 1/4 |
| Final epoch 12 — trained color | 9/12 | 3/6 | 6/6 |
| Final epoch 12 — trained mirror, selected subset | 0/12 | 0/6 | 0/6 |
| Final epoch 12 — mirror, all units | 0/70 | 0/35 | 1/35 |
| Final epoch 12 — held-out cube support | 0/8 | 0/4 | 2/4 |

At best epoch 8, the 29 unselected mirror units contributed 1/58 exact sides and
3/29 changed predictions; at final epoch 12 they contributed 0/58 and 1/29. A
changed wrong answer is not counted as correct. Relative to v9's 12/12 color sides
and 6/6 units, v10 partially forgot the previously demonstrated color behavior
without learning even its selected mirror subset. The exact training and generation
evidence is in
`reports/gemma4/metrics/training_gemma4_color_mirror_wiring_v10.json`,
`reports/gemma4/metrics/scene_signal_audit_gemma4_color_mirror_wiring_v10_best_epoch8.json`,
and
`reports/gemma4/metrics/scene_signal_audit_gemma4_color_mirror_wiring_v10_epoch012.json`.
The config and selection audit remain
`configs/experiments/gemma4_color_mirror_wiring_v10.yaml` and
`reports/gemma4/metrics/training_selection_gemma4_color_mirror_wiring_v10.json`.
No `promotion.json` was created.

#### Gemma v11 full-vocabulary retry — color restored, mirror still failed

V11 is the controlled retry in
`configs/experiments/gemma4_color_mirror_full_vocab_v11.yaml`. It restarts from
the same proven v9 epoch-36 weights and reuses the exact v10 selection and
schedule. The only objective change is a weight-2 hinge requiring the correct
first answer token to exceed the strongest non-target token in Gemma's complete
vocabulary by 1.0; the weight-8 counterfactual candidate hinge remains active.
This uses the existing correct decoder forward and answer labels, so it adds no
environmental text, retrieval, or decoder forward pass. V10 remains the
zero-full-vocabulary-loss ablation.

The MPS run completed 12 epochs, 144 decoder microsteps, and 12 optimizer updates
in 947.428 seconds. Its `best` alias and epoch 12 are byte-identical (adapter
SHA-256 `eee7b3aa8ce2e7584cfe1fc80d8852d4d645b24c156ccd43369cb4ba7e047e22`).
The aggregate teacher-forced gate failed. Color passed at 12/12 candidate and
full-vocabulary sides and 6/6 complete units. Mirror remained at 6/12 candidate
sides and 0/6 units; its full-vocabulary result was 3/12 sides and 0/6 units.

Model-validated greedy generation measured:

| Intervention | Exact sides | Complete units | Predictions changed |
| --- | ---: | ---: | ---: |
| Trained color swap | 12/12 | 6/6 | 6/6 |
| Trained mirror subset | 3/12 | 0/6 | 2/6 |
| Mirror, all units | 7/70 | 0/35 | 10/35 |
| Held-out cube support | 1/8 | 0/4 | 4/4 |

Strict normalized exact remains the promotion score. A secondary parser that
accepts unambiguous verbose relation sentences raises all-mirror scoring to 28/70
sides and 5/35 units, but all five complete units are unselected; the trained
subset remains 0/6. V11 suppressed the `unknown`/unrelated-token mode without
learning the intended physical flip. Its mirror prefix is not structurally
collapsed: final-prefix relative L2 is 0.358191 and 98.55% of elements change.
The exact evidence is
`reports/gemma4/metrics/training_gemma4_color_mirror_full_vocab_v11.json` and
`reports/gemma4/metrics/scene_signal_audit_gemma4_color_mirror_full_vocab_v11_epoch012.json`.
No `promotion.json` was created.

#### Gemma v12 ordered-relation retry — auxiliary margin passed, decoder failed

V12 is defined by
`configs/experiments/gemma4_color_mirror_spatial_relation_v12.yaml`. It preserves
V11's v9 initialization, 24-record deterministic selection, LoRA scope, decoder
losses, and schedule. Its only training change is an ordered target-minus-reference
contrastive loss over dense soft pools of all 256 global scene latents, plus a
scene-only warmup. Exact oracle coordinates remain confined to supervised QA
artifacts; chat still receives the same global question-independent continuous
prefix and no labels, coordinates, or text scene description.

The scene-only warmup stopped early after 21 forward passes and 20 optimizer steps:
all 12 eligible mirror sides exceeded the 0.1 margin, with mean margin 0.246168 and
minimum margin 0.116128. The full run then completed 12 epochs, 144 decoder
microsteps, and 12 main optimizer updates in 1,073.493 seconds (17m 53.493s). The
teacher-forced gate nevertheless failed identically at epochs 8 and 12. Color was
12/12 for both candidate and full-vocabulary first-token scoring, or 6/6 complete
units. Mirror was 6/12 candidate sides and 0/6 complete units, but 0/12
full-vocabulary sides and 0/6 units. Passing the auxiliary relation loss therefore
did not make Gemma decode the ordered scene signal as `left` or `right`.

Model-validated greedy generation produced the same strict results at epoch 8 and
the final/best epoch 12:

| Intervention | Exact sides | Complete units | Predictions changed |
| --- | ---: | ---: | ---: |
| Trained color swap | 12/12 | 6/6 | 6/6 |
| Trained mirror subset | 0/12 | 0/6 | 0/6 |
| Mirror, all units | 0/70 | 0/35 | 0/35 |
| Held-out cube support | 0/8 | 0/4 | 0/4 |

Every mirror answer was the model's literal `unknown`; the audit used no fallback,
had no empty decodes, and exhausted no generation budget. Both audits report zero
checkpoint-contract warnings, validate the native boundary embeddings and BF16
runtime dtype, and prove parity with the model-validated runtime prefix. Epoch 8's
adapter SHA-256 is
`a4c85c14a214e4e594992e489a784cb4bacb64d3dfda519ad3da18b1595d9f22`;
the final epoch-12 and `best` adapter SHA-256 is
`1d46e754873431b11e8dc58066f08f06c17e3bcaa4c47b139358bb0f28ceabb1`.
The exact evidence is
`reports/gemma4/metrics/training_gemma4_color_mirror_spatial_relation_v12.json`,
`reports/gemma4/metrics/scene_signal_audit_gemma4_color_mirror_spatial_relation_v12_epoch008.json`,
and
`reports/gemma4/metrics/scene_signal_audit_gemma4_color_mirror_spatial_relation_v12_best.json`.
No `promotion.json` was created.

#### Gemma v13 frozen-scene decoder banks — integrity passed, behavior failed

V13 is the bounded decoder-capacity falsifier in
[`configs/experiments/gemma4_color_mirror_decoder_banks_v13.yaml`](configs/experiments/gemma4_color_mirror_decoder_banks_v13.yaml).
Its pre-run probe started from the pinned V12 epoch-8 adapter
`a4c85c14a214e4e594992e489a784cb4bacb64d3dfda519ad3da18b1595d9f22`,
froze the scene adapter and inherited rank-4 layer-34 q/o LoRA, and added a
disjoint, exact-zero-output rank-8 q/o bank in layers 30-33. All 12 probed
first-answer vocabulary distributions were bitwise identical before and after
bank installation. The six mirrored units then produced material, aligned
gradients rather than the predicted cancellation: candidate-hinge cancellation
ratio 0.987037 and cosine 0.948397; complete-objective ratio 0.987413 and cosine
0.949781. The no-update probe remains in
[`mirrored_gradient_probe_v13_epoch008.json`](reports/gemma4/metrics/mirrored_gradient_probe_v13_epoch008.json)
(SHA-256 `59638470edf63a8c8b4a450f3a833a7084c171a4147334366fa5016e709533e6`).

Implementation commit `990589363b42b2cd3451ec24f7a912ffac8411f6` adds
schema-2 named LoRA banks while retaining legacy schema-1 loading. It validates
globally disjoint targets, deterministic initialization, per-bank checkpoint
state, trainability, and optimizer membership. V13 freezes the complete scene
state at SHA-256
`690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b`
and the inherited bank at
`dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594`.
The trainable 229,376-parameter extension uses deterministic seed 13008, initial
state hash
`b4ec0518e4759dda33fc93c9c1d4c76f52f1024fd5b8b1667ad1b4ef5da198af`,
and final state hash
`caaf9b13c13b2371463a2cf9d450453f925846b0202bdb0610103b6aa85e435b`.
The combined LoRA state contains 274,432 parameters; base Gemma weights never
enter the optimizer.

A separately labeled one-update smoke completed 12 microsteps in 44.822 seconds.
It preserved both frozen hashes while changing only the extension bank. This is
training-wiring evidence only; runtime reload and prefix parity are established
by the full saved-checkpoint audit below. The full MPS run then completed 12
epochs, 144 microsteps, and 12 optimizer updates in 467.248 seconds. Its clean
recorded source is the same `9905893` implementation commit. At every teacher gate
(epochs 2, 4, 6, 8, 10, and 12), color remained 12/12 for both candidate and
full-vocabulary scoring (6/6 units), while mirror remained 6/12 candidate sides
and 0/6 units, and 0/12 full-vocabulary sides and 0/6 units. `best` is final epoch
12; its adapter and metadata are respectively byte-identical to epoch 12 at
SHA-256 `9b59d15ba9e4d3be8d8a64ea6d9d3071d1e8650333ee8c21c5504e7900353c7c`
and `83ba42f6fc5b8ef2025588f35a3a2bba9a9d7e4074487d85c43ef5b25fc13a7b`.

Saved-and-reloaded, model-validated BF16 greedy generation measured:

| Intervention | Exact sides | Complete units | Predictions changed |
| --- | ---: | ---: | ---: |
| Trained color swap | 12/12 | 6/6 | 6/6 |
| Trained mirror subset | 0/12 | 0/6 | 0/6 |
| Mirror, all units | 0/70 | 0/35 | 0/35 |
| Held-out cube support | 0/8 | 0/4 | 0/4 |

All 70 mirror outputs were the literal model response `unknown`, with no answer
fallback, empty decode, or exhausted token budget. The first audit attempt exposed
an audit-constructor bug, not a checkpoint mutation: FP32 placeholder construction
silently cast persisted native BF16 BOI/EOI buffers and tripped the frozen-state
hash. Commit `a10579d6dadecf8082cc179a201bb1db517656aa` constructs the audit
composer from exact model boundary embeddings, or configured BF16 placeholders
when generation is skipped. The rerun reports zero checkpoint warnings, validates
the loaded BF16 model and native boundaries, and proves chat-runtime prefix parity.

After the BF16 audit fix, the V13 tree passed the full standard suite (310 passed,
15 skipped); a broader Gemma-focused suite passed 92 tests with two benign SWIG
warnings. Focused standard and Gemma audit/boundary suites passed 17 and 26 tests
respectively. Exact machine-readable evidence is
[`training_selection_gemma4_color_mirror_decoder_banks_v13.json`](reports/gemma4/metrics/training_selection_gemma4_color_mirror_decoder_banks_v13.json),
[`training_selection_gemma4_color_mirror_decoder_banks_v13_smoke.json`](reports/gemma4/metrics/training_selection_gemma4_color_mirror_decoder_banks_v13_smoke.json),
[`training_gemma4_color_mirror_decoder_banks_v13_smoke.json`](reports/gemma4/metrics/training_gemma4_color_mirror_decoder_banks_v13_smoke.json),
[`training_gemma4_color_mirror_decoder_banks_v13.json`](reports/gemma4/metrics/training_gemma4_color_mirror_decoder_banks_v13.json)
(file SHA-256 `f86659cbc3ab4407c82d583f2e846c9acf6713c82d89c5dfaded4d91bed6a79c`),
and
[`scene_signal_audit_gemma4_color_mirror_decoder_banks_v13_best.json`](reports/gemma4/metrics/scene_signal_audit_gemma4_color_mirror_decoder_banks_v13_best.json)
(file SHA-256 `042fd7c5b085858ac334aaf40f08533e8f2db1ab0ef258917fa901be224256dc`).
No `promotion.json` was created. V13 therefore falsifies this added-decoder-bank
configuration: its integrity and reload gates pass, but selected mirror, all-mirror,
and held-out support behavior do not. Held-out static QA, Gemma chat and leakage
tests, and semantic robot navigation remain gated.

#### Gemma v14 learning-rate response — transient partial relation fit, gate failed

V14 retains V13's frozen scene adapter, frozen inherited V12 layer-34 bank,
trainable 229,376-parameter layers-30-33 extension, exact V12 epoch-8 restart,
24-record selection, training order, and losses. The only screened intervention
was extension-bank learning rate. The four-arm contract is
[`configs/experiments/gemma4_color_mirror_decoder_banks_v14_lr_sweep.yaml`](configs/experiments/gemma4_color_mirror_decoder_banks_v14_lr_sweep.yaml);
all arms ran from clean source commit
`1ee8b5d13777e74ebdfe1f87e7d8320403ad5fbf` (tree
`b606e85cbb5a786ba2e00f971cf07c174bc5cbef`, empty tracked-diff hash
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
The audited selection and pair-membership hashes were respectively
`7f0714e3151c9ddb57c1da95a457820a833e490c070881a88a9fee4a9168f933`
and `99ee448c23fb71b7269a353a54b2156ac55701847af170597dcc351af15cbcbe`.
Every arm completed exactly four MPS optimizer updates and retained 12/12
full-vocabulary color sides and 6/6 complete color units at epoch 4:

| Learning rate | Recorded screen time (s) | Mirror full-vocab sides | Mirror units | Mean / minimum mirror margin | Screen report SHA-256 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1e-4 | 180.22852345905267 | 0/12 | 0/6 | -19.5859375 / -25.4375 | `88593afc7998785afbe7e4e2c41e56bed7fb90f0f77ee55cd63ca5f5c7f95bfc` |
| 3e-4 | 176.10611624992453 | 0/12 | 0/6 | -18.22395896911621 / -23.78125 | `f651cffc9d42eda7f9f89f010bcda9fee743240e32dfb1f78d405a6086db9ebe` |
| 1e-3 | 179.8313142498955 | 0/12 | 0/6 | -8.165771484375 / -11.78125 | `5b789526563d82d86d29ec8b599ce485de273e939b4e494946168f74dfa4cbe2` |
| 2e-3 | not retained separately | 5/12 | 0/6 | -0.2604166567325592 / -3.9375 | see provenance note below |

The predeclared color-first ranker therefore selected 2e-3. Its historical sweep
summary is
[`lr_sweep_gemma4_color_mirror_decoder_banks_v14.json`](reports/gemma4/metrics/lr_sweep_gemma4_color_mirror_decoder_banks_v14.json),
SHA-256 `b0643e4a09d147702efae58f0559a54fe8f61f98d4ef9d823392a365493ada4a`.
That summary embeds the four epoch-4 arm metrics and their common provenance
contract hash
`f016b57fd79dc3a229e5015a0d146424a16c345810133783a3ad6f960e3f7968`.
Because a selected arm's convenience report path was later overwritten by its
exact continuation, the canonical screen evidence is the independently generated
[`lr_sweep_checkpoint_attestation_gemma4_color_mirror_decoder_banks_v14.json`](reports/gemma4/metrics/lr_sweep_checkpoint_attestation_gemma4_color_mirror_decoder_banks_v14.json),
SHA-256 `9f959226e1d10f16888c5f1b4db165912c286c84c21d98930fe635e3f290359e`.
It reads no historical training-report path: it validates all four intact
`epoch_004` checkpoints and selection manifests, recomputes the trainable-bank
hashes from safetensors, checks optimizer steps and learning rates, and reproduces
the ranking `lr2e3`, `lr1e3`, `lr3e4`, `lr1e4`.
The selected epoch-4 extension-state hash is
`78839fc9683c8b9e4f0227d9c248a5ce916b44967f75424be8d484d50eb07681`;
its adapter, metadata, and optimizer file hashes are respectively
`0e41bc85b4c3e7bb8e9c71d3ed7ff43d8b4f7d3b682fa05c0f5f009b1e0a2203`,
`9168e0e02ff5a92b23f0f6aeece937f17cc2eb37980dc34ffa88a1aa06001fda`,
and `bbffc2feb60a780b43a483faa38ebd9ab740ff3b806861deb5a0161bd1129a7f`.

The selected arm then resumed exactly from epoch 4 with its optimizer and history
and ran eight additional updates. The resume invocation took
351.17732408316806 seconds and produced a 12-epoch, 144-microstep, 12-update
aggregate trace:

| Epoch | Color full-vocab sides / units | Mirror full-vocab sides / units | Mirror mean / minimum margin | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 4 | 12/12 / 6/6 | 5/12 / 0/6 | -0.2604166567325592 / -3.9375 | screen winner |
| 7 | 12/12 / 6/6 | 7/12 / 1/6 | 0.0416666679084301 / -2.3125 | best discrete mirror result |
| 8 | 10/12 / 4/6 | 6/12 / 0/6 | 0.0364583320915699 / -3.125 | color regression |
| 9 | 10/12 / 4/6 | 6/12 / 0/6 | 0.078125 / -2.9375 | color regression |
| 10 | 12/12 / 6/6 | 5/12 / 0/6 | 0.0833333358168602 / -1.375 | scalar-selected `best` |
| 12 | 12/12 / 6/6 | 6/12 / 0/6 | 0.0572916679084301 / -2.25 | final |

The configured `pair_composite_full_vocab_gate_margin` monitor selected epoch 10
at scalar loss 0.700520858168602, despite epoch 7's better discrete result. The
epoch-10 extension-state hash is
`5355e6849adfead434aa021d64af09e62b79af25b8d0075be9994921613c8888`;
the byte-identical `best` adapter, metadata, and optimizer hashes are
`b84c64cadfedb295c9c2806284f06675cc51146e09ea993cabb59c3bb9e15931`,
`da791884ce7a8f89a1c63e4b5a58035b62ff3a16a4a3d47d29d9b7c8112050c2`,
and `6318ec44520ea8166860e3fdd3e6ad1389cee523f3203fdafbb68347a74b5933`.
Final epoch 12 has extension-state hash
`c6483c420210272335d041f1ae4ee7e0e5cdab6e57798d19cd4d2cd539092a1a`
and adapter hash
`2e077d3f46e95898a0b33881bf5e11c7029e198bd775cd9d25ed6e60ce78d6a5`.
Both frozen hashes remain exact: scene state
`690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b`
and inherited bank
`dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594`.

The aggregate trace is
[`training_gemma4_color_mirror_decoder_banks_v14_lr2e3.json`](reports/gemma4/metrics/training_gemma4_color_mirror_decoder_banks_v14_lr2e3.json),
SHA-256 `5efe39c22985908afcd0cf720c06b94b87c5d7f6a4f4fe03ec6ca91b5313d38d`.
Provenance caveat: that same path was an input to the sweep summarizer at epoch 4
and was later overwritten by the exact-resume aggregate report. The checkpoint
attestation is therefore the canonical machine-readable evidence for the screen;
it verifies the historical summary without opening the mutable report paths. The
separate four-update 2e-3 wall time is not present in the surviving artifacts. The
351.177-second value is the eight-update resume invocation, not the original
screen.

No epoch passed the complete color-plus-mirror teacher gate. Under the fail-closed
policy, V14 therefore received no greedy audit, no `promotion.json`, no held-out
control evaluation, and no downstream static chat, leakage, or semantic robot
run. Its transient 7/12 and 1/6 epoch-7 result is evidence of learning response,
not accepted scene understanding.

#### Gemma V15 shared-K/V screen — failed, no extension

V15 kept the complete 256-token scene prefix fixed and question-independent and
added a zero-output 290,816-parameter LoRA bank to Gemma's two physical shared
K/V source layers plus late Q/O projections. Across four optimizer updates,
color remained 12/12 full-vocabulary sides and 6/6 units through epoch 3, while
mirror peaked at 6/12 sides and 0/6 complete units. That missed the predeclared
8/12-side and 2/6-unit continuation gate. Epoch 4 also regressed color to 4/12
sides and 1/6 units. The run was therefore stopped without extra epochs, greedy
generation, promotion, held-out audit, chat, leakage claims, or robot work.

The exact decision and checkpoint hashes are recorded in
[`screen_decision_gemma4_color_mirror_decoder_qkvo_v15.json`](reports/gemma4/metrics/screen_decision_gemma4_color_mirror_decoder_qkvo_v15.json).

#### Gemma V16 global scene residual — failed, no extension

V16 starts from the exact V14 epoch-7 checkpoint and freezes the core scene
encoder, prefix composer, grounding head, and both persisted Gemma LoRA banks.
Its only trainable surface is a 400,000-parameter residual over all 256 scene
tokens. Every output keeps its original token, includes a shared mean influenced
by every scene slot, and receives a persistent spatial-anchor/Fourier feature.
The output projection is exact zero at initialization, so update 0 must reproduce
the V14 prefix bit-for-bit. The module API accepts only scene tokens; it has no
question, answer, retrieval, label, or oracle input. The final adapted prefix is
still computed and hashed before any user question.

The pinned experiment was
[`gemma4_color_mirror_global_scene_residual_v16.yaml`](configs/experiments/gemma4_color_mirror_global_scene_residual_v16.yaml).
Update-zero equivalence was exact for all four scenes, the frozen core and both
LoRA-bank hashes remained unchanged, and the residual changed from its pinned
initial hash to a nonzero trained state. The four teacher-forced updates produced
color/mirror full-vocabulary side counts of 6/6, 6/5, 8/6, and 11/5 out of 12;
the corresponding complete-unit counts were 0/1, 0/1, 2/0, and 5/0 out of 6.
No epoch preserved the required 12/12 and 6/6 color result or reached the 8/12
and 2/6 mirror continuation minimum. Training therefore stopped after exactly
four updates with no extension, generation audit, promotion, held-out claim, or
robot work.

A real load-only runtime probe strictly reconstructed the epoch-4 checkpoint,
built the `[1, 258, 1536]` prefix before any question, applied a nonzero residual,
and recorded zero forbidden/oracle reads. This verifies checkpoint/runtime
integrity only; it is not an answer-correctness or full oracle-deletion result.
The exact decision, per-epoch counts, hashes, and claim limits are recorded in
[`screen_decision_gemma4_color_mirror_global_scene_residual_v16.json`](reports/gemma4/metrics/screen_decision_gemma4_color_mirror_global_scene_residual_v16.json).

#### Gemma V17 residual learning-rate response — failed, no extension

V16's first AdamW update could change only the initially zero 196,608-weight
output projection. At 1e-3, that dense sign-like step produced a residual about
13.4% as large as the frozen prefix RMS; roughly 99% of its energy was an
across-slot mean shift, while pair-specific changes were only about 2% of the
underlying color or mirror scene difference. This explains the immediate color
regression and motivates an optimizer-response test before another architectural
change. The color and mirror output-weight gradients have cosine `+0.281011`, so
direct objective antagonism is not the primary failure; the destructive common
step is. The reproducible offline audit is
[`v16_zero_residual_gradient_audit.json`](reports/gemma4/metrics/v16_zero_residual_gradient_audit.json),
SHA-256 `5e453933df459f7122ff8781bd2881838fb06a47c1a25e0193368edeedcede31`.
It uses supervised QA metadata only as an offline loss probe, executes no
optimizer step, and is not imported by chat runtime.

V17 ran two independent four-update exact restarts at 1e-4 and 3e-4.
Both start from the pinned V14 epoch-7 weights and the same zero-output residual,
use the same 24 records and 12 complete paired units, and retain the same frozen
core, frozen LoRA banks, loss, order, accumulation, and gates. An arm is eligible
for comparison only if it preserves color at 12/12 sides and 6/6 units with
strictly positive minimum margins. The selected eligible arm must also reach at
least 8/12 mirror sides and 2/6 complete mirror units before any extension; no
greedy audit is allowed before the full teacher gate. If neither arm qualifies,
learning-rate tuning stops and the next change must remove the common-shift
failure mode rather than adding epochs or decoder capacity.

The `1e-4` arm had color-eligible epochs 1, 3, and 4. Its strict within-arm
representative was epoch 3: color was 12/12 full-vocabulary sides and 6/6
complete units with minimum margin `+0.5`, while mirror was 6/12 sides and 0/6
units with minimum margin `-2.1875`. At epoch 2, mirror briefly reached the
continuation minimum of 8/12 sides and 2/6 units, but color simultaneously fell
to 11/12 and 5/6 with a negative minimum margin, making that epoch ineligible.

The `3e-4` arm's representative was epoch 4: color was 12/12 and 6/6 with
minimum margin `+2.0625`, while mirror was 5/12 and 0/6 with minimum margin
`-1.25`. Its epoch 2 showed the same tradeoff—mirror reached 8/12 and 2/6 while
color fell to 11/12 and 5/6. The two-stage predeclared ranker therefore selected
`1e-4` epoch 3 across arms, but did not authorize continuation, the full teacher
gate, or greedy generation. Independent inspection verified both input reports,
the clean source commit, identical update-zero prefixes, selection membership,
and frozen scene/LoRA hashes.

The machine-readable decision is
[`residual_lr_response_v17.json`](reports/gemma4/metrics/residual_lr_response_v17.json),
SHA-256 `3f63fd5654fb7120ed7aa9d28414490552eec32b3f170c3b80f992947d4161d9`.
The selected epoch-3 adapter SHA-256 is
`9c09d6b030082d5de771901ca51b4a554ff131ea53ce6ed477272a407d33487c`;
its metadata SHA-256 is
`7f059b17502dcd51917503daf256dc17ec64a4a55e7d9a82dc31c5ab81d80432`.

The immutable arm configs are
[`gemma4_color_mirror_global_scene_residual_v17_lr1e4.yaml`](configs/experiments/gemma4_color_mirror_global_scene_residual_v17_lr1e4.yaml)
and
[`gemma4_color_mirror_global_scene_residual_v17_lr3e4.yaml`](configs/experiments/gemma4_color_mirror_global_scene_residual_v17_lr3e4.yaml).

#### Gemma V18 centered content-gate screen — completed, mirror gate failed

V17 proves that reducing the V16 learning rate does not remove the color/mirror
tradeoff. V18 therefore restarts from the same exact V14 epoch-7 checkpoint with
a new question-independent residual architecture; it does not inherit V16/V17
weights, optimizer moments, or history. For scene slots `x_i`, it projects
normalized content, subtracts the FP32 mean over all 256 slots, gates each slot
with a learned scalar derived from that centered content, combines it with the
persistent spatial Fourier feature, and subtracts the FP32 mean of the resulting
learned delta before retaining the original token through the identity path.
Every output consequently depends on every scene slot, while a common learned
shift is structurally excluded. The API still accepts scene tokens only—never a
question, answer, retrieval query, label, caption, or oracle coordinate.

The new nested residual contract is schema 2, has 400,128 trainable parameters,
starts with an exact-zero output projection, and has pinned initial-state SHA-256
`f7f6353edb6216029bd155e2baab1b5051c85f297a0e6d6b63210354fe0ff0e0`.
Omitting the architecture version still constructs the bit-identical V16 schema-1
module with its original 400,000 parameters and state hash. Strict checkpoint
loading rejects V16↔V18 state migration.

Before training, an offline supervised diagnostic must reproduce the exact
ordered 12 microsteps of epoch 1, accumulate the actual loss gradient, and run
the exact pinned AdamW update on an isolated full-residual clone without
mutating live parameters, optimizer state, or RNG. Each of the four scenes must have a nonzero finite raw and
effective delta, raw common-energy fraction at most `1e-6`, raw slot-varying
fraction at least `0.999999`, effective common-energy fraction at most `1e-3`,
effective slot-varying fraction at least `0.999`, and effective delta/core RMS
at most `0.05`. Both counterfactual scene pairs must receive distinct nonzero
deltas. The looser effective bound is an implementation guard for the
`3.11e-4` common-energy observed when an exactly centered FP32 delta is cast and
added to BF16 tokens; it is not semantic-success evidence.

Execution is deliberately staged. After the no-live-step preflight passes, stage 1
runs exactly one update and stops. Its residual state hash must equal the
preflight prediction before an exact optimizer/history resume may run updates
2–4. The update-one verifier safely reads `optimizer.pt` with PyTorch's
`weights_only` mode and requires exact equality with the preflight's canonical
eight-parameter AdamW manifest: parameter order, named group, 24 moment/step
tensors, step number, and every optimizer option are hash-bound. It also
requires the current clean source provenance to equal both stored records.
Eligible epochs must retain color at 12/12 sides and 6/6 complete units
with positive minimum margins; continuation additionally requires mirror at
least 8/12 and 2/6 in that same epoch. Ties prefer the earlier epoch. Greedy
generation remains forbidden until color and mirror both reach 12/12 and 6/6
with every required minimum margin strictly positive.

The immutable launch contract is
[`gemma4_color_mirror_centered_content_gate_v18.yaml`](configs/experiments/gemma4_color_mirror_centered_content_gate_v18.yaml).
The guarded stages are independently rerunnable as `make gemma4-v18-preflight`,
`make gemma4-v18-stage1`, `make gemma4-v18-verify-update1`,
`make gemma4-v18-resume-screen`, and `make gemma4-v18-select`; the exact chain
is `make gemma4-v18-screen`. Existing checkpoints are reused, but a fresh
preflight and update-one verification still run before a resume. The strict
epoch selector reads only the resolved YAML, the deterministic selection JSON,
and four checkpoint metadata JSON files; it performs no model inference and
loads neither tensor checkpoints nor runtime/oracle artifacts.
The real preflight passed after two MPS-specific diagnostic defects were found,
regression-tested, and fixed before training. Its isolated first AdamW update
predicted residual-state SHA-256
`599a3e8ba334299f71602e8892080e86facfaab3dce2aef7a258f1859747944a`
and canonical optimizer-state SHA-256
`cd19acb2f1bbe133307723125fc943dabc4bafa479fdf610534a95582a06d393`.
The separately executed epoch-1 update matched both hashes exactly, including
all eight parameter states, all 24 step/moment tensors, their order, and every
AdamW option. Only then did the exact optimizer/history resume run epochs 2--4.

V18 did not pass its behavioral screen. The strict selector chose epoch 4:
color reached 12/12 full-vocabulary sides and 6/6 complete units with minimum
full-vocabulary margin `+0.40625`, but mirror reached only 5/12 sides and 0/6
complete units with minimum margin `-1.1875`. Earlier epochs never jointly
preserved color and met the predeclared mirror continuation minimum. The
selector therefore returned
`screen_failed_no_extension_no_greedy_audit`; no extra epochs, free generation,
promotion, held-out evaluation, chat, leakage, or robot claim is authorized.
This falsifies the hypothesis that merely removing the V16 common-shift mode
with a per-slot centered content gate is sufficient to bind mirrored geometry
to left/right language.

The machine-readable evidence is
[`v18_structural_preflight.json`](reports/gemma4/metrics/v18_structural_preflight.json),
[`v18_update1_match.json`](reports/gemma4/metrics/v18_update1_match.json), and
[`v18_epoch_screen.json`](reports/gemma4/metrics/v18_epoch_screen.json). The
final screen file SHA-256 is
`f1d406dd9ba9b93488c07c905235f1045d2d904241f4e8a7c62f9e43d4854aa5`.

#### Gemma V19 signed-X moment screen — staged, not yet behaviorally accepted

V18's frozen scene prefixes already separate the mirror pair more strongly than
the color pair, but its learned residual update aligns about 5.8 times less with
that mirror signal. V19 tests a narrower architectural hypothesis: an explicit
reflection-odd continuous branch may make the existing geometric signal readable
without changing the full-scene tokenizer or language model. It initializes from
the exact V18 epoch-4 checkpoint and freezes the scene encoder, trained V18
residual, both named LoRA banks, and Gemma decoder.

For the frozen V18 centered content `q_i` and the deterministic X coordinate
`s_i` of each of all 256 scene slots, V19 computes
`m = mean_i(s_i q_i)`, broadcasts `s_i tanh(m)` back to every slot, projects it
through one bias-free FP32 `128 -> 1536` matrix, and centers the resulting delta
over all slots before adding it to the V18 token. The only trainable state is that
196,608-parameter matrix. Its exact-zero initial state has SHA-256
`55b7cb21d0ecbe945cabccfacd5b6aa94693743ceee78443f37a5ca0d1ac68b1`.
The branch is constructed before user text, accounts for every scene slot, and
accepts no question, label, caption, object ID, oracle coordinate, or retrieval
query.

The supervised training controller is also pair-specific but remains confined to
the offline trainer. Both opaque pairs use weight-8 candidate and weight-2
full-vocabulary hinges with language NLL disabled. The already-passing retention
pair uses a `0.25` margin, below its frozen V18 minima, so it supplies exactly zero
gradient at update 0; the failing pair keeps the strict `1.0` margins and drives
the signed branch. These policy roles and QA annotations are not runtime inputs.
The trainer records a canonical policy hash, exact selected-unit hash, frozen-base
hashes, and the same question-independent prefix for every question.

V19 is staged like V18: a clean-source, no-live-step preflight must predict the
exact first AdamW state; a separately executed one-update checkpoint must match
that prediction before updates 2--4 may resume. Color must remain 12/12 sides and
6/6 units with positive minima for any epoch to be eligible. No greedy audit,
promotion, static chat, leakage claim, or embodied navigation is allowed unless
the same checkpoint also reaches mirror 12/12 and 6/6 with positive candidate and
full-vocabulary minima. The immutable experiment overlay is
[`gemma4_color_mirror_signed_x_moment_v19.yaml`](configs/experiments/gemma4_color_mirror_signed_x_moment_v19.yaml).
The guarded stages are `make gemma4-v19-preflight`, `make gemma4-v19-stage1`,
`make gemma4-v19-verify-update1`, `make gemma4-v19-resume-screen`, and
`make gemma4-v19-select`; `make gemma4-v19-screen` executes the complete chain.
If—and only if—the four-update selector returns its predeclared continuation-only
decision, `make gemma4-v19-extension` hash-binds the selected checkpoint, resumes
it in a separate namespace through update 12, and runs the strict final selector.
It refuses both unauthorized continuation and overwrite of a partial run.

### Fail-closed Gemma static evaluation and chat

Gemma evaluation must use the isolated Transformers 5 environment and the exact
approved config/checkpoint pair. `GEMMA4_STATIC_CONFIG` and
`GEMMA4_STATIC_CHECKPOINT` are intentionally empty by default. All Gemma prediction,
scoring, control, and chat Make targets fail before inference unless both are
explicitly supplied, architecture-compatible, and accompanied by
`GEMMA4_STATIC_CHECKPOINT/promotion.json`.

The promotion record is a deliberate scientific acceptance artifact, not something
training writes automatically. It must have `status: "accepted"`, contain a
non-empty list of supporting metric artifact paths in `evidence`, and bind the
exact approved config, checkpoint metadata, and adapter with `config_hash`,
`checkpoint_metadata_sha256`, and `checkpoint_adapter_sha256`. No such record
exists today: v9 free-generates its trained color pair exactly but fails both
controls, v10 partially forgets color, and v11 restores color without learning a
complete selected mirror unit or held-out support unit. V12's auxiliary relation
margin did not transfer to generation, and V13's frozen-scene decoder-bank run
preserves color but generates zero correct mirror or held-out support sides. After
a future checkpoint is accepted, the invocation shape will be:

```bash
make gemma4-evaluate-static \
  GEMMA4_STATIC_CONFIG=/path/to/accepted-config.yaml \
  GEMMA4_STATIC_CHECKPOINT=/path/to/accepted-checkpoint \
  GEMMA4_EVAL_SPLIT=test

make chat-gemma4 \
  GEMMA4_STATIC_CONFIG=/path/to/accepted-config.yaml \
  GEMMA4_STATIC_CHECKPOINT=/path/to/accepted-checkpoint \
  SCENE=scene_000005
```

`make gemma4-evaluate-static` runs those prediction and scoring stages in order.
Predictions go to `reports/gemma4/predictions/test.jsonl`; metrics go to
`reports/gemma4/metrics/static_qa_test.json`. The prediction target first creates
`reports/gemma4/questions/test.json`, a sanitized questions-only manifest, then
passes that path explicitly to inference. The inference process never opens the
answer-bearing QA split; only the separate scoring target does. The inference
driver checkpoints each completed opaque `(scene_id, question_id)` atomically and
resumes only when the effective-config, selected-config-file,
adapter-checkpoint, and exact questions-manifest hashes in its provenance sidecar
still match. This avoids silently mixing outputs from different experiments after
an interrupted local run. To inspect only the sanitization stage, run
`make gemma4-prepare-questions` with the same config and split values.

The same accepted adapter can then be run against all continuous-scene controls
with explicit lineage:

```bash
make gemma4-predict-controls \
  GEMMA4_STATIC_CONFIG=/path/to/accepted-config.yaml \
  GEMMA4_STATIC_CHECKPOINT=/path/to/accepted-checkpoint \
  GEMMA4_EVAL_SPLIT=test
```

Historical v1-v11 checkpoints remain inspectable failure or diagnostic artifacts,
but none is a default for these targets and none may be silently treated as
promoted.

## Preserved legacy CLIP/Qwen entry points

```bash
make setup
make doctor
make download-models
make render-smoke-scan
make build-smoke-map
make semantic-sanity
make generate-dataset
make train
make evaluate
make web
make robot-evaluate
make demo-check
make demo
```

These generic commands retain the earlier CLIP/Qwen infrastructure; they are not
the current Gemma primary experiment. `make demo-check` is a finite offline
preflight for prepared artifacts. It checks
the sanitized 24-frame RGB-D manifest, numeric high-dimensional voxel-map headers,
local model snapshots, visualizations, and adapter/config compatibility without
loading model tensors, creating an MPS tensor, starting chat/MCP, or reading the
oracle/QA/feature directories. It also rejects semantic keys and non-opaque frame
filenames in the sanitized manifest. Its machine-readable result is
`reports/metrics/demo_check.json`. For legacy checkpoints this is a compatibility
check, not a behavioral-success gate. For Gemma invocations, the demo wrapper also
requires the explicit accepted promotion record described above.

There is currently **no accepted primary one-command Gemma demo**. Generic
`make demo` preserves the legacy workflow and may resolve an
architecture-compatible legacy checkpoint; that does not promote the legacy model
or establish scene understanding. If its config selects Gemma, both `CONFIG` and
`CHECKPOINT` must be explicit and the checkpoint must carry a valid promotion
record, otherwise the script exits before model inference. A historical legacy
invocation is:

```bash
make demo \
  CONFIG=configs/experiments/multiscene_anticollapse.yaml \
  CHECKPOINT=data/checkpoints/multiscene_anticollapse/best
```

For the preserved legacy finite inference/leakage demonstration (including
temporarily making the oracle unavailable and checking prefix invariance), run
`make demo-leakage` with an explicit compatible legacy checkpoint. That result does
not transfer to Gemma.
Model downloads are an explicit setup step; normal map building uses the pinned
local vision snapshot in offline mode.

`make report` is artifact-only: it never loads model weights or runs inference. Its
current renderer describes the preserved CLIP/Qwen lineage and is not yet a
Gemma-complete report generator. `reports/final_report.md` is therefore prominently
marked legacy and includes a manually audited current Gemma failure summary; do not
read its legacy tables as current primary-model results.

The prohibited oracle-text upper bound and direct-image VLM control are isolated
under `semantic_3d_chat.evaluation`; the primary chat runtime never imports them.
Download their pinned weights and run the controls independently with:

```bash
make download-baselines
make evaluate-oracle-text
make evaluate-direct-images
```

Both prediction commands checkpoint JSONL results after every question and resume
by opaque `(scene_id, question_id)`. The oracle-text command is intentionally
evaluation-only. The direct-image command reads only complete RGB paths from the
sanitized render manifest. Neither is part of `make chat` or the continuous-3D
primary architecture.

### Preserved legacy continuous-scene controls

The evaluation-only control runner builds one global prefix before any question
for each `(condition, scene)` and reuses that exact prefix for every held-out
question. It writes a separate metrics-compatible JSONL plus a prefix-hash
manifest for the primary path, zero/empty prefix, deterministic wrong-scene
prefix, semantic shuffle, position shuffle, geometry-only, semantics without
XYZ, RGB removal, and normal removal:

```bash
.venv/bin/python -m semantic_3d_chat.evaluation.prepare_questions \
  --config configs/experiments/multiscene_anticollapse.yaml \
  --split test \
  --output reports/questions/test.json

.venv/bin/python -m semantic_3d_chat.evaluation.control_predict \
  --config configs/experiments/multiscene_anticollapse.yaml \
  --split test \
  --questions-manifest reports/questions/test.json \
  --checkpoint data/checkpoints/multiscene_anticollapse/best
```

Question preparation is a separate evaluation-side process that reads QA and
copies only `scene_id`, `question_id`, and question text into a strictly
validated, content-hashed manifest. Prediction processes reject extra record or
top-level fields and refuse to open manifests from `qa` or `oracle` directories.
Reference answers and targets are opened later only by the separate scorer.

Use repeated `--condition CONDITION` arguments for a bounded subset, and
`--max-questions-per-scene 4` for a fast wiring check. Results are written under
`reports/predictions/controls/<split>/`; `manifest.json` records every prefix
hash, its opaque source scene, processed voxel count, affected tensors, and the
fact that question-dependent selection was disabled. The default wrong-scene
control uses a deterministic cyclic derangement; explicit pairings can be given
with repeated `--wrong-scene-pair scene_000001=scene_000002` arguments.

Map-level controls are applied in memory after deterministic high-dimensional
map aggregation and before the globally complete scene encoder. They preserve
the row count and never alter the raw map on disk. `semantics_without_xyz` sets
all XYZ inputs to the room center, removing absolute/Fourier position and spatial
block identity while retaining every semantic row. `geometry_only` zeros learned
visual semantics while retaining XYZ, RGB, normals, confidence, and observation
counts. Score any condition with the normal structured scorer, for example:

```bash
.venv/bin/python -m semantic_3d_chat.evaluation.run \
  --config configs/experiments/multiscene.yaml \
  --references data/qa/test.jsonl \
  --predictions reports/predictions/controls/test/position_shuffle.jsonl \
  --output reports/metrics/position_shuffle.json
```

Legacy runtime data is under `data/rendered`, `data/features`, `data/maps`, and
`data/scene_tokens`; Gemma-derived features and maps are isolated below
`data_gemma4`. Semantic oracle specifications and QA supervision are isolated under
`data/oracle` and `data/qa`. The recorded chat file audit and oracle-unavailable test
apply to the legacy `data/checkpoints/best` lineage only; they have not been run for
Gemma v9-v14 and do not transfer to those checkpoints. Their wiring and failure
results are therefore not Gemma leakage-test results.

## Preserved legacy local web interface

The current web command is legacy CLIP/Qwen infrastructure. After a compatible
legacy adapter checkpoint and fused map are available, start it with:

```bash
make web SCENE=scene_000001 CONFIG=configs/default.yaml
```

Open `http://127.0.0.1:8765`. The server constructs the complete continuous scene
prefix before accepting a question and serializes all questions through that same
unchanged prefix. It shows the existing RGB scan montage, fused point-map preview,
numeric reference viewpoint, answer grounding coordinate, confidence, and prefix
hash. These visual panels are for the human only; their pixels are never passed to
the language model.

The web process serves only an inline application shell and an allowlist below
`reports/figures`. It does not fall back to `data/rendered`, and it refuses oracle,
QA, or feature-cache paths. The access audit is written to
`reports/metrics/web_file_access.json` when the server exits. By default it binds
only to loopback; a non-loopback host requires the explicit `--allow-network` flag.

## Embodied-camera precursor and MCP

The first embodied layer is a kinematic camera base with a circular collision
footprint. It reads only anonymous numerical voxel geometry, not scene-generation
metadata. Movement is atomic and bounded; the complete swept segment is checked
against occupied map surfaces and room limits. A pose-dependent point-splat scan
creates an opaque RGB-D artifact, increments persistent voxel observation counts,
updates coverage, and advances a numeric scene-version counter. A callback seam is
provided for replacing this deterministic map reobservation with the full Blender
render → vision patch-token → map-fusion path.

The direct precursor accepts one strict JSON envelope per line:

```bash
make robot SCENE=scene_000001
{"tool":"turn","arguments":{"angle_degrees":30}}
{"tool":"move_forward","arguments":{"distance_meters":0.2}}
{"tool":"scan","arguments":{}}
```

Print constrained schemas with
`semantic-3d-robot --config configs/default.yaml --schemas`. Every result contains
only protocol status, opaque IDs, numerical pose/velocity, collision state,
coverage, and scene version. It never returns an object name, label, caption, or
semantic relationship.

The official Python MCP SDK is pinned at `mcp[cli]==2.0.0`. Start its local stdio
server with `make mcp`; use `--transport streamable-http` only when an HTTP client
is needed. The nine exposed tools are `get_robot_state`, `look`, `turn`,
`move_forward`, `move_backward`, `move_to`, `scan`, `stop`, and `reset_scene`.

Run the reproducible numerical action/collision benchmark with `make
robot-evaluate`. It deliberately does not claim language-conditioned semantic
target navigation: that requires training the robot-state tokens and action policy
on top of the static scene adapter. The current `RobotStateEncoder` maps normalized
position, sine/cosine orientation, velocity, collision, last motion, coverage, and
stop state to configurable continuous tokens that can be appended directly to the
continuous scene prefix.

## Deterministic multi-scene controls

`configs/experiments/multiscene.yaml` defines ten opaque scene IDs, including four
paired controls: red/blue swap, cube on/under the table, left/right mirroring, and
book present/absent. Pair identities, categories, and expected changes are written
only below `data/oracle`; scan rendering receives the opaque scene ID and stable
base render config. Runtime manifest loading rejects semantic keys and requires RGB
and depth filenames derived from opaque frame IDs.

The batch runner caches completed stages. Inspect commands first, then generate and
render independently:

```bash
make multiscene-dry-run
make generate-scene-batch
make render-scene-batch
```

For a bounded Blender-side smoke test, select scenes explicitly:

```bash
.venv/bin/python scripts/generate_scene_batch.py \
  --stage generate --scene scene_000003 --scene scene_000004
```

The default seed remains `20260808` for `scene_000001`; later independent scenes
use stable seed offsets, while both members of each counterfactual pair share one
seed so all stochastic placement factors remain fixed.

Before a multi-scene adapter run, the deterministic selector treats each
expected-change counterfactual question as an indivisible two-scene training
unit, then fills the 48-question budget from the least represented answer type.
This prevents one shuffled scene from seeing a changed question while its paired
scene does not, and keeps scarce support and metric examples. It never reads or
selects validation/test records. Audit the exact aggregate selection without
loading a model or starting MPS training:

```bash
.venv/bin/python -m semantic_3d_chat.training.train_adapter \
  --config configs/experiments/multiscene.yaml --selection-only
```

The report is written to
`reports/metrics/training_selection_multiscene.json`; it includes an opaque
selection hash, per-scene answer-type counts, and complete/incomplete paired-unit
counts, but no questions or answer labels.

## Coordinate system

World coordinates use **X right, Y forward, Z up**, in meters. Runtime camera
coordinates use **x right, y down, z forward**. Blender camera coordinates are
converted with `diag(1, -1, -1, 1)` before RGB-D projection. Depth arrays store
axial camera-Z depth, not Euclidean ray range.

## Licenses

Project code is [MIT-licensed](LICENSE). CLIP weights/code are distributed under MIT terms; Qwen2.5,
SmolVLM-256M, and the selected Gemma 4 E2B checkpoint are Apache-2.0. Model revisions
and downloaded dependency versions are captured in artifacts and lock files.

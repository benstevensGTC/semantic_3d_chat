# Semantic 3D Chat

`semantic_3d_chat` is a local, deterministic research prototype that turns complete
RGB-D room scans into a persistent continuous 3D semantic memory and injects that
memory into a frozen local causal language model. The primary inference path never
receives an object list, caption, textual scene graph, simulator labels, or oracle
metadata.

## Executive status — 2026-08-16

The complete local vertical slice is runnable, including strict static chat, the
persistent embodied MCP process, and a Blender-native semantic-goal rover. V89 is
the promoted static-chat runtime for `scene_000001`; the rover now uses a separate
actual-Gemma closed-loop waypoint policy over the 258-token scene memory and four
numeric robot-state tokens. The runtime-aligned waypoint DAgger V14 checkpoint
is now the model-only operator default after passing fresh local lap, face-cube,
approach-chair, and oracle-isolation checks. This is a working single-room
vertical slice, not held-out rover generalization or project-wide final
acceptance. The authoritative current snapshot is
[`reports/final_report.md`](reports/final_report.md); the long experiment diary
later in this README is retained as historical engineering evidence.

- The selected integrated model is the locally cached
  `google/gemma-4-E2B-it`, not a cloud API. One complete image pass produces a
  48×48 field retaining 3072D float16 middle, late, and language-aligned features.
- Exact RGB-D geometry passes: 6.27e-6 px reprojection RMSE and 1.03e-7 m depth
  round-trip RMSE. The reference map fuses 301,056 observations into 74,699
  occupied 5 cm voxels.
- Across 17 development scenes and 221 semantic queries, zero-shot 3D
  localization reaches 48.42% mean top-1 and 78.28% mean top-k; mean P@k is
  43.17% versus an 8.09% random baseline.
- The promoted strict V89 scene-one runtime supplies one immutable
  `[1,738,1536]` memory directly to Gemma: 736 continuous environmental payload
  tokens plus native BOI/EOI, compiled before the question and reused exactly.
  It has zero question-derived environmental tokens and no retrieval or
  question-conditioned scene readout. It scored 122/138 (88.41%) on its known,
  trained scene-one development set and answered the finite operator smoke
  `yes`, `red`, `left` at 3/3. All model, packaging, oracle-unavailable, and
  runtime-isolation gates passed. This is a real runnable overfit/general-chat
  proof, not held-out-scene evidence.
- Separate V85 scene-disjoint development evidence is 214/384 (55.73%) versus a
  62/384 (16.15%) answer-frequency baseline, with 8 prediction-changing paired
  units. It demonstrates above-prior multiscene transfer, but it is internal
  development evidence; the official test remains unopened. The earlier V54/V55
  41.20% exact run is retained as historical base-checkpoint evidence.
- V94's immutable full-profile post-hoc diagnostic scored 143/216 (66.20%) on
  its six-scene primary arm versus 85/216 (39.35%) with the full scene zeroed.
  That 26.85-point drop is real aggregate scene dependence. Exact binding is
  still weak: paired wrong-scene memory scored 140/216 (only a 1.39-point drop),
  and fully permuting all 736 interior scene tokens remained 143/216 despite
  changing 57 outputs. XYZ and 3072D semantic-row shuffles reduced accuracy to
  132/216 and 131/216; removing RGB improved it to 147/216. Normals were already
  zero and viewpoint is not consumed, so neither is reported as a meaningful
  control. This is a label-isolated, model-free-scored, terminal post-hoc
  diagnostic—not preregistered promotion, held-out/generalization, official, or
  final evidence. V94 remains unpromoted.
- V95 completed its preregistered 40-scene strict-causal training run in
  13,234.8 s on MPS with only a fresh 143,360-parameter bridge trainable. Its
  sealed post-fixed-final known-development gate scored **167/216 (77.31%)**
  primary, **36/216** with zero payload, **127/216** with all 736 interior tokens
  permuted, and **164/216** with paired wrong-scene memory. Control-minus-primary
  mean answer-NLL gaps were +2.296439, +0.616878, and +0.054092 respectively;
  the changed-side paired-wrong gap was +0.486824. The decisive counterfactual
  result remained **13/24 sides, 1/12 complete units, and 2/12 changed
  predictions**, so the preregistered gate failed. V95 was not promoted,
  deferred-final materialization remains locked, and V89 remains the default.
  This is sealed known-development negative evidence—not held-out-final,
  official, generalization, or final-acceptance evidence. The training, structured,
  NLL, final-gate, and evidence-seal SHA-256 values are serialized in
  [`current_metrics.json`](reports/metrics/current_metrics.json).
- V96 completed 285 optimizer updates in 7,231 s and retained the same strict,
  question-independent `[1,738,1536]` continuous scene input. Its separately
  sealed, auth-v2 known-development evaluation scored **174/216 (80.56%)**:
  attribute 25/48, count 41/42, metric 6/6, orientation 6/6, presence 40/42,
  spatial relation 38/48, and support 18/24. Controls scored 36/216 with the
  payload zeroed, 128/216 with all 736 interior tokens permuted, and 165/216
  with paired wrong-scene memory. Primary, zero, permutation, and wrong-scene
  mean answer NLL were 0.277091, 2.616119, 0.882842, and 0.329378; the changed-
  side wrong-minus-primary margin was +0.470579. Counterfactuals reached 16/24
  correct sides and 4/12 complete units, but only 5/12 prediction-changing units;
  24/192 invariant sides also changed. Thus 19/21 gates passed and the two exact
  failures were prediction changes (5, minimum 7) and invariant false changes
  (24, maximum 20). The implementation seal, candidate attestation, prediction,
  label-isolated scoring, and NLL audit recorded zero protected reads and bound
  the local Gemma snapshot, adapter bank, source closure, memories, and outputs.
  V96 was **not** promoted, no deferred-final scenes were materialized, and V89
  remains the default. This is sealed known-development evidence, not held-out
  final, official, generalization, or final-acceptance evidence.
- The current model-only rover is waypoint DAgger V14, checkpoint SHA-256
  `149f5e04de1d8305e642909443f03b96894edc3ece67e4500eacec8f5ca81e7c`.
  On the fresh live lap, Gemma chose all 76 decisions (29 `FACE`, 46 `MOVE_TO`,
  one `STOP`), traveled 18.715408 m, swept 4.272962 m², returned within
  0.048733 m, and incurred zero rejected decisions. It also passed `Face the
  cube, then stop.` in two decisions at 0.201562° yaw error and `Move close to
  the chair, then stop.` with 0.263674 m center progress and 0.431456 m bounding-
  box standoff; the latter took 16 model decisions and eight collision rejections.
  These are live `scene_000001` results, not held-out evidence.
- The corrected Blender operator path is deliberately different from the earlier
  raw-command toy. It starts from the already scanned, complete 74,699-voxel map,
  uses the immutable 256-latent base scene memory (`[1,258,1536]` including
  boundaries), adds four continuous numeric robot-state tokens, and accepts only
  outcome-level language from the user. On every closed-loop step the local
  `google/gemma-4-E2B-it` decoder consumes that complete continuous prefix, the
  raw goal, and numeric action history; learned heads select `MOVE_TO`, `FACE`, or
  `STOP` and emit the waypoint or heading. Deterministic code only transforms the
  exact model output into world coordinates, rejects unsafe proposals, and
  executes accepted primitives. It never selects a destination, route, recovery
  waypoint, heading, or synthetic stop. Rover-camera observations remain disabled
  in this static-map mode. The prior V3 fallbacks and global patrol/between
  planners are retained only as historical code and are not reachable from the
  model-only default runtime. Direct UI motion controls are also disabled.
- The offline oracle-only QA generator now covers object location, containment,
  fixed-yaw viewpoint-relative reasoning, metric distance/comparisons, and
  exact-ray-hit uncertainty. It gates ambiguous or duplicate-category questions,
  retains structured grounding/reference coordinates, and withholds zero-pixel
  objects from answerable rows. This is source plus synthetic-test coverage; no
  existing QA split was regenerated and no model accuracy on these new families
  is claimed yet.
- V66b was an **enhanced readout ablation**: an always-on continuous
  schema-7 controller over the immutable full 256-token scene prefix. It uses no
  question-dependent retrieval and every scene latent contributes, but it appends
  four scene-and-question-conditioned tokens. It is therefore not being claimed
  as the strict fixed-environment-token primary. Its 576-row, 12-fold
  pair-held-out training gate completed at 409/571 supported exact, but only
  37/75 changed-side exact, 5/35 complete changed pairs, and 16/35 prediction
  changes. It failed the locked scene-dependence gate, and the code correctly
  published no checkpoint. No V66b held-out or leakage success is claimed.
- V67 through V71 are now sealed negative results for that numeric scene-control
  line. V70 changed only the global signature from the
  first 8 to the first 32 fixed low-frequency DCT moments over all 256 scene
  latents. Across 12 pair-held-out folds it reached 484/571 supported classes,
  55/75 changed classes, 15/35 complete units, 16/35 prediction changes, and
  51/70 positive own-over-opposite margins. It passed every locked gate except
  prediction changes (minimum 20) and positive margins (minimum 53). This is an
  honest falsification: continuous pair diagnostics improved, but discrete
  scene-conditioned decisions did not. The fail-closed runner used no Gemma
  generation, compiled no atlas, ran no full behavioral evaluation, and wrote no
  checkpoint. V71 then tested the measured complement directly: independent
  8-moment and 32-moment all-latent value branches, separate projections/trunks/
  heads, a fold-trained bounded global fusion scalar, and the exact V69
  `balanced_extrapolation_010` training arm. Across the same 12 isolated folds it
  reached 489/571 supported classes, 57/75 changed classes, 17/35 complete units,
  17/35 prediction changes, 52/70 positive margins, and 28/35 positive pair
  deltas. It again missed only prediction changes (by 3) and positive margins
  (by 1). The fusion stayed effectively equal-weight (0.499998–0.500006), so the
  two independently trained paths did not learn a useful global preference.
  V71 used 767.19 s of numeric fitting and 774.21 s wall time under its 1200 s
  cap; no Gemma generation, atlas, full run, or checkpoint occurred.
  Re-authenticate the immutable V71 preregistration, result, all fold sums,
  source hashes, and absent checkpoint with `make gemma4-v71-authenticate`.
- V72 tested question-adaptive fusion of the same complete 8- and 32-moment
  all-latent branches. Its first pair-disjoint development fold used no held rows
  or held teacher outputs during calibration, but adaptive fusion was worse than
  the frozen 32-moment branch: 1/4 versus 2/4 complete units, 1/4 versus 2/4
  prediction changes, and 5/8 versus 6/8 positive own-over-opposite margins. The
  sealed terminal rule stopped the remaining folds and withheld the full numeric
  screen, internal validation, Gemma generation, and checkpoint publication.
  V72 is an authenticated development-negative mechanism test, not a model.
- The separate fixed-prefix PLE-reader V1--V5 chain is now terminal negative
  evidence. V1 failed a numerical smoke tolerance; V2 and V3 aborted before
  optimizer construction; V4 completed 40 updates and V5 completed 80. V5
  improved answer NLL from 3.235832 to 2.944323 and retained the text control,
  but wrong-prefix positive sides regressed from 30/52 to 28/52 and complete
  changed units from 12/26 to 9/26. Its locked gate therefore skipped greedy
  evaluation and published no checkpoint. The 57--62 deferred and 25--30 final
  scenes remained unopened.
- The fixed-prefix upper-decoder reader V6 did not reach training. Its immutable
  preregistration authorized one zero-update real-model MPS smoke, which failed
  the byte-exact full-sequence-versus-answer-tail selected-logit equality gate in
  12.81 s. It stopped before reader gradients, optimizer construction, training,
  or checkpoint publication. The whole-execution audit recorded 233 unique files,
  zero forbidden reads, and no deferred/final-QA access. The planned 96 updates
  were never run, so V6 supplies no behavioral evidence for the reader.
- V6.1 replaced that invalid byte-equality requirement with a preregistered
  bounded objective-and-gradient comparison and consumed exactly one new
  zero-update MPS smoke. Objective equivalence passed: raw logits differed by at
  most 0.125 (RMS 0.000378110276), per-token NLL differed by 0, and maximum
  Jensen-Shannon divergence was 1.63732948148244e-11. The correct, wrong-prefix,
  and broad gradient comparisons passed individually, but their combined
  first-schedule gradient failed: cosine 0.9998925988237569 was below 0.99999,
  and relative L2 0.014655751591121162 exceeded 0.005. Its 1.000092 norm ratio
  stayed in bounds and the full aggregate gradient was nonzero (L2 0.5059195),
  so this is a bounded numerical-path mismatch—not a missing-gradient failure.
  The fail-closed gate built no optimizer, ran no training, and published no
  checkpoint. The audit recorded 240 files and zero forbidden reads. Exact
  release, attempt, and terminal hashes are `4456ebd1…`, `ec462122…`, and
  `099c1fa6…`.
- V6.2 removed the disputed shape-specialized path and consumed one sealed,
  exact-full-forward training attempt. It completed all 96 AdamW updates in
  872.414 s. Validation answer NLL improved from 3.2358 to 1.9157 and retention
  passed, but causal scene selectivity regressed: expanded positive margins fell
  from 0.6647 to 0.5941, curated complete units fell from 12 to 11 of 26, and
  orientation positive margins collapsed from 0.8571 to 0.1429. The locked gate
  therefore skipped greedy evaluation and published no checkpoint. Its
  whole-execution audit covered 246 loaded files with zero forbidden reads and no
  deferred/final-scene access. Peak process RSS was 6.42 GB and reported MPS
  driver allocation was 15.82 GB. Exact release and terminal hashes are
  `c2cc4110…` and `e86b417d…`.
- V6.3 moved the tiny trainable reader surface into Gemma's shared K/V attention
  projections at physical layers 13--14. Its full-forward gradient screen passed,
  then an eight-update, 40-pair-unit train-only pilot improved positive
  wrong-prefix margins from 48/80 to 49/80, complete paired units from 16/40 to
  18/40, and mean margin from 0.441833 to 0.447165. Retention stayed bounded at
  mean KL 0.000341 with exact next-token top-1 agreement, and the 125-file audit
  had zero forbidden reads and no oracle, validation, or deferred/final access.
  This is a positive diagnostic only: it is non-disjoint, ran no validation or
  generation, and published no runtime checkpoint. It authorized only a
  pair-disjoint V6.4 train-only confirmation; that confirmation has now completed
  and failed.
- V6.4 held out 12 units from three complete physical pairs and all six associated
  scenes, trained on 28 units from 18 disjoint scenes, and completed 12 exact
  full-forward updates in 394.812 s. Training positive margins improved from
  36/56 to 39/56 and complete units from 13/28 to 15/28, but the result did not
  generalize: held complete units stayed 3/12, held mean margin fell by 0.016739,
  and held margin softplus worsened by 0.011246. Those two locked gates failed.
  Retention and its 126-file isolation audit passed with zero forbidden reads and
  no protected inputs. The exact V6.3/V6.4 attention surface is now terminal: no
  continuation, internal validation, generation, runtime promotion, or checkpoint
  is authorized. The current static-reader blocker is causal pair-disjoint
  generalization itself, not a pending run.
- V73 and V74 are terminal negatives; V75 now has positive internal-development,
  promoted-runtime, leakage-isolation, and official-validation evidence. V73's
  full-scene cross-attention reader failed its numeric screen:
  16.19% broad supported accuracy, 4% changed-side accuracy, and 0
  prediction-change units, versus 21.93%, 16%, and 18 for frozen DCT-40. V74
  passed all eight teacher-proxy numeric gates, but real local Gemma fell to 2/16
  versus 6/16 for V54. Its bounded historical-train-only NLL repair lowered NLL
  from 6.610632 to 3.855856 and raised train exact sides from 2/18 to 11/18, yet
  the untouched smoke stayed 6/16—tied with both V54 and wrong-scene—so V74 was
  rejected. V75's nonlinear reader also passed every numeric proxy gate and its
  initial smoke was only 4/16 versus 5/16 wrong-scene. The subsequent V75 NLL
  candidate (`d0127553…`) first reached 9/16 versus 6/16 for V54 in its smoke.
  Its complete 384-row internal-development result is now 295/384 (76.82%) versus
  148/384 (38.54%) for the exact cached V54 baseline. The paired wrong-scene arm
  scored 278/384 (72.40%) against the original targets. More importantly, on the
  52 physically changed sides, the correct scene scored 31/52 against its target;
  after substituting the paired scene, outputs scored only 14/52 against the old
  target but 31/52 against the paired scene's target. Complete changed units were
  6/26 correct-scene/original-target, 0/26 wrong-scene/original-target, and 6/26
  wrong-scene/paired-target; 24/52 outputs changed between scene arms. V54 reached
  only 18/52 changed sides, 0/26 complete units, and 6/26 prediction-changing
  units, versus V75's 31/52, 6/26, and 12/26. This is meaningful causal evidence
  that answers follow the continuous scene input. It remains an **internal
  development** result and is not conflated with the later validation.
- V75 has since been promoted into the sealed two-file controller release
  `data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1`
  (`control.safetensors` plus `runtime_metadata.json`). A live three-question
  isolation run renamed the oracle, training-artifact, and teacher-artifact
  directories so they were unavailable during inference, then restored them.
  It audited 4,198 file reads with zero forbidden accesses, loaded no QA/oracle
  or training artifact, computed the full scene prefix before the first question,
  and reused one prefix hash. The runtime receives no environmental text and does
  no question-dependent scene retrieval.
- The subsequent one-candidate official validation covered 216 questions from six
  held-out validation scenes. V75 scored 167/216 canonical (77.31%) and 76.39%
  normalized exact. Canonical results were attribute 30/48 (62.5%), count 40/42
  (95.24%), metric 6/6, orientation 6/6, presence 38/42 (90.48%), spatial
  relation 28/48 (58.33%), and support 19/24 (79.17%). Counterfactuals reached
  7/12 complete units, 17/24 correct sides, and 8/12 prediction-changing units,
  with success in all three physical-change families. Nine of ten preregistered
  gates passed; the overall validation is nevertheless **failed** because spatial
  relations narrowly missed the 60% minimum. Grounding was weak (2.136 m mean
  error and 0/132 targets within 1 m). Answer references were opened only by the
  isolated scorer, not inference. Official test, deferred-final, and simulator
  oracle inputs remain unopened.
- V75 preserves the complete, prequestion 256-latent base scene prefix and every
  scene latent influences its output. It also appends four question-conditioned
  continuous control tokens, so the total environment-conditioned input is not
  invariant even though the base prefix is. It is therefore a strong continuous-
  scene enhanced readout and a promoted runtime, but not the strict identical-
  total-input primary result or final project acceptance. V75 remains an
  enhanced-readout comparator rather than the strict result.
- The V75 fixed-prefix atlas mechanism has now been executed and hash-pinned
  (`db2f161fb043a3e259e881e2c68c0b8e14e4708805725969809e5dfb68f16725`),
  using the exact sealed V75 controller weights. It compiled all 256 base scene
  latents plus all 480 atlas tokens into one 738-token scene-only prefix before
  any user question; every base latent, probe, and atlas token was preserved.
  The structural check loaded no Gemma model, questions, answers, oracle,
  protected split, or environmental text. This removes the former dependency on
  rejected V66b. A subsequent sealed 16-row Gemma behavior run then compiled all
  16 prefixes before opening questions and kept every 738-token prefix invariant.
  The atlas scored 6/16 (37.5%), exactly tying frozen V54 and trailing direct
  exact V75 at 9/16 (56.25%); prediction-changing paired units were 1/8, 1/8,
  and 2/8 respectively. The predictor audited 119 reads with zero forbidden
  access and never loaded its isolated scorer references. This is negative
  historical-training-pool evidence: the rows were pair- and scene-disjoint but
  12 questions overlapped training, so it is not question-disjoint or official
  validation, and no atlas runtime was promoted.
- V81 turns that layout into a runnable, compile/runtime-separated continuous
  scene memory. Its runtime artifact contains exactly `memory.safetensors` and
  sanitized `runtime_metadata.json`: one bfloat16 `[1,738,1536]` memory with
  canonical prequestion hash
  `a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37`.
  It retains all 480 atlas tokens and all 256 base scene latents. The latest user
  text performs a dense read across all 96 atlas groups; every one of the 384
  scene-value tokens receives positive weight, with no semantic/spatial top-k
  selection, retrieval, or mutation of the serialized scene memory. A real
  three-question local run kept both the 738-token memory and 258-token base
  prefix invariant, audited 4,204 reads with zero forbidden accesses, and worked
  while the oracle directory was unavailable. Chat loaded neither the compile-time
  V75 controller nor the numeric probe bank.
- V81 is nevertheless **experimental and not promoted**. Its bounded 16-row
  historical-development control scored 8/16 versus 6/16 for frozen V54, 3/16
  with shuffled atlas values, 1/16 with an exactly zero environmental payload,
  and 9/16 with the paired wrong scene. It changed 2/8 paired units but failed
  the 9/16 candidate, +3-over-V54, and correct-over-wrong-scene gates. The rows
  were pair- and scene-disjoint but not question-disjoint, and official validation
  was not opened. At that historical point V75 remained the default; strict V89
  later superseded it as the operator default.
- V82 is the first fitted dense learned reader over the exact immutable V81
  memory. It has 688,130 trainable parameters, gives all 384 atlas values and all
  256 base latents positive attention floors, performs no top-k selection or
  question-dependent retrieval, and returns exact zero controls for an empty
  environmental payload. Its pair- and scene-disjoint historical numeric
  development fold covered 384 rows across 16 scenes and reached 0.991352 mean
  control cosine with 0.029239 normalized MSE; shuffled atlas values changed the
  output by 0.112454 RMS. The real local-Gemma behavioral control still failed:
  V82 scored 8/16, versus 6/16 frozen V54, 3/16 shuffled atlas, 1/16 zero
  environment, and 6/16 wrong scene. Although the +2 correct-over-wrong-scene
  gap passed, V82 missed the 9/16 candidate and +3-over-V54 gates, tied V81,
  trailed direct V75's 9/16 comparator, and is **not promoted or officially
  validated**. The config's original `implemented_model_free_preflight_only_...`
  status is retained as a hash-sealed preregistration snapshot; measured outcome
  status lives in the authenticated development and behavioral reports.
- V83 tests the strictest direct fixed-memory form: the exact immutable
  `[1,738,1536]` memory is inserted directly into Gemma's native image-prefix
  slot before the question. All 738 tokens remain visible with exact native
  BOI/EOI and PAD-PLE handling. There is no separate question-conditioned
  environmental reader, retrieval, top-k selection, or control activation, and
  the count of question-derived environmental tokens is zero. This structural
  contract passed, but the isolated 16-row historical-development behavior did
  not: direct V83 scored 6/16, frozen V54 6/16, paired wrong scene 7/16,
  shuffled atlas 6/16, and zero payload 5/16. V83 changed only 1/8
  counterfactual units and failed every promotion gate. It is authenticated but
  **not promoted or officially validated**. V75 was still the default at that
  point; V89 is now the strict default runtime.
- V84.1 establishes the narrower causal wiring result that V83 lacked. A
  preregistered train-only follow-up supplied the same complete immutable
  `[1,738,1536]` memories directly to Gemma, retained zero question-derived
  environmental tokens/readout/retrieval, froze Gemma plus all six inherited
  V54 LoRA banks, and trained only a fresh 55,296-parameter rank-4 LoRA on the
  final layer-34 MLP projection. After a fixed 32 updates with paired-wrong-scene
  margin loss, the identical question produced `on` for `scene_000019` and
  `under` for `scene_000020`. Correct-scene NLLs were 0.0322 and 0.0291, while
  paired-wrong-minus-correct margins were +2.5908 and +1.3780; both scene-memory
  hashes remained byte-identical. This passes the preregistered **two-scene
  wiring/overfit** gate, not held-scene generalization: no development behavior,
  official split, sealed historical behavior set, or oracle was opened, and the
  candidate is explicitly **not runtime-promoted**.
- The optional V78 grounding sidecar also runs on V81 without changing the fixed
  scene-memory hash. A live `Where is the chair?` check scored all 256 scene
  latents, predicted `[-0.7445, 0.4775, 0.6310]` m with 0.278 m map-support
  distance and 0.784 confidence, and audited 5,208 reads with zero forbidden
  accesses. Its text answer was honestly `unknown`, so this is mechanism evidence,
  not a grounding-accuracy result.
- The default `make demo` now launches the promoted strict V89 scene-one runtime;
  `make demo-smoke` is its finite three-question proof. A current local MPS run
  answered `yes`, `red`, and `left`, reused the exact
  `a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37`
  environment-input hash for every question, audited 5,208 reads with zero
  forbidden access, and loaded no training or evaluation report. The independent
  release smoke also passed while the oracle directory was physically unavailable.
  V75 remains available as the stronger scene-disjoint but question-conditioned
  enhanced-readout comparator; its historical five-prompt sample answered
  `yes`, `red`, then the physically wrong `right`, and failed broad support prompts
  closed as `unknown`. Broad free-form list QA remains unsupported.
- V76 is complete and rejected. It trained the raw pre-NLL V75 branch
  (`182481dd…`, not promoted NLL V75 `d0127553…`) on all 40 historical changed
  units / 80 sides for 80 steps. Training answer NLL improved from 7.900908 to
  4.371634 and paired margin from 4.154925 to 5.632363 in 471.26 s, but its held
  smoke reached only 6/16—tied with V54 and the wrong-scene arm—with zero
  correct-over-wrong-scene advantage and four prediction-changing units. It
  published no checkpoint and is superseded by promoted V75. V77 then started
  from promoted NLL V75 and made a bounded historical-training-pool repair over
  72 of 576 rows (28 answer classes, 47 question templates, nine optimizer
  steps). On a separate 24-row training-pool measurement slice, correct-answer
  NLL fell from 2.518216 to 1.604626 and negative-answer margin rose from
  6.736418 to 7.248879. Its 16-row pair-disjoint Gemma smoke reached 9/16 versus
  6/16 for V54 and 6/16 with the wrong scene, but only 2/8 paired units changed
  predictions. The complete 384-row internal screen then reached 299/384 (77.86%)
  versus V75's 295/384 (76.82%), with 35 versus 33 prediction-changing units.
  This four-answer gain has no matched full wrong-scene arm and is internal-only,
  so V77 remains quarantined: no runtime checkpoint was published and official
  validation was not reopened.
- V79 performed the bounded, matched historical relation/counterfactual follow-up.
  It trained the V75 reader for 15 optimizer steps over 120 historical rows, then
  ran a locked 28-row, 14-unit scene-disjoint screen. Correct-scene counts were
  V75 18/28, V77 19/28, and V79 20/28; all wrong-scene arms were 5/28, so the
  correct-minus-wrong gaps improved 13 -> 14 -> 15. However, prediction-changing
  units were 9/14, 10/14, and 9/14: V79 missed the locked best-baseline gate.
  The full 384-row run and runtime publication were blocked. The exact diagnostic
  candidate is `bdd4b6f4…`, but it is not a runtime checkpoint, official result,
  or promotion.
- V78 adds optional continuous numeric grounding without changing the V75 answer
  generator. Its exact two-file diagnostic release contains
  `grounding.safetensors`
  (`3c7914a61e63d80617e7fcfca122e02eec30d15af5a43e910daa0cd6c0b501c4`)
  and `metadata.json`
  (`ea5536dc078b7707000404661c92bdb198dc0c40bbf73cf987d5f94b20464480`).
  A real three-question local isolation run
  (`070feddd71141dfa75f8ca807ec47275225e8b056fce1e2f3862f88e89fc6215`)
  audited 4,204 reads with zero forbidden accesses while the oracle directory
  was unavailable. The prequestion base-prefix hash and the V78 256-token scene
  input were each invariant across questions. Numeric grounding ran, but answer
  text remained V75's output: the two `Where` prompts returned `right` and `red`,
  so this is runtime-integration evidence, not strong conversational location QA.
  The reproducible point-cloud evaluator now reconstructs all 94 historical-held
  predictions exactly without loading full Gemma: mean 3D error is 0.527 m,
  92.55% fall within 1 m, and the mean nearest sanitized-map-voxel distance is
  0.179 m. It renders one prediction/target overlay per held scene using a fixed
  pre-error selection rule; the six-example PNG is deterministic at SHA-256
  `8289bfa9d40097336c834a00555f43aef2e51dfe9b7cd04113f1e81876b0bfb2`.
  Oracle target markers exist only in the evaluation
  report/figure; no runtime artifact is written. Historical held grounding and
  its controls remain internal diagnostics; V78 is neither officially validated
  nor promoted, and its readout remains question-conditioned.
- V78 checkpoint forwarding is wired into the existing passing embodied
  RGB-D scan/map/prefix-refresh path. Lightweight preflight and finite
  scan-plus-answer Make targets exist, but there is no sealed V78-specific
  embodied transcript or navigation score yet.
- The preserved strict V54 web/mechanism comparator prepares and verifies a minimal two-file
  inference release: `adapter.safetensors` plus `runtime_metadata.json`. Its
  manifest declares zero environmental-text inputs and no training metadata.
  This is safer packaging of the existing below-acceptance V54 mechanism demo,
  not model promotion or a claim that static QA acceptance has been reached.
- A fresh embodied observation has already completed the full Blender RGB-D ->
  one complete-image Gemma pass -> map fusion -> scene-prefix refresh transaction
  in 31.31 s. The label-free development navigation policy succeeds on 8/9
  targets with zero collisions; this is not yet held-out or autonomous LLM tool
  use. A separately sealed deterministic seam maps 18 numeric robot-state values
  into 4×1536 continuous tokens. It is an untrained interface proof
  (`task_trained: false`), not a learned navigation controller.
- The opt-in local-Gemma action seam now has a live strict-prefix success: Gemma
  consumed the continuous scene plus numeric robot-state tokens, proposed a valid
  bounded `turn(10°)` JSON call, passed schema/context validation, and executed it
  with no collision or fallback. This is one protocol smoke, not navigation
  accuracy or evidence of a trained tool policy.
- A real decoder-level Gemma tool policy V2.2 was also trained under a sealed
  early gate: 64 optimizer updates / 512 microbatches reduced training loss from
  2.414296 to 0.234181. Across all 2,268 rows in eight held-out scenes it reached
  0.377758 answer-token NLL and 87.13% token accuracy, but only 17.42% exact JSON
  sequences, 26.41% valid schemas, and 24.12% correct tools. Those three locked
  gates failed, so greedy generation and saved-runtime probing were not run and
  no V2.2 runtime checkpoint was published. This is useful negative evidence,
  not a working learned action decoder.
- Historical V3 supervised-action evidence grounds each
  conversational navigation target by comparing local-Gemma text embeddings with
  every voxel in the continuous semantic map, then supplies only numeric target
  state to the bounded controller. It completed 5/6 live tasks with zero
  collisions, action failures, or policy rejections. The grounded-target controls
  are causal but direct scene-prefix controls remain weak, so this is authenticated
  partial evidence rather than a complete navigation claim. Its 5/6 live result
  is now authenticated against byte-exact historical shared-source snapshots;
  compatibility of that sealed run with today's successor source is not claimed.
  V1/V2 remain sealed historical controls.
- The historical V3.3 development runtime passed all 6/6 tasks in one sealed
  run with 28 actions, zero collisions, zero action failures, and zero policy
  rejections. The scan-update task used `scan`, seven bounded `move_to` waypoints,
  and `stop`; it made 2.114 m of target progress and stopped at 0.287 m after a
  real map/prefix update. All 28 decision-context and prefix-chain checks passed,
  runtime environmental-text inputs were empty, and oracle/QA runtime reads were
  zero. This is the supervised continuous-semantic V3 controller combined with
  the deterministic V3.3 numeric planner—not native Gemma function calling. The
  same one-scene benchmark was used to diagnose V3.1/V3.2, so 6/6 is accepted
  development calibration only, not held-out or cross-scene generalization.
- Collision-aware V4.1 was a preregistered single-arm successor to V3. It added
  24 anonymous robot-frame clearance rays, a collision-risk head, and an exact
  mask that redirects an unsafe movement to the highest-ranked safe nonterminal
  action. Its held-out action accuracy was 93.47%, scan-update accuracy 91.03%,
  collision-risk accuracy 98.23%, and exact unsafe-motion rejection 100% (21/21),
  but shuffled clearance reduced the obstacle/scan-update mean by only 0.049565
  against the preregistered 0.10 minimum. It therefore passed 13/14 gates, was
  rejected, wrote no checkpoint, and was never exposed to the live benchmark.
- The official MCP 2.0 stdio boundary exposes all nine numerical tools, rejects
  malformed or out-of-range calls without changing state, and transactionally
  resets the loaded scene's robot pose, renderer coverage, persistent map, and
  continuous prefix. Cross-scene reset remains intentionally fail-closed.
  Successful `look`, `turn`, `move_forward`, `move_backward`, and `move_to` calls
  now auto-scan and transactionally refresh the map and continuous prefix when
  enabled; tests cover all five actions and prove rejected collisions do not scan.
  A live direct-runtime 15° turn has now captured `o_000002` through the second
  complete-image encoder call, advanced both scene and map to version 2, and
  changed both the persistent map and continuous prefix. The 51.89 s run audited
  125 loaded files with zero forbidden accesses and no oracle/QA reads.
  The actual official-SDK MCP 2.0 stdio boundary has now also run a live explicit
  scan followed by a 15° turn and automatic scan with the V54 base and active V75
  continuous controller. It advanced map version `0 -> 1 -> 2`, increased source
  voxels `74,699 -> 74,897 -> 75,594`, and processed 50,176 valid-depth pixels in
  each complete-image observation. Map, base-prefix, V75 control, active-prefix,
  robot-state/token, and binding hashes changed after both observations while the
  robot-state encoder identity stayed fixed. The 57.29 s run audited 4,178 reads
  with zero forbidden accesses and returned no environmental text or semantic
  labels. This is a one-scene integration smoke, not a navigation success-rate
  claim.
- A separately sealed four-record direct embodied conversation reproduced the
  map-version `0 -> 1 -> 2` explicit-scan/15°-turn-autoscan chain, preserved a
  prequestion scene K/V cache, then generated `yes` at version 2 using exactly the
  refreshed active-prefix hash produced by the turn. Both observations had 50,176
  valid-depth pixels; all environmental-text input arrays were empty, and the
  123-read audit had zero forbidden accesses. This proves a local Gemma answer can
  be bound to newly observed continuous scene state; it does not establish broad
  conversational-navigation competence.
- The historical camera-refresh `ConversationalEmbodiedAgent` also crossed the actual
  official-SDK MCP stdio process boundary. Real smokes on `scene_000001` and
  `scene_000031` each executed `scan -> turn -> stop`, accepted exactly four
  continuous binding refreshes, and produced map versions `0 -> 1 -> 2 -> 2`.
  Scan and turn/autoscan changed each scene prefix; stop preserved the unchanged
  scene prefix while changing the numeric robot-state binding. Both runs returned
  numeric structured receipts only, exposed no environmental text, and audited
  4,182 reads apiece with zero forbidden accesses. The scenes ended with distinct
  scene-prefix hashes. This proves two-scene transport and continuous-state refresh,
  not held-out semantic instruction-following or navigation accuracy.
- A historical finite natural-language semantic MCP successor closed those two seams in
  one real `scene_000001` episode. For `Face the chair, then stop.`, selective
  local-Gemma tied-token rows grounded only the user-supplied target phrase against
  every current map voxel; the V3 numeric alignment interlock then issued
  `scan -> turn(45°) -> turn(21.923°) -> stop` exclusively through the official
  MCP 2.0 stdio subprocess. The run stopped at a fresh 0.325° continuous-grounding
  residual after 85.73 s. Each observation advanced the map exactly once and
  changed the map, scene-prefix, and active-prefix hashes; stop preserved the map
  and scene prefix while changing the numeric robot-state prefix. It scored all
  `74,897 -> 75,468 -> 76,220` active voxels, returned no semantic text in tool
  receipts, and both process audits passed with zero forbidden reads (93 client,
  4,185 server). A physically separate post-run oracle scorer measured 0.146°
  chair-heading error and zero collisions. This policy explicitly uses neither
  Gemma function calling nor the learned V3 action head: it is selective-Gemma
  continuous grounding plus deterministic numeric convergence over real MCP.
  It is one development scene and one instruction family; newly fused views also
  moved the semantic target XYZ substantially, so broader grounding stability
  remains an important limitation despite the correct final heading.
- Historical hybrid semantic face-target navigation completed 2/2 development episodes on
  `scene_000001` and `scene_000031`, with zero collisions, zero forbidden runtime
  reads, and no runtime oracle inputs. V3 supplies the learned bounded action;
  after its turn output stalls, a numeric convergence interlock uses only the
  continuously grounded target XYZ and robot yaw, then requires fresh grounding
  inside a 3° deadband before stopping. Final continuous-grounding residuals were
  0.262° and 0.162°. A separate evaluation-only oracle scorer measured physical
  heading errors of 6.579° and 3.302°, both below its 20° threshold. The preserved
  learned-only predecessor timed out after 12 steps (0/1), so this is honestly a
  **hybrid learned-plus-numeric** result—not evidence that V3 alone converged.
  It covers two deterministic development scenes and one instruction family.
  The 256-latent static base memory remains question-independent, but V75's four
  continuous control tokens and navigation grounding are question-conditioned;
  this embodied result therefore does not satisfy strict identical-total-input
  invariance.
- The separate two-scene approach comparison preserves the failed V2 result and
  its V3 successor. V2 passed 1/2: on `scene_000031` it moved 1.220 m but hit an
  exact collision rejection and terminated without stopping. V3 passed 2/2 with
  all action receipts successful, zero collisions, zero forbidden reads, and no
  runtime oracle or environmental-text input. Scene 1 completed the ordinary
  semantic-standoff goal at 0.482 m after 0.700 m of motion. Scene 31 moved
  1.287 m, made 1.272 m of evaluator-measured center progress, and finished
  0.292 m from the oracle bounding box, but its semantic target distance was
  still 0.763 m: it passed through an explicit numeric-map
  `collision_limited_safe_stop`, not the ordinary 0.5 m semantic-standoff rule.
  Oracle target identity and geometry were scorer-only. Both are deterministic
  development results, not held-out evidence or promoted runtimes. The exact
  historical V3 policy source is preserved at
  `reports/gemma4/evidence/navigation_policy_v3_sources/navigation_policy_v3.py`
  (`4e687161…`).
- The V3 approach paths are now preserved in the hash-authenticated
  [trajectory figure](reports/gemma4/figures/embodied_approach_v3_trajectories.png)
  (`6bbe03c6…`) and its
  [machine summary](reports/gemma4/examples/embodied_approach_v3_trajectories.json)
  (`2b1482c0…`). The deterministic post-hoc generator reads only the two pinned
  runtime result JSON files—no oracle, QA, scene metadata, semantic map, or model
  files—and runs no new inference. Scene 1 is ordinary semantic-standoff
  completion; scene 31 is visibly and explicitly a collision-limited
  closest-safe stop at 0.763 m, **not ordinary 0.5 m standoff success**. Rebuild
  and re-authenticate both artifacts with
  `make embodied-approach-v3-trajectories`.

### Global-map semantic-goal Blender rover

Blender is the preferred operator surface for the corrected embodied
demonstration. It shows the furnished room, an unmistakable toy rover, the rover
trajectory, and a sampled rendering of the persistent semantic point map in a
normal perspective 3D viewport. The `Gemma Rover` sidebar is a high-level goal
conversation, not a remote-control pad: it keeps a scrollable user/assistant/
agent transcript, reports thinking/execution state, and exposes the scene-token
shape, hash, and norm without filling the screen with research analytics.
Every motion row identifies the actual selected action, exact continuous output,
MOVE/FACE/STOP probabilities and raw logits, causal scene/robot/history/prompt
token counts, and abbreviated output/prefix/checkpoint hashes. The Blender
scrollback is sized to retain a complete worst-case 128-decision turn; extra
scene-memory diagnostics remain collapsed by default.

The operator path is:

```text
24 complete RGB-D room views
  -> one Gemma vision pass per complete image
  -> 48 x 48 x 3072 spatial patch fields
  -> exact depth/pose projection and 5 cm fusion
  -> 74,699 persistent semantic voxels
  -> every occupied spatial block -> 256 global scene latents
  -> fixed [1,258,1536] continuous Gemma scene prefix
  + four numeric robot-state tokens + numeric action-history tokens
  -> actual Gemma causal forward for every closed-loop decision
  -> learned MOVE_TO / FACE / STOP heads
  -> exact model waypoint/heading execution or safety rejection
```

The 3072D voxel payload retains 768D middle-layer, 768D late-layer, and
1536D language-aligned Gemma features in float16. The full scene prefix is built
before a goal, after every occupied block has contributed. There is no runtime
target-query retrieval, question-dependent scene crop, nearby-point selection,
object inventory, caption, or simulator label. A phrase such as `the chair`
enters only as part of the user's raw goal; it can influence Gemma's action
decision, but it cannot alter or select the already-fixed scene prefix.

User input should describe an outcome, for example `Face the chair`, `Move close
to the bowl`, or `Do a lap around the room`. Gemma selects every intermediate
waypoint, heading, route change, and goal-completing STOP. The host may reject an
out-of-bounds or colliding proposal, but it may not clamp it, reroute it, insert a
recovery waypoint, or turn failure into STOP. After a rejection, the next action
must come from another actual Gemma forward over the unchanged complete scene,
the refreshed numeric robot state, and the recorded failed-action row. Turns and
waypoint moves are internal receipts rather than commands the user must compose.

The V14 waypoint checkpoint binds
`history_parameterization: selected_action_parameters_goal_progress_v2`. Its
16D numeric history contains only the selected action's parameters: `MOVE_TO`
keeps its actual
robot-frame waypoint and uses current/result yaw as a neutral heading, `FACE`
keeps its requested heading and zeros the inactive waypoint, and `STOP` zeros
the inactive waypoint and uses current yaw. A rejected `MOVE_TO` still retains
the exact rejected waypoint. Four further values summarize only numeric action
receipts: accepted path length, signed swept area, return distance, and rejection
streak. They contain no target, label, oracle value, route, or stop rule. Raw
Gemma outputs remain unchanged in the signed step receipts; canonicalization
affects only the history supplied to the next Gemma forward. Older checkpoints
lacking this contract fail closed.

This mode does **not** use the rover's current camera view to choose actions.
`initial_scan` and `auto_scan_after_motion` are both disabled: the precomputed map
and its scene-prefix hash remain fixed while only the continuous robot-state tokens
refresh after motion. The Blender viewport and point-map overlay are human displays
and are never fed back to Gemma.

```bash
make rover-3d-check  # model-free: starts neither Gemma nor Blender
make rover-3d        # one command: local backend + Blender 3D viewport
make rover-gemma-mcp-check # authenticate the one-tool model-owned MCP surface
make rover-gemma-mcp       # stdio MCP: navigate(goal), with no motor tools exposed
```

If Blender is installed as the normal macOS application but is not on `PATH`, use:

```bash
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender make rover-3d-check
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender make rover-3d
```

The live command authenticates the sanitized scene, base semantic map, local
Gemma snapshot, continuous-memory checkpoints, numeric robot-state encoder, and
the configured two-file waypoint-policy checkpoint before starting anything. It then
starts the loopback-only backend at `127.0.0.1:8770`, builds the complete scene
prefix, and launches the sanitized `.blend` with the sidebar. The policy
checkpoint contains only the numeric history projector, decision token, and
action/waypoint/heading heads; frozen Gemma weights remain in the pinned local
model snapshot. Its runtime contract requires an actual Gemma causal forward,
all 258 scene tokens, four robot tokens, raw user text, numeric history, no
environmental text, and no runtime planner.

The earlier failure mode was primarily incorrect integration, not proof that
Gemma E2B was too small: the UI exposed low-level controls and routed text through
an untrained one-step JSON decoder, while several high-level goals were actually
completed by deterministic planners. The model-only launcher rejects that decoder,
disables direct UI tool calls, and requires high-level goals. The local Gemma
decoder owns the entire movement sequence; deterministic code is restricted to
exact coordinate conversion, validation, collision rejection, and primitive
execution.

The previous 47-waypoint patrol and 6/6 V3.3 measurements are historical hybrid
planner evidence and must not be attributed to the new model-only movement path.
The runtime-aligned V14 policy trains on 7,115 rows from `scene_000001` and
reaches 99.9719% action accuracy, 0.004970 m mean waypoint error, and 0.094552°
mean heading error there. Its cache contains 96 validation rows from two disjoint
scenes, but the reported configured 24-row disjoint control is only 12.5%
action-accurate, with 0.122765 m mean waypoint error, 29.5241° mean heading
error, and zero STOP recall. The passing live tests below therefore demonstrate
a runnable one-room controller, not broad unseen-room transfer.

The fresh model-only closed-loop V14 lap **passed** live acceptance. Gemma made
76 decisions (46 `MOVE_TO`, 29 `FACE`, and its own final `STOP`), traveled
18.715408 m, swept 4.272962 m², returned 0.048733 m from its start, and produced
zero rejected decisions. The same checkpoint passed the chair approach with
0.263674 m center progress and 0.431456 m bounding-box standoff despite eight
safely rejected collision attempts over 16 decisions; it faced the cube with
0.201562° yaw error in two decisions. Every accepted primitive used Gemma's
exact selected action and numeric argument. The runs used local inference and
the fixed scene-prefix hash
`52c33298140845d341fa2b4568f2c6e960279495890e08455caafa7d5bbc9c95`;
they used no cloud model, oracle runtime input, deterministic route planner,
fallback, action substitution, or synthetic STOP. The isolated runtime also
passed with the source oracle directory physically unavailable. Evidence:
[`lap`](reports/gemma4/metrics/gemma_waypoint_dagger_v14_live_acceptance.json),
[`chair`](reports/gemma4/metrics/gemma_waypoint_dagger_v14_approach_chair_score.json),
[`cube`](reports/gemma4/metrics/gemma_waypoint_dagger_v14_face_cube_score.json),
and [`oracle isolation`](reports/gemma4/metrics/gemma_waypoint_dagger_v14_oracle_isolation.json).

A separate visual acceptance run used the Blender sidebar itself and supplied
only `Move closer to the chair and stop.` Gemma emitted 17 causal decisions
(10 `MOVE_TO`, six `FACE`, and its own `STOP`); nine were executed and eight
collision-risk proposals were rejected without replacement. The toy rover moved
and replayed the returned trajectory in the actual furnished 3D viewport. See the
[`machine-readable UI result`](reports/gemma4/metrics/blender_rover_v14_live_acceptance.json)
and the authenticated [completed Blender view](reports/gemma4/figures/blender_rover_v14_approach_chair_complete.png).

Closing Blender also stops a backend created by this command. A compatible
backend that was already running is reused and left running. If port 8770 belongs
to another service, the launcher refuses to kill it and explains how to select
another port.

### Browser compatibility surface

The browser route remains available for local API/debug compatibility, but the
Blender viewport above is the preferred real-3D experience:

```bash
make rover-demo-check  # finite/read-only; loads neither Gemma nor Blender
make rover-demo        # opens http://127.0.0.1:8770
```

It uses the same high-level-only, static-map controller: no initial rover scan,
no motion-triggered camera input, no untrained JSON fallback, and no user-facing
manual motion tools. The displayed images remain human-only. The earlier
turn-right browser smoke belongs to the superseded low-level integration and is
not evidence for the corrected semantic-goal path. To expose the bounded numeric
execution boundary to an external local MCP client, run `make rover-demo-mcp` in
its own terminal; those tools are an internal agent protocol, not the intended
human language interface. For the corrected model-owned MCP contract, run
`make rover-gemma-mcp`: it exposes only `navigate(goal)`. The exact goal text is
passed to the same V14 closed loop, and the structured response includes numeric
state, each authenticated Gemma decision, accepted/rejected status, and prefix/
checkpoint hashes—but no object labels, caption, oracle data, deterministic route,
or client-supplied motor command. The authenticated model-free server check and
official-SDK dispatch result are recorded in
[`gemma_goal_mcp_preflight.json`](reports/gemma4/metrics/gemma_goal_mcp_preflight.json).

Current commands:

```bash
make doctor                    # works before setup; records available local tools
make setup                     # locked support env + separately pinned Gemma env
make download-models           # fetch the pinned 10.25 GB Gemma checkpoint once
make demo-artifacts-check-fast # inspect preserved V54/comparator artifact inventory
make prepare-demo-runtime      # materialize the preserved V54 comparator release
make demo                      # strict V89: interactive TTY, finite/exit-safe in CI
make demo-smoke                # finite promoted V89 scene-one three-question proof
make chat                      # interactive promoted strict V89 scene-one chat
make web                       # strict V54 point-map browser comparator
make robot                     # interactive learned embodied Gemma loop
make mcp                       # official local semantic embodied MCP server
make rover-demo-check          # finite room/scan/Gemma/rover/UI readiness gate
make rover-demo                # high-level browser compatibility/debug UI
make rover-demo-mcp            # optional numeric-only official MCP stdio server
make rover-gemma-mcp-check     # authenticate high-level-only V14 Gemma MCP
make rover-gemma-mcp           # one MCP navigate(goal) tool; Gemma owns all motion
make rover-3d-check            # model-free Blender + local backend readiness gate
make rover-3d                  # preferred real-3D Blender rover operator UI
make rover-live-verify         # score high-level goals against a fresh running backend
make demo-check                # model-free V89 + V3.3 + embodied-MCP readiness
make embodied-check            # authenticate V3.3 evidence and MCP session inputs
make demo-leakage              # V89 oracle-unavailable + exact-input invariance proof
make v89-demo-check            # authenticate V89 without loading Gemma
make v89-demo                  # finite V89 proof
make v89-demo-chat             # interactive V89 chat
make v89-demo-leakage          # authenticate isolated V89 runtime smoke
make strict-demo-chat        # legacy interactive strict V54 fixed-prefix chat
make strict-demo-leakage     # oracle removal + exact full-prefix invariance
make strict-web              # loopback browser UI over that same strict prefix
make strict-web-check        # validate UI inputs without loading Gemma
make v75-fixed-atlas-mechanism-check # rebuild the V75 738-token structural proof
make v75-fixed-atlas-behavior-preflight # authenticate bounded behavior inputs
make v81-reader-check      # model-free sealed reader/layout preflight
make v81-scene-memory-check # authenticate exact two-file 738-token scene memory
make v81-scene-memory-demo # finite experimental V81 chat
make v81-scene-memory-chat # interactive experimental V81 chat
make v81-scene-memory-leakage # oracle deletion + fixed-memory invariance
make v82-reader-preflight  # authenticate dense-reader shape/floors/zero control
make v82-reader-evaluate   # pair/scene-disjoint numeric development diagnostic
make v82-historical-score  # bounded real-Gemma 16-row control (gate failed)
make v82-chat              # experimental V82 chat; not the promoted default
make v83-check             # exact direct 738-token native-Gemma layout tests
make v83-chat              # experimental strict direct-memory chat; not promoted
make v83-historical-score  # read create-once 16-row negative behavior result
make v84-pair-margin-check  # authenticate passed strict two-scene causal wiring
make v84-pair-margin-result # print measured fixed-update result; no model load
make strict-atlas-build      # legacy runtime build; no promoted atlas checkpoint yet
make strict-atlas-evaluate   # behavior only after a separately authenticated build
make strict-atlas-v2-auth    # read-only V2 layout/exposure hash authentication
make v78-grounding-check     # authenticate optional V78 + unchanged V75 answer path
make v78-grounding-demo      # finite local V75 answer plus V78 numeric grounding
make v78-grounding-chat      # interactive optional-grounding mode
make v78-grounding-leakage   # oracle deletion + base/V78 scene-token invariance
make v78-grounding-held-pointcloud # exact 94-row replay + six held-map overlays
make v78-grounding-embodied-check # authenticate embodied checkpoint forwarding
make v78-grounding-embodied-once  # finite scan then one grounded answer + audit
make ple-reader-prereg-auth  # read-only rank-4 reader design authentication
PYTHONPATH=src .venv/bin/python -m semantic_3d_chat.evaluation.fixed_prefix_ple_v54_evidence
PYTHONPATH=src .venv/bin/python -m semantic_3d_chat.evaluation.gemma4_tool_decoder_v2_2_evidence
PYTHONPATH=src .venv/bin/python -m semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_evidence
make current-report          # rebuild claim-bounded JSON + Markdown
make gemma4-v70-authenticate # verify sealed 32-moment negative result
make gemma4-v71-authenticate # verify sealed 8+32-branch negative result
PYTHONPATH=src .venv-gemma4/bin/python -c 'from semantic_3d_chat.evaluation.v72_development_authentication import authenticate_v72_development_negative as a; import json; print(json.dumps(a(), indent=2, sort_keys=True))'
PYTHONPATH=src .venv-gemma4/bin/python -m semantic_3d_chat.evaluation.fixed_prefix_attention_reader_v6_3_evidence authenticate
PYTHONPATH=src .venv-gemma4/bin/python -m semantic_3d_chat.evaluation.fixed_prefix_attention_reader_v6_4_evidence authenticate
make research-demo-check     # exact schema-7 preflight; currently fails closed
make research-demo           # finite enhanced readout only after a sealed successor
make research-demo-chat      # interactive form; currently fails closed
make research-demo-leakage   # oracle/training removal + prefix invariance
make gemma4-semantic-navigation SCENE=scene_000001
make gemma4-embodied-chat-learned-check # read-only V3/controller/model preflight
make gemma4-embodied-chat-learned SCENE=scene_000001
make gemma4-embodied-chat-learned-once  # finite 12-step-capped instruction
make gemma4-embodied-chat-llm SCENE=scene_000001
make gemma4-embodied-mcp-check SCENE=scene_000001 # read-only access preflight
make gemma4-embodied-mcp-live-smoke # heavy SDK + Gemma + Blender scan/turn proof
make gemma4-embodied-mcp-conversation-check # finite semantic MCP policy preflight
make gemma4-embodied-mcp-conversation # natural instruction -> real stdio actions
make gemma4-embodied-mcp-conversation-score # separate oracle-only heading score
make embodied-demo-check SCENE=scene_000001 # model/renderer-free session preflight
make embodied-demo-smoke SCENE=scene_000001 # finite same-stdio five-command proof
make embodied-demo-smoke-inspect SCENE=scene_000001 # model-free saved-run auth
make embodied-demo-smoke-score SCENE=scene_000001 # separate oracle geometry score
make embodied-demo SCENE=scene_000001 # persistent conversational MCP session
make conversation-mcp-smoke # conversational agent -> official MCP stdio proof
make gemma4-embodied-mcp SCENE=scene_000001
make mcp-stdio-smoke          # finite official-SDK transport/safety/leakage proof
make navigation-policy-v3-demo # read-only exact V3 evidence authentication (5/6)
make navigation-policy-v3-3-check # historical V3.3 seal; rejects after runtime-source evolution
make navigation-policy-v4-1-result # authenticate rejected 13/14-gate V4.1 arm
make navigation-policy-v2-demo # read-only hash check + measured V2 result (4/6)
# Fresh V2 run; choose a new ID because the launcher refuses every overwrite:
make navigation-policy-benchmark NAVIGATION_POLICY_RUN_ID=learned_v2_local_001
```

Running `./scripts/run_full_demo.sh` with no stack-selection options is equivalent
to `make demo`. Its explicit `--config` form is retained for the legacy/configurable
launcher used by the `legacy-demo*` targets.

`make demo` first runs the machine doctor and `make demo-check`. That model-free
gate authenticates the exact two-file V89 release, runs the current strict
continuous-prefix embodied implementation tests, and preflights the official
numeric-only MCP 2.0 server plus persistent conversational-session inputs. The
historical V3.3 6/6 artifact remains preserved, but its source seal predates the
current runtime and is not presented as current navigation acceptance. A fresh
held-out navigation claim remains blocked until a successor static release is
promoted. The demo then prints the point-map,
point-cloud, browser-UI, and MCP commands. With
terminal stdin and stdout it enters strict V89 chat; under CI, pipes, or redirected
input it runs the finite three-question V89 smoke and exits. It does not
automatically open a GUI or start a second blocking server. `make demo-smoke`
always selects that finite form, while `make demo-leakage` authenticates the
create-once smoke executed with the oracle directory physically unavailable.
The preserved `make strict-web` comparator can materialize the exact two-file V54
checkpoint when its local source artifacts are present. `make
demo-artifacts-check-fast` verifies
exact hashes for all project artifacts and verifies the pinned model by local
revision, required filenames, and exact model-weight size without loading Gemma
or running Blender;
`make demo-artifacts-check` additionally hashes every artifact, including the
10.25 GB model weights. The expected sizes and SHA-256 digests are tracked in
`configs/runtime/demo_artifacts_v1.json`.

The repository currently has no public distribution URL for the ignored 57.4 MB
V89 adapter, its approximately 2.2 MB immutable scene memory, the 53 MB V54
source adapter, 8 MB V75 comparator, 410 MB sanitized scene map, or human-only
preview files. Therefore a clone on this already-prepared Mac is one-command
runnable, while a different fresh machine still needs those exact hash-bound
artifacts transferred locally or must reproduce the applicable earlier phases.
The readiness check reports every missing path and expected size/hash; it does
not invent a download source or claim fresh-clone portability. The pinned Gemma
weights are separately obtainable with `make download-gemma4-weights`.

The unqualified phase commands now select the Gemma research stack consistently:

```bash
make generate-smoke-scene
make render-smoke-scan
make build-smoke-map
make semantic-sanity
make generate-dataset
make train
make evaluate \
  GEMMA4_STATIC_CONFIG=/path/to/promoted-runtime.yaml \
  GEMMA4_STATIC_CHECKPOINT=/path/to/promoted-checkpoint \
  GEMMA4_STATIC_REFERENCES=/physically/separate/test.jsonl
```

The build and training commands are explicit development operations; `make demo`
does not depend on them or overwrite prepared releases. In particular, `make train`
trains the configured Gemma adapter experiment—it does not pretend to regenerate
the sealed historical V75 or V89 releases. `make evaluate` creates fresh questions-only
inference input and predictions before scoring, and fails before inference when the
explicit protected reference file or accepted static adapter is unavailable. The
corresponding preserved CLIP/Qwen commands use the `legacy-*` prefix documented
below.

V2 remains the default for the generic training/evaluation `navigation-policy-*`
Make targets; the conversational learned navigator selects V3 explicitly. V1
remains reproducible by explicitly overriding
the config, checkpoint, embodied config, task manifest, scoring sidecar, output
paths, and run ID. The V2 demo target reads only the sanitized checkpoint,
user-authored tasks, sealed inference journal, clean access audit, and numeric
score; it does not open oracle, QA, or navigation-training data and writes no
files.
The V3 demo target is also read-only: it authenticates the exact historical
shared-source bytes bound by the V3 journal, training-trace hashes, sanitized
two-file checkpoint, held-out controls, oracle-removal audit, separated score,
and trajectory artifacts. It explicitly does not claim that the sealed live run
used today's successor runtime source. `navigation-policy-v3-3-check` is the
corresponding current read-only successor check: it recomputes all gates for the
single preregistered V3.3 development run (6/6 tasks, zero collisions, zero action
failures, and zero policy rejections) without loading Gemma or Blender and without
opening oracle or QA files. This authenticates development evidence; it is not a
held-out navigation claim. `navigation-policy-v4-1-result` verifies
the V4/V4.1 preregistrations, mechanical serialization incident, terminal
13/14-gate rejection, absent checkpoint, and absence of any V4.1 live run.

The default `make demo` now uses promoted V89: Gemma receives the exact immutable
738-token `[1,738,1536]` continuous scene memory directly, with zero
question-derived environmental tokens and no query-conditioned scene readout.
V75 remains the scene-disjoint enhanced-readout comparator; its four continuous
control tokens are question-conditioned. The preserved `make strict-demo` /
`make strict-demo-chat` V54 path is a below-acceptance mechanism comparator
(41.20% exact development behavior with failed counterfactual/grounding gates).
V89 is a strong single-scene development proof, not held-scene final acceptance.

The strict browser path shows the fused RGB point-map raster and chat side by
side on `http://127.0.0.1:8766`. It binds loopback only, builds the complete
prefix before starting HTTP, returns the same environment-input hash on every
answer, and blocks oracle, QA, rendered frames, feature caches, training data,
and scorer-only files. The displayed raster is a human visualization and is
never supplied to Gemma.

The stronger strict mechanism is implemented as a fixed continuous key/value
atlas. The offline compiler has now been executed with the exact sealed V75
controller over 96 continuous probes before any user question. It appends all 96
keys plus four values per key to the complete 256 scene latents, producing a
738-token environment input. The hash-pinned run preserved every base latent,
probe, and atlas token and loaded no Gemma model, question, answer, oracle,
protected split, or environmental text. Thus, the mechanism no longer waits on
or uses rejected V66b. Behavioral accuracy was deliberately unmeasured at that
structural-check stage. The subsequent bounded Gemma diagnostic measured
6/16 for the fixed atlas, tied with V54 and below direct V75's 9/16, with only
1/8 prediction-changing units. All 16 prefixes remained invariant and the
predictor/scorer boundary stayed isolated, but 12 prompts overlapped historical
training. The compiler is therefore structurally sound while this behavior is a
negative, non-promoted result—not evidence for an atlas runtime.

An isolated, versioned Atlas V2 layout contract now orders that same complete
memory as `[BOI][all 480 atlas key/value tokens][all 256 base scene latents][EOI]`.
Nothing is compressed, selected, or retrieved, and compilation still occurs
before user text. This places all 256 base latents inside the direct 512-token
window of the final prompt token for inclusive prompt lengths 57--64; the V1
ordering places zero base latents inside that local window. This is an exact
sliding-attention exposure calculation, not behavioral evidence: V2 compilation
is disabled, no accepted sealed controller or V2 checkpoint exists, and no
accuracy improvement is claimed. The contract is in
`configs/experiments/gemma4_strict_fixed_prefix_atlas_v2.yaml`.

The original Atlas-dependent reader follow-up is a rank-4, unmerged FP32 LoRA on
Gemma 4's `model.language_model.per_layer_model_projection`: 41,984 trainable
parameters, answer-only cross entropy, and a same-question wrong-scene-prefix
hinge. It is deliberately design-only. Training, generation, and checkpoint
publication are unauthorized until an independently accepted Atlas V2 artifact
supplies concrete checkpoint, weights, runtime-metadata, and acceptance-report
digests. The contract also requires an empty-prefix control, wrong-scene-prefix
answer following, oracle removal, invariant pre-question prefix hashes, and
non-environmental text-retention checks. Its hash-pinned preregistration is
[`gemma4_fixed_prefix_ple_reader_preregistration_v1.json`](reports/gemma4/metrics/gemma4_fixed_prefix_ple_reader_preregistration_v1.json).
A separate V54 reader family was later executed through five independently
sealed versions and failed its causal scene-selectivity gates as summarized
above. It does not satisfy or promote the Atlas-dependent design. None of these
read-only authentication commands loads model weights or writes artifacts:

```bash
make strict-atlas-v2-auth
make ple-reader-prereg-auth
PYTHONPATH=src .venv/bin/python -m semantic_3d_chat.evaluation.fixed_prefix_ple_v54_evidence
PYTHONPATH=src .venv/bin/python -m semantic_3d_chat.evaluation.gemma4_tool_decoder_v2_2_evidence
PYTHONPATH=src .venv/bin/python -m semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_evidence
```

`scripts/run_research_demo.sh` accepts only the exact two-file sealed schema-7
checkpoint whose saved-runtime generation gate passed. It does not fall back to a
legacy model or an unsealed training artifact. Because V66b failed and no such
checkpoint exists, its
nonzero exit is deliberate. This launcher is an enhanced-readout experiment; a
separate strict launcher must require one complete environment-conditioned token
sequence computed before and reused unchanged across questions.

The machine-readable summary is
[`reports/metrics/current_metrics.json`](reports/metrics/current_metrics.json).
It is generated from an explicit allowlist that contains no oracle, QA,
scorer-only, or deferred-final-scene paths.

### Historical experiment diary

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
also failed at 5/12 mirror sides and 0/6 units; V19's reflection-odd global
moment reached 6/12 and 0/6. V20 was rejected before a live update because its
original BF16 eligibility statistic was confounded by quantization phase. V21's
corrected phase-aware signed-X local field retained color at 12/12 sides and 6/6
units and improved mirror to 8/12 and 2/6. V22's margin rebalance regressed to
7/12 and 1/6. V23 then froze the V21 scene stack and adapted Gemma's real shared
K/V path: its selected update 2 retained color at 12/12 and 6/6 and reached the
project's strongest mirror result so far, 10/12 sides and 4/6 complete units.
Updates 3--8 did not improve that peak, and no epoch reached the required 12/12,
6/6 mirror gate. V23 therefore closed at its preregistered limit without greedy
generation or promotion. At V23, static chat, leakage, and navigation remained
gated; the later V89 and embodied results supersede that historical status.

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
- The deployed V89 release freezes Gemma and all 11 authenticated LoRA banks at
  runtime (872,448 adapter parameters total). It inserts the V81-derived immutable
  `[1,738,1536]` memory directly into Gemma's native image-prefix slot: all 480
  atlas tokens and all 256 base latents, bounded by native BOI/EOI. There are zero
  question-derived environmental tokens, no question-conditioned readout, and no
  retrieval. The decoder receives continuous scene memory and the user's question—
  never an environmental caption, label list, or oracle metadata. Earlier V9--V23
  banks remain part of the authenticated historical lineage.

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

### Gemma 4 E2B primary path (V89 strict runtime promoted; held-scene acceptance open)

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

`make chat-gemma4` has no checkpoint fallback. It resolves only the immutable
`configs/runtime/primary.json` pointer for a conventional promoted static adapter;
that pointer remains absent because no such adapter passed every strict gate. The
operator-facing `make chat` and `make demo` commands instead select the separately
sealed V89 strict scene-one release described above. V89 passes its model and
oracle-unavailable runtime gates with an identical direct scene memory for every
question, but is trained single-scene development evidence, not held-scene final
acceptance. V75 remains available as the question-conditioned comparator.

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

#### Gemma V19 signed-X moment screen — completed, failed, and not promoted

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

The completed V19 screen did **not** authorize that branch. Its selected epoch 4
retained the color control at 12/12 full-vocabulary sides and 6/6 complete units
with minimum margin `+0.5`, but the mirror pair reached only 6/12 sides and 0/6
complete units with minimum margin `-1.0`. Candidate and full-vocabulary mirror
counts were identical and no pair flipped both predictions, so the remaining
failure is directional scene-to-answer binding rather than an unrelated-token
competition artifact. The strict result is
`screen_failed_no_extension_no_greedy_audit`; V19 has no promotion, chat,
held-out, leakage, or robot authorization. Its machine-readable decision is
[`v19_epoch_screen.json`](reports/gemma4/metrics/v19_epoch_screen.json).

#### Gemma V20 signed-X local-field screen — architecture-only staged test

V20 restarts from the exact frozen V18 epoch-4 checkpoint; it does not load,
stack, or continue V19's trained signed-X state. The losses, opaque pair IDs,
question ordering, optimizer, scene encoder, V18 residual, LoRA banks, and Gemma
decoder remain fixed. The single intended causal change is removal of V19's
global signed-moment bottleneck. For every one of the 256 slots, V20 multiplies
the centered local content by that slot's deterministic signed-X anchor, applies
the same shared bias-free `128 -> 1536` projection, and FP32-centers the resulting
delta over all slots. The Halton-derived slots are not exactly reflection-closed,
so this is a signed-coordinate local field, not a claim of exact equivariance.
All slots remain represented and the branch still receives no question, answer,
caption, object label, oracle coordinate, or retrieval query.

Only the zero-initialized 196,608-parameter projection is trainable. Its state
also contains a fixed V2 architecture marker, preventing a V19 state from being
mistaken for V20. The structural preflight evaluates both raw FP32 and effective
BF16 deltas, local dependence, spatial rank, mirror-versus-color selectivity,
frozen-state hashes, and the exact predicted first AdamW weight, module state,
and optimizer moments without taking a live optimizer step. A separate update-1
checkpoint must match those hashes exactly before updates 2--4 can resume.
The conservative
`maximum_per_scene_raw_and_effective_delta_to_core_rms_ratio: 0.01` policy
applies independently to both the pre-cast and decoder-visible residuals.

The four-update selector is report-only and accepts only exact cumulative
updates 1--4 plus the exact `v20_update1_match.json` authorization. Report-only
here means no Gemma load or inference: the selector deliberately reads each
adapter with safetensors and each one-matrix AdamW state with
`torch.load(weights_only=True, map_location="cpu")`. It validates the signed-X
and frozen tensor hashes, requires optimizer step `N` at update `N`, and binds
the actual adapter, metadata, and optimizer file hashes into every screen row.
The update-1 report hash, rich-preflight reduction, source/config provenance,
and exact epoch-1 artifacts are carried transitively into any extension launch
and final report. Color must remain 12/12 sides and 6/6 units with positive
candidate and full-vocabulary minimum margins. Continuation requires at least
8/12 mirror sides and 2/6 mirror units; complete teacher-forced acceptance
requires both pairs at 12/12 and 6/6 with every minimum margin positive. Greedy
generation is forbidden until that complete gate passes. A continuation-only
result may run through update 8 in an isolated namespace; a failed screen runs
no extension.

The immutable overlay is
[`gemma4_color_mirror_signed_x_local_field_v20.yaml`](configs/experiments/gemma4_color_mirror_signed_x_local_field_v20.yaml).
Individual guarded stages are `make gemma4-v20-preflight`,
`make gemma4-v20-stage1`, `make gemma4-v20-verify-update1`,
`make gemma4-v20-resume-screen`, and `make gemma4-v20-select`. Run the exact
four-update chain with `make gemma4-v20-screen`. Only after its selector emits
the continuation-only decision may `make gemma4-v20-extension` authorize and
run the isolated update-8 branch. Cached completed checkpoints are reused, but
fresh report-only evidence is required before a resume; incomplete namespaces
are never overwritten.

The real V20 preflight is now complete and **failed before any live optimizer
was constructed or stepped**. All zero-output identity, source/config binding,
local-dependence, all-slot coverage, spatial-rank, gradient-isolation, finite
BF16 visibility, and per-scene `0.01` magnitude checks passed. Raw FP32
mirror-versus-color normalized selectivity was `1.8667`, but the preregistered
total-norm BF16 statistic was only `0.5066`, below its `1.5` threshold. The
failure is preserved as `preflight_failed_no_optimizer_step`; V20 has no
checkpoint, continuation, greedy audit, promotion, chat, or robot authorization.

Post-failure diagnosis found no arithmetic mismatch. Each scene's exact
decoder-boundary rounding error was approximately `0.001` RMS. That noise floor
inflated the small color-pair differential from `0.000319` raw RMS to `0.001416`
effective RMS, while the larger mirror differential changed from `0.002109` to
`0.002539`. Thus subtracting two independently rounded responses confounds the
intended pair signal with base-dependent quantization phase. V20's immutable
decision is not being relaxed or reinterpreted. A full-model FP16 MPS
compatibility probe was attempted twice: one load crashed with exit 139, and
the other remained stuck at 0% weight materialization for over six minutes
before being stopped. It never constructed a model or ran inference, so FP16
was rejected as unreliable on this machine. A separately versioned V21
experiment retains the stable BF16 path and the legacy total-norm statistic as
a diagnostic, while gating eligibility on raw selectivity, per-scene precision
survival, signal-aligned evidence with explicit phase-null controls, and exact
predicted-update behavior through the frozen Gemma model.

#### Gemma V21 BF16 local-field screen — phase-aware estimator

V21 is a fresh restart from the exact V18 epoch-4 checkpoint and loads no V20
trained state (none exists). It keeps V20's 256-slot signed-X local field,
zero-initialized 196,608-parameter projection, optimizer, exact twelve-unit
order, pair losses, V18 scene bridge, frozen LoRA banks, and BF16 Gemma model
path. V21 makes no model-path or training-policy change relative to V20; it
changes only the preregistered eligibility estimator after V20 demonstrated
that subtracting independently rounded scene responses confounds signal with
base-token quantization phase.

The V21 preflight remains no-live-step and adds two controls. A phase-aware
audit decomposes the raw pair signal, exact decoder-visible pair response, and
orthogonal quantization component; it also evaluates shared-base and
common-delta nulls. Separately, the exact isolated first AdamW weight is run
through frozen Gemma before any checkpoint is authorized. The color pair must
remain a strict positive 12/12-side, 6/6-unit pass before and after that
predicted update, and the mirror pair's configured weighted margin-hinge
objective must strictly improve. The original total-norm pair statistic remains
reported, but is explicitly diagnostic-only and cannot authorize or veto V21.

Run the stages with `make gemma4-v21-preflight`, `make gemma4-v21-stage1`,
`make gemma4-v21-verify-update1`, `make gemma4-v21-resume-screen`, and
`make gemma4-v21-select`; `make gemma4-v21-screen` executes the guarded chain.
No real optimizer step can run unless the BF16 structural, phase, functional,
source-provenance, and zero-output-identity checks all pass.

The completed V21 chain was produced from clean source commit `806309b`. Its
preflight and exact update-1 verifier passed, and the update-4 selector
authorized only the preregistered isolated continuation through update 8. The
color retention control stayed at 12/12 full-vocabulary sides and 6/6 units at
every update. Mirror performance improved from 7/12 sides and 1/6 units at
update 1 to 8/12 sides and 2/6 units at update 3, then remained at 8/12 and 2/6
through update 8. The mirror mean full-vocabulary margin increased from
`0.140625` to `0.552083`, but its minimum margin remained negative (`-0.75` at
update 8). The base and extension trainer processes recorded `177.01 s` and
`237.24 s`, respectively.

The final strict decision is
`conditional_limit_reached_no_greedy_audit`: the full teacher-forced gate did
not pass, so no greedy audit, promotion record, static chat authorization, or
embodied-phase authorization was created. The immutable evidence seal is
`reports/gemma4/metrics/v21_final_summary.json`. It binds the original source,
config, reports, exact selected update-8 checkpoint hashes, trajectory, timing,
and both superseded setup attempts. Validate the complete local archive with:

```bash
PYTHONPATH=src python -m semantic_3d_chat.evaluation.v21_archive_validator
```

The archive validator intentionally does not compare the current Git HEAD with
the historical V21 source commit; it validates the recorded clean provenance
and byte hashes instead, so later versioned experiments cannot invalidate the
sealed result. Use `--summary-only` in a checkout that does not contain the
generated evidence and checkpoint files.

#### Gemma V22 local-field screen — margin-rebalanced controlled restart

V21's optimizer did not stall: its projection changed on every update, its
weight RMS grew smoothly, its training loss fell, and its mirror mean margin
continued rising after the discrete 8/12-side plateau. The remaining shared
gradient nevertheless kept optimizing every mirror side toward margin `1.0`,
including already-correct sides, while the acceptance rule only requires every
side to be strictly positive. V22 tests that objective-allocation hypothesis
before adding more trainable capacity.

V22 is a clean restart from the same exact V18 epoch-4 checkpoint with fresh
signed-X state and optimizer history. It is identical to V21 in model, BF16
path, 256-token full-scene prefix, architecture, source weights, opaque unit
order, loss weights, learning rate, four-update screen, conditional update-8
cap, and strict behavioral gates. Its only gradient-defining change is that
`pair_000003` uses candidate and full-vocabulary target margins of `0.25`
instead of `1.0`. This does not relax selection: promotion still requires both
pairs at 12/12 full-vocabulary sides and 6/6 complete units, with every minimum
candidate and full-vocabulary margin strictly positive. No question-dependent
scene processing, retrieval, or runtime oracle access is permitted.

The immutable overlay is
[`gemma4_color_mirror_signed_x_local_field_margin_rebalanced_v22.yaml`](configs/experiments/gemma4_color_mirror_signed_x_local_field_margin_rebalanced_v22.yaml).
Run `make gemma4-v22-screen`; run `make gemma4-v22-extension` only if the
four-update selector emits its continuation-only decision. V21 and V22 use
different config hashes, controller identities, report paths, checkpoint
namespaces, nested policy hashes, and extension manifests, so their evidence
cannot authorize one another. Greedy evaluation remains blocked until V22's
complete teacher-forced gate passes.

The completed V22 run was produced from clean source commit `cffeb36`.
Preflight and exact update-1 verification passed, and color remained at 12/12
full-vocabulary sides and 6/6 complete units throughout the four-update screen.
Mirror remained at 7/12 sides and 1/6 complete units at every update, below the
preregistered continuation floor of 8/12 and 2/6. The selector chose update 3
on margin ranking (`0.322917` mean and `-0.625` minimum full-vocabulary mirror
margin) and emitted `screen_failed_no_extension_no_greedy_audit`. Training took
`176.03 s`. No update-8 extension, greedy audit, promotion, static chat, or
embodied-phase authorization was created.

The source-HEAD-independent seal is
`reports/gemma4/metrics/v22_final_summary.json`. Validate its exact preflight,
update-1, four checkpoint epochs, selector decision, and denial state with:

```bash
PYTHONPATH=src python -m semantic_3d_chat.evaluation.v22_archive_validator
```

Use `--summary-only` when generated evidence and checkpoints are unavailable.

### V23 shared-K/V falsifier — completed, partial improvement, gate failed

V23 starts from the sealed V21 epoch-8 weights rather than the weaker V22
checkpoint. It loads the 256-token scene adapter, global residual, signed-X
local field, and both existing decoder banks exactly, freezes all of them, and
adds one deterministic zero-output rank-4/alpha-8 LoRA bank to Gemma 4's real
shared K/V projections at physical layers 13 and 14. This is 30,720 trainable
FP32 parameters; no question-dependent retrieval or oracle runtime input is
introduced. All learning-rate fields are pinned to `3e-4`.

The staged run completed from clean training source commit `2a8cd07`; its
extension controller was clean commit `39c12a1`. Update 1 matched the predicted
state exactly. The four-update screen selected update 2 and authorized only the
predeclared bounded continuation. Before any new update, epochs 3--4 were replayed
from update 2 in an isolated checkout: adapter tensors, decoded AdamW state,
history, metrics, and normalized metadata matched exactly. Raw `torch.save` ZIP
bytes differed because the temporary archive root name differed; both raw hashes
are recorded, while replay identity correctly uses the decoded tensor state.

Color stayed at 12/12 full-vocabulary sides and 6/6 complete units through all
eight updates. Mirror followed this exact side/unit trajectory:

| Update | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mirror sides | 8/12 | **10/12** | 9/12 | 9/12 | 9/12 | 9/12 | 8/12 | 8/12 |
| Mirror units | 2/6 | **4/6** | 3/6 | 3/6 | 3/6 | 3/6 | 2/6 | 2/6 |

Update 2 is therefore the strict selected checkpoint. Its mirror mean
full-vocabulary margin is `0.5729167`, but its minimum is still `-1.0`; it did
not pass the complete gate. The final selector emitted
`conditional_limit_reached_no_greedy_audit`. No greedy audit, promotion, static
chat, Gemma leakage/oracle-deletion inference, or semantic embodied-agent result
was authorized. The primary plus novel training segments took `377.0263 s` total
(the exact replay is an additional control and is not included in that total).

The source-HEAD-independent seal is
`reports/gemma4/metrics/v23_final_summary.json`. Validate the tracked seal alone,
or all eleven reports, eight checkpoint epochs, the exact-replay chain, preserved
superseded raw-container attempt, and denial state, with:

```bash
PYTHONPATH=src python -m semantic_3d_chat.evaluation.v23_archive_validator --summary-only
PYTHONPATH=src python -m semantic_3d_chat.evaluation.v23_archive_validator
```

### V28 learned post-stack dense sidecar

V26 established that a calibrated residual over every occupied voxel localizes
held-out bowl and cabinet regions, while V27 showed that any tested fixed additive
sidecar scale damaged V24's mirrored-room decoder control. V28 keeps the complete
V24 scene stack unchanged, pools the frozen V26 all-voxel signal into the same 256
spatial slots, and applies a learned adapter only after V24's global and signed-X
residuals. Its output projection and bounded full-dimensional channel gain both
start at exact zero; no question, label, caption, oracle relation, or text prototype
enters this numerical route.

The hash-pinned update-zero candidate is
`data_gemma4/checkpoints/gemma4_v28_post_stack_sidecar/candidate_zero`. The measured
update-zero screen processed every coarse voxel in four control scenes, reproduced
V24 scene-token and prefix tensors bit-for-bit, reproduced teacher-forced details
exactly, and retained 12/12 color plus 10/12 mirror full-vocabulary sides. Its
machine-readable audit is
`reports/gemma4/metrics/v28_post_stack_update_zero_screen.json`.

Run the bounded workflow with:

```bash
make gemma4-v28-build-candidate
make gemma4-v28-update-zero
make gemma4-v28-train-stage-a
make gemma4-v28-select-stage-a
make gemma4-v28-evaluate
```

Stage A freezes Gemma, V24, the calibrated V26 bridge, inherited LoRA banks,
composer, grounding head, and the adapter's input projections. It trains only the
zero-initialized output projection and channel gain for at most four optimizer
steps using broad answer-token NLL, one question per batch, and 12-way gradient
accumulation. It caches one question-independent base/sidecar scene tensor pair per
scene before consuming questions. Selection loads Gemma once, evaluates update 0
and all four trained updates, and requires improved validation NLL while retaining
12/12 color sides, at least 10/12 mirror sides, no new negative control sides, and
bounded prefix drift. Evaluation reads `selected_checkpoint` from
`reports/gemma4/metrics/v28_stage_a_selection.json`; it never silently substitutes
the trainer's NLL-only `best` alias. The default evaluation is the validation split;
override `GEMMA4_V28_EVAL_SPLIT` and `GEMMA4_V28_EVAL_CHECKPOINT` explicitly for a
different bounded measurement. These research targets do not create a promotion
record or authorize chat.

An optional selector-gated Stage B can adapt a deliberately tiny decoder surface
after Stage A passes. It adds one fresh deterministic zero-output rank-4/alpha-8
LoRA bank to only Gemma language-layer 13 and 14 query projections (36,864
parameters). The approved sidecar, complete scene stack, Gemma base, and all four
inherited LoRA banks remain frozen and hash-checked. Each scene is encoded once
before questions, and the same full-scene tokens drive broad answer-token NLL for
four bounded updates. Update zero must reproduce the selected Stage-A validation
NLL within `1e-7`. The second selector rejects any update that fails 12 color
sides, 10 mirror sides, introduces a new negative causal side, or does not strictly
improve validation NLL:

```bash
make gemma4-v28-train-stage-b
make gemma4-v28-select-stage-b
make gemma4-v28-evaluate-stage-b
```

The Stage-B config is
`configs/experiments/gemma4_color_mirror_post_stack_decoder_stage_b_v28.yaml`;
selection is written to `reports/gemma4/metrics/v28_stage_b_selection.json`.
Like Stage A, this optional research branch never creates a promotion record or
authorizes chat by itself.

V29 reuses that exact Stage-B architecture for the first scene-disjoint
`diverse20` development run. Its config is
`configs/experiments/gemma4_diverse20_post_stack_decoder_stage_b_v29.yaml`.
The trainer is locked to `data_diverse20/qa`: scenes 11-18 are training, scenes
19-24 are validation, and deferred scenes 25-30 must be absent from both the
split manifest and QA records. The lock verifies the split fingerprint, balanced
question counts (192 train and 216 validation), exact scene coverage, an empty
`test.jsonl`, and the absence of deferred scene IDs before Gemma is loaded. Four
48-question accumulation windows cover one deterministic pass over the 192
training questions. Because Stage A's scalar validation NLL came from a different
dataset, V29 does not compare those incomparable scalars; it instead verifies the
fresh bank's zero B tensors and bit-exact per-target outputs, then records the new
validation NLL as update zero.

Model selection still runs the pre-V29 V24/V28 color and mirror controls from a
physically separate config and requires 12/12 color sides, at least 10/12 mirror
sides, no new negative side, and lower V29 validation NLL. There is deliberately
no final-test target yet:

```bash
make gemma4-v29-train-diverse-stage-b
make gemma4-v29-select-diverse-stage-b
make gemma4-v29-evaluate-diverse-validation
```

V29's bounded decoder-only adaptation improved validation NLL but did not fix
the causal failure: free-generation exact accuracy remained `0.375`, spatial
extraction remained `0.5625`, all 12 answer-changing validation units remained
wrong on at least one side, and only 7 of 216 outputs changed from update zero.
V30 is therefore a separate development experiment, not a mutation or promotion
of V29. Its config is
`configs/experiments/gemma4_diverse20_joint_pair_v30.yaml`.

V30 pins the selector-approved V29 update by selector, adapter, and runtime
metadata hashes. It freezes every inherited scene/decoder tensor except the
post-stack sidecar's `output_projection.weight` and `channel_gain` (198,144
parameters), and adds a disjoint exact-zero rank-8/alpha-16 query LoRA bank at
Gemma language layers 18-21 (131,072 parameters). The complete bounded surface
is 329,216 parameters. The fresh decoder bank is proven bit-exact at update zero;
all 14 development scene prefixes and the 216-question validation NLL must also
exactly reproduce V29 before an optimizer step.

The cache boundary is deliberately before the trainable sidecar: every numeric
map is processed in full once, producing a frozen base tensor and an all-voxel
aligned tensor before any question is read. Each training cycle presents every
changed-answer unit atomically three times. Both scene prefixes are scored
against their correct and swapped canonical answers with a configurable NLL
margin, alongside a deterministic answer-type-balanced broad-NLL subset. No QA,
oracle, label, caption, or retrieval result enters chat runtime, and scenes 25-30
remain absent.

Selection retains the old 12/12 color and 10/12 mirror gates, forbids new
negative sides, requires lower diverse validation NLL, requires paired-margin
improvement, and runs development-only greedy checks on all 24 changed rows plus
a deterministic broad retention subset. One newly correct complete changed unit
is recorded only as development progress. Chat promotion requires at least 6 of
12 complete changed validation units and no aggregate exact-accuracy regression;
missing that bar means another repair iteration, not promotion.

```bash
make gemma4-v30-train-joint-pair
make gemma4-v30-select-joint-pair
```

There is intentionally no V30 final-test target. Deferred scenes 25-30 are not
generated or opened until the architecture is locked by the development gates.

### Fail-closed Gemma static evaluation and chat

Production Gemma chat uses the isolated Transformers 5 environment and the
standalone [runtime config](configs/runtime/gemma4_primary.yaml). That file has no
`_base_` inheritance and contains only numeric architecture/model/runtime fields;
it does not contain QA paths, oracle paths, experiment controls, answer categories,
scene variants, or object-label vocabulary. The CLI and web server reject files
below `configs/experiments` before opening them, refuse an implicit `best`
checkpoint, and require a hash-valid promotion for every direct runtime pair.

Promotion is a deliberate offline scientific acceptance action, never a trainer
side effect. Schema-2 `promotion.json` is written beside one exact checkpoint only
after all of the following independently supplied artifacts validate:

- the selector passed, explicitly set `chat_promotion_eligible: true`, selected
  that exact checkpoint/update, retained every required control, and did not touch
  final-test scenes;
- a complete, scene-disjoint held-out test report is above its declared chance
  control, covers every reference with no missing or extra prediction, evaluates
  counterfactuals and grounding, and binds the runtime config, adapter, runtime
  metadata, metric file, and prediction file by SHA-256;
- the executable leakage report passed with the oracle directory actually renamed
  and unavailable, no forbidden file access, at least three questions, and one
  prefix hash computed before and unchanged across all questions.

The interactive process never opens those evidence reports. It reads only the
standalone runtime config, numeric map, model snapshot, adapter,
`runtime_metadata.json`, and the compact hash/numeric promotion attestation. A
single `configs/runtime/primary.json` pointer additionally binds the exact runtime
config, checkpoint directory, and promotion-file hash. Tampering with any bound
runtime input or the pointer fails before model loading.

There is intentionally no conventional schema-2 promotion or
`configs/runtime/primary.json` pointer today. That promotion path therefore fails
closed until real selector, held-out-final, and leakage artifacts exist. This does
not negate the separately packaged V89 strict scene-one operator release. The
held-out wrapper is itself created from the real primary and
empty-scene-prefix metric files, prediction JSONL files, their immutable
provenance sidecars, and a nonempty scene-disjoint test split. It re-hashes and
cross-checks every file, verifies complete prediction coverage, and requires
counterfactual and grounding cases plus primary exact accuracy strictly above the
empty-prefix control:

```bash
make gemma4-create-final-evidence \
  GEMMA4_PROMOTION_CHECKPOINT=/path/to/selected-checkpoint \
  GEMMA4_FINAL_METRICS=/path/to/test-metrics.json \
  GEMMA4_FINAL_PREDICTIONS=/path/to/test-predictions.jsonl \
  GEMMA4_FINAL_PREDICTION_PROVENANCE=/path/to/test-predictions.jsonl.provenance.json \
  GEMMA4_FINAL_CHANCE_METRICS=/path/to/empty-prefix-metrics.json \
  GEMMA4_FINAL_CHANCE_PREDICTIONS=/path/to/empty-prefix.jsonl \
  GEMMA4_FINAL_CHANCE_PROVENANCE=/path/to/empty-prefix.jsonl.provenance.json \
  GEMMA4_FINAL_SPLIT_MANIFEST=/path/to/frozen-splits.json
```

Once that report and the independent leakage report pass, the one-time operator
promotion command is:

```bash
make gemma4-create-chat-promotion \
  GEMMA4_PROMOTION_CHECKPOINT=/path/to/selected-checkpoint \
  GEMMA4_PROMOTION_SELECTOR_REPORT=/path/to/selector.json \
  GEMMA4_PROMOTION_FINAL_EVIDENCE=/path/to/held-out-final-evidence.json \
  GEMMA4_PROMOTION_LEAKAGE_REPORT=/path/to/leakage.json
```

The creator refuses to overwrite either an existing checkpoint promotion or
primary pointer. Validate and launch the installed pair with:

```bash
make gemma4-validate-chat-promotion
make chat-gemma4 SCENE=scene_000005
make web-gemma4 SCENE=scene_000005
make demo-gemma4 SCENE=scene_000005
```

Until the conventional pointer exists, those four `*-gemma4` commands stop without
inference. The unqualified `make chat` and `make demo` use the separately
authenticated V89 release. An explicit conventional direct launch is also allowed
only for an already promoted pair:

```bash
PYTHONPATH=src .venv-gemma4/bin/python -m semantic_3d_chat.chat.cli \
  --config configs/runtime/gemma4_primary.yaml \
  --checkpoint /path/to/selected-checkpoint \
  --scene scene_000005
```

Development/static scoring targets retain explicit `GEMMA4_STATIC_CONFIG`,
`GEMMA4_STATIC_CHECKPOINT`, and `GEMMA4_STATIC_REFERENCES` inputs and do not confer
promotion by themselves. The reference path is deliberately never inferred: if
the physically separate answer-bearing JSONL is unavailable, `make evaluate`
stops before inference rather than scoring stale predictions or silently reading
another split.

`make evaluate` (or its explicit `make gemma4-evaluate-static` equivalent) runs
those prediction and scoring stages in order once all three inputs are supplied.
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
  GEMMA4_STATIC_REFERENCES=/physically/separate/test.jsonl \
  GEMMA4_EVAL_SPLIT=test
```

Historical v1-v11 checkpoints remain inspectable failure or diagnostic artifacts,
but none is a default for these targets and none may be silently treated as
promoted.

## Preserved legacy CLIP/Qwen entry points

```bash
make legacy-setup
make doctor
make legacy-download-models
make legacy-render-smoke-scan
make legacy-build-smoke-map
make legacy-semantic-sanity
make legacy-generate-dataset
make legacy-train
make legacy-evaluate
make legacy-chat
make legacy-web
make legacy-robot
make legacy-robot-evaluate
make legacy-mcp
make legacy-demo-check
make legacy-demo
```

These namespaced commands retain the earlier CLIP/Qwen infrastructure; they are not
the current Gemma primary experiment. `make legacy-demo-check` is a finite offline
preflight for prepared artifacts. It checks
the sanitized 24-frame RGB-D manifest, numeric high-dimensional voxel-map headers,
local model snapshots, visualizations, and adapter/config compatibility without
loading model tensors, creating an MPS tensor, starting chat/MCP, or reading the
oracle/QA/feature directories. It also rejects semantic keys and non-opaque frame
filenames in the sanitized manifest. Its machine-readable result is
`reports/metrics/demo_check.json`. For legacy checkpoints this is a compatibility
check, not a behavioral-success gate. For Gemma invocations, the demo wrapper also
requires the explicit accepted promotion record described above.

There is currently no conventional static-adapter entry in
`configs/runtime/primary.json`, so `make demo-gemma4` remains wired to that
promotion contract and fails closed. This does not describe the separate sealed
V89 scene-one release: the default `make demo` authenticates that release and
starts strict direct-memory chat over the exact immutable 738-token scene memory.
V89 is the promoted static-chat runtime, with its trained single-scene/no-held-out
scope explicitly disclosed. V75 remains the question-conditioned, scene-disjoint
enhanced-readout comparator. The preserved
CLIP/Qwen workflow is namespaced under `legacy-*`; it may resolve an
architecture-compatible legacy checkpoint, but that does not promote the legacy
model or establish scene understanding. A historical legacy invocation is:

```bash
make legacy-demo \
  CONFIG=configs/experiments/multiscene_anticollapse.yaml \
  CHECKPOINT=data/checkpoints/multiscene_anticollapse/best
```

For the preserved legacy finite inference/leakage demonstration (including
temporarily making the oracle unavailable and checking prefix invariance), run
`make legacy-demo-leakage` with an explicit compatible legacy checkpoint. That
result does not transfer to Gemma. The unqualified `make demo-leakage` exercises
the current promoted V89 oracle-unavailable smoke and exact-input invariance proof.
Model downloads are an explicit setup step; normal map building uses the pinned
local vision snapshot in offline mode.

`make report` is artifact-only: it never loads model weights or runs inference. It
rebuilds the claim-bounded current Gemma metrics snapshot and Markdown report from
an explicit source allowlist. The older CLIP/Qwen report generator remains
available as `make legacy-report`.

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

The diverse20 direct-image control can reuse the already-cached
`google/gemma-4-E2B-it` checkpoint in the isolated Transformers 5 environment—no
second VLM download is needed. It is explicitly a **prohibited primary
substitute** because direct RGB chat bypasses depth projection, persistent 3D
fusion, and question-independent scene tokens. The first command limits only the
number of validation questions; every selected question still receives all 24
complete frames in deterministic manifest order. The config rejects download-
enabled loading and supplies no depth, segmentation, oracle facts, or captions:

```bash
make evaluate-direct-images-gemma4-smoke       # 2 validation questions by default
make evaluate-direct-images-gemma4-validation  # all 216 development-validation questions
```

The validation command uses the versioned
`gemma4_decoder_kv_scene_prefix_v1` optimization. For each scene, Gemma processes
all 24 complete frames plus the fixed instruction once, stopping immediately
after the fixed `Question:` anchor. That continuous causal KV state is built
without a question and privately cloned for each of the scene's 36 questions.
This is exactly the same causal factorization as the uncached prompt, but avoids
repeating a roughly 6,200-token multimodal prefill 36 times. Prediction rows
record the cache contract and a scene-level prefix identity hash; JSONL
checkpoint/resume remains per question. Pass `--no-scene-cache` directly to the
module only for uncached parity debugging.

Override `GEMMA4_DIRECT_BASELINE_SMOKE_LIMIT` for a larger resumable smoke slice.
Scenes 25–30 remain deferred and neither command references their test split.

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

The `legacy-web` command is CLIP/Qwen infrastructure. The unqualified `make web`
now launches the strict Gemma V54 point-map comparator instead. After a compatible
legacy adapter checkpoint and fused map are available, start it with:

```bash
make legacy-web SCENE=scene_000001 CONFIG=configs/default.yaml
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

## Historical embodied-camera precursor and MCP

> **Historical architecture:** this section preserves the earlier camera-refresh
> and low-level MCP experiments for reproducibility. It does not describe the
> current `make rover-3d` operator. The Blender semantic-goal rover starts from a
> fixed pre-scanned map, sets both initial and post-motion scanning to false, and
> does not use the rover camera as a control input.

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
make legacy-robot SCENE=scene_000001
{"tool":"turn","arguments":{"angle_degrees":30}}
{"tool":"move_forward","arguments":{"distance_meters":0.2}}
{"tool":"scan","arguments":{}}
```

Print constrained schemas with
`semantic-3d-robot --config configs/default.yaml --schemas`. Every result contains
only protocol status, opaque IDs, numerical pose/velocity, collision state,
coverage, and scene version. It never returns an object name, label, caption, or
semantic relationship.

The embodied conversation CLI also has an opt-in local-Gemma action selector:

```bash
make gemma4-embodied-chat-llm SCENE=scene_000001
```

For action turns, Gemma receives the cached continuous scene plus numeric
robot-state tokens and must emit exactly one bounded JSON tool call. Duplicate
keys, trailing prose, non-finite values, stale prefix attestations, and limit
violations fail before execution; retries are capped at two. This seam is
currently marked `untrained_tool_selection_seam`, so it demonstrates safe local
model integration rather than claimed navigation competence. The default policy
fails closed; deterministic parsing is available only through an explicit CLI
fallback option.

The historical task-trained conversational MCP target is the V3 continuous
semantic-grounded controller; the current high-level Blender rover instead uses
the model-only waypoint DAgger V14 policy documented above:

```bash
make gemma4-embodied-chat-learned-check
make gemma4-embodied-chat-learned SCENE=scene_000001
# Or run one finite instruction, capped at 12 refreshed policy/action steps:
make gemma4-embodied-chat-learned-once \
  GEMMA4_LEARNED_NAVIGATION_COMMAND="Move closer to the chair, then stop."
```

The historical refresh experiments inherited the hash-sealed `embodied_v54.yaml`
contract and enabled one automatic complete RGB-D scan after accepted motion.
The current `configs/runtime/embodied_live.yaml` instead configures the static-map
Blender rover and disables both initial and post-motion scans. Archived reports
retain their original configuration hashes for evidence authentication.

V3 receives the complete cached continuous scene prefix, continuous numeric
robot-state tokens, and the user's literal instruction. For target-bearing
navigation instructions it embeds only the user-supplied target phrase, scores
every voxel in the active semantic map, and supplies the learned controller with
the resulting numeric target state. It does not load an object inventory,
caption, scene graph, segmentation labels, QA data, or oracle metadata. Each
step emits the policy attestation, map/query hashes, scored-voxel count, numeric
tool receipt, and refreshed-prefix binding. The same instruction is re-evaluated
after every successful bounded action until the controller selects `stop`, an
action or policy fails, or the configurable 12-step cap is reached. The older
`gemma4-embodied-chat-llm` target remains available and remains explicitly marked
`untrained_tool_selection_seam`; it is not the current learned-navigation path.

The official Python MCP SDK is pinned at `mcp[cli]==2.0.0`. Start its local stdio
server with `make mcp`; use `--transport streamable-http` only when an HTTP client
is needed. The nine exposed tools are `get_robot_state`, `look`, `turn`,
`move_forward`, `move_backward`, `move_to`, `scan`, `stop`, and `reset_scene`.

The finite semantic process-boundary demonstration is:

```bash
make gemma4-embodied-mcp-conversation-check SCENE=scene_000001
make gemma4-embodied-mcp-conversation SCENE=scene_000001 \
  GEMMA4_LEARNED_NAVIGATION_COMMAND="Face the chair, then stop."
make gemma4-embodied-mcp-conversation-score SCENE=scene_000001
```

The first command loads no Gemma tensors and starts no renderer or transport. The
second starts the production semantic MCP subprocess and uses a separate client
process for natural-language target parsing, selective local-Gemma text embedding,
fresh all-voxel grounding, and V3 numeric alignment decisions. All robot actions
cross stdio; the subprocess owns complete-image RGB-D encoding, semantic-map fusion,
full scene-prefix reconstruction, and numeric robot-state token refresh. The third
command is evaluation-only and refuses to open its oracle until the runtime result
and both process-lifetime access audits have passed authentication. The emitted
policy record deliberately says `learned_v3_action_head_used: false` and
`gemma_native_function_calling_used: false`; hashes received over MCP are not
misrepresented as the unavailable prefix tensors required by the learned action
head.

For the useful repeated-command form, run:

```bash
make embodied-demo-check SCENE=scene_000001
make embodied-demo-smoke SCENE=scene_000001
make embodied-demo-smoke-inspect SCENE=scene_000001
make embodied-demo-smoke-score SCENE=scene_000001
make embodied-demo SCENE=scene_000001
```

The historical smoke command is a deterministic finite heavy proof that sends
face-chair, approach-chair, scan, state, and standalone stop through the same
stdio session;
its result and two process-lifetime audit files are separate from the interactive
demo's artifacts. It automatically runs the strict post-run inspector. The
`embodied-demo-smoke-inspect` command repeats only that model-free authentication
against an existing report. It verifies the exact command order, every passed
turn, positive translation, every-voxel grounding, receipt-to-receipt map/prefix
refresh, strict numeric-only receipts, two hash-bound zero-forbidden access audits,
and the final safety-latched stop. It deliberately opens no semantic map, Gemma
weights, Blender asset, QA/training file, or oracle geometry; target-distance
scoring remains a separate later oracle-only step. `embodied-demo` keeps one
official-MCP stdio subprocess, robot episode, persistent
semantic map, and continuous scene-prefix binding alive. The prompt accepts
`Face the <target>`, `Look at the <target>`, `Turn toward the <target>`, `Approach
the <target>`, `Move closer to the <target>`, `Walk toward the <target>`, `scan`,
`get robot state`, and `stop`; the optional suffix `then stop` is accepted on
target goals. There is no built-in object vocabulary: the target phrase comes
only from the user's command, is embedded with selectively loaded local-Gemma
token rows, and is scored against every active-map voxel. Only the resulting
numeric turn/movement arguments cross MCP. In these archived auto-scan experiments,
successful motion captured a new complete RGB-D image, fused it into the map, and
rebuilt the scene prefix. The current Blender operator intentionally does none of
those camera updates: its fixed scene prefix remains byte-identical and only
numeric robot tokens refresh.

Reaching a face or approach goal verifies zero numeric linear/angular velocity
and acknowledges the state through MCP, but deliberately does not consume the
episode-wide stop latch; this permits a face command followed by an approach,
scan, or another target. A standalone `stop` invokes the real MCP safety latch and
rejects later motion. Exiting or EOF also latches stop before closing stdio. This
entry point reuses the tested V3 **numeric alignment/approach interlocks**, not the
learned V3 action head, and does not claim Gemma function calling. The local Gemma
rows provide continuous target grounding; action choice is deterministic numeric
geometry plus the anonymous collision field.

A non-interactive same-session sequence is available for reproducible diagnosis:

```bash
PYTHONPATH=src .venv-gemma4/bin/python \
  -m semantic_3d_chat.robot.conversational_mcp_agent \
  --config configs/runtime/embodied_live.yaml \
  --scene scene_000001 \
  --runtime-asset data/runtime_assets/scene_000001/s_000001.blend \
  --command "Face the chair, then stop." \
  --command "Move closer to the chair, then stop." \
  --command scan \
  --command "get robot state" \
  --command stop
```

Both client and server lifetime file audits are authenticated in the saved
session report. MCP receipts are strict numeric/protocol records and contain no
target phrase, category label, caption, object ID, QA data, or oracle metadata.

The first completed `scene_000001` finite same-session smoke passed all five
turns in 162.51 seconds. Its action transcript was initial scan, two bounded
turns, stationary face acknowledgment, two collision-limited forward moves,
stationary approach acknowledgment, explicit scan, state query, standalone stop,
and final state verification. It moved 0.7457 m with zero collisions, refreshed
the continuous map/scene prefix for all six observations, scored every active-map
voxel at every target decision, and finished with the stop latch set. The
model-free inspector passed all 15 checks with zero forbidden reads in both
process audits. The separate oracle-only score reduced chair-center XY distance
from 1.3436 m to 0.5979 m, measured 0.7457 m progress, 0.146° face-heading error,
0.329° final heading error, and zero collisions. This is one deterministic
development-scene integration proof, not held-out navigation generalization.

Run the reproducible numerical action/collision benchmark with `make
robot-evaluate`. A separate label-free semantic policy embeds the user's target
text with the local Gemma tied input-token rows, scores every voxel against the
map's continuous language-aligned visual feature, plans through anonymous geometry,
and executes only bounded numerical robot actions. Run its leakage-blocked policy
and physically separate oracle scorer with:

```bash
make gemma4-semantic-navigation SCENE=scene_000001
```

On the deterministic development room this reached the correct target bounding
box for 8/9 instructions (88.89%), finished within the 0.85 m navigation standoff
for 8/9, had zero collisions, and scored 0/9 on the cyclic wrong-target control.
The cabinet instruction was the sole miss. This is a development-scene vertical
slice, not held-out navigation generalization. Its arrival scan is the sanitized
map reobservation path; the separate `gemma4-embodied-smoke` command proves the
fresh Blender RGB-D → one full-image Gemma pass → 3D fusion → scene-prefix refresh
path.

The current `RobotStateEncoder` maps normalized position, sine/cosine orientation,
velocity, collision, last motion, coverage, and stop state to configurable
continuous tokens that can be appended directly to the continuous scene prefix.

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

### Isolated diverse28 development expansion

`configs/experiments/diverse28.yaml` extends `diverse20` with four new atomic
training pairs at opaque scene IDs 31--38: book on/under the table, mirrored
left/right geometry, picture on wall/floor, and floor-lamp present/removed. It
uses new deterministic pair seeds and varied layouts/colors. The resulting
split trains on scenes 11--18 plus 31--38, keeps validation scenes 19--24
unchanged, and retains scenes 25--30 as the deferred final test. The inherited
artifact paths reuse `data/oracle`, `data/rendered`, and `data_gemma4/maps`; only
the generated QA dataset is isolated at `data_diverse28/qa`.

These development targets name only scenes 31--38 and never pass the deferred
test unlock flag. Inspect, generate, render, and then build balanced QA with:

```bash
make diverse28-dry-run
make diverse28-generate-expansion
make diverse28-render-expansion
make diverse28-generate-dataset
```

The equivalent batch and QA entry points are:

```bash
.venv/bin/python scripts/generate_scene_batch.py \
  --config configs/experiments/diverse28.yaml --stage generate --split train
.venv/bin/python scripts/generate_scene_batch.py \
  --config configs/experiments/diverse28.yaml --stage render --split train
PYTHONPATH=src .venv/bin/python -m semantic_3d_chat.data.qa_generator \
  --config configs/experiments/diverse28.yaml
```

The first two direct commands also visit inherited training scenes 11--18, but
the content-matched cache leaves them unchanged. QA generation fails closed if
any selected development scene or its exact depth-ray visibility evidence is
missing. Counterfactual questions remain selected as indivisible two-scene
units with the inherited per-scene balancing limits.

### V31 expanded-pair development run

V31 consumes the isolated `data_diverse28/qa` dataset after that expansion is
materialized. It is a fresh branch from the selector-approved V29 `update_004`,
not a continuation of the unsuccessful V30 weights. The audited V30
exact-zero/frozen-state engine is reused unchanged: only the sidecar output
projection/gain and the disjoint rank-8 Gemma query bank are trainable.

The strict contract in
`configs/experiments/gemma4_diverse28_joint_pair_v31.yaml` requires exactly 384
questions from training scenes 11--18 plus 31--38, exactly the existing 216
questions from validation scenes 19--24, an empty deferred `test.jsonl`, and no
loaded artifact from scenes 25--30. The training lock also requires all 25
answer-changing units across the eight declared physical-change types. It pins
both the diverse28 source config and
the approved V29 selector/checkpoint hashes. V31 runs eight bounded cycles,
evaluates and checkpoints every cycle, uses clipped zero-weight-decay updates,
and raises the sidecar/decoder learning rates to `1e-4`/`2e-4`. This is a
stronger but still narrow 329,216-parameter repair surface.

```bash
make gemma4-v31-preflight-expanded-pair
make gemma4-v31-train-expanded-pair
make gemma4-v31-select-expanded-pair
```

The selector independently inspects update zero and all eight intermediates.
Development selection requires a strict validation-NLL improvement, retention
of the old 12/12 color and 10/12 mirror sides, no new negative sides, broad-QA
non-regression, and at least one newly correct complete changed validation pair.
That is only evidence of development progress. Chat promotion remains a
separate, harder gate requiring at least 6/12 changed pairs and no aggregate
validation exact-accuracy regression. There is intentionally no V31 final-test
target.

### Conditional V32 true-microstep optimizer repair

V32 is prepared only as a fallback and is conditional on an independently
rejected V31 selector report. Its trainer fails closed while V31 is pending and
also refuses to run if V31 passes. It does not continue V31 (or V30): it starts
again from the hash-pinned, selector-approved V29 `update_004`, reconstructs the
same exact-zero cache, and permits only new training scenes 31--38 to use the
bit-exact repeated derived-prefix path. Validation remains scenes 19--24 and
final scenes 25--30 remain deferred and unopened.

The V31 evidence showed that each full cycle became one giant clipped optimizer
step: validation NLL improved while changed validation pairs remained at 0/12,
with pre-clip norms around 18--20 against a clip of 1. V32 repairs that optimizer
granularity. It makes 80 real AdamW updates. Every update combines one balanced
broad QA record with one indivisible changed-answer pair unit; all 25 units
recur at least three times. The learning rates are `2.5e-5` for the sidecar
surfaces and `2e-5` for the fresh decoder bank, with zero weight decay
and the existing norm-1 clip. The authorized parameter surface remains exactly
329,216 parameters.

```bash
make gemma4-v32-preflight-microstep
# The next command is authorized only after V31's selector writes a rejection.
make gemma4-v32-train-microstep
make gemma4-v32-select-microstep
```

V32 saves update zero and optimizer steps 8, 16, ..., 80. The selector must
independently inspect all eleven arms and verifies that each saved Adam state
records the claimed number of true optimizer updates. Each candidate must strictly improve on
the selected V29 validation NLL, retain the old 12/12 color and 10/12 mirror
teacher controls without a new negative side, avoid broad-QA regression, and
solve at least 1/12 complete changed validation pairs. That is a development
gate only. Chat promotion separately requires at least 6/12 changed pairs and
no aggregate exact-accuracy regression. There is intentionally no V32-specific
final-test bypass target.
An interrupted `make gemma4-v32-train-microstep` resumes from the latest
contiguous complete saved arm. Resume validates the config, V31-rejection hash,
schedule hash, data boundary, full adapter state, and Adam step counters before
performing the next micro-update; a partial or forged checkpoint fails closed.

### V33 environmental-only sidecar recovery

V32 was formally rejected: decoder updates reduced validation NLL but did not
produce a complete non-mirror counterfactual answer. V33 therefore restarts
from the same hash-pinned V29 `update_004` and freezes Gemma, every LoRA bank,
the base-normalization/projection route, and the rest of the scene stack. Its
only trainable surface is 404,608 parameters on the continuous all-voxel
dense-sidecar path: output projection plus channel gain (198,144), sidecar
normalization plus projection (199,808), and position projection (6,656).

The 100 true AdamW updates contain one broad record and one indivisible pair
unit each, so all 25 changed training units recur exactly four times. The three
parameter groups use learning rates `2.5e-5`, `1e-4`, and `1e-4`, zero weight
decay, and independently measured/clipped norm-1 gradients. Saved arms are
0, 8, ..., 96, and 100. Actual adapted continuous-prefix distances are audited
at every saved arm; weak book/picture separations must improve by at least 25%
without increasing unrelated-scene separation by more than 25%.

```bash
make gemma4-v33-preflight-environmental
make gemma4-v33-train-environmental
make gemma4-v33-select-environmental
```

The preflight is pinned to the exact terminal V32 rejection SHA-256 and refuses
any existing final-scene 25--30 footprint. At update 64, training saves the
causal checkpoint and then refuses to continue unless at least one non-mirror
teacher-forced unit is complete and both book and picture margin advantages
are positive. A failed update-64 arm remains valid evidence for the defined,
disabled 199,808-parameter base-route follow-up; resume cannot bypass the gate.

Chat promotion still requires at least 6/12 complete changed validation units,
at least one in each book/mirror/picture family, all old 12 color and 10 mirror
sides with no new negatives, and no aggregate exact-accuracy regression. There
is no V33 final-test bypass target.

### V34 bounded base-route isolation

V33 stopped as designed immediately after saving update 64. Its two weak
non-mirror families regressed instead of crossing the causal gate: the book
teacher margin moved from `0.065371` to `-0.006955`, the picture margin moved
from `-0.054688` to `-0.177734`, and non-mirror completed units remained zero.
The weak-pair adapted-prefix RMS changed by only `1.000267x`, essentially the
same as unrelated scenes (`1.000553x`). The metadata/tensor-only terminal seal
pins all four update-64 files, proves there is no update 72 or later, verifies
the V29/V32/config/schedule lineage, and opens no model, QA, map, oracle, or
final-scene artifact:

```bash
make gemma4-v33-seal-update64
```

V34 is one bounded architectural isolation, not an indefinite hyperparameter
search. It starts from the exact numbered V33 `update_064`; update zero must be
bit-exact in every checkpoint tensor, every training-scene prefix, and the
216-row validation NLL. Gemma, all LoRA banks, every V33-learned environmental
tensor, structural buffers, and all other scene modules remain frozen. Only
`base_norm.{weight,bias}` and `base_projection.{weight,bias}` train: 199,808
parameters split into fresh Adam groups at learning rates `2.5e-5` and `1e-4`,
each with its own norm-1 clip and zero weight decay.

The language schedule still covers all 25 atomic changed QA units in 64 true
microsteps (11 units twice and 14 three times). Its additional question-free
geometry objective is reduced separately over the eight unique changed
physical training-scene pairs and all 112 other pairs among the same 16 train
scenes. Distances are divided by fixed update-zero RMS baselines. The loss uses
the bounded log-selectivity ranking
`log(changed_ratio) - mean(log(unrelated_ratio))`, a two-sided unrelated-scene
log trust penalty, a changed-ratio cap, and a source-centered prefix mean/norm
trust region. It never uses validation IDs, questions, answers, or oracle
environment inputs.

```bash
make gemma4-v34-preflight-base-surface
make gemma4-v34-train-base-surface
make gemma4-v34-select-base-surface
```

Update 32 is a train-only causal stop: geometric-mean selectivity must exceed
`1.02`, at least 6/8 physical changes must exceed `1.02`, none may fall below
`0.98`, and unrelated median/P90 drift must remain within the same two-sided
`1.02` trust region. Validation is logged but cannot decide whether training
continues. Only after a complete bounded run does the independent selector
replay all nine arms on development validation. Retention and aggregate
non-regression are measured against approved V29; improvements are measured
against V33 update 64. Chat promotion additionally requires 6/12 changed
pairs, every book/mirror/picture family, 12 color and 10 mirror control sides,
no new V29-relative negatives, and aggregate exact non-regression. Rich checks
remain separately audited while the outward promotion attestation preserves
the exact three-field final-once protocol. There is no V34 final-test bypass.

V34 stopped as designed after saving update 32 because the train-only causal
gate failed. Changed-pair selectivity was `1.00003981590271x`, no physical pair
exceeded the required `1.02x` ratio, and the inherited 199,808-parameter base
surface therefore did not expose the missing scene distinctions. The terminal
seal verifies the exact five saved arms, all four update-32 file hashes, every
optimizer step, and the four changed base tensors while proving all inherited
tensors stayed frozen. It loads no Gemma weights, maps, QA, oracle data, or
final scenes:

```bash
make gemma4-v34-seal-update32
```

This failure does not authorize chat or final evaluation. It conditionally
authorizes only the exact-zero V35 block-token cross-residual experiment; no
other follow-up architecture is authorized by the seal.

### V35 bounded all-block cross-residual

V34 established that the inherited low-rank/base surfaces did not expose the
changed physical facts. V35 is the bounded high-rank repair. It restarts from
the exact immutable V33 `update_064` (tensor-state SHA-256
`cb7bb3b48ace60212ee5c7f326839bf2ddd993810417de45c9a9cbc666313fe6`);
V34 update 32 is never a weight source. Its terminal report merely authorizes
this one architecture.

For each of the 22 development scenes, the question-free cache retains exact
post-V33 scene tokens as float32 CPU tensors, every occupied-block token as
float16 CPU, and the repeated normalized XYZ position for every block token.
It checks `processed_voxels == voxel_count` and
`token_count == tokens_per_block * occupied_blocks`. The six validation-scene
caches are used only to prove exact update-zero prefix identity; all optimizer
losses, residual penalties, separation diagnostics, and continuation gates use
only the 16 training scenes. Training never opens `validation.jsonl`, an oracle
file, or a deferred final-scene artifact.

The new module sends all block tokens into all 256 scene slots with four-head
FP32 cross-attention, a 1% uniform attention floor, normalized spatial bias,
and a bounded `0.25 * tanh` residual. Its four matrices
`w_q/w_k/w_v/w_o` contain exactly 983,040 parameters. `w_o` starts at exact
zero, making update zero bit-identical to V33. Step 1 trains only `w_o` at
`2.5e-5`; subsequent steps open Q/K/V at `1e-4`. Gemma, every LoRA bank, and
the complete V33 scene stack remain frozen and hash-checked.

The 100-step schedule presents every one of the 25 atomic changed QA units
exactly four times plus one balanced unchanged broad example per step. Its
locked loss is
`0.25*broad_CE + 0.5*pair_CE + 4*side_hinge(0.5) + 8*cross_prefix_flip_hinge(0.25) + 0.001*mean((delta/0.05)^2)`.
The cross-prefix term scores the exact differing answer tokens under both
physical scene prefixes; it is not a vector-distance surrogate.

```bash
make gemma4-v35-preflight-block-cross
make gemma4-v35-train-block-cross
make gemma4-v35-select-block-cross
```

Update 32 is a train-only causal gate requiring at least `1.02x` geometric
changed-prefix selectivity, 6/8 changed physical pairs above `1.02x`, no pair
below `0.98x`, bounded unrelated drift, strictly improved training mean margin
and completed-unit count, and residual RMS at most `0.10`. Update 64 additionally
requires at least 8/25 completed changed QA units and at least one completion in
each of the book-support, mirror-left/right, and picture-support training
families. A failed gate is saved as evidence and cannot be bypassed by resume.
Validation QA is evaluated only afterward by an independent selector. V35 has
no chat or final-test bypass target.

#### V35 post-training selector

The independent selector refuses to load validation QA or Gemma until the
checkpoint directory contains exactly `update_000, update_008, ..., update_096,
update_100`. It inspects every adapter tensor, sanitized runtime metadata file,
and saved Adam state; independently replays the training-only update-32 and
update-64 gates; and requires both gates to have passed. Validation therefore
cannot influence training continuation.

With one local Gemma load, the selector builds the complete question-free
all-block cache and evaluates teacher-forced NLL, pair margins, and continuous
prefix separation for every numbered arm. Greedy generation is bounded to
updates 32, 64, and 100. Exact V33 update 64 is the improvement baseline;
approved V29 is the color/mirror/no-new-negative, broad-retention, and aggregate
accuracy baseline. A development arm must strictly improve validation NLL,
mean pair margin, and passed pair count; separate every book, mirror, and
picture family more than unrelated scenes; retain old controls; and complete a
non-mirror teacher unit. Chat promotion additionally requires 6/12 greedy
counterfactual completions, one in every family, aggregate exact non-regression,
and question-independent prefix/leakage attestations. The outward promotion
record retains exactly the three fields required by the final-once controller;
all richer requirements are recorded separately. This target never reads or
creates deferred final scenes.

V35 stopped as designed after saving update 32 because its first train-only
causal gate failed. The changed-pair prefix-selectivity geometric mean was
`1.000036597251892x`, with `0/8` physical pairs reaching the required `1.02x`;
book-support and picture-support each still had zero complete training units.
The slight training-margin improvement (`1.2638574838638306` to
`1.32265043258667`) and completion increase (`8` to `9`) therefore do not
constitute evidence that the scene prefixes became causally discriminative.

The deterministic terminal seal checks every byte of all five saved arms,
replays the exact failure from metadata, validates all four staged Adam states,
proves all 168 inherited V33 tensors stayed bit-exact, and proves the only
changed tensors were `w_q/w_k/w_v/w_o` while all seven persistent block-core
buffers remained exact. It opens no Gemma weights, QA, scene maps, oracle data,
or deferred final scenes:

```bash
make gemma4-v35-seal-update32
```

This seal does not authorize selection, chat, or final evaluation. It permits
only a fresh-Adam V36 experiment sourced from exact V35 update 32: updates 1--8
may train the already-present exact-zero rank-8
`extension_v30_joint_pair_query` bank on Q projections in language layers
18--21; updates 9--100 may jointly train that same bank and the four V35 block
matrices. No new LoRA bank or other inherited V33 tensor is authorized to
change.

### V36 bounded joint decoder readout

V35 proved that the all-block route can improve train-only answer margins, but
its Euclidean prefix-distance gate did not measure whether Gemma could decode
those small continuous changes. V36 isolates that readout hypothesis. It loads
the exact stopped V35 `update_032` adapter (full tensor-state SHA-256
`1fe8f278460faeb1e13d9da09051a497965a566565c79a4f6ea28c56a9120326`),
including learned block-core state `75af9958...`; it does not restart from the
zero V35 core and never loads V35 Adam momentum. The inherited V33 stack
remains exact.

The only language surface is the existing 131,072-parameter
`extension_v30_joint_pair_query` bank: rank 8, alpha 16, Q projections in
Gemma layers 18--21. Its B matrices are still exact zero at V36 update zero.
Updates 1--8 train only that bank at `2e-5`. Updates 9--100 jointly train the
bank, block W_o at `2.5e-5`, and block Q/K/V at `1e-4`, with fresh Adam, zero
weight decay, and separate gradient clipping. The total authorized surface is
1,114,112 parameters. A 167-tensor hash includes every nonauthorized tensor
and all seven persistent core buffers, excluding only the four core matrices
and eight query-bank tensors.

V36 deliberately reuses V35's exact 100-step schedule and loss, so decoder
readout is the changed factor. The six validation maps are used only for a
question-free source-prefix replay; every loss and continuation metric uses
the 16 training scenes, and `validation.jsonl` is never opened. Update 16
requires improvement beyond the exact V35 source plus bounded broad NLL and
residual drift. Update 32 raises the teacher thresholds and requires at least
one complete book, mirror, and picture unit. Update 64 additionally requires
train-only greedy completion of at least 6/25 units, every priority family,
and broad greedy retention. Euclidean separation remains descriptive rather
than a continuation gate.

```bash
make gemma4-v36-preflight-joint-block-cross
make gemma4-v36-train-joint-block-cross
make gemma4-v36-select-joint-block-cross
```

The V36 post-training selector is the first process permitted to open validation QA,
and only after all numbered arms through update 100 and all three train-only
gates pass. It inspects every tensor and persistent buffer, freshly sanitized
runtime metadata, and every staged Adam moment before loading Gemma. It scores
teacher-forced evidence for all 14 arms and bounded greedy evidence only at
updates 32, 64, and 100. Chat promotion still requires at least 6/12 complete
development counterfactual units, at least one book, mirror, and picture unit,
approved-V29 control and aggregate retention, and prefix-invariance/leakage
attestations. V36 has no final-evaluation or chat bypass.

V36 stopped safely at its first train-only gate after update 16. Teacher-forced
completion remained `9/25` against a required `10/25`, and complete physical-pair
coverage was `4/8` against a required `5/8`. Cross-prefix completion improved to
`16/25`, positive sides to `34/50`, and mean cross-prefix margin to
`1.4565558433532715`; broad train NLL stayed bounded at `2.915099874138832`
(`0.9824921284356688x` its source), but those partial gains cannot authorize
development selection or chat.

The update-16 terminal seal pins all V35 source and V36 update 0/8/16 files,
full/core/query-bank/frozen tensor hashes, sanitized runtime metadata, and every
fresh staged-Adam moment. It proves update 8 changed only the eight V30 query-bank
tensors, update 16 changed only those tensors plus the four block matrices, and
all 167 nonauthorized tensors stayed bit-exact. The report process loads no Gemma,
QA, scene maps, oracle metadata, or final scenes and only hashes the protected
legacy selection artifact:

```bash
make gemma4-v36-seal-update16
```

The only conditional successor is a bounded V37 continuation of the already
learned 30,720-parameter `extension_v23_shared_kv` bank. That rank-4, alpha-8
bank targets K/V projections only in layers 13 and 14, the last non-shared
sliding/full K/V producers consumed by Gemma's upper shared-KV layers. No fresh
bank is legal because duplicate LoRA target paths are forbidden, and layers
18--21 do not expose operative K/V projection modules. V37 must freeze the V36
query bank, learned block core, scene stack, composer, Gemma base, and every
other tensor; start fresh Adam; preserve the exact question-independent scene
prefix; and pass its preregistered train-only gates before validation can open.

### One-shot held-out final evaluation

The generic final-once controller is dormant until an independent selector
both selects one exact checkpoint and sets `chat_promotion_eligible: true`.
Development success or a zero exit status is insufficient. Before any command
can write scenes 25--30, the controller validates that attestation, the exact
checkpoint path, standalone runtime config, deferred scene plan, untouched
final footprint, and development-QA hashes. Preflight is read-only:

```bash
make gemma4-final-once-preflight \
  GEMMA4_FINAL_ONCE_SELECTOR_REPORT=reports/gemma4/metrics/v32_microstep_selection.json \
  GEMMA4_FINAL_ONCE_CHECKPOINT=/exact/selector-chosen/checkpoint
```

Only after that succeeds, the complete resumable run is:

```bash
make gemma4-final-once \
  GEMMA4_FINAL_ONCE_SELECTOR_REPORT=reports/gemma4/metrics/v32_microstep_selection.json \
  GEMMA4_FINAL_ONCE_CHECKPOINT=/exact/selector-chosen/checkpoint
```

It atomically seals config chains, code, selector, checkpoint, split plan,
output paths, and the SHA-256 of every file in the exact pinned local Gemma
snapshot before invoking Blender. It then generates and renders exactly the six
deferred scenes, loads Gemma once for all full-image feature extraction, builds
all maps, creates the scene-disjoint 216-question final split with exactly four
answer-changing units for each of the three physical counterfactual pairs, and
creates a questions-only manifest. Final supervision is generated first in an
isolated evaluation directory; its train/validation bytes must reproduce the
sealed development hashes, and only test/split artifacts are published. It runs
complete primary and empty-prefix inference, scores counterfactuals and
grounding, performs the real oracle-rename
leakage test, and publishes promotion plus the primary pointer only when the
primary run beats the empty-prefix control on aggregate text accuracy,
counterfactual pair accuracy, changed-answer rate, and grounding error. The
leakage prefix hash must also equal the prefix used by primary evaluation for
that exact scene. The fail-closed minimums are 50% pair accuracy, 50%
changed-when-expected rate, 100% grounding coverage, mean grounding error no
larger than half the room diagonal, and at least 10% lower grounding error than
the empty-prefix run. Every stage has a SHA-256 receipt. An interrupted command
may be rerun; changed inputs or outputs fail instead of being mixed. For a bounded
operator checkpoint, set `GEMMA4_FINAL_ONCE_STOP_AFTER=render` (or another stage)
and rerun later without that variable. No final-once command should run while
another process may access `data/oracle`, because the leakage stage deliberately
renames that directory during inference.

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

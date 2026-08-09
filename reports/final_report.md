# Semantic 3D Chat — First Proof-of-Concept Report

> **PRIMARY-RESULT WARNING — LEGACY REPORT BODY.** The CLIP/Qwen tables and chat
> evidence preserved below are superseded as the project's primary model path. They
> remain useful failure/control evidence, but they are not Gemma 4 results and do
> not establish scene-conditioned understanding. The current primary Gemma 4 E2B
> path has a working dense 3D map and semantic-localization result, but no adapter
> checkpoint has passed its wiring gate. There is no accepted Gemma static-chat or
> one-command-demo result yet. The controlled v8 LoRA fallback is implemented but
> has not yet produced a real MPS result.

Generated from local artifacts on `2026-08-09T03:34:21+00:00`. This report does not run models and does not infer missing measurements.

## 1. Research question

Can a local language model answer questions about a synthetic room when the environment reaches it only as continuous, spatially fused visual embeddings and geometry—without a caption, object list, textual scene graph, simulator labels, or question-dependent retrieval?

### Current primary Gemma 4 E2B outcome

The primary representation is 24 complete-image Gemma observations, each retaining
a 48×48 field of middle 768D, late 768D, and native projected 1536D features
(3072D float16 total), fused into a 5 cm persistent map and deterministically
aggregated to 15 cm tokenizer input. Every occupied block feeds the
question-independent `signal_preserving_resampler_v3`, producing 256 scene
latents at model dimension 384 before projection to Gemma's 1536D hidden space.

Gemma v7 completed 12 epochs, 72 decoder microsteps, and 12 optimizer updates in
567.844 seconds. The best checkpoint was epoch 7: 3/6 changed units, 9/12 correct
sides, 3/6 prediction flips, 3/6 wrong-prefix flips, mean ranking margin 0.439616,
minimum margin -0.71875, and hinge 0.314453. The final checkpoint again reached
3/6 changed units and 9/12 sides, with hinge 0.386068 and minimum margin -1.484375.
The required teacher-forced counterfactual gate therefore failed. The higher
4/6 changed-unit count at epoch 3 also failed because its hinge was 0.716146 and its
minimum margin was -3.59375. Runs v1-v6 and the v6 resumes through epochs 18 and 24
also failed; none is promoted.

Gemma held-out QA, free generation, interactive chat, prefix invariance,
oracle-deletion inference, and language-conditioned robot navigation remain
unmeasured. The exact source hash loaded by the already-running v7 process was not
captured. The implementation hashes in
`gemma4/metrics/gemma4_color_wiring_v7_failure.json` are post-run audited snapshots
and do not prove that later padding or audit/resume fixes executed in v7.

#### v8 controlled fallback (implemented; real MPS run unmeasured)

The controlled v8 fallback leaves the native boundaries and complete continuous
scene prefix unchanged. It adapts only layer 34
`model.language_model.layers.34.self_attn.q_proj` and `o_proj`, using rank 4,
alpha 8, dropout 0, LoRA learning rate `1e-4`, and weight decay 0. This is 45,056
FP32 A/B-only parameters (180,224 bytes, approximately 176 KiB). Strict config,
optimizer, checkpoint, SHA/tamper, resume, chat-load, and scene-signal-audit paths
have test coverage. No behavioral or performance result is claimed until the real
MPS run finishes. Reproduce selection and training with:

```bash
PYTHONPATH=src .venv-gemma4/bin/python -m semantic_3d_chat.training.train_adapter \
  --config configs/experiments/gemma4_color_wiring_v8.yaml --selection-only
PYTHONPATH=src .venv-gemma4/bin/python -m semantic_3d_chat.training.train_adapter \
  --config configs/experiments/gemma4_color_wiring_v8.yaml
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
for Gemma v7 and cannot be transferred to any Gemma checkpoint.

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
Fluent chat samples must not be treated as evidence of scene understanding; only the structured held-out and control measurements support behavioral claims.

## 25. Preserved legacy prefix-invariance evidence

PASS for checkpoint `data/checkpoints/best`. Prefix `f4ed6bc9cbf75bd878bcebab04e432ad3eb04ef236aadd6659931c8f214b7c9b` was constructed before the first question and remained identical across 3 questions.

## 26. Preserved legacy oracle-deletion evidence

PASS for checkpoint `data/checkpoints/best`. The oracle directory was atomically renamed away during local inference, no forbidden path was opened, answers completed, and the directory was restored. This result is not automatically transferred to a different checkpoint without rerunning the test.

## 27. Exact remaining limitations

- Gemma v7 failed its six-unit teacher-forced counterfactual gate; no Gemma
  checkpoint is behaviorally promoted.
- No Gemma held-out QA, free-generation, interactive-chat, prefix-invariance, or
  oracle-deletion inference result exists.
- The exact source hash loaded by the v7 process was not captured; current source
  hashes are post-run audited snapshots.
- The v1 multi-scene adapter is scene-content-insensitive despite its raw held-out accuracy; wrong-scene and content-shuffle controls invalidate a scene-understanding claim for that checkpoint.
- The v2 structural diagnostic preserves more scene signal, but no explicitly v2-tagged held-out QA artifact is available yet.
- Selected-run wall-clock training time is not recorded.
- Peak training memory is not recorded.
- Expected-change counterfactual consistency is zero.
- The direct multi-view image baseline is not scored.
- The prohibited oracle-text upper bound is not scored.
- The robot benchmark covers numeric mechanics and MCP wiring only; language-conditioned semantic target navigation is unmeasured.
- The deterministic robot scan is a pose-dependent numerical map reobservation, not an arbitrary-pose Blender render plus CLIP remapping.
- A center scan reconstructs visible surfaces but cannot reveal occluded rear surfaces.
- Legacy CLIP patch semantics missed its top-k query set for: book.

## 28. Recommended next experiments

1. Require the next Gemma adapter to pass every teacher-forced changed-unit,
   prediction-flip, wrong-prefix-flip, margin, and hinge threshold before creating a
   hash-bound promotion record.
2. Only after promotion, run Gemma held-out prediction, continuous-scene controls,
   free generation, prefix invariance, oracle deletion, and interactive chat.
3. Run the direct multi-view VLM and isolated oracle-text upper-bound baselines.
4. Train and evaluate language-conditioned target-facing and approach behavior without returning semantic labels through tools.
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

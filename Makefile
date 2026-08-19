PYTHON := .venv/bin/python
GEMMA4_PYTHON := .venv-gemma4/bin/python
GEMMA4_CONFIG ?= configs/gemma4_e2b.yaml
GEMMA4_MODEL := google/gemma-4-E2B-it
GEMMA4_REVISION := 3e22461f65e89153144f8adb70e3b8c2cc9845a7
# Intentionally empty: chat/evaluation accepts only an explicitly selected,
# behaviorally promoted Gemma checkpoint/config pair.
GEMMA4_STATIC_CONFIG ?=
GEMMA4_STATIC_CHECKPOINT ?=
GEMMA4_STATIC_REFERENCES ?=
GEMMA4_CONTROL_CONFIG ?= configs/runtime/gemma4_v56_question_control.yaml
GEMMA4_CONTROL_BASE_CHECKPOINT ?= data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000
GEMMA4_CONTROL_CHECKPOINT ?=
GEMMA4_CONTROL_TEACHER_ARTIFACT ?= data_gemma4/training/v58_teachers_pair31_32
GEMMA4_RUNTIME_CONFIG ?= configs/runtime/gemma4_primary.yaml
GEMMA4_PRIMARY_POINTER ?= configs/runtime/primary.json
GEMMA4_PROMOTION_CHECKPOINT ?=
GEMMA4_PROMOTION_SELECTOR_REPORT ?=
GEMMA4_PROMOTION_FINAL_EVIDENCE ?=
GEMMA4_PROMOTION_LEAKAGE_REPORT ?=
GEMMA4_FINAL_METRICS ?=
GEMMA4_FINAL_PREDICTIONS ?=
GEMMA4_FINAL_PREDICTION_PROVENANCE ?=
GEMMA4_FINAL_CHANCE_METRICS ?=
GEMMA4_FINAL_CHANCE_PREDICTIONS ?=
GEMMA4_FINAL_CHANCE_PROVENANCE ?=
GEMMA4_FINAL_SPLIT_MANIFEST ?=
GEMMA4_FINAL_EVIDENCE_OUTPUT ?= reports/gemma4/metrics/held_out_final_evidence.json
GEMMA4_EVAL_SPLIT ?= test
GEMMA4_QUESTIONS_MANIFEST ?= reports/gemma4/questions/$(GEMMA4_EVAL_SPLIT).json
GEMMA4_STATIC_PREDICTIONS ?= reports/gemma4/predictions/$(GEMMA4_EVAL_SPLIT).jsonl
GEMMA4_STATIC_METRICS ?= reports/gemma4/metrics/static_qa_$(GEMMA4_EVAL_SPLIT).json
GEMMA4_CONTROL_PREDICTIONS ?= reports/gemma4/predictions/controls/$(GEMMA4_EVAL_SPLIT)
GEMMA4_CONTROL_REFERENCES ?=
GEMMA4_CONTROL_METRICS ?= reports/gemma4/metrics/controls/$(GEMMA4_EVAL_SPLIT)
GEMMA4_V18_CONFIG := configs/experiments/gemma4_color_mirror_centered_content_gate_v18.yaml
GEMMA4_V18_NAMESPACE := gemma4_color_mirror_centered_content_gate_v18
GEMMA4_V18_CHECKPOINT_ROOT := data_gemma4/checkpoints/$(GEMMA4_V18_NAMESPACE)
GEMMA4_V18_PREFLIGHT := reports/gemma4/metrics/v18_structural_preflight.json
GEMMA4_V18_UPDATE1_REPORT := reports/gemma4/metrics/v18_update1_match.json
GEMMA4_V18_SELECTION := reports/gemma4/metrics/training_selection_$(GEMMA4_V18_NAMESPACE).json
GEMMA4_V18_SCREEN_REPORT := reports/gemma4/metrics/v18_epoch_screen.json
GEMMA4_V19_CONFIG := configs/experiments/gemma4_color_mirror_signed_x_moment_v19.yaml
GEMMA4_V19_NAMESPACE := gemma4_color_mirror_signed_x_moment_v19
GEMMA4_V19_CHECKPOINT_ROOT := data_gemma4/checkpoints/$(GEMMA4_V19_NAMESPACE)
GEMMA4_V19_PREFLIGHT := reports/gemma4/metrics/v19_structural_preflight.json
GEMMA4_V19_UPDATE1_REPORT := reports/gemma4/metrics/v19_update1_match.json
GEMMA4_V19_SELECTION := reports/gemma4/metrics/training_selection_$(GEMMA4_V19_NAMESPACE).json
GEMMA4_V19_SCREEN_REPORT := reports/gemma4/metrics/v19_epoch_screen.json
GEMMA4_V19_EXTENSION_NAMESPACE := gemma4_color_mirror_signed_x_moment_v19_extension_u12
GEMMA4_V19_EXTENSION_ROOT := data_gemma4/checkpoints/$(GEMMA4_V19_EXTENSION_NAMESPACE)
GEMMA4_V19_EXTENSION_MANIFEST := reports/gemma4/metrics/v19_extension_launch.json
GEMMA4_V19_EXTENSION_REPORT := reports/gemma4/metrics/v19_extension_final.json
GEMMA4_V20_CONFIG := configs/experiments/gemma4_color_mirror_signed_x_local_field_v20.yaml
GEMMA4_V20_NAMESPACE := gemma4_color_mirror_signed_x_local_field_v20
GEMMA4_V20_CHECKPOINT_ROOT := data_gemma4/checkpoints/$(GEMMA4_V20_NAMESPACE)
GEMMA4_V20_PREFLIGHT := reports/gemma4/metrics/v20_structural_preflight.json
GEMMA4_V20_UPDATE1_REPORT := reports/gemma4/metrics/v20_update1_match.json
GEMMA4_V20_SELECTION := reports/gemma4/metrics/training_selection_$(GEMMA4_V20_NAMESPACE).json
GEMMA4_V20_SCREEN_REPORT := reports/gemma4/metrics/v20_epoch_screen.json
GEMMA4_V20_EXTENSION_NAMESPACE := gemma4_color_mirror_signed_x_local_field_v20_extension_u8
GEMMA4_V20_EXTENSION_ROOT := data_gemma4/checkpoints/$(GEMMA4_V20_EXTENSION_NAMESPACE)
GEMMA4_V20_EXTENSION_MANIFEST := reports/gemma4/metrics/v20_extension_launch.json
GEMMA4_V20_EXTENSION_REPORT := reports/gemma4/metrics/v20_extension_final.json
GEMMA4_V21_CONFIG := configs/experiments/gemma4_color_mirror_signed_x_local_field_phase_aware_v21.yaml
GEMMA4_V21_NAMESPACE := gemma4_color_mirror_signed_x_local_field_phase_aware_v21
GEMMA4_V21_CHECKPOINT_ROOT := data_gemma4/checkpoints/$(GEMMA4_V21_NAMESPACE)
GEMMA4_V21_PREFLIGHT := reports/gemma4/metrics/v21_structural_preflight.json
GEMMA4_V21_UPDATE1_REPORT := reports/gemma4/metrics/v21_update1_match.json
GEMMA4_V21_SELECTION := reports/gemma4/metrics/training_selection_$(GEMMA4_V21_NAMESPACE).json
GEMMA4_V21_SCREEN_REPORT := reports/gemma4/metrics/v21_epoch_screen.json
GEMMA4_V21_EXTENSION_NAMESPACE := gemma4_v21_phase_aware_local_field_extension_u8
GEMMA4_V21_EXTENSION_ROOT := data_gemma4/checkpoints/$(GEMMA4_V21_EXTENSION_NAMESPACE)
GEMMA4_V21_EXTENSION_MANIFEST := reports/gemma4/metrics/v21_extension_launch.json
GEMMA4_V21_EXTENSION_REPORT := reports/gemma4/metrics/v21_extension_final.json
GEMMA4_V22_CONFIG := configs/experiments/gemma4_color_mirror_signed_x_local_field_margin_rebalanced_v22.yaml
GEMMA4_V22_NAMESPACE := gemma4_v22_margin_rebalanced_local_field
GEMMA4_V22_CHECKPOINT_ROOT := data_gemma4/checkpoints/$(GEMMA4_V22_NAMESPACE)
GEMMA4_V22_PREFLIGHT := reports/gemma4/metrics/v22_structural_preflight.json
GEMMA4_V22_UPDATE1_REPORT := reports/gemma4/metrics/v22_update1_match.json
GEMMA4_V22_SELECTION := reports/gemma4/metrics/training_selection_$(GEMMA4_V22_NAMESPACE).json
GEMMA4_V22_SCREEN_REPORT := reports/gemma4/metrics/v22_epoch_screen.json
GEMMA4_V22_EXTENSION_NAMESPACE := gemma4_v22_margin_rebalanced_local_field_extension_u8
GEMMA4_V22_EXTENSION_ROOT := data_gemma4/checkpoints/$(GEMMA4_V22_EXTENSION_NAMESPACE)
GEMMA4_V22_EXTENSION_MANIFEST := reports/gemma4/metrics/v22_extension_launch.json
GEMMA4_V22_EXTENSION_REPORT := reports/gemma4/metrics/v22_extension_final.json
GEMMA4_V23_CONFIG := configs/experiments/gemma4_color_mirror_shared_kv_v23.yaml
GEMMA4_V23_NAMESPACE := gemma4_v23_shared_kv
GEMMA4_V23_CHECKPOINT_ROOT := data_gemma4/checkpoints/$(GEMMA4_V23_NAMESPACE)
GEMMA4_V23_PREFLIGHT := reports/gemma4/metrics/v23_structural_preflight.json
GEMMA4_V23_UPDATE1_REPORT := reports/gemma4/metrics/v23_update1_match.json
GEMMA4_V23_SCREEN_REPORT := reports/gemma4/metrics/v23_epoch_screen.json
GEMMA4_V23_EXTENSION_NAMESPACE := gemma4_v23_shared_kv_extension_u8
GEMMA4_V23_EXTENSION_ROOT := data_gemma4/checkpoints/$(GEMMA4_V23_EXTENSION_NAMESPACE)
GEMMA4_V23_EXTENSION_MANIFEST := reports/gemma4/metrics/v23_extension_launch.json
GEMMA4_V23_EXTENSION_REPLAY := reports/gemma4/metrics/v23_extension_replay.json
GEMMA4_V23_EXTENSION_REPORT := reports/gemma4/metrics/v23_extension_final.json
GEMMA4_V24_CONFIG := configs/experiments/gemma4_color_mirror_shared_query_v24.yaml
GEMMA4_V24_NAMESPACE := gemma4_v24_shared_query
GEMMA4_V24_CHECKPOINT_ROOT := data_gemma4/checkpoints/$(GEMMA4_V24_NAMESPACE)
GEMMA4_V24_PREFLIGHT := reports/gemma4/metrics/v24_structural_preflight.json
GEMMA4_V24_UPDATE1_REPORT := reports/gemma4/metrics/v24_update1_match.json
GEMMA4_V24_SCREEN_REPORT := reports/gemma4/metrics/v24_epoch_screen.json
GEMMA4_V24_EXTENSION_NAMESPACE := gemma4_v24_shared_query_extension_u8
GEMMA4_V24_EXTENSION_ROOT := data_gemma4/checkpoints/$(GEMMA4_V24_EXTENSION_NAMESPACE)
GEMMA4_V24_EXTENSION_MANIFEST := reports/gemma4/metrics/v24_extension_launch.json
GEMMA4_V24_EXTENSION_REPLAY := reports/gemma4/metrics/v24_extension_replay.json
GEMMA4_V24_EXTENSION_REPORT := reports/gemma4/metrics/v24_extension_final.json
GEMMA4_V25_CONFIG := configs/experiments/gemma4_color_mirror_dense_alignment_v25.yaml
GEMMA4_V25_NAMESPACE := gemma4_v25_dense_alignment
GEMMA4_V25_CHECKPOINT_ROOT := data_gemma4/checkpoints/$(GEMMA4_V25_NAMESPACE)
GEMMA4_V25_PREFLIGHT := reports/gemma4/metrics/v25_structural_preflight.json
GEMMA4_V25_CALIBRATION_REPORT := reports/gemma4/metrics/v25_dense_alignment_calibration.json
GEMMA4_V25_CALIBRATION_BRIDGE := reports/gemma4/artifacts/v25_dense_alignment_bridge.safetensors
GEMMA4_V25_CALIBRATION_DECISION := reports/gemma4/metrics/v25_dense_alignment_calibration_decision.json
GEMMA4_V25_UPDATE1_REPORT := reports/gemma4/metrics/v25_update1_match.json
GEMMA4_V25_SCREEN_REPORT := reports/gemma4/metrics/v25_epoch_screen.json
GEMMA4_V26_CONFIG := configs/experiments/gemma4_color_mirror_dense_alignment_v26.yaml
GEMMA4_V26_NAMESPACE := gemma4_v26_dense_alignment
GEMMA4_V26_CHECKPOINT_ROOT := data_gemma4/checkpoints/$(GEMMA4_V26_NAMESPACE)
GEMMA4_V26_PREFLIGHT := reports/gemma4/metrics/v26_structural_preflight.json
GEMMA4_V26_CALIBRATION_REPORT := reports/gemma4/metrics/v26_dense_alignment_calibration.json
GEMMA4_V26_CALIBRATION_BRIDGE := reports/gemma4/artifacts/v26_dense_alignment_bridge.safetensors
GEMMA4_V26_CALIBRATION_DECISION := reports/gemma4/metrics/v26_dense_alignment_calibration_decision.json
GEMMA4_V26_UPDATE1_REPORT := reports/gemma4/metrics/v26_update1_match.json
GEMMA4_V26_SCREEN_REPORT := reports/gemma4/metrics/v26_epoch_screen.json
GEMMA4_V28_CONFIG := configs/experiments/gemma4_color_mirror_post_stack_sidecar_v28.yaml
GEMMA4_V28_ROOT := data_gemma4/checkpoints/gemma4_v28_post_stack_sidecar
GEMMA4_V28_CANDIDATE := $(GEMMA4_V28_ROOT)/candidate_zero
GEMMA4_V28_UPDATE_ZERO_REPORT := reports/gemma4/metrics/v28_post_stack_update_zero_screen.json
GEMMA4_V28_STAGE_A_ROOT := $(GEMMA4_V28_ROOT)/stage_a
GEMMA4_V28_STAGE_A_SELECTION := reports/gemma4/metrics/v28_stage_a_selection.json
GEMMA4_V28_EVAL_CHECKPOINT ?=
GEMMA4_V28_EVAL_SPLIT ?= validation
GEMMA4_V28_QUESTIONS := reports/gemma4/questions/v28_$(GEMMA4_V28_EVAL_SPLIT).json
GEMMA4_V28_PREDICTIONS := reports/gemma4/predictions/v28_$(GEMMA4_V28_EVAL_SPLIT).jsonl
GEMMA4_V28_METRICS := reports/gemma4/metrics/v28_$(GEMMA4_V28_EVAL_SPLIT).json
GEMMA4_V28_STAGE_B_CONFIG := configs/experiments/gemma4_color_mirror_post_stack_decoder_stage_b_v28.yaml
GEMMA4_V28_STAGE_B_ROOT := data_gemma4/checkpoints/gemma4_v28_post_stack_decoder_stage_b
GEMMA4_V28_STAGE_B_SELECTION := reports/gemma4/metrics/v28_stage_b_selection.json
GEMMA4_V28_STAGE_B_EVAL_SPLIT ?= validation
GEMMA4_V28_STAGE_B_QUESTIONS := reports/gemma4/questions/v28_stage_b_$(GEMMA4_V28_STAGE_B_EVAL_SPLIT).json
GEMMA4_V28_STAGE_B_PREDICTIONS := reports/gemma4/predictions/v28_stage_b_$(GEMMA4_V28_STAGE_B_EVAL_SPLIT).jsonl
GEMMA4_V28_STAGE_B_METRICS := reports/gemma4/metrics/v28_stage_b_$(GEMMA4_V28_STAGE_B_EVAL_SPLIT).json
GEMMA4_V29_CONFIG := configs/experiments/gemma4_diverse20_post_stack_decoder_stage_b_v29.yaml
GEMMA4_V29_ROOT := data_gemma4/checkpoints/gemma4_v29_diverse20_post_stack_decoder_stage_b
GEMMA4_V29_SELECTION := reports/gemma4/metrics/v29_diverse_stage_b_selection.json
GEMMA4_V29_QUESTIONS := reports/gemma4/questions/v29_diverse_validation.json
GEMMA4_V29_PREDICTIONS := reports/gemma4/predictions/v29_diverse_validation.jsonl
GEMMA4_V29_METRICS := reports/gemma4/metrics/v29_diverse_validation.json
GEMMA4_V30_CONFIG := configs/experiments/gemma4_diverse20_joint_pair_v30.yaml
GEMMA4_V30_ROOT := data_gemma4/checkpoints/gemma4_v30_diverse20_joint_pair
GEMMA4_V30_SELECTION := reports/gemma4/metrics/v30_joint_pair_selection.json
GEMMA4_V31_CONFIG := configs/experiments/gemma4_diverse28_joint_pair_v31.yaml
GEMMA4_V31_ROOT := data_gemma4/checkpoints/gemma4_v31_diverse28_joint_pair
GEMMA4_V31_SELECTION := reports/gemma4/metrics/v31_joint_pair_selection.json
GEMMA4_V32_CONFIG := configs/experiments/gemma4_diverse28_microstep_v32.yaml
GEMMA4_V32_ROOT := data_gemma4/checkpoints/gemma4_v32_diverse28_microstep
GEMMA4_V32_SELECTION := reports/gemma4/metrics/v32_microstep_selection.json
GEMMA4_V33_CONFIG := configs/experiments/gemma4_diverse28_environmental_sidecar_v33.yaml
GEMMA4_V33_ROOT := data_gemma4/checkpoints/gemma4_v33_diverse28_environmental_sidecar
GEMMA4_V33_SELECTION := reports/gemma4/metrics/v33_environmental_sidecar_selection.json
GEMMA4_V33_TERMINAL_GATE := reports/gemma4/metrics/v33_update64_terminal_gate.json
GEMMA4_V34_CONFIG := configs/experiments/gemma4_diverse28_base_surface_v34.yaml
GEMMA4_V34_ROOT := data_gemma4/checkpoints/gemma4_v34_diverse28_base_surface
GEMMA4_V34_SELECTION := reports/gemma4/metrics/v34_base_surface_selection.json
GEMMA4_V34_TERMINAL_GATE := reports/gemma4/metrics/v34_update32_terminal_gate.json
GEMMA4_V35_CONFIG := configs/experiments/gemma4_diverse28_block_cross_v35.yaml
GEMMA4_V35_ROOT := data_gemma4/checkpoints/gemma4_v35_diverse28_block_cross
GEMMA4_V35_SELECTION := reports/gemma4/metrics/v35_block_cross_selection.json
GEMMA4_V35_TERMINAL_GATE := reports/gemma4/metrics/v35_update32_terminal_gate.json
GEMMA4_V36_CONFIG := configs/experiments/gemma4_diverse28_joint_block_cross_v36.yaml
GEMMA4_V36_ROOT := data_gemma4/checkpoints/gemma4_v36_diverse28_joint_block_cross
GEMMA4_V36_SELECTION := reports/gemma4/metrics/v36_joint_block_cross_selection.json
GEMMA4_V36_TERMINAL_GATE := reports/gemma4/metrics/v36_update16_terminal_gate.json
GEMMA4_FINAL_ONCE_SELECTOR_REPORT ?=
GEMMA4_FINAL_ONCE_CHECKPOINT ?=
GEMMA4_FINAL_ONCE_WORK_ROOT ?= reports/gemma4/final_once
GEMMA4_FINAL_ONCE_STOP_AFTER ?=
BLENDER ?= blender
CONFIG ?= configs/default.yaml
MCP_STDIO_SMOKE_OUTPUT ?= reports/metrics/mcp_stdio_transport.json
BATCH_CONFIG ?= configs/experiments/multiscene.yaml
DIVERSE28_CONFIG ?= configs/experiments/diverse28.yaml
DIVERSE28_NEW_SCENES := scene_000031 scene_000032 scene_000033 scene_000034 scene_000035 scene_000036 scene_000037 scene_000038
DIVERSE28_SCENE_ARGS := $(foreach scene,$(DIVERSE28_NEW_SCENES),--scene $(scene))
SCENE ?= scene_000001
CHECKPOINT ?=
GEMMA4_EMBODIED_CONFIG ?= configs/runtime/embodied_live.yaml
GEMMA4_EMBODIED_CHECKPOINT ?= data_gemma4/runtime/checkpoints/gemma4_v54_release_v1
GEMMA4_EMBODIED_CONTROL_CHECKPOINT ?= data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1
GEMMA4_EMBODIED_CONTROL_CONFIG ?= configs/runtime/gemma4_v56_question_control.yaml
GEMMA4_V78_GROUNDING_CHECKPOINT ?= data_gemma4/runtime/checkpoints/gemma4_v78_grounding_diagnostic_release_v1
GEMMA4_V81_SCENE_MEMORY ?= data_gemma4/runtime/scene_memories/v81/$(SCENE)
GEMMA4_V81_PROBE_BANK ?= reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/probe_bank
GEMMA4_V82_CONFIG ?= configs/experiments/gemma4_v82_strict_dense_learned_reader.yaml
GEMMA4_V82_TRAIN_CACHE ?= reports/gemma4/artifacts/v82_strict_dense_reader/train_cache
GEMMA4_V82_DEVELOPMENT_CACHE ?= reports/gemma4/artifacts/v82_strict_dense_reader/development_cache
GEMMA4_V82_CANDIDATE ?= reports/gemma4/artifacts/v82_strict_dense_reader/candidate
GEMMA4_V82_METRICS ?= reports/gemma4/metrics/v82_strict_dense_reader_development.json
GEMMA4_V83_SCENE_MEMORY ?= $(GEMMA4_V81_SCENE_MEMORY)
GEMMA4_V84_PAIR_MARGIN_CONFIG ?= configs/experiments/gemma4_v84_strict_fixed_memory_pair_margin.yaml
GEMMA4_V84_PAIR_MARGIN_REPORT ?= reports/gemma4/metrics/gemma4_v84_strict_bridge_pair_margin_wiring.json
GEMMA4_V85_CONFIG ?= configs/experiments/gemma4_v85_strict_multiscene.yaml
GEMMA4_V85_TRAINING_REPORT ?= reports/gemma4/metrics/gemma4_v85_strict_multiscene_training.json
GEMMA4_V85_DEVELOPMENT_SCORE ?= reports/gemma4/metrics/gemma4_v85_strict_multiscene_development.json
GEMMA4_V96_MCP_BRIDGE_HOOK ?= configs/runtime/gemma4_v96_explicit_candidate_mcp_bridge.yaml
GEMMA4_V96_MCP_BASE_CHECKPOINT ?= reports/gemma4/artifacts/v85_strict_runtime_candidate
GEMMA4_V96_MCP_SCENE_MEMORY ?= reports/gemma4/artifacts/v85_strict_runtime_candidate_memory/$(SCENE)
GEMMA4_V96_MCP_CHECK_REPORT ?= reports/gemma4/metrics/v96_explicit_candidate_embodied_mcp_check_$(SCENE).json
GEMMA4_V96_MCP_LIVE_AUDIT ?= reports/gemma4/metrics/v96_explicit_candidate_embodied_mcp_live_$(SCENE).json
GEMMA4_V96_MCP_LIVE_RESULT ?= reports/gemma4/metrics/v96_explicit_candidate_embodied_mcp_live_smoke_$(SCENE).json
GEMMA4_V96_MCP_PERSISTENT_MAP ?= data_gemma4/robot/v96_explicit_candidate/$(SCENE)/semantic_map.npz
GEMMA4_V96_MCP_SCAN_OUTPUT ?= data_gemma4/robot/v96_explicit_candidate/$(SCENE)/scans
GEMMA4_V96_ROBOT_SCENE ?= scene_000001
GEMMA4_V96_ROBOT_ASSET ?= data/runtime_assets/$(GEMMA4_V96_ROBOT_SCENE)/$(subst scene_,s_,$(GEMMA4_V96_ROBOT_SCENE)).blend
GEMMA4_V96_ROBOT_MEMORY ?= reports/gemma4/artifacts/v85_strict_runtime_candidate_memory/$(GEMMA4_V96_ROBOT_SCENE)
GEMMA4_V96_ROBOT_AUDIT ?= reports/gemma4/metrics/v96_explicit_candidate_embodied_mcp_check_$(GEMMA4_V96_ROBOT_SCENE).json
GEMMA4_V96_ROBOT_MAP ?= data_gemma4/robot/v96_explicit_candidate/$(GEMMA4_V96_ROBOT_SCENE)/semantic_map.npz
GEMMA4_V96_ROBOT_SCANS ?= data_gemma4/robot/v96_explicit_candidate/$(GEMMA4_V96_ROBOT_SCENE)/scans
GEMMA4_V96_ROBOT_EVIDENCE ?= reports/gemma4/metrics/gemma4_v96_embodied_mcp_preflight_evidence.json
GEMMA4_V96_INTEGRATION_METRICS ?= reports/gemma4/metrics/gemma4_v96_measured_integration.json
GEMMA4_V96_INTEGRATION_REPORT ?= reports/gemma4/gemma4_v96_measured_report.md
GEMMA4_V96_LIVE_ROBOT_EVIDENCE ?= reports/gemma4/metrics/v96_explicit_candidate_embodied_mcp_live_smoke_scene_000025.json
GEMMA4_V96_LIVE_INTEGRATION_METRICS ?= reports/gemma4/metrics/gemma4_v96_measured_live_integration.json
GEMMA4_V96_LIVE_INTEGRATION_REPORT ?= reports/gemma4/gemma4_v96_measured_live_report.md
GEMMA4_ROBOT_STATE_CHECKPOINT ?= data_gemma4/checkpoints/robot_state_numeric_v1
GEMMA4_NAVIGATION_POLICY_CHECKPOINT ?= data_gemma4/checkpoints/navigation_policy_v3
GEMMA4_NAVIGATION_POLICY_VERSION ?= 3
GEMMA4_LEARNED_NAVIGATION_MAX_STEPS ?= 12
GEMMA4_LEARNED_NAVIGATION_COMMAND ?= Face the chair, then stop.
GEMMA4_LEARNED_NAVIGATION_RESULT ?= reports/gemma4/metrics/embodied_conversation_learned_$(SCENE).json
RUNTIME_SCENE_ASSET ?= data/runtime_assets/$(SCENE)/$(subst scene_,s_,$(SCENE)).blend
GEMMA4_EMBODIED_SMOKE_REPORT ?= reports/gemma4/metrics/embodied_runtime_smoke_$(SCENE).json
GEMMA4_EMBODIED_MCP_CHECK_REPORT ?= reports/gemma4/metrics/embodied_mcp_check_$(SCENE).json
CONVERSATION_MCP_SMOKE_REPORT ?= reports/gemma4/metrics/conversation_mcp_stdio_smoke_$(SCENE).json
GEMMA4_CONVERSATIONAL_MCP_RESULT ?= reports/gemma4/metrics/conversational_mcp_face_$(SCENE).json
GEMMA4_CONVERSATIONAL_MCP_PREFLIGHT ?= reports/gemma4/metrics/conversational_mcp_face_preflight_$(SCENE).json
GEMMA4_CONVERSATIONAL_MCP_CLIENT_AUDIT ?= reports/gemma4/metrics/conversational_mcp_client_access_$(SCENE).json
GEMMA4_CONVERSATIONAL_MCP_SERVER_AUDIT ?= reports/gemma4/metrics/conversational_mcp_server_access_$(SCENE).json
GEMMA4_CONVERSATIONAL_MCP_SCORE ?= reports/gemma4/metrics/conversational_mcp_face_oracle_score_$(SCENE).json
GEMMA4_EMBODIED_DEMO_RESULT ?= reports/gemma4/metrics/conversational_mcp_session_$(SCENE).json
GEMMA4_EMBODIED_DEMO_PREFLIGHT ?= reports/gemma4/metrics/conversational_mcp_session_preflight_$(SCENE).json
GEMMA4_EMBODIED_DEMO_CLIENT_AUDIT ?= reports/gemma4/metrics/conversational_mcp_session_client_access_$(SCENE).json
GEMMA4_EMBODIED_DEMO_SERVER_AUDIT ?= reports/gemma4/metrics/conversational_mcp_session_server_access_$(SCENE).json
GEMMA4_EMBODIED_DEMO_SMOKE_RESULT ?= reports/gemma4/metrics/conversational_mcp_session_smoke_$(SCENE).json
GEMMA4_EMBODIED_DEMO_SMOKE_CLIENT_AUDIT ?= reports/gemma4/metrics/conversational_mcp_session_smoke_client_access_$(SCENE).json
GEMMA4_EMBODIED_DEMO_SMOKE_SERVER_AUDIT ?= reports/gemma4/metrics/conversational_mcp_session_smoke_server_access_$(SCENE).json
GEMMA4_EMBODIED_DEMO_SMOKE_INSPECTION ?= reports/gemma4/metrics/conversational_mcp_session_smoke_inspection_$(SCENE).json
GEMMA4_EMBODIED_DEMO_SMOKE_SCORE ?= reports/gemma4/metrics/conversational_mcp_session_smoke_oracle_score_$(SCENE).json
GEMMA4_EMBODIED_DEMO_SMOKE_SCORING_SPEC ?= configs/benchmarks/oracle/conversational_mcp_session_$(SCENE).json
GEMMA4_V78_EMBODIED_AUDIT_REPORT ?= reports/gemma4/metrics/v78_embodied_optional_grounding_access_$(SCENE).json
GEMMA4_SEMANTIC_NAV_REPORT ?= reports/gemma4/metrics/semantic_navigation_$(SCENE).json
GEMMA4_SEMANTIC_NAV_RUNTIME_REPORT ?= reports/gemma4/metrics/semantic_navigation_runtime_$(SCENE).json
NAVIGATION_POLICY_CONFIG ?= configs/experiments/navigation_policy_v2.yaml
NAVIGATION_POLICY_CHECKPOINT ?= data_gemma4/checkpoints/navigation_policy_v2
NAVIGATION_POLICY_RUN_ID ?= learned_v2_demo
NAVIGATION_EMBODIED_CONFIG ?= configs/runtime/embodied_navigation_v2.yaml
NAVIGATION_TASKS ?= configs/benchmarks/llm_navigation_v2_scene_000001.json
NAVIGATION_SCORING_SPEC ?= configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json
NAVIGATION_POLICY_CONTROLS_OUTPUT ?= reports/gemma4/metrics/navigation_policy_v2_controls_local.json
NAVIGATION_POLICY_AUDIT_OUTPUT ?= reports/gemma4/metrics/navigation_policy_v2_runtime_audit_local.json
NAVIGATION_CONTEXT_JOURNAL ?= reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3.json
NAVIGATION_CONTEXT_AUDIT_OUTPUT ?= reports/gemma4/metrics/navigation_continuous_context_v3.json
NAVIGATION_V3_1_PREREGISTRATION ?= reports/gemma4/metrics/navigation_policy_v3_1_runtime_preregistration.json
NAVIGATION_V3_1_RESULT ?= reports/gemma4/metrics/navigation_policy_v3_1_runtime_acceptance.json
NAVIGATION_V3_1_CONTEXT_AUDIT ?= reports/gemma4/metrics/navigation_continuous_context_v3_1.json
CURRENT_METRICS ?= reports/metrics/current_metrics.json
CURRENT_REPORT ?= reports/final_report.md
RESEARCH_DEMO_CONTROL_CHECKPOINT ?= data_gemma4/checkpoints/gemma4_v66b_allrow_always_on_control
ROVER_DEMO_HOST ?= 127.0.0.1
ROVER_DEMO_PORT ?= 8770
BLENDER_ROVER_BACKEND_TIMEOUT ?= 900

.PHONY: doctor setup setup-main legacy-setup download-models legacy-download-models download-baselines setup-gemma4-probe download-gemma4-config download-gemma4-weights gemma4-probe gemma4-probe-test extract-gemma4-scene build-gemma4-map gemma4-semantic-sanity gemma4-extract-smoke gemma4-build-smoke-map train-gemma4 gemma4-v18-preflight gemma4-v18-stage1 gemma4-v18-verify-update1 gemma4-v18-resume-screen gemma4-v18-select gemma4-v18-screen gemma4-v19-preflight gemma4-v19-stage1 gemma4-v19-verify-update1 gemma4-v19-resume-screen gemma4-v19-select gemma4-v19-screen gemma4-v19-prepare-extension gemma4-v19-run-extension gemma4-v19-select-extension gemma4-v19-extension gemma4-v20-preflight gemma4-v20-stage1 gemma4-v20-verify-update1 gemma4-v20-resume-screen gemma4-v20-select gemma4-v20-screen gemma4-v20-prepare-extension gemma4-v20-run-extension gemma4-v20-select-extension gemma4-v20-extension gemma4-v21-preflight gemma4-v21-stage1 gemma4-v21-verify-update1 gemma4-v21-resume-screen gemma4-v21-select gemma4-v21-screen gemma4-v21-prepare-extension gemma4-v21-run-extension-replay gemma4-v21-verify-extension-replay gemma4-v21-run-extension gemma4-v21-select-extension gemma4-v21-extension gemma4-v22-preflight gemma4-v22-stage1 gemma4-v22-verify-update1 gemma4-v22-resume-screen gemma4-v22-select gemma4-v22-screen gemma4-v22-prepare-extension gemma4-v22-run-extension gemma4-v22-select-extension gemma4-v22-extension gemma4-v23-preflight gemma4-v23-stage1 gemma4-v23-verify-update1 gemma4-v23-resume-screen gemma4-v23-select gemma4-v23-screen gemma4-v23-prepare-extension gemma4-v23-run-extension-replay gemma4-v23-verify-extension-replay gemma4-v23-run-extension gemma4-v23-select-extension gemma4-v23-extension gemma4-v24-preflight gemma4-v24-stage1 gemma4-v24-verify-update1 gemma4-v24-resume-screen gemma4-v24-select gemma4-v24-screen gemma4-v24-prepare-extension gemma4-v24-run-extension-replay gemma4-v24-verify-extension-replay gemma4-v24-run-extension gemma4-v24-select-extension gemma4-v24-extension gemma4-v25-preflight gemma4-v25-calibrate gemma4-v25-verify-calibration gemma4-v25-stage1 gemma4-v25-verify-update1 gemma4-v25-resume-screen gemma4-v25-select gemma4-v25-screen gemma4-v26-preflight gemma4-v26-calibrate gemma4-v26-verify-calibration gemma4-v26-stage1 gemma4-v26-verify-update1 gemma4-v26-resume-screen gemma4-v26-select gemma4-v26-screen require-gemma4-promoted chat-gemma4 gemma4-prepare-questions gemma4-predict-static gemma4-score-static gemma4-evaluate-static gemma4-predict-controls gemma4-score-controls gemma4-chat-static generate-smoke-scene render-smoke-scan generate-scene-batch render-scene-batch multiscene-dry-run build-smoke-map semantic-sanity generate-dataset train evaluate evaluate-oracle-text evaluate-direct-images chat web robot robot-evaluate mcp report demo demo-smoke demo-check demo-leakage test v75-official-validation-figures
.PHONY: legacy-generate-smoke-scene legacy-render-smoke-scan legacy-build-smoke-map legacy-semantic-sanity legacy-generate-dataset legacy-train legacy-evaluate legacy-chat legacy-web legacy-robot legacy-robot-evaluate legacy-mcp
.PHONY: gemma4-v28-build-candidate gemma4-v28-update-zero gemma4-v28-screen gemma4-v28-train-stage-a gemma4-v28-select-stage-a gemma4-v28-evaluate
.PHONY: v78-grounding-internal-held-figure v78-grounding-held-pointcloud v75-fixed-atlas-mechanism-check v85-development-figure v86-accuracy-figure v87-accuracy-figure v88-accuracy-figure
.PHONY: v75-fixed-atlas-behavior-prepare v75-fixed-atlas-behavior-preflight v75-fixed-atlas-behavior-predict v75-fixed-atlas-behavior-score v75-fixed-atlas-behavior-full v75-fixed-atlas-behavior-result
.PHONY: gemma4-v28-train-stage-b gemma4-v28-select-stage-b gemma4-v28-stage-b gemma4-v28-evaluate-stage-b
.PHONY: gemma4-v29-train-diverse-stage-b gemma4-v29-select-diverse-stage-b gemma4-v29-evaluate-diverse-validation
.PHONY: gemma4-v30-train-joint-pair gemma4-v30-select-joint-pair
.PHONY: gemma4-v31-preflight-expanded-pair gemma4-v31-train-expanded-pair gemma4-v31-select-expanded-pair
.PHONY: gemma4-v32-preflight-microstep gemma4-v32-train-microstep gemma4-v32-select-microstep
.PHONY: gemma4-v33-preflight-environmental gemma4-v33-train-environmental gemma4-v33-select-environmental
.PHONY: gemma4-v33-seal-update64 gemma4-v34-preflight-base-surface gemma4-v34-train-base-surface gemma4-v34-select-base-surface gemma4-v34-seal-update32
.PHONY: gemma4-v35-preflight-block-cross gemma4-v35-train-block-cross gemma4-v35-select-block-cross gemma4-v35-seal-update32
.PHONY: gemma4-v36-preflight-joint-block-cross gemma4-v36-train-joint-block-cross gemma4-v36-select-joint-block-cross gemma4-v36-seal-update16
.PHONY: require-gemma4-final-once-inputs gemma4-final-once-preflight gemma4-final-once
.PHONY: evaluate-direct-images-gemma4-smoke evaluate-direct-images-gemma4-validation
.PHONY: chat-question-control leakage-question-control
.PHONY: diverse28-dry-run diverse28-generate-expansion diverse28-render-expansion diverse28-generate-dataset
.PHONY: gemma4-create-final-evidence gemma4-create-chat-promotion gemma4-validate-chat-promotion require-gemma4-primary chat-gemma4 web-gemma4 demo-gemma4
.PHONY: export-runtime-scene gemma4-create-robot-state-checkpoint gemma4-embodied-smoke gemma4-semantic-navigation gemma4-embodied-chat gemma4-embodied-chat-llm gemma4-embodied-chat-learned gemma4-embodied-chat-learned-check gemma4-embodied-chat-learned-once embodied-approach-score embodied-approach-v3-score embodied-approach-v3-trajectories gemma4-embodied-mcp gemma4-embodied-mcp-check gemma4-embodied-mcp-live-smoke gemma4-embodied-mcp-conversation-check gemma4-embodied-mcp-conversation gemma4-embodied-mcp-conversation-score
.PHONY: v96-explicit-candidate-embodied-mcp-check v96-explicit-candidate-embodied-mcp
.PHONY: v96-explicit-candidate-embodied-mcp-live-smoke
.PHONY: embodied-check embodied-demo-check embodied-demo-smoke embodied-demo-smoke-inspect embodied-demo-smoke-score embodied-demo
.PHONY: navigation-policy-generate-traces navigation-policy-train navigation-policy-evaluate navigation-policy-controls navigation-policy-audit navigation-policy-context-audit navigation-policy-benchmark navigation-policy-v2-demo navigation-policy-v3-demo navigation-policy-v3-1-preregister navigation-policy-v3-1-authenticate navigation-policy-v3-1-benchmark navigation-policy-v3-1-result navigation-policy-v3-3-check navigation-policy-v4-1-result
.PHONY: current-report legacy-report demo-artifacts-check demo-artifacts-check-fast prepare-demo-runtime research-demo-check research-demo research-demo-chat research-demo-leakage
.PHONY: strict-demo-check strict-demo strict-demo-chat strict-demo-leakage v75-demo-check v75-demo v75-demo-chat v75-demo-leakage v89-demo-check v89-demo v89-demo-chat v89-demo-leakage v78-grounding-prepare v78-grounding-check v78-grounding-demo v78-grounding-chat v78-grounding-leakage legacy-demo legacy-demo-check legacy-demo-leakage
.PHONY: v78-grounding-embodied-check v78-grounding-embodied-once
.PHONY: strict-atlas-check strict-atlas-build strict-atlas-chat strict-atlas-evaluate strict-atlas-v2-auth ple-reader-prereg-auth
.PHONY: strict-web-check strict-web mcp-stdio-smoke
.PHONY: rover-demo-check rover-demo rover-demo-mcp rover-gemma-mcp-check rover-gemma-mcp blender-rover-demo-check blender-rover-demo rover-3d-check rover-3d rover-live-verify
.PHONY: lens-check lens-build lens-scan lens-perceive lens-understand lens-ask lens-ask-3d lens-locate-3d lens-drive lens-all lens-rooms lens-batch lens-train-grounding lens-ground lens-topdown lens-compare lens-phrases lens-train-points lens-point-sweep lens-point-summary lens-rope3d-locate lens-rope3d-relations
.PHONY: v15-check v15-traces v15-cache v15-train v15-sealed-score v15-heldout-plan v15-heldout-rollout v15-heldout-score v15-probe v15-summary
.PHONY: v81-reader-check v81-scene-memory-compile v81-scene-memory-check v81-scene-memory-demo v81-scene-memory-chat v81-scene-memory-leakage v81-historical-predict v81-historical-score conversation-mcp-smoke
.PHONY: v82-reader-preflight v82-reader-prepare-train v82-reader-fit v82-reader-prepare-development v82-reader-evaluate v82-chat v82-historical-predict v82-historical-score
.PHONY: v83-check v83-chat v83-historical-predict v83-historical-score
.PHONY: v84-pair-margin-preregister v84-pair-margin-train v84-pair-margin-check v84-pair-margin-result
.PHONY: v85-preregister v85-preflight v85-check v85-train v85-evaluate v85-result
.PHONY: gemma4-v67-screen gemma4-v67-full
.PHONY: gemma4-v68-preregister gemma4-v68-screen gemma4-v68-full
.PHONY: gemma4-v69-preregister gemma4-v69-screen gemma4-v69-full
.PHONY: gemma4-v70-preregister gemma4-v70-screen gemma4-v70-authenticate
.PHONY: gemma4-v71-preregister gemma4-v71-screen gemma4-v71-authenticate
.PHONY: score-multi-position-ablation

doctor:
	./scripts/doctor.sh

setup:
	bash scripts/setup.sh all

setup-main:
	bash scripts/setup.sh main

legacy-setup: setup-main

download-models: download-gemma4-weights

legacy-download-models:
	$(PYTHON) scripts/download_models.py --config $(CONFIG)

download-baselines:
	$(PYTHON) scripts/download_baseline_models.py --config $(CONFIG)

setup-gemma4-probe:
	bash scripts/setup.sh gemma4

# Metadata-only: downloads pinned config.json (~5 KB) and reads hub file sizes.
# It never downloads the 10.25 GB model.safetensors checkpoint.
download-gemma4-config:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.vision.gemma4_probe --download-config

# Explicit 10.25 GB pinned checkpoint download. All production inference is offline afterward.
download-gemma4-weights:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/download_gemma4_weights.py

# Runs fully offline after download-gemma4-config. Tiny random models execute on CPU.
gemma4-probe:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.vision.gemma4_probe

gemma4-probe-test:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest tests/test_gemma4_probe.py tests/test_gemma4_backends.py tests/test_gemma4_training.py tests/test_gemma4_semantic_sanity.py tests/test_config_paths.py

chat-question-control:
	@test -n "$(GEMMA4_CONTROL_CHECKPOINT)" || { echo "Set GEMMA4_CONTROL_CHECKPOINT to a validated two-file control checkpoint." >&2; exit 2; }
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.chat.question_control_cli \
		--config $(GEMMA4_CONTROL_CONFIG) \
		--scene $(SCENE) \
		--base-checkpoint $(GEMMA4_CONTROL_BASE_CHECKPOINT) \
		--control-checkpoint $(GEMMA4_CONTROL_CHECKPOINT)

leakage-question-control:
	@test -n "$(GEMMA4_CONTROL_CHECKPOINT)" || { echo "Set GEMMA4_CONTROL_CHECKPOINT to a validated two-file control checkpoint." >&2; exit 2; }
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.question_control_leakage \
		--config $(GEMMA4_CONTROL_CONFIG) \
		--scene $(SCENE) \
		--base-checkpoint $(GEMMA4_CONTROL_BASE_CHECKPOINT) \
		--control-checkpoint $(GEMMA4_CONTROL_CHECKPOINT) \
		--teacher-artifact $(GEMMA4_CONTROL_TEACHER_ARTIFACT) \
		--output reports/gemma4/metrics/question_control_leakage.json

# Primary Gemma feature/mapping path: reads shared sanitized renders and writes
# derived artifacts only below data_gemma4.
extract-gemma4-scene:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.vision.encoder --config $(GEMMA4_CONFIG) --scene $(SCENE) --offline

build-gemma4-map:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/build_map.py --config $(GEMMA4_CONFIG) --scene $(SCENE)

# Evaluation only: mean bare-category token embeddings are compared with the
# final 1536D native projected visual stream. No full Gemma model is loaded.
gemma4-semantic-sanity:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.gemma4_semantic_sanity --config $(GEMMA4_CONFIG) --scene $(SCENE) --offline

gemma4-extract-smoke: extract-gemma4-scene

gemma4-build-smoke-map: build-gemma4-map

train-gemma4:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_CONFIG)

# V18 is intentionally staged. The preflight always reruns against the current
# clean commit; completed checkpoint stages are reused and then reverified.
gemma4-v18-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v18_structural_preflight --config $(GEMMA4_V18_CONFIG) --report $(GEMMA4_V18_PREFLIGHT)

gemma4-v18-stage1: gemma4-v18-preflight
	@if [ -f "$(GEMMA4_V18_CHECKPOINT_ROOT)/epoch_001/metadata.json" ]; then \
		echo "Reusing cached V18 epoch_001; the verifier will bind it to the fresh preflight."; \
	elif [ -e "$(GEMMA4_V18_CHECKPOINT_ROOT)" ]; then \
		echo "Incomplete V18 checkpoint root exists without epoch_001 metadata: $(GEMMA4_V18_CHECKPOINT_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V18_CONFIG) --epochs 1; \
	fi

gemma4-v18-verify-update1: gemma4-v18-stage1
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v18_update1_verifier --config $(GEMMA4_V18_CONFIG) --preflight $(GEMMA4_V18_PREFLIGHT) --checkpoint $(GEMMA4_V18_CHECKPOINT_ROOT)/epoch_001 --report $(GEMMA4_V18_UPDATE1_REPORT)

gemma4-v18-resume-screen: gemma4-v18-verify-update1
	@if [ -f "$(GEMMA4_V18_CHECKPOINT_ROOT)/epoch_004/metadata.json" ]; then \
		echo "Reusing cached V18 epoch_004; strict selection will validate cumulative history."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V18_CONFIG) --resume $(GEMMA4_V18_CHECKPOINT_ROOT)/epoch_001 --epochs 4; \
	fi

gemma4-v18-select: gemma4-v18-resume-screen
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v18_epoch_selector --config $(GEMMA4_V18_CONFIG) --selection $(GEMMA4_V18_SELECTION) --epoch 1=$(GEMMA4_V18_CHECKPOINT_ROOT)/epoch_001/metadata.json --epoch 2=$(GEMMA4_V18_CHECKPOINT_ROOT)/epoch_002/metadata.json --epoch 3=$(GEMMA4_V18_CHECKPOINT_ROOT)/epoch_003/metadata.json --epoch 4=$(GEMMA4_V18_CHECKPOINT_ROOT)/epoch_004/metadata.json --output $(GEMMA4_V18_SCREEN_REPORT)

gemma4-v18-screen: gemma4-v18-select

# V19 freezes the complete V18 epoch-4 base and stages one exact update of the
# reflection-odd signed-X output matrix before allowing updates 2--4.
gemma4-v19-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v19_structural_preflight --config $(GEMMA4_V19_CONFIG) --report $(GEMMA4_V19_PREFLIGHT)

gemma4-v19-stage1: gemma4-v19-preflight
	@if [ -f "$(GEMMA4_V19_CHECKPOINT_ROOT)/epoch_001/metadata.json" ]; then \
		echo "Reusing cached V19 epoch_001; the verifier will bind it to the fresh preflight."; \
	elif [ -e "$(GEMMA4_V19_CHECKPOINT_ROOT)" ]; then \
		echo "Incomplete V19 checkpoint root exists without epoch_001 metadata: $(GEMMA4_V19_CHECKPOINT_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V19_CONFIG) --epochs 1; \
	fi

gemma4-v19-verify-update1: gemma4-v19-stage1
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v19_update1_verifier --config $(GEMMA4_V19_CONFIG) --preflight $(GEMMA4_V19_PREFLIGHT) --checkpoint $(GEMMA4_V19_CHECKPOINT_ROOT)/epoch_001 --report $(GEMMA4_V19_UPDATE1_REPORT)

gemma4-v19-resume-screen: gemma4-v19-verify-update1
	@if [ -f "$(GEMMA4_V19_CHECKPOINT_ROOT)/epoch_004/metadata.json" ]; then \
		echo "Reusing cached V19 epoch_004; strict selection will validate cumulative history."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V19_CONFIG) --resume $(GEMMA4_V19_CHECKPOINT_ROOT)/epoch_001 --epochs 4; \
	fi

gemma4-v19-select: gemma4-v19-resume-screen
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v19_epoch_selector --config $(GEMMA4_V19_CONFIG) --selection $(GEMMA4_V19_SELECTION) --epoch 1=$(GEMMA4_V19_CHECKPOINT_ROOT)/epoch_001/metadata.json --epoch 2=$(GEMMA4_V19_CHECKPOINT_ROOT)/epoch_002/metadata.json --epoch 3=$(GEMMA4_V19_CHECKPOINT_ROOT)/epoch_003/metadata.json --epoch 4=$(GEMMA4_V19_CHECKPOINT_ROOT)/epoch_004/metadata.json --output $(GEMMA4_V19_SCREEN_REPORT)

gemma4-v19-screen: gemma4-v19-select

# This branch is reachable only when the strict four-update selector authorizes
# continuation but still forbids greedy generation. It resumes the selected
# checkpoint into an isolated namespace so the original screen is never overwritten.
gemma4-v19-prepare-extension: gemma4-v19-select
	@if [ -e "$(GEMMA4_V19_EXTENSION_ROOT)" ]; then \
		test -f "$(GEMMA4_V19_EXTENSION_MANIFEST)" || { echo "V19 extension root exists without its authorization manifest." >&2; exit 2; }; \
		echo "Reusing cached V19 extension authorization; final selection will revalidate every hash."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v19_extension_controller prepare --config $(GEMMA4_V19_CONFIG) --screen $(GEMMA4_V19_SCREEN_REPORT) --output $(GEMMA4_V19_EXTENSION_MANIFEST); \
	fi

gemma4-v19-run-extension: gemma4-v19-prepare-extension
	@if [ -f "$(GEMMA4_V19_EXTENSION_ROOT)/epoch_012/metadata.json" ]; then \
		echo "Reusing cached V19 update-12 extension; strict final selection will validate it."; \
	elif [ -e "$(GEMMA4_V19_EXTENSION_ROOT)" ]; then \
		echo "Incomplete V19 extension root exists; refusing to overwrite it: $(GEMMA4_V19_EXTENSION_ROOT)" >&2; \
		exit 2; \
	else \
		resume_checkpoint="$$(PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"])' "$(GEMMA4_V19_EXTENSION_MANIFEST)")"; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V19_CONFIG) --resume "$$resume_checkpoint" --output-namespace $(GEMMA4_V19_EXTENSION_NAMESPACE) --epochs 12; \
	fi

gemma4-v19-select-extension: gemma4-v19-run-extension
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v19_extension_controller select-final --manifest $(GEMMA4_V19_EXTENSION_MANIFEST) --output $(GEMMA4_V19_EXTENSION_REPORT)

gemma4-v19-extension: gemma4-v19-select-extension

# V20 is a fresh architecture-only restart from the exact V18 epoch-4 source.
# It preserves each slot's local signed-X content instead of reducing all slots
# to V19's single moment. Unsafe optimizer stages are cached but always rebound
# to fresh report-only preflight and update-one evidence before any resume.
gemma4-v20-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v20_structural_preflight --config $(GEMMA4_V20_CONFIG) --report $(GEMMA4_V20_PREFLIGHT)

gemma4-v20-stage1: gemma4-v20-preflight
	@if [ -f "$(GEMMA4_V20_CHECKPOINT_ROOT)/epoch_001/metadata.json" ]; then \
		echo "Reusing cached V20 epoch_001; the verifier will bind it to the fresh preflight."; \
	elif [ -e "$(GEMMA4_V20_CHECKPOINT_ROOT)" ]; then \
		echo "Incomplete V20 checkpoint root exists without epoch_001 metadata: $(GEMMA4_V20_CHECKPOINT_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V20_CONFIG) --epochs 1; \
	fi

gemma4-v20-verify-update1: gemma4-v20-stage1
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v20_update1_verifier --config $(GEMMA4_V20_CONFIG) --preflight $(GEMMA4_V20_PREFLIGHT) --checkpoint $(GEMMA4_V20_CHECKPOINT_ROOT)/epoch_001 --report $(GEMMA4_V20_UPDATE1_REPORT)

gemma4-v20-resume-screen: gemma4-v20-verify-update1
	@if [ -f "$(GEMMA4_V20_CHECKPOINT_ROOT)/epoch_004/metadata.json" ]; then \
		echo "Reusing cached V20 epoch_004; strict selection will validate cumulative history."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V20_CONFIG) --resume $(GEMMA4_V20_CHECKPOINT_ROOT)/epoch_001 --epochs 4; \
	fi

gemma4-v20-select: gemma4-v20-resume-screen
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v20_epoch_selector --config $(GEMMA4_V20_CONFIG) --selection $(GEMMA4_V20_SELECTION) --update1-report $(GEMMA4_V20_UPDATE1_REPORT) --epoch 1=$(GEMMA4_V20_CHECKPOINT_ROOT)/epoch_001/metadata.json --epoch 2=$(GEMMA4_V20_CHECKPOINT_ROOT)/epoch_002/metadata.json --epoch 3=$(GEMMA4_V20_CHECKPOINT_ROOT)/epoch_003/metadata.json --epoch 4=$(GEMMA4_V20_CHECKPOINT_ROOT)/epoch_004/metadata.json --output $(GEMMA4_V20_SCREEN_REPORT)

gemma4-v20-screen: gemma4-v20-select

# The isolated update-8 branch is reachable only after the strict four-update
# selector authorizes continuation while still denying greedy generation.
gemma4-v20-prepare-extension: gemma4-v20-select
	@if [ -e "$(GEMMA4_V20_EXTENSION_ROOT)" ]; then \
		test -f "$(GEMMA4_V20_EXTENSION_MANIFEST)" || { echo "V20 extension root exists without its authorization manifest." >&2; exit 2; }; \
		echo "Reusing cached V20 extension authorization; final selection will revalidate every hash."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v20_extension_controller prepare --config $(GEMMA4_V20_CONFIG) --screen $(GEMMA4_V20_SCREEN_REPORT) --output $(GEMMA4_V20_EXTENSION_MANIFEST); \
	fi

gemma4-v20-run-extension: gemma4-v20-prepare-extension
	@if [ -f "$(GEMMA4_V20_EXTENSION_ROOT)/epoch_008/metadata.json" ]; then \
		echo "Reusing cached V20 update-8 extension; strict final selection will validate it."; \
	elif [ -e "$(GEMMA4_V20_EXTENSION_ROOT)" ]; then \
		echo "Incomplete V20 extension root exists; refusing to overwrite it: $(GEMMA4_V20_EXTENSION_ROOT)" >&2; \
		exit 2; \
	else \
		resume_checkpoint="$$(PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"])' "$(GEMMA4_V20_EXTENSION_MANIFEST)")"; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V20_CONFIG) --resume "$$resume_checkpoint" --output-namespace $(GEMMA4_V20_EXTENSION_NAMESPACE) --epochs 8; \
	fi

gemma4-v20-select-extension: gemma4-v20-run-extension
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v20_extension_controller select-final --manifest $(GEMMA4_V20_EXTENSION_MANIFEST) --output $(GEMMA4_V20_EXTENSION_REPORT)

gemma4-v20-extension: gemma4-v20-select-extension

# V21 keeps V20's local field, exact training schedule, and stable BF16 Gemma
# path. Its preflight replaces the confounded total-norm eligibility statistic
# with phase-aware precision and exact predicted-functional gates.
gemma4-v21-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v21_structural_preflight --config $(GEMMA4_V21_CONFIG) --report $(GEMMA4_V21_PREFLIGHT)

gemma4-v21-stage1: gemma4-v21-preflight
	@if [ -f "$(GEMMA4_V21_CHECKPOINT_ROOT)/epoch_001/metadata.json" ]; then \
		echo "Reusing cached V21 epoch_001; the verifier will bind it to the fresh preflight."; \
	elif [ -e "$(GEMMA4_V21_CHECKPOINT_ROOT)" ]; then \
		echo "Incomplete V21 checkpoint root exists without epoch_001 metadata: $(GEMMA4_V21_CHECKPOINT_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V21_CONFIG) --epochs 1; \
	fi

gemma4-v21-verify-update1: gemma4-v21-stage1
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v21_update1_verifier --config $(GEMMA4_V21_CONFIG) --preflight $(GEMMA4_V21_PREFLIGHT) --checkpoint $(GEMMA4_V21_CHECKPOINT_ROOT)/epoch_001 --output $(GEMMA4_V21_UPDATE1_REPORT)

gemma4-v21-resume-screen: gemma4-v21-verify-update1
	@if [ -f "$(GEMMA4_V21_CHECKPOINT_ROOT)/epoch_004/metadata.json" ]; then \
		echo "Reusing cached V21 epoch_004; strict selection will validate cumulative history."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V21_CONFIG) --resume $(GEMMA4_V21_CHECKPOINT_ROOT)/epoch_001 --epochs 4; \
	fi

gemma4-v21-select: gemma4-v21-resume-screen
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v21_epoch_selector --config $(GEMMA4_V21_CONFIG) --selection $(GEMMA4_V21_SELECTION) --update1-report $(GEMMA4_V21_UPDATE1_REPORT) --epoch 1=$(GEMMA4_V21_CHECKPOINT_ROOT)/epoch_001/metadata.json --epoch 2=$(GEMMA4_V21_CHECKPOINT_ROOT)/epoch_002/metadata.json --epoch 3=$(GEMMA4_V21_CHECKPOINT_ROOT)/epoch_003/metadata.json --epoch 4=$(GEMMA4_V21_CHECKPOINT_ROOT)/epoch_004/metadata.json --output $(GEMMA4_V21_SCREEN_REPORT)

gemma4-v21-screen: gemma4-v21-select

# Updates 5-8 are a conditional, isolated continuation. Preparation succeeds
# only when the exact four-update selector chose a color-eligible checkpoint
# with >=8/12 mirror sides and >=2/6 mirror units but not the full-teacher gate.
gemma4-v21-prepare-extension: gemma4-v21-select
	@if [ -e "$(GEMMA4_V21_EXTENSION_ROOT)" ]; then \
		test -f "$(GEMMA4_V21_EXTENSION_MANIFEST)" || { echo "V21 extension root exists without its authorization manifest." >&2; exit 2; }; \
		echo "Reusing cached V21 extension authorization; final selection will revalidate every hash."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v21_extension_controller prepare --config $(GEMMA4_V21_CONFIG) --screen $(GEMMA4_V21_SCREEN_REPORT) --output $(GEMMA4_V21_EXTENSION_MANIFEST); \
	fi

gemma4-v21-run-extension: gemma4-v21-prepare-extension
	@if [ -f "$(GEMMA4_V21_EXTENSION_ROOT)/epoch_008/metadata.json" ]; then \
		echo "Reusing cached V21 update-8 extension; strict final selection will validate it."; \
	elif [ -e "$(GEMMA4_V21_EXTENSION_ROOT)" ]; then \
		echo "Incomplete V21 extension root exists; refusing to overwrite it: $(GEMMA4_V21_EXTENSION_ROOT)" >&2; \
		exit 2; \
	else \
		resume_checkpoint="$$(PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"])' "$(GEMMA4_V21_EXTENSION_MANIFEST)")"; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V21_CONFIG) --resume "$$resume_checkpoint" --output-namespace $(GEMMA4_V21_EXTENSION_NAMESPACE) --epochs 8; \
	fi

gemma4-v21-select-extension: gemma4-v21-run-extension
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v21_extension_controller select-final --manifest $(GEMMA4_V21_EXTENSION_MANIFEST) --output $(GEMMA4_V21_EXTENSION_REPORT)

gemma4-v21-extension: gemma4-v21-select-extension

# V22 is an exact V21 restart with only pair_000003's candidate/full-vocabulary
# targets rebalanced to 0.25. Its evidence and checkpoint namespaces are fully
# isolated; no V21 authorization artifact is accepted as a V22 authorization.
gemma4-v22-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v22_structural_preflight --config $(GEMMA4_V22_CONFIG) --report $(GEMMA4_V22_PREFLIGHT)

gemma4-v22-stage1: gemma4-v22-preflight
	@if [ -f "$(GEMMA4_V22_CHECKPOINT_ROOT)/epoch_001/metadata.json" ]; then \
		echo "Reusing cached V22 epoch_001; the verifier will bind it to the fresh preflight."; \
	elif [ -e "$(GEMMA4_V22_CHECKPOINT_ROOT)" ]; then \
		echo "Incomplete V22 checkpoint root exists without epoch_001 metadata: $(GEMMA4_V22_CHECKPOINT_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V22_CONFIG) --epochs 1; \
	fi

gemma4-v22-verify-update1: gemma4-v22-stage1
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v22_update1_verifier --config $(GEMMA4_V22_CONFIG) --preflight $(GEMMA4_V22_PREFLIGHT) --checkpoint $(GEMMA4_V22_CHECKPOINT_ROOT)/epoch_001 --output $(GEMMA4_V22_UPDATE1_REPORT)

gemma4-v22-resume-screen: gemma4-v22-verify-update1
	@if [ -f "$(GEMMA4_V22_CHECKPOINT_ROOT)/epoch_004/metadata.json" ]; then \
		echo "Reusing cached V22 epoch_004; strict selection will validate cumulative history."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V22_CONFIG) --resume $(GEMMA4_V22_CHECKPOINT_ROOT)/epoch_001 --epochs 4; \
	fi

gemma4-v22-select: gemma4-v22-resume-screen
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v22_epoch_selector --config $(GEMMA4_V22_CONFIG) --selection $(GEMMA4_V22_SELECTION) --update1-report $(GEMMA4_V22_UPDATE1_REPORT) --epoch 1=$(GEMMA4_V22_CHECKPOINT_ROOT)/epoch_001/metadata.json --epoch 2=$(GEMMA4_V22_CHECKPOINT_ROOT)/epoch_002/metadata.json --epoch 3=$(GEMMA4_V22_CHECKPOINT_ROOT)/epoch_003/metadata.json --epoch 4=$(GEMMA4_V22_CHECKPOINT_ROOT)/epoch_004/metadata.json --output $(GEMMA4_V22_SCREEN_REPORT)

gemma4-v22-screen: gemma4-v22-select

gemma4-v22-prepare-extension: gemma4-v22-select
	@if [ -e "$(GEMMA4_V22_EXTENSION_ROOT)" ]; then \
		test -f "$(GEMMA4_V22_EXTENSION_MANIFEST)" || { echo "V22 extension root exists without its authorization manifest." >&2; exit 2; }; \
		echo "Reusing cached V22 extension authorization; final selection will revalidate every hash."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v22_extension_controller prepare --config $(GEMMA4_V22_CONFIG) --screen $(GEMMA4_V22_SCREEN_REPORT) --output $(GEMMA4_V22_EXTENSION_MANIFEST); \
	fi

gemma4-v22-run-extension: gemma4-v22-prepare-extension
	@if [ -f "$(GEMMA4_V22_EXTENSION_ROOT)/epoch_008/metadata.json" ]; then \
		echo "Reusing cached V22 update-8 extension; strict final selection will validate it."; \
	elif [ -e "$(GEMMA4_V22_EXTENSION_ROOT)" ]; then \
		echo "Incomplete V22 extension root exists; refusing to overwrite it: $(GEMMA4_V22_EXTENSION_ROOT)" >&2; \
		exit 2; \
	else \
		resume_checkpoint="$$(PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"])' "$(GEMMA4_V22_EXTENSION_MANIFEST)")"; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V22_CONFIG) --resume "$$resume_checkpoint" --output-namespace $(GEMMA4_V22_EXTENSION_NAMESPACE) --epochs 8; \
	fi

gemma4-v22-select-extension: gemma4-v22-run-extension
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v22_extension_controller select-final --manifest $(GEMMA4_V22_EXTENSION_MANIFEST) --output $(GEMMA4_V22_EXTENSION_REPORT)

gemma4-v22-extension: gemma4-v22-select-extension

# V23 loads the sealed V21 epoch-8 scene/residual stack and older decoder
# banks as frozen weights, then trains only one zero-output rank-4 shared-K/V
# bank (30,720 FP32 parameters). A report-only verifier must authorize stage 2.
gemma4-v23-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v23_shared_kv_controller preflight --config $(GEMMA4_V23_CONFIG) --output $(GEMMA4_V23_PREFLIGHT)

gemma4-v23-stage1: gemma4-v23-preflight
	@if [ -f "$(GEMMA4_V23_CHECKPOINT_ROOT)/epoch_001/metadata.json" ]; then \
		echo "Reusing cached V23 epoch_001; the verifier will bind it to the fresh preflight."; \
	elif [ -e "$(GEMMA4_V23_CHECKPOINT_ROOT)" ]; then \
		echo "Incomplete V23 checkpoint root exists without epoch_001 metadata: $(GEMMA4_V23_CHECKPOINT_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V23_CONFIG) --epochs 1; \
	fi

gemma4-v23-verify-update1: gemma4-v23-stage1
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v23_shared_kv_controller verify-update1 --config $(GEMMA4_V23_CONFIG) --preflight $(GEMMA4_V23_PREFLIGHT) --checkpoint $(GEMMA4_V23_CHECKPOINT_ROOT)/epoch_001 --output $(GEMMA4_V23_UPDATE1_REPORT)

gemma4-v23-resume-screen: gemma4-v23-verify-update1
	@PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if report.get("stage_2_authorized") is True else 2)' "$(GEMMA4_V23_UPDATE1_REPORT)" || { echo "V23 update 1 did not retain the preregistered color/mirror stage-2 gate; refusing updates 2--4." >&2; exit 2; }
	@if [ -f "$(GEMMA4_V23_CHECKPOINT_ROOT)/epoch_004/metadata.json" ]; then \
		echo "Reusing cached V23 epoch_004; strict selection will validate all four updates."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V23_CONFIG) --resume $(GEMMA4_V23_CHECKPOINT_ROOT)/epoch_001 --epochs 4; \
	fi

gemma4-v23-select: gemma4-v23-resume-screen
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v23_shared_kv_controller select --config $(GEMMA4_V23_CONFIG) --update1-report $(GEMMA4_V23_UPDATE1_REPORT) --epoch 1=$(GEMMA4_V23_CHECKPOINT_ROOT)/epoch_001/metadata.json --epoch 2=$(GEMMA4_V23_CHECKPOINT_ROOT)/epoch_002/metadata.json --epoch 3=$(GEMMA4_V23_CHECKPOINT_ROOT)/epoch_003/metadata.json --epoch 4=$(GEMMA4_V23_CHECKPOINT_ROOT)/epoch_004/metadata.json --output $(GEMMA4_V23_SCREEN_REPORT)

gemma4-v23-screen: gemma4-v23-select

# V24 freezes the sealed V23 epoch-2 stack and trains only one deterministic
# rank-4 query bank on physical layers 28/29 (36,864 FP32 parameters). The
# report-only update-1 verifier must authorize the remaining bounded screen.
gemma4-v24-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v24_shared_query_controller preflight --config $(GEMMA4_V24_CONFIG) --output $(GEMMA4_V24_PREFLIGHT)

gemma4-v24-stage1: gemma4-v24-preflight
	@if [ -f "$(GEMMA4_V24_CHECKPOINT_ROOT)/epoch_001/metadata.json" ]; then \
		echo "Reusing cached V24 epoch_001; the verifier will bind it to the fresh preflight."; \
	elif [ -e "$(GEMMA4_V24_CHECKPOINT_ROOT)" ]; then \
		echo "Incomplete V24 checkpoint root exists without epoch_001 metadata: $(GEMMA4_V24_CHECKPOINT_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V24_CONFIG) --epochs 1; \
	fi

gemma4-v24-verify-update1: gemma4-v24-stage1
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v24_shared_query_controller verify-update1 --config $(GEMMA4_V24_CONFIG) --preflight $(GEMMA4_V24_PREFLIGHT) --checkpoint $(GEMMA4_V24_CHECKPOINT_ROOT)/epoch_001 --output $(GEMMA4_V24_UPDATE1_REPORT)

gemma4-v24-resume-screen: gemma4-v24-verify-update1
	@PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if report.get("stage_2_authorized") is True else 2)' "$(GEMMA4_V24_UPDATE1_REPORT)" || { echo "V24 update 1 did not retain the preregistered color/mirror stage-2 gate; refusing updates 2--4." >&2; exit 2; }
	@if [ -f "$(GEMMA4_V24_CHECKPOINT_ROOT)/epoch_004/metadata.json" ]; then \
		echo "Reusing cached V24 epoch_004; strict selection will validate all four updates."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V24_CONFIG) --resume $(GEMMA4_V24_CHECKPOINT_ROOT)/epoch_001 --epochs 4; \
	fi

gemma4-v24-select: gemma4-v24-resume-screen
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v24_shared_query_controller select --config $(GEMMA4_V24_CONFIG) --update1-report $(GEMMA4_V24_UPDATE1_REPORT) --epoch 1=$(GEMMA4_V24_CHECKPOINT_ROOT)/epoch_001/metadata.json --epoch 2=$(GEMMA4_V24_CHECKPOINT_ROOT)/epoch_002/metadata.json --epoch 3=$(GEMMA4_V24_CHECKPOINT_ROOT)/epoch_003/metadata.json --epoch 4=$(GEMMA4_V24_CHECKPOINT_ROOT)/epoch_004/metadata.json --output $(GEMMA4_V24_SCREEN_REPORT)

gemma4-v24-screen: gemma4-v24-select

# V25 adds one all-voxel rank-eight visual-to-language bridge to the sealed
# V24 epoch-1 stack. Preflight authorizes only bounded semantic calibration;
# update-1 verification must independently authorize paired QA updates 2--4.
gemma4-v25-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v25_dense_alignment_controller preflight --config $(GEMMA4_V25_CONFIG) --output $(GEMMA4_V25_PREFLIGHT)

# Standalone reproducibility probe. The trainer repeats the same fail-closed
# calibration internally before QA and does not consume this report/bridge.
gemma4-v25-calibrate: gemma4-v25-preflight
	@set +e; \
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.dense_alignment_calibration --config $(GEMMA4_V25_CONFIG) --bridge-output $(GEMMA4_V25_CALIBRATION_BRIDGE) --report-output $(GEMMA4_V25_CALIBRATION_REPORT); \
	status=$$?; \
	set -e; \
	if [ "$$status" -ne 0 ] && [ "$$status" -ne 2 ]; then exit "$$status"; fi; \
	test -f "$(GEMMA4_V25_CALIBRATION_REPORT)" || { echo "V25 calibration produced no decision report." >&2; exit 2; }

gemma4-v25-verify-calibration: gemma4-v25-calibrate
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v25_dense_alignment_controller verify-calibration --config $(GEMMA4_V25_CONFIG) --calibration $(GEMMA4_V25_CALIBRATION_REPORT) --output $(GEMMA4_V25_CALIBRATION_DECISION)

gemma4-v25-stage1: gemma4-v25-verify-calibration
	@PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if report.get("paired_qa_stage_authorized") is True else 2)' "$(GEMMA4_V25_CALIBRATION_DECISION)" || { echo "V25 calibration is deterministically denied at the pinned 20-step limit; refusing to launch paired-QA training." >&2; exit 2; }
	@if [ -f "$(GEMMA4_V25_CHECKPOINT_ROOT)/epoch_001/metadata.json" ]; then \
		echo "Reusing cached V25 epoch_001; the verifier will bind it to the fresh preflight."; \
	elif [ -e "$(GEMMA4_V25_CHECKPOINT_ROOT)" ]; then \
		echo "Incomplete V25 checkpoint root exists without epoch_001 metadata: $(GEMMA4_V25_CHECKPOINT_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V25_CONFIG) --epochs 1; \
	fi

gemma4-v25-verify-update1: gemma4-v25-stage1
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v25_dense_alignment_controller verify-update1 --config $(GEMMA4_V25_CONFIG) --preflight $(GEMMA4_V25_PREFLIGHT) --checkpoint $(GEMMA4_V25_CHECKPOINT_ROOT)/epoch_001 --output $(GEMMA4_V25_UPDATE1_REPORT)

gemma4-v25-resume-screen: gemma4-v25-verify-update1
	@PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if report.get("stage_2_authorized") is True else 2)' "$(GEMMA4_V25_UPDATE1_REPORT)" || { echo "V25 update 1 did not pass calibration/localization and the preregistered teacher-forced stage-2 gate; refusing updates 2--4." >&2; exit 2; }
	@if [ -f "$(GEMMA4_V25_CHECKPOINT_ROOT)/epoch_004/metadata.json" ]; then \
		echo "Reusing cached V25 epoch_004; strict selection will validate all four updates."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V25_CONFIG) --resume $(GEMMA4_V25_CHECKPOINT_ROOT)/epoch_001 --epochs 4; \
	fi

gemma4-v25-select: gemma4-v25-resume-screen
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v25_dense_alignment_controller select --config $(GEMMA4_V25_CONFIG) --update1-report $(GEMMA4_V25_UPDATE1_REPORT) --epoch 1=$(GEMMA4_V25_CHECKPOINT_ROOT)/epoch_001/metadata.json --epoch 2=$(GEMMA4_V25_CHECKPOINT_ROOT)/epoch_002/metadata.json --epoch 3=$(GEMMA4_V25_CHECKPOINT_ROOT)/epoch_003/metadata.json --epoch 4=$(GEMMA4_V25_CHECKPOINT_ROOT)/epoch_004/metadata.json --output $(GEMMA4_V25_SCREEN_REPORT)

gemma4-v25-screen: gemma4-v25-select

# V26 corrects V25's calibration split: gradients use only scenes 1/2/9/10,
# scenes 7/8 are semantic validation, and scenes 3/4 plus final QA test scenes
# 5/6 are rejected by the map/oracle loader recorder.  No paired-QA optimizer
# may run until the exact selected calibration report and tensor-only bridge
# pass the controller's byte-level verification.
gemma4-v26-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v26_dense_alignment_controller preflight --config $(GEMMA4_V26_CONFIG) --output $(GEMMA4_V26_PREFLIGHT)

gemma4-v26-calibrate: gemma4-v26-preflight
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.dense_alignment_calibration --config $(GEMMA4_V26_CONFIG) --bridge-output $(GEMMA4_V26_CALIBRATION_BRIDGE) --report-output $(GEMMA4_V26_CALIBRATION_REPORT)

gemma4-v26-verify-calibration: gemma4-v26-calibrate
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v26_dense_alignment_controller verify-calibration --config $(GEMMA4_V26_CONFIG) --preflight $(GEMMA4_V26_PREFLIGHT) --calibration $(GEMMA4_V26_CALIBRATION_REPORT) --bridge $(GEMMA4_V26_CALIBRATION_BRIDGE) --output $(GEMMA4_V26_CALIBRATION_DECISION)

gemma4-v26-stage1: gemma4-v26-verify-calibration
	@PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if report.get("paired_qa_stage_authorized") is True and report.get("final_qa_test_untouched") is True else 2)' "$(GEMMA4_V26_CALIBRATION_DECISION)" || { echo "V26 exact calibration/access gate did not authorize paired-QA update 1." >&2; exit 2; }
	@if [ -f "$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_001/metadata.json" ] && [ -f "$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_001/runtime_metadata.json" ] && [ -f "$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_001/adapter.safetensors" ] && [ -f "$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_001/optimizer.pt" ]; then \
		echo "Reusing complete cached V26 epoch_001; strict verification will audit metadata, adapter, and optimizer hashes."; \
	elif [ -e "$(GEMMA4_V26_CHECKPOINT_ROOT)" ]; then \
		echo "Incomplete V26 checkpoint root exists without all epoch_001 artifacts: $(GEMMA4_V26_CHECKPOINT_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V26_CONFIG) --epochs 1; \
	fi

gemma4-v26-verify-update1: gemma4-v26-stage1
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v26_dense_alignment_controller verify-update1 --config $(GEMMA4_V26_CONFIG) --preflight $(GEMMA4_V26_PREFLIGHT) --calibration-decision $(GEMMA4_V26_CALIBRATION_DECISION) --checkpoint $(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_001 --output $(GEMMA4_V26_UPDATE1_REPORT)

# Updates 2--4 remain inaccessible until update 1 passes calibration provenance,
# teacher forcing, frozen-state, adapter-state, and sanitized runtime-sidecar gates.
gemma4-v26-resume-screen: gemma4-v26-verify-update1
	@PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if report.get("stage_2_authorized") is True and report.get("final_qa_test_untouched") is True else 2)' "$(GEMMA4_V26_UPDATE1_REPORT)" || { echo "V26 update 1 did not authorize updates 2--4." >&2; exit 2; }
	@if [ -f "$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_004/metadata.json" ] && [ -f "$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_004/runtime_metadata.json" ] && [ -f "$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_004/adapter.safetensors" ] && [ -f "$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_004/optimizer.pt" ]; then \
		echo "Reusing cached V26 epoch_004; selector will audit every checkpoint artifact."; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V26_CONFIG) --resume $(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_001 --epochs 4; \
	fi

gemma4-v26-select: gemma4-v26-resume-screen
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v26_dense_alignment_controller select --config $(GEMMA4_V26_CONFIG) --update1-report $(GEMMA4_V26_UPDATE1_REPORT) --epoch 1=$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_001 --epoch 2=$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_002 --epoch 3=$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_003 --epoch 4=$(GEMMA4_V26_CHECKPOINT_ROOT)/epoch_004 --output $(GEMMA4_V26_SCREEN_REPORT)

gemma4-v26-screen: gemma4-v26-select

# V28 installs the frozen calibrated V26 all-voxel stream beside the exact V24
# scene stack, then trains only a zero-output post-stack adapter. The candidate
# and every optimizer stage are isolated and refuse partial-directory reuse.
gemma4-v28-build-candidate:
	@if [ -f "$(GEMMA4_V28_CANDIDATE)/adapter.safetensors" ] && [ -f "$(GEMMA4_V28_CANDIDATE)/metadata.json" ] && [ -f "$(GEMMA4_V28_CANDIDATE)/runtime_metadata.json" ]; then \
		echo "Reusing complete hash-bound V28 update-zero candidate."; \
	elif [ -e "$(GEMMA4_V28_CANDIDATE)" ]; then \
		echo "Incomplete V28 candidate exists; refusing to overwrite it: $(GEMMA4_V28_CANDIDATE)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) scripts/build_post_stack_sidecar_candidate.py --config "$(GEMMA4_V28_CONFIG)" --base-checkpoint data_gemma4/checkpoints/gemma4_v24_shared_query/epoch_001 --bridge reports/gemma4/artifacts/v26_dense_alignment_bridge.safetensors --output "$(GEMMA4_V28_CANDIDATE)"; \
	fi

gemma4-v28-update-zero: gemma4-v28-build-candidate
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v28_post_stack_screen --config "$(GEMMA4_V28_CONFIG)" --checkpoint "$(GEMMA4_V28_CANDIDATE)" --output "$(GEMMA4_V28_UPDATE_ZERO_REPORT)"

gemma4-v28-screen: gemma4-v28-update-zero

gemma4-v28-train-stage-a: gemma4-v28-update-zero
	@PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if report.get("passed") is True else 2)' "$(GEMMA4_V28_UPDATE_ZERO_REPORT)" || { echo "V28 update-zero equivalence/control screen failed; refusing Stage A." >&2; exit 2; }
	@if [ -f "$(GEMMA4_V28_STAGE_A_ROOT)/update_004/adapter.safetensors" ] && [ -f "$(GEMMA4_V28_STAGE_A_ROOT)/update_004/metadata.json" ] && [ -f "$(GEMMA4_V28_STAGE_A_ROOT)/update_004/runtime_metadata.json" ] && [ -f "$(GEMMA4_V28_STAGE_A_ROOT)/best/adapter.safetensors" ]; then \
		echo "Reusing complete four-update V28 Stage-A run."; \
	elif [ -e "$(GEMMA4_V28_STAGE_A_ROOT)" ]; then \
		echo "Incomplete V28 Stage-A root exists; refusing to overwrite it: $(GEMMA4_V28_STAGE_A_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_post_stack_sidecar --config "$(GEMMA4_V28_CONFIG)" --candidate "$(GEMMA4_V28_CANDIDATE)" --output "$(GEMMA4_V28_STAGE_A_ROOT)"; \
	fi

gemma4-v28-select-stage-a: gemma4-v28-train-stage-a
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v28_stage_a_selector --config "$(GEMMA4_V28_CONFIG)" --checkpoint-root "$(GEMMA4_V28_STAGE_A_ROOT)" --output "$(GEMMA4_V28_STAGE_A_SELECTION)"

# Optional Stage B opens only a fresh, exact-zero rank-4 query-LoRA bank after
# the Stage-A causal selector approves a nonzero sidecar update. The trainer
# freezes and hash-checks the selected scene stack and all inherited banks.
gemma4-v28-train-stage-b: gemma4-v28-select-stage-a
	@PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); valid=report.get("artifact") == "v28_post_stack_sidecar_stage_a_selection" and report.get("passed") is True and isinstance(report.get("selected_update"), int) and report["selected_update"] > 0 and bool(report.get("selected_checkpoint")); sys.exit(0 if valid else 2)' "$(GEMMA4_V28_STAGE_A_SELECTION)" || { echo "V28 Stage-A selector did not approve a nonzero checkpoint; refusing Stage B." >&2; exit 2; }
	@complete=1; \
	for update in 000 001 002 003 004; do \
		for file in adapter.safetensors metadata.json runtime_metadata.json; do \
			test -f "$(GEMMA4_V28_STAGE_B_ROOT)/update_$$update/$$file" || complete=0; \
		done; \
	done; \
	if [ "$$complete" = 1 ] && [ -f "$(GEMMA4_V28_STAGE_B_ROOT)/best/adapter.safetensors" ]; then \
		echo "Reusing complete four-update V28 Stage-B run; causal selection will rerun."; \
	elif [ -e "$(GEMMA4_V28_STAGE_B_ROOT)" ]; then \
		echo "Incomplete V28 Stage-B root exists; refusing to overwrite it: $(GEMMA4_V28_STAGE_B_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_post_stack_decoder --config "$(GEMMA4_V28_STAGE_B_CONFIG)" --output "$(GEMMA4_V28_STAGE_B_ROOT)" --stage-a-selection "$(GEMMA4_V28_STAGE_A_SELECTION)"; \
	fi

gemma4-v28-select-stage-b: gemma4-v28-train-stage-b
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v28_stage_b_selector --config "$(GEMMA4_V28_STAGE_B_CONFIG)" --checkpoint-root "$(GEMMA4_V28_STAGE_B_ROOT)" --output "$(GEMMA4_V28_STAGE_B_SELECTION)"

gemma4-v28-stage-b: gemma4-v28-select-stage-b

gemma4-v28-evaluate-stage-b: gemma4-v28-select-stage-b
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.prepare_questions --config "$(GEMMA4_V28_STAGE_B_CONFIG)" --split "$(GEMMA4_V28_STAGE_B_EVAL_SPLIT)" --output "$(GEMMA4_V28_STAGE_B_QUESTIONS)" --force
	@checkpoint="$$(PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); assert report.get("artifact") == "v28_post_stack_decoder_stage_b_selection" and report.get("passed") is True and report.get("selected_checkpoint"); print(report["selected_checkpoint"])' "$(GEMMA4_V28_STAGE_B_SELECTION)")"; \
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.predict --config "$(GEMMA4_V28_STAGE_B_CONFIG)" --split "$(GEMMA4_V28_STAGE_B_EVAL_SPLIT)" --questions-manifest "$(GEMMA4_V28_STAGE_B_QUESTIONS)" --checkpoint "$$checkpoint" --output "$(GEMMA4_V28_STAGE_B_PREDICTIONS)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.run --config "$(GEMMA4_V28_STAGE_B_CONFIG)" --references "data/qa/$(GEMMA4_V28_STAGE_B_EVAL_SPLIT).jsonl" --predictions "$(GEMMA4_V28_STAGE_B_PREDICTIONS)" --output "$(GEMMA4_V28_STAGE_B_METRICS)"

# V29 trains only on scenes 11-18 and validates only on scenes 19-24. Its
# config and trainer fail closed if scenes 25-30 appear in any development QA
# input. Selection still gates every update on the older V24/V28 color/mirror
# controls, which are loaded from a separate immutable control config.
gemma4-v29-train-diverse-stage-b: gemma4-v28-select-stage-a
	@PYTHONPATH=src $(GEMMA4_PYTHON) -c 'from semantic_3d_chat.config import load_config; from semantic_3d_chat.training.train_post_stack_decoder import load_stage_b_qa_records; c=load_config("$(GEMMA4_V29_CONFIG)"); load_stage_b_qa_records(c, max_train_questions=None, max_validation_questions=None)'
	@complete=1; \
	for update in 000 001 002 003 004; do \
		for file in adapter.safetensors metadata.json runtime_metadata.json; do \
			test -f "$(GEMMA4_V29_ROOT)/update_$$update/$$file" || complete=0; \
		done; \
	done; \
	if [ "$$complete" = 1 ] && [ -f "$(GEMMA4_V29_ROOT)/best/adapter.safetensors" ]; then \
		echo "Reusing complete four-update V29 diverse Stage-B run; causal selection will rerun."; \
	elif [ -e "$(GEMMA4_V29_ROOT)" ]; then \
		echo "Incomplete V29 Stage-B root exists; refusing to overwrite it: $(GEMMA4_V29_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_post_stack_decoder --config "$(GEMMA4_V29_CONFIG)" --output "$(GEMMA4_V29_ROOT)" --stage-a-selection "$(GEMMA4_V28_STAGE_A_SELECTION)"; \
	fi

gemma4-v29-select-diverse-stage-b: gemma4-v29-train-diverse-stage-b
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v28_stage_b_selector --config "$(GEMMA4_V29_CONFIG)" --checkpoint-root "$(GEMMA4_V29_ROOT)" --output "$(GEMMA4_V29_SELECTION)"

gemma4-v29-evaluate-diverse-validation: gemma4-v29-select-diverse-stage-b
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.prepare_questions --config "$(GEMMA4_V29_CONFIG)" --split validation --output "$(GEMMA4_V29_QUESTIONS)" --force
	@checkpoint="$$(PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json; report=json.load(open("$(GEMMA4_V29_SELECTION)", encoding="utf-8")); assert report.get("passed") is True and report.get("selected_checkpoint"); print(report["selected_checkpoint"])')"; \
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.predict --config "$(GEMMA4_V29_CONFIG)" --split validation --questions-manifest "$(GEMMA4_V29_QUESTIONS)" --checkpoint "$$checkpoint" --output "$(GEMMA4_V29_PREDICTIONS)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.run --config "$(GEMMA4_V29_CONFIG)" --references data_diverse20/qa/validation.jsonl --predictions "$(GEMMA4_V29_PREDICTIONS)" --output "$(GEMMA4_V29_METRICS)"

# V30 is a development-only repair of V29's counterfactual invariance. It
# reuses the locked scenes 11-24 and refuses deferred scenes 25-30. Every
# answer-changing unit is atomic and oversampled; the only trainable tensors
# are the sidecar output projection/gain and a fresh disjoint decoder bank.
gemma4-v30-train-joint-pair:
	@PYTHONPATH=src $(GEMMA4_PYTHON) -c 'from semantic_3d_chat.config import load_config; from semantic_3d_chat.training.train_joint_pair_v30 import require_approved_v29_source,v30_contract,v30_settings; c=load_config("$(GEMMA4_V30_CONFIG)"); v30_contract(c); v30_settings(c); require_approved_v29_source(c)'
	@complete=1; \
	for update in 000 001 002 003 004; do \
		for file in adapter.safetensors metadata.json runtime_metadata.json; do \
			test -f "$(GEMMA4_V30_ROOT)/update_$$update/$$file" || complete=0; \
		done; \
	done; \
	if [ "$$complete" = 1 ] && [ -f "$(GEMMA4_V30_ROOT)/best/adapter.safetensors" ]; then \
		echo "Reusing complete four-update V30 joint-pair run; selection will rerun."; \
	elif [ -e "$(GEMMA4_V30_ROOT)" ]; then \
		echo "Incomplete V30 root exists; refusing to overwrite it: $(GEMMA4_V30_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_joint_pair_v30 --config "$(GEMMA4_V30_CONFIG)" --output "$(GEMMA4_V30_ROOT)"; \
	fi

gemma4-v30-select-joint-pair: gemma4-v30-train-joint-pair
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v30_joint_pair_selector --config "$(GEMMA4_V30_CONFIG)" --checkpoint-root "$(GEMMA4_V30_ROOT)" --output "$(GEMMA4_V30_SELECTION)"

# V31 starts again from approved V29 update_004; it never continues V30.
# Training expands only to scenes 11-18 plus 31-38, preserves validation
# scenes 19-24, and fails closed if any deferred scene 25-30 is materialized in
# QA.  All nine arms (update zero plus eight cycles) are selected independently.
gemma4-v31-preflight-expanded-pair:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_joint_pair_v31 --config "$(GEMMA4_V31_CONFIG)" --preflight-only

gemma4-v31-train-expanded-pair: gemma4-v31-preflight-expanded-pair
	@complete=1; \
	for update in 000 001 002 003 004 005 006 007 008; do \
		for file in adapter.safetensors metadata.json runtime_metadata.json; do \
			test -f "$(GEMMA4_V31_ROOT)/update_$$update/$$file" || complete=0; \
		done; \
	done; \
	if [ "$$complete" = 1 ] && [ -f "$(GEMMA4_V31_ROOT)/best/adapter.safetensors" ]; then \
		echo "Reusing complete eight-cycle V31 expanded-pair run; selection will rerun."; \
	elif [ -e "$(GEMMA4_V31_ROOT)" ]; then \
		echo "Incomplete V31 root exists; refusing to overwrite it: $(GEMMA4_V31_ROOT)" >&2; \
		exit 2; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_joint_pair_v31 --config "$(GEMMA4_V31_CONFIG)" --output "$(GEMMA4_V31_ROOT)"; \
	fi

gemma4-v31-select-expanded-pair: gemma4-v31-train-expanded-pair
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v31_joint_pair_selector --config "$(GEMMA4_V31_CONFIG)" --checkpoint-root "$(GEMMA4_V31_ROOT)" --output "$(GEMMA4_V31_SELECTION)"

# V32 is prepared as a conditional optimizer repair, not as an automatic
# continuation. The trainer itself refuses to run until the audited V31
# selector report exists and rejects V31. It then starts afresh from approved
# V29 and performs 80 true broad+pair micro-updates, saving every eight steps.
gemma4-v32-preflight-microstep:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_microstep_v32 --config "$(GEMMA4_V32_CONFIG)" --preflight-only

gemma4-v32-train-microstep: gemma4-v32-preflight-microstep
	@complete=1; \
	for update in 000 008 016 024 032 040 048 056 064 072 080; do \
		for file in adapter.safetensors metadata.json runtime_metadata.json; do \
			test -f "$(GEMMA4_V32_ROOT)/update_$$update/$$file" || complete=0; \
		done; \
		if [ "$$update" != 000 ]; then test -f "$(GEMMA4_V32_ROOT)/update_$$update/optimizer.pt" || complete=0; fi; \
	done; \
	for file in adapter.safetensors metadata.json runtime_metadata.json; do \
		test -f "$(GEMMA4_V32_ROOT)/best/$$file" || complete=0; \
	done; \
	if [ "$$complete" = 1 ]; then \
		echo "Reusing complete 80-step V32 microstep run; selection will rerun."; \
	elif [ -e "$(GEMMA4_V32_ROOT)" ]; then \
		echo "Resuming incomplete V32 run from its latest complete saved arm."; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_microstep_v32 --config "$(GEMMA4_V32_CONFIG)" --output "$(GEMMA4_V32_ROOT)" --resume-latest; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_microstep_v32 --config "$(GEMMA4_V32_CONFIG)" --output "$(GEMMA4_V32_ROOT)"; \
	fi

gemma4-v32-select-microstep: gemma4-v32-train-microstep
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v32_microstep_selector --config "$(GEMMA4_V32_CONFIG)" --checkpoint-root "$(GEMMA4_V32_ROOT)" --output "$(GEMMA4_V32_SELECTION)"

# V33 is the environmental-only response to V32's pinned terminal rejection.
# It starts afresh from approved V29 update_004, freezes Gemma and all LoRA
# banks, and trains exactly 404,608 dense-sidecar parameters for 100 true steps.
gemma4-v33-preflight-environmental:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_environmental_sidecar_v33 --config "$(GEMMA4_V33_CONFIG)" --preflight-only

gemma4-v33-train-environmental: gemma4-v33-preflight-environmental
	@complete=1; \
	for update in 000 008 016 024 032 040 048 056 064 072 080 088 096 100; do \
		for file in adapter.safetensors metadata.json runtime_metadata.json; do \
			test -f "$(GEMMA4_V33_ROOT)/update_$$update/$$file" || complete=0; \
		done; \
		if [ "$$update" != 000 ]; then test -f "$(GEMMA4_V33_ROOT)/update_$$update/optimizer.pt" || complete=0; fi; \
	done; \
	if [ "$$complete" = 1 ]; then \
		echo "Reusing complete 100-step V33 environmental-only run; selection will rerun."; \
	elif [ -e "$(GEMMA4_V33_ROOT)" ]; then \
		echo "Resuming V33 from its latest complete audited arm."; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_environmental_sidecar_v33 --config "$(GEMMA4_V33_CONFIG)" --output "$(GEMMA4_V33_ROOT)" --resume-latest; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_environmental_sidecar_v33 --config "$(GEMMA4_V33_CONFIG)" --output "$(GEMMA4_V33_ROOT)"; \
	fi

gemma4-v33-select-environmental: gemma4-v33-train-environmental
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v33_environmental_selector --config "$(GEMMA4_V33_CONFIG)" --checkpoint-root "$(GEMMA4_V33_ROOT)" --output "$(GEMMA4_V33_SELECTION)"

# V33 stopped causally at update 64. Seal that exact failure without loading a
# model or scene data, then permit only the bounded four-tensor V34 base route.
gemma4-v33-seal-update64:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v33_terminal_gate --config "$(GEMMA4_V33_CONFIG)" --checkpoint-root "$(GEMMA4_V33_ROOT)" --output "$(GEMMA4_V33_TERMINAL_GATE)"

gemma4-v34-preflight-base-surface: gemma4-v33-seal-update64
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_base_surface_v34 --config "$(GEMMA4_V34_CONFIG)" --preflight-only

gemma4-v34-train-base-surface: gemma4-v34-preflight-base-surface
	@complete=1; \
	for update in 000 008 016 024 032 040 048 056 064; do \
		for file in adapter.safetensors metadata.json runtime_metadata.json; do \
			test -f "$(GEMMA4_V34_ROOT)/update_$$update/$$file" || complete=0; \
		done; \
		if [ "$$update" != 000 ]; then test -f "$(GEMMA4_V34_ROOT)/update_$$update/optimizer.pt" || complete=0; fi; \
	done; \
	if [ "$$complete" = 1 ]; then \
		echo "Reusing complete bounded 64-step V34 base-route run; selection will rerun."; \
	elif [ -e "$(GEMMA4_V34_ROOT)" ]; then \
		echo "Resuming V34 from its latest complete audited arm."; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_base_surface_v34 --config "$(GEMMA4_V34_CONFIG)" --output "$(GEMMA4_V34_ROOT)" --resume-latest; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_base_surface_v34 --config "$(GEMMA4_V34_CONFIG)" --output "$(GEMMA4_V34_ROOT)"; \
	fi

gemma4-v34-select-base-surface: gemma4-v34-train-base-surface
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v34_base_surface_selector --config "$(GEMMA4_V34_CONFIG)" --checkpoint-root "$(GEMMA4_V34_ROOT)" --output "$(GEMMA4_V34_SELECTION)"

# V34 stopped causally at its train-only update-32 selectivity gate. This
# metadata/tensor-only seal can authorize only the exact-zero V35 block route.
gemma4-v34-seal-update32:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v34_terminal_gate --config "$(GEMMA4_V34_CONFIG)" --checkpoint-root "$(GEMMA4_V34_ROOT)" --output "$(GEMMA4_V34_TERMINAL_GATE)"

# V35 always restarts from exact V33 update 64. The V34 seal is an immutable
# authorization receipt only; failed V34 update-32 weights are never loaded.
gemma4-v35-preflight-block-cross: gemma4-v34-seal-update32
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_block_cross_v35 --config "$(GEMMA4_V35_CONFIG)" --preflight-only

gemma4-v35-train-block-cross: gemma4-v35-preflight-block-cross
	@complete=1; \
	for update in 000 008 016 024 032 040 048 056 064 072 080 088 096 100; do \
		for file in adapter.safetensors metadata.json runtime_metadata.json; do \
			test -f "$(GEMMA4_V35_ROOT)/update_$$update/$$file" || complete=0; \
		done; \
		if [ "$$update" != 000 ]; then test -f "$(GEMMA4_V35_ROOT)/update_$$update/optimizer.pt" || complete=0; fi; \
	done; \
	if [ "$$complete" = 1 ]; then \
		echo "Reusing complete bounded 100-step V35 all-block run; selection must rerun independently."; \
	elif [ -e "$(GEMMA4_V35_ROOT)" ]; then \
		echo "Resuming V35 from its latest complete audited arm."; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_block_cross_v35 --config "$(GEMMA4_V35_CONFIG)" --output "$(GEMMA4_V35_ROOT)" --resume-latest; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_block_cross_v35 --config "$(GEMMA4_V35_CONFIG)" --output "$(GEMMA4_V35_ROOT)"; \
	fi

# Development validation remains unreachable until the trainer has produced
# all exact numbered arms through update 100 and both train-only gates pass.
gemma4-v35-select-block-cross: gemma4-v35-train-block-cross
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v35_block_cross_selector --config "$(GEMMA4_V35_CONFIG)" --checkpoint-root "$(GEMMA4_V35_ROOT)" --output "$(GEMMA4_V35_SELECTION)"

# V35 stopped causally at its train-only update-32 selectivity gate. This
# report-only seal loads no Gemma, QA, maps, oracle data, or final scenes and
# can authorize only the exact bounded V36 joint existing-LoRA route.
gemma4-v35-seal-update32:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v35_terminal_gate --config "$(GEMMA4_V35_CONFIG)" --checkpoint-root "$(GEMMA4_V35_ROOT)" --output "$(GEMMA4_V35_TERMINAL_GATE)"

# V36 continues only the exact learned V35 update-32 block matrices and the
# existing exact-zero V30 query bank. V35 optimizer momentum is never loaded.
gemma4-v36-preflight-joint-block-cross: gemma4-v35-seal-update32
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_joint_block_cross_v36 --config "$(GEMMA4_V36_CONFIG)" --preflight-only

gemma4-v36-train-joint-block-cross: gemma4-v36-preflight-joint-block-cross
	@complete=1; \
	for update in 000 008 016 024 032 040 048 056 064 072 080 088 096 100; do \
		for file in adapter.safetensors metadata.json runtime_metadata.json; do \
			test -f "$(GEMMA4_V36_ROOT)/update_$$update/$$file" || complete=0; \
		done; \
		if [ "$$update" != 000 ]; then test -f "$(GEMMA4_V36_ROOT)/update_$$update/optimizer.pt" || complete=0; fi; \
	done; \
	if [ "$$complete" = 1 ]; then \
		echo "Reusing complete bounded 100-step V36 joint run; selection must rerun independently."; \
	elif [ -e "$(GEMMA4_V36_ROOT)" ]; then \
		echo "Resuming V36 from its latest complete audited arm."; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_joint_block_cross_v36 --config "$(GEMMA4_V36_CONFIG)" --output "$(GEMMA4_V36_ROOT)" --resume-latest; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_joint_block_cross_v36 --config "$(GEMMA4_V36_CONFIG)" --output "$(GEMMA4_V36_ROOT)"; \
	fi

# Development validation remains sealed until all 100 updates and all three
# train-only continuation gates are complete and independently replayable.
gemma4-v36-select-joint-block-cross: gemma4-v36-train-joint-block-cross
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v36_joint_block_cross_selector --config "$(GEMMA4_V36_CONFIG)" --checkpoint-root "$(GEMMA4_V36_ROOT)" --output "$(GEMMA4_V36_SELECTION)"

# V36 stopped at its first train-only causal gate. This tensor-only seal loads
# no Gemma, QA, maps, oracle data, or final scenes and cannot select V36.
gemma4-v36-seal-update16:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v36_terminal_gate --config "$(GEMMA4_V36_CONFIG)" --checkpoint-root "$(GEMMA4_V36_ROOT)" --output "$(GEMMA4_V36_TERMINAL_GATE)"

# Development-set measurement only. This target bypasses the promotion gate so
# a bounded research candidate can be measured; it does not authorize chat.
gemma4-v28-evaluate: gemma4-v28-select-stage-a
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.prepare_questions --config "$(GEMMA4_V28_CONFIG)" --split "$(GEMMA4_V28_EVAL_SPLIT)" --output "$(GEMMA4_V28_QUESTIONS)" --force
	@checkpoint="$(strip $(GEMMA4_V28_EVAL_CHECKPOINT))"; \
	if [ -z "$$checkpoint" ]; then \
		checkpoint="$$(PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); assert report.get("passed") is True and report.get("selected_checkpoint"); print(report["selected_checkpoint"])' "$(GEMMA4_V28_STAGE_A_SELECTION)")"; \
	fi; \
	test -f "$$checkpoint/adapter.safetensors"; \
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.predict --config "$(GEMMA4_V28_CONFIG)" --split "$(GEMMA4_V28_EVAL_SPLIT)" --questions-manifest "$(GEMMA4_V28_QUESTIONS)" --checkpoint "$$checkpoint" --output "$(GEMMA4_V28_PREDICTIONS)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.run --config "$(GEMMA4_V28_CONFIG)" --references "data/qa/$(GEMMA4_V28_EVAL_SPLIT).jsonl" --predictions "$(GEMMA4_V28_PREDICTIONS)" --output "$(GEMMA4_V28_METRICS)"

# V23's control plane was added after its four-update evidence was produced.
# The trainer therefore runs from a temporary detached worktree at the exact
# source commit recorded by the screen. Only ignored artifact children are
# linked into that checkout, so train_adapter's strict provenance check remains
# enabled. Arguments: resume checkpoint, target epoch, copy final summaries.
define run-v23-pinned-trainer
	@if [ "$(4)" = "replay" ] && [ -f "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_004/metadata.json" ]; then \
		echo "Reusing cached V23 replay; the verifier will require exact epochs 003/004."; exit 0; \
	elif [ "$(4)" = "replay" ] && { [ -e "$(GEMMA4_V23_EXTENSION_ROOT)" ] || [ -L "$(GEMMA4_V23_EXTENSION_ROOT)" ]; }; then \
		echo "Partial V23 replay root exists; refusing to overwrite it: $(GEMMA4_V23_EXTENSION_ROOT)" >&2; exit 2; \
	elif [ "$(4)" = "novel" ] && [ -f "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_008/metadata.json" ]; then \
		echo "Reusing cached V23 update-8 branch; final selection will validate all epochs."; exit 0; \
	elif [ "$(4)" = "novel" ] && [ ! -f "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_004/metadata.json" ]; then \
		echo "V23 replay checkpoint epoch_004 is unavailable." >&2; exit 2; \
	elif [ "$(4)" = "novel" ] && { [ -e "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_005" ] || [ -L "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_005" ] || [ -e "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_006" ] || [ -L "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_006" ] || [ -e "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_007" ] || [ -L "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_007" ] || [ -e "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_008" ] || [ -L "$(GEMMA4_V23_EXTENSION_ROOT)/epoch_008" ]; }; then \
		echo "Partial V23 novel-update branch exists; refusing to overwrite it." >&2; exit 2; \
	fi; \
	set -eu; \
	training_commit="$$(PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_source_provenance"]["head_commit"])' "$(GEMMA4_V23_EXTENSION_MANIFEST)")"; \
	worktree_parent="$$(mktemp -d "$${TMPDIR:-/tmp}/semantic_3d_chat_v23.XXXXXX")"; \
	worktree="$$worktree_parent/source"; \
	cleanup() { \
		if [ -e "$$worktree/.git" ]; then git worktree remove --force "$$worktree" >/dev/null 2>&1 || true; fi; \
		rmdir "$$worktree_parent" >/dev/null 2>&1 || true; \
	}; \
	trap cleanup EXIT HUP INT TERM; \
	git worktree add --detach "$$worktree" "$$training_commit"; \
	mkdir "$$worktree/data_gemma4"; \
	ln -s "$(CURDIR)/data_gemma4/checkpoints" "$$worktree/data_gemma4/checkpoints"; \
	ln -s "$(CURDIR)/data_gemma4/features" "$$worktree/data_gemma4/features"; \
	ln -s "$(CURDIR)/data_gemma4/maps" "$$worktree/data_gemma4/maps"; \
	mkdir -p "$$worktree/data"; \
	ln -s "$(CURDIR)/data/qa" "$$worktree/data/qa"; \
	(cd "$$worktree" && PYTHONPATH=src "$(CURDIR)/$(GEMMA4_PYTHON)" -c 'import json; from semantic_3d_chat.config import PROJECT_ROOT; from semantic_3d_chat.training.source_provenance import capture_git_source_provenance, require_clean_committed_source; observed=capture_git_source_provenance(PROJECT_ROOT); require_clean_committed_source(observed); expected=json.load(open("$(CURDIR)/$(GEMMA4_V23_EXTENSION_MANIFEST)", encoding="utf-8"))["training_source_provenance"]; assert observed == expected, (observed, expected)'); \
	(cd "$$worktree" && PYTHONPATH=src "$(CURDIR)/$(GEMMA4_PYTHON)" -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V23_CONFIG) --resume "$(1)" --output-namespace $(GEMMA4_V23_EXTENSION_NAMESPACE) --epochs "$(2)"); \
	if [ "$(3)" = "copy-reports" ]; then \
		cp "$$worktree/reports/gemma4/metrics/training_$(GEMMA4_V23_EXTENSION_NAMESPACE).json" "$(CURDIR)/reports/gemma4/metrics/"; \
		cp "$$worktree/reports/gemma4/metrics/training_selection_$(GEMMA4_V23_EXTENSION_NAMESPACE).json" "$(CURDIR)/reports/gemma4/metrics/"; \
	fi
endef

# Preparation recomputes the complete V23 screen and authorizes only an exact
# epoch-2 source in a fresh isolated namespace.
gemma4-v23-prepare-extension:
	@if [ -e "$(GEMMA4_V23_EXTENSION_ROOT)" ] || [ -L "$(GEMMA4_V23_EXTENSION_ROOT)" ]; then \
		test -f "$(GEMMA4_V23_EXTENSION_MANIFEST)" || { echo "V23 extension root exists without its authorization manifest." >&2; exit 2; }; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v23_extension_controller validate-launch --manifest $(GEMMA4_V23_EXTENSION_MANIFEST); \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v23_extension_controller prepare --config $(GEMMA4_V23_CONFIG) --screen $(GEMMA4_V23_SCREEN_REPORT) --output $(GEMMA4_V23_EXTENSION_MANIFEST); \
	fi

# Stage A replays only updates 3 and 4 from immutable primary epoch 2.
gemma4-v23-run-extension-replay: gemma4-v23-prepare-extension
	$(call run-v23-pinned-trainer,data_gemma4/checkpoints/$(GEMMA4_V23_NAMESPACE)/epoch_002,4,no-copy,replay)

gemma4-v23-verify-extension-replay: gemma4-v23-run-extension-replay
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v23_extension_controller verify-replay --manifest $(GEMMA4_V23_EXTENSION_MANIFEST) --output $(GEMMA4_V23_EXTENSION_REPLAY)

# Stage B cannot execute until replay artifacts exactly match primary 003/004.
gemma4-v23-run-extension: gemma4-v23-verify-extension-replay
	@PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v23_extension_controller authorize-stage-b --replay $(GEMMA4_V23_EXTENSION_REPLAY)
	$(call run-v23-pinned-trainer,data_gemma4/checkpoints/$(GEMMA4_V23_EXTENSION_NAMESPACE)/epoch_004,8,copy-reports,novel)

gemma4-v23-select-extension: gemma4-v23-run-extension
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v23_extension_controller select-final --manifest $(GEMMA4_V23_EXTENSION_MANIFEST) --replay $(GEMMA4_V23_EXTENSION_REPLAY) --output $(GEMMA4_V23_EXTENSION_REPORT)

gemma4-v23-extension: gemma4-v23-select-extension

# V24's screen selected epoch 1. Replay updates 2--4 from that exact state in
# an isolated namespace and require decoded optimizer/history equivalence
# before running novel updates 5--8. As with V23, the trainer executes from the
# immutable training-source commit recorded by the launch manifest.
define run-v24-pinned-trainer
	@if [ "$(4)" = "replay" ] && [ -f "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_004/metadata.json" ]; then \
		echo "Reusing cached V24 replay; the verifier will require exact epochs 002--004."; exit 0; \
	elif [ "$(4)" = "replay" ] && { [ -e "$(GEMMA4_V24_EXTENSION_ROOT)" ] || [ -L "$(GEMMA4_V24_EXTENSION_ROOT)" ]; }; then \
		echo "Partial V24 replay root exists; refusing to overwrite it: $(GEMMA4_V24_EXTENSION_ROOT)" >&2; exit 2; \
	elif [ "$(4)" = "novel" ] && [ -f "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_008/metadata.json" ]; then \
		echo "Reusing cached V24 update-8 branch; final selection will validate all epochs."; exit 0; \
	elif [ "$(4)" = "novel" ] && [ ! -f "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_004/metadata.json" ]; then \
		echo "V24 replay checkpoint epoch_004 is unavailable." >&2; exit 2; \
	elif [ "$(4)" = "novel" ] && { [ -e "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_005" ] || [ -L "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_005" ] || [ -e "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_006" ] || [ -L "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_006" ] || [ -e "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_007" ] || [ -L "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_007" ] || [ -e "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_008" ] || [ -L "$(GEMMA4_V24_EXTENSION_ROOT)/epoch_008" ]; }; then \
		echo "Partial V24 novel-update branch exists; refusing to overwrite it." >&2; exit 2; \
	fi; \
	set -eu; \
	training_commit="$$(PYTHONPATH=src $(GEMMA4_PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_source_provenance"]["head_commit"])' "$(GEMMA4_V24_EXTENSION_MANIFEST)")"; \
	worktree_parent="$$(mktemp -d "$${TMPDIR:-/tmp}/semantic_3d_chat_v24.XXXXXX")"; \
	worktree="$$worktree_parent/source"; \
	cleanup() { \
		if [ -e "$$worktree/.git" ]; then git worktree remove --force "$$worktree" >/dev/null 2>&1 || true; fi; \
		rmdir "$$worktree_parent" >/dev/null 2>&1 || true; \
	}; \
	trap cleanup EXIT HUP INT TERM; \
	git worktree add --detach "$$worktree" "$$training_commit"; \
	mkdir "$$worktree/data_gemma4"; \
	ln -s "$(CURDIR)/data_gemma4/checkpoints" "$$worktree/data_gemma4/checkpoints"; \
	ln -s "$(CURDIR)/data_gemma4/features" "$$worktree/data_gemma4/features"; \
	ln -s "$(CURDIR)/data_gemma4/maps" "$$worktree/data_gemma4/maps"; \
	mkdir -p "$$worktree/data"; \
	ln -s "$(CURDIR)/data/qa" "$$worktree/data/qa"; \
	(cd "$$worktree" && PYTHONPATH=src "$(CURDIR)/$(GEMMA4_PYTHON)" -c 'import json; from semantic_3d_chat.config import PROJECT_ROOT; from semantic_3d_chat.training.source_provenance import capture_git_source_provenance, require_clean_committed_source; observed=capture_git_source_provenance(PROJECT_ROOT); require_clean_committed_source(observed); expected=json.load(open("$(CURDIR)/$(GEMMA4_V24_EXTENSION_MANIFEST)", encoding="utf-8"))["training_source_provenance"]; assert observed == expected, (observed, expected)'); \
	(cd "$$worktree" && PYTHONPATH=src "$(CURDIR)/$(GEMMA4_PYTHON)" -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_V24_CONFIG) --resume "$(1)" --output-namespace $(GEMMA4_V24_EXTENSION_NAMESPACE) --epochs "$(2)"); \
	if [ "$(3)" = "copy-reports" ]; then \
		cp "$$worktree/reports/gemma4/metrics/training_$(GEMMA4_V24_EXTENSION_NAMESPACE).json" "$(CURDIR)/reports/gemma4/metrics/"; \
		cp "$$worktree/reports/gemma4/metrics/training_selection_$(GEMMA4_V24_EXTENSION_NAMESPACE).json" "$(CURDIR)/reports/gemma4/metrics/"; \
	fi
endef

gemma4-v24-prepare-extension:
	@if [ -e "$(GEMMA4_V24_EXTENSION_ROOT)" ] || [ -L "$(GEMMA4_V24_EXTENSION_ROOT)" ]; then \
		test -f "$(GEMMA4_V24_EXTENSION_MANIFEST)" || { echo "V24 extension root exists without its authorization manifest." >&2; exit 2; }; \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v24_extension_controller validate-launch --manifest $(GEMMA4_V24_EXTENSION_MANIFEST); \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v24_extension_controller prepare --config $(GEMMA4_V24_CONFIG) --screen $(GEMMA4_V24_SCREEN_REPORT) --output $(GEMMA4_V24_EXTENSION_MANIFEST); \
	fi

gemma4-v24-run-extension-replay: gemma4-v24-prepare-extension
	$(call run-v24-pinned-trainer,data_gemma4/checkpoints/$(GEMMA4_V24_NAMESPACE)/epoch_001,4,no-copy,replay)

gemma4-v24-verify-extension-replay: gemma4-v24-run-extension-replay
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v24_extension_controller verify-replay --manifest $(GEMMA4_V24_EXTENSION_MANIFEST) --output $(GEMMA4_V24_EXTENSION_REPLAY)

gemma4-v24-run-extension: gemma4-v24-verify-extension-replay
	@PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v24_extension_controller authorize-stage-b --replay $(GEMMA4_V24_EXTENSION_REPLAY)
	$(call run-v24-pinned-trainer,data_gemma4/checkpoints/$(GEMMA4_V24_EXTENSION_NAMESPACE)/epoch_004,8,copy-reports,novel)

gemma4-v24-select-extension: gemma4-v24-run-extension
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v24_extension_controller select-final --manifest $(GEMMA4_V24_EXTENSION_MANIFEST) --replay $(GEMMA4_V24_EXTENSION_REPLAY) --output $(GEMMA4_V24_EXTENSION_REPORT)

gemma4-v24-extension: gemma4-v24-select-extension

# No Gemma checkpoint currently satisfies this gate. A future accepted pair must
# be supplied explicitly and carry a hash-bound promotion.json beside the adapter.
require-gemma4-promoted:
	@if [ -z "$(strip $(GEMMA4_STATIC_CONFIG))" ] || [ -z "$(strip $(GEMMA4_STATIC_CHECKPOINT))" ]; then \
		echo "No accepted Gemma checkpoint is configured. Supply both GEMMA4_STATIC_CONFIG and GEMMA4_STATIC_CHECKPOINT after behavioral promotion." >&2; \
		exit 2; \
	fi
	@test -x "$(GEMMA4_PYTHON)"
	@test -f "$(GEMMA4_STATIC_CONFIG)"
	@test -f "$(GEMMA4_STATIC_CHECKPOINT)/adapter.safetensors"
	@PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.chat.promotion validate --runtime-config "$(GEMMA4_STATIC_CONFIG)" --checkpoint "$(GEMMA4_STATIC_CHECKPOINT)" >/dev/null

# Promotion is deliberately a separate offline action. It requires the selector,
# held-out final evaluation, and oracle-deletion leakage reports for one exact
# checkpoint/config pair. Empty variables make this target fail before writing.
gemma4-create-final-evidence:
	@if [ -z "$(strip $(GEMMA4_PROMOTION_CHECKPOINT))" ] || \
	    [ -z "$(strip $(GEMMA4_FINAL_METRICS))" ] || \
	    [ -z "$(strip $(GEMMA4_FINAL_PREDICTIONS))" ] || \
	    [ -z "$(strip $(GEMMA4_FINAL_PREDICTION_PROVENANCE))" ] || \
	    [ -z "$(strip $(GEMMA4_FINAL_CHANCE_METRICS))" ] || \
	    [ -z "$(strip $(GEMMA4_FINAL_CHANCE_PREDICTIONS))" ] || \
	    [ -z "$(strip $(GEMMA4_FINAL_CHANCE_PROVENANCE))" ] || \
	    [ -z "$(strip $(GEMMA4_FINAL_SPLIT_MANIFEST))" ]; then \
		echo "Final evidence requires complete primary/chance predictions, provenance, metrics, checkpoint, and split paths." >&2; \
		exit 2; \
	fi
	@test -x "$(PYTHON)"
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.chat.promotion create-final-evidence \
		--runtime-config "$(GEMMA4_RUNTIME_CONFIG)" \
		--checkpoint "$(GEMMA4_PROMOTION_CHECKPOINT)" \
		--metrics "$(GEMMA4_FINAL_METRICS)" \
		--predictions "$(GEMMA4_FINAL_PREDICTIONS)" \
		--prediction-provenance "$(GEMMA4_FINAL_PREDICTION_PROVENANCE)" \
		--chance-metrics "$(GEMMA4_FINAL_CHANCE_METRICS)" \
		--chance-predictions "$(GEMMA4_FINAL_CHANCE_PREDICTIONS)" \
		--chance-prediction-provenance "$(GEMMA4_FINAL_CHANCE_PROVENANCE)" \
		--split-manifest "$(GEMMA4_FINAL_SPLIT_MANIFEST)" \
		--output "$(GEMMA4_FINAL_EVIDENCE_OUTPUT)"

# The only supported transition that can unlock deferred scenes 25--30. The
# selector and exact selected checkpoint are deliberately mandatory. The
# controller revalidates chat_promotion_eligible before writing its immutable
# launch seal and before invoking any Blender/final-scene command.
require-gemma4-final-once-inputs:
	@if [ -z "$(strip $(GEMMA4_FINAL_ONCE_SELECTOR_REPORT))" ] || \
	    [ -z "$(strip $(GEMMA4_FINAL_ONCE_CHECKPOINT))" ]; then \
		echo "Final-once requires explicit GEMMA4_FINAL_ONCE_SELECTOR_REPORT and GEMMA4_FINAL_ONCE_CHECKPOINT." >&2; \
		exit 2; \
	fi
	@test -x "$(PYTHON)"
	@test -x "$(GEMMA4_PYTHON)"
	@test -f "$(GEMMA4_FINAL_ONCE_SELECTOR_REPORT)"
	@test -f "$(GEMMA4_FINAL_ONCE_CHECKPOINT)/adapter.safetensors"

gemma4-final-once-preflight: require-gemma4-final-once-inputs
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.final_once preflight \
		--dataset-config "$(DIVERSE28_CONFIG)" \
		--runtime-config "$(GEMMA4_RUNTIME_CONFIG)" \
		--selector-report "$(GEMMA4_FINAL_ONCE_SELECTOR_REPORT)" \
		--checkpoint "$(GEMMA4_FINAL_ONCE_CHECKPOINT)" \
		--work-root "$(GEMMA4_FINAL_ONCE_WORK_ROOT)" \
		--primary-pointer "$(GEMMA4_PRIMARY_POINTER)" \
		--python "$(PYTHON)" --gemma-python "$(GEMMA4_PYTHON)" --blender "$(BLENDER)"

gemma4-final-once: require-gemma4-final-once-inputs
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.final_once run \
		--dataset-config "$(DIVERSE28_CONFIG)" \
		--runtime-config "$(GEMMA4_RUNTIME_CONFIG)" \
		--selector-report "$(GEMMA4_FINAL_ONCE_SELECTOR_REPORT)" \
		--checkpoint "$(GEMMA4_FINAL_ONCE_CHECKPOINT)" \
		--work-root "$(GEMMA4_FINAL_ONCE_WORK_ROOT)" \
		--primary-pointer "$(GEMMA4_PRIMARY_POINTER)" \
		--python "$(PYTHON)" --gemma-python "$(GEMMA4_PYTHON)" --blender "$(BLENDER)" $(if $(strip $(GEMMA4_FINAL_ONCE_STOP_AFTER)),--stop-after "$(GEMMA4_FINAL_ONCE_STOP_AFTER)",)

gemma4-create-chat-promotion:
	@if [ -z "$(strip $(GEMMA4_PROMOTION_CHECKPOINT))" ] || \
	    [ -z "$(strip $(GEMMA4_PROMOTION_SELECTOR_REPORT))" ] || \
	    [ -z "$(strip $(GEMMA4_PROMOTION_FINAL_EVIDENCE))" ] || \
	    [ -z "$(strip $(GEMMA4_PROMOTION_LEAKAGE_REPORT))" ]; then \
		echo "Promotion requires explicit checkpoint, selector, held-out-final, and leakage evidence paths." >&2; \
		exit 2; \
	fi
	@test -x "$(PYTHON)"
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.chat.promotion create \
		--runtime-config "$(GEMMA4_RUNTIME_CONFIG)" \
		--checkpoint "$(GEMMA4_PROMOTION_CHECKPOINT)" \
		--selector-report "$(GEMMA4_PROMOTION_SELECTOR_REPORT)" \
		--final-evidence "$(GEMMA4_PROMOTION_FINAL_EVIDENCE)" \
		--leakage-report "$(GEMMA4_PROMOTION_LEAKAGE_REPORT)" \
		--primary-pointer "$(GEMMA4_PRIMARY_POINTER)"

gemma4-validate-chat-promotion:
	@test -x "$(PYTHON)"
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.chat.promotion validate \
		--primary-pointer "$(GEMMA4_PRIMARY_POINTER)" >/dev/null

require-gemma4-primary: gemma4-validate-chat-promotion
	@test -x "$(GEMMA4_PYTHON)"

chat-gemma4: require-gemma4-primary
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.chat.cli \
		--primary-pointer "$(GEMMA4_PRIMARY_POINTER)" --scene "$(SCENE)"

web-gemma4: require-gemma4-primary
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.chat.web_app \
		--primary-pointer "$(GEMMA4_PRIMARY_POINTER)" --scene "$(SCENE)"

demo-gemma4: require-gemma4-primary
	./scripts/run_full_demo.sh --primary-pointer "$(GEMMA4_PRIMARY_POINTER)" --scene "$(SCENE)"

# Reproducible static evaluation for a behaviorally promoted checkpoint only.
# Historical failed wiring runs remain available as artifacts, not defaults.
gemma4-prepare-questions: require-gemma4-promoted
	@if [ -z "$(strip $(GEMMA4_STATIC_REFERENCES))" ]; then \
		echo "Question preparation requires explicit GEMMA4_STATIC_REFERENCES; no answer-bearing split is inferred." >&2; \
		exit 2; \
	fi
	@test -x "$(GEMMA4_PYTHON)"
	@test -f "$(GEMMA4_STATIC_CONFIG)"
	@test -f "$(GEMMA4_STATIC_REFERENCES)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.prepare_questions --config "$(GEMMA4_STATIC_CONFIG)" --split $(GEMMA4_EVAL_SPLIT) --qa "$(GEMMA4_STATIC_REFERENCES)" --output $(GEMMA4_QUESTIONS_MANIFEST) --force

gemma4-predict-static: gemma4-prepare-questions
	@test -x "$(GEMMA4_PYTHON)"
	@test -f "$(GEMMA4_STATIC_CONFIG)"
	@test -f "$(GEMMA4_STATIC_CHECKPOINT)/adapter.safetensors"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.predict --config "$(GEMMA4_STATIC_CONFIG)" --split $(GEMMA4_EVAL_SPLIT) --questions-manifest $(GEMMA4_QUESTIONS_MANIFEST) --checkpoint "$(GEMMA4_STATIC_CHECKPOINT)" --output $(GEMMA4_STATIC_PREDICTIONS)

gemma4-score-static: require-gemma4-promoted
	@if [ -z "$(strip $(GEMMA4_STATIC_REFERENCES))" ]; then \
		echo "Evaluation references are intentionally not implicit. Set GEMMA4_STATIC_REFERENCES to the physically separate QA JSONL." >&2; \
		exit 2; \
	fi
	@test -f "$(GEMMA4_STATIC_PREDICTIONS)"
	@test -f "$(GEMMA4_STATIC_REFERENCES)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.run --config "$(GEMMA4_STATIC_CONFIG)" --references "$(GEMMA4_STATIC_REFERENCES)" --predictions $(GEMMA4_STATIC_PREDICTIONS) --output $(GEMMA4_STATIC_METRICS)

gemma4-evaluate-static:
	@if [ -z "$(strip $(GEMMA4_STATIC_REFERENCES))" ]; then \
		echo "Evaluation refused before inference: set GEMMA4_STATIC_REFERENCES to the physically separate QA JSONL." >&2; \
		exit 2; \
	fi
	@test -f "$(GEMMA4_STATIC_REFERENCES)"
	@$(MAKE) --no-print-directory gemma4-predict-static \
		GEMMA4_STATIC_CONFIG="$(GEMMA4_STATIC_CONFIG)" \
		GEMMA4_STATIC_CHECKPOINT="$(GEMMA4_STATIC_CHECKPOINT)" \
		GEMMA4_STATIC_REFERENCES="$(GEMMA4_STATIC_REFERENCES)" \
		GEMMA4_EVAL_SPLIT="$(GEMMA4_EVAL_SPLIT)" \
		GEMMA4_QUESTIONS_MANIFEST="$(GEMMA4_QUESTIONS_MANIFEST)" \
		GEMMA4_STATIC_PREDICTIONS="$(GEMMA4_STATIC_PREDICTIONS)"
	@test -f "$(GEMMA4_STATIC_PREDICTIONS)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.run --config "$(GEMMA4_STATIC_CONFIG)" --references "$(GEMMA4_STATIC_REFERENCES)" --predictions $(GEMMA4_STATIC_PREDICTIONS) --output $(GEMMA4_STATIC_METRICS)

gemma4-predict-controls: gemma4-prepare-questions
	@test -x "$(GEMMA4_PYTHON)"
	@test -f "$(GEMMA4_STATIC_CONFIG)"
	@test -f "$(GEMMA4_STATIC_CHECKPOINT)/adapter.safetensors"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.control_predict --config "$(GEMMA4_STATIC_CONFIG)" --split $(GEMMA4_EVAL_SPLIT) --questions-manifest $(GEMMA4_QUESTIONS_MANIFEST) --checkpoint "$(GEMMA4_STATIC_CHECKPOINT)" --output-dir $(GEMMA4_CONTROL_PREDICTIONS)

gemma4-score-controls:
	@if [ -z "$(strip $(GEMMA4_CONTROL_REFERENCES))" ]; then echo "Set GEMMA4_CONTROL_REFERENCES to the physically separate evaluation QA JSONL." >&2; exit 2; fi
	@test -f "$(GEMMA4_CONTROL_PREDICTIONS)/manifest.json"
	@test -f "$(GEMMA4_CONTROL_REFERENCES)"
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.control_score --manifest "$(GEMMA4_CONTROL_PREDICTIONS)/manifest.json" --references "$(GEMMA4_CONTROL_REFERENCES)" --output-dir "$(GEMMA4_CONTROL_METRICS)"

gemma4-chat-static: require-gemma4-promoted
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.chat.cli --config "$(GEMMA4_STATIC_CONFIG)" --scene $(SCENE) --checkpoint "$(GEMMA4_STATIC_CHECKPOINT)"

generate-smoke-scene:
	$(BLENDER) --background --python-exit-code 1 --python blender/generate_scene.py -- --config $(GEMMA4_CONFIG) --scene $(SCENE)

render-smoke-scan: generate-smoke-scene
	$(BLENDER) --background data/oracle/$(SCENE)/scene.blend --python-exit-code 1 --python blender/render_scan.py -- --config $(GEMMA4_CONFIG) --scene $(SCENE)

legacy-generate-smoke-scene:
	$(BLENDER) --background --python-exit-code 1 --python blender/generate_scene.py -- --config $(CONFIG) --scene $(SCENE)

legacy-render-smoke-scan: legacy-generate-smoke-scene
	$(BLENDER) --background data/oracle/$(SCENE)/scene.blend --python-exit-code 1 --python blender/render_scan.py -- --config $(CONFIG) --scene $(SCENE)

generate-scene-batch:
	$(PYTHON) scripts/generate_scene_batch.py --config $(BATCH_CONFIG) --stage generate

render-scene-batch:
	$(PYTHON) scripts/generate_scene_batch.py --config $(BATCH_CONFIG) --stage render

multiscene-dry-run:
	$(PYTHON) scripts/generate_scene_batch.py --config $(BATCH_CONFIG) --stage all --dry-run

# Development-only data expansion. These targets name scenes 31--38
# explicitly and contain no switch capable of unlocking deferred scenes 25--30.
diverse28-dry-run:
	$(PYTHON) scripts/generate_scene_batch.py --config $(DIVERSE28_CONFIG) --stage all $(DIVERSE28_SCENE_ARGS) --dry-run

diverse28-generate-expansion:
	$(PYTHON) scripts/generate_scene_batch.py --config $(DIVERSE28_CONFIG) --stage generate $(DIVERSE28_SCENE_ARGS)

diverse28-render-expansion:
	$(PYTHON) scripts/generate_scene_batch.py --config $(DIVERSE28_CONFIG) --stage render $(DIVERSE28_SCENE_ARGS)

diverse28-generate-dataset:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.data.qa_generator --config $(DIVERSE28_CONFIG)

build-smoke-map:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.vision.encoder --config $(GEMMA4_CONFIG) --scene $(SCENE) --offline
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/build_map.py --config $(GEMMA4_CONFIG) --scene $(SCENE)

semantic-sanity:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.gemma4_semantic_sanity --config $(GEMMA4_CONFIG) --scene $(SCENE) --offline

generate-dataset:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.data.qa_generator --config $(GEMMA4_CONFIG)

train:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_adapter --config $(GEMMA4_CONFIG)

evaluate:
	@$(MAKE) --no-print-directory gemma4-evaluate-static \
		GEMMA4_STATIC_CONFIG="$(GEMMA4_STATIC_CONFIG)" \
		GEMMA4_STATIC_CHECKPOINT="$(GEMMA4_STATIC_CHECKPOINT)" \
		GEMMA4_STATIC_REFERENCES="$(GEMMA4_STATIC_REFERENCES)" \
		GEMMA4_EVAL_SPLIT="$(GEMMA4_EVAL_SPLIT)"

legacy-build-smoke-map:
	$(PYTHON) -m semantic_3d_chat.vision.encoder --config $(CONFIG) --scene $(SCENE) --offline
	$(PYTHON) scripts/build_map.py --config $(CONFIG) --scene $(SCENE)

legacy-semantic-sanity:
	$(PYTHON) -m semantic_3d_chat.evaluation.semantic_sanity --config $(CONFIG) --scene $(SCENE)

legacy-generate-dataset:
	$(PYTHON) -m semantic_3d_chat.data.qa_generator --config $(CONFIG)

legacy-train:
	$(PYTHON) -m semantic_3d_chat.training.train_adapter --config $(CONFIG)

# Historical scorer-only behavior is retained explicitly; callers must create
# the legacy predictions first instead of mistaking this for an inference run.
legacy-evaluate:
	@test -f reports/predictions/test.jsonl || { \
		echo "Legacy evaluation requires reports/predictions/test.jsonl; generate predictions explicitly before scoring." >&2; \
		exit 2; \
	}
	$(PYTHON) -m semantic_3d_chat.evaluation.run --config $(CONFIG)

evaluate-oracle-text:
	$(PYTHON) -m semantic_3d_chat.evaluation.oracle_text_baseline --config $(CONFIG)
	$(PYTHON) -m semantic_3d_chat.evaluation.run --config $(CONFIG) --references data/qa/test.jsonl --predictions reports/predictions/oracle_text.jsonl --output reports/metrics/oracle_text.json

evaluate-direct-images:
	$(PYTHON) -m semantic_3d_chat.evaluation.direct_multiview_baseline --config $(CONFIG)
	$(PYTHON) -m semantic_3d_chat.evaluation.run --config $(CONFIG) --references data/qa/test.jsonl --predictions reports/predictions/direct_multiview.jsonl --output reports/metrics/direct_multiview.json

# PROHIBITED PRIMARY SUBSTITUTE: direct RGB-image control only. The smoke limit
# restricts questions, never views; each question still receives all 24 complete
# deterministic validation frames. Loading is fail-closed to the cached checkpoint.
GEMMA4_DIRECT_BASELINE_CONFIG ?= configs/experiments/gemma4_diverse20_direct_multiview_baseline.yaml
GEMMA4_DIRECT_BASELINE_SMOKE_LIMIT ?= 2
GEMMA4_DIRECT_BASELINE_REFERENCES := data_diverse20/qa/validation.jsonl
GEMMA4_DIRECT_BASELINE_SMOKE_REFERENCES := reports/gemma4/references/direct_multiview_diverse20_validation_smoke.jsonl
GEMMA4_DIRECT_BASELINE_SMOKE_PREDICTIONS := reports/gemma4/predictions/direct_multiview_diverse20_validation_smoke.jsonl
GEMMA4_DIRECT_BASELINE_SMOKE_METRICS := reports/gemma4/metrics/direct_multiview_diverse20_validation_smoke.json
GEMMA4_DIRECT_BASELINE_VALIDATION_PREDICTIONS := reports/gemma4/predictions/direct_multiview_diverse20_validation.jsonl
GEMMA4_DIRECT_BASELINE_VALIDATION_METRICS := reports/gemma4/metrics/direct_multiview_diverse20_validation.json

evaluate-direct-images-gemma4-smoke:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.direct_multiview_baseline --config "$(GEMMA4_DIRECT_BASELINE_CONFIG)" --references "$(GEMMA4_DIRECT_BASELINE_REFERENCES)" --output "$(GEMMA4_DIRECT_BASELINE_SMOKE_PREDICTIONS)" --limit $(GEMMA4_DIRECT_BASELINE_SMOKE_LIMIT) --selected-references-output "$(GEMMA4_DIRECT_BASELINE_SMOKE_REFERENCES)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.run --config "$(GEMMA4_DIRECT_BASELINE_CONFIG)" --references "$(GEMMA4_DIRECT_BASELINE_SMOKE_REFERENCES)" --predictions "$(GEMMA4_DIRECT_BASELINE_SMOKE_PREDICTIONS)" --output "$(GEMMA4_DIRECT_BASELINE_SMOKE_METRICS)"

evaluate-direct-images-gemma4-validation:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.direct_multiview_baseline --config "$(GEMMA4_DIRECT_BASELINE_CONFIG)" --references "$(GEMMA4_DIRECT_BASELINE_REFERENCES)" --output "$(GEMMA4_DIRECT_BASELINE_VALIDATION_PREDICTIONS)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.run --config "$(GEMMA4_DIRECT_BASELINE_CONFIG)" --references "$(GEMMA4_DIRECT_BASELINE_REFERENCES)" --predictions "$(GEMMA4_DIRECT_BASELINE_VALIDATION_PREDICTIONS)" --output "$(GEMMA4_DIRECT_BASELINE_VALIDATION_METRICS)"

# PROHIBITED PRIMARY SUBSTITUTE: exact oracle text supplied only to the isolated
# evaluation baseline. The production chat package never imports this module.
.PHONY: evaluate-oracle-text-gemma4-validation
GEMMA4_ORACLE_BASELINE_CONFIG ?= configs/experiments/diverse20.yaml
GEMMA4_ORACLE_BASELINE_REFERENCES := data_diverse20/qa/validation.jsonl
GEMMA4_ORACLE_BASELINE_PREDICTIONS := reports/gemma4/predictions/oracle_text_diverse20_validation.jsonl
GEMMA4_ORACLE_BASELINE_METRICS := reports/gemma4/metrics/oracle_text_diverse20_validation.json

evaluate-oracle-text-gemma4-validation:
	./scripts/run_oracle_text_upper_bound_v55.sh
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.run --config "$(GEMMA4_ORACLE_BASELINE_CONFIG)" --references "$(GEMMA4_ORACLE_BASELINE_REFERENCES)" --predictions "$(GEMMA4_ORACLE_BASELINE_PREDICTIONS)" --output "$(GEMMA4_ORACLE_BASELINE_METRICS)"

# Preserved legacy CLIP/Qwen runtime. The unqualified chat target below uses
# the promoted strict V89 direct-memory release instead.
legacy-chat:
	@if [ "$$(PYTHONPATH=src $(PYTHON) -c 'import sys; from semantic_3d_chat.config import load_config; print(load_config(sys.argv[1]).get("language", {}).get("backend", "auto"))' "$(CONFIG)")" = "gemma4" ]; then \
		echo "legacy-chat accepts only the preserved CLIP/Qwen configuration." >&2; \
		exit 2; \
	fi
	@checkpoint="$(CHECKPOINT)"; \
	if [ -z "$$checkpoint" ]; then \
		checkpoint="$$( $(PYTHON) scripts/demo_check.py --config "$(CONFIG)" --scene "$(SCENE)" --resolve-checkpoint )" || exit $$?; \
	fi; \
	$(PYTHON) -m semantic_3d_chat.chat.cli --config "$(CONFIG)" --scene "$(SCENE)" --checkpoint "$$checkpoint"

chat: v89-demo-chat

legacy-web:
	@if [ "$$(PYTHONPATH=src $(PYTHON) -c 'import sys; from semantic_3d_chat.config import load_config; print(load_config(sys.argv[1]).get("language", {}).get("backend", "auto"))' "$(CONFIG)")" = "gemma4" ]; then \
		echo "legacy-web accepts only the preserved CLIP/Qwen configuration." >&2; \
		exit 2; \
	fi
	@checkpoint="$(CHECKPOINT)"; \
	if [ -z "$$checkpoint" ]; then \
		checkpoint="$$( $(PYTHON) scripts/demo_check.py --config "$(CONFIG)" --scene "$(SCENE)" --resolve-checkpoint )" || exit $$?; \
	fi; \
	$(PYTHON) -m semantic_3d_chat.chat.web_app --config "$(CONFIG)" --scene "$(SCENE)" --checkpoint "$$checkpoint"

# Human-facing point-map UI over the strict V54 comparator. V89 is the default
# answer runtime; its interactive CLI is `make chat`.
web: strict-web

legacy-robot:
	$(PYTHON) -m semantic_3d_chat.robot.agent_loop --config $(CONFIG) --scene $(SCENE)

robot: gemma4-embodied-chat-learned

robot-evaluate:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.robot_benchmark --config $(GEMMA4_CONFIG) --scene $(SCENE)

legacy-robot-evaluate:
	$(PYTHON) -m semantic_3d_chat.evaluation.robot_benchmark --config $(CONFIG) --scene $(SCENE)

# Offline generation-side operation: this is the only target below that opens
# the oracle-authored Blender source. The resulting opaque runtime asset is
# authenticated and thereafter opened without the oracle tree.
export-runtime-scene:
	@mkdir -p "$(dir $(RUNTIME_SCENE_ASSET))"
	$(BLENDER) --background --disable-autoexec "data/oracle/$(SCENE)/scene.blend" --python-exit-code 1 --python blender/export_runtime_scene.py -- --scene "$(SCENE)" --output "$(RUNTIME_SCENE_ASSET)"

gemma4-create-robot-state-checkpoint:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/create_robot_state_checkpoint.py \
		--output "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		--output-dim 1536 \
		--hidden-dim 256 \
		--token-count 4 \
		--seed 20260812 \
		--output-scale 0.02 \
		--report reports/gemma4/metrics/robot_state_numeric_v1.json

gemma4-embodied-smoke:
	@test -f "$(RUNTIME_SCENE_ASSET)" || { echo "Missing sanitized runtime asset; run make export-runtime-scene first." >&2; exit 2; }
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/run_embodied_runtime_smoke.py --config "$(GEMMA4_EMBODIED_CONFIG)" --checkpoint "$(GEMMA4_EMBODIED_CHECKPOINT)" --asset "$(RUNTIME_SCENE_ASSET)" --robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" --scene "$(SCENE)" --output "$(GEMMA4_EMBODIED_SMOKE_REPORT)"

gemma4-semantic-navigation:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/run_semantic_navigation_benchmark.py --config "$(GEMMA4_EMBODIED_CONFIG)" --scene "$(SCENE)" --runtime-output "$(GEMMA4_SEMANTIC_NAV_RUNTIME_REPORT)" --output "$(GEMMA4_SEMANTIC_NAV_REPORT)"

navigation-policy-generate-traces:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/generate_navigation_policy_traces.py --config "$(NAVIGATION_POLICY_CONFIG)"

navigation-policy-train:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/train_navigation_policy.py --config "$(NAVIGATION_POLICY_CONFIG)"

navigation-policy-evaluate:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/evaluate_navigation_policy.py --config "$(NAVIGATION_POLICY_CONFIG)" --checkpoint "$(NAVIGATION_POLICY_CHECKPOINT)"

navigation-policy-controls:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/evaluate_navigation_policy_controls.py --config "$(NAVIGATION_POLICY_CONFIG)" --checkpoint "$(NAVIGATION_POLICY_CHECKPOINT)" --output "$(NAVIGATION_POLICY_CONTROLS_OUTPUT)"

navigation-policy-audit:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/audit_navigation_policy_runtime.py --config "$(NAVIGATION_POLICY_CONFIG)" --checkpoint "$(NAVIGATION_POLICY_CHECKPOINT)" --output "$(NAVIGATION_POLICY_AUDIT_OUTPUT)"

# Oracle-free verification that every bounded decision consumed the current
# complete scene prefix and numeric robot tokens, including after a map scan.
navigation-policy-context-audit:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/audit_navigation_continuous_context.py --journal "$(NAVIGATION_CONTEXT_JOURNAL)" --output "$(NAVIGATION_CONTEXT_AUDIT_OUTPUT)"

navigation-policy-benchmark:
	NAVIGATION_RUN_ID="$(NAVIGATION_POLICY_RUN_ID)" NAVIGATION_POLICY_CHECKPOINT="$(NAVIGATION_POLICY_CHECKPOINT)" NAVIGATION_EMBODIED_CONFIG="$(NAVIGATION_EMBODIED_CONFIG)" NAVIGATION_TASKS="$(NAVIGATION_TASKS)" NAVIGATION_SCORING_SPEC="$(NAVIGATION_SCORING_SPEC)" ./scripts/run_learned_navigation_benchmark.sh

# Read-only, hash-pinned display of the strongest completed V2 run. This opens
# no oracle/QA/training tree and never rewrites the sealed journal or metrics.
navigation-policy-v2-demo:
	NAVIGATION_RUN_ID="learned_v2" NAVIGATION_POLICY_CHECKPOINT="data_gemma4/checkpoints/navigation_policy_v2" NAVIGATION_EMBODIED_CONFIG="configs/runtime/embodied_navigation_v2.yaml" NAVIGATION_TASKS="configs/benchmarks/llm_navigation_v2_scene_000001.json" NAVIGATION_SCORING_SPEC="configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json" ./scripts/run_learned_navigation_benchmark.sh --check

# Read-only authentication of the exact historical V3 source snapshot, dataset,
# checkpoint, controls, oracle-removal audit, sealed journal, score, and plot.
# This intentionally makes no compatibility claim about current successor code.
navigation-policy-v3-demo:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.navigation_policy_v3_evidence

# V3.1 keeps the accepted V3 checkpoint and changes only the bounded numeric
# runtime grammar for scan-then-approach terminal goals.  Its live outputs use
# new paths; the immutable V3 journal and score are inputs, never overwritten.
navigation-policy-v3-1-preregister:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.navigation_policy_v3_1_preregistration preregister --output "$(NAVIGATION_V3_1_PREREGISTRATION)"

navigation-policy-v3-1-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.navigation_policy_v3_1_preregistration authenticate --preregistration "$(NAVIGATION_V3_1_PREREGISTRATION)"

navigation-policy-v3-1-benchmark: navigation-policy-v3-1-authenticate
	NAVIGATION_RUN_ID="learned_v3_1" NAVIGATION_POLICY_VERSION="3" NAVIGATION_POLICY_CHECKPOINT="data_gemma4/checkpoints/navigation_policy_v3" NAVIGATION_EMBODIED_CONFIG="configs/runtime/embodied_navigation_v2.yaml" NAVIGATION_TASKS="configs/benchmarks/llm_navigation_v2_scene_000001.json" NAVIGATION_SCORING_SPEC="configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json" ./scripts/run_learned_navigation_benchmark.sh
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/audit_navigation_continuous_context.py --journal "reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3_1.json" --output "$(NAVIGATION_V3_1_CONTEXT_AUDIT)"
	$(MAKE) navigation-policy-v3-1-result

navigation-policy-v3-1-result:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.navigation_policy_v3_1_preregistration result --preregistration "$(NAVIGATION_V3_1_PREREGISTRATION)" --output "$(NAVIGATION_V3_1_RESULT)"

# Read-only authentication of the sole preregistered V3.3 development run.
# This recomputes every gate from the sealed journal, inference audit, numeric
# score, and continuous-context audit. It loads neither Gemma nor Blender and
# never opens oracle or QA files.
navigation-policy-v3-3-check:
	@test -f "reports/gemma4/metrics/navigation_policy_v3_3_runtime_preregistration.json" || { echo "Missing sealed V3.3 preregistration evidence." >&2; exit 2; }
	@test -f "reports/gemma4/metrics/navigation_policy_v3_3_runtime_acceptance.json" || { echo "Missing sealed V3.3 acceptance result; the create-once benchmark has not completed." >&2; exit 2; }
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.navigation_policy_v3_3_preregistration \
		authenticate-result \
		--preregistration reports/gemma4/metrics/navigation_policy_v3_3_runtime_preregistration.json \
		--result reports/gemma4/metrics/navigation_policy_v3_3_runtime_acceptance.json

# Read-only authentication of the terminal preregistered V4.1 negative result.
# The sole arm passed 13/14 gates, published no checkpoint, and never ran live.
navigation-policy-v4-1-result:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.navigation_policy_v41_result

gemma4-embodied-chat:
	EMBODIED_CONFIG="$(GEMMA4_EMBODIED_CONFIG)" EMBODIED_CONTROL_CONFIG="$(GEMMA4_EMBODIED_CONTROL_CONFIG)" EMBODIED_SCENE="$(SCENE)" EMBODIED_BASE_CHECKPOINT="$(GEMMA4_EMBODIED_CHECKPOINT)" EMBODIED_CONTROL_CHECKPOINT="$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" EMBODIED_RUNTIME_ASSET="$(RUNTIME_SCENE_ASSET)" EMBODIED_ROBOT_STATE_CHECKPOINT="$(GEMMA4_ROBOT_STATE_CHECKPOINT)" ./scripts/run_embodied_conversation.sh

gemma4-embodied-chat-llm:
	EMBODIED_CONFIG="$(GEMMA4_EMBODIED_CONFIG)" EMBODIED_CONTROL_CONFIG="$(GEMMA4_EMBODIED_CONTROL_CONFIG)" EMBODIED_SCENE="$(SCENE)" EMBODIED_BASE_CHECKPOINT="$(GEMMA4_EMBODIED_CHECKPOINT)" EMBODIED_CONTROL_CHECKPOINT="$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" EMBODIED_RUNTIME_ASSET="$(RUNTIME_SCENE_ASSET)" EMBODIED_ROBOT_STATE_CHECKPOINT="$(GEMMA4_ROBOT_STATE_CHECKPOINT)" ./scripts/run_embodied_conversation.sh --llm-tool-policy

# Current task-trained conversational target. V3 grounds user-supplied target
# phrases against every active-map voxel and then runs a bounded refreshed loop.
gemma4-embodied-chat-learned:
	EMBODIED_CONFIG="$(GEMMA4_EMBODIED_CONFIG)" EMBODIED_CONTROL_CONFIG="$(GEMMA4_EMBODIED_CONTROL_CONFIG)" EMBODIED_SCENE="$(SCENE)" EMBODIED_BASE_CHECKPOINT="$(GEMMA4_EMBODIED_CHECKPOINT)" EMBODIED_CONTROL_CHECKPOINT="$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" EMBODIED_RUNTIME_ASSET="$(RUNTIME_SCENE_ASSET)" EMBODIED_ROBOT_STATE_CHECKPOINT="$(GEMMA4_ROBOT_STATE_CHECKPOINT)" EMBODIED_NAVIGATION_POLICY_CHECKPOINT="$(GEMMA4_NAVIGATION_POLICY_CHECKPOINT)" EMBODIED_NAVIGATION_POLICY_VERSION="$(GEMMA4_NAVIGATION_POLICY_VERSION)" ./scripts/run_embodied_conversation.sh --navigation-max-steps "$(GEMMA4_LEARNED_NAVIGATION_MAX_STEPS)"

# Read-only finite validation: authenticates the sanitized policy and exact
# local-model binding without loading Gemma/Blender or changing map/robot state.
gemma4-embodied-chat-learned-check:
	EMBODIED_CONFIG="$(GEMMA4_EMBODIED_CONFIG)" EMBODIED_CONTROL_CONFIG="$(GEMMA4_EMBODIED_CONTROL_CONFIG)" EMBODIED_SCENE="$(SCENE)" EMBODIED_BASE_CHECKPOINT="$(GEMMA4_EMBODIED_CHECKPOINT)" EMBODIED_CONTROL_CHECKPOINT="$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" EMBODIED_RUNTIME_ASSET="$(RUNTIME_SCENE_ASSET)" EMBODIED_ROBOT_STATE_CHECKPOINT="$(GEMMA4_ROBOT_STATE_CHECKPOINT)" EMBODIED_NAVIGATION_POLICY_CHECKPOINT="$(GEMMA4_NAVIGATION_POLICY_CHECKPOINT)" EMBODIED_NAVIGATION_POLICY_VERSION="$(GEMMA4_NAVIGATION_POLICY_VERSION)" ./scripts/run_embodied_conversation.sh --check

# Finite real inference over one configurable natural-language instruction.
gemma4-embodied-chat-learned-once:
	EMBODIED_CONFIG="$(GEMMA4_EMBODIED_CONFIG)" EMBODIED_CONTROL_CONFIG="$(GEMMA4_EMBODIED_CONTROL_CONFIG)" EMBODIED_SCENE="$(SCENE)" EMBODIED_BASE_CHECKPOINT="$(GEMMA4_EMBODIED_CHECKPOINT)" EMBODIED_CONTROL_CHECKPOINT="$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" EMBODIED_RUNTIME_ASSET="$(RUNTIME_SCENE_ASSET)" EMBODIED_ROBOT_STATE_CHECKPOINT="$(GEMMA4_ROBOT_STATE_CHECKPOINT)" EMBODIED_NAVIGATION_POLICY_CHECKPOINT="$(GEMMA4_NAVIGATION_POLICY_CHECKPOINT)" EMBODIED_NAVIGATION_POLICY_VERSION="$(GEMMA4_NAVIGATION_POLICY_VERSION)" ./scripts/run_embodied_conversation.sh --human --navigation-max-steps "$(GEMMA4_LEARNED_NAVIGATION_MAX_STEPS)" --command "$(GEMMA4_LEARNED_NAVIGATION_COMMAND)" --result-output "$(GEMMA4_LEARNED_NAVIGATION_RESULT)"

# Evaluation-only oracle scorer. Runtime artifacts are fully validated before
# either oracle file is opened, and oracle content never enters chat inference.
embodied-approach-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.embodied_approach_score --case scene_000001=reports/gemma4/metrics/embodied_conversation_hybrid_approach_chair_v2_scene_000001.json --case scene_000031=reports/gemma4/metrics/embodied_conversation_hybrid_approach_chair_v2_scene_000031.json --oracle-root data/oracle --target-category chair --output reports/gemma4/metrics/embodied_navigation_hybrid_approach_oracle_score.json

# Successor score keeps the V2 1/2 collision failure immutable and separately
# scores the collision-limited safe-stop implementation on the same two scenes.
embodied-approach-v3-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.embodied_approach_score --case scene_000001=reports/gemma4/metrics/embodied_conversation_hybrid_approach_chair_v3_scene_000001.json --case scene_000031=reports/gemma4/metrics/embodied_conversation_hybrid_approach_chair_v3_scene_000031.json --oracle-root data/oracle --target-category chair --output reports/gemma4/metrics/embodied_navigation_hybrid_approach_v3_oracle_score.json

# Deterministic post-hoc visualization of the two hash-pinned runtime results.
# The plotter opens only those result JSON files: no oracle, QA, map, or model.
embodied-approach-v3-trajectories:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.embodied_approach_v3_trajectories --figure reports/gemma4/figures/embodied_approach_v3_trajectories.png --output reports/gemma4/examples/embodied_approach_v3_trajectories.json

gemma4-embodied-mcp:
	@test -f "$(RUNTIME_SCENE_ASSET)" || { echo "Missing sanitized runtime asset; run make export-runtime-scene first." >&2; exit 2; }
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.mcp_server.server --config "$(GEMMA4_EMBODIED_CONFIG)" --scene "$(SCENE)" --checkpoint "$(GEMMA4_EMBODIED_CHECKPOINT)" $(if $(strip $(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)),--control-checkpoint "$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" --control-runtime-config "$(GEMMA4_EMBODIED_CONTROL_CONFIG)",) --runtime-asset "$(RUNTIME_SCENE_ASSET)" --robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)"

# Finite semantic-runtime preflight. It authenticates the high-dimensional map,
# exact V54 release, sanitized Blender asset, numeric robot-state encoder, nine
# bounded MCP tools, and blocking file audit without loading Gemma or Blender,
# starting a transport, or changing robot/map state.
gemma4-embodied-mcp-check:
	@test -f "$(RUNTIME_SCENE_ASSET)" || { echo "Missing sanitized runtime asset; run make export-runtime-scene first." >&2; exit 2; }
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.mcp_server.server \
		--config "$(GEMMA4_EMBODIED_CONFIG)" \
		--scene "$(SCENE)" \
		--checkpoint "$(GEMMA4_EMBODIED_CHECKPOINT)" \
		$(if $(strip $(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)),--control-checkpoint "$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" --control-runtime-config "$(GEMMA4_EMBODIED_CONTROL_CONFIG)",) \
		--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
		--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		--audit-report "$(GEMMA4_EMBODIED_MCP_CHECK_REPORT)" \
		--check

# Explicit V96 embodied bridge over candidate-format inputs. Before any model
# load, isolated children must authenticate known-development, deferred-final,
# and the promoted six-scene runtime-leakage release. This target does not alter
# the default MCP/runtime variables and refuses to fall back to V54/V89.
v96-explicit-candidate-embodied-mcp-check:
	@test -f "$(RUNTIME_SCENE_ASSET)" || { echo "Missing sanitized runtime asset; run make export-runtime-scene first." >&2; exit 2; }
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.mcp_server.server \
		--config "$(GEMMA4_EMBODIED_CONFIG)" \
		--scene "$(SCENE)" \
		--checkpoint "$(GEMMA4_V96_MCP_BASE_CHECKPOINT)" \
		--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
		--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		--v96-candidate-bridge-hook "$(GEMMA4_V96_MCP_BRIDGE_HOOK)" \
		--v96-scene-memory "$(GEMMA4_V96_MCP_SCENE_MEMORY)" \
		--allow-explicit-v96-candidate \
		--persistent-map "$(GEMMA4_V96_MCP_PERSISTENT_MAP)" \
		--scan-output-directory "$(GEMMA4_V96_MCP_SCAN_OUTPUT)" \
		--audit-report "$(GEMMA4_V96_MCP_CHECK_REPORT)" \
		--check

# Heavy local server. MCP exposes exactly the same nine numeric tools; scans
# transactionally compile a new complete 738-token memory before map commit.
# Direct V96 answer generation with the additional robot-state tokens remains
# deliberately disabled until a separate live layout test authenticates it.
v96-explicit-candidate-embodied-mcp:
	@test -f "$(RUNTIME_SCENE_ASSET)" || { echo "Missing sanitized runtime asset; run make export-runtime-scene first." >&2; exit 2; }
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.mcp_server.server \
		--config "$(GEMMA4_EMBODIED_CONFIG)" \
		--scene "$(SCENE)" \
		--checkpoint "$(GEMMA4_V96_MCP_BASE_CHECKPOINT)" \
		--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
		--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		--v96-candidate-bridge-hook "$(GEMMA4_V96_MCP_BRIDGE_HOOK)" \
		--v96-scene-memory "$(GEMMA4_V96_MCP_SCENE_MEMORY)" \
		--allow-explicit-v96-candidate \
		--persistent-map "$(GEMMA4_V96_MCP_PERSISTENT_MAP)" \
		--scan-output-directory "$(GEMMA4_V96_MCP_SCAN_OUTPUT)" \
		--audit-report "$(GEMMA4_V96_MCP_LIVE_AUDIT)"

# Finite official-SDK proof over one promoted held-out scene. The runner and
# server independently authenticate the promoted V96 release before transport
# or model load, then call only get_robot_state, scan, and turn. It exercises
# transactional numeric 738-token refresh and deliberately makes no claim
# about direct V96 answers with the additional robot-state tokens.
v96-explicit-candidate-embodied-mcp-live-smoke: SCENE = scene_000025
v96-explicit-candidate-embodied-mcp-live-smoke: GEMMA4_V96_MCP_SCENE_MEMORY = reports/gemma4/artifacts/v95_deferred_final/memory_cache/$(SCENE)
v96-explicit-candidate-embodied-mcp-live-smoke: v96-release-verify
	@test -f "$(RUNTIME_SCENE_ASSET)" || { echo "Missing sanitized runtime asset; run make export-runtime-scene SCENE=$(SCENE) first." >&2; exit 2; }
	@test -d "$(GEMMA4_V96_MCP_SCENE_MEMORY)" || { echo "Missing authenticated held-out V96 source memory: $(GEMMA4_V96_MCP_SCENE_MEMORY)" >&2; exit 2; }
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		scripts/run_v96_candidate_mcp_live_smoke.py \
		--config "$(GEMMA4_EMBODIED_CONFIG)" \
		--scene "$(SCENE)" \
		--base-checkpoint "$(GEMMA4_V96_MCP_BASE_CHECKPOINT)" \
		--bridge-hook "$(GEMMA4_V96_MCP_BRIDGE_HOOK)" \
		--scene-memory "$(GEMMA4_V96_MCP_SCENE_MEMORY)" \
		--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
		--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		--audit-report "$(GEMMA4_V96_MCP_LIVE_AUDIT)" \
		--output "$(GEMMA4_V96_MCP_LIVE_RESULT)" \
		--python "$(GEMMA4_PYTHON)"

# Heavy finite proof over the official MCP stdio transport. This loads local
# Gemma 4 and V75, renders two complete RGB-D observations with Blender, fuses
# them into the persistent high-dimensional map, and verifies that every
# continuous scene/controller/robot binding changes without textual leakage.
gemma4-embodied-mcp-live-smoke:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/run_semantic_mcp_live_smoke.py \
		--config "$(GEMMA4_EMBODIED_CONFIG)" \
		--scene "$(SCENE)" \
		--base-checkpoint "$(GEMMA4_EMBODIED_CHECKPOINT)" \
		--control-checkpoint "$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" \
		--control-runtime-config "$(GEMMA4_EMBODIED_CONTROL_CONFIG)" \
		--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
		--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)"

# Finite no-model preflight for the natural-language MCP loop. The dependency
# separately authenticates the production semantic MCP server. This command
# validates the target grammar, sanitized assets, selective local Gemma text
# encoder, and explicitly records that neither Gemma function calling nor the
# learned V3 action head is claimed by this integration path.
gemma4-embodied-mcp-conversation-check: gemma4-embodied-mcp-check
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.robot.conversational_mcp_agent \
		--config "$(GEMMA4_EMBODIED_CONFIG)" \
		--scene "$(SCENE)" \
		--instruction "$(GEMMA4_LEARNED_NAVIGATION_COMMAND)" \
		--base-checkpoint "$(GEMMA4_EMBODIED_CHECKPOINT)" \
		--control-checkpoint "$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" \
		--control-runtime-config "$(GEMMA4_EMBODIED_CONTROL_CONFIG)" \
		--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
		--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		--max-steps "$(GEMMA4_LEARNED_NAVIGATION_MAX_STEPS)" \
		--output "$(GEMMA4_CONVERSATIONAL_MCP_PREFLIGHT)" \
		--check

# Real end-to-end process-boundary proof: user instruction -> selective-Gemma
# all-voxel grounding -> V3 numeric convergence decisions -> official MCP
# stdio scan/turn/stop calls -> RGB-D map fusion and continuous-prefix refresh.
gemma4-embodied-mcp-conversation: gemma4-embodied-mcp-conversation-check
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.robot.conversational_mcp_agent \
		--config "$(GEMMA4_EMBODIED_CONFIG)" \
		--scene "$(SCENE)" \
		--instruction "$(GEMMA4_LEARNED_NAVIGATION_COMMAND)" \
		--base-checkpoint "$(GEMMA4_EMBODIED_CHECKPOINT)" \
		--control-checkpoint "$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" \
		--control-runtime-config "$(GEMMA4_EMBODIED_CONTROL_CONFIG)" \
		--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
		--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		--max-steps "$(GEMMA4_LEARNED_NAVIGATION_MAX_STEPS)" \
		--output "$(GEMMA4_CONVERSATIONAL_MCP_RESULT)" \
		--client-audit-report "$(GEMMA4_CONVERSATIONAL_MCP_CLIENT_AUDIT)" \
		--server-audit-report "$(GEMMA4_CONVERSATIONAL_MCP_SERVER_AUDIT)"

# Physically separate post-run oracle scorer. Runtime evidence and both process
# audits are validated before this evaluator opens the oracle/task files.
gemma4-embodied-mcp-conversation-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.conversational_mcp_face_score \
		--runtime-result "$(GEMMA4_CONVERSATIONAL_MCP_RESULT)" \
		--scene-oracle "data/oracle/$(SCENE)/oracle.json" \
		--scoring-spec "configs/benchmarks/oracle/llm_navigation_v2_$(SCENE).json" \
		--output "$(GEMMA4_CONVERSATIONAL_MCP_SCORE)"

# Model-free, renderer-free preflight for the persistent conversational entry
# point. The dependency authenticates the same production MCP assets and tool
# inventory without starting stdio or changing robot/map state.
embodied-demo-check: gemma4-embodied-mcp-check
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.robot.conversational_mcp_agent \
		--config "$(GEMMA4_EMBODIED_CONFIG)" \
		--scene "$(SCENE)" \
		--interactive \
		--base-checkpoint "$(GEMMA4_EMBODIED_CHECKPOINT)" \
		--control-checkpoint "$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" \
		--control-runtime-config "$(GEMMA4_EMBODIED_CONTROL_CONFIG)" \
		--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
		--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		--max-steps "$(GEMMA4_LEARNED_NAVIGATION_MAX_STEPS)" \
		--output "$(GEMMA4_EMBODIED_DEMO_PREFLIGHT)" \
		--check

# Deterministic finite heavy smoke over one stdio session. It exercises both
# semantic target skills, an explicit scan, a read-only state query, and the
# safety-latched standalone stop while keeping its artifacts separate from the
# interactive demo.
embodied-demo-smoke: embodied-demo-check
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.robot.conversational_mcp_agent \
		--config "$(GEMMA4_EMBODIED_CONFIG)" \
		--scene "$(SCENE)" \
		--command "Face the chair, then stop." \
		--command "Move closer to the chair, then stop." \
		--command "scan" \
		--command "get robot state" \
		--command "stop" \
		--base-checkpoint "$(GEMMA4_EMBODIED_CHECKPOINT)" \
		--control-checkpoint "$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" \
		--control-runtime-config "$(GEMMA4_EMBODIED_CONTROL_CONFIG)" \
		--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
		--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		--max-steps "$(GEMMA4_LEARNED_NAVIGATION_MAX_STEPS)" \
		--output "$(GEMMA4_EMBODIED_DEMO_SMOKE_RESULT)" \
		--client-audit-report "$(GEMMA4_EMBODIED_DEMO_SMOKE_CLIENT_AUDIT)" \
		--server-audit-report "$(GEMMA4_EMBODIED_DEMO_SMOKE_SERVER_AUDIT)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.conversational_mcp_session_inspect \
		--runtime-result "$(GEMMA4_EMBODIED_DEMO_SMOKE_RESULT)" \
		--output "$(GEMMA4_EMBODIED_DEMO_SMOKE_INSPECTION)"

# Re-authenticate an existing finite smoke without loading models, maps,
# renderer state, training/QA data, or oracle geometry.
embodied-demo-smoke-inspect:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.conversational_mcp_session_inspect \
		--runtime-result "$(GEMMA4_EMBODIED_DEMO_SMOKE_RESULT)" \
		--output "$(GEMMA4_EMBODIED_DEMO_SMOKE_INSPECTION)"

# Physically separate evaluation-only geometry score. The scorer first
# recomputes/authenticates the model-free inspection and both runtime audits;
# only then may it open the oracle/spec files below.
embodied-demo-smoke-score: embodied-demo-smoke-inspect
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.conversational_mcp_session_oracle_score \
		--runtime-result "$(GEMMA4_EMBODIED_DEMO_SMOKE_RESULT)" \
		--inspection-result "$(GEMMA4_EMBODIED_DEMO_SMOKE_INSPECTION)" \
		--scene-oracle "data/oracle/$(SCENE)/oracle.json" \
		--scoring-spec "$(GEMMA4_EMBODIED_DEMO_SMOKE_SCORING_SPEC)" \
		--output "$(GEMMA4_EMBODIED_DEMO_SMOKE_SCORE)"

# One persistent official-MCP stdio process for repeated face, approach, scan,
# state, and explicit-stop commands. Goal convergence is stationary/unlatched;
# standalone stop is safety-latched and ends further motion for the episode.
embodied-demo: embodied-demo-check
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.robot.conversational_mcp_agent \
		--config "$(GEMMA4_EMBODIED_CONFIG)" \
		--scene "$(SCENE)" \
		--interactive \
		--base-checkpoint "$(GEMMA4_EMBODIED_CHECKPOINT)" \
		--control-checkpoint "$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" \
		--control-runtime-config "$(GEMMA4_EMBODIED_CONTROL_CONFIG)" \
		--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
		--robot-state-checkpoint "$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		--max-steps "$(GEMMA4_LEARNED_NAVIGATION_MAX_STEPS)" \
		--output "$(GEMMA4_EMBODIED_DEMO_RESULT)" \
		--client-audit-report "$(GEMMA4_EMBODIED_DEMO_CLIENT_AUDIT)" \
		--server-audit-report "$(GEMMA4_EMBODIED_DEMO_SERVER_AUDIT)"

# Complete model-free embodied implementation gate. The historical V3.3 result
# remains preserved as an experiment, but its old source seal does not
# authenticate the newer runtime sources. The current gate therefore verifies
# the strict promoted-release wrapper and the production numeric-only MCP
# surface without making a current navigation-success claim.
embodied-check: v96-embodied-check embodied-demo-check
	@echo "Embodied implementation readiness: PASS (strict continuous-prefix bridge and official MCP 2.0 numeric-only surface); held-out navigation acceptance: PENDING a promoted successor static release."

legacy-mcp:
	$(PYTHON) -m semantic_3d_chat.mcp_server.server --config $(CONFIG) --scene $(SCENE)

mcp: gemma4-embodied-mcp

# Finite official-SDK stdio integration test. This starts the real local MCP
# server, lists all numeric tools, executes one bounded turn, verifies rejected
# calls are state-preserving, resets the scene, and scans tool receipts for
# semantic-label leakage.
mcp-stdio-smoke:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.mcp_transport_smoke \
		--config "$(CONFIG)" \
		--scene "$(SCENE)" \
		--output "$(MCP_STDIO_SMOKE_OUTPUT)"

current-report:
	$(PYTHON) scripts/build_current_report.py --output "$(CURRENT_METRICS)" --markdown-output "$(CURRENT_REPORT)"

v75-official-validation-figures:
	$(PYTHON) scripts/plot_v75_official_validation.py

v78-grounding-internal-held-figure:
	$(PYTHON) scripts/plot_v78_grounding_internal_held.py

v85-development-figure:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v85_development_figure

v86-accuracy-figure:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v86_accuracy_figure

v87-accuracy-figure:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v87_accuracy_figure

v88-accuracy-figure:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v88_accuracy_figure

# Reproduce the sealed 94-row historical-held V78 score without full Gemma
# inference, then overlay deterministic predictions/targets on sanitized RGB
# point clouds. Oracle target coordinates are evaluation-only and the command
# refuses to write its report or figure below the runtime data tree.
v78-grounding-held-pointcloud:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v78_grounding_pointcloud

v75-fixed-atlas-mechanism-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/check_v75_fixed_atlas_mechanism.py

# Historical-internal, no-promotion V75 fixed-prefix atlas diagnostic. The
# preflight authenticates numeric artifacts without loading Gemma. Prepare,
# predictor, and scorer are no-overwrite by construction; `full` is therefore
# an explicit one-shot pipeline, not a cache-overwriting convenience target.
v75-fixed-atlas-behavior-prepare:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/prepare_v75_fixed_atlas_behavior.py

v75-fixed-atlas-behavior-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/check_v75_fixed_atlas_behavior.py

v75-fixed-atlas-behavior-predict: v75-fixed-atlas-behavior-preflight
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/predict_v75_fixed_atlas_behavior.py

v75-fixed-atlas-behavior-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/score_v75_fixed_atlas_behavior.py

v75-fixed-atlas-behavior-full:
	@$(MAKE) --no-print-directory v75-fixed-atlas-behavior-prepare
	@$(MAKE) --no-print-directory v75-fixed-atlas-behavior-predict
	@$(MAKE) --no-print-directory v75-fixed-atlas-behavior-score

# Read-only authentication of the exact completed one-shot diagnostic. This
# target has no dependency on prepare, predictor, scorer, Gemma, or scene maps.
v75-fixed-atlas-behavior-result:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v75_fixed_atlas_behavior_result

demo-artifacts-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/check_demo_artifacts.py

demo-artifacts-check-fast:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/check_demo_artifacts.py --fast

prepare-demo-runtime:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/prepare_demo_runtime.py
	@$(MAKE) --no-print-directory demo-artifacts-check-fast

score-multi-position-ablation:
	$(PYTHON) scripts/score_center_vs_multi_position.py

report: current-report

legacy-report:
	$(PYTHON) scripts/build_report.py --config $(CONFIG)

research-demo-check:
	RESEARCH_CONTROL_CHECKPOINT="$(RESEARCH_DEMO_CONTROL_CHECKPOINT)" ./scripts/run_research_demo.sh --check --scene "$(SCENE)"

research-demo:
	RESEARCH_CONTROL_CHECKPOINT="$(RESEARCH_DEMO_CONTROL_CHECKPOINT)" ./scripts/run_research_demo.sh --scene "$(SCENE)"

research-demo-chat:
	RESEARCH_CONTROL_CHECKPOINT="$(RESEARCH_DEMO_CONTROL_CHECKPOINT)" ./scripts/run_research_demo.sh --interactive --scene "$(SCENE)"

research-demo-leakage:
	RESEARCH_CONTROL_CHECKPOINT="$(RESEARCH_DEMO_CONTROL_CHECKPOINT)" ./scripts/run_research_demo.sh --leakage --scene "$(SCENE)"

strict-demo-check:
	./scripts/run_strict_fixed_prefix_demo.sh --check --scene "$(SCENE)"

strict-demo:
	./scripts/run_strict_fixed_prefix_demo.sh --scene "$(SCENE)"

strict-demo-chat:
	./scripts/run_strict_fixed_prefix_demo.sh --interactive --scene "$(SCENE)"

strict-demo-leakage:
	./scripts/run_strict_fixed_prefix_demo.sh --leakage --scene "$(SCENE)"

# Promoted primary static path. V54 still constructs the immutable complete
# scene prefix; the sealed schema-75 controller adds learned continuous
# question/scene interaction without textifying or retrieving the environment.
v75-demo-check:
	./scripts/run_v75_question_control_demo.sh --check --scene "$(SCENE)"

v75-demo:
	./scripts/run_v75_question_control_demo.sh --scene "$(SCENE)"

v75-demo-chat:
	./scripts/run_v75_question_control_demo.sh --interactive --scene "$(SCENE)"

v75-demo-leakage:
	./scripts/run_v75_question_control_demo.sh --leakage --scene "$(SCENE)"

# Stricter experimental static path. A separate compile step materializes one
# exact 738-token numeric memory before chat. The chat process then loads only
# that sealed memory, the sanitized map/base checkpoint, and local Gemma.
v81-reader-check:
	./scripts/run_v81_strict_fixed_prefix_reader.sh --check

v81-scene-memory-compile:
	V81_DEMO_SCENE_MEMORY="$(GEMMA4_V81_SCENE_MEMORY)" \
		V81_DEMO_PROBE_BANK="$(GEMMA4_V81_PROBE_BANK)" \
		./scripts/run_v81_scene_memory_demo.sh --compile --scene "$(SCENE)" --scene-memory "$(GEMMA4_V81_SCENE_MEMORY)"

v81-scene-memory-check:
	V81_DEMO_SCENE_MEMORY="$(GEMMA4_V81_SCENE_MEMORY)" \
		./scripts/run_v81_scene_memory_demo.sh --check --scene "$(SCENE)" --scene-memory "$(GEMMA4_V81_SCENE_MEMORY)"

v81-scene-memory-demo:
	V81_DEMO_SCENE_MEMORY="$(GEMMA4_V81_SCENE_MEMORY)" \
		./scripts/run_v81_scene_memory_demo.sh --scene "$(SCENE)" --scene-memory "$(GEMMA4_V81_SCENE_MEMORY)"

v81-scene-memory-chat:
	V81_DEMO_SCENE_MEMORY="$(GEMMA4_V81_SCENE_MEMORY)" \
		./scripts/run_v81_scene_memory_demo.sh --interactive --scene "$(SCENE)" --scene-memory "$(GEMMA4_V81_SCENE_MEMORY)"

v81-scene-memory-leakage:
	V81_DEMO_SCENE_MEMORY="$(GEMMA4_V81_SCENE_MEMORY)" \
		./scripts/run_v81_scene_memory_demo.sh --leakage --scene "$(SCENE)" --scene-memory "$(GEMMA4_V81_SCENE_MEMORY)"

# Create-once historical development smoke. Prediction and answer-bearing
# scoring are separate processes; neither opens protected/final splits.
v81-historical-predict:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-c 'from semantic_3d_chat.evaluation.v81_historical_behavior import predict; predict()'

v81-historical-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-c 'from semantic_3d_chat.evaluation.v81_historical_behavior import score; import json; print(json.dumps(score(), indent=2, sort_keys=True))'

# V82 is an additive diagnostic reader over the exact sealed V81 memory.  It
# trains on the historical optimization fold only; held development preparation
# and scoring are separate create-once steps.
v82-reader-preflight:
	@if [ -f reports/gemma4/metrics/gemma4_v82_strict_dense_reader_preflight.json ]; then \
		PYTHONPATH=src $(GEMMA4_PYTHON) scripts/preflight_v82_strict_dense_reader.py \
			--config "$(GEMMA4_V82_CONFIG)" --no-write; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) scripts/preflight_v82_strict_dense_reader.py \
			--config "$(GEMMA4_V82_CONFIG)"; \
	fi

v82-reader-prepare-train:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/prepare_v82_reader_cache.py \
		--config "$(GEMMA4_V82_CONFIG)" --split train \
		--output "$(GEMMA4_V82_TRAIN_CACHE)"

v82-reader-fit:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.training.train_v82_dense_reader \
		--config "$(GEMMA4_V82_CONFIG)" --cache "$(GEMMA4_V82_TRAIN_CACHE)" \
		--output "$(GEMMA4_V82_CANDIDATE)"

v82-reader-prepare-development:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/prepare_v82_reader_cache.py \
		--config "$(GEMMA4_V82_CONFIG)" --split historical-development \
		--output "$(GEMMA4_V82_DEVELOPMENT_CACHE)"

v82-reader-evaluate:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.evaluate_v82_dense_reader \
		--config "$(GEMMA4_V82_CONFIG)" --cache "$(GEMMA4_V82_DEVELOPMENT_CACHE)" \
		--candidate "$(GEMMA4_V82_CANDIDATE)" --output "$(GEMMA4_V82_METRICS)"

v82-chat:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.chat.v82_scene_memory_cli \
		--config configs/runtime/gemma4_v54.yaml --scene "$(SCENE)" \
		--base-checkpoint data_gemma4/runtime/checkpoints/gemma4_v54_release_v1 \
		--scene-memory "$(GEMMA4_V81_SCENE_MEMORY)" \
		--reader-checkpoint "$(GEMMA4_V82_CANDIDATE)"

# First result is create-once.  Predictor opens answer-free historical-dev
# questions only; scorer references are physically blocked until the next step.
v82-historical-predict:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v82_historical_behavior \
		--reader "$(GEMMA4_V82_CANDIDATE)"

v82-historical-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) -c \
		'from semantic_3d_chat.evaluation.v82_historical_behavior import score; import json; print(json.dumps(score(), indent=2, sort_keys=True))'

# V83 is the strictest fixed-memory baseline: Gemma receives the exact 738
# tokens directly, with no question-derived reader, control tokens, or retrieval.
v83-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q tests/test_v83_direct_scene_memory.py

v83-chat:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.chat.v83_direct_scene_memory_cli \
		--config configs/runtime/gemma4_v54.yaml --scene "$(SCENE)" \
		--base-checkpoint data_gemma4/runtime/checkpoints/gemma4_v54_release_v1 \
		--scene-memory "$(GEMMA4_V83_SCENE_MEMORY)"

# Create-once pair-disjoint historical smoke. The predictor cannot open
# answer-bearing references; scoring remains a separate model-free process.
v83-historical-predict:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v83_direct_historical_behavior

v83-historical-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) -c \
		'from semantic_3d_chat.evaluation.v83_direct_historical_behavior import score; import json; print(json.dumps(score(), indent=2, sort_keys=True))'

# V84.1 is a train-only two-scene causal wiring result. It retains the exact
# V83 memory and adds only one fresh final-layer LoRA bank. It is not a runtime.
v84-pair-margin-preregister:
	@if [ -f reports/gemma4/metrics/gemma4_v84_strict_bridge_pair_margin_preregistration.json ]; then \
		PYTHONPATH=src $(GEMMA4_PYTHON) -c \
		'from semantic_3d_chat.training.train_v84_pair_margin_followup import authenticate_pair_margin_preregistration_v84, load_pair_margin_config_v84; authenticate_pair_margin_preregistration_v84(load_pair_margin_config_v84())'; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v84_pair_margin_preregistration \
		--config "$(GEMMA4_V84_PAIR_MARGIN_CONFIG)"; \
	fi

v84-pair-margin-train: v84-pair-margin-preregister
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v84_pair_margin_followup \
		--config "$(GEMMA4_V84_PAIR_MARGIN_CONFIG)"

v84-pair-margin-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q tests/test_v84_strict_bridge.py
	@test -f "$(GEMMA4_V84_PAIR_MARGIN_REPORT)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -c \
		'import json, pathlib; p=json.loads(pathlib.Path("$(GEMMA4_V84_PAIR_MARGIN_REPORT)").read_text()); assert p["passed"] and all(p["gates"].values()) and p["runtime_promotion_authorized"] is False'

v84-pair-margin-result:
	PYTHONPATH=src $(GEMMA4_PYTHON) -c \
		'import json, pathlib; p=json.loads(pathlib.Path("$(GEMMA4_V84_PAIR_MARGIN_REPORT)").read_text()); print(json.dumps({"status": p["status"], "optimizer_updates": p["optimizer_updates"], "initial_mean_nll": p["initial_mean_correct_scene_nll"], "final_mean_nll": p["final_mean_correct_scene_nll"], "final_rows": p["final_rows"], "gates": p["gates"], "runtime_promotion_authorized": p["runtime_promotion_authorized"]}, indent=2, sort_keys=True))'

# V85 is the preregistered one-epoch, 24-scene continuation of V84's exact
# 738-token direct memory. Development opens only after fixed update 72.
v85-preregister:
	@if [ -f reports/gemma4/metrics/gemma4_v85_strict_multiscene_preregistration.json ]; then \
		PYTHONPATH=src $(GEMMA4_PYTHON) -c \
		'from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import authenticate_preregistration_v85, load_config_v85; authenticate_preregistration_v85(load_config_v85())'; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v85_strict_multiscene_preflight \
		preregister --config "$(GEMMA4_V85_CONFIG)"; \
	fi

v85-preflight: v85-preregister
	@if [ -f reports/gemma4/metrics/gemma4_v85_strict_multiscene_cpu_preflight.json ]; then \
		PYTHONPATH=src $(GEMMA4_PYTHON) -c \
		'from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import authenticate_cpu_preflight_v85, load_config_v85; authenticate_cpu_preflight_v85(load_config_v85())'; \
	else \
		PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v85_strict_multiscene_preflight \
		preflight --config "$(GEMMA4_V85_CONFIG)"; \
	fi

v85-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q tests/test_v85_strict_multiscene.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v85_strict_multiscene_preflight.py \
		src/semantic_3d_chat/evaluation/evaluate_v85_strict_multiscene.py \
		src/semantic_3d_chat/training/train_v85_strict_multiscene.py \
		tests/test_v85_strict_multiscene.py

v85-train: v85-preflight v85-check
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v85_strict_multiscene \
		--config "$(GEMMA4_V85_CONFIG)"

v85-evaluate:
	@test -f "$(GEMMA4_V85_TRAINING_REPORT)"
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.evaluate_v85_strict_multiscene \
		--config "$(GEMMA4_V85_CONFIG)"

v85-result:
	@test -f "$(GEMMA4_V85_DEVELOPMENT_SCORE)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -c \
		'import json, pathlib; p=json.loads(pathlib.Path("$(GEMMA4_V85_DEVELOPMENT_SCORE)").read_text()); print(json.dumps({"status": p["status"], "metrics": p["metrics"], "separate_leakage_runtime_packaging_authorized": p["separate_leakage_runtime_packaging_authorized"], "runtime_promotion_authorized": p["runtime_promotion_authorized"]}, indent=2, sort_keys=True))'

# Real ConversationalEmbodiedAgent -> official MCP stdio server -> numeric-only
# action receipts -> refreshed continuous scene/robot binding.
conversation-mcp-smoke:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		scripts/run_conversation_mcp_smoke.py \
			--scene "$(SCENE)" \
			--runtime-asset "$(RUNTIME_SCENE_ASSET)" \
			--output "$(CONVERSATION_MCP_SMOKE_REPORT)"

# Optional numeric-grounding diagnostic. V75 remains the answer generator;
# this target is intentionally separate from the promoted default demo.
v78-grounding-prepare:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/prepare_v78_grounding_runtime.py

v78-grounding-check:
	./scripts/run_v78_grounding_demo.sh --check --scene "$(SCENE)" --grounding-checkpoint "$(GEMMA4_V78_GROUNDING_CHECKPOINT)"

v78-grounding-demo:
	./scripts/run_v78_grounding_demo.sh --scene "$(SCENE)" --grounding-checkpoint "$(GEMMA4_V78_GROUNDING_CHECKPOINT)"

v78-grounding-chat:
	./scripts/run_v78_grounding_demo.sh --interactive --scene "$(SCENE)" --grounding-checkpoint "$(GEMMA4_V78_GROUNDING_CHECKPOINT)"

v78-grounding-leakage:
	./scripts/run_v78_grounding_demo.sh --leakage --scene "$(SCENE)" --grounding-checkpoint "$(GEMMA4_V78_GROUNDING_CHECKPOINT)"

# Lightweight authentication of the static V54/V75/V78 binding followed by
# embodied input/CLI preflight. Neither command loads Gemma or changes map state.
v78-grounding-embodied-check:
	./scripts/run_v78_grounding_demo.sh --check --scene "$(SCENE)" --grounding-checkpoint "$(GEMMA4_V78_GROUNDING_CHECKPOINT)"
	EMBODIED_PYTHON="$(GEMMA4_PYTHON)" \
		EMBODIED_CONFIG="$(GEMMA4_EMBODIED_CONFIG)" \
		EMBODIED_CONTROL_CONFIG="$(GEMMA4_EMBODIED_CONTROL_CONFIG)" \
		EMBODIED_SCENE="$(SCENE)" \
		EMBODIED_BASE_CHECKPOINT="$(GEMMA4_EMBODIED_CHECKPOINT)" \
		EMBODIED_CONTROL_CHECKPOINT="$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" \
		EMBODIED_GROUNDING_CHECKPOINT="$(GEMMA4_V78_GROUNDING_CHECKPOINT)" \
		EMBODIED_RUNTIME_ASSET="$(RUNTIME_SCENE_ASSET)" \
		EMBODIED_ROBOT_STATE_CHECKPOINT="$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		./scripts/run_embodied_conversation.sh --check

# Explicit finite live proof: one rendered/fused scan followed by one V75 answer
# with optional V78 numeric grounding over the refreshed environmental prefix.
v78-grounding-embodied-once:
	EMBODIED_PYTHON="$(GEMMA4_PYTHON)" \
		EMBODIED_CONFIG="$(GEMMA4_EMBODIED_CONFIG)" \
		EMBODIED_CONTROL_CONFIG="$(GEMMA4_EMBODIED_CONTROL_CONFIG)" \
		EMBODIED_SCENE="$(SCENE)" \
		EMBODIED_BASE_CHECKPOINT="$(GEMMA4_EMBODIED_CHECKPOINT)" \
		EMBODIED_CONTROL_CHECKPOINT="$(GEMMA4_EMBODIED_CONTROL_CHECKPOINT)" \
		EMBODIED_GROUNDING_CHECKPOINT="$(GEMMA4_V78_GROUNDING_CHECKPOINT)" \
		EMBODIED_RUNTIME_ASSET="$(RUNTIME_SCENE_ASSET)" \
		EMBODIED_ROBOT_STATE_CHECKPOINT="$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
		./scripts/run_embodied_conversation.sh \
			--human \
			--audit-report "$(GEMMA4_V78_EMBODIED_AUDIT_REPORT)" \
			--command "scan" \
			--command "Where is the chair?"

strict-atlas-check:
	./scripts/run_strict_fixed_prefix_atlas.sh check

strict-atlas-build:
	./scripts/run_strict_fixed_prefix_atlas.sh build

strict-atlas-chat:
	STRICT_SCENE="$(SCENE)" ./scripts/run_strict_fixed_prefix_atlas.sh chat

strict-atlas-evaluate:
	./scripts/run_strict_fixed_prefix_atlas.sh evaluate --split validation

# Read-only structural/design authentication. These targets hash exact sources,
# configs, tests, and preregistration evidence; they load no model and authorize
# neither Atlas V2 compilation nor PLE-reader training.
strict-atlas-v2-auth:
	PYTHONPATH=src $(PYTHON) scripts/build_current_report.py --check-atlas-v2-only

ple-reader-prereg-auth:
	PYTHONPATH=src $(PYTHON) scripts/build_current_report.py --check-ple-reader-only

strict-web-check:
	./scripts/run_strict_fixed_prefix_web.sh --check --scene "$(SCENE)"

strict-web:
	./scripts/run_strict_fixed_prefix_web.sh --scene "$(SCENE)"

gemma4-v67-screen:
	./scripts/run_gemma4_v67_pair_objective.sh screen

gemma4-v67-full:
	./scripts/run_gemma4_v67_pair_objective.sh full

gemma4-v68-preregister:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v68_regularized_pair_preregistration --output reports/gemma4/metrics/v68_regularized_pair_preregistration.json

gemma4-v68-screen:
	./scripts/run_gemma4_v68_regularized_pair.sh screen

gemma4-v68-full:
	./scripts/run_gemma4_v68_regularized_pair.sh full

gemma4-v69-preregister:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v69_pair_augmentation_preregistration --output reports/gemma4/metrics/v69_pair_augmentation_preregistration.json

gemma4-v69-screen:
	./scripts/run_gemma4_v69_pair_augmentation.sh screen

gemma4-v69-full:
	./scripts/run_gemma4_v69_pair_augmentation.sh full

gemma4-v70-preregister:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v70_low_frequency_moments_preregistration --output reports/gemma4/metrics/v70_low_frequency_moments_preregistration.json

gemma4-v70-screen:
	./scripts/run_gemma4_v70_low_frequency_moments.sh

gemma4-v70-authenticate: current-report
	$(PYTHON) -c 'import json; from pathlib import Path; p=json.loads(Path("$(CURRENT_METRICS)").read_text()); v=p["v70_low_frequency_moments_numeric_screen"]; assert v["measurement_authenticated"] is True and v["status"] == "authenticated_numeric_screen_failed_no_publication" and all(v["authentication_checks"].values()); print("V70 authenticated: failed closed; no generation, atlas, full run, or checkpoint")'

gemma4-v71-preregister:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v71_multiscale_preregistration --output reports/gemma4/metrics/v71_multiscale_preregistration.json

gemma4-v71-screen:
	./scripts/run_gemma4_v71_multiscale.sh

gemma4-v71-authenticate:
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v71_result_authentication

test:
	$(PYTHON) -m pytest

# Default operator entry point: inspect the machine, authenticate the promoted
# strict V89 release, print human-visualization and optional service commands,
# then start interactive chat only when attached to a TTY. Redirected/CI
# launches run a finite three-question strict-memory demo and exit.
demo:
	./scripts/doctor.sh
	@$(MAKE) --no-print-directory demo-check SCENE="$(SCENE)"
	@echo "RGB point-map preview: reports/gemma4/figures/$(SCENE)/map_rgb.png"
	@echo "3D point cloud: reports/gemma4/figures/$(SCENE)/map_rgb.ply"
	@echo "Scan montage: reports/gemma4/figures/scan_montage.png"
	@echo "Open human-only visuals (optional): open reports/gemma4/figures/$(SCENE)/map_rgb.png reports/gemma4/figures/scan_montage.png"
	@echo "Browser UI comparator in another terminal (V54 strict prefix): make strict-web SCENE=$(SCENE)"
	@echo "Strict 738-token direct-memory chat (primary): make chat SCENE=$(SCENE)"
	@echo "Question-conditioned V75 comparator: make v75-demo-chat SCENE=$(SCENE)"
	@echo "Embodied readiness (model-free): make embodied-check SCENE=$(SCENE)"
	@echo "Embodied MCP server in another terminal: make gemma4-embodied-mcp SCENE=$(SCENE)"
	@if [ -t 0 ] && [ -t 1 ]; then \
		echo "Starting interactive promoted V89 strict continuous-memory chat. Type exit or quit to stop."; \
		./scripts/run_v89_strict_scene1_demo.sh --interactive --scene "$(SCENE)"; \
	else \
		echo "No interactive TTY detected; running the finite V89 strict demo instead."; \
		./scripts/run_v89_strict_scene1_demo.sh --scene "$(SCENE)"; \
	fi

# Finite, non-interactive three-question smoke. This remains useful for CI and
# scripted demonstrations; it is intentionally separate from the default chat.
demo-smoke:
	./scripts/run_v89_strict_scene1_demo.sh --scene "$(SCENE)"

demo-check: v89-demo-check embodied-check
	@echo "Prepared local demo: PASS (promoted strict V89 static chat and accepted V14 same-room Gemma waypoint rover; held-out navigation generalization remains pending)."

demo-leakage:
	./scripts/run_v89_strict_scene1_demo.sh --leakage --scene "$(SCENE)"

# Human-facing local embodied demonstration. The check is finite, read-only,
# model-free, and renderer-free. The live target opens a loopback-only browser
# UI and then loads Gemma, the complete continuous room memory, and the bounded
# rover controller in one local process.
rover-demo-check:
	./scripts/run_local_rover_demo.sh --check --scene "$(SCENE)" --host "$(ROVER_DEMO_HOST)" --port "$(ROVER_DEMO_PORT)" --no-open

rover-demo:
	./scripts/run_local_rover_demo.sh --scene "$(SCENE)" --host "$(ROVER_DEMO_HOST)" --port "$(ROVER_DEMO_PORT)"

# Optional official MCP transport for external local clients. This is a
# separate process/entry point so the normal browser demo does not load a
# second 10.25 GB Gemma instance on a 24 GB Mac.
rover-demo-mcp: rover-demo-check
	@echo "Starting the official numeric-only rover MCP server on stdio. The browser UI is not started by this target."
	@$(MAKE) --no-print-directory gemma4-embodied-mcp SCENE="$(SCENE)"

# Model-owned navigation transport. This deliberately exposes one high-level
# `navigate(goal)` tool and no user/client motor primitives. The promoted local
# Gemma policy selects every exact FACE/MOVE_TO/STOP decision; deterministic
# code only executes or safety-rejects that decision.
rover-gemma-mcp-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.mcp_server.gemma_goal_server --check --scene "$(SCENE)"

rover-gemma-mcp: rover-gemma-mcp-check
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.mcp_server.gemma_goal_server --scene "$(SCENE)"

# Preferred human-facing demo: Blender is the real 3D viewport and talks only
# to the loopback numeric rover API. The launcher owns and cleans up the heavy
# backend process it starts, while safely reusing an already matching backend.
blender-rover-demo-check:
	BLENDER="$(BLENDER)" BLENDER_ROVER_BACKEND_TIMEOUT="$(BLENDER_ROVER_BACKEND_TIMEOUT)" \
		./scripts/run_blender_rover_demo.sh --check --scene "$(SCENE)" --host "$(ROVER_DEMO_HOST)" --port "$(ROVER_DEMO_PORT)"

blender-rover-demo:
	BLENDER="$(BLENDER)" BLENDER_ROVER_BACKEND_TIMEOUT="$(BLENDER_ROVER_BACKEND_TIMEOUT)" \
		./scripts/run_blender_rover_demo.sh --scene "$(SCENE)" --host "$(ROVER_DEMO_HOST)" --port "$(ROVER_DEMO_PORT)"

# Short operator aliases.
rover-3d-check: blender-rover-demo-check

rover-3d: blender-rover-demo

# ---------------------------------------------------------------------------
# Spatial Lens: author a room, perceive it, ask about it, drive in it.
#
# Zero on-device training. Perception is Gemma's vision encoder plus its VQA;
# reasoning is Gemma reading the metric map it built. The author's words for the
# furniture are written to a scorer-only path and never reach the model.
# ---------------------------------------------------------------------------
LENS_ROOM ?= studio
LENS_SPEC ?= rooms/$(LENS_ROOM).json
LENS_GOAL ?= Drive to the bookshelf and stop beside it.
LENS_METRICS ?= reports/gemma4/metrics
LENS_ROOM_COUNT ?= 12
LENS_ROOM_PREFIX ?= room
LENS_HOLDOUT ?= 8
LENS_ROOMS ?= studio

lens-check:
	$(PYTHON) -m pytest -q tests/test_spatial_lens.py tests/test_spatial_grounding.py \
		tests/test_rope3d.py tests/test_point_grounding_data.py
	$(PYTHON) -m ruff check \
		src/semantic_3d_chat/spatial_lens/ \
		src/semantic_3d_chat/language/rope3d_patch.py \
		scripts/lens_train_points.py \
		scripts/lens_cache_phrases.py \
		scripts/lens_eval_rope3d_locate.py \
		scripts/lens_eval_rope3d_relations.py \
		scripts/build_point_grounding_summary.py \
		scripts/lens_make_rooms.py \
		scripts/lens_batch.py \
		scripts/lens_train_grounding.py \
		scripts/lens_ground.py \
		scripts/lens_topdown_baseline.py \
		scripts/lens_compare_methods.py \
		scripts/lens_ask_3d.py \
		scripts/lens_locate_3d.py \
		scripts/lens_build_room.py \
		scripts/lens_scan_room.py \
		scripts/lens_perceive.py \
		scripts/lens_understand.py \
		scripts/lens_ask.py \
		scripts/lens_drive.py \
		blender/build_authored_room.py \
		blender/scan_authored_room.py \
		tests/test_spatial_lens.py \
		tests/test_spatial_grounding.py

lens-build:
	PYTHONPATH=src $(PYTHON) scripts/lens_build_room.py --spec $(LENS_SPEC) --force

lens-scan:
	PYTHONPATH=src $(PYTHON) scripts/lens_scan_room.py --room $(LENS_ROOM) --force

lens-perceive:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_perceive.py --room $(LENS_ROOM) --force

lens-understand:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_understand.py --room $(LENS_ROOM) --force

lens-rooms:
	$(PYTHON) scripts/lens_make_rooms.py --count $(LENS_ROOM_COUNT) --prefix $(LENS_ROOM_PREFIX)

lens-batch:
	PYTHONPATH=src $(PYTHON) scripts/lens_batch.py --rooms $(LENS_ROOMS)

lens-train-grounding:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_train_grounding.py --holdout $(LENS_HOLDOUT)

lens-ground:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_ground.py --room $(LENS_ROOM) --controls \
		--output $(LENS_METRICS)/spatial_lens_$(LENS_ROOM)_ground.json

lens-topdown:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_topdown_baseline.py --rooms $(LENS_ROOMS) \
		--output $(LENS_METRICS)/spatial_lens_topdown_gemma.json

lens-compare:
	PYTHONPATH=src $(PYTHON) scripts/lens_compare_methods.py

lens-phrases:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_cache_phrases.py

lens-train-points:
	PYTHONPATH=src $(PYTHON) scripts/lens_train_points.py --holdout $(LENS_HOLDOUT)

lens-point-sweep:
	./scripts/lens_point_sweep.sh

lens-point-summary:
	PYTHONPATH=src $(PYTHON) scripts/build_point_grounding_summary.py

lens-rope3d-locate:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_eval_rope3d_locate.py

lens-rope3d-relations:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_eval_rope3d_relations.py

lens-ask-3d:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_ask_3d.py --room $(LENS_ROOM) --controls \
		--output $(LENS_METRICS)/spatial_lens_$(LENS_ROOM)_qa3d.json

lens-locate-3d:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_locate_3d.py --room $(LENS_ROOM) --controls \
		--output $(LENS_METRICS)/spatial_lens_$(LENS_ROOM)_locate3d.json

lens-ask:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_ask.py --room $(LENS_ROOM) --show-map

lens-drive:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/lens_drive.py --room $(LENS_ROOM) \
		--goal "$(LENS_GOAL)" \
		--output $(LENS_METRICS)/spatial_lens_$(LENS_ROOM)_drive.json

# Everything from an authored JSON room to a perceived, queryable scene graph.
lens-all: lens-build lens-scan lens-perceive lens-understand


# ---------------------------------------------------------------------------
# V15 room generalization
#
# Every V4-V14 waypoint result was measured in the single room the policy was
# fitted to. V15 trains across 27 rooms and measures 8 scene-disjoint
# development rooms plus 6 rooms that are sealed until the checkpoint is
# frozen. These targets rebuild that pipeline from scratch; each stage refuses
# to overwrite an existing artifact, so remove the previous one deliberately.
# ---------------------------------------------------------------------------
GEMMA4_V15_CONFIG ?= configs/experiments/gemma_waypoint_policy_v15_general.yaml
GEMMA4_V15_CHECKPOINT ?= data_gemma4/checkpoints/gemma_waypoint_policy_v15_general
GEMMA4_V15_SEALED_DATASET ?= data_gemma4/training/gemma_waypoint_policy_v15_sealed
GEMMA4_V15_METRICS ?= reports/gemma4/metrics

v15-check:
	$(PYTHON) -m pytest -q \
		tests/test_v15_general_trace_dataset.py \
		tests/test_v15_shuffled_scene_control.py \
		tests/test_v15_heldout_closed_loop.py \
		tests/test_v15_summary_builder.py \
		tests/test_v15_recorded_result.py \
		tests/test_gemma_waypoint_training.py \
		tests/test_gemma_waypoint_runtime.py
	$(PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v15_heldout_closed_loop.py \
		src/semantic_3d_chat/evaluation/v15_scene_token_probe.py \
		src/semantic_3d_chat/training/gemma_waypoint_policy.py \
		src/semantic_3d_chat/training/gemma_waypoint_trace_generator.py \
		scripts/evaluate_v15_heldout_closed_loop.py \
		scripts/evaluate_gemma_waypoint_policy.py \
		scripts/generate_gemma_waypoint_traces.py \
		tests/test_v15_general_trace_dataset.py \
		tests/test_v15_shuffled_scene_control.py \
		tests/test_v15_heldout_closed_loop.py \
		tests/test_v15_summary_builder.py \
		tests/test_v15_recorded_result.py \
		scripts/build_gemma_waypoint_v15_summary.py

v15-traces:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/generate_gemma_waypoint_traces.py \
		--config $(GEMMA4_V15_CONFIG) --profile general
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/generate_gemma_waypoint_traces.py \
		--config $(GEMMA4_V15_CONFIG) --profile general_sealed \
		--destination $(GEMMA4_V15_SEALED_DATASET)

v15-cache:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/cache_gemma_waypoint_hidden.py \
		--config $(GEMMA4_V15_CONFIG) --gemma-batch-size 4 --forward-chunk-size 128

v15-train:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/train_gemma_waypoint_policy.py \
		--config $(GEMMA4_V15_CONFIG) \
		--metrics $(GEMMA4_V15_METRICS)/gemma_waypoint_policy_v15_general_training.json

# Sealed unseen-room offline decisions, including the shuffled-scene control.
v15-sealed-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/evaluate_gemma_waypoint_policy.py \
		--config $(GEMMA4_V15_CONFIG) \
		--dataset $(GEMMA4_V15_SEALED_DATASET) \
		--checkpoint $(GEMMA4_V15_CHECKPOINT) \
		--split validation \
		--sample-limit 270 \
		--condition wrong_scene_prefix --condition zero_scene_prefix \
		--condition shuffled_scene_prefix --condition zero_history \
		--output $(GEMMA4_V15_METRICS)/gemma_waypoint_v15_sealed_controls.json

# Closed-loop goals in the sealed rooms. `plan` reads oracle geometry, `rollout`
# is audited to prove it cannot, and `score` joins the two.
v15-heldout-plan:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/evaluate_v15_heldout_closed_loop.py plan \
		--tasks $(GEMMA4_V15_METRICS)/gemma_waypoint_v15_heldout_tasks.json \
		--targets reports/gemma4/scorer_only/gemma_waypoint_v15_heldout_targets.json

v15-heldout-rollout:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/evaluate_v15_heldout_closed_loop.py rollout \
		--tasks $(GEMMA4_V15_METRICS)/gemma_waypoint_v15_heldout_tasks.json \
		--navigation-checkpoint $(GEMMA4_V15_CHECKPOINT) \
		--output $(GEMMA4_V15_METRICS)/gemma_waypoint_v15_heldout_runtime.json

v15-heldout-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) scripts/evaluate_v15_heldout_closed_loop.py score \
		--rollouts $(GEMMA4_V15_METRICS)/gemma_waypoint_v15_heldout_runtime.json \
		--targets reports/gemma4/scorer_only/gemma_waypoint_v15_heldout_targets.json \
		--output $(GEMMA4_V15_METRICS)/gemma_waypoint_v15_heldout_score.json

# Every field is read back out of the artifacts above and each one is hashed,
# so the summary cannot drift from the evidence. Absent stages are reported.
v15-probe:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.v15_scene_token_probe \
		--prefix-cache data_gemma4/scene_tokens/gemma_waypoint_policy_v15_rooms \
		--output $(GEMMA4_V15_METRICS)/gemma_waypoint_v15_scene_token_probe.json

v15-summary:
	$(PYTHON) scripts/build_gemma_waypoint_v15_summary.py \
		--checkpoint $(GEMMA4_V15_CHECKPOINT) \
		--sealed-dataset $(GEMMA4_V15_SEALED_DATASET) \
		--output $(GEMMA4_V15_METRICS)/gemma_waypoint_v15_summary.json

# Run against a separately started, fresh rover backend (action_count must be 0).
# This verifier reads no oracle data. It requires an actual Gemma forward for
# every waypoint/heading/STOP and rejects any deterministic route substitution.
rover-live-verify:
	./scripts/verify_gemma_waypoint_rover_live.py --base-url "http://$(ROVER_DEMO_HOST):$(ROVER_DEMO_PORT)"

# Preserved generic legacy infrastructure.
legacy-demo:
	./scripts/run_full_demo.sh --config $(CONFIG) --scene $(SCENE) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

legacy-demo-check:
	./scripts/run_full_demo.sh --check --config $(CONFIG) --scene $(SCENE) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

legacy-demo-leakage:
	./scripts/run_full_demo.sh --non-interactive --config $(CONFIG) --scene $(SCENE) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)
# V87's runtime is a separate post-model-gate release surface.  These targets
# do not alter the default demo.  ``prepare`` remains fail-closed until the
# sealed V87 evaluation passes every preregistered model gate; ``promote`` also
# requires the independent oracle-unavailable child-process smoke.
.PHONY: v87-runtime-check v87-runtime-authenticate v87-runtime-prepare
.PHONY: v87-runtime-verify-candidate v87-runtime-smoke v87-runtime-promote
.PHONY: v87-runtime-verify v87-runtime-cleanup-failed-candidate

v87-runtime-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q tests/test_v87_strict_runtime.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/chat/v87_strict_scene1_runtime.py \
		src/semantic_3d_chat/chat/v87_strict_scene1_cli.py \
		src/semantic_3d_chat/evaluation/v87_strict_runtime_release.py \
		tests/test_v87_strict_runtime.py

v87-runtime-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v87_strict_runtime_release authenticate

v87-runtime-prepare: v87-runtime-authenticate
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v87_strict_runtime_release prepare

v87-runtime-verify-candidate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v87_strict_runtime_release verify-candidate

v87-runtime-smoke: v87-runtime-verify-candidate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v87_strict_runtime_release smoke

v87-runtime-promote:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v87_strict_runtime_release promote

v87-runtime-verify:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v87_strict_runtime_release verify

v87-runtime-cleanup-failed-candidate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v87_strict_runtime_release cleanup-failed-candidate

# V89's post-gate release is isolated from the sealed trainer/evaluator. It is
# now the default demo; packaging still refuses until every fixed-final model
# gate passes, and promotion additionally requires the oracle-renamed smoke.
.PHONY: v89-runtime-check v89-runtime-authenticate v89-runtime-prepare
.PHONY: v89-runtime-verify-candidate v89-runtime-smoke v89-runtime-promote
.PHONY: v89-runtime-verify v89-runtime-cleanup-failed-candidate v89-chat

v89-runtime-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_strict_direct_release_core.py \
		tests/test_v89_strict_runtime_skeleton.py \
		tests/test_v89_strict_runtime_release.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/strict_direct_release_core.py \
		src/semantic_3d_chat/chat/v89_strict_scene1_runtime.py \
		src/semantic_3d_chat/chat/v89_strict_scene1_cli.py \
		src/semantic_3d_chat/evaluation/v89_strict_runtime_skeleton.py \
		src/semantic_3d_chat/evaluation/v89_strict_runtime_release.py \
		tests/test_v89_strict_runtime_skeleton.py \
		tests/test_v89_strict_runtime_release.py

v89-runtime-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v89_strict_runtime_release authenticate

v89-runtime-prepare: v89-runtime-authenticate
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v89_strict_runtime_release prepare

v89-runtime-verify-candidate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v89_strict_runtime_release verify-candidate

v89-runtime-smoke: v89-runtime-verify-candidate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v89_strict_runtime_release smoke

v89-runtime-promote:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v89_strict_runtime_release promote

v89-runtime-verify:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v89_strict_runtime_release verify

v89-runtime-cleanup-failed-candidate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v89_strict_runtime_release cleanup-failed-candidate

v89-chat:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.chat.v89_strict_scene1_cli --scene scene_000001

v89-demo-check:
	./scripts/run_v89_strict_scene1_demo.sh --check --scene "$(SCENE)"

v89-demo:
	./scripts/run_v89_strict_scene1_demo.sh --scene "$(SCENE)"

v89-demo-chat:
	./scripts/run_v89_strict_scene1_demo.sh --interactive --scene "$(SCENE)"

v89-demo-leakage:
	./scripts/run_v89_strict_scene1_demo.sh --leakage --scene "$(SCENE)"

# V90 is the fixed-schedule conversational repair of the promoted V89 stack.
# These offline targets remain create-once and fail closed; V89 stays the
# operator default until V90 passes its preregistered model and isolated-runtime
# gates and is packaged as a separate sanitized release.
.PHONY: v90-check v90-derive v90-preregister v90-preflight v90-authenticate
.PHONY: v90-train v90-evaluate

v90-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q tests/test_v90_scene1_conversational.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v90_scene1_conversational_preflight.py \
		src/semantic_3d_chat/training/train_v90_scene1_conversational.py \
		src/semantic_3d_chat/evaluation/evaluate_v90_scene1_conversational.py \
		tests/test_v90_scene1_conversational.py

v90-derive:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v90_scene1_conversational_preflight derive

v90-preregister:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v90_scene1_conversational_preflight preregister

v90-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v90_scene1_conversational_preflight preflight

v90-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v90_scene1_conversational_preflight authenticate

v90-train: v90-authenticate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v90_scene1_conversational

v90-evaluate:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.evaluate_v90_scene1_conversational

# V91 is the preregistered evidence-weighted continuation after V90's measured
# conversational gate failure. It keeps the exact direct scene memory and all
# parent banks frozen, and remains offline/create-once until every gate passes.
.PHONY: v91-check v91-derive v91-preregister v91-preflight v91-authenticate
.PHONY: v91-train v91-evaluate v91-runtime-check v91-release-authenticate
.PHONY: v91-release-prepare v91-release-verify-candidate v91-release-smoke
.PHONY: v91-release-promote v91-release-verify v91-demo

v91-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q tests/test_v91_scene1_conversational.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v91_scene1_conversational_preflight.py \
		src/semantic_3d_chat/training/train_v91_scene1_conversational_repair.py \
		src/semantic_3d_chat/evaluation/evaluate_v91_scene1_conversational_repair.py \
		tests/test_v91_scene1_conversational.py

v91-derive:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v91_scene1_conversational_preflight derive

v91-preregister:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v91_scene1_conversational_preflight preregister

v91-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v91_scene1_conversational_preflight preflight

v91-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v91_scene1_conversational_preflight authenticate

v91-train: v91-authenticate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v91_scene1_conversational_repair

v91-evaluate:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.evaluate_v91_scene1_conversational_repair

# V92 freezes the exact failed V91 candidate and adds one disjoint retention-
# aware conversational repair bank. Its artifacts are create-once and remain
# evaluation-only unless every preregistered gate passes.
.PHONY: v92-check v92-derive v92-preregister v92-preflight v92-authenticate
.PHONY: v92-train v92-evaluate

v92-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v92_scene1_retention_conversation.py \
		tests/test_v92_evaluator.py \
		tests/test_v92_trainer_contract.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v92_scene1_retention_conversation_preflight.py \
		src/semantic_3d_chat/training/train_v92_scene1_retention_conversation_repair.py \
		src/semantic_3d_chat/evaluation/evaluate_v92_scene1_retention_conversation_repair.py \
		tests/test_v92_scene1_retention_conversation.py \
		tests/test_v92_evaluator.py \
		tests/test_v92_trainer_contract.py

v92-derive:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight derive

v92-preregister:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight preregister

v92-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight preflight

v92-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight authenticate

v92-train: v92-authenticate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v92_scene1_retention_conversation_repair

v92-evaluate:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.evaluate_v92_scene1_retention_conversation_repair

# V93 freezes the exact failed V92 fourteen-bank stack and adds one disjoint
# termination/paraphrase repair bank. Its artifacts are create-once and remain
# evaluation-only unless every preregistered model gate passes.
.PHONY: v93-check v93-derive v93-preregister v93-preflight v93-authenticate
.PHONY: v93-train v93-evaluate

v93-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v93_scene1_termination_paraphrase.py \
		tests/test_v93_evaluator.py \
		tests/test_v93_trainer_contract.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v93_scene1_termination_paraphrase_preflight.py \
		src/semantic_3d_chat/training/train_v93_scene1_termination_paraphrase_repair.py \
		src/semantic_3d_chat/evaluation/evaluate_v93_scene1_termination_paraphrase_repair.py \
		tests/test_v93_scene1_termination_paraphrase.py \
		tests/test_v93_evaluator.py \
		tests/test_v93_trainer_contract.py

v93-derive:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight derive

v93-preregister:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight preregister

v93-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight preflight

v93-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight authenticate

v93-train: v93-authenticate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v93_scene1_termination_paraphrase_repair

v93-evaluate:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.evaluate_v93_scene1_termination_paraphrase_repair

# V91 runtime packaging is deliberately separate from offline evaluation.
# Authentication and the oracle-absent smoke must pass before the explicit
# promote target can materialize a release; none of these targets change the
# repository's current default demo.
v91-runtime-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v91_strict_runtime_skeleton.py \
		tests/test_v91_strict_runtime_release.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/chat/v91_strict_scene1_runtime.py \
		src/semantic_3d_chat/chat/v91_strict_scene1_cli.py \
		src/semantic_3d_chat/evaluation/v91_strict_runtime_release.py \
		tests/test_v91_strict_runtime_skeleton.py \
		tests/test_v91_strict_runtime_release.py

v91-release-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v91_strict_runtime_release authenticate

v91-release-prepare:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v91_strict_runtime_release prepare

v91-release-verify-candidate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v91_strict_runtime_release verify-candidate

v91-release-smoke:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v91_strict_runtime_release smoke

v91-release-promote:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v91_strict_runtime_release promote

v91-release-verify:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v91_strict_runtime_release verify

v91-demo:
	./scripts/run_v91_strict_scene1_demo.sh --scene "$(SCENE)"

# V92 runtime packaging remains a separate, fail-closed release workflow.
# These targets do not change the repository default. The prepare/smoke/promote
# chain is usable only after the exact create-once V92 evaluation passes.
.PHONY: v92-runtime-check v92-release-authenticate v92-release-prepare
.PHONY: v92-release-verify-candidate v92-release-smoke v92-release-promote
.PHONY: v92-release-verify v92-demo

v92-runtime-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v92_strict_runtime_skeleton.py \
		tests/test_v92_strict_runtime_release.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/chat/v92_strict_scene1_runtime.py \
		src/semantic_3d_chat/chat/v92_strict_scene1_cli.py \
		src/semantic_3d_chat/evaluation/v92_strict_runtime_release.py \
		tests/test_v92_strict_runtime_skeleton.py \
		tests/test_v92_strict_runtime_release.py

v92-release-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v92_strict_runtime_release authenticate

v92-release-prepare:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v92_strict_runtime_release prepare

v92-release-verify-candidate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v92_strict_runtime_release verify-candidate

v92-release-smoke:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v92_strict_runtime_release smoke

v92-release-promote:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v92_strict_runtime_release promote

v92-release-verify:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v92_strict_runtime_release verify

v92-demo:
	./scripts/run_v92_strict_scene1_demo.sh --scene "$(SCENE)"

# V94 is a sealed full-40-scene continuation from the exact V85 strict
# runtime. Prediction is intentionally separate from the label-bearing scorer.
.PHONY: v94-check v94-derive v94-preregister v94-preflight v94-authenticate
.PHONY: v94-topology-smoke v94-train v94-compile-eval-memory v94-predict v94-score v94-evidence v94-evaluate

v94-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v94_strict_multiscene_preflight.py \
		tests/test_v94_trainer.py \
		tests/test_evaluate_v94_strict_multiscene_full40.py \
		tests/test_v94_strict_multiscene_evidence.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v94_strict_multiscene_preflight.py \
		src/semantic_3d_chat/training/train_v94_strict_multiscene_full40.py \
		src/semantic_3d_chat/evaluation/evaluate_v94_strict_multiscene_full40.py \
		src/semantic_3d_chat/evaluation/v94_strict_multiscene_evidence.py \
		tests/test_v94_strict_multiscene_preflight.py \
		tests/test_v94_trainer.py \
		tests/test_evaluate_v94_strict_multiscene_full40.py \
		tests/test_v94_strict_multiscene_evidence.py

v94-derive:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strict_multiscene_preflight derive

v94-preregister:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strict_multiscene_preflight preregister

v94-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strict_multiscene_preflight cpu-preflight

v94-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strict_multiscene_preflight authenticate

v94-topology-smoke: v94-authenticate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v94_strict_multiscene_full40 --topology-smoke

v94-train: v94-authenticate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v94_strict_multiscene_full40

v94-compile-eval-memory:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strict_multiscene_evidence --compile-cache

v94-predict:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.evaluate_v94_strict_multiscene_full40 predict

v94-score:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.evaluate_v94_strict_multiscene_full40 score

v94-evidence:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strict_multiscene_evidence --require-score

v94-evaluate: v94-train
	$(MAKE) v94-compile-eval-memory
	$(MAKE) v94-predict
	$(MAKE) v94-score
	$(MAKE) v94-evidence

# Independent, post-hoc V94 causal diagnostics. These targets are terminal and
# cannot alter the sealed V94 gates, package a runtime, or authorize promotion.
# The representative profile needs no map recompilation and runs 36 questions
# across correct, zero, paired-wrong, and all-736-interior-token permutation.
# The full profile adds semantic-row, XYZ-only, and RGB controls. Normals are
# already exactly zero in all six maps and viewpoint is not consumed, so both
# are authenticated unsupported no-ops rather than fabricated ablation arms.
.PHONY: v94-strong-causal-check v94-strong-causal-core-predict
.PHONY: v94-strong-causal-core-authenticate v94-strong-causal-core-score
.PHONY: v94-strong-causal-core-evaluate v94-strong-causal-compile
.PHONY: v94-strong-causal-cache-authenticate v94-strong-causal-full-predict
.PHONY: v94-strong-causal-full-authenticate v94-strong-causal-full-score
.PHONY: v94-strong-causal-full-evaluate

v94-strong-causal-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v94_strong_causal_ablations.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v94_strong_causal_ablations.py \
		tests/test_v94_strong_causal_ablations.py

v94-strong-causal-core-predict:
	@echo "Terminal post-hoc V94 diagnostic only; cannot authorize release or promotion."
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strong_causal_ablations \
		predict --profile representative-core

v94-strong-causal-core-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strong_causal_ablations \
		authenticate --profile representative-core

v94-strong-causal-core-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strong_causal_ablations \
		score --profile representative-core

v94-strong-causal-core-evaluate:
	$(MAKE) v94-strong-causal-core-predict
	$(MAKE) v94-strong-causal-core-authenticate
	$(MAKE) v94-strong-causal-core-score

v94-strong-causal-compile:
	@echo "Compiling terminal post-hoc map controls; sealed V94 artifacts remain untouched."
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strong_causal_ablations compile-controls

v94-strong-causal-cache-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strong_causal_ablations authenticate-cache

v94-strong-causal-full-predict:
	@echo "Terminal post-hoc V94 diagnostic only; cannot authorize release or promotion."
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strong_causal_ablations predict --profile full

v94-strong-causal-full-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strong_causal_ablations authenticate --profile full

v94-strong-causal-full-score:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v94_strong_causal_ablations score --profile full

v94-strong-causal-full-evaluate:
	$(MAKE) v94-strong-causal-compile
	$(MAKE) v94-strong-causal-cache-authenticate
	$(MAKE) v94-strong-causal-full-predict
	$(MAKE) v94-strong-causal-full-authenticate
	$(MAKE) v94-strong-causal-full-score

# V95 is a fixed-final causal-memory successor to V94's exact failed,
# non-promoted optimization parent.  Sealing and CPU topology checks do not
# load Gemma; the long MPS training run remains an explicit separate target.
.PHONY: v95-check v95-seal-parent-evidence v95-derive v95-preregister
.PHONY: v95-preflight v95-authenticate v95-topology-smoke v95-train
.PHONY: v95-known-development-python-check
.PHONY: v95-known-development-predict
.PHONY: v95-known-development-authenticate-predictions
.PHONY: v95-known-development-score
.PHONY: v95-known-development-authenticate-score
.PHONY: v95-known-development-nll
.PHONY: v95-known-development-authenticate-nll
.PHONY: v95-known-development-seal
.PHONY: v95-known-development-authenticate-final
.PHONY: v95-known-development-evaluate
.PHONY: v95-deferred-final-check v95-deferred-final-preflight
.PHONY: v95-deferred-final-unlock v95-deferred-final-authenticate
.PHONY: v95-deferred-final-template v95-deferred-final-materialization-preflight
.PHONY: v95-deferred-final-preregister-materialization
.PHONY: v95-deferred-final-authenticate-materialization-preregistration
.PHONY: v95-deferred-final-generate v95-deferred-final-render
.PHONY: v95-deferred-final-features v95-deferred-final-maps
.PHONY: v95-deferred-final-qa-raw v95-deferred-final-qa-select
.PHONY: v95-deferred-final-memory v95-deferred-final-questions

v95-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v95_strict_causal_successor.py \
		tests/test_v95_trainer.py \
		tests/test_v95_known_development_harness.py \
		tests/test_v95_deferred_final.py \
		tests/test_v95_deferred_final_materialization.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v95_strict_causal_successor_preflight.py \
		src/semantic_3d_chat/evaluation/v95_known_development_common.py \
		src/semantic_3d_chat/evaluation/predict_v95_known_development.py \
		src/semantic_3d_chat/evaluation/authenticate_v95_known_development.py \
		src/semantic_3d_chat/evaluation/score_v95_known_development.py \
		src/semantic_3d_chat/evaluation/nll_v95_known_development.py \
		src/semantic_3d_chat/evaluation/seal_v95_known_development.py \
		src/semantic_3d_chat/evaluation/v95_deferred_final_qa.py \
		src/semantic_3d_chat/evaluation/v95_deferred_final_materialization.py \
		src/semantic_3d_chat/evaluation/v95_deferred_final.py \
		src/semantic_3d_chat/training/train_v95_strict_causal_successor.py \
		tests/test_v95_strict_causal_successor.py \
		tests/test_v95_trainer.py \
		tests/test_v95_known_development_harness.py \
		tests/test_v95_deferred_final.py \
		tests/test_v95_deferred_final_materialization.py

v95-seal-parent-evidence:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight \
		seal-parent-evidence

v95-derive:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight derive

v95-preregister:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight preregister

v95-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight cpu-preflight

v95-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight authenticate

v95-topology-smoke: v95-authenticate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v95_strict_causal_successor \
		--topology-smoke

v95-train: v95-authenticate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v95_strict_causal_successor

# V95 known-development evaluation is deliberately split across process
# boundaries. Prediction is label-blind; structured scoring is model-free;
# NLL is the only separately authorized model process that opens the pinned
# labels; sealing/final authentication reopen neither labels nor Gemma.
# Every public artifact is create-once and each stage reauthenticates its
# complete prerequisite bundle before doing any work.
v95-known-development-python-check:
	@test -x .venv-gemma4/bin/python || { \
		echo "Missing required V95 interpreter: .venv-gemma4/bin/python" >&2; \
		exit 1; \
	}

v95-known-development-predict: v95-known-development-python-check v95-authenticate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.predict_v95_known_development

v95-known-development-authenticate-predictions: v95-known-development-python-check
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.authenticate_v95_known_development prediction

v95-known-development-score: v95-known-development-authenticate-predictions
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.score_v95_known_development

v95-known-development-authenticate-score: v95-known-development-python-check
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.authenticate_v95_known_development structured

v95-known-development-nll: v95-known-development-authenticate-score
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.nll_v95_known_development

v95-known-development-authenticate-nll: v95-known-development-python-check
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.authenticate_v95_known_development nll

v95-known-development-seal: v95-known-development-authenticate-score v95-known-development-authenticate-nll
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.seal_v95_known_development seal

v95-known-development-authenticate-final: v95-known-development-python-check
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.seal_v95_known_development authenticate

v95-known-development-evaluate: v95-known-development-python-check
	$(MAKE) v95-known-development-predict
	$(MAKE) v95-known-development-authenticate-predictions
	$(MAKE) v95-known-development-score
	$(MAKE) v95-known-development-authenticate-score
	$(MAKE) v95-known-development-nll
	$(MAKE) v95-known-development-authenticate-nll
	$(MAKE) v95-known-development-seal
	$(MAKE) v95-known-development-authenticate-final

# V95 cannot use the legacy gemma4-final-once transition: that controller
# requires a promoted adapter.safetensors checkpoint and legacy selector.
# These model-free targets authenticate V95's immutable bridge plus its passing
# sealed known-development gate. The outcome-independent materialization recipe
# is sealed before labels. Every later stage reauthenticates both seals before
# running its fixed command; none of these check/preregister targets generates.
v95-deferred-final-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v95_deferred_final.py \
		tests/test_v95_deferred_final_materialization.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v95_deferred_final_qa.py \
		src/semantic_3d_chat/evaluation/v95_deferred_final_materialization.py \
		src/semantic_3d_chat/evaluation/v95_deferred_final.py \
		tests/test_v95_deferred_final.py \
		tests/test_v95_deferred_final_materialization.py

v95-deferred-final-preregister-materialization: v95-known-development-python-check
	PYTHONPATH=src $(PYTHON) \
		-m semantic_3d_chat.evaluation.v95_deferred_final_materialization preregister

v95-deferred-final-authenticate-materialization-preregistration: v95-known-development-python-check
	PYTHONPATH=src $(PYTHON) \
		-m semantic_3d_chat.evaluation.v95_deferred_final_materialization authenticate

v95-deferred-final-preflight: v95-known-development-authenticate-final v95-deferred-final-authenticate-materialization-preregistration
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v95_deferred_final preflight

v95-deferred-final-unlock: v95-known-development-authenticate-final v95-deferred-final-authenticate-materialization-preregistration
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v95_deferred_final unlock

v95-deferred-final-authenticate: v95-known-development-python-check
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v95_deferred_final authenticate

v95-deferred-final-template: v95-deferred-final-authenticate
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v95_deferred_final template

v95-deferred-final-materialization-preflight: v95-deferred-final-authenticate
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v95_deferred_final materialization-preflight

v95-deferred-final-generate: v95-deferred-final-authenticate
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v95_deferred_final_materialization run-stage --stage generate

v95-deferred-final-render: v95-deferred-final-generate
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v95_deferred_final_materialization run-stage --stage render

v95-deferred-final-features: v95-deferred-final-render
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v95_deferred_final_materialization run-stage --stage features

v95-deferred-final-maps: v95-deferred-final-features
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v95_deferred_final_materialization run-stage --stage maps

v95-deferred-final-memory: v95-deferred-final-maps
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v95_deferred_final_materialization run-stage --stage memory

v95-deferred-final-qa-raw: v95-deferred-final-memory
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v95_deferred_final_materialization run-stage --stage qa_raw

v95-deferred-final-qa-select: v95-deferred-final-qa-raw
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v95_deferred_final_materialization run-stage --stage qa_select

v95-deferred-final-questions: v95-deferred-final-qa-select
	PYTHONPATH=src $(PYTHON) -m semantic_3d_chat.evaluation.v95_deferred_final_materialization run-stage --stage questions

# V96 is a fixed-final repair successor to V95. Its model-free derivation
# opens only training data plus V95's sealed aggregate evidence. Sealing and
# execution remain explicit; `v96-train` requires authenticated persisted
# topology evidence and never performs checkpoint selection on development.
.PHONY: v96-check v96-authenticate-parent v96-derive v96-preregister
.PHONY: v96-preflight v96-authenticate v96-topology-smoke
.PHONY: v96-authenticate-topology v96-train

v96-check:
	PYTHONPATH=src $(PYTHON) -m pytest -q \
		tests/test_v96_atomic_pair_repair.py \
		tests/test_v96_trainer.py
	PYTHONPATH=src $(PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v96_atomic_pair_repair_preflight.py \
		src/semantic_3d_chat/training/train_v96_atomic_pair_repair.py \
		tests/test_v96_atomic_pair_repair.py \
		tests/test_v96_trainer.py

v96-authenticate-parent:
	PYTHONPATH=src $(PYTHON) \
		-m semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight authenticate-parent

v96-derive:
	PYTHONPATH=src $(PYTHON) \
		-m semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight derive

v96-preregister:
	PYTHONPATH=src $(PYTHON) \
		-m semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight preregister

v96-preflight:
	PYTHONPATH=src $(PYTHON) \
		-m semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight cpu-preflight

v96-authenticate:
	PYTHONPATH=src $(PYTHON) \
		-m semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight authenticate

v96-topology-smoke: v96-authenticate
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v96_atomic_pair_repair --topology-smoke

v96-authenticate-topology:
	PYTHONPATH=src $(PYTHON) \
		-m semantic_3d_chat.training.train_v96_atomic_pair_repair \
		--authenticate-topology-smoke

v96-train: v96-authenticate-topology
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.training.train_v96_atomic_pair_repair

# V96 known-development follows the same strict process split as V95, while
# adding the preregistered 192-side stable-invariant false-change control.
# The implementation seal is an explicit create-once prerequisite and is not
# produced by the evaluation target itself.
.PHONY: v96-known-development-check v96-known-development-python-check
.PHONY: v96-known-development-seal-implementation
.PHONY: v96-known-development-authenticate-implementation
.PHONY: v96-known-development-seal-candidate-attestation
.PHONY: v96-known-development-authenticate-candidate-attestation
.PHONY: v96-known-development-predict
.PHONY: v96-known-development-authenticate-predictions
.PHONY: v96-known-development-score
.PHONY: v96-known-development-authenticate-score
.PHONY: v96-known-development-nll
.PHONY: v96-known-development-authenticate-nll
.PHONY: v96-known-development-seal
.PHONY: v96-known-development-authenticate-final
.PHONY: v96-known-development-evaluate

v96-known-development-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v96_known_development_harness.py \
		tests/test_v96_known_development_implementation_seal.py \
		tests/test_v96_known_development_auth_v2.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v96_evaluation_io_v2.py \
		src/semantic_3d_chat/evaluation/v96_known_development_candidate_attestation.py \
		src/semantic_3d_chat/evaluation/v96_known_development_common_v2.py \
		src/semantic_3d_chat/evaluation/v96_known_development_implementation_v2.py \
		src/semantic_3d_chat/evaluation/predict_v96_known_development_v2.py \
		src/semantic_3d_chat/evaluation/authenticate_v96_known_development_v2.py \
		src/semantic_3d_chat/evaluation/score_v96_known_development_v2.py \
		src/semantic_3d_chat/evaluation/nll_v96_known_development_v2.py \
		src/semantic_3d_chat/evaluation/seal_v96_known_development_v2.py \
		src/semantic_3d_chat/evaluation/v96_known_development_common.py \
		src/semantic_3d_chat/evaluation/v96_known_development_implementation.py \
		src/semantic_3d_chat/evaluation/predict_v96_known_development.py \
		src/semantic_3d_chat/evaluation/authenticate_v96_known_development.py \
		src/semantic_3d_chat/evaluation/score_v96_known_development.py \
		src/semantic_3d_chat/evaluation/nll_v96_known_development.py \
		src/semantic_3d_chat/evaluation/seal_v96_known_development.py \
		tests/test_v96_known_development_harness.py \
		tests/test_v96_known_development_implementation_seal.py \
		tests/test_v96_known_development_auth_v2.py

v96-known-development-python-check:
	@test -x .venv-gemma4/bin/python || { \
		echo "Missing required V96 interpreter: .venv-gemma4/bin/python" >&2; \
		exit 1; \
	}

v96-known-development-seal-implementation: v96-known-development-python-check
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.v96_known_development_implementation_v2 seal

v96-known-development-authenticate-implementation: v96-known-development-python-check
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.v96_known_development_implementation_v2 authenticate

v96-known-development-seal-candidate-attestation: v96-known-development-authenticate-implementation
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.v96_known_development_candidate_attestation seal

v96-known-development-authenticate-candidate-attestation: v96-known-development-authenticate-implementation
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.v96_known_development_candidate_attestation authenticate

v96-known-development-predict: v96-known-development-authenticate-candidate-attestation v96-authenticate-topology
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.predict_v96_known_development_v2

v96-known-development-authenticate-predictions: v96-known-development-authenticate-implementation
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.authenticate_v96_known_development_v2 prediction

v96-known-development-score: v96-known-development-authenticate-predictions
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.score_v96_known_development_v2

v96-known-development-authenticate-score: v96-known-development-authenticate-implementation
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.authenticate_v96_known_development_v2 structured

v96-known-development-nll: v96-known-development-authenticate-score
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.nll_v96_known_development_v2

v96-known-development-authenticate-nll: v96-known-development-authenticate-implementation
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.authenticate_v96_known_development_v2 nll

v96-known-development-seal: v96-known-development-authenticate-score v96-known-development-authenticate-nll
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.seal_v96_known_development_v2 seal

v96-known-development-authenticate-final: v96-known-development-authenticate-implementation
	PYTHONPATH=src .venv-gemma4/bin/python \
		-m semantic_3d_chat.evaluation.seal_v96_known_development_v2 authenticate

v96-known-development-evaluate: v96-known-development-authenticate-implementation
	$(MAKE) v96-known-development-predict
	$(MAKE) v96-known-development-authenticate-predictions
	$(MAKE) v96-known-development-score
	$(MAKE) v96-known-development-authenticate-score
	$(MAKE) v96-known-development-nll
	$(MAKE) v96-known-development-authenticate-nll
	$(MAKE) v96-known-development-seal
	$(MAKE) v96-known-development-authenticate-final

# V96 reuses V95's already-sealed, outcome-independent physical recipe. The
# QA-selection child is authorization-repaired to use V96; all other physical
# child commands remain byte-identical. These targets enforce the wrapper:
# a complete V96 known-development PASS and a separately invoked create-once
# unlock are mandatory. No check, preflight, or authentication target creates
# a deferred scene, and no materialization target auto-creates the unlock.
.PHONY: v96-deferred-final-check
.PHONY: v96-deferred-final-authenticate-materialization-preregistration
.PHONY: v96-deferred-final-seal-evaluation-preregistration
.PHONY: v96-deferred-final-authenticate-evaluation-preregistration
.PHONY: v96-deferred-final-preflight v96-deferred-final-unlock
.PHONY: v96-deferred-final-authenticate v96-deferred-final-template
.PHONY: v96-deferred-final-materialization-preflight
.PHONY: v96-deferred-final-generate v96-deferred-final-render
.PHONY: v96-deferred-final-features v96-deferred-final-maps
.PHONY: v96-deferred-final-memory v96-deferred-final-qa-raw
.PHONY: v96-deferred-final-qa-select v96-deferred-final-questions
.PHONY: v96-deferred-final-authenticate-materialized-predictor
.PHONY: v96-deferred-final-authenticate-materialized-label
.PHONY: v96-deferred-final-predict-v96 v96-deferred-final-authenticate-v96
.PHONY: v96-deferred-final-predict-v94 v96-deferred-final-authenticate-v94
.PHONY: v96-deferred-final-score v96-deferred-final-authenticate-score
.PHONY: v96-deferred-final-nll v96-deferred-final-authenticate-nll
.PHONY: v96-deferred-final-seal v96-deferred-final-authenticate-final
.PHONY: v96-deferred-final-evaluate

v96-deferred-final-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v96_deferred_final.py \
		tests/test_v96_deferred_final_materialization.py \
		tests/test_v96_deferred_final_qa.py \
		tests/test_v96_deferred_final_harness.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/evaluation/v96_deferred_final.py \
		src/semantic_3d_chat/evaluation/v96_deferred_final_materialization.py \
		src/semantic_3d_chat/evaluation/v96_deferred_final_qa.py \
		src/semantic_3d_chat/evaluation/v96_deferred_final_evaluation.py \
		src/semantic_3d_chat/evaluation/v96_deferred_final_common.py \
		src/semantic_3d_chat/evaluation/predict_v96_deferred_final.py \
		src/semantic_3d_chat/evaluation/score_v96_deferred_final.py \
		src/semantic_3d_chat/evaluation/nll_v96_deferred_final.py \
		src/semantic_3d_chat/evaluation/seal_v96_deferred_final.py \
		src/semantic_3d_chat/evaluation/authenticate_v96_deferred_final.py \
		tests/test_v96_deferred_final.py \
		tests/test_v96_deferred_final_materialization.py \
		tests/test_v96_deferred_final_qa.py \
		tests/test_v96_deferred_final_harness.py

v96-deferred-final-authenticate-materialization-preregistration: v96-known-development-python-check
	PYTHONPATH=src $(PYTHON) \
		-m semantic_3d_chat.evaluation.v95_deferred_final_materialization authenticate

v96-deferred-final-seal-evaluation-preregistration: v96-known-development-authenticate-final v96-deferred-final-authenticate-materialization-preregistration
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_evaluation seal

v96-deferred-final-authenticate-evaluation-preregistration: v96-known-development-python-check
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_evaluation authenticate

v96-deferred-final-preflight: v96-known-development-authenticate-final v96-deferred-final-authenticate-materialization-preregistration v96-deferred-final-authenticate-evaluation-preregistration
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final preflight

v96-deferred-final-unlock: v96-known-development-authenticate-final v96-deferred-final-authenticate-materialization-preregistration v96-deferred-final-authenticate-evaluation-preregistration
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final unlock

v96-deferred-final-authenticate: v96-known-development-python-check
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final authenticate

v96-deferred-final-template: v96-deferred-final-authenticate
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final template

v96-deferred-final-materialization-preflight: v96-deferred-final-authenticate
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_materialization preflight

v96-deferred-final-generate: v96-deferred-final-authenticate
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_materialization run-stage --stage generate

v96-deferred-final-render: v96-deferred-final-generate
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_materialization run-stage --stage render

v96-deferred-final-features: v96-deferred-final-render
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_materialization run-stage --stage features

v96-deferred-final-maps: v96-deferred-final-features
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_materialization run-stage --stage maps

v96-deferred-final-memory: v96-deferred-final-maps
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_materialization run-stage --stage memory

v96-deferred-final-qa-raw: v96-deferred-final-memory
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_materialization run-stage --stage qa_raw

v96-deferred-final-qa-select: v96-deferred-final-qa-raw
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_materialization run-stage --stage qa_select

v96-deferred-final-questions: v96-deferred-final-qa-select
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_deferred_final_materialization run-stage --stage questions

v96-deferred-final-authenticate-materialized-predictor: v96-deferred-final-questions
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.authenticate_v96_deferred_final materialized-predictor

v96-deferred-final-authenticate-materialized-label: v96-deferred-final-questions
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.authenticate_v96_deferred_final materialized-label

v96-deferred-final-predict-v96: v96-deferred-final-authenticate-materialized-predictor
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.predict_v96_deferred_final v96

v96-deferred-final-authenticate-v96: v96-deferred-final-authenticate-materialized-predictor
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.authenticate_v96_deferred_final predictions-v96

v96-deferred-final-predict-v94: v96-deferred-final-authenticate-materialized-predictor
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.predict_v96_deferred_final v94

v96-deferred-final-authenticate-v94: v96-deferred-final-authenticate-materialized-predictor
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.authenticate_v96_deferred_final predictions-v94

v96-deferred-final-score: v96-deferred-final-authenticate-v96 v96-deferred-final-authenticate-v94
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.score_v96_deferred_final score

v96-deferred-final-authenticate-score: v96-deferred-final-authenticate-v96 v96-deferred-final-authenticate-v94
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.authenticate_v96_deferred_final structured

v96-deferred-final-nll: v96-deferred-final-authenticate-v96 v96-deferred-final-authenticate-v94
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.nll_v96_deferred_final measure

v96-deferred-final-authenticate-nll: v96-deferred-final-authenticate-v96 v96-deferred-final-authenticate-v94
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.authenticate_v96_deferred_final nll

v96-deferred-final-seal: v96-deferred-final-authenticate-score v96-deferred-final-authenticate-nll
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.seal_v96_deferred_final seal

v96-deferred-final-authenticate-final: v96-known-development-python-check
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.authenticate_v96_deferred_final final

v96-deferred-final-evaluate: v96-deferred-final-authenticate-evaluation-preregistration
	$(MAKE) v96-deferred-final-unlock
	$(MAKE) v96-deferred-final-questions
	$(MAKE) v96-deferred-final-predict-v96
	$(MAKE) v96-deferred-final-authenticate-v96
	$(MAKE) v96-deferred-final-predict-v94
	$(MAKE) v96-deferred-final-authenticate-v94
	$(MAKE) v96-deferred-final-score
	$(MAKE) v96-deferred-final-authenticate-score
	$(MAKE) v96-deferred-final-nll
	$(MAKE) v96-deferred-final-authenticate-nll
	$(MAKE) v96-deferred-final-seal
	$(MAKE) v96-deferred-final-authenticate-final

# V96 runtime release is a separate post-held-out leakage gate. None of these
# targets changes the project-wide default (V89). Candidate packaging requires
# every known-development and deferred-final gate; promotion additionally
# requires the six-scene oracle-physically-unavailable child-process smoke.
V96_RELEASE_SCENE ?= scene_000025
.PHONY: v96-runtime-check v96-release-authenticate v96-release-prepare
.PHONY: v96-release-verify-candidate v96-release-smoke v96-release-promote
.PHONY: v96-release-verify v96-release-cleanup-failed-candidate
.PHONY: v96-release-recover-oracles v96-release-cleanup-partial-release
.PHONY: v96-chat v96-demo v96-demo-check v96-demo-leakage
.PHONY: v96-robot-evidence v96-report-check v96-report
.PHONY: v96-handoff-check v96-handoff-demo
.PHONY: v96-report-live-check v96-report-live v96-handoff-live
.PHONY: v96-handoff-live-demo

v96-runtime-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q \
		tests/test_v96_explicit_candidate_runtime.py \
		tests/test_v96_candidate_robot_bridge.py \
		tests/test_v96_strict_runtime_release.py \
		tests/test_v96_final_reporting.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/chat/v96_explicit_candidate_runtime.py \
		src/semantic_3d_chat/chat/v96_explicit_candidate_authorize.py \
		src/semantic_3d_chat/chat/v96_explicit_candidate_cli.py \
		src/semantic_3d_chat/chat/v96_strict_multiscene_runtime.py \
		src/semantic_3d_chat/chat/v96_strict_multiscene_cli.py \
		src/semantic_3d_chat/mcp_server/server.py \
		src/semantic_3d_chat/robot/runtime_refresh.py \
		src/semantic_3d_chat/robot/semantic_mapping.py \
		src/semantic_3d_chat/robot/v96_candidate_refresh.py \
		src/semantic_3d_chat/evaluation/v96_candidate_mcp_live_smoke.py \
		src/semantic_3d_chat/evaluation/v96_strict_runtime_release.py \
		src/semantic_3d_chat/evaluation/v96_final_reporting.py \
		scripts/run_v96_candidate_mcp_live_smoke.py \
		tests/test_v96_explicit_candidate_runtime.py \
		tests/test_v96_candidate_robot_bridge.py \
		tests/test_v96_strict_runtime_release.py \
		tests/test_v96_final_reporting.py

v96-release-authenticate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_strict_runtime_release authenticate

v96-release-prepare: v96-release-authenticate
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_strict_runtime_release prepare

v96-release-verify-candidate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_strict_runtime_release verify-candidate

v96-release-smoke: v96-release-verify-candidate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_strict_runtime_release smoke

v96-release-promote:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_strict_runtime_release promote

v96-release-verify:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_strict_runtime_release verify

v96-release-cleanup-failed-candidate:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_strict_runtime_release cleanup-failed-candidate

v96-release-recover-oracles:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_strict_runtime_release recover-oracles

v96-release-cleanup-partial-release:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_strict_runtime_release cleanup-partial-release

v96-chat:
	./scripts/run_v96_strict_multiscene_demo.sh --interactive --scene "$(V96_RELEASE_SCENE)"

v96-demo:
	./scripts/run_v96_strict_multiscene_demo.sh --scene "$(V96_RELEASE_SCENE)"

v96-demo-check:
	./scripts/run_v96_strict_multiscene_demo.sh --check --scene "$(V96_RELEASE_SCENE)"

v96-demo-leakage:
	./scripts/run_v96_strict_multiscene_demo.sh --leakage --scene "$(V96_RELEASE_SCENE)"

# This is deliberately model-free MCP readiness evidence. It authenticates the
# released V96 candidate, numeric-only tools, and forbidden-read audit without
# executing navigation or claiming embodied task success.
v96-robot-evidence: v96-release-verify
	V96_ROBOT_PYTHON="$(GEMMA4_PYTHON)" \
	V96_ROBOT_SCENE="$(GEMMA4_V96_ROBOT_SCENE)" \
	V96_ROBOT_CONFIG="$(GEMMA4_EMBODIED_CONFIG)" \
	V96_ROBOT_CHECKPOINT="$(GEMMA4_V96_MCP_BASE_CHECKPOINT)" \
	V96_ROBOT_ASSET="$(GEMMA4_V96_ROBOT_ASSET)" \
	V96_ROBOT_STATE="$(GEMMA4_ROBOT_STATE_CHECKPOINT)" \
	V96_ROBOT_HOOK="$(GEMMA4_V96_MCP_BRIDGE_HOOK)" \
	V96_ROBOT_MEMORY="$(GEMMA4_V96_ROBOT_MEMORY)" \
	V96_ROBOT_MAP="$(GEMMA4_V96_ROBOT_MAP)" \
	V96_ROBOT_SCANS="$(GEMMA4_V96_ROBOT_SCANS)" \
	V96_ROBOT_AUDIT="$(GEMMA4_V96_ROBOT_AUDIT)" \
	V96_ROBOT_EVIDENCE="$(GEMMA4_V96_ROBOT_EVIDENCE)" \
	./scripts/run_v96_embodied_preflight_evidence.sh

v96-report-check: v96-robot-evidence
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_final_reporting check \
		--python "$(GEMMA4_PYTHON)" \
		--robot-evidence "$(GEMMA4_V96_ROBOT_EVIDENCE)"

v96-report: v96-report-check
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_final_reporting build \
		--python "$(GEMMA4_PYTHON)" \
		--robot-evidence "$(GEMMA4_V96_ROBOT_EVIDENCE)" \
		--metrics-output "$(GEMMA4_V96_INTEGRATION_METRICS)" \
		--markdown-output "$(GEMMA4_V96_INTEGRATION_REPORT)"

# Safe one-command post-training handoff. The check remains model-free; the
# demo target loads Gemma only after every sealed result and release gate passes.
v96-handoff-check: v96-report-check v96-demo-check
	@echo "V96 handoff authenticated; V89 remains the project-wide default."
	@echo "V96 embodied evidence is MCP readiness only; navigation is not measured."

v96-handoff-demo: v96-handoff-check v96-report
	./scripts/run_v96_strict_multiscene_demo.sh --scene "$(V96_RELEASE_SCENE)"

# Optional measured numeric-refresh addendum. Generate its create-once input
# explicitly with `make v96-explicit-candidate-embodied-mcp-live-smoke`; these
# targets authenticate but never silently launch that heavy Gemma/Blender run.
v96-report-live-check: v96-report-check
	@test -f "$(GEMMA4_V96_LIVE_ROBOT_EVIDENCE)" || { \
		echo "Missing finite V96 numeric MCP evidence; run make v96-explicit-candidate-embodied-mcp-live-smoke first." >&2; \
		exit 2; \
	}
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_final_reporting check \
		--python "$(GEMMA4_PYTHON)" \
		--robot-evidence "$(GEMMA4_V96_ROBOT_EVIDENCE)" \
		--live-robot-evidence "$(GEMMA4_V96_LIVE_ROBOT_EVIDENCE)"

v96-report-live: v96-report-live-check
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_final_reporting build \
		--python "$(GEMMA4_PYTHON)" \
		--robot-evidence "$(GEMMA4_V96_ROBOT_EVIDENCE)" \
		--live-robot-evidence "$(GEMMA4_V96_LIVE_ROBOT_EVIDENCE)" \
		--metrics-output "$(GEMMA4_V96_LIVE_INTEGRATION_METRICS)" \
		--markdown-output "$(GEMMA4_V96_LIVE_INTEGRATION_REPORT)"

v96-handoff-live: v96-report-live-check v96-demo-check
	@echo "V96 finite numeric MCP refresh authenticated; conversational navigation remains unmeasured."

v96-handoff-live-demo: v96-handoff-live v96-report-live
	./scripts/run_v96_strict_multiscene_demo.sh --scene "$(V96_RELEASE_SCENE)"

# Downstream promoted-release embodied path. These commands remain inert until
# the strict static release is promoted. The held-out runtime receives only the
# create-once sanitized task files; scorer-only categories and oracle JSON stay
# outside its six isolated scene processes.
V96_EMBODIED_SCENE ?= scene_000025
V96_EMBODIED_ASSET ?= data/runtime_assets/$(V96_EMBODIED_SCENE)/$(subst scene_,s_,$(V96_EMBODIED_SCENE)).blend
V96_EMBODIED_INSTRUCTION ?= Face the chair, then stop.
V96_EMBODIED_RESULT ?= reports/gemma4/metrics/v96_promoted_embodied_$(V96_EMBODIED_SCENE).json
V96_EMBODIED_AUDIT ?= reports/gemma4/metrics/v96_promoted_embodied_access_$(V96_EMBODIED_SCENE).json
V96_EMBODIED_PERSISTENT_MAP ?= data_gemma4/runtime/v96_embodied/robot/$(V96_EMBODIED_SCENE)/semantic_map.npz
V96_EMBODIED_SCAN_OUTPUT ?= data_gemma4/runtime/v96_embodied/scans/$(V96_EMBODIED_SCENE)
V96_EMBODIED_PREREGISTRATION ?= reports/gemma4/metrics/v96_embodied_navigation_preregistration.json
V96_EMBODIED_RUNTIME_INPUT_ROOT ?= reports/gemma4/embodied/runtime_inputs/v96
V96_EMBODIED_RUNTIME_RESULT_ROOT ?= reports/gemma4/embodied/runtime_results/v96
V96_EMBODIED_SCRATCH_ROOT ?= data_gemma4/runtime/v96_embodied/heldout_scratch
V96_EMBODIED_HELDOUT_SCORE ?= reports/gemma4/metrics/v96_embodied_navigation_score.json
V96_EMBODIED_ORACLE_ROOT ?= data/oracle

.PHONY: v96-embodied-check v96-embodied-heldout-preflight
.PHONY: v96-embodied-heldout-preregister v96-embodied-heldout-authenticate
.PHONY: v96-embodied-heldout-run v96-embodied-heldout-score v96-embodied-live

v96-embodied-check:
	PYTHONPATH=src $(GEMMA4_PYTHON) -m pytest -q tests/test_v96_embodied_downstream.py
	PYTHONPATH=src $(GEMMA4_PYTHON) -m ruff check \
		src/semantic_3d_chat/robot/v96_release_embodied.py \
		src/semantic_3d_chat/robot/v96_release_action.py \
		src/semantic_3d_chat/robot/v96_co_resident_mcp_agent.py \
		src/semantic_3d_chat/robot/v96_embodied_task_runner.py \
		src/semantic_3d_chat/robot/v96_release_agent_cli.py \
		src/semantic_3d_chat/robot/v96_runtime_source_contract.py \
		src/semantic_3d_chat/evaluation/v96_embodied_heldout.py \
		scripts/run_v96_embodied_heldout.py \
		tests/test_v96_embodied_downstream.py

# Readiness only: writes nothing and fails closed until the static release is
# promoted. Preregistration remains a separate explicit evaluator command.
v96-embodied-heldout-preflight:
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_embodied_heldout preflight

# Create-once scientific protocol. On the current authenticated V96 NO-GO,
# the release prerequisite fails before any preregistration or held-out file is
# created. If a future V96 release is validly promoted, run these in order.
v96-embodied-heldout-preregister: v96-release-verify
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_embodied_heldout preregister \
		--output "$(V96_EMBODIED_PREREGISTRATION)"

v96-embodied-heldout-authenticate: v96-release-verify
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_embodied_heldout authenticate \
		--preregistration "$(V96_EMBODIED_PREREGISTRATION)"

v96-embodied-heldout-run: v96-embodied-heldout-authenticate
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_embodied_heldout run \
		--preregistration "$(V96_EMBODIED_PREREGISTRATION)" \
		--runtime-input-root "$(V96_EMBODIED_RUNTIME_INPUT_ROOT)" \
		--runtime-result-root "$(V96_EMBODIED_RUNTIME_RESULT_ROOT)" \
		--scratch-root "$(V96_EMBODIED_SCRATCH_ROOT)"

v96-embodied-heldout-score: v96-embodied-heldout-authenticate
	PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.evaluation.v96_embodied_heldout score \
		--preregistration "$(V96_EMBODIED_PREREGISTRATION)" \
		--runtime-result-root "$(V96_EMBODIED_RUNTIME_RESULT_ROOT)" \
		--oracle-root "$(V96_EMBODIED_ORACLE_ROOT)" \
		--output "$(V96_EMBODIED_HELDOUT_SCORE)"

v96-embodied-live: v96-release-verify
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src $(GEMMA4_PYTHON) \
		-m semantic_3d_chat.robot.v96_release_agent_cli \
		--scene "$(V96_EMBODIED_SCENE)" \
		--runtime-asset "$(V96_EMBODIED_ASSET)" \
		--persistent-map "$(V96_EMBODIED_PERSISTENT_MAP)" \
		--scan-output "$(V96_EMBODIED_SCAN_OUTPUT)" \
		--instruction "$(V96_EMBODIED_INSTRUCTION)" \
		--output "$(V96_EMBODIED_RESULT)" \
		--audit-report "$(V96_EMBODIED_AUDIT)"

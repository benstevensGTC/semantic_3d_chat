PYTHON := .venv/bin/python
GEMMA4_PYTHON := .venv-gemma4/bin/python
GEMMA4_CONFIG ?= configs/gemma4_e2b.yaml
GEMMA4_MODEL := google/gemma-4-E2B-it
GEMMA4_REVISION := 3e22461f65e89153144f8adb70e3b8c2cc9845a7
# Intentionally empty: chat/evaluation accepts only an explicitly selected,
# behaviorally promoted Gemma checkpoint/config pair.
GEMMA4_STATIC_CONFIG ?=
GEMMA4_STATIC_CHECKPOINT ?=
GEMMA4_EVAL_SPLIT ?= test
GEMMA4_QUESTIONS_MANIFEST ?= reports/gemma4/questions/$(GEMMA4_EVAL_SPLIT).json
GEMMA4_STATIC_PREDICTIONS ?= reports/gemma4/predictions/$(GEMMA4_EVAL_SPLIT).jsonl
GEMMA4_STATIC_METRICS ?= reports/gemma4/metrics/static_qa_$(GEMMA4_EVAL_SPLIT).json
GEMMA4_CONTROL_PREDICTIONS ?= reports/gemma4/predictions/controls/$(GEMMA4_EVAL_SPLIT)
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
BLENDER := blender
CONFIG ?= configs/default.yaml
BATCH_CONFIG ?= configs/experiments/multiscene.yaml
SCENE ?= scene_000001
CHECKPOINT ?=

.PHONY: doctor setup download-models download-baselines setup-gemma4-probe download-gemma4-config download-gemma4-weights gemma4-probe gemma4-probe-test extract-gemma4-scene build-gemma4-map gemma4-semantic-sanity gemma4-extract-smoke gemma4-build-smoke-map train-gemma4 gemma4-v18-preflight gemma4-v18-stage1 gemma4-v18-verify-update1 gemma4-v18-resume-screen gemma4-v18-select gemma4-v18-screen gemma4-v19-preflight gemma4-v19-stage1 gemma4-v19-verify-update1 gemma4-v19-resume-screen gemma4-v19-select gemma4-v19-screen gemma4-v19-prepare-extension gemma4-v19-run-extension gemma4-v19-select-extension gemma4-v19-extension gemma4-v20-preflight gemma4-v20-stage1 gemma4-v20-verify-update1 gemma4-v20-resume-screen gemma4-v20-select gemma4-v20-screen gemma4-v20-prepare-extension gemma4-v20-run-extension gemma4-v20-select-extension gemma4-v20-extension require-gemma4-promoted chat-gemma4 gemma4-prepare-questions gemma4-predict-static gemma4-score-static gemma4-evaluate-static gemma4-predict-controls gemma4-chat-static generate-smoke-scene render-smoke-scan generate-scene-batch render-scene-batch multiscene-dry-run build-smoke-map semantic-sanity generate-dataset train evaluate evaluate-oracle-text evaluate-direct-images chat web robot robot-evaluate mcp report demo demo-check demo-leakage test

doctor:
	./scripts/doctor.sh

setup:
	uv sync --extra dev --extra mcp

download-models:
	$(PYTHON) scripts/download_models.py --config $(CONFIG)

download-baselines:
	$(PYTHON) scripts/download_baseline_models.py --config $(CONFIG)

setup-gemma4-probe:
	@test -x "$(GEMMA4_PYTHON)" || uv venv --python 3.12 --seed .venv-gemma4
	uv pip install --python $(GEMMA4_PYTHON) -r requirements-gemma4-probe.txt

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
	@PYTHONPATH=src $(GEMMA4_PYTHON) scripts/demo_check.py --config "$(GEMMA4_STATIC_CONFIG)" --scene "$(SCENE)" --checkpoint "$(GEMMA4_STATIC_CHECKPOINT)" --resolve-checkpoint --require-promotion >/dev/null

chat-gemma4: require-gemma4-promoted
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.chat.cli --config "$(GEMMA4_STATIC_CONFIG)" --scene "$(SCENE)" --checkpoint "$(GEMMA4_STATIC_CHECKPOINT)"

# Reproducible static evaluation for a behaviorally promoted checkpoint only.
# Historical failed wiring runs remain available as artifacts, not defaults.
gemma4-prepare-questions: require-gemma4-promoted
	@test -x "$(GEMMA4_PYTHON)"
	@test -f "$(GEMMA4_STATIC_CONFIG)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.prepare_questions --config "$(GEMMA4_STATIC_CONFIG)" --split $(GEMMA4_EVAL_SPLIT) --output $(GEMMA4_QUESTIONS_MANIFEST) --force

gemma4-predict-static: gemma4-prepare-questions
	@test -x "$(GEMMA4_PYTHON)"
	@test -f "$(GEMMA4_STATIC_CONFIG)"
	@test -f "$(GEMMA4_STATIC_CHECKPOINT)/adapter.safetensors"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.predict --config "$(GEMMA4_STATIC_CONFIG)" --split $(GEMMA4_EVAL_SPLIT) --questions-manifest $(GEMMA4_QUESTIONS_MANIFEST) --checkpoint "$(GEMMA4_STATIC_CHECKPOINT)" --output $(GEMMA4_STATIC_PREDICTIONS)

gemma4-score-static: require-gemma4-promoted
	@test -f "$(GEMMA4_STATIC_PREDICTIONS)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.run --config "$(GEMMA4_STATIC_CONFIG)" --references data/qa/$(GEMMA4_EVAL_SPLIT).jsonl --predictions $(GEMMA4_STATIC_PREDICTIONS) --output $(GEMMA4_STATIC_METRICS)

gemma4-evaluate-static: gemma4-predict-static
	@test -f "$(GEMMA4_STATIC_PREDICTIONS)"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.run --config "$(GEMMA4_STATIC_CONFIG)" --references data/qa/$(GEMMA4_EVAL_SPLIT).jsonl --predictions $(GEMMA4_STATIC_PREDICTIONS) --output $(GEMMA4_STATIC_METRICS)

gemma4-predict-controls: gemma4-prepare-questions
	@test -x "$(GEMMA4_PYTHON)"
	@test -f "$(GEMMA4_STATIC_CONFIG)"
	@test -f "$(GEMMA4_STATIC_CHECKPOINT)/adapter.safetensors"
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.evaluation.control_predict --config "$(GEMMA4_STATIC_CONFIG)" --split $(GEMMA4_EVAL_SPLIT) --questions-manifest $(GEMMA4_QUESTIONS_MANIFEST) --checkpoint "$(GEMMA4_STATIC_CHECKPOINT)" --output-dir $(GEMMA4_CONTROL_PREDICTIONS)

gemma4-chat-static: require-gemma4-promoted
	PYTHONPATH=src $(GEMMA4_PYTHON) -m semantic_3d_chat.chat.cli --config "$(GEMMA4_STATIC_CONFIG)" --scene $(SCENE) --checkpoint "$(GEMMA4_STATIC_CHECKPOINT)"

generate-smoke-scene:
	$(BLENDER) --background --python blender/generate_scene.py -- --config $(CONFIG) --scene $(SCENE)

render-smoke-scan: generate-smoke-scene
	$(BLENDER) --background data/oracle/$(SCENE)/scene.blend --python blender/render_scan.py -- --config $(CONFIG) --scene $(SCENE)

generate-scene-batch:
	$(PYTHON) scripts/generate_scene_batch.py --config $(BATCH_CONFIG) --stage generate

render-scene-batch:
	$(PYTHON) scripts/generate_scene_batch.py --config $(BATCH_CONFIG) --stage render

multiscene-dry-run:
	$(PYTHON) scripts/generate_scene_batch.py --config $(BATCH_CONFIG) --stage all --dry-run

build-smoke-map:
	$(PYTHON) -m semantic_3d_chat.vision.encoder --config $(CONFIG) --scene $(SCENE) --offline
	$(PYTHON) scripts/build_map.py --config $(CONFIG) --scene $(SCENE)

semantic-sanity:
	$(PYTHON) -m semantic_3d_chat.evaluation.semantic_sanity --config $(CONFIG) --scene $(SCENE)

generate-dataset:
	$(PYTHON) -m semantic_3d_chat.data.qa_generator --config $(CONFIG)

train:
	$(PYTHON) -m semantic_3d_chat.training.train_adapter --config $(CONFIG)

evaluate:
	$(PYTHON) -m semantic_3d_chat.evaluation.run --config $(CONFIG)

evaluate-oracle-text:
	$(PYTHON) -m semantic_3d_chat.evaluation.oracle_text_baseline --config $(CONFIG)
	$(PYTHON) -m semantic_3d_chat.evaluation.run --config $(CONFIG) --references data/qa/test.jsonl --predictions reports/predictions/oracle_text.jsonl --output reports/metrics/oracle_text.json

evaluate-direct-images:
	$(PYTHON) -m semantic_3d_chat.evaluation.direct_multiview_baseline --config $(CONFIG)
	$(PYTHON) -m semantic_3d_chat.evaluation.run --config $(CONFIG) --references data/qa/test.jsonl --predictions reports/predictions/direct_multiview.jsonl --output reports/metrics/direct_multiview.json

# Preserved legacy CLIP/Qwen runtime. Use chat-gemma4 for the fail-closed Gemma path.
chat:
	@if [ "$$(PYTHONPATH=src $(PYTHON) -c 'import sys; from semantic_3d_chat.config import load_config; print(load_config(sys.argv[1]).get("language", {}).get("backend", "auto"))' "$(CONFIG)")" = "gemma4" ]; then \
		echo "The generic chat target is legacy-only. Use chat-gemma4 with an explicit promoted Gemma config/checkpoint." >&2; \
		exit 2; \
	fi
	@checkpoint="$(CHECKPOINT)"; \
	if [ -z "$$checkpoint" ]; then \
		checkpoint="$$( $(PYTHON) scripts/demo_check.py --config "$(CONFIG)" --scene "$(SCENE)" --resolve-checkpoint )" || exit $$?; \
	fi; \
	$(PYTHON) -m semantic_3d_chat.chat.cli --config "$(CONFIG)" --scene "$(SCENE)" --checkpoint "$$checkpoint"

web:
	@if [ "$$(PYTHONPATH=src $(PYTHON) -c 'import sys; from semantic_3d_chat.config import load_config; print(load_config(sys.argv[1]).get("language", {}).get("backend", "auto"))' "$(CONFIG)")" = "gemma4" ]; then \
		echo "The generic web target is legacy-only until a promoted Gemma checkpoint exists." >&2; \
		exit 2; \
	fi
	@checkpoint="$(CHECKPOINT)"; \
	if [ -z "$$checkpoint" ]; then \
		checkpoint="$$( $(PYTHON) scripts/demo_check.py --config "$(CONFIG)" --scene "$(SCENE)" --resolve-checkpoint )" || exit $$?; \
	fi; \
	$(PYTHON) -m semantic_3d_chat.chat.web_app --config "$(CONFIG)" --scene "$(SCENE)" --checkpoint "$$checkpoint"

robot:
	$(PYTHON) -m semantic_3d_chat.robot.agent_loop --config $(CONFIG) --scene $(SCENE)

robot-evaluate:
	$(PYTHON) -m semantic_3d_chat.evaluation.robot_benchmark --config $(CONFIG) --scene $(SCENE)

mcp:
	$(PYTHON) -m semantic_3d_chat.mcp_server.server --config $(CONFIG) --scene $(SCENE)

report:
	$(PYTHON) scripts/build_report.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest

# Generic legacy demo infrastructure. If CONFIG selects Gemma, the script
# requires an explicit checkpoint plus its accepted promotion record.
demo:
	./scripts/run_full_demo.sh --config $(CONFIG) --scene $(SCENE) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

demo-check:
	./scripts/run_full_demo.sh --check --config $(CONFIG) --scene $(SCENE) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

demo-leakage:
	./scripts/run_full_demo.sh --non-interactive --config $(CONFIG) --scene $(SCENE) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

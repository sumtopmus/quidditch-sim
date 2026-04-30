# Quidditch-Sim Makefile
# Usage:  make <target>  [RUN_NAME=ppo_hoop]  [TRIAL=20240416_143022]
#
# Run layout:  runs/<RUN_NAME>/<trial>/   (trial = auto-timestamp per training run)

CONDA_ENV  ?= uav
RUN_NAME   ?= ppo_hoop
TRIAL      ?=
PRETRAIN   ?=
CHECKPOINT ?=
RUNS_DIR   := runs
MODELS_DIR := models

# Resolve the trial directory for eval / promote:
#   TRIAL= given on CLI          → runs/$(RUN_NAME)/$(TRIAL)
#   RUN_NAME= given on CLI only  → latest trial inside that run
#   nothing given                → latest trial across all runs
_LATEST_IN_RUN  = $(shell ls -d $(RUNS_DIR)/$(RUN_NAME)/* 2>/dev/null | sort -t/ -k3 | tail -1)
_LATEST_OVERALL = $(shell ls -d $(RUNS_DIR)/*/* 2>/dev/null | sort -t/ -k3 | tail -1)
_TRIAL_DIR      = $(strip $(if $(TRIAL),\
                    $(RUNS_DIR)/$(RUN_NAME)/$(TRIAL),\
                    $(if $(filter command line,$(origin RUN_NAME)),\
                      $(_LATEST_IN_RUN),\
                      $(_LATEST_OVERALL))))
_LATEST_CKPT    = $(shell ls -1 "$(_TRIAL_DIR)/checkpoints/"*.zip 2>/dev/null | sort -V | tail -1 | sed 's/\.zip$$//')
# Run name extracted from the resolved trial dir (e.g. runs/ppo_hoop_randstart/... → ppo_hoop_randstart)
_RESUME_RUN     = $(word 2,$(subst /, ,$(_TRIAL_DIR)))

# Resolve the conda binary: prefer $CONDA_EXE (set by `conda init`), fall back to PATH.
CONDA := $(or $(CONDA_EXE),$(shell command -v conda 2>/dev/null))
ifeq ($(CONDA),)
$(error conda not found — activate a conda shell or set CONDA_EXE)
endif

# Run a command inside the conda env, streaming output in real time.
CONDA_RUN := $(CONDA) run --no-capture-output -n $(CONDA_ENV)
PYTHON    := $(CONDA_RUN) python
# macOS: mujoco.viewer.launch_passive() requires mjpython (owns the Cocoa main
# thread).  mjpython is a wrapper installed by the mujoco pip package; use it
# only for targets that open the interactive viewer.
MJPYTHON  := $(CONDA_RUN) mjpython

# ──────────────────────────────────────────────────────────────────────────────
.PHONY: help check check-viewer hover waypoint train resume eval eval-headless tensorboard promote repro install clean list-runs

.DEFAULT_GOAL := help

help: ## 📋 Show available targets
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Override variables:  make train RUN_NAME=my_run  PRETRAIN=models/...  CONDA_ENV=uav"

# ──────────────────────────────────────────────────────────────────────────────

check: ## ✅ Validate env headless (fast, no window)
	@$(PYTHON) scripts/check_env.py

check-viewer: ## 🪟 Validate env with MuJoCo viewer (interactive camera)
	@$(MJPYTHON) scripts/check_env.py --viewer

hover: ## 🚁 MuJoCo hover smoke test (opens viewer)
	@$(MJPYTHON) demo/hover_demo.py

waypoint: ## 📍 Fly a waypoint triangle with marker spheres (opens viewer)
	@$(MJPYTHON) demo/waypoint_demo.py

train: ## 🚀 Run PPO training  [RUN_NAME=...] [PRETRAIN=models/...] [overrides config]
	@$(PYTHON) scripts/train_ppo.py \
	  $(if $(filter command line,$(origin RUN_NAME)),--run-name $(RUN_NAME)) \
	  $(if $(PRETRAIN),--pretrain $(PRETRAIN)/best_model)

resume: ## ▶️  Resume from latest checkpoint  [RUN_NAME=...] [TRIAL=...] [CHECKPOINT=path/to/ckpt]
	@ckpt="$(or $(CHECKPOINT),$(_LATEST_CKPT))"; \
	 test -n "$$ckpt" || { echo "ERROR: no checkpoint found in $(_TRIAL_DIR)/checkpoints/ — check RUN_NAME= and TRIAL="; exit 1; }; \
	 $(PYTHON) scripts/train_ppo.py --run-name "$(_RESUME_RUN)" --resume "$$ckpt"

eval: ## 🎯 Evaluate best model visually  [RUN_NAME=...] [TRIAL=...] [EPISODES=10]
	@$(MJPYTHON) scripts/eval_ppo.py --model $(_TRIAL_DIR)/best_model --episodes $(or $(EPISODES),10)

eval-headless: ## 📈 Evaluate best model headless  [RUN_NAME=...] [TRIAL=...] [EPISODES=50]
	@$(PYTHON) scripts/eval_ppo.py --model $(_TRIAL_DIR)/best_model --no-render --episodes $(or $(EPISODES),50)

tensorboard: ## 📊 Launch TensorBoard — all runs, or [RUN_NAME=...] for one config
	@PYTHONWARNINGS=ignore $(CONDA_RUN) tensorboard \
	  --logdir $(if $(filter command line,$(origin RUN_NAME)),$(RUNS_DIR)/$(RUN_NAME),$(RUNS_DIR)) \
	  2>&1 | grep --line-buffered -v "pkg_resources\|TensorFlow installation not found\|experimental fast data\|--load_fast\|issues on GitHub\|tensorflow/tensorboard\|^[[:space:]]*$$"

# ──────────────────────────────────────────────────────────────────────────────

promote: ## 🏆 Promote best model to models/  [RUN_NAME=...] [TRIAL=...]
	@dir="$(_TRIAL_DIR)"; \
	 [ -n "$$dir" ] || { echo "ERROR: no trials found in $(RUNS_DIR)/"; exit 1; }; \
	 src="$$dir/best_model.zip"; \
	 test -f "$$src" || { echo "ERROR: $$src not found — run 'make train' first"; exit 1; }; \
	 label=$$(echo "$$dir" | sed 's|$(RUNS_DIR)/||'); \
	 dest="$(MODELS_DIR)/$$(echo $$label | tr '/' '_')"; \
	 mkdir -p "$$dest"; \
	 cp "$$src"                   "$$dest/best_model.zip"; \
	 [ -f "$$dir/info.toml" ]            && cp "$$dir/info.toml"            "$$dest/run_info.toml"        || true; \
	 [ -f "$$dir/config_snapshot.toml" ] && cp "$$dir/config_snapshot.toml" "$$dest/config.toml" || true; \
	 echo ""; \
	 echo "  Trial:    $$dir"; \
	 echo "  Promoted  →  $$dest/"; \
	 echo ""; \
	 echo "  To reproduce this config:"; \
	 echo "    make repro MODEL=$$(echo $$label | tr '/' '_')"; \
	 echo ""; \
	 echo "  To commit:"; \
	 echo "    git add $$dest && git commit -m 'model: promote $$label best model'"

repro: ## 🔄 Restore config/training.toml from a promoted model  [MODEL=...]
	@test -n "$(MODEL)" || { echo "ERROR: specify MODEL=<name>  (see 'make list-runs')"; exit 1; }; \
	 src="$(MODELS_DIR)/$(MODEL)/config.toml"; \
	 test -f "$$src" || { echo "ERROR: $$src not found — model promoted before config snapshots were added?"; exit 1; }; \
	 cp "$$src" config/training.toml; \
	 echo "Restored config/training.toml from $$src"

# ──────────────────────────────────────────────────────────────────────────────

install: ## 📦 Create or update the $(CONDA_ENV) conda env from environment.yml
	@$(CONDA) env create -f environment.yml 2>/dev/null || $(CONDA) env update -f environment.yml --prune
	@if [ ! -f config/training.toml ]; then \
	   cp templates/training.toml config/training.toml; \
	   echo "Created config/training.toml from templates/training.toml."; \
	 else \
	   echo "config/training.toml already exists — not overwritten."; \
	 fi
	@echo "Done. Verify with: make check"

list-runs: ## 🗂️  List training runs grouped by config name
	@echo "=== $(RUNS_DIR)/ ==="; \
	 configs=$$(ls -1 $(RUNS_DIR) 2>/dev/null); \
	 if [ -z "$$configs" ]; then echo "  (none)"; \
	 else for cfg in $$configs; do \
	   echo "  $$cfg/"; \
	   ls -1t "$(RUNS_DIR)/$$cfg" 2>/dev/null | sed 's/^/    /'; \
	 done; fi; \
	 echo ""; \
	 echo "=== $(MODELS_DIR)/ ==="; \
	 mdirs=$$(ls -1d $(MODELS_DIR)/*/ 2>/dev/null); \
	 if [ -n "$$mdirs" ]; then \
	   for d in $$mdirs; do \
	     echo "  $$d"; \
	     ls -1 "$$d" 2>/dev/null | sed 's/^/      /'; \
	   done; \
	 else echo "  (none — run 'make promote' after a successful training run)"; fi

clean: ## 🧹 Remove __pycache__ and .pyc files
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean."

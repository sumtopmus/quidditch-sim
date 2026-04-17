# Quidditch-Sim Makefile
# Usage:  make <target>  [RUN_NAME=ppo_hoop]  [TRIAL=20240416_143022]
#
# Run layout:  runs/<RUN_NAME>/<trial>/   (trial = auto-timestamp per training run)

CONDA_ENV  ?= uav
RUN_NAME   ?= ppo_hoop
TRIAL      ?=
RUNS_DIR   := runs
MODELS_DIR := models

# Resolve the trial directory for eval / promote:
#   TRIAL= given on CLI          → runs/$(RUN_NAME)/$(TRIAL)
#   RUN_NAME= given on CLI only  → latest trial inside that run
#   nothing given                → latest trial across all runs
_LATEST_IN_RUN  = $(shell ls -d $(RUNS_DIR)/$(RUN_NAME)/* 2>/dev/null | sort | tail -1)
_LATEST_OVERALL = $(shell ls -d $(RUNS_DIR)/*/* 2>/dev/null | sort | tail -1)
_TRIAL_DIR      = $(strip $(if $(TRIAL),\
                    $(RUNS_DIR)/$(RUN_NAME)/$(TRIAL),\
                    $(if $(filter command line,$(origin RUN_NAME)),\
                      $(_LATEST_IN_RUN),\
                      $(_LATEST_OVERALL))))

# Run a command inside the conda env, streaming output in real time.
CONDA_RUN := conda run --no-capture-output -n $(CONDA_ENV)
PYTHON    := $(CONDA_RUN) python

# ──────────────────────────────────────────────────────────────────────────────
.PHONY: help check check-gui train eval eval-headless tensorboard promote repro install clean list-runs

.DEFAULT_GOAL := help

help: ## 📋 Show available targets
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Override variables:  make train RUN_NAME=my_run   CONDA_ENV=uav"

# ──────────────────────────────────────────────────────────────────────────────

check: ## ✅ Validate env headless (fast, no window)
	@$(PYTHON) check_env.py

check-gui: ## 🪟 Validate env with PyBullet GUI (interactive camera)
	@$(PYTHON) check_env.py --gui

train: ## 🚀 Run PPO training  [RUN_NAME=ppo_hoop]
	@$(PYTHON) train_ppo.py --run-name $(RUN_NAME)

eval: ## 🎯 Evaluate best model visually  [RUN_NAME=...] [TRIAL=...] [EPISODES=10]
	@$(PYTHON) eval_ppo.py --model $(_TRIAL_DIR)/best_model --episodes $(or $(EPISODES),10)

eval-headless: ## 📈 Evaluate best model headless  [RUN_NAME=...] [TRIAL=...] [EPISODES=50]
	@$(PYTHON) eval_ppo.py --model $(_TRIAL_DIR)/best_model --no-render --episodes $(or $(EPISODES),50)

tensorboard: ## 📊 Launch TensorBoard — all runs, or [RUN_NAME=...] for one config
	@$(CONDA_RUN) tensorboard --logdir $(if $(filter command line,$(origin RUN_NAME)),$(RUNS_DIR)/$(RUN_NAME),$(RUNS_DIR))

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
	@conda env create -f environment.yml 2>/dev/null || conda env update -f environment.yml --prune
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

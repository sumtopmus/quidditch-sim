# Quidditch-Sim Makefile
# Usage:  make <target>  [RUN_NAME=ppo_hoop_v1]

CONDA_ENV  ?= uav
RUN_NAME   ?= ppo_hoop_v1
RUNS_DIR   := runs
MODELS_DIR := models

# Use RUN_NAME if passed on the command line, otherwise pick the
# most-recently-modified subdirectory of RUNS_DIR automatically.
_LATEST_RUN = $(if $(filter command line,$(origin RUN_NAME)),$(RUN_NAME),$(shell ls -t $(RUNS_DIR) 2>/dev/null | head -1))

# Run a command inside the conda env, streaming output in real time.
CONDA_RUN := conda run --no-capture-output -n $(CONDA_ENV)
PYTHON    := $(CONDA_RUN) python

# ──────────────────────────────────────────────────────────────────────────────
.PHONY: help check check-gui train eval eval-headless tensorboard promote install clean list-runs

.DEFAULT_GOAL := help

help: ## 📋 Show available targets
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Override variables:  make train RUN_NAME=my_run   CONDA_ENV=uav"

# ──────────────────────────────────────────────────────────────────────────────

check: ## ✅ Validate env headless (fast, no window)
	$(PYTHON) check_env.py

check-gui: ## 🪟 Validate env with PyBullet GUI (interactive camera)
	$(PYTHON) check_env.py --gui

train: ## 🚀 Run PPO training  [RUN_NAME=ppo_hoop_v1]
	$(PYTHON) train_ppo.py --run-name $(RUN_NAME)

eval: ## 🎯 Evaluate best model visually  [RUN_NAME=latest] [EPISODES=10]
	$(PYTHON) eval_ppo.py --model $(RUNS_DIR)/$(_LATEST_RUN)/best_model --episodes $(or $(EPISODES),10)

eval-headless: ## 📈 Evaluate best model headless (stats only)  [RUN_NAME=latest] [EPISODES=50]
	$(PYTHON) eval_ppo.py --model $(RUNS_DIR)/$(_LATEST_RUN)/best_model --no-render --episodes $(or $(EPISODES),50)

tensorboard: ## 📊 Launch TensorBoard for RUN_NAME  [RUN_NAME=ppo_hoop_v1]
	$(CONDA_RUN) tensorboard --logdir $(RUNS_DIR)/$(RUN_NAME)/tb

# ──────────────────────────────────────────────────────────────────────────────

promote: ## 🏆 Promote best model to models/ (latest run or RUN_NAME=...)
	@run="$(_LATEST_RUN)"; \
	 [ -n "$$run" ] || { echo "ERROR: no runs found in $(RUNS_DIR)/"; exit 1; }; \
	 src="$(RUNS_DIR)/$$run/best_model.zip"; \
	 test -f "$$src" || { echo "ERROR: $$src not found — run 'make train' first"; exit 1; }; \
	 mkdir -p $(MODELS_DIR); \
	 dest="$(MODELS_DIR)/$${run}_best.zip"; \
	 cp "$$src" "$$dest"; \
	 echo ""; \
	 echo "  Run:      $$run"; \
	 echo "  Promoted  $$src  →  $$dest"; \
	 echo ""; \
	 echo "  To commit:"; \
	 echo "    git add $$dest && git commit -m 'model: promote $$run best model'"

# ──────────────────────────────────────────────────────────────────────────────

install: ## 📦 Create or update the $(CONDA_ENV) conda env from environment.yml
	conda env create -f environment.yml 2>/dev/null || conda env update -f environment.yml --prune
	@echo "Done. Verify with: make check"

list-runs: ## 🗂️  List available training runs and their saved models
	@echo "=== $(RUNS_DIR)/ ==="; \
	 runs=$$(ls -1 $(RUNS_DIR) 2>/dev/null); \
	 if [ -n "$$runs" ]; then echo "$$runs"; else echo "  (none)"; fi; \
	 echo ""; \
	 echo "=== $(MODELS_DIR)/ ==="; \
	 zips=$$(find $(MODELS_DIR) -maxdepth 1 -name "*.zip" 2>/dev/null); \
	 if [ -n "$$zips" ]; then echo "$$zips" | xargs ls -lh; \
	 else echo "  (none — run 'make promote' after a successful training run)"; fi

clean: ## 🧹 Remove __pycache__ and .pyc files
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean."

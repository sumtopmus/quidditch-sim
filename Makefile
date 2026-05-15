# Quidditch-Sim Makefile
# Usage:
#   make train EXP=blue_v5                                          # ladder rung
#   make train EXP=canary_team OVERRIDES="trainer.total_timesteps=500"
#   make resume RUN_NAME=ppo_hoop_blue_5 EXP=blue_v5
#
# Hydra owns run-dir layout: runs/<run_name>/<YYYYMMDD_HHMMSS>/
#   ├── .hydra/{config,overrides,hydra,meta}.yaml
#   ├── best_model.zip
#   └── checkpoints/

CONDA_ENV  ?= uav
RUN_NAME   ?=
TRIAL      ?=
CHECKPOINT ?=
EXP        ?=
OVERRIDES  ?=
RUNS_DIR   := runs
MODELS_DIR := models

# Resolve the trial directory for eval / promote / lineage:
#   TRIAL= given on CLI          → runs/$(RUN_NAME)/$(TRIAL)
#   RUN_NAME= given on CLI only  → latest trial inside that run
#   nothing given                → latest trial across all runs
_LATEST_IN_RUN  = $(shell find $(RUNS_DIR)/$(RUN_NAME) -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -t/ -k3 | tail -1)
_LATEST_OVERALL = $(shell find $(RUNS_DIR) -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort -t/ -k3 | tail -1)
_TRIAL_DIR      = $(strip $(if $(TRIAL),\
                    $(RUNS_DIR)/$(RUN_NAME)/$(TRIAL),\
                    $(if $(filter command line,$(origin RUN_NAME)),\
                      $(_LATEST_IN_RUN),\
                      $(_LATEST_OVERALL))))
_LATEST_CKPT    = $(shell ls -1 "$(_TRIAL_DIR)/checkpoints/"*.zip 2>/dev/null | sort -V | tail -1 | sed 's/\.zip$$//')

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
.PHONY: help test test-fast test-warm camera-test demo train resume eval eval-headless lineage promote install clean list-runs eval-team sweep sweep-agent sweep-agents

.DEFAULT_GOAL := help

help: ## 📋 Show available targets
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make train EXP=blue_v5"
	@echo "  make train EXP=canary_team OVERRIDES=\"trainer.total_timesteps=500\""
	@echo "  make resume RUN_NAME=ppo_hoop_blue_5 EXP=blue_v5"

# ──────────────────────────────────────────────────────────────────────────────

test: ## ✅ Run all tests (unit + integration)
	@$(PYTHON) -m pytest

test-fast: ## ⚡ Unit tests only (skip slow integration canaries)
	@$(PYTHON) -m pytest tests/unit

test-warm: ## ✅ Warm-start preserves single-agent behavior  MODEL=<run-name>
	@test -n "$(MODEL)" || { echo "ERROR: MODEL=<run-name> required (see 'make list-runs')"; exit 1; }; \
	 MODEL="$(MODEL)" $(PYTHON) -m pytest tests/integration/test_warm_start.py

CAM ?= grid
camera-test: ## 🎥 Render hover flight as 2x2 grid → mp4 (CAM=grid|fixed|north|east|south|west|top|fpv|tpv|port|starboard)
	@$(PYTHON) demo/camera_test.py --cam $(CAM)

demo: ## 🎮 Pick a demo to run (hover, waypoint, scenarios) — opens viewer
	@$(MJPYTHON) demo/menu.py

# ──────────────────────────────────────────────────────────────────────────────

train: ## 🚀 Run PPO training  EXP=<experiment-name>  [OVERRIDES="key=val ..."]
	@test -n "$(EXP)" || { echo "ERROR: EXP=<experiment-name> required (see conf/experiment/)"; exit 1; }
	@$(PYTHON) -m scripts.train +experiment=$(EXP) $(OVERRIDES)

resume: ## ▶️  Resume from latest checkpoint  RUN_NAME=...  [EXP=...]  [OVERRIDES=...]
	@test -n "$(RUN_NAME)" || { echo "ERROR: RUN_NAME=<name> required"; exit 1; }
	@$(PYTHON) -m scripts.train \
	  $(if $(EXP),+experiment=$(EXP),) \
	  init=resume init.parent_run=$(RUN_NAME) \
	  run_name=$(RUN_NAME) $(OVERRIDES)

eval: ## 🎯 Evaluate best model visually  [RUN_NAME=...] [TRIAL=...] [EPISODES=10]
	@$(MJPYTHON) scripts/eval_ppo.py --model $(_TRIAL_DIR)/best_model --episodes $(or $(EPISODES),10)

eval-headless: ## 📈 Evaluate best model headless  [RUN_NAME=...] [TRIAL=...] [EPISODES=50]
	@$(PYTHON) scripts/eval_ppo.py --model $(_TRIAL_DIR)/best_model --no-render --episodes $(or $(EPISODES),50)

lineage: ## ⛓  Walk pretrain ancestry  [RUN_NAME=...] [TRIAL=...] [TARGET=<path-or-uri>] [LOCAL=1] [BOTH=1]
	@target="$(or $(TARGET),$(_TRIAL_DIR))"; \
	 test -n "$$target" || { echo "ERROR: pass TARGET=<path-or-uri> or RUN_NAME=..."; exit 1; }; \
	 $(PYTHON) -m scripts.lineage "$$target" \
	   $(if $(LOCAL),--local) $(if $(BOTH),--both)

# ──────────────────────────────────────────────────────────────────────────────

promote: ## 🏆 Promote best model — alias on wandb + copy to models/  [RUN_NAME=...] [TRIAL=...]
	@dir="$(_TRIAL_DIR)"; \
	 [ -n "$$dir" ] || { echo "ERROR: no trials found in $(RUNS_DIR)/"; exit 1; }; \
	 test -f "$$dir/best_model.zip" || { echo "ERROR: $$dir/best_model.zip not found — run 'make train EXP=...' first"; exit 1; }; \
	 $(PYTHON) -m scripts.promote "$$dir" --models-root "$(MODELS_DIR)"

# ──────────────────────────────────────────────────────────────────────────────

install: ## 📦 Create/update the $(CONDA_ENV) conda env
	@$(CONDA) env create -f environment.yml 2>/dev/null || $(CONDA) env update -f environment.yml --prune
	@echo "Done. Verify with: make test"

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

# ──────────────────────────────────────────────────────────────────────────────

eval-team: ## 🎯 Head-to-head eval  RED=<spec>  BLUE=<spec>  [EPISODES=N] [GUI=1] [DETERMINISTIC=1] [LEARNER=red_0|blue_0] [LEARNER_FRAME_STACK=N] [RANDOMISE_START=1]
	@test -n "$(RED)" -a -n "$(BLUE)" || { echo "ERROR: RED=<spec> BLUE=<spec> required"; exit 1; }; \
	 $(if $(GUI),$(MJPYTHON),$(PYTHON)) scripts/eval_team.py --red "$(RED)" --blue "$(BLUE)" \
	   $(if $(EPISODES),--episodes $(EPISODES)) \
	   $(if $(GUI),--gui) \
	   $(if $(DETERMINISTIC),--deterministic) \
	   $(if $(LEARNER),--learner $(LEARNER)) \
	   $(if $(LEARNER_FRAME_STACK),--learner-frame-stack $(LEARNER_FRAME_STACK)) \
	   $(if $(RANDOMISE_START),--randomise-start)

# ── Sweeps ───────────────────────────────────────────────────────────────────

WANDB_PROJECT ?= drone-quidditch
SWEEP ?=
ID    ?=
N     ?= 1

sweep: ## 🔁 Create a wandb sweep controller  SWEEP=<name> (file under sweeps/)
	@test -n "$(SWEEP)" || { echo "ERROR: SWEEP=<name> required (see sweeps/)"; exit 1; }
	@test -f "sweeps/$(SWEEP).yaml" || { echo "ERROR: sweeps/$(SWEEP).yaml not found"; exit 1; }
	$(CONDA_RUN) wandb sweep --project $(WANDB_PROJECT) sweeps/$(SWEEP).yaml

sweep-agent: ## 🤖 Run one sweep agent  ID=<sweep_id>
	@test -n "$(ID)" || { echo "ERROR: ID=<sweep_id> required (copy from 'make sweep' output)"; exit 1; }
	$(CONDA_RUN) wandb agent $(ID)

sweep-agents: ## 🤖🤖 Run N parallel sweep agents  ID=<sweep_id> N=<n>
	@test -n "$(ID)" || { echo "ERROR: ID=<sweep_id> required"; exit 1; }
	@echo "Spawning $(N) agents.  Single-machine: N=1 is the sane default for CPU laptop trainings."
	@for i in $$(seq 1 $(N)); do $(CONDA_RUN) wandb agent $(ID) & done; wait

clean: ## 🧹 Remove __pycache__ and .pyc files
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean."

# Quidditch-Sim Makefile — infrastructure & test targets only.
#
# Day-to-day workflow (train / eval / resume / promote / lineage / tensorboard /
# repro / list-runs / demos) lives in the TUI launcher: `make ui` or
# `python -m tui`. The TUI shells out to scripts/*.py; those scripts remain
# directly callable for scripting / CI.

CONDA_ENV ?= uav

# Resolve the conda binary: prefer $CONDA_EXE (set by `conda init`), fall back to PATH.
CONDA := $(or $(CONDA_EXE),$(shell command -v conda 2>/dev/null))
ifeq ($(CONDA),)
$(error conda not found — activate a conda shell or set CONDA_EXE)
endif

CONDA_RUN := $(CONDA) run --no-capture-output -n $(CONDA_ENV)
PYTHON    := $(CONDA_RUN) python

.PHONY: help ui demo test test-fast test-warm camera-test install configs clean

.DEFAULT_GOAL := help

help: ## 📋 Show available targets
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "All day-to-day actions (train / eval / promote / …) are in the TUI: 'make ui'."

# ──────────────────────────────────────────────────────────────────────────────
# Interactive launcher

ui: ## 🎛  Interactive launcher (TUI dashboard)
	@$(PYTHON) -m tui

demo: ## 🎮 Open TUI on the Demo group
	@$(PYTHON) -m tui --group Demo

# ──────────────────────────────────────────────────────────────────────────────
# Tests

test: ## ✅ Run all tests (unit + integration)
	@$(PYTHON) -m pytest

test-fast: ## ⚡ Unit tests only (skip slow integration canaries)
	@$(PYTHON) -m pytest tests/unit

test-warm: ## ✅ Warm-start preserves single-agent behavior  MODEL=<run-name>
	@test -n "$(MODEL)" || { echo "ERROR: MODEL=<run-name> required (see TUI list-runs)"; exit 1; }; \
	 MODEL="$(MODEL)" $(PYTHON) -m pytest tests/integration/test_warm_start.py

CAM ?= grid
camera-test: ## 🎥 Render hover flight as 2x2 grid → mp4 (CAM=grid|fixed|north|east|south|west|top|fpv|tpv|port|starboard)
	@$(PYTHON) demo/camera_test.py --cam $(CAM)

# ──────────────────────────────────────────────────────────────────────────────
# Setup

install: ## 📦 Create/update the $(CONDA_ENV) conda env + populate config/ from templates
	@$(CONDA) env create -f environment.yml 2>/dev/null || $(CONDA) env update -f environment.yml --prune
	@$(MAKE) --no-print-directory configs
	@echo "Done. Verify with: make test"

configs: ## 🛠  Populate config/ from templates/ (idempotent — never overwrites)
	@mkdir -p config
	@for f in training camera; do \
	   if [ ! -f config/$$f.toml ]; then \
	     cp templates/$$f.toml config/$$f.toml; \
	     echo "config/$$f.toml << templates/$$f.toml."; \
	   else \
	     echo "config/$$f.toml already exists — not overwritten"; \
	   fi; \
	 done

clean: ## 🧹 Remove __pycache__ and .pyc files
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean."

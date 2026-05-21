# Quidditch-Sim Makefile — Slice 1 (Part 3 ML infra) reshape.
#
# Day-to-day commands moved to:
#   dsim --help                            inspection + dispatch (Typer CLI)
#   python -m scripts.train +experiment=X  Hydra entrypoint (composable)
#   python -m scripts.eval_team  …         Hydra entrypoint
#   python -m scripts.eval_battery …       Hydra entrypoint
#   python -m scripts.eval_solo  …         Hydra entrypoint
#   make tui                               opens the controller TUI (Slice 2)

CONDA_ENV  ?= uav
EXP        ?=
OVERRIDES  ?=

CONDA := $(or $(CONDA_EXE),$(shell command -v conda 2>/dev/null))
ifeq ($(CONDA),)
$(error conda not found — activate a conda shell or set CONDA_EXE)
endif

CONDA_RUN := $(CONDA) run --no-capture-output -n $(CONDA_ENV)
PYTHON    := $(CONDA_RUN) python
MJPYTHON  := $(CONDA_RUN) mjpython

.PHONY: help install clean test test-fast test-warm tui train

.DEFAULT_GOAL := help

help: ## 📋 Show targets + pointers to dsim and Hydra entrypoints
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Other day-to-day commands (no make wrapper):"
	@echo "  dsim --help                              List dsim subcommands"
	@echo "  python -m scripts.eval_team +eval_team=default +learner=blue \\"
	@echo "      learner.uri=<…> opponent=beeline_red"
	@echo "  python -m scripts.eval_battery +eval_battery=default \\"
	@echo "      eval_battery.candidate=<uri>"
	@echo ""

install: ## ⚙️  Create conda env + install dsim editable
	conda env create -f environment.yml || conda env update -f environment.yml
	$(CONDA_RUN) pip install -e .

clean: ## 🧹 Remove build artifacts and __pycache__
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info

test: ## ✅ Full test suite (unit + integration)
	@$(PYTHON) -m pytest

test-fast: ## ⚡ Unit tests only (skip @pytest.mark.slow)
	@$(PYTHON) -m pytest -m "not slow"

test-warm: ## ✅ Warm-start integration test  MODEL=<run-name>
	@test -n "$(MODEL)" || { echo "ERROR: MODEL=<run-name> required (see 'dsim inventory')"; exit 1; }; \
	 MODEL="$(MODEL)" $(PYTHON) -m pytest tests/core/policies/test_warm_start.py

tui: ## 🖼  Open the controller TUI (Slice 2; subprocess slots use mjpython when needed)
	@$(PYTHON) -m dsim tui

train: ## 🚀 Launch a training run  EXP=<name> [OVERRIDES="key=val key=val"]
	@test -n "$(EXP)" || { echo "ERROR: EXP=<experiment-name> required (ls conf/experiment/)"; exit 1; }
	@$(PYTHON) -m scripts.train +experiment=$(EXP) $(OVERRIDES)

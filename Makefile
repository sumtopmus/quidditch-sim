# Quidditch-Sim Makefile — Slice 1 (Part 3 ML infra) reshape.
#
# Day-to-day commands moved to:
#   dsim --help                            inspection + dispatch (Typer CLI)
#   python -m scripts.train +experiment=X  Hydra entrypoint (composable)
#   python -m scripts.eval_team  …         Hydra entrypoint
#   python -m scripts.eval_battery …       Hydra entrypoint
#   python -m scripts.eval_solo  …         Hydra entrypoint
#   make tui                               opens the controller TUI (Slice 2)

EXP        ?=
OVERRIDES  ?=

# Resolve uv: prefer PATH, else error out.
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
$(error uv not found — install from https://docs.astral.sh/uv/)
endif

# Run a command inside the uv-managed venv, streaming output in real time.
UV_RUN   := $(UV) run
PYTHON   := $(UV_RUN) python
# macOS: mujoco.viewer.launch_passive() requires mjpython (owns the Cocoa main
# thread). mjpython is a console script installed by the mujoco pip wheel; use
# it only for targets that open the interactive viewer.
MJPYTHON := $(UV_RUN) mjpython

.PHONY: help install clean test test-fast tui train

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

install: ## ⚙️  Sync the uv-managed venv from pyproject.toml + uv.lock
	@$(UV) sync
	@echo "Done. Verify with: make test"

clean: ## 🧹 Remove build artifacts and __pycache__
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info

test: ## ✅ Full test suite (unit + integration)
	@$(PYTHON) -m pytest

test-fast: ## ⚡ Unit tests only (skip @pytest.mark.slow)
	@$(PYTHON) -m pytest -m "not slow"

tui: ## 🖼  Open the controller TUI (Slice 2; subprocess slots use mjpython when needed)
	@$(PYTHON) -m dsim tui

train: ## 🚀 Launch a training run  EXP=<name> [OVERRIDES="key=val key=val"]
	@test -n "$(EXP)" || { echo "ERROR: EXP=<experiment-name> required (ls conf/experiment/)"; exit 1; }
	@$(PYTHON) -m scripts.train +experiment=$(EXP) $(OVERRIDES)

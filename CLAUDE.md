# CLAUDE.md — orientation for AI agents

## CLI surface (ML infra Part 3, merged 2026-06)

Three surfaces by purpose:
- `dsim <cmd>` — Typer CLI for inspection (inventory, lineage, list-runs,
  obs-preflight, obs-specs, describe-run) and dispatch (resume, promote,
  sweep). Single binary; `dsim --help` lists all.
- `python -m scripts.<name>` — Hydra apps for composable runs:
  `train`, `eval_team`, `eval_solo`, `eval_battery`.
- `make <target>` — chores only: `install`, `clean`, `test*`, `tui`, `train`.

See `README.md` "CLI surface" section for the migration map.

## Conventions

- Toolchain: **uv** (migrated off conda).  Run things via `uv run <cmd>`, or
  the `make` / `dsim` wrappers (both call `uv run`).  `make install` = `uv sync`.
  The interactive MuJoCo viewer needs `uv run mjpython` (Cocoa main thread).
- Config: Hydra `conf/` group tree (no TOML).  Experiments are YAMLs under
  `conf/experiment/`; `make train EXP=<name>` is the daily entrypoint.
- Experiment tracking: **W&B** (TensorBoard retired).  Online by default;
  `WANDB_MODE=disabled` in `tests/conftest.py` keeps the suite offline.
- Commit-type taxonomy: `feat:`, `refactor:`, `fix:`, `test:`, `docs:`,
  `build:`, `chore:`.
- Test layout mirrors source: `tests/<package>/<module>/test_<name>.py`.
- Slow / integration tests are gated by `@pytest.mark.slow`; `make
  test-fast` skips them.
- `WANDB_MODE=disabled` is set in `tests/conftest.py` so the suite stays
  offline.

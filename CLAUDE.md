# CLAUDE.md — orientation for AI agents

## CLI surface (Slice 1 of ML infra Part 3, landed 2026-05-19)

Three surfaces by purpose:
- `dsim <cmd>` — Typer CLI for inspection (inventory, lineage, list-runs,
  obs-preflight, obs-specs, describe-run) and dispatch (resume, promote,
  sweep). Single binary; `dsim --help` lists all.
- `python -m scripts.<name>` — Hydra apps for composable runs:
  `train`, `eval_team`, `eval_solo`, `eval_battery`.
- `make <target>` — chores only: `install`, `clean`, `test*`, `tui`, `train`.

See `README.md` "CLI surface" section for the migration map.

## Conventions

- Conda env: `uav`.  Activate before running anything (`conda activate uav`).
- Commit-type taxonomy: `feat:`, `refactor:`, `fix:`, `test:`, `docs:`,
  `build:`, `chore:`.
- Test layout mirrors source: `tests/<package>/<module>/test_<name>.py`.
- Slow / integration tests are gated by `@pytest.mark.slow`; `make
  test-fast` skips them.
- `WANDB_MODE=disabled` is set in `tests/conftest.py` so the suite stays
  offline.

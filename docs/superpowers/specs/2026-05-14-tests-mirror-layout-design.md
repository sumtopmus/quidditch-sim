# Tests Mirror Layout — Design

**Date:** 2026-05-14
**Branch:** `feature/tests-mirror-layout`
**Scope:** Pure refactor of `tests/` directory layout. No test logic changes, no fixture changes, no coverage additions.

## Motivation

`tests/` is currently split horizontally by test type (`unit/` vs `integration/`) with all 28 test files in two flat directories. The source tree is nested (`core/{mjcf,drone,policies}/`, `envs/quidditch/rewards/`, `scripts/`, `demo/`, top-level `config_schema.py`). Mirroring the source tree under `tests/` and replacing the `unit/integration/` dirs with the existing `slow` pytest marker brings the layout in line with mainstream pytest conventions — predictable navigation ("where's the test for `envs/quidditch/scene.py`?" answerable from the path alone) without overhauling the test infrastructure that already works.

The `slow` marker already exists in `pyproject.toml` and is applied to all 7 integration tests via `pytestmark = pytest.mark.slow`. The marker-based unit-vs-slow distinction is already half-built; this refactor finishes it and removes the redundant directory split.

## Goal (and non-goal)

**In scope.** Move test files into a mirrored layout. Delete `tests/unit/` and `tests/integration/` shells. Update two Makefile target paths. Delete redundant `__init__.py` files. Update the `slow` marker description in `pyproject.toml` to formalize its role as the unit-vs-integration axis.

**Out of scope.** No test logic edits. No fixture refactors. No new tests. No coverage tooling. No marker renames. No `tests/` → `test/` rename. No changes to `conftest.py` contents. No work on the `feature/tui-launcher` worktree (left to resolve later).

## Success criteria

1. `make test` produces the same test count and pass/fail signature as before the refactor.
2. `make test-fast` runs the unit subset only (formerly path-based, now marker-based).
3. `make test-warm MODEL=...` still works against the relocated `test_warm_start.py`.
4. Canary fingerprints byte-identical: `step 434 / reward 7.3837` (single-agent scoring canary), team canary unchanged.
5. Every test file's path encodes which source module it primarily exercises (with documented exceptions for top-level cross-cutting tests).

## Target layout

```
tests/
├── conftest.py              # unchanged — hydra_compose + build_team_world + helpers
├── test_imports.py          # unchanged — top-level discovery smoke at tests/ root
├── test_config_schema.py    # from unit/ — tests top-level config_schema.py
├── test_config_loading.py   # from unit/ — tests Hydra composition
├── test_meta_yaml.py        # from unit/ — tests .hydra/meta.yaml writer
├── test_migrate_legacy_models.py   # from unit/ — tests scripts/migrate_legacy_models.py
├── core/
│   ├── test_disable_motors.py      # from unit/ — exercises core/quadrotor.py
│   ├── test_render_smoke.py        # from unit/ — exercises core/world.py render path
│   └── policies/
│       ├── test_warm_start_by_spec.py  # from unit/ — exercises core/policies/warm_start.py
│       └── test_warm_start.py          # from integration/ (@slow) — same source, integration angle
├── envs/quidditch/
│   ├── test_crash_detector.py       # from unit/
│   ├── test_tag_state_machine.py    # from unit/
│   ├── test_team_mjcf.py            # from unit/ — primarily exercises envs/quidditch/scene.py
│   ├── test_obs_spec.py             # from unit/
│   ├── test_obs_compat.py           # from unit/
│   ├── test_augmented_obs.py        # from unit/
│   ├── test_env_factories.py        # from unit/
│   ├── test_opponent_env_world.py   # from unit/
│   ├── test_simple_env_contract.py  # from integration/ (@slow)
│   ├── test_scoring_canary.py       # from integration/ (@slow)
│   ├── test_team_env_canary.py      # from integration/ (@slow)
│   ├── test_crash_aftermath.py      # from integration/ (@slow)
│   ├── test_take_down.py            # from integration/ (@slow)
│   └── rewards/
│       └── test_reward_stack.py     # from unit/
└── scripts/
    ├── test_train_common_video.py     # from unit/ — exercises scripts/_train_common.py
    ├── test_video_callback_team.py    # from unit/ — exercises scripts/callbacks.py
    └── test_scripted_demos.py         # from integration/ (@slow) — exercises demo/ scripts
```

### Placement decisions worth recording

- **Test file names are preserved.** A test file's name describes what it tests (a behavior or feature), not its source filename. Renaming `test_disable_motors.py` → `test_quadrotor.py` would obscure the test's actual coverage, and the project already has legitimate cases of multiple test files per source file (`core/policies/warm_start.py` has both `test_warm_start.py` and `test_warm_start_by_spec.py`, deliberately splitting integration vs spec-level coverage).
- **Test files live where their primary subject lives.** `test_team_mjcf.py` builds a team scene via `envs/quidditch/scene.py` (which internally uses `core/mjcf/`) — it lives under `tests/envs/quidditch/` because the scene module is the test's primary subject. `test_render_smoke.py` exercises `core/world.py`'s render path — it lives under `tests/core/`.
- **The four config/migrate tests stay at `tests/` root.** They test the top-level `config_schema.py`, the `conf/` Hydra tree, the `.hydra/meta.yaml` writer (a cross-cutting Hydra convention, not owned by one script), and `scripts/migrate_legacy_models.py` (a top-level migration tool, not a runtime script). A nested `tests/conf/` is rejected because `conf/` is not a Python package and would imply otherwise; nesting these would be a directory created for one or two files.
- **`test_scripted_demos.py` lives at `tests/scripts/test_scripted_demos.py`, not `tests/demo/`.** It's a single file that exercises the scripted-demo runner machinery; creating a one-file `tests/demo/` directory pays nesting cost without payoff. If a second demo-related test arrives, splitting out `tests/demo/` is a one-line follow-up.
- **`test_imports.py` keeps its tests/ root placement.** Its docstring already explains: it exercises pytest discovery itself, so it lives outside any subdir.

## Marker convention

Keep the existing `slow` marker. All 7 integration tests already declare `pytestmark = pytest.mark.slow` (or include it in a marker list); no test-file edits are needed for marker semantics.

`pyproject.toml` change: tighten the marker description to make explicit that `slow` is the canonical unit-vs-integration axis:

```diff
 markers = [
-    "slow: integration tests with real episode loops (full canaries, warm-start)",
+    "slow: integration tests — real episode loops, full canaries, warm-start. `make test-fast` skips these via -m 'not slow'.",
 ]
```

Rejected alternative: renaming `slow` → `integration`. Both names are widely understood; `slow` matches existing usage across 7 files and avoids churn. The semantics, not the spelling, matter.

## `__init__.py` policy

**Keep `tests/__init__.py`.** Eight test files import helpers via `from tests.conftest import build_team_world, set_body_state, hydra_compose, ...`. That import path requires `tests` to be a Python package, i.e. `tests/__init__.py` must exist. Deleting it would break those imports and turn this into a code-edit refactor rather than a pure layout move.

**Delete `tests/unit/__init__.py` and `tests/integration/__init__.py`** — they vanish with their parent directories.

**Do not create `__init__.py` files in the new mirrored subdirs** (`tests/core/`, `tests/envs/`, `tests/envs/quidditch/`, `tests/envs/quidditch/rewards/`, `tests/core/policies/`, `tests/scripts/`). Pytest's rootdir-relative discovery handles unique `test_*.py` basenames without them. After the mirror, all 28 test filenames remain unique.

Net effect: one `__init__.py` survives (`tests/__init__.py`), two are deleted with their dirs, none are added.

## Makefile updates

```diff
 test: ## ✅ Run all tests (unit + integration)
        @$(PYTHON) -m pytest

-test-fast: ## ⚡ Unit tests only (skip slow integration canaries)
-       @$(PYTHON) -m pytest tests/unit
+test-fast: ## ⚡ Unit tests only (skip slow integration canaries)
+       @$(PYTHON) -m pytest -m "not slow"

 test-warm: ## ✅ Warm-start preserves single-agent behavior  MODEL=<run-name>
        @test -n "$(MODEL)" || { echo "ERROR: MODEL=<run-name> required (see 'make list-runs')"; exit 1; }; \
-        MODEL="$(MODEL)" $(PYTHON) -m pytest tests/integration/test_warm_start.py
+        MODEL="$(MODEL)" $(PYTHON) -m pytest tests/core/policies/test_warm_start.py
```

`make test` is unchanged (pytest reads `testpaths = ["tests"]` from `pyproject.toml`).

## Risk profile

This is the lowest-risk refactor on the active task list. There are no behavioral changes, only file moves and three line edits in `Makefile` + `pyproject.toml`. The two ways it could go wrong:

1. **`tests/integration/test_warm_start.py` is currently broken** (per `brain/index.md` Known Issues) — gated on `MODEL=` so the default suite passes. The refactor preserves this state; the Makefile target updates to point at the new path, and the test remains broken in the same way. Not a regression introduced here.
2. **`__file__` references.** Already audited: the only `__file__` use in `tests/` is at `tests/conftest.py:87` (`Path(__file__).parent.parent / "conf"` to resolve the Hydra config dir). `conftest.py` does not move, so this path stays valid. No other test file uses `__file__`.

## Verification

After the refactor:

1. Diff the file list: `git diff develop --stat -- tests/` shows only renames (no content changes).
2. `make test` count matches develop's count exactly (114 tests at time of writing, but pin the actual count from a clean `make test` run on develop before/after).
3. Pass/fail set is identical: `pytest --collect-only -q` output diffed before/after produces only path changes.
4. Canary fingerprints pinned in test source remain assertable: `step 434 / reward 7.3837` (single-agent), team canary unchanged.
5. `make test-fast` runs only non-`@slow` tests; `make test-warm MODEL=<any-valid-run>` invokes `tests/core/policies/test_warm_start.py`.
6. `from tests.conftest import ...` imports (8 files) still resolve, confirming `tests/__init__.py` was preserved.

## Out of scope follow-ups

- `feature/tui-launcher` rebase considerations — that worktree touches `tests/` lightly; integration with this refactor handled when/if that branch resumes.
- Renaming the `slow` marker to `integration` — punted; cosmetic.
- Adding a `tests/demo/` directory if a second demo test is added.
- Coverage tooling, watch mode, or any test-infrastructure additions.

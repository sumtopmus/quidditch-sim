# Tests Mirror Layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `tests/` so the directory layout mirrors the source tree, replacing the `tests/unit/` vs `tests/integration/` split with the already-existing `slow` pytest marker.

**Architecture:** Pure file-move refactor. Twenty-six test files relocate from `tests/{unit,integration}/test_*.py` to `tests/<source-mirror>/test_*.py` via `git mv` (preserving rename history). Two `__init__.py` files (`tests/unit/__init__.py`, `tests/integration/__init__.py`) are deleted alongside their now-empty parent dirs. `tests/__init__.py` is preserved because eight tests import helpers via `from tests.conftest import ...`. Two Makefile target paths and one `pyproject.toml` marker description are edited. No test code, fixture code, or `conftest.py` content changes.

**Tech Stack:** Python, pytest, GNU Make. Repo lives in a git worktree at `worktrees/feature/tests-mirror-layout/` on branch `feature/tests-mirror-layout` (off `develop`).

**Spec:** [`docs/superpowers/specs/2026-05-14-tests-mirror-layout-design.md`](../specs/2026-05-14-tests-mirror-layout-design.md) (committed as `88a872f`).

**Working directory for ALL tasks:** `worktrees/feature/tests-mirror-layout/`. The umbrella path is `/Users/shurioque/Library/Mobile Documents/com~apple~CloudDocs/Projects/drone-sim/`. All commands assume `pwd` is the worktree root.

---

## File Inventory (locked at plan-write time)

**Source paths to relocate (26 files):**

From `tests/unit/`:
- `test_augmented_obs.py` → `tests/envs/quidditch/test_augmented_obs.py`
- `test_config_loading.py` → `tests/test_config_loading.py`
- `test_config_schema.py` → `tests/test_config_schema.py`
- `test_crash_detector.py` → `tests/envs/quidditch/test_crash_detector.py`
- `test_disable_motors.py` → `tests/core/test_disable_motors.py`
- `test_env_factories.py` → `tests/envs/quidditch/test_env_factories.py`
- `test_meta_yaml.py` → `tests/test_meta_yaml.py`
- `test_migrate_legacy_models.py` → `tests/test_migrate_legacy_models.py`
- `test_obs_compat.py` → `tests/envs/quidditch/test_obs_compat.py`
- `test_obs_spec.py` → `tests/envs/quidditch/test_obs_spec.py`
- `test_opponent_env_world.py` → `tests/envs/quidditch/test_opponent_env_world.py`
- `test_render_smoke.py` → `tests/core/test_render_smoke.py`
- `test_reward_stack.py` → `tests/envs/quidditch/rewards/test_reward_stack.py`
- `test_tag_state_machine.py` → `tests/envs/quidditch/test_tag_state_machine.py`
- `test_team_mjcf.py` → `tests/envs/quidditch/test_team_mjcf.py`
- `test_train_common_video.py` → `tests/scripts/test_train_common_video.py`
- `test_video_callback_team.py` → `tests/scripts/test_video_callback_team.py`
- `test_warm_start_by_spec.py` → `tests/core/policies/test_warm_start_by_spec.py`

From `tests/integration/`:
- `test_crash_aftermath.py` → `tests/envs/quidditch/test_crash_aftermath.py`
- `test_scoring_canary.py` → `tests/envs/quidditch/test_scoring_canary.py`
- `test_scripted_demos.py` → `tests/scripts/test_scripted_demos.py`
- `test_simple_env_contract.py` → `tests/envs/quidditch/test_simple_env_contract.py`
- `test_take_down.py` → `tests/envs/quidditch/test_take_down.py`
- `test_team_env_canary.py` → `tests/envs/quidditch/test_team_env_canary.py`
- `test_warm_start.py` → `tests/core/policies/test_warm_start.py`

**Paths to delete:**
- `tests/unit/__init__.py` (`git rm`)
- `tests/integration/__init__.py` (`git rm`)
- `tests/unit/` (empty after moves — `rmdir`)
- `tests/integration/` (empty after moves — `rmdir`)

**Paths to create (directories only — pytest needs no `__init__.py` here):**
- `tests/core/`
- `tests/core/policies/`
- `tests/envs/`
- `tests/envs/quidditch/`
- `tests/envs/quidditch/rewards/`
- `tests/scripts/`

**Paths to keep unchanged:**
- `tests/__init__.py` (load-bearing — `from tests.conftest import ...` in 8 files)
- `tests/conftest.py` (helpers — referenced by `__file__` to resolve `conf/`; staying put preserves the reference)
- `tests/test_imports.py` (top-level discovery smoke)

**Files to modify (3 line edits across 2 files):**
- `Makefile` — `test-fast` and `test-warm` target bodies
- `pyproject.toml` — `slow` marker description string

---

## Task 1: Capture pre-refactor baseline

This task establishes the comparison baseline for "byte-identical pass/fail signature." No commit — these are local artifacts the implementing engineer references during Task 5 verification.

**Files:**
- Create: `$TMPDIR/tests-mirror-baseline-collect.txt` (local scratch — not committed)
- Create: `$TMPDIR/tests-mirror-baseline-count.txt` (local scratch — not committed)

- [ ] **Step 1: Verify clean worktree state**

```bash
git status --short
git branch --show-current
```

Expected output:
```
(empty — no uncommitted changes)
feature/tests-mirror-layout
```

If output is not empty, stop and resolve before proceeding.

- [ ] **Step 2: Capture pytest collection output for diff comparison**

```bash
python -m pytest --collect-only -q > "$TMPDIR/tests-mirror-baseline-collect.txt" 2>&1
echo "Exit: $?"
head -5 "$TMPDIR/tests-mirror-baseline-collect.txt"
tail -5 "$TMPDIR/tests-mirror-baseline-collect.txt"
wc -l "$TMPDIR/tests-mirror-baseline-collect.txt"
```

Expected: Exit code 0. Last line of the file reports the test count (e.g. `114 tests collected in 0.42s`). Save the count for Task 5.

- [ ] **Step 3: Capture full test run baseline**

```bash
python -m pytest 2>&1 | tee "$TMPDIR/tests-mirror-baseline-count.txt" | tail -3
```

Expected: A summary line like `===== 114 passed in N.NNs =====` (or with `skipped` entries — the `test_warm_start.py` integration test is `@pytest.mark.skip`-gated on `MODEL=` per spec §Risk profile, so 1 skip is normal). Pin the exact `<N> passed[, <M> skipped]` line in your notes — Task 5 must reproduce it.

- [ ] **Step 4: Confirm baseline test-fast also passes**

```bash
python -m pytest tests/unit 2>&1 | tail -3
```

Expected: Pass count strictly less than the full count from Step 3 (no `@slow` tests run). Pin this count too.

---

## Task 2: Switch `make test-fast` to the `slow` marker

The marker switch is independent of the file moves and works on the current layout — all integration tests already declare `pytestmark = pytest.mark.slow`. Doing this first proves the marker plumbing works end-to-end before any files relocate, keeping each later checkpoint cleanly verifiable.

**Files:**
- Modify: `Makefile` (the `test-fast` target body)

- [ ] **Step 1: Read the target you're about to edit**

```bash
grep -n -A1 "^test-fast" Makefile
```

Expected output:
```
68:test-fast: ## ⚡ Unit tests only (skip slow integration canaries)
69-	@$(PYTHON) -m pytest tests/unit
```

- [ ] **Step 2: Edit `Makefile` to use the marker**

In `Makefile`, replace the body of the `test-fast` target. Change this line:

```makefile
test-fast: ## ⚡ Unit tests only (skip slow integration canaries)
	@$(PYTHON) -m pytest tests/unit
```

To this:

```makefile
test-fast: ## ⚡ Unit tests only (skip slow integration canaries)
	@$(PYTHON) -m pytest -m "not slow"
```

Use the `Edit` tool with `old_string = "test-fast: ## ⚡ Unit tests only (skip slow integration canaries)\n\t@$(PYTHON) -m pytest tests/unit"` and `new_string = "test-fast: ## ⚡ Unit tests only (skip slow integration canaries)\n\t@$(PYTHON) -m pytest -m \"not slow\""`. Make sure the indentation is a literal tab (Makefile requirement), not spaces.

- [ ] **Step 3: Verify `make test-fast` count matches the baseline**

```bash
make test-fast 2>&1 | tail -3
```

Expected: Same passed/skipped numbers as Task 1 Step 4. If the count differs, an integration test is missing its `@slow` marker — abort and investigate before proceeding.

- [ ] **Step 4: Verify `make test` still works**

```bash
make test 2>&1 | tail -3
```

Expected: Same total as Task 1 Step 3.

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "$(cat <<'EOF'
refactor(tests): switch test-fast target to slow-marker selection

Replaces `pytest tests/unit` with `pytest -m "not slow"`. Equivalent
selection because every integration test already declares
`pytestmark = pytest.mark.slow`. Prepares for the upcoming tests/
directory mirror-layout refactor — once unit/ and integration/
dirs disappear, the path-based selection would break.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Verify the commit landed and is GPG-signed:

```bash
git log -1 --show-signature 2>&1 | head -5
```

Expected: `Good "git" signature for ...`.

---

## Task 3: Relocate test files to mirrored layout

The 26 moves happen in a single commit. `git mv` preserves rename detection so the resulting diff shows renames, not delete+add pairs. After the moves, the now-empty `tests/unit/` and `tests/integration/` directories are removed.

**Files:**
- Create (directories): `tests/core/`, `tests/core/policies/`, `tests/envs/quidditch/rewards/`, `tests/scripts/`
- Move (via `git mv`): 26 files per the inventory above
- Delete (via `git rm`): `tests/unit/__init__.py`, `tests/integration/__init__.py`
- Delete (via `rmdir`): `tests/unit/`, `tests/integration/` (filesystem only — git tracks files, not empty dirs)

- [ ] **Step 1: Create the destination directory tree**

```bash
mkdir -p tests/core/policies tests/envs/quidditch/rewards tests/scripts
ls -d tests/core tests/core/policies tests/envs/quidditch tests/envs/quidditch/rewards tests/scripts
```

Expected: All five paths listed, no errors. (Note: `mkdir -p tests/envs/quidditch/rewards` creates `tests/envs/` and `tests/envs/quidditch/` implicitly.)

- [ ] **Step 2: Move the four "root-level" tests out of `tests/unit/`**

These four files belong at the `tests/` root because they test cross-cutting / top-level concerns (per spec §Placement decisions).

```bash
git mv tests/unit/test_config_loading.py        tests/test_config_loading.py
git mv tests/unit/test_config_schema.py         tests/test_config_schema.py
git mv tests/unit/test_meta_yaml.py             tests/test_meta_yaml.py
git mv tests/unit/test_migrate_legacy_models.py tests/test_migrate_legacy_models.py
```

Verify:

```bash
git status --short | grep -E "^R" | wc -l
```

Expected: `4` (four renames detected).

- [ ] **Step 3: Move the `core/` tests**

```bash
git mv tests/unit/test_disable_motors.py        tests/core/test_disable_motors.py
git mv tests/unit/test_render_smoke.py          tests/core/test_render_smoke.py
git mv tests/unit/test_warm_start_by_spec.py    tests/core/policies/test_warm_start_by_spec.py
git mv tests/integration/test_warm_start.py    tests/core/policies/test_warm_start.py
```

Verify:

```bash
git status --short | grep -E "^R" | wc -l
```

Expected: `8` (cumulative).

- [ ] **Step 4: Move the `envs/quidditch/` tests (unit subset)**

```bash
git mv tests/unit/test_augmented_obs.py         tests/envs/quidditch/test_augmented_obs.py
git mv tests/unit/test_crash_detector.py        tests/envs/quidditch/test_crash_detector.py
git mv tests/unit/test_env_factories.py         tests/envs/quidditch/test_env_factories.py
git mv tests/unit/test_obs_compat.py            tests/envs/quidditch/test_obs_compat.py
git mv tests/unit/test_obs_spec.py              tests/envs/quidditch/test_obs_spec.py
git mv tests/unit/test_opponent_env_world.py    tests/envs/quidditch/test_opponent_env_world.py
git mv tests/unit/test_tag_state_machine.py     tests/envs/quidditch/test_tag_state_machine.py
git mv tests/unit/test_team_mjcf.py             tests/envs/quidditch/test_team_mjcf.py
git mv tests/unit/test_reward_stack.py          tests/envs/quidditch/rewards/test_reward_stack.py
```

- [ ] **Step 5: Move the `envs/quidditch/` tests (integration subset, all `@slow`)**

```bash
git mv tests/integration/test_crash_aftermath.py      tests/envs/quidditch/test_crash_aftermath.py
git mv tests/integration/test_scoring_canary.py       tests/envs/quidditch/test_scoring_canary.py
git mv tests/integration/test_simple_env_contract.py  tests/envs/quidditch/test_simple_env_contract.py
git mv tests/integration/test_take_down.py            tests/envs/quidditch/test_take_down.py
git mv tests/integration/test_team_env_canary.py      tests/envs/quidditch/test_team_env_canary.py
```

- [ ] **Step 6: Move the `scripts/` tests**

```bash
git mv tests/unit/test_train_common_video.py        tests/scripts/test_train_common_video.py
git mv tests/unit/test_video_callback_team.py       tests/scripts/test_video_callback_team.py
git mv tests/integration/test_scripted_demos.py     tests/scripts/test_scripted_demos.py
```

- [ ] **Step 7: Confirm all 26 files moved, none missed**

```bash
git status --short | grep -E "^R" | wc -l
ls tests/unit/ 2>/dev/null
ls tests/integration/ 2>/dev/null
```

Expected:
- Rename count: `26`.
- `tests/unit/` contains only `__init__.py` (and possibly `__pycache__/` — ignored by git).
- `tests/integration/` contains only `__init__.py` (and possibly `__pycache__/`).

If any `test_*.py` files remain in `tests/unit/` or `tests/integration/`, you missed one — cross-reference against the inventory and `git mv` it before proceeding.

- [ ] **Step 8: Remove the now-vestigial `__init__.py` files**

```bash
git rm tests/unit/__init__.py tests/integration/__init__.py
```

Expected: `rm 'tests/unit/__init__.py'` and `rm 'tests/integration/__init__.py'` confirmation lines.

- [ ] **Step 9: Remove the now-empty directories (filesystem-level)**

Pycache dirs may still exist; clear them first since git doesn't, then rmdir:

```bash
rm -rf tests/unit/__pycache__ tests/integration/__pycache__
rmdir tests/unit tests/integration
ls tests/unit 2>&1
ls tests/integration 2>&1
```

Expected: `rmdir` succeeds silently. Both `ls` commands report "No such file or directory".

- [ ] **Step 10: Verify `make test` still passes with the new layout**

```bash
make test 2>&1 | tail -3
```

Expected: Same `passed[, skipped]` numbers as the Task 1 baseline. If counts diverge, pytest discovery is finding a different set of tests — abort and investigate (most likely cause: a missed move, or a test that shadows another's basename).

- [ ] **Step 11: Verify `make test-fast` also passes**

```bash
make test-fast 2>&1 | tail -3
```

Expected: Same `not slow` count as Task 1 Step 4.

- [ ] **Step 12: Verify `from tests.conftest import ...` still resolves**

Spec §`__init__.py` policy step 6: confirm preserved `tests/__init__.py` keeps the 8 importing tests happy. The above `make test` would have failed on `ImportError` if it didn't — but make it explicit:

```bash
grep -l "from tests.conftest import" tests/**/*.py tests/*.py | wc -l
python -c "from tests.conftest import build_team_world, set_body_state, hydra_compose; print('OK')"
```

Expected: `8` files matched; `OK` printed.

- [ ] **Step 13: Commit the moves**

```bash
git add -A tests/
git status --short | head -30
```

Verify the `git status` output shows only `R` (rename) and `D` (deleted) entries — no `M` (modified) or `??` (untracked). If there are unexpected entries, stop and investigate.

```bash
git commit -m "$(cat <<'EOF'
refactor(tests): mirror source tree, drop unit/integration dirs

Relocate all 26 test files from tests/{unit,integration}/ to a
layout that mirrors the source tree (tests/core/, tests/envs/quidditch/,
tests/scripts/, plus four cross-cutting tests at tests/ root).
Delete tests/{unit,integration}/__init__.py and the now-empty parent
dirs. tests/__init__.py is preserved (load-bearing for the 8 tests
that import via `from tests.conftest import ...`).

The unit-vs-integration axis now lives on the existing `slow`
pytest marker; the previous commit switched `make test-fast` to
use it. No test logic, fixture, or conftest changes — pure
file-move refactor.

See docs/superpowers/specs/2026-05-14-tests-mirror-layout-design.md
for the full design rationale.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log -1 --show-signature 2>&1 | head -5
```

Expected: Commit lands; `Good "git" signature` confirmation.

---

## Task 4: Update `test-warm` Makefile path + `slow` marker description

Two small edits across two files. Both depend on the new layout existing (so they come after Task 3) but are independent of each other.

**Files:**
- Modify: `Makefile` (the `test-warm` target body)
- Modify: `pyproject.toml` (the `slow` marker description string)

- [ ] **Step 1: Update the `test-warm` target path**

In `Makefile`, replace the old pytest path in the `test-warm` target body.

Use the `Edit` tool with:
- `old_string = "MODEL=\"$(MODEL)\" $(PYTHON) -m pytest tests/integration/test_warm_start.py"`
- `new_string = "MODEL=\"$(MODEL)\" $(PYTHON) -m pytest tests/core/policies/test_warm_start.py"`

Then verify:

```bash
grep -n -A2 "^test-warm" Makefile
```

Expected output contains `tests/core/policies/test_warm_start.py`, not `tests/integration/test_warm_start.py`.

- [ ] **Step 2: Update the `slow` marker description in `pyproject.toml`**

Use the `Edit` tool. The literal file content before and after is:

Before:
```toml
    "slow: integration tests with real episode loops (full canaries, warm-start)",
```

After:
```toml
    "slow: integration tests — real episode loops, full canaries, warm-start. `make test-fast` skips these via -m 'not slow'.",
```

Pass these exact strings as `old_string` and `new_string` to the `Edit` tool — backticks, em-dash, and embedded double-quote work as literal characters in JSON parameter values; no escaping required beyond what the tool's JSON encoding handles itself.

Verify:

```bash
grep "slow:" pyproject.toml
```

Expected: One line containing the new description with the `make test-fast` reference.

- [ ] **Step 3: Verify `pyproject.toml` is still parseable**

```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml').read()); print('OK')"
```

Expected: `OK` printed (no `TOMLDecodeError`).

- [ ] **Step 4: Verify `make test` still passes**

```bash
make test 2>&1 | tail -3
```

Expected: Same count as the Task 1 baseline.

- [ ] **Step 5: Verify `make test-warm` shape (without an actual MODEL)**

```bash
make test-warm 2>&1 | head -3
```

Expected: First line is `ERROR: MODEL=<run-name> required (see 'make list-runs')`. This proves the target's argument guard fires correctly — running the actual warm-start integration test requires a checkpoint and is out of scope here (and is the test currently known-broken per spec §Risk profile, regardless).

- [ ] **Step 6: Commit**

```bash
git add Makefile pyproject.toml
git commit -m "$(cat <<'EOF'
refactor(tests): point test-warm at new path + tighten slow marker doc

Updates the `make test-warm` target to invoke the relocated
tests/core/policies/test_warm_start.py (was tests/integration/).
Tightens the `slow` marker description in pyproject.toml to make
the canonical use of `make test-fast` (which uses -m "not slow")
explicit. Closes out the tests/ mirror-layout refactor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log -1 --show-signature 2>&1 | head -5
```

Expected: Commit lands, GPG-signed.

---

## Task 5: Post-refactor verification against baseline

Final cross-check against the baselines from Task 1. No commit — purely diagnostic.

**Files:** None modified.

- [ ] **Step 1: Confirm test count is unchanged**

```bash
make test 2>&1 | tail -3
```

Expected: Same `<N> passed[, <M> skipped]` summary as Task 1 Step 3.

- [ ] **Step 2: Confirm collection set is identical (only paths differ)**

```bash
python -m pytest --collect-only -q > "$TMPDIR/tests-mirror-post-collect.txt" 2>&1
echo "Exit: $?"
```

Compare the post output to the baseline. The diff should consist entirely of path renames — each old path (e.g. `tests/unit/test_crash_detector.py::test_*`) replaced by the new path (`tests/envs/quidditch/test_crash_detector.py::test_*`), with no tests added or removed.

A quick way to confirm: extract just the test node IDs (drop the directory part of the path) from both files and diff them. They should be identical.

```bash
sed -E 's|^tests/[^:]+::|::|' "$TMPDIR/tests-mirror-baseline-collect.txt" | sort > /tmp/baseline-nodes.txt
sed -E 's|^tests/[^:]+::|::|' "$TMPDIR/tests-mirror-post-collect.txt"     | sort > /tmp/post-nodes.txt
diff /tmp/baseline-nodes.txt /tmp/post-nodes.txt
echo "Exit: $?"
```

Expected: Exit `0`, empty diff output. Any non-zero diff means a test was added, removed, or renamed in its own file — investigate before declaring success.

- [ ] **Step 3: Confirm `make test-fast` post-refactor count matches the pre-refactor `not slow` count**

```bash
make test-fast 2>&1 | tail -3
```

Expected: Same count as Task 1 Step 4 (the pre-refactor `pytest tests/unit` count).

- [ ] **Step 4: Confirm only `tests/` and the two config files changed against develop**

```bash
git diff develop --stat -- tests/ Makefile pyproject.toml
git diff develop --stat | grep -v -E "(tests/|Makefile|pyproject.toml|docs/superpowers/)"
```

Expected:
- First command lists 26 renames (`tests/...` → `tests/...`), 2 deletions (`__init__.py`), `Makefile`, `pyproject.toml`. The diff sizes should be small (no `+200 -180` line shuffles — just renames).
- Second command produces no output (nothing else changed).

- [ ] **Step 5: Sanity-confirm `from tests.conftest import` import path still works**

```bash
python -c "from tests.conftest import build_team_world, set_body_state, hydra_compose, qpos_addr; print('OK')"
grep -rn "from tests.conftest" tests/ | wc -l
```

Expected: `OK` printed. 8 import sites confirmed.

- [ ] **Step 6: Clean up baseline scratch files**

```bash
rm -f "$TMPDIR/tests-mirror-baseline-collect.txt" "$TMPDIR/tests-mirror-baseline-count.txt" "$TMPDIR/tests-mirror-post-collect.txt" /tmp/baseline-nodes.txt /tmp/post-nodes.txt
```

- [ ] **Step 7: Final commit summary**

```bash
git log --oneline develop..HEAD
```

Expected: Three commits on the branch, in order:
```
<hash3> refactor(tests): point test-warm at new path + tighten slow marker doc
<hash2> refactor(tests): mirror source tree, drop unit/integration dirs
<hash1> refactor(tests): switch test-fast target to slow-marker selection
```

Plus the pre-existing spec commit (`88a872f docs: spec for tests/ mirror-layout refactor`).

---

## Done criteria

All checkboxes above are checked, and:
- `git log --oneline develop..HEAD` shows 4 commits (1 spec + 3 refactor).
- `make test` and `make test-fast` both pass with the pinned baseline counts.
- `git diff develop --stat -- tests/` shows only renames + 2 deletions (no content modifications).
- No content edits in any `test_*.py` file or `conftest.py`.

At this point the branch is ready for merge into `develop` (per project convention: `git -C repo merge --no-ff feature/tests-mirror-layout`) — but the merge itself is **not** part of this plan, and should happen after the user has reviewed the branch.

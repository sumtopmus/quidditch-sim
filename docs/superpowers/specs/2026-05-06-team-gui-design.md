# Team GUI Eval — Design

**Branch:** `feature/team-gui`
**Date:** 2026-05-06
**Status:** Spec

## Problem

`scripts/eval_team.py` is currently headless-only. After training a Red learner via `make train-team-red-warm`, the user has no way to *watch* it play against a chosen Blue in the MuJoCo viewer; they can only get aggregate stats over N episodes. The single-agent counterpart (`scripts/eval_ppo.py`) already supports a viewer mode and the `QuidditchTeamEnv` already accepts `render_mode="human"` — the missing piece is a thin CLI surface and a few wiring changes.

## Goals

1. Add an opt-in `--gui` flag to `scripts/eval_team.py` that opens the MuJoCo passive viewer, runs episodes through it, and idles after the last one until the user closes the window.
2. Add an opt-in `--deterministic` flag that propagates to all `frozen:` opponents (Red, Blue, or those nested inside a `mixture:` spec), defaulting to `True` when `--gui` is set and `False` otherwise.
3. Preserve existing `make eval-team` headless-stats behavior byte-for-byte when neither flag is passed.

## Non-Goals

- No video recording or screenshot capture in this change. The team-env video callback gap (called out in `brain/index.md`) is a separate follow-up that may land on this same branch later.
- No new test suite — the viewer cannot run in CI, and the determinism plumbing is too thin to justify mocking SB3 model loading. Manual smoke is the verification path.
- No changes to `scripts/train_team_ppo.py` semantics. Its existing `from_spec(...)` calls continue to use the default `deterministic=False`.
- No new opponent types or reward changes.

## Design

### CLI surface

Two new argparse flags on `scripts/eval_team.py`:

```
--gui                 Open MuJoCo passive viewer; idle after last episode.
                      When set, --episodes default drops to 5 and
                      --deterministic default flips to True.
--deterministic       Pass deterministic=True into every frozen: opponent
                      constructed from --red / --blue specs (including
                      frozen leaves inside mixture:). Default False unless
                      --gui is set.
```

Default resolution order for `--episodes` (post-parse, since defaults depend on `--gui`):

1. If user passed `--episodes` explicitly → honor it.
2. Else if `--gui` set → 5.
3. Else → 100 (current behavior).

Same shape for `--deterministic`:

1. If user passed `--deterministic` explicitly → honor it.
2. Else if `--gui` set → `True`.
3. Else → `False`.

To detect "explicitly passed" cleanly, both flags are declared with `default=None` (sentinel) and resolved by hand after `parse_args()`:

- `--episodes`: standard `type=int, default=None` — argparse already returns `None` when omitted.
- `--deterministic`: `action="store_const", const=True, default=None` (not `store_true`) — `store_true` would default to `False` and we'd lose the "unset vs explicit-False" distinction. There's no `--no-deterministic` form because explicit-False has no use case (the only way to get stochastic behavior is to omit the flag in non-GUI mode, which is the default anyway).
- `--gui`: plain `action="store_true"` — boolean, no sentinel needed.

### Episode-loop changes

The existing per-episode loop (`while env.agents:`) is unchanged. Two surrounding changes:

1. Env construction:
   ```python
   render_mode = "human" if args.gui else None
   env = QuidditchTeamEnv(cfg=cfg, render_mode=render_mode)
   ```
   `QuidditchTeamEnv` already wires `render_mode == "human"` through to `World(render=True)`, so the viewer opens on the first `reset()` call inside the existing loop. No additional viewer-management code is needed.

2. Idle on shutdown:
   ```python
   if args.gui:
       env._world.idle()
   env.close()
   ```
   `World.idle()` already exists and blocks until the viewer window closes — same pattern `eval_ppo.py:155` uses (`env._quad.idle()` reaches the same world via the quadrotor). For team-env we go through `env._world` directly because the team env owns the world, not a single quadrotor.

### Determinism propagation

Currently `from_spec()` in [envs/quidditch/opponents.py](../../../envs/quidditch/opponents.py) only constructs `FrozenPolicyOpponent` with its constructor default (`deterministic=False`). To thread the flag without per-spec syntax (e.g. `frozen:path:deterministic=true`), we add a single kwarg:

```python
def from_spec(spec: str, *, deterministic: bool = False) -> Opponent:
    ...
    if spec.startswith("mixture:"):
        ...
        components.append((float(w_str), from_spec(sub_spec, deterministic=deterministic)))
        return MixtureOpponent(components)

    if spec.startswith("frozen:"):
        return FrozenPolicyOpponent(
            model_path=spec[len("frozen:"):],
            deterministic=deterministic,
        )
    ...
```

Scripted opponents (`beeline_*`, `intercepter_*`, `zero`) ignore the kwarg — they're already deterministic functions of observation. Only frozen leaves change behavior.

`scripts/train_team_ppo.py` calls `from_spec(args.opponent)` without the kwarg, so it retains the default `False` — training-time opponent stochasticity is unchanged.

### Makefile target

[Makefile](../../../Makefile) `eval-team` target gains two optional env-var passthroughs:

```makefile
eval-team: ## Head-to-head eval  RED=<spec>  BLUE=<spec>  [EPISODES=N] [GUI=1] [DETERMINISTIC=1]
	@test -n "$(RED)" -a -n "$(BLUE)" || { echo "ERROR: RED=<spec> BLUE=<spec> required"; exit 1; }; \
	 $(PYTHON) scripts/eval_team.py --red "$(RED)" --blue "$(BLUE)" \
	   $(if $(EPISODES),--episodes $(EPISODES)) \
	   $(if $(GUI),--gui) \
	   $(if $(DETERMINISTIC),--deterministic)
```

Note the `--episodes` flag is also moved behind the `$(if ...)` guard so the GUI-mode default (5) takes effect when `EPISODES` is unset. This is a behavior change for headless callers that previously relied on the implicit `100` from the Makefile — but the python script's argparse default is also `100` (after resolution), so headless behavior is identical.

## File changes

| File | Change |
|---|---|
| `envs/quidditch/opponents.py` | Add `deterministic: bool = False` kwarg to `from_spec`; thread through `MixtureOpponent` recursion and into `FrozenPolicyOpponent` constructor. |
| `scripts/eval_team.py` | Add `--gui` and `--deterministic` argparse flags (sentinel `None` defaults); resolve `--episodes` and `--deterministic` defaults post-parse based on `--gui`; construct env with `render_mode`; call `env._world.idle()` before `env.close()` when `--gui`. |
| `Makefile` | Extend `eval-team` target with optional `GUI=1` and `DETERMINISTIC=1` env-var passthroughs; move `--episodes` behind `$(if ...)` guard; update help text. |

## Build sequence

Three commits on `feature/team-gui`:

1. **`docs(team-gui): add design spec`** — this document.
2. **`refactor(opponents): thread deterministic flag through from_spec`** — single-purpose change to `envs/quidditch/opponents.py`. Easy to review in isolation.
3. **`feat(eval-team): add --gui and --deterministic flags`** — touches `scripts/eval_team.py` and `Makefile`. Depends on commit 2.

The branch stays open after commit 3 — a follow-up effort to add video snapshotting to team training (still in scoping; see `brain/index.md` "no video callback yet" follow-up) may land on the same branch before merge.

## Verification

Manual smoke (cannot run in CI):

```bash
# In the worktree:
cd worktrees/feature/team-gui
conda activate uav

# Trained Red, default Blue (the one Red was trained against), 1 episode, viewer open:
make eval-team \
  RED=frozen:runs/ppo_hoop_rand_start/20260506_103058/best_model \
  BLUE=beeline_blue \
  GUI=1 EPISODES=1
```

Expected:
- MuJoCo viewer opens.
- Red drone starts at randomized arena position (per `TeamConfig.randomise_red_start=True`); Blue at defender position.
- Episode runs ≤ 30 s; viewer stays open after termination/truncation; closes on user input.
- Stats summary still printed after viewer closes.

Negative checks (no regression):

```bash
# Pure headless, no flags — should be byte-identical to pre-change output:
make eval-team RED=beeline_red BLUE=beeline_blue EPISODES=10
```

Expected: same per-line stats output as before this change (modulo run-to-run variance from the shared seed loop, which is unchanged).

## Risks

- **Sentinel-based default resolution is more error-prone than argparse defaults.** Mitigation: keep the resolution logic to a small, explicit block right after `parse_args()`; add a comment pointing at this spec.
- **Viewer pacing.** `QuidditchTeamEnv.reset()` already calls `time.sleep(1)` when `render_mode == "human"` ([team_env.py:221](../../../envs/quidditch/team_env.py)), so resets between episodes pause for visual readability. No additional pacing needed.
- **`env._world` access is private.** Single-agent eval reaches into `env._quad`; we mirror that. If world ownership ever moves out of the env, both paths break together — acceptable.
- **Models with mismatched obs shape.** A `frozen:` model trained on the old 16-d single-agent obs will fail when loaded against the team env's 22-d obs. This is unrelated to this change (existing eval_team has the same failure mode) but worth noting: the GUI doesn't make it any worse.

## Out of scope (deliberately)

- Video recording / per-episode mp4 export from team eval.
- A CI-runnable headless smoke test for `--gui` (would require a virtual display + viewer mocking).
- Per-spec deterministic syntax (e.g. `frozen:path:deterministic=true`) — single global flag is enough until proven otherwise.
- A separate `make watch-team` target. The `GUI=1` modifier is sufficient.

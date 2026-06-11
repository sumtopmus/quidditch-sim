# RLlib League Step 4 — PFSP + Population Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Step 3's uniform frozen-opponent sampling with PFSP (prioritized fictitious self-play) driven by per-opponent win-rates, and replace stop-at-cap with prune-the-most-dominated-member, completing Step 4 of the RLlib league migration.

**Architecture:** All changes live in `rllib/league.py` (pure helpers + `LeagueCallback`) plus `conf/league/default.yaml`. Win-rates per (main vs frozen-opponent) matchup are logged per-episode by a new `LeagueCallback.on_episode_end` hook through the existing MetricsLogger pathway, read back off the `on_train_result` result dict, converted to PFSP sampling probabilities `∝ (1 − winrate)^p` with a uniform floor, and baked into a freshly-rebuilt `policy_mapping_fn` every iteration. Pruning is two-phase: a dominated member leaves the mapping fn immediately and is physically `remove_module`d after a grace period (in-flight episodes keep their old mapping until they end — RLlib guarantee).

**Tech Stack:** Ray RLlib new API stack (RLModule/Learner, `MultiAgentEpisode.module_for`, `Algorithm.remove_module`), Hydra `conf/` tree, pytest. Branch: `feature/rllib-league-step-4` (this worktree), off `develop` at `3b2313e` (Step 3 merge).

**Specs:** `docs/superpowers/specs/2026-06-07-rllib-league-migration-design.md` (§B3 "PFSP progression", build-sequence step 4) and `docs/superpowers/specs/2026-06-10-rllib-snapshot-league-step-3-design.md` (whose Non-goals defer PFSP + pruning here).

---

## Design decisions (settled while writing this plan)

There is no separate Step-4 spec; the parent spec's §B3 defines the requirements and these decisions close the implementation-level gaps. Flag any of them in review if they look wrong.

| # | Axis | Decision |
|---|------|----------|
| D1 | Win-rate source | **In-training per-matchup metrics** (extends Step-3's Q3 "reuse in-training metrics"). `LeagueCallback.on_episode_end` logs `league_wr_vs_<opponent_id>` = main's win (1.0/0.0) for every main-vs-frozen episode, via `metrics_logger.log_value(..., reduce="mean", window=100)` — a rolling window, more responsive than the default lifetime EMA. `on_train_result` reads the values back with the existing `read_metric`. **No callback-side EMA, no separate league-state file** (Step-3 principle preserved). |
| D2 | Unseen / restored opponents | Default win-rate **0.5** (`WINRATE_DEFAULT`) until the matchup has data. On restore, metric state resets → PFSP re-warms from uniform-ish; accepted (bounded, same class as Step 3's cooldown-reset-on-restore). |
| D3 | PFSP formula | `P(o) = (1 − floor) · (1 − wr_o)^p / Σ + floor / N`; if every opponent is fully dominated (Σ of raw weights = 0), fall back to uniform. Knobs: `pfsp_exponent` (default 2.0), `pfsp_uniform_floor` (default 0.1). |
| D4 | Mapping-fn refresh | Rebuilt + reinstalled on **every** `on_train_result` (cheap: overwrites `runner.config.policy_mapping_fn`, the same mechanism Step 3 uses on restore). In-flight episodes keep their matchup — RLlib caches the agent→module mapping per episode. |
| D5 | Pruning policy | **Prune-at-cap:** when a snapshot is due and the population is at cap, evict the most-dominated member — highest main-winrate among those `≥ prune_winrate_threshold` (default 0.8). If no member qualifies, **don't snapshot** (Step-3 stop-at-cap behavior preserved for still-challenging populations). |
| D6 | Removal safety | **Two-phase:** the victim is excluded from the mapping fn immediately (pending state) and physically removed via `Algorithm.remove_module` after `prune_grace_iters` iterations (default 2). Rationale: `remove_module`'s docstring says ongoing episodes keep their old mapping till episode end — removing the module out from under such an episode would KeyError the env runner. Constraint to keep documented in the YAML: `prune_grace_iters × train_batch_size_per_learner ≥ episode length in env steps` (real config: 2 × 8192 ≫ 30 s × 120 Hz = 3600 ✓). |
| D7 | Version reuse | Snapshot version numbers are **never reused within a run** (`LeagueCallback` keeps a per-side version floor, since `next_version` over post-prune ids could recycle a pruned id and inherit its stale metric window). Across a restore where the highest version was pruned pre-checkpoint, reuse is possible — accepted, same class as D2. |
| D8 | Determinism plumbing | The per-episode rolls move from the two-draw `_episode_rng_roll` to salted single draws `_episode_roll(episode, salt)` (`zlib.crc32(f"{id}:{salt}")`). The old scheme's second draw had only ~4300 distinct values (crc32 is 32-bit; `seed // 1_000_003` exhausts it) and offers no independent third draw, which the weighted member pick needs. |

**Non-goals (unchanged from the specs):** no dedicated eval battery for the snapshot trigger (Step 5, per parent build sequence), no reward anneal / curriculum levers / promotion adapter (Step 5), no AlphaStar-lite exploiters, no obs changes, no scripted seeds.

## File structure

| File | Change |
|------|--------|
| `rllib/league.py` | All Step-4 logic: `_episode_roll`, `WINRATE_DEFAULT`, `WINRATE_WINDOW`, `pfsp_weights`, `weighted_pick`, `winrate_key`, `matchup_outcome`, `collect_winrates`, `prune_candidate`, `report_league`; extend `make_league_mapping_fn` (optional `winrates` param); extend `LeagueCallback` (`on_episode_end`, per-iteration refresh, prune-at-cap, pending removals, version floor). |
| `conf/league/default.yaml` | Add `pfsp_exponent`, `pfsp_uniform_floor`, `prune_winrate_threshold`, `prune_grace_iters`. |
| `tests/rllib/test_league.py` | Unit tests for every helper + callback behavior; extend `_FakeAlgo` with `remove_module`. |
| `tests/rllib/test_smoke_train.py` | New slow integration test: snapshot → cap → prune → physical removal → checkpoint/restore. |

`rllib/config_builder.py` needs **no change**: `make_league_mapping_fn` gains only an optional parameter, and `LeagueCallback` is already wired when `cfg.league.enabled`.

## Execution notes

- Run everything via `uv run …` from the worktree root (`worktrees/feature/rllib-league-step-4/`). `uv`/`make`/`dsim` need the sandbox disabled (uv cache lives in `~/.cache/uv`).
- Fast loop: `uv run pytest tests/rllib/test_league.py -v`. Full fast suite: `make test-fast`. Slow integration: `uv run pytest tests/rllib/test_smoke_train.py -v -m slow` (W&B is disabled by `tests/conftest.py`).
- Commits are GPG-signed by default; use the `feat:`/`test:`/`docs:` taxonomy as written in each task.

---

### Task 1: Salted per-episode rolls (`_episode_roll`)

Replaces `_episode_rng_roll` (two coupled draws) with independent salted draws, keeping mapping determinism per episode id. Pure refactor — all existing mapping tests must stay green.

**Files:**
- Modify: `rllib/league.py` (replace `_episode_rng_roll`, rewrite `make_league_mapping_fn` internals)
- Test: `tests/rllib/test_league.py`

- [x] **Step 1: Write the failing test**

Append to `tests/rllib/test_league.py` (the `_episode` helper already exists there):

```python
def test_episode_roll_deterministic_salted_and_bounded():
    ep = _episode("abc")
    r = L._episode_roll(ep, "mode")
    assert r == L._episode_roll(ep, "mode")          # deterministic
    assert 0.0 <= r < 1.0
    # different salts and different episode ids give different draws
    assert r != L._episode_roll(ep, "member")
    assert r != L._episode_roll(_episode("xyz"), "mode")
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/rllib/test_league.py::test_episode_roll_deterministic_salted_and_bounded -v`
Expected: FAIL — `AttributeError: module 'rllib.league' has no attribute '_episode_roll'`

- [x] **Step 3: Implement**

In `rllib/league.py`, replace the whole `_episode_rng_roll` function with:

```python
def _episode_roll(episode, salt: str) -> float:
    """Deterministic [0,1) draw keyed on (episode id, salt), so both agents in
    one episode see the same matchup and salts give independent draws.
    zlib.crc32 is stable across processes, unlike hash() under PYTHONHASHSEED.
    """
    eid = str(getattr(episode, "id_", episode))
    return zlib.crc32(f"{eid}:{salt}".encode()) / 2**32
```

and rewrite the inner `league_mapping_fn` in `make_league_mapping_fn` to use it (factory signature unchanged in this task):

```python
    def league_mapping_fn(agent_id, episode, **kw):
        mode = "live"
        if _episode_roll(episode, "mode") >= live_fraction:
            if _episode_roll(episode, "side") < 0.5:
                mode = "red_exploits" if blue_pop else "live"
            else:
                mode = "blue_exploits" if red_pop else "live"
        if mode == "live":
            return "main_red" if agent_id == "red_0" else "main_blue"
        member = _episode_roll(episode, "member")
        if mode == "red_exploits":
            if agent_id == "red_0":
                return "main_red"
            return blue_pop[int(member * len(blue_pop)) % len(blue_pop)]
        # blue_exploits
        if agent_id == "red_0":
            return red_pop[int(member * len(red_pop)) % len(red_pop)]
        return "main_blue"
```

- [x] **Step 4: Run the league unit tests**

Run: `uv run pytest tests/rllib/test_league.py -v`
Expected: ALL PASS (the distribution-based mapping tests tolerate the new draw values).

- [x] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -m "refactor(rllib): salted per-episode rolls for the league mapping fn"
```

### Task 2: `pfsp_weights` pure helper

**Files:**
- Modify: `rllib/league.py`
- Test: `tests/rllib/test_league.py`

- [x] **Step 1: Write the failing tests**

```python
def test_pfsp_weights_prioritizes_hard_opponents():
    probs = L.pfsp_weights({"a": 0.9, "b": 0.1}, exponent=2.0, floor=0.0)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    # raw weights (1-0.9)^2 = 0.01 vs (1-0.1)^2 = 0.81 -> b gets ~98.8%
    assert probs["b"] > 0.95 > probs["a"] > 0.0


def test_pfsp_weights_uniform_floor_is_anti_forgetting():
    probs = L.pfsp_weights({"a": 1.0, "b": 0.0}, exponent=2.0, floor=0.2)
    # 'a' is fully dominated; the floor still guarantees it floor/N mass
    assert abs(probs["a"] - 0.1) < 1e-9
    assert abs(probs["b"] - 0.9) < 1e-9


def test_pfsp_weights_all_dominated_falls_back_to_uniform():
    assert L.pfsp_weights({"a": 1.0, "b": 1.0}, exponent=2.0, floor=0.0) == {
        "a": 0.5, "b": 0.5}


def test_pfsp_weights_empty():
    assert L.pfsp_weights({}, exponent=2.0, floor=0.1) == {}
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/rllib/test_league.py -k pfsp_weights -v`
Expected: FAIL — `AttributeError: ... no attribute 'pfsp_weights'`

- [x] **Step 3: Implement**

Add to `rllib/league.py` (below `next_version`):

```python
WINRATE_DEFAULT = 0.5   # assumed winrate vs an opponent with no data yet


def pfsp_weights(
    winrates: dict[str, float], exponent: float, floor: float
) -> dict[str, float]:
    """Normalized PFSP sampling probabilities over frozen opponents.

    P(o) ∝ (1 − winrate_vs_o)^exponent — concentrate on opponents the main
    struggles against — mixed with a uniform floor so dominated opponents keep
    nonzero mass (anti-forgetting). All-dominated (Σ raw = 0) → uniform.
    """
    if not winrates:
        return {}
    raw = {m: (1.0 - min(max(w, 0.0), 1.0)) ** exponent for m, w in winrates.items()}
    total = sum(raw.values())
    n = len(raw)
    if total <= 0.0:
        return {m: 1.0 / n for m in raw}
    return {m: (1.0 - floor) * v / total + floor / n for m, v in raw.items()}
```

- [x] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/rllib/test_league.py -k pfsp_weights -v`
Expected: 4 PASS

- [x] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -m "feat(rllib): PFSP sampling-weight helper (exponent + uniform floor)"
```

### Task 3: `weighted_pick` pure helper

**Files:**
- Modify: `rllib/league.py`
- Test: `tests/rllib/test_league.py`

- [x] **Step 1: Write the failing test**

```python
def test_weighted_pick_respects_cumulative_probs():
    items, probs = ["a", "b"], {"a": 0.25, "b": 0.75}
    assert L.weighted_pick(items, probs, 0.0) == "a"
    assert L.weighted_pick(items, probs, 0.24) == "a"
    assert L.weighted_pick(items, probs, 0.25) == "b"
    assert L.weighted_pick(items, probs, 0.999) == "b"
    # float-accumulation guard: a roll at/above the total still returns an item
    assert L.weighted_pick(items, probs, 1.0) == "b"
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/rllib/test_league.py::test_weighted_pick_respects_cumulative_probs -v`
Expected: FAIL — `AttributeError: ... no attribute 'weighted_pick'`

- [x] **Step 3: Implement**

Add to `rllib/league.py` (below `pfsp_weights`):

```python
def weighted_pick(items: list[str], probs: dict[str, float], roll: float) -> str:
    """Pick the item whose cumulative-probability bucket contains `roll`."""
    acc = 0.0
    for item in items:
        acc += probs[item]
        if roll < acc:
            return item
    return items[-1]   # roll == 1.0 or float round-off past the total
```

- [x] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/rllib/test_league.py::test_weighted_pick_respects_cumulative_probs -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -m "feat(rllib): deterministic weighted pick for PFSP opponent draws"
```

### Task 4: PFSP-weighted `make_league_mapping_fn`

The factory gains an optional `winrates` param. `None`/missing entries default to `WINRATE_DEFAULT` → behaves like uniform sampling, so `config_builder.py` and the Step-3 call sites need no change.

**Files:**
- Modify: `rllib/league.py:make_league_mapping_fn`
- Test: `tests/rllib/test_league.py`

- [x] **Step 1: Write the failing tests**

```python
def test_mapping_pfsp_concentrates_on_hard_opponents():
    ids = {"main_red", "main_blue", "blue_pop_v1", "blue_pop_v2"}
    cfg = {"live_fraction": 0.0, "pfsp_exponent": 2.0, "pfsp_uniform_floor": 0.0}
    # main_red dominates v1 (wr 0.9) and struggles vs v2 (wr 0.1)
    fn = L.make_league_mapping_fn(
        ids, cfg, winrates={"blue_pop_v1": 0.9, "blue_pop_v2": 0.1})
    picks = [fn("blue_0", _episode(eid)) for eid in range(4000)]
    frozen = [p for p in picks if p != "main_blue"]
    assert frozen, "expected red-exploits episodes"
    assert frozen.count("blue_pop_v2") / len(frozen) > 0.9   # ~0.988 expected


def test_mapping_without_winrates_is_uniform():
    ids = {"main_red", "main_blue", "blue_pop_v1", "blue_pop_v2"}
    fn = L.make_league_mapping_fn(ids, {"live_fraction": 0.0})
    picks = [fn("blue_0", _episode(eid)) for eid in range(4000)]
    frozen = [p for p in picks if p != "main_blue"]
    assert 0.4 < frozen.count("blue_pop_v1") / len(frozen) < 0.6
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/rllib/test_league.py -k "pfsp_concentrates or without_winrates" -v`
Expected: FAIL — `TypeError: make_league_mapping_fn() got an unexpected keyword argument 'winrates'`

- [x] **Step 3: Implement**

Replace `make_league_mapping_fn` in `rllib/league.py` with:

```python
def make_league_mapping_fn(
    module_ids: Iterable[str],
    league_cfg: dict,
    winrates: Optional[dict[str, float]] = None,
):
    """Build the per-episode matchup fn (PFSP sampling, Step 4).

    With prob `live_fraction`: main_red vs main_blue (both learn). Otherwise
    split 50/50: 'red exploits' (main_red vs a blue_pop member) or 'blue
    exploits' (a red_pop member vs main_blue). Frozen members are drawn PFSP:
    P(o) ∝ (1 − winrate_vs_o)^pfsp_exponent + uniform floor; opponents with no
    win-rate data yet weigh in at WINRATE_DEFAULT, so winrates=None degrades to
    uniform (the Step-3 behavior). An empty opposite population makes that
    exploit mode fall back to live. Closes over plain lists/floats/dicts only,
    so it pickles across the Ray boundary.
    """
    red_pop = population_members(module_ids, RED_POP_RE)
    blue_pop = population_members(module_ids, BLUE_POP_RE)
    live_fraction = float(league_cfg.get("live_fraction", 0.5))
    exponent = float(league_cfg.get("pfsp_exponent", 2.0))
    floor = float(league_cfg.get("pfsp_uniform_floor", 0.1))
    wr = winrates or {}
    red_probs = pfsp_weights(
        {m: wr.get(m, WINRATE_DEFAULT) for m in red_pop}, exponent, floor)
    blue_probs = pfsp_weights(
        {m: wr.get(m, WINRATE_DEFAULT) for m in blue_pop}, exponent, floor)

    def league_mapping_fn(agent_id, episode, **kw):
        mode = "live"
        if _episode_roll(episode, "mode") >= live_fraction:
            if _episode_roll(episode, "side") < 0.5:
                mode = "red_exploits" if blue_pop else "live"
            else:
                mode = "blue_exploits" if red_pop else "live"
        if mode == "live":
            return "main_red" if agent_id == "red_0" else "main_blue"
        member = _episode_roll(episode, "member")
        if mode == "red_exploits":
            if agent_id == "red_0":
                return "main_red"
            return weighted_pick(blue_pop, blue_probs, member)
        # blue_exploits
        if agent_id == "red_0":
            return weighted_pick(red_pop, red_probs, member)
        return "main_blue"

    return league_mapping_fn
```

- [x] **Step 4: Run the full league test file**

Run: `uv run pytest tests/rllib/test_league.py -v`
Expected: ALL PASS (Step-3 mapping tests exercise the `winrates=None` uniform path).

- [x] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -m "feat(rllib): PFSP-weighted opponent sampling in the league mapping fn"
```

### Task 5: Per-matchup win logging (`matchup_outcome` + `on_episode_end`)

`LeagueCallback` gains an episode hook that logs the main's win against the specific frozen opponent it faced. The matchup comes from `episode.module_for(agent_id)` (`MultiAgentEpisode` caches the per-episode agent→module mapping); the scored flag is reused from `ScoreMetricsCallback`'s `score_acc` accumulator in `episode.custom_data` (both callbacks are always installed together by `config_builder` when the league is enabled — guard defensively anyway).

**Files:**
- Modify: `rllib/league.py`
- Test: `tests/rllib/test_league.py`

- [x] **Step 1: Write the failing tests**

```python
def test_matchup_outcome_main_vs_frozen():
    # red main vs frozen blue: red's win = it scored
    assert L.matchup_outcome("main_red", "blue_pop_v1", scored=True) == (
        "league_wr_vs_blue_pop_v1", 1.0)
    assert L.matchup_outcome("main_red", "blue_pop_v1", scored=False) == (
        "league_wr_vs_blue_pop_v1", 0.0)
    # blue main vs frozen red: blue's win = it prevented the score
    assert L.matchup_outcome("red_pop_v3", "main_blue", scored=False) == (
        "league_wr_vs_red_pop_v3", 1.0)
    assert L.matchup_outcome("red_pop_v3", "main_blue", scored=True) == (
        "league_wr_vs_red_pop_v3", 0.0)


def test_matchup_outcome_live_episode_is_none():
    assert L.matchup_outcome("main_red", "main_blue", scored=True) is None


class _FakeMetricsLogger:
    def __init__(self):
        self.logged = []

    def log_value(self, key, value, **kw):
        self.logged.append((key, value))


def _ended_episode(red_mod, blue_mod, scored):
    mapping = {"red_0": red_mod, "blue_0": blue_mod}
    return types.SimpleNamespace(
        id_="ep1",
        custom_data={"score_acc": {"scored": scored, "min_dist": 1.0}},
        module_for=lambda aid, m=mapping: m[aid],
    )


def test_on_episode_end_logs_main_vs_frozen_winrate():
    cb = L.LeagueCallback()
    logger = _FakeMetricsLogger()
    cb.on_episode_end(
        episode=_ended_episode("main_red", "blue_pop_v1", scored=True),
        metrics_logger=logger)
    assert ("league_wr_vs_blue_pop_v1", 1.0) in logger.logged


def test_on_episode_end_skips_live_and_unaccumulated_episodes():
    cb = L.LeagueCallback()
    logger = _FakeMetricsLogger()
    cb.on_episode_end(
        episode=_ended_episode("main_red", "main_blue", scored=True),
        metrics_logger=logger)
    ep = _ended_episode("main_red", "blue_pop_v1", scored=True)
    ep.custom_data = {}   # ScoreMetricsCallback absent -> no acc -> skip
    cb.on_episode_end(episode=ep, metrics_logger=logger)
    assert logger.logged == []
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/rllib/test_league.py -k "matchup_outcome or on_episode_end" -v`
Expected: FAIL — `AttributeError: ... no attribute 'matchup_outcome'`

- [x] **Step 3: Implement**

Add to `rllib/league.py` (below `weighted_pick`):

```python
# Rolling window for the per-matchup winrate means. More responsive than the
# MetricsLogger default lifetime EMA (coeff 0.01), which would take hundreds of
# episodes per matchup to move off its initial value. A module constant (not a
# league_cfg knob) because on_episode_end runs on env runners, where the
# callback never sees on_algorithm_init / the league cfg.
WINRATE_WINDOW = 100


def winrate_key(opponent_id: str) -> str:
    """Metric key carrying the main's winrate vs one frozen opponent. The main
    is implied by the population the opponent belongs to (main_red plays
    blue_pop members, main_blue plays red_pop members)."""
    return f"league_wr_vs_{opponent_id}"


def matchup_outcome(
    red_module: Optional[str], blue_module: Optional[str], scored: bool
) -> Optional[tuple[str, float]]:
    """(winrate metric key, main's win 1.0/0.0) for a main-vs-frozen episode.

    Red main wins when it scores; Blue main wins when it prevents the score
    (the same semantics as red_score_rate / blue_prevention_rate). Live
    main-vs-main episodes carry no PFSP signal -> None.
    """
    if red_module == "main_red" and blue_module and BLUE_POP_RE.match(blue_module):
        return winrate_key(blue_module), 1.0 if scored else 0.0
    if blue_module == "main_blue" and red_module and RED_POP_RE.match(red_module):
        return winrate_key(red_module), 0.0 if scored else 1.0
    return None
```

and add the hook to `LeagueCallback`:

```python
    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        """Log the main's win vs the specific frozen opponent it faced, feeding
        the PFSP winrate estimates. Reuses ScoreMetricsCallback's score_acc
        (always installed alongside this callback by config_builder); episodes
        without it are skipped rather than guessed at."""
        if metrics_logger is None:
            return
        acc = episode.custom_data.get("score_acc")
        if acc is None:
            return
        outcome = matchup_outcome(
            episode.module_for("red_0"),
            episode.module_for("blue_0"),
            bool(acc["scored"]),
        )
        if outcome is not None:
            key, win = outcome
            metrics_logger.log_value(
                key, win, reduce="mean", window=WINRATE_WINDOW)
```

- [x] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/rllib/test_league.py -v`
Expected: ALL PASS

- [x] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -m "feat(rllib): per-matchup winrate logging on episode end"
```

### Task 6: Win-rate collection + per-iteration PFSP refresh + diagnostics

`on_train_result` now (a) collects winrates off the result, (b) reinstalls a PFSP-weighted mapping fn every iteration, and (c) reports `league/wr_vs_*` + `league/pfsp_p_*` diagnostics into the result dict (mutating `result` in `on_train_result` propagates to Tune/W&B).

**Files:**
- Modify: `rllib/league.py` (`collect_winrates`, `report_league`, `LeagueCallback.on_train_result`, `LeagueCallback._refresh`)
- Test: `tests/rllib/test_league.py`

- [x] **Step 1: Write the failing tests**

```python
def test_collect_winrates_reads_only_population_keys():
    ids = {"main_red", "main_blue", "red_pop_v1", "blue_pop_v1"}
    result = {"env_runners": {"league_wr_vs_red_pop_v1": 0.8, "noise": 1.0}}
    assert L.collect_winrates(result, ids) == {"red_pop_v1": 0.8}


def test_report_league_writes_winrates_and_pfsp_probs():
    result = {}
    ids = {"main_red", "main_blue", "blue_pop_v1", "blue_pop_v2"}
    L.report_league(result, ids, {"blue_pop_v1": 0.9},
                    {"pfsp_exponent": 2.0, "pfsp_uniform_floor": 0.0})
    league = result["league"]
    assert league["wr_vs_blue_pop_v1"] == 0.9
    assert league["wr_vs_blue_pop_v2"] == L.WINRATE_DEFAULT   # unseen
    total = league["pfsp_p_blue_pop_v1"] + league["pfsp_p_blue_pop_v2"]
    assert abs(total - 1.0) < 1e-9
    assert league["pfsp_p_blue_pop_v2"] > league["pfsp_p_blue_pop_v1"]


def test_on_train_result_refreshes_mapping_fn_every_iteration():
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "blue_pop_v1"},
                     iteration=5, league_cfg=_LEAGUE_CFG)
    cb.on_algorithm_init(algorithm=algo)
    before = algo.env_runner_group.refreshed
    cb.on_train_result(algorithm=algo,
                       result={"env_runners": {"red_score_rate": 0.0,
                                               "blue_prevention_rate": 0.0}})
    assert algo.env_runner_group.refreshed > before   # PFSP weights reinstalled
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/rllib/test_league.py -k "collect_winrates or report_league or refreshes_mapping" -v`
Expected: FAIL — `AttributeError: ... no attribute 'collect_winrates'`

- [x] **Step 3: Implement**

Add to `rllib/league.py` (below `matchup_outcome`):

```python
def collect_winrates(result: dict, module_ids: Iterable[str]) -> dict[str, float]:
    """Current winrate estimate per frozen member, read off the train result.
    Members whose matchup has no logged data yet are omitted (callers default
    them to WINRATE_DEFAULT)."""
    out: dict[str, float] = {}
    for regex in (RED_POP_RE, BLUE_POP_RE):
        for opp in population_members(module_ids, regex):
            value = read_metric(result, winrate_key(opp))
            if value is not None:
                out[opp] = value
    return out


def report_league(
    result: dict, module_ids: Iterable[str],
    winrates: dict[str, float], league_cfg: dict,
) -> None:
    """Write league diagnostics into the train result (surfaces in Tune/W&B):
    per-member winrate estimates and the PFSP probabilities actually in force."""
    league = result.setdefault("league", {})
    exponent = float(league_cfg.get("pfsp_exponent", 2.0))
    floor = float(league_cfg.get("pfsp_uniform_floor", 0.1))
    for regex in (RED_POP_RE, BLUE_POP_RE):
        pop = population_members(module_ids, regex)
        probs = pfsp_weights(
            {m: winrates.get(m, WINRATE_DEFAULT) for m in pop}, exponent, floor)
        for m in pop:
            league[f"wr_vs_{m}"] = winrates.get(m, WINRATE_DEFAULT)
            league[f"pfsp_p_{m}"] = probs[m]
```

Replace `LeagueCallback.on_train_result` with (and add `_refresh`):

```python
    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        if not self._cfg:
            self._cfg = self._league_cfg(algorithm)
        it = int(algorithm.iteration)
        active_ids = _algo_module_ids(algorithm)
        winrates = collect_winrates(result, active_ids)
        for side, metric_name, main_id, regex in _SIDES:
            metric = read_metric(result, metric_name)
            if metric is None:
                continue
            if should_snapshot(
                metric=metric,
                threshold=float(self._cfg["snapshot_threshold"]),
                iters_since_last=it - self._last_snapshot_iter[side],
                cooldown=int(self._cfg["min_iters_between_snapshots"]),
                pop_size=len(population_members(active_ids, regex)),
                cap=int(self._cfg["population_cap"]),
            ):
                self._snapshot(algorithm, main_id, regex, active_ids)
                self._last_snapshot_iter[side] = it
                active_ids = _algo_module_ids(algorithm)
        # PFSP weights move every iteration -> rebuild + reinstall the mapping
        # fn each time (the same cheap config overwrite the restore path uses).
        # This also supersedes the uniform fn add_module just installed when a
        # snapshot fired above.
        self._refresh(algorithm, active_ids, winrates)
        report_league(result, active_ids, winrates, self._cfg)

    def _refresh(self, algorithm, module_ids, winrates) -> None:
        _refresh_mapping_fn(
            algorithm, make_league_mapping_fn(module_ids, self._cfg, winrates))
```

- [x] **Step 4: Run the full league test file**

Run: `uv run pytest tests/rllib/test_league.py -v`
Expected: ALL PASS (the Step-3 callback tests tolerate the extra refresh + result mutation).

- [x] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -m "feat(rllib): per-iteration PFSP weight refresh + league diagnostics"
```

### Task 7: `prune_candidate` pure helper

**Files:**
- Modify: `rllib/league.py`
- Test: `tests/rllib/test_league.py`

- [x] **Step 1: Write the failing tests**

```python
def test_prune_candidate_picks_most_dominated():
    pop = ["red_pop_v1", "red_pop_v2", "red_pop_v3"]
    wr = {"red_pop_v1": 0.95, "red_pop_v2": 0.85, "red_pop_v3": 0.4}
    assert L.prune_candidate(pop, wr, threshold=0.8) == "red_pop_v1"


def test_prune_candidate_none_when_population_still_challenging():
    pop = ["red_pop_v1", "red_pop_v2"]
    wr = {"red_pop_v1": 0.6}   # v2 unseen -> WINRATE_DEFAULT (0.5)
    assert L.prune_candidate(pop, wr, threshold=0.8) is None
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/rllib/test_league.py -k prune_candidate -v`
Expected: FAIL — `AttributeError: ... no attribute 'prune_candidate'`

- [x] **Step 3: Implement**

Add to `rllib/league.py` (below `collect_winrates`):

```python
def prune_candidate(
    pop: list[str], winrates: dict[str, float], threshold: float
) -> Optional[str]:
    """The most-dominated member — highest main-winrate among those clearing
    `threshold` — or None when every member still puts up a fight (then the
    population is full of useful opponents and Step-3 stop-at-cap applies)."""
    dominated = [
        (winrates.get(m, WINRATE_DEFAULT), m) for m in pop
        if winrates.get(m, WINRATE_DEFAULT) >= threshold
    ]
    return max(dominated)[1] if dominated else None
```

- [x] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/rllib/test_league.py -k prune_candidate -v`
Expected: 2 PASS

- [x] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -m "feat(rllib): prune-candidate selection (most-dominated member)"
```

### Task 8: Prune-at-cap + two-phase removal in `LeagueCallback`

The callback gains pending-removal state and a per-side version floor. At cap, a dominated member is marked pending (immediately excluded from new matchups), the new snapshot proceeds, and the victim is physically removed `prune_grace_iters` iterations later.

**Files:**
- Modify: `rllib/league.py` (`LeagueCallback.__init__`, `on_algorithm_init`, `on_train_result`, `_snapshot`, new `_process_pending_removals`)
- Test: `tests/rllib/test_league.py` (extend `_FakeAlgo` + `_LEAGUE_CFG`)

- [x] **Step 1: Extend the fakes and write the failing tests**

In `tests/rllib/test_league.py`, add to `_FakeAlgo.__init__`: `self.removed = []`, and add the method:

```python
    def remove_module(self, module_id, **kw):
        self.removed.append(module_id)
        self._module._ids.discard(module_id)
```

Replace `_LEAGUE_CFG` with:

```python
_LEAGUE_CFG = {"snapshot_threshold": 0.7, "min_iters_between_snapshots": 20,
               "population_cap": 5, "live_fraction": 0.5,
               "pfsp_exponent": 2.0, "pfsp_uniform_floor": 0.1,
               "prune_winrate_threshold": 0.8, "prune_grace_iters": 2}
```

Add the tests:

```python
def _prune_cfg(**over):
    return {**_LEAGUE_CFG, "population_cap": 1,
            "min_iters_between_snapshots": 0, **over}


def test_callback_prunes_most_dominated_at_cap(monkeypatch):
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "blue_pop_v1"},
                     iteration=0, league_cfg=_prune_cfg())
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "blue_prevention_rate": 0.9,        # blue snapshot due, blue_pop at cap
        "red_score_rate": 0.0,              # red side quiet
        "league_wr_vs_blue_pop_v1": 0.95,   # v1 dominated -> prunable
    }})
    assert "blue_pop_v2" in algo.added                  # snapshot proceeded
    assert "blue_pop_v1" in cb._pending_removal         # victim marked...
    assert algo.removed == []                           # ...not yet removed


def test_pending_removal_executes_after_grace(monkeypatch):
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "blue_pop_v1"},
                     iteration=0, league_cfg=_prune_cfg(prune_grace_iters=2))
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "blue_prevention_rate": 0.9, "red_score_rate": 0.0,
        "league_wr_vs_blue_pop_v1": 0.95}})
    quiet = {"env_runners": {"blue_prevention_rate": 0.0, "red_score_rate": 0.0}}
    algo.iteration = 2
    cb.on_train_result(algorithm=algo, result=dict(quiet))   # grace not elapsed
    assert algo.removed == []
    algo.iteration = 3
    cb.on_train_result(algorithm=algo, result=dict(quiet))   # 3 - 1 >= 2 -> due
    assert algo.removed == ["blue_pop_v1"]
    assert "blue_pop_v1" not in cb._pending_removal


def test_no_snapshot_at_cap_when_no_member_is_dominated(monkeypatch):
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "blue_pop_v1"},
                     iteration=0, league_cfg=_prune_cfg())
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "blue_prevention_rate": 0.9, "red_score_rate": 0.0,
        "league_wr_vs_blue_pop_v1": 0.4}})   # still challenging -> keep
    assert algo.added == []                  # Step-3 stop-at-cap preserved
    assert cb._pending_removal == {}


def test_snapshot_version_never_reused_after_prune(monkeypatch):
    monkeypatch.setattr(L, "_snapshot_spec", lambda algo, main_id: object())
    cb = L.LeagueCallback()
    algo = _FakeAlgo({"main_red", "main_blue", "blue_pop_v1"},
                     iteration=0, league_cfg=_prune_cfg())
    cb.on_algorithm_init(algorithm=algo)
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result={"env_runners": {
        "blue_prevention_rate": 0.9, "red_score_rate": 0.0,
        "league_wr_vs_blue_pop_v1": 0.95}})
    # the active pop at snapshot time was empty (v1 pending), yet the version
    # floor prevents recycling v1 for a brand-new policy
    assert "blue_pop_v2" in algo.added
    assert "blue_pop_v1" not in algo.added
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/rllib/test_league.py -k "prunes_most_dominated or pending_removal or no_snapshot_at_cap or never_reused" -v`
Expected: FAIL — `AttributeError: 'LeagueCallback' object has no attribute '_pending_removal'` (and `test_callback_prunes_most_dominated_at_cap` sees no snapshot added because the pop is at cap).

- [x] **Step 3: Implement**

In `rllib/league.py`, update `LeagueCallback.__init__`:

```python
    def __init__(self):
        super().__init__()
        self._cfg: dict = {}
        self._last_snapshot_iter = {"red": 0, "blue": 0}
        # Two-phase pruning: victim module id -> iteration it was marked. A
        # pending member is out of the mapping fn immediately but only
        # physically removed after the grace period, because in-flight episodes
        # keep their old agent->module mapping until they end (RLlib contract)
        # and would KeyError on a module yanked mid-episode.
        self._pending_removal: dict[str, int] = {}
        # Highest snapshot version ever issued per side this run; prevents a
        # pruned id from being recycled (which would inherit the stale metric
        # window of the dead policy).
        self._version_floor = {"red": 0, "blue": 0}
```

Update `on_algorithm_init` (full replacement):

```python
    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        self._cfg = self._league_cfg(algorithm)
        it = int(algorithm.iteration)
        self._last_snapshot_iter = {"red": it, "blue": it}
        # Restore note: pending removals and the version floor reset here. A
        # victim checkpointed mid-grace rejoins the active population (one
        # member over cap until the next prune) and a pruned max-version id can
        # be reused across a restore — both bounded, same class as the
        # cooldown reset.
        self._pending_removal = {}
        module_ids = _algo_module_ids(algorithm)
        self._version_floor = {
            "red": next_version(module_ids, RED_POP_RE) - 1,
            "blue": next_version(module_ids, BLUE_POP_RE) - 1,
        }
        _refresh_mapping_fn(algorithm, make_league_mapping_fn(module_ids, self._cfg))
```

Replace `on_train_result` and `_snapshot`, and add `_process_pending_removals`:

```python
    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        if not self._cfg:
            self._cfg = self._league_cfg(algorithm)
        it = int(algorithm.iteration)
        self._process_pending_removals(algorithm, it)
        active_ids = _algo_module_ids(algorithm) - set(self._pending_removal)
        winrates = collect_winrates(result, active_ids)
        for side, metric_name, main_id, regex in _SIDES:
            metric = read_metric(result, metric_name)
            if metric is None:
                continue
            pop = population_members(active_ids, regex)
            victim = None
            if len(pop) >= int(self._cfg["population_cap"]):
                victim = prune_candidate(
                    pop, winrates,
                    float(self._cfg.get("prune_winrate_threshold", 0.8)))
            if should_snapshot(
                metric=metric,
                threshold=float(self._cfg["snapshot_threshold"]),
                iters_since_last=it - self._last_snapshot_iter[side],
                cooldown=int(self._cfg["min_iters_between_snapshots"]),
                pop_size=len(pop) - (1 if victim else 0),
                cap=int(self._cfg["population_cap"]),
            ):
                if victim is not None:
                    self._pending_removal[victim] = it
                    active_ids = active_ids - {victim}
                self._snapshot(algorithm, main_id, side, regex, active_ids)
                self._last_snapshot_iter[side] = it
                active_ids = _algo_module_ids(algorithm) - set(self._pending_removal)
        # PFSP weights move every iteration -> rebuild + reinstall the mapping
        # fn each time (the same cheap config overwrite the restore path uses).
        # This also supersedes the uniform fn add_module just installed when a
        # snapshot fired above, and drops pending victims from new matchups.
        self._refresh(algorithm, active_ids, winrates)
        report_league(result, active_ids, winrates, self._cfg)
        result["league"]["pending_removals"] = len(self._pending_removal)

    def _process_pending_removals(self, algorithm, it: int) -> None:
        """Physically remove victims whose grace period has elapsed. By now no
        env runner routes new episodes to them (excluded from the mapping fn
        since mark time) and in-flight episodes from before the mark have ended
        (grace_iters x batch >= episode length — see conf/league/default.yaml)."""
        grace = int(self._cfg.get("prune_grace_iters", 2))
        due = [m for m, marked in self._pending_removal.items()
               if it - marked >= grace]
        for victim in due:
            algorithm.remove_module(
                module_id=victim,
                new_should_module_be_updated=["main_red", "main_blue"],
            )
            del self._pending_removal[victim]

    def _snapshot(self, algorithm, main_id, side, regex, module_ids) -> None:
        ver = max(next_version(module_ids, regex), self._version_floor[side] + 1)
        self._version_floor[side] = ver
        new_id = f"{side}_pop_v{ver}"
        # Uniform fn at add time is fine — the trailing _refresh in
        # on_train_result reinstalls the PFSP-weighted fn in the same call.
        new_mapping_fn = make_league_mapping_fn(module_ids | {new_id}, self._cfg)
        # Add a fresh (main-arch) module, kept out of the gradient update, and
        # refresh the mapping fn across the env-runner group in the same call.
        algorithm.add_module(
            module_id=new_id,
            module_spec=_snapshot_spec(algorithm, main_id),
            new_should_module_be_updated=["main_red", "main_blue"],
            new_agent_to_module_mapping_fn=new_mapping_fn,
        )
        # Copy the live main's weights into the frozen snapshot on the learner —
        # the authoritative copy, excluded from the gradient update — via the
        # canonical RLlib self-play set_state path.
        main_state = algorithm.get_module(main_id).get_state()
        algorithm.set_state(
            {"learner_group": {"learner": {"rl_module": {new_id: main_state}}}}
        )
        # set_state's learner->env-runner sync is inference-only and lands on the
        # *next* iteration, so the freshly-added env-runner module would serve the
        # frozen opponent with random weights for one iteration. Copy main->snap
        # directly on every env runner (incl. the local one) so the opponent has
        # the main's weights from its very next episode.
        def _seed_snapshot(runner, m=main_id, n=new_id):
            runner.module[n].set_state(runner.module[m].get_state())

        algorithm.env_runner_group.foreach_env_runner(
            _seed_snapshot, local_env_runner=True
        )
```

(The `_snapshot` body below `new_mapping_fn` is unchanged from Step 3 — only the signature, the version-floor id computation, and the comment above `new_mapping_fn` are new.)

- [x] **Step 4: Run the full league test file**

Run: `uv run pytest tests/rllib/test_league.py -v`
Expected: ALL PASS. Note `test_callback_snapshots_red_when_threshold_cleared` (Step 3) still passes: empty pop → no victim → version floor 0 → `red_pop_v1` as before.

- [x] **Step 5: Commit**

```bash
git add rllib/league.py tests/rllib/test_league.py
git commit -m "feat(rllib): prune-at-cap with two-phase removal + version floor"
```

### Task 9: League config knobs

**Files:**
- Modify: `conf/league/default.yaml`

(No schema change needed — the `league` group is plain YAML read via `league_cfg.get(...)`; `config_schema.py` doesn't cover it.)

- [x] **Step 1: Update the YAML**

Replace the full contents of `conf/league/default.yaml` with:

```yaml
# Snapshot-population league knobs (Steps 3+4).
#   snapshot_threshold:          win-rate (red_score_rate / blue_prevention_rate)
#                                a main must clear to freeze a snapshot.
#   min_iters_between_snapshots: cooldown so a main doesn't snapshot every iter.
#   population_cap:              max frozen members per side. At cap, the most
#                                dominated member is pruned to make room — or,
#                                if none clears prune_winrate_threshold, no
#                                snapshot is taken (Step-3 stop-at-cap).
#   live_fraction:               P(main-vs-main episode); rest splits 50/50 into
#                                red/blue exploits a PFSP-sampled frozen member.
#   pfsp_exponent:               p in P(o) ∝ (1 − winrate_vs_o)^p — concentrate
#                                on opponents the main still loses to.
#   pfsp_uniform_floor:          probability mass spread uniformly over the
#                                population (anti-forgetting coverage).
#   prune_winrate_threshold:     a member is "dominated" (prunable) once the
#                                opposite main beats it at this rate.
#   prune_grace_iters:           iterations between dropping a victim from the
#                                mapping fn and physically removing its module.
#                                Must satisfy: grace × train_batch_size_per_learner
#                                ≥ episode length in env steps (in-flight episodes
#                                keep their old matchup until they end).
enabled: true
snapshot_threshold: 0.7
min_iters_between_snapshots: 20
population_cap: 5
live_fraction: 0.5
pfsp_exponent: 2.0
pfsp_uniform_floor: 0.1
prune_winrate_threshold: 0.8
prune_grace_iters: 2
```

- [x] **Step 2: Sanity-check composition + fast suite**

Run: `uv run python -c "from omegaconf import OmegaConf; c = OmegaConf.load('conf/league/default.yaml'); print(dict(c))"`
Expected: dict with all 9 keys.

Run: `uv run pytest tests/rllib -m "not slow" -v`
Expected: ALL PASS

- [x] **Step 3: Commit**

```bash
git add conf/league/default.yaml
git commit -m "feat(conf): PFSP + pruning knobs in the league config group"
```

### Task 10: Integration smoke test (snapshot → PFSP → prune → restore)

A few-iteration real run, `@pytest.mark.slow`, mirroring Step 3's `test_league_snapshots_freezes_and_restores`. With `population_cap=1` and zeroed thresholds: iteration 1 snapshots `*_pop_v1`; iteration 2 snapshots `*_pop_v2` and marks `v1` pending; iteration 3 physically removes `v1` (grace 0 → due at the next `on_train_result`). In-flight safety holds in-test: episodes are 120 steps (1.0 s), each 256-step iteration finishes them before the removal at iteration 3's end.

**Files:**
- Modify: `tests/rllib/test_smoke_train.py`

- [x] **Step 1: Write the test**

Append to `tests/rllib/test_smoke_train.py`:

```python
@pytest.mark.slow
def test_league_pfsp_prunes_and_restores(tmp_path):
    """cap=1 + zero thresholds: iter 1 snapshots v1; iter 2 snapshots v2 and
    marks v1 pending (out of new matchups, module still present); iter 3
    physically removes v1. League diagnostics appear in the result; the
    surviving member round-trips a checkpoint."""
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.league import RED_POP_RE, BLUE_POP_RE, population_members
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    cfg = OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0",
                       "policies_to_train": ["main_red", "main_blue"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "learned"}}},
        # Short episodes so several complete per 256-step batch -> metrics flow
        # and no episode straddles the prune (120 steps << 256/iteration).
        "curriculum": {"randomise_start": True, "episode_seconds": 1.0},
        "league": {"enabled": True, "snapshot_threshold": 0.0,
                   "min_iters_between_snapshots": 0, "population_cap": 1,
                   "live_fraction": 0.5, "pfsp_exponent": 2.0,
                   "pfsp_uniform_floor": 0.1, "prune_winrate_threshold": 0.0,
                   "prune_grace_iters": 0},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        algo.train()   # iter 1: at least one side snapshots v1
        ids1 = set(algo.env_runner.module.keys())
        assert population_members(ids1, RED_POP_RE) or \
            population_members(ids1, BLUE_POP_RE), f"no snapshot after iter 1: {ids1}"

        result2 = algo.train()   # iter 2: at cap -> v2 added, v1 marked pending
        ids2 = set(algo.env_runner.module.keys())
        capped = [(regex, population_members(ids2, regex))
                  for regex in (RED_POP_RE, BLUE_POP_RE)
                  if len(population_members(ids2, regex)) >= 2]
        assert capped, f"expected a side holding v1 (pending) + v2: {ids2}"
        regex, members = capped[0]
        victim, survivor = members[0], members[-1]
        league = result2.get("league", {})
        assert any(k.startswith("pfsp_p_") for k in league), league
        assert league.get("pending_removals", 0) >= 1

        algo.train()   # iter 3: grace (0) elapsed -> victim physically removed
        ids3 = set(algo.env_runner.module.keys())
        assert victim not in ids3, f"{victim} not removed: {ids3}"
        assert survivor in ids3

        ckpt = algo.save(str(tmp_path / "ckpt")).checkpoint.path
    finally:
        algo.stop()

    from ray.rllib.algorithms.algorithm import Algorithm
    algo2 = Algorithm.from_checkpoint(ckpt)
    try:
        ids4 = set(algo2.env_runner.module.keys())
        assert survivor in ids4, "surviving member missing after restore"
        assert victim not in ids4, "pruned member resurrected by restore"
    finally:
        algo2.stop()
```

- [x] **Step 2: Run it**

Run: `uv run pytest tests/rllib/test_smoke_train.py::test_league_pfsp_prunes_and_restores -v`
Expected: PASS (takes a few minutes — three training iterations + a restore). If `remove_module` misbehaves on this Ray version, this test is the gate (see Risks).

- [x] **Step 3: Run the Step-3 slow tests to confirm no regression**

Run: `uv run pytest tests/rllib/test_smoke_train.py -v`
Expected: ALL PASS (4 tests).

- [x] **Step 4: Commit**

```bash
git add tests/rllib/test_smoke_train.py
git commit -m "test(rllib): integration smoke — PFSP league prunes and restores"
```

### Task 11: Full verification + brain update

- [x] **Step 1: Full fast suite**

Run from the worktree root: `make test-fast`
Expected: ALL PASS (the 5 macOS-render failures only appear in non-GUI shells and are environmental).

- [ ] **Step 2: Update the brain** (lives at the umbrella level, `../../../brain/` from the worktree — NOT committed to the repo)

- `brain/index.md` — "Current State": Step 4 (PFSP + pruning) implemented on `feature/rllib-league-step-4`; "Active Priorities" #1: next is a validation league run, then Step 5 (curriculum + promotion).
- `brain/changelog.md` — new `[2026-06-10]` entry (or extend): files touched, the D1–D8 design decisions in brief, test counts.
- `brain/tasks.md` — mark Step 4 implementation done; add "validation league run with PFSP" + "Step 5 plan".
- `brain/decisions.md` — record D1 (in-training winrate estimator, no state file), D5/D6 (prune-at-cap, two-phase removal), D7 (version floor).

- [ ] **Step 3: Hand off for verification before merge**

Per project convention, **do not merge to develop yet**: the behavioral gate for Step 4 is a real league run showing PFSP weights shifting and a prune firing (user-confirmed), mirroring how Steps 1–3 were gated. Offer the user a validation run via `make train EXP=rllib_league` and stop.

---

## Risks

- **`Algorithm.remove_module` mid-training** is the least-exercised RLlib path here (Step 3's analogue was `add_module`). The two-phase grace design sidesteps the known in-flight-episode hazard; the Task-10 integration test is the gate. Fallback if removal proves broken on this Ray version: keep victims pending forever (mapping-fn exclusion only — sampling-correct, memory bounded by run length, not by cap) and file the physical removal as a follow-up.
- **Winrate estimates are training-distribution biased** (PFSP concentrates episodes on hard opponents, so easy opponents' estimates go stale). The uniform floor guarantees a trickle of episodes against every member, refreshing all estimates. This is the standard PFSP estimator; the dedicated eval battery (Step 5) will replace it where precision matters (promotion).
- **`result` mutation in `on_train_result`** reaching Tune/W&B relies on RLlib calling the hook before returning the result dict (true on the current new API stack; the Task-10 assertion on `result2["league"]` guards it).

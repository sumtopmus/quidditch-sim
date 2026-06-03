"""Eval engine: run a learner through N episodes of one scenario.

Used by:
  - scripts/eval_team.py (Hydra entrypoint, Task 10)
  - scripts/eval_battery.py (new Hydra entrypoint, Task 13)

run_scenario() is the single entry point.  Pure function (no print, no
sys.exit); returns a ScenarioResult with per-episode detail + aggregates.

Implementation notes:
  - Uses envs.quidditch.team_env.QuidditchTeamEnv directly.
  - For non-scripted learners, loads the PPO via PPO.load(learner_uri); the
    obs spec + n_stack are read from the learner's .hydra/config.yaml via
    core.run_context.load_run_context.
  - learner_id + learner_spec are first-class config fields; team_env builds
    the per-agent obs natively.  Frame-stacking via FrameStackWrapper when
    the learner's recorded n_stack > 1.
  - Crash-aftermath plumbed through cfg.crash_aftermath_seconds.
  - For `learner_uri='scripted:<spec>'` learners, uses envs.quidditch.opponents.
    from_spec(<spec>) as the policy (tests-only escape hatch — real callers
    pass model paths).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

TERMINAL_BUCKETS = (
    "drone_drone_crash",
    "red_floor", "blue_floor",
    "red_wall",  "blue_wall",
    "red_oob",   "blue_oob",
    "timeout",
    "score",
)


@dataclass(frozen=True)
class ScenarioSpec:
    opponent: str                  # "beeline_red" / "intercepter_red:lookahead=0.5" / "frozen" / ...
    opponent_model_path: str | None  # required when opponent == "frozen"
    randomise_start: bool
    n_episodes: int
    crash_aftermath_seconds: float = 0.0
    deterministic: bool = True
    learner_id: str = "blue_0"
    seed: int = 0


@dataclass(frozen=True)
class EpisodeResult:
    length: int
    reward_learner: float
    reward_opponent: float
    terminal_cause: str            # one of TERMINAL_BUCKETS
    take_down_fired: bool
    score_at_episode_end: int | None  # learner side scored (1), opponent side (-1), neither (None)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: ScenarioSpec
    episodes: list[EpisodeResult]
    win_rate: float
    mean_reward_learner: float
    mean_reward_opponent: float
    take_down_rate: float
    terminal_cause_counts: dict[str, int]
    mean_episode_length: float


def run_scenario(
    learner_uri: str,
    scenario: ScenarioSpec,
    *,
    render: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> ScenarioResult:
    """Run `scenario.n_episodes` episodes; aggregate into a ScenarioResult.

    `learner_uri` may be:
      - filesystem path to a best_model.zip (with or without .zip suffix)
      - wandb:// URI (resolves via scripts._artifact_io.resolve_parent)
      - "scripted:<spec>" for tests (e.g. "scripted:beeline_blue")
    """
    from envs.quidditch.obs_spec import build_spec_from_block_names
    from envs.quidditch.opponents import (
        FrameStackWrapper, OpponentControlledEnv, from_spec,
    )
    from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

    is_scripted_learner = learner_uri.startswith("scripted:")
    learner_blocks: list[str] = []
    learner_spec_name: str | None = None
    learner_n_stack: int = 1

    if not is_scripted_learner:
        from core.run_context import load_run_context
        from scripts._artifact_io import resolve_parent
        # Resolve weights path; load_run_context wants the run dir, not the .zip.
        model_path = resolve_parent(learner_uri)
        model_path_p = Path(str(model_path))
        if model_path_p.is_file():
            run_dir = model_path_p.parent
        else:
            run_dir = model_path_p
        ctx = load_run_context(run_dir)
        cfg = ctx["cfg"]
        obs = cfg.get("obs") if hasattr(cfg, "get") else None
        if obs is not None and hasattr(obs, "get"):
            learner_spec_name = str(obs.get("name", "DUEL_V1_BODY"))
            learner_n_stack = int(obs.get("n_stack", 1))
            learner_blocks = list(obs.get("blocks") or [])

    if learner_blocks:
        learner_spec = build_spec_from_block_names(learner_blocks)
    elif learner_spec_name:
        # Pre-2026-05-18 schema: only obs.name was recorded.  Look up the
        # blocks via conf/obs/*.yaml by matching the name field.
        from core.obs_compat import _spec_by_name
        learner_spec = _spec_by_name(learner_spec_name)
    else:
        from core.obs_compat import _spec_by_name
        learner_spec = _spec_by_name("DUEL_V1_BODY")

    team_cfg = TeamConfig(
        randomise_red_start=scenario.randomise_start,
        crash_aftermath_seconds=scenario.crash_aftermath_seconds,
    )
    render_mode = "human" if render else None
    team_env = QuidditchTeamEnv(
        cfg=team_cfg, render_mode=render_mode,
        learner_id=scenario.learner_id,
        learner_spec=learner_spec,
    )

    # Build the opponent.  scenario.opponent == "frozen" requires opponent_model_path;
    # any other spec passes through from_spec.
    if scenario.opponent == "frozen":
        if scenario.opponent_model_path is None:
            raise ValueError("scenario.opponent == 'frozen' requires opponent_model_path")
        opp = from_spec(f"frozen:{scenario.opponent_model_path}",
                        deterministic=scenario.deterministic)
    else:
        opp = from_spec(scenario.opponent, deterministic=scenario.deterministic)

    # Build the learner side.  Two paths: scripted (tests) vs PPO (real).
    if is_scripted_learner:
        learner_spec_str = learner_uri[len("scripted:"):]
        oce = OpponentControlledEnv(team_env, learner_id=scenario.learner_id, opponent=opp)
        env = oce  # no frame-stacking for scripted learners
        scripted_policy = from_spec(learner_spec_str, deterministic=scenario.deterministic)
        model = None
    else:
        from stable_baselines3 import PPO
        oce = OpponentControlledEnv(team_env, learner_id=scenario.learner_id, opponent=opp)
        env = (FrameStackWrapper(oce, n_stack=learner_n_stack)
               if learner_n_stack > 1 else oce)
        model = PPO.load(str(model_path))
        scripted_policy = None

    rng = np.random.default_rng(scenario.seed)
    episodes: list[EpisodeResult] = []
    bucket_counter: Counter[str] = Counter()
    take_down_count = 0

    for ep_idx in range(scenario.n_episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=seed)
        ep_len = 0
        rew_learner = 0.0
        rew_opp = 0.0
        take_down_fired = False
        terminal_cause = "timeout"
        score_at_end: int | None = None

        while True:
            if model is not None:
                action, _ = model.predict(obs, deterministic=scenario.deterministic)
            else:
                action = scripted_policy.act(obs)
            obs, r, term, trunc, info = env.step(action)
            ep_len += 1
            rew_learner += float(r)

            team_infos = oce.last_team_infos
            red_info = team_infos.get("red_0", {}) or {}
            blue_info = team_infos.get("blue_0", {}) or {}
            # Opponent reward: not reliably available in OCE single-agent mode.
            # Leave as 0 — the per-episode mean carries the learner side only.

            if red_info.get("take_down_fired") or blue_info.get("take_down_fired"):
                take_down_fired = True

            if term or trunc:
                terminal_cause = _classify_terminal(red_info, blue_info)
                if terminal_cause == "score":
                    if blue_info.get("scored"):
                        score_at_end = 1 if scenario.learner_id == "blue_0" else -1
                    elif red_info.get("scored"):
                        score_at_end = -1 if scenario.learner_id == "blue_0" else 1
                break

        if take_down_fired:
            take_down_count += 1
        bucket_counter[terminal_cause] += 1

        episodes.append(EpisodeResult(
            length=ep_len,
            reward_learner=rew_learner,
            reward_opponent=rew_opp,
            terminal_cause=terminal_cause,
            take_down_fired=take_down_fired,
            score_at_episode_end=score_at_end,
        ))
        if progress_cb is not None:
            progress_cb(ep_idx + 1, scenario.n_episodes)

    n = max(len(episodes), 1)
    learner_side_scored = sum(1 for e in episodes if e.score_at_episode_end == 1)
    win_rate = learner_side_scored / n
    take_down_rate = take_down_count / n
    mean_R_l = float(np.mean([e.reward_learner for e in episodes])) if episodes else 0.0
    mean_R_o = float(np.mean([e.reward_opponent for e in episodes])) if episodes else 0.0
    mean_ep_len = float(np.mean([e.length for e in episodes])) if episodes else 0.0

    team_env.close()

    return ScenarioResult(
        scenario=scenario,
        episodes=episodes,
        win_rate=win_rate,
        mean_reward_learner=mean_R_l,
        mean_reward_opponent=mean_R_o,
        take_down_rate=take_down_rate,
        terminal_cause_counts=dict(bucket_counter),
        mean_episode_length=mean_ep_len,
    )


def _classify_terminal(red_info: dict, blue_info: dict) -> str:
    """Map a terminal step's two info dicts to a single TERMINAL_BUCKETS bucket."""
    if red_info.get("scored") or blue_info.get("scored"):
        return "score"
    if red_info.get("drone_drone_crash") or blue_info.get("drone_drone_crash"):
        return "drone_drone_crash"
    if red_info.get("red_floor"):       return "red_floor"
    if red_info.get("red_wall_crash"):  return "red_wall"
    if red_info.get("red_oob"):         return "red_oob"
    if blue_info.get("blue_floor"):     return "blue_floor"
    if blue_info.get("blue_wall_crash"):return "blue_wall"
    if blue_info.get("blue_oob"):       return "blue_oob"
    return "timeout"

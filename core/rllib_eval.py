"""RLlib head-to-head eval (Step 5c).

Loads main_red + main_blue from a checkpoint dir and runs the Step-5b eval
battery head-to-head on the team env. Used by scripts/eval_team.py's RLlib
branch. The env is rebuilt from the checkpoint's recorded .hydra/config.yaml so
obs blocks / team config match training; the reward stack is irrelevant to the
battery's outcome metrics, so eval lets the env build its default team stack.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def _env_config_from_run(run_dir: Path) -> dict:
    """env_config for make_team_env, read from the run's .hydra/config.yaml."""
    cfg = OmegaConf.load(Path(run_dir) / ".hydra" / "config.yaml")
    obs = cfg.get("obs") or {}
    ma = cfg.get("multiagent") or {}
    cur = cfg.get("curriculum") or {}
    team_cfg: dict = {}
    if cur:
        team_cfg["randomise_red_start"] = bool(cur.get("randomise_start", True))
        team_cfg["episode_seconds"] = float(cur.get("episode_seconds", 30.0))
        rsp = cur.get("red_start_pos")
        if rsp is not None:
            team_cfg["red_start_pos"] = [float(v) for v in rsp]
    return {
        "learner_id": str(ma.get("learner_id", "red_0")),
        "obs_blocks": list(obs.get("blocks") or []),
        "team_cfg": team_cfg,
        "reward_stack": None,
        "league": None,
    }


def run_rllib_battery(checkpoint_dir: Path, run_dir: Path, *,
                      n_episodes: int, seed: int) -> dict:
    """Load both mains from the checkpoint and run the head-to-head battery."""
    from envs.quidditch.rllib_env import make_team_env
    from core.rllib_checkpoint import load_rl_module
    from rllib.eval_battery import rollout_battery, module_action_fn

    env = make_team_env(_env_config_from_run(run_dir))
    try:
        red = module_action_fn(load_rl_module(checkpoint_dir, "main_red"))
        blue = module_action_fn(load_rl_module(checkpoint_dir, "main_blue"))
        return rollout_battery(env, red, blue, n_episodes=n_episodes, seed=seed)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

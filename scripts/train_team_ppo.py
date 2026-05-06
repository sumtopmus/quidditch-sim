"""Training entry for QuidditchTeamEnv.  Wraps the team env in an
OpponentControlledEnv so SB3 PPO sees a single-agent Gym env.

Run:
    conda activate uav
    python scripts/train_team_ppo.py --learner red_0 --opponent beeline_blue
    python scripts/train_team_ppo.py --learner blue_0 --opponent frozen:models/.../best_model.zip
    python scripts/train_team_ppo.py --learner red_0 --opponent beeline_blue \
        --warm-start models/ppo_hoop_rand_start_20260505_174509/best_model.zip
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.opponents import OpponentControlledEnv, from_spec
from scripts._train_common import (
    make_run_dir, build_callbacks, load_config, write_run_info,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QuidditchTeamEnv via SB3 PPO")
    p.add_argument("--learner", choices=["red_0", "blue_0"], required=True)
    p.add_argument("--opponent", required=True, help="opponent spec, e.g. 'beeline_blue'")
    p.add_argument("--warm-start", default="", help="path to old single-agent best_model.zip")
    p.add_argument("--config", default="config/training.toml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-envs",    type=int, default=None)
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--seed",      type=int, default=None)
    return p.parse_args()


def make_env_fn(*, cfg: TeamConfig, learner_id: str, opponent_spec: str):
    def _thunk():
        team = QuidditchTeamEnv(cfg=cfg)
        opp = from_spec(opponent_spec)
        return OpponentControlledEnv(team, learner_id=learner_id, opponent=opp)
    return _thunk


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_name = args.run_name or config["training"].get("run_name", "team_red")
    run_dir = make_run_dir(run_name=run_name, runs_root="runs")

    n_envs = args.n_envs    or config["training"].get("n_envs", 8)
    seed   = args.seed      if args.seed is not None else config["training"].get("seed", 42)

    env_team_cfg = config.get("env", {}).get("team", {})
    cfg = TeamConfig(
        red_prefix       = env_team_cfg.get("red_prefix",       "red_0"),
        blue_prefix      = env_team_cfg.get("blue_prefix",      "blue_0"),
        hoop_prefix      = env_team_cfg.get("hoop_prefix",      "hoop_0"),
        midpoint_alpha   = env_team_cfg.get("midpoint_alpha",   0.5),
        tag_radius       = env_team_cfg.get("tag_radius",       0.3),
        tag_cooldown_s   = env_team_cfg.get("tag_cooldown_s",   1.0),
        crash_vel_thr    = env_team_cfg.get("crash_vel_thr",    1.5),
        walls_collide    = env_team_cfg.get("walls_collide",    True),
        randomise_red_start = config.get("env", {}).get("randomise_start", True),
        episode_seconds     = config.get("env", {}).get("episode_seconds", 30.0),
    )

    env_fns = [
        make_env_fn(cfg=cfg, learner_id=args.learner, opponent_spec=args.opponent)
        for _ in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    ppo_kwargs: dict = {
        k: v for k, v in config["training"].get("ppo", {}).items()
    }
    if args.lr is not None:
        ppo_kwargs["learning_rate"] = args.lr
    elif "lr" in ppo_kwargs:
        ppo_kwargs["learning_rate"] = ppo_kwargs.pop("lr")
    ppo_kwargs.setdefault("learning_rate", 3e-4)

    if args.warm_start:
        from core.policies.warm_start import warm_start_ppo
        ws_cfg = config.get("training", {}).get("team", {}).get("warm_start", {})
        model = warm_start_ppo(
            old_checkpoint=args.warm_start,
            new_env=vec_env,
            new_input_dim=22, old_input_dim=16,
            new_dim_init_scale=ws_cfg.get("new_dim_init_scale", 0.01),
            tensorboard_log=str(run_dir),
            seed=seed,
            verbose=1,
            **ppo_kwargs,
        )
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            tensorboard_log=str(run_dir),
            seed=seed,
            verbose=1,
            **ppo_kwargs,
        )

    callbacks = build_callbacks(
        run_dir=run_dir,
        eval_env_fn=make_env_fn(cfg=cfg, learner_id=args.learner,
                                 opponent_spec=args.opponent),
        config=config,
        n_envs=n_envs,
    )

    total_timesteps = args.timesteps or config["training"].get("total_timesteps", 5_000_000)

    started = datetime.now()
    write_run_info(
        run_dir, config=config, args=args,
        extra={"learner": args.learner, "opponent_spec": args.opponent,
               "warm_start_from": args.warm_start},
        started=started,
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    progress_bar=True)
    finally:
        model.save(str(run_dir / "final_model"))
        elapsed_s = (datetime.now() - started).total_seconds()
        write_run_info(
            run_dir, config=config, args=args,
            extra={"learner": args.learner, "opponent_spec": args.opponent,
                   "warm_start_from": args.warm_start},
            started=started, elapsed_s=elapsed_s,
            steps_trained=int(model.num_timesteps),
        )


if __name__ == "__main__":
    main()

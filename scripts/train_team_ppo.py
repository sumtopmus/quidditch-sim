"""Training entry for QuidditchTeamEnv.  Wraps the team env in an
OpponentControlledEnv so SB3 PPO sees a single-agent Gym env.

Run modes:
    # Cold start
    python scripts/train_team_ppo.py --learner red_0 --opponent beeline_blue

    # Cold start, frozen-policy opponent
    python scripts/train_team_ppo.py --learner blue_0 \\
        --opponent frozen:models/.../best_model.zip

    # Warm-start from a single-agent checkpoint (16->22 input-layer surgery)
    python scripts/train_team_ppo.py --learner red_0 --opponent beeline_blue \\
        --warm-start models/ppo_hoop_rand_start_20260505_174509/best_model.zip

    # Resume an existing team-play trial from its latest checkpoint.  --learner
    # and --opponent are read from the parent trial's info.toml [extra] block
    # if not given on the CLI.  Config's lr overrides the checkpoint's lr.
    python scripts/train_team_ppo.py \\
        --resume runs/ppo_hoop_blue_1/20260507_194423/checkpoints/ppo_10000000_steps.zip
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# macOS conda ships multiple copies of libomp; suppress the duplicate-init abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.opponents import OpponentControlledEnv, from_spec
from scripts._train_common import (
    make_run_dir, build_callbacks, load_config, write_run_info,
)
from scripts.callbacks import ResumeProgressCallback

import tomllib


def _read_parent_extra(checkpoint_path: str) -> tuple[str | None, str | None]:
    """Walk from a checkpoint path up to its trial dir and read learner/opponent
    from info.toml [extra]. Returns (None, None) if anything is missing."""
    trial_dir = Path(checkpoint_path).resolve().parent.parent
    info = trial_dir / "info.toml"
    if not info.exists():
        return None, None
    try:
        data = tomllib.loads(info.read_text())
    except Exception:
        return None, None
    extra = data.get("extra", {})
    return extra.get("learner"), extra.get("opponent_spec")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QuidditchTeamEnv via SB3 PPO")
    p.add_argument("--learner", choices=["red_0", "blue_0"], required=False, default=None)
    p.add_argument("--opponent", required=False, default=None,
                   help="opponent spec, e.g. 'beeline_blue'")
    p.add_argument("--config", default="config/training.toml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-envs",    type=int, default=None)
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--seed",      type=int, default=None)
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print SB3 training logs instead of showing a rich progress bar.",
    )

    # --warm-start and --resume are mutually exclusive: warm-start does the
    # 16->22 input-layer surgery from a single-agent checkpoint; resume picks
    # up an existing team-play trial mid-flight (same I/O dimensions, keeps
    # optimizer state, keeps step counter).
    g = p.add_mutually_exclusive_group()
    g.add_argument("--warm-start", default="",
                   help="path to old single-agent best_model.zip (input-layer surgery)")
    g.add_argument("--resume", default=None, metavar="PATH",
                   help="path to a checkpoint .zip to resume from. Keeps the step "
                        "counter; trains for the remaining steps to total_timesteps. "
                        "Loads --learner / --opponent from the parent trial's info.toml "
                        "if not given on the CLI.")
    return p.parse_args()


def make_env_fn(*, cfg: TeamConfig, learner_id: str, opponent_spec: str):
    def _thunk():
        team = QuidditchTeamEnv(cfg=cfg)
        opp = from_spec(opponent_spec)
        return OpponentControlledEnv(team, learner_id=learner_id, opponent=opp)
    return _thunk


def main() -> None:
    args = parse_args()

    if args.resume is None:
        if args.learner is None or args.opponent is None:
            print("ERROR: --learner and --opponent are required unless --resume is given",
                  file=sys.stderr)
            sys.exit(2)

    if args.resume is not None:
        if args.learner is None or args.opponent is None:
            parent_learner, parent_opponent = _read_parent_extra(args.resume)
            args.learner = args.learner or parent_learner
            args.opponent = args.opponent or parent_opponent
        if args.learner is None or args.opponent is None:
            print(
                f"ERROR: could not resolve --learner/--opponent from parent of {args.resume}. "
                "Pass them on the CLI.",
                file=sys.stderr,
            )
            sys.exit(2)

    config = load_config(args.config)

    if not args.warm_start:
        ws_from = config.get("training", {}).get("team", {}).get("warm_start_from", "").strip()
        if ws_from:
            args.warm_start = f"{ws_from.rstrip('/')}/best_model"

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
    # Frame-stack so the MLP can derive closing-rate signals across time steps.
    # The eval / video envs in build_callbacks must match this stack depth.
    frame_stack = int(config["training"].get("frame_stack", 1))
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

    ppo_kwargs: dict = {
        k: v for k, v in config["training"].get("ppo", {}).items()
    }
    if args.lr is not None:
        ppo_kwargs["learning_rate"] = args.lr
    elif "lr" in ppo_kwargs:
        ppo_kwargs["learning_rate"] = ppo_kwargs.pop("lr")
    ppo_kwargs.setdefault("learning_rate", 3e-4)

    total_timesteps = args.timesteps or config["training"].get("total_timesteps", 5_000_000)

    verbose = 1 if args.verbose else 0

    resumed_at: int | None = None
    if args.resume is not None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶️  Resuming from {args.resume}")
        # Override learning_rate from config so e.g. lr=1e-4 takes effect on
        # resume.  Other PPO hyperparameters keep their checkpoint values —
        # changing gamma/n_steps/etc. mid-training breaks invariants the value
        # function and rollout buffer rely on.
        model = PPO.load(
            args.resume,
            env=vec_env,
            tensorboard_log=str(run_dir),
            verbose=verbose,
            learning_rate=ppo_kwargs["learning_rate"],
        )
        resumed_at = int(model.num_timesteps)
        remaining = max(total_timesteps - resumed_at, 0)
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] ⏱  Checkpoint at "
            f"{resumed_at:,} steps; {remaining:,} remaining to {total_timesteps:,}"
        )
    elif args.warm_start:
        from core.policies.warm_start import warm_start_ppo
        ws_cfg = config.get("training", {}).get("team", {}).get("warm_start", {})
        model = warm_start_ppo(
            old_checkpoint=args.warm_start,
            new_env=vec_env,
            new_input_dim=22, old_input_dim=16,
            new_dim_init_scale=ws_cfg.get("new_dim_init_scale", 0.01),
            tensorboard_log=str(run_dir),
            seed=seed,
            verbose=verbose,
            **ppo_kwargs,
        )
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            tensorboard_log=str(run_dir),
            seed=seed,
            verbose=verbose,
            **ppo_kwargs,
        )

    # Video env factory: mirrors the eval env but in rgb_array mode so
    # the offscreen renderer produces frames.  build_callbacks only
    # appends the video callback when video_env_fn is provided.
    def _make_video_env():
        team = QuidditchTeamEnv(cfg=cfg, render_mode="rgb_array")
        opp  = from_spec(args.opponent, deterministic=True)
        env = OpponentControlledEnv(
            team, learner_id=args.learner, opponent=opp,
        )
        if frame_stack > 1:
            from envs.quidditch.opponents import FrameStackWrapper
            return FrameStackWrapper(env, n_stack=frame_stack)
        return env

    callbacks = build_callbacks(
        run_dir=run_dir,
        eval_env_fn=make_env_fn(cfg=cfg, learner_id=args.learner,
                                 opponent_spec=args.opponent),
        config=config,
        n_envs=n_envs,
        video_env_fn=_make_video_env,
        frame_stack=frame_stack,
        verbose=verbose,
        total_timesteps=args.timesteps,
        kind="team",
        learner=args.learner,
        opponent_spec=args.opponent,
    )

    resume_info = (
        {"checkpoint": args.resume, "resumed_at": resumed_at}
        if args.resume is not None else None
    )

    started = datetime.now()
    write_run_info(
        run_dir, config=config, args=args,
        extra={"learner": args.learner, "opponent_spec": args.opponent,
               "warm_start_from": args.warm_start},
        resume=resume_info,
        started=started,
    )

    extra_callbacks = (
        [ResumeProgressCallback(total_timesteps)]
        if args.resume is not None and not args.verbose else []
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[*callbacks, *extra_callbacks],
            reset_num_timesteps=args.resume is None,
            progress_bar=not args.verbose and args.resume is None,
        )
    finally:
        model.save(str(run_dir / "final_model"))
        elapsed_s = (datetime.now() - started).total_seconds()
        write_run_info(
            run_dir, config=config, args=args,
            extra={"learner": args.learner, "opponent_spec": args.opponent,
                   "warm_start_from": args.warm_start},
            resume=resume_info,
            started=started, elapsed_s=elapsed_s,
            steps_trained=int(model.num_timesteps),
        )


if __name__ == "__main__":
    main()

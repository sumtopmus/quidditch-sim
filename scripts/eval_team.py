"""Head-to-head eval: drives QuidditchTeamEnv with two Opponent specs.

Outputs match summary: red score rate, mean tag count, mean tag duration,
drone-drone crash rate, mean episode length.

Two modes:
    1. Symmetric (default) — both sides driven via `from_spec(...)` opponents.
       Works for scripted-vs-scripted and for frozen checkpoints whose policy
       was trained on the raw 22-d team_env obs (e.g. red_v1).

    2. Learner-mode (`--learner red_0|blue_0` + `--learner-frame-stack N`) —
       routes the learner side through `OpponentControlledEnv` + optional
       `FrameStackWrapper` so it sees the exact 25-d augmented + N-stacked
       obs it was trained on (e.g. blue_v4 with frame_stack=3).  The matching
       --red / --blue arg must be a `frozen:...` spec; the other side becomes
       the scripted opponent inside OCE.

Run:
    conda activate uav
    python scripts/eval_team.py --red beeline_red --blue beeline_blue --episodes 100
    python scripts/eval_team.py --red beeline_red \
        --blue frozen:models/ppo_hoop_blue_4_20260511_202612/best_model \
        --learner blue_0 --learner-frame-stack 3 --gui
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# macOS conda ships multiple copies of libomp; suppress the duplicate-init abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.opponents import (
    FrameStackWrapper,
    OpponentControlledEnv,
    from_spec,
)


LEARNER_CHOICES = ("red_0", "blue_0")


def _classify_terminal(red_info: dict, blue_info: dict) -> str:
    """Map a terminal step's two info dicts to a single bucket name."""
    if red_info.get("scored") or blue_info.get("scored"):
        return "score"
    if red_info.get("drone_drone_crash") or blue_info.get("drone_drone_crash"):
        return "drone_drone"
    if red_info.get("red_floor"):       return "red_floor"
    if red_info.get("red_wall_crash"):  return "red_wall"
    if red_info.get("red_oob"):         return "red_oob"
    if blue_info.get("blue_floor"):     return "blue_floor"
    if blue_info.get("blue_wall_crash"):return "blue_wall"
    if blue_info.get("blue_oob"):       return "blue_oob"
    return "timeout"


def _run_symmetric(env: QuidditchTeamEnv, red_opp, blue_opp, args, rng) -> dict:
    """Both sides driven by `from_spec(...)` opponents; raw 22-d obs to both."""
    summary: Counter = Counter()
    tag_count_per_ep: list[int] = []
    tag_steps_per_ep: list[int] = []
    ep_lengths:        list[int] = []
    red_totals:        list[float] = []
    blue_totals:       list[float] = []

    for _ep in range(args.episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=seed)
        red_opp.reset(); blue_opp.reset()
        tag_count = 0; tag_steps = 0
        red_total = 0.0; blue_total = 0.0
        step = 0
        info: dict = {"red_0": {}, "blue_0": {}}
        while env.agents:
            actions = {"red_0": red_opp.act(obs["red_0"]),
                       "blue_0": blue_opp.act(obs["blue_0"])}
            obs, rew, term, trunc, info = env.step(actions)
            step += 1
            red_total  += rew["red_0"]
            blue_total += rew["blue_0"]
            if info["red_0"].get("tag_entry"):  tag_count += 1
            if info["red_0"].get("tag_during"): tag_steps += 1
            if any(term.values()) or any(trunc.values()):
                summary[_classify_terminal(info["red_0"], info["blue_0"])] += 1
                break
        ep_lengths.append(step)
        tag_count_per_ep.append(tag_count)
        tag_steps_per_ep.append(tag_steps)
        red_totals.append(red_total)
        blue_totals.append(blue_total)

    return {
        "summary": summary, "tag_count_per_ep": tag_count_per_ep,
        "tag_steps_per_ep": tag_steps_per_ep, "ep_lengths": ep_lengths,
        "red_totals": red_totals, "blue_totals": blue_totals,
    }


def _run_learner(
    team_env: QuidditchTeamEnv,
    learner_id: str,
    learner_model_path: str,
    opp,
    args,
    rng,
) -> dict:
    """Learner side runs through OCE+FrameStack so it sees its training obs."""
    from stable_baselines3 import PPO

    oce = OpponentControlledEnv(team_env, learner_id=learner_id, opponent=opp)
    env = FrameStackWrapper(oce, n_stack=args.learner_frame_stack) \
        if args.learner_frame_stack > 1 else oce

    model = PPO.load(learner_model_path)

    summary: Counter = Counter()
    tag_count_per_ep: list[int] = []
    tag_steps_per_ep: list[int] = []
    ep_lengths:        list[int] = []
    red_totals:        list[float] = []
    blue_totals:       list[float] = []

    for _ep in range(args.episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=seed)
        # opp.reset() runs inside OCE.reset(); team_env reset there too.
        tag_count = 0; tag_steps = 0
        red_total = 0.0; blue_total = 0.0
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _learner_rew, term, trunc, info_learner = env.step(action)
            step += 1
            # Both teams' info dicts (with role-specific keys) come from OCE.
            team_infos = oce.last_team_infos
            red_info  = team_infos.get("red_0",  {})
            blue_info = team_infos.get("blue_0", {})
            # Tag flags live in both info dicts identically — read either.
            if red_info.get("tag_entry"):  tag_count += 1
            if red_info.get("tag_during"): tag_steps += 1
            # Per-step rewards aren't returned by OCE for both sides; we
            # don't have a cheap way to recover Red's reward in learner mode,
            # so the per-episode totals reflect only the learner.
            red_total  += 0.0 if learner_id == "blue_0" else float(_learner_rew)
            blue_total += float(_learner_rew) if learner_id == "blue_0" else 0.0
            if term or trunc:
                summary[_classify_terminal(red_info, blue_info)] += 1
                break
        ep_lengths.append(step)
        tag_count_per_ep.append(tag_count)
        tag_steps_per_ep.append(tag_steps)
        red_totals.append(red_total)
        blue_totals.append(blue_total)

    return {
        "summary": summary, "tag_count_per_ep": tag_count_per_ep,
        "tag_steps_per_ep": tag_steps_per_ep, "ep_lengths": ep_lengths,
        "red_totals": red_totals, "blue_totals": blue_totals,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--red",  required=True)
    p.add_argument("--blue", required=True)
    p.add_argument("--episodes", type=int, default=None,
                   help="default: 100 headless / 5 with --gui")
    p.add_argument("--episode-seconds", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gui", action="store_true",
                   help="open MuJoCo passive viewer; idle after last episode")
    p.add_argument("--deterministic", action="store_const", const=True, default=None,
                   help="pass deterministic=True to all frozen: opponents "
                        "(default: True with --gui, False otherwise)")
    p.add_argument("--crash-aftermath-seconds", type=float, default=None,
                   help="extra seconds with Red's motors cut after a drone-drone ram, "
                        "so the crash stays on screen (default: 1.0 with --gui, 0.0 otherwise)")
    p.add_argument("--learner", choices=LEARNER_CHOICES, default=None,
                   help="if set, route this side through OpponentControlledEnv + "
                        "FrameStackWrapper so it sees the 25-d augmented obs it was "
                        "trained on; the matching --red/--blue arg must be frozen:...")
    p.add_argument("--learner-frame-stack", type=int, default=1,
                   help="n_stack for the learner's obs (must match training; e.g. blue_v4 → 3)")
    p.add_argument("--randomise-start", action="store_true",
                   help="random uniform Red start (default: fixed at origin)")
    args = p.parse_args()

    if args.episodes is None:
        args.episodes = 5 if args.gui else 100
    if args.deterministic is None:
        args.deterministic = bool(args.gui)
    if args.crash_aftermath_seconds is None:
        args.crash_aftermath_seconds = 1.0 if args.gui else 0.0

    cfg = TeamConfig(
        randomise_red_start=args.randomise_start,
        episode_seconds=args.episode_seconds,
        crash_aftermath_seconds=args.crash_aftermath_seconds,
    )
    render_mode = "human" if args.gui else None
    team_env = QuidditchTeamEnv(cfg=cfg, render_mode=render_mode)
    rng = np.random.default_rng(args.seed)

    if args.learner is None:
        red_opp  = from_spec(args.red,  deterministic=args.deterministic)
        blue_opp = from_spec(args.blue, deterministic=args.deterministic)
        result = _run_symmetric(team_env, red_opp, blue_opp, args, rng)
    else:
        learner_spec = args.blue if args.learner == "blue_0" else args.red
        opp_spec     = args.red  if args.learner == "blue_0" else args.blue
        if not learner_spec.startswith("frozen:"):
            raise SystemExit(
                f"--learner {args.learner} requires --{args.learner[:-2]} to be a "
                f"'frozen:path' spec; got {learner_spec!r}"
            )
        learner_model_path = learner_spec[len("frozen:"):]
        opp = from_spec(opp_spec, deterministic=args.deterministic)
        result = _run_learner(team_env, args.learner, learner_model_path, opp, args, rng)

    n = args.episodes
    summary = result["summary"]
    print(f"\n=== {n} episodes  red={args.red}  blue={args.blue}"
          f"{f'  learner={args.learner}/fs={args.learner_frame_stack}' if args.learner else ''} ===")
    print(f"score rate (Red):       {summary['score']/n:>5.1%}")
    print(f"drone-drone crashes:    {summary['drone_drone']/n:>5.1%}")
    print(f"red floor / wall / OOB: "
          f"{summary['red_floor']/n:>5.1%} / "
          f"{summary['red_wall']/n:>5.1%} / "
          f"{summary['red_oob']/n:>5.1%}")
    print(f"blue floor / wall / OOB:"
          f"{summary['blue_floor']/n:>5.1%} / "
          f"{summary['blue_wall']/n:>5.1%} / "
          f"{summary['blue_oob']/n:>5.1%}")
    print(f"timeouts:               {summary['timeout']/n:>5.1%}")
    print(f"mean episode length:    {np.mean(result['ep_lengths']):.1f} steps")
    print(f"mean tag count/ep:      {np.mean(result['tag_count_per_ep']):.2f}")
    print(f"mean tag-steps/ep:      {np.mean(result['tag_steps_per_ep']):.2f}")
    print(f"red reward mean ± std:  {np.mean(result['red_totals']):+.3f} ± {np.std(result['red_totals']):.3f}")
    print(f"blue reward mean ± std: {np.mean(result['blue_totals']):+.3f} ± {np.std(result['blue_totals']):.3f}")

    if args.gui:
        team_env._world.idle()
    team_env.close()


if __name__ == "__main__":
    main()

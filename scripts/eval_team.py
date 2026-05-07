"""Head-to-head eval: drives QuidditchTeamEnv with two Opponent specs.

Outputs match summary: red score rate, mean tag count, mean tag duration,
drone-drone crash rate, mean episode length.

Run:
    conda activate uav
    python scripts/eval_team.py --red beeline_red --blue beeline_blue --episodes 100
    python scripts/eval_team.py --red frozen:models/red_v2/best_model \
                                --blue frozen:models/blue_v1/best_model --episodes 50
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
from envs.quidditch.opponents import from_spec


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
    args = p.parse_args()

    if args.episodes is None:
        args.episodes = 5 if args.gui else 100
    if args.deterministic is None:
        args.deterministic = bool(args.gui)

    red_opp  = from_spec(args.red,  deterministic=args.deterministic)
    blue_opp = from_spec(args.blue, deterministic=args.deterministic)

    cfg = TeamConfig(randomise_red_start=True, episode_seconds=args.episode_seconds)
    render_mode = "human" if args.gui else None
    env = QuidditchTeamEnv(cfg=cfg, render_mode=render_mode)

    rng = np.random.default_rng(args.seed)
    summary: Counter = Counter()
    tag_count_per_ep: list[int] = []
    tag_steps_per_ep: list[int] = []
    ep_lengths:        list[int] = []
    red_totals:        list[float] = []
    blue_totals:       list[float] = []

    for _ep in range(args.episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=seed)
        red_opp.reset()
        blue_opp.reset()
        tag_count = 0
        tag_steps = 0
        red_total = 0.0
        blue_total = 0.0
        step = 0
        info: dict = {"red_0": {}, "blue_0": {}}
        while env.agents:
            actions = {"red_0": red_opp.act(obs["red_0"]),
                       "blue_0": blue_opp.act(obs["blue_0"])}
            obs, rew, term, trunc, info = env.step(actions)
            step += 1
            red_total  += rew["red_0"]
            blue_total += rew["blue_0"]
            if info["red_0"].get("tag_entry"):
                tag_count += 1
            if info["red_0"].get("tag_during"):
                tag_steps += 1
            if any(term.values()) or any(trunc.values()):
                if info["red_0"].get("scored"):                  summary["score"] += 1
                elif info["red_0"].get("drone_drone_crash"):     summary["drone_drone"] += 1
                elif info["red_0"].get("red_floor"):             summary["red_floor"] += 1
                elif info["red_0"].get("red_wall_crash"):        summary["red_wall"] += 1
                elif info["red_0"].get("red_oob"):               summary["red_oob"] += 1
                elif info["blue_0"].get("blue_floor"):           summary["blue_floor"] += 1
                elif info["blue_0"].get("blue_wall_crash"):      summary["blue_wall"] += 1
                elif info["blue_0"].get("blue_oob"):             summary["blue_oob"] += 1
                else:                                            summary["timeout"] += 1
                break
        ep_lengths.append(step)
        tag_count_per_ep.append(tag_count)
        tag_steps_per_ep.append(tag_steps)
        red_totals.append(red_total)
        blue_totals.append(blue_total)

    n = args.episodes
    print(f"\n=== {n} episodes  red={args.red}  blue={args.blue} ===")
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
    print(f"mean episode length:    {np.mean(ep_lengths):.1f} steps")
    print(f"mean tag count/ep:      {np.mean(tag_count_per_ep):.2f}")
    print(f"mean tag-steps/ep:      {np.mean(tag_steps_per_ep):.2f}")
    print(f"red reward mean ± std:  {np.mean(red_totals):+.3f} ± {np.std(red_totals):.3f}")
    print(f"blue reward mean ± std: {np.mean(blue_totals):+.3f} ± {np.std(blue_totals):.3f}")

    if args.gui:
        env._world.idle()
    env.close()


if __name__ == "__main__":
    main()

"""Asserts warm-start preserves the old policy's behavior on obs[:16]
when obs[16:] is zeroed out.

The augmented input columns are initialized small (default σ=0.01), so
the new policy's logits should be very close to the old policy's on
the same obs[:16].  We test action-distribution similarity on a fixed
batch of obs vectors.

Run:
    conda activate uav
    python scripts/check_team_warm.py --old models/<run>/best_model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.opponents import OpponentControlledEnv, BeelineBlue
from core.policies.warm_start import warm_start_ppo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True, help="path to old single-agent best_model")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--tol", type=float, default=0.1)
    args = p.parse_args()

    old = PPO.load(args.old)

    def _env_fn():
        return OpponentControlledEnv(
            QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False)),
            learner_id="red_0", opponent=BeelineBlue(),
        )
    new_env = DummyVecEnv([_env_fn])
    new = warm_start_ppo(
        old_checkpoint=args.old,
        new_env=new_env,
        new_input_dim=22, old_input_dim=16,
        new_dim_init_scale=0.01,
    )

    rng = np.random.default_rng(42)
    obs_seed = new_env.reset()
    diffs: list[float] = []
    for _ in range(args.n_samples):
        obs_zeroed = obs_seed.copy()
        obs_zeroed[..., 16:] = 0.0
        action,     _ = new.predict(obs_zeroed,         deterministic=True)
        old_action, _ = old.predict(obs_seed[..., :16], deterministic=True)
        diffs.append(float(np.linalg.norm(action - old_action)))
        random_a = rng.uniform(-1, 1, size=action.shape).astype(np.float32)
        obs_seed, _, _, _ = new_env.step(random_a)

    mean_diff = float(np.mean(diffs))
    max_diff  = float(np.max(diffs))
    print(f"OK warm_start: mean ||a_new − a_old|| = {mean_diff:.4f}, max = {max_diff:.4f}")
    assert mean_diff < args.tol, f"warm-started policy drifts {mean_diff:.4f} > tol {args.tol}"
    new_env.close()


if __name__ == "__main__":
    main()

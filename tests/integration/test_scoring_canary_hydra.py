"""Scoring canary, Hydra-composed.  Phase 4 — runs in parallel with the
existing test_scoring_canary.py.  Phase 5 replaces it.

Locked fingerprint: SCORED at step 434 / total reward 7.3837.
"""
from __future__ import annotations

import numpy as np
from hydra.utils import instantiate

from envs.quidditch.constants import HOOP_CENTER, HOOP_OUTWARD_NORMAL
from envs.quidditch.simple_env import QuidditchSimpleEnv
from tests.conftest import hydra_compose


def test_scripted_flyaway_scores_through_hoop_hydra() -> None:
    """Same scripted flight as test_scoring_canary.py, but reward stack is
    composed via Hydra.  In Phase 4 the env still builds its own stack from
    Python constants; this test verifies the Hydra-composed stack instantiates
    AND the env's own arithmetic still yields the canary fingerprint.
    """
    with hydra_compose(experiment="canary_single") as cfg:
        reward_stack = instantiate(cfg.reward, _convert_="all")
        assert len(reward_stack.terms) == 3
        env = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    try:
        obs, _ = env.reset()

        approach_point = np.array([0.0, 0.0, float(HOOP_CENTER[2])], dtype=np.float64)
        through_point = HOOP_CENTER + 0.7 * HOOP_OUTWARD_NORMAL
        phase2 = False

        scored_at_step: int | None = None
        total_reward = 0.0
        for step in range(env._max_steps):
            pos = obs[9:12].astype(np.float64)
            if not phase2 and np.linalg.norm(pos - approach_point) < 0.3:
                phase2 = True
            target = through_point if phase2 else approach_point
            vec = target - pos
            if np.linalg.norm(vec) < 0.01:
                action = np.zeros(4, dtype=np.float32)
            else:
                action = np.array([
                    np.clip(vec[0] / 0.2, -1.0, 1.0),
                    np.clip(vec[1] / 0.2, -1.0, 1.0),
                    0.0,
                    np.clip(vec[2] / 0.1, -1.0, 1.0),
                ], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info.get("scored"):
                scored_at_step = step
                break
            if terminated or truncated:
                break
        assert scored_at_step == 434, f"scored_at_step={scored_at_step}"
        assert abs(total_reward - 7.3837) < 1e-3, f"total_reward={total_reward}"
    finally:
        env.close()

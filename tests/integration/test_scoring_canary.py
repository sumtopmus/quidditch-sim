"""Single-drone scripted scoring canary.

Two-phase script: climb to hoop altitude at the arena centre, then push
past the hoop along its outward normal.  Aiming directly at HOOP_CENTER
is unreliable under the MuJoCo controller — z setpoint scale (0.1) lags
xy scale (0.2), so the drone arrives at hoop plane still below the
aperture and goes around it.

Ported from part 3 of scripts/check_env.py.

Locked fingerprint (2026-05-06):
    SCORED at step 434 / total reward 7.3837
"""
from __future__ import annotations

import numpy as np
import pytest

from envs.quidditch.constants import HOOP_CENTER, HOOP_OUTWARD_NORMAL
from envs.quidditch.simple_env import QuidditchSimpleEnv

pytestmark = pytest.mark.slow


def test_scripted_flyaway_scores_through_hoop() -> None:
    env = QuidditchSimpleEnv(render_mode=None, randomise_start=False)
    try:
        obs, _ = env.reset()

        approach_point = np.array(
            [0.0, 0.0, float(HOOP_CENTER[2])], dtype=np.float64
        )
        through_point = HOOP_CENTER + 0.7 * HOOP_OUTWARD_NORMAL  # 0.7 m past hoop
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

        assert scored_at_step == 434, (
            f"expected SCORED at step 434, got {scored_at_step}"
        )
        assert total_reward == pytest.approx(7.3837, abs=1e-4)
    finally:
        env.close()

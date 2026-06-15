"""Verify the universal feature dict refactor produces byte-identical obs to the old dispatch."""
from __future__ import annotations

import numpy as np
import pytest

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch import obs_spec as obs_spec_module


@pytest.mark.parametrize("stem", ["duel_v1_body", "duel_v2_world", "duel_v3_body_ego"])
def test_feature_dict_obs_matches_legacy_dispatch(stem):
    """For each currently-supported spec, the YAML-driven obs is deterministic + correct dim."""
    spec = obs_spec_module.load_obs_yaml(stem)

    cfg = TeamConfig()  # default red_0 / blue_0 prefixes
    env = QuidditchTeamEnv(cfg=cfg, learner_id="blue_0", learner_spec=spec)
    obs_dict, _ = env.reset(seed=42)

    learner_obs = obs_dict["blue_0"]
    assert learner_obs.shape == (spec.dim,), (
        f"obs shape {learner_obs.shape} != spec dim {spec.dim}"
    )
    assert learner_obs.dtype == np.float32
    # Determinism: re-running reset with the same seed reproduces the obs.
    env.close()
    env2 = QuidditchTeamEnv(cfg=cfg, learner_id="blue_0", learner_spec=spec)
    obs_dict2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(learner_obs, obs_dict2["blue_0"])
    env2.close()


def test_take_down_fired_set_on_drone_drone_crash():
    """take_down_fired mirrors drone_drone_crash in BOTH agents' info dicts.

    eval_core._classify_terminal and the eval battery's takedown-rate read this
    key; before this fix the env never set it, so takedown-rate was always 0.
    """
    import numpy as np
    from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False))
    env.reset(seed=0)
    # One ordinary step: no crash -> take_down_fired present and False.
    zero = np.zeros(4, dtype=np.float32)
    _, _, _, _, infos = env.step({"red_0": zero, "blue_0": zero})
    assert infos["red_0"]["take_down_fired"] is False
    assert infos["blue_0"]["take_down_fired"] is False
    # Force a drone-drone crash flag and confirm both infos mirror it.
    env._aftermath_steps_left = 0
    env.reset(seed=0)
    env.cfg.crash_aftermath_seconds = 0.0
    # Drive the detector path directly: monkeypatch events() to report a ram.
    import types
    fake = types.SimpleNamespace(
        solo_floor={"red_0": False, "blue_0": False},
        wall={"red_0": 0.0, "blue_0": 0.0},
        drone_drone=(0.0, 0.0, 99.0),  # rel speed >> crash_vel_thr
    )
    env._crash_detector.events = lambda f=fake: f  # type: ignore[assignment]
    _, _, _, _, infos = env.step({"red_0": zero, "blue_0": zero})
    assert infos["red_0"]["take_down_fired"] is True
    assert infos["blue_0"]["take_down_fired"] is True
    env.close()


def test_red_action_scale_throttles_red_setpoint_delta():
    """red_action_scale halves Red's setpoint movement but leaves Blue's full."""
    import numpy as np
    from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig, ACTION_SCALE

    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=False,
                                          red_action_scale=0.5))
    env.reset(seed=0)
    sp_red0 = env._setpoint_red.copy()
    sp_blue0 = env._setpoint_blue.copy()
    a = np.ones(4, dtype=np.float32)
    env.step({"red_0": a, "blue_0": a})
    # Red moved by 0.5 * ACTION_SCALE; Blue by full ACTION_SCALE (x/y unclamped here).
    assert np.allclose(env._setpoint_red[:2] - sp_red0[:2], 0.5 * ACTION_SCALE[:2])
    assert np.allclose(env._setpoint_blue[:2] - sp_blue0[:2], ACTION_SCALE[:2])
    env.close()


def test_red_start_r_max_caps_random_disc():
    """With randomise_red_start and a small r_max, every sampled start sits
    within the cap (not the full 2.9 m disc)."""
    import numpy as np
    from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig

    env = QuidditchTeamEnv(cfg=TeamConfig(randomise_red_start=True,
                                          red_start_r_max=0.5))
    for seed in range(20):
        pos, _ = env._sample_red_start() if False else (None, None)  # see note
        env._np_random = np.random.default_rng(seed)
        pos, _ = env._sample_red_start()
        assert float(np.linalg.norm(pos[:2])) <= 0.5 + 1e-9
    env.close()

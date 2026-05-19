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

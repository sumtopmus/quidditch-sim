"""Verify env factories accept obs_blocks + obs_name and build the right spec."""
from envs.quidditch.env_factories import SimpleEnvFactory, TeamEnvFactory


def test_simple_env_factory_accepts_obs_blocks():
    factory = SimpleEnvFactory(
        n_envs=1, randomise_start=False, episode_seconds=10.0,
        obs_blocks=["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
                    "UNIT_TO_GOAL", "SIGNED_DIST_NORM"],
        obs_name="SIMPLE_ENV_OBS",
    )
    assert factory.obs_blocks == ["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
                                  "UNIT_TO_GOAL", "SIGNED_DIST_NORM"]
    assert factory.obs_name == "SIMPLE_ENV_OBS"


def test_team_env_factory_accepts_obs_blocks():
    """team_cfg is a stub — we only test field plumbing here."""
    factory = TeamEnvFactory(
        n_envs=1, team_cfg=None, learner_id="blue_0", opponent_spec="beeline_red",
        obs_blocks=["ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
                    "UNIT_TO_GOAL", "VEC_TO_HOOP", "OPP_POS_REL",
                    "OPP_VEL_REL_WORLD", "CLOSING_RATE"],
        obs_name="DUEL_V2_WORLD",
        frame_stack=3,
    )
    assert factory.obs_name == "DUEL_V2_WORLD"
    assert len(factory.obs_blocks) == 9

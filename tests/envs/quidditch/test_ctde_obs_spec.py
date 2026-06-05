from envs.quidditch import obs_spec
from envs.quidditch.obs_spec import BLOCK_BY_NAME, build_spec_from_block_names

CTDE_ACTOR = ["ANG_VEL", "ANG_POS", "LIN_VEL_WORLD", "LIN_POS",
              "VEC_TO_HOOP", "OPP_POS_REL", "OPP_VEL_REL_WORLD",
              "CLOSING_RATE", "TIME_REMAINING"]
CTDE_CRITIC = ["OPP_NEXT_ACTION", "SELF_FUTURE_DISP", "OPP_FUTURE_REL",
               "SCORE_PRED", "TAKEDOWN_PRED", "RED_POS_ABS", "BLUE_POS_ABS",
               "TAG_STATE_ONEHOT", "TERMINAL_MARGINS"]


def test_new_blocks_registered():
    for name in CTDE_ACTOR + CTDE_CRITIC:
        assert name in BLOCK_BY_NAME, name


def test_actor_and_critic_dims():
    actor = build_spec_from_block_names(CTDE_ACTOR)
    critic = build_spec_from_block_names(CTDE_CRITIC)
    assert actor.dim == 23
    assert critic.dim == 28

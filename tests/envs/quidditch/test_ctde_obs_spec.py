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


import numpy as np
from envs.quidditch.obs_spec import pack_normalized, NORM_BY_BLOCK, ARENA_NORM


def test_pack_normalized_scales_known_blocks_and_passes_through_unknown():
    spec = build_spec_from_block_names(["LIN_POS", "TAG_STATE_ONEHOT"])
    values = {"lin_pos": np.array([3.0, 0.0, 0.0], np.float32),
              "tag_state_onehot": np.array([1.0, 0.0], np.float32)}
    out = pack_normalized(spec, values, NORM_BY_BLOCK)
    # lin_pos divided by ARENA_NORM (=3) -> 1.0; tag_state passes through (no scale).
    np.testing.assert_allclose(out[:3], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[3:], [1.0, 0.0], atol=1e-6)


def test_plain_pack_unchanged_for_flat_specs():
    # The flat path must stay byte-identical: pack() ignores NORM_BY_BLOCK.
    spec = build_spec_from_block_names(["LIN_POS"])
    out = obs_spec.pack(spec, {"lin_pos": np.array([3.0, 0.0, 0.0], np.float32)})
    np.testing.assert_allclose(out, [3.0, 0.0, 0.0])

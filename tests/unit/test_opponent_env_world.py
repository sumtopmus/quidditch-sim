"""OpponentControlledEnv (and FrameStackWrapper around it) expose the
underlying World via a passthrough ``_world`` property so the video callback
can render through the wrapper.  gym.Wrapper.__getattr__ skips underscore-
prefixed names, so this only works because both wrappers add the property
explicitly."""
from __future__ import annotations

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.opponents import FrameStackWrapper, OpponentControlledEnv, from_spec


def test_opponent_env_exposes_world_property() -> None:
    team = QuidditchTeamEnv(cfg=TeamConfig(), render_mode="rgb_array")
    env = OpponentControlledEnv(
        team,
        learner_id="red_0",
        opponent=from_spec("beeline_blue"),
    )
    env.reset(seed=0)

    # _world on the wrapper is the same object as on the inner team env.
    assert env._world is team._world
    # And it's a usable World — has the rendering API the callback needs.
    assert hasattr(env._world, "render_cells")
    assert hasattr(env._world, "render_frame")


def test_frame_stack_wrapper_forwards_world_property() -> None:
    """Regression guard: video callback accesses env._world.render_cells
    through the FrameStackWrapper.  Without an explicit property,
    gym.Wrapper.__getattr__ refuses underscore names and the callback
    crashes mid-training (observed 2026-05-11)."""
    team = QuidditchTeamEnv(cfg=TeamConfig(), render_mode="rgb_array")
    inner = OpponentControlledEnv(
        team,
        learner_id="blue_0",
        opponent=from_spec("zero"),
    )
    env = FrameStackWrapper(inner, n_stack=3)
    env.reset(seed=0)

    assert env._world is team._world
    assert hasattr(env._world, "render_cells")

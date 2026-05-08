"""OpponentControlledEnv exposes the underlying World via a passthrough
``_world`` property so the video callback can render through the wrapper."""
from __future__ import annotations

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.opponents import OpponentControlledEnv, from_spec


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

from gymnasium import spaces
from envs.quidditch.env_factories import TeamEnvFactory
from envs.quidditch.team_env import TeamConfig
from scripts._train_common import build_callbacks


def _legacy_cfg():
    return {"training": {"eval": {"eval_freq_steps": 1000, "n_eval_episodes": 1},
                         "callbacks": {"checkpoint_freq_steps": 1000,
                                       "video_every_n_evals": 2, "video_fps": 20}}}


def test_eval_env_is_dict_stacked_for_ctde(tmp_path):
    f = TeamEnvFactory(n_envs=1, team_cfg=TeamConfig(randomise_red_start=False),
                       learner_id="blue_0", opponent_spec="beeline_red",
                       obs_blocks=[], obs_name="CTDE_V1", frame_stack=3,
                       ctde_mode=True, obs_stem="ctde_v1")
    cbs = build_callbacks(run_dir=tmp_path, eval_env_fn=f._make_thunk(),
                          config=_legacy_cfg(), n_envs=1, frame_stack=3,
                          ctde_mode=True)
    eval_cb = next(c for c in cbs if hasattr(c, "eval_env"))
    assert isinstance(eval_cb.eval_env.observation_space, spaces.Dict)
    assert eval_cb.eval_env.observation_space["actor"].shape == (69,)

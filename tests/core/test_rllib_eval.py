"""Step-5c: RLlib eval scenario — env spec read from the checkpoint's .hydra."""
from __future__ import annotations

from pathlib import Path

from core.rllib_eval import _env_config_from_run


def test_env_config_from_run_reads_obs_blocks(tmp_path):
    run_dir = tmp_path / "runs" / "rllib_league_step5" / "20260611_130631"
    hydra = run_dir / ".hydra"; hydra.mkdir(parents=True)
    (hydra / "config.yaml").write_text(
        "obs:\n  name: DUEL_V1_BODY\n  n_stack: 1\n"
        "  blocks: [ANG_VEL, ANG_POS]\n"
        "multiagent:\n  learner_id: red_0\n"
        "curriculum:\n  randomise_start: false\n  episode_seconds: 30.0\n"
        "  red_start_pos: [0.5, 0.0, 2.0]\n")
    ec = _env_config_from_run(run_dir)
    assert ec["obs_blocks"] == ["ANG_VEL", "ANG_POS"]
    assert ec["learner_id"] == "red_0"
    assert ec["team_cfg"]["randomise_red_start"] is False
    assert ec["reward_stack"] is None     # eval builds the default team stack

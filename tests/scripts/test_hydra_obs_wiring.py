"""Smoke test: Hydra-compose a full config and verify obs.blocks propagates to the factory."""
from pathlib import Path

from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_hydra_compose_team_carries_obs_blocks():
    with initialize_config_dir(str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["+experiment=canary_team"])
    assert list(cfg.obs.blocks) == [
        "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
        "UNIT_TO_GOAL", "VEC_TO_HOOP", "OPP_POS_REL",
        "OPP_VEL_REL_WORLD", "CLOSING_RATE",
    ]
    assert list(cfg.env.obs_blocks) == list(cfg.obs.blocks)
    assert cfg.env.obs_name == "DUEL_V2_WORLD"


def test_hydra_compose_simple_carries_obs_blocks():
    with initialize_config_dir(str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["+experiment=canary_single"])
    assert list(cfg.env.obs_blocks) == list(cfg.obs.blocks)
    assert cfg.env.obs_name == "SIMPLE_ENV_OBS"

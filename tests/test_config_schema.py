"""Schema validation tests."""
from __future__ import annotations


def test_init_config_mode_values():
    from config_schema import InitConfig
    # scratch is the only init mode after the SB3 retirement (Step 6).
    assert InitConfig(mode="scratch").mode == "scratch"


def test_register_configs_runs_without_error():
    from config_schema import register_configs
    register_configs()  # idempotent (HydraConfigStore.store overwrites by name)


def test_top_level_config_has_description_field():
    """The top-level Config schema must carry an optional `description` string
    so experiment YAMLs can set `description: |...` without Hydra struct
    rejection.  Empty default = use auto-template in MODEL.md.
    """
    from omegaconf import OmegaConf
    from config_schema import Config

    cfg = OmegaConf.structured(Config)
    assert "description" in cfg
    assert cfg.description == ""


def test_curriculum_schema_has_difficulty_levers_and_schedules():
    from config_schema import CurriculumConfig
    c = CurriculumConfig()
    assert c.red_action_scale == 1.0
    assert c.red_start_r_max is None
    assert c.dense_scale_schedule is None
    assert c.red_action_scale_schedule is None
    assert c.red_start_r_max_schedule is None

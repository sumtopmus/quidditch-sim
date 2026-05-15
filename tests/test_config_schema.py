"""Schema validation tests."""
from __future__ import annotations


def test_trainer_config_defaults():
    from config_schema import TrainerConfig
    c = TrainerConfig()
    assert c.n_steps == 1024
    assert 0.0 < c.lr < 1.0


def test_init_config_mode_values():
    from config_schema import InitConfig
    InitConfig(mode="scratch")
    InitConfig(mode="pretrain", parent="x")
    InitConfig(mode="resume", parent_run="r")
    InitConfig(mode="warm_start", parent="x", obs_surgery=True)


def test_register_configs_runs_without_error():
    from config_schema import register_configs
    register_configs()  # idempotent (HydraConfigStore.store overwrites by name)

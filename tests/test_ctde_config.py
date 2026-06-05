from config_schema import ObsConfig


def test_obs_config_has_ctde_fields_with_back_compat_defaults():
    c = ObsConfig()
    assert c.obs_mode == "flat"        # default keeps the existing path
    assert c.actor_blocks == []
    assert c.critic_blocks == []

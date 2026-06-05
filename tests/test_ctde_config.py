from config_schema import ObsConfig


def test_obs_config_has_ctde_fields_with_back_compat_defaults():
    c = ObsConfig()
    assert c.obs_mode == "flat"        # default keeps the existing path
    assert c.actor_blocks == []
    assert c.critic_blocks == []


from config_schema import PolicyConfig


def test_policy_config_defaults():
    p = PolicyConfig()
    assert p.policy_class == "MlpPolicy"
    assert p.share_features_extractor is True

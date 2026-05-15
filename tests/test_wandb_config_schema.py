"""WandbConfig is registered and conf/wandb/default.yaml composes cleanly."""
from tests.conftest import hydra_compose


def test_wandb_default_composes() -> None:
    with hydra_compose(experiment="canary_team") as cfg:
        assert cfg.wandb.project == "drone-quidditch"
        assert cfg.wandb.entity_override is None
        assert list(cfg.wandb.tags_extra) == []
        assert cfg.wandb.notes == ""
        assert cfg.wandb.log_gradients is False


def test_wandb_config_in_defaults_list() -> None:
    """conf/config.yaml's defaults list must include `wandb: default`."""
    with hydra_compose(experiment="canary_single") as cfg:
        # If `wandb: default` weren't in defaults, cfg.wandb wouldn't exist.
        assert "wandb" in cfg
        assert cfg.wandb.project == "drone-quidditch"

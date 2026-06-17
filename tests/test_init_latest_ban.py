"""InitConfig is scratch-only after the SB3 retirement (migration Step 6):
the pretrain/resume/warm_start modes, parent loading, and the `:latest`-alias
ban were all removed with the SB3 training path."""
from config_schema import InitConfig


def test_scratch_is_the_only_mode() -> None:
    cfg = InitConfig(mode="scratch")
    assert cfg.mode == "scratch"

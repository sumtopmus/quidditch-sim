"""init_wandb builds the right kwargs.  No actual wandb.init is called —
we mock at the SDK boundary so the test stays offline and fast."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import hydra_compose


@pytest.fixture
def fake_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs" / "ppo_hoop_blue_5" / "20260514_120000"
    d.mkdir(parents=True)
    return d


def _call_init(role: str, cfg_overrides: list[str] | None = None, env: dict | None = None):
    """Helper: compose canary_team cfg, patch wandb.init, return its kwargs."""
    from scripts._wandb_init import init_wandb
    with hydra_compose(experiment="canary_team", overrides=cfg_overrides or []) as cfg:
        with patch("wandb.init") as mock:
            mock.return_value = MagicMock()
            saved_env = dict(os.environ)
            try:
                if env:
                    os.environ.update(env)
                init_wandb(cfg, run_dir=Path("runs/ppo_hoop_blue_5/20260514_120000"),
                           role=role)
            finally:
                os.environ.clear()
                os.environ.update(saved_env)
        return mock.call_args.kwargs


def test_name_and_id_match_run_dir_basename() -> None:
    kw = _call_init("train")
    # cfg.run_name comes from canary_team.yaml ("canary_team"); timestamp is
    # the basename of run_dir.
    assert kw["name"] == "_canary_team_20260514_120000"
    assert kw["id"] == "_canary_team_20260514_120000"


def test_group_is_run_name_not_timestamped() -> None:
    kw = _call_init("train")
    assert kw["group"] == "_canary_team"


def test_job_type_train_when_no_sweep_env() -> None:
    kw = _call_init("train")
    assert kw["job_type"] == "train"


def test_job_type_sweep_train_when_wandb_sweep_id_set() -> None:
    kw = _call_init("train", env={"WANDB_SWEEP_ID": "abc123"})
    assert kw["job_type"] == "sweep-train"


def test_job_type_eval_when_role_eval() -> None:
    kw = _call_init("eval")
    assert kw["job_type"] == "eval"


def test_tags_include_obs_name_init_mode_learner() -> None:
    kw = _call_init("train")
    tags = set(kw["tags"])
    # canary_team.yaml: env.learner_id=blue_0, obs=DUEL_V2_WORLD,
    # init=scratch, opponent=beeline_red.
    assert "blue_0" in tags
    assert "DUEL_V2_WORLD" in tags
    assert "scratch" in tags
    assert "beeline_red" in tags


def test_tags_extra_appended() -> None:
    kw = _call_init("train", cfg_overrides=["wandb.tags_extra=[ablation,smoke]"])
    tags = set(kw["tags"])
    assert "ablation" in tags
    assert "smoke" in tags


def test_config_is_resolved_dict() -> None:
    kw = _call_init("train")
    cfg_dict = kw["config"]
    # Must be a dict (resolved OmegaConf), not a DictConfig — wandb's
    # auto-flatten works on plain dicts.
    assert isinstance(cfg_dict, dict)
    # Nested keys preserved.
    assert "trainer" in cfg_dict
    assert "lr" in cfg_dict["trainer"]


def test_mode_from_wandb_mode_env() -> None:
    # The conftest already sets WANDB_MODE=disabled.
    kw = _call_init("train")
    assert kw["mode"] == "disabled"


def test_resume_allow_so_make_resume_reattaches() -> None:
    kw = _call_init("train")
    assert kw["resume"] == "allow"


def test_dir_is_str_of_run_dir() -> None:
    kw = _call_init("train")
    assert kw["dir"] == "runs/ppo_hoop_blue_5/20260514_120000"


def test_project_from_cfg() -> None:
    kw = _call_init("train")
    assert kw["project"] == "drone-quidditch"


def test_project_override_via_env() -> None:
    kw = _call_init("train", env={"WANDB_PROJECT": "scratch-project"})
    assert kw["project"] == "scratch-project"


def test_entity_from_env_default_none() -> None:
    # Test: with no WANDB_ENTITY env and no override, entity is None.
    saved = os.environ.pop("WANDB_ENTITY", None)
    try:
        kw = _call_init("train")
        assert kw["entity"] is None
    finally:
        if saved is not None:
            os.environ["WANDB_ENTITY"] = saved


def test_entity_override_from_cfg() -> None:
    kw = _call_init("train", cfg_overrides=["wandb.entity_override=team-quidditch"])
    assert kw["entity"] == "team-quidditch"

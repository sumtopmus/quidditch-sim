"""Structured Hydra config dataclasses.

The schemas here cover the *data-only* groups (trainer, eval, init,
curriculum, obs).  Instantiated groups (env factories, opponents, reward
terms) use _target_ instantiation — the Python class itself is the schema.

`register_configs()` registers everything with Hydra's ConfigStore so YAML
files in conf/ are validated against these schemas at compose time.
Validation catches typos and wrong types before training starts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class TrainerConfig:
    """PPO hyperparameters (matches stable_baselines3.PPO __init__)."""
    n_steps: int = 1024
    batch_size: int = 512
    n_epochs: int = 6
    lr: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    total_timesteps: int = 20_000_000


@dataclass
class VideoConfig:
    enabled: bool = True
    every_n_evals: int = 2
    fps: int = 20
    grid: bool = True
    cells: list[str] = field(default_factory=lambda: ["south", "east", "top", "fixed"])
    cell_width: int = 960
    cell_height: int = 540


@dataclass
class EvalConfig:
    eval_freq_steps: int = 200_000
    n_eval_episodes: int = 5
    checkpoint_freq_steps: int = 50_000
    video: VideoConfig = field(default_factory=VideoConfig)


@dataclass
class InitConfig:
    """Mutex enforced by Hydra group choice (one conf/init/*.yaml at a time).

    Per-mode validity:
      - scratch:    all parent fields null
      - pretrain:   parent required; obs_surgery=False
      - resume:     parent_run required; parent_checkpoint optional
      - warm_start: parent required; obs_surgery=True
    """
    mode: str = "scratch"
    parent: str | None = None
    parent_run: str | None = None
    parent_checkpoint: str | None = None
    obs_surgery: bool = False
    new_dim_init_scale: float = 0.01  # only used when mode=warm_start


@dataclass
class CurriculumConfig:
    randomise_start: bool = True
    episode_seconds: float = 30.0


@dataclass
class ObsConfig:
    """Names a canonical ObsSpec from envs.quidditch.obs_spec.SPEC_BY_NAME."""
    name: str = "AUGMENTED_OBS"
    n_stack: int = 3


def register_configs() -> None:
    """Register schemas with Hydra's ConfigStore.

    Called once from scripts/train.py before @hydra.main fires.  Idempotent
    because ConfigStore.store overwrites by (group, name).
    """
    cs = ConfigStore.instance()
    cs.store(group="trainer",    name="schema", node=TrainerConfig)
    cs.store(group="eval",       name="schema", node=EvalConfig)
    cs.store(group="init",       name="schema", node=InitConfig)
    cs.store(group="curriculum", name="schema", node=CurriculumConfig)
    cs.store(group="obs",        name="schema", node=ObsConfig)

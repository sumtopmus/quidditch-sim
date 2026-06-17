"""Structured Hydra config dataclasses.

The schemas here cover the *data-only* groups (eval, init, curriculum, obs).
Instantiated groups (opponents, reward terms) use _target_ instantiation —
the Python class itself is the schema.

`register_configs()` registers everything with Hydra's ConfigStore so YAML
files in conf/ are validated against these schemas at compose time.
Validation catches typos and wrong types before training starts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


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
    """Pure-scratch league: training always starts from random init.

    (Pre-RLlib SB3 modes — pretrain/resume/warm_start, parent loading, and the
    `:latest`-alias ban — were retired with the SB3 path in migration Step 6.)
    """
    mode: str = "scratch"


@dataclass
class CurriculumConfig:
    randomise_start: bool = True
    episode_seconds: float = 30.0
    # Fixed-start lever (only read when randomise_start is False): explicit Red
    # spawn [x, y, z] and yaw.  None → origin (legacy fixed-start behavior).
    red_start_pos: list[float] | None = None
    red_start_yaw: float = 0.0
    # Difficulty levers (Step 5a) — static initial values; schedules below anneal
    # them at runtime. red_start_r_max=None → full random-start disc.
    red_action_scale: float = 1.0
    red_start_r_max: float | None = None
    # Anneal schedules as [[timestep, value], ...] (RLlib's schedule shape).
    # None → no anneal (the static value above holds for the whole run).
    dense_scale_schedule: list[list[float]] | None = None
    red_action_scale_schedule: list[list[float]] | None = None
    red_start_r_max_schedule: list[list[float]] | None = None


@dataclass
class ObsConfig:
    """Names a canonical ObsSpec; `blocks` carries the ordered ObsBlock identifiers."""
    name: str = "DUEL_V2_WORLD"
    n_stack: int = 3
    blocks: list[str] = field(default_factory=list)


@dataclass
class WandbConfig:
    """W&B integration knobs.

    Read at runtime by scripts/_wandb_init.py.  Defaults target this
    project's wandb workspace; an experiment YAML can override tags_extra
    for ad-hoc filtering.
    """
    project: str = "drone-quidditch"
    entity_override: str | None = None         # null → WANDB_ENTITY env / default
    tags_extra: list[str] = field(default_factory=list)
    notes: str = ""
    log_gradients: bool = False
    # Suppress non-essential console chatter (the "Encoding video..." spinner
    # that fires on every wandb.Video.encode, repeated per-cam per-eval).  Run
    # banner + "View run at..." links are kept.  Set to False for debugging.
    quiet: bool = True


@dataclass
class Config:
    """Top-level Hydra config schema.

    Data-only groups reference their respective schemas above.  Instantiated
    groups (env, reward, opponent) carry `Any` since they're target-instantiated
    and lack static schemas.  `description` is the optional free-text override
    for the MODEL.md Summary section; empty string falls back to auto-template.
    """
    run_name: str = "_adhoc"
    seed: int = 42
    description: str = ""
    eval: EvalConfig = field(default_factory=EvalConfig)
    init: InitConfig = field(default_factory=InitConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    env: Any = None
    reward: Any = None
    opponent: Any = None


def register_configs() -> None:
    """Register schemas with Hydra's ConfigStore.

    Called once from scripts/train.py before @hydra.main fires.  Idempotent
    because ConfigStore.store overwrites by (group, name).
    """
    cs = ConfigStore.instance()
    cs.store(group="eval",       name="schema", node=EvalConfig)
    cs.store(group="init",       name="schema", node=InitConfig)
    cs.store(group="curriculum", name="schema", node=CurriculumConfig)
    cs.store(group="obs",        name="schema", node=ObsConfig)
    cs.store(group="wandb",      name="schema", node=WandbConfig)

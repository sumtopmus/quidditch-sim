"""RLlib + Tune training entrypoint (new API stack).

Parallel to scripts/train.py (SB3) during the migration. Builds a PPOConfig
from Hydra, injects the instantiated reward stack into the env_config, and
hands the loop to Tune with the native W&B logger.
"""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import RunConfig, CheckpointConfig

from config_schema import register_configs
from rllib.config_builder import build_ppo_config
from rllib.runtime import ray_init_for_project

register_configs()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ray_init_for_project()

    reward_stack = instantiate(cfg.reward, _convert_="all") if cfg.get("reward") else None
    # Pass the built stack to the builder directly — OmegaConf is struct-mode and
    # rejects arbitrary class instances, so it can't ride inside cfg.
    ppo_config = build_ppo_config(cfg, reward_stack=reward_stack)

    callbacks = []
    if cfg.tune.wandb.enabled:
        callbacks.append(WandbLoggerCallback(project=cfg.tune.wandb.project))

    # Anchor Tune's output inside the Hydra run dir (runs/<run_name>/<ts>/),
    # next to .hydra/ — without this, Tune defaults to ~/ray_results. Must be
    # absolute (Tune rejects relative storage paths); hydra.job.chdir=false
    # keeps cwd at the repo root, so resolve() yields the checked-out tree.
    storage_path = Path(HydraConfig.get().runtime.output_dir).resolve()

    tuner = tune.Tuner(
        ppo_config.algo_class,
        param_space=ppo_config,
        run_config=RunConfig(
            name="tune",
            storage_path=str(storage_path),
            stop={"num_env_steps_sampled_lifetime": int(cfg.algo.total_timesteps)},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=int(cfg.tune.checkpoint_frequency),
                checkpoint_at_end=bool(cfg.tune.checkpoint_at_end),
            ),
            callbacks=callbacks,
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    main()

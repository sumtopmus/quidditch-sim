"""RLlib + Tune training entrypoint (new API stack).

Parallel to scripts/train.py (SB3) during the migration. Builds a PPOConfig
from Hydra, injects the instantiated reward stack into the env_config, and
hands the loop to Tune with the native W&B logger.
"""
from __future__ import annotations

import hydra
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
    # Stash the built stack so the builder can put it into env_config.
    cfg_with_reward = cfg.copy()
    cfg_with_reward.reward_stack = reward_stack  # type: ignore[attr-defined]

    ppo_config = build_ppo_config(cfg_with_reward)

    callbacks = []
    if cfg.tune.wandb.enabled:
        callbacks.append(WandbLoggerCallback(project=cfg.tune.wandb.project))

    tuner = tune.Tuner(
        ppo_config.algo_class,
        param_space=ppo_config,
        run_config=RunConfig(
            name=cfg.run_name,
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

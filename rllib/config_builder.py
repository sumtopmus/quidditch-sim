"""Hydra DictConfig -> new-stack PPOConfig.

Maps the conf/ tree (algo, obs, multiagent, reward) onto RLlib's
AlgorithmConfig builder. Registers the env under a stable name. main_red is a
learned PPO RLModule; main_blue (and future frozen members) are
ScriptedRLModules excluded from policies_to_train.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.registry import register_env

from envs.quidditch.rllib_env import make_team_env
from envs.quidditch.rllib_modules import ScriptedRLModule

_ENV_NAME = "quidditch_team"


def _ensure_env_registered() -> None:
    register_env(_ENV_NAME, lambda env_config: make_team_env(env_config))


def build_ppo_config(cfg: DictConfig) -> PPOConfig:
    _ensure_env_registered()
    ma = cfg.multiagent
    obs_blocks = list(cfg.obs.blocks)
    reward_stack = cfg.get("reward_stack")  # injected by the entrypoint (built dataclass)

    # Per-policy module specs.
    module_specs: dict[str, RLModuleSpec] = {}
    for name, spec in ma.modules.items():
        if spec.kind == "learned":
            module_specs[name] = RLModuleSpec()  # default PPO torch module
        elif spec.kind == "scripted":
            module_specs[name] = RLModuleSpec(
                module_class=ScriptedRLModule,
                model_config={"opponent_spec": spec.opponent_spec},
            )
        else:
            raise ValueError(f"unknown module kind: {spec.kind!r}")

    mapping = OmegaConf.to_container(ma.mapping, resolve=True)

    def policy_mapping_fn(agent_id, episode, **kw):
        return mapping[agent_id]

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            _ENV_NAME,
            env_config={
                "learner_id": ma.learner_id,
                "obs_blocks": obs_blocks,
                "team_cfg": {},
                "reward_stack": reward_stack,
            },
        )
        .framework("torch")
        .env_runners(num_env_runners=int(cfg.algo.num_env_runners))
        .multi_agent(
            policies=set(ma.modules.keys()),
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(ma.policies_to_train),
        )
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=module_specs))
        .training(
            lr=float(cfg.algo.lr),
            gamma=float(cfg.algo.gamma),
            lambda_=float(cfg.algo.lambda_),
            clip_param=float(cfg.algo.clip_param),
            entropy_coeff=float(cfg.algo.entropy_coeff),
            num_epochs=int(cfg.algo.num_epochs),
            minibatch_size=int(cfg.algo.minibatch_size),
            train_batch_size_per_learner=int(cfg.algo.train_batch_size_per_learner),
        )
        .debugging(seed=int(cfg.seed))
    )
    return config

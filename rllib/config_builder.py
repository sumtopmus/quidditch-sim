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
from rllib.metrics import ScoreMetricsCallback
from rllib.league import LeagueCallback, make_league_mapping_fn
from rllib.eval_battery import EvalBatteryCallback

_ENV_NAME = "quidditch_team"


def _ensure_env_registered() -> None:
    register_env(_ENV_NAME, lambda env_config: make_team_env(env_config))


def _team_cfg_from(cfg: DictConfig) -> dict:
    """Assemble TeamConfig overrides from cfg.env.team_env_params + cfg.curriculum.

    Mirrors scripts/train.py:_build_team_cfg for the RLlib path. Returns plain
    Python values only (the dict crosses Ray's serialization boundary into the
    env-runner workers). Empty when neither group is composed (the bare
    unit-test cfg), so make_team_env falls back to TeamConfig defaults.
    """
    out: dict = {}
    env = cfg.get("env")
    if env is not None and env.get("team_env_params") is not None:
        p = env.team_env_params
        out.update(
            red_prefix=p.red_prefix,
            blue_prefix=p.blue_prefix,
            hoop_prefix=p.hoop_prefix,
            midpoint_alpha=float(p.midpoint_alpha),
            tag_radius=float(p.tag_radius),
            tag_cooldown_s=float(p.tag_cooldown_s),
            crash_vel_thr=float(p.crash_vel_thr),
            walls_collide=bool(p.walls_collide),
        )
    cur = cfg.get("curriculum")
    if cur is not None:
        out["randomise_red_start"] = bool(cur.randomise_start)
        out["episode_seconds"] = float(cur.episode_seconds)
        rsp = cur.get("red_start_pos")
        if rsp is not None:
            out["red_start_pos"] = [float(v) for v in rsp]
        out["red_start_yaw"] = float(cur.get("red_start_yaw") or 0.0)
        out["red_action_scale"] = float(cur.get("red_action_scale") or 1.0)
        rrm = cur.get("red_start_r_max")
        out["red_start_r_max"] = float(rrm) if rrm is not None else None
    return out


def _curriculum_dict_from(cfg: DictConfig) -> dict:
    """Anneal schedules for CurriculumCallback. Plain Python (crosses the Ray
    boundary into env_config). Empty when no curriculum group is composed."""
    cur = cfg.get("curriculum")
    if cur is None:
        return {}
    out: dict = {}
    for key in ("dense_scale_schedule", "red_action_scale_schedule",
                "red_start_r_max_schedule"):
        sched = cur.get(key)
        if sched is not None:
            out[key] = [[float(t), float(v)] for t, v in sched]
    return out


def build_ppo_config(cfg: DictConfig, reward_stack=None) -> PPOConfig:
    _ensure_env_registered()
    ma = cfg.multiagent
    obs_blocks = list(cfg.obs.blocks)
    # reward_stack is a live (instantiated) RewardStack object, passed in by the
    # entrypoint rather than stuffed into cfg — OmegaConf rejects arbitrary
    # class instances as config values.

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

    # entropy_coeff: a schedule [[ts, val], ...] (anneal-to-zero stabilizer)
    # takes precedence over the scalar when present.
    if cfg.algo.get("entropy_coeff_schedule") is not None:
        entropy_coeff = OmegaConf.to_container(
            cfg.algo.entropy_coeff_schedule, resolve=True)
    else:
        entropy_coeff = float(cfg.algo.entropy_coeff)

    league_cfg = cfg.get("league")
    league_on = bool(league_cfg is not None and league_cfg.get("enabled"))
    if league_on:
        league_dict = OmegaConf.to_container(league_cfg, resolve=True)
        # Populations start empty: only the two mains exist at build time.
        policy_mapping_fn = make_league_mapping_fn(set(ma.modules.keys()), league_dict)
        if league_dict.get("eval_enabled"):
            # Order: EvalBatteryCallback writes result["eval"] BEFORE LeagueCallback
            # reads it in the same on_train_result sweep.
            callbacks = [ScoreMetricsCallback, EvalBatteryCallback, LeagueCallback]
        else:
            callbacks = [ScoreMetricsCallback, LeagueCallback]
    else:
        league_dict = None

        def policy_mapping_fn(agent_id, episode, **kw):
            return mapping[agent_id]

        callbacks = ScoreMetricsCallback

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
                "team_cfg": _team_cfg_from(cfg),
                "reward_stack": reward_stack,
                "league": league_dict,
                "curriculum": _curriculum_dict_from(cfg),
            },
        )
        .framework("torch")
        .callbacks(callbacks)
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
            entropy_coeff=entropy_coeff,
            num_epochs=int(cfg.algo.num_epochs),
            minibatch_size=int(cfg.algo.minibatch_size),
            train_batch_size_per_learner=int(cfg.algo.train_batch_size_per_learner),
            grad_clip=cfg.algo.get("grad_clip"),
        )
        .debugging(seed=int(cfg.seed))
    )
    return config

"""Unified Hydra entrypoint for both single-agent and team PPO training.

  python -m scripts.train +experiment=blue_v5
  python -m scripts.train +experiment=blue_v5 trainer.lr=1e-4
  python -m scripts.train -m +experiment=blue_v5 trainer.lr=1e-4,3e-4,5e-4

Group choice selects the env (single vs team), reward stack, obs spec,
opponent, init mode, curriculum, eval cadence; experiment YAMLs compose
named ladder rungs.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Allow `python -m scripts.train` and direct invocation alike.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# macOS conda libomp guard (matches train_ppo.py / train_team_ppo.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# SB3 emits an unconditional UserWarning when train (SubprocVecEnv) and eval
# (DummyVecEnv) types differ.  Intentional here; install before model.learn.
warnings.filterwarnings(
    "ignore",
    message="Training and eval env are not of the same type",
    category=UserWarning,
)

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO

from config_schema import register_configs
from envs.quidditch.obs_spec import SPEC_BY_NAME
from envs.quidditch.team_env import TeamConfig
from scripts._artifact_io import resolve_parent
from scripts._train_common import (
    append_meta_yaml_final_stats,
    build_callbacks,
    check_obs_compat,
    read_parent_chain_total_from_hydra,
    write_meta_yaml,
)


def _build_team_cfg(cfg: DictConfig) -> TeamConfig:
    """Map cfg.env.team_env_params + cfg.curriculum → TeamConfig dataclass."""
    p = cfg.env.team_env_params
    return TeamConfig(
        red_prefix          = p.red_prefix,
        blue_prefix         = p.blue_prefix,
        hoop_prefix         = p.hoop_prefix,
        midpoint_alpha      = p.midpoint_alpha,
        tag_radius          = p.tag_radius,
        tag_cooldown_s      = p.tag_cooldown_s,
        crash_vel_thr       = p.crash_vel_thr,
        walls_collide       = p.walls_collide,
        randomise_red_start = cfg.curriculum.randomise_start,
        episode_seconds     = cfg.curriculum.episode_seconds,
    )


_SCRIPTED_NAMES = {
    "BeelineRed": "beeline_red",
    "BeelineBlue": "beeline_blue",
    "IntercepterBlue": "intercepter_blue",
    "ZeroOpponent": "zero",
}


def _opponent_spec_from_cfg(cfg: DictConfig) -> str:
    """Re-serialize cfg.opponent into the legacy from_spec() string form."""
    kind = cfg.opponent.get("kind", "scripted")
    if kind == "scripted":
        target = cfg.opponent._target_.rsplit(".", 1)[-1]
        return _SCRIPTED_NAMES[target]
    if kind == "frozen":
        return f"frozen:{cfg.opponent.model_path}"
    if kind == "mixture":
        parts: list[str] = []
        for pair in cfg.opponent.mixture:
            weight, sub = pair[0], pair[1]
            sub_name = sub._target_.rsplit(".", 1)[-1]
            if sub_name == "FrozenPolicyOpponent":
                parts.append(f"{weight}*frozen:{sub.model_path}")
            else:
                parts.append(f"{weight}*{_SCRIPTED_NAMES[sub_name]}")
        return "mixture:" + ",".join(parts)
    raise ValueError(f"Unknown opponent kind: {kind}")


def _build_env_factory(cfg: DictConfig):
    """Instantiate the env factory, injecting team_cfg + opponent_spec.

    `team_env_params` is metadata for train.py (to build TeamConfig); it is
    NOT a factory kwarg, so we strip it before instantiate().
    """
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    extra: dict = {}
    if cfg.env._target_.endswith("TeamEnvFactory"):
        env_cfg.pop("team_env_params", None)
        extra["team_cfg"] = _build_team_cfg(cfg)
        extra["opponent_spec"] = _opponent_spec_from_cfg(cfg)
    return instantiate(env_cfg, **extra)


def _check_obs_compat_from_hydra(parent_hydra: Path, current_spec, current_n_stack: int):
    """Read parent's obs spec from .hydra/config.yaml, compare to current."""
    cfg_path = parent_hydra / "config.yaml"
    parent_cfg = OmegaConf.load(cfg_path)
    parent_spec_name = parent_cfg.obs.name
    parent_n_stack = int(parent_cfg.obs.n_stack)
    if parent_spec_name not in SPEC_BY_NAME:
        raise ValueError(f"Parent obs name {parent_spec_name!r} not in SPEC_BY_NAME registry")
    parent_spec = SPEC_BY_NAME[parent_spec_name]
    if parent_spec.blocks != current_spec.blocks or parent_n_stack != current_n_stack:
        raise SystemExit(
            f"Obs spec mismatch:\n"
            f"  parent: {parent_spec_name} (n_stack={parent_n_stack})\n"
            f"  current: {[b.name for b in current_spec.blocks]} (n_stack={current_n_stack})\n"
            f"Use init=warm_start for surgical extension."
        )


def _read_hydra_obs(parent_hydra: Path) -> tuple[str | None, int | None]:
    cfg_path = parent_hydra / "config.yaml"
    if not cfg_path.exists():
        return None, None
    parent_cfg = OmegaConf.load(cfg_path)
    return parent_cfg.obs.name, int(parent_cfg.obs.n_stack)


def _build_or_load_model(cfg: DictConfig, vec_env, run_dir: Path, seed: int):
    """Dispatch on cfg.init.mode to build PPO from scratch / pretrain / resume / warm_start."""
    ppo_kwargs = {
        "n_steps":       cfg.trainer.n_steps,
        "batch_size":    cfg.trainer.batch_size,
        "n_epochs":      cfg.trainer.n_epochs,
        "learning_rate": cfg.trainer.lr,
        "gamma":         cfg.trainer.gamma,
        "gae_lambda":    cfg.trainer.gae_lambda,
        "clip_range":    cfg.trainer.clip_range,
        "ent_coef":      cfg.trainer.ent_coef,
    }
    current_spec = SPEC_BY_NAME[cfg.obs.name]
    frame_stack = int(cfg.obs.n_stack)

    if cfg.init.mode == "scratch":
        return PPO(
            "MlpPolicy", vec_env,
            tensorboard_log=None, seed=seed, verbose=0, **ppo_kwargs,
        ), 0

    if cfg.init.mode == "pretrain":
        parent = resolve_parent(cfg.init.parent)
        parent_hydra = parent.parent / ".hydra"
        if not parent_hydra.exists() and (parent.parent.parent / ".hydra").exists():
            parent_hydra = parent.parent.parent / ".hydra"
        if not parent_hydra.exists():
            # Legacy fallback: parent has info.toml (pre-Phase-6-migration model).
            legacy_info = parent.parent / "info.toml"
            if not legacy_info.exists():
                legacy_info = parent.parent.parent / "info.toml"
            if legacy_info.exists():
                check_obs_compat(legacy_info, current=current_spec,
                                  current_n_stack=frame_stack, surgery=False)
            else:
                raise FileNotFoundError(
                    f"Parent at {parent.parent} has no .hydra/ — run "
                    f"scripts/migrate_legacy_models.py first."
                )
        else:
            _check_obs_compat_from_hydra(parent_hydra, current_spec, frame_stack)
        model = PPO.load(str(parent), env=vec_env, tensorboard_log=None, verbose=0, **ppo_kwargs)
        chain_total = read_parent_chain_total_from_hydra(str(parent)) or int(model.num_timesteps)
        return model, chain_total

    if cfg.init.mode == "resume":
        run_root = Path("runs") / cfg.init.parent_run
        if not run_root.exists():
            raise FileNotFoundError(f"init.parent_run={cfg.init.parent_run}: no such dir under runs/")
        latest_trial = max(run_root.iterdir(), key=lambda p: p.name)
        ckpt = cfg.init.parent_checkpoint
        if ckpt is None:
            ckpts = sorted((latest_trial / "checkpoints").glob("*.zip"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints in {latest_trial}/checkpoints/")
            ckpt = str(ckpts[-1])
        parent_hydra = latest_trial / ".hydra"
        if parent_hydra.exists():
            _check_obs_compat_from_hydra(parent_hydra, current_spec, frame_stack)
        model = PPO.load(ckpt, env=vec_env, tensorboard_log=None, verbose=0,
                          learning_rate=cfg.trainer.lr)
        return model, 0

    if cfg.init.mode == "warm_start":
        from core.policies.warm_start import warm_start_ppo_by_spec
        parent = resolve_parent(cfg.init.parent)
        parent_hydra = parent.parent / ".hydra"
        if not parent_hydra.exists() and (parent.parent.parent / ".hydra").exists():
            parent_hydra = parent.parent.parent / ".hydra"
        parent_spec_name, parent_n_stack = _read_hydra_obs(parent_hydra)
        parent_spec = SPEC_BY_NAME[parent_spec_name] if parent_spec_name else SPEC_BY_NAME["SIMPLE_ENV_OBS"]
        model = warm_start_ppo_by_spec(
            old_checkpoint=str(parent),
            new_env=vec_env,
            parent_spec=parent_spec,
            parent_n_stack=parent_n_stack or 1,
            current_spec=current_spec,
            current_n_stack=frame_stack,
            new_dim_init_scale=cfg.init.new_dim_init_scale,
            tensorboard_log=None, seed=seed, verbose=0, **ppo_kwargs,
        )
        return model, 0

    raise ValueError(f"Unknown init.mode={cfg.init.mode}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)

    seed = int(cfg.seed)

    # 1) Build env factory (instantiate cfg.env with reward + opponent injection)
    # _convert_=all so each term in cfg.reward.terms is built into a real
    # Python dataclass (default convert=none leaves them as DictConfig).
    reward_stack = instantiate(cfg.reward, _convert_="all")
    env_factory = _build_env_factory(cfg)
    # Pass reward stack to factory for Phase 5 wiring; envs still build their
    # own stack from Python constants in Phase 4 (canary preservation).
    env_factory.reward_stack = reward_stack
    train_env = env_factory.build_train_env()

    # 1.5) Initialize wandb run (after Hydra cfg is composed, before model build)
    from scripts._wandb_init import init_wandb
    wandb_run = init_wandb(cfg, run_dir=run_dir, role="train")

    # 2) Build / load model
    model, parent_chain_total = _build_or_load_model(cfg, train_env, run_dir, seed)

    # 3) Write meta.yaml start fields
    write_meta_yaml(
        run_dir,
        parent_chain_total=parent_chain_total,
        init_mode=cfg.init.mode,
        parent_path=str(cfg.init.parent) if cfg.init.parent else None,
    )

    # 4) Build callbacks (adapt cfg.eval.* to the legacy dict shape that
    #    build_callbacks consumes; Phase 5 may inline build_callbacks here).
    legacy_cfg = {
        "training": {
            "eval":      {"eval_freq_steps": cfg.eval.eval_freq_steps,
                          "n_eval_episodes": cfg.eval.n_eval_episodes},
            "callbacks": {"checkpoint_freq_steps": cfg.eval.checkpoint_freq_steps,
                          "video_every_n_evals":   cfg.eval.video.every_n_evals,
                          "video_fps":             cfg.eval.video.fps,
                          "video": {"grid":        cfg.eval.video.grid,
                                    "cells":       list(cfg.eval.video.cells),
                                    "cell_width":  cfg.eval.video.cell_width,
                                    "cell_height": cfg.eval.video.cell_height}},
        }
    }

    is_team = cfg.env._target_.endswith("TeamEnvFactory")
    if is_team:
        from envs.quidditch.team_env import QuidditchTeamEnv
        from envs.quidditch.opponents import OpponentControlledEnv, from_spec
        team_cfg = _build_team_cfg(cfg)
        opp_spec = _opponent_spec_from_cfg(cfg)
        learner = env_factory.learner_id
        def eval_env_fn():
            team = QuidditchTeamEnv(cfg=team_cfg)
            opp = from_spec(opp_spec)
            return OpponentControlledEnv(team, learner_id=learner, opponent=opp)
    else:
        from envs.quidditch.simple_env import QuidditchSimpleEnv
        rs = bool(cfg.curriculum.randomise_start)
        eps = float(cfg.curriculum.episode_seconds)
        def eval_env_fn():
            return QuidditchSimpleEnv(render_mode=None, randomise_start=rs, episode_seconds=eps)

    video_env_fn = env_factory.build_video_env_fn() if cfg.eval.video.enabled else None
    frame_stack = int(cfg.obs.n_stack)

    callbacks = build_callbacks(
        run_dir=run_dir,
        eval_env_fn=eval_env_fn,
        config=legacy_cfg,
        n_envs=cfg.env.n_envs,
        video_env_fn=video_env_fn,
        verbose=0,
        frame_stack=frame_stack,
    )

    # Wandb integration callback: hooks SB3's internal logger and forwards
    # every recorded value to wandb.log.  log="gradients" emits weight+grad
    # histograms; cfg-gated for cost.  In WANDB_MODE=disabled this is a no-op.
    from wandb.integration.sb3 import WandbCallback as _WandbCallback
    callbacks.append(
        _WandbCallback(
            verbose=0,
            log="gradients" if cfg.wandb.log_gradients else None,
            model_save_path=None,   # we handle artifact upload ourselves in Phase C
        )
    )

    # 5) Train
    started = datetime.now()
    best_eval_reward: float | None = None
    try:
        model.learn(
            total_timesteps=cfg.trainer.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=(cfg.init.mode != "resume"),
            progress_bar=True,
        )
    finally:
        elapsed_s = (datetime.now() - started).total_seconds()
        completed_steps = int(model.num_timesteps)
        model.save(str(run_dir / "final_model"))

        # EvalCallback tracks best_mean_reward on the callback instance.
        # Reach in for the value if eval ran.
        for cb in callbacks:
            if hasattr(cb, "best_mean_reward"):
                rwd = float(cb.best_mean_reward)
                if rwd > -1e9:                                # SB3 sentinel default
                    best_eval_reward = rwd
                break

        append_meta_yaml_final_stats(
            run_dir,
            wall_time_s=elapsed_s,
            completed_steps=completed_steps,
            best_eval_reward=best_eval_reward,
        )

        # Render MODEL.md before the artifact log so the upload picks it up.
        # Best-effort: a doc-gen failure logs a warning but does not fail the
        # training run.
        try:
            from scripts._render_model_doc import render_model_doc
            doc = render_model_doc(run_dir)
            (run_dir / "MODEL.md").write_text(doc)
            log.info("wrote %s", run_dir / "MODEL.md")
        except Exception as e:  # noqa: BLE001
            log.warning("MODEL.md generation failed (training succeeded): %s", e)

        # Log the best_model + .hydra/ as a wandb artifact (no-op in disabled).
        from scripts._artifact_io import log_run_artifact
        log_run_artifact(
            run=wandb_run,
            run_dir=run_dir,
            cfg=cfg,
            parent_chain_total=parent_chain_total,
            best_eval_reward=best_eval_reward,
        )

        import wandb as _wandb
        _wandb.finish()


if __name__ == "__main__":
    register_configs()
    main()

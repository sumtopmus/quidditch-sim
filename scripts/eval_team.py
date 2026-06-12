"""Head-to-head eval — Hydra entrypoint.

Usage (Hydra):
    python -m scripts.eval_team +eval_team=default \
        +learner=blue learner.uri=models/ppo_hoop_blue_4_*/best_model \
        opponent=beeline_red \
        eval.gui=true eval.crash_aftermath_seconds=3.0 eval.n_episodes=5

The argparse surface (--learner/--learner-frame-stack/--blue/--red/--gui/
--crash-aftermath-seconds/--episodes/--deterministic/--randomise-start)
is gone; all those flags are now Hydra overrides on the config groups.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# macOS: multiple libomp copies can coexist across Python distributions;
# suppress the duplicate-init abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hydra
from omegaconf import DictConfig

from core.eval_core import ScenarioResult, ScenarioSpec, run_scenario


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    learner = cfg.learner
    opp_cfg = cfg.opponent

    # opponent_model_path: only set when the chosen opponent is `frozen`.
    opponent_model_path = (
        str(opp_cfg.model_path)
        if hasattr(opp_cfg, "get") and opp_cfg.get("model_path") not in (None, "???")
        else None
    )

    scenario = ScenarioSpec(
        opponent=_opponent_spec_from_cfg(opp_cfg),
        opponent_model_path=opponent_model_path,
        randomise_start=bool(cfg.eval.randomise_start),
        n_episodes=int(cfg.eval.n_episodes),
        crash_aftermath_seconds=float(cfg.eval.crash_aftermath_seconds),
        deterministic=bool(cfg.eval.deterministic),
        learner_id=str(learner.id),
        seed=int(cfg.eval.seed),
    )

    from core.rllib_checkpoint import find_latest_checkpoint_dir, is_rllib_checkpoint
    from pathlib import Path as _Path

    learner_uri = str(learner.uri)
    ckpt = (_Path(learner_uri) if is_rllib_checkpoint(learner_uri)
            else find_latest_checkpoint_dir(_Path(learner_uri)))
    if ckpt is not None:
        from core.rllib_eval import run_rllib_battery
        run_dir = ckpt
        # Walk up to the run-ts dir that holds .hydra/ (checkpoint dirs nest
        # under tune/<trial>/).
        for parent in ckpt.parents:
            if (parent / ".hydra" / "config.yaml").exists():
                run_dir = parent
                break
        metrics = run_rllib_battery(
            ckpt, run_dir,
            n_episodes=int(cfg.eval.n_episodes), seed=int(cfg.eval.seed))
        _print_rllib_summary(metrics)
        return

    result = run_scenario(
        learner_uri=learner_uri,
        scenario=scenario,
        render=bool(cfg.eval.gui),
    )

    _print_summary(result)


def _opponent_spec_from_cfg(opp_cfg: DictConfig) -> str:
    """Read the `spec` string from the chosen /opponent YAML."""
    if hasattr(opp_cfg, "get") and opp_cfg.get("spec"):
        return str(opp_cfg.spec)
    # Fall back to deriving from _target_ (informational; should never hit in
    # practice since every opponent YAML now carries `spec`).
    target = str(opp_cfg.get("_target_", "")) if hasattr(opp_cfg, "get") else ""
    return target.rsplit(".", 1)[-1].lower() if target else "unknown"


def _print_rllib_summary(m: dict) -> None:
    print("\n=== RLlib head-to-head (main_red vs main_blue) ===")
    print(f"  red score-rate:        {m['eval_red_score_rate']:.2%}")
    print(f"  blue prevention-rate:  {m['eval_blue_prevention_rate']:.2%}")
    print(f"  take-down rate:        {m['eval_takedown_rate']:.2%}")
    print(f"  mean episode length:   {m['eval_mean_ep_len']:.1f}")
    print(f"  terminal buckets:")
    for key, v in sorted(m.items()):
        if key.startswith("eval_terminal_") and v:
            print(f"    {key[len('eval_terminal_'):]:24s} {int(v)}")


def _print_summary(result: ScenarioResult) -> None:
    n = len(result.episodes)
    print(f"\n=== {result.scenario.opponent}  (n={n}) ===")
    print(f"  win_rate:               {result.win_rate:.2%}")
    print(f"  mean reward (learner):  {result.mean_reward_learner:+.3f}")
    print(f"  mean reward (opponent): {result.mean_reward_opponent:+.3f}")
    print(f"  take-down rate:         {result.take_down_rate:.2%}")
    print(f"  mean episode length:    {result.mean_episode_length:.1f}")
    print(f"  terminal buckets:")
    for cause, k in sorted(result.terminal_cause_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {cause:24s} {k}")


if __name__ == "__main__":
    main()

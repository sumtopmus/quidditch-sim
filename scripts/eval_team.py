"""Head-to-head eval — Hydra entrypoint (RLlib-only).

Usage (Hydra):
    python -m scripts.eval_team +eval_team=default \
        +learner=red learner.uri=runs/<run>/<ts>/tune/<trial>/checkpoint_NNN \
        eval.n_episodes=5

The SB3 eval path (core.eval_core.run_scenario over OpponentControlledEnv) was
retired in migration Step 6; eval_team now loads an RLlib checkpoint and runs
the native main_red-vs-main_blue battery.
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    learner = cfg.learner

    from core.rllib_checkpoint import find_latest_checkpoint_dir, is_rllib_checkpoint

    learner_uri = str(learner.uri)
    ckpt = (Path(learner_uri) if is_rllib_checkpoint(learner_uri)
            else find_latest_checkpoint_dir(Path(learner_uri)))
    if ckpt is None:
        raise SystemExit(
            f"eval_team: {learner_uri!r} is not an RLlib checkpoint. "
            f"The SB3 eval path was retired in migration Step 6."
        )

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


if __name__ == "__main__":
    main()

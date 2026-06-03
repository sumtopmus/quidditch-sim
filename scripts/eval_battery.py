"""Run a candidate through a battery of scenarios; emit local Markdown/CSV/JSONL report.

Usage:
    python -m scripts.eval_battery +eval_battery=default \
        candidate=models/ppo_hoop_blue_4_20260511_202612/best_model

    python -m scripts.eval_battery +eval_battery=quick \
        candidate=wandb://ppo_hoop_blue_4:prod

Writes:
    <hydra.run.dir>/eval_report/{summary.md,results.csv,per_episode.jsonl}
unless `eval_battery.output_dir` is set explicitly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hydra
from omegaconf import DictConfig

from core.eval_core import ScenarioSpec, run_scenario
from core.eval_report import write_report
from core.run_context import load_run_context
from scripts._artifact_io import resolve_parent


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    eb = cfg.eval_battery
    candidate_uri = str(eb.candidate)

    candidate_dir = resolve_parent(candidate_uri, metadata_only=True)
    ctx = load_run_context(candidate_dir)
    candidate_meta = _candidate_summary_from_ctx(candidate_dir, ctx)

    scenario_results = []
    for s in eb.scenarios:
        spec = ScenarioSpec(
            opponent=str(s.opponent),
            opponent_model_path=(str(s.opponent_model_path)
                                 if s.opponent_model_path else None),
            randomise_start=bool(s.randomise_start),
            n_episodes=int(s.n_episodes),
            crash_aftermath_seconds=float(s.crash_aftermath_seconds),
            deterministic=bool(s.deterministic),
            learner_id=str(s.learner_id),
            seed=int(eb.seed),
        )
        result = run_scenario(learner_uri=candidate_uri, scenario=spec, render=False)
        scenario_results.append(result)

    out_dir = (Path(eb.output_dir) if eb.output_dir
               else Path(_hydra_run_dir()) / "eval_report")
    write_report(scenario_results, output_dir=out_dir, candidate=candidate_meta)
    print(f"\nReport written to {out_dir}/")
    print(f"  summary.md       — Markdown summary")
    print(f"  results.csv      — one row per scenario")
    print(f"  per_episode.jsonl — per-episode detail")


def _candidate_summary_from_ctx(d: Path, ctx) -> dict:
    cfg = ctx["cfg"]
    meta = ctx.get("meta") or {}
    wandb_meta = ctx.get("wandb_meta") or {}
    obs = cfg.get("obs") if hasattr(cfg, "get") else None
    obs_spec = "?"
    n_stack = 1
    if obs is not None and hasattr(obs, "get"):
        obs_spec = str(obs.get("name", "?"))
        n_stack = int(obs.get("n_stack", 1))
    return {
        "name": d.name,
        "short_name": _short_name(d.name),
        "obs_spec": obs_spec,
        "n_stack": n_stack,
        "parent_chain_total": int(meta.get("parent_chain_total", 0) or 0),
        "source": "vendored" if "/.cache/" not in str(d) else "cache",
        "wandb_alias": wandb_meta.get("alias") if isinstance(wandb_meta, dict) else None,
    }


def _short_name(full: str) -> str:
    if full.startswith("ppo_hoop_"):
        full = full[len("ppo_hoop_"):]
    parts = full.rsplit("_", 2)
    if (len(parts) == 3 and len(parts[1]) == 8 and parts[1].isdigit()
            and len(parts[2]) == 6 and parts[2].isdigit()):
        full = parts[0]
    return full


def _hydra_run_dir() -> str:
    """Return Hydra's `runtime.output_dir` for the active run."""
    from hydra.core.hydra_config import HydraConfig
    return HydraConfig.get().runtime.output_dir


if __name__ == "__main__":
    main()

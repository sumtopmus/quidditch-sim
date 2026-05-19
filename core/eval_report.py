"""Battery report writer: Markdown + CSV + JSONL.

Public API:
    write_report(scenario_results, output_dir, candidate)

Writes three files under <output_dir>/:
  - summary.md       Markdown with candidate header + one row per scenario
                     + per-scenario terminal-cause breakdown.
  - results.csv      Flat CSV; one row per scenario; all aggregates as
                     columns.  Grep-friendly.
  - per_episode.jsonl  One JSON object per episode.  For ad-hoc re-aggregation.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.eval_core import ScenarioResult


def write_report(
    scenario_results: list[ScenarioResult],
    output_dir: Path,
    candidate: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_md(scenario_results, output_dir / "summary.md", candidate)
    _write_results_csv(scenario_results, output_dir / "results.csv")
    _write_per_episode_jsonl(scenario_results, output_dir / "per_episode.jsonl")


def _write_summary_md(
    results: list[ScenarioResult],
    path: Path,
    candidate: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append(f"# Eval battery — {candidate['short_name']}")
    lines.append("")
    lines.append(f"**Run:** `{candidate['name']}`  ·  "
                 f"**Obs:** `{candidate['obs_spec']}` × n_stack={candidate['n_stack']}")
    lines.append(f"**Chain total:** {candidate['parent_chain_total']:,} steps  ·  "
                 f"**Source:** {candidate['source']}  ·  "
                 f"**Alias:** `{candidate.get('wandb_alias') or '—'}`")
    lines.append("")
    lines.append("## Per-scenario summary")
    lines.append("")
    lines.append("| Opponent | Start | n_eps | win% | mean R (lrn) | mean R (opp) | take-down% | mean ep len |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        start = "random" if r.scenario.randomise_start else "fixed"
        lines.append(
            f"| {r.scenario.opponent} | {start} | {len(r.episodes)} | "
            f"{r.win_rate * 100:.1f}% | {r.mean_reward_learner:+.3f} | "
            f"{r.mean_reward_opponent:+.3f} | {r.take_down_rate * 100:.1f}% | "
            f"{r.mean_episode_length:.1f} |"
        )
    lines.append("")
    lines.append("## Terminal-cause breakdown")
    lines.append("")
    for r in results:
        start = "random" if r.scenario.randomise_start else "fixed"
        lines.append(f"### {r.scenario.opponent} ({start} start)")
        lines.append("")
        for cause, n in sorted(r.terminal_cause_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{cause}**: {n}")
        lines.append("")
    path.write_text("\n".join(lines))


def _write_results_csv(results: list[ScenarioResult], path: Path) -> None:
    fields = [
        "opponent", "opponent_model_path", "randomise_start", "n_episodes",
        "deterministic", "crash_aftermath_seconds", "learner_id", "seed",
        "win_rate", "mean_reward_learner", "mean_reward_opponent",
        "take_down_rate", "mean_episode_length",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {**asdict(r.scenario),
                   "win_rate": r.win_rate,
                   "mean_reward_learner": r.mean_reward_learner,
                   "mean_reward_opponent": r.mean_reward_opponent,
                   "take_down_rate": r.take_down_rate,
                   "mean_episode_length": r.mean_episode_length}
            w.writerow({k: row.get(k) for k in fields})


def _write_per_episode_jsonl(results: list[ScenarioResult], path: Path) -> None:
    with path.open("w") as f:
        for r in results:
            for i, ep in enumerate(r.episodes):
                obj = {
                    "scenario": r.scenario.opponent,
                    "randomise_start": r.scenario.randomise_start,
                    "ep_idx": i,
                    "length": ep.length,
                    "reward_learner": ep.reward_learner,
                    "reward_opponent": ep.reward_opponent,
                    "terminal_cause": ep.terminal_cause,
                    "take_down_fired": ep.take_down_fired,
                    "score_at_episode_end": ep.score_at_episode_end,
                }
                f.write(json.dumps(obj) + "\n")

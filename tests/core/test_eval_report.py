"""Behavior contract for core.eval_report.write_report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from core.eval_core import EpisodeResult, ScenarioResult, ScenarioSpec


def _make_scenario_result(opp: str, n: int = 3, win_rate: float = 0.5) -> ScenarioResult:
    spec = ScenarioSpec(
        opponent=opp, opponent_model_path=None, randomise_start=True,
        n_episodes=n, deterministic=True, learner_id="blue_0", seed=0,
    )
    episodes = [
        EpisodeResult(length=500 + i, reward_learner=1.0 + i,
                      reward_opponent=-0.5,
                      terminal_cause="drone_drone_crash" if i == 0 else "timeout",
                      take_down_fired=(i == 0),
                      score_at_episode_end=None)
        for i in range(n)
    ]
    return ScenarioResult(
        scenario=spec, episodes=episodes,
        win_rate=win_rate, mean_reward_learner=2.0, mean_reward_opponent=-0.5,
        take_down_rate=1.0 / n,
        terminal_cause_counts={"drone_drone_crash": 1, "timeout": n - 1},
        mean_episode_length=500 + (n - 1) / 2,
    )


def test_write_report_creates_all_three_files(tmp_path: Path) -> None:
    from core.eval_report import write_report
    candidate = {
        "name": "ppo_hoop_blue_4_20260511_202612",
        "short_name": "blue_4",
        "obs_spec": "DUEL_V2_WORLD",
        "n_stack": 3,
        "parent_chain_total": 30_007_296,
        "source": "vendored",
        "wandb_alias": "prod",
    }
    results = [
        _make_scenario_result("beeline_red"),
        _make_scenario_result("intercepter_red:lookahead=0.5"),
    ]
    write_report(results, output_dir=tmp_path, candidate=candidate)
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "per_episode.jsonl").exists()


def test_summary_md_includes_candidate_header_and_one_row_per_scenario(tmp_path: Path) -> None:
    from core.eval_report import write_report
    candidate = {"name": "x", "short_name": "x", "obs_spec": "DUEL_V2_WORLD",
                 "n_stack": 3, "parent_chain_total": 1, "source": "vendored",
                 "wandb_alias": None}
    results = [_make_scenario_result("beeline_red"),
               _make_scenario_result("zero_red")]
    write_report(results, output_dir=tmp_path, candidate=candidate)
    md = (tmp_path / "summary.md").read_text()
    assert "DUEL_V2_WORLD" in md
    assert "beeline_red" in md
    assert "zero_red" in md
    # Each opponent row appears once in the summary table.
    assert md.count("| beeline_red") == 1
    assert md.count("| zero_red") == 1


def test_results_csv_one_row_per_scenario(tmp_path: Path) -> None:
    from core.eval_report import write_report
    candidate = {"name": "x", "short_name": "x", "obs_spec": "DUEL_V2_WORLD",
                 "n_stack": 3, "parent_chain_total": 1, "source": "vendored",
                 "wandb_alias": None}
    results = [_make_scenario_result("a"), _make_scenario_result("b"),
               _make_scenario_result("c")]
    write_report(results, output_dir=tmp_path, candidate=candidate)
    with (tmp_path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {r["opponent"] for r in rows} == {"a", "b", "c"}


def test_per_episode_jsonl_one_object_per_episode(tmp_path: Path) -> None:
    from core.eval_report import write_report
    candidate = {"name": "x", "short_name": "x", "obs_spec": "DUEL_V2_WORLD",
                 "n_stack": 3, "parent_chain_total": 1, "source": "vendored",
                 "wandb_alias": None}
    results = [_make_scenario_result("beeline_red", n=3),
               _make_scenario_result("zero_red", n=2)]
    write_report(results, output_dir=tmp_path, candidate=candidate)
    lines = (tmp_path / "per_episode.jsonl").read_text().splitlines()
    assert len(lines) == 5
    rows = [json.loads(line) for line in lines]
    assert {r["scenario"] for r in rows} == {"beeline_red", "zero_red"}
    assert all("terminal_cause" in r for r in rows)

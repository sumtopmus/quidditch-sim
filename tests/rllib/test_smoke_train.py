"""End-to-end smoke tests for the RLlib path. Marked slow."""
from __future__ import annotations

import pytest


@pytest.mark.slow
def test_cartpole_one_iteration():
    """Ray + RLlib new stack runs one PPO iteration on this machine.

    This is the install/plumbing gate — it touches no project code. If this
    fails, the Ray/Python-3.13/macOS stack is the problem, not our adapters.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(num_env_runners=1)
        .training(train_batch_size_per_learner=256, minibatch_size=64, num_epochs=1)
    )
    algo = config.build_algo()
    try:
        result = algo.train()
        assert "env_runners" in result
    finally:
        algo.stop()


@pytest.mark.slow
def test_team_skeleton_trains_and_restores(tmp_path):
    """main_red trains on the team env vs frozen hover blue; checkpoint round-trips."""
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    cfg = OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0", "policies_to_train": ["main_red"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "scripted", "opponent_spec": "zero"}}},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        algo.train()
        ckpt = algo.save(str(tmp_path / "ckpt")).checkpoint.path
        algo.restore_from_path(ckpt)
    finally:
        algo.stop()


@pytest.mark.slow
def test_league_snapshots_freezes_and_restores(tmp_path):
    """A league run with the threshold forced low takes a snapshot: the frozen
    module's weights match the main at snapshot time, do NOT change across the
    next iteration, and survive a checkpoint round-trip with membership intact."""
    import numpy as np
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.league import RED_POP_RE, BLUE_POP_RE, population_members
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    cfg = OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0",
                       "policies_to_train": ["main_red", "main_blue"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "learned"}}},
        # Short episodes (env steps at 120 Hz) so several complete inside the
        # 256-step batch -> ScoreMetricsCallback reliably logs the win-rate
        # metrics that drive the snapshot trigger.
        "curriculum": {"randomise_start": True, "episode_seconds": 1.0},
        # threshold 0 + cooldown 0 -> a snapshot fires on the first iteration
        # for whichever side records a metric.
        "league": {"enabled": True, "snapshot_threshold": 0.0,
                   "min_iters_between_snapshots": 0, "population_cap": 5,
                   "live_fraction": 0.5},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        algo.train()  # iteration 1: metrics logged -> at least one snapshot added
        # The MultiRLModule (whose keys are the population) lives on the local
        # env runner; Algorithm.get_module(id) only returns a single sub-module.
        ids = set(algo.env_runner.module.keys())
        pops = population_members(ids, RED_POP_RE) + population_members(ids, BLUE_POP_RE)
        assert pops, "expected at least one frozen snapshot after iter 1"
        snap_id = pops[0]
        main_id = "main_red" if snap_id.startswith("red") else "main_blue"

        # Snapshot weights equal the main's weights now (copied at snapshot time).
        def _flat(state):
            return np.concatenate([np.ravel(v) for v in state.values()
                                   if hasattr(v, "shape")])
        snap0 = _flat(algo.get_module(snap_id).get_state())

        algo.train()  # iteration 2: main updates; frozen snapshot must NOT.
        snap1 = _flat(algo.get_module(snap_id).get_state())
        assert np.allclose(snap0, snap1), "frozen snapshot weights changed"

        ckpt = algo.save(str(tmp_path / "ckpt")).checkpoint.path
    finally:
        algo.stop()

    # Restore into a fresh algo. Algorithm.from_checkpoint rebuilds the full
    # MultiRLModule from the checkpoint's saved per-module specs, so the frozen
    # population members come back — membership is derived from the standard
    # module checkpoint with no separate league-state file. (build_algo() +
    # restore_from_path would only reload the two mains the config declares.)
    from ray.rllib.algorithms.algorithm import Algorithm
    algo2 = Algorithm.from_checkpoint(ckpt)
    try:
        ids2 = set(algo2.env_runner.module.keys())
        assert snap_id in ids2, "snapshot module missing after restore"
    finally:
        algo2.stop()


@pytest.mark.slow
def test_league_pfsp_prunes_and_restores(tmp_path):
    """cap=1 + zero thresholds: iter 1 snapshots v1; iter 2 snapshots v2 and
    marks v1 pending (out of new matchups, module still present); iter 3
    physically removes v1. League diagnostics appear in the result; the
    surviving member round-trips a checkpoint."""
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.league import RED_POP_RE, BLUE_POP_RE, population_members
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    cfg = OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0",
                       "policies_to_train": ["main_red", "main_blue"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "learned"}}},
        # Short episodes so several complete per 256-step batch -> metrics flow
        # and no episode straddles the prune (120 steps << 256/iteration).
        "curriculum": {"randomise_start": True, "episode_seconds": 1.0},
        "league": {"enabled": True, "snapshot_threshold": 0.0,
                   "min_iters_between_snapshots": 0, "population_cap": 1,
                   "live_fraction": 0.5, "pfsp_exponent": 2.0,
                   "pfsp_uniform_floor": 0.1, "prune_winrate_threshold": 0.0,
                   "prune_grace_iters": 0},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        algo.train()   # iter 1: at least one side snapshots v1
        ids1 = set(algo.env_runner.module.keys())
        assert population_members(ids1, RED_POP_RE) or \
            population_members(ids1, BLUE_POP_RE), f"no snapshot after iter 1: {ids1}"

        result2 = algo.train()   # iter 2: at cap -> v2 added, v1 marked pending
        ids2 = set(algo.env_runner.module.keys())
        capped = [(regex, population_members(ids2, regex))
                  for regex in (RED_POP_RE, BLUE_POP_RE)
                  if len(population_members(ids2, regex)) >= 2]
        assert capped, f"expected a side holding v1 (pending) + v2: {ids2}"
        regex, members = capped[0]
        victim, survivor = members[0], members[-1]
        league = result2.get("league", {})
        assert any(k.startswith("pfsp_p_") for k in league), league
        assert league.get("pending_removals", 0) >= 1

        algo.train()   # iter 3: grace (0) elapsed -> victim physically removed
        ids3 = set(algo.env_runner.module.keys())
        assert victim not in ids3, f"{victim} not removed: {ids3}"
        assert survivor in ids3

        ckpt = algo.save(str(tmp_path / "ckpt")).checkpoint.path
    finally:
        algo.stop()

    from ray.rllib.algorithms.algorithm import Algorithm
    algo2 = Algorithm.from_checkpoint(ckpt)
    try:
        ids4 = set(algo2.env_runner.module.keys())
        assert survivor in ids4, "surviving member missing after restore"
        assert victim not in ids4, "pruned member resurrected by restore"
    finally:
        algo2.stop()


@pytest.mark.slow
def test_eval_battery_runs_and_writes_metrics(tmp_path):
    """A league run with the eval battery enabled writes clean eval_* metrics
    into the train result, and the snapshot gate consumes them (threshold 0)."""
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.eval_battery import read_eval
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    cfg = OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0",
                       "policies_to_train": ["main_red", "main_blue"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "learned"}}},
        "curriculum": {"randomise_start": True, "episode_seconds": 1.0},
        "league": {"enabled": True, "snapshot_threshold": 0.0,
                   "min_iters_between_snapshots": 0, "population_cap": 5,
                   "live_fraction": 0.5,
                   "eval_enabled": True, "eval_interval_iters": 1,
                   "eval_episodes": 3, "eval_seed": 0},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        result = algo.train()   # iter 1: battery runs, writes eval_* metrics
        srate = read_eval(result, "eval_red_score_rate")
        prev = read_eval(result, "eval_blue_prevention_rate")
        assert srate is not None and 0.0 <= float(srate) <= 1.0
        assert prev is not None and abs((srate + prev) - 1.0) < 1e-6
        assert read_eval(result, "eval_episodes") == 3.0
    finally:
        algo.stop()


@pytest.mark.slow
def test_curriculum_anneal_reaches_env(tmp_path):
    """CurriculumCallback pushes the scheduled dense_scale + red_action_scale
    onto the live env. With a schedule that hits 0 almost immediately, after one
    iteration the local env runner's reward stack reports the annealed value."""
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.runtime import ray_init_for_project

    ray_init_for_project()
    cfg = OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0",
                       "policies_to_train": ["main_red", "main_blue"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "learned"}}},
        # dense_scale drops to 0 by step 1; red_action_scale ramps 0.5 -> 1.0.
        "curriculum": {"randomise_start": True, "episode_seconds": 1.0,
                       "red_action_scale": 0.5,
                       "dense_scale_schedule": [[0, 1.0], [1, 0.0]],
                       "red_action_scale_schedule": [[0, 0.5], [10_000, 1.0]]},
        "league": {"enabled": True, "snapshot_threshold": 0.0,
                   "min_iters_between_snapshots": 0, "population_cap": 5,
                   "live_fraction": 0.5},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        algo.train()  # one iteration -> CurriculumCallback.on_train_result fired
        from rllib.curriculum import _inner_team_envs
        # The new-stack runner wraps envs in a SyncVectorMultiAgentEnv; reach the
        # live QuidditchTeamEnv(s) via the same seam the callback pushes through.
        inners = _inner_team_envs(algo.env_runner)
        assert inners, "could not reach the live QuidditchTeamEnv on the runner"
        inner = inners[0]
        assert inner._reward_stack.dense_scale == 0.0     # annealed to zero
        assert inner.cfg.red_action_scale > 0.5           # ramped up off the floor
    finally:
        algo.stop()


@pytest.mark.slow
def test_eval_team_battery_loads_from_checkpoint_dir(tmp_path):
    """Train a tiny league, save a checkpoint, then load both mains from the
    on-disk checkpoint dir and run the battery — the Step-5c eval path."""
    from pathlib import Path
    from omegaconf import OmegaConf
    from rllib.config_builder import build_ppo_config
    from rllib.runtime import ray_init_for_project
    from core.rllib_checkpoint import load_rl_module
    from rllib.eval_battery import rollout_battery, module_action_fn
    from envs.quidditch.rllib_env import make_team_env

    ray_init_for_project()
    cfg = OmegaConf.create({
        "seed": 0,
        "obs": {"name": "DUEL_V1_BODY", "n_stack": 1, "blocks": [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "SIGNED_DIST_NORM", "OPP_POS_REL", "OPP_VEL_REL_BODY",
        ]},
        "algo": {"lr": 5e-5, "gamma": 0.99, "lambda_": 0.95, "clip_param": 0.2,
                 "entropy_coeff": 0.01, "num_epochs": 1, "minibatch_size": 64,
                 "train_batch_size_per_learner": 256, "num_env_runners": 0,
                 "total_timesteps": 256},
        "multiagent": {"learner_id": "red_0",
                       "policies_to_train": ["main_red", "main_blue"],
                       "mapping": {"red_0": "main_red", "blue_0": "main_blue"},
                       "modules": {"main_red": {"kind": "learned"},
                                   "main_blue": {"kind": "learned"}}},
        "curriculum": {"randomise_start": True, "episode_seconds": 1.0},
        "reward_stack": None,
    })
    algo = build_ppo_config(cfg).build_algo()
    try:
        algo.train()
        ckpt = algo.save(str(tmp_path / "ckpt")).checkpoint.path
    finally:
        algo.stop()

    # A direct algo.save writes the checkpoint dir at the given path (the
    # checkpoint_<N> naming is a Tune-only convention); the returned path IS the
    # RLlib checkpoint dir. find_latest_checkpoint_dir's checkpoint_<N> discovery
    # is exercised by the synthetic core/run-listing tests; here we load from the
    # real Ray-produced checkpoint via the same is_rllib_checkpoint detection.
    from core.rllib_checkpoint import is_rllib_checkpoint
    found = Path(ckpt)
    assert is_rllib_checkpoint(found)
    env = make_team_env({"learner_id": "red_0",
                         "obs_blocks": list(cfg.obs.blocks),
                         "team_cfg": {"randomise_red_start": True,
                                      "episode_seconds": 1.0},
                         "reward_stack": None})
    try:
        red = module_action_fn(load_rl_module(found, "main_red"))
        blue = module_action_fn(load_rl_module(found, "main_blue"))
        m = rollout_battery(env, red, blue, n_episodes=2, seed=0)
    finally:
        env.close()
    assert 0.0 <= m["eval_red_score_rate"] <= 1.0
    assert abs(m["eval_red_score_rate"] + m["eval_blue_prevention_rate"] - 1.0) < 1e-6

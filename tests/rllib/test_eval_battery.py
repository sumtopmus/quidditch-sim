"""Step-5b eval battery: pure aggregation + rollout + callback wiring."""
from __future__ import annotations

import rllib.eval_battery as EB


def test_battery_metrics_honest_prevention_is_length_independent():
    acc = EB.init_battery_acc()
    # 4 episodes: 1 score, 3 prevented. Episode lengths vary wildly; prevention
    # must depend ONLY on the binary scored flag, not on length.
    EB.fold_episode(acc, scored=True,  take_down=False, bucket="score",    length=900)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="timeout",  length=10)
    EB.fold_episode(acc, scored=False, take_down=True,  bucket="drone_drone_crash", length=50)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="red_oob",  length=5)
    m = EB.battery_metrics(acc)
    assert m["eval_red_score_rate"] == 0.25
    assert m["eval_blue_prevention_rate"] == 0.75
    assert m["eval_takedown_rate"] == 0.25
    assert abs(m["eval_mean_ep_len"] - (900 + 10 + 50 + 5) / 4) < 1e-9


def test_battery_metrics_terminal_histogram():
    acc = EB.init_battery_acc()
    EB.fold_episode(acc, scored=True,  take_down=False, bucket="score",   length=100)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="timeout", length=100)
    EB.fold_episode(acc, scored=False, take_down=False, bucket="timeout", length=100)
    m = EB.battery_metrics(acc)
    assert m["eval_terminal_score"] == 1
    assert m["eval_terminal_timeout"] == 2
    # every bucket is represented (zeros included) so W&B panels are stable
    assert m["eval_terminal_red_oob"] == 0


def test_battery_metrics_empty_is_safe():
    m = EB.battery_metrics(EB.init_battery_acc())
    assert m["eval_red_score_rate"] == 0.0
    assert m["eval_blue_prevention_rate"] == 0.0
    assert m["eval_mean_ep_len"] == 0.0


class _StubEnv:
    """Scripted multi-agent env: each episode runs `script` steps, then ends
    with the given terminal infos. Mimics QuidditchMultiAgentEnv's step shape
    (obs/rew/term/trunc/infos dicts with __all__) without MuJoCo."""

    def __init__(self, episodes):
        # episodes: list of (n_steps, scored, take_down, terminal_cause)
        self._episodes = episodes
        self._idx = -1

    def reset(self, *, seed=None, options=None):
        self._idx += 1
        self._step = 0
        return {"red_0": [0.0], "blue_0": [0.0]}, {}

    def step(self, action_dict):
        n_steps, scored, take_down, cause = self._episodes[self._idx]
        self._step += 1
        done = self._step >= n_steps
        obs = {"red_0": [0.0], "blue_0": [0.0]}
        rew = {"red_0": 0.0, "blue_0": 0.0}
        term = {"red_0": done, "blue_0": done, "__all__": done}
        trunc = {"red_0": False, "blue_0": False, "__all__": False}
        infos = {"red_0": {}, "blue_0": {}}
        if done:
            # Stamp the terminal info so _classify_terminal returns `cause` and
            # `scored`/`take_down_fired` read correctly.
            if cause == "score":
                infos["red_0"]["scored"] = True
                infos["blue_0"]["scored"] = True
            elif cause == "drone_drone_crash":
                infos["red_0"]["drone_drone_crash"] = True
                infos["blue_0"]["drone_drone_crash"] = True
            elif cause == "red_oob":
                infos["red_0"]["red_oob"] = True
            infos["red_0"]["take_down_fired"] = take_down
            infos["blue_0"]["take_down_fired"] = take_down
        return obs, rew, term, trunc, infos


def test_rollout_battery_aggregates_outcomes():
    env = _StubEnv([
        (3, True,  False, "score"),
        (5, False, False, "timeout"),
        (2, False, True,  "drone_drone_crash"),
    ])
    # Action callables are never inspected by the stub; identity is fine.
    m = EB.rollout_battery(env, lambda o: o, lambda o: o, n_episodes=3, seed=0)
    assert m["eval_red_score_rate"] == 1 / 3
    assert m["eval_blue_prevention_rate"] == 2 / 3
    assert m["eval_takedown_rate"] == 1 / 3
    assert m["eval_terminal_score"] == 1
    assert m["eval_terminal_timeout"] == 1
    assert m["eval_terminal_drone_drone_crash"] == 1
    assert abs(m["eval_mean_ep_len"] - (3 + 5 + 2) / 3) < 1e-9


def test_module_action_fn_returns_deterministic_mean():
    import numpy as np
    import torch
    from ray.rllib.core.columns import Columns

    class _FakeModule:
        """Emits action_dist_inputs = [mean(4), log_std(4)] for a 1-row batch."""
        def forward_inference(self, batch):
            n = batch[Columns.OBS].shape[0]
            mean = torch.arange(4, dtype=torch.float32).repeat(n, 1)  # [0,1,2,3]
            log_std = torch.zeros(n, 4)
            return {Columns.ACTION_DIST_INPUTS: torch.cat([mean, log_std], dim=1)}

    fn = EB.module_action_fn(_FakeModule())
    a = fn(np.zeros(8, dtype=np.float32))
    assert isinstance(a, np.ndarray)
    assert a.shape == (4,)
    assert np.allclose(a, [0.0, 1.0, 2.0, 3.0])   # the mean, not a sample


def test_module_action_fn_stochastic_sampling_is_graded_and_seeded():
    """deterministic=False samples from the action distribution (mean + std*N(0,1)).

    The fixed-start eval battery with a deterministic (mean) policy produces
    byte-identical episodes -> a binary eval_red_score_rate, which defeats the
    graded asymmetric snapshot threshold. Stochastic sampling makes the metric a
    graded fraction of the policy's true competence; a seeded rng keeps it
    reproducible across evals of the same policy.
    """
    import numpy as np
    import torch
    from ray.rllib.core.columns import Columns

    class _FakeModule:
        """mean=0, log_std=0 (std=1) -> samples are unit-Gaussian around 0."""
        def forward_inference(self, batch):
            n = batch[Columns.OBS].shape[0]
            return {Columns.ACTION_DIST_INPUTS: torch.zeros(n, 8)}

    m = _FakeModule()
    obs = np.zeros(8, dtype=np.float32)
    # deterministic (default) -> the mean (zeros).
    assert np.allclose(EB.module_action_fn(m)(obs), np.zeros(4))
    # stochastic with a seeded rng -> a sample, NOT the mean; same seed reproduces.
    s1 = EB.module_action_fn(m, deterministic=False, rng=np.random.default_rng(0))(obs)
    s2 = EB.module_action_fn(m, deterministic=False, rng=np.random.default_rng(0))(obs)
    assert s1.shape == (4,)
    assert not np.allclose(s1, np.zeros(4))    # sampled, not the mean
    assert np.allclose(s1, s2)                 # seeded -> reproducible
    # consecutive draws from one rng differ -> graded trajectories across episodes
    fn = EB.module_action_fn(m, deterministic=False, rng=np.random.default_rng(1))
    assert not np.allclose(fn(obs), fn(obs))


import types


def _eval_algo(league_cfg, modules=("main_red", "main_blue")):
    """Minimal Algorithm stand-in for EvalBatteryCallback: exposes env_config,
    iteration, and get_module."""
    return types.SimpleNamespace(
        iteration=1,
        config=types.SimpleNamespace(env_config={
            "learner_id": "red_0",
            "obs_blocks": ["ANG_VEL"],
            "team_cfg": {},
            "reward_stack": None,
            "league": league_cfg,
        }),
        get_module=lambda mid=None: object(),
    )


def test_eval_callback_runs_on_cadence_and_caches(monkeypatch):
    calls = {"n": 0}

    def fake_battery(env, act_red, act_blue, *, n_episodes, seed):
        calls["n"] += 1
        return {"eval_red_score_rate": 0.3, "eval_blue_prevention_rate": 0.7}

    # Stub the env build + rollout so the test needs no MuJoCo.
    monkeypatch.setattr(EB, "_build_eval_env", lambda env_config: object())
    monkeypatch.setattr(EB, "rollout_battery", fake_battery)
    monkeypatch.setattr(EB, "module_action_fn", lambda m, **kw: (lambda o: o))

    cb = EB.EvalBatteryCallback()
    algo = _eval_algo({"eval_enabled": True, "eval_interval_iters": 5,
                       "eval_episodes": 4, "eval_seed": 0})

    # iter 1: always evaluates.
    r1 = {}
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result=r1)
    assert calls["n"] == 1
    assert EB.read_eval(r1, "eval_red_score_rate") == 0.3

    # iter 3: not on cadence -> no new battery, but cached metrics still written.
    r3 = {}
    algo.iteration = 3
    cb.on_train_result(algorithm=algo, result=r3)
    assert calls["n"] == 1                                   # not re-run
    assert EB.read_eval(r3, "eval_blue_prevention_rate") == 0.7  # cached

    # iter 10: 10 % 5 == 0 hits a cadence boundary -> re-run.
    r10 = {}
    algo.iteration = 10
    cb.on_train_result(algorithm=algo, result=r10)
    assert calls["n"] == 2


def test_eval_callback_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(EB, "_build_eval_env",
                        lambda env_config: (_ for _ in ()).throw(AssertionError))
    cb = EB.EvalBatteryCallback()
    algo = _eval_algo({"eval_enabled": False})
    result = {}
    algo.iteration = 1
    cb.on_train_result(algorithm=algo, result=result)   # must not build an env
    assert "eval" not in result

"""RLlib MultiAgentEnv adapter for QuidditchTeamEnv.

QuidditchTeamEnv is already a PettingZoo ParallelEnv with per-agent obs/action
spaces and dict-keyed reset/step. RLlib's MultiAgentEnv contract is nearly
identical, with one addition: the terminateds/truncateds dicts must carry an
"__all__" key signalling whole-episode termination. This adapter adds it and
otherwise passes everything through.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from envs.quidditch.team_env import QuidditchTeamEnv, TeamConfig
from envs.quidditch.obs_spec import build_spec_from_block_names


class QuidditchMultiAgentEnv(MultiAgentEnv):
    """Thin RLlib wrapper around QuidditchTeamEnv."""

    def __init__(self, inner: QuidditchTeamEnv) -> None:
        super().__init__()
        self._inner = inner
        self.possible_agents = list(inner.possible_agents)
        self.agents = list(inner.agents)
        self.observation_spaces: dict[str, spaces.Box] = dict(inner.observation_spaces)
        self.action_spaces: dict[str, spaces.Box] = dict(inner.action_spaces)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, infos = self._inner.reset(seed=seed, options=options)
        self.agents = list(self._inner.agents)
        return obs, infos

    def step(self, action_dict: dict[str, np.ndarray]):
        obs, rew, term, trunc, infos = self._inner.step(action_dict)
        term = dict(term)
        trunc = dict(trunc)
        term["__all__"] = all(term.get(a, False) for a in self.possible_agents)
        trunc["__all__"] = all(trunc.get(a, False) for a in self.possible_agents)
        self.agents = [a for a in self._inner.agents]
        return obs, rew, term, trunc, infos


def make_team_env(env_config: dict[str, Any]) -> QuidditchMultiAgentEnv:
    """Env creator for tune.register_env.

    Reads EVERYTHING from env_config — no closure capture survives Ray's
    serialization boundary, so the creator must be self-contained.

    Expected keys:
      learner_id:  str           (default "red_0")
      obs_blocks:  list[str]     (the learner's ObsSpec block names)
      team_cfg:    dict | None   (overrides for TeamConfig fields)
      reward_stack: RewardStack | None
    """
    learner_id = env_config.get("learner_id", "red_0")
    obs_blocks = list(env_config["obs_blocks"])
    team_cfg_overrides = dict(env_config.get("team_cfg", {}) or {})
    reward_stack = env_config.get("reward_stack")

    learner_spec = build_spec_from_block_names(obs_blocks)
    cfg = TeamConfig(**team_cfg_overrides)
    inner = QuidditchTeamEnv(
        cfg=cfg,
        reward_stack=reward_stack,
        learner_id=learner_id,
        learner_spec=learner_spec,
    )
    return QuidditchMultiAgentEnv(inner)

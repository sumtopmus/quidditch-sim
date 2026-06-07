"""Non-learning RLModule that defers to a scripted Opponent.

Wraps the existing Opponent protocol (envs.quidditch.opponents) so scripted
and frozen policies plug into RLlib's multi-agent stack as modules that are
NEVER in policies_to_train. Returns actions directly via Columns.ACTIONS,
bypassing the action-distribution sampling step.
"""
from __future__ import annotations

import numpy as np
import torch

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule

from envs.quidditch.opponents import from_spec


class ScriptedRLModule(TorchRLModule):
    """A frozen module driven by a scripted/frozen Opponent.

    model_config keys:
        opponent_spec: str  — passed to envs.quidditch.opponents.from_spec
                              (e.g. "zero", "beeline_blue", "frozen:path").
    """

    def setup(self) -> None:
        spec = self.model_config["opponent_spec"]
        self._opponent = from_spec(spec)
        self._opponent.reset()

    def _act(self, batch: dict) -> dict:
        obs = batch[Columns.OBS].detach().cpu().numpy()
        actions = np.stack([self._opponent.act(row) for row in obs]).astype(np.float32)
        return {Columns.ACTIONS: torch.from_numpy(actions)}

    def _forward_inference(self, batch, **kw):
        return self._act(batch)

    def _forward_exploration(self, batch, **kw):
        return self._act(batch)

    def _forward_train(self, batch, **kw):
        raise RuntimeError("ScriptedRLModule is non-learning; do not add it to policies_to_train.")

"""Non-learning RLModule that defers to a scripted Opponent.

Wraps the existing Opponent protocol (envs.quidditch.opponents) so scripted
and frozen policies plug into RLlib's multi-agent stack as modules that are
NEVER in policies_to_train. Returns actions directly via Columns.ACTIONS,
bypassing the action-distribution sampling step.

Subclasses the framework-agnostic ``RLModule`` (NOT ``TorchRLModule``) on
purpose: the TorchLearner builds an optimizer for every module that is an
``nn.Module`` (``rl_module_is_compatible`` => ``isinstance(module, nn.Module)``),
and a parameter-less torch module makes Adam raise "optimizer got an empty
parameter list". A plain ``RLModule`` is skipped by that loop, exactly like
RLlib's own ``RandomRLModule``. Forward methods still return torch tensors so
the connector pipeline and tests see the same dtype as the learned module.
"""
from __future__ import annotations

import numpy as np
import torch

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

from envs.quidditch.opponents import from_spec


class ScriptedRLModule(RLModule):
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
        obs = batch[Columns.OBS]
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        obs = np.asarray(obs)
        actions = np.stack([self._opponent.act(row) for row in obs]).astype(np.float32)
        return {Columns.ACTIONS: torch.from_numpy(actions)}

    def _forward_inference(self, batch, **kw):
        return self._act(batch)

    def _forward_exploration(self, batch, **kw):
        return self._act(batch)

    def _forward_train(self, batch, **kw):
        raise RuntimeError("ScriptedRLModule is non-learning; do not add it to policies_to_train.")

    def compile(self, *args, **kwargs):
        """No-op for parity with TorchRLModule's compile hook."""
        pass

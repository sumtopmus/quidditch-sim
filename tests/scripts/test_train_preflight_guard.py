"""When init.mode=pretrain + obs mismatch, the preflight warning fires
before the strict-raise."""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_pretrain_with_obs_mismatch_logs_preflight_then_exits(tmp_path) -> None:
    """blue_4 is DUEL_V2_WORLD n_stack=3; tell the experiment to use
    DUEL_V1_BODY n_stack=1 (which is a real spec from blue_v1/red_v1
    but incompatible).  pretrain (not warm_start) ⇒ strict-raise expected.
    """
    result = subprocess.run(
        [sys.executable, "-m", "scripts.train",
         "+experiment=canary_team",
         "init.mode=pretrain",
         "init.parent=models/ppo_hoop_blue_4_20260511_202612/best_model",
         "obs=duel_v1_body",   # forces a mismatch vs the V2 parent
         "trainer.total_timesteps=10",
         f"hydra.run.dir={tmp_path}/run",
        ],
        capture_output=True, text=True,
    )
    # Should exit non-zero (the existing check_obs_compat strict-raises).
    assert result.returncode != 0, result.stdout
    out = result.stdout + result.stderr
    assert "preflight WARNING" in out
    assert "surgery_required" in out

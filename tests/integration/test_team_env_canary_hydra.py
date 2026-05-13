"""Team env canary, Hydra-composed.  Asserts the team_env's per-step reward
fingerprint matches when the reward stack is instantiated from team_v2.yaml.

Phase 4 — runs in parallel with the existing test_team_env_canary.py.
Phase 5 collapses to a single canonical test.
"""
from __future__ import annotations

from hydra.utils import instantiate

from tests.conftest import hydra_compose


def test_team_v2_stack_instantiates_via_hydra():
    """Composing the canary YAML and instantiating cfg.reward yields a
    RewardStack with 9 terms matching the team_v2 composition."""
    with hydra_compose(experiment="canary_team") as cfg:
        stack = instantiate(cfg.reward, _convert_="all")
    assert len(stack.terms) == 9
    term_names = [type(t).__name__ for t in stack.terms]
    expected = [
        "TagEntryPulse", "ProximityGradedTag", "ClosingVelInTagZone",
        "HoopDistancePenalty", "ZeroSumDistMirror", "HoopAnchor",
        "ScoreEvent", "TakeDown", "CrashEvent",
    ]
    assert term_names == expected


def test_single_agent_stack_instantiates_via_hydra():
    """canary_single's reward stack should be a 3-term single-agent stack."""
    with hydra_compose(experiment="canary_single") as cfg:
        stack = instantiate(cfg.reward, _convert_="all")
    assert len(stack.terms) == 3
    term_names = [type(t).__name__ for t in stack.terms]
    assert term_names == ["HoopDistancePenalty", "ScoreEvent", "CrashEvent"]

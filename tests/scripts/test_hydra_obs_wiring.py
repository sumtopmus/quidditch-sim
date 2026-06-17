"""Smoke test: Hydra-composes the default config and verifies the obs group
resolves to the expected DUEL_V2_WORLD block list.

(The pre-RLlib SB3 version also asserted obs.blocks propagated to the env
factory; the factory was retired in migration Step 6, so this now just guards
the default obs composition.)"""
from tests.conftest import hydra_compose


def test_hydra_compose_default_carries_obs_blocks():
    with hydra_compose() as cfg:
        assert list(cfg.obs.blocks) == [
            "ANG_VEL", "ANG_POS", "LIN_VEL_BODY", "LIN_POS",
            "UNIT_TO_GOAL", "VEC_TO_HOOP", "OPP_POS_REL",
            "OPP_VEL_REL_WORLD", "CLOSING_RATE",
        ]
        assert cfg.obs.name == "DUEL_V2_WORLD"

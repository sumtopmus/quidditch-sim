"""Canonical Quidditch arena constants — single source of truth.

Previously these were duplicated between envs/quidditch_simple_env.py and
core/quadrotor.py defaults.  Anything that needs to know "where is the hoop"
or "how big is the arena" should import from here.
"""

from __future__ import annotations

import numpy as np


# ── Arena ────────────────────────────────────────────────────────────────────
ARENA_RADIUS: float = 3.0  # m (6 m diameter)
ARENA_WALL_HEIGHT: float = 4.5  # m

# ── Hoop geometry ────────────────────────────────────────────────────────────
# Vertical ring at x=2, y=0, z=2; outward normal points along +x (away from
# arena centre).  Drones score by crossing the ring from -x to +x.
HOOP_CENTER         = np.array([2.0, 0.0, 2.0], dtype=np.float64)
HOOP_OUTWARD_NORMAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)
HOOP_DIAMETER: float = 0.5  # m
HOOP_RADIUS:   float = HOOP_DIAMETER / 2.0  # 0.25 m

# ── Scoring trigger volume ───────────────────────────────────────────────────
# Half-length (along the hoop's outward normal) of the cylinder used for
# mj_geomDistance-based hoop scoring.  See decisions.md 2026-04-30 for the ADR.
HOOP_SCORE_TUBE_HALF_LEN: float = 0.1  # ± m around the hoop plane

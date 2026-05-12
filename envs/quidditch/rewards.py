"""Canonical Quidditch reward magnitudes — single source of truth.

Physical / geometric thresholds that *gate* reward events
(TAG_RADIUS, CRASH_VEL_THR, TAG_COOLDOWN_SECONDS, …) live in constants.py.
Only the per-event reward / penalty magnitudes (and the shaping coefficient
that ties them together) live here.

Sign convention: each constant carries its own sign.  A *_REWARD constant is
positive (gain to the recipient); a *_PENALTY constant is negative (loss to
the recipient).  This avoids the legacy `+= X / -= X` sign-flip pattern and
lets the two sides of a zero-sum event be tuned independently if needed.
"""

from __future__ import annotations

# ── Shared (simple_env + team_env) ──────────────────────────────────────────
SCORE_REWARD:           float = 10.0  # drone scores through the hoop
CRASH_PENALTY:          float = -20.0  # drone hits wall, floor, or goes OOB
DIST_REWARD_SCALE:      float = 0.01  # multiplier on −dist/ARENA_RADIUS shaping
# Blue-only hoop anchor: a per-step pull toward the hoop so the defender
# doesn't follow red to the arena edge and leave the goal undefended.
# Added in 2026-05-11 after observing blue learn to "fly to a safe corner" —
# the midpoint shaping alone wasn't enough to keep blue near the hoop when
# red wandered far from it.
HOOP_ANCHOR_SCALE:      float = 0.005

# ── Team-only (team_env) ────────────────────────────────────────────────────
# Soft-tag (proximity-based): zero-sum between blue (defender) and red (attacker).
TAG_ENTRY_REWARD:           float = 5.0   # one-shot pulse on first entry into zone
# Per-step bonus inside the tag zone, graded by proximity to red center:
#   bonus = TAG_DURATION_REWARD_MAX * max(0, 1 - dist/tag_radius)
# Peaks at contact, decays to 0 at the zone boundary, so PPO has a gradient
# that points *toward* contact instead of a flat plateau inside the sphere.
TAG_DURATION_REWARD_MAX:    float = 0.05
# Per-step bonus inside the tag zone for closing on red, in (m/s)⁻¹:
#   bonus = CLOSING_VEL_REWARD_SCALE * max(0, -d(dist)/dt)
# Rewards driving in faster than red can flee — the prerequisite for crossing
# CRASH_VEL_THR (1.5 m/s) and triggering a take-down (+TAKE_DOWN_REWARD).
CLOSING_VEL_REWARD_SCALE:   float = 0.05

# Take-down: drone-drone collision with |v_rel| > CRASH_VEL_THR.
# Currently zero-sum (same magnitude, opposite signs); split into two named
# constants so they can diverge later without touching team_env's reward code.
TAKE_DOWN_REWARD:       float = 20.0  # aggressor (the drone that took the other down)
TAKE_DOWN_PENALTY:      float = -20.0  # victim (the drone that was taken down)

# Defender shaping/tactics: target midpoint = α·red_pos + (1−α)·hoop_center.
# Lives here (not in constants.py) because it parameterises a reward term,
# not the scene geometry.
DEFAULT_MIDPOINT_ALPHA: float = 0.5

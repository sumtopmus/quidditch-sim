"""Shared imperative helpers for unit + integration tests.

Plain functions, not pytest fixtures — tests call them imperatively because
no test in the suite needs automatic teardown beyond what try/finally
already gives them.
"""
from __future__ import annotations

import mujoco

from core.drone.cf2x import cf2x_assets, cf2x_fragment
from core.world import World
from envs.quidditch.constants import (
    ARENA_RADIUS,
    ARENA_WALL_HEIGHT,
    HOOP_CENTER,
    HOOP_OUTWARD_NORMAL,
    HOOP_RADIUS,
)
from envs.quidditch.crash import CrashDetector
from envs.quidditch.scene import arena_wall_fragment, hoop_fragment


def build_team_world() -> tuple[World, CrashDetector]:
    """Build a two-drone team world (red_0 + blue_0) and its crash detector."""
    fragments = [
        cf2x_assets(with_collision_meshes=True),
        cf2x_fragment(prefix="red_0",  with_collisions=True, with_tag_sphere=True),
        cf2x_fragment(prefix="blue_0", with_collisions=True, with_tag_sphere=True),
        arena_wall_fragment(ARENA_RADIUS, ARENA_WALL_HEIGHT, with_collisions=True),
        hoop_fragment("hoop_0", HOOP_CENTER, HOOP_OUTWARD_NORMAL, HOOP_RADIUS),
    ]
    world = World(fragments)
    return world, CrashDetector(world, ["red_0", "blue_0"])


def qpos_addr(world: World, body_name: str) -> int:
    """qpos address of body's first joint (free joint = 7 entries: xyz + quat)."""
    bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jnt = int(world.model.body_jntadr[bid])
    return int(world.model.jnt_qposadr[jnt])


def set_body_state(
    world: World,
    body: str,
    pos: tuple[float, float, float],
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Write body's free-joint position, identity quat, and linear velocity."""
    qa = qpos_addr(world, body)
    world.data.qpos[qa : qa + 3] = pos
    world.data.qpos[qa + 3 : qa + 7] = (1.0, 0.0, 0.0, 0.0)
    bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, body)
    jnt = int(world.model.body_jntadr[bid])
    qva = int(world.model.jnt_dofadr[jnt])
    world.data.qvel[qva : qva + 3] = vel
    world.data.qvel[qva + 3 : qva + 6] = 0.0


# ── Hydra config composition for tests ──────────────────────────────────────
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def hydra_compose(experiment: str | None = None, overrides: list[str] | None = None):
    """Compose a Hydra config without running @hydra.main.

    Usage:
        with hydra_compose(experiment="canary_team") as cfg:
            stack = instantiate(cfg.reward)

    Each call uses a fresh Hydra context (initialize() is idempotent only when
    cleared; this context-manager handles cleanup).
    """
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    # Resolve conf/ relative to the repo root, not pytest's cwd.
    conf_path = (Path(__file__).parent.parent / "conf").resolve()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # register_configs must run before compose for schema validation to fire.
    from config_schema import register_configs
    register_configs()

    with hydra.initialize_config_dir(config_dir=str(conf_path), version_base=None):
        ov = list(overrides or [])
        if experiment is not None:
            ov.append(f"+experiment={experiment}")
        cfg = hydra.compose(config_name="config", overrides=ov)
        yield cfg

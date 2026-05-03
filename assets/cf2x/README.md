# Bitcraze Crazyflie 2 — vendored visual mesh assets

The seven `cf2_<i>.obj` files plus 32 `cf2_collision_<i>.obj` files are
meshes for the Bitcraze Crazyflie 2, copied verbatim from the
**MuJoCo Menagerie** distribution:

- Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/bitcraze_crazyflie_2
- Upstream commit: `affef0836947b64cc06c4ab1cbf0152835693374` (2026-04-16)
- Files vendored:
  - `cf2_0.obj` through `cf2_6.obj` (visuals — always loaded)
  - `cf2_collision_0.obj` through `cf2_collision_31.obj` (collision hulls
    — opt-in; loaded only when `cf2x_assets(with_collision_meshes=True)`)
  - `LICENSE` (Apache 2.0; covers all of the above)

## What we use, and what we don't

We take the **visuals** (7 `.obj` files) and the **collision hulls**
(32 `.obj` files). We deliberately skip:

- Menagerie's MJCF (`cf2.xml`, `scene.xml`). Our scene is built
  programmatically by `core.mjcf.build_mjcf` from `SceneFragment` objects;
  the cf2x drone fragment is `core.drone.cf2x.cf2x_fragment(...)` plus
  `core.drone.cf2x.cf2x_assets()`.
- Menagerie's actuator model (`<motor gear=...>`). We use an RPM-based
  thrust + reaction-torque model whose constants live in
  `core/drone/cf2x.py` (`THRUST_COEF`, `TORQUE_COEF`, `MAX_RPM`, `ARM`)
  and were originally taken from PyFlyt's `cf2x.yaml`.

## Collision meshes — opt-in

The 32 `cf2_collision_<i>.obj` files are vendored alongside the 7 visual
meshes but are loaded into the MJCF **only** when
`cf2x_assets(with_collision_meshes=True)` is called.  The default
(`False`) skips them entirely — no `<mesh>` declarations emitted, no
`.obj` bytes loaded into the in-memory asset dict.

Likewise, collision-mesh **geoms** are emitted only when
`cf2x_fragment(prefix, with_collisions=True)`.  When True, 32
`<geom mesh="cf2_collision_<i>" contype="1" conaffinity="1" group="3"/>`
geoms are inserted in the drone's `<body>` between the visual meshes
and the IMU site.

The default `contype`/`conaffinity` is `1`/`1`: the drone collides with
anything else carrying bit 1.  Currently only the floor (the worldbody
plane) carries bit 1; the hoop and arena wall use bit 0.  So enabling
the flag yields **drone-floor + drone-drone collisions** in multi-drone
scenes.  `group="3"` keeps the hulls hidden in viewers by default
(groups 3+ are off), so flipping the flag doesn't suddenly overlay
collision hulls on the visual meshes.

The single-drone `QuidditchSimpleEnv` does **not** set the flags —
it stays fully non-colliding, byte-identical to the pre-collision
build.

## What we adopted from Menagerie

- The **inertia tensor** (`IXX = IYY = 2.3951e-5`, `IZZ = 3.2347e-5`),
  which is more physically accurate per the Crazyflie 2 mechanical specs
  than PyFlyt's older defaults (`1.4e-5`, `2.17e-5`).
- The **mass** (`0.027 kg` — same value PyFlyt also uses).
- The **material colours** (rgba) attached to each of the seven mesh
  components — `propeller_plastic`, `medium_gloss_plastic`,
  `polished_gold`, `polished_plastic`, `burnished_chrome`,
  `body_frame_plastic`, `white`.

## Licensing

The `LICENSE` file in this directory is the Apache 2.0 license carried
forward from `bitcraze_crazyflie_2/LICENSE` in MuJoCo Menagerie.

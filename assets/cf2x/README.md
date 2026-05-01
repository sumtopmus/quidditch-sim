# Bitcraze Crazyflie 2 — vendored visual mesh assets

The seven `cf2_<i>.obj` files are visual meshes for the Bitcraze Crazyflie 2,
copied verbatim from the **MuJoCo Menagerie** distribution:

- Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/bitcraze_crazyflie_2
- Upstream commit: `affef0836947b64cc06c4ab1cbf0152835693374` (2026-04-16)
- Files vendored: `cf2_0.obj` through `cf2_6.obj` plus `LICENSE`

## What we use, and what we don't

We take only the **visuals** — the seven `.obj` files. We deliberately skip:

- The 32 `cf2_collision_*.obj` files. The drone in this project is
  non-colliding everywhere (`contype=0 conaffinity=0`) so the collision
  meshes are unnecessary.
- Menagerie's MJCF (`cf2.xml`, `scene.xml`). Our scene is built
  programmatically by `core.mjcf.build_mjcf` from `SceneFragment` objects;
  the cf2x drone fragment is `core.drone.cf2x.cf2x_fragment(...)` plus
  `core.drone.cf2x.cf2x_assets()`.
- Menagerie's actuator model (`<motor gear=...>`). We use an RPM-based
  thrust + reaction-torque model whose constants live in
  `core/drone/cf2x.py` (`THRUST_COEF`, `TORQUE_COEF`, `MAX_RPM`, `ARM`)
  and were originally taken from PyFlyt's `cf2x.yaml`.

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

"""Layout B: station-based line. The arm only does membrane handling; everything else is a fixed mechanism.

Stations (all animated kinematically by control/line_ctrl.py):
  - membrane magazine (spring-fed stack; the arm picks the top membrane)
  - 6-position filtration manifold: lifting funnels with clean-in-place (CIP) on every position
  - gantry doser (Y rail, X slide, Z nozzle) with disposable tips: vessel rack -> funnel
  - plate hotel -> shuttle -> transfer spot (lid lifter) -> output stack
  - tip wash station (sanitant dip + air dry) for the arm's tweezer tips
  - sterile-water and sanitant reservoirs, waste bin, operator pass-through hatch

Collision: every fixture is visual only (contype/conaffinity 0). The membrane (flex) only touches the tile
surfaces (frit tops, agar, magazine), as in cell.py. Moving parts are driven by qpos overrides.
"""
import warnings

import numpy as np
import mujoco

from scene.cell import (lookat_xyaxes, tile_bodies, panda_spec, outer_ring, MEMBRANE_RINGS, TILE, FRIT_TOP,
                        AGAR_TOP, PLATE_PITCH, MAGAZINE_TOP, HOME_Q, home, STEEL, FRIT_RGBA, PP_CLEAR,
                        LIQUID, AGAR)
from scene.membrane import flexcomp_xml

N_POS = 6
FRIT_X, PITCH = 0.54, 0.085
FRIT_YS = [PITCH * (i - (N_POS - 1) / 2) for i in range(N_POS)]
N_SAMPLES = 8

MAG = (0.30, -0.30)
SPOT = (0.30, 0.32)                  # open plate at the transfer spot
LIDPARK = (0.17, 0.32)
HOTEL = (0.10, 0.50)
OUT = (0.46, 0.50)
JUNCTION_X = 0.30                    # shuttle path: hotel -> (0.30, 0.50) -> spot ; spot -> (0.30, 0.50) -> output
TIPWASH = (0.27, 0.12)
VESSEL_X = 0.72
VESSEL_YS = [0.07 * (k - (N_SAMPLES - 1) / 2) for k in range(N_SAMPLES)]
TIPRACK = (0.66, 0.335)
WASTE = (0.66, 0.43)
GANTRY_X = 0.82                      # Y rail location; nozzle world x = NOZ_X0 + dx
NOZ_X0 = 0.78
NOZ_Z0 = 0.50                        # nozzle tip world z at dz = 0
BEAM_Z = 0.60
PARK_Q = None

CLEAR = "contype=\"0\" conaffinity=\"0\""
DARK = "0.3 0.35 0.45 1"
WATER = "0.35 0.65 0.95 0.55"
SANI = "0.95 0.65 0.25 0.55"


def funnel_xml(i, y, r_in=0.028, h=0.08, wall=0.002, n=20):
    """Lifting funnel i: slide joint flift{i}, liquid geom funnel{i}_liquid (alpha 0 until used)."""
    r_mid = r_in + wall / 2
    seg = 2 * np.pi * r_mid / n / 2 * 1.08
    g = []
    for k in range(n):
        a = 2 * np.pi * k / n
        g.append(f'<geom type="box" size="{wall/2:.4f} {seg:.4f} {h/2:.4f}" pos="{r_mid*np.cos(a):.4f} {r_mid*np.sin(a):.4f} {h/2:.4f}" '
                 f'quat="{np.cos(a/2):.4f} 0 0 {np.sin(a/2):.4f}" rgba="{PP_CLEAR}" {CLEAR}/>')
    g.append(f'<geom type="cylinder" size="{r_in+0.006:.4f} 0.005" pos="0 0 0.005" rgba="{DARK}" {CLEAR}/>')
    g.append(f'<geom type="cylinder" size="{r_in+0.004:.4f} 0.002" pos="0 0 {h:.4f}" rgba="{DARK}" {CLEAR}/>')
    g.append(f'<geom name="funnel{i}_liquid" type="cylinder" size="{r_in-0.0005:.4f} 0.0005" pos="0 0 0.0015" rgba="{LIQUID[:-4]}0" {CLEAR}/>')
    return (f'<body name="funnel{i}" pos="{FRIT_X} {y:.4f} 0.099" gravcomp="1">'
            f'<joint name="flift{i}" type="slide" axis="0 0 1" range="0 0.4" damping="1"/>{"".join(g)}</body>')


def fine_tiles(prefix, z_top, solref="0.005 1"):
    """19 overlapping thin tiles (centre + 6 + 12) so that no single tile body carries more than ~12 of the
    membrane's vertex contacts (MuJoCo caps flex contacts at 50 per flex-body pair)."""
    cells = [(0.0, 0.0, 0.0105)]
    cells += [(0.0165 * np.cos(a), 0.0165 * np.sin(a), 0.0095) for a in np.arange(6) * np.pi / 3]
    cells += [(0.0285 * np.cos(a), 0.0285 * np.sin(a), 0.0095) for a in (np.arange(12) + 0.5) * np.pi / 6]
    return "\n".join(
        f'<body name="{prefix}_t{i}" pos="{cx:.4f} {cy:.4f} {z_top - 0.0005:.4f}">'
        f'<geom type="cylinder" size="{r:.4f} 0.0005" rgba="0.5 0.5 0.5 1" group="3" contype="{TILE}" conaffinity="{TILE}" '
        f'friction="1.5 0.01 0.0001" solref="{solref}"/></body>' for i, (cx, cy, r) in enumerate(cells))


def plate_xml(k, pos, closed=True):
    """Plate k (mocap body: kinematic, so the membrane cannot push it) with agar tiles, hidden rigid-membrane disc pm{k} and its lid."""
    tiles = fine_tiles(f"agar{k}", AGAR_TOP)
    return f"""
    <body name="plate{k}" mocap="true" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}">
      <geom type="cylinder" size="0.045 0.006" pos="0 0 0.006" rgba="{PP_CLEAR}" {CLEAR} mass="0.03"/>
      <geom type="cylinder" size="0.043 0.0055" pos="0 0 0.0065" rgba="{AGAR}" {CLEAR} mass="0"/>
      <geom name="pm{k}" type="cylinder" size="0.0235 0.0003" pos="0 0 {AGAR_TOP + 0.0004:.4f}" rgba="0.95 0.95 0.9 0" {CLEAR} mass="0"/>
      {tiles}
    </body>
    <body name="lid{k}" mocap="true" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2] + 0.0064:.4f}">
      <geom type="cylinder" size="0.047 0.0012" pos="0 0 0.0088" rgba="{PP_CLEAR}" {CLEAR} mass="0.008"/>
      <geom type="cylinder" size="0.0475 0.004" pos="0 0 0.0048" rgba="0.88 0.9 0.95 0.18" {CLEAR} mass="0"/>
    </body>"""


def base_xml():
    fx = FRIT_X
    # --- manifold positions
    pos_xml = []
    for i, y in enumerate(FRIT_YS):
        pos_xml.append(f'<body name="frit{i}" pos="{fx} {y:.4f} 0.08">'
                       f'<geom type="cylinder" size="0.025 0.019" rgba="{FRIT_RGBA}"/>'
                       f'<geom type="cylinder" size="0.032 0.003" pos="0 0 0.016" rgba="{DARK}"/>'
                       f'<geom name="fm{i}" type="cylinder" size="0.0235 0.0003" pos="0 0 0.0204" rgba="0.95 0.95 0.9 0" {CLEAR} mass="0"/>'
                       f'</body>')
        pos_xml.append(tile_bodies(f"frit{i}", fx, y, FRIT_TOP, 0.025))
        pos_xml.append(funnel_xml(i, y))
        pos_xml.append(f'<geom name="led{i}" type="sphere" size="0.007" pos="{fx - 0.058:.4f} {y:.4f} 0.075" rgba="0.5 0.5 0.5 1" {CLEAR}/>')
        pos_xml.append(f'<geom type="cylinder" size="0.004 0.04" pos="{fx:.4f} {y:.4f} 0.012" rgba="0.25 0.25 0.28 1" {CLEAR}/>')

    # --- vessels (septum caps), tip rack
    vessels = []
    for k, y in enumerate(VESSEL_YS):
        vessels.append(f'<body name="vessel{k}" pos="{VESSEL_X} {y:.4f} 0.03">'
                       f'<geom type="cylinder" size="0.02 0.04" pos="0 0 0.04" rgba="{PP_CLEAR}" {CLEAR}/>'
                       f'<geom name="vessel{k}_liquid" type="cylinder" size="0.0185 0.025" pos="0 0 0.026" rgba="{LIQUID}" {CLEAR}/>'
                       f'<geom type="cylinder" size="0.0215 0.006" pos="0 0 0.086" rgba="0.2 0.45 0.7 1" {CLEAR}/>'
                       f'<geom type="cylinder" size="0.008 0.002" pos="0 0 0.093" rgba="0.1 0.1 0.12 1" {CLEAR}/></body>')
    tips = "".join(f'<geom name="tip{j}" type="capsule" size="0.003 0.012" pos="{TIPRACK[0] + 0.012 * (j % 4 - 1.5):.4f} {TIPRACK[1] + 0.012 * (j // 4 - 0.5):.4f} 0.052" rgba="0.8 0.85 0.9 0.9" {CLEAR}/>'
                   for j in range(N_SAMPLES))

    # --- plate hotel (static outline) and rails
    def rail(p0, p1, w=0.012):
        c = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        hx, hy = abs(p1[0] - p0[0]) / 2 + w, abs(p1[1] - p0[1]) / 2 + w
        return f'<geom type="box" size="{hx:.4f} {hy:.4f} 0.0015" pos="{c[0]:.4f} {c[1]:.4f} 0.0015" rgba="0.55 0.58 0.62 1" {CLEAR}/>'
    rails = rail(HOTEL, (JUNCTION_X, HOTEL[1])) + rail((JUNCTION_X, HOTEL[1]), SPOT) + rail((JUNCTION_X, OUT[1]), OUT)

    plates = "".join(plate_xml(k, (HOTEL[0], HOTEL[1], (N_SAMPLES - 1 - k) * PLATE_PITCH + 0.002)) for k in range(N_SAMPLES))

    # --- magazine stack discs
    mag_discs = "".join(f'<geom name="ms{j}" type="cylinder" size="0.0235 0.0003" pos="0 0 {0.0275 + j * 0.0006:.4f}" rgba="0.97 0.97 0.95 1" {CLEAR} mass="0"/>'
                        for j in range(N_SAMPLES))

    cams = {
        "line_a": ((1.32, -0.72, 0.85), (0.46, 0.12, 0.08), (0, 0, 1)),
        "line_b": ((0.47, 0.03, 1.20), (0.47, 0.03, 0.0), (1, 0, 0)),
        "line_c": ((1.15, 0.65, 0.65), (0.48, 0.05, 0.08), (0, 0, 1)),
        "line_d": ((0.72, 0.92, 0.52), (0.30, 0.38, 0.04), (0, 0, 1)),
        "oblique": ((1.25, -0.45, 0.80), (0.38, 0.02, 0.17), (0, 0, 1)),
        "spot_side": ((SPOT[0] + 0.005, SPOT[1] - 0.15, 0.06), (SPOT[0], SPOT[1], 0.018), (0, 0, 1)),
        "frit_side": ((FRIT_X - 0.12, FRIT_YS[0] - 0.12, 0.17), (FRIT_X, FRIT_YS[0], 0.10), (0, 0, 1)),
    }
    cam_xml = "\n".join(f'<camera name="{n}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" xyaxes="{lookat_xyaxes(p, t, u)}" fovy="48"/>'
                        for n, (p, t, u) in cams.items())

    posts = "\n".join(f'<geom type="box" size="0.012 0.012 0.45" pos="{x} {y} 0.45" rgba="0.6 0.62 0.66 1" {CLEAR}/>'
                      for x in (-0.22, 0.92) for y in (-0.72, 0.72))

    return f"""<mujoco model="membrane_line">
  <option timestep="0.002" integrator="discrete" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="-140" elevation="-25"/>
    <headlight ambient="0.38 0.38 0.38" diffuse="0.5 0.5 0.5"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.92 0.94 0.97" rgb2="0.6 0.65 0.72" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.78 0.79 0.8" rgb2="0.72 0.73 0.74" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="8 8"/>
    <texture name="bench" type="2d" builtin="flat" rgb1="0.93 0.93 0.91" width="64" height="64"/>
    <material name="bench" texture="bench" specular="0.3" shininess="0.4"/>
  </asset>
  <worldbody>
    <light pos="0.5 -0.3 1.8" dir="0 0.2 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light pos="1.2 0.8 1.4" dir="-0.6 -0.4 -0.6" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.01" pos="0 0 -0.75" material="floor"/>
    <geom name="table" type="box" size="0.58 0.74 0.02" pos="0.35 0 -0.02" material="bench"/>
    <geom type="box" size="0.58 0.74 0.36" pos="0.35 0 -0.40" rgba="0.55 0.57 0.6 1"/>
    <geom type="box" size="0.004 0.72 0.45" pos="-0.22 0 0.45" rgba="0.8 0.88 0.95 0.12" {CLEAR}/>
    {cam_xml}

    <!-- manifold block with 6 positions -->
    <body name="manifold" pos="{fx} 0 0.04">
      <geom type="box" size="0.05 0.27 0.04" rgba="{STEEL}" {CLEAR}/>
      <geom type="cylinder" size="0.008 0.27" pos="-0.062 0 -0.02" quat="0.7071 0.7071 0 0" rgba="0.25 0.25 0.28 1" {CLEAR}/>
    </body>
    {''.join(pos_xml)}

    <!-- membrane magazine -->
    <body name="magazine" pos="{MAG[0]} {MAG[1]} 0">
      <geom type="cylinder" size="0.03 0.01" pos="0 0 0.01" rgba="{STEEL}" {CLEAR}/>
      <geom type="cylinder" size="0.026 0.0075" pos="0 0 0.0195" rgba="0.5 0.52 0.56 1" {CLEAR}/>
      {mag_discs}
      <geom type="box" size="0.002 0.002 0.03" pos="0 0.031 0.03" rgba="{STEEL}" {CLEAR}/>
      <geom type="box" size="0.002 0.002 0.03" pos="0 -0.031 0.03" rgba="{STEEL}" {CLEAR}/>
    </body>
    {tile_bodies("magazine", MAG[0], MAG[1], MAGAZINE_TOP, 0.025)}

    <!-- tip wash station: bath + air knife -->
    <body name="tipwash" pos="{TIPWASH[0]} {TIPWASH[1]} 0">
      <geom type="box" size="0.05 0.04 0.012" pos="0 0 0.012" rgba="0.5 0.6 0.7 1" {CLEAR}/>
      <geom name="tw_bath" type="cylinder" size="0.022 0.003" pos="0 0 0.0255" rgba="{SANI}" {CLEAR}/>
      <geom type="box" size="0.004 0.004 0.03" pos="0.032 0 0.04" rgba="{STEEL}" {CLEAR}/>
      <geom name="tw_air" type="cylinder" size="0.0035 0.01" pos="0.024 0 0.048" quat="0.7071 0 0.7071 0" rgba="0.75 0.9 1 0" {CLEAR}/>
    </body>

    <!-- vessel rack, hatch, tip rack, waste -->
    <body name="rack" pos="{VESSEL_X} 0 0.015"><geom type="box" size="0.035 0.31 0.015" rgba="0.85 0.86 0.88 1" {CLEAR}/></body>
    {''.join(vessels)}
    <geom type="box" size="0.05 0.04 0.012" pos="{TIPRACK[0]} {TIPRACK[1]} 0.012" rgba="0.8 0.8 0.84 1" {CLEAR}/>
    {tips}
    <body name="waste" pos="{WASTE[0]} {WASTE[1]} 0">
      <geom type="box" size="0.05 0.04 0.04" pos="0 0 0.04" rgba="0.3 0.3 0.32 1" {CLEAR}/>
      <geom type="box" size="0.044 0.034 0.002" pos="0 0 0.082" rgba="0.12 0.12 0.14 1" {CLEAR}/>
    </body>
    <geom name="hatch" type="box" size="0.012 0.32 0.13" pos="0.915 0 0.15" rgba="0.35 0.6 0.85 0.25" {CLEAR}/>
    <geom type="box" size="0.02 0.33 0.01" pos="0.915 0 0.285" rgba="0.5 0.52 0.56 1" {CLEAR}/>

    <!-- reservoirs and pipes -->
    <body name="tanks" pos="-0.12 0 0">
      <geom type="cylinder" size="0.06 0.15" pos="0 -0.34 0.15" rgba="0.8 0.85 0.9 0.35" {CLEAR}/>
      <geom name="tank_w" type="cylinder" size="0.057 0.14" pos="0 -0.34 0.14" rgba="{WATER}" {CLEAR}/>
      <geom type="cylinder" size="0.06 0.15" pos="0 -0.2 0.15" rgba="0.8 0.85 0.9 0.35" {CLEAR}/>
      <geom name="tank_s" type="cylinder" size="0.057 0.14" pos="0 -0.2 0.14" rgba="{SANI}" {CLEAR}/>
    </body>
    <geom type="box" size="0.33 0.004 0.004" pos="0.21 -0.34 0.006" rgba="0.4 0.6 0.85 1" {CLEAR}/>
    <geom type="box" size="0.004 0.04 0.004" pos="0.54 -0.30 0.006" rgba="0.4 0.6 0.85 1" {CLEAR}/>
    <geom type="box" size="0.30 0.004 0.004" pos="0.18 -0.20 0.016" rgba="0.95 0.65 0.25 1" {CLEAR}/>

    <!-- shuttle rails, hotel outline, output marker -->
    {rails}
    <geom type="box" size="0.055 0.055 0.0015" pos="{HOTEL[0]} {HOTEL[1]} 0.0015" rgba="0.5 0.52 0.56 1" {CLEAR}/>
    <geom type="box" size="0.055 0.055 0.0015" pos="{OUT[0]} {OUT[1]} 0.0015" rgba="0.5 0.52 0.56 1" {CLEAR}/>
    <geom type="box" size="0.055 0.055 0.0015" pos="{SPOT[0]} {SPOT[1]} 0.0015" rgba="0.5 0.52 0.56 1" {CLEAR}/>
    {plates}

    <!-- plate shuttle carriage (drives with the plate) -->
    <body name="shuttle" pos="{HOTEL[0]} {HOTEL[1]} 0" gravcomp="1">
      <joint name="sx" type="slide" axis="1 0 0" range="-1 1" damping="1"/>
      <joint name="sy" type="slide" axis="0 1 0" range="-1 1" damping="1"/>
      <joint name="sz" type="slide" axis="0 0 1" range="-0.2 0.4" damping="1"/>
      <geom type="box" size="0.055 0.055 0.002" pos="0 0 -0.002" rgba="0.2 0.45 0.7 1" {CLEAR}/>
    </body>

    <!-- lid lifter: horizontal bar with a suction cup -->
    <body name="lidg" pos="{LIDPARK[0]} {SPOT[1]} 0" gravcomp="1">
      <joint name="lgx" type="slide" axis="1 0 0" range="-0.3 0.3" damping="1"/>
      <geom type="box" size="0.012 0.075 0.01" pos="0 0.075 0.19" rgba="{STEEL}" {CLEAR}/>
      <geom type="box" size="0.012 0.012 0.1" pos="0 0.15 0.1" rgba="{STEEL}" {CLEAR}/>
      <body name="lidz" pos="0 0 0.18" gravcomp="1">
        <joint name="lgz" type="slide" axis="0 0 1" range="-0.2 0.01" damping="1"/>
        <geom type="cylinder" size="0.004 0.03" pos="0 0 0.02" rgba="0.3 0.3 0.33 1" {CLEAR}/>
        <geom type="cylinder" size="0.022 0.004" pos="0 0 -0.012" rgba="0.15 0.15 0.18 1" {CLEAR}/>
      </body>
    </body>

    <!-- doser gantry: Y rail with carriage, X slide, Z nozzle -->
    <geom type="box" size="0.02 0.5 0.02" pos="{GANTRY_X} 0 {BEAM_Z + 0.04}" rgba="0.6 0.62 0.66 1" {CLEAR}/>
    <geom type="box" size="0.02 0.02 0.3" pos="{GANTRY_X} -0.5 0.3" rgba="0.6 0.62 0.66 1" {CLEAR}/>
    <geom type="box" size="0.02 0.02 0.3" pos="{GANTRY_X} 0.5 0.3" rgba="0.6 0.62 0.66 1" {CLEAR}/>
    <body name="dcar" pos="{GANTRY_X} 0 {BEAM_Z}" gravcomp="1">
      <joint name="dy" type="slide" axis="0 1 0" range="-0.46 0.46" damping="1"/>
      <geom type="box" size="0.04 0.045 0.03" pos="0 0 0.02" rgba="0.25 0.5 0.75 1" {CLEAR}/>
      <geom type="box" size="0.2 0.012 0.012" pos="-0.2 0 -0.012" rgba="{STEEL}" {CLEAR}/>
      <body name="dslide" pos="{NOZ_X0 - GANTRY_X} 0 -0.03" gravcomp="1">
        <joint name="dx" type="slide" axis="1 0 0" range="-0.34 0.02" damping="1"/>
        <geom type="box" size="0.03 0.025 0.02" pos="0 0 0" rgba="0.25 0.5 0.75 1" {CLEAR}/>
        <body name="dnoz" pos="0 0 {NOZ_Z0 - BEAM_Z + 0.03}" gravcomp="1">
          <joint name="dz" type="slide" axis="0 0 1" range="-0.45 0.02" damping="1"/>
          <geom name="dshaft" type="cylinder" size="0.004 0.05" pos="0 0 0.05" rgba="{STEEL}" {CLEAR}/>
          <geom name="dtip" type="capsule" size="0.003 0.012" pos="0 0 0.012" rgba="0.8 0.85 0.9 0" {CLEAR}/>
        </body>
      </body>
    </body>

    <!-- membrane (single flex, teleported between stations) -->
    {flexcomp_xml(pos=(MAG[0], MAG[1], MAGAZINE_TOP + 0.0004), contype=TILE)}
  </worldbody>
</mujoco>"""


def build():
    spec = mujoco.MjSpec.from_string(base_xml())
    frame = spec.worldbody.add_frame(pos=[0, 0, 0])
    frame.attach_body(panda_spec().body("link0"), "panda/", "")
    for k in range(outer_ring(MEMBRANE_RINGS - 1)[0], outer_ring()[-1] + 1):
        spec.add_equality(name=f"grip{k}", type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1="panda/hand", name2=f"membrane_{k}", active=False, solref=[0.004, 1])
    return spec


def load():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attach conflict")
        m = build().compile()
    d = mujoco.MjData(m)
    home(m, d)
    return m, d

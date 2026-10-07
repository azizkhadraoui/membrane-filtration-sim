"""Variant A cell: Panda on a table, 6-position filter block, plates, vessels, magazine.

Units metres; origin at the arm base, table top at z = 0.

Collision layout
- The membrane (flex) only collides with "tile" geoms (bit 8): frit tops, agar, magazine top.
  MuJoCo caps flex contacts at 50 per flex-body pair, so each tile surface is 9 separate bodies.
- Rigid objects use the default bit 1; tweezer tips use bit 2 and touch nothing.

Modes
- "transfer": membrane on frit 0 after filtration, open plate at the work spot, lid beside it.
- "sequence": membrane in the magazine, plate closed on top of the input stack, funnel 0
  parked at the wash station, vessel 0 closed. Used for the full per-sample sequence.
"""
from pathlib import Path
import numpy as np
import mujoco

from scene.membrane import flexcomp_xml

ROOT = Path(__file__).resolve().parents[1]
PANDA_XML = ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

FRIT_X, FRIT_PITCH, N_POS = 0.54, 0.09, 6
FRIT_TOP = 0.10
FRIT_YS = [FRIT_PITCH * (i - (N_POS - 1) / 2) for i in range(N_POS)]
PLATE_XY = (0.30, 0.32)          # work spot for the open plate
AGAR_TOP = 0.012                 # agar surface above the plate base
PLATE_RIM = 0.014                # lid rests here
STACK_IN_XY, STACK_OUT_XY = (0.18, 0.46), (0.48, 0.46)
PLATE_PITCH = 0.0145             # stacked plate height
MAGAZINE_XY, MAGAZINE_TOP = (0.30, -0.34), 0.0326
WASH_XY, WASH_TOP = (0.70, 0.40), 0.062
VESSEL_X, VESSEL_YS = 0.70, np.linspace(-0.18, 0.18, 5)
LID_SPOT = (0.19, 0.30)
HOME_Q = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
TILE = 8                         # contype/conaffinity bit shared by membrane and tiles

STEEL = "0.72 0.74 0.76 1"
FRIT_RGBA = "0.55 0.55 0.52 1"
PP_CLEAR = "0.88 0.9 0.95 0.35"
LIQUID = "0.95 0.82 0.35 0.55"
AGAR = "0.86 0.62 0.38 0.95"


def lookat_xyaxes(pos, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return " ".join(f"{v:.4f}" for v in (*x, *y))


def tile_bodies(prefix, x, y, z_top, radius, solref="0.002 1"):
    """9 thin disc tiles (centre + ring of 8) covering a disc of `radius`, each its own body.

    Collision only (group 3, hidden) so they don't z-fight with the membrane."""
    cells = [(0.0, 0.0, 0.45 * radius)]
    cells += [(0.68 * radius * np.cos(a), 0.68 * radius * np.sin(a), 0.42 * radius)
              for a in np.arange(8) * np.pi / 4]
    return "\n".join(
        f'<body name="{prefix}_t{i}" pos="{x + cx:.4f} {y + cy:.4f} {z_top - 0.0005:.4f}">'
        f'<geom type="cylinder" size="{r:.4f} 0.0005" rgba="0.5 0.5 0.5 1" group="3" '
        f'contype="{TILE}" conaffinity="{TILE}" friction="1.5 0.01 0.0001" solref="{solref}"/></body>'
        for i, (cx, cy, r) in enumerate(cells))


def funnel_geoms(r_in=0.028, h=0.08, wall=0.002, n=20, liquid=None, name=None):
    """Hollow cylinder from n wall boxes, plus a bayonet collar and a liquid column.

    With `name`, the liquid geom is always present (named f"{name}_liquid") so its level can be
    animated by editing geom_size/geom_pos at runtime."""
    r_mid = r_in + wall / 2
    seg = 2 * np.pi * r_mid / n / 2 * 1.08
    g = []
    for k in range(n):
        a = 2 * np.pi * k / n
        g.append(f'<geom type="box" size="{wall/2:.4f} {seg:.4f} {h/2:.4f}" '
                 f'pos="{r_mid*np.cos(a):.4f} {r_mid*np.sin(a):.4f} {h/2:.4f}" '
                 f'quat="{np.cos(a/2):.4f} 0 0 {np.sin(a/2):.4f}" rgba="{PP_CLEAR}" mass="0.002"/>')
    g.append(f'<geom type="cylinder" size="{r_in+0.006:.4f} 0.005" pos="0 0 0.005" '
             f'rgba="0.3 0.35 0.45 1" contype="0" conaffinity="0" mass="0.01"/>')
    g.append(f'<geom type="cylinder" size="{r_in+0.004:.4f} 0.002" pos="0 0 {h:.4f}" '
             f'rgba="0.3 0.35 0.45 1" contype="0" conaffinity="0" mass="0.005"/>')
    if liquid or name:
        lv = max(liquid or 0.0, 0.0005)
        nm = f'name="{name}_liquid" ' if name else ""
        g.append(f'<geom {nm}type="cylinder" size="{r_in-0.0005:.4f} {lv/2:.5f}" pos="0 0 {lv/2+0.001:.5f}" '
                 f'rgba="{LIQUID if liquid else "0.95 0.82 0.35 0"}" contype="0" conaffinity="0" mass="0"/>')
    return "\n".join(g)


def plate_geoms(with_agar=True):
    """Closed-plate base geometry, origin at the plate bottom (static stacks)."""
    g = [f'<geom type="cylinder" size="0.045 0.006" pos="0 0 0.006" rgba="{PP_CLEAR}"/>']
    if with_agar:
        g.append(f'<geom type="cylinder" size="0.043 0.004" pos="0 0 0.006" rgba="{AGAR}" contype="0" conaffinity="0"/>')
    return "".join(g)


def plate_stack_geoms(n):
    g = []
    for k in range(n):
        z0 = k * PLATE_PITCH
        g.append(f'<geom type="cylinder" size="0.045 0.006" pos="0 0 {z0+0.006:.4f}" rgba="{PP_CLEAR}"/>')
        g.append(f'<geom type="cylinder" size="0.043 0.004" pos="0 0 {z0+0.006:.4f}" rgba="{AGAR}" '
                 f'contype="0" conaffinity="0"/>')
        g.append(f'<geom type="cylinder" size="0.047 0.0012" pos="0 0 {z0+0.0133:.4f}" rgba="{PP_CLEAR}"/>')
    return "\n".join(g)


def work_plate_xml(pos, agar_rgba=AGAR):
    """Free plate with agar; its 9 agar tiles are child bodies so they ride along."""
    x, y, z = pos
    tiles = tile_bodies("agar", 0, 0, AGAR_TOP, 0.043, solref="0.005 1")
    return f"""
    <body name="plate" pos="{x:.4f} {y:.4f} {z:.4f}">
      <freejoint name="plate"/>
      <geom name="plate_base" type="cylinder" size="0.045 0.007" pos="0 0 0.007" rgba="0 0 0 0" group="3" mass="0.03"/>
      <geom type="cylinder" size="0.045 0.006" pos="0 0 0.006" rgba="{PP_CLEAR}" contype="0" conaffinity="0" mass="0"/>
      <geom name="agar_visual" type="cylinder" size="0.043 0.0055" pos="0 0 0.0065" rgba="{agar_rgba}" contype="0" conaffinity="0" mass="0"/>
      {tiles}
    </body>"""


def lid_xml(pos):
    x, y, z = pos
    return f"""
    <body name="lid" pos="{x:.4f} {y:.4f} {z:.4f}">
      <freejoint name="lid"/>
      <geom name="lid_top" type="cylinder" size="0.047 0.0012" pos="0 0 0.0088" rgba="{PP_CLEAR}" mass="0.008"/>
      <geom type="cylinder" size="0.0475 0.004" pos="0 0 0.0048" rgba="0.88 0.9 0.95 0.18" contype="0" conaffinity="0" mass="0"/>
    </body>"""


def base_xml(mode="transfer", membrane_offset=(0.0, 0.0), membrane_yaw=0.0, plate_offset=(0.0, 0.0),
             dr=None, filling=(None, 0.06, 0.035, 0.015, 0.004, None), **membrane):
    dr = dr or {}
    fx = FRIT_X
    frits = []
    for i, y in enumerate(FRIT_YS):
        frits.append(f'<body name="frit{i}" pos="{fx} {y:.4f} 0.08">'
                     f'<geom type="cylinder" size="0.025 0.019" rgba="{FRIT_RGBA}"/>'
                     f'<geom type="cylinder" size="0.032 0.003" pos="0 0 0.016" rgba="0.3 0.35 0.45 1"/></body>')
        frits.append(tile_bodies(f"frit{i}", fx, y, FRIT_TOP, 0.025))
        if i > 0 and filling[i] is not None:
            frits.append(f'<body name="funnel{i}" pos="{fx} {y:.4f} 0.099">{funnel_geoms(liquid=filling[i])}</body>')

    agar_rgba = dr.get("agar_rgba", AGAR)
    if mode == "transfer":
        px, py = PLATE_XY[0] + plate_offset[0], PLATE_XY[1] + plate_offset[1]
        plate = work_plate_xml((px, py, 0.0), agar_rgba)
        lid = lid_xml((LID_SPOT[0], LID_SPOT[1], -0.0076))          # upside-down-ish, resting on table
        stack_in = plate_stack_geoms(10)
        mem_pos = (fx + membrane_offset[0], FRIT_YS[0] + membrane_offset[1], FRIT_TOP + 0.0004)
        funnel0 = (WASH_XY[0], WASH_XY[1], WASH_TOP)
    else:
        sx, sy = STACK_IN_XY
        top = 9 * PLATE_PITCH
        plate = work_plate_xml((sx, sy, top), agar_rgba)
        lid = lid_xml((sx, sy, top + PLATE_RIM - 0.0076))
        stack_in = plate_stack_geoms(9)
        mem_pos = (MAGAZINE_XY[0] + membrane_offset[0], MAGAZINE_XY[1] + membrane_offset[1], MAGAZINE_TOP + 0.0004)
        funnel0 = (WASH_XY[0], WASH_XY[1], WASH_TOP)

    vessels = []
    for k, y in enumerate(VESSEL_YS):
        body = (f'<geom type="cylinder" size="0.02 0.04" pos="0 0 0.04" rgba="{PP_CLEAR}" mass="0.02"/>'
                f'<geom name="vessel{k}_liquid" type="cylinder" size="0.0185 0.025" pos="0 0 0.026" rgba="{LIQUID}" '
                f'contype="0" conaffinity="0" mass="0"/>')
        if k == 0:
            vessels.append(f'<body name="vessel0" pos="{VESSEL_X} {y:.3f} 0.03"><freejoint name="vessel0"/>{body}</body>'
                           f'<body name="cap0" pos="{VESSEL_X} {y:.3f} 0.11"><freejoint name="cap0"/>'
                           f'<geom type="cylinder" size="0.0215 0.007" pos="0 0 0.007" rgba="0.75 0.15 0.15 1" mass="0.005"/>'
                           f'<geom type="box" size="0.0225 0.002 0.0055" pos="0 0 0.007" rgba="0.6 0.1 0.1 1" contype="0" conaffinity="0" mass="0"/></body>')
        else:
            vessels.append(f'<body name="vessel{k}" pos="{VESSEL_X} {y:.3f} 0.03">{body}'
                           f'<geom type="cylinder" size="0.0215 0.007" pos="0 0 0.087" rgba="0.75 0.15 0.15 1"/></body>')

    cj = dr.get("cam_jitter", np.zeros(3))
    cams = {
        "overhead": ((0.45, 0.0, 1.25), (0.45, 0.0, 0.0), (1, 0, 0)),
        "oblique": ((1.25, -0.45, 0.80), (0.38, 0.02, 0.17), (0, 0, 1)),
        "front": ((1.45, 0.0, 0.55), (0.40, 0.0, 0.10), (0, 0, 1)),
        "membrane_closeup": ((0.68, FRIT_YS[0] - 0.12, 0.22), (fx, FRIT_YS[0], FRIT_TOP), (0, 0, 1)),
        "plate_side": ((PLATE_XY[0] + 0.005, PLATE_XY[1] - 0.15, 0.06), (PLATE_XY[0], PLATE_XY[1], 0.018), (0, 0, 1)),
        "plate_closeup": ((PLATE_XY[0] + 0.02, PLATE_XY[1] - 0.17, 0.13), (PLATE_XY[0] - 0.01, PLATE_XY[1], 0.02), (0, 0, 1)),
        "plate_top": ((PLATE_XY[0], PLATE_XY[1], 0.32), (PLATE_XY[0], PLATE_XY[1], 0.0), (1, 0, 0)),
        "policy_top": ((0.95 + cj[0], 0.04 + cj[1], 0.80 + cj[2]), (0.40, 0.05, 0.0), (0, 0, 1)),
        "policy_plate": ((PLATE_XY[0] + cj[0] * 0.5, PLATE_XY[1] - 0.13 + cj[1] * 0.5, 0.17 + cj[2] * 0.5),
                         (PLATE_XY[0], PLATE_XY[1], 0.01), (0, 0, 1)),
        "magazine_closeup": ((MAGAZINE_XY[0] + 0.16, MAGAZINE_XY[1] - 0.12, 0.17), (MAGAZINE_XY[0], MAGAZINE_XY[1], 0.03), (0, 0, 1)),
        "block_closeup": ((0.80, -0.30, 0.30), (0.50, -0.20, 0.12), (0, 0, 1)),
    }
    fov = {"plate_top": 30, "policy_top": 52}
    cam_xml = "\n".join(f'<camera name="{n}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" xyaxes="{lookat_xyaxes(p, t, u)}" '
                        f'fovy="{fov.get(n, 45)}"/>' for n, (p, t, u) in cams.items())
    posts = "\n".join(f'<geom type="box" size="0.012 0.012 0.45" pos="{x} {y} 0.45" rgba="0.6 0.62 0.66 1"/>'
                      for x in (-0.22, 0.92) for y in (-0.72, 0.72))
    light_pos = dr.get("light_pos", (0.4, -0.3, 1.6))
    light_diff = dr.get("light_diffuse", 0.6)
    bench = dr.get("bench_rgb", (0.93, 0.93, 0.91))
    return f"""<mujoco model="membrane_cell">
  <option timestep="0.002" integrator="discrete" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="-140" elevation="-25"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.92 0.94 0.97" rgb2="0.6 0.65 0.72" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.78 0.79 0.8" rgb2="0.72 0.73 0.74" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="8 8"/>
    <texture name="bench" type="2d" builtin="flat" rgb1="{bench[0]:.3f} {bench[1]:.3f} {bench[2]:.3f}" width="64" height="64"/>
    <material name="bench" texture="bench" specular="0.3" shininess="0.4"/>
  </asset>
  <worldbody>
    <light pos="{light_pos[0]:.3f} {light_pos[1]:.3f} {light_pos[2]:.3f}" dir="0 0.2 -1" diffuse="{light_diff:.3f} {light_diff:.3f} {light_diff:.3f}" castshadow="true"/>
    <light pos="1.2 0.8 1.2" dir="-0.6 -0.4 -0.6" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.01" pos="0 0 -0.75" material="floor"/>
    <geom name="table" type="box" size="0.58 0.74 0.02" pos="0.35 0 -0.02" material="bench"/>
    <geom type="box" size="0.58 0.74 0.36" pos="0.35 0 -0.40" rgba="0.55 0.57 0.6 1"/>
    {posts}
    <geom type="box" size="0.58 0.012 0.012" pos="0.35 -0.72 0.9" rgba="0.6 0.62 0.66 1"/>
    <geom type="box" size="0.58 0.012 0.012" pos="0.35 0.72 0.9" rgba="0.6 0.62 0.66 1"/>
    <geom type="box" size="0.004 0.72 0.45" pos="-0.22 0 0.45" rgba="0.8 0.88 0.95 0.12" contype="0" conaffinity="0"/>
    {cam_xml}

    <body name="filter_block" pos="{fx} 0 0.04">
      <geom type="box" size="0.05 0.29 0.04" rgba="{STEEL}"/>
      <geom type="cylinder" size="0.008 0.15" pos="-0.06 0 -0.02" quat="0.7071 0.7071 0 0" rgba="0.25 0.25 0.28 1"/>
    </body>
    {''.join(frits)}
    <body name="funnel0" pos="{funnel0[0]:.4f} {funnel0[1]:.4f} {funnel0[2]:.4f}">
      <freejoint name="funnel0"/>
      {funnel_geoms(name="funnel0")}
    </body>

    {plate}
    {lid}
    <body name="stack_in" pos="{STACK_IN_XY[0]} {STACK_IN_XY[1]} 0">{stack_in}</body>
    <body name="stack_out" pos="{STACK_OUT_XY[0]} {STACK_OUT_XY[1]} 0">{plate_stack_geoms(3)}</body>

    <body name="magazine" pos="{MAGAZINE_XY[0]} {MAGAZINE_XY[1]} 0">
      <geom type="cylinder" size="0.03 0.01" pos="0 0 0.01" rgba="{STEEL}"/>
      <geom type="cylinder" size="0.0235 0.006" pos="0 0 0.026" rgba="0.97 0.97 0.95 1"/>
      <geom type="cylinder" size="0.0236 0.0003" pos="0 0 0.0323" rgba="0.35 0.55 0.85 1"/>
      <geom type="box" size="0.002 0.002 0.03" pos="0 0.031 0.03" rgba="{STEEL}"/>
      <geom type="box" size="0.002 0.002 0.03" pos="0 -0.031 0.03" rgba="{STEEL}"/>
    </body>
    {tile_bodies("magazine", MAGAZINE_XY[0], MAGAZINE_XY[1], MAGAZINE_TOP, 0.025)}

    <body name="vessel_rack" pos="{VESSEL_X} 0 0.015"><geom type="box" size="0.035 0.22 0.015" rgba="0.85 0.86 0.88 1"/></body>
    {''.join(vessels)}

    <body name="wash_station" pos="{WASH_XY[0]} {WASH_XY[1]} 0.03">
      <geom type="box" size="0.06 0.06 0.03" rgba="0.5 0.6 0.7 1"/>
      <geom type="cylinder" size="0.03 0.002" pos="0 0 0.031" rgba="0.2 0.25 0.3 1"/>
    </body>

    <body name="waste" pos="0.18 -0.52 0">
      <geom type="box" size="0.07 0.07 0.002" pos="0 0 0.002" rgba="0.3 0.3 0.32 1"/>
      <geom type="box" size="0.07 0.003 0.05" pos="0 0.067 0.05" rgba="0.3 0.3 0.32 1"/>
      <geom type="box" size="0.07 0.003 0.05" pos="0 -0.067 0.05" rgba="0.3 0.3 0.32 1"/>
      <geom type="box" size="0.003 0.07 0.05" pos="0.067 0 0.05" rgba="0.3 0.3 0.32 1"/>
      <geom type="box" size="0.003 0.07 0.05" pos="-0.067 0 0.05" rgba="0.3 0.3 0.32 1"/>
    </body>

    {flexcomp_xml(pos=mem_pos, yaw=membrane_yaw, contype=TILE, **membrane)}
  </worldbody>
</mujoco>"""


TCP_Z = 0.1205       # tweezer tip end, in the hand frame
TIP_GROUP = 2        # contype/conaffinity bit for tips: they never touch anything
MEMBRANE_RINGS = 4


def outer_ring(rings=MEMBRANE_RINGS):
    start = 1 + sum(6 * r for r in range(1, rings))
    return list(range(start, start + 6 * rings))


def panda_spec():
    """Menagerie Panda with tweezer tips, a TCP site at the tip end and a wrist camera."""
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    panda.meshdir = str(PANDA_XML.parent / "assets")
    # tweezer tips: 12 mm capsules past the finger pads; pads stop colliding (plan section 7)
    for side in ("left_finger", "right_finger"):
        b = panda.body(side)
        for g in b.geoms:
            if g.classname.name.startswith("fingertip_pad"):
                g.contype, g.conaffinity = 0, 0
        b.add_geom(name=f"{side}_tip", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0015, 0, 0],
                   fromto=[0, 0.0025, 0.050, 0, 0.0025, 0.062], rgba=[0.15, 0.15, 0.15, 1],
                   friction=[1.5, 0.01, 0.0001], contype=TIP_GROUP, conaffinity=TIP_GROUP)
    # a real Franka compensates gravity in its controller; without this the servos sag 3-7 mm
    for b in panda.bodies:
        b.gravcomp = 1
    hand = panda.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_Z], size=[0.003, 0, 0], rgba=[1, 0, 0, 1], group=4)
    xyaxes = [float(v) for v in lookat_xyaxes((0.09, 0, -0.01), (0, 0, 0.17), up=(1, 0, 0)).split()]
    hand.add_camera(name="wrist", pos=[0.09, 0, -0.01], xyaxes=xyaxes, fovy=70)
    return panda


def build(**cfg):
    spec = mujoco.MjSpec.from_string(base_xml(**cfg))
    frame = spec.worldbody.add_frame(pos=[0, 0, 0])
    frame.attach_body(panda_spec().body("link0"), "panda/", "")
    # grasp by attachment (plan section 4.3): one inactive connect per vertex of the outer two
    # rings; the expert clamps a small edge patch (fixes the sheet's slope like tweezer jaws, not
    # a hinge) and writes the anchors into eq_data at grasp time
    for k in range(outer_ring(MEMBRANE_RINGS - 1)[0], outer_ring()[-1] + 1):
        spec.add_equality(name=f"grip{k}", type=mujoco.mjtEq.mjEQ_CONNECT,
                          objtype=mujoco.mjtObj.mjOBJ_BODY, name1="panda/hand",
                          name2=f"membrane_{k}", active=False, solref=[0.004, 1])
    return spec


def home(m, d, q=HOME_Q):
    for i, qi in enumerate(q, start=1):
        d.qpos[m.joint(f"panda/joint{i}").qposadr[0]] = qi
        d.ctrl[m.actuator(f"panda/actuator{i}").id] = qi
    for f in ("finger_joint1", "finger_joint2"):
        d.qpos[m.joint(f"panda/{f}").qposadr[0]] = 0.04
    d.ctrl[m.actuator("panda/actuator8").id] = 255
    mujoco.mj_forward(m, d)


def load(q0=HOME_Q, **cfg):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attach conflict")
        m = build(**cfg).compile()
    d = mujoco.MjData(m)
    home(m, d, q0)
    return m, d

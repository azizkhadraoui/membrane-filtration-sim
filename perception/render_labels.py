"""Synthetic segmentation dataset for Phase D perception.

Renders RGB + per-pixel class labels from the MuJoCo cell with heavy domain randomisation:
free cameras around the frit block / plate (close, oblique, "person filming a bench"),
lights, agar / bench / funnel / liquid / membrane colours, membrane + plate pose, and
membrane states (on frit, under a funnel, on agar, mid-transfer from the scripted expert,
in the magazine of the `sequence` layout).

    python perception/render_labels.py                 # full dataset (4 worker processes)
    python perception/render_labels.py --preview       # one small scene of each kind

Output: perception/data/seg/<kind>_<seed>.npz with img (N,S,S,3) uint8 and lbl (N,S,S) uint8.

Labelling rule: first visible surface. MuJoCo's segmentation pass draws translucent geoms
opaque, so agar seen through a closed lid is "lid", a membrane under a funnel is "funnel".
Hidden collision tiles (group 3) are never drawn and never labelled.
"""
import argparse
import re
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

CLASSES = ["background", "membrane", "frit", "funnel", "plate", "lid", "agar"]
OUT = ROOT / "perception" / "data" / "seg"
S = 256

AGAR_PALETTE = np.array([
    [0.86, 0.62, 0.38],   # sim default (amber)
    [0.55, 0.30, 0.55],   # purple (e.g. TGE / chromogenic) - seen in the client video
    [0.90, 0.82, 0.45],   # yellow / straw - seen in the client video
    [0.75, 0.20, 0.20],   # blood / red media
    [0.90, 0.70, 0.75],   # pink
    [0.80, 0.80, 0.70],   # pale / plate count
    [0.35, 0.25, 0.20],   # dark media
    [0.55, 0.65, 0.40],   # greenish
])
LIQUID_PALETTE = np.array([[0.95, 0.82, 0.35], [0.55, 0.80, 0.95], [0.85, 0.9, 0.95], [0.95, 0.75, 0.80],
                           [0.45, 0.75, 0.85]])


# --------------------------------------------------------------------------------------------
def geom_classes(m):
    """Class id per geom (first visible surface rule). Unnamed geoms are classified by body."""
    import mujoco
    cls = np.zeros(m.ngeom, np.uint8)
    groups = dict(dish=[], agar=[], lid=[], funnel_wall=[], funnel_collar=[], liquid=[], frit=[], block=[])
    for g in range(m.ngeom):
        if m.geom_group[g] >= 3:
            continue
        b = m.body(m.geom_bodyid[g]).name
        gname = m.geom(g).name
        size = m.geom_size[g]
        if re.fullmatch(r"frit\d", b):
            cls[g] = 2; groups["frit"].append(g)
        elif b == "filter_block":
            cls[g] = 2; groups["block"].append(g)
        elif re.fullmatch(r"funnel\d", b):
            cls[g] = 3
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                groups["funnel_wall"].append(g)
            elif gname.endswith("_liquid") or abs(size[0] - 0.0275) < 1e-4:
                groups["liquid"].append(g)
            else:
                groups["funnel_collar"].append(g)
        elif b == "plate":
            if gname == "agar_visual":
                cls[g] = 6; groups["agar"].append(g)
            else:
                cls[g] = 4; groups["dish"].append(g)
        elif b == "lid":
            cls[g] = 5; groups["lid"].append(g)
        elif b in ("stack_in", "stack_out"):
            r = size[0]
            if abs(r - 0.043) < 1e-4:
                cls[g] = 6; groups["agar"].append(g)
            elif abs(r - 0.047) < 1e-4:
                cls[g] = 5; groups["lid"].append(g)
            else:
                cls[g] = 4; groups["dish"].append(g)
    return cls, {k: np.array(v, int) for k, v in groups.items()}


def fix_plate_visual(m):
    """The work plate's dish visual and agar tops are coplanar (z = 12 mm) and z-fight; lower the
    dish visual top by 1 mm. Visual-only geom (contype 0), physics unchanged."""
    pid = m.body("plate").id
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == pid and m.geom_group[g] < 3 and m.geom(g).name != "agar_visual":
            if abs(m.geom_size[g, 0] - 0.045) < 1e-4:
                m.geom_size[g, 2] = 0.0055
                m.geom_pos[g, 2] = 0.0055


def jitter(rng, c, a):
    return np.clip(np.asarray(c) + rng.uniform(-a, a, 3), 0, 1)


def randomize_appearance(m, rng, groups, lights=True):
    """Runtime domain randomisation on the in-memory model (no reload)."""
    import mujoco
    gr = m.geom_rgba
    agar = jitter(rng, AGAR_PALETTE[rng.integers(len(AGAR_PALETTE))], 0.08)
    gr[groups["agar"], :3] = agar
    gr[groups["agar"], 3] = rng.uniform(0.85, 1.0)
    pp = jitter(rng, [0.9, 0.92, 0.95], 0.08)
    gr[groups["dish"], :3] = pp; gr[groups["dish"], 3] = rng.uniform(0.2, 0.6)
    gr[groups["lid"], :3] = pp; gr[groups["lid"], 3] = rng.uniform(0.12, 0.45)
    wall = jitter(rng, [0.9, 0.9, 0.92], 0.1)
    gr[groups["funnel_wall"], :3] = wall; gr[groups["funnel_wall"], 3] = rng.uniform(0.25, 0.85)
    collar = rng.choice([[0.3, 0.35, 0.45], [0.92, 0.92, 0.92], [0.6, 0.6, 0.62], [0.2, 0.3, 0.7]])
    gr[groups["funnel_collar"], :3] = jitter(rng, collar, 0.06)
    liq = jitter(rng, LIQUID_PALETTE[rng.integers(len(LIQUID_PALETTE))], 0.08)
    vis = gr[groups["liquid"], 3] > 0
    gr[groups["liquid"], :3] = liq
    gr[groups["liquid"][vis], 3] = rng.uniform(0.25, 0.75)
    gr[groups["frit"], :3] = jitter(rng, rng.choice([[0.55, 0.55, 0.52], [0.75, 0.75, 0.75], [0.35, 0.37, 0.42]]), 0.08)
    gr[groups["block"], :3] = jitter(rng, [0.72, 0.74, 0.76], 0.15)
    # membrane: white cellulose ester, sometimes grey / tinted (gridded, stained)
    mem = np.full(3, rng.uniform(0.82, 1.0))
    if rng.uniform() < 0.15:
        mem = jitter(rng, mem * [0.85, 0.95, 1.0], 0.08)
    m.flex_rgba[0, :3] = mem
    # bench
    bid = m.mat("bench").id
    tone = rng.choice([[0.93, 0.93, 0.91], [0.75, 0.77, 0.8], [0.6, 0.62, 0.65], [0.85, 0.82, 0.75], [0.45, 0.47, 0.5]])
    m.mat_rgba[bid, :3] = jitter(rng, tone, 0.06)
    if lights:
        m.light_pos[0] = [0.4 + rng.uniform(-0.6, 0.6), -0.3 + rng.uniform(-0.6, 0.6), 1.6 + rng.uniform(-0.6, 0.3)]
        m.light_diffuse[0] = rng.uniform(0.2, 0.95)
        m.light_diffuse[1] = rng.uniform(0.0, 0.45)
        m.vis.headlight.ambient[:] = rng.uniform(0.12, 0.5)
        m.vis.headlight.diffuse[:] = rng.uniform(0.15, 0.6)
        m.vis.global_.fovy = rng.uniform(35, 65)


def random_camera(rng, target, dist=(0.12, 0.5)):
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(target) + rng.normal(0, [0.02, 0.02, 0.008])
    lo, hi = dist
    cam.distance = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    cam.azimuth = float(rng.uniform(0, 360))
    cam.elevation = float(-rng.uniform(15, 88)) if rng.uniform() < 0.85 else float(-rng.uniform(5, 15))
    return cam


class Shooter:
    def __init__(self, m, size=S):
        import mujoco
        self.m = m
        self.ren = mujoco.Renderer(m, size, size)
        self.cls, self.groups = geom_classes(m)
        self.imgs, self.lbls = [], []

    def shot(self, d, cam, min_fg=0.03):
        import mujoco
        self.ren.disable_segmentation_rendering()
        self.ren.update_scene(d, camera=cam)
        rgb = self.ren.render().copy()
        self.ren.enable_segmentation_rendering()
        self.ren.update_scene(d, camera=cam)
        seg = self.ren.render()
        self.ren.disable_segmentation_rendering()
        oid, oty = seg[..., 0], seg[..., 1]
        lbl = np.zeros(oid.shape, np.uint8)
        g = oty == mujoco.mjtObj.mjOBJ_GEOM
        lbl[g] = self.cls[oid[g]]
        lbl[oty == mujoco.mjtObj.mjOBJ_FLEX] = 1
        if (lbl > 0).mean() < min_fg:            # nothing of interest in view
            return False
        self.imgs.append(rgb); self.lbls.append(lbl)
        return True

    def save(self, path):
        if self.imgs:
            np.savez_compressed(path, img=np.stack(self.imgs), lbl=np.stack(self.lbls))
        self.ren.close()
        return len(self.imgs)


# --------------------------------------------------------------------------------------------
def settle(m, d, n=300):
    import mujoco
    for _ in range(n):
        mujoco.mj_step(m, d)


def set_free(m, d, joint, xyz, quat=None):
    a = m.joint(joint).qposadr[0]
    d.qpos[a:a + 3] = xyz
    if quat is not None:
        d.qpos[a + 3:a + 7] = quat
    d.qvel[m.joint(joint).dofadr[0]:m.joint(joint).dofadr[0] + 6] = 0


def set_liquid(m, d, level, name="funnel0_liquid", rgba_alpha=0.55):
    g = m.geom(name).id
    lv = max(level, 0.0005)
    m.geom_size[g, 1] = lv / 2
    m.geom_pos[g, 2] = lv / 2 + 0.001
    m.geom_rgba[g, 3] = rgba_alpha if level > 0.001 else 0.0


def move_membrane(m, d, target_xy, z, yaw):
    """Teleport all flex vertices (3 slide joints each) to a disc at target_xy, height z."""
    import mujoco
    mujoco.mj_forward(m, d)
    v = d.flexvert_xpos.copy()
    c = v.mean(0)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    new = v.copy()
    new[:, :2] = (v[:, :2] - c[:2]) @ R.T + target_xy
    new[:, 2] = v[:, 2] - c[2] + z
    for k, bid in enumerate(m.flex_vertbodyid[m.flex_vertadr[0]:m.flex_vertadr[0] + m.flex_vertnum[0]]):
        ja = m.body_jntadr[bid]
        for i in range(3):
            d.qpos[m.jnt_qposadr[ja + i]] += new[k, i] - v[k, i]
            d.qvel[m.jnt_dofadr[ja + i]] = 0
    mujoco.mj_forward(m, d)


def load_quiet(**cfg):
    from scene.cell import load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load(**cfg)


def scene_targets(m, d):
    import mujoco
    mujoco.mj_forward(m, d)
    mem = d.flexvert_xpos.mean(0)
    plate = d.body("plate").xpos + [0, 0, 0.012]
    frits = [d.body(f"frit{i}").xpos + [0, 0, 0.02] for i in range(6)]
    return mem, plate, frits


def views(rng, sh, m, d, n, weights):
    """Render n random views; weights over targets (membrane, plate, frit, lid, block, stack)."""
    import mujoco
    mem, plate, frits = scene_targets(m, d)
    tg = dict(membrane=mem, plate=plate, lid=d.body("lid").xpos + [0, 0, 0.005],
              block=np.array([0.54, rng.uniform(-0.25, 0.25), 0.09]),
              stack=d.body("stack_in").xpos + [0, 0, 0.08], frit=None)
    keys = list(weights)
    p = np.array([weights[k] for k in keys], float); p /= p.sum()
    done, tries = 0, 0
    while done < n and tries < 3 * n:
        tries += 1
        randomize_appearance(m, rng, sh.groups)
        mujoco.mj_forward(m, d)
        k = keys[rng.choice(len(keys), p=p)]
        t = frits[rng.integers(6)] if k == "frit" else tg[k]
        dist = (0.25, 0.8) if k in ("block", "stack") else (0.1, 0.45)
        done += sh.shot(d, random_camera(rng, t, dist))
    return done


# --------------------------------------------------------------------------------------------
def gen(job):
    kind, seed = job
    path = OUT / f"{kind}_{seed:04d}.npz"
    if path.exists():
        return kind, seed, -1, 0.0
    import mujoco
    t0 = time.time()
    rng = np.random.default_rng([seed, len(kind), ord(kind[0])])
    disc = lambda r: (lambda a, rr: (rr * np.cos(a), rr * np.sin(a)))(rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform()))
    cfg = dict(mode="sequence" if kind == "sequence" else "transfer",
               membrane_offset=disc(0.008), membrane_yaw=float(rng.uniform(0, 2 * np.pi)),
               plate_offset=disc(0.03))
    if kind == "transfer":
        from transfer.randomize import sample
        cfg, expert = sample(seed, perturb=bool(seed % 2))
    m, d = load_quiet(**cfg)
    fix_plate_visual(m)
    sh = Shooter(m)
    n_views = 28

    if kind == "frit":
        r = rng.uniform()
        if r < 0.3:      # filtration in progress: funnel 0 locked on frit 0 over the membrane
            set_free(m, d, "funnel0", [*d.body("frit0").xpos[:2], 0.099], [1, 0, 0, 0])
            set_liquid(m, d, rng.uniform(0, 0.06))
        elif r < 0.5:    # funnel 0 on another frit
            set_free(m, d, "funnel0", [*d.body(f"frit{rng.choice([5])}").xpos[:2], 0.099], [1, 0, 0, 0])
            set_liquid(m, d, rng.uniform(0, 0.06))
        settle(m, d, 300)
        views(rng, sh, m, d, n_views, dict(membrane=5, frit=2, block=1.5, plate=1, lid=0.4))
    elif kind == "agar":
        settle(m, d, 50)
        plate_xy = d.body("plate").xpos[:2]
        off = np.array(disc(0.004 if rng.uniform() < 0.7 else 0.02))
        move_membrane(m, d, plate_xy + off, d.body("plate").xpos[2] + 0.0135, rng.uniform(0, 2 * np.pi))
        if rng.uniform() < 0.12:   # closed plate: lid on top
            set_free(m, d, "lid", [*plate_xy, 0.014 - 0.0076 + 0.0005], [1, 0, 0, 0])
        if rng.uniform() < 0.3:
            set_free(m, d, "funnel0", [*d.body("frit0").xpos[:2], 0.099], [1, 0, 0, 0])
            set_liquid(m, d, rng.uniform(0, 0.05))
        settle(m, d, 300)
        views(rng, sh, m, d, n_views, dict(membrane=5, plate=2, lid=1, frit=1, block=0.5))
    elif kind == "sequence":
        settle(m, d, 300)
        views(rng, sh, m, d, n_views, dict(membrane=3, stack=3, frit=1.5, block=1, plate=2))
    elif kind == "transfer":
        from control.primitives import Robot
        from transfer.expert import transfer
        randomize_appearance(m, rng, sh.groups)
        mujoco.mj_forward(m, d)
        robot = Robot(m, d)
        state = dict(n=0)

        def hook(rb):
            state["n"] += 1
            if state["n"] % 70:
                return
            randomize_appearance(m, rng, sh.groups, lights=False)
            mem = d.flexvert_xpos.mean(0)
            if not np.isfinite(mem).all():
                return
            t = mem if rng.uniform() < 0.75 else d.body("plate").xpos + [0, 0, 0.012]
            sh.shot(d, random_camera(rng, t, (0.12, 0.5)))
        robot.wait(0.5)
        robot.hooks.append(hook)
        try:
            transfer(robot, **{k: v for k, v in expert.items() if k != "perturb_mode"})
        except Exception as e:
            print("expert failed", seed, repr(e)[:100], flush=True)
        robot.hooks.clear()
        if np.isfinite(d.flexvert_xpos).all():
            views(rng, sh, m, d, 20, dict(membrane=5, plate=2, frit=1))
    n = sh.save(path)
    return kind, seed, n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--n_frit", type=int, default=64)
    ap.add_argument("--n_agar", type=int, default=48)
    ap.add_argument("--n_seq", type=int, default=16)
    ap.add_argument("--n_transfer", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.preview:
        jobs = [("frit", 9000), ("agar", 9000), ("sequence", 9000)]
    else:
        jobs = ([("transfer", 100 + i) for i in range(a.n_transfer)] + [("frit", i) for i in range(a.n_frit)]
                + [("agar", i) for i in range(a.n_agar)] + [("sequence", i) for i in range(a.n_seq)])
    t0 = time.time()
    total = 0
    with Pool(a.procs) as pool:
        for kind, seed, n, dt in pool.imap_unordered(gen, jobs):
            total += max(n, 0)
            print(f"{kind:9s} {seed:5d} frames {n:4d}  {dt:6.1f}s  total {total}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

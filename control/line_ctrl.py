"""Controller for the station-based line (scene/line.py).

- Stations are animated by generator "processes" that are ticked every physics step (Robot hook).
- The arm only does: dry membrane magazine -> frit, wet membrane frit -> agar (roll-on), tip wash.
- One flex membrane stands in for every membrane: it is teleported between stations and swapped with a rigid
  disc whenever the membrane is not being handled (see swap_*). The wet transfer runs the real flex.
- A single "zone" lock keeps the arm and the doser out of each other's way at the manifold.

    python control/line_ctrl.py --samples 2 --filt 20        # quick test
    python control/line_ctrl.py                              # full run -> results/line_*.mp4, line_run.json
"""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco

from scene.line import (load, FRIT_X, FRIT_YS, FRIT_TOP, N_POS, N_SAMPLES, MAG, SPOT, LIDPARK, HOTEL, OUT,
                        JUNCTION_X, TIPWASH, VESSEL_X, VESSEL_YS, TIPRACK, WASTE, NOZ_X0, NOZ_Z0, WATER, SANI)
from scene.cell import AGAR_TOP, PLATE_PITCH, MAGAZINE_TOP, ROOT, LIQUID
from control.primitives import Robot, Recorder, min_jerk, down_rotation
from transfer.expert import transfer

PLATE_Z = lambda k: (N_SAMPLES - 1 - k) * PLATE_PITCH + 0.002
FUNNEL_UP = 0.40
LIFT_T = 2.0
LED = dict(idle=(0.25, 0.75, 0.35), clamp=(0.95, 0.85, 0.2), filter=(0.25, 0.5, 0.95), ready=(0.95, 0.55, 0.1),
           cip=(0.2, 0.85, 0.9), error=(0.9, 0.2, 0.2), done=(0.5, 0.5, 0.5))
STATE_LABEL = dict(idle="idle", wait_load="loading", loaded="clamping", wait_dose="queued", dosing="dosing",
                   filtering="filtering", ready="ready", wet="transfer", cip="cleaning", error="fault")


class Proc:
    def __init__(self, gen, name):
        self.gen, self.name, self.wake, self.cond = gen, name, 0.0, None


def rgba_str(rgb, a):
    return np.array([*rgb, a], dtype=np.float32)


class Line:
    def __init__(self, m, d, robot, n_samples=N_SAMPLES, filt=(70.0, 110.0), seed=0, prep_lead=25.0, arm_speed=2.0):
        self.m, self.d, self.robot = m, d, robot
        self.dt = m.opt.timestep
        self.rng = np.random.default_rng(seed)
        self.n_samples, self.prep_lead = n_samples, prep_lead
        self.arm_speed = arm_speed
        self.procs = []
        self.zone = None
        self.events, self.jobs = [], []
        self.t_start = 0.0
        self.verbose = True

        # driven joints
        self.jq = {}
        for nm in [f"flift{i}" for i in range(N_POS)] + ["sx", "sy", "sz", "lgx", "lgz", "dy", "dx", "dz"]:
            j = m.joint(nm)
            self.jq[nm] = (int(j.qposadr[0]), int(j.dofadr[0]))
        self.jhold = {nm: float(d.qpos[qa]) for nm, (qa, _) in self.jq.items()}
        # kinematic bodies (plates, lids are mocap bodies)
        self.fq = {}
        self.fhold = {}
        for k in range(N_SAMPLES):
            for nm in (f"plate{k}", f"lid{k}"):
                mid = int(m.body_mocapid[m.body(nm).id])
                self.fq[nm] = mid
                self.fhold[nm] = np.concatenate([d.mocap_pos[mid], d.mocap_quat[mid]]).copy()
        self.carry = []                                    # [(body name, offset xyz)] riding on the shuttle
        self.cup_carry = None                              # lid riding on the lid-lifter cup
        self.jhold["flift0"] = 0.0
        for i in range(N_POS):
            self.jhold[f"flift{i}"] = 0.0
        self.jhold["dx"] = 0.0
        self.jhold["dz"] = 0.0
        self.jhold["lgz"] = -0.05

        self.grip_eqs = [int(m.equality(nm).id) for nm in [m.equality(i).name for i in range(m.neq)] if nm.startswith("grip")]
        # flex membrane bookkeeping
        f0 = 0
        nv = int(m.flex_vertnum[0])
        vb = m.flex_vertbodyid[int(m.flex_vertadr[0]):int(m.flex_vertadr[0]) + nv]
        self.vbody = [int(b) for b in vb]
        self.vq = [int(m.jnt_qposadr[m.body_jntadr[b]]) for b in self.vbody]
        self.vd = [int(m.jnt_dofadr[m.body_jntadr[b]]) for b in self.vbody]
        self.rest = None                                   # flat pose on the magazine (set in begin)
        self.mem_xyz = {}                                  # frit i -> stored settled vertex positions
        self.mem_left = n_samples

        # samples and positions
        self.samples = [dict(id=s, vessel=s, plate=s, vol=float(self.rng.uniform(8, 40)),
                             filt=float(self.rng.uniform(*filt)), t={}) for s in range(n_samples)]
        self.next_sample = 0
        self.P = [dict(state="idle", sample=None, filt_frac=0.0, plate_req=False, plate_ready=False) for _ in range(N_POS)]
        self.dose_q, self.plate_in_q, self.plate_out_q = [], [], []
        self.spot_owner = None
        self.out_count = 0
        self.tips_used = 0
        self.cip_cycles = 0
        self.water, self.sani = 1.0, 1.0
        self.arm_busy_s = 0.0
        self.arm_label = ""
        self.plates_taken = 0
        # state trace for offline rendering: everything the renderer needs is recorded at trace_dt
        self.trace_dt, self.next_trace = 0.1, 0.0
        self.trace_q, self.trace_mp, self.trace_mq, self.trace_g, self.trace_hud = [], [], [], [], []
        names = [m.geom(i).name for i in range(m.ngeom)]
        self.trace_gids = [i for i, nm in enumerate(names) if nm and nm.startswith(
            ("led", "funnel", "vessel", "fm", "pm", "ms", "tip", "dtip", "tw_", "tank_", "dshaft"))]
        self.results = []
        self.done_samples = 0
        self.m.geom_rgba[m.geom("dtip").id][3] = 0.0
        for i in range(N_POS):
            self.set_led(i, "idle")

    # ------------------------------------------------------------------ process machinery
    def spawn(self, gen, name=""):
        self.procs.append(Proc(gen, name))

    def tick(self, robot=None):
        t = self.d.time
        for p in list(self.procs):
            if p.cond is not None:
                if not p.cond():
                    continue
                p.cond = None
            elif t < p.wake - 1e-9:
                continue
            try:
                y = next(p.gen)
            except StopIteration:
                self.procs.remove(p)
                continue
            if callable(y):
                p.cond = y
            else:
                p.wake = t + float(y)
        d = self.d
        for nm, (qa, da) in self.jq.items():
            d.qpos[qa] = self.jhold[nm]
            d.qvel[da] = 0.0
        self._follow()
        for nm, mid in self.fq.items():
            d.mocap_pos[mid] = self.fhold[nm][:3]
            d.mocap_quat[mid] = self.fhold[nm][3:]
        if self.trace_dt and d.time >= self.next_trace:
            self.next_trace += self.trace_dt
            self.snap()
        # doser shaft spans from the tip up to the beam
        gid = self.m.geom("dshaft").id
        tip_z = NOZ_Z0 + self.jhold["dz"]
        half = max((0.68 - tip_z) / 2, 0.01)
        self.m.geom_size[gid][1] = half
        self.m.geom_pos[gid][2] = half

    def hud_state(self):
        return dict(t=self.d.time - self.t_start, out=self.out_count, n=self.n_samples, arm_busy=self.arm_busy_s,
                    cip=self.cip_cycles, mem_left=self.mem_left, plates_taken=self.plates_taken, tips_used=self.tips_used,
                    sani=self.sani, water=self.water, arm_label=self.arm_label,
                    P=[dict(state=P["state"], sample=None if P["sample"] is None else P["sample"]["id"],
                            filt=P["filt_frac"]) for P in self.P])

    def snap(self):
        m, d = self.m, self.d
        g = self.trace_gids
        self.trace_q.append(d.qpos.astype(np.float32).copy())
        self.trace_mp.append(d.mocap_pos.astype(np.float32).copy())
        self.trace_mq.append(d.mocap_quat.astype(np.float32).copy())
        self.trace_g.append(np.concatenate([m.geom_rgba[g].ravel(), m.geom_size[g].ravel(), m.geom_pos[g].ravel(),
                                            [m.flex_rgba[0][3]]]).astype(np.float32))
        self.trace_hud.append(self.hud_state())

    def save_trace(self, path):
        np.savez_compressed(path, q=np.array(self.trace_q), mp=np.array(self.trace_mp), mq=np.array(self.trace_mq),
                            g=np.array(self.trace_g), gids=np.array(self.trace_gids), dt=self.trace_dt)
        Path(str(path).replace(".npz", "_hud.json")).write_text(json.dumps(self.trace_hud))

    def _follow(self):
        sx, sy, sz = self.jhold["sx"], self.jhold["sy"], self.jhold["sz"]
        for nm, off in self.carry:
            self.fhold[nm][:3] = (HOTEL[0] + sx + off[0], HOTEL[1] + sy + off[1], sz + off[2])
            self.fhold[nm][3:] = (1, 0, 0, 0)
        if self.cup_carry:
            lx = LIDPARK[0] + self.jhold["lgx"]
            cup_bottom = 0.164 + self.jhold["lgz"]
            self.fhold[self.cup_carry][:3] = (lx, SPOT[1], cup_bottom - 0.01)
            self.fhold[self.cup_carry][3:] = (1, 0, 0, 0)

    def move(self, targets, T):
        """Smooth simultaneous move of driven joints. targets: dict name -> value."""
        v0 = {n: self.jhold[n] for n in targets}
        n = max(1, int(T / self.dt))
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            for nm, v in targets.items():
                self.jhold[nm] = v0[nm] + (v - v0[nm]) * s
            yield 0

    def event(self, kind, **kw):
        self.events.append(dict(t=round(self.d.time - self.t_start, 2), kind=kind, **kw))

    # ------------------------------------------------------------------ small helpers
    def set_led(self, i, st):
        self.m.geom_rgba[self.m.geom(f"led{i}").id] = rgba_str(LED[st], 1.0)

    def set_state(self, i, st, led=None):
        self.P[i]["state"] = st
        if led:
            self.set_led(i, led)

    def liquid(self, name, h, rgba=None, base=0.0015):
        gid = self.m.geom(name).id
        h = max(h, 0.0005)
        self.m.geom_size[gid][1] = h / 2
        self.m.geom_pos[gid][2] = base + h / 2
        if rgba is not None:
            if isinstance(rgba, str):
                rgba = [float(x) for x in rgba.split()]
            self.m.geom_rgba[gid] = np.array(rgba, dtype=np.float32)

    def hide(self, name, hide=True, rgb=None):
        gid = self.m.geom(name).id
        self.m.geom_rgba[gid][3] = 0.0 if hide else (1.0 if rgb is None else 1.0)

    def flex_alpha(self, a):
        self.m.flex_rgba[0][3] = a

    def flex_set(self, P):
        for v, (b, qa, da) in enumerate(zip(self.vbody, self.vq, self.vd)):
            self.d.qpos[qa:qa + 3] = P[v] - self.m.body_pos[b]
            self.d.qvel[da:da + 3] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def flex_xyz(self):
        return self.d.flexvert_xpos.copy()

    def nozzle_goto(self, x, y, z, speed=0.5, zspeed=0.35):
        """Cartesian move of the doser nozzle tip (world coords)."""
        tgt = dict(dy=y, dx=x - NOZ_X0, dz=z - NOZ_Z0)
        cur = np.array([NOZ_X0 + self.jhold["dx"], self.jhold["dy"], NOZ_Z0 + self.jhold["dz"]])
        dxy = np.hypot(x - cur[0], y - cur[1])
        T = max(dxy / speed * 1.875, abs(z - cur[2]) / zspeed * 1.875, 0.4)
        yield from self.move(tgt, T)

    # ------------------------------------------------------------------ doser
    def doser_loop(self):
        TRAVEL = 0.40
        tip_j = 0
        while True:
            yield lambda: bool(self.dose_q)
            i = self.dose_q.pop(0)
            s = self.P[i]["sample"]
            t0 = self.d.time
            self.set_state(i, "dosing", "clamp")
            # 1. take a disposable tip
            rx = TIPRACK[0] + 0.012 * (tip_j % 4 - 1.5)
            ry = TIPRACK[1] + 0.012 * (tip_j // 4 - 0.5)
            yield from self.nozzle_goto(rx, ry, TRAVEL)
            yield from self.nozzle_goto(rx, ry, 0.075)
            self.hide(f"tip{tip_j}")
            self.m.geom_rgba[self.m.geom("dtip").id][3] = 0.95
            tip_j += 1
            self.tips_used += 1
            yield 0.3
            yield from self.nozzle_goto(rx, ry, TRAVEL)
            # 2. aspirate through the septum
            vx, vy = VESSEL_X, VESSEL_YS[s["vessel"]]
            yield from self.nozzle_goto(vx, vy, TRAVEL)
            yield from self.nozzle_goto(vx, vy, 0.135, zspeed=0.3)
            h0 = 0.05
            n = 60
            for q in range(n + 1):
                self.liquid(f"vessel{s['vessel']}_liquid", h0 * (1 - q / n) + 0.004 * q / n, base=0.001)
                yield 0.05
            yield from self.nozzle_goto(vx, vy, TRAVEL)
            # 3. visit the funnel (needs the manifold zone)
            yield lambda: self.zone is None
            self.zone = "doser"
            fy = FRIT_YS[i]
            yield from self.nozzle_goto(FRIT_X, fy, TRAVEL)
            yield from self.nozzle_goto(FRIT_X, fy, 0.215, zspeed=0.3)
            hv = 0.015 + 0.0012 * s["vol"]
            n = 60
            for q in range(n + 1):
                self.liquid(f"funnel{i}_liquid", hv * q / n, LIQUID)
                yield 0.05
            yield from self.nozzle_goto(FRIT_X, fy, TRAVEL)
            self.zone = None
            self.P[i]["hv"] = hv
            self.set_state(i, "filtering", "filter")
            s["t"]["dosed"] = round(self.d.time - self.t_start, 1)
            self.spawn(self.filter_proc(i), f"filter{i}")
            # 4. eject tip into the waste and go home
            yield from self.nozzle_goto(WASTE[0], WASTE[1], TRAVEL)
            yield from self.nozzle_goto(WASTE[0], WASTE[1], 0.16)
            self.m.geom_rgba[self.m.geom("dtip").id][3] = 0.0
            yield 0.3
            yield from self.nozzle_goto(NOZ_X0, 0.0, NOZ_Z0 - 0.02)
            self.jobs.append(dict(kind="dose", pos=i, sample=s["id"], t0=round(t0 - self.t_start, 1), dur=round(self.d.time - t0, 1)))

    # ------------------------------------------------------------------ filtration, funnels, CIP
    def filter_proc(self, i):
        s = self.P[i]["sample"]
        T, hv = s["filt"], self.P[i]["hv"]
        t0 = self.d.time
        while True:
            frac = (self.d.time - t0) / T
            if frac >= 1:
                break
            self.P[i]["filt_frac"] = frac
            self.liquid(f"funnel{i}_liquid", hv * (1 - frac) ** 1.4, LIQUID)
            if not self.P[i]["plate_req"] and T - (self.d.time - t0) <= self.prep_lead:
                self.P[i]["plate_req"] = True
                self.plate_in_q.append(i)
            yield 0.2
        self.P[i]["filt_frac"] = 1.0
        self.liquid(f"funnel{i}_liquid", 0.0005, LIQUID)
        if not self.P[i]["plate_req"]:
            self.P[i]["plate_req"] = True
            self.plate_in_q.append(i)
        s["t"]["filtered"] = round(self.d.time - self.t_start, 1)
        self.set_state(i, "ready", "ready")

    def funnel_after_arm(self, i, t_job0):
        """Lower funnel i once the arm has left the manifold."""
        yield lambda: self.robot.tcp_pos()[0] < 0.45 and self.d.time > t_job0 + 3.0
        yield from self.move({f"flift{i}": 0.0}, LIFT_T)

    def after_dry(self, i):
        yield 0.0
        self.set_state(i, "loaded", "clamp")
        yield lambda: abs(self.jhold[f"flift{i}"]) < 1e-6
        self.set_state(i, "wait_dose", "clamp")
        self.dose_q.append(i)

    def cip_proc(self, i):
        t0 = self.d.time
        s = self.P[i]["sample"]
        self.set_state(i, "cip", "cip")
        yield lambda: abs(self.jhold[f"flift{i}"]) < 1e-6
        name = f"funnel{i}_liquid"
        for color, label, hold in ((WATER, "water", 3.0), (SANI, "sanitant", 5.0), (WATER, "water", 2.0)):
            n = 30
            for q in range(n + 1):
                self.liquid(name, 0.035 * q / n, rgba=None)
                self.m.geom_rgba[self.m.geom(name).id] = np.array([float(x) for x in color.split()], dtype=np.float32)
                yield 0.1
            yield hold
            if color is SANI:
                self.sani = max(self.sani - 0.012, 0.0)
                self._tank("tank_s", self.sani)
            else:
                self.water = max(self.water - 0.01, 0.0)
                self._tank("tank_w", self.water)
            for q in range(n + 1):
                self.liquid(name, 0.035 * (1 - q / n))
                yield 0.07
        # air dry
        for _ in range(12):
            self.set_led(i, "cip")
            yield 0.5
            self.set_led(i, "idle")
            yield 0.5
        self.liquid(name, 0.0005, LIQUID)
        self.m.geom_rgba[self.m.geom(name).id][3] = 0.0
        self.cip_cycles += 1
        s["t"]["clean"] = round(self.d.time - self.t_start, 1)
        self.jobs.append(dict(kind="cip", pos=i, sample=s["id"], t0=round(t0 - self.t_start, 1), dur=round(self.d.time - t0, 1)))
        self.P[i].update(state="idle", sample=None, filt_frac=0.0, plate_req=False, plate_ready=False)
        self.set_led(i, "idle")

    def _tank(self, nm, frac):
        gid = self.m.geom(nm).id
        h = 0.14 * frac
        self.m.geom_size[gid][2] = max(h, 0.001)
        self.m.geom_pos[gid][2] = max(h, 0.001)

    # ------------------------------------------------------------------ plate hotel, shuttle, lid lifter
    def plate_manager(self):
        while True:
            yield lambda: bool(self.plate_out_q) or (bool(self.plate_in_q) and self.spot_owner is None)
            t0 = self.d.time
            if self.plate_out_q:
                i, k = self.plate_out_q.pop(0)
                yield from self.plate_out(k)
                kind = "plate_out"
            else:
                i = self.plate_in_q.pop(0)
                k = self.P[i]["sample"]["plate"]
                self.spot_owner = i
                yield from self.plate_in(k)
                self.P[i]["plate_ready"] = True
                kind = "plate_in"
            self.jobs.append(dict(kind=kind, pos=i, sample=k, t0=round(t0 - self.t_start, 1), dur=round(self.d.time - t0, 1)))

    def plate_in(self, k):
        pn, ln = f"plate{k}", f"lid{k}"
        self.plates_taken += 1
        zk = PLATE_Z(k)
        yield from self.move({"sz": zk}, 1.4)
        self.carry = [(pn, (0, 0, 0)), (ln, (0, 0, 0.0064))]
        yield from self.move({"sz": 0.0}, 1.0)
        yield from self.move({"sx": JUNCTION_X - HOTEL[0]}, 2.0)
        yield from self.move({"sy": SPOT[1] - HOTEL[1]}, 2.0)
        # lid off
        yield from self.move({"lgx": SPOT[0] - LIDPARK[0]}, 1.8)
        yield from self.move({"lgz": -0.1476}, 1.0)
        self.carry = [(pn, (0, 0, 0))]
        self.cup_carry = ln
        yield from self.move({"lgz": -0.06}, 1.0)
        yield from self.move({"lgx": 0.0}, 1.8)
        yield from self.move({"lgz": -0.1548}, 1.0)
        self.cup_carry = None
        self.fhold[ln][:3] = (LIDPARK[0], SPOT[1], -0.0008)
        yield from self.move({"lgz": -0.05}, 0.8)

    def plate_out(self, k):
        pn, ln = f"plate{k}", f"lid{k}"
        # lid back on
        yield from self.move({"lgz": -0.1548}, 0.8)
        self.cup_carry = ln
        yield from self.move({"lgz": -0.06}, 0.8)
        yield from self.move({"lgx": SPOT[0] - LIDPARK[0]}, 1.8)
        yield from self.move({"lgz": -0.1476}, 1.0)
        self.cup_carry = None
        self.carry = [(pn, (0, 0, 0)), (ln, (0, 0, 0.0064))]
        yield from self.move({"lgz": -0.05}, 0.8)
        yield from self.move({"lgx": 0.0}, 1.6)
        # shuttle to the output stack
        yield from self.move({"sy": OUT[1] - HOTEL[1]}, 2.0)
        self.spot_owner = None
        yield from self.move({"sx": OUT[0] - HOTEL[0]}, 2.0)
        zo = self.out_count * PLATE_PITCH + 0.002
        yield from self.move({"sz": zo}, 1.0)
        self.carry = []
        self.fhold[pn][:3] = (OUT[0], OUT[1], zo)
        self.fhold[ln][:3] = (OUT[0], OUT[1], zo + 0.0064)
        self.out_count += 1
        yield from self.move({"sz": 0.0}, 0.8)
        yield from self.move({"sx": 0.0, "sy": 0.0}, 3.0)

    # ------------------------------------------------------------------ arm jobs
    def arm_park(self):
        self.arm_label = ""
        self.robot.move_to([0.30, -0.02, 0.42], down_rotation([0, 1, 0]), vmax=0.5 * self.arm_speed)

    def run_transfer(self, **kw):
        """transfer() from a clean start; one retry from the park pose if the grasp pose is not reached."""
        for attempt in (0, 1):
            self._arm_reset_gripper()
            self.arm_park() if attempt == 1 or not self.at_park() else None
            try:
                return transfer(self.robot, insets=False, slowmo=False, speed=self.arm_speed, **kw)
            except Exception as e:
                self._arm_reset()
                print("transfer retry:", repr(e)[:90], flush=True)
                err = e
        return dict(success=False, error=repr(err)[:90])

    def at_park(self):
        return np.linalg.norm(self.robot.tcp_pos() - np.array([0.30, -0.02, 0.42])) < 0.02

    def _arm_reset_gripper(self):
        self.d.eq_active[self.grip_eqs] = 0          # only the grip clamps; the membrane's own edge constraints stay on
        self.d.ctrl[self.robot.grip_act] = 255.0

    def arm_choose(self):
        """Pick the next arm job or None."""
        if self.zone == "doser":
            return None
        for i in range(N_POS):
            P = self.P[i]
            if P["state"] == "ready" and P["plate_ready"]:
                return ("wet", i)
        if self.next_sample < self.n_samples and len(self.dose_q) < 2:
            waiting = sum(1 for P in self.P if P["state"] in ("loaded", "wait_dose"))
            if waiting < 2:
                for i in range(N_POS):
                    if self.P[i]["state"] == "idle":
                        return ("dry", i)
        return None

    def swap_to_disc(self, name_geom, local_xyz):
        gid = self.m.geom(name_geom).id
        self.m.geom_pos[gid][0], self.m.geom_pos[gid][1], self.m.geom_pos[gid][2] = local_xyz
        self.m.geom_rgba[gid][3] = 1.0

    def dry_job(self, i):
        r, d = self.robot, self.d
        s = self.samples[self.next_sample]
        self.next_sample += 1
        P = self.P[i]
        P.update(sample=s, state="wait_load")
        self.set_led(i, "clamp")
        t0 = d.time
        s["t"]["loaded0"] = round(t0 - self.t_start, 1)
        self.zone = "arm"
        self.arm_label = f"Arm: dry membrane to P{i + 1}"
        self.spawn(self.move({f"flift{i}": FUNNEL_UP}, LIFT_T), "up")
        # magazine: top disc becomes the flex membrane
        top = self.mem_left - 1
        self.hide(f"ms{top}")
        self.flex_set(self.rest)
        self.flex_alpha(1.0)
        self.mem_left -= 1
        self.spawn(self.funnel_after_arm(i, t0), "down")
        res = self.run_transfer(dest=(FRIT_X, FRIT_YS[i], FRIT_TOP))
        # freeze the placed membrane as a rigid disc on the frit; the flex returns to the magazine
        V = self.flex_xyz()
        self.mem_xyz[i] = V
        c = V.mean(0)
        self.swap_to_disc(f"fm{i}", (c[0] - FRIT_X, c[1] - FRIT_YS[i], 0.0204 + (c[2] - 0.1004)))
        self.flex_set(self.rest)
        self.flex_alpha(0.0)
        s["dry"] = {k: res.get(k) for k in ("success", "centroid_offset_mm", "defect", "error") if k in res}
        print(f"[t={d.time - self.t_start:5.0f}s] dry S{s['id']+1} -> P{i+1}: {s['dry']}  centre={c.round(3)}", flush=True)
        if res.get("error"):
            print("DRY ERROR", i, res["error"], flush=True)
        self.zone = None
        self.jobs.append(dict(kind="arm_dry", pos=i, sample=s["id"], t0=round(t0 - self.t_start, 1), dur=round(d.time - t0, 1)))
        self.arm_busy_s += d.time - t0
        self.spawn(self.after_dry(i), "after_dry")

    def _arm_reset(self):
        self._arm_reset_gripper()

    def wet_job(self, i):
        r, d = self.robot, self.d
        P = self.P[i]
        s = P["sample"]
        k = s["plate"]
        t0 = d.time
        s["t"]["wet0"] = round(t0 - self.t_start, 1)
        self.set_state(i, "wet", "ready")
        self.zone = "arm"
        self.arm_label = f"Arm: wet membrane P{i + 1} to agar"
        self.spawn(self.move({f"flift{i}": FUNNEL_UP}, LIFT_T), "up")
        # the rigid disc on the frit becomes the flex again, exactly where it was
        self.hide(f"fm{i}")
        self.flex_set(self.mem_xyz[i])
        self.flex_alpha(1.0)
        self.spawn(self.funnel_after_arm(i, t0), "down")
        res = self.run_transfer(dest=(SPOT[0], SPOT[1], AGAR_TOP))
        V = self.flex_xyz()
        c = V.mean(0)
        self.swap_to_disc(f"pm{k}", (c[0] - SPOT[0], c[1] - SPOT[1], AGAR_TOP + 0.0004 + max(c[2] - AGAR_TOP - 0.0003, 0.0)))
        self.flex_set(self.rest)
        self.flex_alpha(0.0)
        s["wet"] = {kk: res.get(kk) for kk in ("success", "centroid_offset_mm", "flatness_mm", "max_lift_mm", "fold", "air_pocket", "overhang", "defect", "error") if kk in res}
        if res.get("error"):
            print("WET ERROR", i, res["error"], flush=True)
        s["t"]["transferred"] = round(d.time - self.t_start, 1)
        print(f"[t={d.time - self.t_start:5.0f}s] wet S{s['id']+1} P{i+1}: {s['wet']}", flush=True)
        self.results.append(dict(sample=s["id"], **s["wet"]))
        self.P[i]["plate_ready"] = False
        self.plate_out_q.append((i, k))
        self.spawn(self.cip_proc(i), "cip")
        self.zone = None
        t_arm = d.time - t0
        self.arm_label = "Arm: tip wash (sanitant dip, air dry)"
        # tip wash: dip in sanitant, air dry
        self.tip_wash()
        self.zone = None
        self.jobs.append(dict(kind="arm_wet", pos=i, sample=s["id"], t0=round(t0 - self.t_start, 1), dur=round(t_arm, 1)))
        self.jobs.append(dict(kind="tip_wash", pos=i, sample=s["id"], t0=round(t0 + t_arm - self.t_start, 1), dur=round(d.time - t0 - t_arm, 1)))
        self.arm_busy_s += d.time - t0

    def tip_wash(self):
        r, v = self.robot, self.arm_speed
        bx, by = TIPWASH
        r.move_to([bx, by, 0.09], vmax=0.5 * v)
        r.move_to([bx, by, 0.034], vmax=0.2 * v)
        r.wait(1.0)
        r.move_to([bx, by, 0.09], vmax=0.25 * v)
        r.move_to([bx + 0.024, by, 0.058], vmax=0.2 * v)
        self.m.geom_rgba[self.m.geom("tw_air").id][3] = 0.6
        r.wait(1.5)
        self.m.geom_rgba[self.m.geom("tw_air").id][3] = 0.0

    # ------------------------------------------------------------------ main loop
    def finished(self):
        return self.out_count >= self.n_samples and not self.plate_out_q and not self.procs_busy()

    def procs_busy(self):
        return any(p.name.startswith(("filter", "cip", "up", "down", "after")) for p in self.procs)

    def run(self, max_t=4000.0):
        r = self.robot
        self.t_start = self.d.time
        r.hooks.append(self.tick)
        self.spawn(self.doser_loop(), "doser")
        self.spawn(self.plate_manager(), "plates")
        r.wait(0.5)
        self.rest = self.flex_xyz()
        self.flex_alpha(0.0)
        parked = False
        last_log = -1e9
        while self.d.time - self.t_start < max_t and not self.finished():
            if self.verbose and self.d.time - last_log >= 30.0:
                last_log = self.d.time
                print(f"[t={self.d.time - self.t_start:6.0f}s] out={self.out_count} next={self.next_sample} zone={self.zone} "
                      f"states={[P['state'] for P in self.P]} doseq={self.dose_q} plin={self.plate_in_q} plout={self.plate_out_q} "
                      f"spot={self.spot_owner}", flush=True)
            job = self.arm_choose()
            if job is None:
                if not parked:
                    self.arm_park()
                    parked = True
                r.wait(0.1)
                continue
            parked = False
            if job[0] == "dry":
                self.dry_job(job[1])
            else:
                self.wet_job(job[1])
        r.wait(2.0)
        self.t_end = self.d.time - self.t_start
        return self.summary()

    def summary(self):
        T = self.t_end
        by = {}
        for j in self.jobs:
            by.setdefault(j["kind"], []).append(j["dur"])
        stats = {k: dict(n=len(v), mean=round(float(np.mean(v)), 1), max=round(float(np.max(v)), 1)) for k, v in by.items()}
        done = [s for s in self.samples if "transferred" in s["t"]]
        cyc = None
        if len(done) >= 3:
            ts = sorted(s["t"]["transferred"] for s in done)
            cyc = round(float(np.mean(np.diff(ts))), 1)
        return dict(total_s=round(T, 1), samples=len(done), steady_cycle_s=cyc, arm_busy_s=round(self.arm_busy_s, 1),
                    arm_util=round(self.arm_busy_s / T, 3), stats=stats, cip_cycles=self.cip_cycles, tips=self.tips_used,
                    results=self.results)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--filt", type=float, default=0.0, help="fixed filtration time (s); default random 70-110")
    ap.add_argument("--video", default="")
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--cams", default="line_a,line_b")
    a = ap.parse_args()
    m, d = load()
    robot = Robot(m, d)
    filt = (a.filt, a.filt) if a.filt > 0 else (70.0, 110.0)
    line = Line(m, d, robot, n_samples=a.samples, filt=filt)
    t0 = time.perf_counter()
    summ = line.run()
    print("wall %.0f s" % (time.perf_counter() - t0))
    print(json.dumps({k: v for k, v in summ.items() if k != "results"}, indent=1))
    print("results:", summ["results"])

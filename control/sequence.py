"""Phase A: the full per-sample sequence on one filtration position, in the "sequence" layout.

plate from input stack -> work spot, lid off | membrane magazine -> frit (roll-on) |
funnel on + bayonet twist | open vessel | dose (pour, liquid animated) | filtration (time-lapse) |
funnel off -> wash station | membrane frit -> agar (roll-on) | lid on | plate -> output stack

Rigid objects are carried kinematically (Robot.carry). The 90 mm Petri dish is wider than the
Franka Hand's 80 mm stroke, so plates and lids are pinched at the rim (a custom rim gripper would
be needed on the real cell; noted on the results page).

    python control/sequence.py              # video at 4x -> results/sequence.mp4, durations -> results/step_durations.csv
    python control/sequence.py --no-video
"""
import csv
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco
import mink

from scene.cell import (load, ROOT, FRIT_X, FRIT_YS, FRIT_TOP, PLATE_XY, PLATE_RIM, LID_SPOT,
                        STACK_OUT_XY, PLATE_PITCH, VESSEL_X, VESSEL_YS, WASH_XY, WASH_TOP)
from control.primitives import Robot, Recorder, GRIP_OPEN, GRIP_CLOSED, down_rotation, axis_angle
from transfer.expert import transfer

SAFE_Z = 0.30
CAP_SPOT = (VESSEL_X, -0.27)
FUNNEL_Z = 0.099               # funnel origin when seated on a frit collar
FILTRATION_MODEL_S = 180.0     # filtration time used in the throughput model (sim shows a time-lapse)


def width_ctrl(w):
    """Gripper ctrl for a finger opening of w metres (0..0.08)."""
    return float(np.clip(w / 0.08 * 255, 0, 255))


def set_level(m, geom, h, base=0.001):
    g = m.geom(geom).id
    h = max(h, 1e-4)
    m.geom_size[g][1] = h / 2
    m.geom_pos[g][2] = base + h / 2
    m.geom_rgba[g][3] = 0.55 if h > 6e-4 else 0.0


class Sequence:
    def __init__(self, robot, rec=None):
        self.r, self.m, self.d, self.rec = robot, robot.m, robot.d, rec
        self.n_steps, self.k = 10, 0
        self.results = {}

    # -- helpers ----------------------------------------------------------------------------
    def caption(self, text):
        if self.rec:
            self.rec.caption = f"{self.k}/{self.n_steps}  {text}"

    def se3(self, R, p):
        return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), np.asarray(p, float))

    def choose_R(self, u, poses):
        """Of the two equivalent finger orientations, take the one the IK can reach at all `poses`."""
        cands = (down_rotation(u), down_rotation(-np.asarray(u)))
        return min(cands, key=lambda R: self.r.ik.residual([self.se3(R, p) for p in poses], iters=150))

    def go(self, pos, R=None, vmax=0.35):
        """Lift to SAFE_Z, travel, descend to `pos`."""
        cur = self.r.target.translation()
        if cur[2] < SAFE_Z - 1e-3:
            self.r.move_to([cur[0], cur[1], SAFE_Z], vmax=0.25)
        self.r.move_to([pos[0], pos[1], SAFE_Z], R, vmax=vmax)
        self.r.move_to(pos, vmax=0.15)

    def pick(self, bodies, grasp, R, width, approach=0.06):
        self.go(np.asarray(grasp) + [0, 0, approach], R)
        self.r.move_to(grasp, vmax=0.06)
        self.r.gripper(width_ctrl(width), 0.3)
        self.r.carry(*bodies)

    def place(self, body, origin, approach=0.04, open_to=0.08):
        """Move so carried `body` lands with its origin at `origin`, release, back off."""
        delta = np.asarray(origin) - self.d.body(body).xpos
        goal = self.r.target.translation() + delta
        self.go(goal + [0, 0, approach])
        self.r.move_to(goal, vmax=0.05)
        self.r.drop()
        self.r.gripper(width_ctrl(open_to), 0.3)
        self.r.move_to(goal + [0, 0, approach], vmax=0.1)

    def twist(self, angle, duration=1.0):
        R = axis_angle([0, 0, 1], angle) @ self.r.target.rotation().as_matrix()
        self.r.move_to(self.r.target.translation(), R, wmax=abs(angle) / duration * 1.875)

    def step(self, name, label):
        self.k += 1
        self.caption(label)
        return self.r.timed(name)

    # -- the sample -------------------------------------------------------------------------
    def run(self):
        r, m, d = self.r, self.m, self.d
        frit = np.array([FRIT_X, FRIT_YS[0]])
        set_level(m, "funnel0_liquid", 0.0)

        with self.step("plate_prep", "Plate from input stack to work spot, lid off"):
            p = d.body("plate").xpos.copy()
            grasp = p + [-0.045, 0, 0.008]
            R = self.choose_R([1, 0, 0], [grasp, np.r_[PLATE_XY[0] - 0.045, PLATE_XY[1], 0.010]])
            self.pick(["plate", "lid"], grasp, R, 0.012)
            self.place("plate", [PLATE_XY[0], PLATE_XY[1], 0.001], open_to=0.03)
            lid = d.body("lid").xpos.copy()
            grasp = lid + [-0.047, 0, 0.0088]
            self.pick(["lid"], grasp, R, 0.010)
            self.place("lid", [LID_SPOT[0], LID_SPOT[1], -0.0076 + 0.001], open_to=0.03)

        with self.step("membrane_to_frit", "Membrane from magazine onto frit (roll-on)"):
            res = transfer(r, rec=self.rec, dest=(frit[0], frit[1], FRIT_TOP), insets=False, slowmo=False,
                           captions=dict(approach=self._c("Membrane: approach magazine"), grasp=self._c("Membrane: grasp edge"),
                                         lift=self._c("Membrane: lift"), transit=self._c("Membrane: to frit"),
                                         touchdown=self._c("Membrane: far edge down"), roll_on=self._c("Membrane: roll-on onto frit"),
                                         release=self._c("Membrane: release"), retract=self._c("Membrane: retract"),
                                         settled=self._c("Membrane on frit")))
            self.results["membrane_on_frit"] = res

        with self.step("funnel_on", "Funnel from wash station onto frit, bayonet twist 30 deg"):
            f = d.body("funnel0").xpos.copy()
            grasp = f + [0, 0, 0.06]
            seat = np.r_[frit, FUNNEL_Z + 0.001]
            R = self.choose_R([0, 1, 0], [grasp, seat + [0, 0, 0.06]])
            self.pick(["funnel0"], grasp, R, 0.062)
            delta = seat - d.body("funnel0").xpos
            goal = r.target.translation() + delta
            self.go(goal + [0, 0, 0.03])
            r.move_to(goal, vmax=0.03)
            self.twist(np.radians(30))
            r.drop()
            r.gripper(width_ctrl(0.08), 0.3)
            r.move_to(r.target.translation() + [0, 0, 0.05], vmax=0.15)

        with self.step("open_vessel", "Open sample vessel (quarter-turn cap off)"):
            c = d.body("cap0").xpos.copy()
            grasp = c + [0, 0, 0.003]
            R = self.choose_R([0, 1, 0], [grasp, np.r_[CAP_SPOT, 0.01]])
            self.pick(["cap0"], grasp, R, 0.044)
            self.twist(np.radians(-90))
            r.move_to(r.target.translation() + [0, 0, 0.04], vmax=0.1)
            self.place("cap0", [CAP_SPOT[0], CAP_SPOT[1], 0.001], open_to=0.07)

        with self.step("dose", "Dose sample into funnel (pour, level animated)"):
            v = d.body("vessel0").xpos.copy()
            grasp = v + [0, 0, 0.065]
            home_v = v.copy()
            R = self.choose_R([0, 1, 0], [grasp, np.r_[frit[0] + 0.02, frit[1], 0.25]])
            self.pick(["vessel0"], grasp, R, 0.042)
            self.go(np.r_[frit[0] + 0.035, frit[1], 0.255])
            # tilt about the axis that keeps the mouth over the funnel; pick the reachable sign
            best = None
            for ax in ([0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]):
                Rt = axis_angle(ax, np.radians(105)) @ r.target.rotation().as_matrix()
                e = r.ik.residual([self.se3(Rt, r.target.translation())], iters=150)
                if best is None or e < best[0]:
                    best = (e, Rt)
            r.move_to(r.target.translation(), best[1], wmax=1.2)
            for i in range(31):                                  # pour: levels animated over ~3 s
                x = i / 30
                set_level(m, "vessel0_liquid", 0.05 * (1 - x) + 0.004 * x)
                set_level(m, "funnel0_liquid", 0.035 * x)
                r.wait(0.1)
            r.move_to(r.target.translation(), R, wmax=1.2)
            self.place("vessel0", home_v + [0, 0, 0.001], open_to=0.07)

        with self.step("filtration", "Vacuum filtration (3 min, time-lapse); arm free"):
            r.move_to(r.target.translation() + [0, 0, 0.0], vmax=0.2)
            self.go([0.35, -0.05, SAFE_Z])
            for i in range(41):
                set_level(m, "funnel0_liquid", 0.035 * (1 - i / 40))
                r.wait(0.1)

        with self.step("funnel_off", "Funnel off -> wash station (rinse + disinfection off-arm)"):
            f = d.body("funnel0").xpos.copy()
            grasp = f + [0, 0, 0.06]
            self.r.move_to(r.target.translation(), vmax=0.1)
            R = self.choose_R([0, 1, 0], [grasp, np.r_[WASH_XY, WASH_TOP + 0.06]])
            R = axis_angle([0, 0, 1], np.radians(30)) @ R
            self.pick(["funnel0"], grasp, R, 0.062, approach=0.03)
            self.twist(np.radians(-30))
            r.move_to(r.target.translation() + [0, 0, 0.03], vmax=0.03)
            self.place("funnel0", [WASH_XY[0], WASH_XY[1], WASH_TOP + 0.001])

        with self.step("transfer", "Membrane frit -> agar (roll-on)"):
            res = transfer(r, rec=self.rec, insets=False, slowmo=False,
                           captions=dict(approach=self._c("Transfer: approach membrane edge"), grasp=self._c("Transfer: grasp edge"),
                                         lift=self._c("Transfer: peel off frit"), transit=self._c("Transfer: to agar plate"),
                                         touchdown=self._c("Transfer: far edge touches agar"), roll_on=self._c("Transfer: roll-on"),
                                         release=self._c("Transfer: release"), retract=self._c("Transfer: retract"),
                                         settled=self._c("Membrane on agar")))
            self.results["membrane_on_agar"] = res

        with self.step("lid_and_stack", "Lid on, plate to output stack"):
            lid = d.body("lid").xpos.copy()
            grasp = lid + [-0.047, 0, 0.0088]
            R = self.choose_R([1, 0, 0], [grasp, np.r_[STACK_OUT_XY[0] - 0.045, STACK_OUT_XY[1], 3 * PLATE_PITCH + 0.01]])
            self.pick(["lid"], grasp, R, 0.010)
            pl = d.body("plate").xpos.copy()
            self.place("lid", [pl[0], pl[1], pl[2] + PLATE_RIM - 0.0076 + 0.001], open_to=0.03)
            grasp = d.body("plate").xpos.copy() + [-0.045, 0, 0.008]
            self.pick(["plate", "lid"], grasp, R, 0.012)
            self.place("plate", [STACK_OUT_XY[0], STACK_OUT_XY[1], 3 * PLATE_PITCH + 0.001], open_to=0.03)
            self.go([0.35, 0.0, SAFE_Z + 0.1])
        self.k = self.n_steps
        self.caption("Sample done: plate in output stack, ready for incubation")
        r.wait(1.0)
        return self.results

    def _c(self, text):
        return f"{self.k}/{self.n_steps}  {text}"


def main(video=True):
    m, d = load(mode="sequence")
    out = ROOT / "results"
    rec = Recorder(m, camera="oblique", inset=None, overlay=False, path=out / "sequence.mp4") if video else None
    if rec:
        rec.slowmo = 0.25                                       # 4x speed
    robot = Robot(m, d, recorder=rec)
    robot.wait(0.5)
    seq = Sequence(robot, rec)
    t0 = time.perf_counter()
    res = seq.run()
    print(f"sim {d.time:.1f} s, wall {time.perf_counter() - t0:.0f} s")
    for k, v in res.items():
        print(k, {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    top = [(n, b - a) for n, a, b in robot.log if n in {"plate_prep", "membrane_to_frit", "funnel_on", "open_vessel", "dose",
                                                        "filtration", "funnel_off", "transfer", "lid_and_stack"}]
    with open(out / "step_durations.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "seconds", "note"])
        for n, s in top:
            if n == "filtration":
                w.writerow([n, f"{FILTRATION_MODEL_S:.1f}", f"model value; sim shows a {s:.1f} s time-lapse"])
            else:
                w.writerow([n, f"{s:.1f}", "measured in sim"])
            print(f"  {n:18s} {s:6.1f} s")
    if rec:
        print("wrote", rec.save(), rec.n, "frames")


if __name__ == "__main__":
    main(video="--no-video" not in sys.argv)

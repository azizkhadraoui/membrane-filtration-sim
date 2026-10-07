"""10 Hz policy interface for the membrane transfer (plan Phase C).

Observation: RGB from the top-down policy camera and the wrist camera (IMG x IMG), plus state =
7 arm joint positions + gripper opening.
Action: absolute TCP target pose for the next tick (position 3 + rotation 6D 6) + gripper
command (1 = closed). Between ticks the target is interpolated linearly at the 500 Hz physics rate.
Grasp rule (same for expert and policy): closing within GRASP_TOL of a membrane edge vertex
clamps the edge patch to the hand; opening releases it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco
import mink

from scene.cell import load, outer_ring, FRIT_X, FRIT_YS
from control.primitives import Robot, GRIP_OPEN, GRIP_CLOSED, down_rotation
from transfer.expert import attach, grip_patch, transfer
from transfer.metrics import evaluate
from transfer.randomize import sample
from scene.cell import AGAR_TOP

CAMS = ("policy_top", "panda/wrist", "policy_plate")
IMG = 128
HZ = 10
GRASP_TOL = 0.005
ACT_DIM, STATE_DIM = 10, 8


def rot6(R):
    return np.concatenate([R[:, 0], R[:, 1]])


def from_rot6(r6):
    a1, a2 = np.asarray(r6[:3], float), np.asarray(r6[3:6], float)
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    b2 = a2 - b1 * (b1 @ a2)
    b2 /= np.linalg.norm(b2) + 1e-9
    return np.column_stack([b1, b2, np.cross(b1, b2)])


class TransferEnv:
    def __init__(self, seed, scale=1.0, visual=True, render=True):
        self.seed = seed
        cfg, self.expert_kwargs = sample(seed, scale=scale, visual=visual)
        self.m, self.d = load(**cfg)
        self.robot = Robot(self.m, self.d)
        self.renderer = mujoco.Renderer(self.m, IMG, IMG) if render else None
        self.attached = []
        self.released = False
        self.grip_cmd = 0.0
        rng = np.random.default_rng(seed + 7919)
        # start hovering above the frit (not recorded): randomised pre-grasp pose
        hover = np.array([FRIT_X + rng.uniform(-0.03, 0.03), FRIT_YS[0] + rng.uniform(-0.03, 0.03),
                          0.17 + rng.uniform(0, 0.04)])
        a = rng.uniform(-0.3, 0.3)
        v = np.array([-np.cos(a), np.sin(a), 0.0])
        # of the two equivalent finger orientations take the one the IK reaches without hitting
        # the joint-7 limit (otherwise the wrist winds up and the arm self-collides)
        se3 = lambda R: mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), hover)
        R = min((down_rotation(v), down_rotation(-v)), key=lambda R: self.robot.ik.residual([se3(R)], iters=200))
        self.robot.move_to(hover, R, vmax=0.4)
        if np.linalg.norm(self.robot.tcp_pos() - hover) > 0.005:
            raise RuntimeError("start pose not reached")
        self.robot.wait(0.3)

    # -- observation ------------------------------------------------------------------------
    def images(self):
        out = []
        for cam in CAMS:
            self.renderer.update_scene(self.d, camera=cam)
            out.append(self.renderer.render().copy())
        return np.stack(out)                                   # (2, IMG, IMG, 3) uint8

    def state(self):
        d, r = self.d, self.robot
        grip = d.qpos[self.m.joint("panda/finger_joint1").qposadr[0]] + d.qpos[self.m.joint("panda/finger_joint2").qposadr[0]]
        return np.r_[d.qpos[r.qadr], grip].astype(np.float32)

    def target_action(self):
        t = self.robot.target
        return np.r_[t.translation(), rot6(t.rotation().as_matrix()), self.grip_cmd].astype(np.float32)

    # -- policy control ---------------------------------------------------------------------
    def _gripper(self, cmd):
        closed = cmd > 0.5
        if closed and not self.grip_cmd > 0.5:
            self.d.ctrl[self.robot.grip_act] = GRIP_CLOSED
        if closed and not self.attached and not self.released:
            # the tips are closed: grasp as soon as they are at an edge (not only on the closing tick)
            v = self.d.flexvert_xpos
            outer = outer_ring()
            dist = np.linalg.norm(v[outer] - self.robot.tcp_pos(), axis=1)
            i0 = int(np.argmin(dist))
            if dist[i0] < GRASP_TOL + 0.0015:                   # TCP rides 1.5 mm above the vertex
                self.attached = attach(self.robot, grip_patch(v, outer, i0))
        elif not closed and self.grip_cmd > 0.5:
            self.d.ctrl[self.robot.grip_act] = GRIP_OPEN
            if self.attached:
                self.d.eq_active[self.attached] = 0
                self.attached = []
                self.released = True
        self.grip_cmd = float(closed)

    def step(self, action):
        """Apply one 10 Hz action: interpolate the TCP target to the commanded pose."""
        self._gripper(action[9])
        start = self.robot.target
        goal = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(from_rot6(action[3:9])),
                                                      np.asarray(action[:3], float))
        n = int(round(1.0 / HZ / self.robot.dt))
        for i in range(1, n + 1):
            self.robot.target = start.interpolate(goal, i / n)
            self.robot._step()

    def evaluate(self, settle=1.0):
        self.robot.wait(settle)
        if not np.isfinite(self.d.flexvert_xpos).all():
            return dict(success=False, defect="sim_error")
        p = self.d.body("plate").xpos
        return evaluate(self.d.flexvert_xpos.copy(), p[:2], p[2] + AGAR_TOP)

    # -- expert demonstration ---------------------------------------------------------------
    def record_expert(self, noise_mm=0.0, noise_tau=0.6, seed=0):
        """Like record_expert_clean, with DART noise: the executed target is offset by a smooth random
        process (Ornstein-Uhlenbeck, std noise_mm, time constant noise_tau); the recorded label is the
        clean expert target, and the recorded `ref` is the executed target. The policy thus sees
        off-course states paired with the corrective command."""
        if noise_mm <= 0:
            return self.record_expert_clean()
        r, d = self.robot, self.d
        rng = np.random.default_rng(seed + 424242)
        sig, a = noise_mm * 1e-3, np.exp(-r.dt / noise_tau)
        b = sig * np.sqrt(1 - a * a)
        imgs, states, targets, refs = [], [], [], []
        next_t = [d.time]

        def noise(robot):
            robot.exec_offset = a * robot.exec_offset + b * rng.standard_normal(3)

        def rec(robot):
            if d.time + 1e-9 >= next_t[0]:
                next_t[0] += 1.0 / HZ
                self.grip_cmd = float(d.ctrl[r.grip_act] < 128)
                imgs.append(self.images())
                states.append(self.state())
                targets.append(self.target_action())
                refs.append((r.target.translation() + r.exec_offset).astype(np.float32))
        r.hooks += [noise, rec]
        rec(r)
        res = transfer(r, grasp_tol=0.0065, **self.expert_kwargs)
        r.hooks.remove(noise); r.hooks.remove(rec)
        r.exec_offset = np.zeros(3)
        T = len(targets) - int(1.0 * HZ)
        return dict(images=np.stack(imgs[:T]), state=np.stack(states[:T]), action=np.stack(targets[1:T + 1]),
                    ref=np.stack(refs[:T])), res

    def record_expert_clean(self):
        """Run the scripted expert, sampling obs and commanded targets at 10 Hz.
        Returns dict(images (T,2,H,W,3), state (T,8), action (T,10)) and the expert metrics."""
        r, d = self.robot, self.d
        imgs, states, targets = [], [], []
        next_t = [d.time]

        def hook(robot):
            if d.time + 1e-9 >= next_t[0]:
                next_t[0] += 1.0 / HZ
                self.grip_cmd = float(d.ctrl[r.grip_act] < 128)
                imgs.append(self.images())
                states.append(self.state())
                targets.append(self.target_action())
        r.hooks.append(hook)
        hook(r)
        res = transfer(r, **self.expert_kwargs)                 # includes 1 s settle + evaluation
        r.hooks.remove(hook)
        # stop the episode once the arm has retracted (drop the 1 s settle phase); action[k] is
        # the commanded target at tick k+1
        T = len(targets) - int(1.0 * HZ)
        return dict(images=np.stack(imgs[:T]), state=np.stack(states[:T]),
                    action=np.stack(targets[1:T + 1])), res

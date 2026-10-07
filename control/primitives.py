"""Motion primitives on the cell. Each returns elapsed sim time and is logged for the cycle-time CSV."""
from pathlib import Path
import numpy as np
import mujoco
import mink

from control.ik import ArmIK

GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0


def min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def axis_angle(axis, angle):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def down_rotation(finger_axis):
    """Hand z pointing down, fingers opening along the horizontal `finger_axis`."""
    y = np.array([finger_axis[0], finger_axis[1], 0.0])
    y /= np.linalg.norm(y)
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


class Robot:
    def __init__(self, m, d, recorder=None):
        self.m, self.d, self.recorder = m, d, recorder
        self.dt = m.opt.timestep
        self.qadr = [m.joint(f"panda/joint{i}").qposadr[0] for i in range(1, 8)]
        self.act = [m.actuator(f"panda/actuator{i}").id for i in range(1, 8)]
        self.grip_act = m.actuator("panda/actuator8").id
        self.tcp_site = m.site("panda/tcp").id
        q0 = np.r_[d.qpos[self.qadr], 0.04, 0.04]
        self.ik = ArmIK(q0)
        self.target = self.ik.tcp()
        self.log = []                                  # (step name, t_start, t_end)
        self.hooks = []                                # called every physics step
        self.carried = {}                              # body id -> (rel pos, rel quat, saved contype/conaffinity)
        self.exec_offset = np.zeros(3)                 # added to the target actually executed (DART noise)
        self.hand = m.body("panda/hand").id

    # -- low level --------------------------------------------------------------------------
    def _step(self, n=1):
        for _ in range(n):
            tgt = self.target
            if self.exec_offset.any():
                tgt = mink.SE3.from_rotation_and_translation(tgt.rotation(), tgt.translation() + self.exec_offset)
            self.d.ctrl[self.act] = self.ik.step(tgt, self.dt)
            self._update_carried()
            mujoco.mj_step(self.m, self.d)
            for h in self.hooks:
                h(self)
            if self.recorder:
                self.recorder.tick(self.d)

    # -- kinematic carrying of rigid objects (stand-in for a firm grasp) ------------------------
    def _body_pose(self, b):
        return self.d.xpos[b].copy(), self.d.xquat[b].copy()

    def carry(self, *names):
        """Attach free bodies to the hand at their current relative pose; they stop colliding."""
        hp, hq = self._body_pose(self.hand)
        hq_inv = np.zeros(4); mujoco.mju_negQuat(hq_inv, hq)
        for name in names:
            b = self.m.body(name).id
            bp, bq = self._body_pose(b)
            rp = np.zeros(3); mujoco.mju_rotVecQuat(rp, bp - hp, hq_inv)
            rq = np.zeros(4); mujoco.mju_mulQuat(rq, hq_inv, bq)
            geoms = np.flatnonzero(self.m.geom_bodyid == b)
            saved = (self.m.geom_contype[geoms].copy(), self.m.geom_conaffinity[geoms].copy())
            self.m.geom_contype[geoms] = 0; self.m.geom_conaffinity[geoms] = 0
            self.carried[b] = (rp, rq, geoms, saved)

    def drop(self, *names):
        """Release carried bodies (all if no names): collisions restored, velocity zeroed."""
        ids = [self.m.body(n).id for n in names] if names else list(self.carried)
        for b in ids:
            rp, rq, geoms, (ct, ca) = self.carried.pop(b)
            self.m.geom_contype[geoms] = ct; self.m.geom_conaffinity[geoms] = ca
            j = self.m.body_jntadr[b]
            self.d.qvel[self.m.jnt_dofadr[j]:self.m.jnt_dofadr[j] + 6] = 0

    def _update_carried(self):
        if not self.carried:
            return
        hp, hq = self._body_pose(self.hand)
        for b, (rp, rq, _, _) in self.carried.items():
            j = self.m.body_jntadr[b]
            qa, va = self.m.jnt_qposadr[j], self.m.jnt_dofadr[j]
            p = np.zeros(3); mujoco.mju_rotVecQuat(p, rp, hq)
            q = np.zeros(4); mujoco.mju_mulQuat(q, hq, rq)
            self.d.qpos[qa:qa + 3] = hp + p
            self.d.qpos[qa + 3:qa + 7] = q
            self.d.qvel[va:va + 6] = 0

    def tcp_pos(self):
        return self.d.site_xpos[self.tcp_site].copy()

    def tcp_rot(self):
        return self.d.site_xmat[self.tcp_site].reshape(3, 3).copy()

    def timed(self, name):
        robot = self

        class _T:
            def __enter__(self):
                self.t0 = robot.d.time

            def __exit__(self, *exc):
                robot.log.append((name, self.t0, robot.d.time))
        return _T()

    # -- primitives -------------------------------------------------------------------------
    def move_to(self, pos, rot=None, vmax=0.25, wmax=1.0, settle_tol=1e-3, settle_max=0.3):
        t0 = self.d.time
        start = self.target
        R = start.rotation() if rot is None else mink.SO3.from_matrix(np.asarray(rot))
        goal = mink.SE3.from_rotation_and_translation(R, np.asarray(pos, float))
        dist = np.linalg.norm(goal.translation() - start.translation())
        ang = np.linalg.norm((start.rotation().inverse() @ R).log())
        T = max(1.875 * dist / vmax, 1.875 * ang / wmax, 0.1)
        n = max(1, int(round(T / self.dt)))
        for i in range(1, n + 1):
            self.target = start.interpolate(goal, min_jerk(i / n))
            self._step()
        for _ in range(int(settle_max / self.dt)):
            if np.linalg.norm(self.tcp_pos() - goal.translation()) < settle_tol:
                break
            self._step()
        return self.d.time - t0

    def follow(self, path, duration):
        """Track path(x) -> (pos, rot) for x in [0, 1], time-scaled with min-jerk."""
        t0 = self.d.time
        n = max(1, int(round(duration / self.dt)))
        for i in range(1, n + 1):
            pos, rot = path(min_jerk(i / n))
            self.target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot), pos)
            self._step()
        return self.d.time - t0

    def gripper(self, ctrl, duration=0.4):
        t0 = self.d.time
        self.d.ctrl[self.grip_act] = ctrl
        self._step(int(duration / self.dt))
        return self.d.time - t0

    def wait(self, duration):
        t0 = self.d.time
        self._step(int(duration / self.dt))
        return self.d.time - t0


class Recorder:
    """Renders the main camera at `fps` of sim time, with an optional picture-in-picture inset.

    `slowmo` > 1 captures more frames per sim second, so playback at `fps` is slowed down.
    """

    def __init__(self, m, camera="oblique", inset=None, fps=30, size=(720, 1280), inset_scale=0.36, overlay=True,
                 path=None):
        self.m, self.camera, self.inset, self.fps, self.overlay = m, camera, inset, fps, overlay
        self.path, self.writer, self.n = path, None, 0  # with `path`, frames stream to disk instead of RAM
        self.meta = []                                 # per frame: sim time, caption, speed
        self.renderer = mujoco.Renderer(m, *size)
        self.ih, self.iw = int(size[0] * inset_scale), int(size[1] * inset_scale)
        self.inset_renderer = mujoco.Renderer(m, self.ih, self.iw)
        self.slowmo = 1.0
        self.next_t = 0.0
        self.frames = []
        self.caption = ""

    def tick(self, d):
        if d.time + 1e-9 < self.next_t:
            return
        self.next_t = d.time + 1.0 / (self.fps * self.slowmo)
        self.renderer.update_scene(d, camera=self.camera)
        frame = self.renderer.render().copy()
        if self.inset:
            self.inset_renderer.update_scene(d, camera=self.inset)
            img = self.inset_renderer.render()
            H, W = frame.shape[:2]
            frame[H - self.ih - 16:H - 16, W - self.iw - 16:W - 16] = img
            frame[H - self.ih - 18:H - self.ih - 16, W - self.iw - 18:W - 14] = 255
        if getattr(self, 'hud', None):
            frame = self.hud(frame)
        self.meta.append(dict(t=round(d.time, 3), caption=self.caption, slowmo=self.slowmo))
        if self.overlay and (self.caption or self.slowmo != 1.0):
            import cv2
            speed = (f"   [{self.slowmo:g}x slow motion]" if self.slowmo > 1 else
                     f"   [{1 / self.slowmo:g}x speed]" if self.slowmo < 1 else "")
            text = self.caption + speed
            text += f"   t = {d.time:5.1f} s"
            cv2.putText(frame, text, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        self.n += 1
        if self.path:
            if self.writer is None:
                import imageio.v2 as iio2
                self.writer = iio2.get_writer(self.path, fps=self.fps, codec="libx264", quality=8, macro_block_size=8)
            self.writer.append_data(frame)
        else:
            self.frames.append(frame)

    def save(self, path=None):
        import imageio.v3 as iio
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            path = self.path
        else:
            iio.imwrite(path, np.stack(self.frames), fps=self.fps, codec="libx264", quality=8)
        import json
        Path(path).with_suffix(".json").write_text(json.dumps(self.meta))
        return path

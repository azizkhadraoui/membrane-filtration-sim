"""Differential IK (mink) on a Panda-only model, used as a kinematic planner.

The IK keeps its own configuration and is not reset from the sim each step: targets are
smooth (min-jerk), so the integrated joint trajectory is smooth and the position servos track it.
Solving on a separate model keeps the membrane's 183 vertex DOFs out of the QP.
"""
import numpy as np
import mink

from scene.cell import panda_spec


class ArmIK:
    def __init__(self, q0, vmax=2.0):
        self.model = panda_spec().compile()
        self.cfg = mink.Configuration(self.model, np.asarray(q0, float))
        self.task = mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=0.5,
                                   lm_damping=1e-3)
        self.posture = mink.PostureTask(self.model, cost=1e-3)
        self.posture.set_target(self.cfg.q.copy())
        self.limits = [mink.ConfigurationLimit(self.model),
                       mink.VelocityLimit(self.model, {f"joint{i}": vmax for i in range(1, 8)})]

    def tcp(self):
        return self.cfg.get_transform_frame_to_world("tcp", "site")

    def step(self, target, dt):
        self.task.set_target(target)
        v = mink.solve_ik(self.cfg, [self.task, self.posture], dt, "daqp", damping=1e-3,
                          limits=self.limits)
        self.cfg.integrate_inplace(v, dt)
        return self.cfg.q[:7].copy()

    def residual(self, poses, iters=300, dt=0.01):
        """Dry-run the IK from the current configuration through `poses` (list of SE3) on a copy;
        return the worst final position error. Used to pick between equivalent grasp orientations."""
        cfg = mink.Configuration(self.model, self.cfg.q.copy())
        task = mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=0.5, lm_damping=1e-3)
        worst = 0.0
        for pose in poses:
            task.set_target(pose)
            for _ in range(iters):
                v = mink.solve_ik(cfg, [task, self.posture], dt, "daqp", damping=1e-3, limits=self.limits)
                cfg.integrate_inplace(v, dt)
            err = np.linalg.norm(cfg.get_transform_frame_to_world("tcp", "site").translation() - pose.translation())
            worst = max(worst, err)
        return worst

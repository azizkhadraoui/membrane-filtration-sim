"""Scripted expert: membrane from frit 0 to the open agar plate with roll-on placement.

Grasp by attachment (plan §4.3): tips close on the edge, then three adjacent outer-ring vertices
are connected to the hand. The sheet hangs from that edge; the far edge is servoed onto the agar
first, then the grasped edge sweeps forward and down so the contact line rolls across the membrane.

    python transfer/expert.py            # run once, write results/transfer_expert.mp4
    python transfer/expert.py --no-video
"""
import csv
import sys
import time
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco
import mink

from scene.cell import load, outer_ring, AGAR_TOP, ROOT, MEMBRANE_RINGS
from control.primitives import Robot, Recorder, GRIP_OPEN, GRIP_CLOSED, down_rotation, axis_angle
from transfer.metrics import evaluate, VERT_R

R_MEM = 0.0235
TIP_CLEAR = 0.0015      # TCP rides this far above the grasped vertex (tip capsule radius)


def attach(robot, verts):
    m, d = robot.m, robot.d
    hand = m.body("panda/hand").id
    Rh, ph = d.xmat[hand].reshape(3, 3), d.xpos[hand]
    ids = []
    for k in verts:
        e = m.equality(f"grip{k}").id
        m.eq_data[e, 0:3] = Rh.T @ (d.flexvert_xpos[k] - ph)    # anchor in hand frame
        m.eq_data[e, 3:6] = 0                                   # vertex body origin
        d.eq_active[e] = 1
        ids.append(e)
    return ids


def grip_patch(v, outer, i0):
    """Edge vertex, its two ring neighbours and the two nearest next-ring vertices: a clamped
    patch fixes the sheet's slope at the tips, like tweezer jaws (three edge vertices act as a hinge)."""
    n = len(outer)
    edge = [outer[(i0 + o) % n] for o in (-1, 0, 1)]
    inner = outer_ring(MEMBRANE_RINGS - 1)
    near = np.argsort(np.linalg.norm(v[inner] - v[outer[i0]], axis=1))[:2]
    return edge + [inner[i] for i in near]


def transfer(robot, grasp_angle=np.pi, tilt0=np.radians(45), slack=0.9, sweep_time=4.0,
             place_offset=(0.0, 0.0), release_height=0.0, rec=None, dest=None, captions=None, insets=True, slowmo=True, grasp_tol=0.003, speed=1.0, **_):
    """grasp_angle: direction (world yaw) of the grasped edge from the membrane centre.

    The fingers open along that direction u and the sheet extends from the tips towards -u.
    Roll-on: tips tilted by `tilt0` so the sheet slopes down to the far edge, far edge lands at
    the -u side of the plate, then the grasped edge travels +u and down along the sheet's
    remaining length while the tilt goes to zero, so the contact line sweeps far -> near.
    `place_offset` and `release_height` exist to produce deliberate failures (randomize.perturb).
    """
    m, d = robot.m, robot.d
    sweep_time = sweep_time / speed

    def mv(pos, rot=None, vmax=0.25, **kw):
        return robot.move_to(pos, rot, vmax=vmax * speed, **kw)
    # destination surface: the agar plate by default, or dest=(x, y, surface_z) (e.g. a frit)
    if dest is None:
        plate = d.body("plate").xpos[:2].copy() + np.asarray(place_offset)
        agar_z = d.body("plate").xpos[2] + AGAR_TOP
    else:
        plate, agar_z = np.asarray(dest[:2], float) + np.asarray(place_offset), float(dest[2])
    agar_v = agar_z + VERT_R
    C = dict(approach="Approach membrane edge on frit", grasp="Grasp edge (tweezer tips + attachment)",
             lift="Peel off frit and lift", transit="Transit to agar plate", touchdown="Far edge touches agar first",
             roll_on="Roll-on: contact line sweeps across", release="Release", retract="Retract", settled="Settled on agar")
    C.update(captions or {})
    v = d.flexvert_xpos
    c = v.mean(0)
    outer = outer_ring()
    ang = np.arctan2(v[outer, 1] - c[1], v[outer, 0] - c[0])
    i0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - grasp_angle))))))
    g_idx, far = outer[i0], outer[(i0 + len(outer) // 2) % len(outer)]
    u = np.array([np.cos(grasp_angle), np.sin(grasp_angle), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    G = v[g_idx].copy()
    F_final = np.r_[plate - R_MEM * u[:2], agar_v]
    G_final = np.r_[plate + R_MEM * u[:2], agar_v + TIP_CLEAR + 0.0005 + release_height]
    L = 2 * R_MEM
    hover = F_final + L * np.cos(tilt0) * u
    # the gripper is symmetric, so fingers along +u or -u grip the same; but one of the two needs
    # joint 6/7 beyond their limits for the tilted placement. Dry-run the IK through the key poses.
    se3 = lambda R, p: mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), np.asarray(p, float))
    tilt_about = lambda R, a: axis_angle(np.cross(u, z), a) @ R   # rotates the sheet's tangent (-u) towards -z

    def worst(R):
        return robot.ik.residual([se3(R, G + [0, 0, TIP_CLEAR]),
                                  se3(tilt_about(R, tilt0), [hover[0], hover[1], agar_v + 0.03]),
                                  se3(R, G_final)])
    R0 = min((down_rotation(u), down_rotation(-u)), key=worst)
    tilt = lambda a: tilt_about(R0, a)

    def cap(text, inset=None, slow=1.0):
        if rec:
            rec.caption = text
            if slowmo:
                rec.slowmo = slow
            if inset and insets:
                rec.inset = inset

    cap(C["approach"], "membrane_closeup")
    with robot.timed("approach"):
        mv(G + [0, 0, 0.06], R0, vmax=0.35)
        mv(G + [0, 0, TIP_CLEAR], vmax=0.05)
    if np.linalg.norm(robot.tcp_pos() - (G + [0, 0, TIP_CLEAR])) > grasp_tol:
        raise RuntimeError(f"grasp pose not reached: tcp {robot.tcp_pos().round(4)} vs {(G + [0, 0, TIP_CLEAR]).round(4)}")
    t_grasp = d.time
    cap(C["grasp"])
    with robot.timed("grasp"):
        robot.gripper(GRIP_CLOSED, 0.3)
        eqs = attach(robot, grip_patch(v, outer, i0))
    cap(C["lift"])
    with robot.timed("lift"):
        mv(G + [0, 0, 0.02], vmax=0.02)
        mv(G + [0, 0, 0.15], vmax=0.25)
    cap(C["transit"], "plate_side")
    with robot.timed("transit"):
        mv([hover[0], hover[1], 0.25], vmax=0.35)
        mv([hover[0], hover[1], agar_v + 0.07], tilt(tilt0), vmax=0.2)
        robot.wait(0.6 / speed)                                       # let the hanging sheet stop swinging
    cap(C["touchdown"], slow=2.0)
    with robot.timed("touchdown"):
        for _ in range(400):
            F = d.flexvert_xpos[far]
            if F[2] <= agar_v + 0.0008:
                break
            tgt = robot.target.translation().copy()
            tgt[:2] += np.clip(0.3 * (F_final[:2] - F[:2]), -0.002, 0.002)
            tgt[2] -= 0.0015
            mv(tgt, vmax=0.05, settle_max=0)
    cap(C["roll_on"], slow=4.0)
    with robot.timed("roll_on"):
        p0 = d.flexvert_xpos[far].copy()
        G0 = robot.target.translation().copy()

        def path(x):
            s = x * L                                         # laid length
            a = tilt0 * (1 - x)
            g = np.r_[p0[:2], G_final[2]] + s * u + (L - s) * slack * (np.cos(a) * u + np.sin(a) * z)
            g[:2] += (G_final[:2] - (p0[:2] + L * u[:2])) * x  # land exactly on G_final
            blend = max(0.0, 1 - x / 0.3) ** 2                # start from where touchdown left the tips
            return g + blend * (G0 - path0), tilt(a)
        path0 = np.r_[p0[:2], G_final[2]] + L * slack * (np.cos(tilt0) * u + np.sin(tilt0) * z)
        robot.follow(path, sweep_time)
    t_release = d.time
    cap(C["release"], slow=2.0)
    with robot.timed("release"):
        d.eq_active[eqs] = 0
        robot.gripper(GRIP_OPEN, 0.3)
        mv(robot.target.translation() + [0, 0, 0.01], vmax=0.02)
        cap(C["retract"])
        mv(robot.target.translation() + [0, 0, 0.12], vmax=0.25)
    cap(C["settled"])
    robot.wait(1.0 / speed)
    r = evaluate(d.flexvert_xpos.copy(), plate - np.asarray(place_offset), agar_z)
    r["transfer_time_s"] = t_release - t_grasp
    return r


def main(video=True, camera="oblique", overlay=True, seed=None, name="transfer_expert.mp4", inset=None):
    from transfer.randomize import sample
    cfg, ex = sample(seed) if seed is not None else (dict(mode="transfer"), {})
    m, d = load(**cfg)
    out = ROOT / "results"
    rec = Recorder(m, camera=camera, inset=inset or "membrane_closeup", overlay=overlay, path=out / name) if video else None
    robot = Robot(m, d, recorder=rec)
    if rec:
        rec.caption = "Membrane settles on frit"
    robot.wait(0.5)
    t0 = time.perf_counter()
    r = transfer(robot, rec=rec, insets=inset is None, **ex)
    wall = time.perf_counter() - t0
    for k, v in r.items():
        print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")
    print(f"sim {d.time:.1f} s, wall {wall:.1f} s")
    with open(out / "transfer_steps.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "seconds"])
        for name, a, b in robot.log:
            w.writerow([name, f"{b - a:.2f}"])
            print(f"  step {name:10s} {b - a:5.2f} s")
    if rec:
        print("wrote", rec.save(), rec.n, "frames")
    return r


if __name__ == "__main__":
    cam = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--camera=")), "oblique")
    name = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), "transfer_expert.mp4")
    inset = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--inset=")), None)
    main(video="--no-video" not in sys.argv, camera=cam, overlay="--clean" not in sys.argv, name=name, inset=inset)

"""Render a camera view of a saved line trace (no physics; fast to iterate on framing).

    python video/render_line.py --tag full --cam line_a --speed 8 [--t0 0 --t1 600] [--out results/x.mp4] [--stills 5]
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco
import imageio.v2 as iio2

from scene.line import load, update_telescopes
from control.line_hud import make_hud

RES = Path(__file__).resolve().parents[1] / "results"


def load_trace(tag):
    z = np.load(RES / f"line_{tag}_trace.npz")
    hud = json.loads((RES / f"line_{tag}_trace_hud.json").read_text())
    return {k: z[k] for k in z.files}, hud


def apply(m, d, tr, idx):
    G = len(tr["gids"])
    g = tr["g"][idx]
    gids = tr["gids"]
    m.geom_rgba[gids] = g[:4 * G].reshape(G, 4)
    m.geom_size[gids] = g[4 * G:7 * G].reshape(G, 3)
    m.geom_pos[gids] = g[7 * G:10 * G].reshape(G, 3)
    m.flex_rgba[0][3] = g[10 * G]
    d.qpos[:] = tr["q"][idx]
    d.qvel[:] = 0
    d.mocap_pos[:] = tr["mp"][idx]
    d.mocap_quat[:] = tr["mq"][idx]
    update_telescopes(m, d)
    mujoco.mj_forward(m, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--cam", default="line_a")
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--out", default="")
    ap.add_argument("--stills", type=int, default=0, help="write N evenly spaced PNG stills instead of a video")
    ap.add_argument("--no-hud", action="store_true")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    m, d = load()
    tr, hudtr = load_trace(a.tag)
    n, dt = len(tr["q"]), float(tr["dt"])
    t1 = min(a.t1, (n - 1) * dt)
    hud = make_hud(speed_label=f"{a.speed:g}x speed")
    ren = mujoco.Renderer(m, H, W)

    def frame_at(t):
        idx = min(int(round(t / dt)), n - 1)
        apply(m, d, tr, idx)
        ren.update_scene(d, camera=a.cam)
        fr = ren.render()
        return fr if a.no_hud else hud(fr, hudtr[idx])

    if a.stills:
        outdir = RES / "line_stills"
        outdir.mkdir(exist_ok=True)
        for k, t in enumerate(np.linspace(a.t0, t1, a.stills)):
            iio2.imwrite(outdir / f"{a.tag}_{a.cam}_{k:02d}.png", frame_at(t))
        print("wrote", a.stills, "stills to", outdir)
        return
    out = Path(a.out) if a.out else RES / f"line_{a.tag}_{a.cam}.mp4"
    w = iio2.get_writer(out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
    step = a.speed / a.fps
    t, k = a.t0, 0
    while t <= t1:
        w.append_data(frame_at(t))
        t += step
        k += 1
    w.close()
    print("wrote", out, k, "frames")


if __name__ == "__main__":
    main()

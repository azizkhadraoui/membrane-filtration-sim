"""Run the scripted expert over randomised episodes in parallel; log metrics and end-state crops.

    python transfer/eval_expert.py --n 100                       # nominal ranges (Phase B acceptance)
    python transfer/eval_expert.py --n 100 --scale 1.2 --tag wide
    python transfer/eval_expert.py --n 240 --perturb --tag perturbed   # failures for the defect classifier
"""
import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_episode(args):
    seed, scale, perturb, tag, speed = args
    import mujoco
    import imageio.v3 as iio
    from scene.cell import load
    from control.primitives import Robot
    from transfer.expert import transfer
    from transfer.randomize import sample

    cfg, expert = sample(seed, scale=scale, perturb=perturb)
    expert['speed'] = speed
    m, d = load(**cfg)
    robot = Robot(m, d)
    robot.wait(0.5)
    t0 = time.perf_counter()
    try:
        r = transfer(robot, **expert)
    except Exception as e:                       # a solver blow-up counts as a failure, not a crash
        r = dict(success=False, defect="sim_error", error=repr(e)[:200])
    r["wall_s"] = time.perf_counter() - t0
    out = ROOT / "results" / "episodes" / tag
    out.mkdir(parents=True, exist_ok=True)
    if np.isfinite(d.flexvert_xpos).all():
        # top-down crop centred on the plate for the defect classifier
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = d.body("plate").xpos + [0, 0, 0.012]
        cam.distance, cam.azimuth, cam.elevation = 0.16, 90.0, -90.0
        with mujoco.Renderer(m, 224, 224) as ren:
            ren.update_scene(d, camera=cam)
            iio.imwrite(out / f"{seed:05d}.png", ren.render())
        np.save(out / f"{seed:05d}_verts.npy", d.flexvert_xpos.astype(np.float32))
    row = dict(seed=seed, **{k: (json.dumps(v) if isinstance(v, (tuple, list, dict)) else v) for k, v in cfg.items()},
               **{f"expert_{k}": v for k, v in expert.items()}, **r)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--tag", default="nominal")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    jobs = [(a.seed0 + i, a.scale, a.perturb, a.tag, a.speed) for i in range(a.n)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(run_episode, jobs)):
            rows.append(row)
            print(f"[{i+1}/{a.n}] seed {row['seed']} {row.get('defect')} "
                  f"off {row.get('centroid_offset_mm', float('nan')):.1f} mm  ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["seed"])
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "seed", k))
    out = ROOT / "results" / f"expert_{a.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    succ = np.mean([bool(r.get("success")) for r in rows])
    from collections import Counter
    print(f"\n{a.tag}: success {succ*100:.0f}% over {len(rows)}  defects {dict(Counter(r.get('defect') for r in rows))}")
    print("wrote", out)


if __name__ == "__main__":
    main()

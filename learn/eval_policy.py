"""Evaluate the trained ACT policy in sim on held-out randomisation (20 % wider than training).

    python learn/eval_policy.py --n 100 --scale 1.2           # -> results/policy_eval.csv
    python learn/eval_policy.py --video 31337,31338 --scale 1.2  # rollout clips -> results/policy_<seed>.mp4
"""
import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "learn" / "act_ckpt.pt"
HORIZON = 320                  # 32 s at 10 Hz; expert demos are ~180-260 steps


def rollout(seed, scale, ckpt=CKPT, recorder_cam=None, device="cuda", ensemble_m=0.01):
    import torch
    torch.set_num_threads(2)
    from learn.env import TransferEnv
    from learn.act import Policy
    from control.primitives import Recorder
    env = TransferEnv(seed, scale=scale, visual=True)
    rec = None
    if recorder_cam:
        rec = Recorder(env.m, camera=recorder_cam, inset="plate_side", overlay=False,
                       path=ROOT / "results" / f"policy_{seed}.mp4")
        rec.caption = "Learned policy (ACT), held-out randomisation"
        env.robot.recorder = rec
    pol = Policy(ckpt, device=device, ensemble_m=ensemble_m)
    t0 = time.perf_counter()
    grasped_at = released_at = None
    t_end = HORIZON
    for t in range(HORIZON):
        a = pol(env.images(), env.state(), env.robot.target.translation())
        env.step(a)
        if env.attached and grasped_at is None:
            grasped_at = t
        if grasped_at is not None and not env.attached and released_at is None:
            released_at = t
        # done once the gripper has let go and the arm has lifted clear of the plate
        if released_at is not None and env.robot.tcp_pos()[2] > env.d.body("plate").xpos[2] + 0.08:
            t_end = t + 1
            break
    res = env.evaluate()
    c = env.d.flexvert_xpos[:, :2].mean(0) - env.d.body("plate").xpos[:2]   # placement error vector (world x, y)
    res.update(off_dx_mm=float(c[0] * 1e3), off_dy_mm=float(c[1] * 1e3))
    res.update(seed=seed, steps=t_end, grasped=grasped_at is not None, released=released_at is not None,
               transfer_time_s=(released_at - grasped_at) / 10 if released_at is not None else float("nan"),
               wall_s=round(time.perf_counter() - t0, 1))
    if rec:
        res["video"] = str(rec.save())
    return res


def _job(args):
    try:
        return rollout(*args)
    except Exception as e:
        return dict(seed=args[0], success=False, defect="error", error=repr(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--scale", type=float, default=1.2)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--video", default="")
    ap.add_argument("--camera", default="oblique")
    ap.add_argument("--tag", default="policy_eval")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--m", type=float, default=0.01, help="temporal-ensembling weight; negative favours newer predictions")
    a = ap.parse_args()
    if a.video:
        vrows = []
        for s in map(int, a.video.split(",")):
            r = rollout(s, a.scale, ckpt=Path(a.ckpt), recorder_cam=a.camera, ensemble_m=a.m)
            vrows.append(r)
            print(s, r.get("defect"), r.get("centroid_offset_mm"), r.get("video"), flush=True)
        keys = sorted({k for r in vrows for k in r}, key=lambda k: (k != "seed", k))
        with open(ROOT / "results" / f"{a.tag}_videos.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(vrows)
        return
    jobs = [(a.seed0 + i, a.scale, Path(a.ckpt), None, "cuda", a.m) for i in range(a.n)]
    rows, t0 = [], time.time()
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            print(f"[{i+1}/{a.n}] seed {r['seed']} {r.get('defect')} off {r.get('centroid_offset_mm', float('nan')):.1f} mm "
                  f"steps {r.get('steps')} ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["seed"])
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "seed", k))
    with open(ROOT / "results" / f"{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"success {np.mean([bool(r.get('success')) for r in rows])*100:.0f}% over {len(rows)}; "
          f"defects {dict(Counter(r.get('defect') for r in rows))}")


if __name__ == "__main__":
    main()

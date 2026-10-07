"""Roll out the scripted expert with visual domain randomisation; save successful episodes.

    python learn/gen_dataset.py --n 400            # -> learn/data/ep_XXXXX.npz + learn/data/index.csv
"""
import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

DATA_DEFAULT = Path(__file__).resolve().parent / "data"


def run(args):
    seed, out, noise = args
    from learn.env import TransferEnv
    t0 = time.perf_counter()
    try:
        env = TransferEnv(seed, scale=1.0, visual=True)
        ep, res = env.record_expert(noise_mm=noise, seed=seed)
    except Exception as e:
        return dict(seed=seed, success=False, defect="error", error=repr(e)[:200], steps=0, wall=0)
    keep = res.get("success") if noise <= 0 else res.get("centroid_offset_mm", 999) < 100   # labels are clean either way
    if keep:
        out = Path(out)
        np.save(out / f"ep_{seed:05d}_img.npy", ep.pop("images"))          # uncompressed: memory-mapped in training
        np.savez(out / f"ep_{seed:05d}.npz", **ep)
    return dict(seed=seed, success=bool(res.get("success")), defect=res.get("defect"),
                steps=len(ep["action"]), wall=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--out", default=str(DATA_DEFAULT))
    ap.add_argument("--noise", type=float, default=0.0, help="DART noise std in mm")
    a = ap.parse_args()
    data_dir = Path(a.out)
    data_dir.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    with Pool(a.procs) as pool:
        todo = [(s, str(data_dir), a.noise) for s in range(a.seed0, a.seed0 + a.n) if not (data_dir / f"ep_{s:05d}.npz").exists()]
        rows += [dict(seed=s, success=True, defect="flat", steps=len(np.load(data_dir / f"ep_{s:05d}.npz")["action"]), wall=0)
                 for s in range(a.seed0, a.seed0 + a.n) if s not in {t[0] for t in todo}]
        for i, r in enumerate(pool.imap_unordered(run, todo)):
            rows.append(r)
            print(f"[{i+1}/{a.n}] seed {r['seed']} {r['defect']} {r['steps']} steps ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["seed"])
    with open(data_dir / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "success", "defect", "steps", "wall", "error"])
        w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r["success"]]
    print(f"kept {len(ok)}/{len(rows)} successful demos, mean length {np.mean([r['steps'] for r in ok]):.0f} steps")


if __name__ == "__main__":
    main()

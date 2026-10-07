"""Rebuild a results/expert_<tag>.csv from saved end states (results/episodes/<tag>/*_verts.npy).

Used when an evaluation run is stopped early: each episode already saved its crop and final
vertex positions, and the plate pose follows from the seed, so metrics can be recomputed.
    python transfer/relabel.py perturbed --perturb
"""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scene.cell import PLATE_XY, AGAR_TOP
from transfer.metrics import evaluate
from transfer.randomize import sample

ROOT = Path(__file__).resolve().parents[1]


def main(tag, perturb):
    rows = []
    for f in sorted((ROOT / "results" / "episodes" / tag).glob("*_verts.npy")):
        seed = int(f.stem.split("_")[0])
        cfg, ex = sample(seed, perturb=perturb)
        plate = np.array(PLATE_XY) + np.array(cfg["plate_offset"])
        r = evaluate(np.load(f).astype(float), plate, AGAR_TOP)
        rows.append(dict(seed=seed, **{f"expert_{k}": v for k, v in ex.items()}, **r, relabelled=True))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "seed", k))
    out = ROOT / "results" / f"expert_{tag}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} episodes -> {out}; defects {dict(Counter(r['defect'] for r in rows))}")


if __name__ == "__main__":
    main(sys.argv[1], "--perturb" in sys.argv)

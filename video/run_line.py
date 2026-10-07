"""Run the line simulation once and save its state trace (rendered offline by video/render_line.py).

    python video/run_line.py --samples 8 --tag full
-> results/line_<tag>_trace.npz, line_<tag>_trace_hud.json, line_<tag>_run.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene.line import load, N_SAMPLES
from control.primitives import Robot
from control.line_ctrl import Line

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--filt", type=float, default=0.0, help="fixed filtration time (s); default random 70-110")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m, d = load()
    robot = Robot(m, d)
    filt = (a.filt, a.filt) if a.filt > 0 else (70.0, 110.0)
    line = Line(m, d, robot, n_samples=a.samples, filt=filt, seed=a.seed)
    t0 = time.perf_counter()
    summ = line.run()
    wall = time.perf_counter() - t0
    line.save_trace(RES / f"line_{a.tag}_trace.npz")
    out = dict(summary=summ, jobs=line.jobs, events=line.events, wall_s=round(wall),
               samples=line.samples, filt_range=filt, positions=len(line.P))
    (RES / f"line_{a.tag}_run.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"wall {wall:.0f} s; sim {summ['total_s']} s; samples {summ['samples']}; steady cycle {summ['steady_cycle_s']} s; "
          f"arm util {summ['arm_util']}; trace frames {len(line.trace_q)}")


if __name__ == "__main__":
    main()

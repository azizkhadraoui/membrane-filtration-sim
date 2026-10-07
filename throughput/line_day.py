"""Full-shift model of the station-based line (layout B), driven by durations measured in the MuJoCo run.

Resources: 1 arm, 1 doser, 1 plate shuttle/lid lifter, 1 transfer spot, N filtration positions, and a manifold
zone lock shared by the arm and the doser (as in the simulation). Per-sample flow:
  position -> arm loads dry membrane -> funnel clamps -> doser doses -> filtration (lognormal, with stalls)
  -> plate prepared at the spot (overlaps the end of filtration) -> arm wet transfer + tip wash
  -> position cleans itself (CIP) -> plate leaves via the shuttle.
Failures: a transfer fails with probability p_fail and is retried once (costs arm time + a discarded membrane).
Consumables: membrane pack, plate hotel, tip box, sanitant tank; the operator refills through the hatch.

    python throughput/line_day.py                 # table + results/line_day.json
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import simpy

ROOT = Path(__file__).resolve().parents[1]

# durations (s) measured in the sim; overwritten from results/line_full_run.json when present
DUR = dict(arm_dry=22.0, arm_wet=30.0, tip_wash=9.0, dose=28.0, dose_zone=13.0, plate_in=14.0, plate_out=16.0, cip=37.0)
CAP = dict(membranes=150, plates=100, tips=96, sanitant_samples=400)     # per operator refill


def load_measured(tag="full"):
    p = ROOT / "results" / f"line_{tag}_run.json"
    if not p.exists():
        return dict(DUR), False
    stats = json.loads(p.read_text())["summary"]["stats"]
    d = dict(DUR)
    for k in ("arm_dry", "arm_wet", "tip_wash", "dose", "plate_in", "plate_out", "cip"):
        if k in stats:
            d[k] = stats[k]["mean"]
    d["dose_zone"] = d["dose"] * 0.46
    return d, True


def run(n_pos=6, shift_h=8.0, filt_median=180.0, stall_p=0.04, stall_s=600.0, p_fail=0.02, dur=None, n_arms=1,
        seed=0, demand=None, lead=25.0, backlog=True):
    """Simulate one shift of samples arriving as a backlog (or at `demand` samples/hour)."""
    dur = dur or DUR
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    arm = simpy.PriorityResource(env, n_arms)
    doser = simpy.Resource(env, 1)
    zone = simpy.Resource(env, 1)
    shuttle = simpy.PriorityResource(env, 1)
    spot = simpy.Resource(env, 1)
    positions = simpy.Resource(env, n_pos)
    stats = dict(done=0, fails=0, arm_busy=0.0, doser_busy=0.0, cycle=[], finish=[], stalls=0, discarded=0)
    horizon = shift_h * 3600.0

    def filt_time():
        t = float(rng.lognormal(np.log(filt_median), 0.25))
        if rng.random() < stall_p:
            stats["stalls"] += 1
            t += stall_s
        return t

    def use_arm(t, prio=1, zone_held=True):
        with arm.request(priority=prio) as a:
            yield a
            if zone_held:
                with zone.request() as z:
                    yield z
                    yield env.timeout(t)
            else:
                yield env.timeout(t)
            stats["arm_busy"] += t

    def sample(idx):
        with positions.request() as pos:
            yield pos
            t_in = env.now
            yield from use_arm(dur["arm_dry"], prio=1)
            # dosing: tip + aspirate happen outside the zone, the funnel visit needs it
            with doser.request() as ds:
                yield ds
                yield env.timeout(dur["dose"] - dur["dose_zone"])
                with zone.request() as z:
                    yield z
                    yield env.timeout(dur["dose_zone"])
                stats["doser_busy"] += dur["dose"]
            t_filt = filt_time()
            plate_ready = env.event()
            plate_done = env.event()

            def plate_proc():
                yield env.timeout(max(0.0, t_filt - lead))
                with spot.request() as sp:
                    yield sp
                    with shuttle.request(priority=1) as s:
                        yield s
                        yield env.timeout(dur["plate_in"])
                    plate_ready.succeed()
                    yield plate_done                       # spot is held until the plate has left
            env.process(plate_proc())
            yield env.timeout(t_filt)
            yield plate_ready
            fails = 0
            while True:
                yield from use_arm(dur["arm_wet"], prio=0)
                if rng.random() < p_fail and fails == 0:
                    fails += 1
                    stats["fails"] += 1
                    stats["discarded"] += 1
                    continue
                break
            yield from use_arm(dur["tip_wash"], prio=0, zone_held=False)
            # plate out (shuttle, off the arm's critical path); the position cleans itself meanwhile
            def out():
                with shuttle.request(priority=0) as s:
                    yield s
                    yield env.timeout(dur["plate_out"])
                plate_done.succeed()
            env.process(out())
            yield env.timeout(dur["cip"])
            stats["done"] += 1
            stats["finish"].append(env.now)
            stats["cycle"].append(env.now - t_in)

    n_total = int(shift_h * 400) if backlog else int(demand * shift_h)
    def arrivals():
        for i in range(n_total):
            env.process(sample(i))
            if not backlog:
                yield env.timeout(3600.0 / demand)
            else:
                yield env.timeout(0)
    env.process(arrivals())
    env.run(until=horizon)
    fin = np.array(stats["finish"])
    per_hour = stats["done"] / shift_h
    mid = fin[(fin > 0.15 * horizon) & (fin < 0.95 * horizon)]
    steady = (len(mid) - 1) / ((mid[-1] - mid[0]) / 3600.0) if len(mid) > 5 else per_hour
    return dict(positions=n_pos, arms=n_arms, per_hour=per_hour, steady_per_hour=steady, min_per_sample=60.0 / steady,
                done=stats["done"], arm_util=stats["arm_busy"] / horizon, doser_util=stats["doser_busy"] / horizon,
                stalls=stats["stalls"], fails=stats["fails"], membranes_used=stats["done"] + stats["discarded"])


def autonomy(rate_per_hour, cap=CAP, extra_membranes=1.03):
    """Hours between operator visits for each consumable at the given throughput."""
    return {k: round(v / (rate_per_hour * (extra_membranes if k == "membranes" else 1.0)), 1) for k, v in cap.items()}


if __name__ == "__main__":
    dur, measured = load_measured()
    print("durations (%s):" % ("measured in sim" if measured else "defaults"), {k: round(v, 1) for k, v in dur.items()})
    rows = []
    print(f"{'positions':>9} {'samples/h':>10} {'min/sample':>11} {'100 samples':>12} {'arm util':>9} {'doser util':>11}")
    for n in (2, 3, 4, 6, 8, 10, 12):
        rs = [run(n_pos=n, dur=dur, seed=s) for s in range(5)]
        r = {k: float(np.mean([x[k] for x in rs])) for k in ("steady_per_hour", "min_per_sample", "arm_util", "doser_util", "per_hour")}
        r["positions"] = n
        rows.append(r)
        print(f"{n:>9} {r['steady_per_hour']:>10.1f} {r['min_per_sample']:>11.2f} {100 / r['steady_per_hour']:>10.1f} h {r['arm_util'] * 100:>8.0f}% {r['doser_util'] * 100:>10.0f}%")
    base = [run(n_pos=6, dur=dur, seed=s, filt_median=600.0, stall_p=0.0) for s in range(3)]
    slow = float(np.mean([b["steady_per_hour"] for b in base]))
    print("\n10 min filtration, 6 positions:", round(slow, 1), "samples/h")
    ref = next(r for r in rows if r["positions"] == 6)
    demands = {"100 samples/day": 100 / 24.0, "300 samples/day": 300 / 24.0, "full speed (6 positions)": ref["steady_per_hour"]}
    aut = {k: autonomy(v) for k, v in demands.items()}
    for k, v in aut.items():
        print(f"autonomy at {k}: hours between refills", v)
    out = ROOT / "results" / "line_day.json"
    out.write_text(json.dumps(dict(durations=dur, measured=measured, rows=rows, slow_6=slow,
                                   autonomy=aut, cap=CAP), indent=1))
    print("wrote", out)

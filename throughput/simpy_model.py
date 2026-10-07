"""Phase E: cycle time per sample vs number of parallel filtration positions.

Step durations are placeholders until Phase A logs real ones to results/step_durations.csv.
"""
import csv, statistics
from pathlib import Path
import simpy

# placeholders; replaced by results/step_durations.csv written by control/sequence.py
STEP = dict(plate_prep=20, membrane_to_frit=25, funnel_on=12, open_vessel=12, dose=10,
            filtration=180, funnel_off=12, transfer=18, lid_and_stack=25)
PRE = ("plate_prep", "membrane_to_frit", "funnel_on", "open_vessel", "dose")   # arm busy, position held
POST = ("funnel_off", "transfer", "lid_and_stack")                             # arm busy, then position frees
# funnel rinse/disinfection happens at the wash station, off the arm's critical path
DURATIONS_CSV = Path(__file__).resolve().parents[1] / "results" / "step_durations.csv"


def load_steps():
    steps = dict(STEP, _measured=False)
    if DURATIONS_CSV.exists():
        with open(DURATIONS_CSV) as f:
            for row in csv.DictReader(f):
                if row["step"] in steps:
                    steps[row["step"]] = float(row["seconds"])
        steps["_measured"] = True
    return steps


def sample(env, arms, positions, done, steps):
    with positions.request() as pos:
        yield pos
        t0 = env.now                                    # latency = time occupying a position
        with arms.request() as a:
            yield a
            for k in PRE:
                yield env.timeout(steps[k])
        yield env.timeout(steps["filtration"])          # arm is free here
        with arms.request(priority=0) as a:             # unloading beats loading
            yield a
            for k in POST:
                yield env.timeout(steps[k])
    done.append(env.now - t0)


def run(n_positions, n_arms=1, n_samples=100, steps=None, filtration=None):
    steps = dict(steps or load_steps())
    if filtration is not None:
        steps["filtration"] = filtration
    env = simpy.Environment()
    arms = simpy.PriorityResource(env, n_arms)
    positions = simpy.Resource(env, n_positions)
    done = []
    for _ in range(n_samples):
        env.process(sample(env, arms, positions, done, steps))
    env.run()
    return env.now / n_samples, statistics.mean(done)   # throughput cycle, per-sample latency


if __name__ == "__main__":
    steps = load_steps()
    arm_time = sum(steps[k] for k in PRE + POST)
    print(f"arm-busy time per sample: {arm_time:.0f} s  (lower bound on cycle with 1 arm)")
    for n in range(1, 8):
        cyc, lat = run(n, steps=steps)
        print(f"{n} positions: {cyc/60:.2f} min/sample throughput, {lat/60:.1f} min latency")

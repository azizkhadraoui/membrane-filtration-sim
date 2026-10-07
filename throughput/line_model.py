"""Station-based line model: arm only for membrane handling, fixed stations for the rest.

Compares three architectures over 100 samples:
  A  arm does everything (current demo; measured step times)
  B  line: fixed dosing head, lifting funnels with clean-in-place, plate shuttle; arm does the two membrane moves
  C  cassette consumable (membrane stays in a rigid base that snaps onto a media cassette): no wet-membrane handling
"""
import simpy
import statistics

FILTRATION = 180.0
CIP = 40.0            # rinse + sanitant + air-dry of a fixed funnel/frit, in place, position blocked
N_SAMPLES = 100


def run(arch, n_pos, filtration=FILTRATION, n_samples=N_SAMPLES):
    env = simpy.Environment()
    arm = simpy.PriorityResource(env, 1)
    doser = simpy.Resource(env, 1)
    shuttle = simpy.Resource(env, 1)
    positions = simpy.Resource(env, n_pos)
    done = []

    def use(res, t, prio=1):
        req = res.request(priority=prio) if isinstance(res, simpy.PriorityResource) else res.request()
        with req:
            yield req
            yield env.timeout(t)

    def sample_A(i):
        t0 = env.now
        with positions.request() as pos:
            yield pos
            yield from use(arm, 31 + 22 + 19 + 15 + 27)          # plate prep, membrane->frit, funnel on, open vessel, dose
            yield env.timeout(filtration)
            yield from use(arm, 17 + 23 + 38, prio=0)             # funnel off, transfer, lid + stack
        done.append(env.now - t0)

    def sample_B(i):
        t0 = env.now
        with positions.request() as pos:
            yield pos
            yield from use(arm, 12)                               # dry membrane: dispenser -> frit
            yield from use(doser, 15)                             # funnel lowers, fixed head doses from the vessel rack
            yield env.timeout(filtration)
            yield from use(arm, 18, prio=0)                       # funnel lifts; wet transfer frit -> agar (roll-on)
            env.process(cip(pos))                                 # position cleans itself; released afterwards
            cleaned = env.event(); cip_events[id(pos)] = cleaned
            yield cleaned
        env.process(use(shuttle, 15))                             # lid on, plate out: off the critical path
        done.append(env.now - t0)

    cip_events = {}

    def cip(pos):
        yield env.timeout(CIP)
        cip_events.pop(id(pos)).succeed()

    def sample_C(i):
        t0 = env.now
        with positions.request() as pos:
            yield pos
            yield from use(arm, 10)                               # single-use funnel/membrane unit onto manifold
            yield from use(doser, 15)
            yield env.timeout(filtration)
            yield from use(arm, 12, prio=0)                       # funnel off to waste; membrane base onto media cassette
        env.process(use(shuttle, 10))
        done.append(env.now - t0)

    fn = dict(A=sample_A, B=sample_B, C=sample_C)[arch]
    for i in range(n_samples):
        env.process(fn(i))
    env.run()
    return env.now / n_samples, statistics.mean(done), env.now / 3600


if __name__ == "__main__":
    print("throughput cycle (min/sample) and hours for 100 samples")
    print(f"{'positions':>10} | {'A arm-only':>18} | {'B line + arm':>18} | {'C cassette':>18}")
    for n in (2, 4, 6, 8, 12):
        cells = []
        for arch in "ABC":
            cyc, lat, hours = run(arch, n)
            cells.append(f"{cyc/60:5.2f} min  {hours:4.1f} h")
        print(f"{n:>10} | " + " | ".join(cells))
    print("\nslow filtration (10 min):")
    for n in (6, 8, 12):
        cells = [f"{run(arch, n, filtration=600)[0]/60:5.2f} min" for arch in "ABC"]
        print(f"{n:>10} | " + " | ".join(f"{c:>18}" for c in cells))

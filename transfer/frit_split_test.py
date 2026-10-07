"""Does splitting the frit into several geoms lift the 50-contacts-per-pair cap?"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco, numpy as np
from scene.membrane import flexcomp_xml


def frit_geoms(split):
    # base cylinder (no collision with membrane) + thin top tiles that carry the contacts
    if split == 1:
        return '<geom type="cylinder" size="0.025 0.02" friction="1.5 0.01 0.0001"/>'
    g = ['<geom type="cylinder" size="0.025 0.0195" pos="0 0 -0.0005" friction="1.5 0.01 0.0001"/>']
    # concentric annulus split into `split` angular sectors, built from thin boxes is fiddly;
    # use a hex pattern of thin cylinders covering r=0.025 (centre + 6 around)
    r = 0.0105
    centres = [(0, 0)] + [(2 * r * np.cos(a), 2 * r * np.sin(a)) for a in np.arange(6) * np.pi / 3]
    if split == 19:
        centres = [(0, 0)]
        r = 0.0062
        for ring, n in ((1, 6), (2, 12)):
            for k in range(n):
                a = 2 * np.pi * k / n
                centres.append((ring * 2 * r * np.cos(a), ring * 2 * r * np.sin(a)))
    for x, y in centres:
        g.append(f'<geom type="cylinder" size="{r*1.18:.4f} 0.0005" pos="{x:.4f} {y:.4f} 0.0195" '
                 f'friction="1.5 0.01 0.0001"/>')
    return "\n      ".join(g)


for rings in (4, 5, 6):
    for split in (1, 7, 19):
        xml = f"""<mujoco><option timestep="0.002" integrator="discrete" cone="elliptic" impratio="10"/>
  <worldbody><geom type="plane" size="1 1 .01"/>
    <body name="frit" pos="0 0 0.08">{frit_geoms(split)}</body>
    {flexcomp_xml(rings=rings)}
  </worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
        t0 = time.perf_counter(); maxcon = 0
        for _ in range(500):
            mujoco.mj_step(m, d); maxcon = max(maxcon, d.ncon)
        z = d.flexvert_xpos[:, 2]
        print(f"rings={rings} ({m.nflexvert:3d} v) frit geoms={split:2d}: ncon {maxcon:3d}, "
              f"z-range {1e3*(z.max()-z.min()):5.2f} mm, min below top {1e3*(0.10-z.min()):5.2f} mm, "
              f"wall {time.perf_counter()-t0:5.2f}s / 1.0s sim")

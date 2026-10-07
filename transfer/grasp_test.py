"""Plan §4.4 verification: settle membrane on frit, attach +x edge vertex, lift and drag."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco, numpy as np
from scene.membrane import flexcomp_xml


def build(rings=4, edge_vertex=None):
    xml = f"""<mujoco>
  <option timestep="0.002" integrator="discrete" cone="elliptic" impratio="10"/>
  <worldbody>
    <geom type="plane" size="1 1 .01"/>
    <body name="frit" pos="0 0 0.08">
      <geom type="cylinder" size="0.025 0.02" friction="1.5 0.01 0.0001"/>
    </body>
    <body name="tool" mocap="true" pos="0.0235 0 0.1004">
      <geom type="sphere" size="0.002" contype="0" conaffinity="0" rgba="1 0 0 1"/>
    </body>
    {flexcomp_xml(rings=rings)}
  </worldbody>
  <equality>
    <connect name="grasp" body1="tool" body2="membrane_{edge_vertex}" anchor="0 0 0" active="false"/>
  </equality>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def run(rings):
    # find +x edge vertex index from geometry rather than hard-coding
    from scene.membrane import disc_mesh
    pts, _ = disc_mesh(rings=rings)
    vi = int(np.argmax(pts[:, 0]))
    m = build(rings, vi); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
    t0 = time.perf_counter()
    maxcon = 0
    for _ in range(500):
        mujoco.mj_step(m, d); maxcon = max(maxcon, d.ncon)
    z_settle = d.flexvert_xpos[:, 2].copy()
    d.eq_active[eq] = 1
    for k in range(750):
        d.mocap_pos[0] = [0.0235 + 0.05 * k / 750, 0, 0.1004 + 0.03 * k / 750]
        mujoco.mj_step(m, d)
    wall = time.perf_counter() - t0
    v = d.flexvert_xpos
    frit_top = 0.10
    print(f"rings={rings} nvert={m.nflexvert} vi={vi} | settle: min z-frit {1e3*(z_settle.min()-frit_top):+.2f} mm, "
          f"max sag {1e3*(z_settle.max()-z_settle.min()):.2f} mm, max ncon {maxcon} | "
          f"after lift: grasped {v[vi,2]:.4f}, centre {v[0,2]:.4f}, min {v[:,2].min():.4f} | "
          f"wall {wall:.2f}s for {500+750}*2ms = {1250*0.002:.1f}s sim")


if __name__ == "__main__":
    for r in (4, 5, 6):
        run(r)

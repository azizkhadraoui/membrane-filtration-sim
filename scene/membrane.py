"""Flex membrane generator: 47 mm disc as a 2D flexcomp (type="direct")."""
import numpy as np


def disc_mesh(radius=0.0235, rings=4):
    pts = [(0.0, 0.0, 0.0)]
    ring_start = [0]
    for r in range(1, rings + 1):
        n = 6 * r
        ring_start.append(len(pts))
        for k in range(n):
            a = 2 * np.pi * k / n
            pts.append((radius * r / rings * np.cos(a), radius * r / rings * np.sin(a), 0.0))
    tris = [(0, 1 + k, 1 + (k + 1) % 6) for k in range(6)]
    for r in range(2, rings + 1):
        # zipper between rings: walk both rings in angle order, always advancing the one whose
        # next vertex comes first. Covers the annulus exactly once, all triangles CCW (+z).
        s0, n0 = ring_start[r - 1], 6 * (r - 1)
        s1, n1 = ring_start[r], 6 * r
        i = j = 0
        while i < n0 or j < n1:
            a0, a1 = s0 + i % n0, s1 + j % n1
            if j < n1 and (i == n0 or (j + 1) / n1 <= (i + 1) / n0):
                tris.append((a0, a1, s1 + (j + 1) % n1)); j += 1
            else:
                tris.append((a0, a1, s0 + (i + 1) % n0)); i += 1
    return np.array(pts), np.array(tris)


def flexcomp_xml(name="membrane", pos=(0, 0, 0.1004), radius=0.0235, rings=4,
                 mass=0.0004, young=2e7, thickness=1.2e-4, edge_solref="0.004 1", edge_solimp="0.99 0.999 0.0001", yaw=0.0, contype=1):
    pts, tris = disc_mesh(radius, rings)
    point = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts)
    elem = " ".join(" ".join(map(str, t)) for t in tris)
    return f"""
    <flexcomp name="{name}" type="direct" dim="2" pos="{pos[0]} {pos[1]} {pos[2]}" axisangle="0 0 1 {np.degrees(yaw):.4f}"
              radius="0.0003" mass="{mass}" rgba="0.95 0.95 0.9 1"
              point="{point}" element="{elem}">
      <edge equality="true" damping="0.002" solref="{edge_solref}" solimp="{edge_solimp}"/>
      <contact selfcollide="none" internal="false" condim="3" contype="{contype}" conaffinity="{contype}"
               solref="0.002 1" solimp="0.95 0.99 0.0005"/>
      <elasticity young="{young}" poisson="0.3" thickness="{thickness}" elastic2d="bend"/>
    </flexcomp>"""

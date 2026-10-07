"""Plan §6 metrics on the membrane after placement, from d.flexvert_xpos."""
from collections import defaultdict
import numpy as np

from scene.membrane import disc_mesh
from scene.cell import outer_ring

VERT_R = 0.0003           # flex vertex radius: a vertex resting on agar sits this far above it
AGAR_R = 0.043


def _mesh_graph(rings=4):
    pts, tris = disc_mesh(rings=rings)
    edges = {tuple(sorted(e)) for t in tris for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    nbr = defaultdict(set)
    for a, b in edges:
        nbr[a].add(b); nbr[b].add(a)
    rest = {e: np.linalg.norm(pts[e[0]] - pts[e[1]]) for e in edges}
    return edges, nbr, rest


EDGES, NBR, REST = _mesh_graph()
OUTER = set(outer_ring())
INTERIOR = [i for i in NBR if i not in OUTER]


def _components(nodes):
    nodes, seen, comps = set(nodes), set(), []
    for n in nodes:
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            u = stack.pop(); comp.append(u)
            for w in NBR[u] & nodes:
                if w not in seen:
                    seen.add(w); stack.append(w)
        comps.append(comp)
    return comps


def evaluate(v, plate_xy, agar_z):
    """v: (nvert, 3) vertex positions. Returns dict of metrics + 'success' + 'defect' label."""
    lift = v[:, 2] - (agar_z + VERT_R)
    centroid = v[:, :2].mean(0)
    off = float(np.linalg.norm(centroid - np.asarray(plate_xy)))
    radial = np.linalg.norm(v[:, :2] - np.asarray(plate_xy), axis=1)

    # fold: interior vertex sticking up above all its neighbours, or non-adjacent vertices touching
    peak = any(lift[i] - max(lift[j] for j in NBR[i]) > 0.0015 for i in INTERIOR)
    dd = np.linalg.norm(v[:, None] - v[None], axis=2)
    # rest spacing of the 61-vertex mesh is ~6 mm, so non-adjacent vertices within 2 mm = folded over
    touching = any((a, b) not in EDGES for a, b in np.argwhere(np.triu(dd < 0.002, 1)))
    fold = bool(peak or touching)

    pockets = [c for c in _components([i for i in INTERIOR if lift[i] > 0.0008]) if len(c) >= 3]
    stretch = max(np.linalg.norm(v[a] - v[b]) / REST[(a, b)] - 1 for a, b in EDGES)

    r = dict(
        centroid_offset_mm=off * 1e3,
        flatness_mm=float(lift.std() * 1e3),
        max_lift_mm=float(lift.max() * 1e3),
        fold=fold,
        air_pocket=bool(pockets),
        overhang=bool((radial > AGAR_R).any()),
        max_stretch_pct=float(stretch * 100),
        tear=bool(stretch > 0.05),
    )
    r["on_target"] = r["centroid_offset_mm"] <= 3.0
    r["success"] = bool(r["on_target"] and r["flatness_mm"] < 1.0 and r["max_lift_mm"] < 2.0
                        and not fold and not r["air_pocket"] and not r["overhang"])
    # single label for the Phase D defect classifier (plan's classes)
    r["defect"] = ("torn" if r["tear"] else "folded" if fold else "bubble" if r["air_pocket"]
                   else "misaligned" if not r["on_target"] or r["overhang"]
                   else "flat" if r["success"] else "lifted")
    return r

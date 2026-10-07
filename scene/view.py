"""Open the cell in the interactive MuJoCo viewer, or render stills with --stills.

    python scene/view.py            # interactive window (double-click to select, ctrl+drag to perturb)
    python scene/view.py --stills   # results/scene_<camera>.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
from scene.cell import load, ROOT


def stills(m, d, settle_steps=300):
    import imageio.v3 as iio
    for _ in range(settle_steps):
        mujoco.mj_step(m, d)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with mujoco.Renderer(m, 720, 1280) as r:
        for cam in ("oblique", "front", "overhead", "membrane_closeup", "panda/wrist"):
            r.update_scene(d, camera=cam)
            path = out / f"scene_{cam.replace('panda/', '')}.png"
            iio.imwrite(path, r.render())
            print("wrote", path)


if __name__ == "__main__":
    m, d = load()
    print(f"bodies {m.nbody}  geoms {m.ngeom}  flex verts {m.nflexvert}  actuators {m.nu}")
    if "--stills" in sys.argv:
        stills(m, d)
    else:
        import mujoco.viewer
        mujoco.viewer.launch(m, d)

# Membrane filtration automation: simulation study

MuJoCo simulation of a robot cell that automates membrane filtration (47 mm membrane, funnel and frit, agar plate)
for a food-microbiology lab. Two concepts:

- **v1, arm-only cell** (`scene/cell.py`): a Franka Panda does every step. Scripted expert for the membrane transfer
  (98% success), an ACT imitation policy trained on its demonstrations, perception on synthetic data, SimPy cycle-time model.
- **v2, Line B** (`scene/line.py`): fixed stations (gantry doser, lifting funnels with clean-in-place, plate hotel and
  shuttle, tip wash); the arm only handles the membrane. Station scheduler, shift model, offline video renderer.

Simulation only. The membrane is a thin-shell approximation, not a model of wet-membrane physics.

## Setup

```
python -m venv .venv && .venv\Scripts\activate        # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie assets/mujoco_menagerie
```

GPU PyTorch (CUDA build) is only needed for `learn/` and `perception/`. LeRobot is not used.

## Layout

| Folder | Contents |
|---|---|
| `scene/` | cell.py (v1), line.py (Line B), membrane.py (flex membrane generator), view.py |
| `control/` | ik.py (mink), primitives.py, sequence.py (v1 full sample), line_ctrl.py and line_hud.py (Line B) |
| `transfer/` | expert.py (edge grasp + roll-on), metrics.py, randomize.py, eval_expert.py |
| `learn/` | env.py, gen_dataset.py, act.py, train_act.py, eval_policy.py |
| `perception/` | segmentation, defect classifier, stall detector |
| `throughput/` | simpy_model.py (v1), line_day.py (Line B shift model), plots |
| `video/` | assemble.py (v1 video), run_line.py, render_line.py, assemble_line.py (Line B) |
| `report/` | page and PDF builders |
| `results/` | small result tables (CSV, JSON, PDF) |

## Main commands

```
python control/sequence.py                                   # v1: one full sample, video + step times
python transfer/eval_expert.py --n 100                       # expert, randomised runs
python learn/gen_dataset.py --n 300 --noise 2.0 --out learn/data_dart
python learn/train_act.py --data learn/data_dart --steps 30000
python learn/eval_policy.py --n 100 --scale 1.2              # learned policy, unseen seeds
python video/run_line.py --samples 8 --tag full              # Line B: simulate, save state trace
python throughput/line_day.py                                # Line B: shift model from measured times
python video/render_line.py --tag full --cam line_a --speed 8
python video/assemble_line.py                                # 127 s demo video
python report/build_line_pdf.py
```

## Notes learned the hard way

- MuJoCo caps flex contacts at 50 per flex-body pair: frit, agar and magazine surfaces are built from many small tile bodies.
- Plates and lids are mocap bodies. A free body held up only by qpos overrides gets pushed down by the membrane.
- Never zero `d.eq_active` wholesale: it disables the membrane's own edge-length constraints. Only the `grip*` equalities.
- Membrane parameters in `scene/membrane.py` were tuned for stability, not for wet-membrane physics.

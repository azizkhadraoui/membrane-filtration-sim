"""Episode randomisation for the transfer task (plan Phase B/C).

Nominal: membrane +-3 mm and any yaw on the frit, plate +-5 mm.
`scale` widens the ranges (1.2 = the 20 %-wider held-out evaluation set).
`perturb` also randomises the expert's own parameters so some placements fail, which gives
the Phase D defect classifier folded / bubble / misaligned examples with labels from the metrics.
`visual` adds domain randomisation (lighting, colours, camera jitter) for the learned policy.
"""
import numpy as np


def disc(rng, r):
    a, rr = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform())
    return (float(rr * np.cos(a)), float(rr * np.sin(a)))


def sample(seed, scale=1.0, perturb=False, visual=False):
    rng = np.random.default_rng(seed)
    cfg = dict(mode="transfer",
               membrane_offset=disc(rng, 0.003 * scale),
               membrane_yaw=float(rng.uniform(0, 2 * np.pi)),
               plate_offset=disc(rng, 0.005 * scale))
    if visual:
        cfg["dr"] = dict(
            light_pos=(0.4 + rng.uniform(-0.3, 0.3), -0.3 + rng.uniform(-0.3, 0.3), 1.6 + rng.uniform(-0.2, 0.2)),
            light_diffuse=float(rng.uniform(0.4, 0.8)),
            bench_rgb=tuple(float(c) for c in np.clip(0.9 + rng.uniform(-0.12, 0.06, 3), 0, 1)),
            agar_rgba=" ".join(f"{c:.3f}" for c in (*np.clip(np.array([0.86, 0.62, 0.38]) + rng.uniform(-0.08, 0.08, 3), 0, 1), 0.95)),
            cam_jitter=rng.uniform(-0.01, 0.01, 3),
        )
    expert = dict(grasp_angle=float(np.pi + rng.uniform(-0.15, 0.15) * scale))
    if perturb:
        mode = rng.choice(["fast", "flat", "steep", "offset", "slack", "early"])
        expert.update(dict(
            fast=dict(sweep_time=float(rng.uniform(0.4, 1.2))),
            flat=dict(tilt0=float(np.radians(rng.uniform(0, 15)))),
            steep=dict(tilt0=float(np.radians(rng.uniform(70, 85))), slack=float(rng.uniform(1.0, 1.25))),
            offset=dict(place_offset=disc(rng, rng.uniform(0.004, 0.025))),
            slack=dict(slack=float(rng.uniform(0.45, 0.7))),
            early=dict(release_height=float(rng.uniform(0.006, 0.02))),
        )[mode])
        expert["perturb_mode"] = str(mode)
    return cfg, expert

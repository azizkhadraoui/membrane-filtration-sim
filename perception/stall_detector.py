"""Stalled-filtration detector: track the liquid level in a funnel from images, flag "no progress".

    python perception/stall_detector.py            # sim scenarios -> results/perception/stall_sim.png
    python perception/stall_detector.py --real     # also the client video -> stall_real.png

Level estimation (pure CV, no simulator state):
  1. ROI: the funnel is located once, on a setup frame of the mounted empty funnel, by the
     Phase D segmentation model (connected "funnel" component nearest the image centre); fixed
     camera afterwards. The same setup frame gives an empty-funnel saturation profile that is
     subtracted (removes static coloured parts such as the collar). The real clip has no such
     frame, so there the raw profile is used.
  2. Inside the ROI, per-row "liquid score" = mean HSV saturation over the central columns
     (tinted sample vs clear / white plastic and air). Rows whose score exceeds an adaptive
     threshold (air reference = top rows of the ROI) count as liquid; the level is the highest
     liquid row of the run that starts at the funnel bottom, as a fraction of ROI height.
  3. A 3-frame median smooths single-frame glitches (glare, occlusion).

Stall rule: liquid present (level > EMPTY) and max-min of the level over the last WINDOW
real minutes < DELTA. A funnel that drains to empty is "done", never "stalled".

Time compression: the sim does not simulate filtration hydraulics. The liquid column of
funnel0 is animated by a scripted level(t) in *real* minutes and one frame is rendered per
10 s of real time, so a 30-minute run is 180 frames (a few seconds of rendering).
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

RES = ROOT / "results" / "perception"
REAL = ROOT / "perception" / "real_frames"
WINDOW_MIN = 10.0     # plan: "> 10 min -> fault"
DELTA = 0.03          # fraction of ROI height
EMPTY = 0.04
FRAME_S = 10.0        # real seconds per rendered frame
L0 = 0.06             # sim liquid column (m) at the start: 100 ml water + sample, roughly


# ---------------------------------------------------------------------------------- CV
def liquid_profile(roi_rgb):
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    w = roi_rgb.shape[1]
    sat = hsv[:, w // 4: 3 * w // 4, 1] / 255.0
    return cv2.GaussianBlur(sat.mean(1)[:, None], (1, 5), 0)[:, 0]


def estimate_level(roi_rgb, air_rows=0.12, k=0.06, ref=None):
    """Level as a fraction of ROI height (0 = funnel bottom = ROI bottom edge).
    `ref`: optional saturation profile of the mounted *empty* funnel (one calibration frame taken
    at setup); subtracting it removes static coloured parts such as the funnel's bottom collar."""
    prof = liquid_profile(roi_rgb)
    if ref is not None:
        # grey-dilate the reference by +-3 rows so 1-2 px camera jitter of edges is not "liquid"
        refd = np.max(np.stack([np.roll(ref, k) for k in range(-3, 4)]), 0)
        prof = np.clip(prof - refd, 0, None)
    H = len(prof)
    air = np.median(prof[: max(2, int(air_rows * H))])
    thr = air + max(k, 0.35 * (np.percentile(prof, 95) - air))
    liq = prof > thr
    # walk up from the bottom (skip a few rows of collar), tolerate short gaps
    start = int(0.03 * H)
    top, gap = None, 0
    for r in range(H - 1 - start, -1, -1):
        if liq[r]:
            top, gap = r, 0
        else:
            gap += 1
            if gap > max(3, H // 25) and top is not None:
                break
            if gap > H // 3 and top is None:          # bottom collar can hide up to ~1/4 of the ROI
                break
    if top is None:
        return 0.0
    return float((H - top) / H)


class StallDetector:
    def __init__(self, window_min=WINDOW_MIN, delta=DELTA, empty=EMPTY):
        self.window, self.delta, self.empty = window_min, delta, empty
        self.t, self.raw, self.lv = [], [], []

    def update(self, t_min, level):
        self.t.append(t_min); self.raw.append(level)
        self.lv.append(float(np.median(self.raw[-3:])))
        t = np.array(self.t); lv = np.array(self.lv)
        if lv[-1] <= self.empty:
            return "done"
        if t[-1] - t[0] < self.window:
            return "running"
        w = lv[t >= t[-1] - self.window]
        return "STALL" if w.max() - w.min() < self.delta else "running"


def funnel_roi(rgb, net=None):
    """Bounding box (x0, y0, x1, y1) of the funnel nearest the image centre, from the seg model."""
    from perception.train_seg import predict
    cls, _ = predict(net, rgb)
    mask = (cls == 3).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    H, W = mask.shape
    best = min(range(1, n), key=lambda i: (np.hypot(*(cent[i] - [W / 2, H / 2])) - 0.2 * np.sqrt(stats[i, 4])))
    x, y, w, h, _ = stats[best]
    return int(x), int(y), int(x + w), int(y + h)


# ---------------------------------------------------------------------------------- sim
SCENARIOS = {
    # level(t) as a fraction of the initial column, t in real minutes
    "normal (drains in ~4 min)": lambda t: np.clip(1 - t / 4.0, 0, 1) ** 1.6,
    "slow but progressing (~14 min)": lambda t: np.clip(1 - t / 14.0, 0, 1) ** 1.3,
    "stalled (membrane clogs)": lambda t: 0.45 + 0.55 * np.exp(-t / 1.5),
}


def render_scenarios(total_min=30.0, seed=0):
    import mujoco
    from scene.cell import load, FRIT_YS
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, d = load(mode="transfer")
    for _ in range(300):
        mujoco.mj_step(m, d)
    a = m.joint("funnel0").qposadr[0]
    d.qpos[a:a + 3] = [*d.body("frit0").xpos[:2], 0.099]
    d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
    g = m.geom("funnel0_liquid").id
    m.geom_rgba[g] = [0.95, 0.82, 0.35, 0.55]
    ren = mujoco.Renderer(m, 320, 240)
    times = np.arange(0, total_min + 1e-9, FRAME_S / 60)
    out = {}
    # setup calibration frame: funnel mounted, empty (liquid geom hidden)
    m.geom_rgba[g, 3] = 0.0
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.54, FRIT_YS[0], 0.14]; cam.distance, cam.azimuth, cam.elevation = 0.24, 180.0, -12.0
    ren.update_scene(d, camera=cam)
    out["_empty"] = ren.render().copy()
    for name, f in SCENARIOS.items():
        frames, gt = [], []
        for t in times:
            lv = L0 * float(f(t))
            m.geom_size[g, 1] = max(lv, 0.0005) / 2
            m.geom_pos[g, 2] = max(lv, 0.0005) / 2 + 0.001
            m.geom_rgba[g, 3] = 0.55 if lv > 0.001 else 0.0
            m.light_diffuse[0] = 0.6 + rng.normal(0, 0.03)           # flicker
            mujoco.mj_forward(m, d)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.54, FRIT_YS[0], 0.14] + rng.normal(0, 0.0008, 3)   # mount vibration
            cam.distance, cam.azimuth, cam.elevation = 0.24, 180.0 + rng.normal(0, 0.3), -12.0
            ren.update_scene(d, camera=cam)
            img = ren.render().astype(np.float32)
            img = np.clip(img + rng.normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)
            frames.append(img); gt.append(lv / 0.08)                  # funnel wall height 80 mm
        out[name] = (times, frames, np.array(gt))
    ren.close()
    return out


def run_sim(net):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = render_scenarios()
    empty = data.pop("_empty")
    roi = funnel_roi(empty, net)
    ref = liquid_profile(empty[roi[1]:roi[3], roi[0]:roi[2]])
    fig, axes = plt.subplots(len(data), 2, figsize=(11, 3.1 * len(data)), gridspec_kw=dict(width_ratios=[3.2, 1]))
    summary = {}
    for row, (name, (times, frames, gt)) in enumerate(data.items()):
        x0, y0, x1, y1 = roi
        det = StallDetector()
        est, flags = [], []
        for t, fr in zip(times, frames):
            lv = estimate_level(fr[y0:y1, x0:x1], ref=ref)
            est.append(lv); flags.append(det.update(t, lv))
        est = np.array(est); sm = np.array(det.lv)
        st = np.array([f == "STALL" for f in flags])
        t_flag = float(times[st.argmax()]) if st.any() else None
        done = [f == "done" for f in flags]
        t_done = float(times[int(np.argmax(done))]) if any(done) else None
        # level estimate vs truth, both as a fraction of ROI / funnel height (scale differs a bit)
        ax = axes[row, 0]
        ax.plot(times, gt, "--", color="0.55", lw=1.2, label="scripted level (eval only)")
        ax.plot(times, est, ".", color="#6a8dbf", ms=3, alpha=0.6, label="image estimate (raw)")
        ax.plot(times, sm, "-", color="#1f4e8c", lw=1.8, label="image estimate (median-3)")
        if st.any():
            ax.fill_between(times, 0, 1, where=st, color="#d62728", alpha=0.15, transform=ax.get_xaxis_transform(),
                            label="STALL flagged")
            ax.axvline(t_flag, color="#d62728", lw=1)
            ax.text(t_flag + 0.3, 0.9, f"stall flagged at {t_flag:.1f} min", color="#d62728", fontsize=9)
        if t_done is not None:
            ax.axvline(t_done, color="#2ca02c", lw=1)
            ax.text(t_done + 0.3, 0.8, f"empty / done at {t_done:.1f} min", color="#2ca02c", fontsize=9)
        ax.set_ylim(0, 1); ax.set_xlim(0, times[-1])
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("level (fraction of height)")
        if row == len(data) - 1:
            ax.set_xlabel("real time (min); 1 rendered frame = 10 s")
        if row == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
        k = int(np.searchsorted(times, 3.0))
        im = frames[k].copy()
        cv2.rectangle(im, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 1)
        axes[row, 1].imshow(im); axes[row, 1].axis("off")
        axes[row, 1].set_title("frame at 3 min, ROI from seg model", fontsize=8)
        mae = float(np.mean(np.abs(sm - gt)))
        summary[name] = dict(stall_flag_min=t_flag, done_min=t_done, level_mae_vs_script=mae)
    fig.suptitle(f"Stalled-filtration detector (sim): flag if level changes < {DELTA:.0%} of height over "
                 f"{WINDOW_MIN:.0f} min while liquid remains", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "stall_sim.png", dpi=130)
    plt.close(fig)
    return dict(roi=roi, window_min=WINDOW_MIN, delta=DELTA, empty=EMPTY, frame_s=FRAME_S, scenarios=summary)


# ---------------------------------------------------------------------------------- real
def run_real(fps=2.0, t_range=(8.0, 66.0)):
    """Client filtration video: handheld phone, so register every frame to a reference frame
    (ORB + RANSAC homography) and read the level in a fixed ROI on the bottom funnel.
    The ROI is drawn by hand once on the reference frame (documented, not tuned on the output)."""
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = REAL / "video_eHX_Ej245P4.mp4"
    rd = imageio.get_reader(str(path))
    vfps = rd.get_meta_data()["fps"]
    frames, times = [], []
    for t in np.arange(t_range[0], t_range[1], 1 / fps):
        try:
            frames.append(rd.get_data(int(round(t * vfps))).copy()); times.append(float(t))
        except IndexError:
            break
    rd.close()
    ref_i = int(np.argmin(np.abs(np.array(times) - 20.0)))
    ref = frames[ref_i]
    H, W = ref.shape[:2]
    # ROI on the bottom (blue) funnel, in reference-frame pixels: chosen by eye on the reference frame
    roi = REAL_ROI
    orb = cv2.ORB_create(3000)
    gref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    # features only on static structure (left half: manifold, bottles); hands move on the right
    fmask = np.zeros_like(gref); fmask[:, : int(0.55 * W)] = 255
    kr, dr = orb.detectAndCompute(gref, fmask)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    levels, ok, warped = [], [], []
    for fr in frames:
        g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        k, dd = orb.detectAndCompute(g, fmask)
        Hm = None
        if dd is not None and len(k) > 20:
            mt = sorted(bf.match(dd, dr), key=lambda x: x.distance)[:400]
            if len(mt) > 20:
                src = np.float32([k[x.queryIdx].pt for x in mt]); dst = np.float32([kr[x.trainIdx].pt for x in mt])
                Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                if inl is None or inl.sum() < 25:
                    Hm = None
        if Hm is None:
            levels.append(np.nan); ok.append(False); warped.append(None); continue
        wimg = cv2.warpPerspective(fr, Hm, (W, H))
        x0, y0, x1, y1 = roi
        levels.append(estimate_level(wimg[y0:y1, x0:x1])); ok.append(True); warped.append(wimg)
    times = np.array(times); levels = np.array(levels)
    # No stall flag here: the clip is ~1 min long and does not contain a full drain, so a 10-min
    # window cannot fire and a shortened window would only produce a meaningless flag.
    fig, ax = plt.subplots(1, 4, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[2.6, 1, 1, 1]))
    ax[0].plot(times, levels, ".", color="#6a8dbf", ms=4, label="raw estimate")
    sm = np.full(len(levels), np.nan)
    good = np.isfinite(levels)
    if good.sum() > 2:
        from scipy.ndimage import median_filter
        sm[good] = median_filter(levels[good], 5, mode="nearest")
    ax[0].plot(times, sm, "-", color="#1f4e8c", lw=1.8, label="median-5")
    for t, o in zip(times, ok):
        if not o:
            ax[0].axvline(t, color="0.85", lw=2, zorder=0)
    ax[0].set_ylim(0, 1); ax[0].set_xlabel("video time (s)"); ax[0].set_ylabel("level (fraction of ROI height)")
    ax[0].set_title("Client video, bottom funnel: image-based level (grey = registration failed).\n"
                    "Water poured ~11 s; no full drain within the clip, so no stall test possible.", fontsize=9, loc="left")
    ax[0].legend(fontsize=8)
    for j, tt in enumerate((times[0] + 2, 20.0, times[-1] - 4)):
        i = int(np.nanargmin(np.abs(times - tt)))
        im = (warped[i] if warped[i] is not None else frames[i]).copy()
        x0, y0, x1, y1 = roi
        cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 0), 2)
        if np.isfinite(levels[i]):
            yl = int(y1 - levels[i] * (y1 - y0))
            cv2.line(im, (x0 - 8, yl), (x1 + 8, yl), (255, 255, 0), 2)
        ax[1 + j].imshow(im[: int(0.75 * H)]); ax[1 + j].axis("off")
        ax[1 + j].set_title(f"t = {times[i]:.0f} s (registered)", fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "stall_real.png", dpi=120)
    plt.close(fig)
    return dict(video="video_eHX_Ej245P4.mp4", t_range_s=list(t_range), fps_sampled=fps, n_frames=len(frames),
                registered_frac=float(np.mean(ok)), level_first=float(np.nanmedian(levels[:6])),
                level_last=float(np.nanmedian(levels[-6:])), roi_ref_frame=list(roi),
                level_median_12_50s=float(np.nanmedian(levels[(times >= 12) & (times <= 50)])),
                level_min_max_12_60s=[float(np.nanmin(levels[(times >= 12) & (times <= 60)])),
                                      float(np.nanmax(levels[(times >= 12) & (times <= 60)]))],
                stall_flag="not applied: clip ~1 min, no complete drain visible")


REAL_ROI = (65, 268, 125, 358)   # x0, y0, x1, y1 on the t=20 s reference frame (360x640): inner bottom funnel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--no-sim", action="store_true")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    out = {}
    if not a.no_sim:
        from perception.train_seg import load_model
        net = load_model()
        out["sim"] = run_sim(net)
        print(json.dumps(out["sim"], indent=1))
    if a.real:
        out["real"] = run_real()
        print(json.dumps(out["real"], indent=1))
    p = RES / "stall_metrics.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()

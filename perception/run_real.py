"""Run the synthetic-trained segmentation model on the client's real footage.

    python perception/run_real.py

Writes to results/perception/:
  real_contact_sheet.png   every extracted real frame with its prediction (no selection)
  real_overlay_<name>.png  a fixed, pre-chosen list of frames (chosen by content before any
                           prediction was looked at; see OVERLAYS)
  sim_val_examples.png     held-out synthetic frames: image / ground truth / prediction
  real_metrics.json        point-label accuracy on hand-placed points (perception/real_points.json)
                           and predicted class fractions per frame
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2
import imageio.v3 as iio

from perception.render_labels import CLASSES, OUT as DATA
from perception.train_seg import load_model, predict

RES = ROOT / "results" / "perception"
REAL = ROOT / "perception" / "real_frames"
PAL = np.array([[0, 0, 0], [255, 0, 255], [40, 200, 40], [30, 120, 255], [255, 190, 0], [0, 230, 230], [235, 50, 50]],
               np.uint8)
# chosen by content before running the model: funnels + plates, membrane on frit, membrane over a
# plate, pipetting into funnel, disinfection (bottle pouring), funnel held close to the camera
OVERLAYS = {
    "funnels_plates": "filtration/ref_t20.png",
    "membrane_on_frit": "filtration/f_28.png",
    "membrane_to_plate": "filtration/f_29.png",
    "pipetting": "filtration/f_12.png",
    "disinfection_pour": "disinfection/d_03.png",
    "funnel_in_hand": "disinfection/d_15.png",
}


def overlay(rgb, cls, alpha=0.5):
    col = PAL[cls]
    out = rgb.astype(np.float32).copy()
    m = cls > 0
    out[m] = (1 - alpha) * out[m] + alpha * col[m]
    # class boundaries
    edge = cv2.morphologyEx(cls, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[edge] = 255
    return out.astype(np.uint8)


def legend(h, w=150):
    img = np.full((h, w, 3), 255, np.uint8)
    for i, c in enumerate(CLASSES):
        y = 20 + 26 * i
        cv2.rectangle(img, (8, y - 12), (28, y + 6), PAL[i].tolist(), -1)
        cv2.rectangle(img, (8, y - 12), (28, y + 6), (0, 0, 0), 1)
        cv2.putText(img, c, (36, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def text(img, s, y=18, scale=0.5):
    cv2.putText(img, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def frames():
    fs = sorted((REAL / "filtration").glob("*.png")) + sorted((REAL / "disinfection").glob("*.png"))
    return [f for f in fs]


def main():
    net = load_model()
    pts = json.loads((ROOT / "perception" / "real_points.json").read_text())
    preds, metrics = {}, dict(frames={}, points={})
    for f in frames():
        rgb = iio.imread(f)[..., :3]
        cls, prob = predict(net, rgb)
        key = f"{f.parent.name}/{f.name}"
        preds[key] = (rgb, cls, prob)
        frac = np.bincount(cls.ravel(), minlength=len(CLASSES)) / cls.size
        metrics["frames"][key] = {c: round(float(v), 4) for c, v in zip(CLASSES, frac)}
    # point-label check
    allp = []
    for key, plist in pts.items():
        if key.startswith("_"):
            continue
        rgb, cls, prob = preds[key]
        for x, y, c in plist:
            p = CLASSES[cls[y, x]]
            allp.append(dict(frame=key, x=x, y=y, true=c, pred=p, ok=p == c))
    per = {}
    for c in CLASSES:
        sel = [p for p in allp if p["true"] == c]
        if sel:
            per[c] = dict(n=len(sel), correct=int(sum(p["ok"] for p in sel)),
                          predicted_as={q: sum(p["pred"] == q for p in sel) for q in CLASSES if any(p["pred"] == q for p in sel)})
    fg = [p for p in allp if p["true"] != "background"]
    metrics["points"] = dict(n=len(allp), accuracy=float(np.mean([p["ok"] for p in allp])),
                             foreground_accuracy=float(np.mean([p["ok"] for p in fg])), per_class=per, points=allp,
                             note="hand-placed by eye on 3 frames before predictions were viewed; coarse sanity check, not a benchmark")
    # overlays
    for name, key in OVERLAYS.items():
        rgb, cls, prob = preds[key]
        ov = overlay(rgb, cls)
        conf = prob.max(-1)
        confv = cv2.applyColorMap((255 * conf).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1]
        if key in pts:
            for x, y, c in pts[key]:
                ok = CLASSES[cls[y, x]] == c
                cv2.circle(ov, (x, y), 5, (255, 255, 255), -1)
                cv2.circle(ov, (x, y), 4, PAL[CLASSES.index(c)].tolist(), -1)
                if not ok:
                    cv2.drawMarker(ov, (x, y), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 12, 2)
        panel = np.concatenate([text(rgb.copy(), "real frame"), text(ov, "prediction"),
                                text(confv.copy(), "max softmax"), legend(rgb.shape[0])], 1)
        iio.imwrite(RES / f"real_overlay_{name}.png", panel)
    # contact sheet: every frame, image above prediction
    th = 200
    tiles = []
    for key, (rgb, cls, _) in preds.items():
        w = int(rgb.shape[1] * th / rgb.shape[0])
        a = cv2.resize(rgb, (w, th), interpolation=cv2.INTER_AREA)
        b = cv2.resize(overlay(rgb, cls), (w, th), interpolation=cv2.INTER_AREA)
        tiles.append(text(np.concatenate([a, b], 0), key.split("/")[1][:-4], 14, 0.4))
    W = tiles[0].shape[1]
    tiles = [cv2.resize(t, (W, t.shape[0])) for t in tiles]
    ncol = 13
    rows = []
    for i in range(0, len(tiles), ncol):
        r = tiles[i:i + ncol]
        r += [np.full_like(tiles[0], 255)] * (ncol - len(r))
        rows.append(np.concatenate(r, 1))
    sheet = np.concatenate(rows, 0)
    leg = cv2.resize(legend(200, 150), (150, 200))
    hdr = np.full((40, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(hdr, "All extracted real frames (no selection): top = frame, bottom = synthetic-trained U-Net prediction",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    sheet = np.concatenate([hdr, sheet], 0)
    sheet[40:240, -150:] = leg
    iio.imwrite(RES / "real_contact_sheet.png", sheet)
    # held-out synthetic examples
    segm = json.loads((RES / "seg_metrics.json").read_text())
    rng = np.random.default_rng(1)
    rows = []
    for fn in rng.choice(segm["val_scene_files"], 6, replace=False):
        z = np.load(DATA / fn)
        i = rng.integers(len(z["img"]))
        img, gt = z["img"][i], z["lbl"][i]
        pr, _ = predict(net, img)
        rows.append(np.concatenate([text(img.copy(), fn[:-4], 14, 0.4), overlay(img, gt), overlay(img, pr)], 1))
    hdr = np.full((28, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(hdr, "held-out sim: image | ground truth | prediction", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA)
    a, b = np.concatenate(rows[:3], 0), np.concatenate(rows[3:], 0)
    iio.imwrite(RES / "sim_val_examples.png", np.concatenate([np.concatenate([hdr, hdr], 1), np.concatenate([a, b], 1)], 0))
    (RES / "real_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics["points"].items() if k != "points"}, indent=1))


if __name__ == "__main__":
    main()

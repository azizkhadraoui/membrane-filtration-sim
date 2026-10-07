"""Figures for the PDF report -> report/pdf_fig/*.png (all data read from results/ and learn/)."""
import csv
import json
from pathlib import Path

import numpy as np
import imageio.v2 as iio2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report" / "pdf_fig"
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
FONT = "C:/Windows/Fonts/seguisb.ttf"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(length=0)


def frames_at(clip, picks, size=(640, 360)):
    """picks: list of (predicate on caption meta, fraction within matching span, label)."""
    meta = json.loads(Path(clip).with_suffix(".json").read_text())
    want = []
    for pred, frac, label in picks:
        idx = [i for i, m in enumerate(meta) if pred(m.get("caption", ""))]
        want.append((idx[int(frac * (len(idx) - 1))] if idx else 0, label))
    order = sorted(set(i for i, _ in want))
    got = {}
    r = iio2.get_reader(clip)
    for i, fr in enumerate(r):
        if i in order:
            got[i] = fr
        if i >= order[-1]:
            break
    r.close()
    out = []
    for i, label in want:
        im = Image.fromarray(got[i]).resize(size, Image.LANCZOS)
        d = ImageDraw.Draw(im, "RGBA")
        f = ImageFont.truetype(FONT, 22)
        tw = d.textlength(label, font=f)
        d.rounded_rectangle([12, 12, 36 + tw, 50], 7, fill=(17, 22, 28, 215))
        d.text((24, 16), label, font=f, fill=(255, 255, 255))
        out.append(im)
    return out


def grid(images, cols, path, gap=8):
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), "white")
    for k, im in enumerate(images):
        canvas.paste(im, ((k % cols) * (w + gap), (k // cols) * (h + gap)))
    canvas.save(path, quality=90)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
    boxes = {
        "cell": (2, 30, 20, 11, "Sim cell\nMuJoCo 3.14, Panda,\nflex membrane"),
        "ctrl": (27, 30, 20, 11, "Control\nmink IK, primitives,\nkinematic carry"),
        "seq": (52, 30, 20, 11, "Full sequence\n9 steps, measured\nstep times"),
        "simpy": (77, 30, 21, 11, "Throughput model\nSimPy, cycle time\nvs positions"),
        "expert": (27, 15, 20, 11, "Scripted expert\nedge clamp +\nroll-on"),
        "eval": (2, 15, 20, 11, "Randomised eval\nmetrics from\nvertex positions"),
        "data": (52, 15, 20, 11, "Demonstrations\n298 episodes, 3 cams,\nnoise-injected"),
        "act": (77, 15, 21, 11, "ACT policy\ntrain 30k steps,\neval 100 runs"),
        "perc": (27, 0.5, 20, 11, "Perception\nsynthetic labels,\nU-Net, classifiers"),
        "real": (52, 0.5, 20, 11, "Client footage\n2 videos from the\nuse-case page"),
        "out": (77, 0.5, 21, 11, "Deliverables\n90 s video,\nresults page"),
    }
    col = {"cell": S1, "ctrl": S1, "seq": S1, "simpy": S1, "expert": S2, "eval": S2, "data": S2, "act": S2,
           "perc": S3, "real": S3, "out": INK2}
    for k, (x, y, w, h, t) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc=SURFACE, ec=col[k], lw=1.4))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.35)
    def arrow(a, b, dy=0):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=0, shrinkB=0))
    arrow((22.4, 35.5), (26.6, 35.5)); arrow((47.4, 35.5), (51.6, 35.5)); arrow((72.4, 35.5), (76.6, 35.5))
    arrow((37, 29.6), (37, 26.4)); arrow((26.6, 20.5), (22.4, 20.5)); arrow((47.4, 20.5), (51.6, 20.5)); arrow((72.4, 20.5), (76.6, 20.5))
    arrow((47.4, 6), (51.6, 6)); arrow((72.4, 6), (76.6, 6)); arrow((12, 14.6), (30, 11.9))
    for y, lab, c in ((42.6, "Cell and throughput", S1), (27.5, "Deformable transfer and learning", S2), (12.4, "Perception", S3)):
        pass
    fig.savefig(OUT / "pipeline.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_offsets():
    def offs(name):
        rs = list(csv.DictReader(open(RES / name)))
        v = np.array([float(r["centroid_offset_mm"]) for r in rs if r.get("centroid_offset_mm")])
        return v[v < 100]
    e, p2, p3 = offs("expert_nominal.csv"), offs("policy_eval_iter2.csv"), offs("policy_eval.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200, facecolor="white")
    style(ax)
    bins = np.arange(0, 20.5, 1.0)
    x = bins[:-1]
    for k, (v, c, lab) in enumerate(((e, S1, f"Scripted expert, nominal (n={len(e)})"),
                                     (p2, S2, f"Policy, previous iteration (n={len(p2)} transferred)"),
                                     (p3, S3, f"Policy, final (n={len(p3)} transferred)"))):
        h, _ = np.histogram(v, bins)
        ax.bar(x + 0.2 + k * 0.29, h, width=0.27, color=c, label=lab, zorder=3)
    ax.set_xticks(np.arange(0, 21, 2))
    ax.axvline(3, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=4)
    ax.text(3.2, ax.get_ylim()[1] * 0.92, "3 mm tolerance", fontsize=8.5, color=INK)
    ax.set_xlabel("Membrane centroid offset from plate centre (mm)")
    ax.set_ylabel("Runs")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "offsets.png", facecolor="white")
    plt.close(fig)


def fig_training():
    rel = json.loads((ROOT / "learn" / "train_log.json").read_text())
    ab = json.loads((ROOT / "learn" / "train_log_abs20k.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=200, facecolor="white")
    style(ax)
    for log, c, name in ((rel, S1, "Final policy (relative targets)"),):
        st = [r["step"] for r in log if r["step"] >= 500]
        ax.plot(st, [r["val_l1"] for r in log if r["step"] >= 500], color=c, lw=1.8, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Training step"); ax.set_ylabel("Validation L1 (normalised)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "training.png", facecolor="white")
    plt.close(fig)


def fig_strips():
    seq = RES / "sequence.mp4"
    picks = [(lambda c: "Plate from input" in c, 0.3, "1  Plate in, lid off"),
             (lambda c: "Membrane: roll-on" in c, 0.5, "2  Membrane onto frit"),
             (lambda c: "Funnel from wash" in c, 0.75, "3  Funnel on, twist"),
             (lambda c: "Dose sample" in c, 0.55, "5  Dose (pour)"),
             (lambda c: "Transfer: roll-on" in c, 0.5, "8  Membrane to agar"),
             (lambda c: "Lid on" in c, 0.8, "9  Plate to output stack")]
    grid(frames_at(seq, picks), 3, OUT / "sequence.jpg")
    tr = RES / "transfer_clean.mp4"
    picks = [(lambda c: "Peel off" in c, 0.6, "Peel and lift"),
             (lambda c: "Far edge" in c, 0.7, "Far edge touches first"),
             (lambda c: "Roll-on" in c, 0.45, "Roll-on"),
             (lambda c: "Settled" in c, 0.9, "Flat on agar")]
    grid(frames_at(tr, picks), 2, OUT / "rollon.jpg")


def fig_policy_grid():
    r = iio2.get_reader(RES / "membrane_demo.mp4")
    target = int(64.3 * 30)
    for i, fr in enumerate(r):
        if i == target:
            Image.fromarray(fr).save(OUT / "policy_grid.jpg", quality=90)
            break
    r.close()


if __name__ == "__main__":
    fig_pipeline(); fig_offsets(); fig_training(); fig_strips(); fig_policy_grid()
    print(sorted(p.name for p in OUT.iterdir()))

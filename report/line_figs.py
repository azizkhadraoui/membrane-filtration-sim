"""Figures for the line (layout B): Gantt of the measured run, arm-only vs line comparison, shift model.

    python report/line_figs.py   -> report/pdf_fig/line_*.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report" / "pdf_fig"
OUT.mkdir(exist_ok=True)

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#c9ccd1"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})


def gantt(tag="full", path=None, size=(11.5, 5.2), dpi=200):
    run = json.loads((RES / f"line_{tag}_run.json").read_text())
    jobs, samples = run["jobs"], run["samples"]
    T = run["summary"]["total_s"]
    lanes = ["Arm"] + ["Doser", "Plate shuttle"] + [f"Position {i + 1}" for i in range(run["positions"])]
    y = {n: len(lanes) - 1 - k for k, n in enumerate(lanes)}
    col = dict(arm_dry=BLUE, arm_wet=BLUE, tip_wash=BLUE, dose=ORANGE, plate_in=AQUA, plate_out=AQUA, cip=MAGENTA)
    fig, ax = plt.subplots(figsize=size, dpi=dpi, facecolor="white")
    ax.set_facecolor(SURF)
    h = 0.62
    for j in jobs:
        k = j["kind"]
        if k in ("arm_dry", "arm_wet", "tip_wash"):
            lane = "Arm"
        elif k == "dose":
            lane = "Doser"
        elif k in ("plate_in", "plate_out"):
            lane = "Plate shuttle"
        else:
            lane = f"Position {j['pos'] + 1}"
        light = 0.55 if k == "tip_wash" else 1.0
        ax.barh(y[lane], j["dur"], left=j["t0"], height=h, color=col[k], alpha=light, edgecolor="white", linewidth=0.6, zorder=3)
        if j["dur"] > 14 and lane != "Arm":
            ax.text(j["t0"] + j["dur"] / 2, y[lane], {"cip": "clean", "dose": f"S{j['sample'] + 1}", "plate_in": "in", "plate_out": "out"}.get(k, ""),
                    ha="center", va="center", fontsize=7.5, color="white", zorder=4)
        if lane == "Arm" and k != "tip_wash" and j["dur"] > 10:
            ax.text(j["t0"] + j["dur"] / 2, y[lane], f"S{j['sample'] + 1}", ha="center", va="center", fontsize=7.5, color="white", zorder=4)
    for s in samples:
        t = s["t"]
        if "dosed" in t and "filtered" in t:
            # find position of this sample from its cip job
            pos = next((j["pos"] for j in jobs if j["kind"] == "cip" and j["sample"] == s["id"]), None)
            if pos is not None:
                ax.barh(y[f"Position {pos + 1}"], t["filtered"] - t["dosed"], left=t["dosed"], height=h, color=GREY, edgecolor="white", linewidth=0.6, zorder=2)
                ax.text((t["dosed"] + t["filtered"]) / 2, y[f"Position {pos + 1}"], f"filtering S{s['id'] + 1}", ha="center", va="center", fontsize=7.5, color=INK2, zorder=4)
    ax.set_yticks([y[n] for n in lanes]); ax.set_yticklabels(lanes)
    ax.set_xlim(0, T)
    ax.set_xlabel("Simulated time (s); filtration is time-compressed to 70-110 s")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA, GREY, MAGENTA)]
    ax.legend(handles, ["Arm: membrane moves", "Doser", "Plate in/out", "Filtration", "Clean-in-place"], ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path or OUT / "line_gantt.png", facecolor="white")
    plt.close(fig)


def compare(path=None):
    """Arm-only cell (v1) vs line, shift model at 6 positions plus bar of arm time per sample."""
    day = json.loads((RES / "line_day.json").read_text())
    rows = day["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8), dpi=200, facecolor="white", gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axs[0]
    ax.set_facecolor(SURF)
    d = day["durations"]
    v1 = 191.0
    line_arm = d["arm_dry"] + d["arm_wet"] + d["tip_wash"]
    ax.barh([1], [v1], color=INK2, height=0.5)
    ax.barh([0], [line_arm], color=BLUE, height=0.5)
    ax.text(v1 + 3, 1, f"{v1:.0f} s", va="center", color=INK, fontsize=10)
    ax.text(line_arm + 3, 0, f"{line_arm:.0f} s", va="center", color=INK, fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Line B\n(membrane only)", "Arm-only cell\n(v1)"])
    ax.set_xlim(0, 230); ax.set_xlabel("Arm busy time per sample (s)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax = axs[1]
    ax.set_facecolor(SURF)
    pos = [r["positions"] for r in rows]
    ax.plot(pos, [r["steady_per_hour"] for r in rows], color=BLUE, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label="Line B (shift model)")
    ax.axhline(60.0 / 3.2, color=INK2, lw=1.2, ls=(0, (5, 3)))
    ax.text(pos[-1], 60.0 / 3.2 + 1.5, "arm-only cell (v1): 3.2 min per sample, 19 per hour at most", fontsize=8.5, color=INK2, ha="right")
    ax.axhline(60.0 / 5.0, color=ORANGE, lw=1.2, ls=(0, (2, 2)))
    ax.text(pos[-1], 60.0 / 5.0 - 4.5, "client target: 5 min per sample, 12 per hour", fontsize=8.5, color=ORANGE, ha="right")
    ax.set_xlabel("Parallel filtration positions"); ax.set_ylabel("Samples per hour")
    ax.set_ylim(0, max(r["steady_per_hour"] for r in rows) * 1.15)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path or OUT / "line_compare.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    gantt()
    if (RES / "line_day.json").exists():
        compare()
    print(sorted(p.name for p in OUT.glob("line_*")))

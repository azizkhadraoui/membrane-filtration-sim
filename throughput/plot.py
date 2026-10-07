"""Cycle time per sample vs parallel filtration positions (static PNG for the video and page).

Series: 1 arm @ 3 min filtration, 1 arm @ 10 min (slow/stalled filtration), 2 arms @ 3 min.
Durations come from results/step_durations.csv (Phase A sim) when present.
Also writes results/cycle_time.json for the interactive chart on the results page.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simpy_model import run, load_steps, PRE, POST

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [("1 arm, 3 min filtration", dict(n_arms=1, filtration=180), "#2a78d6"),
          ("1 arm, 10 min filtration", dict(n_arms=1, filtration=600), "#eb6834"),
          ("2 arms, 3 min filtration", dict(n_arms=2, filtration=180), "#1baf7a")]
NS = list(range(1, 9))


def compute():
    steps = load_steps()
    data = {name: [run(n, steps=steps, **kw)[0] / 60 for n in NS] for name, kw, _ in SERIES}
    arm = sum(steps[k] for k in PRE + POST)
    return steps, data, arm


def main(out=ROOT / "results" / "cycle_time.png", size=(12.8, 7.2), dpi=100):
    steps, data, arm = compute()
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 15})
    fig, ax = plt.subplots(figsize=size, dpi=dpi, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for name, _, col in SERIES:
        y = data[name]
        ax.plot(NS, y, color=col, lw=2.5, marker="o", ms=8, mec=SURFACE, mew=2, label=name, zorder=3)
    ax.axhline(5, color=INK, lw=1.5, ls=(0, (6, 4)), zorder=2)
    ax.text(1.05, 5.12, "5 min/sample target", color=INK, fontsize=13, va="bottom")
    ax.set_xlabel("Parallel filtration positions", color=INK2)
    ax.set_ylabel("Throughput cycle (min per sample)", color=INK2)
    ax.set_xticks(NS)
    ax.set_xlim(0.7, 8.3)
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.08)
    ax.grid(axis="y", color=GRID, lw=1); ax.grid(axis="x", visible=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    src = "step times measured in the sim cell" if steps.get("_measured") else "placeholder step times"
    ax.set_title(f"Cycle time per sample vs filtration positions  ({src}; arm busy {arm:.0f} s per sample)",
                 color=INK, fontsize=15, loc="left", pad=14)
    ax.legend(frameon=False, loc="upper right", fontsize=13, labelcolor=INK2)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.11)
    fig.savefig(out, facecolor=SURFACE)
    (ROOT / "results" / "cycle_time.json").write_text(json.dumps(dict(
        positions=NS, series=[dict(name=n, color=c, values=[round(v, 3) for v in data[n]]) for n, _, c in SERIES],
        arm_busy_s=arm, steps={k: v for k, v in steps.items() if not k.startswith("_")}, measured=steps.get("_measured"))))
    print("wrote", out)
    for n, _, _ in SERIES:
        print(f"  {n:28s}", " ".join(f"{v:4.2f}" for v in data[n]))


if __name__ == "__main__":
    main()

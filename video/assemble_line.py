"""Cut the Line B demo video from camera renders of the saved trace, charts and cards.

    python video/assemble_line.py [tag]   -> results/line_demo.mp4
Needs: results/line_<tag>_{line_a,line_b,line_c}.mp4 (same speed), line_<tag>_run.json, line_day.json,
       report/pdf_fig/line_gantt.png, line_compare.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as iio2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "pdf_fig"
W, H, FPS = 1280, 720, 30
FONTS = Path("C:/Windows/Fonts")
INK, PAPER, ACCENT, MUTED = (17, 22, 28), (245, 246, 244), (42, 120, 214), (112, 116, 122)


def font(size, kind="body"):
    name = {"display": "bahnschrift.ttf", "body": "segoeui.ttf", "semi": "seguisb.ttf"}[kind]
    return ImageFont.truetype(str(FONTS / name), size)


def card(title, lines=(), eyebrow=None, dark=False, foot=None, big=None):
    bg, fg, mu = (INK, PAPER, (170, 176, 184)) if dark else (PAPER, INK, MUTED)
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    x, y = 96, 150
    if eyebrow:
        d.text((x, y - 50), eyebrow.upper(), font=font(22, "semi"), fill=ACCENT)
    for ln in title.split("\n"):
        d.text((x, y), ln, font=font(54, "display"), fill=fg)
        y += 66
    y += 18
    for ln in lines:
        d.text((x, y), ln, font=font(27), fill=mu)
        y += 40
    if foot:
        d.text((x, H - 70), foot, font=font(19), fill=mu)
    return np.asarray(im)


def stat_card(title, rows, eyebrow=None, foot=None, dark=True):
    """rows: list of (label, value, sub) rendered as a table of big numbers."""
    bg, fg, mu = (INK, PAPER, (170, 176, 184)) if dark else (PAPER, INK, MUTED)
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    if eyebrow:
        d.text((96, 70), eyebrow.upper(), font=font(22, "semi"), fill=ACCENT)
    d.text((96, 104), title, font=font(46, "display"), fill=fg)
    y = 220
    for lab, val, sub in rows:
        d.text((96, y + 4), lab, font=font(27), fill=fg)
        if sub:
            d.text((96, y + 40), sub, font=font(19), fill=mu)
        vw = d.textlength(val, font=font(44, "display"))
        d.text((W - 96 - vw, y - 2), val, font=font(44, "display"), fill=fg)
        d.line([96, y + 78, W - 96, y + 78], fill=(60, 66, 74) if dark else (214, 217, 213), width=1)
        y += 96
    if foot:
        d.text((96, H - 62), foot, font=font(19), fill=mu)
    return np.asarray(im)


def fig_frame(path, caption=None, bg=(252, 252, 251)):
    im = Image.open(path).convert("RGB")
    s = min((W - 80) / im.width, (H - 150) / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    out = Image.new("RGB", (W, H), bg)
    out.paste(im, ((W - im.width) // 2, 40))
    if caption:
        d = ImageDraw.Draw(out)
        d.text((60, H - 80), caption, font=font(24, "semi"), fill=INK)
    return np.asarray(out)


def lower_third(frame, title, sub=None):
    im = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    ft, fs = font(28, "semi"), font(20)
    bw = int(max(d.textlength(title, font=ft), d.textlength(sub or "", font=fs)) + 52)
    bh = 62 if not sub else 94
    y0 = H - bh - 84
    x0 = 372
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], 10, fill=(17, 22, 28, 205))
    d.rectangle([x0, y0 + 14, x0 + 4, y0 + bh - 14], fill=ACCENT + (255,))
    d.text((x0 + 24, y0 + 12), title, font=ft, fill=(255, 255, 255, 255))
    if sub:
        d.text((x0 + 24, y0 + 52), sub, font=fs, fill=(205, 210, 216, 255))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


class Cut:
    def __init__(self, path):
        self.w = iio2.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
        self.n, self.marks = 0, []

    def put(self, fr):
        self.w.append_data(np.ascontiguousarray(fr[:, :, :3]))
        self.n += 1

    def hold(self, fr, secs):
        for _ in range(int(secs * FPS)):
            self.put(fr)

    def mark(self, name):
        self.marks.append((name, round(self.n / FPS, 1)))

    def close(self):
        self.w.close()


def main(tag="full"):
    run = json.loads((RES / f"line_{tag}_run.json").read_text())
    day = json.loads((RES / "line_day.json").read_text())
    S = run["summary"]
    st = S["stats"]
    cams = {c: RES / f"line_{tag}_{c}.mp4" for c in ("line_a", "line_b", "line_c", "line_d")}
    readers = {c: iio2.get_reader(p) for c, p in cams.items() if p.exists()}
    n_frames = min(r.count_frames() for r in readers.values())
    rows6 = next(r for r in day["rows"] if r["positions"] == 6)
    cyc_sim = S.get("steady_cycle_s")
    arm_line = st["arm_dry"]["mean"] + st["arm_wet"]["mean"] + st["tip_wash"]["mean"]

    cut = Cut(RES / "line_demo.mp4")
    cut.mark("title")
    cut.hold(card("Line B: a process,\nnot a human imitation",
                  ["The arm only does what needs hands: the wet membrane.",
                   "Dosing, plates, funnels and cleaning are fixed, fast stations."],
                  eyebrow="Membrane filtration cell · simulation", foot="MuJoCo simulation · Franka Panda · no hardware"), 5)
    cut.mark("problem")
    cut.hold(stat_card("Why change the concept", [
        ("Arm does every manual step", "3.2 min / sample", "v1: 191 s of arm time, fixed by what a technician does, in sequence"),
        ("Arm only for the membrane", f"{rows6['min_per_sample']:.1f} min / sample", f"Line B at 6 positions, {arm_line:.0f} s of arm time per sample (shift model, measured step times)"),
        ("Disinfection and supply", "built in", "clean-in-place on every position, tip wash after every sample, one-shift buffers"),
    ], eyebrow="Client feedback"), 8)

    cut.mark("tour")
    cut.hold(card("The cell", [
        "Membrane magazine  ·  6 lifting funnels with clean-in-place",
        "Gantry doser with disposable tips  ·  plate hotel, shuttle, lid lifter",
        "Tip wash for the tweezers  ·  sterile-water and sanitant tanks",
        "Operator hatch: racks of vessels in, plates out"], eyebrow="Layout"), 7)

    # ---- main run, camera schedule over the time-synchronised renders
    cut.mark("run")
    sched = [("line_a", 0.00), ("line_c", 0.14), ("line_a", 0.27), ("line_b", 0.40), ("line_c", 0.52), ("line_a", 0.64), ("line_d", 0.78), ("line_a", 0.90)]
    caps = [(0.00, "Samples enter one by one", "Each membrane goes dry from the magazine onto a free position"),
            (0.14, "Gantry doser", "Disposable tip, aspirates through the vessel septum, dispenses into the clamped funnel"),
            (0.27, "Positions run in parallel", "Filtration needs no arm; the arm serves whichever position is ready"),
            (0.40, "Top view: stations working at the same time", "Doser, plate shuttle and arm overlap"),
            (0.52, "Membrane transfer", "Funnel lifts, the arm rolls the wet membrane onto agar"),
            (0.64, "Clean-in-place while the arm moves on", "Water rinse, sanitant, rinse, air dry on the position itself"),
            (0.78, "Plate shuttle", "Lid off, plate in; lid on, plate to the output stack"),
            (0.90, "Tip wash", "Tweezer tips dipped in sanitant and air-dried after every sample")]
    cam_order = sched
    for k in range(n_frames):
        frac = k / max(n_frames - 1, 1)
        cam = [c for c, f0 in cam_order if frac >= f0][-1]
        fr = readers[cam].get_data(k) if cam in readers else readers["line_a"].get_data(k)
        # captions: pull the schedule's caption for this segment
        seg = [cp for cp in caps if frac >= cp[0]][-1]
        if frac - seg[0] < 0.07:
            fr = lower_third(fr, seg[1], seg[2])
        cut.put(fr)

    cut.mark("gantt")
    cut.hold(fig_frame(FIG / "line_gantt.png", "Measured in the simulation: the stations overlap, the arm is the only shared resource"), 8)
    cut.mark("results")
    cut.hold(stat_card("Result", [
        ("Membrane transfers flat and centred", f"{sum(1 for r in S['results'] if r.get('success'))} of {len(S['results'])}", "8 samples through 6 positions, measured in the sim run"),
        ("Arm time per sample", f"{arm_line:.0f} s", f"v1 arm-only cell: 191 s  ·  dose {st['dose']['mean']:.0f} s on the doser, CIP {st['cip']['mean']:.0f} s per position, in parallel"),
        ("Throughput at 6 positions", f"{rows6['steady_per_hour']:.0f} / h", f"shift model, measured step times, real filtration times (180 s, with stalls); target 12 / h"),
    ], eyebrow="Line B"), 9)
    cut.mark("compare")
    cut.hold(fig_frame(FIG / "line_compare.png", "More positions do not help an arm-only cell; the line scales until the arm is full"), 7)
    aut = day["autonomy"]
    cut.mark("supply")
    cut.hold(stat_card("Continuous supply", [
        ("Membranes (150 per magazine)", f"{aut['100 samples/day']['membranes']:.0f} h", "autonomy at 100 samples per day"),
        ("Plates (100 in the hotel)", f"{aut['100 samples/day']['plates']:.0f} h", "closed plates leave through the same hatch, ready for the incubator"),
        ("Tips and sanitant", f"{min(aut['100 samples/day']['tips'], aut['100 samples/day']['sanitant_samples']):.0f} h", "operator visit once per day; the cell stops accepting samples before a buffer runs dry"),
    ], eyebrow="Materials"), 8)
    cut.mark("next")
    cut.hold(card("What the simulation does not settle", [
        "Wet-membrane physics, real tweezer tips, and the sanitiser choice",
        "Next: real membrane tests on the Franka; price a dosing head, a manifold",
        "with lifting funnels and a plate shuttle against this concept."],
        eyebrow="Next", dark=True, foot="Station times and the shift model are from the simulation; filtration is time-compressed in the run."), 8)
    cut.close()
    print("wrote", RES / "line_demo.mp4", f"{cut.n / FPS:.0f} s")
    for n, t in cut.marks:
        print(f"  {t:6.1f} s  {n}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")

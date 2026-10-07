"""Cut the 90 s demo video (plan section 8.1) from rendered clips, stills and charts.

    python video/assemble.py     # -> results/membrane_demo.mp4 (1280x720, 30 fps)

Inputs (missing ones are skipped with a placeholder card and a warning):
  results/sequence.mp4 (+ .json captions), results/step_durations.csv   full per-sample sequence
  results/transfer_clean.mp4 (+ .json)                                   expert roll-on, slow motion
  results/policy_<seed>.mp4, results/policy_eval.csv                     learned-policy rollouts
  results/expert_*.csv                                                   expert success
  results/perception/*.png, results/perception/summary.json              perception on real footage
  results/cycle_time.png                                                 throughput chart
  perception/real_frames/*                                               client footage frames
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as iio2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
W, H, FPS = 1280, 720, 30
FONTS = Path("C:/Windows/Fonts")
INK, PAPER, ACCENT, MUTED = (17, 22, 28), (245, 246, 244), (42, 120, 214), (112, 116, 122)
GOOD, BAD = (12, 140, 60), (208, 59, 59)


def font(size, kind="body"):
    name = {"display": "bahnschrift.ttf", "body": "segoeui.ttf", "semi": "seguisb.ttf", "bold": "segoeuib.ttf"}[kind]
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except OSError:
        return ImageFont.load_default()


def warn(msg):
    print("WARNING:", msg, file=sys.stderr)


# -- frame helpers ----------------------------------------------------------------------------
def fit(img, w=W, h=H, bg=PAPER):
    """Letterbox an RGB array into w x h."""
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    im = im.convert("RGB")
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    out = Image.new("RGB", (w, h), bg)
    out.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return np.asarray(out)


def lower_third(frame, title, sub=None, tag=None):
    im = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    ft, fs = font(30, "semi"), font(21)
    tw = d.textlength(title, font=ft)
    sw = d.textlength(sub, font=fs) if sub else 0
    bw = int(max(tw, sw) + 48)
    bh = 66 if not sub else 98
    y0 = H - bh - 28
    d.rounded_rectangle([28, y0, 28 + bw, y0 + bh], 10, fill=(17, 22, 28, 205))
    d.rectangle([28, y0 + 14, 32, y0 + bh - 14], fill=ACCENT + (255,))
    d.text((52, y0 + 14), title, font=ft, fill=(255, 255, 255, 255))
    if sub:
        d.text((52, y0 + 56), sub, font=fs, fill=(205, 210, 216, 255))
    if tag:
        tf = font(20, "semi")
        tw2 = d.textlength(tag, font=tf)
        d.rounded_rectangle([28, 24, 60 + tw2, 62], 8, fill=(17, 22, 28, 205))
        d.text((44, 31), tag, font=tf, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def card(title, sub=None, eyebrow=None, foot=None, dark=False):
    bg, fg, mu = (INK, PAPER, (170, 176, 184)) if dark else (PAPER, INK, MUTED)
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    x = 96
    y = 220
    if eyebrow:
        d.text((x, y - 56), eyebrow.upper(), font=font(22, "semi"), fill=ACCENT)
    for line in title.split("\n"):
        d.text((x, y), line, font=font(60, "display"), fill=fg)
        y += 74
    if sub:
        y += 18
        for line in sub.split("\n"):
            d.text((x, y), line, font=font(28), fill=mu)
            y += 40
    if foot:
        d.text((x, H - 80), foot, font=font(20), fill=mu)
    return np.asarray(im)


class Cut:
    def __init__(self, path):
        self.w = iio2.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
        self.last = None
        self.n = 0
        self.marks = []

    def put(self, frame, fade_in=0):
        frame = np.ascontiguousarray(frame[:, :, :3])
        if fade_in and self.last is not None and self._fade_left > 0:
            a = 1 - self._fade_left / fade_in
            frame = (self.last_section * (1 - a) + frame * a).astype(np.uint8)
            self._fade_left -= 1
        self.w.append_data(frame)
        self.last = frame
        self.n += 1

    def section(self, name, fade=12):
        self.marks.append((name, self.n / FPS))
        self.last_section = self.last.astype(np.float32) if self.last is not None else None
        self._fade_left = fade if self.last is not None else 0
        return fade

    def hold(self, frame, secs, fade=0):
        for _ in range(int(secs * FPS)):
            self.put(frame, fade)

    def close(self):
        self.w.close()


def clip_frames(path, out_secs, start=0.0, end=1.0):
    """Yield frames of `path` resampled to out_secs (start/end as fractions of the clip)."""
    r = iio2.get_reader(path)
    n = r.count_frames()
    a, b = int(start * (n - 1)), int(end * (n - 1))
    want = np.linspace(a, b, int(out_secs * FPS)).round().astype(int)
    meta = []
    side = Path(path).with_suffix(".json")
    if side.exists():
        meta = json.loads(side.read_text())
    j = 0
    for i, fr in enumerate(r):
        while j < len(want) and want[j] == i:
            yield fr, (meta[i] if i < len(meta) else {}), i / max(n - 1, 1)
            j += 1
        if j >= len(want):
            break
    r.close()


def read_csv(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def rate(rows):
    return (100 * np.mean([r.get("success") == "True" for r in rows]), len(rows)) if rows else (float("nan"), 0)


def flat_rate(rows):
    """Laid flat anywhere on the agar: no fold, air pocket, overhang or tear, and actually transferred."""
    if not rows:
        return float("nan")
    ok = [all(r.get(k) == "False" for k in ("fold", "air_pocket", "overhang", "tear"))
          and float(r.get("centroid_offset_mm", "nan") or "nan") < 100 for r in rows]
    return 100 * np.mean(ok)


# -- sections ---------------------------------------------------------------------------------
def s_intro(cut):
    f = cut.section("intro", 0)
    cut.hold(card("Automated membrane filtration\nin simulation",
                  "Food microbiology lab  ·  sugar-solution samples  ·  up to 100 per day\nFranka arm cell, deformable membrane, learned transfer policy",
                  eyebrow="Proof of concept · simulation only", foot="Membrane filtration, ~100 samples/day, manual today"), 4.0, f)
    vid = ROOT / "perception" / "real_frames" / "video_eHX_Ej245P4.mp4"
    if not vid.exists():
        warn("client video missing")
        cut.hold(card("Today the process is manual", "Open vessel, dose 3-40 ml, filter, transfer the wet 47 mm membrane to agar"), 6.0)
        return
    f = cut.section("real", 10)
    base = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(base)
    d.text((96, 150), "TODAY", font=font(22, "semi"), fill=ACCENT)
    d.text((96, 190), "Every step is manual", font=font(54, "display"), fill=PAPER)
    y = 290
    for line in ("Open the vessel, dose 3-40 ml onto a 47 mm membrane",
                 "Vacuum-filter, rinse, take the funnel off",
                 "Lift the wet membrane onto agar with tweezers",
                 "Close, stack for incubation, disinfect the funnel"):
        d.ellipse([98, y + 12, 108, y + 22], fill=ACCENT)
        d.text((126, y), line, font=font(25), fill=(205, 210, 216))
        y += 50
    d.text((96, H - 90), "Client footage from the use-case page", font=font(19), fill=(140, 146, 154))
    vh = 640
    for fr, meta, x in clip_frames(vid, 6.0, start=0.12, end=0.20):
        im = Image.fromarray(fr).convert("RGB")
        im = im.resize((int(im.width * vh / im.height), vh), Image.LANCZOS)
        frame = base.copy()
        frame.paste(im, (W - im.width - 96, (H - vh) // 2))
        cut.put(np.asarray(frame), f)


def s_sequence(cut):
    path = RES / "sequence.mp4"
    if not path.exists():
        warn("sequence.mp4 missing"); cut.hold(card("Full per-sample sequence", "(clip missing)"), 20); return
    steps = read_csv(RES / "step_durations.csv")
    label = dict(plate_prep="Plate in, lid off", membrane_to_frit="Membrane onto frit", funnel_on="Funnel on + twist",
                 open_vessel="Open vessel", dose="Dose sample", filtration="Filtration", funnel_off="Funnel to wash",
                 transfer="Membrane to agar", lid_and_stack="Lid on, plate out")
    f = cut.section("sequence")
    for fr, meta, x in clip_frames(path, 20.0):
        cap = meta.get("caption", "")
        k = int(cap.split("/")[0]) if cap[:2].strip().isdigit() else 0
        text = cap.split("  ", 1)[1] if "  " in cap else cap
        im = Image.fromarray(fr).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        px, py, pw = W - 330, 24, 302
        d.rounded_rectangle([px, py, px + pw, py + 72 + 30 * len(steps)], 10, fill=(17, 22, 28, 200))
        d.text((px + 16, py + 10), "Step times in sim (s)", font=font(19, "semi"), fill=(255, 255, 255, 255))
        for i, s in enumerate(steps):
            y = py + 42 + 30 * i
            cur = (i + 1) == k
            col = (255, 255, 255, 255) if cur else (170, 176, 184, 255)
            if cur:
                d.rounded_rectangle([px + 8, y - 3, px + pw - 8, y + 25], 6, fill=ACCENT + (255,))
            name = label.get(s["step"], s["step"])
            val = f'{float(s["seconds"]):.0f}' + ("*" if s["step"] == "filtration" else "")
            d.text((px + 18, y), name, font=font(18), fill=col)
            d.text((px + pw - 18 - d.textlength(val, font=font(18, "semi")), y), val, font=font(18, "semi"), fill=col)
        d.text((px + 16, py + 46 + 30 * len(steps)), "* model value, shown as time-lapse", font=font(15), fill=(170, 176, 184, 255))
        frame = np.asarray(Image.alpha_composite(im, ov).convert("RGB"))
        cut.put(lower_third(frame, text or "Full per-sample sequence", "One complete sample on one filtration position",
                            tag="Sim  ·  sped up"), f)


def s_transfer(cut):
    path = RES / "transfer_clean.mp4"
    if not path.exists():
        warn("transfer_clean.mp4 missing"); cut.hold(card("Deformable membrane transfer", "(clip missing)"), 20); return
    nom, wide = read_csv(RES / "expert_nominal.csv"), read_csv(RES / "expert_wide.csv")
    (rn, nn), (rw, nw) = rate(nom), rate(wide)
    f = cut.section("transfer")
    total = 20.0
    for fr, meta, x in clip_frames(path, total, start=0.30, end=1.0):
        sm = meta.get("slowmo", 1.0)
        tag = f"Sim  ·  {sm:g}x slow motion" if sm > 1 else "Sim  ·  real time"
        cap = meta.get("caption", "") or "Scripted expert"
        frame = lower_third(fr, cap, "Wet membrane = thin elastic sheet (approximation); grasp by edge clamp", tag=tag)
        if x > 0.86 and nn:
            frame = lower_third(fr, f"Scripted expert: {rn:.0f}% flat and on target",
                                f"{nn} randomised runs (+{rw:.0f}% on {nw} runs with 20% wider ranges)", tag=tag)
        cut.put(frame, f)


def s_policy(cut, seeds):
    clips = [(s, RES / f"policy_{s}.mp4") for s in seeds if (RES / f"policy_{s}.mp4").exists()]
    rows = {int(r["seed"]): r for r in read_csv(RES / "policy_eval_videos.csv")}
    if not clips:
        warn("no policy clips"); cut.hold(card("Learned policy", "(clips missing)"), 20); return
    readers = [list(clip_frames(p, 15.0)) for _, p in clips[:4]]
    f = cut.section("policy")
    for i in range(int(15.0 * FPS)):
        canvas = Image.new("RGB", (W, H), INK)
        for j, frames in enumerate(readers):
            fr = frames[min(i, len(frames) - 1)][0]
            tile = Image.fromarray(fr).resize((W // 2 - 6, H // 2 - 6), Image.LANCZOS)
            x0, y0 = (j % 2) * (W // 2) + 3, (j // 2) * (H // 2) + 3
            canvas.paste(tile, (x0, y0))
            d = ImageDraw.Draw(canvas)
            r = rows.get(clips[j][0], {})
            ok = r.get("success") == "True"
            done = i > 0.85 * 15 * FPS
            offv = float(r.get('centroid_offset_mm', 'nan') or 'nan')
            why = "not grasped" if offv >= 100 else f"{r.get('defect', '?')} {offv:.0f} mm off"
            lab = (f"success · {offv:.1f} mm from centre" if ok else f"failure · {why}") if done else "running"
            col = (GOOD if ok else BAD) if done else (80, 86, 94)
            tw = d.textlength(lab, font=font(18, "semi"))
            d.rounded_rectangle([x0 + 12, y0 + 12, x0 + 32 + tw, y0 + 44], 7, fill=col)
            d.text((x0 + 22, y0 + 15), lab, font=font(18, "semi"), fill=(255, 255, 255))
        frame = lower_third(np.asarray(canvas), "Learned vision policy (ACT) on held-out randomisation",
                            "Inputs: top, wrist and plate cameras + joint state  ·  output: tip pose + gripper at 10 Hz")
        cut.put(frame, f)
    # metrics card
    pol = read_csv(RES / "policy_eval.csv")
    nom, wide = read_csv(RES / "expert_nominal.csv"), read_csv(RES / "expert_wide.csv")
    im = Image.fromarray(card("Membrane transfer, 100 runs each", None, eyebrow="Results", dark=True))
    d = ImageDraw.Draw(im)
    hx = [W - 360, W - 96]
    d.text((hx[0] - d.textlength("Laid flat", font=font(20, "semi")), 318), "Laid flat", font=font(20, "semi"), fill=(170, 176, 184))
    d.text((hx[1] - d.textlength("+ centred 3 mm", font=font(20, "semi")), 318), "+ centred 3 mm", font=font(20, "semi"), fill=(170, 176, 184))
    rows_t = [("Scripted expert (state-based), nominal ranges", nom),
              ("Scripted expert, ranges 20% wider", wide),
              ("Learned policy (vision), ranges 20% wider, unseen", pol)]
    y = 360
    for name, rs in rows_t:
        d.text((96, y), name, font=font(28), fill=PAPER)
        for x, val in zip(hx, (flat_rate(rs), rate(rs)[0])):
            t = f"{val:.0f}%" if rs else "n/a"
            d.text((x - d.textlength(t, font=font(40, "display")), y - 6), t, font=font(40, "display"), fill=PAPER)
        d.line([96, y + 48, W - 96, y + 48], fill=(60, 66, 74), width=1)
        y += 70
    d.text((96, y + 10), "Laid flat = no fold, air pocket, overhang or tear. The policy's limit is placement precision.",
           font=font(20), fill=(170, 176, 184))
    f = cut.section("metrics")
    cut.hold(np.asarray(im), 5.0, f)


def s_perception(cut):
    pdir = RES / "perception"
    summ = json.loads((pdir / "summary.json").read_text()) if (pdir / "summary.json").exists() else {}
    seg = summ.get("segmentation", {})
    pts = summ.get("real_eval", {}).get("points", {})
    miou = seg.get("val_miou", float("nan"))
    fg = pts.get("foreground_accuracy", float("nan"))
    for i, name in enumerate(("real_overlay_funnels_plates.png", "real_overlay_membrane_on_frit.png")):
        p = pdir / name
        if not p.exists():
            warn(f"missing {name}"); continue
        f = cut.section(f"perception{i}", 8)
        im = Image.open(p).convert("RGB")
        im = im.crop((0, 0, int(im.width * 720 / 1230), im.height))   # real frame + prediction panels
        img = fit(np.asarray(im), bg=INK)
        cut.hold(lower_third(img, "Sim-trained segmentation on client footage: does not transfer yet",
                             f"Sim validation mIoU {miou:.2f}  ·  real footage: {fg*100:.0f}% of hand-checked object points correct",
                             tag="Real footage"), 3.5, f)
    p = pdir / "stall_sim.png"
    if p.exists():
        st = summ.get("stall_detector", {}).get("sim", {}).get("scenarios", {})
        flag = next((v.get("stall_flag_min") for k, v in st.items() if "stall" in k and v.get("stall_flag_min")), None)
        f = cut.section("stall", 8)
        sub = (f"Clogged membrane flagged at {flag:.1f} min; normal and slow drains not flagged" if flag else
               "Level unchanged over a 10-minute window raises the flag")
        cut.hold(lower_third(fit(np.asarray(Image.open(p).convert("RGB")), bg=(252, 252, 251)),
                             "Stalled-filtration flag from the liquid level (sim)", sub + "  ·  real clip too short to validate"), 3.0, f)


def s_chart(cut):
    p = RES / "cycle_time.png"
    if not p.exists():
        warn("cycle_time.png missing"); return
    f = cut.section("chart")
    cut.hold(fit(np.asarray(Image.open(p).convert("RGB")), bg=(252, 252, 251)), 5.0, f)


def s_outro(cut):
    f = cut.section("outro")
    cut.hold(card("Next: the same stack on your Franka",
                  "Membrane physics is the open question; this is how we close it.\n"
                  "50-100 teleoperated demos on real membranes  ·  same metrics  ·  same perception",
                  eyebrow="What the sim does not prove",
                  foot="The membrane is a stable thin-shell approximation tuned for solver stability, not for wet-membrane physics.",
                  dark=True), 5.0, f)


def main():
    seeds = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else [])]
    out = RES / "membrane_demo.mp4"
    cut = Cut(out)
    s_intro(cut)
    s_sequence(cut)
    s_transfer(cut)
    s_policy(cut, seeds)
    s_perception(cut)
    s_chart(cut)
    s_outro(cut)
    cut.close()
    print(f"wrote {out}: {cut.n / FPS:.1f} s")
    for name, t in cut.marks:
        print(f"  {t:5.1f} s  {name}")


if __name__ == "__main__":
    main()

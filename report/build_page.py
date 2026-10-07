"""Build the one-page results report (plan section 8.2) from the result files.

    python report/build_page.py      # -> report/site/index.html + report/site/media/*

Everything shown is read from results/; nothing is typed in by hand except the scope text,
the known gaps and the Franka test plan.
"""
import csv
import html
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SITE = ROOT / "report" / "site"
MEDIA = SITE / "media"


def rows(name):
    p = RES / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def num(r, k):
    try:
        v = float(r.get(k, "nan"))
        return v
    except ValueError:
        return float("nan")


def summarize(rs):
    if not rs:
        return None
    ok = [r.get("success") == "True" for r in rs]
    offs = np.array([num(r, "centroid_offset_mm") for r in rs])
    offs = offs[np.isfinite(offs) & (offs < 100)]            # transferred runs only
    tt = np.array([num(r, "transfer_time_s") for r in rs])
    tt = tt[np.isfinite(tt)]
    flat = [all(r.get(k) == "False" for k in ("fold", "air_pocket", "overhang", "tear")) and num(r, "centroid_offset_mm") < 100
            for r in rs]
    moved = [num(r, "centroid_offset_mm") < 100 for r in rs]
    return dict(n=len(rs), success=100 * np.mean(ok), flat=100 * np.mean(flat), moved=100 * np.mean(moved),
                w5=100 * np.mean([num(r, "centroid_offset_mm") <= 5 for r in rs]),
                defects=Counter("not_grasped" if num(r, "centroid_offset_mm") >= 100 else r.get("defect") for r in rs),
                off_med=float(np.median(offs)) if len(offs) else float("nan"),
                off_p95=float(np.percentile(offs, 95)) if len(offs) else float("nan"),
                t_mean=float(tt.mean()) if len(tt) else float("nan"))


def still(src, name, width=1100, quality=82):
    """Copy an image into media/ as a compressed JPEG; returns the relative path or None."""
    src = Path(src)
    if not src.exists():
        return None
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    out = MEDIA / f"{name}.jpg"
    im.save(out, quality=quality, optimize=True)
    return f"media/{name}.jpg"


def defect_text(dc):
    if not dc:
        return "Not run."
    pc = dc.get("per_class", {})
    ar = dc.get("accept_reject", {})
    parts = [f"{k} {v['recall']:.0%} of {v['support']}" for k, v in pc.items() if v.get("support", 0) >= 5]
    return (f"Stratified 5-fold cross-validation on {dc.get('n_images', '?')} sim crops. Recall by class: "
            + ", ".join(parts) + f". As accept/reject: {ar.get('false_accepts', '?')} false accepts, "
            f"{ar.get('false_rejects', '?')} false rejects. Only one folded example exists, so folds are not yet evaluated.")


def stall_text(sd):
    sc = sd.get("sim", {}).get("scenarios", {})
    flag = next((v.get("stall_flag_min") for k, v in sc.items() if v.get("stall_flag_min")), None)
    t = ("In sim, a clogged membrane is flagged at %.1f min while normal and slow drains are not. " % flag) if flag else ""
    return t + "The client clip is about a minute long with no complete drain, so stall detection on real footage is not validated."


def e(s):
    return html.escape(str(s))


def pct(s):
    return "n/a" if s is None else f"{s['success']:.0f}%"


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(exist_ok=True)

    nominal, wide, pert = summarize(rows("expert_nominal.csv")), summarize(rows("expert_wide.csv")), summarize(rows("expert_perturbed.csv"))
    policy = summarize(rows("policy_eval.csv"))
    policy_prev = summarize(rows("policy_eval_iter2.csv"))
    steps = rows("step_durations.csv")
    cyc = json.loads((RES / "cycle_time.json").read_text()) if (RES / "cycle_time.json").exists() else None
    perc = json.loads((RES / "perception" / "summary.json").read_text()) if (RES / "perception" / "summary.json").exists() else {}
    train_log = json.loads((ROOT / "learn" / "train_log.json").read_text()) if (ROOT / "learn" / "train_log.json").exists() else []
    n_demos = len([p for p in (ROOT / "learn" / "data_dart").glob("ep_*.npz")])

    video = None
    if (RES / "membrane_demo_web.mp4").exists():
        shutil.copy(RES / "membrane_demo_web.mp4", MEDIA / "membrane_demo.mp4")
        video = "media/membrane_demo.mp4"
    poster = still(RES / "stills" / "cell_overview.png", "poster", width=1280) or still(RES / "scene_oblique.png", "poster")
    cell = poster
    rollon = still(RES / "stills" / "rollon.png", "rollon")
    perc_imgs = []
    captions = {
        "real_overlay_funnels_plates": "Funnels and plates on the bench. Funnels are only partly found; plates and agar are missed.",
        "real_overlay_membrane_on_frit": "Membrane being handled. Most of the scene is mislabelled; the perforated bench reads as plate and lid.",
        "real_overlay_membrane_to_plate": "Membrane carried to the plate. The agar plates are not found.",
        "real_overlay_disinfection_pour": "Disinfection clip. Bottle caps and hands are confused with funnels.",
    }
    for name in captions:
        p = RES / "perception" / f"{name}.png"
        if p.exists():
            perc_imgs.append((still(p, name, width=900), name))
    defect_cm = still(RES / "perception" / "defect_confusion.png", "defect_confusion", width=800)
    stall = still(RES / "perception" / "stall_sim.png", "stall_sim", width=900)

    # ---- transfer table ----
    def defect_cells(s):
        if not s:
            return "<td colspan=6>not run</td>"
        d = s["defects"]
        cols = ["flat", "misaligned", "folded", "bubble", "not_grasped"]
        other = sum(v for k, v in d.items() if k not in cols)
        return "".join(f"<td class=num>{d.get(c, 0)}</td>" for c in cols) + f"<td class=num>{other}</td>"

    def trow(label, sub, s):
        if not s:
            return f"<tr><th scope=row>{e(label)}<span class=sub>{e(sub)}</span></th><td class=num>n/a</td><td class=num>-</td><td class=num>-</td><td class=num>-</td>{defect_cells(s)}</tr>"
        return (f"<tr><th scope=row>{e(label)}<span class=sub>{e(sub)}</span></th>"
                f"<td class='num strong'>{s['success']:.0f}%</td><td class=num>{s['flat']:.0f}%</td><td class=num>{s['n']}</td>"
                f"<td class=num>{s['off_med']:.1f} / {s['off_p95']:.1f}</td>{defect_cells(s)}</tr>")

    transfer_rows = "".join([
        trow("Scripted expert", "state-based, nominal ranges", nominal),
        trow("Scripted expert", "ranges 20% wider", wide),
        trow("Learned policy (ACT)", "vision only, ranges 20% wider, unseen seeds", policy),
        trow("Learned policy, previous iteration", "no recovery data, two cameras; same seeds", policy_prev),
        trow("Expert, perturbed on purpose", "wrong tilt, speed, offset, release height", pert),
    ])

    # ---- steps table + chart data ----
    label = dict(plate_prep="Plate in from stack, lid off", membrane_to_frit="Membrane magazine to frit (roll-on)",
                 funnel_on="Funnel on, bayonet twist", open_vessel="Open sample vessel", dose="Dose (pour)",
                 filtration="Vacuum filtration", funnel_off="Funnel off to wash station",
                 transfer="Membrane frit to agar (roll-on)", lid_and_stack="Lid on, plate to output stack")
    step_rows = "".join(
        f"<tr><td>{e(label.get(s['step'], s['step']))}</td><td class=num>{float(s['seconds']):.0f} s</td>"
        f"<td class=muted>{'arm free (model value)' if s['step'] == 'filtration' else 'arm busy'}</td></tr>" for s in steps)
    arm_busy = sum(float(s["seconds"]) for s in steps if s["step"] != "filtration")
    cyc_json = json.dumps(cyc) if cyc else "null"

    perception_html = ""
    if perc_imgs:
        figs = "".join(f"<figure><img src='{src}' alt='Segmentation overlay on client footage, {e(name)}' loading=lazy>"
                       f"<figcaption>{e(captions[name])}</figcaption></figure>"
                       for src, name in perc_imgs if src)
        perception_html = f"<div class=gallery>{figs}</div>"

    def pget(*keys, fmt="{:.2f}", default="n/a"):
        v = perc
        for k in keys:
            if not isinstance(v, dict) or k not in v:
                return default
            v = v[k]
        try:
            return fmt.format(v)
        except (ValueError, TypeError):
            return e(v)

    final_val = train_log[-1]["val_l1"] if train_log else None

    page = TEMPLATE
    subs = {
        "VIDEO": (f"<video controls preload=metadata poster='{poster}' src='{video}'></video>" if video
                  else f"<img src='{poster}' alt='Simulated Franka cell'>"),
        "EXPERT_NOM": pct(nominal), "EXPERT_N": str(nominal["n"] if nominal else 0),
        "POLICY_FLAT": f"{policy['flat']:.0f}%" if policy else "n/a",
        "POLICY_MOVED": f"{policy['moved']:.0f}%" if policy else "n/a",
        "POLICY_OFF": f"{policy['off_med']:.1f}" if policy else "n/a",
        "POLICY_W5": f"{policy['w5']:.0f}%" if policy else "n/a",
        "POLICY": pct(policy), "POLICY_N": str(policy["n"] if policy else 0),
        "CYCLE": f"{min(cyc['series'][0]['values']):.1f}" if cyc else "n/a",
        "ARM": f"{arm_busy:.0f}",
        "TRANSFER_ROWS": transfer_rows,
        "STEP_ROWS": step_rows,
        "CYC_JSON": cyc_json,
        "PERCEPTION": perception_html,
        "SEG_MIOU": pget("segmentation", "val_miou"),
        "DEFECT_TEXT": defect_text(perc.get("defect_classifier", {})),
        "REAL_FG": pget("real_eval", "points", "foreground_accuracy", fmt="{:.0%}"),
        "REAL_N": pget("real_eval", "points", "n", fmt="{}"),
        "STALL_TEXT": stall_text(perc.get("stall_detector", {})),
        "DEFECT_CM": f"<img src='{defect_cm}' alt='Defect classifier confusion matrix'>" if defect_cm else "",
        "STALL": f"<img src='{stall}' alt='Stalled-filtration detector: liquid level over time'>" if stall else "",
        "N_DEMOS": str(n_demos),
        "VAL_L1": f"{final_val:.3f}" if final_val is not None else "n/a",
        "STEPS": f"{train_log[-1]['step']:,}" if train_log else "n/a",
        "ROLLON": f"<img src='{rollon}' alt='Roll-on placement of the membrane onto agar'>" if rollon else "",
        "CELL": f"<img src='{cell}' alt='Simulated cell with Franka arm, filter block, plates and vessels'>" if cell else "",
    }
    for k, v in subs.items():
        page = page.replace("{{" + k + "}}", v)
    (SITE / "index.html").write_text(page, encoding="utf-8")
    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"wrote {SITE / 'index.html'}  (site {size / 1e6:.1f} MB)")


TEMPLATE = (Path(__file__).resolve().parent / "template.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    main()

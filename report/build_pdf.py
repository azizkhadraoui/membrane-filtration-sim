"""Build the technical report PDF -> results/Membrane_Filtration_Sim_Report.pdf

Numbers are read from results/, learn/ and results/perception/; run report/pdf_figures.py first.
"""
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether, NextPageTemplate, CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "pdf_fig"
OUT = RES / "Membrane_Filtration_Sim_Report.pdf"

# ---------------------------------------------------------------- fonts & styles
F = "C:/Windows/Fonts/"
for name, file in (("UI", "segoeui.ttf"), ("UI-Semi", "seguisb.ttf"), ("UI-Bold", "segoeuib.ttf"),
                   ("UI-Italic", "segoeuii.ttf"), ("Mono", "consola.ttf")):
    pdfmetrics.registerFont(TTFont(name, F + file))
pdfmetrics.registerFontFamily("UI", normal="UI", bold="UI-Semi", italic="UI-Italic", boldItalic="UI-Semi")

INK = colors.HexColor("#12171c")
INK2 = colors.HexColor("#4d5560")
MUTED = colors.HexColor("#7c838c")
RULE = colors.HexColor("#d6d9d5")
ACCENT = colors.HexColor("#2a78d6")
TINT = colors.HexColor("#f2f4f3")
GOOD = colors.HexColor("#0b7a35")
BAD = colors.HexColor("#c03636")

ST = dict(
    body=ParagraphStyle("body", fontName="UI", fontSize=9.6, leading=14.2, textColor=INK, spaceAfter=6),
    small=ParagraphStyle("small", fontName="UI", fontSize=8.2, leading=11.4, textColor=INK2, spaceAfter=4),
    cap=ParagraphStyle("cap", fontName="UI-Italic", fontSize=8.2, leading=11, textColor=MUTED, spaceBefore=3, spaceAfter=10),
    h1=ParagraphStyle("h1", fontName="UI-Semi", fontSize=17, leading=21, textColor=INK, spaceBefore=4, spaceAfter=10),
    h2=ParagraphStyle("h2", fontName="UI-Semi", fontSize=12, leading=16, textColor=INK, spaceBefore=12, spaceAfter=5),
    h3=ParagraphStyle("h3", fontName="UI-Semi", fontSize=10, leading=14, textColor=INK2, spaceBefore=8, spaceAfter=3),
    bullet=ParagraphStyle("bullet", fontName="UI", fontSize=9.6, leading=14, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=3),
    cell=ParagraphStyle("cell", fontName="UI", fontSize=8.4, leading=11, textColor=INK),
    cellb=ParagraphStyle("cellb", fontName="UI-Semi", fontSize=8.4, leading=11, textColor=INK),
    cellh=ParagraphStyle("cellh", fontName="UI-Semi", fontSize=7.8, leading=10, textColor=INK2),
    code=ParagraphStyle("code", fontName="Mono", fontSize=8, leading=11, textColor=INK, backColor=TINT,
                        borderPadding=(5, 6, 5, 6), leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=10),
    callout=ParagraphStyle("callout", fontName="UI", fontSize=9.6, leading=14, textColor=INK, backColor=TINT,
                           borderPadding=(8, 10, 8, 10), leftIndent=10, rightIndent=10, spaceBefore=6, spaceAfter=14),
    toc1=ParagraphStyle("toc1", fontName="UI", fontSize=10, leading=17, textColor=INK),
)
W = A4[0] - 4.2 * cm


def P(text, s="body"):
    return Paragraph(text, ST[s])


def bullets(items, s="bullet"):
    return [Paragraph(t, ST[s], bulletText="•") for t in items]


def code(text):
    return Paragraph(text.replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"), ST["code"])


def table(rows, widths, header=True, num_cols=(), bold_first=False, zebra=False):
    data = []
    for i, r in enumerate(rows):
        row = []
        for j, c in enumerate(r):
            st = ST["cellh"] if (header and i == 0) else (ST["cellb"] if bold_first and j == 0 else ST["cell"])
            if j in num_cols and not (header and i == 0):
                st = ParagraphStyle("n", parent=st, alignment=2)
            if j in num_cols and header and i == 0:
                st = ParagraphStyle("nh", parent=st, alignment=2)
            row.append(Paragraph(str(c), st))
        data.append(row)
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.9, INK2), ("BACKGROUND", (0, 0), (-1, 0), TINT)]
    if zebra:
        style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafbfa")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def fig(path, width=1.0, caption=None, max_h=None):
    p = Path(path)
    if not p.exists():
        return [P(f"<i>[missing figure: {p.name}]</i>", "small")]
    w, h = PILImage.open(p).size
    dw = W * width
    dh = dw * h / w
    if max_h and dh > max_h:
        dh = max_h; dw = dh * w / h
    out = [Image(str(p), width=dw, height=dh)]
    if caption:
        out.append(P(caption, "cap"))
    return [KeepTogether(out)]


# ---------------------------------------------------------------- data
def rows(name):
    p = RES / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def fnum(r, k):
    try:
        return float(r.get(k))
    except (TypeError, ValueError):
        return float("nan")


def summ(rs):
    off = np.array([fnum(r, "centroid_offset_mm") for r in rs])
    moved = off < 100
    flat = [all(r.get(k) == "False" for k in ("fold", "air_pocket", "overhang", "tear")) and m for r, m in zip(rs, moved)]
    o = off[moved]
    tt = np.array([fnum(r, "transfer_time_s") for r in rs]); tt = tt[np.isfinite(tt)]
    return dict(n=len(rs), succ=100 * np.mean([r.get("success") == "True" for r in rs]), flat=100 * np.mean(flat),
                moved_pct=100 * np.mean(moved),
                moved=int(moved.sum()), med=np.median(o), p95=np.percentile(o, 95), w5=int((o <= 5).sum()),
                t=tt.mean() if len(tt) else float("nan"),
                defects=Counter("not grasped" if x >= 100 else r.get("defect") for r, x in zip(rs, off)))


NOM, WIDE, PERT, POL, POL2 = (summ(rows(n)) for n in ("expert_nominal.csv", "expert_wide.csv", "expert_perturbed.csv",
                                                       "policy_eval.csv", "policy_eval_iter2.csv"))
_pr = rows("policy_eval.csv")
POL["bias"] = (np.mean([fnum(r, "off_dx_mm") for r in _pr]), np.mean([fnum(r, "off_dy_mm") for r in _pr]))
POL["scatter"] = float(np.mean([np.std([fnum(r, "off_dx_mm") for r in _pr]), np.std([fnum(r, "off_dy_mm") for r in _pr])]))
PERT_MODES = Counter((r["expert_perturb_mode"], r["defect"]) for r in rows("expert_perturbed.csv"))
STEPS = rows("step_durations.csv")
CYC = json.loads((RES / "cycle_time.json").read_text())
PERC = json.loads((RES / "perception" / "summary.json").read_text())
TF = json.loads((RES / "policy_teacher_forced.json").read_text())
LOG = json.loads((ROOT / "learn" / "train_log.json").read_text())
_DEMOS = sorted((ROOT / "learn" / "data_dart").glob("ep_*.npz"))
N_DEMOS = len(_DEMOS)
DEMO_FRAMES = sum(len(np.load(f)["action"]) for f in _DEMOS)
ARM = sum(float(s["seconds"]) for s in STEPS if s["step"] != "filtration")
SEG = PERC["segmentation"]
PTS = PERC["real_eval"]["points"]
DC = PERC["defect_classifier"]
STALL = PERC["stall_detector"]["sim"]["scenarios"]
one_arm = CYC["series"][0]["values"]
slow = CYC["series"][1]["values"]
two_arm = CYC["series"][2]["values"]

# ---------------------------------------------------------------- document
class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=2.2 * cm,
                         bottomMargin=2.0 * cm, title="Membrane Filtration Automation: Simulation Study",
                         author="Simulation study team", subject="Technical report")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=self.cover),
                               PageTemplate("body", [frame], onPage=self.decor)])

    def cover(self, c, d):
        c.saveState()
        c.setFillColor(INK)
        c.rect(0, A4[1] - 9.5 * cm, A4[0], 9.5 * cm, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.rect(2.1 * cm, A4[1] - 9.5 * cm - 0.18 * cm, 3.2 * cm, 0.18 * cm, fill=1, stroke=0)
        c.restoreState()

    def decor(self, c, d):
        c.saveState()
        c.setFont("UI", 7.6); c.setFillColor(MUTED)
        c.drawString(2.1 * cm, A4[1] - 1.3 * cm, "Membrane filtration automation · simulation study · technical report")
        c.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, f"{d.page}")
        c.setStrokeColor(RULE); c.setLineWidth(0.5)
        c.line(2.1 * cm, A4[1] - 1.45 * cm, A4[0] - 2.1 * cm, A4[1] - 1.45 * cm)
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name == "h1":
            self.notify("TOCEntry", (0, f.getPlainText(), self.page))


def H1(t):
    return [CondPageBreak(6 * cm), Paragraph(t, ST["h1"])]


def build():
    s = []
    # ---- cover
    cover_title = ParagraphStyle("ct", fontName="UI-Semi", fontSize=27, leading=33, textColor=colors.white)
    cover_sub = ParagraphStyle("cs", fontName="UI", fontSize=12.5, leading=18, textColor=colors.HexColor("#c9d1da"))
    cover_eb = ParagraphStyle("ce", fontName="UI-Semi", fontSize=9, leading=12, textColor=colors.HexColor("#8fb8ec"))
    s += [Spacer(1, 0.6 * cm), Paragraph("TECHNICAL REPORT · SIMULATION ONLY", cover_eb), Spacer(1, 0.35 * cm),
          Paragraph("Automating membrane filtration<br/>with a robot arm: a simulation study", cover_title), Spacer(1, 0.4 * cm),
          Paragraph("What was built, what it shows, what it does not, and how to proceed", cover_sub),
          Spacer(1, 2.3 * cm)]
    s += fig(FIG / "sequence.jpg", 1.0, None)
    meta = [["Subject", "Food-microbiology lab, membrane filtration of sugar-solution samples (Birdwave use case)"],
            ["Robot", "Franka Emika Panda (simulated), Variant A: arm with tools"],
            ["Simulator", "MuJoCo 3.14 with deformable (flex) membrane; mink differential IK"],
            ["Status", "Simulation proof of concept complete; no hardware used"],
            ["Date", date(2026, 10, 5).strftime("%d %B %Y")]]
    s += [Spacer(1, 0.6 * cm), table(meta, [0.16, 0.84], header=False, bold_first=True)]
    s += [NextPageTemplate("body"), PageBreak()]

    # ---- contents
    toc = TableOfContents()
    toc.levelStyles = [ST["toc1"]]
    s += [Paragraph("Contents", ParagraphStyle("h1c", parent=ST["h1"])), toc, PageBreak()]

    # ---- 1 executive summary
    s += H1("1  Executive summary")
    s += [P("A food producer's microbiology lab filters up to 100 sugar-solution samples per day by hand. An integrator is preparing a paid "
            "feasibility study to automate it, and asked whether an AI-driven robot arm could take on the manipulation, above all the "
            "transfer of the wet 47&nbsp;mm membrane from the filtration frit onto an agar plate. With no hardware available, we built the "
            "whole cell in simulation to answer as much of that question as a simulator can, and to have a working pipeline ready "
            "for the real robot.")]
    s += [P("What we built", "h2")]
    s += bullets([
        "A simulated Variant A cell: Franka Panda, 6-position filter block, plate stacks, membrane magazine, sample vessels, wash station, and a deformable membrane.",
        "The complete per-sample sequence, from plate in to plate out (9 steps), executed by the arm with measured step times.",
        "A scripted expert for the critical membrane transfer (edge grasp and roll-on placement), tested on hundreds of randomised runs with objective quality metrics.",
        "A learned vision policy (ACT) trained on the expert's demonstrations and evaluated on unseen, wider randomisation.",
        "A throughput model (SimPy) of cycle time against the number of parallel filtration positions.",
        "Perception: segmentation trained on synthetic images and tested on the lab's own videos, a placement-defect classifier, and a stalled-filtration detector.",
        "A 90-second demo video and an interactive results page for the integrator.",
    ])
    s += [P("Key results", "h2")]
    s += [table([
        ["Question", "Result", "Confidence"],
        ["Can an arm cell reach and perform every step?", f"Yes. Full sequence runs end to end; arm busy {ARM:.0f}&nbsp;s per sample.", "High for kinematics and layout"],
        ["Does one arm meet 5 min per sample?", f"Yes: {min(one_arm):.1f}&nbsp;min/sample from 2 parallel positions; about {min(one_arm)*100/60:.1f}&nbsp;h for 100 samples.", "Medium; step times are conservative"],
        ["Can the membrane be laid flat by the arm?", f"Scripted expert: {NOM['succ']:.0f}% flat and centred (nominal), {WIDE['succ']:.0f}% with 20% wider ranges.", "Sim only; wet physics not modelled"],
        ["Can a learned vision policy do it?", f"Mostly: {POL['succ']:.0f}% flat and centred within 3&nbsp;mm, {POL['flat']:.0f}% laid flat, {POL['w5']} of {POL['n']} within 5&nbsp;mm (up from {POL2['succ']:.0f}% in the previous iteration).", "Sim only; the real policy is retrained"],
        ["Does sim-trained perception work on real footage?", f"No, not yet: {PTS['foreground_accuracy']*100:.0f}% of hand-checked object points correct (sim validation mIoU {SEG['val_miou']:.2f}).", "Needs real labelled data"],
    ], [0.30, 0.48, 0.22])]
    s += [Spacer(1, 6), P("<b>Bottom line.</b> The cell concept, the sequence and the throughput target hold up in simulation, and the hardest step "
                         "has a working manipulation strategy. The two things simulation cannot settle are the behaviour of a real wet membrane "
                         "and real-world perception; both need the real Franka and real consumables. Section 13 lays out how to get there: a one-week "
                         "Franka test with 50 to 100 teleoperated demonstrations, using the same pipeline, metrics and policy code.", "callout")]

    # ---- 2 background
    s += H1("2  Background and goals")
    s += [P("The process, per sample: open the sample vessel, dose 3 to 40&nbsp;ml onto a 47&nbsp;mm membrane sitting on a frit inside a funnel, "
            "vacuum-filter, rinse, remove the funnel, transfer the membrane onto an agar plate, close the plate, stack it for incubation, and "
            "disinfect the funnel and frit before the next sample. The lab targets about 5 minutes per sample and up to 100 samples per day. "
            "Filtration can take minutes, so several filtrations must run in parallel. The enclosure is semi-sterile, explicitly not GMP. "
            "The robot architecture is still open: Variant A (arm with tool changer) or Variant B (gantry with fixed tools).")]
    s += [P("The integrator's contact is preparing a priced first investigation for the lab and may offer remote access to a Franka arm. "
            "The purpose of this study was to give him evidence, within about three weeks and without hardware, that the manipulation and AI "
            "part is credible, and to show exactly where the remaining risk sits.")]
    s += [P("Goals of the simulation study", "h3")] + bullets([
        "Show a plausible arm-based cell layout and a complete per-sample motion sequence.",
        "Estimate cycle time against the 5-minute target from measured step durations.",
        "Demonstrate the membrane transfer with a deformable membrane, including a learned vision policy.",
        "Test a perception stack on the client's real footage.",
        "Package the evidence as a short video and a results page, scoped honestly.",
    ])

    # ---- 3 scope
    s += H1("3  What the simulation can and cannot show")
    s += [table([
        ["Shown in simulation", "Not shown (open questions)"],
        ["Cell layout and reach for an arm-based cell", "That a wet MCE membrane behaves like the simulated sheet"],
        ["A complete per-sample motion sequence", "Adhesion, surface tension and air entrapment on agar"],
        ["Cycle time against the 5-minute target", "Real gripper or tweezer performance"],
        ["A learned vision policy performing the transfer in sim", "Sim-to-real transfer of that policy"],
        ["Perception trained on synthetic data, run on real footage", "Reliability under real lab lighting and occlusion"],
        ["The whole software stack, end to end", "Production readiness, sterility, cleaning validation"],
    ], [0.5, 0.5])]
    s += [Spacer(1, 6), P("The membrane is a stable thin-shell approximation tuned for solver stability, not for wet-membrane physics. "
                         "The physics gap is exactly what the Franka phase is for.", "callout")]

    # ---- 4 overview
    s += H1("4  System overview")
    s += [P("The work is organised as one Python code base (<font name='Mono'>membrane-sim/</font>). The cell and control layer feed both the "
            "full-sequence demo and the transfer experiments; the scripted expert provides evaluation baselines, defect labels and the "
            "demonstrations used to train the learned policy; perception is trained on synthetic renders of the same cell and tested on the "
            "lab's footage.")]
    s += fig(FIG / "pipeline.png", 1.0, "Figure 1. Pipeline. Blue: cell and throughput. Orange: deformable transfer and learning. Green: perception.")
    s += [P("Compute: one Windows laptop (16 CPU threads, 32&nbsp;GB RAM, RTX 4060 Laptop GPU with 8&nbsp;GB). Everything ran locally; "
            "no cloud resources were used.", "small")]

    # ---- 5 cell
    s += H1("5  The simulated cell")
    s += [P("Units are metres, with the arm base at the origin and the table top at z&nbsp;=&nbsp;0. The layout mirrors a plausible Variant A "
            "cell. Movable objects (work plate, lid, one funnel, one vessel and its cap) have free joints; the rest are static.")]
    s += [table([
        ["Element", "Position / geometry", "Notes"],
        ["Franka Panda", "base at origin; Menagerie model", "gravity-compensated; 12&nbsp;mm tweezer tips added to the fingers"],
        ["Filter block", "6 frits at x&nbsp;=&nbsp;0.54, pitch 0.09&nbsp;m; frit top z&nbsp;=&nbsp;0.10", "positions 1–4 hold filled funnels in the scene"],
        ["Work plate (90&nbsp;mm)", "(0.30, 0.32); agar surface 12&nbsp;mm above the base", "free body; agar carried as child bodies"],
        ["Plate stacks", "input (0.18, 0.46), output (0.48, 0.46)", "10 and 3 plates"],
        ["Membrane magazine", "(0.30, −0.34)", "stack of 47&nbsp;mm membranes; top is a contact surface"],
        ["Vessel rack", "x&nbsp;=&nbsp;0.70, 5 vessels", "vessel 0 and its cap are free bodies"],
        ["Wash station", "(0.70, 0.40)", "funnel parking; disinfection is off the arm"],
        ["Cameras", "overview, plate side, top policy camera, wrist", "1280×720 for video, 128×128 for the policy"],
    ], [0.2, 0.42, 0.38])]
    s += fig(RES / "stills" / "cell_overview.png", 0.85, "Figure 2. The simulated cell during the funnel and dosing steps.")
    s += [P("5.1  The membrane model", "h2"),
          P("The membrane is a 47&nbsp;mm disc meshed with 61 vertices (four rings) as a MuJoCo 2D flex with bending elasticity. Each vertex is "
            "a small body, so the sheet can drape, hang and fold. Contact uses only the vertex spheres (0.3&nbsp;mm radius). Self-collision is "
            "off for stability.")]
    s += [table([
        ["Parameter", "Value", "Why"],
        ["Mass, thickness", "0.4&nbsp;g, 0.12&nbsp;mm", "typical 47&nbsp;mm MCE membrane"],
        ["Young's modulus", "20&nbsp;MPa (2e7&nbsp;Pa)", "plausible for wet nitrocellulose; 0.2&nbsp;MPa crumpled like tissue"],
        ["Edge constraint solref / solimp", "0.004&nbsp;1 / 0.99&nbsp;0.999&nbsp;0.0001", "default (0.02) let the sheet stretch 25–60% when lifted; now under 1%"],
        ["Integrator, timestep", "discrete, 2&nbsp;ms", "required by flex elasticity in MuJoCo 3.x"],
        ["Frit friction", "1.5", "stands in for wet adhesion"],
    ], [0.3, 0.3, 0.4])]
    s += [P("5.2  Contact surfaces", "h2"),
          P("MuJoCo caps contacts between a flex and any single body at 50. With 61 vertices on a one-piece frit, 11 vertices were left "
            "unsupported and the sheet sagged. Splitting the frit into several geoms on one body did not help; splitting it into nine thin "
            "tile <i>bodies</i> did (sag dropped from 0.5&nbsp;mm to 0.03&nbsp;mm at 61 vertices, and from 21&nbsp;mm to 0.03&nbsp;mm at 127). "
            "Frits, agar and the magazine top are all built this way. The tiles are invisible and collide only with the membrane.")]
    s += [P("5.3  Grasping and carrying", "h2"),
          P("Pinching a 0.3&nbsp;mm-radius vertex between two capsules is numerically fragile, so the membrane is grasped by attachment, a "
            "standard cloth-simulation technique. When the tips close at an edge vertex, five vertices (three on the edge and the two nearest "
            "on the next ring) are constrained to the hand. Attaching only three edge vertices behaves like a hinge and the sheet folds; the "
            "five-vertex patch fixes the sheet's slope at the tips, like tweezer jaws. Rigid objects are carried kinematically once gripped. "
            "The 90&nbsp;mm Petri dish is wider than the Franka Hand's 80&nbsp;mm stroke, so dishes and lids are pinched at the rim: the real "
            "cell needs a rim gripper or a tool change.")]

    # ---- 6 sequence
    s += H1("6  Control and the full per-sample sequence")
    s += [P("The arm is driven by differential inverse kinematics (mink) on a separate Panda-only model, used as a kinematic planner: "
            "smooth minimum-jerk targets for the tool point are converted into joint targets for the Panda's position servos at 500&nbsp;Hz. "
            "Two control details mattered:")]
    s += bullets([
        "<b>Gravity compensation.</b> Without it the servos sagged 3–7&nbsp;mm; with it the tip tracks to 0.2&nbsp;mm. A real Franka compensates gravity internally.",
        "<b>Choosing the grasp orientation.</b> A two-finger gripper is symmetric, so every grasp has two equivalent hand orientations. One of them can wind joint 7 into its limit or need wrist angles the Panda cannot reach. The controller dry-runs the IK through the key poses for both orientations and keeps the reachable one.",
    ])
    s += fig(FIG / "sequence.jpg", 1.0, "Figure 3. Six of the nine steps of the per-sample sequence.")
    label = dict(plate_prep="Plate from input stack to work spot, lid off", membrane_to_frit="Membrane from magazine onto frit (roll-on)",
                 funnel_on="Funnel from wash station onto frit, 30° bayonet twist", open_vessel="Open sample vessel (quarter-turn cap)",
                 dose="Dose sample into funnel (pour)", filtration="Vacuum filtration", funnel_off="Funnel to wash station",
                 transfer="Membrane from frit onto agar (roll-on)", lid_and_stack="Lid on, plate to output stack")
    seq_rows = [["#", "Step", "Time", "Arm"]]
    for i, st in enumerate(STEPS, 1):
        seq_rows.append([i, label.get(st["step"], st["step"]), f"{float(st['seconds']):.0f}&nbsp;s",
                         "free (model value)" if st["step"] == "filtration" else "busy"])
    seq_rows.append(["", "<b>Arm busy per sample</b>", f"<b>{ARM:.0f}&nbsp;s</b>", ""])
    s += [table(seq_rows, [0.05, 0.6, 0.12, 0.23], num_cols=(2,))]
    s += [Spacer(1, 6), P("Step times come from the simulated run with deliberately conservative speeds (0.25–0.35&nbsp;m/s in transit, "
                         "slower near objects), so they are an upper bound. Dosing and filtration are shown as animated liquid levels; pipetting, "
                         "vacuum and rinsing are not simulated. The 180&nbsp;s filtration is a model value. In both transfers within the sequence "
                         "the membrane landed flat (0.46&nbsp;mm off centre on the frit, 0.85&nbsp;mm on the agar).", "small")]

    # ---- 7 transfer
    s += H1("7  The membrane transfer")
    s += [P("The transfer is the step the lab flagged as hardest and the one most likely to decide feasibility. Technicians lift the wet "
            "membrane with tweezers, let it hang, touch the far edge onto the agar first and lower it so the contact line rolls across the "
            "sheet, which avoids trapped air. The scripted expert reproduces that motion:")]
    s += bullets([
        "Approach the membrane edge facing the robot and close the tips (edge patch attached).",
        "Peel the sheet off the frit slowly and lift it; it hangs from the tips.",
        "Over the plate, tilt the tips 45° so the sheet slopes down, and lower until the far edge touches the agar (closed-loop on the far-edge height).",
        "Sweep the gripped edge forward and down along the sheet's remaining length while the tilt goes to zero, then release and retract.",
    ])
    s += fig(FIG / "rollon.jpg", 0.82, "Figure 4. Roll-on placement, side view (inset: cell overview).")
    s += [P("7.1  Quality metrics", "h2"), P("Every run is scored automatically from the final vertex positions:")]
    s += [table([
        ["Metric", "Definition", "Pass"],
        ["On target", "membrane centroid within 3&nbsp;mm of plate centre", "yes"],
        ["Flatness", "standard deviation of vertex height above the agar", "&lt; 1.0&nbsp;mm"],
        ["Max lift", "highest vertex above the agar", "&lt; 2.0&nbsp;mm"],
        ["Fold", "an interior vertex 1.5&nbsp;mm above all its neighbours, or non-adjacent vertices within 2&nbsp;mm", "none"],
        ["Air pocket (proxy)", "connected patch of 3+ interior vertices more than 0.8&nbsp;mm above the agar", "none"],
        ["Overhang / tear", "vertex outside the agar disc / any edge stretched more than 5%", "none"],
    ], [0.2, 0.62, 0.18])]
    res_block = [P("7.2  Results", "h2")]
    def drow(name, d):
        dd = d["defects"]
        return [name, f"{d['succ']:.0f}%", f"{d['flat']:.0f}%", d["n"], f"{d['med']:.1f} / {d['p95']:.1f}",
                dd.get("misaligned", 0), dd.get("bubble", 0), dd.get("folded", 0), dd.get("not grasped", 0)]
    res_block += [table([["Controller and test set", "Success", "Laid flat", "Runs", "Offset mm med / p95", "Misal.", "Bubble", "Fold", "Not grasped"],
                 drow("Expert, nominal randomisation", NOM), drow("Expert, ranges 20% wider", WIDE),
                 drow("Expert, deliberately perturbed", PERT), drow("Learned policy (ACT, final), 20% wider, unseen", POL),
                 drow("Learned policy, previous iteration", POL2)],
                [0.29, 0.08, 0.08, 0.06, 0.15, 0.07, 0.07, 0.06, 0.14], num_cols=(1, 2, 3, 4, 5, 6, 7, 8))]
    res_block += [Spacer(1, 4), P("Nominal randomisation: membrane position ±3&nbsp;mm and any rotation on the frit, plate position ±5&nbsp;mm, grasp "
                         "direction ±0.15&nbsp;rad. Offsets are over runs where the membrane reached the plate. Mean grasp-to-release time "
                         f"for the expert: {NOM['t']:.1f}&nbsp;s.", "small")]
    s += [KeepTogether(res_block)]
    s += [P("7.3  What the perturbed runs reveal", "h2"),
          P("To create labelled defects, the expert was run with one parameter deliberately wrong per run. The outcome is informative "
            "about the simulator as much as about the strategy:")]
    def pm(mode):
        c = {k[1]: v for k, v in PERT_MODES.items() if k[0] == mode}
        n = sum(c.values())
        return f"{c.get('flat', 0)}/{n} flat" + "".join(f", {v} {k}" for k, v in c.items() if k != "flat")
    s += [table([
        ["Perturbation", "Outcome"],
        ["No tilt (0–15°): straight lowering", pm("flat")],
        ["Too much slack in the sweep", pm("slack")],
        ["Very fast sweep (0.4–1.2&nbsp;s)", pm("fast")],
        ["Early release (6–20&nbsp;mm above agar)", pm("early")],
        ["Steep tilt (70–85°) with slack", pm("steep")],
        ["Aim offset 4–25&nbsp;mm", pm("offset")],
    ], [0.45, 0.55])]
    s += [Spacer(1, 4), P("<b>Interpretation.</b> In simulation, lowering the membrane without any roll-on still lays it flat. The simulated sheet "
                         "has no air to trap and no surface tension, so the main reason technicians roll the membrane on cannot show up here. "
                         "The roll-on is kept because it is the technicians' method, not because the simulation proves it necessary. Whether "
                         "it matters, and how fast it can be done, is a question for the real membrane.", "callout")]

    # ---- 8 throughput
    s += H1("8  Throughput")
    s += [P("A discrete-event model (SimPy) runs 100 samples through N filtration positions served by one or two arms, using the step "
            "times measured in Section 6. The arm is busy during loading and unloading; during filtration it serves other positions. "
            "Funnel rinsing and disinfection happen at the wash station, off the arm.")]
    s += fig(RES / "cycle_time.png", 0.95, "Figure 5. Throughput cycle (minutes per sample) against parallel filtration positions.")
    s += [table([["Positions"] + [str(p) for p in CYC["positions"]],
                  ["1 arm, 3 min filtration"] + [f"{v:.1f}" for v in one_arm],
                  ["1 arm, 10 min filtration"] + [f"{v:.1f}" for v in slow],
                  ["2 arms, 3 min filtration"] + [f"{v:.1f}" for v in two_arm]],
                 [0.28] + [0.09] * 8, num_cols=tuple(range(1, 9)))]
    s += [Spacer(1, 6)] + bullets([
        f"With one arm the cell levels off at {min(one_arm):.1f}&nbsp;min per sample from two positions on: the arm, not filtration, is the bottleneck. 100 samples take about {min(one_arm)*100/60:.1f}&nbsp;hours.",
        "If filtrations slow toward 10 minutes (a clogging-prone matrix), four positions recover the same rate.",
        f"A second arm, or a faster arm, roughly halves the cycle ({min(two_arm):.1f}&nbsp;min with two arms).",
        "Largest arm-time items: lid on and plate out, plate in and lid off, and dosing. Plate handling is the obvious target for a dedicated device (for example a plate shuttle with lid lifter).",
    ])

    # ---- 9 learned policy
    s += H1("9  Learned vision policy (ACT)")
    s += [P("ACT (Action Chunking with Transformers, Zhao et al. 2023) is a widely used network for robot imitation learning. At each 10&nbsp;Hz "
            "tick it takes camera images and the robot state and predicts the next 20 actions (a 2-second chunk); overlapping chunks are "
            "averaged (temporal ensembling) for smooth motion. Inside, a ResNet18 encodes each image, a transformer combines image features "
            "with the robot state, and a transformer decoder outputs the action chunk; a CVAE encoder used only in training captures the "
            "variation between demonstrations. It learns purely by copying demonstrations, which is why it suits real robots: 50–100 "
            "teleoperated demonstrations are typically enough for a single task.")]
    s += [table([
        ["Item", "This study (final policy)"],
        ["Inputs", "top camera, wrist camera and a close plate camera, 128×128&nbsp;px each; 7 joint angles, gripper opening, current tip target"],
        ["Outputs", "tip position (relative to the current target), tip orientation (6D), gripper command; 20-step chunks at 10&nbsp;Hz"],
        ["Grasp rule", "closing the gripper within 5&nbsp;mm of a membrane edge vertex clamps the edge patch (same rule for expert and policy)"],
        ["Data", f"{N_DEMOS} expert demonstrations with injected noise (2&nbsp;mm), {DEMO_FRAMES:,} frames; randomised lighting, bench and agar colour, camera pose"],
        ["Model", "21.9&nbsp;M parameters; ResNet18 (ImageNet) backbone; 4 encoder, 4 decoder layers; latent 32"],
        ["Training", f"{LOG[-1]['step']:,} steps, batch 48, AdamW 1e-4, about 1&nbsp;h on the laptop GPU"],
        ["Evaluation", "100 unseen seeds with randomisation ranges 20% wider than training"],
    ], [0.18, 0.82])]
    s += [P("ACT was implemented directly in PyTorch for this study. The LeRobot library was not used because its current release pins an "
            "older PyTorch that would have replaced the CUDA build on this machine. The dataset is plain NumPy and easy to convert to a "
            "LeRobot dataset later.", "small")]
    s += [P("9.1  Three iterations", "h2"),
          P("<b>Iteration 1</b> predicted absolute tip positions in world coordinates. It failed: about half of the runs never grasped the "
            "membrane and the rest landed it 12–26&nbsp;mm off centre (evaluation stopped after 22 runs with no success). Absolute positions "
            "spanning about 0.6&nbsp;m left too little resolution for millimetre placement, and the gripper command arrived a tick late, "
            "after the tips had started to lift, so the grasp rule never fired.")]
    s += [P("<b>Iteration 2</b> predicts each chunk relative to the current tip target and receives that target as an input, and the "
            "grasp fires on any tick where the gripper is commanded closed at an edge. It imitated the expert closely when fed the expert's "
            f"own observations ({TF['iter2_relative']['median']:.1f}&nbsp;mm median error), yet acting on its own it landed the membrane a median "
            f"{POL2['med']:.1f}&nbsp;mm off centre and missed the grasp in {POL2['defects'].get('not grasped', 0)} of {POL2['n']} runs. The cause was "
            "compounding error: small deviations take the robot into states the demonstrations never showed, and a policy trained only on "
            "perfect demonstrations has never seen how to recover.")]
    s += [P("<b>Iteration 3</b> (final) addresses exactly that. While recording demonstrations, the expert's executed motion was pushed off "
            "course by a smooth random offset (2&nbsp;mm standard deviation per axis), but the recorded label was always the expert's clean, "
            "corrective target (a technique known as DART). The policy therefore sees off-course states paired with the way back. A third "
            "camera close to the plate (about 1.3&nbsp;mm per pixel, against about 6&nbsp;mm for the top camera) was added for placement, "
            f"and the dataset grew to {N_DEMOS} demonstrations.")]
    tf1, tf2, tf3 = TF["iter1_absolute"], TF["iter2_relative"], TF["iter3_dart"]
    s += [table([
        ["Measure", "Iteration 1", "Iteration 2", "Iteration 3 (final)"],
        ["Closed loop: success (centred within 3&nbsp;mm and flat)", "0 of 22", f"{POL2['succ']:.0f} of {POL2['n']}", f"<b>{POL['succ']:.0f} of {POL['n']}</b>"],
        ["Closed loop: laid flat anywhere on the agar", "n/a", f"{POL2['flat']:.0f}%", f"{POL['flat']:.0f}%"],
        ["Closed loop: membrane grasped and transferred", "about 50%", f"{POL2['moved_pct']:.0f}%", f"{POL['moved_pct']:.0f}%"],
        ["Closed loop: median placement offset", "about 20&nbsp;mm", f"{POL2['med']:.1f}&nbsp;mm", f"{POL['med']:.1f}&nbsp;mm"],
        ["Closed loop: runs within 5&nbsp;mm", "0", f"{POL2['w5']}", f"{POL['w5']}"],
        ["Teacher-forced tip error, median (held-out demos)", f"{tf1['median']:.1f}&nbsp;mm", f"{tf2['median']:.1f}&nbsp;mm", f"{tf3['median']:.1f}&nbsp;mm"],
    ], [0.43, 0.17, 0.17, 0.23], num_cols=(1, 2, 3))]
    s += [Spacer(1, 4), P("Teacher-forced error measures how closely the policy copies the labels on held-out demonstrations. It rises for "
                         "iteration 3 because its labels include deliberate corrections, which are harder to predict; closed-loop "
                         "performance, which is what matters, improves sharply. Copying precisely matters less than knowing how to recover.", "small")]
    s += fig(FIG / "offsets.png", 0.95, "Figure 6. Placement offset for the scripted expert and the two learned-policy iterations that transferred the membrane.")
    s += [P("9.2  What limits the final policy", "h2"),
          P(f"All failures of the final policy are placements just outside the 3&nbsp;mm tolerance: {POL['moved_pct']:.0f}% of runs transfer the "
            f"membrane, {POL['flat']:.0f}% lay it flat, and {POL['w5']} of {POL['n']} land within 5&nbsp;mm. The placement error has no systematic "
            f"bias (mean {POL['bias'][0]:+.1f}&nbsp;mm in x and {POL['bias'][1]:+.1f}&nbsp;mm in y) and a scatter of about {POL['scatter']:.1f}&nbsp;mm "
            "per axis. Re-weighting the temporal ensembling toward newer predictions made no measurable difference (20 against 19 successes on "
            "the same 30 seeds). The remaining levers are more demonstrations, higher image resolution and longer training, each cheap to try on "
            "a GPU cluster; on the real robot the policy is retrained on real images anyway.")]
    s += fig(FIG / "policy_grid.jpg", 0.9, "Figure 7. Four rollouts of the final policy on unseen randomisation: three successes and one placement 4.3&nbsp;mm off centre.")

    # ---- 10 perception
    s += H1("10  Perception")
    s += [P("10.1  Segmentation trained on synthetic images", "h2"),
          P(f"A U-Net with a ResNet18 encoder was trained on {PERC['seg_dataset']['total_frames']:,} rendered images (256×256) with automatic "
            "per-pixel labels, under randomised cameras, lighting, colours and membrane states (on frit, mid-transfer, on agar). "
            f"On held-out simulated scenes it reaches mIoU {SEG['val_miou']:.2f}.")]
    iou = SEG["val_iou"]
    s += [table([["Class"] + list(iou.keys()), ["IoU (sim)"] + [f"{v:.2f}" for v in iou.values()]],
                [0.16] + [0.12] * 7, num_cols=tuple(range(1, 8)))]
    s += [P("10.2  On the lab's own footage", "h2"),
          P("Both videos from the use-case page were downloaded and 52 frames extracted. As there are no real annotations, "
            f"{PTS['n']} points were placed by hand on three frames before the predictions were viewed. Only "
            f"{PTS['foreground_accuracy']*100:.0f}% of the object points were labelled correctly. Real funnels (conical, translucent) mostly read as "
            "background, agar plates are not found, the perforated bench reads as plate and lid, and hands are sometimes taken for funnels. "
            "The model is confidently wrong, so its confidence cannot be used to filter mistakes.")]
    s += fig(RES / "perception" / "real_overlay_funnels_plates.png", 0.85,
             "Figure 8. Client footage (left), sim-trained segmentation (middle), confidence (right). Crosses mark hand-checked points that were wrong.")
    s += [P("<b>Conclusion.</b> The real lab looks different from the simulated one: conical funnels on a manifold, coloured agar, a perforated "
            "bench, hands everywhere. Domain randomisation alone does not bridge that. A few hundred labelled real frames, which the Franka "
            "phase produces anyway, should be enough to fine-tune the same network.", "callout")]
    s += [P("10.3  Placement-defect classifier", "h2")]
    pc = DC["per_class"]; ar = DC["accept_reject"]
    s += [P(f"A ResNet18 classifier labels top-down crops of the placed membrane, using the metric labels as ground truth "
            f"({DC['n_images']} crops from the nominal, wide and perturbed runs; stratified 5-fold cross-validation). Used as an "
            f"accept/reject check it made {ar['false_accepts']} false accepts and {ar['false_rejects']} false rejects. Recall: "
            + ", ".join(f"{k} {v['recall']*100:.0f}% (n={v['support']})" for k, v in pc.items() if v["support"] >= 5)
            + ". Only one folded example exists, so folds are not yet evaluated.")]
    s += fig(RES / "perception" / "defect_confusion.png", 0.62, "Figure 9. Defect classifier confusion matrix (sim crops).")
    s += [P("10.4  Stalled-filtration detector", "h2")]
    sc = [(k, v) for k, v in STALL.items()]
    s += [P("The liquid level in the funnel is read from images (colour saturation in a funnel region found by the segmentation model). "
            "A stall is flagged when the level changes by less than 3% of the funnel height over 10 minutes while liquid remains. "
            "Time is compressed in simulation (one rendered frame stands for 10&nbsp;s).")]
    s += [table([["Simulated scenario", "Result"]] + [
        [k, (f"stall flagged at {v['stall_flag_min']:.1f}&nbsp;min" if v.get("stall_flag_min") else
             f"no flag; drained at {v['done_min']:.1f}&nbsp;min" if v.get("done_min") is not None else "no flag")] for k, v in sc],
        [0.55, 0.45])]
    s += [Spacer(1, 4), P("On the real filtration clip the level reading is plausible (frames registered 95% of the time), but the clip is about "
                         "a minute long with no complete drain, so stall detection on real footage is not validated.", "small")]

    # ---- 11 deliverables
    s += H1("11  Deliverables and how to reproduce")
    s += bullets([
        "<b>Demo video</b>: <font name='Mono'>results/membrane_demo.mp4</font>, 90&nbsp;s, 1280×720, captioned, following the planned storyboard.",
        "<b>Results page</b> (private until shared): https://claude.ai/artifact/YZW9a6ZthNNuVhyu97gDRf",
        "<b>This report</b>: <font name='Mono'>results/Membrane_Filtration_Sim_Report.pdf</font>.",
        "<b>Policy checkpoint</b>: <font name='Mono'>learn/act_ckpt.pt</font>; perception weights in <font name='Mono'>perception/</font>.",
        "<b>Raw results</b>: per-run CSV files in <font name='Mono'>results/</font>, perception metrics in <font name='Mono'>results/perception/</font>.",
    ])
    s += [P("Repository layout", "h3"), table([
        ["Folder", "Contents"],
        ["scene/", "cell.py (layout, modes, collision tiles, Panda with tips), membrane.py (flex generator), view.py"],
        ["control/", "ik.py (mink IK), primitives.py (motions, carrying, video recorder), sequence.py (full sample)"],
        ["transfer/", "expert.py (edge grasp, roll-on), metrics.py, randomize.py, eval_expert.py, relabel.py"],
        ["learn/", "env.py (10&nbsp;Hz policy interface), gen_dataset.py, act.py, train_act.py, eval_policy.py"],
        ["perception/", "render_labels.py, train_seg.py, run_real.py, defect_cls.py, stall_detector.py, real_frames/"],
        ["throughput/", "simpy_model.py, plot.py"],
        ["video/, report/", "assemble.py (90&nbsp;s cut); build_page.py, pdf_figures.py, build_pdf.py"],
    ], [0.18, 0.82])]
    s += [P("Main commands (from <font name='Mono'>membrane-sim/</font>, with <font name='Mono'>.venv</font> active)", "h3"),
          code("python control/sequence.py                      # full sample, video + step_durations.csv\n"
               "python transfer/eval_expert.py --n 100           # expert, nominal randomisation\n"
               "python learn/gen_dataset.py --n 300 --noise 2.0 --out learn/data_dart\n"
               "python learn/train_act.py --data learn/data_dart --steps 30000\n"
               "python learn/eval_policy.py --n 100 --scale 1.2  # evaluate the policy\n"
               "python throughput/plot.py                        # cycle-time chart\n"
               "python video/assemble.py 50007,50012,50022,50002 # 90 s video\n"
               "python report/build_page.py; python report/pdf_figures.py; python report/build_pdf.py")]

    # ---- 12 limitations
    s += H1("12  Limitations and risks")
    s += [table([
        ["Area", "Limitation", "Consequence / mitigation"],
        ["Membrane physics", "Thin elastic sheet; no wetting, adhesion, surface tension or air", "Roll-on benefit and real failure modes untested; first thing to measure on hardware"],
        ["Grasp", "Edge patch attached by constraint; tips do not physically pinch", "Real tweezer or gripper design is open; test candidate tips on real membranes"],
        ["Rigid handling", "Objects carried kinematically; plates pinched at the rim", "A rim gripper, plate shuttle or tool change is needed for 90&nbsp;mm dishes"],
        ["Process steps", "Dosing, vacuum, rinsing and disinfection animated or off the arm", "Dosing tool (pipette vs pour) and vessel opening (cap vs septum) still to decide"],
        ["Learned policy", f"{POL['succ']:.0f}% within the 3&nbsp;mm tolerance; random placement scatter of about 2&nbsp;mm", "More data and resolution (Section 13.1); the real-robot policy is retrained anyway"],
        ["Perception", "Does not transfer to real footage", "Fine-tune on labelled real frames from the Franka phase"],
        ["Throughput", "Step times from conservative sim motions; no failures or retries modelled", "Treat as an upper bound on arm time; add failure rates once measured"],
        ["Scale of evidence", f"{N_DEMOS} demonstrations, one training run per iteration, laptop compute", "More data, runs and ablations are cheap on a GPU cluster"],
    ], [0.16, 0.42, 0.42])]

    # ---- 13 how to proceed
    s += H1("13  How to proceed")
    s += [P("The simulation has done its job: the concept, the sequence and the throughput are credible, and the tooling is ready. The "
            "remaining risk sits in physics and perception, which only hardware can resolve. Work splits into three tracks: optional "
            "simulation improvements, the Franka test, and the feasibility study for the lab.")]
    s += [P("13.1  Near-term simulation work (optional, about one week)", "h2")]
    s += [table([
        ["Work item", "Expected effect", "Effort"],
        ["500–1,000 noise-injected demonstrations and longer training, run on the GPU cluster", "Reduces the remaining 2&nbsp;mm placement scatter", "1–2 days"],
        ["Higher-resolution plate and wrist views (224–256&nbsp;px)", "Finer placement", "1 day + retraining"],
        ["DAgger-style relabelling of the policy's own rollouts by the expert", "Further recovery data where the policy actually drifts", "2–3 days"],
        ["Diffusion Policy or a small VLA (e.g. SmolVLA) as comparison", "Shows a second policy family", "2–3 days"],
        ["Variant B (gantry) cell and second-arm variant in the throughput model", "Supports the architecture decision", "2 days"],
        ["Pipette dosing tool and septum vessels", "Removes the pour approximation; answers Challenge 1", "2–3 days"],
        ["Faster, collision-aware motions", "Tighter cycle-time estimate", "1–2 days"],
    ], [0.5, 0.36, 0.14])]
    s += [P("Noise-injected demonstrations and the plate camera are already done (Section 9.1). The remaining items are worth doing only "
            "if they change a decision.", "small")]

    s += [P("13.2  Franka phase (about one week once access and consumables exist)", "h2"),
          P("The simulated policy will not transfer to a real arm as is. What carries over is the pipeline: the same 10&nbsp;Hz policy "
            "interface, the same ACT code, the same metric definitions and the same perception stack. The plan:")]
    s += [table([
        ["Day", "Activity", "Output"],
        ["1", "Mount an overhead and a wrist camera in the sim layout; calibrate; set up the policy server and action-chunk streaming", "Working data and control loop"],
        ["1–2", "Characterise the real membrane: lift, hang and lay-down with candidate tweezer tips; film it", "First answer to the physics question"],
        ["2–3", "Collect 50–100 teleoperated transfer demonstrations on real membranes and agar", "Real dataset in the sim format"],
        ["3–4", "Label 200–300 real frames (membrane, funnel, plate, agar); fine-tune segmentation and the defect check", "Perception that works on the rig"],
        ["4", "Train ACT on real data, optionally co-trained with the sim demonstrations", "Real-robot policy"],
        ["5", "Run 50 rollouts scored with the same metrics, judged by the perception stack; record video", "Success rate and failure modes on hardware"],
    ], [0.08, 0.6, 0.32])]
    s += [P("Success criteria for the Franka test", "h3")] + bullets([
        "Membrane laid flat (no fold, no visible air pocket) in at least 80% of 50 rollouts, centred within 3&nbsp;mm in at least 70%.",
        "Transfer time under 30&nbsp;s from grasp to release.",
        "Perception agrees with a human judge on flat/defect for at least 90% of placements.",
    ])
    s += [P("Technical setup to agree with the integrator", "h3")] + bullets([
        "libfranka's real-time loop runs on the arm's PC; policy inference can run remotely within about 100&nbsp;ms latency if it sends 10–20-step action chunks.",
        "Two camera streams (overhead and wrist) over VPN; someone on site for resets and safety.",
        "Consumables at the arm: frit base and funnels, membranes, agar plates, tweezers or tip inserts.",
    ])

    s += [P("13.3  Feasibility study for the lab", "h2"),
          P("If the Franka test is positive, this work maps onto the integrator's paid feasibility study as follows:")]
    s += [table([
        ["Work package", "Content", "Builds on"],
        ["WP1 Process and requirements", "Sample matrix, volumes, filtration-time distribution, vessel type, traceability, cleaning rules", "Section 2; questions below"],
        ["WP2 Cell concept and throughput", "Variant A vs B layout, number of positions and arms, plate handling device, cycle-time model with measured times", "Sections 5, 6 and 8"],
        ["WP3 Membrane handling", "Tip or gripper design, transfer strategy, success criteria, failure handling", "Sections 7 and 13.2"],
        ["WP4 Perception and QC", "Membrane presence and placement check, QR traceability, stalled-filtration detection", "Section 10"],
        ["WP5 Disinfection concept", "Ethanol (explosion risk, extraction cost), H<sub>2</sub>O<sub>2</sub> (cost), UV (shadowing): select and size", "Integrator's open headache"],
        ["WP6 Risk register and budget", "Remaining technical risks, costs, schedule to a pilot cell", "All of the above"],
    ], [0.24, 0.52, 0.24])]

    s += [P("13.4  Open questions for the lab and the integrator", "h2")] + bullets([
        "Is the arm a Panda or an FR3; which gripper (Franka Hand, Robotiq, custom); is suction available?",
        "Which cameras are available and where can they be mounted?",
        "Can inference run on our side with streamed action chunks, or must everything run on the control PC?",
        "Sample vessels: screw caps, snap caps or septa? Dosing by pour, pipette or peristaltic pump?",
        "Typical and worst-case filtration times for the sugar solutions, and how often filtrations stall today.",
        "Membrane type and supplier (MCE, gridded?), agar plate type, and the current manual failure rate at transfer.",
        "Traceability format (QR on vessel and plate?), and the cleaning and disinfection rules between samples.",
    ])
    s += [KeepTogether([P("13.5  Decision points", "h2"), table([
        ["Gate", "Question", "Go if"],
        ["After the Franka test", "Can a real wet membrane be transferred reliably by an arm?", "Success criteria in 13.2 met, or a clear tip-design fix identified"],
        ["After WP2", "Variant A or B?", "Cycle time and cost favour one; plate handling solved"],
        ["After WP5", "Can the cell be cleaned to the lab's standard without ethanol extraction?", "A disinfection method fits cost and safety constraints"],
    ], [0.2, 0.42, 0.38])])]

    # ---- appendix
    s += H1("Appendix A  Corrections to the original plan")
    s += [P("Several statements in the original plan did not hold when tested, and were corrected in the code:")]
    s += [table([
        ["Plan said", "Found", "Fix"],
        ["Contact cap of 50 is per geom pair; keep 60 vertices", "Cap is per flex-body pair; even 61 vertices left 11 unsupported", "Contact surfaces as 9 tile bodies; finer meshes now possible"],
        ["Membrane mesh generator (disc_mesh)", "Half the triangles faced down and there were holes between rings", "Rewritten as an angle-ordered zipper triangulation"],
        ["Membrane parameters (young 2e5, default edge constraint)", "Sheet stretched 25–60% when lifted and crumpled like tissue", "Edge solref 0.004, solimp 0.99/0.999, young 2e7"],
        ["Grasp by attaching one or a few edge vertices", "Acts as a hinge; the sheet folds", "Clamp a five-vertex edge patch"],
        ["rings=4 runs in real time", "About real time without rendering on this laptop, slower with video", "Budget compute accordingly"],
        ["Use LeRobot for dataset and ACT", "Its pinned PyTorch would replace the CUDA build", "ACT implemented in PyTorch; NumPy dataset"],
        ["Absolute end-effector actions", "Insufficient precision; grasp timing missed", "Relative chunks, tip target as input, robust grasp trigger"],
        ["Clean scripted demonstrations", "Policy could not recover from its own small errors", "Noise-injected demonstrations (DART) and a plate camera"],
    ], [0.3, 0.37, 0.33])]
    s += H1("Appendix B  Software and hardware")
    s += [table([
        ["Component", "Version / detail"],
        ["MuJoCo", "3.14.0 (flex with elasticity, discrete integrator)"],
        ["Robot model", "MuJoCo Menagerie franka_emika_panda, with tweezer tips, TCP site and wrist camera added"],
        ["IK", "mink 1.3 with the daqp QP solver"],
        ["Learning", "PyTorch 2.14 (CUDA 12.6), torchvision 0.29"],
        ["Other", "SimPy, NumPy, OpenCV, imageio/ffmpeg, matplotlib, ReportLab"],
        ["Machine", "Windows 10, 16 CPU threads, 32&nbsp;GB RAM, RTX 4060 Laptop GPU (8&nbsp;GB)"],
    ], [0.22, 0.78])]
    return s


if __name__ == "__main__":
    doc = Doc(OUT)
    doc.multiBuild(build())
    print("wrote", OUT)

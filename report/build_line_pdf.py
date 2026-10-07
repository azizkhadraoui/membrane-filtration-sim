"""Line B concept note (PDF) for the client: architecture, measured simulation results, supply and disinfection.

    python report/build_line_pdf.py   -> results/Line_B_Concept_Note.pdf
Run after video/run_line.py, throughput/line_day.py and report/line_figs.py. Stills: results/line_stills/*.png
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pdf as bp
from build_pdf import P, bullets, table, fig, ST, W, H1, RES, FIG
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Spacer, PageBreak, NextPageTemplate, KeepTogether, Paragraph
from reportlab.pdfbase.pdfmetrics import stringWidth

OUT = RES / "Line_B_Concept_Note.pdf"
run = json.loads((RES / "line_full_run.json").read_text())
day = json.loads((RES / "line_day.json").read_text())
S = run["summary"]
stt = S["stats"]
rows = {r["positions"]: r for r in day["rows"]}
r6 = rows[6]
arm_line = stt["arm_dry"]["mean"] + stt["arm_wet"]["mean"] + stt["tip_wash"]["mean"]
aut = day["autonomy"]
n_ok = sum(1 for r in S["results"] if r.get("success"))
STILLS = RES / "line_stills"


def still(name):
    p = STILLS / name
    return p if p.exists() else None


class Doc(bp.Doc):
    def decor(self, c, d):
        c.saveState()
        c.setFont("UI", 7.6); c.setFillColor(bp.MUTED)
        c.drawString(2.1 * cm, A4[1] - 1.3 * cm, "Membrane filtration automation · Line B concept note")
        c.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, f"{d.page}")
        c.setStrokeColor(bp.RULE); c.setLineWidth(0.5)
        c.line(2.1 * cm, A4[1] - 1.45 * cm, A4[0] - 2.1 * cm, A4[1] - 1.45 * cm)
        c.restoreState()

    def afterFlowable(self, f):
        pass


def build():
    s = []
    ct = ParagraphStyle("ct", fontName="UI-Semi", fontSize=26, leading=32, textColor=bp.colors.white)
    cs = ParagraphStyle("cs", fontName="UI", fontSize=12.5, leading=18, textColor=bp.colors.HexColor("#c9d1da"))
    ce = ParagraphStyle("ce", fontName="UI-Semi", fontSize=9, leading=12, textColor=bp.colors.HexColor("#8fb8ec"))
    s += [Spacer(1, 0.6 * cm), Paragraph("CONCEPT NOTE · SIMULATION RESULTS", ce), Spacer(1, 0.35 * cm),
          Paragraph("Line B: automating membrane filtration<br/>as a process, not a human imitation", ct), Spacer(1, 0.4 * cm),
          Paragraph("Fixed stations for everything routine, the arm only for the wet membrane, with built-in "
                    "clean-in-place and one-shift material buffers", cs), Spacer(1, 2.1 * cm)]
    if still("full_line_a_03.png"):
        s += fig(still("full_line_a_03.png"), 1.0, None)
    s += [Spacer(1, 0.5 * cm), table([
        ["Question", "Answer from the simulation"],
        ["Throughput", f"{r6['min_per_sample']:.1f} min per sample ({r6['steady_per_hour']:.0f} samples/h) at 6 positions, against {60 / 12:.0f} min targeted and 3.2 min for the arm-only cell"],
        ["Arm workload", f"{arm_line:.0f} s of arm time per sample (arm-only cell: 191 s); the rest runs on fixed stations in parallel"],
        ["Disinfection", "clean-in-place on every filtration position, tip wash after every sample, disposable dosing tips; no ethanol or UV"],
        ["Supply", f"one operator visit per day at 100 samples/day (membranes {aut['100 samples/day']['membranes']:.0f} h, plates {aut['100 samples/day']['plates']:.0f} h, tips {aut['100 samples/day']['tips']:.0f} h)"],
        ["Status", "simulation only; membrane physics, tip design and sanitiser validation need hardware"],
    ], [0.18, 0.82], header=True, bold_first=True)]
    s += [NextPageTemplate("body"), PageBreak()]

    # ---------------------------------------------------------------- 1
    s += [Paragraph("1  What changed and why", ST["h1"])]
    s += [P("The first simulation automated the process the way a technician performs it: one arm, every step in sequence. It worked "
            "(the membrane transfer succeeds in 98% of randomised runs, and the full per-sample sequence runs end to end) but it "
            "inherits the technician's productivity: the arm is busy for 191 s per sample, so the cell is limited to about 3.2 minutes per "
            "sample however many filtration positions are added. Two further points were raised: the cell must be supplied continuously, "
            "and the tools must be disinfected continuously.")]
    s += [P("The design principle behind Line B is to keep the arm for the one step that needs dexterity, lifting and laying down the "
            "wet membrane, and to give every other step to a simple fixed mechanism that can be fast, cleanable and cheap. The "
            "arm's cycle is then the only shared resource, and it is short.")]
    s += [table([
        ["Step", "Arm-only cell (v1)", "Line B"],
        ["Dosing", "arm opens the vessel and pours", "gantry doser, disposable tip, through a pierceable septum"],
        ["Funnel handling", "arm carries funnels between frit and wash", "6 funnels fixed on the manifold, they lift pneumatically"],
        ["Funnel cleaning", "wash station, ethanol or UV questions", "clean-in-place on the position: rinse, sanitant, rinse, air dry"],
        ["Plates", "arm moves plate and lid (69 s per sample)", "plate hotel, shuttle and lid lifter, overlapped with filtration"],
        ["Membrane", "arm, from a magazine, twice", "arm, from a magazine, twice (unchanged: needs hands)"],
        ["Tools", "not addressed", "tweezer tip wash and dry after every sample"],
    ], [0.17, 0.38, 0.45], bold_first=True)]

    # ---------------------------------------------------------------- 2
    s += H1("2  The cell")
    s += [P("Everything below was built and run in the physics simulation (MuJoCo, Franka Panda); fixed mechanisms are animated, the membrane "
            "and the arm are simulated with contact. The arm reaches every membrane station; the doser, shuttle and funnels are fixed.")]
    if still("full_line_b_03.png"):
        s += fig(still("full_line_b_03.png"), 0.95, "Figure 1. Top view during a run: positions P1 to P6 on the manifold, gantry doser above the vessel rack, plate hotel, shuttle and output stack on the left.")
    s += [table([
        ["Station", "Function", "How it is modelled"],
        ["Membrane magazine", "spring-fed stack; the arm picks the top membrane", "flex membrane, rigid discs for the stack"],
        ["Filtration manifold, 6 positions", "frit, lifting funnel, valve, status LED", "slide-driven funnels; liquid level animated"],
        ["Gantry doser", "Y rail, X slide, Z nozzle; one disposable tip per sample", "kinematic, 3 axes, tip rack and waste"],
        ["Plate hotel, shuttle, lid lifter", "stack of closed plates in; open plate at the transfer spot; closed plate out", "kinematic, lid and plate follow the carriage"],
        ["Tip wash", "sanitant bath and air knife for the tweezer tips", "the arm visits it after each wet transfer"],
        ["Tanks and hatch", "sterile water and sanitant reservoirs; operator hatch", "level indicators, counters on the status board"],
    ], [0.24, 0.42, 0.34], bold_first=True)]

    # ---------------------------------------------------------------- 3
    s += H1("3  Measured behaviour")
    s += [P(f"A run of 8 samples through 6 positions in the simulation ({S['total_s']:.0f} s simulated, filtration time-compressed to 70 to 110 s per "
            f"sample). {n_ok} of {len(S['results'])} wet transfers were laid flat and centred, with the arm running at twice the speed of the first study "
            "(re-checked on 48 randomised transfers: 96% success). Station times measured in the run:")]
    names = dict(arm_dry="Arm: dry membrane from magazine to frit", arm_wet="Arm: wet membrane from frit to agar", tip_wash="Arm: tip wash",
                 dose="Doser: tip, aspirate, dispense, eject", plate_in="Shuttle: plate in, lid off", plate_out="Shuttle: lid on, plate out",
                 cip="Position: clean-in-place")
    trows = [["Step", "Mean (s)", "Longest (s)", "Runs"]]
    for k in ("arm_dry", "arm_wet", "tip_wash", "dose", "plate_in", "plate_out", "cip"):
        if k in stt:
            trows.append([names[k], f"{stt[k]['mean']:.0f}", f"{stt[k]['max']:.0f}", stt[k]["n"]])
    s += [table(trows, [0.58, 0.14, 0.16, 0.12], num_cols=(1, 2, 3))]
    s += [Spacer(1, 6)]
    s += fig(FIG / "line_gantt.png", 1.0, "Figure 2. Timeline of the run. The arm (top lane) is the only shared resource; doser, shuttle and positions overlap with it.")
    s += [P(f"<b>Throughput.</b> With these step times, real filtration times (median 180 s, with a 4% chance of a 10-minute stall) and a 2% "
            f"transfer retry rate, a simulated 8-hour shift gives the following (mean of 5 runs):")]
    t = [["Positions", "Samples per hour", "Minutes per sample", "100 samples take", "Arm busy", "Doser busy"]]
    for n in (2, 3, 4, 6, 8, 12):
        r = rows[n]
        t.append([n, f"{r['steady_per_hour']:.0f}", f"{r['min_per_sample']:.2f}", f"{100 / r['steady_per_hour']:.1f} h", f"{r['arm_util'] * 100:.0f}%", f"{r['doser_util'] * 100:.0f}%"])
    s += [table(t, [0.14, 0.18, 0.2, 0.18, 0.15, 0.15], num_cols=(0, 1, 2, 3, 4, 5))]
    s += [Spacer(1, 6)]
    s += fig(FIG / "line_compare.png", 1.0, "Figure 3. Left: arm busy time per sample. Right: shift-model throughput against positions, with the arm-only cell and the client target.")
    r2, r8 = rows[2], rows[8]
    s += [P(f"With slow filtrations (10 minutes, no variation) six positions still deliver {day['slow_6']:.0f} samples per hour. Throughput "
            f"levels off at about {rows[12]['steady_per_hour']:.0f} per hour beyond 8 positions: arm ({r8['arm_util'] * 100:.0f}% busy) and doser "
            f"({r8['doser_util'] * 100:.0f}%) both need the manifold zone, so they take turns, and each sample still needs its own arm moves. "
            "Further gains would come from separate access paths for doser and arm, or faster arm cycles. "
            f"The lab's actual need, 100 samples per day, is about 4 samples per hour, so the cell would work in short bursts and sit idle "
            f"most of the day: even 2 positions give {r2['steady_per_hour']:.0f} per hour, above the 12 per hour set by the 5-minute target. "
            "More positions buy headroom for peaks and slow filtrations, not routine throughput.", "body")]

    # ---------------------------------------------------------------- 4
    s += H1("4  Continuous supply of materials")
    s += [P("Every consumable has a buffer sized for at least one day at the lab's rate, and the cell stops accepting new samples before any buffer "
            "runs dry. The operator works through a hatch, so the enclosure stays closed during normal operation.")]
    cap = day["cap"]
    rate = {"100 samples/day": 100 / 24.0, "300 samples/day": 300 / 24.0}
    s += [table([
        ["Consumable", "Buffer", "At 100 samples/day", "At 300 samples/day", "At full speed"],
        ["Membranes (sterile, in magazine)", f"{cap['membranes']}", f"{aut['100 samples/day']['membranes']:.0f} h", f"{aut['300 samples/day']['membranes']:.0f} h", f"{aut['full speed (6 positions)']['membranes']:.1f} h"],
        ["Agar plates (hotel)", f"{cap['plates']}", f"{aut['100 samples/day']['plates']:.0f} h", f"{aut['300 samples/day']['plates']:.0f} h", f"{aut['full speed (6 positions)']['plates']:.1f} h"],
        ["Dosing tips (rack)", f"{cap['tips']}", f"{aut['100 samples/day']['tips']:.0f} h", f"{aut['300 samples/day']['tips']:.0f} h", f"{aut['full speed (6 positions)']['tips']:.1f} h"],
        ["Sanitant and rinse water", "see Section 5", f"{aut['100 samples/day']['sanitant_samples']:.0f} h", f"{aut['300 samples/day']['sanitant_samples']:.0f} h", f"{aut['full speed (6 positions)']['sanitant_samples']:.1f} h"],
        ["Sample vessels (racks through the hatch)", "barcode-scanned racks", "continuous", "continuous", "continuous"],
    ], [0.34, 0.16, 0.17, 0.17, 0.16], num_cols=(1, 2, 3, 4), bold_first=True)]
    s += [Spacer(1, 6)] + bullets([
        "<b>In:</b> racks of septum-capped vessels, membrane magazines and plate stacks through the hatch; one vessel barcode is read at entry and linked to the plate barcode at exit.",
        "<b>Out:</b> closed plates stacked for the incubator (or into a hotel that docks to it); used tips to a sealed autoclave bag; waste water to drain.",
        "<b>Monitoring:</b> every buffer is counted; the scheduler refuses new samples below a threshold and raises an operator request, so the cell never starves mid-sample.",
    ])

    # ---------------------------------------------------------------- 5
    s += H1("5  Disinfection of the tools")
    s += [P("The three disinfection options discussed so far each have a drawback: ethanol needs explosion-proof extraction, hydrogen peroxide is "
            "costly, and UV leaves shadows. Line B reduces the problem to places where a liquid can reach everything, and avoids ethanol:")]
    s += [table([
        ["Surface", "Concept", "Duration in the simulation"],
        ["Funnel and frit (sample contact)", "clean-in-place on the position: sterile-water rinse, sanitant, rinse, air dry, funnel closed", f"{stt['cip']['mean']:.0f} s per position, parallel to the arm"],
        ["Tweezer tips (membrane contact)", "dip in a sanitant bath, then air knife, after every wet transfer", f"{stt['tip_wash']['mean']:.0f} s, inside the arm cycle"],
        ["Dosing", "one disposable tip per sample; the nozzle never touches the sample", "none"],
        ["Plate and lid", "closed in the hotel, opened only at the transfer spot", "none"],
    ], [0.27, 0.5, 0.23], bold_first=True)]
    s += [Spacer(1, 6), P("<b>Choice of sanitant.</b> The concept works with a non-flammable sanitant (hot water at 80 °C or above, peracetic acid, a quaternary "
                         "ammonium or hypochlorous solution), which removes the ATEX question. Which one is acceptable depends on membrane and "
                         "frit compatibility and on residue limits, to be settled with the lab. <b>Validation.</b> A process blank is filtered at fixed "
                         "intervals (sterile water through a cleaned position, then onto agar and incubated); a growing blank flags the position "
                         "and triggers an extra cycle.")]

    # ---------------------------------------------------------------- 6
    s += H1("6  Variants and decisions")
    s += [table([
        ["Variant", "Idea", "Strengths", "Risks"],
        ["B (this note)", "fixed stations, arm only for the membrane", "fast, fully standard Petri dishes and membranes, cleaning on the position", "wet membrane handling still needs the arm; more mechanisms to build"],
        ["C: cassette", "single-use funnel with membrane in a rigid base that snaps onto a media cassette; no wet membrane handling", "removes the hardest manipulation and most of the cleaning", "proprietary consumables and per-sample cost; lab must accept cassettes instead of its own agar"],
        ["A (v1)", "arm does everything", "proven in simulation, flexible", "3.2 min per sample, arm-limited"],
    ], [0.14, 0.29, 0.29, 0.28], bold_first=True)]
    s += [Spacer(1, 6), P("<b>Decisions for the lab.</b> (1) Will standard Petri dishes with its own agar remain, or are cassettes acceptable? (2) Can vessels "
                         "switch to septum caps? (3) Typical and worst-case filtration times and stall rates. (4) Cleaning standard between samples and "
                         "the sanitant. (5) Should plates go straight into an incubator?")]

    # ---------------------------------------------------------------- 7
    s += H1("7  What the simulation does not settle, and next steps")
    s += bullets([
        "Wet-membrane behaviour (adhesion, surface tension, air entrapment) is not modelled; it is measured first on the real Franka.",
        "Tweezer tip design, funnel sealing and vacuum behaviour of the manifold, dosing accuracy and carry-over: hardware questions.",
        "Fixed stations are animated, not designed; they are the basis for a cost estimate and a concept sketch, not a mechanical design.",
        "Filtration is time-compressed in the run; the shift model uses real times and the step durations measured in the run.",
    ])
    s += [P("Proposed next steps", "h3")] + bullets([
        "<b>Week 1:</b> real-membrane transfer tests on the Franka (Section 7 of the first report); price the doser, the lifting-funnel manifold and the plate shuttle.",
        "<b>Week 2:</b> choose sanitant and test clean-in-place on one funnel and frit with process blanks; decide Petri dish versus cassette.",
        "<b>Then:</b> detailed cell layout, risk register and budget for a pilot cell.",
    ])
    return s


if __name__ == "__main__":
    doc = Doc(OUT)
    doc.multiBuild(build())
    print("wrote", OUT)

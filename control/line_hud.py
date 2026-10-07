"""Status-board overlay for the line video (drawn on every recorded frame)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from control.line_ctrl import LED, STATE_LABEL

F = "C:/Windows/Fonts/"
_f = {}


def font(size, kind="segoeui.ttf"):
    key = (size, kind)
    if key not in _f:
        _f[key] = ImageFont.truetype(F + kind, size)
    return _f[key]


INK = (14, 18, 24, 215)
TXT = (240, 243, 246, 255)
MUT = (165, 172, 180, 255)
ACC = (85, 152, 231, 255)


def make_hud(title="Line B: the arm only handles the membrane", speed_label="8x speed"):
    """Returns f(frame, st) -> frame, where st is one entry of the HUD trace (Line.hud_state())."""

    def hud(frame, st):
        im = Image.fromarray(frame).convert("RGBA")
        W, H = im.size
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        t = st['t']

        # --- title strip
        tw = d.textlength(title, font=font(26, "seguisb.ttf"))
        d.rounded_rectangle([20, 18, 44 + tw, 62], 9, fill=INK)
        d.text((32, 22), title, font=font(26, "seguisb.ttf"), fill=TXT)
        clock = f"t = {int(t // 60):d}:{int(t % 60):02d}   {speed_label}"
        cw = d.textlength(clock, font=font(19))
        d.rounded_rectangle([20, 70, 44 + cw, 104], 8, fill=INK)
        d.text((32, 74), clock, font=font(19), fill=MUT)

        # --- positions panel (right)
        pw, ph = 330, 62 + 44 * len(st['P']) + 12
        x0, y0 = W - pw - 20, 18
        d.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], 10, fill=INK)
        d.text((x0 + 16, y0 + 10), "Filtration positions", font=font(19, "seguisb.ttf"), fill=TXT)
        for i, P in enumerate(st['P']):
            y = y0 + 46 + 44 * i
            pst = P["state"]
            col = {"idle": LED["idle"], "wait_load": LED["clamp"], "loaded": LED["clamp"], "wait_dose": LED["clamp"],
                   "dosing": LED["clamp"], "filtering": LED["filter"], "ready": LED["ready"], "wet": LED["ready"],
                   "cip": LED["cip"], "error": LED["error"]}[pst]
            col = tuple(int(255 * c) for c in col) + (255,)
            d.ellipse([x0 + 16, y + 6, x0 + 32, y + 22], fill=col)
            d.text((x0 + 42, y), f"P{i + 1}", font=font(18, "seguisb.ttf"), fill=TXT)
            lab = STATE_LABEL[pst]
            if P["sample"] is not None:
                lab += f"  ·  S{P['sample'] + 1}"
            d.text((x0 + 86, y), lab, font=font(18), fill=TXT if pst != "idle" else MUT)
            if pst == "filtering":
                bx0, bx1 = x0 + 86, x0 + pw - 18
                d.rounded_rectangle([bx0, y + 27, bx1, y + 33], 3, fill=(60, 66, 74, 255))
                d.rounded_rectangle([bx0, y + 27, bx0 + (bx1 - bx0) * P["filt"], y + 33], 3, fill=col)

        # --- bottom-left counters
        arm_pct = 100.0 * st["arm_busy"] / max(t, 1.0)
        rows = [("Samples out", f"{st['out']} / {st['n']}"),
                ("Arm busy", f"{arm_pct:.0f}%"),
                ("CIP cycles", f"{st['cip']}"),
                ("Membranes left", f"{st['mem_left']}"),
                ("Plates left", f"{st['n'] - st['plates_taken']}"),
                ("Tips left", f"{st['n'] - st['tips_used']}"),
                ("Sanitant / water", f"{st['sani'] * 100:.0f}% / {st['water'] * 100:.0f}%")]
        bw, bh = 330, 46 + 30 * len(rows)
        bx, by = 20, H - bh - 20
        d.rounded_rectangle([bx, by, bx + bw, by + bh], 10, fill=INK)
        d.text((bx + 16, by + 10), "Cell status", font=font(19, "seguisb.ttf"), fill=TXT)
        for r, (a, b) in enumerate(rows):
            y = by + 44 + 30 * r
            d.text((bx + 16, y), a, font=font(17), fill=MUT)
            d.text((bx + bw - 16 - d.textlength(b, font=font(17, "seguisb.ttf")), y), b, font=font(17, "seguisb.ttf"), fill=TXT)

        # --- what the arm is doing
        cap = st['arm_label']
        if cap:
            cw = d.textlength(cap, font=font(21, "seguisb.ttf"))
            d.rounded_rectangle([W - 40 - cw - 20, H - 66, W - 20, H - 20], 9, fill=INK)
            d.ellipse([W - 34 - cw - 20 + 2, H - 52, W - 34 - cw - 20 + 16, H - 38], fill=ACC)
            d.text((W - 34 - cw + 8, H - 61), cap, font=font(21, "seguisb.ttf"), fill=TXT)
        return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))

    return hud

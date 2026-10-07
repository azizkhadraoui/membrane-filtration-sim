"""Pull stills for the results page from rendered clips (uses the caption sidecars to find moments).

    python report/stills.py  -> results/stills/{cell_overview,rollon}.png
"""
import json
from pathlib import Path

import imageio.v2 as iio2

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "stills"


def frame_where(clip, pred, frac=0.5):
    """First..last frames whose caption meta satisfies pred; return the one at `frac` of that span."""
    meta = json.loads(Path(clip).with_suffix(".json").read_text())
    idx = [i for i, m in enumerate(meta) if pred(m)]
    if not idx:
        return None
    target = idx[int(frac * (len(idx) - 1))]
    r = iio2.get_reader(clip)
    for i, fr in enumerate(r):
        if i == target:
            r.close()
            return fr
    r.close()


def main():
    OUT.mkdir(exist_ok=True)
    seq, tr = RES / "sequence.mp4", RES / "transfer_clean.mp4"
    if seq.with_suffix(".json").exists():
        fr = frame_where(seq, lambda m: "Funnel on" in m.get("caption", "") or "Dose" in m.get("caption", ""), 0.6)
        if fr is not None:
            iio2.imwrite(OUT / "cell_overview.png", fr)
    if tr.with_suffix(".json").exists():
        fr = frame_where(tr, lambda m: "Roll-on" in m.get("caption", ""), 0.45)
        if fr is not None:
            iio2.imwrite(OUT / "rollon.png", fr)
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()

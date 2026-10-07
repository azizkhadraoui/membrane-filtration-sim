"""Collect the Phase D metrics files into results/perception/summary.json.

    python perception/make_summary.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

RES = ROOT / "results" / "perception"
REAL = ROOT / "perception" / "real_frames"


def jload(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def dataset_stats():
    by_kind = {}
    for f in sorted((ROOT / "perception" / "data" / "seg").glob("*.npz")):
        k = f.stem.split("_")[0]
        with np.load(f) as z:
            n = z["lbl"].shape[0]
        d = by_kind.setdefault(k, dict(scenes=0, frames=0))
        d["scenes"] += 1; d["frames"] += n
    return dict(resolution="256x256", total_frames=sum(v["frames"] for v in by_kind.values()), by_kind=by_kind)


def main():
    seg = jload("seg_metrics.json") or {}
    real = jload("real_metrics.json") or {}
    defect = jload("defect_metrics.json")
    stall = jload("stall_metrics.json")
    s = dict(
        real_footage=dict(
            page="https://atlas.birdwave.io/use-cases/automating-membrane-filtration-in-a-food-microbiology-lab",
            videos={
                "eHX_Ej245P4": dict(title="Filtration - Membrane Filtration in Microbiology", obtained=True,
                                    file="perception/real_frames/video_eHX_Ej245P4.mp4", resolution="360x640", duration_s=75.8,
                                    note="default yt-dlp client reported 'not available'; android client gave a 360p stream",
                                    frames_extracted=len(list((REAL / "filtration").glob("f_*.png")))),
                "5eLQqS8ND3U": dict(title="Disinfection between samples - Membrane Filtration in Microbiology", obtained=True,
                                    file="perception/real_frames/video_5eLQqS8ND3U.mp4", resolution="608x1080", duration_s=44.5,
                                    frames_extracted=len(list((REAL / "disinfection").glob("d_*.png")))),
            },
            page_images="5 gallery images saved (perception/real_frames/page/); all are diagrams, not photos"),
        seg_dataset=dataset_stats(),
        segmentation=dict(model="U-Net, ImageNet resnet18 encoder", weights="perception/seg_unet_r18.pt",
                          val_split="by scene file, stratified by scene kind",
                          train_frames=seg.get("train_frames"), val_frames=seg.get("val_frames"),
                          val_miou=seg.get("val_miou"), val_miou_foreground=seg.get("val_miou_foreground"),
                          val_iou=seg.get("val_iou"), val_miou_by_kind=seg.get("val_miou_by_kind"),
                          train_time_s=seg.get("train_time_s"), peak_gpu_gb=seg.get("peak_gpu_gb"),
                          train_time_note="wall time 82 min: epochs 1-6 and 11-14 took ~95-100 s each, epochs 7-10 ~1000 s each while the data memmap was paged under system RAM pressure; GPU compute ~23 min"),
        real_eval=dict(points={k: v for k, v in real.get("points", {}).items() if k != "points"},
                       inference_scale_sweep_short_side=jload("real_scale_sweep.json"),
                       note="deliverable overlays use short side 256 (set before evaluation); the sweep is a sensitivity check on the same 46 points, not a tuned result"),
        defect_classifier=defect and {k: v for k, v in defect.items() if k != "confusion"},
        stall_detector=stall,
    )
    (RES / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1)[:3000])


if __name__ == "__main__":
    main()

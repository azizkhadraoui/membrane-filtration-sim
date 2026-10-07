"""End-state defect classifier on the top-down plate crops written by transfer/eval_expert.py.

    python perception/defect_cls.py            # train on whatever expert_{nominal,wide,perturbed}.csv exist
    python perception/defect_cls.py --wait     # first wait for results/expert_perturbed.csv to appear

Re-runnable: it re-reads all CSVs and crops each time and overwrites its outputs. Without the
perturbed set the result is marked PRELIMINARY (almost no defect examples).

Data: results/episodes/{nominal,wide,perturbed}/<seed>.png, labels = `defect` column of
results/expert_{tag}.csv. `sim_error` rows are solver blow-ups, not a visual class: dropped.
Classes with fewer than MIN_CLASS examples are merged into "other_defect" (reported).

Model: ImageNet resnet18, fine-tuned at 224x224. Evaluation: stratified 5-fold cross-validation
(every image is scored exactly once by a model that never saw it), 1/sqrt(n) class-balanced sampling,
rotation/flip/colour augmentation (the crop is top-down, so any rotation is a valid view).
Also reports the binary "accept (flat) vs reject (any defect)" decision, which is what a cell
controller would act on.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import imageio.v3 as iio

RES = ROOT / "results" / "perception"
TAGS = ["nominal", "wide", "perturbed"]
MIN_CLASS = 8
EPOCHS = 25
WEIGHTS = ROOT / "perception" / "defect_resnet18.pt"


def wait_for_data(timeout_s=3600):
    p = ROOT / "results" / "expert_perturbed.csv"
    t0 = time.time()
    while not p.exists():
        if time.time() - t0 > timeout_s:
            raise SystemExit("results/expert_perturbed.csv did not appear")
        print("waiting for", p, flush=True)
        time.sleep(30)
    size = -1
    while p.stat().st_size != size:            # wait until the writer is done
        size = p.stat().st_size
        time.sleep(10)


def load():
    rows = []
    for tag in TAGS:
        f = ROOT / "results" / f"expert_{tag}.csv"
        if not f.exists():
            print("missing", f); continue
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            img = ROOT / "results" / "episodes" / tag / f"{int(r.seed):05d}.png"
            rows.append(dict(tag=tag, seed=int(r.seed), defect=str(r.defect), img=str(img), exists=img.exists(),
                             mode=str(r.get("perturb_mode", r.get("mode", "")))))
    return pd.DataFrame(rows)


def make_model(n):
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, n)
    return net


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def augment(x):
    B = x.shape[0]
    dev = x.device
    k = int(torch.randint(4, (1,)))
    x = torch.rot90(x, k, (2, 3))
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    ang = torch.empty(B, device=dev).uniform_(-np.pi / 4, np.pi / 4)
    s = torch.empty(B, device=dev).uniform_(0.92, 1.08)
    t = torch.empty(2, B, device=dev).uniform_(-0.04, 0.04)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = s * torch.cos(ang); th[:, 0, 1] = -s * torch.sin(ang); th[:, 0, 2] = t[0]
    th[:, 1, 0] = s * torch.sin(ang); th[:, 1, 1] = s * torch.cos(ang); th[:, 1, 2] = t[1]
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    x = x * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
    x = x * torch.empty(B, 3, 1, 1, device=dev).uniform_(0.93, 1.07)
    x = x + torch.randn_like(x) * 0.01
    return x.clamp(0, 1)


def train_one(X, y, n_cls, dev, epochs=EPOCHS, seed=0):
    torch.manual_seed(seed)
    net = make_model(n_cls).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    bs = 32
    steps = epochs * max(1, len(X) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.15)
    cnt = np.bincount(y, minlength=n_cls).astype(float)
    pw = (1 / np.sqrt(np.maximum(cnt, 1)))[y]; pw /= pw.sum()     # softened class balancing (1/sqrt n)
    Xt = torch.from_numpy(X)
    mean, std = MEAN.to(dev), STD.to(dev)
    for ep in range(epochs):
        net.train()
        for _ in range(max(1, len(X) // bs)):
            idx = np.random.choice(len(X), bs, p=pw)
            xb = Xt[idx].to(dev).permute(0, 3, 1, 2).float().div(255)
            xb = (augment(xb) - mean) / std
            yb = torch.from_numpy(y[idx]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net(xb).float(), yb, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return net


@torch.no_grad()
def predict(net, X, dev, tta=True):
    net.eval()
    mean, std = MEAN.to(dev), STD.to(dev)
    out = []
    for i in range(0, len(X), 64):
        xb = torch.from_numpy(X[i:i + 64]).to(dev).permute(0, 3, 1, 2).float().div(255)
        xb = (xb - mean) / std
        p = 0
        for k in (range(4) if tta else [0]):
            p = p + net(torch.rot90(xb, k, (2, 3))).softmax(1)
        out.append((p / (4 if tta else 1)).cpu().numpy())
    return np.concatenate(out)


def plot_confusion(cm, names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2.5, 1.0 * len(names) + 2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}" if cm[i].sum() else "", ha="center", va="center",
                    fontsize=8, color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels([f"{n} (n={cm[i].sum()})" for i, n in enumerate(names)])
    ax.set_xlabel("predicted"); ax.set_ylabel("true (metrics label)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    if "--wait" in sys.argv:
        wait_for_data()
    df = load()
    preliminary = not (df.tag == "perturbed").any()
    pre = "PRELIMINARY (no perturbed set yet) - " if preliminary else ""
    raw_counts = df.defect.value_counts().to_dict()
    n_missing_img = int((~df.exists).sum())
    df = df[df.exists & (df.defect != "sim_error")].reset_index(drop=True)
    counts = df.defect.value_counts()
    merged = [c for c, n in counts.items() if n < MIN_CLASS and c != "flat"]
    df["cls"] = df.defect.where(~df.defect.isin(merged), "other_defect")
    order = ["flat", "misaligned", "folded", "bubble", "lifted", "torn", "other_defect"]
    names = [c for c in order if c in set(df.cls)] + sorted(set(df.cls) - set(order))
    y = df.cls.map({n: i for i, n in enumerate(names)}).to_numpy()
    X = np.stack([iio.imread(p)[..., :3] for p in df.img])
    print("raw label counts", raw_counts, "| used", dict(df.cls.value_counts()), "| merged into other_defect:", merged,
          flush=True)
    dev = "cuda"
    # stratified k-fold (classes with < k members are spread round-robin)
    k = 5
    rng = np.random.default_rng(0)
    fold = np.zeros(len(y), int)
    for c in range(len(names)):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        fold[idx] = (np.arange(len(idx)) + rng.integers(k)) % k
    P = np.zeros((len(y), len(names)))
    t0 = time.time()
    for f in range(k):
        tr, te = fold != f, fold == f
        net = train_one(X[tr], y[tr], len(names), dev, seed=f)
        P[te] = predict(net, X[te], dev)
        print(f"fold {f}: acc {np.mean(P[te].argmax(1) == y[te]):.3f}  [{time.time() - t0:.0f}s]", flush=True)
        del net; torch.cuda.empty_cache()
    pred = P.argmax(1)
    cm = np.zeros((len(names), len(names)), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    per = {}
    for i, n in enumerate(names):
        tp = cm[i, i]; sup = cm[i].sum(); pp = cm[:, i].sum()
        per[n] = dict(support=int(sup), recall=float(tp / sup) if sup else None,
                      precision=float(tp / pp) if pp else None)
    acc = float(np.mean(pred == y))
    bal = float(np.mean([v["recall"] for v in per.values() if v["recall"] is not None]))
    # accept/reject view
    flat = names.index("flat")
    yt, yp = y != flat, pred != flat
    binary = dict(reject_recall=float((yt & yp).sum() / max(yt.sum(), 1)),
                  accept_recall=float((~yt & ~yp).sum() / max((~yt).sum(), 1)),
                  accuracy=float(np.mean(yt == yp)), n_defect=int(yt.sum()), n_flat=int((~yt).sum()),
                  false_accepts=int((yt & ~yp).sum()), false_rejects=int((~yt & yp).sum()))
    plot_confusion(cm, names, RES / "defect_confusion.png",
                   f"Defect classifier (resnet18), stratified 5-fold CV on sim crops\n"
                   f"accuracy {acc:.1%}, balanced accuracy {bal:.1%}, n={len(y)}")
    # final model on all data
    net = train_one(X, y, len(names), dev, seed=99)
    torch.save(dict(state_dict=net.state_dict(), classes=names), WEIGHTS)
    res = dict(preliminary=preliminary, sources=sorted(set(df.tag)), n_images=len(y), raw_label_counts=raw_counts, images_missing=n_missing_img,
               dropped=["sim_error (solver failure, not a visual class)"], merged_into_other_defect=merged,
               classes=names, cv="stratified 5-fold", accuracy=acc, balanced_accuracy=bal, per_class=per,
               confusion=cm.tolist(), accept_reject=binary, by_source={
                   t: float(np.mean(pred[df.tag.to_numpy() == t] == y[df.tag.to_numpy() == t]))
                   for t in TAGS if (df.tag == t).any()},
               train_time_s=time.time() - t0, weights=str(WEIGHTS.relative_to(ROOT)))
    (RES / "defect_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "confusion"}, indent=1))


if __name__ == "__main__":
    main()

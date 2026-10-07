"""Train a small U-Net (ImageNet resnet18 encoder) on the synthetic segmentation set.

    python perception/train_seg.py                 # train, write perception/seg_unet_r18.pt
    python perception/train_seg.py --epochs 2      # smoke test

Validation split is by *scene file* (whole scenes / expert runs held out), stratified by kind,
so near-duplicate views of one scene never sit on both sides.
Image-level augmentation runs on the GPU per batch: random 192 px affine crops (scale/rotate/flip),
label-aware membrane recolouring, hand/coat-coloured occluders, colour jitter
(brightness, contrast, saturation, hue), grey, gaussian blur, sensor noise.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from perception.render_labels import CLASSES, OUT as DATA

WEIGHTS = ROOT / "perception" / "seg_unet_r18.pt"
RES = ROOT / "results" / "perception"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------------- model
class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class UNetR18(nn.Module):
    def __init__(self, n_classes=len(CLASSES), pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)          # /2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # /4 64, /8 128, /16 256, /32 512
        self.u4 = Up(512, 256, 256)
        self.u3 = Up(256, 128, 128)
        self.u2 = Up(128, 64, 64)
        self.u1 = Up(64, 64, 48)
        self.head = nn.Sequential(nn.Conv2d(48 + 3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, n_classes, 1))

    def forward(self, x):
        s0 = self.stem(x)
        e1 = self.l1(self.pool(s0))
        e2 = self.l2(e1)
        e3 = self.l3(e2)
        e4 = self.l4(e3)
        y = self.u4(e4, e3)
        y = self.u3(y, e2)
        y = self.u2(y, e1)
        y = self.u1(y, s0)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat([y, x], 1))


def load_model(path=WEIGHTS, device="cuda"):
    net = UNetR18(pretrained=False)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(device).eval()


@torch.no_grad()
def predict(net, rgb, device="cuda", short=256):
    """rgb: HxWx3 uint8 of any size -> (HxW class map, HxWxC softmax) at the input resolution.
    Resized so the short side is `short`, both sides rounded to multiples of 32."""
    H, W = rgb.shape[:2]
    sc = short / min(H, W)
    h, w = int(round(H * sc / 32)) * 32, int(round(W * sc / 32)) * 32
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(device)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    x = (x - MEAN.to(device)) / STD.to(device)
    p = net(x).softmax(1)
    p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[0]
    return p.argmax(0).cpu().numpy().astype(np.uint8), p.permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------------------------- data
def load_data(val_frac=0.12, seed=0):
    files = sorted(DATA.glob("*.npz"))
    rng = np.random.default_rng(seed)
    by_kind = {}
    for f in files:
        by_kind.setdefault(f.stem.split("_")[0], []).append(f)
    tr, va = [], []
    for k, fs in by_kind.items():
        fs = list(fs); rng.shuffle(fs)
        nv = max(1, int(round(val_frac * len(fs))))
        va += fs[:nv]; tr += fs[nv:]

    def cat(fs, name):
        """Concatenate into disk-backed .npy memmaps (keeps RAM use low; the OS page cache does the rest)."""
        ns = []
        for f in fs:
            with np.load(f) as z:
                ns.append(z["lbl"].shape[0])
        N = sum(ns)
        ip, lp = DATA.parent / f"cache_{name}_img.npy", DATA.parent / f"cache_{name}_lbl.npy"
        img = np.lib.format.open_memmap(ip, "w+", np.uint8, (N, 256, 256, 3))
        lbl = np.lib.format.open_memmap(lp, "w+", np.uint8, (N, 256, 256))
        kinds, k = [], 0
        for f, n in zip(fs, ns):
            with np.load(f) as z:
                img[k:k + n] = z["img"]; lbl[k:k + n] = z["lbl"]
            kinds += [f.stem.split("_")[0]] * n
            k += n
        img.flush(); lbl.flush(); del img, lbl
        return np.load(ip, mmap_mode="r"), np.load(lp, mmap_mode="r"), kinds
    return cat(tr, "train"), cat(va, "val"), [f.name for f in tr], [f.name for f in va]


MEMBRANE_TINTS = torch.tensor([[0.35, 0.75, 0.75], [0.45, 0.8, 0.55], [0.4, 0.6, 0.9], [0.7, 0.7, 0.7],
                               [0.95, 0.95, 0.9], [0.25, 0.25, 0.28]])
OCCLUDER_TINTS = torch.tensor([[0.85, 0.65, 0.55], [0.6, 0.42, 0.32], [0.95, 0.95, 0.97], [0.35, 0.45, 0.85],
                               [0.55, 0.75, 0.9], [0.2, 0.2, 0.22]])


def recolour_membrane(x, y, p=0.35):
    """Label-aware: give the membrane a random tint (gridded / coloured membranes exist; the
    client's video shows a teal-looking membrane), keeping the rendered shading."""
    B = x.shape[0]
    dev = x.device
    sel = torch.rand(B, device=dev) < p
    if not sel.any():
        return x
    m = (y == 1)[:, None] & sel[:, None, None, None]
    lum = x.mean(1, keepdim=True)
    ref = (lum * m).sum((2, 3), keepdim=True) / m.sum((2, 3), keepdim=True).clamp(min=1)
    tint = MEMBRANE_TINTS.to(dev)[torch.randint(len(MEMBRANE_TINTS), (B,), device=dev)][:, :, None, None]
    tint = (tint + torch.empty(B, 3, 1, 1, device=dev).uniform_(-0.08, 0.08)).clamp(0, 1)
    new = tint * (lum / ref.clamp(min=1e-3)).clamp(0.3, 1.5)
    return torch.where(m, new.clamp(0, 1), x)


def occluders(x, y, p=0.35):
    """Random ellipses in hand / glove / lab-coat colours, labelled background (the real
    footage is full of hands); shaded with a soft gradient so they are not flat stickers."""
    B, _, H, W = x.shape
    dev = x.device
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
    for b in torch.nonzero(torch.rand(B, device=dev) < p).flatten().tolist():
        for _ in range(int(torch.randint(1, 3, (1,)))):
            cx, cy = (torch.rand(2, device=dev) * 2.4 - 1.2).tolist()
            a, c = (torch.rand(2, device=dev) * 0.5 + 0.12).tolist()
            th = float(torch.rand(1)) * np.pi
            u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
            v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
            m = (u / a) ** 2 + (v / c) ** 2 < 1
            col = OCCLUDER_TINTS.to(dev)[int(torch.randint(len(OCCLUDER_TINTS), (1,)))]
            shade = 0.75 + 0.35 * (1 - ((u / a) ** 2 + (v / c) ** 2)).clamp(0, 1)
            x[b] = torch.where(m, (col[:, None, None] * shade).clamp(0, 1), x[b])
            y[b][m] = 0
    return x, y


def augment(x, y, g, out=None):
    """x: B3HW float [0,1] on GPU, y: BHW long. Geometric + photometric augmentation.
    `out`: output crop size (training uses 192 random affine crops of the 256 renders)."""
    B, _, H, W = x.shape
    if out:
        H = W = out
    dev = x.device
    # random scale (zoom in/out) + translation + flip via affine grid
    s = torch.exp(torch.empty(B, device=dev).uniform_(np.log(0.45), np.log(1.1)))   # x 192/256 crop: ~0.6..1.5 zoom
    ang = torch.empty(B, device=dev).uniform_(-0.35, 0.35)
    tx, ty = torch.empty(2, B, device=dev).uniform_(-0.3, 0.3)
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = s * torch.cos(ang) * flip; th[:, 0, 1] = -s * torch.sin(ang); th[:, 0, 2] = tx
    th[:, 1, 0] = s * torch.sin(ang) * flip; th[:, 1, 1] = s * torch.cos(ang); th[:, 1, 2] = ty
    grid = F.affine_grid(th, (B, 3, H, W), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="reflection", align_corners=False)[:, 0].long()
    x = recolour_membrane(x, y)
    x, y = occluders(x, y)
    # colour jitter
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    sat = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.4, 1.6)
    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * sat
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * c + m
    x = x * b
    # hue-ish: random per-channel gain + small channel permutation sometimes
    x = x * torch.empty(B, 3, 1, 1, device=dev).uniform_(0.8, 1.2)
    perm = torch.rand(B, device=dev) < 0.15
    if perm.any():
        x[perm] = x[perm][:, torch.randperm(3, device=dev)]
    grey = torch.rand(B, device=dev) < 0.08
    x[grey] = x[grey].mean(1, keepdim=True).expand(-1, 3, -1, -1)
    # blur
    k = 7
    sig = torch.empty(B, device=dev).uniform_(0.1, 2.0) * (torch.rand(B, device=dev) < 0.6)
    r = torch.arange(k, device=dev) - k // 2
    ker = torch.exp(-(r[None] ** 2) / (2 * sig[:, None].clamp(min=0.1) ** 2)); ker = ker / ker.sum(1, keepdim=True)
    xb = x.reshape(1, B * 3, H, W)
    kh = ker.repeat_interleave(3, 0)[:, None, None, :]
    kv = ker.repeat_interleave(3, 0)[:, None, :, None]
    xb = F.conv2d(F.pad(xb, (k // 2, k // 2, 0, 0), mode="reflect"), kh, groups=B * 3)
    xb = F.conv2d(F.pad(xb, (0, 0, k // 2, k // 2), mode="reflect"), kv, groups=B * 3)
    x = xb.reshape(B, 3, H, W)
    # noise
    x = x + torch.randn_like(x) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0, 0.05)
    return x.clamp(0, 1), y


def confusion(pred, y, n=len(CLASSES)):
    return torch.bincount((y * n + pred).flatten(), minlength=n * n).reshape(n, n)


def iou_from_conf(cm):
    cm = cm.double()
    inter = cm.diag()
    union = cm.sum(0) + cm.sum(1) - inter
    return (inter / union.clamp(min=1)).cpu().numpy(), (union > 0).cpu().numpy()


# ---------------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    (xtr, ytr, ktr), (xva, yva, kva), ftr, fva = load_data()
    print(f"train {len(xtr)} frames ({len(ftr)} scenes)  val {len(xva)} frames ({len(fva)} scenes)", flush=True)
    freq = sum(np.bincount(ytr[i:i + 200].reshape(-1), minlength=len(CLASSES)) for i in range(0, len(ytr), 200)).astype(float)
    freq /= freq.sum()
    w = torch.tensor((1 / np.sqrt(freq + 1e-4)) / (1 / np.sqrt(freq + 1e-4)).mean(), dtype=torch.float32, device=dev)
    print("class pixel freq", dict(zip(CLASSES, freq.round(4))), "weights", w.cpu().numpy().round(2), flush=True)
    net = UNetR18().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(xtr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    mean, std = MEAN.to(dev), STD.to(dev)
    g = torch.Generator(device=dev)
    hist = []
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        tot = 0.0
        nb = len(xtr) // a.bs
        def gather(i):
            idx = np.sort(perm[i * a.bs:(i + 1) * a.bs].numpy())
            return torch.from_numpy(xtr[idx]).pin_memory(), torch.from_numpy(ytr[idx]).pin_memory()
        fut = pool.submit(gather, 0)
        for i in range(nb):
            xb, yb = fut.result()
            if i + 1 < nb:
                fut = pool.submit(gather, i + 1)
            x = xb.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div(255)
            y = yb.to(dev, non_blocking=True).long()
            x, y = augment(x, y, g, out=192)
            x = (x - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            out = out.float()
            ce = F.cross_entropy(out, y, weight=w)
            p = out.softmax(1)
            oh = F.one_hot(y, len(CLASSES)).permute(0, 3, 1, 2).float()
            dice = 1 - ((2 * (p * oh).sum((0, 2, 3)) + 1) / (p.sum((0, 2, 3)) + oh.sum((0, 2, 3)) + 1)).mean()
            loss = ce + dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item()
        miou, ious = evaluate(net, xva, yva)
        hist.append(dict(epoch=ep + 1, loss=tot / (len(xtr) // a.bs), val_miou=miou))
        print(f"ep {ep + 1:2d} loss {hist[-1]['loss']:.3f} val mIoU {miou:.3f} "
              + " ".join(f"{c[:4]}={v:.2f}" for c, v in zip(CLASSES, ious))
              + f"  [{time.time() - t0:.0f}s, {torch.cuda.max_memory_allocated() / 1e9:.2f} GB]", flush=True)
        torch.save(net.state_dict(), WEIGHTS)
    miou, ious = evaluate(net, xva, yva)
    per_kind = {}
    for k in sorted(set(kva)):
        sel = np.where(np.array([kk == k for kk in kva]))[0]
        per_kind[k] = evaluate(net, xva[sel], yva[sel])[0]
    res = dict(train_frames=len(xtr), val_frames=len(xva), train_scenes=len(ftr), val_scenes=len(fva),
               val_scene_files=fva, epochs=a.epochs, val_miou=miou, val_miou_foreground=float(np.mean(ious[1:])),
               val_iou=dict(zip(CLASSES, map(float, ious))), val_miou_by_kind=per_kind,
               class_pixel_freq_train=dict(zip(CLASSES, map(float, freq))), history=hist,
               train_time_s=time.time() - t0, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "seg_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("history", "val_scene_files")}, indent=1))


@torch.no_grad()
def evaluate(net, x, y, bs=48):
    net.eval()
    dev = "cuda"
    cm = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long, device=dev)
    mean, std = MEAN.to(dev), STD.to(dev)
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(dev).permute(0, 3, 1, 2).float().div(255)
        yb = torch.from_numpy(np.ascontiguousarray(y[i:i + bs])).to(dev).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net((xb - mean) / std).argmax(1)
        cm += confusion(p, yb)
    ious, present = iou_from_conf(cm)
    return float(ious[present].mean()), ious


if __name__ == "__main__":
    main()

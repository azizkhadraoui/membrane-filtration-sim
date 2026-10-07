"""Train ACT on the expert demos in learn/data.

    python learn/train_act.py --steps 40000
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from learn.act import ACT, act_loss

HERE = Path(__file__).resolve().parent


class Frames:
    """Episode images, concatenated logically; memory-mapped when stored as ep_*_img.npy."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.starts = np.cumsum([0] + [len(a) for a in arrays[:-1]])
        a0 = arrays[0]
        self.shape = (sum(len(a) for a in arrays),) + a0.shape[1:]
        self.nbytes = int(np.prod(self.shape))

    def __getitem__(self, g):
        e = np.searchsorted(self.starts, g, side="right") - 1
        return np.stack([self.arrays[ei][gi - self.starts[ei]] for ei, gi in zip(e, g)])


def load_data(data_dir, max_eps=None):
    files = sorted(Path(data_dir).glob("ep_*.npz"))[:max_eps]
    imgs, states, acts, refs, starts = [], [], [], [], []
    n = 0
    for f in files:
        z = np.load(f)
        side = f.with_name(f.stem + "_img.npy")
        imgs.append(np.load(side, mmap_mode="r") if side.exists() else z["images"])
        a = z["action"].astype(np.float32)
        states.append(z["state"]); acts.append(a)
        if "ref" in z:
            refs.append(z["ref"].astype(np.float32))           # executed target (DART demos)
        else:                                                  # clean demos: previous commanded target
            refs.append(np.vstack([a[:1, :3], a[:-1, :3]]))
        starts.append(n); n += len(a)
    lengths = np.array([len(a) for a in acts])
    return (Frames(imgs), np.concatenate(states).astype(np.float32), np.concatenate(acts),
            np.concatenate(refs), np.array(starts), lengths, [f.name for f in files])


def augment(x):
    """x: (B, C, 3, H, W) float. Random shift (pad 6, crop) + mild brightness/contrast jitter."""
    B, C, _, H, W = x.shape
    x = x.flatten(0, 1)
    p = 6
    x = F.pad(x, (p, p, p, p), mode="replicate")
    ox, oy = torch.randint(0, 2 * p + 1, (2, x.shape[0]))
    idx_y = (oy[:, None] + torch.arange(H)).to(x.device)
    idx_x = (ox[:, None] + torch.arange(W)).to(x.device)
    x = x[torch.arange(x.shape[0], device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
          idx_y[:, None, :, None], idx_x[:, None, None, :]]
    b = 1 + 0.15 * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 2
    c = 0.08 * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 2
    x = (x * b + c).clamp(0, 1)
    return x.view(B, C, 3, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_eps", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "act_ckpt.pt"))
    ap.add_argument("--data", default=str(HERE / "data"))
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0); np.random.seed(0)

    imgs, states, acts, ref, starts, lengths, names = load_data(a.data, a.max_eps)
    n_eps = len(starts)
    states = np.concatenate([states, ref], 1)          # policy also sees where its target is
    val_eps = np.arange(n_eps)[::20]                               # 5% held-out episodes for val loss
    train_eps = np.setdiff1d(np.arange(n_eps), val_eps)
    print(f"{n_eps} episodes, {len(acts)} frames, images {imgs.nbytes/1e9:.1f} GB", flush=True)
    # action stats over relative chunks (position relative to the current target)
    rng = np.random.default_rng(0)
    gs = rng.integers(0, len(acts), 20000)
    ep_of = np.searchsorted(starts, gs, side="right") - 1
    tt = gs - starts[ep_of]
    off = rng.integers(0, a.chunk, len(gs))
    j = starts[ep_of] + np.minimum(tt + off, lengths[ep_of] - 1)
    samp = acts[j].copy()
    samp[:, :3] -= ref[gs]
    stats = dict(s_mean=states.mean(0), s_std=states.std(0) + 1e-3,
                 a_mean=samp.mean(0), a_std=samp.std(0) + 1e-3)
    print("relative action std (pos, mm):", (stats["a_std"][:3] * 1e3).round(1), flush=True)
    s_mean, s_std, a_mean, a_std = (torch.tensor(stats[k], device=dev) for k in ("s_mean", "s_std", "a_mean", "a_std"))

    cfg = dict(model=dict(state_dim=states.shape[1], act_dim=acts.shape[1], n_cams=imgs.shape[1],
                          chunk=a.chunk, img=imgs.shape[2]), train=vars(a), episodes=names, rel=True)
    model = ACT(**cfg["model"]).to(dev)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": rest, "lr": a.lr}, {"params": bb, "lr": a.lr * 0.1}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s / a.steps))))
    k = a.chunk

    def batch(eps):
        e = np.random.choice(eps, a.batch)
        t = (np.random.rand(a.batch) * lengths[e]).astype(int)
        g = starts[e] + t
        idx = np.minimum(t[:, None] + np.arange(k), lengths[e][:, None] - 1) + starts[e][:, None]
        mask = (t[:, None] + np.arange(k)) < lengths[e][:, None]
        im = torch.from_numpy(imgs[g]).to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float() / 255.0
        st = (torch.from_numpy(states[g]).to(dev) - s_mean) / s_std
        chunk = acts[idx].copy()
        chunk[:, :, :3] -= ref[g][:, None]
        ac = (torch.from_numpy(chunk).to(dev) - a_mean) / a_std
        return im, st, ac, torch.from_numpy(mask).to(dev)

    log = []
    t0 = time.time()
    scaler_dtype = torch.bfloat16
    for step in range(1, a.steps + 1):
        model.train()
        im, st, ac, mask = batch(train_eps)
        im = augment(im)
        with torch.autocast("cuda", dtype=scaler_dtype):
            pred, mu, logvar = model(im, st, ac, mask)
        loss, l1, kl = act_loss(pred.float(), ac, mask.float(), mu.float(), logvar.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == 1:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=scaler_dtype):
                vl = []
                for _ in range(4):
                    im, st, ac, mask = batch(val_eps)
                    p, _, _ = model(im, st)
                    vl.append(((p.float() - ac).abs().mean(-1) * mask).sum().item() / mask.sum().item())
            rec = dict(step=step, l1=l1.item(), kl=kl.item(), val_l1=float(np.mean(vl)), min=round((time.time() - t0) / 60, 1))
            log.append(rec)
            print(rec, flush=True)
        if step % 5000 == 0 or step == a.steps:
            torch.save(dict(model=model.state_dict(), cfg=cfg, stats=stats, log=log), a.out)
    Path(a.out).with_suffix(".log.json").write_text(json.dumps(log))
    print("saved", a.out)


if __name__ == "__main__":
    main()

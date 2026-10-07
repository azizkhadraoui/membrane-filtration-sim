"""Compact ACT (Action Chunking with Transformers, Zhao et al. 2023) in plain torch.

Same structure as the paper: ResNet18 image tokens + state token + CVAE latent token ->
transformer encoder; a transformer decoder with `chunk` learned queries predicts the next
`chunk` actions; the CVAE encoder (training only) embeds the ground-truth chunk into z.
Inference uses z = 0 and temporal ensembling over overlapping chunks.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sine_pos_2d(h, w, dim):
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))
    out = [torch.sin(x.flatten()[:, None] * omega), torch.cos(x.flatten()[:, None] * omega),
           torch.sin(y.flatten()[:, None] * omega), torch.cos(y.flatten()[:, None] * omega)]
    return torch.cat(out, 1)                                   # (h*w, dim)


class ACT(nn.Module):
    def __init__(self, state_dim=8, act_dim=10, n_cams=2, chunk=20, d=256, latent=32,
                 enc_layers=4, dec_layers=4, vae_layers=4, heads=8, ff=1024, dropout=0.1, img=128):
        super().__init__()
        self.chunk, self.latent, self.act_dim = chunk, latent, act_dim
        rn = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(rn.children())[:-2])     # (B,512,H/32,W/32)
        self.proj = nn.Conv2d(512, d, 1)
        fh = img // 32
        self.register_buffer("img_pos", sine_pos_2d(fh, fh, d))
        self.cam_embed = nn.Parameter(torch.zeros(n_cams, 1, d))
        self.state_proj = nn.Linear(state_dim, d)
        self.latent_proj = nn.Linear(latent, d)
        self.extra_pos = nn.Parameter(torch.zeros(2, d))              # latent, state tokens
        layer = lambda: nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer(), enc_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True), dec_layers)
        self.queries = nn.Parameter(torch.randn(chunk, d) * 0.02)
        self.head = nn.Linear(d, act_dim)
        # CVAE posterior encoder
        self.vae = nn.TransformerEncoder(layer(), vae_layers)
        self.vae_cls = nn.Parameter(torch.zeros(1, 1, d))
        self.vae_act = nn.Linear(act_dim, d)
        self.vae_state = nn.Linear(state_dim, d)
        self.vae_pos = nn.Parameter(torch.randn(chunk + 2, d) * 0.02)
        self.vae_out = nn.Linear(d, 2 * latent)

    def image_tokens(self, images):
        """images: (B, n_cams, 3, H, W) float in [0,1]."""
        B, C = images.shape[:2]
        x = images.flatten(0, 1)
        x = (x - IMAGENET_MEAN.to(x)) / IMAGENET_STD.to(x)
        f = self.proj(self.backbone(x))                              # (B*C, d, h, w)
        f = f.flatten(2).transpose(1, 2) + self.img_pos              # (B*C, h*w, d)
        f = f.view(B, C, -1, f.shape[-1]) + self.cam_embed[None]
        return f.flatten(1, 2)                                       # (B, C*h*w, d)

    def forward(self, images, state, actions=None, mask=None):
        B = state.shape[0]
        if actions is not None:
            tok = torch.cat([self.vae_cls.expand(B, -1, -1), self.vae_state(state)[:, None],
                             self.vae_act(actions)], 1) + self.vae_pos
            pad = None if mask is None else torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=state.device), ~mask], 1)
            h = self.vae(tok, src_key_padding_mask=pad)[:, 0]
            mu, logvar = self.vae_out(h).chunk(2, -1)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            z = torch.zeros(B, self.latent, device=state.device)
        extra = torch.stack([self.latent_proj(z), self.state_proj(state)], 1) + self.extra_pos
        mem = self.encoder(torch.cat([extra, self.image_tokens(images)], 1))
        out = self.decoder(self.queries.expand(B, -1, -1), mem)
        return self.head(out), mu, logvar


def act_loss(pred, target, mask, mu, logvar, kl_weight=10.0):
    l1 = (F.l1_loss(pred, target, reduction="none").mean(-1) * mask).sum() / mask.sum()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
    return l1 + kl_weight * kl, l1, kl


class Policy:
    """Wraps a trained ACT checkpoint with normalisation and temporal ensembling."""

    def __init__(self, ckpt_path, device="cpu", ensemble_m=0.01):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.cfg = ck["cfg"]
        self.model = ACT(**self.cfg["model"]).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.stats = {k: torch.as_tensor(v, device=device) for k, v in ck["stats"].items()}
        self.device, self.m = device, ensemble_m
        self.reset()

    def reset(self):
        self.t = 0
        self.buf = []                                                # (start_t, chunk actions)

    @torch.no_grad()
    def __call__(self, images, state, ref=None):
        """images: (n_cams, H, W, 3) uint8; state: (state_dim,); ref: current commanded TCP position
        (needed by relative-action checkpoints). Returns one absolute action (act_dim,)."""
        rel = self.cfg.get("rel", False)
        if rel:
            state = np.concatenate([state, ref]).astype(np.float32)
        images = images[:self.cfg["model"].get("n_cams", len(images))]   # older checkpoints use fewer cameras
        img = torch.as_tensor(images, device=self.device).permute(0, 3, 1, 2).float()[None] / 255.0
        st = (torch.as_tensor(state, device=self.device).float() - self.stats["s_mean"]) / self.stats["s_std"]
        a, _, _ = self.model(img, st[None])
        a = (a[0] * self.stats["a_std"] + self.stats["a_mean"]).cpu().numpy()
        if rel:
            a[:, :3] += np.asarray(ref)
        self.buf.append((self.t, a))
        self.buf = [(s, c) for s, c in self.buf if self.t - s < len(c)]
        preds = np.stack([c[self.t - s] for s, c in self.buf])        # oldest first
        w = np.exp(-self.m * np.arange(len(preds)))
        self.t += 1
        return (preds * w[:, None]).sum(0) / w.sum()

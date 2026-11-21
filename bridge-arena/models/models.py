"""三个造桥生成模型：自回归 Transformer、扩散模型、卷积 VAE。

设计向量：GridLevel 的 121 根候选杆件，每根取 0..3（空/木/钢/索）。
Transformer 按 4 类 token 自回归；Diffusion 在 121×4 one-hot 上扩散；
CNN-VAE 把设计排成 4 种杆型 × 4 种材料的 16 通道网格。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CAND = 121
N_CLASS = 4
N_GRID = (4, 4, 11)          # (kind, row, col)


# ============================================================ Transformer
class CausalBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class BridgeTransformer(nn.Module):
    def __init__(self, dim=128, layers=4, heads=4, dropout=0.1,
                 n_cand=N_CAND, n_class=N_CLASS):
        super().__init__()
        self.n_cand = n_cand
        self.tok = nn.Embedding(n_class + 1, dim)   # 0..3 + BOS
        self.pos = nn.Parameter(torch.zeros(1, n_cand, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([CausalBlock(dim, heads, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_class)
        mask = torch.triu(torch.ones(n_cand, n_cand, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, tokens):
        L = tokens.shape[1]
        x = self.tok(tokens) + self.pos[:, :L]
        for blk in self.blocks:
            x = blk(x, self.mask[:L, :L])
        return self.head(self.ln(x))      # [B, L, 4]

    @torch.no_grad()
    def sample(self, n, device, temperature=0.9, generator=None):
        self.eval()
        tokens = torch.full((n, 1), N_CLASS, dtype=torch.long, device=device)
        for _ in range(self.n_cand):
            logits = self(tokens)[:, -1] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens[:, 1:]


# ============================================================ Diffusion
class SinusoidalTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EpsMLP(nn.Module):
    def __init__(self, dim=512, time_dim=128):
        super().__init__()
        D = N_CAND * N_CLASS
        self.time = nn.Sequential(SinusoidalTime(time_dim), nn.Linear(time_dim, time_dim),
                                  nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.inp = nn.Linear(D + time_dim, dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            for _ in range(4)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(),
                                 nn.Linear(dim, D))

    def forward(self, x, t):
        h = self.inp(torch.cat([x, self.time(t)], dim=-1))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


class GaussianDiffusion:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T, device=device)
        self.alphas = 1.0 - self.betas
        self.ab = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        ab = self.ab[t][:, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n, device, steps=100, generator=None, temperature=1.0):
        model.eval()
        x = torch.randn(n, N_CAND * N_CLASS, device=device, generator=generator) * temperature
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        for i, t in enumerate(ts):
            tt = torch.full((n,), int(t), device=device, dtype=torch.long)
            eps = model(x, tt)
            ab_t = self.ab[t]
            ab_prev = self.ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1.5, 1.5)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        logits = x.view(n, N_CAND, N_CLASS)
        return logits.argmax(-1)


# ============================================================ CNN VAE
class ConvVAE(nn.Module):
    def __init__(self, latent=32, ch=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(16, ch, 3, stride=2, padding=1), nn.SiLU(),      # 4x11 -> 2x6
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.SiLU(),  # -> 1x3
            nn.Flatten())
        feat = ch * 2 * 1 * 3
        self.fc_mu = nn.Linear(feat, latent)
        self.fc_lv = nn.Linear(feat, latent)
        self.fc = nn.Linear(latent, feat)
        self.ch, self.latent = ch, latent
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1), nn.SiLU(),   # 1x3 -> 2x6
            nn.ConvTranspose2d(ch, 16, 4, stride=2, padding=1),                 # -> 4x12
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        h = self.fc(z).view(-1, self.ch * 2, 1, 3)
        return self.dec(h)[:, :, :4, :11]

    def forward(self, x):
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, lv

    @torch.no_grad()
    def sample(self, n, device, generator=None):
        z = torch.randn(n, self.latent, device=device, generator=generator)
        logits = self.decode(z).view(n, 4, 4, 4, 11)   # kind, class, row, col
        return logits.argmax(2)                        # [n, kind, row, col]

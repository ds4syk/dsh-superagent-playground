"""训练三个造桥模型（关卡 1）。

python scripts/train.py --data results/dataset_l1.npz --out results/checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena_bridge.design import LEVEL_GRID  # noqa: E402
from models.models import (BridgeTransformer, ConvVAE, EpsMLP, GaussianDiffusion,  # noqa: E402
                           N_CAND, N_CLASS)

G = LEVEL_GRID[1]


def onehot_design(classes: np.ndarray) -> np.ndarray:
    """[N,121] -> [N,484] one-hot（kind-major：每根杆件的 4 个类别）。"""
    out = np.zeros((len(classes), N_CAND * N_CLASS), dtype=np.float32)
    for k in range(N_CLASS):
        out[:, k::N_CLASS] = (classes == k)
    return out


def design_to_grid(classes: np.ndarray) -> np.ndarray:
    """[N,121] -> [N,16,4,11] one-hot（kind, class, row, col），供 CNN 编码器输入。"""
    out = np.zeros((len(classes), 16, 4, 11), dtype=np.float32)
    for i, c in enumerate(classes):
        g = G.to_grid(c)                       # [4,4,11] 每格类别 0..3
        for kind in range(4):
            for cls in range(4):
                out[i, kind * 4 + cls] = (g[kind] == cls)
    return out


def design_to_class_grid(classes: np.ndarray) -> np.ndarray:
    """[N,121] -> [N,4,4,11] 类别标签（0..3），供 CNN 交叉熵监督。"""
    return np.stack([G.to_grid(c) for c in classes]).astype(np.int64)


def set_seed(seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path, topk=6000, seed=0):
    d = np.load(path)
    designs = d["designs"].astype(np.int64)
    score = d["score"].astype(np.float32)
    passed = d["passed"].astype(bool)
    idx = np.where(passed)[0]
    idx = idx[np.argsort(-score[idx])][:topk]
    designs, score = designs[idx], score[idx]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(designs))
    nv = max(64, int(len(designs) * 0.05))
    va, tr = perm[:nv], perm[nv:]
    print(f"passed pool {passed.sum()}, selected {len(designs)}, train {len(tr)} val {len(va)}")
    return designs[tr], designs[va], score[tr]


# ---------------------------------------------------------------- transformer
def train_transformer(xtr, xva, dev, out, args):
    model = BridgeTransformer(dim=args.dim, layers=args.layers, heads=args.heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    steps = max(1, len(xtr) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps)
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr)), batch_size=args.batch,
                    shuffle=True, drop_last=True)
    print(f"transformer params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    hist = []
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        for (c,) in dl:
            c = c.to(dev)
            bos = torch.full((c.shape[0], 1), N_CLASS, device=dev, dtype=torch.long)
            tok = torch.cat([bos, c[:, :-1]], dim=1)
            logits = model(tok)
            loss = F.cross_entropy(logits.reshape(-1, N_CLASS), c.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                v = torch.from_numpy(xva).to(dev)
                bos = torch.full((v.shape[0], 1), N_CLASS, device=dev, dtype=torch.long)
                tok = torch.cat([bos, v[:, :-1]], dim=1)
                vl = F.cross_entropy(model(tok).reshape(-1, N_CLASS), v.reshape(-1)).item()
            hist.append(dict(epoch=ep, train=tot / nb, val=vl))
            print(f"  ep {ep:4d} train {tot/nb:.4f} val {vl:.4f}", flush=True)
    torch.save(model.state_dict(), os.path.join(out, "transformer.pt"))
    return hist


# ---------------------------------------------------------------- diffusion
def train_diffusion(xtr, xva, dev, out, args):
    x0 = torch.from_numpy(onehot_design(xtr) * 2 - 1)
    model = EpsMLP(dim=args.mlp_dim).to(dev)
    diff = GaussianDiffusion(1000, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps = max(1, len(x0) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps)
    dl = DataLoader(TensorDataset(x0), batch_size=args.batch, shuffle=True, drop_last=True)
    print(f"diffusion params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    hist = []
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        for (x,) in dl:
            x = x.to(dev)
            t = torch.randint(0, 1000, (x.shape[0],), device=dev)
            eps = torch.randn_like(x)
            xt = diff.q_sample(x, t, eps)
            loss = F.mse_loss(model(xt, t), eps)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        if ep % 20 == 0 or ep == args.epochs - 1:
            hist.append(dict(epoch=ep, train=tot / nb))
            print(f"  ep {ep:4d} train {tot/nb:.4f}", flush=True)
    torch.save(model.state_dict(), os.path.join(out, "diffusion.pt"))
    return hist


# ---------------------------------------------------------------- cnn
def train_cnn(xtr, xva, dev, out, args):
    gtr = torch.from_numpy(design_to_grid(xtr))
    ctr = torch.from_numpy(design_to_class_grid(xtr))
    model = ConvVAE(latent=args.latent).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = max(1, len(gtr) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps)
    dl = DataLoader(TensorDataset(gtr, ctr), batch_size=args.batch, shuffle=True, drop_last=True)
    print(f"cnn-vae params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    hist = []
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        beta = min(0.05, 0.05 * ep / max(1, args.epochs // 4))
        for g, c in dl:
            g = g.to(dev); c = c.to(dev)
            logits, mu, lv = model(g)
            ce = F.cross_entropy(logits.view(-1, 4, 4, 4, 11), c)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = ce + beta * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += ce.item(); nb += 1
        if ep % 20 == 0 or ep == args.epochs - 1:
            hist.append(dict(epoch=ep, ce=tot / nb))
            print(f"  ep {ep:4d} ce {tot/nb:.4f}", flush=True)
    torch.save(model.state_dict(), os.path.join(out, "cnn.pt"))
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/dataset_l1.npz")
    ap.add_argument("--out", default="results/checkpoints")
    ap.add_argument("--model", default="all", choices=["all", "transformer", "diffusion", "cnn"])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--topk", type=int, default=6000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp_dim", type=int, default=512)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__}", flush=True)
    xtr, xva, _ = load_data(args.data, args.topk, args.seed)
    meta = {}
    t0 = time.time()
    if args.model in ("all", "transformer"):
        t = time.time(); meta["transformer"] = train_transformer(xtr, xva, dev, args.out, args)
        meta["transformer_time"] = time.time() - t
    if args.model in ("all", "diffusion"):
        t = time.time(); meta["diffusion"] = train_diffusion(xtr, xva, dev, args.out, args)
        meta["diffusion_time"] = time.time() - t
    if args.model in ("all", "cnn"):
        t = time.time(); meta["cnn"] = train_cnn(xtr, xva, dev, args.out, args)
        meta["cnn_time"] = time.time() - t
    meta["total_time"] = time.time() - t0
    meta["n_train"] = int(len(xtr))
    meta["config"] = vars(args)
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"done in {meta['total_time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

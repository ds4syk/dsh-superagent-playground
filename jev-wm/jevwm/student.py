"""A tiny transformer policy used as the distilled 'learner'.

The point is not to beat a frontier model: it is to show that a few thousand
JEV-generated action labels are enough to train a policy whose inference is
several orders of magnitude cheaper than querying the decision model.
"""

from __future__ import annotations

import math
import random
import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: Sequence[str], min_count: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in texts:
        for tok in tokenize(t):
            counts[tok] = counts.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if c >= min_count:
            vocab[tok] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 192) -> List[int]:
    ids = [vocab.get(t, 1) for t in tokenize(text)][:max_len]
    if not ids:
        ids = [1]
    return ids


class TinyPolicy(nn.Module):
    def __init__(self, vocab_size: int, n_actions: int = 9, d: int = 128,
                 layers: int = 4, heads: int = 4, max_len: int = 192):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                           dropout=0.1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_actions)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids == 0
        b, t = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(t, device=ids.device))[None]
        x = self.encoder(x, src_key_padding_mask=mask)
        lengths = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        x = (x * (~mask).unsqueeze(-1)).sum(dim=1) / lengths
        return self.head(self.norm(x))


def train_student(texts: Sequence[str], labels: Sequence[int], epochs: int = 60,
                  batch_size: int = 64, lr: float = 1e-3, val_frac: float = 0.2,
                  seed: int = 0, device: str = "cpu", verbose: bool = True,
                  n_actions: int = 9) -> Tuple[TinyPolicy, Dict, Dict]:
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    vocab = build_vocab([texts[i] for i in train_idx])
    model = TinyPolicy(len(vocab), n_actions=n_actions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def batch(indices):
        xs = [encode_text(texts[i], vocab) for i in indices]
        ys = [labels[i] for i in indices]
        m = max(len(x) for x in xs)
        ids = torch.zeros(len(xs), m, dtype=torch.long)
        for r, x in enumerate(xs):
            ids[r, :len(x)] = torch.tensor(x)
        return ids.to(device), torch.tensor(ys, dtype=torch.long, device=device)

    best_state, best_val = None, -1.0
    for ep in range(epochs):
        model.train()
        rng.shuffle(train_idx)
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(train_idx), batch_size):
            chunk = train_idx[i:i + batch_size]
            ids, ys = batch(chunk)
            logits = model(ids)
            loss = nn.functional.cross_entropy(logits, ys)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(chunk)
            correct += (logits.argmax(-1) == ys).sum().item()
            total += len(chunk)
        sched.step()
        model.eval()
        with torch.no_grad():
            ids, ys = batch(val_idx)
            val_acc = (model(ids).argmax(-1) == ys).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"epoch {ep:3d} train_acc {correct/total:.3f} val_acc {val_acc:.3f} loss {loss_sum/total:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    meta = {"vocab": vocab, "val_acc": best_val, "n_train": len(train_idx), "n_val": n_val}
    return model, meta, {"val_acc": best_val, "n_train": len(train_idx)}


def save_student(path: str, model: TinyPolicy, meta: Dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta,
                "n_actions": model.head.out_features,
                "d": model.embed.embedding_dim,
                "layers": len(model.encoder.layers)}, path)


def load_student(path: str, device: str = "cpu") -> Tuple[TinyPolicy, Dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = TinyPolicy(len(meta["vocab"]), n_actions=ckpt["n_actions"],
                       d=ckpt["d"], layers=ckpt["layers"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta

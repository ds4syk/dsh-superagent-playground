"""Frozen V-JEPA 2 ViT-L encoder: build, load, preprocess and encode clips."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

VJEPA2_REPO = os.environ.get("VJEPA2_REPO", "/home/syk/labs/JEPA/repos/vjepa2")
if VJEPA2_REPO not in sys.path:
    sys.path.insert(0, VJEPA2_REPO)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def _clean_key(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        out[k] = v
    return out


def build_encoder(
    ckpt_path: str | Path,
    num_frames: int = 16,
    img_size: int = 256,
    device: str = "cpu",
) -> torch.nn.Module:
    from src.models import vision_transformer as vit

    encoder = vit.vit_large(
        img_size=(img_size, img_size),
        patch_size=16,
        num_frames=num_frames,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "target_encoder" if "target_encoder" in ckpt else "encoder"
    sd = _clean_key(ckpt[key])
    msg = encoder.load_state_dict(sd, strict=False)
    info = dict(
        ckpt_key=key,
        num_loaded=len(sd),
        missing=list(msg.missing_keys),
        unexpected=list(msg.unexpected_keys),
        params=sum(p.numel() for p in encoder.parameters()),
        num_frames=num_frames,
    )
    print(f"[vjepa2] loaded {info['num_loaded']} tensors from '{key}' "
          f"({info['params']/1e6:.1f}M params), missing={len(info['missing'])}, "
          f"unexpected={len(info['unexpected'])}", flush=True)
    encoder.eval()
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, info


@torch.no_grad()
def encode_clips(
    encoder: torch.nn.Module,
    clips: np.ndarray,
    device: str = "cpu",
    batch_size: int = 8,
    tokens_per_tubelet: int = 256,
    pool: str = "mean",
) -> np.ndarray:
    """``clips`` uint8 ``[B,16,256,256,3]`` -> features ``[B,8,1024]``.

    Each output timestep is the spatial mean of the 16x16 token grid of one
    tubelet (2 input frames).
    """
    out = []
    n = len(clips)
    for start in range(0, n, batch_size):
        batch = clips[start : start + batch_size]
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
        x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)  # [B,C,T,H,W]
        x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        tokens = encoder(x)  # [B, T/2*16*16, 1024]
        B, N, D = tokens.shape
        T = N // tokens_per_tubelet
        grid = tokens.reshape(B, T, 16, 16, D)
        feat = grid.mean(dim=(2, 3))
        if pool == "meanmax":
            feat = torch.cat([feat, grid.amax(dim=(2, 3))], dim=-1)
        elif pool != "mean":
            raise ValueError(pool)
        out.append(feat.cpu().numpy())
    return np.concatenate(out, axis=0)

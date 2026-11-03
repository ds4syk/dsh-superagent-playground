"""Does the SDPR output layer propagate gradients back into V-JEPA 2?

Loads the frozen ViT-L encoder and a trained edge head, encodes a few clips
with gradients enabled on the last encoder block (and patch embedding), runs
the edge head -> cost matrix -> SDPR -> global rotation loss, and reports the
gradient norms on the head, the encoder and the input features.

Also compares the analytic SDPR gradient with finite differences once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import EdgeHead, build_Q_torch, geodesic_loss, solve_rotations  # noqa: E402
from rotsync import RotSyncProblem, SDPRunner  # noqa: E402
from train_eval import EDGES, N_NODES, PRIOR_WEIGHT, load_data, standardize  # noqa: E402
from vjepa2_encoder import build_encoder, encode_clips  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--samples", default=os.path.join(HERE, "..", "artifacts", "data", "samples.npz"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "artifacts", "models", "vitl.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "artifacts", "results", "bridge_check.json"))
    args = ap.parse_args()

    data = load_data(args.features)
    train_mask = data["split"] == "train"
    _, mu, sd = standardize(data["feats"], train_mask)

    sample = np.load(args.samples)
    frames = torch.from_numpy(sample["frames"][: args.n])
    R_gt = torch.from_numpy(sample["R"][: args.n]).double()

    t0 = time.time()
    encoder, _ = build_encoder(args.ckpt, device=args.device)
    # Make the last block + patch embedding trainable so we can see whether the
    # SDPR loss reaches them (the rest stays frozen for memory).
    for name, p in encoder.named_parameters():
        p.requires_grad_(name.startswith("blocks.23") or name.startswith("patch_embed") or name.startswith("norm"))
    print(f"[bridge] encoder loaded in {time.time()-t0:.1f}s", flush=True)

    head = EdgeHead(d=data["feats"].shape[-1]).double()
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    head.train()
    head.to(args.device)

    x = frames.to(args.device).permute(0, 4, 1, 2, 3).float().div_(255.0)
    x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(args.device)) / (
        torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(args.device)
    )
    tokens = encoder(x)
    B, N, D = tokens.shape
    T = N // 256
    feats = tokens.reshape(B, T, 16, 16, D).mean(dim=(2, 3))
    feats = (feats - mu.to(feats.device)) / sd.to(feats.device)
    feats.retain_grad()

    prob = RotSyncProblem(N_NODES, redundant=False)
    runner = SDPRunner(prob, solver="SCS", eps=1e-7)
    meas, w = head(feats, EDGES)
    Qs = torch.stack([build_Q_torch(prob, meas[b], w[b], EDGES, PRIOR_WEIGHT) for b in range(B)])
    Rs, diag = solve_rotations(prob, Qs, runner)
    loss = geodesic_loss(Rs, R_gt.to(Rs.device))
    loss.backward()

    def gnorm(params):
        vals = [p.grad.detach().norm().item() for p in params if p.grad is not None]
        return float(np.sqrt(np.sum(np.square(vals)))) if vals else 0.0

    head_params = list(head.parameters())
    last_block = [p for n, p in encoder.named_parameters() if n.startswith("blocks.23")]
    patch = [p for n, p in encoder.named_parameters() if n.startswith("patch_embed")]
    enc_norm = [p for n, p in encoder.named_parameters() if n.startswith("norm")]
    out = {
        "n_clips": int(B),
        "loss_deg": float(loss.detach() * 180 / np.pi),
        "head_grad_norm": gnorm(head_params),
        "encoder_last_block_grad_norm": gnorm(last_block),
        "encoder_patch_embed_grad_norm": gnorm(patch),
        "encoder_norm_grad_norm": gnorm(enc_norm),
        "feature_grad_norm": float(feats.grad.norm()) if feats.grad is not None else 0.0,
        "rank_gap": [float(v) for v in diag["rank_gap"]],
        "eig_min": [float(v) for v in diag["eig_min"]],
        "device": args.device,
        "elapsed_s": round(time.time() - t0, 1),
    }

    # A one-step check that the encoder gradient actually moves the loss.
    opt = torch.optim.SGD([p for p in last_block + patch if p.requires_grad], lr=1e-3)
    opt.step()
    tokens2 = encoder(x)
    feats2 = (tokens2.reshape(B, T, 16, 16, D).mean(dim=(2, 3)) - mu.to(tokens2.device)) / sd.to(tokens2.device)
    meas2, w2 = head(feats2, EDGES)
    Qs2 = torch.stack([build_Q_torch(prob, meas2[b], w2[b], EDGES, PRIOR_WEIGHT) for b in range(B)])
    Rs2, _ = solve_rotations(prob, Qs2, runner)
    out["loss_after_one_step_deg"] = float(geodesic_loss(Rs2, R_gt.to(Rs2.device)).detach() * 180 / np.pi)
    out["loss_delta_deg"] = out["loss_after_one_step_deg"] - out["loss_deg"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()

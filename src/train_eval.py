"""Train/evaluate the V-JEPA 2 -> SDPR output bridge and baselines.

Methods
-------
``direct``    per-node MLP head on V-JEPA 2 features, no structured layer.
``twostage``  edge head trained only on relative-rotation loss, then a
              *frozen* SDPR rotation-synchronization layer produces the
              global output (no gradient through the layer).
``e2e``       the same edge head fine-tuned end-to-end through SDPR with the
              global rotation loss (the "bridge").
``oracle``    ground-truth relative rotations through SDPR (upper bound).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import (  # noqa: E402
    EdgeHead,
    NodeHead,
    build_Q_torch,
    chordal_sync,
    geodesic_loss,
    project_so3_torch,
    solve_rotations,
)
from rotsync import RotSyncProblem, SDPRunner  # noqa: E402

N_NODES = 8
EDGES = [(i, i + 1) for i in range(N_NODES - 1)] + [(i, i + 2) for i in range(N_NODES - 2)]
PRIOR_WEIGHT = 10.0


def load_data(path: str):
    d = np.load(path, allow_pickle=True)
    return {
        "feats": torch.tensor(d["feats"], dtype=torch.float64),
        "R": torch.tensor(d["R"], dtype=torch.float64),
        "split": np.asarray(d["split"]),
        "pano": np.asarray(d["pano"]),
        "idx": np.asarray(d["idx"]),
    }


def standardize(feats: torch.Tensor, train_mask: np.ndarray):
    mu = feats[train_mask].mean(dim=0, keepdim=True)
    sd = feats[train_mask].std(dim=0, keepdim=True).clamp_min(1e-6)
    return (feats - mu) / sd, mu, sd


def relative_targets(R: torch.Tensor) -> torch.Tensor:
    # [B,N,3,3] -> [B,E,3,3]
    return torch.stack([R[:, i] @ R[:, j].transpose(-1, -2) for i, j in EDGES], dim=1)


def relative_error_deg(meas: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    Rg = relative_targets(R)
    tr = (meas * Rg).sum(dim=(-2, -1)).clamp(-3.0, 3.0)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)))


def evaluate(
    head,
    prob,
    runner,
    feats,
    R,
    mask,
    kind: str,
    batch_size: int = 16,
    collect: bool = False,
) -> dict:
    if head is not None:
        head.eval()
    idxs = np.where(mask)[0]
    errs, rels, cert = [], [], []
    preds, gts, all_rank, all_eig = [], [], [], []
    t_solve = []
    with torch.no_grad():
        for start in range(0, len(idxs), batch_size):
            bidx = idxs[start : start + batch_size]
            f = feats[bidx]
            Rg = R[bidx]
            if kind == "direct":
                Rp = head(f)
                rels.append(torch.zeros(len(bidx)))
            elif kind == "ls":
                meas, w = head(f, EDGES)
                Rp = chordal_sync(meas, w, EDGES, N_NODES)
                # gauge alignment (Wahba) to the ground truth (test-time only)
                H = torch.einsum("bnij,bnik->bjk", Rg, Rp)
                U, _, Vt = torch.linalg.svd(H)
                D = torch.ones((len(bidx), 3), dtype=Rp.dtype)
                D[:, 2] = torch.sign(torch.det(U @ Vt))
                G = (U * D.unsqueeze(-2)) @ Vt
                Rp = G.unsqueeze(1) @ Rp
                rels.append(relative_error_deg(meas, Rg).mean(dim=1))
            else:
                if kind == "oracle":
                    meas = relative_targets(Rg)
                    w = torch.ones(meas.shape[:2], dtype=torch.float64)
                else:
                    meas, w = head(f, EDGES)
                Qs = torch.stack(
                    [build_Q_torch(prob, meas[b], w[b], EDGES, PRIOR_WEIGHT) for b in range(len(bidx))]
                )
                t0 = time.time()
                Rp, diag = solve_rotations(prob, Qs, runner)
                t_solve.append((time.time() - t0) / len(bidx))
                all_rank.append(diag["rank_gap"])
                all_eig.append(diag["eig_min"])
                rels.append(relative_error_deg(meas, Rg).mean(dim=1))
            e = torch.rad2deg(
                torch.acos(
                    (
                        ((Rp * Rg).sum(dim=(-2, -1)) - 1.0) / 2.0
                    ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                )
            )
            errs.append(e)
            preds.append(Rp)
            gts.append(Rg)
    err = torch.cat(errs).numpy()
    rel = torch.cat(rels).numpy()
    out = {
        "n": int(err.size),
        "mean_deg": float(err.mean()),
        "median_deg": float(np.median(err)),
        "p90_deg": float(np.percentile(err, 90)),
        "max_deg": float(err.max()),
        "per_node_mean_deg": err.mean(axis=0).round(4).tolist(),
        "rel_mean_deg": float(rel.mean()),
    }
    if all_rank:
        rank = np.concatenate(all_rank)
        eig = np.concatenate(all_eig)
        out.update(
            cert_rate=float(np.mean((rank < 1e-6) & (eig > -1e-5))),
            cert_rate_strict=float(np.mean((rank < 1e-6) & (eig > -1e-7))),
            rank_gap_max=float(rank.max()),
            eig_min=float(eig.min()),
            sdp_fwd_s=float(np.mean(t_solve)),
        )
    if collect:
        out["_pred"] = torch.cat(preds).numpy()
        out["_gt"] = torch.cat(gts).numpy()
        out["_idx"] = idxs
        out["_rel"] = rel
        if all_rank:
            out["_rank"] = np.concatenate(all_rank)
            out["_eig"] = np.concatenate(all_eig)
    return out


def train_edge_head(feats, R, train_mask, val_mask, prob, runner, epochs, batch_size, lr, tag):
    head = EdgeHead(d=feats.shape[-1]).double()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_idx = np.where(train_mask)[0]
    hist = []
    for ep in range(1, epochs + 1):
        head.train()
        perm = np.random.permutation(tr_idx)
        losses = []
        for start in range(0, len(perm), batch_size):
            bidx = perm[start : start + batch_size]
            f = feats[bidx]
            Rg = R[bidx]
            meas, _ = head(f, EDGES)
            loss = geodesic_loss(meas, relative_targets(Rg))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        sched.step()
        with torch.no_grad():
            vf = feats[val_mask]
            vm, _ = head(vf, EDGES)
            val_rel = float(geodesic_loss(vm, relative_targets(R[val_mask])) * 180 / np.pi)
        rec = {
            "epoch": ep,
            "train_rel_loss_deg": float(np.mean(losses) * 180 / np.pi),
            "val_rel_deg": val_rel,
        }
        hist.append(rec)
        print(
            f"[twostage] ep{ep:02d} rel_loss={rec['train_rel_loss_deg']:.2f}deg "
            f"val_rel={val_rel:.2f}deg", flush=True
        )
    return head, hist


def finetune_e2e(head, feats, R, train_mask, val_mask, prob, runner, epochs, batch_size, lr, tag):
    head = EdgeHead(d=feats.shape[-1]).double()
    head.load_state_dict(torch.load(tag, map_location="cpu"))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_idx = np.where(train_mask)[0]
    hist = []
    best = None
    for ep in range(1, epochs + 1):
        head.train()
        perm = np.random.permutation(tr_idx)
        losses = []
        for start in range(0, len(perm), batch_size):
            bidx = perm[start : start + batch_size]
            f = feats[bidx]
            Rg = R[bidx]
            meas, w = head(f, EDGES)
            Qs = torch.stack(
                [build_Q_torch(prob, meas[b], w[b], EDGES, PRIOR_WEIGHT) for b in range(len(bidx))]
            )
            Rp, _ = solve_rotations(prob, Qs, runner)
            loss = geodesic_loss(Rp, Rg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            losses.append(float(loss) * 180 / np.pi)
        sched.step()
        ev = evaluate(head, prob, runner, feats, R, val_mask, "sdpr", batch_size=16)
        rec = {
            "epoch": ep,
            "train_global_loss_deg": float(np.mean(losses)),
            "val_mean_deg": ev["mean_deg"],
            "val_cert_rate": ev.get("cert_rate", None),
        }
        hist.append(rec)
        print(
            f"[e2e] ep{ep:02d} train={rec['train_global_loss_deg']:.2f}deg "
            f"val={ev['mean_deg']:.2f}deg cert={ev.get('cert_rate'):.2f}",
            flush=True,
        )
        if best is None or ev["mean_deg"] < best[0]:
            best = (ev["mean_deg"], {k: v.clone() for k, v in head.state_dict().items()})
    if best is not None:
        head.load_state_dict(best[1])
    return head, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--method", required=True, choices=["direct", "twostage", "e2e", "oracle"])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "artifacts", "results"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--init", default=None, help="two-stage checkpoint for e2e")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    data = load_data(args.features)
    feats_raw, R, split = data["feats"], data["R"], data["split"]
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = (split == "test_id") | (split == "test_ood")
    feats, mu, sd = standardize(feats_raw, train_mask)

    prob = RotSyncProblem(N_NODES, redundant=False)
    runner = SDPRunner(prob, solver="SCS", eps=1e-7)
    # warm up CVXPY canonicalisation once
    t0 = time.time()
    _ = runner.solve(prob.build_Q(np.stack([np.eye(3)] * len(EDGES)), np.ones(len(EDGES)), EDGES, PRIOR_WEIGHT))
    print(f"[setup] SDP warmup {time.time()-t0:.2f}s", flush=True)

    head = None
    hist = []
    if args.method in ("twostage", "e2e"):
        epochs = args.epochs or (40 if args.method == "twostage" else 10)
        bs = args.batch_size or (64 if args.method == "twostage" else 8)
        lr = args.lr or (1e-3 if args.method == "twostage" else 3e-4)
        if args.method == "twostage":
            head, hist = train_edge_head(
                feats, R, train_mask, val_mask, prob, runner, epochs, bs, lr, args.tag
            )
            ckpt = os.path.join(args.out_dir, f"edgehead_twostage_{args.tag}.pt")
            torch.save(head.state_dict(), ckpt)
            print(f"[twostage] saved {ckpt}", flush=True)
        else:
            assert args.init, "e2e needs --init <two-stage checkpoint>"
            head, hist = finetune_e2e(
                head, feats, R, train_mask, val_mask, prob, runner, epochs, bs, lr, args.init
            )
            ckpt = os.path.join(args.out_dir, f"edgehead_e2e_{args.tag}.pt")
            torch.save(head.state_dict(), ckpt)
            print(f"[e2e] saved {ckpt}", flush=True)
    elif args.method == "direct":
        head = NodeHead(d=feats.shape[-1]).double()
        epochs = args.epochs or 40
        bs = args.batch_size or 64
        lr = args.lr or 1e-3
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        tr_idx = np.where(train_mask)[0]
        for ep in range(1, epochs + 1):
            head.train()
            perm = np.random.permutation(tr_idx)
            losses = []
            for start in range(0, len(perm), bs):
                bidx = perm[start : start + bs]
                Rp = head(feats[bidx])
                loss = geodesic_loss(Rp, R[bidx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss) * 180 / np.pi)
            sched.step()
            ev = evaluate(head, prob, runner, feats, R, val_mask, "direct")
            print(f"[direct] ep{ep:02d} train={np.mean(losses):.2f} val={ev['mean_deg']:.2f}", flush=True)
            hist.append({"epoch": ep, "train_deg": float(np.mean(losses)), "val_deg": ev["mean_deg"]})
        ckpt = os.path.join(args.out_dir, f"nodehead_direct_{args.tag}.pt")
        torch.save(head.state_dict(), ckpt)

    kind = "direct" if args.method == "direct" else ("oracle" if args.method == "oracle" else "sdpr")
    results = {"method": args.method, "tag": args.tag, "history": hist}
    for split_name in ["train", "val", "test_id", "test_ood"]:
        mask = split == split_name
        ev = evaluate(head, prob, runner, feats, R, mask, kind, collect=True)
        pred = ev.pop("_pred")
        gt = ev.pop("_gt")
        idx = ev.pop("_idx")
        rel = ev.pop("_rel")
        rank = ev.pop("_rank", None)
        eig = ev.pop("_eig", None)
        results[split_name] = ev
        if args.method in ("twostage", "e2e"):
            ev_ls = evaluate(head, prob, runner, feats, R, mask, "ls", collect=False)
            results[split_name + "_chordal"] = ev_ls
        np.savez(
            os.path.join(args.out_dir, f"pred_{args.method}_{split_name}_{args.tag}.npz"),
            pred=pred,
            gt=gt,
            idx=idx,
            rel=rel,
            rank=rank if rank is not None else np.array([]),
            eig=eig if eig is not None else np.array([]),
        )
        print(f"[{args.method}] {split_name}: {json.dumps(ev)}", flush=True)
    with open(os.path.join(args.out_dir, f"metrics_{args.method}_{args.tag}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

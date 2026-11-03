"""Extract frozen V-JEPA 2 ViT-L features for the rotating-camera benchmark.

Run on a GPU node through Slurm, e.g.::

    sbatch --job-name=sdpr-jepa-feats tools/slurm/run_gpu.sbatch \
        python /path/to/src/extract_features.py --device cuda \
        --out /path/to/artifacts/features/vjepa2_vitl.npz

The panorama is built from UCF101 frames (see ``panorama.build_panorama``);
the held-out panorama seed is used only for the OOD test split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from panorama import build_panorama, make_sequence, geodesic_deg  # noqa: E402
from vjepa2_encoder import build_encoder, encode_clips  # noqa: E402

UCF_ROOT = os.environ.get(
    "UCF101_ROOT", "/mnt/storage1/syk/jepa/datasets/ucf101_alt/UCF-101"
)
PANO_DIR = os.path.join(HERE, "..", "artifacts", "assets")
PANO_IMAGES = {seed: os.path.join(PANO_DIR, f"pano_{seed}.jpg") for seed in (1, 2, 3, 7)}

SPLITS = {
    "train": [1, 2, 3],
    "val": [1, 2, 3],
    "test_id": [1, 2, 3],
    "test_ood": [7],
}
SPLIT_COUNTS = {"train": 160, "val": 30, "test_id": 40, "test_ood": 120}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "artifacts", "models", "vitl.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "artifacts", "features", "vjepa2_vitl.npz"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sample-out", default=os.path.join(HERE, "..", "artifacts", "data", "samples.npz"))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply per-panorama sequence counts (smoke tests)")
    ap.add_argument("--pool", default="mean", choices=["mean", "meanmax"])
    args = ap.parse_args()

    t_start = time.time()
    print(f"[extract] building encoder from {args.ckpt} on {args.device}", flush=True)
    encoder, info = build_encoder(args.ckpt, device=args.device)

    # Build each panorama once; sequences are drawn with a split-specific RNG.
    panos = {}
    for seed in sorted({s for seeds in SPLITS.values() for s in seeds}):
        img = PANO_IMAGES.get(seed)
        if img is None or not os.path.exists(img):
            print(f"[extract] panorama seed={seed} (UCF fallback)", flush=True)
            panos[seed] = build_panorama(UCF_ROOT, seed=seed)
        else:
            print(f"[extract] panorama seed={seed} <- {img}", flush=True)
            panos[seed] = build_panorama(image_path=img)

    all_feats, all_R, all_split, all_pano, all_idx = [], [], [], [], []
    for split, seeds in SPLITS.items():
        n_per = max(1, int(round(SPLIT_COUNTS[split] * args.scale)))
        seed_offset = {"train": 0, "val": 1000, "test_id": 2000, "test_ood": 3000}[split]
        for pano_seed in seeds:
            rng = np.random.default_rng(seed_offset + 100000 * pano_seed)
            frames_list, R_list = [], []
            for _ in range(n_per):
                f, R = make_sequence(panos[pano_seed], rng)
                frames_list.append(f)
                R_list.append(R)
            frames = np.stack(frames_list)
            Rs = np.stack(R_list)
            print(f"[extract] {split} pano={pano_seed}: {len(frames)} clips", flush=True)
            t = time.time()
            feats = encode_clips(
                encoder, frames, device=args.device, batch_size=args.batch_size, pool=args.pool
            )
            print(f"[extract] encoded in {time.time()-t:.1f}s", flush=True)
            all_feats.append(feats)
            all_R.append(Rs)
            all_split += [split] * len(frames)
            all_pano += [pano_seed] * len(frames)
            all_idx += list(range(len(frames)))
            if split == "test_id" and pano_seed == seeds[0] and len(frames_list) >= 6:
                np.savez(
                    args.sample_out,
                    frames=np.stack([frames_list[i] for i in range(6)]),
                    R=np.stack([R_list[i] for i in range(6)]),
                )

    feats = np.concatenate(all_feats)
    Rs = np.concatenate(all_R)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        feats=feats.astype(np.float32),
        R=Rs.astype(np.float32),
        split=np.array(all_split),
        pano=np.array(all_pano),
        idx=np.array(all_idx),
    )
    meta = dict(
        checkpoint=os.path.abspath(args.ckpt),
        encoder_params=info["params"],
        num_loaded=info["num_loaded"],
        missing=len(info["missing"]),
        unexpected=len(info["unexpected"]),
        splits={k: int(v) for k, v in SPLIT_COUNTS.items()},
        pano_seeds={k: v for k, v in SPLITS.items()},
        feat_shape=list(feats.shape),
        elapsed_s=round(time.time() - t_start, 1),
        node=os.uname().nodename,
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        per_node_step_deg_mean=float(np.mean(geodesic_deg(Rs[:, :-1], Rs[:, 1:]))),
    )
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[extract] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

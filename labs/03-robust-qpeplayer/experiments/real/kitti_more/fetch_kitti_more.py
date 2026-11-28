#!/usr/bin/env python3
"""Fetch the remaining KITTI *odometry* sequences 01/02/03/04/06/08/09/10.

The odometry Velodyne archive is a single 84.8 GB zip (``data_odometry_velodyne.zip``)
mirrored on HuggingFace (``sangramrout/velodyne``).  Downloading it is out of
scope; the mirror's CDN honours HTTP ``Range`` so we stream only the ~35 frames
per sequence that the motion protocol touches (same plan as
``../kitti_odom/fetch_kitti_odom.py``).  Poses (``dataset/poses/NN.txt``) and
calib (``dataset/sequences/NN/calib.txt``) for all 11 sequences are already in
the small 1.3 / 0.6 MB official zips, so those are extracted locally.

Writes into the shared data cache ``/mnt/storage1/syk/data/kitti_odom/``
(poses/, calib/, sequences/NN/velodyne/, manifest_more.json).  No repository
file is modified outside this worktree.

Run:  /home/syk/miniconda3/envs/WM/bin/python fetch_kitti_more.py
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "kitti_odom"
sys.path.insert(0, str(PARENT))

from httpzip import HTTPRangeFile  # noqa: E402

DATA = Path("/mnt/storage1/syk/data/kitti_odom")
VELO_ZIP = ("https://huggingface.co/datasets/sangramrout/velodyne/resolve/main/"
            "data_odometry_velodyne.zip")
POSES_ZIP = DATA / "data_odometry_poses.zip"
CALIB_ZIP = DATA / "data_odometry_calib.zip"

SEQS = ["01", "02", "03", "04", "06", "08", "09", "10"]
N_PAIRS = 10
GAP_BASES = 3
GAPS = [1, 2, 4, 8]


def main_plan(n_frames: int):
    """Frame indices touched by the protocol, per sequence (same as kitti_odom)."""
    step = max(2, (n_frames - 2) // N_PAIRS)
    pairs = [(k * step, k * step + 1) for k in range(N_PAIRS)]
    bases = [0, n_frames // 3, (2 * n_frames) // 3]
    sweep = [(b, b + g) for b in bases for g in GAPS if b + g < n_frames]
    need = sorted({i for p in pairs for i in p} | {i for p in sweep for i in p})
    return pairs, sweep, need


def extract_small(seq):
    (DATA / "calib").mkdir(parents=True, exist_ok=True)
    (DATA / "poses").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(CALIB_ZIP) as z:
        member = f"dataset/sequences/{seq}/calib.txt"
        (DATA / "calib" / f"{seq}.txt").write_bytes(z.read(member))
    with zipfile.ZipFile(POSES_ZIP) as z:
        member = f"dataset/poses/{seq}.txt"
        (DATA / "poses" / f"{seq}.txt").write_bytes(z.read(member))


def main():
    t0 = time.time()
    for seq in SEQS:
        extract_small(seq)
        print(f"[calib/poses] {seq} ready", flush=True)

    remote = HTTPRangeFile(VELO_ZIP, block=1 << 20)
    print(f"[remote] velodyne zip {remote.size} B opened in {time.time()-t0:.1f}s",
          flush=True)
    zf = zipfile.ZipFile(remote)
    names = set(zf.namelist())
    print(f"[remote] {len(names)} members listed in {time.time()-t0:.1f}s",
          flush=True)

    manifest = {}
    for seq in SEQS:
        n_frames = len((DATA / "poses" / f"{seq}.txt").read_text().splitlines())
        pairs, sweep, need = main_plan(n_frames)
        outdir = DATA / "sequences" / seq / "velodyne"
        outdir.mkdir(parents=True, exist_ok=True)
        fetched, nbytes, missing = [], 0, []
        seq_t0 = time.time()
        for i in need:
            member = f"dataset/sequences/{seq}/velodyne/{i:06d}.bin"
            dst = outdir / f"{i:06d}.bin"
            if dst.exists() and dst.stat().st_size > 1000:
                continue
            if member not in names:
                missing.append(member)
                continue
            with zf.open(member) as src, open(dst, "wb") as fh:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            nbytes += dst.stat().st_size
            fetched.append(i)
        manifest[seq] = dict(n_frames=n_frames, pairs=pairs, sweep=sweep,
                             frames=need, fetched=fetched, fetched_bytes=nbytes,
                             missing=missing,
                             elapsed_s=round(time.time() - seq_t0, 1),
                             remote_http_requests=remote._n_requests)
        print(f"[seq {seq}] {n_frames} GT frames, {len(need)} needed, "
              f"fetched {len(fetched)} ({nbytes/1e6:.1f} MB) in "
              f"{time.time()-seq_t0:.1f}s, missing={len(missing)}", flush=True)

    manifest["_meta"] = dict(
        source=VELO_ZIP, poses=str(POSES_ZIP), calib=str(CALIB_ZIP),
        remote_zip_bytes=remote.size, elapsed_s=round(time.time() - t0, 1),
        n_remote_requests=remote._n_requests,
        note="range-streamed only the protocol frames with 1 MiB range blocks "
             "(central directory + member headers + frame payloads); the 84.8 GB "
             "zip is never downloaded in full")
    (DATA / "manifest_more.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tot = sum(v.get("fetched_bytes", 0) for v in manifest.values()
              if isinstance(v, dict))
    print(f"[done] {tot/1e6:.1f} MB in {time.time()-t0:.1f}s -> manifest_more.json",
          flush=True)


if __name__ == "__main__":
    main()

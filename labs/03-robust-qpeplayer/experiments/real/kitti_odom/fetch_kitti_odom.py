#!/usr/bin/env python3
"""Fetch the KITTI *odometry* frames we need for sequences 00 / 05 / 07.

Why this and not "more raw drives"?  The requested axis is KITTI **00-10**,
i.e. the odometry benchmark sequences with *exact* ground-truth poses.  The
official odometry Velodyne archive is ~80 GB and there is no per-sequence
split on the official site; the only per-sequence HuggingFace mirrors found
(``yujie2696/kitti_odometry_0X``, 2026-01) contain **images + calib only**, no
``velodyne/``, so they cannot be used.

What *is* usable: ``sangramrout/velodyne`` mirrors the official
``data_odometry_velodyne.zip`` (84.8 GB, 43,574 members, one member per frame)
and its CDN honours HTTP ``Range`` at ~6 MB/s, plus it mirrors the official
``data_odometry_poses.zip`` (1.3 MB, ``dataset/poses/NN.txt``) and
``data_odometry_calib.zip`` (0.6 MB, per-sequence ``calib.txt`` with ``Tr:``).
So we stream just the ~30 frames per sequence that the protocol touches
(~60 MB/sequence) instead of 80 GB.

Layout written under ``/mnt/storage1/syk/data/kitti_odom/``::

    poses/NN.txt                       # 12-number 3x4 T_w_cam0, exact GT
    calib/NN.txt                       # P0..P3 + Tr (velodyne -> cam0)
    sequences/NN/velodyne/000000.bin   # only the extracted frames
    manifest.json                      # frames, byte counts, timings

Run:  /home/syk/miniconda3/envs/WM/bin/python fetch_kitti_odom.py
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from httpzip import HTTPRangeFile  # noqa: E402

DATA = Path("/mnt/storage1/syk/data/kitti_odom")
VELO_ZIP = ("https://huggingface.co/datasets/sangramrout/velodyne/resolve/main/"
            "data_odometry_velodyne.zip")
POSES_ZIP = ("https://huggingface.co/datasets/sangramrout/velodyne/resolve/main/"
             "data_odometry_poses.zip")
CALIB_ZIP = ("https://huggingface.co/datasets/sangramrout/velodyne/resolve/main/"
             "data_odometry_calib.zip")

SEQS = ["00", "05", "07"]
N_PAIRS = 10
GAP_BASES = 3          # base frames for the motion (frame-gap) sweep
GAPS = [1, 2, 4, 8]


def main_plan(n_frames: int):
    """Frame indices touched by the protocol, per sequence."""
    step = max(2, (n_frames - 2) // N_PAIRS)
    pairs = [(k * step, k * step + 1) for k in range(N_PAIRS)]
    bases = [0, n_frames // 3, (2 * n_frames) // 3]
    sweep = [(b, b + g) for b in bases for g in GAPS if b + g < n_frames]
    need = sorted({i for p in pairs for i in p} | {i for p in sweep for i in p})
    return pairs, sweep, need


def fetch_small(name, url):
    import requests
    path = DATA / name
    if path.exists() and path.stat().st_size > 1000:
        return path
    DATA.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.proxies = {"http": "http://127.0.0.1:7899", "https": "http://127.0.0.1:7899"}
    t0 = time.time()
    r = sess.get(url, timeout=300)
    r.raise_for_status()
    path.write_bytes(r.content)
    print(f"downloaded {name} {len(r.content)} B in {time.time()-t0:.1f}s", flush=True)
    return path


def main():
    t0 = time.time()
    calib_zip = fetch_small("data_odometry_calib.zip", CALIB_ZIP)
    poses_zip = fetch_small("data_odometry_poses.zip", POSES_ZIP)
    for seq in SEQS:
        (DATA / "calib").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(calib_zip) as z:
            member = f"dataset/sequences/{seq}/calib.txt"
            if member in z.namelist():
                (DATA / "calib" / f"{seq}.txt").write_bytes(z.read(member))
        with zipfile.ZipFile(poses_zip) as z:
            member = f"dataset/poses/{seq}.txt"
            (DATA / "poses").mkdir(parents=True, exist_ok=True)
            (DATA / "poses" / f"{seq}.txt").write_bytes(z.read(member))
        print(f"calib/poses for {seq} ready", flush=True)

    manifest = {}
    remote = HTTPRangeFile(VELO_ZIP)
    print(f"remote velodyne zip: {remote.size} B, opened in {time.time()-t0:.1f}s", flush=True)
    zf = zipfile.ZipFile(remote)
    names = set(zf.namelist())
    print(f"{len(names)} remote members listed in {time.time()-t0:.1f}s", flush=True)

    for seq in SEQS:
        n_frames = int((DATA / "poses" / f"{seq}.txt").read_text().count("\n"))
        n_frames = max(n_frames, sum(1 for _ in open(DATA / "poses" / f"{seq}.txt")))
        pairs, sweep, need = main_plan(n_frames)
        outdir = DATA / "sequences" / seq / "velodyne"
        outdir.mkdir(parents=True, exist_ok=True)
        got, fetched, nbytes = [], [], 0
        for i in need:
            member = f"dataset/sequences/{seq}/velodyne/{i:06d}.bin"
            dst = outdir / f"{i:06d}.bin"
            got.append(i)
            if dst.exists() and dst.stat().st_size > 1000:
                continue
            if member not in names:
                raise FileNotFoundError(member)
            info = zf.getinfo(member)
            with zf.open(info) as src, open(dst, "wb") as fh:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            nbytes += dst.stat().st_size
            fetched.append(i)
        manifest[seq] = dict(n_frames=n_frames, pairs=pairs, sweep=sweep,
                             frames=got, fetched=fetched, fetched_bytes=nbytes,
                             remote_http_requests=remote._n_requests)
        print(f"seq {seq}: {n_frames} frames in GT, {len(pairs)} pairs, "
              f"{len(need)} frames needed, fetched {len(fetched)} "
              f"({nbytes/1e6:.1f} MB), total {time.time()-t0:.1f}s", flush=True)

    manifest["_meta"] = dict(
        source=VELO_ZIP, poses=POSES_ZIP, calib=CALIB_ZIP,
        remote_zip_bytes=remote.size, elapsed_s=round(time.time() - t0, 1),
        note="only the frames the protocol needs are extracted; no 80 GB download")
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v.get("fetched_bytes") for k, v in manifest.items()
                      if isinstance(v, dict)}, indent=2))
    print("wrote", DATA / "manifest.json")


if __name__ == "__main__":
    main()

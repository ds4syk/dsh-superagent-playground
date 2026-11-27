#!/usr/bin/env python3
"""Fast per-frame range fetcher for the KITTI odometry Velodyne mirror.

Why not ``zipfile.ZipFile(HTTPRangeFile)``?  Each block read costs one HTTP
round-trip, and through the local proxy a 1 MiB range request takes ~2 s
(latency-bound), so streaming 33 frames per sequence sequentially ran at
~5 s/frame.  The Velodyne members are ZIP_STORED, so each frame can instead be
fetched with **one** range request spanning the local header + stored payload
(~2 MB), and frames can be fetched in parallel.

Only the frames the motion protocol touches are fetched; poses/calib come from
the small official zips already in the data cache.  No 84.8 GB download.

Run:  /home/syk/miniconda3/envs/WM/bin/python fetch_fast.py 01 02 ... 10
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "kitti_odom"
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(HERE))

from httpzip import HTTPRangeFile, PROXY  # noqa: E402
from fetch_kitti_more import DATA, VELO_ZIP, main_plan  # noqa: E402

SEQS = sys.argv[1:] or ["01", "02", "03", "04", "06", "08", "09", "10"]
LOCAL_HDR = 30
MAX_EXTRA = 1 << 12
WORKERS = 8


def frame_bytes(sess, base_url, info, member):
    """One range request: local header + stored payload; returns raw bytes."""
    name_len = len(member.encode())
    start = info.header_offset
    end = start + LOCAL_HDR + name_len + MAX_EXTRA + info.compress_size
    last = None
    for attempt in range(4):
        try:
            r = sess.get(base_url, headers={"Range": f"bytes={start}-{end - 1}"},
                         timeout=180)
            r.raise_for_status()
            blob = r.content
            if len(blob) < LOCAL_HDR + name_len + info.compress_size:
                raise RuntimeError(f"short read {len(blob)}")
            nlen = int.from_bytes(blob[26:28], "little")
            elen = int.from_bytes(blob[28:30], "little")
            off = LOCAL_HDR + nlen + elen
            payload = blob[off:off + info.compress_size]
            if len(payload) != info.compress_size:
                raise RuntimeError(f"payload {len(payload)} != {info.compress_size}")
            if zlib.crc32(payload) & 0xFFFFFFFF != info.CRC:
                raise RuntimeError("crc mismatch")
            return payload
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{member}: {last}")


def main():
    t0 = time.time()
    remote = HTTPRangeFile(VELO_ZIP)
    import zipfile
    zf = zipfile.ZipFile(remote)
    names = set(zf.namelist())
    print(f"[remote] {len(names)} members listed in {time.time()-t0:.1f}s", flush=True)
    base_url = remote.url
    remote.close()

    sess = requests.Session()
    sess.proxies = {"http": PROXY, "https": PROXY}
    sess.headers.update({"User-Agent": "rangezip-fast/1.0"})
    ad = requests.adapters.HTTPAdapter(pool_connections=WORKERS,
                                       pool_maxsize=WORKERS, max_retries=0)
    sess.mount("https://", ad)
    sess.mount("http://", ad)

    manifest = {}
    for seq in SEQS:
        n_frames = len((DATA / "poses" / f"{seq}.txt").read_text().splitlines())
        pairs, sweep, need = main_plan(n_frames)
        outdir = DATA / "sequences" / seq / "velodyne"
        outdir.mkdir(parents=True, exist_ok=True)
        todo = []
        for i in need:
            member = f"dataset/sequences/{seq}/velodyne/{i:06d}.bin"
            dst = outdir / f"{i:06d}.bin"
            if dst.exists() and dst.stat().st_size > 1000:
                continue
            if member not in names:
                raise FileNotFoundError(member)
            todo.append((i, member, zf.getinfo(member)))
        seq_t0 = time.time()
        ok, nbytes = [], 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(frame_bytes, sess, base_url, info, member): (i, member)
                    for i, member, info in todo}
            for fut in as_completed(futs):
                i, member = futs[fut]
                payload = fut.result()
                (outdir / f"{i:06d}.bin").write_bytes(payload)
                ok.append(i)
                nbytes += len(payload)
        manifest[seq] = dict(n_frames=n_frames, pairs=pairs, sweep=sweep,
                             frames=need, fetched=sorted(ok),
                             fetched_bytes=nbytes,
                             elapsed_s=round(time.time() - seq_t0, 1))
        print(f"[seq {seq}] {len(need)} needed, fetched {len(ok)} "
              f"({nbytes/1e6:.1f} MB) in {time.time()-seq_t0:.1f}s", flush=True)

    (DATA / "manifest_fast.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {sum(v['fetched_bytes'] for v in manifest.values())/1e6:.1f} MB "
          f"in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

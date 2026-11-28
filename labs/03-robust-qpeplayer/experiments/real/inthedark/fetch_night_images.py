#!/usr/bin/env python3
"""Parallel zip range-fetch for a *window* of UTIAS In-The-Dark night frames.

Why this exists
---------------
`fetch_zip_members.py` extracts one member per `curl` call, serially.  On the
night archives that is ~7 KB/s per stream and one FTP round-trip per image, so
22 images (10 stereo pairs) take >15 min.  The server is per-connection
limited, not total-bandwidth limited: N concurrent range requests aggregate.

This script parses the zip central directory once (from a cached `.tmp/cd.bin`
when present, otherwise a 128 KB tail probe) and then range-fetches the
requested members through a thread pool, decompressing raw deflate and
verifying size + CRC32 per member.  It also handles the zip64 0xFFFFFFFF
offsets that cover ~22% of the night-run image entries.

Usage:
  python fetch_night_images.py --url ftp://.../run_000018.zip --size 2605865200 \
      --out-dir /data/run_000018 --window 2529 2539 --workers 12
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import struct
import subprocess
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fetch_zip_members as F  # noqa: E402


def parse_cd(cd: bytes) -> dict:
    ents, p = {}, 0
    while cd[p:p + 4] == b"PK\x01\x02":
        (_, _, _, method, _, _, crc, csize, usize, nlen, elen, clen,
         _, _, _, lho) = struct.unpack_from("<HHHHHHIIIHHHHHII", cd, p + 4)
        name = cd[p + 46:p + 46 + nlen].decode()
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if 0xFFFFFFFF in (usize, csize, lho):
            usize, csize, lho = F._zip64_extra(extra, usize, csize, lho)
        ents[name] = dict(method=method, csize=csize, usize=usize, lho=lho,
                          nlen=nlen, elen=elen, crc=crc)
        p += 46 + nlen + elen + clen
    return ents


def get_entries(url: str, size: int, out_dir: pathlib.Path) -> dict:
    tmp = out_dir / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    cd_path = tmp / "cd.bin"
    if not cd_path.exists():
        tail = tmp / "tail.bin"
        F.fetch_range(url, size - 128 * 1024, size - 1, tail)
        blob = tail.read_bytes()
        i = blob.rfind(b"PK\x05\x06")
        if i < 0:
            raise SystemExit("no EOCD in tail")
        _, _, _, _, _, cd_size, cd_off, _ = struct.unpack_from("<IHHHHIIH", blob, i)
        base = size - len(blob)
        F.fetch_range(url, cd_off, min(cd_off + cd_size + 64, size - 1), cd_path)
    return parse_cd(cd_path.read_bytes())


def fetch_one(url: str, m: str, ent: dict, out: pathlib.Path,
              tmp: pathlib.Path, idx: int) -> tuple[str, bool, str]:
    part = tmp / f"m{idx}.bin"
    need = 30 + ent["nlen"] + ent["elen"] + ent["csize"]
    cmd = ["curl", "-sS", "--fail", "--retry", "4", "--retry-delay", "2",
           "-r", f"{ent['lho']}-{ent['lho'] + need - 1}", "-o", str(part), url]
    try:
        subprocess.run(cmd, check=True)
        blob = part.read_bytes()
        nlen, elen = struct.unpack_from("<HH", blob, 26)
        raw = blob[30 + nlen + elen:30 + nlen + elen + ent["csize"]]
        data = (zlib.decompressobj(-15).decompress(raw) if ent["method"] == 8
                else raw)
        if len(data) != ent["usize"]:
            return m, False, f"size {len(data)} != {ent['usize']}"
        if zlib.crc32(data) & 0xFFFFFFFF != ent["crc"]:
            return m, False, "crc mismatch"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return m, True, f"{len(data)} B"
    except subprocess.CalledProcessError as e:
        return m, False, f"curl rc={e.returncode}"
    finally:
        part.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", nargs=2, type=int, required=True,
                    help="inclusive vertex id range")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out_dir = pathlib.Path(a.out_dir)
    ents = get_entries(a.url, a.size, out_dir)
    v0, v1 = a.window
    members = [f"images/{s}/{v:06d}.png" for v in range(v0, v1 + 1)
               for s in ("left", "right")]
    missing = [m for m in members if m not in ents]
    if missing:
        raise SystemExit(f"not in archive: {missing[:3]}")
    todo = [m for m in members if not (out_dir / m).exists()]
    print(f"{len(members)} members, {len(todo)} to fetch, {a.workers} workers",
          flush=True)
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_one, a.url, m, ents[m], out_dir / m,
                          out_dir / ".tmp", i): m
                for i, m in enumerate(todo)}
        for fut in cf.as_completed(futs):
            m, good, info = fut.result()
            ok += good
            print(f"  [{'ok' if good else 'FAIL'}] {m}: {info}", flush=True)
    print(f"fetched {ok}/{len(todo)} ok", flush=True)
    if ok != len(todo):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

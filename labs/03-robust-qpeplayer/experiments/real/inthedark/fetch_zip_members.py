#!/usr/bin/env python3
"""Extract selected members of a remote (FTP/HTTP) zip via byte-range requests.

Motivation: the official UTIAS zips are 590-905 MB and direct FTP throughput is
~14 KB/s, yet the *metadata* members we need (transforms_temporal.txt,
timestamps_images.txt, gps.txt) are 9-130 KB compressed.  A zip central
directory lets us locate each member's local header offset, so we can fetch a
few KB instead of the whole archive.  Works for any server honouring Range / FTP
REST (the UTIAS FTP does: `Accept-ranges: bytes`).

Usage:
  $PY fetch_zip_members.py --url ftp://.../run_000000.zip --size 904605594 \
      --member transforms_temporal.txt --member timestamps_images.txt --out-dir .
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess
import zlib

CHUNK = 30 + 4096  # local header + generous name/extra allowance


def fetch_range(url: str, start: int, end: int, out: pathlib.Path) -> bytes:
    """Inclusive byte range [start, end] -> bytes (curl handles FTP REST)."""
    subprocess.run(["curl", "-s", "--fail", "-r", f"{start}-{end}", "-o", str(out), url],
                   check=True)
    return out.read_bytes()


def central_directory(url: str, size: int, tmp: pathlib.Path):
    tail = fetch_range(url, size - 128 * 1024, size - 1, tmp / "tail.bin")
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise SystemExit("no EOCD in tail; not a zip or archive too small for tail probe")
    base = size - len(tail)
    _, _, _, n_disk, n_tot, cd_size, cd_off, _ = struct.unpack_from("<IHHHHIIH", tail, i)
    if cd_off == 0xFFFFFFFF:
        raise SystemExit("zip64 archive: not supported by this probe")
    cd = tail[cd_off - base:] if cd_off >= base else fetch_range(
        url, cd_off, min(cd_off + cd_size + 64, size - 1), tmp / "cd.bin")
    ents, p = {}, 0
    while cd[p:p + 4] == b"PK\x01\x02":
        (_, _, _, method, _, _, crc, csize, usize, nlen, elen, clen,
         _, _, _, lho) = struct.unpack_from("<HHHHHHIIIHHHHHII", cd, p + 4)
        name = cd[p + 46:p + 46 + nlen].decode()
        ents[name] = dict(method=method, csize=csize, usize=usize, lho=lho,
                          nlen=nlen, elen=elen, crc=crc)
        p += 46 + nlen + elen + clen
    if len(ents) != n_tot:
        raise SystemExit(f"central directory parse mismatch: {len(ents)} != {n_tot}")
    return ents


def extract(url: str, ent: dict, out: pathlib.Path, tmp: pathlib.Path) -> pathlib.Path:
    need = 30 + ent["nlen"] + ent["elen"] + ent["csize"]
    blob = fetch_range(url, ent["lho"], ent["lho"] + need - 1, tmp / "member.bin")
    nlen, elen = struct.unpack_from("<HH", blob, 26)
    start = 30 + nlen + elen
    raw = blob[start:start + ent["csize"]]
    if ent["method"] == 8:
        data = zlib.decompressobj(-15).decompress(raw)
    elif ent["method"] == 0:
        data = raw
    else:
        raise SystemExit(f"unsupported compression method {ent['method']}")
    if len(data) != ent["usize"]:
        raise SystemExit(f"size mismatch {len(data)} != {ent['usize']}")
    if zlib.crc32(data) & 0xFFFFFFFF != ent["crc"]:
        raise SystemExit("CRC mismatch: range extraction corrupt")
    out.write_bytes(data)
    print(f"  {out.name}: {len(data)} B (crc ok, deflate {ent['csize']} B)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--member", action="append", default=[])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    tmp = pathlib.Path(a.out_dir)
    (tmp / ".tmp").mkdir(parents=True, exist_ok=True)
    ents = central_directory(a.url, a.size, tmp / ".tmp")
    if a.list or not a.member:
        for name, e in ents.items():
            if not name.startswith("images/"):
                print(f"{name:40s} csize={e['csize']:>9d} usize={e['usize']:>9d} lho={e['lho']}")
            elif a.list:
                print(f"{name:40s} csize={e['csize']:>9d} usize={e['usize']:>9d} lho={e['lho']}")
        return
    for m in a.member:
        if m not in ents:
            raise SystemExit(f"member {m!r} not in archive")
        extract(a.url, ents[m], tmp / pathlib.Path(m).name, tmp / ".tmp")


if __name__ == "__main__":
    main()

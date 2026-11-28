#!/usr/bin/env python3
"""Minimal HTTP range reader so ``zipfile`` can open a remote 84 GB zip.

The KITTI odometry Velodyne mirror we found (``sangramrout/velodyne`` on
HuggingFace) ships one monolithic ``data_odometry_velodyne.zip`` (84.8 GB)
instead of per-sequence archives.  Downloading it is out of scope, but the
mirror's CDN honours ``Range`` requests at ~6 MB/s, so we only fetch the
handful of ``velodyne/NNNNNN.bin`` members that the protocol needs.

This class implements just enough of the ``io.RawIOBase`` seek/read API for
``zipfile.ZipFile`` (it reads the central directory at the end of the file and
then seeks to individual member headers).  Reads go through a small LRU block
cache so the sequential member scan does not re-fetch bytes.
"""
from __future__ import annotations

import io
from collections import OrderedDict

import requests

PROXY = "http://127.0.0.1:7899"
CHUNK = 8 * 2 ** 20  # 8 MiB blocks


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block=CHUNK, proxy=PROXY, timeout=120):
        self.url = url
        self.block = int(block)
        self.timeout = timeout
        self.sess = requests.Session()
        self.sess.proxies = {"http": proxy, "https": proxy}
        self.sess.headers.update({"User-Agent": "rangezip/1.0"})
        self.pos = 0
        self._cache = OrderedDict()
        self._n_requests = 0
        r = self.sess.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout)
        r.raise_for_status()
        cr = r.headers.get("Content-Range", "")
        r.close()
        if "/" not in cr:
            raise RuntimeError(f"server did not answer Range request: {cr!r}")
        self.size = int(cr.rsplit("/", 1)[1])
        # resolve redirects once so each block request hits the CDN directly
        self.url = r.url

    # -- io plumbing -----------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = int(off)
        elif whence == 1:
            self.pos += int(off)
        elif whence == 2:
            self.pos = self.size + int(off)
        else:
            raise ValueError(whence)
        return self.pos

    def _block(self, k):
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        lo = k * self.block
        hi = min(lo + self.block, self.size) - 1
        for attempt in range(4):
            try:
                r = self.sess.get(self.url, headers={"Range": f"bytes={lo}-{hi}"},
                                  timeout=self.timeout)
                r.raise_for_status()
                data = r.content
                if len(data) != hi - lo + 1:
                    raise RuntimeError(f"short block {len(data)} != {hi-lo+1}")
                break
            except Exception:
                if attempt == 3:
                    raise
        self._n_requests += 1
        self._cache[k] = data
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = []
        remaining = n
        while remaining > 0:
            k, off = divmod(self.pos, self.block)
            blk = self._block(k)
            take = min(remaining, len(blk) - off)
            out.append(blk[off:off + take])
            self.pos += take
            remaining -= take
        return b"".join(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def close(self):
        try:
            self.sess.close()
        finally:
            super().close()

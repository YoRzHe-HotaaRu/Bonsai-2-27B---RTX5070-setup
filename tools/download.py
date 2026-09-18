#!/usr/bin/env python3
"""Resumable, multi-threaded downloader using only the Python standard library.

Why this exists: on this machine, schannel-based clients (curl, git,
Invoke-WebRequest) fail with SEC_E_NO_CREDENTIALS inside the sandboxed shell,
while Python's bundled OpenSSL stack works. This tool therefore talks HTTPS
directly via urllib, downloads a file with several parallel HTTP range
requests, resumes across runs via a block bitmap sidecar, and verifies the
resulting file against a known SHA-256 when one is supplied.

Usage:
    python tools/download.py URL DEST [--size BYTES] [--sha256 HEX] [--threads N]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
import urllib.error
import urllib.request

BLOCK = 4 * 1024 * 1024
UA = "bonsai-setup/1.0"


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def fetch_range(url: str, start: int, end: int, attempts: int = 4) -> bytes:
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": UA},
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read: {len(data)} != {end - start + 1}")
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"range {start}-{end} failed after {attempts} attempts: {last}")


class Downloader:
    def __init__(self, url: str, dest: str, size: int, threads: int, sha256: str | None) -> None:
        self.url = url
        self.dest = dest
        self.size = size
        self.threads = threads
        self.sha256 = sha256.lower() if sha256 else None
        self.nblocks = (size + BLOCK - 1) // BLOCK
        self.bitmap_path = dest + ".blocks"
        self.bitmap = bytearray(self.nblocks)
        self.lock = threading.Lock()
        self.next_block = 0
        self.done_bytes = 0
        self.errors: list[str] = []
        self.stop = False
        self._load_bitmap()

    def _load_bitmap(self) -> None:
        if os.path.exists(self.bitmap_path):
            with open(self.bitmap_path, "rb") as f:
                raw = f.read()
            for i, b in enumerate(raw[: self.nblocks]):
                if b:
                    self.bitmap[i] = 1
                    self.done_bytes += self._block_len(i)
            if self.done_bytes:
                print(f"  resume: {self.done_bytes / 1e9:.2f} GB already present")

    def _block_len(self, i: int) -> int:
        start = i * BLOCK
        return min(BLOCK, self.size - start)

    def _save_bitmap(self) -> None:
        tmp = self.bitmap_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(bytes(self.bitmap))
        os.replace(tmp, self.bitmap_path)

    def _worker(self, fh) -> None:
        while not self.stop:
            with self.lock:
                if self.next_block >= self.nblocks:
                    return
                i = self.next_block
                self.next_block += 1
            start = i * BLOCK
            end = start + self._block_len(i) - 1
            try:
                data = fetch_range(self.url, start, end)
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.errors.append(str(e))
                    self.stop = True
                return
            with self.lock:
                fh.seek(start)
                fh.write(data)
                self.bitmap[i] = 1
                self.done_bytes += len(data)

    def run(self) -> bool:
        if os.path.exists(self.dest) and os.path.getsize(self.dest) == self.size and not any(self.bitmap):
            print("  file already complete, verifying hash only")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(self.dest)), exist_ok=True)
            with open(self.dest, "r+b" if os.path.exists(self.dest) else "wb") as fh:
                if not os.path.exists(self.dest) or os.fstat(fh.fileno()).st_size != self.size:
                    fh.truncate(self.size)
                t0 = time.time()
                base = self.done_bytes
                threads = [threading.Thread(target=self._worker, args=(fh,), daemon=True)
                           for _ in range(self.threads)]
                for t in threads:
                    t.start()
                while any(t.is_alive() for t in threads):
                    time.sleep(3)
                    with self.lock:
                        moved = self.done_bytes - base
                        dt = time.time() - t0
                        pct = 100 * self.done_bytes / self.size
                        rate = moved / 1e6 / dt if dt else 0
                        eta = (self.size - self.done_bytes) / 1e6 / max(rate, 0.01) / 60
                        print(f"  {pct:6.2f}%  {self.done_bytes / 1e9:.2f}/{self.size / 1e9:.2f} GB"
                              f"  {rate:5.1f} MB/s  eta {eta:4.1f} min", flush=True)
                        self._save_bitmap()
                for t in threads:
                    t.join()
            if self.errors:
                print("  FAILED: " + self.errors[0])
                return False
            self._save_bitmap()

        if os.path.getsize(self.dest) != self.size:
            print(f"  size mismatch: {os.path.getsize(self.dest)} != {self.size}")
            return False

        if self.sha256:
            print("  verifying sha256 ...", flush=True)
            h = hashlib.sha256()
            with open(self.dest, "rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest().lower() != self.sha256:
                print(f"  SHA256 MISMATCH\n    got      {h.hexdigest()}\n    expected {self.sha256}")
                return False
            print("  sha256 OK")

        try:
            os.remove(self.bitmap_path)
        except OSError:
            pass
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--sha256", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    size = args.size or remote_size(args.url)
    print(f"downloading {args.url}")
    print(f"        -> {args.dest}  ({size / 1e9:.2f} GB, {args.threads} threads)")
    dl = Downloader(args.url, args.dest, size, args.threads, args.sha256)
    return 0 if dl.run() else 1


if __name__ == "__main__":
    sys.exit(main())

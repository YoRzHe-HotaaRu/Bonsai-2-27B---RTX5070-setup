#!/usr/bin/env python3
"""Second-pass probe: corrected release asset names + real throughput test."""

from __future__ import annotations

import concurrent.futures as cf
import time
import urllib.error
import urllib.request

FORK = "PrismML-Eng/llama.cpp"
TAG = "prism-b10687-5d80cff"
BASE = f"https://github.com/{FORK}/releases/download/{TAG}"
MODEL_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf"
TIMEOUT = 60

# Correct names: the tag already begins with "prism-", so binaries are
# llama-prism-b10687-... while the GPU cudart packs are cudart-llama-bin-...
CANDIDATES = [
    f"llama-prism-b10687-5d80cff-bin-win-cuda-13.3-x64.zip",
    f"llama-prism-b10687-5d80cff-bin-win-cuda-12.4-x64.zip",
    "llama-bin-win-cpu-x64.zip",
    "llama-bin-win-vulkan-x64.zip",
    "llama-bin-win-hip-radeon-x64.zip",
    "cudart-llama-bin-win-cuda-13.3-x64.zip",
]


def head(url: str) -> tuple[int, int]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "probe/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status, int(r.headers.get("Content-Length") or 0)


def timed_get(url: str, start: int, length: int) -> tuple[int, float]:
    end = start + length - 1
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "probe/1.0"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = r.read()
    return len(data), time.time() - t0


def main() -> None:
    print("=" * 78)
    print("A. Corrected Windows binary URLs")
    print("=" * 78)
    for name in CANDIDATES:
        try:
            status, length = head(f"{BASE}/{name}")
            print(f"  OK    {name:<56} {length / 1e6:>9.1f} MB")
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code}  {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERR   {name}: {type(e).__name__}")

    print()
    print("=" * 78)
    print("B. Throughput: single 8 MiB range")
    print("=" * 78)
    n, dt = timed_get(MODEL_URL, 0, 8 * 1024 * 1024)
    single = (n / 1e6) / dt
    print(f"  {n / 1e6:.2f} MB in {dt:.2f}s -> {single:.1f} MB/s")

    print()
    print("=" * 78)
    print("C. Throughput: 8 parallel 2 MiB ranges")
    print("=" * 78)
    jobs = [(i * 32 * 1024 * 1024, 2 * 1024 * 1024) for i in range(8)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(timed_get, MODEL_URL, s, l) for s, l in jobs]
        total = sum(f.result()[0] for f in futs)
    dt = time.time() - t0
    par = (total / 1e6) / dt
    print(f"  {total / 1e6:.2f} MB in {dt:.2f}s -> {par:.1f} MB/s aggregate")

    best = max(single, par)
    print()
    print(f"  best observed: {best:.1f} MB/s")
    for label, size_gb in (("PQ2_0", 7.21), ("PTQ1_0", 5.95), ("mmproj-Q8_0", 0.63)):
        print(f"  {label:<12} {size_gb:>5.2f} GB -> {size_gb * 1000 / max(best, 0.01) / 60:>6.1f} min")


if __name__ == "__main__":
    main()

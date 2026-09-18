#!/usr/bin/env python3
"""Connectivity / asset availability probe for the Bonsai 2 27B setup.

Stdlib only. Uses Python's own OpenSSL stack, which works in environments
where schannel-based clients (curl, git, Invoke-WebRequest) are blocked.

Checks:
  1. Every asset attached to the pinned PrismML llama.cpp fork release.
  2. Whether each candidate Windows CUDA binary URL resolves (HEAD / Range GET).
  3. The exact byte size and reachability of each GGUF in the model repo.
  4. Download throughput on the model repo (1 MiB range request).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

FORK = "PrismML-Eng/llama.cpp"
TAG = "prism-b10687-5d80cff"
MODEL_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
TIMEOUT = 45

CANDIDATE_BINARIES = [
    f"llama-prism-{TAG}-bin-win-cuda-13.3-x64.zip",
    f"llama-prism-{TAG}-bin-win-cuda-12.4-x64.zip",
    "llama-bin-win-cpu-x64.zip",
    "llama-bin-win-vulkan-x64.zip",
    "cudart-llama-bin-win-cuda-13.3-x64.zip",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
]

MODEL_FILES = [
    "Ternary-Bonsai-2-27B-PQ2_0.gguf",
    "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
    "Ternary-Bonsai-2-27B-mmproj-BF16.gguf",
]


def head(url: str) -> tuple[int, int, str]:
    """Return (status, content-length, final-url) following redirects."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "net-probe/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        length = int(r.headers.get("Content-Length") or 0)
        return r.status, length, r.url


def main() -> int:
    print("=" * 78)
    print(f"1. Assets attached to {FORK} release {TAG}")
    print("=" * 78)
    api = f"https://api.github.com/repos/{FORK}/releases/tags/{TAG}"
    try:
        with urllib.request.urlopen(api, timeout=TIMEOUT) as r:
            rel = json.load(r)
        for a in rel.get("assets", []):
            print(f"  {a['name']:<58} {a['size'] / 1e6:>10.1f} MB  ({a['state']})")
    except urllib.error.HTTPError as e:
        print(f"  API error: HTTP {e.code}")
    except Exception as e:  # noqa: BLE001
        print(f"  API error: {type(e).__name__}: {e}")

    print()
    print("=" * 78)
    print("2. Candidate Windows binary URLs")
    print("=" * 78)
    base = f"https://github.com/{FORK}/releases/download/{TAG}"
    available = []
    for name in CANDIDATE_BINARIES:
        url = f"{base}/{name}"
        try:
            status, length, _ = head(url)
            print(f"  OK    {name:<56} {length / 1e6:>9.1f} MB")
            available.append((name, length, url))
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERR   {name}: {type(e).__name__}")

    print()
    print("=" * 78)
    print("3. Model files in " + MODEL_REPO)
    print("=" * 78)
    for name in MODEL_FILES:
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{name}"
        try:
            status, length, _ = head(url)
            print(f"  OK    {name:<48} {length / 1e9:>7.2f} GB")
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERR   {name}: {type(e).__name__}")

    print()
    print("=" * 78)
    print("4. Throughput sample (1 MiB range request on PQ2_0)")
    print("=" * 78)
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1048575", "User-Agent": "net-probe/1.0"})
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read()
        dt = time.time() - t0
        mbps = (len(data) / 1e6) / dt if dt else 0
        print(f"  {len(data) / 1e6:.2f} MB in {dt:.2f}s -> {mbps:.1f} MB/s")
        print(f"  projected time for 7.21 GB: {7206 / max(mbps, 0.01) / 60:.1f} min")
    except Exception as e:  # noqa: BLE001
        print(f"  ERR: {type(e).__name__}: {e}")

    print()
    print("=== available binaries ===")
    for name, length, url in available:
        print(f"{name}\t{length}\t{url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

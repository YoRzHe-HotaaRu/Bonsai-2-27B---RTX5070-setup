#!/usr/bin/env python3
"""Fetch and unpack the PrismML llama.cpp fork binaries for Windows x64 CUDA 13.3.

Uses the release tag the Bonsai demo pins (prism-b10683-d8f26ee) and verifies
each asset against the SHA-256 the GitHub API publishes for it.

Result:
    bin/cuda/llama-cli.exe, llama-server.exe, ggml*.dll, cudart DLLs ...
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download import Downloader  # noqa: E402

FORK = "PrismML-Eng/llama.cpp"
TAG = "prism-b10683-d8f26ee"
# Selected by predicate, not by a constructed name: the release asset names embed
# the tag (which already starts with "prism-"), and the cudart pack name itself
# ends with "-bin-win-cuda-13.3-x64.zip", so suffix matching alone is ambiguous.
WANT: list[tuple[str, object]] = [
    ("llama.cpp CUDA 13.3 binaries",
     lambda n: n.startswith("llama-") and n.endswith("-bin-win-cuda-13.3-x64.zip")),
    ("CUDA 13.3 runtime DLLs",
     lambda n: n.startswith("cudart-") and n.endswith("win-cuda-13.3-x64.zip")),
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_BIN = os.path.join(ROOT, "bin", "cuda")
DEST_DL = os.path.join(ROOT, "downloads")


def release_assets() -> dict[str, dict]:
    url = f"https://api.github.com/repos/{FORK}/releases/tags/{TAG}"
    with urllib.request.urlopen(url, timeout=60) as r:
        rel = json.load(r)
    return {a["name"]: a for a in rel.get("assets", [])}


def extract(zip_path: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        # Release zips are flat except some are wrapped in a single folder.
        for n in names:
            if n.endswith("/"):
                continue
            target = os.path.join(dest, os.path.basename(n))
            with z.open(n) as src, open(target, "wb") as out:
                out.write(src.read())
    print(f"  extracted {len([n for n in names if not n.endswith('/')])} files -> {dest}")


def main() -> int:
    os.makedirs(DEST_DL, exist_ok=True)
    os.makedirs(DEST_BIN, exist_ok=True)
    assets = release_assets()

    wanted: list[str] = []
    for label, pred in WANT:
        matches = sorted(n for n in assets if pred(n) and "arm64" not in n)
        if not matches:
            print(f"  MISSING asset for {label}")
            print("  available:", ", ".join(sorted(assets)))
            return 1
        print(f"  {label}: {matches[0]}")
        wanted.append(matches[0])

    for name in wanted:
        a = assets[name]
        digest = (a.get("digest") or "").removeprefix("sha256:") or None
        url = a["browser_download_url"]
        dest = os.path.join(DEST_DL, name)
        print(f"{name}  {a['size'] / 1e6:.1f} MB  sha256={digest or 'n/a'}")
        if os.path.exists(dest) and os.path.getsize(dest) == a["size"]:
            print("  already downloaded")
        else:
            dl = Downloader(url, dest, a["size"], threads=8, sha256=digest)
            if not dl.run():
                print("  download failed")
                return 1
        extract(dest, DEST_BIN)

    print()
    print("contents of bin/cuda:")
    for f in sorted(os.listdir(DEST_BIN)):
        size = os.path.getsize(os.path.join(DEST_BIN, f))
        print(f"  {f:<44} {size / 1e6:>9.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

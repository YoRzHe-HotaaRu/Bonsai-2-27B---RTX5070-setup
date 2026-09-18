#!/usr/bin/env python3
"""List every PrismML-Eng/llama.cpp release and its assets, so we can find the
release that actually carries the Windows CUDA binary zips."""

from __future__ import annotations

import json
import urllib.request

API = "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases?per_page=8"
TIMEOUT = 60


def main() -> None:
    with urllib.request.urlopen(API, timeout=TIMEOUT) as r:
        releases = json.load(r)
    for rel in releases:
        print("=" * 78)
        print(f"tag={rel['tag_name']}  published={rel.get('published_at')}  draft={rel['draft']}  prerelease={rel['prerelease']}")
        assets = rel.get("assets", [])
        if not assets:
            print("  (no assets attached)")
        for a in assets:
            print(f"  {a['name']:<60} {a['size'] / 1e6:>9.1f} MB  {a['state']}")
    print()
    print("tags:", ", ".join(r["tag_name"] for r in releases))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect the running llama-server web UI: what reasoning controls does it ship?

llama-server serves its embedded UI gzip-encoded and answers 415 unless the
client advertises gzip, so this probe always sends Accept-Encoding: gzip and
decompresses manually.

Saves the fetched bundles under webui-probe/ and reports where reasoning-related
identifiers appear, so the composer's real capabilities can be read off the code
that is actually being served rather than guessed from documentation.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8080"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "webui-probe")

KEYWORDS = [
    "Reasoning effort", "reasoning_effort", "reasoningEffort", "lightbulb",
    "thinking_budget_tokens", "reasoning_budget", "supports_reasoning_effort",
    "preserve_thinking", "enable_thinking",
]


def fetch(path_or_url: str) -> str:
    url = path_or_url if path_or_url.startswith("http") else urllib.parse.urljoin(BASE + "/", path_or_url.lstrip("/"))
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/javascript,text/css,*/*",
            "Accept-Encoding": "gzip",
            "User-Agent": "ui-probe/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            raw = r.read()
            enc = (r.headers.get("Content-Encoding") or "").lower()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:200]
        raise RuntimeError(f"HTTP {e.code} for {url}: {body}") from None
    if "gzip" in enc or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)

    for _ in range(60):
        try:
            fetch("/health")
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    else:
        print("server did not come up")
        return 1

    html = fetch("/")
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"index.html: {len(html)} bytes")
    print()

    assets = re.findall(r'(?:src|href)="([^"]+\.(?:js|css))"', html)
    print("bundles:", assets)
    print()

    for a in assets:
        try:
            body = fetch(a)
        except Exception as e:  # noqa: BLE001
            print(f"{a}: ERROR {e}")
            continue
        name = a.split("/")[-1]
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(body)
        print(f"--- {a}  ({len(body)} bytes) -> {name} ---")
        for kw in KEYWORDS:
            hits = [m.start() for m in re.finditer(re.escape(kw), body, re.I)]
            if not hits:
                continue
            print(f"  {kw!r}: {len(hits)} hit(s)")
            for h in hits[:3]:
                s = max(0, h - 150)
                snippet = body[s:h + 200].replace("\n", " ")
                print(f"      ...{snippet}...")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

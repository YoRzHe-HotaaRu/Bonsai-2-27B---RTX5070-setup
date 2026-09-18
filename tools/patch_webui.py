#!/usr/bin/env python3
"""Build a patched copy of the llama-server web UI with the reasoning selector
made unconditional, ready to hand to llama-server via --path.

Why: the shipped UI renders its reasoning-effort dropdown (lightbulb + current
level, choices Default / Off / Low 512 / Medium 2048 / High 8192 / Max) only
when `modelSupportsThinking` is true, a value the UI derives by regex-scanning
the model's chat template. That single gate is the only thing that can hide the
control, so this removes it. Everything else about the control is untouched: it
still writes per-conversation reasoningEffort / thinkingEnabled state and still
sends enable_thinking + thinking_budget_tokens to the server.

Requires a running server to pull the assets from (see tools/probe_webui.py).

Output: webui/ laid out exactly as the app expects.
"""

from __future__ import annotations

import gzip
import os
import re
import shutil
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "webui-probe")
DEST = os.path.join(ROOT, "webui")
BASE = "http://127.0.0.1:8080"

EXTRA_ASSETS = ["favicon.ico", "favicon.svg", "apple-touch-icon-180x180.png", "manifest.webmanifest"]

GATE = re.compile(r"\b[A-Za-z_$][\w$]*\.modelSupportsThinking\s*&&\s*")


def fetch(path: str) -> bytes:
    req = urllib.request.Request(
        BASE + path,
        headers={"Accept-Encoding": "gzip", "Accept": "*/*", "User-Agent": "ui-patch/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        if "gzip" in (r.headers.get("Content-Encoding") or "").lower() or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
    return raw


def main() -> int:
    index_path = os.path.join(SRC, "index.html")
    if not os.path.exists(index_path):
        print("webui-probe/index.html missing; run tools/probe_webui.py first")
        return 1

    with open(index_path, encoding="utf-8") as f:
        html = f.read()

    bundle = None
    for attr in ("href", "src"):
        m = re.search(rf'{attr}="([^"]+\.js)"', html)
        if m:
            bundle = m.group(1).split("/")[-1]
            break
    css_m = re.search(r'href="([^"]+\.css)"', html)
    css = css_m.group(1).split("/")[-1] if css_m else None
    print(f"index.html -> bundle={bundle} css={css}")
    if not bundle:
        print("no JS bundle referenced by index.html")
        return 1

    if os.path.isdir(DEST):
        shutil.rmtree(DEST)
    os.makedirs(os.path.join(DEST, "_app", "immutable", "assets"), exist_ok=True)

    with open(os.path.join(DEST, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    if css:
        with open(os.path.join(SRC, css), encoding="utf-8") as f:
            css_body = f.read()
        with open(os.path.join(DEST, "_app", "immutable", "assets", css), "w", encoding="utf-8") as f:
            f.write(css_body)
        print(f"copied {css}")

    with open(os.path.join(SRC, bundle), encoding="utf-8") as f:
        js = f.read()
    before = len(GATE.findall(js))
    js, n = GATE.subn("", js)
    after = len(GATE.findall(js))
    print(f"gate occurrences: {before} -> {after} ({n} removed)")

    out_bundle = os.path.join(DEST, "_app", "immutable", bundle)
    with open(out_bundle, "w", encoding="utf-8") as f:
        f.write(js)
    print(f"wrote {out_bundle} ({len(js)} chars)")

    for extra in EXTRA_ASSETS:
        try:
            data = fetch("/" + extra)
        except Exception as e:  # noqa: BLE001
            print(f"  {extra}: not fetched ({type(e).__name__})")
            continue
        rel = extra.split("/")[-1]
        with open(os.path.join(DEST, rel), "wb") as f:
            f.write(data)
        print(f"  {extra}: {len(data)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Locate the composer's reasoning-effort control inside the served UI bundle.

The bundle is minified Svelte. Rather than guess, this extracts the windows
around the identifiers that decide whether the control renders, and writes them
to webui-probe/ for reading.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "webui-probe", "bundle.oP6qBMGT.js")
OUT = os.path.join(ROOT, "webui-probe")

TOKENS = [
    "HI(",                      # the template-scanning capability check
    "reasoningEffortDefault",
    "Reasoning effort",
    "reasoning_control",
    "lightbulb-off",
]


def main() -> int:
    with open(BUNDLE, encoding="utf-8", errors="replace") as f:
        body = f.read()
    print(f"bundle: {len(body)} chars\n")

    for tok in TOKENS:
        hits = [m.start() for m in re.finditer(re.escape(tok), body)]
        print(f"=== {tok!r}: {len(hits)} hit(s) ===")
        for i, h in enumerate(hits[:6]):
            s = max(0, h - 260)
            snippet = body[s:h + 300].replace("\n", " ")
            print(f"  [{i}] ...{snippet}...")
        print()

    # Full window around the composer control, for close reading.
    for tok, name in (("Reasoning effort", "window-reasoning-effort.txt"),
                      ("HI(", "window-capability-check.txt")):
        h = body.find(tok)
        if h < 0:
            continue
        s = max(0, h - 12000)
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(body[s:h + 12000])
        print(f"wrote {name} (window {s}..{h + 12000})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

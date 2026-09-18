#!/usr/bin/env python3
"""Find where the shipped reasoning-effort control is mounted in the UI."""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "webui-probe", "bundle.oP6qBMGT.js")

TOKENS = [
    "isSelected(",          # the dropdown menu that lists the levels
    "getReasoningEffort",
    "setReasoningEffort",
    "K9(",                  # the lightbulb icon function
    "pSt",                  # the levels array Default/Off/Low/Medium/High/Max
    "modelSupportsThinking",
]


def main() -> int:
    with open(BUNDLE, encoding="utf-8", errors="replace") as f:
        body = f.read()

    for tok in TOKENS:
        hits = [m.start() for m in re.finditer(re.escape(tok), body)]
        print(f"=== {tok!r}: {len(hits)} hit(s) ===")
        for h in hits[:5]:
            s = max(0, h - 300)
            print("   ..." + body[s:h + 340].replace("\n", " ") + "...")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

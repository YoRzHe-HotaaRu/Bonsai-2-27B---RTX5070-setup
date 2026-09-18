#!/usr/bin/env python3
"""Run the web UI's own reasoning-detection logic against this server's template.

The served UI decides whether to render the reasoning-effort picker with:

    supportsThinking = HI(props.chat_template)

where HI() checks, in order:
  1. each of ["enable_thinking", "reasoning_effort", "thinking_budget"] appearing
     inside a {{...}} or {%...%} tag,
  2. four "if ... thinking/enable" tag patterns,
  3. a list of literal reasoning-tag pairs (<think>...</think> etc).

This script replays exactly those rules against /props so we know whether the
control is being hidden by the detection logic or by something else.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import urllib.request

BASE = "http://127.0.0.1:8080"

KEYWORD_TOKENS = ["enable_thinking", "reasoning_effort", "thinking_budget"]
TAG_PATTERNS = [
    r"\{%-?\s*if\s+\(?\s*\w*enable[\s_]+\w*(thinking|think|reasoning)",
    r"\{%-?\s*if\s+\w*(thinking|reasoning)\s*(is not|==|!=)",
    r"\{%-?\s*if\s+not\s+\w*enable",
    r"\{%-?\s*if\s+ns\.enable_thinking",
]
LITERAL_PAIRS = [
    ("<think>", "</think>"),
    ("<|channel>thought", "<|channel|>"),
    ("<|think|>", "</|think|>"),
    ("<seed:think|>", "</seed:think|>"),
    ("<think></think>", None),
]


def get(path: str) -> str:
    req = urllib.request.Request(BASE + path, headers={"Accept-Encoding": "gzip", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=90) as r:
        raw = r.read()
        if "gzip" in (r.headers.get("Content-Encoding") or "").lower() or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def hi(template: str) -> tuple[bool, list[str]]:
    """Replay of the UI's HI() function. Returns (supported, reasons)."""
    reasons: list[str] = []
    if not template:
        return False, ["template is empty"]
    for kw in KEYWORD_TOKENS:
        pattern = rf"(\{{{{[^{{}}]*\b{kw}\b[^{{}}]*\}}}}|\{{%[^{{}}]*\b{kw}\b[^{{}}]*%\}})"
        if re.search(pattern, template, re.I):
            reasons.append(f"token {kw!r} found inside a template tag")
    for p in TAG_PATTERNS:
        if re.search(p, template, re.I):
            reasons.append(f"tag pattern matched: {p}")
    for open_tag, close_tag in LITERAL_PAIRS:
        if open_tag in template and (close_tag is None or close_tag in template):
            reasons.append(f"literal reasoning tags present: {open_tag}")
    return bool(reasons), reasons


def main() -> int:
    props = json.loads(get("/props"))
    template = props.get("chat_template") or ""
    print(f"chat_template length: {len(template)} chars")
    print(f"chat_template_caps.supports_reasoning_effort: "
          f"{props.get('chat_template_caps', {}).get('supports_reasoning_effort')}")
    print(f"default reasoning_format: {props.get('default_generation_settings', {}).get('params', {}).get('reasoning_format')}")
    print()

    supported, reasons = hi(template)
    print(f"HI(template) => {supported}")
    for r in reasons:
        print(f"   - {r}")

    print()
    print("=== effort -> budget map (oIe) and related constants from the bundle ===")
    sys.path.insert(0, __file__.rsplit("\\", 1)[0])
    import os
    bundle = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "webui-probe", "bundle.oP6qBMGT.js")
    with open(bundle, encoding="utf-8", errors="replace") as f:
        body = f.read()
    for tok in ("oIe=", "supportsThinking", "reasoningEffortDefault"):
        hits = [m.start() for m in re.finditer(re.escape(tok), body)]
        print(f"\n{tok!r}: {len(hits)} hit(s)")
        for h in hits[:4]:
            print("   ..." + body[max(0, h - 200):h + 260].replace("\n", " ") + "...")
    return 0


if __name__ == "__main__":
    sys.exit(main())

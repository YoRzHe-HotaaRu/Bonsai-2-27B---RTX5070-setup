#!/usr/bin/env python3
"""Show how the client-side output cap interacts with thinking tokens.

Sends the same question at several max_tokens values and reports, for each:
the finish reason, how many tokens were generated, and how the output split
between the reasoning channel and the visible answer. The point is to show that
a thinking model spends the output budget on reasoning first, so a client cap
smaller than the reasoning length yields an empty or truncated reply even though
the server itself imposed no limit.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

QUESTION = "Explain in three sentences why ternary weights save memory."


def ask(port: int, max_tokens: int, budget: int | None = None) -> dict:
    body: dict = {
        "model": "bonsai-2-27b",
        "messages": [{"role": "user", "content": QUESTION}],
        "max_tokens": max_tokens,
    }
    if budget is not None:
        body["thinking_budget_tokens"] = budget
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--caps", default="256,1024,4096")
    args = ap.parse_args()

    print(f"{'max_tokens':>10} | {'finish':<8} | {'tokens':>6} | {'reasoning':>9} | {'answer':>6} | verdict")
    print("-" * 78)
    for cap in (int(c) for c in args.caps.split(",")):
        d = ask(args.port, cap)
        if "error" in d:
            print(f"{cap:>10} | ERROR: {str(d['error'])[:50]}")
            continue
        ch = d["choices"][0]
        msg = ch["message"]
        rc = msg.get("reasoning_content") or ""
        content = (msg.get("content") or "").strip()
        t = d.get("timings", {})
        verdict = "complete" if ch.get("finish_reason") == "stop" else "TRUNCATED"
        if not content:
            verdict = "NO ANSWER AT ALL (thinking ate the whole budget)"
        print(f"{cap:>10} | {ch.get('finish_reason'):<8} | {t.get('predicted_n'):>6} | "
              f"{len(rc):>9} | {len(content):>6} | {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

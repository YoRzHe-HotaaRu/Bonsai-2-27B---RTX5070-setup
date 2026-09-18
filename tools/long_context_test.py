#!/usr/bin/env python3
"""End-to-end deep-context check against a running llama-server.

Loads a long prompt with a needle buried early, asks for the needle, and reports
prefill/decode throughput plus whether the answer was correct. Loading a big
context is not the same as being able to use one: if the KV cache has spilled to
system memory, prefill throughput collapses and retrieval fails.

Usage:
    python tools/long_context_test.py [--target-tokens 16000] [--port 8080]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog. "
NEEDLE = "IMPORTANT: The vault code is 7391-KX. Remember it."
QUESTION = "What is the vault code? Reply with only the code."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=16000)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    # FILLER is roughly 10 tokens, so repeats ~= target/10.
    repeats = max(1, args.target_tokens // 10)
    parts = [FILLER * (repeats // 3), NEEDLE, FILLER * (repeats - repeats // 3)]
    prompt = "".join(parts) + "\n\n" + QUESTION

    body = {
        "model": "bonsai-2-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_control": True,
    }

    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    print(f"prompt: ~{len(prompt)} chars, needle at ~{100 * (repeats // 3) // max(repeats, 1)}% depth")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    wall = time.time() - t0

    if "choices" not in d:
        print("ERROR:", str(d)[:300])
        return 1

    msg = d["choices"][0]["message"]
    answer = (msg.get("content") or "").strip()
    t = d.get("timings", {})
    print(f"prompt tokens      : {t.get('prompt_n')}")
    print(f"prefill            : {round(t.get('prompt_per_second', 0), 1)} t/s")
    print(f"decode             : {round(t.get('predicted_per_second', 0), 1)} t/s")
    print(f"wall clock         : {wall:.1f}s")
    print(f"answer             : {answer[:60]!r}")
    print(f"needle retrieved   : {'YES' if '7391' in answer else 'NO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Logging proxy that sits between a client and llama-server.

Purpose: find out what a client actually sends. UIs and agent harnesses often
impose an output cap that is either hidden or hardcoded, and llama-server's own
log does not record the incoming ``max_tokens``. Point the client at this proxy
instead and every request is printed with the fields that matter.

Usage:
    python tools/log_proxy.py --port 8081 [--upstream http://127.0.0.1:8080]
                                     [--log proxy-log.txt]

Then configure the client with base URL http://127.0.0.1:8081/v1 and reproduce.
Streaming (SSE) responses are passed through unbuffered, so the client behaves
exactly as it would against the server directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://127.0.0.1:8080"
LOGFILE: str | None = None

INTERESTING = (
    "model", "max_tokens", "max_completion_tokens", "n_predict", "stream",
    "temperature", "top_p", "top_k", "thinking_budget_tokens", "reasoning_budget",
    "reasoning_control", "reasoning_format", "chat_template_kwargs", "tool_choice",
)


def log(line: str) -> None:
    print(line, flush=True)
    if LOGFILE:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence per-request stderr noise
        pass

    def do_GET(self) -> None:
        self.forward("GET")

    def do_POST(self) -> None:
        self.forward("POST")

    def forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        stamp = time.strftime("%H:%M:%S")
        if body and "chat/completions" in self.path:
            try:
                j = json.loads(body)
                fields = {k: j[k] for k in INTERESTING if k in j}
                fields["messages"] = len(j.get("messages") or [])
                if j.get("tools"):
                    fields["tools"] = len(j["tools"])
                log(f"REQ  {stamp} {self.path} | max_tokens={j.get('max_tokens', '<ABSENT -> server default -1>')} "
                    f"stream={j.get('stream', False)} | {json.dumps(fields, ensure_ascii=False)[:400]}")
            except Exception:  # noqa: BLE001
                log(f"REQ  {stamp} {self.path} | unparseable body, {len(body)} bytes")
        else:
            log(f"REQ  {stamp} {method} {self.path} | {len(body)} bytes")

        req = urllib.request.Request(
            UPSTREAM + self.path,
            data=body if body else None,
            method=method,
        )
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "accept-encoding", "connection"):
                continue
            req.add_header(k, v)

        try:
            upstream = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            upstream = e

        status = getattr(upstream, "status", None) or getattr(upstream, "code", 200)
        ctype = upstream.headers.get("Content-Type", "application/json")
        clen = upstream.headers.get("Content-Length")

        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if clen:
            self.send_header("Content-Length", clen)
        else:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        collected = bytearray()
        try:
            while True:
                chunk = upstream.read(2048)
                if not chunk:
                    break
                collected += chunk
                if clen:
                    self.wfile.write(chunk)
                else:
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log(f"RESP {stamp} | client disconnected early")
            return
        finally:
            upstream.close()

        if not clen:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        text = collected.decode("utf-8", "replace")
        note = ""
        if "finish_reason" in text:
            idx = text.rfind('"finish_reason"')
            note = text[idx:idx + 40].replace("\n", " ")
        if '"usage"' in text or '"timings"' in text:
            try:
                obj = json.loads(text)
                t = obj.get("timings") or obj.get("usage") or {}
                note += f" | tokens={t.get('predicted_n') or t.get('completion_tokens')}"
            except Exception:  # noqa: BLE001
                pass
        log(f"RESP {stamp} | {len(collected)} bytes | {note}")


def main() -> int:
    global UPSTREAM, LOGFILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--upstream", default="http://127.0.0.1:8080")
    ap.add_argument("--log", default=None, help="also append every line to this file")
    args = ap.parse_args()

    UPSTREAM = args.upstream.rstrip("/")
    LOGFILE = args.log

    log(f"logging proxy on http://127.0.0.1:{args.port}  ->  {UPSTREAM}")
    log("point the client at this port instead of the server; Ctrl+C to stop")
    if LOGFILE:
        log(f"appending to {LOGFILE}")
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

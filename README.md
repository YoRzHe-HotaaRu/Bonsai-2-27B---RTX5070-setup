# Bonsai 2 27B on an RTX 5070 (12 GB) — which pack, and how to run it

Status: **environment provisioned and verified on this machine.**
GPU: NVIDIA GeForce RTX 5070, 12226 MiB, driver 591.86, compute capability 12.0 (Blackwell `sm_120`).

---

## 1. The short answer

| Question | Answer |
| --- | --- |
| Which file do I download? | **`Ternary-Bonsai-2-27B-PQ2_0.gguf`** — 7.21 GB, the 2.13 bpw pack. Measured fastest on this card on both axes: 2.4x the prefill and 16% more decode than `PTQ1_0` (§8). |
| Do I also want the vision pack? | **`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`** — 0.63 GB, only needed for image input. Verified working (§8). |
| Which runtime? | **PrismML's llama.cpp fork, Windows x64 CUDA 13.3 build** (`prism-b10683-d8f26ee`). |
| Do I need the CUDA toolkit? | **No.** The prebuilt binary loads natively on `sm_120` — verified below. |
| Do I need to build from source? | **No.** |
| Can I use Ollama / LM Studio / stock llama.cpp? | **No** — see §3. |

Both `PQ2_0` and `PTQ1_0` are present locally (12.25 GB of weights total), so the
pack choice was settled by measurement on this card rather than by the vendor
table alone. `PTQ1_0` earns its place only as the long-context option: it costs
1.1 GiB less VRAM. Delete it if you never need more than 32K context.

---

## 2. What the model actually is

Bonsai 2 27B (Prism ML, Apache 2.0) is a **ternary** build of `Qwen/Qwen3.8-27B`:
weights from `{−1, 0, +1}` with one FP16 scale per group of 128, stored in a
blockwise-Hadamard-rotated basis. 27.36B parameters total; 26.2M of them
(the linear-attention recurrent state and normalization weights) stay above
ternary, for a true 1.72 bits/weight. Architecture is unchanged from the base
model: hybrid attention (~75% linear / ~25% full), 262,144-token context, plus a
27-block vision tower shipped separately.

Vendor figures: 98.2% of the FP16 model's 14-benchmark average (84.78 vs 86.32)
at ~1/9th the size; math 96.57 vs 97.06; coding 89.42 vs 89.07; BFCL v3 tool
calling 74.92 vs 76.74.

### Files in `prism-ml/Ternary-Bonsai-2-27B-gguf`

| File | Size | Bits/weight | Verdict |
| --- | --- | --- | --- |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7.21 GB | 2.13 | **Primary pick.** One trit per 2-bit slot: cheaper to unpack, faster prefill everywhere, and the vendor's recommended pack for Blackwell decode. |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5.95 GB | 1.75 | Dense trits: 17% less weight traffic per step, more arithmetic per step. The vendor says it wins where memory is the binding constraint (Ada, L4); **measured here it loses on both axes to `PQ2_0`** (§8), so its only value on this card is the 1.1 GiB of VRAM it saves for longer contexts. |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 0.63 GB | — | Optional vision tower. Loaded only when images are sent (0.59 GiB VRAM, or keep it in RAM). |
| `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` | 0.93 GB | — | Reference vision tower. Larger, no quality benefit worth the VRAM here. |
| `Ternary-Bonsai-2-27B-F16.gguf` | 53.8 GB | 16.0 | **Do not download.** Needs ~54 GB of VRAM. It exists as the accuracy reference. |
| `prism-ml/Ternary-Bonsai-2-27B-gguf-dev`: `...-Q2_0-prism-fork-required.gguf` | 7.6 GB | 2.25 | **Do not use.** Testing artifact for the upstreaming work. Worse: upstream llama.cpp loads it without any warning and emits gibberish, because `Q2_0` and `qwen35` are both already known upstream. |

---

## 3. Why the runtime must be PrismML's fork

Every Bonsai 2 band stores weights in a rotated basis and needs a matching
activation (FWHT) transform at runtime. That is not upstream yet, and the two
shipped packings use Prism-private tensor types:

```
141  or  PQ2_0   :  2.13 bpw quantization (group 128, Prism)
143  or  PTQ1_0  :  1.75 bpw ternarization (group 128, Prism)   <- reported by this build
```

Those ids sit past upstream's `GGML_TYPE_COUNT`, so stock llama.cpp **refuses**
the files outright (a safe failure). The dangerous case is the dev repo's `Q2_0`
file, which upstream happily loads and turns into noise.

Consequently: **Ollama, LM Studio, Jan, Docker Model Runner and vLLM cannot run
these GGUFs.** They bundle their own upstream `llama.cpp`; the "run with Ollama /
LM Studio" snippets on the Hugging Face model page do not apply to Bonsai 2.
Any wrapper that lets you point at a custom `llama-server.exe` binary is fine.

---

## 4. Pipeline as provisioned here

1. **Binaries** — from release `prism-b10683-d8f26ee` (the tag the official
   `Bonsai-demo` repository pins as its known-good build):
   - `llama-prism-b10683-d8f26ee-bin-win-cuda-13.3-x64.zip` (144.9 MB)
   - `cudart-llama-bin-win-cuda-13.3-x64.zip` (391.0 MB — CUDA 13.3 runtime DLLs)
   - Both verified against the SHA-256 digests published by the GitHub API.
   - Extracted to `bin\cuda\` (67 files: `llama-server.exe`, `llama-cli.exe`,
     `llama-bench.exe`, `ggml-cuda.dll` at 148 MB, `mtmd.dll` for vision, ...).
2. **Weights** — pulled from Hugging Face with SHA-256 verification into
   `models\bonsai2-27B\`.
3. **Verification** — the CUDA backend enumerates the card:

```
$ bin\cuda\llama-cli.exe --list-devices
Available devices:
  CUDA0: NVIDIA GeForce RTX 5070 (12226 MiB, 11017 MiB free)
```

Why CUDA 13.3 and not the 12.4 build: `sm_120` (Blackwell consumer) support
started with CUDA 12.8, so a CUDA 12.4 build has no native kernels for this card
and would at best fall back to PTX JIT. CUDA 13.3 targets it directly.

> Note: the newest fork tag (`prism-b10687-5d80cff`) had not finished uploading
> its binary zips when this was set up — it carried only the `cudart` packs.
> `prism-b10683-d8f26ee` is both complete and the demo's pinned release. If you
> ever bump the tag, take the `win-cuda-13.3-x64` asset and re-verify `--list-devices`.

---

## 5. VRAM budget on a 12 GB card

Measured, not estimated — `llama-cli -v` allocation lines for `PQ2_0` with
`-ngl 99 -fa on -c 32768` (65/65 layers offloaded to the GPU):

| Component | Measured |
| --- | --- |
| GPU weights (`load_tensors: CUDA0 model buffer size`) | 6539.67 MiB |
| Host-mapped embedding tensor (`CPU_Mapped`) | 322.07 MiB — system RAM, not VRAM |
| KV cache, 32K, F16 | 2048.00 MiB (32768 cells × 64 KiB; 16 full-attention layers; K 1024 + V 1024) |
| Recurrent state, linear-attention path (f32) | 149.62 MiB |
| Compute buffers | 166.02 MiB GPU + 52.02 MiB host |
| Output buffer | 0.95 MiB |
| **VRAM total, text-only** | **≈ 8.70 GiB** |
| **+ vision projector on GPU (`mmproj` Q8_0)** | **≈ 9.29 GiB** |

The card reports 12226 MiB total with 11017 MiB (10.76 GiB) free alongside the
running desktop, so the recommended configuration fits. Note the 322 MiB of
embeddings live in host memory: `pq2_0` cannot use the preferred `CUDA_Host`
buffer type, so that tensor is CPU-mapped and does not consume VRAM.

Those figures are what `llama.cpp` accounts for itself. Actual VRAM in use is
higher, because the CUDA context, cuBLAS workspaces and the vision projector are
not in its buffer list — `nvidia-smi` reports **10824 MiB of 12227 MiB while
serving 32K context with vision enabled**, against a desktop baseline of ~838 MiB
before the model loads. Budget ~10.0 GiB attributable to the server and leave
~1.4 GiB of headroom; the derived table below uses the same accounting as the
measured numbers, so read its totals as a lower bound and confirm with
`nvidia-smi` after a change.

Base cost without KV is ~6.70 GiB for `PQ2_0` and ~5.58 GiB for `PTQ1_0` (measured
GPU model buffer 6539.67 / 5395.33 MiB plus state and compute buffers); each token
costs 64 KiB of KV in F16 or about 18 KiB with the Q4_0 cache:

Real VRAM at each context size, measured with `tools\measure-context.ps1`
(which starts the server, samples `nvidia-smi` and runs a request; desktop
baseline ~600 MiB, card total 12227 MiB, `PQ2_0`, one slot, vision loaded):

| Context | KV cache | Vision | Server VRAM | Free after load |
| --- | --- | --- | --- | --- |
| 32,768 | F16 | GPU | 10037 MiB | 1567 MiB |
| 65,536 | F16 | GPU | 11208 MiB | 415 MiB |
| 65,536 | Q4_0 | GPU | 9398 MiB | 2224 MiB |
| 98,304 | Q4_0 | GPU | 10119 MiB | 1494 MiB |
| 131,072 | Q4_0 | GPU | 10869 MiB | 753 MiB |
| 131,072 | Q4_0 | projector in RAM | 10018 MiB | 1619 MiB |
| 196,608 | Q4_0 | GPU | 11241 MiB | 398 MiB |
| 262,144 | Q4_0 | GPU | 11222 MiB | 401 MiB |

`PTQ1_0` shifts every row down by about 1.1 GiB (its GPU model buffer measures
5395.33 MiB against 6539.67 MiB), at the cost of roughly half the prefill speed
(§8).

Practical guidance:

- **Recommended maximum: `-Context 98304 -Kv4`.** 96K with 1.5 GiB of headroom,
  verified at depth rather than merely loaded: a 16,041-token prompt with a
  needle at 33% depth prefilled at 1184.7 t/s (full speed, so nothing had spilled
  to system memory) and was answered correctly. Decode at that depth measured
  41.1 t/s against ~60 t/s shallow.
- **Safest long-context option: `-Context 65536 -Kv4`.** 2.2 GiB spare, enough
  that a GPU-accelerated browser cannot push the driver into paging.
- **If you specifically want 128K:** add `-VisionOnCPU`, which keeps 1.6 GiB free.
- **192K and 262K load, but do not run them.** They land at ~11.2 GiB with about
  400 MiB spare, and on WDDM an over-committed allocation pages to system memory
  instead of failing loudly, so the symptom is unexplained slowness rather than an
  error you can act on.
- **Above ~48K the Q4_0 cache wins outright.** 64K at F16 costs 11208 MiB while
  96K at Q4_0 costs 10119 MiB, so the quantized cache buys 50% more context *and*
  a gigabyte of headroom. It is a memory mechanism, not a speed one. The vendor's
  KV-CACHE.md documents an optional mean-centering bias for quality at long
  context (`--kv-mean-center`, built by the demo's `make_kv_bias.sh`).
- Default remains `-Context 32768`, single slot, projector in VRAM.

---

## 6. Commands

```powershell
# provision / repair (idempotent, resumes partial downloads)
.\scripts\setup.ps1
.\scripts\setup.ps1 -AllPacks          # also fetch PTQ1_0 for the A/B

# chat UI + OpenAI-compatible API on http://127.0.0.1:8080
.\scripts\start-server.ps1
.\scripts\start-server.ps1 -Pack PTQ1_0 -Context 65536 -Kv4
.\scripts\start-server.ps1 -VisionOnCPU -ReasoningBudget 2048

# one-shot prompt (no server)
.\scripts\run-cli.ps1 -Prompt "Explain ternary quantization in two sentences."

# throughput, both packs, markdown table -> bench-results.md
.\scripts\bench.ps1

# how much VRAM a given context actually costs (starts, measures, stops)
.\tools\measure-context.ps1 -Context 98304 -Kv4
.\tools\measure-context.ps1 -Context 131072 -Kv4 -VisionOnCPU

# deep-context check against a running server: prefill speed + needle retrieval
python tools\long_context_test.py --target-tokens 16000
```

API call:

```powershell
curl.exe http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{\"model\":\"bonsai\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'
```

---

## 7. Thinking selector in the chat UI

Bonsai 2 thinks by default, and the bundled web UI already ships a control for
it: a lightbulb plus the current level, sitting in the message box beside the
model chip. Its ladder:

| Level | Behaviour |
| --- | --- |
| Default | whatever the server was started with (now `xhigh` effort, unlimited budget) |
| Off | thinking disabled (`enable_thinking: false`) |
| Low | 512-token budget |
| Medium | 2,048 |
| High | 8,192 |
| Max | unlimited budget — the same as Default here; the effort string stays `xhigh`, since the template rejects `max` |

The choice is per conversation, the default is remembered in
`localStorage["LlamaUi.reasoningEffortDefault"]`, and each request carries
`thinking_budget_tokens` plus `reasoning_control`.

Verified end to end against this server, replaying the payloads the picker
sends for the same question ("how many r's in strawberry?"):

| Selector | `reasoning_content` | Tokens generated | Answer |
| --- | --- | --- | --- |
| Off | 0 chars | 2 | 3 (correct) |
| Medium (2048) | 382 chars | 151 | 3 (correct) |

So the budget is honoured and the thinking is routed to its own channel rather
than into the visible answer.

The UI renders that control only when it believes the model can reason, which it
decides by regex-scanning the chat template from `/props`. For this model the
scan passes — `enable_thinking` and `reasoning_effort` both appear inside
template tags, and the literal `<think>` pair is present — so a fresh browser
session should show it.

**If the composer shows only the model chip**, work down this list:

1. Open the UI in a private window first. Site state for `localhost:8080`
   (`localStorage` and cached app state) outlives server changes, and a private
   window shows what a fresh visitor gets.
2. If it is still missing, serve the patched UI, which removes that gate:

   ```powershell
   .\scripts\start-server.ps1 -PatchedUi
   ```

   `webui\` is a copy of the served UI with both `modelSupportsThinking` guards
   deleted (`tools\patch_webui.py` builds it, and the result passes
   `node --check`), handed to llama-server through `--path`. The reasoning
   dropdown then renders unconditionally with its own wiring untouched.

### Server-side equivalents

Useful without a UI, or to set the default the UI starts from:

| Flag | Effect |
| --- | --- |
| `-Reasoning on|off|auto` | `--reasoning`; `auto` (default) detects from the template |
| `-ReasoningEffort <level>` | `--reasoning-effort`: this model's template accepts only `xhigh`, `low`, `medium` |
| `-ReasoningBudget N` | `--reasoning-budget`: -1 unlimited (default), 0 immediate end, N cap |
| `-ReasoningFormat none|deepseek` | `--reasoning-format`: `deepseek` moves thoughts to `message.reasoning_content` |
| `-ReasoningPreserve` | keeps the reasoning trace in the full history, not just the last turn |

> **Do not pass `max`, `high` or `minimal` as an effort level.** The Bonsai 2
> chat template raises on anything outside its three supported values, and the
> request fails outright with
> `Jinja Exception: Unexpected reasoning effort max. Supported types are xhigh (default), medium, and low.`
> Verified: 0 tokens generated. **`xhigh` is the top level this model offers.**
> "Max" in the web UI picker is a different axis — an unlimited token *budget* —
> and that is already the default.

The defaults in `start-server.ps1` are thinking on, effort pinned to `xhigh`
(the highest the template accepts), and an unlimited thinking budget, all passed
explicitly so the launch banner states the effective configuration. A plain API
call then returns the trace in `message.reasoning_content` (measured: 320
characters) beside a clean `message.content` (`3`), so the web UI's collapsible
thinking block and ordinary API clients both work as-is.

### Pointing another client or agent harness at this server

| Field | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8080/v1` |
| API protocol | `openai-completions` (verified against `/v1/chat/completions`) |
| API key | any non-empty placeholder; the server configures none by default |
| Model id | `bonsai-2-27b` (set by `-Alias`; without it the API advertises the full `.gguf` path, which is a poor thing to store in a client config) |

Notes for agent use, all measured on this server:

- **Start the server first.** It is a local process, not a hosted endpoint; a
  client will simply get connection refused while it is down.
- **Tool calling is native and works**: `finish_reason: "tool_calls"` with valid
  JSON arguments (`get_weather {"city":"Paris"}` for a Paris weather question).
  This needs `--jinja`, which the script always passes.
- **Thinking does not pollute `message.content`.** With thinking on, the answer
  came back as plain `3` and the trace arrived separately in
  `reasoning_content`; `-ReasoningBudget N` bounds it.
- **Budget your output tokens.** Uncapped thinking is generated first, so a small
  `max_tokens` can be spent entirely on reasoning before any answer is emitted.
  Allow a few thousand, or cap the budget.
- **One slot by default.** `-np 1` means a harness that fires parallel requests
  will serialise them; `-np 2 -Context 32768 -Kv4` fits two 16K conversations in
  the same VRAM envelope.
- **Multimodal**: `/v1/models` reports capabilities `completion` and
  `multimodal`, so `image_url` content parts are accepted.
- `/v1/responses` also exists and returns a valid response object, but
  `/v1/chat/completions` is the fully exercised path.

Use `127.0.0.1` rather than `localhost` in the base URL: the server binds IPv4
loopback, and a client that resolves `localhost` to `::1` first will fail to
connect.

---

## 8. Measured on this machine

`llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3`, raw output in `bench-results.md`:

| Pack | Size | pp512 (t/s) | tg128 (t/s) |
| --- | --- | --- | --- |
| `PQ2_0` (2.13 bpw) | 6.70 GiB | **1281.55** ± 43.72 | **61.75** ± 0.13 |
| `PTQ1_0` (1.75 bpw) | 5.53 GiB | 533.70 ± 4.35 | 53.31 ± 0.11 |

**PQ2_0 wins on both axes on this card**, by 2.4x on prompt processing and 16% on
generation, despite being the larger file. That matches the vendor's guidance
("PQ2_0 … is faster at prompt processing everywhere … the faster decode on …
the Blackwell cards"): `PTQ1_0` moves 17% less weight data per step, but
unpacking dense trits costs more arithmetic than reading 2-bit slots, and
batch-1 decode on this GPU is limited by instruction throughput rather than by
the 672 GB/s of memory bandwidth. The 5070 is less bandwidth-starved relative to
its compute than a 5090, which makes the arithmetic penalty of the denser pack
show up clearly.

Practical consequence: **use `PQ2_0` as the default.** Reach for `PTQ1_0` only
when the ~1.1 GiB of VRAM it saves is what stands between you and a longer
context (64K with an F16 KV cache, or ~100K comfortably), and accept roughly
1/2 the prefill speed and ~14% slower generation for it.

End-to-end checks with the same binary, same card:

| Check | Result |
| --- | --- |
| `llama-cli --list-devices` | `CUDA0: NVIDIA GeForce RTX 5070 (12226 MiB, 11017 MiB free)` |
| Text prompt, thinking off, 32K context | correct answer, generation 57-59 t/s reported by the CLI |
| Vision: 730x515 PNG screenshot through `mmproj-Q8_0` | image described correctly (banner text, three repo entries, metadata), prompt 757 t/s |
| `scripts\start-server.ps1`, then `/health` and `/props` | `{"status":"ok"}`; `n_ctx = 32768`, `slots = 1`, build `b10683-d8f26eec7` |
| `/v1/chat/completions` | correct answer, 59.0 t/s generation, 10824 MiB VRAM in use with the desktop running |

CLI-reported prompt speeds vary a lot with prompt length (short prompts are
dominated by fixed per-request overhead), so treat the `llama-bench` row as the
authoritative throughput figure and the CLI numbers as sanity checks.

Vendor reference points for the same metrics: RTX 5090 (32 GB) 129.9 t/s `tg128`
and 3931.9 t/s `pp512` on `PQ2_0`. Against those, this machine sits at ~48% of
the 5090's decode and ~33% of its prefill — consistent with roughly 1/3 of the
SM count and 672 GB/s versus 1792 GB/s of bandwidth.

---

## 9. Operating notes

- **Sampling (from the model card).** Thinking mode: `temp 1.0`, `top_p 0.95`,
  `top_k 20`. Instruct mode: `temp 0.7`, `top_p 0.80`, `top_k 20`,
  `presence_penalty 1.5`. These are baked into the scripts and into the GGUF
  metadata (`general.sampling.*`), so a client that reads model defaults agrees.
- **It is a reasoning model.** Thinking is on by default; most of a slow answer is
  reasoning tokens, not prefill. `-ReasoningBudget 2048` (or the lightbulb picker
  in the built-in web UI) is the lever. Reasoning effort defaults to `xhigh`;
  `low` is not supported and behaves like `xhigh`.
- **Reasoning is dropped from history by default.** The server logs a hint at
  startup that the chat template supports preserving it — add `--reasoning-preserve`
  (pass it through `start-server.ps1` as an extra argument) if you want earlier
  turns' thinking kept in context.
- **`-c` is shared across slots.** `llama-server` defaults to `-np 4`, which would
  split a 32K context into 8K per conversation. The scripts pass `-np 1`; raise it
  deliberately if you want concurrent chats.
- **Prefix caching** works across turns, so follow-up questions are cheap. The
  optional DSpark speculative decoding (~0.6 GB drafter, vendor-measured 1.8-2.4x
  on CUDA for this family) disables cross-request prefix reuse and forces one
  slot — worthwhile for single-shot code/math, a regression for multi-turn chat.
- **Vision** costs one prefill per new image; subsequent questions about the same
  image hit the cache. On CUDA the scripts leave the image-token cap uncapped.
- Tool calling is native OpenAI-style `tool_calls` (`--jinja`), with full
  round-trips; the built-in web UI also has an MCP client.

---

## 10. Workspace layout

```
Bonsai-2 27B - RTX5070\
├── README.md                     this document
├── bench-results.md              generated by scripts\bench.ps1
├── test-image.png                sample used for the vision check (§8)
├── server-patched.log            log of the most recent server run
├── bin\cuda\                     PrismML fork binaries + CUDA 13.3 DLLs
├── models\bonsai2-27B\           PQ2_0, PTQ1_0, mmproj-Q8_0
├── webui\                        patched UI, served when -PatchedUi is used
├── webui-probe\                  raw copy of the built-in UI + bundles (patch source)
├── downloads\                    the release zips (safe to delete, ~536 MB)
├── scripts\
│   ├── setup.ps1                 download + verify binaries and weights
│   ├── start-server.ps1          llama-server, tuned for 12 GB
│   ├── run-cli.ps1               one-shot prompt
│   └── bench.ps1                 llama-bench over all local packs
└── tools\
    ├── download.py               stdlib parallel/resumable downloader + SHA-256
    ├── get_binaries.py           pinned fork release fetch + extract
    ├── probe_webui.py            fetch the served UI and its bundles
    ├── patch_webui.py            build webui\ (removes the thinking gate)
    ├── measure-context.ps1       real VRAM cost of a context size
    ├── long_context_test.py      deep-context prefill speed + needle retrieval
    ├── find_reasoning_ui.py      locate reasoning code inside the UI bundle
    ├── find_reasoning_ui2.py     locate where that code is mounted
    ├── test_thinking_detection.py  replay the UI's chat-template scan
    ├── net_probe.py              connectivity and asset availability probe
    ├── net_probe2.py             throughput measurement
    └── list_releases.py          enumerate fork releases and assets
```

`tools\*.py` exist because this machine's shell environment cannot use
schannel-based TLS clients (curl, git, `Invoke-WebRequest` fail with
`SEC_E_NO_CREDENTIALS`), while Python's bundled OpenSSL works. Run the scripts
from a normal PowerShell window and they are unnecessary.

## 11. Sources

- Model card: https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
- Collection: https://huggingface.co/collections/prism-ml/bonsai-2
- Demo / source of truth for flags: https://github.com/PrismML-Eng/Bonsai-demo
- Fork: https://github.com/PrismML-Eng/llama.cpp (release `prism-b10683-d8f26ee`)
- Pack formats: `MODEL-FORMATS.md` in Bonsai-demo

# Max-performance recipe: Bonsai 2 27B on an RTX 5070 (12 GB)

Every number below was measured on this machine, not estimated. The recipe is
already the default in `scripts\start-server.ps1`, so the recipe *is* the command.

---

## 1. The command

```powershell
cd "G:\DeepSeek-Platform-New\Bonsai-2 27B - RTX5070"
.\scripts\start-server.ps1
```

Open <http://127.0.0.1:8080> (or point a client at `http://127.0.0.1:8080/v1` with
model id `bonsai-2-27b`). Stop it with:

```powershell
Get-Process llama-server | Stop-Process -Force
```

## 2. What that line actually launches

```
bin\cuda\llama-server.exe
  -m  models\bonsai2-27B\Ternary-Bonsai-2-27B-PQ2_0.gguf
  --mmproj models\bonsai2-27B\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf
  --host 127.0.0.1 --port 8080
  -ngl 99 -fa on -b 8192 -ub 2048 -c 65536 -np 1
  --cache-type-k q4_0 --cache-type-v q4_0
  --kv-mean-center models\bonsai2-27B\kv-mean-center.gguf
  --temp 1.0 --top-p 0.95 --top-k 20
  --reasoning-budget 8192 --reasoning-effort medium
  --jinja --alias bonsai-2-27b
```

## 3. Why each setting

| Setting | Value | Why, measured |
| --- | --- | --- |
| Weights | `PQ2_0` (ternary g128, 2.13 bpw) | Wins on both axes against `PTQ1_0`: pp512 1284 vs 521, tg128 61.1 vs 51.3. `PTQ1_0` was deleted after measurement. |
| Context | 65536 | Fits with ~1.4 GiB spare. 96K also fits (10119 MiB) if you need the room. |
| KV cache | `q4_0` (both K and V) | At 64K an F16 cache needs 4 GiB and leaves only 415 MiB free. Cost of the quantized cache: 1.4% of decode (59.49 vs 60.35). |
| KV bias | mean-centering | The vendor's quality booster for a quantized cache. Built from `tools\kv-bias-corpus.txt`; applied automatically when the file exists. |
| Batch / ubatch | 8192 / 2048 | Best measured: pp512 1294 and tg128 61.08, against 1246 / 59.14 at the llama.cpp defaults of 2048 / 512. Later re-runs of the same configuration landed at 1252 / 59.1, so read these as ranges (see section 4). |
| GPU offload | `-ngl 99` | All 65 layers on the GPU. There is no CPU fallback path that stays fast. |
| Flash attention | `-fa on` | Required for a quantized KV cache, and faster than the alternative. |
| Slots | `-np 1` | `-c` is the total pool shared across slots. One slot means one conversation gets all 64K. |
| Thinking budget | 8192 | The web UI picker's "High" preset. `max`, `high` and `minimal` are **invalid effort levels** for this template and fail the request outright. |
| Thinking effort | `medium` | The vendor's shorter/faster setting, chosen for interactive use. `xhigh` is the top level this template accepts when a task deserves maximum deliberation; `high` and `max` are **invalid** and fail every request. |
| Sampling | temp 1.0 / top-p 0.95 / top-k 20 | The model card's thinking-mode defaults, also carried in the GGUF metadata. |
| Model id | `--alias bonsai-2-27b` | Without it the API advertises the full `.gguf` path as the model id. |
| Tool calling | `--jinja` | Native OpenAI `tool_calls`, verified round-trip. |

## 4. What to expect

| Measurement | Result |
| --- | --- |
| `pp512` (prefill, bench) | 1252 to 1294 t/s across runs |
| `tg128` (generation, bench) | 59.1 to 61.1 t/s across runs |
| Decode on a real request, thinking on | 57.6 t/s |
| Prefill, 16K-token prompt | 1185 t/s, needle retrieved correctly |
| Cold start | 25 to 40 seconds to read 6.7 GB of weights |
| VRAM in use | 10754 MiB of 12227, with the desktop running (~1.4 GiB spare) |
| Context actually served | 65536 tokens, 1 slot |

Treat those as ranges, not constants: repeated runs of the identical
configuration varied by about 3% on this machine, which is normal for a desktop
with a compositor, a browser and whatever else is resident. Compare
configurations using the same session, not numbers from different days.

**The ceiling, so you do not chase it.** One token requires streaming 6.70 GiB of
weights. At 672 GB/s that is a hard limit of about 93 t/s even at 100% of peak
bandwidth, and ternary kernels do not reach peak. 60 to 61 t/s is the practical
maximum for this model on this card. Speculative decoding, the usual trick for
going faster, is not available for Bonsai 2: the vendor ships no drafter for it.

## 5. Optional extras

```powershell
# the reasoning-effort selector in the web UI, forced to render
.\scripts\start-server.ps1 -PatchedUi

# more context: 96K keeps 1.5 GiB spare; 128K needs the projector in RAM
.\scripts\start-server.ps1 -Context 98304
.\scripts\start-server.ps1 -Context 131072 -VisionOnCPU

# the last 1.4% of decode: F16 KV cache, only viable at 32K or less
.\scripts\start-server.ps1 -F16Kv -Context 32768

# shorter thinking for interactive work: the picker's "Medium" budget and the
# brief-effort setting
.\scripts\start-server.ps1 -ReasoningBudget 2048 -ReasoningEffort medium

# unlimited thinking (the picker's "Max"), still at the top effort level
.\scripts\start-server.ps1 -ReasoningBudget -1

# skip the KV bias
.\scripts\start-server.ps1 -NoKvBias
```

If 70 to 80 t/s is a hard requirement rather than a preference, the only lever is
a smaller model. The previous-generation 1-bit `Bonsai-27B-Q1_0` (already
downloaded, 3.53 GiB) measured **79.19 t/s** bench and 76.5 t/s through the API,
at a visible cost in reliability. Command in `README.md` section 8.

## 6. Truncated replies ("output token limit reached")

A client that reports a cut-off reply is hitting **its own `max_tokens`**, and a
reasoning model makes that far more likely: thinking tokens are generated before
the answer and count against the same cap. Thinking length varies per question
(768 to 2566 characters for the *same* question across runs here), so a fixed
small cap is hit unpredictably.

Measured on this server, same question, three server configurations:

**Thinking on, 8192 budget (the shipped default)**

| Client cap | finish | tokens | reasoning | answer | Result |
| --- | --- | --- | --- | --- | --- |
| 256 | `length` | 256 | 768 chars | 355 chars | truncated mid-answer |
| 1024 | `stop` | 508 | 1562 chars | 401 chars | complete |
| 4096 | `stop` | 400 | 973 chars | 578 chars | complete |

**Thinking on, `-ReasoningBudget 1024`**

| Client cap | finish | tokens | reasoning | answer | Result |
| --- | --- | --- | --- | --- | --- |
| 256 | `length` | 256 | 1025 chars | **0 chars** | no answer at all |
| 512 | `length` | 512 | 1946 chars | **0 chars** | no answer at all |
| 1024 | `stop` | 344 | 927 chars | 537 chars | complete |
| 4096 | `stop` | 639 | 1981 chars | 502 chars | complete |

**Thinking off (`-Reasoning off`)**

| Client cap | finish | tokens | reasoning | answer | Result |
| --- | --- | --- | --- | --- | --- |
| 256 | `stop` | 94 | 0 | 475 chars | complete |
| 1024 | `stop` | 113 | 0 | 579 chars | complete |

The rule that falls out of it: **the client cap must exceed the thinking budget
plus the answer length.** When the cap is at or below the thinking length, the
reply can contain no answer at all, which is what a "reply was cut off" message
usually is.

Fixes, in order of preference:

1. **Raise the client's max output tokens** to 8192 or higher. This is the real
   fix; everything below is a workaround for a client you cannot configure.
2. **Bound thinking server-side** so total output is predictable:
   `.\scripts\start-server.ps1 -ReasoningBudget 2048`, and keep the client cap at
   4096 or more. A budget is graceful: when it is reached, the server forces the
   end of thinking so the model still answers. A client cap is not: it just stops
   the stream.
3. **Turn thinking off for that client**: `.\scripts\start-server.ps1 -Reasoning off`.
   Answers complete even under a 256-token cap, and the same question returned in
   1.84 s instead of several seconds, because there is no reasoning to generate
   first. Best choice for tight agent loops.

Diagnose which side is capping you: read `finish_reason` in the API response
(`length` means a cap was hit) and compare `timings.predicted_n` with your
configured cap. If they are equal, the client capped it. The server console also
prints per-request eval stats, so a suspiciously round token count every time is
the client's limit.

`python tools\truncation_test.py --caps 256,1024,4096` reproduces all of the
above against whatever server is running.

## 7. Rebuilding the KV bias

Do this after changing the model file or the calibration corpus:

```powershell
$env:LLAMA_ATTN_ROT_DISABLE = "1"
.\bin\cuda\llama-kv-mean-center.exe `
  -m .\models\bonsai2-27B\Ternary-Bonsai-2-27B-PQ2_0.gguf `
  -f .\tools\kv-bias-corpus.txt `
  -o .\models\bonsai2-27B\kv-mean-center.gguf -c 512
```

The bias was calibrated with attention K-rotation disabled, and inference must
match or the loader rejects it by design. `start-server.ps1` sets that flag for
the server process only, and restores it afterwards. A 5 KB corpus is enough; if
your work has a distinctive register, add a page of it to the corpus and rebuild.

## 8. Do not

- Do not run these weights on stock `llama.cpp`, Ollama, LM Studio or vLLM. They
  cannot read the `PQ2_0` / `PTQ1_0` tensor types; upstream loads the dev-repo
  `Q2_0` file and silently emits gibberish.
- Do not pass `-ReasoningEffort max`, `high` or `minimal`. Jinja exception, zero
  tokens generated.
- Do not exceed `-Context 131072`. Beyond that the allocation stops fitting and
  Windows starts paging instead of failing, which looks like unexplained
  slowness rather than an error.
- Do not run two servers on the same port. The second one loads the model, holds
  VRAM, and never receives a request. The launcher now refuses to start instead.
- Do not judge speed from the web UI without accounting for thinking: most of a
  slow answer is reasoning tokens. Cap the budget before blaming the hardware.

## 9. Verify it yourself

```powershell
# effective context, slots, build
curl.exe -s http://127.0.0.1:8080/props | python -c "import sys,json;d=json.load(sys.stdin);print(d['default_generation_settings']['n_ctx'], d.get('total_slots'))"

# throughput, both directions
.\scripts\bench.ps1

# real VRAM cost of a context size (starts, measures, stops)
.\tools\measure-context.ps1 -Context 65536

# is a long context actually usable at depth? prefill speed + needle retrieval
python tools\long_context_test.py --target-tokens 16000
```

## 10. Cheat sheet

| Action | Command |
| --- | --- |
| Start (recipe) | `.\scripts\start-server.ps1` |
| Start with the effort picker in the UI | `.\scripts\start-server.ps1 -PatchedUi` |
| Start deeper | `.\scripts\start-server.ps1 -Context 98304` |
| Stop | `Get-Process llama-server \| Stop-Process -Force` |
| Open the chat UI | <http://127.0.0.1:8080> |
| API endpoint | `http://127.0.0.1:8080/v1/chat/completions` |
| Model id for clients | `bonsai-2-27b` |
| One-shot prompt, no server | `.\scripts\run-cli.ps1 -Prompt "..."` |
| Re-provision from scratch | `.\scripts\setup.ps1` |

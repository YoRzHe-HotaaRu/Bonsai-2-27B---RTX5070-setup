<#
.SYNOPSIS
    Start Bonsai 2 27B (ternary) as an OpenAI-compatible llama.cpp server.

.DESCRIPTION
    Uses the PrismML llama.cpp fork binaries in bin\cuda. Stock llama.cpp cannot
    read the PQ2_0 / PTQ1_0 tensor types; see README.md before swapping the binary.

    Defaults are the measured best configuration for a 12 GB RTX 5070 (see
    RECIPE.md): PQ2_0 weights, 64K context with a Q4_0 KV cache, all layers on
    GPU, flash attention on, batch 8192 / ubatch 2048, single slot, thinking on
    at medium effort capped to 8192 tokens.

.EXAMPLE
    .\scripts\start-server.ps1
    .\scripts\start-server.ps1 -Context 98304 -VisionOnCPU
    .\scripts\start-server.ps1 -F16Kv -Context 32768
    .\scripts\start-server.ps1 -ReasoningBudget 2048 -PatchedUi
#>
[CmdletBinding()]
param(
    [ValidateSet('PQ2_0', 'PTQ1_0')]
    [string]$Pack = 'PQ2_0',

    [int]$Context = 65536,
    [int]$Port = 8080,
    [string]$HostAddress = '127.0.0.1',
    [int]$Parallel = 1,

    # Measured on this card: 8192/2048 gave pp512 1294 t/s and tg128 61.08 t/s,
    # against 1246/59.14 at the llama.cpp defaults of 2048/512. Larger ubatches
    # cost compute-buffer VRAM, so this is verified to fit at 64K in RECIPE.md.
    [int]$BatchSize = 8192,
    [int]$UBatchSize = 2048,

    # Model id advertised through the API. Without it, llama-server reports the
    # full path to the .gguf, which is a poor model id for a client to store.
    [string]$Alias = 'bonsai-2-27b',

    [switch]$NoVision,
    [switch]$VisionOnCPU,

    # Q4_0 KV cache is on by default: at 64K context the F16 cache needs 4 GiB and
    # leaves only ~400 MiB of VRAM free. Measured cost of Q4_0 against F16 is
    # about 1% of decode (59.49 vs 60.35 t/s at depth 0), so it is nearly free.
    # Use -F16Kv to force F16, which is sensible only at 32K context or less.
    [switch]$Kv4,
    [switch]$F16Kv,

    # Skip the K-cache mean-centering bias even when the file is present.
    [switch]$NoKvBias,

    # 8192 is the web UI picker's "High" preset. 'high' is NOT a valid effort
    # level for this model: passing it raises a Jinja exception and fails the
    # request, which is why the effort stays on a supported value.
    [int]$ReasoningBudget = 8192,
    [switch]$PatchedUi,

    # Thinking controls. Defaults reproduce the vendor's tested configuration:
    # thinking on, template default effort, reasoning left unparsed in content.
    [ValidateSet('on', 'off', 'auto')]
    [string]$Reasoning = 'auto',

    # This model's chat template accepts only xhigh, medium and low. Anything
    # else raises a Jinja exception and fails the request outright: verified
    # with both 'max' and 'high', each producing no output at all. medium is the
    # vendor's shorter/faster setting and the default here; xhigh is the maximum
    # when a task deserves the deliberation.
    [ValidateSet('default', 'low', 'medium', 'xhigh')]
    [string]$ReasoningEffort = 'medium',

    # 'none' (default) leaves thoughts inline in message.content; 'deepseek'
    # moves them to message.reasoning_content for API clients. The built-in web
    # UI asks for parsing per request, so it is unaffected either way.
    [ValidateSet('none', 'deepseek')]
    [string]$ReasoningFormat = 'none',

    [switch]$ReasoningPreserve,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path $PSScriptRoot -Parent
$BinDir = Join-Path $Root 'bin\cuda'
$Server = Join-Path $BinDir 'llama-server.exe'
$ModelDir = Join-Path $Root 'models\bonsai2-27B'
$Model = Join-Path $ModelDir "Ternary-Bonsai-2-27B-$Pack.gguf"
$Mmproj = Join-Path $ModelDir 'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf'

if (-not (Test-Path $Server)) {
    Write-Error "llama-server.exe not found at $Server. Run .\scripts\setup.ps1 first."
}
if (-not (Test-Path $Model)) {
    Write-Error "Model not found: $Model. Run: python tools\download.py <url> <dest>"
}

# ggml-cuda.dll and the cudart DLLs live next to the executable, so the loading
# directory has to be on PATH for the process to find them.
$env:Path = "$BinDir;$env:Path"

# KV cache estimate: Bonsai 2's hybrid attention costs ~64 KiB/token at F16,
# ~18 KiB/token with the Q4_0 cache.
$kvKiB = if ($F16Kv) { 64 } else { 18 }
$kvGiB = [math]::Round($Context * $kvKiB / 1MB, 2)

$serverArgs = @(
    '-m', $Model,
    '--host', $HostAddress,
    '--port', "$Port",
    '-ngl', '99',
    '-fa', 'on',
    '-b', "$BatchSize",
    '-ub', "$UBatchSize",
    '-c', "$Context",
    '-np', "$Parallel",
    '--temp', '1.0',
    '--top-p', '0.95',
    '--top-k', '20',
    '--jinja'
)

if ($Alias) { $serverArgs += @('--alias', $Alias) }

if (-not $F16Kv) {
    $serverArgs += @('--cache-type-k', 'q4_0', '--cache-type-v', 'q4_0')
}
if (-not $NoVision -and (Test-Path $Mmproj)) {
    $serverArgs += @('--mmproj', $Mmproj)
    if ($VisionOnCPU) { $serverArgs += '--no-mmproj-offload' }
}
# Passed unconditionally so the effective setting is visible in the launch line:
# budget -1 means unrestricted thinking, the maximum available for this model.
$serverArgs += @('--reasoning-budget', "$ReasoningBudget")
if ($Reasoning -ne 'auto') { $serverArgs += @('--reasoning', $Reasoning) }
if ($ReasoningEffort -ne 'default') { $serverArgs += @('--reasoning-effort', $ReasoningEffort) }
if ($ReasoningFormat -ne 'none') { $serverArgs += @('--reasoning-format', $ReasoningFormat) }
if ($ReasoningPreserve) { $serverArgs += '--reasoning-preserve' }

# Patched web UI: a copy of the built-in UI served from disk (webui\) with the
# modelSupportsThinking gate removed, so the reasoning-effort selector in the
# composer is always visible. Build it with tools\patch_webui.py.
$UiDir = Join-Path $Root 'webui'
$usingPatchedUi = $false
if ($PatchedUi) {
    if (-not (Test-Path (Join-Path $UiDir 'index.html'))) {
        Write-Error "Patched UI not found at $UiDir. Build it: python tools\probe_webui.py (server running), then python tools\patch_webui.py"
    }
    $serverArgs += @('--path', $UiDir)
    $usingPatchedUi = $true
}

if ($ExtraArgs) { $serverArgs += $ExtraArgs }

# K-cache mean-centering bias: a quality booster for the quantized KV cache,
# built once with tools\kv-bias-corpus.txt via llama-kv-mean-center.exe. The
# vendor calibrates it with attention K-rotation disabled and requires inference
# to match, so the env var is set for the server process and restored afterwards.
# Applied automatically when the file exists, matching the demo's behaviour.
$KvBias = Join-Path $ModelDir 'kv-mean-center.gguf'
$usingKvBias = (-not $F16Kv) -and (-not $NoKvBias) -and (Test-Path $KvBias)
if ($usingKvBias) {
    $serverArgs += @('--kv-mean-center', $KvBias)
}

# Pre-flight: a second server cannot bind the port, yet it still loads the model
# and holds VRAM while the first one keeps answering requests. The result is a
# flag change that appears to do nothing. Refuse to start instead.
$alreadyUp = $false
try {
    $null = Invoke-WebRequest -Uri "http://${HostAddress}:$Port/health" -TimeoutSec 3 -ErrorAction Stop
    $alreadyUp = $true
} catch {
    $alreadyUp = $false
}
if ($alreadyUp) {
    Write-Host "[ERR] Something is already serving http://${HostAddress}:$Port/health." -ForegroundColor Red
    Write-Host '      Stop it first:  Get-Process llama-server | Stop-Process -Force' -ForegroundColor Yellow
    Write-Host '      Or pick another port:  .\scripts\start-server.ps1 -Port 8081' -ForegroundColor Yellow
    exit 1
}

Write-Host '=== Bonsai 2 27B server ===' -ForegroundColor Cyan
Write-Host "  pack         : $Pack"
Write-Host "  model        : $(Split-Path $Model -Leaf)"
Write-Host "  context      : $Context tokens (KV cache ~$kvGiB GiB, $(if ($F16Kv) { 'F16' } else { 'Q4_0' }))"
Write-Host "  slots        : $Parallel (context is shared across slots)"
Write-Host "  vision       : $(if ($NoVision) { 'off' } else { if (Test-Path $Mmproj) { if ($VisionOnCPU) { 'on, projector in system RAM' } else { 'on, projector on GPU (+0.59 GiB)' } } else { 'no mmproj file' } })"
Write-Host "  web UI       : $(if ($usingPatchedUi) { 'patched copy from webui\ (reasoning selector forced visible)' } else { 'built-in' })"
$thinkDesc = @(
    "reasoning=$Reasoning",
    "effort=$ReasoningEffort",
    "budget=$(if ($ReasoningBudget -lt 0) { 'unlimited' } else { $ReasoningBudget })"
)
if ($ReasoningFormat -ne 'none') { $thinkDesc += "format=$ReasoningFormat" }
if ($ReasoningPreserve) { $thinkDesc += 'preserve' }
Write-Host "  thinking     : $($thinkDesc -join ', ')"
Write-Host "  endpoint     : http://${HostAddress}:$Port  (chat UI, /v1/chat/completions)"
Write-Host "  model id     : $Alias   (what clients should send as 'model')"
Write-Host "  KV bias      : $(if ($usingKvBias) { 'mean-centering applied (kv-mean-center.gguf)' } elseif ($NoKvBias) { 'skipped (-NoKvBias)' } elseif ($F16Kv) { 'not applicable (F16 KV)' } else { 'no bias file built yet' })"
Write-Host '  If VRAM is short: lower -Context, add -VisionOnCPU, or switch to -Pack PTQ1_0.'
Write-Host ''

$priorRotDisable = $env:LLAMA_ATTN_ROT_DISABLE
if ($usingKvBias) { $env:LLAMA_ATTN_ROT_DISABLE = '1' }
try {
    & $Server @serverArgs
} finally {
    # Never leak the flag into the parent shell.
    if ($null -eq $priorRotDisable) {
        Remove-Item Env:LLAMA_ATTN_ROT_DISABLE -ErrorAction SilentlyContinue
    } else {
        $env:LLAMA_ATTN_ROT_DISABLE = $priorRotDisable
    }
}

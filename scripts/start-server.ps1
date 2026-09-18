<#
.SYNOPSIS
    Start Bonsai 2 27B (ternary) as an OpenAI-compatible llama.cpp server.

.DESCRIPTION
    Uses the PrismML llama.cpp fork binaries in bin\cuda. Stock llama.cpp cannot
    read the PQ2_0 / PTQ1_0 tensor types; see README.md before swapping the binary.

    Defaults are tuned for a 12 GB RTX 5070: all layers on GPU, flash attention on,
    32K context in a single slot, thinking-mode sampling from the model card.

.EXAMPLE
    .\scripts\start-server.ps1
    .\scripts\start-server.ps1 -Pack PTQ1_0 -Context 65536 -Kv4
    .\scripts\start-server.ps1 -VisionOnCPU -ReasoningBudget 2048
#>
[CmdletBinding()]
param(
    [ValidateSet('PQ2_0', 'PTQ1_0')]
    [string]$Pack = 'PQ2_0',

    [int]$Context = 32768,
    [int]$Port = 8080,
    [string]$HostAddress = '127.0.0.1',
    [int]$Parallel = 1,

    [switch]$NoVision,
    [switch]$VisionOnCPU,
    [switch]$Kv4,
    [int]$ReasoningBudget = -1,
    [switch]$PatchedUi,

    # Thinking controls. Defaults reproduce the vendor's tested configuration:
    # thinking on, template default effort, reasoning left unparsed in content.
    [ValidateSet('on', 'off', 'auto')]
    [string]$Reasoning = 'auto',

    [ValidateSet('default', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')]
    [string]$ReasoningEffort = 'default',

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
$kvKiB = if ($Kv4) { 18 } else { 64 }
$kvGiB = [math]::Round($Context * $kvKiB / 1MB, 2)

$serverArgs = @(
    '-m', $Model,
    '--host', $HostAddress,
    '--port', "$Port",
    '-ngl', '99',
    '-fa', 'on',
    '-c', "$Context",
    '-np', "$Parallel",
    '--temp', '1.0',
    '--top-p', '0.95',
    '--top-k', '20',
    '--jinja'
)

if ($Kv4) {
    $serverArgs += @('--cache-type-k', 'q4_0', '--cache-type-v', 'q4_0')
}
if (-not $NoVision -and (Test-Path $Mmproj)) {
    $serverArgs += @('--mmproj', $Mmproj)
    if ($VisionOnCPU) { $serverArgs += '--no-mmproj-offload' }
}
if ($ReasoningBudget -ge 0) {
    $serverArgs += @('--reasoning-budget', "$ReasoningBudget")
}
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
Write-Host "  context      : $Context tokens (KV cache ~$kvGiB GiB, $(if ($Kv4) { 'Q4_0' } else { 'F16' }))"
Write-Host "  slots        : $Parallel (context is shared across slots)"
Write-Host "  vision       : $(if ($NoVision) { 'off' } else { if (Test-Path $Mmproj) { if ($VisionOnCPU) { 'on, projector in system RAM' } else { 'on, projector on GPU (+0.59 GiB)' } } else { 'no mmproj file' } })"
Write-Host "  web UI       : $(if ($usingPatchedUi) { 'patched copy from webui\ (reasoning selector forced visible)' } else { 'built-in' })"
$thinkDesc = @("reasoning=$Reasoning", "effort=$ReasoningEffort")
if ($ReasoningBudget -ge 0) { $thinkDesc += "budget=$ReasoningBudget" }
if ($ReasoningFormat -ne 'none') { $thinkDesc += "format=$ReasoningFormat" }
if ($ReasoningPreserve) { $thinkDesc += 'preserve' }
Write-Host "  thinking     : $($thinkDesc -join ', ')"
Write-Host "  endpoint     : http://${HostAddress}:$Port  (chat UI, /v1/chat/completions)"
Write-Host '  If VRAM is short: lower -Context, add -Kv4, or add -VisionOnCPU.'
Write-Host ''

& $Server @serverArgs

<#
.SYNOPSIS
    Start llama-server at a given context size, measure real VRAM use, run one
    request, then stop it.

.DESCRIPTION
    llama.cpp reports its own buffer allocations, but the CUDA context, cuBLAS
    workspaces and the vision projector are not in that accounting, and on
    WDDM an oversized allocation can silently page to system memory instead of
    failing. This measures what nvidia-smi actually sees, so a context size can
    be chosen from evidence rather than arithmetic.

.EXAMPLE
    .\tools\measure-context.ps1 -Context 131072 -Kv4
    .\tools\measure-context.ps1 -Context 65536
    .\tools\measure-context.ps1 -Context 131072 -Kv4 -VisionOnCPU
#>
[CmdletBinding()]
param(
    [int]$Context = 32768,
    [switch]$Kv4,
    [switch]$VisionOnCPU,
    [int]$Port = 8090,
    [int]$TimeoutSec = 240,
    [int]$MaxTokens = 16
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path $PSScriptRoot -Parent
$Exe = Join-Path $Root 'bin\cuda\llama-server.exe'
$Model = Join-Path $Root 'models\bonsai2-27B\Ternary-Bonsai-2-27B-PQ2_0.gguf'
$Mmproj = Join-Path $Root 'models\bonsai2-27B\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf'
$Log = Join-Path $Root "measure-$Context.log"
$Err = Join-Path $Root "measure-$Context.err"
$Req = Join-Path $Root 'measure-req.json'

function Get-Vram {
    $v = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    return [int]($v -replace '[^0-9]', '')
}

$serverArgs = @(
    '-m', $Model, '--mmproj', $Mmproj,
    '-ngl', '99', '-fa', 'on', '-c', "$Context", '-np', '1',
    '--alias', 'bonsai-2-27b', '--jinja',
    '--temp', '1.0', '--top-p', '0.95', '--top-k', '20',
    '--reasoning-budget', '-1', '--reasoning-effort', 'xhigh',
    '--host', '127.0.0.1', '--port', "$Port"
)
if ($Kv4) { $serverArgs += @('--cache-type-k', 'q4_0', '--cache-type-v', 'q4_0') }
if ($VisionOnCPU) { $serverArgs += '--no-mmproj-offload' }

$label = "ctx=$Context kv=$(if ($Kv4) { 'q4_0' } else { 'f16' }) vision=$(if ($VisionOnCPU) { 'cpu' } else { 'gpu' })"
$before = Get-Vram

# Start-Process joins the array with spaces and does NOT quote, so any argument
# containing a space (this workspace path does) must be quoted by hand.
$quoted = $serverArgs | ForEach-Object { if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ } }

$proc = Start-Process -FilePath $Exe -ArgumentList $quoted -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $Log -RedirectStandardError $Err

$up = $false
for ($i = 0; $i -lt [int]($TimeoutSec / 3); $i++) {
    Start-Sleep -Seconds 3
    $code = curl.exe -s -o NUL -w "%{http_code}" "http://127.0.0.1:$Port/health"
    if ($code -eq '200') { $up = $true; break }
    if ($proc.HasExited) { break }
}

if (-not $up) {
    Write-Host "$label  -> FAILED TO START" -ForegroundColor Red
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Seconds 2
    Get-Content $Err -Tail 15 -ErrorAction SilentlyContinue
    exit 1
}

$after = Get-Vram
$nCtx = curl.exe -s "http://127.0.0.1:$Port/props" |
    python -c "import sys,json;d=json.load(sys.stdin);print(d['default_generation_settings']['n_ctx'])"

'{"model":"bonsai-2-27b","messages":[{"role":"user","content":"Say OK."}],"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}' |
    Set-Content $Req -Encoding utf8
$resp = curl.exe -s --max-time 300 "http://127.0.0.1:$Port/v1/chat/completions" `
    -H "Content-Type: application/json" --data-binary "@$Req"
Remove-Item $Req -ErrorAction SilentlyContinue
$verdict = $resp | python -c "import sys,json;d=json.load(sys.stdin);print('answered' if 'choices' in d else 'ERROR ' + str(d)[:150])"

Write-Host $label -ForegroundColor Cyan
Write-Host "  n_ctx reported    : $nCtx"
Write-Host "  VRAM before/after : $before / $after MiB   (server uses $($after - $before) MiB)"
Write-Host "  free after load   : $(12227 - $after) MiB"
Write-Host "  request           : $verdict"

Stop-Process -Id $proc.Id -Force
Start-Sleep -Seconds 4

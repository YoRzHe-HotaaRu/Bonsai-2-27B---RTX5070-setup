<#
.SYNOPSIS
    Benchmark every Bonsai 2 pack present in models\bonsai2-27B with llama-bench.

.DESCRIPTION
    Reports prompt processing (pp) and token generation (tg) throughput, the same
    two numbers the model card publishes, so results here are comparable with the
    vendor table. Flags match the shipped recipe (RECIPE.md): Q4_0 KV cache and
    batch 8192 / ubatch 2048, both measured best on this card. Writes a markdown
    table to bench-results.md.

.EXAMPLE
    .\scripts\bench.ps1
    .\scripts\bench.ps1 -PromptTokens 2048 -GenTokens 256 -Repetitions 5
#>
[CmdletBinding()]
param(
    [int]$PromptTokens = 512,
    [int]$GenTokens = 128,
    [int]$Repetitions = 3,
    [int]$Context = 4096
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path $PSScriptRoot -Parent
$BinDir = Join-Path $Root 'bin\cuda'
$Bench = Join-Path $BinDir 'llama-bench.exe'
$ModelDir = Join-Path $Root 'models\bonsai2-27B'
$OutFile = Join-Path $Root 'bench-results.md'

if (-not (Test-Path $Bench)) { Write-Error "llama-bench.exe not found. Run .\scripts\setup.ps1 first." }

$env:Path = "$BinDir;$env:Path"

$models = @(Get-ChildItem -Path $ModelDir -Filter '*Ternary-Bonsai-2-27B-*.gguf' -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notlike '*mmproj*' })
if ($models.Count -eq 0) { Write-Error "No model packs found in $ModelDir" }

$modelArgs = @()
foreach ($m in $models) { $modelArgs += @('-m', $m.FullName) }

$benchArgs = $modelArgs + @(
    '-ngl', '99',
    '-fa', '1',
    # recipe configuration: Q4_0 KV cache, batch 8192 / ubatch 2048
    '-ctk', 'q4_0',
    '-ctv', 'q4_0',
    '-b', '8192',
    '-ub', '2048',
    '-p', "$PromptTokens",
    '-n', "$GenTokens",
    '-r', "$Repetitions",
    '-o', 'md'
)

Write-Host "Benchmarking $($models.Count) pack(s): $(($models | ForEach-Object Name) -join ', ')" -ForegroundColor Cyan
Write-Host ""

& $Bench @benchArgs 2>&1 | Tee-Object -FilePath $OutFile

if ($LASTEXITCODE -ne 0) {
    Write-Warning "llama-bench exited with code $LASTEXITCODE; retrying without -fa."
    $benchArgs = $benchArgs | Where-Object { $_ -ne '-fa' -and $_ -ne '1' }
    & $Bench @benchArgs 2>&1 | Tee-Object -FilePath $OutFile
}

Write-Host ""
Write-Host "Results written to $OutFile" -ForegroundColor Green

<#
.SYNOPSIS
    Provision (or repair) the Bonsai 2 27B runtime on this machine.

.DESCRIPTION
    1. Downloads the pinned PrismML llama.cpp fork binaries (Windows x64, CUDA 13.3)
       and their CUDA runtime DLLs, verifying the SHA-256 published by the GitHub API.
    2. Downloads the GGUF weight packs that are missing, verifying SHA-256.
    3. Confirms the CUDA backend can see the GPU.

    Idempotent: already-complete files are skipped, partial files resume.

.EXAMPLE
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -AllPacks          # also fetch PTQ1_0 for an A/B comparison
    .\scripts\setup.ps1 -SkipBinaries
#>
[CmdletBinding()]
param(
    [switch]$AllPacks,
    [switch]$SkipBinaries,
    [int]$Threads = 12
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path $PSScriptRoot -Parent
$Tools = Join-Path $Root 'tools'
$ModelDir = Join-Path $Root 'models\bonsai2-27B'
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) { Write-Error 'python not found on PATH. Install Python 3.11+ first.' }

$REPO = 'prism-ml/Ternary-Bonsai-2-27B-gguf'
$HF = "https://huggingface.co/$REPO/resolve/main"

# name, bytes, sha256 (published by the Hugging Face API)
$packs = @(
    @{ Name = 'Ternary-Bonsai-2-27B-PQ2_0.gguf';        Size = 7206168928; Sha = '3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1'; Always = $true  },
    @{ Name = 'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf';  Size = 629246976;  Sha = '6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903'; Always = $true  },
    @{ Name = 'Ternary-Bonsai-2-27B-PTQ1_0.gguf';       Size = 5946648928; Sha = '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'; Always = $false }
)

if (-not $SkipBinaries) {
    Write-Host '=== llama.cpp fork binaries (Windows x64, CUDA 13.3) ===' -ForegroundColor Cyan
    & $python (Join-Path $Tools 'get_binaries.py')
    if ($LASTEXITCODE -ne 0) { Write-Error 'Binary download failed.' }
}

Write-Host ''
Write-Host '=== GGUF weight packs ===' -ForegroundColor Cyan
foreach ($p in $packs) {
    if (-not $p.Always -and -not $AllPacks) { continue }
    $dest = Join-Path $ModelDir $p.Name
    if ((Test-Path $dest) -and (Get-Item $dest).Length -eq $p.Size) {
        Write-Host "  $($p.Name): already complete"
        continue
    }
    & $python (Join-Path $Tools 'download.py') "$HF/$($p.Name)" $dest `
        --size $p.Size --sha256 $p.Sha --threads $Threads
    if ($LASTEXITCODE -ne 0) { Write-Error "Download failed for $($p.Name)" }
}

Write-Host ''
Write-Host '=== GPU visibility check ===' -ForegroundColor Cyan
$env:Path = "$(Join-Path $Root 'bin\cuda');$env:Path"
& (Join-Path $Root 'bin\cuda\llama-cli.exe') --list-devices

Write-Host ''
Write-Host 'Ready. Next:' -ForegroundColor Green
Write-Host '  .\scripts\start-server.ps1            # chat UI + API on http://127.0.0.1:8080'
Write-Host '  .\scripts\run-cli.ps1 -Prompt "hi"    # one-shot prompt'
Write-Host '  .\scripts\bench.ps1                   # throughput table'

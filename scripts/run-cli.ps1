<#
.SYNOPSIS
    One-shot prompt against Bonsai 2 27B, no server required.

.EXAMPLE
    .\scripts\run-cli.ps1 -Prompt "Explain ternary quantization in two sentences."
    .\scripts\run-cli.ps1 -Pack PTQ1_0 -Context 8192 -Prompt "hi" -NoThinking
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Prompt,

    [ValidateSet('PQ2_0', 'PTQ1_0')]
    [string]$Pack = 'PQ2_0',

    [int]$Context = 8192,
    [int]$MaxTokens = 512,
    [switch]$NoThinking,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path $PSScriptRoot -Parent
$BinDir = Join-Path $Root 'bin\cuda'
$Cli = Join-Path $BinDir 'llama-cli.exe'
$Model = Join-Path $Root "models\bonsai2-27B\Ternary-Bonsai-2-27B-$Pack.gguf"

if (-not (Test-Path $Cli)) { Write-Error "llama-cli.exe not found at $Cli. Run .\scripts\setup.ps1 first." }
if (-not (Test-Path $Model)) { Write-Error "Model not found: $Model" }

$env:Path = "$BinDir;$env:Path"

# Thinking mode (model card): temp 1.0 / top-p 0.95 / top-k 20.
# Instruct mode: temp 0.7 / top-p 0.80 / top-k 20 / presence penalty 1.5.
$sampling = if ($NoThinking) {
    @('--temp', '0.7', '--top-p', '0.80', '--top-k', '20', '--presence-penalty', '1.5',
      '--chat-template-kwargs', '{"enable_thinking": false}')
} else {
    @('--temp', '1.0', '--top-p', '0.95', '--top-k', '20')
}

$cliArgs = @(
    '-m', $Model,
    '-ngl', '99',
    '-fa', 'on',
    '-c', "$Context",
    '-n', "$MaxTokens",
    '-st'
) + $sampling + @('-p', $Prompt) + $ExtraArgs

& $Cli @cliArgs

# Push the kernel to Kaggle.
# Always re-copies the source first, so the pushed code cannot drift from the
# working copy - pushing a stale script and then puzzling over the numbers is
# the easiest way to waste a GPU session.
# ASCII only: Windows PowerShell 5.1 reads .ps1 as ANSI.
param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "cvqnn_cifar10.py"
$dst = Join-Path $PSScriptRoot "cvqnn_cifar10.py"

Copy-Item $src $dst -Force
"copied: $src -> $dst"

# Show the mode the kernel will actually run in - cheap guard against
# pushing with a leftover setting.
$mode = (Select-String -Path $src -Pattern '^\s*mode\s*=' | Select-Object -First 1).Line.Trim()
$epochs = (Select-String -Path $src -Pattern '^\s*epochs\s*=' | Select-Object -First 1).Line.Trim()
"kernel will run with:  $mode  |  $epochs"

if ($DryRun) { "dry run - not pushing"; exit 0 }

$env:KAGGLE_API_TOKEN = (Get-Content '\\wsl$\Ubuntu\home\nmakarov\.kaggle\access_token' -Raw).Trim()
$env:WSLENV = "KAGGLE_API_TOKEN"

$raw = wsl -d kali-linux /home/kali/.venvs/kaggle/bin/kaggle kernels push -p /mnt/c/claude_home/research/cvqnn/kaggle 2>&1
(($raw -join "`n") -replace "\x00", "") -split "`n" |
    Where-Object { $_ -notmatch "Unknown key" -and $_.Trim() }

# Push this repository to GitHub.
#
# The token is read from WSL (~/.gh) at call time and injected into the push
# URL from a variable: it is never written into .git/config, never stored on
# the Windows side, and is scrubbed from any output git produces.
#
# ASCII only on purpose - Windows PowerShell 5.1 reads .ps1 files as ANSI,
# so non-ASCII characters here break the parser.
param(
    [string]$Branch = "main",
    [string]$Repo   = "makarovNick/cvqnn",
    [string]$TokenPath = '\\wsl$\Ubuntu\home\nmakarov\.gh'
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $TokenPath)) {
    throw "token not found at $TokenPath"
}
$token = (Get-Content $TokenPath -Raw).Trim()

# keep the stored remote clean; the token lives only in this one URL
git -C $root remote set-url origin "https://github.com/$Repo.git" 2>$null

$out = git -C $root push "https://x-access-token:$token@github.com/$Repo.git" `
        "${Branch}:${Branch}" 2>&1

# scrub the token before anything reaches the console or a log
($out | Out-String) -replace [regex]::Escape($token), "<token>"

if ($LASTEXITCODE -ne 0) { throw "push failed with exit code $LASTEXITCODE" }
"pushed $Branch -> $Repo"

# Poll Kaggle kernel status until it finishes.
# NOTE: ASCII only on purpose - Windows PowerShell 5.1 reads .ps1 as ANSI,
# so non-ASCII text here breaks the parser.
# The token is read from WSL and forwarded via WSLENV: never copied to disk,
# never printed.
param(
    [string]$Slug = "kekekek/cvqnn-cifar-10-phase-quantized-weights",
    [int]$IntervalSec = 180,
    [int]$MaxChecks = 70
)

$env:KAGGLE_API_TOKEN = (Get-Content '\\wsl$\Ubuntu\home\nmakarov\.kaggle\access_token' -Raw).Trim()
$env:WSLENV = "KAGGLE_API_TOKEN"
$kaggle = "/home/kali/.venvs/kaggle/bin/kaggle"

for ($i = 1; $i -le $MaxChecks; $i++) {
    $raw = wsl -d kali-linux $kaggle kernels status $Slug 2>&1
    $line = (($raw -join "`n") -replace "\x00", "") -split "`n" |
            Where-Object { $_ -notmatch "Unknown key" -and $_.Trim() } |
            Select-Object -First 1

    if ($line -notmatch "RUNNING|QUEUED") {
        "[check $i] FINISHED: $line"
        exit 0
    }
    "[check $i] $line"
    Start-Sleep -Seconds $IntervalSec
}
"Poll timeout after $MaxChecks checks - kernel still running."
exit 2

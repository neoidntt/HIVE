# Download the installer "uv" into .hive_tools\uv.exe, after checking its SHA-256 checksum.
# Used by HIVE.bat when the computer has no Python 3.10+; uv then fetches its own Python.
# The URL and file names are the ones launch_hive.py uses.
$ErrorActionPreference = "Stop"
$folder = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$tools = Join-Path $folder ".hive_tools"
$target = Join-Path $tools "uv.exe"
if (Test-Path $target) { exit 0 }

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$base = "https://github.com/astral-sh/uv/releases/latest/download/"
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
$asset = "uv-$arch-pc-windows-msvc.zip"
New-Item -ItemType Directory -Force -Path $tools | Out-Null
$work = Join-Path $tools ("download-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force -Path $work | Out-Null
try {
    Write-Host "Python was not found, so the installer (uv) is downloaded first ..."
    $zip = Join-Path $work $asset
    Invoke-WebRequest -UseBasicParsing -Uri ($base + $asset) -OutFile $zip
    $text = (Invoke-WebRequest -UseBasicParsing -Uri ($base + $asset + ".sha256")).Content
    if ($text -is [byte[]]) { $text = [Text.Encoding]::ASCII.GetString($text) }
    $expected = ($text.Trim() -split "\s+")[0]
    $actual = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash
    if ($expected -ne $actual) { throw "The downloaded installer failed its checksum test." }
    Expand-Archive -Path $zip -DestinationPath (Join-Path $work "x") -Force
    $found = Get-ChildItem -Path (Join-Path $work "x") -Recurse -Filter "uv.exe" | Select-Object -First 1
    if (-not $found) { throw "uv.exe was not found in the download." }
    Copy-Item -Path $found.FullName -Destination $target -Force
} catch {
    Write-Host ("Could not get the installer: " + $_.Exception.Message)
    exit 1
} finally {
    Remove-Item -Recurse -Force -Path $work -ErrorAction SilentlyContinue
}

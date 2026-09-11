# flutter_app/tool/build.ps1
#
# Syncs this repository's flutter_app source into a build copy OUTSIDE OneDrive, then runs
# flutter there.
#
# Why: OneDrive holds handles on build\ and the Flutter tool fails with
#   "Flutter failed to delete a directory at build\unit_test_assets"
# every time it tries to clean. Building in place is not reliable on this machine. The repo
# stays canonical; the build copy is disposable and is never edited by hand.
#
#   .\tool\build.ps1 analyze     static analysis
#   .\tool\build.ps1 test        unit + widget tests
#   .\tool\build.ps1 apk         release APK, copied back to outputs/apk/
#   .\tool\build.ps1 all         analyze, then test, then apk

param(
    [Parameter(Position = 0)]
    [ValidateSet('analyze', 'test', 'apk', 'all')]
    [string]$Task = 'all'
)

$ErrorActionPreference = 'Stop'

$Tools   = 'C:/Users/souri/navpulse-build-tools'
$Flutter = "$Tools/flutter/bin/flutter.bat"
$Build   = "$Tools/indoor-build"
$Repo    = Split-Path -Parent $PSScriptRoot
$ApkOut  = Join-Path (Split-Path -Parent $Repo) 'outputs/apk'

$env:JAVA_HOME    = "$Tools/jdk-17.0.20.1+1"
$env:ANDROID_HOME = "$Tools/android-sdk"

if (-not (Test-Path $Flutter)) { throw "Flutter not found at $Flutter" }

function Sync-Source {
    Write-Host "[sync] $Repo -> $Build" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $Build | Out-Null

    # Only the inputs. build/ and .dart_tool/ in the build copy are left alone so
    # incremental compilation keeps working.
    foreach ($dir in 'lib', 'test', 'assets', 'android') {
        $src = Join-Path $Repo $dir
        if (-not (Test-Path $src)) { continue }
        $dst = Join-Path $Build $dir
        # /MIR mirrors deletions too, so a file removed from the repo does not linger in the
        # build copy and silently keep compiling.
        robocopy $src $dst /MIR /NFL /NDL /NJH /NJS /NP /XD build .dart_tool .gradle | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $dir (exit $LASTEXITCODE)" }
    }
    foreach ($file in 'pubspec.yaml', 'pubspec.lock', 'analysis_options.yaml', 'PRESENTATION.md', 'README.md') {
        $src = Join-Path $Repo $file
        if (Test-Path $src) { Copy-Item $src (Join-Path $Build $file) -Force }
    }
    # robocopy uses exit codes 0-7 for success; reset so the caller's $? is meaningful.
    $global:LASTEXITCODE = 0
}

function Invoke-Flutter {
    param([string[]]$FlutterArgs)
    Push-Location $Build
    # flutter writes progress and its "N issues found" summary to stderr. With
    # $ErrorActionPreference='Stop' the native-command wrapper turns those into terminating
    # errors and the script dies on perfectly normal output, so it is relaxed just here.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        Write-Host "[flutter] $($FlutterArgs -join ' ')" -ForegroundColor Cyan
        & $Flutter @FlutterArgs | ForEach-Object { Write-Host $_ }
        return $LASTEXITCODE
    } finally { $ErrorActionPreference = $prev; Pop-Location }
}

Sync-Source

$failed = @()

if ($Task -eq 'analyze' -or $Task -eq 'all') {
    # analyze exits non-zero on lint infos too, so only treat a real error line as failure.
    # No 2>&1 here: in Windows PowerShell 5.1 redirecting a native exe's stderr wraps each
    # line in an ErrorRecord and trips $ErrorActionPreference='Stop', so the script would
    # abort on lint output. flutter writes its findings to stdout anyway.
    Push-Location $Build
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $out = & $Flutter analyze 2>&1 | Out-String
    $ErrorActionPreference = $prev
    Pop-Location
    Write-Host $out
    if ($out -match '^\s*error\s-' -or $out -match 'error •') { $failed += 'analyze' }
    else { Write-Host "[analyze] no errors" -ForegroundColor Green }
}

if ($Task -eq 'test' -or $Task -eq 'all') {
    if ((Invoke-Flutter @('test')) -ne 0) { $failed += 'test' }
    else { Write-Host "[test] passed" -ForegroundColor Green }
}

if ($Task -eq 'apk' -or $Task -eq 'all') {
    if ((Invoke-Flutter @('build', 'apk', '--release')) -ne 0) {
        $failed += 'apk'
    } else {
        $apk = Join-Path $Build 'build/app/outputs/flutter-apk/app-release.apk'
        if (Test-Path $apk) {
            New-Item -ItemType Directory -Force -Path $ApkOut | Out-Null
            $dest = Join-Path $ApkOut 'navpulse.apk'
            Copy-Item $apk $dest -Force
            $mb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
            Write-Host "[apk] $dest ($mb MB)" -ForegroundColor Green
        } else { $failed += 'apk (artifact missing)' }
    }
}

if ($failed.Count) {
    Write-Host "FAILED: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

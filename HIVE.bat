@echo off
rem Double-click to start HIVE on Windows.
rem
rem First start: downloads what HIVE needs into this folder (the installer "uv", a private Python
rem if the computer has none, and the libraries in requirements.txt), then opens HIVE in your
rem browser. Later starts skip the download and open straight away. Nothing is installed
rem system-wide; deleting this folder removes everything. (HIVE.exe does the same without a
rem command window of your own.)
rem The first time, it asks whether to add a HIVE shortcut to your desktop (Yes / No); the answer is kept in
rem .hive_settings.json.
setlocal EnableExtensions
cd /d "%~dp0"
title HIVE
set "HIVE_LAUNCHER=%~f0"

rem 1. A real Python 3.10 or newer that can run the launcher (it needs only the standard library).
rem    The Microsoft Store "python" shortcut is not one: it fails the version test below.
set "PY="
call :try py -3
if not defined PY call :try python
if not defined PY call :try python3

if defined PY (
    call %PY% launch_hive.py %*
    goto :finished
)

rem 2. No usable Python: get the installer (uv) ourselves; uv then brings its own Python.
if exist ".hive_tools\uv.exe" goto :have_uv
where uv >nul 2>nul && goto :have_uv
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\bootstrap_uv.ps1"
if not exist ".hive_tools\uv.exe" goto :failed
:have_uv
set "UV=uv"
if exist ".hive_tools\uv.exe" set "UV=.hive_tools\uv.exe"
set "UV_CACHE_DIR=%~dp0.hive_tools\cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0.hive_tools\python"
set "NO_COLOR=1"
"%UV%" run --no-project --python 3.12 launch_hive.py %*
goto :finished

:failed
echo.
echo HIVE could not be started: Python 3.10 or newer was not found and the installer could
echo not be downloaded. Check the internet connection, or install Python from
echo https://www.python.org/downloads/ (tick "Add python.exe to PATH") and try again.
if exist "HIVE.exe" (
    echo.
    echo Starting HIVE.exe instead, which needs no Python ...
    start "" "HIVE.exe"
    exit /b 0
)
echo.
pause
exit /b 1

:finished
rem The launcher waits for a key itself after a problem it explains (exit code 1).
if errorlevel 2 (
    echo.
    echo HIVE stopped unexpectedly ^(exit code %errorlevel%^).
    pause
)
exit /b %errorlevel%

:try
rem %1 = command, %2 = optional argument. Sets PY if it is a working Python 3.10+.
where %1 >nul 2>nul || exit /b 1
call %1 %2 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul || exit /b 1
set "PY=%1 %2"
exit /b 0

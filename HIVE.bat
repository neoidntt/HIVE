@echo off
rem Double-click to start HIVE on Windows. The first start installs what HIVE needs.
cd /d "%~dp0"

rem Prefer a real Python (the "py" launcher, then "python"); skip the Microsoft Store stub.
where py >nul 2>nul && (py -3 launch_hive.py %* & goto :done)
where python >nul 2>nul && (python launch_hive.py %* & goto :done)

echo.
echo Python was not found on this computer.
echo Either install Python 3.10 or newer from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), or use HIVE.exe, which needs no Python.
echo.
pause
exit /b 1

:done
if errorlevel 1 pause

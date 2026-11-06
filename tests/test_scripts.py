"""HIVE.sh and HIVE.bat: they start HIVE with a double-click, installing first if needed."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import launch_hive

ROOT = Path(__file__).resolve().parents[1]
POSIX = pytest.mark.skipif(os.name == "nt", reason="bash script")


def _folder(tmp_path) -> Path:
    """A folder with the real HIVE.sh and a stub launcher that only reports it was run."""
    folder = tmp_path / "my hive folder"            # a space on purpose: real folders have them
    folder.mkdir()
    shutil.copy(ROOT / "HIVE.sh", folder / "HIVE.sh")
    (folder / "launch_hive.py").write_text(
        "import sys\nprint('LAUNCHER RAN', *sys.argv[1:])\n")
    return folder


def test_scripts_are_runnable_by_double_click():
    if os.name != "nt":                                   # Windows has no Unix permission bits
        assert (ROOT / "HIVE.sh").stat().st_mode & stat.S_IXUSR
    assert (ROOT / "HIVE.sh").read_text().startswith("#!/bin/bash")
    assert b"\r\n" in (ROOT / "HIVE.bat").read_bytes()         # cmd.exe wants Windows line endings
    assert not re.search(rb"[^\r]\n", (ROOT / "HIVE.bat").read_bytes())


@POSIX
def test_the_bash_script_is_syntactically_valid():
    assert subprocess.run(["bash", "-n", str(ROOT / "HIVE.sh")]).returncode == 0


def test_the_installer_download_matches_the_launcher():
    """The scripts fetch uv from the same place, under the same names, as launch_hive.py."""
    sh = (ROOT / "HIVE.sh").read_text()
    ps1 = (ROOT / "packaging" / "bootstrap_uv.ps1").read_text()
    assert f'UV_DOWNLOAD="{launch_hive.UV_DOWNLOAD}"' in sh
    assert f'$base = "{launch_hive.UV_DOWNLOAD}"' in ps1
    assert f'PYTHON_VERSION="{launch_hive.PYTHON_VERSION}"' in sh
    assert f"--python {launch_hive.PYTHON_VERSION} " in (ROOT / "HIVE.bat").read_text()
    for system, machine in (("Linux", "x86_64"), ("Linux", "aarch64"), ("Darwin", "arm64"),
                            ("Darwin", "x86_64")):
        assert launch_hive.uv_asset(system, machine) in sh
    assert launch_hive.uv_asset("Windows", "AMD64") == "uv-x86_64-pc-windows-msvc.zip"
    assert 'uv-$arch-pc-windows-msvc.zip' in ps1 and '"aarch64"' in ps1 and "Get-FileHash" in ps1


@POSIX
@pytest.mark.skipif(sys.platform != "linux", reason="the Linux terminal list (macOS uses open -a Terminal)")
def test_a_script_started_without_a_terminal_opens_one(tmp_path):
    folder = _folder(tmp_path)
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "x-terminal-emulator").write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > "{tmp_path}/args"\n')
    (fake / "x-terminal-emulator").chmod(0o755)
    env = {**os.environ, "DISPLAY": ":99", "PATH": f"{fake}:{os.environ['PATH']}"}
    env.pop("HIVE_IN_TERMINAL", None)
    done = subprocess.run(["bash", str(folder / "HIVE.sh"), "--setup-only"], env=env,
                          capture_output=True, text=True)       # captured: stdout is not a terminal
    assert done.returncode == 0 and "LAUNCHER RAN" not in done.stdout
    assert (tmp_path / "args").read_text().splitlines() == [
        "-e", "bash", str(folder / "HIVE.sh"), "--setup-only"]


@POSIX
def test_inside_a_terminal_it_runs_the_launcher_with_the_arguments(tmp_path):
    folder = _folder(tmp_path)
    done = subprocess.run(
        ["bash", str(folder / "HIVE.sh"), "--port", "8600"], capture_output=True, text=True,
        env={**os.environ, "HIVE_IN_TERMINAL": "1", "HIVE_NO_SHORTCUT": "1"})
    assert done.returncode == 0 and "LAUNCHER RAN --port 8600" in done.stdout


@POSIX
def test_the_script_tells_the_launcher_which_file_a_desktop_shortcut_should_start(tmp_path):
    folder = _folder(tmp_path)
    (folder / "launch_hive.py").write_text(
        "import os, sys\nprint('LAUNCHER', os.environ.get('HIVE_LAUNCHER'), *sys.argv[1:])\n")
    done = subprocess.run(["bash", str(folder / "HIVE.sh")], capture_output=True, text=True,
                          env={**os.environ, "HIVE_IN_TERMINAL": "1"})
    assert f"LAUNCHER {folder / 'HIVE.sh'}" in done.stdout


def test_the_windows_and_mac_launchers_do_the_same():
    bat = (ROOT / "HIVE.bat").read_text()
    assert 'set "HIVE_LAUNCHER=%~f0"' in bat
    command = (ROOT / "Run HIVE App.command").read_text()
    assert "export HIVE_LAUNCHER=" in command
    assert "launch_hive.py --desktop-prompt always" in command       # asks every time, no "never"
    code = [line for line in command.splitlines() if not line.lstrip().startswith("#")]
    assert not any("remind" in line.lower() or "never" in line.lower() for line in code)   # no such option

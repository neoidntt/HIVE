#!/usr/bin/env python3
"""One-step HIVE launcher: installs what HIVE needs the first time, then starts the app.

Double-click ``HIVE.bat`` (Windows), ``Run HIVE App.command`` (macOS) or ``HIVE.sh``
(Linux), or run ``python launch_hive.py``. A frozen ``HIVE.exe`` built from this file
behaves the same way and needs no Python on the computer.

What happens on the first start (a few minutes, then it is skipped):

1. ``uv`` (a small, fast installer from Astral) is downloaded into ``.hive_tools/``
   and its SHA-256 checksum is checked. If ``uv`` is already installed it is used.
2. ``uv`` creates ``.venv`` with Python 3.12, downloading that Python if the computer
   has none, and installs everything in ``requirements.txt`` into it.
3. HIVE opens in your browser. Everything it downloads, including uv's cache and any
   Python it fetches, stays inside this folder (``.hive_tools`` and ``.venv``).

On later starts only step 3 runs, unless ``requirements.txt`` has changed.

If ``uv`` cannot be downloaded (offline, firewall) and this is not the frozen exe, the
launcher falls back to ``python -m venv`` with the Python that is running it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PYTHON_VERSION = "3.12"
UV_DOWNLOAD = "https://github.com/astral-sh/uv/releases/latest/download/"
READY_MARKER = ".hive_ready"
MIN_PYTHON = (3, 10)


class LauncherError(RuntimeError):
    """A problem the user can act on; the message says how."""


def log(message: str = "") -> None:
    """Print immediately, so progress shows even when output is piped."""
    print(message, flush=True)


def app_dir() -> Path:
    """The HIVE folder: next to this script, or next to ``HIVE.exe`` when frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def venv_python(env: Path) -> Path:
    """The interpreter inside a virtual environment."""
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def uv_asset(system: str | None = None, machine: str | None = None) -> str:
    """Name of the ``uv`` release file for this computer.

    Raises:
        LauncherError: On a platform ``uv`` does not publish a build for.
    """
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "aarch64",
            "aarch64": "aarch64"}.get(machine)
    triple = {
        "windows": "pc-windows-msvc", "darwin": "apple-darwin", "linux": "unknown-linux-gnu",
    }.get(system)
    if arch is None or triple is None:
        raise LauncherError(f"No automatic installer for {system}/{machine}. Install Python "
                            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ and run: pip install -r requirements.txt")
    return f"uv-{arch}-{triple}" + (".zip" if system == "windows" else ".tar.gz")


def requirements_fingerprint(folder: Path) -> str:
    """Hash of what the environment should contain; changes when requirements change."""
    digest = hashlib.sha256()
    digest.update((folder / "requirements.txt").read_bytes())
    digest.update(PYTHON_VERSION.encode())
    return digest.hexdigest()


def verify_checksum(archive: Path, expected_line: str) -> None:
    """Check a download against a ``<sha256>  <name>`` checksum line.

    Raises:
        LauncherError: If the hashes differ (a damaged or altered download).
    """
    expected = expected_line.split()[0].strip().lower()
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise LauncherError(f"{archive.name} failed its checksum; it was not installed. "
                            "Run the launcher again, or install uv yourself from astral.sh/uv.")


def _fetch(url: str, destination: Path) -> None:
    """Download a file, showing a percentage when the size is known."""
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        for block in iter(lambda: response.read(1 << 18), b""):
            out.write(block)
            done += len(block)
            if total and done * 10 // total != shown:
                shown = done * 10 // total
                log(f"    {shown * 10}%")


def extract_uv(archive: Path, target_dir: Path) -> Path:
    """Unpack the ``uv`` executable from its release archive into ``target_dir``."""
    name = "uv.exe" if archive.suffix == ".zip" else "uv"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            member = next(m for m in z.namelist() if Path(m).name == name)
            target.write_bytes(z.read(member))
    else:
        with tarfile.open(archive) as t:
            member = next(m for m in t.getmembers() if Path(m.name).name == name and m.isfile())
            extracted = t.extractfile(member)
            assert extracted is not None
            target.write_bytes(extracted.read())
    target.chmod(0o755)
    return target


def find_uv(folder: Path) -> Path | None:
    """An installed ``uv``, or the copy this launcher downloaded earlier."""
    found = shutil.which("uv")
    if found:
        return Path(found)
    local = folder / ".hive_tools" / ("uv.exe" if os.name == "nt" else "uv")
    return local if local.exists() else None


def install_uv(folder: Path) -> Path:
    """Download ``uv`` into ``.hive_tools`` after checking its checksum."""
    asset = uv_asset()
    tools = folder / ".hive_tools"
    tools.mkdir(exist_ok=True)
    archive = tools / asset
    log(f"Downloading the installer (uv) ...")
    try:
        _fetch(UV_DOWNLOAD + asset, archive)
        with urllib.request.urlopen(UV_DOWNLOAD + asset + ".sha256", timeout=60) as response:
            checksum = response.read().decode()
    except OSError as exc:
        archive.unlink(missing_ok=True)
        raise LauncherError(f"Could not download the installer: {exc}") from exc
    verify_checksum(archive, checksum)
    uv = extract_uv(archive, tools)
    archive.unlink(missing_ok=True)
    return uv


def _run(command: list[str | Path], **kwargs) -> None:
    """Run a command, letting its output through; fail with a readable error."""
    result = subprocess.run([str(c) for c in command], **kwargs)
    if result.returncode != 0:
        raise LauncherError(f"Command failed ({result.returncode}): {' '.join(map(str, command[:4]))} ...")


def environment_ready(folder: Path) -> bool:
    """Whether ``.venv`` exists, matches ``requirements.txt`` and can import HIVE."""
    env = folder / ".venv"
    python = venv_python(env)
    marker = env / READY_MARKER
    if not python.exists() or not marker.exists():
        return False
    if marker.read_text().strip() != requirements_fingerprint(folder):
        return False
    probe = subprocess.run([str(python), "-c", "import streamlit, hive"], cwd=folder,
                           capture_output=True)
    return probe.returncode == 0


def build_environment(folder: Path, force: bool = False) -> Path:
    """Create or update ``.venv`` with everything in ``requirements.txt``.

    Returns:
        The environment's Python interpreter.

    Raises:
        LauncherError: If it cannot be built; the message says what to try.
    """
    env = folder / ".venv"
    python = venv_python(env)
    if force:
        shutil.rmtree(env, ignore_errors=True)
    uv = find_uv(folder)
    if uv is None:
        try:
            uv = install_uv(folder)
        except LauncherError as exc:
            if getattr(sys, "frozen", False):
                raise
            log(f"{exc}\nFalling back to python -m venv with {sys.executable}.")
    requirements = folder / "requirements.txt"
    # Keep uv's download cache and any Python it fetches inside the HIVE folder, so deleting
    # the folder removes everything.
    tools = folder / ".hive_tools"
    isolated = {**os.environ, "UV_CACHE_DIR": str(tools / "cache"),
                "UV_PYTHON_INSTALL_DIR": str(tools / "python")}

    if uv is not None:
        log(f"Preparing Python {PYTHON_VERSION} and the HIVE libraries (first start: a few "
            "minutes, several hundred MB) ...")
        if not python.exists():
            _run([uv, "venv", env, "--python", PYTHON_VERSION], cwd=folder, env=isolated)
        _run([uv, "pip", "install", "--python", python, "-r", requirements], cwd=folder,
             env=isolated)
    else:
        if sys.version_info < MIN_PYTHON:
            raise LauncherError(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is needed.")
        if not python.exists():
            _run([sys.executable, "-m", "venv", env])
        _run([python, "-m", "pip", "install", "-r", requirements], cwd=folder)

    (env / READY_MARKER).write_text(requirements_fingerprint(folder))
    return python


def start_app(folder: Path, python: Path, port: int | None) -> int:
    """Run the Streamlit app and return when the user closes it."""
    command = [str(python), "-m", "streamlit", "run", str(folder / "app.py"),
               "--server.headless=false"]
    if port:
        command.append(f"--server.port={port}")
    log("Starting HIVE. Your browser will open; keep this window open while you work, "
        "and close it (or press Ctrl+C) to stop HIVE.")
    try:
        return subprocess.call(command, cwd=folder)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    """Install if needed, then start HIVE. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Install (first time) and start HIVE")
    parser.add_argument("--setup-only", action="store_true",
                        help="Install the environment but do not start the app")
    parser.add_argument("--reinstall", action="store_true", help="Rebuild .venv from scratch")
    parser.add_argument("--port", type=int, help="Port for the app (default 8501)")
    args = parser.parse_args(argv)

    folder = app_dir()
    log("=" * 54 + "\n  HIVE - High-Processing Imaging Visualization Engine\n" + "=" * 54)
    try:
        if not (folder / "app.py").exists() or not (folder / "requirements.txt").exists():
            raise LauncherError(f"app.py and requirements.txt were not found in {folder}. "
                                "Keep the launcher inside the HIVE folder.")
        if args.reinstall or not environment_ready(folder):
            python = build_environment(folder, force=args.reinstall)
        else:
            python = venv_python(folder / ".venv")
            log("Environment is ready.")
        if args.setup_only:
            log("Setup finished.")
            return 0
        return start_app(folder, python, args.port)
    except LauncherError as exc:
        log(f"\nERROR: {exc}")
        if sys.stdin and sys.stdin.isatty() and os.name == "nt":
            input("\nPress Enter to close this window.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

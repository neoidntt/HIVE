#!/usr/bin/env python3
"""One-step HIVE launcher: installs what HIVE needs the first time, then starts the app.

Double-click ``HIVE.bat`` (Windows), ``Run HIVE App.command`` (macOS) or ``HIVE.sh``
(Linux), or run ``python launch_hive.py``. A frozen ``HIVE.exe`` built from this file
behaves the same way and needs no Python on the computer.

What happens on the first start (a few minutes, then it is skipped):

1. ``uv`` (a small, fast installer from Astral) is downloaded into ``.hive_tools/``
   and its SHA-256 checksum is checked. If ``uv`` is already installed it is used.
2. ``uv`` creates ``.venv`` (see below for an existing one) with Python 3.12, downloading that Python if the computer
   has none, and installs everything in ``requirements.txt`` into it.
3. HIVE starts on a free local port (8501 or the next free one) and the launcher opens it in
   your browser once it answers. If no browser opens, the address is printed. If the app
   fails to start, the last lines of its output are shown and the window waits.
   HIVE opens in your browser. Everything it downloads, including uv's cache and any
   Python it fetches, stays inside this folder (``.hive_tools`` and ``.venv``).

On later starts only step 3 runs, unless ``requirements.txt`` has changed.

An existing ``.venv`` that cannot be used on this computer (made on another system, for
example Linux/macOS vs Windows, or damaged) is never deleted without asking: the launcher
asks whether to replace it, or to keep it and use a separate ``.venv-<system>`` instead
(the default, and the answer when there is no keyboard to ask).

If ``uv`` cannot be downloaded (offline, firewall) and this is not the frozen exe, the
launcher falls back to ``python -m venv`` with the Python that is running it.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path
from typing import Callable, Sequence

PYTHON_VERSION = "3.12"
UV_DOWNLOAD = "https://github.com/astral-sh/uv/releases/latest/download/"
READY_MARKER = ".hive_ready"
MIN_PYTHON = (3, 10)
START_PORT = 8501
START_TIMEOUT = 180          # seconds to wait for the app to answer after it is launched


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


def system_tag() -> str:
    """Short name of this operating system, used for a separate environment folder."""
    return {"windows": "windows", "darwin": "mac", "linux": "linux"}.get(
        platform.system().lower(), "other")


def env_usable(env: Path) -> bool:
    """Whether an environment folder has an interpreter that belongs to this system."""
    return venv_python(env).exists()


def ask_choice(prompt: str, default: str) -> str:
    """Ask on the keyboard; ``default`` when there is nobody to ask."""
    if not (sys.stdin and sys.stdin.isatty()):
        log(prompt + f"\n(no keyboard available: answering '{default}')")
        return default
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return default
    return answer or default


def choose_environment(folder: Path, ask: Callable[[str, str], str] = ask_choice) -> Path:
    """Pick the environment folder to use, asking before anything is deleted.

    * ``.venv`` is used if it works on this computer.
    * A ``.venv-<system>`` that works is used next (the earlier answer "keep").
    * If ``.venv`` exists but cannot be used here (made on another system, or damaged),
      the user chooses: [R]eplace it, or [K]eep it and use ``.venv-<system>`` (default).
    * With no ``.venv`` at all, a fresh one is made.

    Returns:
        The environment folder to build or start.
    """
    main_env = folder / ".venv"
    separate = folder / f".venv-{system_tag()}"
    if env_usable(main_env):
        return main_env
    if env_usable(separate):
        return separate
    if not main_env.exists():
        return main_env

    log(f"\nA folder named .venv already exists in {folder}\n"
        f"but it cannot be used on this computer ({system_tag()}): it has no "
        f"{venv_python(Path('.venv'))}.\n"
        "It was probably created on another system, or is damaged.")
    answer = ask(
        "  [R] Replace it with a new one made for this computer (deletes the old .venv)\n"
        f"  [K] Keep it, and use a separate environment ({separate.name}) instead\n"
        "Choose R or K [K]: ", "k")
    if answer.strip().lower().startswith("r"):
        try:
            shutil.rmtree(main_env)
            return main_env
        except OSError as exc:
            log(f"Could not delete .venv ({exc}). Using {separate.name} instead and leaving "
                ".venv alone.")
    else:
        log(f"Keeping .venv. Using {separate.name}.")
    if separate.exists() and not env_usable(separate):
        shutil.rmtree(separate, ignore_errors=True)       # our own half-made folder
    return separate


def environment_ready(folder: Path, env: Path | None = None) -> bool:
    """Whether the environment exists, matches ``requirements.txt`` and can import HIVE."""
    env = env or folder / ".venv"
    python = venv_python(env)
    marker = env / READY_MARKER
    if not python.exists() or not marker.exists():
        return False
    if marker.read_text().strip() != requirements_fingerprint(folder):
        return False
    probe = subprocess.run([str(python), "-c", "import streamlit, hive"], cwd=folder,
                           capture_output=True)
    return probe.returncode == 0


def build_environment(folder: Path, force: bool = False, env: Path | None = None) -> Path:
    """Create or update the environment with everything in ``requirements.txt``.

    Returns:
        The environment's Python interpreter.

    Raises:
        LauncherError: If it cannot be built; the message says what to try.
    """
    env = env or folder / ".venv"
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
    # NO_COLOR: plain Windows consoles (and Wine) print uv's colour codes as garbage.
    isolated = {**os.environ, "UV_CACHE_DIR": str(tools / "cache"),
                "UV_PYTHON_INSTALL_DIR": str(tools / "python"), "NO_COLOR": "1"}

    if uv is not None:
        log(f"Preparing Python {PYTHON_VERSION} and the HIVE libraries (first start: a few "
            "minutes, several hundred MB) ...")
        if not python.exists():
            # --clear: the choice about an existing folder was made (by the user) before this
            # point; never let uv ask its own question.
            _run([uv, "venv", env, "--python", PYTHON_VERSION, "--clear"], cwd=folder,
                 env=isolated)
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


def find_free_port(start: int = START_PORT, tries: int = 50) -> int:
    """First port from ``start`` that nothing is listening on.

    Raises:
        LauncherError: If none of ``tries`` consecutive ports is free.
    """
    for port in range(start, start + tries):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise LauncherError(f"No free port between {start} and {start + tries - 1}; "
                        "close other programs or pass --port.")


def wait_for_health(port: int, process: subprocess.Popen, timeout: float = START_TIMEOUT) -> bool:
    """Wait until Streamlit answers on ``port``; ``False`` if it exits or times out.

    Asks the IPv4 and the IPv6 loopback address directly instead of the name
    ``localhost``: on Windows and macOS ``localhost`` tries ``::1`` first and a refused
    connection there can use up the whole timeout before IPv4 is tried.
    """
    urls = [f"http://{host}:{port}/_stcore/health" for host in ("127.0.0.1", "[::1]")]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=1.5) as response:
                    if response.status == 200:
                        return True
            except OSError:
                continue
        time.sleep(0.5)
    return False


def serve(command: Sequence[str | Path], cwd: Path, port: int,
          open_browser: Callable[[str], bool] = webbrowser.open,
          timeout: float = START_TIMEOUT) -> int:
    """Run the app, open the browser when it answers, and return when it stops.

    Raises:
        LauncherError: If the app exits or does not answer before ``timeout``; the
            message carries the last lines it printed.
    """
    tail: collections.deque[str] = collections.deque(maxlen=40)
    process = subprocess.Popen(
        [str(c) for c in command], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    if not wait_for_health(port, process, timeout):
        if process.poll() is None:
            process.terminate()
        reader.join(timeout=3)
        raise LauncherError(
            "HIVE did not start. The last lines it printed:\n" + "".join(tail)
            + "\nIf this is not clear, send this text (and your operating system) to whoever "
              "maintains HIVE.")

    url = f"http://localhost:{port}"
    log(f"\nHIVE is running at {url}")
    try:
        opened = open_browser(url)
    except Exception:  # noqa: BLE001 - a missing browser must not stop the app
        opened = False
    if not opened:
        log(f"Could not open a browser automatically. Open this address yourself: {url}")
    log("Keep this window open while you work; close it (or press Ctrl+C) to stop HIVE.")
    try:
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        code = 0
    reader.join(timeout=3)
    return code


def start_app(folder: Path, python: Path, port: int | None) -> int:
    """Start the Streamlit app on a free port and return when the user closes it.

    ``--server.headless=true`` stops Streamlit asking for an e-mail address on first run
    (which would wait for input in a window nobody is looking at); the launcher opens the
    browser itself instead. The server listens on this computer only.
    """
    port = port or find_free_port()
    command = [python, "-m", "streamlit", "run", folder / "app.py", "--server.headless=true",
               f"--server.port={port}", "--server.address=localhost",
               "--browser.gatherUsageStats=false"]
    log(f"Starting HIVE on port {port} ...")
    return serve(command, folder, port)


def main(argv: list[str] | None = None) -> int:
    """Install if needed, then start HIVE. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Install (first time) and start HIVE")
    parser.add_argument("--setup-only", action="store_true",
                        help="Install the environment but do not start the app")
    parser.add_argument("--reinstall", action="store_true",
                        help="Rebuild the environment from scratch (deletes it without asking)")
    parser.add_argument("--port", type=int, help="Port for the app (default 8501)")
    args = parser.parse_args(argv)

    folder = app_dir()
    log("=" * 54 + "\n  HIVE - High-Processing Imaging Visualization Engine\n" + "=" * 54)
    try:
        if not (folder / "app.py").exists() or not (folder / "requirements.txt").exists():
            raise LauncherError(f"app.py and requirements.txt were not found in {folder}. "
                                "Keep the launcher inside the HIVE folder.")
        env = choose_environment(folder)
        if args.reinstall or not environment_ready(folder, env):
            python = build_environment(folder, force=args.reinstall, env=env)
        else:
            python = venv_python(env)
            log("Environment is ready.")
        if args.setup_only:
            log("Setup finished.")
            return 0
        return start_app(folder, python, args.port)
    except LauncherError as exc:
        log(f"\nERROR: {exc}")
        if sys.stdin and sys.stdin.isatty():
            input("\nPress Enter to close this window.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""A native "choose a file / folder" dialog, for opening data where it already is.

The web app runs on the same computer as the browser, so it can ask the operating system for
a file chooser instead of making the user type a path. Each system's own tool is used (none
of them needs an extra Python package), and each is started as a separate process so a GUI
toolkit never runs inside the web server:

* Linux: ``zenity`` (GNOME) or ``kdialog`` (KDE), when installed;
* macOS: ``osascript``;
* Windows: PowerShell with the .NET file dialogs;
* anywhere else ``tkinter``, if this Python has it.

If none is available :func:`available` is ``False`` and the app asks for a typed path.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

_TITLE_FILE = "Choose an image or stack"
_TITLE_FOLDER = "Choose a folder of images"
_TK_SCRIPT = (
    "import sys, tkinter as tk\n"
    "from tkinter import filedialog\n"
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "p = filedialog.askdirectory(title=sys.argv[2]) if sys.argv[1] == 'folder' "
    "else filedialog.askopenfilename(title=sys.argv[2])\n"
    "print(p or '')\n"
)
_WINDOWS_FILE = (
    "Add-Type -AssemblyName System.Windows.Forms; "
    "$d = New-Object System.Windows.Forms.OpenFileDialog; $d.Title = '%s'; "
    "if ($d.ShowDialog() -eq 'OK') { $d.FileName }"
)
_WINDOWS_FOLDER = (
    "Add-Type -AssemblyName System.Windows.Forms; "
    "$d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description = '%s'; "
    "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }"
)


def _tk_importable() -> bool:
    try:
        import tkinter  # noqa: F401
        return True
    except ImportError:
        return False


def _command(kind: str) -> list[str] | None:
    """The command that opens the dialog on this system, or ``None`` if there is none."""
    folder = kind == "folder"
    title = _TITLE_FOLDER if folder else _TITLE_FILE
    if sys.platform == "darwin" and shutil.which("osascript"):
        verb = "choose folder" if folder else "choose file"
        return ["osascript", "-e", f'POSIX path of ({verb} with prompt "{title}")']
    if sys.platform == "win32" and shutil.which("powershell"):
        script = (_WINDOWS_FOLDER if folder else _WINDOWS_FILE) % title
        return ["powershell", "-NoProfile", "-STA", "-Command", script]
    if sys.platform.startswith("linux"):
        if shutil.which("zenity"):
            return ["zenity", "--file-selection", f"--title={title}"] + (["--directory"] if folder else [])
        if shutil.which("kdialog"):
            return ["kdialog", "--getexistingdirectory" if folder else "--getopenfilename", "."]
    if _tk_importable():
        return [sys.executable, "-c", _TK_SCRIPT, kind, title]
    return None


def available() -> bool:
    """Whether a file dialog can be opened on this computer."""
    return _command("file") is not None


def pick(kind: str = "file") -> str | None:
    """Open the dialog and return the chosen path, or ``None`` if cancelled or unavailable.

    Args:
        kind: ``"file"`` or ``"folder"``.

    Returns:
        The path as text.
    """
    if kind not in ("file", "folder"):
        raise ValueError(f"kind must be 'file' or 'folder', got {kind!r}")
    command = _command(kind)
    if command is None:
        return None
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.SubprocessError):
        return None
    chosen = done.stdout.strip().splitlines()
    return chosen[-1].strip() if done.returncode == 0 and chosen and chosen[-1].strip() else None

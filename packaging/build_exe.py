#!/usr/bin/env python3
"""Build the one-file launcher: ``HIVE.exe`` on Windows, ``HIVE`` on macOS/Linux.

The program is ``launch_hive.py`` frozen with PyInstaller, so it carries its own Python
and needs nothing installed. Put the result in the HIVE folder (next to ``app.py``) and
double-click it: it installs HIVE's libraries on first start, then opens the app.

    pip install pyinstaller
    python packaging/build_exe.py

Build it on the system it is for: a Windows .exe has to be built on Windows (the
``Build launcher`` GitHub workflow does this). The file is unsigned, so Windows
SmartScreen may show "unknown publisher" the first time; choose "More info" > "Run anyway".
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def make_icon(target: Path) -> Path | None:
    """Convert the bundled logo to a Windows icon; ``None`` if there is no logo."""
    logo = ROOT / "assets" / "hive_logo.jpg"
    if not logo.exists():
        return None
    from PIL import Image

    with Image.open(logo) as image:
        side = max(image.size)
        square = Image.new("RGB", (side, side), "white")
        square.paste(image.convert("RGB"), ((side - image.width) // 2, (side - image.height) // 2))
        square.save(target, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return target


def main() -> int:
    import PyInstaller.__main__

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        args = [
            "--onefile", "--name", "HIVE", "--console", "--noconfirm", "--clean",
            "--distpath", str(ROOT), "--workpath", str(work / "build"), "--specpath", str(work),
        ]
        if sys.platform == "win32" and (icon := make_icon(work / "hive.ico")):
            args += ["--icon", str(icon)]
        PyInstaller.__main__.run(args + [str(ROOT / "launch_hive.py")])
    built = ROOT / ("HIVE.exe" if sys.platform == "win32" else "HIVE")
    print(f"Built {built}" if built.exists() else "Build failed")
    return 0 if built.exists() else 1


if __name__ == "__main__":
    sys.exit(main())

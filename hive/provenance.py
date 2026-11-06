"""Saved settings and run records, so any result can be reproduced.

Every analysis writes ``hive_run.json`` into its output folder. It holds the full
configuration (including the random seed), the HIVE release, the versions of the
libraries that produced the numbers, and a fingerprint of each input file.
:func:`read_run_record` loads it back into a :class:`~hive.config.HiveConfig`, and
:func:`environment_differences` tells the user if the libraries have changed since
-- the usual reason a "reproduced" result differs in the last decimal places.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import HiveConfig

RUN_RECORD_NAME = "hive_run.json"
_RECORD_FORMAT = 1

# Libraries whose versions can change a number, not just the look of a plot.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy", "scipy", "scikit-image", "pandas", "tifffile", "imagecodecs",
    "matplotlib", "pillow", "czifile", "nd2", "readlif", "roifile", "cellpose",
)
_FULL_HASH_BYTES = 1 << 30      # hash a file completely up to 1 GB ...
_TOTAL_HASH_BUDGET = 4 << 30    # ... and stop hashing after 4 GB in total per run


@dataclass
class RunRecord:
    """A saved run, as read back from disk.

    Attributes:
        config: The configuration that produced the run.
        record: The full JSON content (versions, inputs, timestamp).
        path: Where it was read from.
    """

    config: HiveConfig
    record: dict[str, Any]
    path: Path


def package_versions() -> dict[str, str]:
    """Versions of the installed libraries that affect results.

    Returns:
        Mapping of package name to version; packages that are not installed are
        omitted.
    """
    from importlib import metadata

    found = {}
    for name in _TRACKED_PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def file_fingerprint(path: Path, hash_budget: list[int] | None = None) -> dict[str, Any]:
    """Describe an input file well enough to recognise it again.

    Args:
        path: File to describe.
        hash_budget: One-element list holding the bytes that may still be hashed
            in this run; decremented in place. ``None`` = no limit.

    Returns:
        Dict with ``name``, ``size_bytes`` and ``sha256`` (``None`` when the
        file was too large to hash within the budget).
    """
    size = path.stat().st_size
    entry: dict[str, Any] = {"name": path.name, "size_bytes": size, "sha256": None}
    allowed = size <= _FULL_HASH_BYTES and (hash_budget is None or size <= hash_budget[0])
    if not allowed:
        return entry
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    entry["sha256"] = digest.hexdigest()
    if hash_budget is not None:
        hash_budget[0] -= size
    return entry


def _input_files(input_path: str | Path) -> list[Path]:
    """List the files an input path stands for (one file, or a folder's images)."""
    from .io_utils import collect_image_paths

    path = Path(input_path)
    if path.is_file():
        return [path]
    return collect_image_paths(path) if path.is_dir() else []


def write_run_record(
    output_folder: str | Path, cfg: HiveConfig, notes: dict[str, Any] | None = None,
) -> Path:
    """Write ``hive_run.json`` for an analysis into its output folder.

    Args:
        output_folder: Where the analysis wrote its results.
        cfg: The configuration that was used.
        notes: Extra facts to keep with the run, for example that tiled
            processing replaced CLAHE.

    Returns:
        Path of the written file.
    """
    from . import __release__, __version__

    budget = [_TOTAL_HASH_BUDGET]
    record = {
        "record_format": _RECORD_FORMAT,
        "hive_release": __release__,
        "hive_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "inputs": [file_fingerprint(p, budget) for p in _input_files(cfg.input_path)],
        "notes": notes or {},
        "config": cfg.to_dict(),
    }
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    path = out / RUN_RECORD_NAME
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def read_run_record(path: str | Path) -> RunRecord:
    """Load a saved run (or a bare settings file) back into a configuration.

    Args:
        path: ``hive_run.json``, a folder containing it, or a JSON file holding
            just a configuration dict.

    Returns:
        The configuration and the full record.

    Raises:
        FileNotFoundError: If there is no such file.
        ValueError: If the file is not valid JSON or holds unknown settings.
    """
    path = Path(path)
    if path.is_dir():
        path = path / RUN_RECORD_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No saved settings at {path}")
    return parse_run_record(path.read_text(encoding="utf-8"), path)


def parse_run_record(text: str, source: str | Path = "settings") -> RunRecord:
    """Parse the text of a saved run (for example an uploaded file).

    Args:
        text: JSON text of ``hive_run.json`` or of a bare configuration dict.
        source: Name used in messages.

    Returns:
        The configuration and the full record.

    Raises:
        ValueError: If the text is not valid JSON or holds unknown settings.
    """
    name = Path(source).name
    try:
        content = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"{name} does not hold a settings object")
    config = content.get("config", content)
    return RunRecord(HiveConfig.from_dict(config), content if "config" in content else {},
                     Path(source))


def environment_differences(record: dict[str, Any]) -> list[str]:
    """Compare a saved run's library versions with the ones installed now.

    Args:
        record: The ``record`` of a :class:`RunRecord`.

    Returns:
        One readable line per difference; empty when nothing changed (or the
        record holds no environment).
    """
    saved = record.get("environment", {})
    now = package_versions()
    lines = []
    if saved.get("python") and saved["python"] != sys.version.split()[0]:
        lines.append(f"python: saved {saved['python']}, now {sys.version.split()[0]}")
    for name, version in saved.get("packages", {}).items():
        current = now.get(name)
        if current != version:
            lines.append(f"{name}: saved {version}, now {current or 'not installed'}")
    return lines

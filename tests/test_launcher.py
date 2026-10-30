"""The one-click launcher: the parts that can be checked without a network."""

from __future__ import annotations

import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import launch_hive as launcher


def test_the_right_uv_download_is_chosen_for_each_platform():
    assert launcher.uv_asset("Windows", "AMD64") == "uv-x86_64-pc-windows-msvc.zip"
    assert launcher.uv_asset("Darwin", "arm64") == "uv-aarch64-apple-darwin.tar.gz"
    assert launcher.uv_asset("Linux", "x86_64") == "uv-x86_64-unknown-linux-gnu.tar.gz"
    with pytest.raises(launcher.LauncherError, match="pip install"):
        launcher.uv_asset("Plan9", "mips")


def test_a_damaged_download_is_refused(tmp_path):
    archive = tmp_path / "uv.tar.gz"
    archive.write_bytes(b"genuine contents")
    good = hashlib.sha256(b"genuine contents").hexdigest()
    launcher.verify_checksum(archive, f"{good}  uv.tar.gz\n")
    archive.write_bytes(b"tampered contents")
    with pytest.raises(launcher.LauncherError, match="checksum"):
        launcher.verify_checksum(archive, f"{good}  uv.tar.gz\n")


def test_uv_is_unpacked_from_both_archive_types(tmp_path):
    payload = tmp_path / "uv"
    payload.write_bytes(b"#!/bin/sh\n")
    tar_path = tmp_path / "uv-x.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(payload, arcname="uv-x86_64-unknown-linux-gnu/uv")
    out = launcher.extract_uv(tar_path, tmp_path / "t1")
    assert out.name == "uv" and out.read_bytes() == b"#!/bin/sh\n" and (sys.platform == "win32" or out.stat().st_mode & 0o111)

    zip_path = tmp_path / "uv-x.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("uv.exe", b"MZ")
        z.writestr("uvx.exe", b"other")
    assert launcher.extract_uv(zip_path, tmp_path / "t2").read_bytes() == b"MZ"


def test_the_environment_is_rebuilt_only_when_requirements_change(tmp_path):
    (tmp_path / "requirements.txt").write_text("numpy\n")
    first = launcher.requirements_fingerprint(tmp_path)
    assert first == launcher.requirements_fingerprint(tmp_path)
    (tmp_path / "requirements.txt").write_text("numpy\nscipy\n")
    assert launcher.requirements_fingerprint(tmp_path) != first


def test_an_environment_without_the_ready_marker_is_not_ready(tmp_path):
    (tmp_path / "requirements.txt").write_text("numpy\n")
    assert not launcher.environment_ready(tmp_path)
    python = launcher.venv_python(tmp_path / ".venv")
    python.parent.mkdir(parents=True)
    python.write_text("")
    assert not launcher.environment_ready(tmp_path)                 # no marker yet


def test_the_interpreter_path_matches_the_platform(tmp_path):
    expected = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    assert launcher.venv_python(tmp_path) == tmp_path / expected


def test_the_frozen_exe_looks_for_the_app_next_to_itself(monkeypatch, tmp_path):
    exe = tmp_path / "HIVE.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert launcher.app_dir() == tmp_path


def test_a_launcher_in_the_wrong_folder_explains_itself(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(launcher, "app_dir", lambda: tmp_path)
    assert launcher.main(["--setup-only"]) == 1
    assert "app.py and requirements.txt were not found" in capsys.readouterr().out


def test_setup_is_skipped_when_the_environment_is_ready(monkeypatch, tmp_path, capsys):
    (tmp_path / "app.py").write_text("")
    (tmp_path / "requirements.txt").write_text("numpy\n")
    monkeypatch.setattr(launcher, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(launcher, "environment_ready", lambda folder: True)
    monkeypatch.setattr(launcher, "build_environment",
                        lambda *a, **k: pytest.fail("must not rebuild a ready environment"))
    assert launcher.main(["--setup-only"]) == 0
    assert "Environment is ready" in capsys.readouterr().out


def test_the_wrappers_exist_and_call_the_launcher():
    root = Path(launcher.__file__).parent
    for name in ("HIVE.bat", "HIVE.sh", "Run HIVE App.command"):
        assert "launch_hive.py" in (root / name).read_text(), name
    assert "-r requirements.txt" in (root / "environment.yml").read_text()

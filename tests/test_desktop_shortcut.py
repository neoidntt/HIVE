"""The "Add HIVE to your desktop?" prompt: what it asks, what it remembers, what it makes."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

import launch_hive as lh


class Answers:
    """A stand-in for the console: hands out answers and records the questions asked."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def __call__(self, question):
        self.asked.append(question)
        return self.answers.pop(0)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A HIVE folder and a desktop; shortcut creation just drops a file."""
    folder, desktop = tmp_path / "hive folder", tmp_path / "Desktop"
    folder.mkdir()
    desktop.mkdir()
    monkeypatch.setattr(lh, "desktop_dir", lambda system=None: desktop)
    made = []

    def fake_create(folder_, desktop_, system=None):
        link = lh.shortcut_path(desktop_)
        link.write_text("shortcut")
        made.append(link)
        return link

    monkeypatch.setattr(lh, "create_shortcut", fake_create)
    return folder, desktop, made


def state(folder):
    return lh.load_settings(folder).get("desktop_shortcut")


def test_an_existing_shortcut_means_no_question(world):
    folder, desktop, made = world
    lh.shortcut_path(desktop).write_text("already there")
    ask = Answers()
    lh.offer_desktop_shortcut(folder, ask=ask)
    assert ask.asked == [] and made == [] and state(folder) is None


def test_yes_makes_the_shortcut_and_never_asks_again(world):
    folder, desktop, made = world
    ask = Answers(True)
    lh.offer_desktop_shortcut(folder, ask=ask)
    assert ask.asked == ["Add HIVE to your desktop?"] and len(made) == 1 and state(folder) == "added"
    lh.shortcut_path(desktop).unlink()                      # even if it is deleted later
    again = Answers()
    lh.offer_desktop_shortcut(folder, ask=again)
    assert again.asked == []


def test_no_then_remind_asks_again_next_time(world):
    folder, _, made = world
    ask = Answers(False, True)
    lh.offer_desktop_shortcut(folder, ask=ask)
    assert ask.asked == ["Add HIVE to your desktop?", "Remind you next time you open HIVE?"]
    assert made == [] and state(folder) == "remind"
    second = Answers(True)
    lh.offer_desktop_shortcut(folder, ask=second)           # asked again, and this time Yes
    assert second.asked == ["Add HIVE to your desktop?"] and state(folder) == "added"


def test_no_then_no_never_asks_from_these_launchers(world):
    folder, _, made = world
    lh.offer_desktop_shortcut(folder, ask=Answers(False, False))
    assert state(folder) == "never" and made == []
    ask = Answers()
    lh.offer_desktop_shortcut(folder, ask=ask)
    assert ask.asked == []


def test_the_mac_command_asks_every_time_and_has_no_never_option(world):
    folder, _, made = world
    lh.save_settings(folder, {"desktop_shortcut": "never"})          # the saved choice is ignored
    ask = Answers(False)
    lh.offer_desktop_shortcut(folder, "always", ask=ask)
    assert ask.asked == ["Add HIVE to your desktop?"]                 # no second question
    assert state(folder) == "never" and made == []                    # declining changes nothing
    again = Answers(True)
    lh.offer_desktop_shortcut(folder, "always", ask=again)
    assert len(made) == 1 and state(folder) == "added"
    third = Answers()
    lh.offer_desktop_shortcut(folder, "always", ask=third)            # now there is one: silent
    assert third.asked == []


def test_nobody_to_ask_or_a_failure_saves_nothing_and_never_blocks(world, monkeypatch):
    folder, _, _ = world
    lh.offer_desktop_shortcut(folder, ask=Answers(None))              # no keyboard
    assert state(folder) is None
    monkeypatch.setattr(lh, "create_shortcut", lambda *a, **k: (_ for _ in ()).throw(lh.LauncherError("disk full")))
    lh.offer_desktop_shortcut(folder, ask=Answers(True))               # does not raise
    assert state(folder) is None                                       # so it asks again next time


def test_no_desktop_folder_means_no_question(tmp_path, monkeypatch):
    monkeypatch.setattr(lh, "desktop_dir", lambda system=None: None)
    ask = Answers()
    lh.offer_desktop_shortcut(tmp_path, ask=ask)
    assert ask.asked == []


def test_a_damaged_settings_file_is_ignored(tmp_path):
    (tmp_path / lh.SETTINGS_FILE).write_text("{not json")
    assert lh.load_settings(tmp_path) == {}
    lh.save_settings(tmp_path, {"desktop_shortcut": "never"})
    assert lh.load_settings(tmp_path) == {"desktop_shortcut": "never"}


def test_the_settings_file_is_next_to_the_launcher_and_not_committed():
    root = Path(lh.__file__).parent
    assert lh.settings_path(root) == root / ".hive_settings.json"
    assert ".hive_settings.json" in (root / ".gitignore").read_text()


def test_the_prompt_runs_at_start_with_the_chosen_mode(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(lh, "app_dir", lambda: tmp_path)
    (tmp_path / "app.py").write_text("")
    (tmp_path / "requirements.txt").write_text("")
    monkeypatch.setattr(lh, "offer_desktop_shortcut", lambda folder, mode="settings", **k: calls.append((folder, mode)))
    monkeypatch.setattr(lh, "choose_environment", lambda folder: tmp_path / ".venv")
    monkeypatch.setattr(lh, "environment_ready", lambda folder, env=None: True)
    lh.main(["--setup-only"])
    lh.main(["--setup-only", "--desktop-prompt", "always"])
    assert calls == [(tmp_path, "settings"), (tmp_path, "always")]


# ---- what each system's shortcut is made of ------------------------------------------------

@pytest.mark.skipif(os.name == "nt", reason="the Linux desktop-file writer, with POSIX-style paths")
def test_linux_shortcut_is_a_trusted_executable_desktop_file(tmp_path, monkeypatch):
    folder, desktop = tmp_path / "my hive", tmp_path / "Desktop"
    folder.mkdir()
    desktop.mkdir()
    (folder / "HIVE.sh").write_text("#!/bin/bash\n")
    (folder / "assets").mkdir()
    (folder / "assets" / "hive_logo.jpg").write_bytes(b"x")
    monkeypatch.setenv("HIVE_LAUNCHER", str(folder / "HIVE.sh"))
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: calls.append(cmd))
    monkeypatch.setattr(lh.shutil, "which", lambda name: "/usr/bin/gio")
    link = lh.create_shortcut(folder, desktop, "Linux")
    text = link.read_text()
    assert link == desktop / "HIVE.desktop"
    if os.name != "nt":                                   # Windows has no Unix permission bits
        assert link.stat().st_mode & stat.S_IXUSR
    assert f'Exec="{folder / "HIVE.sh"}"' in text and f"Path={folder}" in text and "Terminal=true" in text
    assert calls == [["gio", "set", str(link), "metadata::trusted", "true"]]


def test_mac_shortcut_is_a_command_file_that_starts_the_launcher(tmp_path, monkeypatch):
    folder, desktop = tmp_path / "my hive", tmp_path / "Desktop"
    folder.mkdir()
    desktop.mkdir()
    target = folder / "Run HIVE App.command"
    target.write_text("#!/bin/bash\n")
    monkeypatch.setenv("HIVE_LAUNCHER", str(target))
    link = lh.create_shortcut(folder, desktop, "Darwin")
    assert link == desktop / "HIVE.command"
    if os.name != "nt":
        assert link.stat().st_mode & stat.S_IXUSR
    assert link.read_text() == f"#!/bin/bash\nexec '{target}' \"$@\"\n"


def test_windows_shortcut_is_made_with_a_powershell_link(tmp_path, monkeypatch):
    folder, desktop = tmp_path / "my hive", tmp_path / "Desktop"
    folder.mkdir()
    desktop.mkdir()
    (folder / "HIVE.bat").write_text("@echo off\n")
    (folder / "HIVE.exe").write_bytes(b"MZ")
    monkeypatch.setenv("HIVE_LAUNCHER", str(folder / "HIVE.bat"))
    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: seen.update(cmd=cmd, env=k["env"]))
    link = lh.create_shortcut(folder, desktop, "Windows")
    assert link == desktop / "HIVE.lnk"
    assert seen["cmd"][0] == "powershell" and "CreateShortcut($env:HIVE_LNK)" in seen["cmd"][-1]
    env = seen["env"]
    assert env["HIVE_LNK"] == str(link) and env["HIVE_TARGET"] == str(folder / "HIVE.bat")
    assert env["HIVE_DIR"] == str(folder) and env["HIVE_ICON"] == f"{folder / 'HIVE.exe'},0"


def test_the_frozen_exe_points_the_shortcut_at_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(lh.sys, "frozen", True, raising=False)
    monkeypatch.setattr(lh.sys, "executable", str(tmp_path / "HIVE.exe"))
    assert lh.launcher_target(tmp_path) == tmp_path / "HIVE.exe"


def test_shortcut_names_per_system(tmp_path):
    assert [lh.shortcut_path(tmp_path, s).name for s in ("Windows", "Darwin", "Linux")] == [
        "HIVE.lnk", "HIVE.command", "HIVE.desktop"]

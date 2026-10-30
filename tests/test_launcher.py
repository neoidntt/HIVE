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
    monkeypatch.setattr(launcher, "environment_ready", lambda folder, env=None: True)
    monkeypatch.setattr(launcher, "build_environment",
                        lambda *a, **k: pytest.fail("must not rebuild a ready environment"))
    assert launcher.main(["--setup-only"]) == 0
    assert "Environment is ready" in capsys.readouterr().out


def test_the_wrappers_exist_and_call_the_launcher():
    root = Path(launcher.__file__).parent
    for name in ("HIVE.bat", "HIVE.sh", "Run HIVE App.command"):
        assert "launch_hive.py" in (root / name).read_text(), name
    assert "-r requirements.txt" in (root / "environment.yml").read_text()


# --- starting the app: the browser must open, and failures must be visible -----------------

STUB_SERVER = """
import http.server, sys, threading, time
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
class S(http.server.HTTPServer):
    def server_bind(self):     # HTTPServer.server_bind does a reverse-DNS lookup that can take
        import socketserver    # tens of seconds on macOS runners; this stub does not need it
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "localhost", self.server_address[1]
server = S(("127.0.0.1", int(sys.argv[1])), H)
threading.Thread(target=server.serve_forever, daemon=True).start()
print("stub server up", flush=True)
time.sleep(float(sys.argv[2]))
"""


def test_a_busy_port_is_skipped():
    import socket
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        port = busy.getsockname()[1]
        assert launcher.find_free_port(port, tries=5) != port


def test_the_browser_is_opened_once_the_app_answers(tmp_path):
    port = launcher.find_free_port(8700)
    opened = []
    code = launcher.serve([sys.executable, "-c", STUB_SERVER, str(port), "1.5"], tmp_path, port,
                          open_browser=lambda url: opened.append(url) or True, timeout=30)
    assert code == 0 and opened == [f"http://localhost:{port}"]


def test_a_missing_browser_prints_the_address_instead_of_failing(tmp_path, capsys):
    port = launcher.find_free_port(8710)
    launcher.serve([sys.executable, "-c", STUB_SERVER, str(port), "1"], tmp_path, port,
                   open_browser=lambda url: False, timeout=30)
    assert f"Open this address yourself: http://localhost:{port}" in capsys.readouterr().out


def test_an_app_that_crashes_shows_its_output_and_does_not_open_a_browser(tmp_path):
    opened = []
    crash = "import sys; print('ModuleNotFoundError: no module named streamlit'); sys.exit(1)"
    with pytest.raises(launcher.LauncherError, match="no module named streamlit"):
        launcher.serve([sys.executable, "-c", crash], tmp_path, 8799,
                       open_browser=lambda url: opened.append(url), timeout=10)
    assert opened == []


def test_an_app_that_never_answers_is_stopped_after_the_timeout(tmp_path):
    with pytest.raises(launcher.LauncherError, match="did not start"):
        launcher.serve([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, 8798,
                       open_browser=lambda url: True, timeout=1.5)


def test_streamlit_is_started_headless_on_this_computer_only(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(launcher, "serve", lambda command, cwd, port, **k: seen.update(
        command=[str(c) for c in command], port=port) or 0)
    launcher.start_app(tmp_path, Path("python"), None)
    assert "--server.headless=true" in seen["command"]      # no first-run e-mail prompt
    assert "--server.address=localhost" in seen["command"]
    assert f"--server.port={seen['port']}" in seen["command"]
    assert str(tmp_path / "app.py") in seen["command"]


# --- an existing .venv that this computer cannot use must never be deleted unasked ----------


def _foreign_venv(folder: Path) -> Path:
    """A .venv as another operating system would have left it: no interpreter at our path."""
    venv = folder / ".venv"
    other = venv / ("bin/python" if sys.platform == "win32" else "Scripts/python.exe")
    other.parent.mkdir(parents=True)
    other.write_text("from another system")
    return venv


def _working_venv(env: Path) -> None:
    python = launcher.venv_python(env)
    python.parent.mkdir(parents=True)
    python.write_text("")


def _never_ask(*a):
    pytest.fail("must not ask when nothing needs deciding")


def test_no_environment_means_a_fresh_dot_venv(tmp_path):
    assert launcher.choose_environment(tmp_path, _never_ask) == tmp_path / ".venv"


def test_a_working_environment_is_used_without_asking(tmp_path):
    _working_venv(tmp_path / ".venv")
    assert launcher.choose_environment(tmp_path, _never_ask) == tmp_path / ".venv"


def test_the_default_keeps_the_foreign_venv_and_uses_a_separate_one(tmp_path):
    venv = _foreign_venv(tmp_path)
    chosen = launcher.choose_environment(tmp_path, lambda prompt, default: default)
    assert chosen == tmp_path / f".venv-{launcher.system_tag()}"
    assert venv.exists() and any(venv.rglob("python*"))          # untouched


def test_the_user_can_choose_to_replace_it(tmp_path):
    venv = _foreign_venv(tmp_path)
    chosen = launcher.choose_environment(tmp_path, lambda prompt, default: "R")
    assert chosen == tmp_path / ".venv" and not venv.exists()


def test_the_prompt_explains_what_is_wrong_and_the_two_choices(tmp_path, capsys):
    _foreign_venv(tmp_path)
    seen = {}
    launcher.choose_environment(tmp_path, lambda prompt, default: seen.update(p=prompt) or "k")
    assert "[R] Replace" in seen["p"] and "[K] Keep" in seen["p"]
    assert "cannot be used on this computer" in capsys.readouterr().out


def test_if_deleting_fails_the_old_one_is_left_alone_and_a_separate_one_is_used(tmp_path, monkeypatch):
    venv = _foreign_venv(tmp_path)
    monkeypatch.setattr(launcher.shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(
        OSError("Directory is not empty")))
    chosen = launcher.choose_environment(tmp_path, lambda prompt, default: "r")
    assert chosen == tmp_path / f".venv-{launcher.system_tag()}" and venv.exists()


def test_the_separate_environment_is_remembered_so_nobody_is_asked_twice(tmp_path):
    _foreign_venv(tmp_path)
    _working_venv(tmp_path / f".venv-{launcher.system_tag()}")
    assert launcher.choose_environment(tmp_path, _never_ask) == \
        tmp_path / f".venv-{launcher.system_tag()}"


def test_no_keyboard_means_keep(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", None)
    assert launcher.ask_choice("Choose R or K [K]: ", "k") == "k"
    assert "no keyboard" in capsys.readouterr().out


def test_uv_is_never_allowed_to_ask_its_own_question():
    source = Path(launcher.__file__).read_text()
    assert '"--clear"' in source and "NO_COLOR" in source


# --- Wine older than 10.1 cannot run numpy >= 2.1 (missing crealf/cimagf) ------------------


def test_wine_versions_are_read_from_their_text():
    assert launcher.parse_wine_version("10.0") == (10, 0)
    assert launcher.parse_wine_version("wine-10.1 (Staging)") == (10, 1)
    assert launcher.parse_wine_version("") == (0,)


def test_old_wine_limits_numpy_and_new_wine_or_real_windows_does_not(monkeypatch):
    monkeypatch.delenv("HIVE_EXTRA_CONSTRAINTS", raising=False)
    for version, expected in (((10, 0), ["numpy<2.1"]), ((9, 22), ["numpy<2.1"]),
                              ((10, 1), []), ((11, 0), []), (None, [])):
        monkeypatch.setattr(launcher, "wine_version", lambda v=version: v)
        assert launcher.active_constraints() == expected, version


def test_extra_constraints_can_be_given_in_the_environment(monkeypatch):
    monkeypatch.setattr(launcher, "wine_version", lambda: None)
    monkeypatch.setenv("HIVE_EXTRA_CONSTRAINTS", "numpy<2.1, scipy<2")
    assert launcher.active_constraints() == ["numpy<2.1", "scipy<2"]


def test_a_new_limit_forces_a_reinstall_so_a_broken_environment_is_repaired(monkeypatch, tmp_path):
    (tmp_path / "requirements.txt").write_text("numpy\n")
    monkeypatch.setattr(launcher, "wine_version", lambda: None)
    without = launcher.requirements_fingerprint(tmp_path)
    monkeypatch.setattr(launcher, "wine_version", lambda: (10, 0))
    assert launcher.requirements_fingerprint(tmp_path) != without


def test_the_limit_is_passed_to_the_installer(monkeypatch, tmp_path):
    (tmp_path / "requirements.txt").write_text("numpy\n")
    (tmp_path / ".venv").mkdir()            # the stand-in installer does not create it
    monkeypatch.setattr(launcher, "wine_version", lambda: (10, 0))
    monkeypatch.setattr(launcher, "find_uv", lambda folder: Path("uv"))
    commands = []
    monkeypatch.setattr(launcher, "_run", lambda command, **kw: commands.append(
        [str(c) for c in command]))
    launcher.build_environment(tmp_path)
    install = next(c for c in commands if "install" in c)
    assert "-c" in install
    assert (tmp_path / ".hive_tools" / "constraints.txt").read_text().strip() == "numpy<2.1"


def test_no_limit_means_no_constraint_file_is_used(monkeypatch, tmp_path):
    (tmp_path / "requirements.txt").write_text("numpy\n")
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(launcher, "wine_version", lambda: None)
    monkeypatch.delenv("HIVE_EXTRA_CONSTRAINTS", raising=False)
    monkeypatch.setattr(launcher, "find_uv", lambda folder: Path("uv"))
    commands = []
    monkeypatch.setattr(launcher, "_run", lambda command, **kw: commands.append(
        [str(c) for c in command]))
    launcher.build_environment(tmp_path)
    assert all("-c" not in c for c in commands if "install" in c)

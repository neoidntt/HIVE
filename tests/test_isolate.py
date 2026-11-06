"""The analysis runs in its own process: results come back, errors keep their type, and a kill
is reported instead of taking the caller down."""

from __future__ import annotations

import os
import signal

import pytest

from hive.isolate import AnalysisStoppedError, run_isolated


def add(a, b, progress=None):
    if progress:
        progress(0.5, "halfway")
        progress(1.0, "done")
    return {"sum": a + b, "pid": os.getpid()}


def fail(progress=None):
    raise FileNotFoundError("nothing here")


def get_killed(progress=None):
    os.kill(os.getpid(), signal.SIGKILL)


def test_the_result_comes_back_from_another_process_with_progress():
    seen = []
    out = run_isolated(add, 2, 3, progress=lambda fraction, message: seen.append((fraction, message)))
    assert out["sum"] == 5 and out["pid"] != os.getpid()
    assert seen == [(0.5, "halfway"), (1.0, "done")]
    assert run_isolated(add, 1, b=1)["sum"] == 2


def test_an_error_keeps_its_type():
    with pytest.raises(FileNotFoundError, match="nothing here"):
        run_isolated(fail, progress=lambda *a: None)


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL")
def test_a_killed_analysis_is_reported_and_the_caller_survives():
    with pytest.raises(AnalysisStoppedError, match="ran out of memory"):
        run_isolated(get_killed, progress=lambda *a: None)
    assert run_isolated(add, 1, 1)["sum"] == 2             # and it can be used again


def test_the_main_script_is_not_run_again_in_the_analysis_process(tmp_path, monkeypatch):
    """Under Streamlit the "main module" is app.py; a spawned process must not re-run it."""
    import sys
    import types

    marker = tmp_path / "ran"
    script = tmp_path / "pretend_app.py"
    script.write_text(f"open({str(marker)!r}, 'w').write('the main script ran again')\n")
    main = types.ModuleType("__main__")
    main.__file__ = str(script)                    # what Streamlit's stand-in for __main__ looks like
    monkeypatch.setitem(sys.modules, "__main__", main)
    assert run_isolated(add, 1, 2)["sum"] == 3
    assert not marker.exists()
    assert sys.modules["__main__"] is main          # and the real one is put back

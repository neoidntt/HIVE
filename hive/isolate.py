"""Run a long analysis in a process of its own, so the web app survives if it is killed.

When a computer runs out of memory, Linux (and macOS, and Windows) ends the biggest process.
If the analysis runs inside the web server, the server is the one that goes, and the browser
shows only "Connection error. Is Streamlit still running?". Run in a separate process instead,
the analysis is the biggest process and the first to go; the server stays up and can say
what happened.

The function and its arguments cross to the new process by pickle, and its return value comes
back the same way, so use this for analyses that return tables and file paths (a 2D batch, a
folder of stacks), not for ones that return whole volumes.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import signal
import sys
import types
from typing import Any, Callable


class AnalysisStoppedError(RuntimeError):
    """The analysis process ended without returning a result (it was killed)."""


def _child(connection, function: Callable, args: tuple, kwargs: dict, wants_progress: bool) -> None:
    if hasattr(os, "setpgrp"):
        os.setpgrp()                    # own process group, so the parent can end its workers too
    try:
        if wants_progress:
            kwargs["progress"] = lambda fraction, message: connection.send(
                ("progress", float(fraction), str(message)))
        connection.send(("done", function(*args, **kwargs)))
    except BaseException as exc:  # noqa: BLE001 - everything is reported to the parent
        try:
            pickle.dumps(exc)
            connection.send(("error", exc))
        except Exception:  # noqa: BLE001 - an exception that cannot be pickled
            connection.send(("error", RuntimeError(f"{type(exc).__name__}: {exc}")))


def _end(process) -> None:
    """Stop the process and anything it started."""
    if process.is_alive():
        if hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
        else:
            process.kill()
    process.join(timeout=5)


def run_isolated(function: Callable, *args: Any, progress: Callable[[float, str], None] | None = None,
                 **kwargs: Any) -> Any:
    """Call ``function(*args, **kwargs)`` in a new process and return what it returns.

    Args:
        function: A function defined at the top level of a module (so it can be pickled).
        progress: Optional ``progress(fraction, message)``. It is passed to ``function`` as its
            ``progress`` argument; the calls are made here, in the calling thread.

    Raises:
        AnalysisStoppedError: If the process was killed (almost always: out of memory).
        Exception: Whatever ``function`` raised, with its original type when it can be pickled.
    """
    context = multiprocessing.get_context("spawn")      # a clean process, not a copy of the server
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(sender, function, args, kwargs, progress is not None))
    # A new "spawn" process re-runs the parent's main script. Under Streamlit that "main script" is
    # app.py itself, which must not run again inside the analysis process, so for the moment the
    # process is started the main module is replaced by an empty one (nothing to re-run).
    real_main = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        process.start()
    finally:
        if real_main is not None:
            sys.modules["__main__"] = real_main
    sender.close()
    try:
        while True:
            try:
                ready = receiver.poll(0.25)
            except (EOFError, OSError):
                ready = False
            if ready:
                try:
                    kind, *payload = receiver.recv()
                except (EOFError, OSError):
                    break                                   # it died while sending
                if kind == "progress":
                    if progress is not None:
                        progress(*payload)
                elif kind == "done":
                    return payload[0]
                else:
                    raise payload[0]
            elif not process.is_alive():
                if receiver.poll(0):                         # a last message before it ended
                    continue
                break
    finally:
        _end(process)
        receiver.close()
    code = process.exitcode
    cause = (" It was stopped by the operating system, almost certainly because the computer ran "
             "out of memory." if code == -getattr(signal, "SIGKILL", 9) else "")
    raise AnalysisStoppedError(
        f"The analysis ended unexpectedly (exit code {code}).{cause} The app itself is still "
        "running. Close other programs (browser tabs use a lot), use fewer parallel workers, or "
        "analyse the images in smaller groups, then run it again.")

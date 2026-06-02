# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Regression tests for issue #37.

On some Windows environments (a Jupyter/IPython ``!`` shell-out, git-bash)
the standard output stream uses a legacy code page such as ``cp1252`` that
cannot encode the Unicode box-drawing and spinner glyphs Rich emits. That
raised ``UnicodeEncodeError`` and crashed the CLI before it did any work.

``_build_console`` reconfigures the stream so those glyphs encode cleanly.
The legacy code page is the trigger regardless of platform, so we reproduce
it portably with an in-memory ``cp1252`` stream.
"""

import io
import sys

import pytest
from ngff_zarr.cli import _build_console

# The opening corner of the panel border Rich draws -- not encodable as cp1252.
_BOX_GLYPH = "╭"


def _cp1252_stream():
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")


def _is_cp1252_encodable(ch):
    try:
        ch.encode("cp1252")
    except UnicodeEncodeError:
        return False
    return True


def test_cp1252_cannot_encode_box_glyph():
    """Guard the premise: this glyph genuinely breaks cp1252 (the #37 trigger)."""
    with pytest.raises(UnicodeEncodeError):
        _BOX_GLYPH.encode("cp1252")


def test_build_console_reconfigures_legacy_stdout(monkeypatch):
    """A cp1252 stdout is reconfigured so Rich's glyphs become encodable."""
    monkeypatch.setattr(sys, "stdout", _cp1252_stream())
    assert sys.stdout.encoding.lower() == "cp1252"

    _build_console()

    # After reconfiguration the glyph that crashed the CLI must encode cleanly.
    _BOX_GLYPH.encode(sys.stdout.encoding)


def test_unfixed_cp1252_spinner_render_crashes():
    """Guard the premise: the CLI's spinner glyphs crash an un-fixed cp1252 stream.

    The CLI's initial panel wraps ``Spinner("point", ...)`` whose frames use
    ``∙``/``●`` (positions 0-2 in the issue #37 traceback). ``safe_box`` does
    not substitute those, so on a cp1252 stream the render raises -- this is the
    failure ``_build_console`` exists to prevent.
    """
    from rich.console import Console
    from rich.spinner import Spinner

    stream = _cp1252_stream()
    console = Console(file=stream, force_terminal=True, legacy_windows=True)
    with pytest.raises(UnicodeEncodeError):
        console.print(Spinner("point", text="Loading input..."))
        stream.flush()


def test_build_console_renders_spinner_without_crashing(monkeypatch):
    """After _build_console, the CLI's spinner renders to a legacy stream cleanly."""
    from rich.console import Console
    from rich.spinner import Spinner

    monkeypatch.setattr(sys, "stdout", _cp1252_stream())
    _build_console()

    # Force the terminal/legacy-windows rendering path that produced the
    # original traceback, bound to the (now reconfigured) stdout.
    console = Console(file=sys.stdout, force_terminal=True, legacy_windows=True)
    console.print(Spinner("point", text="Loading input..."))
    sys.stdout.flush()

    output = sys.stdout.buffer.getvalue().decode("utf-8")
    # Rich emitted spinner glyphs that cp1252 cannot encode -- exactly the
    # characters that crashed the CLI before the stream was reconfigured.
    assert any(not _is_cp1252_encodable(ch) for ch in output)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-32"])
def test_build_console_leaves_utf_stdout_untouched(monkeypatch, encoding):
    """Any capable UTF variant is left as-is and must not be reconfigured."""
    calls = []
    stream = io.TextIOWrapper(io.BytesIO(), encoding=encoding)

    original_reconfigure = stream.reconfigure

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original_reconfigure(*args, **kwargs)

    monkeypatch.setattr(stream, "reconfigure", spy)
    monkeypatch.setattr(sys, "stdout", stream)

    _build_console()

    assert calls == []


def test_build_console_without_reconfigure_does_not_raise(monkeypatch):
    """A legacy stream lacking ``reconfigure`` degrades gracefully, no crash."""
    stream = _cp1252_stream()
    # ``getattr(stream, "reconfigure", None)`` now returns None.
    monkeypatch.setattr(stream, "reconfigure", None)
    monkeypatch.setattr(sys, "stdout", stream)

    _build_console()

    # The stream could not be switched, so it stays cp1252 -- but no exception.
    assert stream.encoding.lower() == "cp1252"


def test_build_console_falls_back_to_errors_replace(monkeypatch):
    """If switching to UTF-8 fails, fall back to replacing un-encodable chars."""
    stream = _cp1252_stream()
    calls = []

    def reconfigure(*args, **kwargs):
        calls.append(kwargs)
        if "encoding" in kwargs:
            raise OSError("cannot change encoding on this stream")

    monkeypatch.setattr(stream, "reconfigure", reconfigure)
    monkeypatch.setattr(sys, "stdout", stream)

    _build_console()

    # First the UTF-8 switch is attempted, then the replace fallback.
    assert {"encoding": "utf-8"} in calls
    assert {"errors": "replace"} in calls


def test_build_console_swallows_reconfigure_failures(monkeypatch):
    """When neither reconfigure attempt works, the CLI still gets a console."""
    stream = _cp1252_stream()

    def reconfigure(*args, **kwargs):
        raise ValueError("stream refuses reconfiguration")

    monkeypatch.setattr(stream, "reconfigure", reconfigure)
    monkeypatch.setattr(sys, "stdout", stream)

    # Both attempts raise; the failures must be swallowed and a Console returned.
    console = _build_console()

    assert console is not None

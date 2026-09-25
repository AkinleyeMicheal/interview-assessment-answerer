"""Passive clipboard capture for interview context.

The candidate copies the problem statement / starter code with their OWN
physical keyboard (nothing is ever injected — injected keystrokes carry the
LLKHF_INJECTED flag and are distinguishable by proctoring hooks), then taps
Problem / Code on the phone. Reading the clipboard from this process uses
ordinary Win32 clipboard APIs and is invisible to web pages: a browser can
only observe paste events fired inside its own document, never external
readers of the system clipboard.

ctypes only — no pywin32 dependency.
"""

from __future__ import annotations

import ctypes
import logging

log = logging.getLogger("quizbox.clipboard")

CF_UNICODETEXT = 13
_OVERLAP_SCAN = 300  # chars scanned at the seam when merging chunked copies
_MIN_OVERLAP = 4     # ignore 1-3 char matches: they glue unrelated chunks

if hasattr(ctypes, "windll"):
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    _user32.OpenClipboard.restype = ctypes.c_bool
    _user32.CloseClipboard.restype = ctypes.c_bool
    _user32.EmptyClipboard.restype = ctypes.c_bool
    _user32.GetClipboardData.argtypes = [ctypes.c_uint]
    _user32.GetClipboardData.restype = ctypes.c_void_p
    _user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    _user32.SetClipboardData.restype = ctypes.c_void_p
    _kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    _kernel32.GlobalAlloc.restype = ctypes.c_void_p
    _kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    _kernel32.GlobalLock.restype = ctypes.c_void_p
    _kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    _kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
else:  # non-Windows (CI) — every operation degrades to a no-op
    _user32 = None
    _kernel32 = None

GMEM_MOVEABLE = 0x0002


def _open() -> bool:
    """Open the clipboard, retrying briefly while another app holds it."""
    if _user32 is None:
        return False
    for _ in range(5):
        if _user32.OpenClipboard(None):
            return True
        import time
        time.sleep(0.04)
    return False


def read_text() -> str:
    """Return the current CF_UNICODETEXT text, or "" when empty/unavailable."""
    if not _open():
        return ""
    try:
        handle = _user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        ptr = _kernel32.GlobalLock(handle)
        if not ptr:
            return ""
        try:
            return ctypes.c_wchar_p(ptr).value or ""
        finally:
            _kernel32.GlobalUnlock(handle)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("clipboard read failed: %s", exc)
        return ""
    finally:
        _user32.CloseClipboard()


def write_text(text: str) -> bool:
    """Replace the clipboard with text (used by tests and the seed helper)."""
    if _user32 is None or not text:
        return False
    if not _open():
        return False
    try:
        if not _user32.EmptyClipboard():
            return False
        data = (text + "\0").encode("utf-16-le")
        size = len(data)
        handle = _kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            return False
        ptr = _kernel32.GlobalLock(handle)
        if not ptr:
            _kernel32.GlobalFree(handle)
            return False
        ctypes.memmove(ptr, data, size)
        _kernel32.GlobalUnlock(handle)
        return bool(_user32.SetClipboardData(CF_UNICODETEXT, handle))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("clipboard write failed: %s", exc)
        return False
    finally:
        _user32.CloseClipboard()


def merge_stash(existing: str, new: str) -> str:
    """Merge a freshly copied clipboard chunk into an existing stash.

    Handles the three real shapes of a chunked problem copy:
    - exact re-copy or contained text -> keep the richer version
    - adjacent chunks with an overlapping seam -> append without duplicating
    - unrelated text -> append on a new line (multi-part statements)
    """
    existing = (existing or "").strip()
    new = (new or "").strip()
    if not new:
        return existing
    if not existing:
        return new
    if new in existing:
        return existing
    if existing in new:
        return new

    tail = existing[-_OVERLAP_SCAN:]
    head = new[:_OVERLAP_SCAN]
    max_overlap = min(len(tail), len(head))
    for size in range(max_overlap, _MIN_OVERLAP - 1, -1):
        if tail[-size:] == head[:size]:
            return existing + new[size:]
    return existing + "\n" + new

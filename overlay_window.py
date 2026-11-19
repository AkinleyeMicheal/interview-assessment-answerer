"""Desktop overlay pane — mirrors the phone UI on top of everything on the PC.

The pane is a frameless, always-on-top, taskbar/Alt-Tab-hidden Qt window
hosting a QtWebEngine view of the same Quizbox app (``/?display=overlay``).
It is read-only by default — every mouse event passes through to the window
beneath — but an explicit **Scroll mode** unlock (``set_interact``) temporarily
makes the answer/conversation panels wheel- and scrollbar-scrollable; the
unlock auto re-locks on Hide, close, or client disconnect.  It is controlled
from the phone's Mirror tab over WebSocket; every control lands in this
module's command queue and executes on a single dedicated GUI thread.

Design notes
------------
- Qt is imported ONLY on the GUI thread (lazy), so importing this module in
  tests or server startup costs nothing.
- ``QTWEBENGINE_CHROMIUM_FLAGS`` is set at import time, BEFORE any Qt/WebEngine
  import: ``--ignore-certificate-errors`` (the server uses a self-signed cert)
  and ``--disable-gpu`` (software rendering is plenty for a mirror page).
- State math (geometry clamping, zoom/opacity bounds) works with no Qt at all
  (``_win is None``), so unit tests never spawn a window.
- The global hotkey (Ctrl+Shift+`) runs on its own plain Win32 thread with a
  GetMessage loop and enqueues toggle commands like any other controller op.
"""

from __future__ import annotations

import ctypes
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future

log = logging.getLogger("quizbox.overlay")

# Must be set before QtWebEngine initializes (module import happens early).
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--ignore-certificate-errors --disable-gpu",
)

# --- hotkey (Win32) -------------------------------------------------------
WM_HOTKEY = 0x0312
HOTKEY_ID = 0x5151
MOD_NOREPEAT = 0x4000
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
VK_OEM_3 = 0xC0  # `~ key on US layouts

# --- pane limits ----------------------------------------------------------
MIN_W, MIN_H = 160, 120
MAX_W, MAX_H = 4096, 2160
SIZE_STEP = 40
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.5, 2.0, 0.1
# Pane starts smaller so desktop text/buttons fit a laptop screen; the
# Mirror-tab Zoom ± control still adjusts this live (phone UI unaffected).
DEFAULT_ZOOM = 0.75
OPACITY_MIN, OPACITY_MAX, OPACITY_STEP = 0.2, 1.0, 0.1
PRESETS: dict[str, tuple[int, int]] = {
    "tiny": (240, 150),
    "compact": (420, 560),
    "standard": (768, 800),
}
DEFAULT_W, DEFAULT_H = 768, 800
# Keep at least this much of the pane on screen when clamping position.
MIN_VISIBLE = 80
HOTKEY_LABEL = "Ctrl+Shift+`"
# Windows: the window shows on the monitor only — it never appears in any
# screen capture (duplicates, screenshots, recordings).  Keeps the pane from
# feeding its own mirrored image back into the captured frame (infinite
# mirror / Droste feedback loop).  Win10 2004+ (user is on Windows 11).
WDA_EXCLUDEFROMCAPTURE = 0x00000011

OVERLAY_URL = "https://127.0.0.1:8000/?display=overlay"

_STOP = object()


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _set_capture_exclusion(hwnd: int, user32=None) -> bool:
    """Make the window invisible to every screen-capture API (WDA_EXCLUDE...).

    Callable from any thread with a live HWND; returns False on failure
    (older Windows, foreign window) — callers degrade gracefully.
    """
    if user32 is None:
        user32 = ctypes.windll.user32
    try:
        ok = bool(user32.SetWindowDisplayAffinity(int(hwnd), WDA_EXCLUDEFROMCAPTURE))
    except Exception as exc:  # noqa: BLE001
        log.warning("SetWindowDisplayAffinity(0x%x) raised for hwnd=%s: %s",
                    WDA_EXCLUDEFROMCAPTURE, hwnd, exc)
        return False
    if not ok:
        err = ctypes.windll.kernel32.GetLastError()
        log.warning("SetWindowDisplayAffinity(0x%x) failed for hwnd=%s (err=%s)",
                    WDA_EXCLUDEFROMCAPTURE, hwnd, err)
    return ok


class OverlayController:
    """Thread-safe façade over the Qt overlay window.

    Public mutators (``open``, ``close``, ``move_by``, ...) are executed on the
    GUI thread via :meth:`submit`; each bumps an internal revision and, when
    ``on_state`` is set, the fresh state dict is handed to that callback (the
    server broadcasts it as an ``overlay_state`` message to every client).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._queue: "queue.Queue" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._hotkey_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._ready = threading.Event()
        self._failure: str | None = None
        self._start_lock = threading.Lock()
        self._rev = 0

        self._app = None      # QApplication, GUI thread only
        self._win = None      # pane QWidget, GUI thread only

        self.url = OVERLAY_URL
        # Called (on the GUI thread) after any state change.
        self.on_state = None
        self._state: dict = {
            "open": False,
            "hidden": False,
            "x": 0,
            "y": 0,
            "w": DEFAULT_W,
            "h": DEFAULT_H,
            "opacity": 1.0,
            "zoom": DEFAULT_ZOOM,
            "positioned": False,
            "hotkey": HOTKEY_LABEL,
            "hotkey_registered": False,
            "capture_excluded": False,
            "interact": False,
        }

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self, timeout: float = 25.0) -> bool:
        """Boot the GUI thread (idempotent).  Returns False on Qt failure."""
        if self._ready.is_set():
            return self._failure is None
        if self._thread is not None:
            return self._ready.wait(timeout) and self._failure is None
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._gui_main, name="qt-gui", daemon=True
                )
                self._thread.start()
        if not self._ready.wait(timeout):
            return False
        return self._failure is None

    def _gui_main(self) -> None:
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication

            # WebEngine requires this attribute BEFORE QApplication exists.
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
            self._app = QApplication.instance() or QApplication([])
            self._hotkey_thread = threading.Thread(
                target=self._hotkey_main, name="overlay-hotkey", daemon=True
            )
            self._hotkey_thread.start()
            self._ready.set()

            while not self._stop_evt.is_set():
                self._drain_once()
                self._app.processEvents()
                time.sleep(0.01)
        except Exception as exc:  # noqa: BLE001 — record and unblock waiters
            log.error("overlay GUI thread failed: %s", exc)
            self._failure = str(exc)
            self._ready.set()

    def _drain_once(self) -> None:
        """Execute every queued command.  Separated for unit tests."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _STOP:
                return
            fn, args, kwargs, fut = item
            rev_before = self._rev
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if fut is not None:
                    fut.set_exception(exc)
                else:
                    log.error("overlay command %s failed: %s", getattr(fn, "__name__", fn), exc)
                continue
            if fut is not None and not fut.cancelled():
                try:
                    fut.set_result(result)
                except Exception:  # noqa: BLE001
                    pass
            self._notify_if_changed(rev_before)

    def _notify_if_changed(self, rev_before: int) -> None:
        if self._rev == rev_before or self.on_state is None:
            return
        try:
            self.on_state(self.status())
        except Exception as exc:  # noqa: BLE001
            log.error("overlay on_state callback failed: %s", exc)

    def submit(self, fn, *args, **kwargs) -> Future:
        """Run ``fn(*args, **kwargs)`` on the GUI thread; returns its Future."""
        fut: Future = Future()
        if not self.start():
            fut.set_exception(RuntimeError(self._failure or "overlay GUI thread unavailable"))
            return fut
        self._queue.put((fn, args, kwargs, fut))
        return fut

    def stop(self) -> None:
        self._stop_evt.set()

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _set(self, **changes) -> dict:
        with self._lock:
            self._state.update(changes)
            self._rev += 1
            return dict(self._state)

    # ------------------------------------------------------------------
    # geometry (pure state math; Qt applied only when the pane exists)
    # ------------------------------------------------------------------
    def _screen_bounds(self) -> tuple[int, int, int, int]:
        """(x, y, w, h) of the primary screen; sane fallback without Qt."""
        app = self._app
        if app is None:
            return 0, 0, 1920, 1080
        geo = app.primaryScreen().availableGeometry()
        return geo.x(), geo.y(), geo.width(), geo.height()

    def _apply_geometry(self) -> None:
        """GUI thread: push state geometry to the widget (if it exists)."""
        if self._win is None:
            return
        with self._lock:
            st = dict(self._state)
        try:
            self._win.setGeometry(st["x"], st["y"], st["w"], st["h"])
        except Exception as exc:  # noqa: BLE001
            log.error("overlay setGeometry failed: %s", exc)

    def _clamped_position(self, x: int, y: int) -> tuple[int, int]:
        st = dict(self._state)
        sx, sy, sw, sh = self._screen_bounds()
        x = clamp(x, sx + MIN_VISIBLE - st["w"], sx + sw - MIN_VISIBLE)
        y = clamp(y, sy, sy + sh - MIN_VISIBLE)
        return int(x), int(y)

    # ------------------------------------------------------------------
    # commands (executed on the GUI thread via submit())
    # ------------------------------------------------------------------
    def open(self) -> dict:
        st = self.status()
        sx, sy, sw, sh = self._screen_bounds()
        # Clamp size to the screen so the pane is never cut off
        # (the 800-high default hangs off a 768-high display).
        w = clamp(st["w"], MIN_W, min(MAX_W, sw))
        h = clamp(st["h"], MIN_H, min(MAX_H, sh))
        if (w, h) != (st["w"], st["h"]):
            self._set(w=w, h=h)
        if not st.get("positioned"):
            # First open: park on the right edge, fully on-screen.
            x = clamp(sx + sw - w - 40, sx, sx + max(0, sw - w))
            y = clamp(sy + 40, sy, sy + max(0, sh - h))
            self._set(x=int(x), y=int(y), positioned=True)
        win = self._ensure_window()
        self._apply_geometry()
        win.open_pane()
        self._apply_capture_exclusion()
        return self._set(open=True, hidden=False, interact=False)

    def close(self) -> dict:
        win = self._win
        if win is not None:
            try:
                win.close_pane()
            except Exception as exc:  # noqa: BLE001
                log.error("overlay close failed: %s", exc)
            self._win = None
        return self._set(
            open=False, hidden=False, capture_excluded=False, interact=False
        )

    def toggle(self) -> dict:
        return self.close() if self.status()["open"] else self.open()

    def move_by(self, dx: int = 0, dy: int = 0) -> dict:
        st = self.status()
        x, y = self._clamped_position(st["x"] + int(dx), st["y"] + int(dy))
        state = self._set(x=x, y=y, positioned=True)
        self._apply_geometry()
        return state

    def move_to(self, x: int, y: int) -> dict:
        x, y = self._clamped_position(int(x), int(y))
        state = self._set(x=x, y=y, positioned=True)
        self._apply_geometry()
        return state

    def resize(self, w: int | None = None, h: int | None = None) -> dict:
        st = self.status()
        sx, sy, sw, sh = self._screen_bounds()
        new_w = clamp(int(w if w is not None else st["w"]), MIN_W, min(MAX_W, sw))
        new_h = clamp(int(h if h is not None else st["h"]), MIN_H, min(MAX_H, sh))
        x, y = self._clamped_position(st["x"], st["y"])  # re-clamp for new size
        state = self._set(w=new_w, h=new_h, x=x, y=y)
        self._apply_geometry()
        return state

    def step(self, axis: str = "w", delta: int = SIZE_STEP) -> dict:
        st = self.status()
        delta = int(delta)
        if axis == "h":
            return self.resize(h=st["h"] + delta)
        return self.resize(w=st["w"] + delta)

    def preset(self, name: str = "standard") -> dict:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name!r} (expected one of {sorted(PRESETS)})")
        w, h = PRESETS[name]
        return self.resize(w=w, h=h)

    def center(self) -> dict:
        st = self.status()
        sx, sy, sw, sh = self._screen_bounds()
        return self.move_to(sx + (sw - st["w"]) // 2, sy + (sh - st["h"]) // 2)

    def zoom_step(self, delta: float = ZOOM_STEP) -> dict:
        st = self.status()
        zoom = round(clamp(st["zoom"] + float(delta), ZOOM_MIN, ZOOM_MAX), 2)
        state = self._set(zoom=zoom)
        self._apply_zoom(zoom)
        return state

    def opacity_step(self, delta: float = OPACITY_STEP) -> dict:
        st = self.status()
        opacity = round(clamp(st["opacity"] + float(delta), OPACITY_MIN, OPACITY_MAX), 2)
        state = self._set(opacity=opacity)
        self._apply_opacity(opacity)
        return state

    def set_hidden(self, enabled: bool | None = None) -> dict:
        st = self.status()
        if not st["open"]:
            # hide-request while closed just opens (panic hotkey semantics)
            if enabled in (None, False):
                return self.open()
            return st
        hidden = (not st["hidden"]) if enabled is None else bool(enabled)
        state = self._set(hidden=hidden)
        win = self._win
        if win is not None:
            if hidden and state.get("interact"):
                # Re-lock BEFORE hiding: Qt only applies window-flag changes
                # to the platform window while the pane is visible (a
                # setWindowFlag on a hidden widget defers the HWND update to
                # the next show, leaving the hidden window unlocked).
                try:
                    state = self.set_interact(False)
                except Exception as exc:  # noqa: BLE001
                    log.error("overlay scroll re-lock failed: %s", exc)
                    state = self._set(interact=False)
            try:
                win.set_hidden(hidden)
            except Exception as exc:  # noqa: BLE001
                log.error("overlay hide failed: %s", exc)
        return state

    def toggle_hidden(self) -> dict:
        return self.set_hidden(None)

    def set_interact(self, enabled: bool) -> dict:
        """Unlock (or restore) native input for the pane's scroll surfaces.

        Scroll mode: dropping ``WindowTransparentForInput`` lets the mouse
        wheel and the native scrollbars of the answer/conversation panels
        work directly on the pane (the page re-enables ``pointer-events``
        for those two surfaces only).  The unlock is fail-closed — it is
        forced back to locked by :meth:`set_hidden` (Hide/hotkey),
        :meth:`close`, and the server on client disconnect.
        """
        win = self._win
        if win is None:
            raise RuntimeError("pane is not open")
        enabled = bool(enabled)
        win.set_interact(enabled)
        win.run_js(
            "document.body.classList.toggle('iv-scrollon', %s)"
            % ("true" if enabled else "false")
        )
        return self._set(interact=enabled)

    # ------------------------------------------------------------------
    # Qt application (GUI thread only, lazy)
    # ------------------------------------------------------------------
    def _ensure_window(self):
        if self._win is not None:
            return self._win
        self._win = _build_pane(self)
        return self._win

    def _apply_zoom(self, zoom: float) -> None:
        win = self._win
        if win is None:
            return
        try:
            win.set_zoom(zoom)
        except Exception as exc:  # noqa: BLE001
            log.error("overlay zoom failed: %s", exc)

    def _apply_opacity(self, opacity: float) -> None:
        win = self._win
        if win is None:
            return
        try:
            win.set_opacity(opacity)
        except Exception as exc:  # noqa: BLE001
            log.error("overlay opacity failed: %s", exc)

    def _apply_capture_exclusion(self) -> None:
        """GUI thread: hide the pane from every screen capture (WDA_EXCLUDE...).

        The pane floats over the very screen it mirrors — without this the
        captured frame contains the previous mirror image and the feed spirals
        into an infinite-mirror (Droste) loop.  Re-applied on every open()
        because the widget is destroyed on close (WA_DeleteOnClose).
        """
        win = self._win
        if win is None:
            return
        try:
            hwnd = int(win.winId())
        except Exception as exc:  # noqa: BLE001
            log.warning("capture exclusion: no hwnd yet (%s)", exc)
            return
        self._set(capture_excluded=_set_capture_exclusion(hwnd))

    # ------------------------------------------------------------------
    # global hotkey thread (plain Win32, no Qt)
    # ------------------------------------------------------------------
    def _hotkey_main(self) -> None:
        try:
            import ctypes.wintypes  # noqa: F401 — must be imported explicitly

            user32 = ctypes.windll.user32
            if not user32.RegisterHotKey(
                None, HOTKEY_ID, MOD_NOREPEAT | MOD_CONTROL | MOD_SHIFT, VK_OEM_3
            ):
                log.warning("overlay hotkey Ctrl+Shift+` could not be registered")
                with self._lock:
                    self._state["hotkey_registered"] = False
                return
            with self._lock:
                self._state["hotkey_registered"] = True
                self._rev += 1
            msg = ctypes.wintypes.MSG()
            while not self._stop_evt.is_set():
                r = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if r in (0, -1):
                    break
                if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                    self._queue.put((self.toggle_hidden, (), {}, None))
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except Exception as exc:  # noqa: BLE001
            log.error("overlay hotkey thread failed: %s", exc)
        finally:
            try:
                ctypes.windll.user32.UnregisterHotKey(None, HOTKEY_ID)
            except Exception:  # noqa: BLE001
                pass


def _build_pane(controller: OverlayController):
    """Create the Qt pane widget.  GUI thread only."""
    from PySide6.QtCore import Qt, QUrl
    from PySide6.QtWidgets import QVBoxLayout, QWidget
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebEngineWidgets import QWebEngineView

    class _Page(QWebEnginePage):
        def certificateError(self, error):  # noqa: N802 — Qt virtual
            try:
                error.acceptCertificate()
                return True
            except Exception:  # noqa: BLE001
                return False

    class _Pane(QWidget):
        def __init__(self, url: str):
            flags = (
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool  # hidden from taskbar + Alt+Tab
                | Qt.WindowType.WindowTransparentForInput  # permanent read-only
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
            super().__init__(None, flags)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self._view = QWebEngineView(self)
            self._view.setPage(_Page(self._view))
            self._view.setUrl(QUrl(url))
            layout.addWidget(self._view)

        # --- controls -------------------------------------------------
        def open_pane(self):
            self.show()
            self.raise_()

        def close_pane(self):
            self.close()

        def set_hidden(self, hidden: bool):
            self.hide() if hidden else self.show()

        def set_zoom(self, zoom: float):
            self._view.setZoomFactor(zoom)

        def set_interact(self, enabled: bool):
            # Scroll mode: toggle Qt's input-transparency flag at runtime.
            # setWindowFlag hides the widget, so re-show exactly when it was
            # visible before (WA_ShowWithoutActivating + the persistent
            # WindowDoesNotAcceptFocus flag keep focus on the app beneath).
            was_visible = self.isVisible()
            self.setWindowFlag(
                Qt.WindowType.WindowTransparentForInput, not enabled
            )
            if was_visible:
                self.show()
                self.raise_()

        def set_opacity(self, opacity: float):
            self.setWindowOpacity(opacity)

        def run_js(self, script: str):
            self._view.page().runJavaScript(script)

    pane = _Pane(controller.url)
    with controller._lock:
        st = dict(controller._state)
    pane.setGeometry(st["x"], st["y"], st["w"], st["h"])
    pane.setWindowOpacity(st["opacity"])
    pane.set_zoom(st["zoom"])
    return pane


_controller: OverlayController | None = None
_controller_lock = threading.Lock()


def get_overlay() -> OverlayController:
    global _controller
    if _controller is None:
        with _controller_lock:
            if _controller is None:
                _controller = OverlayController()
    return _controller

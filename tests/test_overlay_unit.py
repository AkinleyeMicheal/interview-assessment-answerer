"""Unit tests for the desktop overlay controller.

These NEVER spawn Qt: state math runs with ``_win is None`` and the queue is
drained directly via ``_drain_once``.  A FakePane stands in for the widget.
"""

import os
import queue
import sys
from concurrent.futures import Future

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from overlay_window import (
    DEFAULT_ZOOM,
    MIN_H,
    MIN_W,
    OPACITY_MAX,
    OPACITY_MIN,
    PRESETS,
    WDA_EXCLUDEFROMCAPTURE,
    ZOOM_MAX,
    ZOOM_MIN,
    OverlayController,
    _set_capture_exclusion,
    clamp,
    get_overlay,
)


class FakePane:
    def __init__(self):
        self.calls = []
        self.hwnd = 1234

    def winId(self):
        return self.hwnd

    def open_pane(self):
        self.calls.append("open")

    def close_pane(self):
        self.calls.append("close")

    def set_hidden(self, hidden):
        self.calls.append(("hidden", hidden))

    def set_zoom(self, zoom):
        self.calls.append(("zoom", zoom))

    def set_opacity(self, opacity):
        self.calls.append(("opacity", opacity))

    def set_interact(self, enabled):
        self.calls.append(("interact", bool(enabled)))

    def run_js(self, script):
        self.calls.append(("js", script))

    def setGeometry(self, *geo):
        self.calls.append(("geometry", geo))


@pytest.fixture()
def ctrl():
    c = OverlayController()
    yield c
    c.stop()


def _install_fake_pane(c):
    pane = FakePane()

    def fake_ensure():
        c._win = pane
        return pane

    c._ensure_window = fake_ensure
    return pane


class TestHelpers:
    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_presets_within_limits(self):
        for name, (w, h) in PRESETS.items():
            assert w >= MIN_W, name
            assert h >= MIN_H, name

    def test_standard_preset_is_default_size(self):
        from overlay_window import DEFAULT_H, DEFAULT_W

        assert PRESETS["standard"] == (DEFAULT_W, DEFAULT_H)
        assert (DEFAULT_W, DEFAULT_H) == (768, 800)  # +20% over the old 640


class TestGeometryState:
    def test_initial_status(self, ctrl):
        st = ctrl.status()
        assert st["open"] is False
        assert st["hidden"] is False
        assert "click_through" not in st  # pane is permanently read-only
        assert st["capture_excluded"] is False
        assert st["zoom"] == DEFAULT_ZOOM == 0.75  # pane starts compact
        assert st["opacity"] == 1.0
        assert st["hotkey"] == "Ctrl+Shift+`"

    def test_resize_clamps_to_min(self, ctrl):
        st = ctrl.resize(w=10, h=5)
        assert st["w"] == MIN_W
        assert st["h"] == MIN_H

    def test_resize_clamps_to_screen(self, ctrl):
        # without Qt the fallback screen is 1920x1080, so the screen clamp
        # (min(MAX_W, screen_w)) binds before MAX_W does
        st = ctrl.resize(w=99999, h=99999)
        assert st["w"] == 1920
        assert st["h"] <= 2160

    def test_step_axis(self, ctrl):
        st = ctrl.step("w", 40)
        assert st["w"] == 768 + 40
        st = ctrl.step("h", -40)
        assert st["h"] == 800 - 40

    def test_preset_known(self, ctrl):
        st = ctrl.preset("tiny")
        assert (st["w"], st["h"]) == PRESETS["tiny"]

    def test_preset_unknown_raises(self, ctrl):
        with pytest.raises(ValueError):
            ctrl.preset("gigantic")

    def test_zoom_clamps(self, ctrl):
        for _ in range(30):
            st = ctrl.zoom_step(0.1)
        assert st["zoom"] == ZOOM_MAX
        for _ in range(30):
            st = ctrl.zoom_step(-0.1)
        assert st["zoom"] == ZOOM_MIN

    def test_opacity_clamps(self, ctrl):
        for _ in range(20):
            st = ctrl.opacity_step(-0.1)
        assert st["opacity"] == OPACITY_MIN
        for _ in range(20):
            st = ctrl.opacity_step(0.1)
        assert st["opacity"] == OPACITY_MAX

    def test_move_clamps_to_screen(self, ctrl):
        # default screen fallback is 1920x1080; y must stay >= 0
        st = ctrl.move_by(-100000, -100000)
        assert st["x"] <= 1920 - 80
        assert st["y"] >= 0
        # x lower bound: x >= MIN_VISIBLE - w
        st = ctrl.move_by(-100000, 0)
        assert st["x"] >= 80 - ctrl.status()["w"]

    def test_center(self, ctrl):
        st = ctrl.center()
        assert st["x"] > 0
        assert st["y"] > 0


class TestPaneCommands:
    def test_open_close_with_fake_pane(self, ctrl):
        pane = _install_fake_pane(ctrl)
        st = ctrl.open()
        assert st["open"] is True
        assert "open" in pane.calls
        st = ctrl.close()
        assert st["open"] is False
        assert "close" in pane.calls
        assert ctrl._win is None

    def test_open_parks_on_right_edge(self, ctrl):
        _install_fake_pane(ctrl)
        st = ctrl.open()
        # x = screen(1920) - w(768) - 40 = 1112
        assert st["x"] == 1920 - 768 - 40
        assert st["positioned"] is True

    def test_open_clamps_size_and_parks_fully_on_screen(self, ctrl):
        _install_fake_pane(ctrl)
        ctrl._screen_bounds = lambda: (0, 0, 1366, 768)  # user's display
        st = ctrl.open()
        # the 800-high default must be clamped to the screen height
        assert (st["w"], st["h"]) == (768, 768)
        assert st["x"] + st["w"] <= 1366
        assert st["y"] + st["h"] <= 768
        assert st["x"] == 1366 - 768 - 40  # right-edge park preserved

    def test_open_applies_capture_exclusion(self, ctrl, monkeypatch):
        pane = _install_fake_pane(ctrl)
        seen = []
        monkeypatch.setattr(
            "overlay_window._set_capture_exclusion",
            lambda hwnd, user32=None: seen.append(hwnd) or True,
        )
        st = ctrl.open()
        assert seen == [pane.hwnd]
        assert st["capture_excluded"] is True

    def test_capture_exclusion_failure_degrades(self, ctrl, monkeypatch):
        _install_fake_pane(ctrl)
        monkeypatch.setattr(
            "overlay_window._set_capture_exclusion",
            lambda hwnd, user32=None: False,
        )
        st = ctrl.open()
        assert st["open"] is True
        assert st["capture_excluded"] is False

    def test_close_resets_capture_excluded(self, ctrl, monkeypatch):
        _install_fake_pane(ctrl)
        monkeypatch.setattr(
            "overlay_window._set_capture_exclusion",
            lambda hwnd, user32=None: True,
        )
        ctrl.open()
        st = ctrl.close()
        assert st["capture_excluded"] is False

    def test_zoom_and_opacity_reach_pane(self, ctrl):
        pane = _install_fake_pane(ctrl)
        ctrl.open()
        ctrl.zoom_step(0.1)
        ctrl.opacity_step(-0.1)
        assert ("zoom", 0.85) in pane.calls  # 0.75 default + 0.1 step
        assert ("opacity", 0.9) in pane.calls

    def test_no_click_through_mutation_exists(self, ctrl):
        assert not hasattr(ctrl, "set_click_through")

    def test_hidden_toggles(self, ctrl):
        pane = _install_fake_pane(ctrl)
        ctrl.open()
        st = ctrl.toggle_hidden()
        assert st["hidden"] is True
        st = ctrl.toggle_hidden()
        assert st["hidden"] is False
        assert ("hidden", True) in pane.calls
        assert ("hidden", False) in pane.calls

    def test_hide_while_closed_opens(self, ctrl):
        _install_fake_pane(ctrl)
        st = ctrl.toggle_hidden()
        assert st["open"] is True  # panic hotkey semantics: reveal pane

    def test_explicit_hide_while_closed_stays_closed(self, ctrl):
        _install_fake_pane(ctrl)
        st = ctrl.set_hidden(True)
        assert st["open"] is False


def _js_scripts(pane):
    return [c[1] for c in pane.calls if isinstance(c, tuple) and c[0] == "js"]


class TestScrollMode:
    """Scroll mode: unlock native wheel/scrollbar input on the pane."""

    def test_interact_unlock_and_relock(self, ctrl):
        pane = _install_fake_pane(ctrl)
        ctrl.open()
        st = ctrl.set_interact(True)
        assert st["interact"] is True
        assert ("interact", True) in pane.calls
        assert any("iv-scrollon', true" in s for s in _js_scripts(pane))

        st = ctrl.set_interact(False)
        assert st["interact"] is False
        assert ("interact", False) in pane.calls
        assert any("iv-scrollon', false" in s for s in _js_scripts(pane))

    def test_interact_requires_open_pane(self, ctrl):
        _install_fake_pane(ctrl)
        with pytest.raises(RuntimeError, match="not open"):
            ctrl.set_interact(True)

    def test_hide_forces_relock(self, ctrl):
        """Panic Hide must fail-closed back to locked (flag + state + page)."""
        pane = _install_fake_pane(ctrl)
        ctrl.open()
        ctrl.set_interact(True)

        st = ctrl.set_hidden(True)
        assert st["hidden"] is True
        assert st["interact"] is False
        assert ("hidden", True) in pane.calls
        assert ("interact", False) in pane.calls
        assert any("iv-scrollon', false" in s for s in _js_scripts(pane))

    def test_close_resets_interact(self, ctrl):
        pane = _install_fake_pane(ctrl)
        ctrl.open()
        ctrl.set_interact(True)

        st = ctrl.close()
        assert st["open"] is False
        assert st["interact"] is False

    def test_open_starts_locked(self, ctrl):
        _install_fake_pane(ctrl)
        st = ctrl.open()
        assert st["interact"] is False

    def test_status_carries_interact_field(self, ctrl):
        st = ctrl.status()
        assert st["interact"] is False
        assert "click_through" not in st  # pane stays read-only by default


class TestQueueAndNotify:
    def test_drain_executes_and_resolves_future(self, ctrl):
        fut = Future()
        ctrl._queue.put((lambda a, b: a + b, (2, 3), {}, fut))
        ctrl._drain_once()
        assert fut.result(timeout=1) == 5

    def test_drain_propagates_exception(self, ctrl):
        fut = Future()

        def boom():
            raise RuntimeError("kaput")

        ctrl._queue.put((boom, (), {}, fut))
        ctrl._drain_once()
        with pytest.raises(RuntimeError, match="kaput"):
            fut.result(timeout=1)

    def test_state_change_triggers_on_state(self, ctrl):
        seen = []
        ctrl.on_state = seen.append
        ctrl._drain_once()  # empty queue, no-op
        fut = Future()
        ctrl._queue.put((ctrl.move_by, (10, 5), {}, fut))
        ctrl._drain_once()
        fut.result(timeout=1)
        assert len(seen) == 1
        assert seen[0]["x"] == 10
        assert seen[0]["y"] == 5

    def test_read_only_command_does_not_notify(self, ctrl):
        seen = []
        ctrl.on_state = seen.append
        fut = Future()
        ctrl._queue.put((ctrl.status, (), {}, fut))
        ctrl._drain_once()
        fut.result(timeout=1)
        assert seen == []

    def test_submit_fails_fast_when_gui_unavailable(self, ctrl):
        ctrl._ready.set()
        ctrl._failure = "qt exploded"
        fut = ctrl.submit(ctrl.status)
        with pytest.raises(RuntimeError, match="qt exploded"):
            fut.result(timeout=1)


class TestSingleton:
    def test_get_overlay_is_singleton(self):
        assert get_overlay() is get_overlay()
        assert isinstance(get_overlay(), OverlayController)


class TestCaptureExclusionHelper:
    class _Stub:
        def __init__(self, ok):
            self.ok = ok
            self.calls = []

        def SetWindowDisplayAffinity(self, hwnd, affinity):
            self.calls.append((hwnd, affinity))
            return self.ok

    def test_calls_winapi_with_wda_exclude(self):
        stub = self._Stub(True)
        assert _set_capture_exclusion(0x1234, user32=stub) is True
        assert stub.calls == [(0x1234, WDA_EXCLUDEFROMCAPTURE)]
        assert WDA_EXCLUDEFROMCAPTURE == 0x11

    def test_failure_returns_false(self):
        assert _set_capture_exclusion(1, user32=self._Stub(False)) is False

    def test_exception_returns_false(self):
        class _Boom:
            def SetWindowDisplayAffinity(self, hwnd, affinity):
                raise OSError("boom")

        assert _set_capture_exclusion(1, user32=_Boom()) is False

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
    MIN_H,
    MIN_W,
    OPACITY_MAX,
    OPACITY_MIN,
    PRESETS,
    ZOOM_MAX,
    ZOOM_MIN,
    OverlayController,
    clamp,
    get_overlay,
)


class FakePane:
    def __init__(self):
        self.calls = []

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

    def set_click_through(self, enabled):
        self.calls.append(("click_through", enabled))

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


class TestGeometryState:
    def test_initial_status(self, ctrl):
        st = ctrl.status()
        assert st["open"] is False
        assert st["hidden"] is False
        assert st["click_through"] is True
        assert st["zoom"] == 1.0
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
        assert st["w"] == 640 + 40
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
        # x = screen(1920) - w(640) - 40 = 1240
        assert st["x"] == 1920 - 640 - 40
        assert st["positioned"] is True

    def test_zoom_and_opacity_reach_pane(self, ctrl):
        pane = _install_fake_pane(ctrl)
        ctrl.open()
        ctrl.zoom_step(0.1)
        ctrl.opacity_step(-0.1)
        assert ("zoom", 1.1) in pane.calls
        assert ("opacity", 0.9) in pane.calls

    def test_click_through_toggle_reaches_pane(self, ctrl):
        pane = _install_fake_pane(ctrl)
        ctrl.open()
        st = ctrl.set_click_through(False)
        assert st["click_through"] is False
        assert ("click_through", False) in pane.calls

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

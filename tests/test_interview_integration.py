"""Integration tests: REST + WebSocket against the FastAPI app in-process.

External services (audio devices, ElevenLabs, Groq) are replaced with fakes;
the full protocol, pump broadcast, and session wiring are exercised for real.
"""

import os
import queue
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

import interview_session as isession
import server


class FakeSTT:
    def __init__(self):
        self.running = False
        self.sent = []
        self.on_committed = None
        self.on_partial = None
        self.committed_text = ""
        self.commit_calls = 0

    def start(self, on_partial=None, on_committed=None, on_error=None, **kwargs):
        self.running = True
        self.on_committed = on_committed
        self.on_partial = on_partial
        self.kwargs = kwargs

    def stop(self):
        self.running = False

    def send_audio(self, pcm):
        self.sent.append(pcm)

    def commit(self):
        self.commit_calls += 1
        if self.on_committed:
            self.on_committed({"text": self.committed_text})

    def is_running(self):
        return self.running


class FakeCapture:
    def __init__(self):
        self.running = False
        self.callback = None

    def start(self, callback):
        self.running = True
        self.callback = callback
        return {"ok": True, "device": "fake-device"}

    def stop(self):
        self.running = False

    def is_running(self):
        return self.running


class FakeTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text, block=True):
        self.spoken.append(text)

    def stop_playback(self):
        pass


@pytest.fixture()
def client(monkeypatch):
    session = isession.session

    original = {
        "live_stt": session.live_stt,
        "loopback": session.loopback,
        "mic_stt": session.mic_stt,
        "mic_capture": session.mic_capture,
        "tts": session.tts,
    }
    session.live_stt = FakeSTT()
    session.loopback = FakeCapture()
    session.mic_stt = FakeSTT()
    session.mic_capture = FakeCapture()
    session.tts = FakeTTS()

    monkeypatch.setattr(isession, "get_elevenlabs_key", lambda: "test-key")
    monkeypatch.setattr(server, "check_elevenlabs_status",
                        lambda: {"configured": True, "valid": True})

    # make sure nothing is running
    session.stop_live()
    session.stop_mock()
    _drain(session.out_queue)

    with TestClient(server.app) as test_client:
        yield test_client

    session.stop_live()
    session.stop_mock()
    for name, value in original.items():
        setattr(session, name, value)


def _drain(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _recv_until(ws, msg_type, timeout=10.0, actions=()):
    """Receive WS messages until msg_type found (or action result seen)."""
    collected = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = ws.receive_json()
        collected.append(msg)
        if msg.get("type") == msg_type:
            return msg, collected
        if msg.get("type") == "action_result" and msg.get("action") in actions:
            return msg, collected
    raise AssertionError(f"timeout waiting for {msg_type}; got: {collected}")


def _recv_until_all(ws, predicates, timeout=8.0):
    """Collect messages until every predicate(msg) has matched at least once."""
    remaining = list(predicates)
    collected = []
    deadline = time.time() + timeout
    while remaining and time.time() < deadline:
        msg = ws.receive_json()
        collected.append(msg)
        remaining = [p for p in remaining if not p(msg)]
    assert not remaining, f"missing expected messages; got: {collected}"
    return collected


def _recv_mock_listening(ws, timeout=10.0):
    """Receive until the mock session reports listening_answer (mic open)."""
    deadline = time.time() + timeout
    collected = []
    while time.time() < deadline:
        msg = ws.receive_json()
        collected.append(msg)
        if msg.get("type") == "status" and msg.get("state") == "listening_answer":
            return msg, collected
    raise AssertionError(f"timeout waiting for listening_answer; got: {collected}")


class TestREST:
    def test_status_endpoint(self, client):
        r = client.get("/api/interview/status")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] is None
        assert data["elevenlabs"] == {"configured": True, "valid": True}
        assert data["elevenlabs_key_set"] is True

    def test_live_start_stop_rest(self, client):
        r = client.post("/api/interview/live/start")
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True
        assert isession.session.mode == "live"

        r = client.get("/api/interview/status")
        assert r.json()["mode"] == "live"
        assert r.json()["live_running"] is True

        r = client.post("/api/interview/live/stop")
        assert r.status_code == 200
        assert isession.session.mode is None

    def test_mock_start_stop_rest(self, client, monkeypatch):
        monkeypatch.setattr(isession, "generate_mock_question", lambda **kw: "Introduce yourself.")
        r = client.post("/api/interview/mock/start", json={"mic_source": "desktop", "role": "tester"})
        assert r.status_code == 200, r.text
        assert isession.session.mode == "mock"
        client.post("/api/interview/mock/stop")
        assert isession.session.mode is None

    def test_live_start_failure_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(isession, "get_elevenlabs_key", lambda: "")
        r = client.post("/api/interview/live/start")
        assert r.status_code == 400
        assert "ELEVENLABS_API_KEY" in r.json()["detail"]


class TestWebSocket:
    def test_greeting_and_ping(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            greeting = ws.receive_json()
            assert greeting["type"] == "status"
            ws.send_json({"action": "ping"})
            pong = ws.receive_json()
            assert pong["type"] == "pong"

    def test_start_live_via_ws(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()  # greeting
            ws.send_json({"action": "start_live"})
            result, _ = _recv_until(ws, "action_result", actions=("start_live",))
            assert result["ok"] is True
            assert isession.session.mode == "live"

            ws.send_json({"action": "stop_live"})
            result, _ = _recv_until(ws, "action_result", actions=("stop_live",))
            assert result["ok"] is True
            assert isession.session.mode is None

    def test_live_transcript_and_answer_flow(self, client, monkeypatch):
        monkeypatch.setattr(isession, "detect_and_answer", lambda t, **kw: {
            "is_question": True,
            "question": t,
            "answer": "I would answer with a STAR story here.",
        })
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()  # greeting
            ws.send_json({"action": "start_live"})
            _recv_until(ws, "action_result", actions=("start_live",))

            # simulate STT committing a transcript
            isession.session._on_live_committed({"text": "Tell me about a conflict"})

            answer = None
            saw_final_transcript = False
            deadline = time.time() + 10
            while answer is None and time.time() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "transcript" and msg.get("final"):
                    saw_final_transcript = True
                if msg.get("type") == "answer":
                    answer = msg
            assert saw_final_transcript, "no final transcript over websocket"
            assert answer is not None, "no answer over websocket"
            assert answer["answer"] == "I would answer with a STAR story here."
            assert answer["question"] == "Tell me about a conflict"

            ws.send_json({"action": "stop_live"})
            _recv_until(ws, "action_result", actions=("stop_live",))

    def test_mock_flow_via_ws(self, client, monkeypatch):
        questions = iter(["What is your greatest strength?",
                          "Describe a difficult bug you fixed."])
        monkeypatch.setattr(isession, "generate_mock_question", lambda **kw: next(questions))
        monkeypatch.setattr(isession, "evaluate_mock_answer", lambda q, a, **kw: {
            "score": 7, "feedback": "Solid but add metrics.", "improved_answer": "model"
        })

        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()  # greeting
            ws.send_json({"action": "start_mock", "mic_source": "desktop"})
            _recv_until(ws, "action_result", actions=("start_mock",))

            q1, _ = _recv_until(ws, "mock_question")
            assert q1["question_number"] == 1
            assert q1["question"] == "What is your greatest strength?"
            _recv_mock_listening(ws)  # mic opens only after the question is spoken

            # candidate speaks (partial), then presses Done -> commit -> evaluate
            isession.session._on_mock_partial({"text": "My strength is debugging."})
            isession.session.mic_stt.committed_text = "My strength is debugging."
            done = isession.session.finish_mock_answer()
            assert done["ok"] is True

            feedback, _ = _recv_until(ws, "mock_feedback")
            assert feedback["score"] == 7
            assert feedback["user_answer"] == "My strength is debugging."

            q2, _ = _recv_until(ws, "mock_question")
            assert q2["question_number"] == 2

            ws.send_json({"action": "stop_mock"})
            _recv_until(ws, "action_result", actions=("stop_mock",))
            assert isession.session.mode is None

    def test_answer_done_ws_action(self, client, monkeypatch):
        monkeypatch.setattr(isession, "generate_mock_question", lambda **kw: "Q?")
        monkeypatch.setattr(isession, "evaluate_mock_answer", lambda q, a, **kw: {
            "score": 9, "feedback": "great", "improved_answer": "",
        })
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "start_mock", "mic_source": "desktop"})
            _recv_until(ws, "action_result", actions=("start_mock",))
            _recv_until(ws, "mock_question")
            _recv_mock_listening(ws)

            # Done with no speech -> rejected, session keeps running
            ws.send_json({"action": "answer_done"})
            result, _ = _recv_until(ws, "action_result", actions=("answer_done",))
            assert result["ok"] is False
            assert isession.session.mode == "mock"

            # with speech -> accepted, feedback flows
            isession.session._on_mock_partial({"text": "solid answer"})
            isession.session.mic_stt.committed_text = "solid answer"
            ws.send_json({"action": "answer_done"})
            result, _ = _recv_until(ws, "action_result", actions=("answer_done",))
            assert result["ok"] is True
            fb, _ = _recv_until(ws, "mock_feedback")
            assert fb["score"] == 9

            ws.send_json({"action": "stop_mock"})
            _recv_until(ws, "action_result", actions=("stop_mock",))

    def test_solve_screen_ws_action(self, client, monkeypatch):
        import base64 as b64mod
        monkeypatch.setattr(isession, "solve_with_screen", lambda *a, **k: {
            "detected": True, "answer_text": "Walk through edge cases first.",
            "language": "python", "code_solution": "def solve(): ...",
            "next_steps": ["parse input", "handle empty"],
        })
        monkeypatch.setattr(isession, "generate_mock_question", lambda **kw: "Q?")
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "start_mock", "mic_source": "desktop"})
            _recv_until(ws, "action_result", actions=("start_mock",))
            _recv_until(ws, "mock_question")

            image = b64mod.b64encode(b"fake-jpeg-bytes").decode()
            ws.send_json({"action": "solve_screen", "image": image})
            result, _ = _recv_until(ws, "action_result", actions=("solve_screen",))
            assert result["ok"] is True

            solve, _ = _recv_until(ws, "screen_solve")
            assert solve["code"] == "def solve(): ..."
            assert solve["steps"] == ["parse input", "handle empty"]
            assert solve["variant"] == "solve"

            ws.send_json({"action": "next_step", "image": image})
            result, _ = _recv_until(ws, "action_result", actions=("next_step",))
            assert result["ok"] is True
            solve2, _ = _recv_until(ws, "screen_solve")
            assert solve2["variant"] == "next_step"

            ws.send_json({"action": "set_auto_assist", "enabled": False})
            result, _ = _recv_until(ws, "action_result", actions=("set_auto_assist",))
            assert result["auto_screen_assist"] is False

            ws.send_json({"action": "stop_mock"})
            _recv_until(ws, "action_result", actions=("stop_mock",))

    def test_phone_mic_audio_forwarded(self, client, monkeypatch):
        import base64
        monkeypatch.setattr(isession, "generate_mock_question", lambda **kw: "Q?")
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "start_mock", "mic_source": "phone"})
            _recv_until(ws, "action_result", actions=("start_mock",))
            _recv_until(ws, "mock_question")
            _recv_mock_listening(ws)  # phone audio is dropped until listening

            payload = base64.b64encode(b"\x01\x02" * 512).decode()
            ws.send_json({"action": "mic_audio", "data": payload})
            deadline = time.time() + 3
            while time.time() < deadline and not isession.session.mic_stt.sent:
                time.sleep(0.05)
            assert isession.session.mic_stt.sent, "phone mic audio not forwarded"

            ws.send_json({"action": "stop_mock"})
            _recv_until(ws, "action_result", actions=("stop_mock",))

    def test_phone_mic_audio_dropped_while_asking(self, client, monkeypatch):
        import base64
        monkeypatch.setattr(isession, "generate_mock_question", lambda **kw: "Q?")
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "start_mock", "mic_source": "phone"})
            _recv_until(ws, "action_result", actions=("start_mock",))
            # do NOT wait for listening — send while asking/speaking
            isession.session.mock_state = "speaking"
            payload = base64.b64encode(b"\x01\x02" * 512).decode()
            ws.send_json({"action": "mic_audio", "data": payload})
            time.sleep(0.5)
            assert not isession.session.mic_stt.sent, "audio leaked while question was spoken"
            ws.send_json({"action": "stop_mock"})
            _recv_until(ws, "action_result", actions=("stop_mock",))

    def test_unknown_action_error(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "bogus"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "bogus" in msg["message"]

    def test_skip_question(self, client, monkeypatch):
        questions = iter(["Q1", "Q2", "Q3"])
        monkeypatch.setattr(isession, "generate_mock_question", lambda **kw: next(questions))
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "start_mock", "mic_source": "desktop"})
            _recv_until(ws, "action_result", actions=("start_mock",))
            _recv_until(ws, "mock_question")
            ws.send_json({"action": "skip_question"})
            result, _ = _recv_until(ws, "action_result", actions=("skip_question",))
            assert result["ok"] is True
            q2, _ = _recv_until(ws, "mock_question")
            assert q2["question_number"] == 2
            ws.send_json({"action": "stop_mock"})
            _recv_until(ws, "action_result", actions=("stop_mock",))


class TestContextAndProviderWS:
    """Clipboard stash + Groq/Gemini switch over the live WebSocket."""

    def _reset(self):
        s = isession.session
        s.context = {"problem": "", "code": "", "segments": []}
        s.clipboard_reader = None
        s.frame_provider = None
        if s.router.active != "groq":
            s.router.switch("groq")

    def test_stash_capture_ws_action(self, client):
        self._reset()
        long_problem = ("Given an array of integers nums and an integer target, "
                        "return indices of the two numbers that add up to target.")
        isession.session.clipboard_reader = lambda: long_problem
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()  # initial status (now includes provider)
                ws.send_json({"action": "stash_capture", "slot": "problem"})
                collected = _recv_until_all(ws, [
                    lambda m: m.get("type") == "action_result"
                    and m.get("action") == "stash_capture",
                    lambda m: m.get("type") == "context_update",
                ])
                result = next(m for m in collected
                              if m.get("type") == "action_result"
                              and m.get("action") == "stash_capture")
                assert result["ok"] is True
                assert result["context"]["problem"]["chars"] == len(long_problem)
                update = next(m for m in collected if m.get("type") == "context_update")
                assert update["context"]["problem"]["chars"] == len(long_problem)

                ws.send_json({"action": "context_status"})
                status, _ = _recv_until(ws, "action_result", actions=("context_status",))
                assert status["context"]["problem"]["chars"] == len(long_problem)
        finally:
            self._reset()

    def test_stash_capture_rejects_short_clipboard(self, client):
        self._reset()
        isession.session.clipboard_reader = lambda: "nope"
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()
                ws.send_json({"action": "stash_capture", "slot": "problem"})
                result, _ = _recv_until(ws, "action_result", actions=("stash_capture",))
                assert result["ok"] is False
                assert isession.session.context["problem"] == ""
        finally:
            self._reset()

    def test_stash_clear_ws_action(self, client):
        self._reset()
        long_problem = ("Given an array of integers nums and an integer target, "
                        "return indices of the two numbers that add up to target.")
        isession.session.clipboard_reader = lambda: long_problem
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()
                ws.send_json({"action": "stash_capture", "slot": "problem"})
                _recv_until_all(ws, [
                    lambda m: m.get("type") == "action_result"
                    and m.get("action") == "stash_capture",
                    lambda m: m.get("type") == "context_update",
                ])
                ws.send_json({"action": "stash_clear", "slot": "all"})
                cleared, _ = _recv_until(ws, "action_result", actions=("stash_clear",))
                assert cleared["ok"] is True
                assert cleared["context"]["problem"]["chars"] == 0
        finally:
            self._reset()

    def test_add_segment_ws_action(self, client):
        import time as _time
        self._reset()
        isession.session.frame_provider = lambda: (b"seg-jpeg", 7, _time.time())
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()
                ws.send_json({"action": "add_segment"})
                collected = _recv_until_all(ws, [
                    lambda m: m.get("type") == "action_result"
                    and m.get("action") == "add_segment",
                    lambda m: m.get("type") == "context_update",
                ])
                result = next(m for m in collected
                              if m.get("type") == "action_result"
                              and m.get("action") == "add_segment")
                assert result["ok"] is True
                assert result["segments"] == 1
        finally:
            self._reset()

    def test_set_provider_ws_action(self, client):
        self._reset()
        try:
            with client.websocket_connect("/ws/interview") as ws:
                first = ws.receive_json()
                assert first.get("provider") == "groq"
                ws.send_json({"action": "set_provider", "provider": "gemini"})
                collected = _recv_until_all(ws, [
                    lambda m: m.get("type") == "action_result"
                    and m.get("action") == "set_provider",
                    lambda m: m.get("type") == "provider_changed",
                ])
                result = next(m for m in collected
                              if m.get("type") == "action_result"
                              and m.get("action") == "set_provider")
                assert result["ok"] is True
                assert result["active"] == "gemini"
                changed = next(m for m in collected if m.get("type") == "provider_changed")
                assert changed["reason"] == "manual"
                assert changed["active"] == "gemini"

                ws.send_json({"action": "provider_status"})
                status, _ = _recv_until(ws, "action_result", actions=("provider_status",))
                assert status["active"] == "gemini"
        finally:
            self._reset()

    def test_set_provider_rejects_unknown(self, client):
        self._reset()
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()
                ws.send_json({"action": "set_provider", "provider": "openai"})
                result, _ = _recv_until(ws, "action_result", actions=("set_provider",))
                assert result["ok"] is False
                assert isession.session.router.active == "groq"
        finally:
            self._reset()

    def test_failover_event_broadcast(self, client):
        self._reset()
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()
                # simulate groq erroring out while answering on gemini
                isession.session.router.note_served("gemini", failed="groq", error="boom")
                changed, _ = _recv_until(ws, "provider_changed")
                assert changed["reason"] == "fallback"
                assert changed["previous"] == "groq"
                assert changed["active"] == "gemini"
        finally:
            self._reset()


class _FakePane:
    """Widget stand-in for NoQtOverlay (no Qt in tests)."""

    def __init__(self):
        self.calls = []
        self.hwnd = 4321

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
        self.calls.append(("geom", geo))


class NoQtOverlay:
    """The REAL OverlayController state machine with Qt submission bypassed.

    submit() runs commands inline (never boots the GUI thread) and the pane
    is a FakePane, so tests exercise production clamping/notify logic.
    """

    def __init__(self):
        from overlay_window import OverlayController

        self._real = OverlayController()
        self._real._ensure_window = self._ensure_window

    def _ensure_window(self):
        if self._real._win is None:
            self._real._win = _FakePane()
        return self._real._win

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        fut = Future()
        rev_before = self._real._rev
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:
            fut.set_exception(exc)
            return fut
        # mirror _drain_once: state changes trigger the broadcast callback
        self._real._notify_if_changed(rev_before)
        return fut

    def status(self):
        return self._real.status()

    def on_state_set(self, cb):
        self._real.on_state = cb

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestOverlayWS:
    """Mirror-tab overlay actions over the live WebSocket (no Qt)."""

    def _install(self, monkeypatch):
        fake = NoQtOverlay()
        fake.on_state_set(
            lambda st: isession.session.emit({"type": "overlay_state", **st})
        )
        monkeypatch.setattr(server, "overlay", fake)
        # hermetic: never call the real Win32 SetWindowDisplayAffinity here
        excluded = []
        monkeypatch.setattr(
            "overlay_window._set_capture_exclusion",
            lambda hwnd, user32=None: excluded.append(hwnd) or True,
        )
        fake.excluded_hwnds = excluded
        return fake

    def test_hello_includes_overlay_tab_and_stream(self, client, monkeypatch):
        self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "status"
            assert msg["overlay"]["open"] is False
            assert "click_through" not in msg["overlay"]  # read-only pane
            assert msg["overlay"]["interact"] is False  # starts locked
            assert msg["overlay"]["capture_excluded"] is False
            assert msg["tab"] in {"groq_vision", "gemini_vision", "interview", "mirror"}
            assert "stream" in msg
            assert msg["stream"]["running"] in (True, False)

    def test_open_close_lifecycle(self, client, monkeypatch):
        fake = self._install(monkeypatch)
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()
                ws.send_json({"action": "overlay_open"})
                collected = _recv_until_all(ws, [
                    lambda m: m.get("type") == "action_result"
                    and m.get("action") == "overlay_open",
                    lambda m: m.get("type") == "overlay_state"
                    and m.get("open") is True,
                ])
                res = next(m for m in collected
                           if m.get("type") == "action_result"
                           and m.get("action") == "overlay_open")
                assert res["ok"] is True
                assert res["open"] is True
                # pane HWND was excluded from screen capture on open
                assert fake.excluded_hwnds == [4321]
                assert fake.status()["capture_excluded"] is True

                ws.send_json({"action": "overlay_close"})
                closed, _ = _recv_until(ws, "action_result", actions=("overlay_close",))
                assert closed["ok"] is True
                assert closed["open"] is False
                assert fake.status()["open"] is False
                assert fake.status()["capture_excluded"] is False
        finally:
            server._current_tab["tab"] = "groq_vision"

    def test_move_and_step(self, client, monkeypatch):
        fake = self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "overlay_move", "dx": 30, "dy": 15})
            moved, _ = _recv_until(ws, "action_result", actions=("overlay_move",))
            assert moved["x"] == 30
            assert moved["y"] == 15

            ws.send_json({"action": "overlay_step", "axis": "w", "delta": 40})
            stepped, _ = _recv_until(ws, "action_result", actions=("overlay_step",))
            assert stepped["w"] == 768 + 40

            ws.send_json({"action": "overlay_step", "axis": "h", "delta": -40})
            stepped, _ = _recv_until(ws, "action_result", actions=("overlay_step",))
            assert stepped["h"] == 800 - 40

    def test_resize_clamps(self, client, monkeypatch):
        self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "overlay_resize", "w": 5, "h": 99999})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_resize",))
            assert st["w"] == 160
            # no Qt in tests -> fallback screen 1920x1080 binds the max
            assert st["h"] == 1080

    def test_preset_and_unknown_preset(self, client, monkeypatch):
        self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "overlay_preset", "name": "tiny"})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_preset",))
            assert st["ok"] is True
            assert (st["w"], st["h"]) == (240, 150)

            ws.send_json({"action": "overlay_preset", "name": "gigantic"})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_preset",))
            assert st["ok"] is False
            assert "Unknown preset" in st["error"]

    def test_zoom_and_opacity(self, client, monkeypatch):
        self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "overlay_zoom", "delta": 0.1})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_zoom",))
            assert st["zoom"] == 0.85  # 0.75 default + 0.1 step

            for _ in range(12):
                ws.send_json({"action": "overlay_opacity", "delta": -0.1})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_opacity",))
            for _ in range(11):
                st, _ = _recv_until(ws, "action_result", actions=("overlay_opacity",))
            assert st["opacity"] == 0.2  # clamped at the floor

    def test_click_through_action_removed(self, client, monkeypatch):
        self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "overlay_click_through", "enabled": False})
            msg, _ = _recv_until(ws, "error", timeout=5.0)
            assert "Unknown action" in msg["message"]

    def test_hide_toggles_only_when_open(self, client, monkeypatch):
        fake = self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            # hide while closed opens the pane (panic semantics)
            ws.send_json({"action": "overlay_hide"})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_hide",))
            assert st["open"] is True
            assert st["hidden"] is False

            ws.send_json({"action": "overlay_hide"})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_hide",))
            assert st["hidden"] is True

            ws.send_json({"action": "overlay_hide"})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_hide",))
            assert st["hidden"] is False

    def test_overlay_status_is_read_only(self, client, monkeypatch):
        fake = self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "overlay_status"})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_status",))
            assert st["ok"] is True
            assert st["open"] is False
            assert st["hotkey"] == "Ctrl+Shift+`"

    def test_overlay_interact_action(self, client, monkeypatch):
        fake = self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            # closed pane -> surfaced error (the client toasts it)
            ws.send_json({"action": "overlay_interact", "enabled": True})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_interact",))
            assert st["ok"] is False
            assert "not open" in st["error"]

            ws.send_json({"action": "overlay_open"})
            _recv_until(ws, "action_result", actions=("overlay_open",))
            pane = fake._real._win

            ws.send_json({"action": "overlay_interact", "enabled": True})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_interact",))
            assert st["ok"] is True
            assert st["interact"] is True
            assert ("interact", True) in pane.calls

            # the unlock is broadcast to every client
            _recv_until_all(ws, [
                lambda m: m.get("type") == "overlay_state"
                and m.get("interact") is True,
            ])

            ws.send_json({"action": "overlay_interact", "enabled": False})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_interact",))
            assert st["ok"] is True
            assert st["interact"] is False
            assert ("interact", False) in pane.calls

    def test_disconnect_relocks_scroll_mode(self, client, monkeypatch):
        """A dropped client must fail-closed: Scroll mode cannot linger on."""
        fake = self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "overlay_open"})
            _recv_until(ws, "action_result", actions=("overlay_open",))
            ws.send_json({"action": "overlay_interact", "enabled": True})
            st, _ = _recv_until(ws, "action_result", actions=("overlay_interact",))
            assert st["ok"] is True and st["interact"] is True
        # socket gone -> server forces the lock without a client to ask it
        deadline = time.time() + 3
        while time.time() < deadline and fake.status()["interact"]:
            time.sleep(0.05)
        assert fake.status()["interact"] is False
        assert ("interact", False) in fake._real._win.calls

    def test_sync_scroll_broadcast_and_reject(self, client, monkeypatch):
        self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "sync_scroll", "target": "history", "top": 123})
            _recv_until_all(ws, [
                lambda m: m.get("type") == "action_result"
                and m.get("action") == "sync_scroll",
                lambda m: m.get("type") == "scroll_sync"
                and m.get("target") == "history" and m.get("top") == 123,
            ])

            ws.send_json({"action": "sync_scroll", "target": "bogus", "top": 5})
            st, _ = _recv_until(ws, "action_result", actions=("sync_scroll",))
            assert st["ok"] is False
            assert "Unknown scroll target" in st["error"]

    def test_sync_tab_broadcast_and_hello_replay(self, client, monkeypatch):
        self._install(monkeypatch)
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()
                ws.send_json({"action": "sync_tab", "tab": "interview"})
                _recv_until_all(ws, [
                    lambda m: m.get("type") == "action_result"
                    and m.get("action") == "sync_tab",
                    lambda m: m.get("type") == "tab_changed"
                    and m.get("tab") == "interview",
                ])

            # a freshly connected overlay pane boots onto the last tab
            with client.websocket_connect("/ws/interview") as ws:
                hello = ws.receive_json()
                assert hello["tab"] == "interview"
        finally:
            server._current_tab["tab"] = "groq_vision"

    def test_sync_tab_rejects_unknown(self, client, monkeypatch):
        self._install(monkeypatch)
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "sync_tab", "tab": "solitaire"})
            st, _ = _recv_until(ws, "action_result", actions=("sync_tab",))
            assert st["ok"] is False
            assert server._current_tab["tab"] != "solitaire"


class TestMirrorBroadcasts:
    """Display events broadcast to every client so the read-only pane
    shows the same feed, answers, and session state as the phone."""

    def test_accept_image_emits_quiz_result(self, client):
        import base64 as b64mod
        from unittest.mock import patch

        answer = {
            "detected": True, "type": "MCQ", "question": "2+2?",
            "answer": "B", "provider": "groq_vision", "explanation": "",
        }
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            with patch("server.analyze_image", return_value=dict(answer)):
                resp = client.post("/api/accept", json={
                    "image": b64mod.b64encode(b"jpeg").decode(),
                    "mode": "groq_vision",
                    "groq_strategy": "quick",
                })
            assert resp.status_code == 200

            msg, _ = _recv_until(ws, "quiz_result")
            assert msg["result"]["question"] == "2+2?"
            assert msg["result"]["mode_used"] == "groq_vision"
            # the tapper's HTTP response still carries the same payload
            assert resp.json()["answer"] == "B"

    def test_screen_start_stop_emit_stream_state(self, client, monkeypatch):
        class _FakeScreen:
            def __init__(self):
                self.region = None
                self.running = False

            def set_region(self, l, t, r, b):
                self.region = (l, t, r, b)

            def get_region(self):
                return self.region

            def is_running(self):
                return self.running

            def start(self):
                self.running = True

            def stop(self):
                self.running = False

            def get_latest_jpeg(self):
                return None

        monkeypatch.setattr(server, "screen", _FakeScreen())
        monkeypatch.setattr(server, "get_settings", lambda: {
            "OCR_MODE": "groq_vision",
            "SCREEN_REGION_LEFT": 10,
            "SCREEN_REGION_TOP": 20,
            "SCREEN_REGION_RIGHT": 510,
            "SCREEN_REGION_BOTTOM": 420,
        })
        with client.websocket_connect("/ws/interview") as ws:
            hello = ws.receive_json()
            assert hello["stream"]["running"] is False
            assert hello["stream"]["region"] is None

            resp = client.post("/api/screen/start")
            assert resp.json()["started"] is True
            msg, _ = _recv_until(ws, "stream_state")
            assert msg["running"] is True
            assert msg["region"] == {"left": 10, "top": 20, "right": 510, "bottom": 420}

            client.post("/api/screen/stop")
            msg, _ = _recv_until(ws, "stream_state")
            assert msg["running"] is False

    def test_relay_toast_broadcasts_with_nonce(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "relay_toast", "text": "Region: 800x600", "nonce": "n-123"})
            collected = _recv_until_all(ws, [
                lambda m: m.get("type") == "action_result"
                and m.get("action") == "relay_toast",
                lambda m: m.get("type") == "toast" and m.get("nonce") == "n-123",
            ])
            toast = next(m for m in collected if m.get("type") == "toast")
            assert toast["text"] == "Region: 800x600"

    def test_sync_subtab_broadcast_and_reject(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "sync_subtab", "sub": "mock"})
            collected = _recv_until_all(ws, [
                lambda m: m.get("type") == "action_result"
                and m.get("action") == "sync_subtab",
                lambda m: m.get("type") == "subtab_changed" and m.get("sub") == "mock",
            ])
            res = next(m for m in collected if m.get("type") == "action_result")
            assert res["ok"] is True

            ws.send_json({"action": "sync_subtab", "sub": "duet"})
            res, _ = _recv_until(ws, "action_result", actions=("sync_subtab",))
            assert res["ok"] is False

    def test_auto_assist_broadcasts(self, client):
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()
                ws.send_json({"action": "set_auto_assist", "enabled": False})
                collected = _recv_until_all(ws, [
                    lambda m: m.get("type") == "action_result"
                    and m.get("action") == "set_auto_assist",
                    lambda m: m.get("type") == "auto_assist"
                    and m.get("enabled") is False,
                ])
                res = next(m for m in collected if m.get("type") == "action_result")
                assert res["auto_screen_assist"] is False
        finally:
            isession.session.set_auto_assist(True)

    def test_start_stop_live_emits_session_state(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "start_live"})
            collected = _recv_until_all(ws, [
                lambda m: m.get("type") == "action_result"
                and m.get("action") == "start_live",
                lambda m: m.get("type") == "session_state"
                and m.get("state") == "live",
            ])
            state = next(m for m in collected if m.get("type") == "session_state")
            assert state["fresh"] is True
            assert state["provider"]

            ws.send_json({"action": "stop_live"})
            collected = _recv_until_all(ws, [
                lambda m: m.get("type") == "action_result"
                and m.get("action") == "stop_live",
                lambda m: m.get("type") == "session_state"
                and m.get("state") == "idle",
            ])
            stopped = next(m for m in collected if m.get("type") == "session_state")
            assert stopped["fresh"] is False


class TestDesignWS:
    """Design interview mode: design_start / design_next / design_end."""

    def test_greeting_includes_design_state(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            greeting = ws.receive_json()
            assert greeting["type"] == "status"
            assert greeting["design"]["active"] is False
            assert "stage" in greeting["design"]

    def test_design_next_without_session_errors(self, client):
        with client.websocket_connect("/ws/interview") as ws:
            ws.receive_json()
            ws.send_json({"action": "design_next", "force": True})
            res, _ = _recv_until(ws, "action_result", actions=("design_next",))
            assert res["ok"] is False
            assert "active" in res["error"].lower()

    def test_design_start_next_end_round_trip(self, client, monkeypatch):
        session = isession.session
        monkeypatch.setattr(isession, "design_extract", lambda *a, **k: {
            "question": "Design a URL shortener",
            "instructions": ["Clarify requirements", "Bottlenecks"],
            "provider": "groq_vision",
        })
        monkeypatch.setattr(isession, "design_turn", lambda u, s, **k: {
            "stage": "requirements",
            "spoken": "About 500 million URLs a month.",
            "mermaid": "flowchart TD\n  C[Client] --> A[API]",
            "facts": ["500M URLs/month"],
            "decisions": ["Base62 IDs"],
            "next": "Outline the high-level design.",
            "provider": "groq",
        })
        old_problem = session.context["problem"]
        session.context["problem"] = "Design a URL shortener (panel text)"
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()
                ws.send_json({"action": "start_live"})
                _recv_until(ws, "action_result", actions=("start_live",))

                ws.send_json({"action": "design_start"})
                collected = _recv_until_all(ws, [
                    lambda m: m.get("type") == "action_result"
                    and m.get("action") == "design_start",
                    lambda m: m.get("type") == "design_state"
                    and m.get("active") is True,
                ])
                start = next(m for m in collected
                             if m.get("type") == "action_result")
                assert start["ok"] is True
                assert start["question"] == "Design a URL shortener"
                assert start["stage"] == "clarify"

                ws.send_json({"action": "design_next", "force": True})
                collected = _recv_until_all(ws, [
                    lambda m: m.get("type") == "action_result"
                    and m.get("action") == "design_next",
                    lambda m: m.get("type") == "design_answer",
                ])
                answer = next(m for m in collected
                              if m.get("type") == "design_answer")
                assert answer["stage"] == "requirements"
                assert answer["diagram"].startswith("flowchart TD")
                assert answer["facts"] == ["500M URLs/month"]
                assert answer["answer"]["spoken"].startswith("About 500")
                # merged server-side too
                assert session.design["facts"] == ["500M URLs/month"]

                ws.send_json({"action": "design_end"})
                res, _ = _recv_until(ws, "action_result", actions=("design_end",))
                assert res["ok"] is True
                assert session.design["active"] is False
        finally:
            session.context["problem"] = old_problem
            session.design = session._empty_design()

    def test_stop_live_clears_design_session(self, client):
        session = isession.session
        try:
            with client.websocket_connect("/ws/interview") as ws:
                ws.receive_json()
                ws.send_json({"action": "start_live"})
                _recv_until(ws, "action_result", actions=("start_live",))
                session.design.update({
                    "active": True, "question": "Design X",
                    "instructions": [],
                })
                ws.send_json({"action": "stop_live"})
                collected = _recv_until_all(ws, [
                    lambda m: m.get("type") == "action_result"
                    and m.get("action") == "stop_live",
                    lambda m: m.get("type") == "design_state"
                    and m.get("active") is False,
                ])
                assert any(m.get("type") == "design_state" for m in collected)
                assert session.design["active"] is False
        finally:
            session.design = session._empty_design()

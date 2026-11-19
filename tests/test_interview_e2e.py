"""Real end-to-end tests for interview mode.

These use REAL services: ElevenLabs Scribe STT/TTS, WASAPI loopback
capture, and Groq answer generation.  They spend API credits and emit
audio through the PC speakers, so they only run when RUN_E2E=1:

    RUN_E2E=1 python -m pytest tests/test_interview_e2e.py -v
"""

import base64
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E") != "1",
    reason="Real-service E2E is opt-in: RUN_E2E=1",
)

from interview_session import InterviewSession


def _wait_for(q, msg_type, timeout=45.0):
    """Poll out_queue for a message type; return it or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = q.get(timeout=0.3)
        except queue.Empty:
            continue
        if msg.get("type") == msg_type:
            return msg
    return None


def _wait_for_state(q, state, timeout=45.0):
    """Poll out_queue for a status message with the given state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = q.get(timeout=0.3)
        except queue.Empty:
            continue
        if msg.get("type") == "status" and msg.get("state") == state:
            return msg
    return None


def _recv_until_all(ws, predicates, timeout=8.0):
    """Collect WS messages until every predicate(msg) matched at least once."""
    remaining = list(predicates)
    collected = []
    deadline = time.time() + timeout
    while remaining and time.time() < deadline:
        msg = ws.receive_json()
        collected.append(msg)
        remaining = [p for p in remaining if not p(msg)]
    assert not remaining, f"missing expected messages; got: {collected}"
    return collected


def _recv_until(ws, msg_type, timeout=10.0, actions=()):
    """Receive WS messages until msg_type (or an action_result in actions).

    Returns (message, all_messages_collected_so_far).
    """
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


def _playback(pcm_bytes: bytes, sample_rate: int = 16000):
    """Play PCM on a thread so the test can keep consuming messages."""
    import numpy as np
    import sounddevice as sd

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    sd.play(samples, samplerate=sample_rate, blocking=True)


@pytest.fixture()
def live_session():
    s = InterviewSession()
    s.start()
    yield s
    s.shutdown()


class TestLiveE2E:
    def test_loopback_to_answer_chain(self, live_session):
        """Speakers -> WASAPI loopback -> Scribe STT -> Groq -> answer."""
        from elevenlabs_client import ElevenLabsTTS

        result = live_session.start_live()
        assert result.get("ok"), result

        tts = ElevenLabsTTS()
        speech = tts.synthesize_pcm("Tell me about a time you had to learn a new technology quickly.")
        assert speech, "TTS produced no audio"

        # give the loopback stream a moment to start delivering frames
        time.sleep(1.0)
        player = threading.Thread(target=_playback, args=(speech, 16000), daemon=True)
        player.start()

        # 1) transcript must commit once VAD/watchdog fires after playback
        transcript = _wait_for(live_session.out_queue, "transcript", timeout=40)
        # keep draining until a final transcript shows up
        deadline = time.time() + 40
        while (transcript is None or not transcript.get("final")) and time.time() < deadline:
            try:
                msg = live_session.out_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if msg.get("type") == "transcript" and msg.get("final"):
                transcript = msg

        assert transcript is not None, "no transcript committed"
        print(f"\nE2E transcript: {transcript['text']!r}")

        # 2) answer must be generated for the committed question
        answer = _wait_for(live_session.out_queue, "answer", timeout=40)
        assert answer is not None, "no answer generated"
        assert len(answer["answer"]) > 40, "answer too short"
        assert answer["question"]
        print(f"E2E answer: {answer['answer'][:160]}...")

        live_session.stop_live()
        assert live_session.mode is None


class TestMockE2E:
    def test_question_stt_evaluation_chain(self, live_session):
        """Groq question -> TTS -> (injected speech) -> hybrid endpoint -> STT -> Groq evaluation."""
        from elevenlabs_client import ElevenLabsTTS

        result = live_session.start_mock("phone", role="backend engineer", level="mid-level")
        assert result.get("ok"), result

        question = _wait_for(live_session.out_queue, "mock_question", timeout=45)
        assert question is not None, "no mock question generated"
        print(f"\nE2E mock Q: {question['question']!r}")

        # the mic only opens once the question finished speaking
        listening = _wait_for_state(live_session.out_queue, "listening_answer", timeout=30)
        assert listening is not None, "mock never reached listening state"
        assert live_session.mock_state == "listening"

        # synthesize a spoken answer and inject it as if the phone mic sent it
        tts = ElevenLabsTTS()
        speech = tts.synthesize_pcm(
            "In my last role I redesigned our order processing pipeline, cutting "
            "nightly batch time from four hours to twenty minutes and reducing errors by half."
        )
        assert speech, "TTS produced no audio"

        for i in range(0, len(speech), 3200):
            live_session.handle_phone_mic_audio(
                base64.b64encode(speech[i:i + 3200]).decode()
            )
            time.sleep(0.05)

        # hybrid endpointing: silence after the answer auto-commits (6s default)
        user_answer = _wait_for(live_session.out_queue, "mock_user_answer", timeout=40)
        assert user_answer is not None, "answer never endpointed (silence commit missing)"
        print(f"E2E user answer: {user_answer['text'][:120]!r}")

        feedback = _wait_for(live_session.out_queue, "mock_feedback", timeout=40)
        assert feedback is not None, "no evaluation produced"
        assert isinstance(feedback.get("score"), int)
        assert 0 <= feedback["score"] <= 10
        assert feedback.get("feedback")
        print(f"E2E score: {feedback['score']}/10 — {feedback['feedback'][:120]}")

        follow_up = _wait_for(live_session.out_queue, "mock_question", timeout=40)
        assert follow_up is not None, "no follow-up question"
        assert follow_up["question_number"] == 2

        live_session.stop_mock()
        assert live_session.mode is None

    def test_done_button_ends_answer(self, live_session):
        """Manual endpointing: Done button commits the pending partial immediately."""
        result = live_session.start_mock("phone", role="product manager", level="mid-level")
        assert result.get("ok"), result
        assert _wait_for(live_session.out_queue, "mock_question", timeout=45) is not None
        assert _wait_for_state(live_session.out_queue, "listening_answer", timeout=30) is not None

        # inject a short spoken answer, then press Done (no 6s wait)
        from elevenlabs_client import ElevenLabsTTS
        tts = ElevenLabsTTS()
        speech = tts.synthesize_pcm("I would start by clarifying the customer's core pain point before proposing scope cuts.")
        assert speech
        for i in range(0, len(speech), 3200):
            live_session.handle_phone_mic_audio(base64.b64encode(speech[i:i + 3200]).decode())
            time.sleep(0.05)

        # wait until STT partial text exists so Done has something to submit
        deadline = time.time() + 15
        while time.time() < deadline and not live_session._mock_partial_text:
            time.sleep(0.1)
        assert live_session._mock_partial_text, "no STT partial before Done"

        result = live_session.finish_mock_answer()
        assert result["ok"], result

        user_answer = _wait_for(live_session.out_queue, "mock_user_answer", timeout=20)
        assert user_answer is not None, "Done button did not deliver an answer"
        print(f"E2E done-button answer: {user_answer['text'][:100]!r}")
        live_session.stop_mock()


class TestModelAccessE2E:
    def test_groq_models_available(self):
        """Guard against the qwen3.6-27b style 404: the configured models must exist."""
        from gemini_client import GROQ_CODE_MODEL, GROQ_TEXT_FALLBACK_MODEL, GROQ_VISION_MODEL, check_groq_status

        status = check_groq_status()
        assert status.get("valid"), status
        assert status.get("vision_available"), f"vision model unavailable: {GROQ_VISION_MODEL} / {status}"

        import json as _json
        import urllib.request
        from config import get_settings
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {get_settings()['GROQ_API_KEY']}",
                     "User-Agent": "Quizbox/1.0"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            ids = {m["id"] for m in _json.loads(response.read().decode())["data"]}
        assert GROQ_VISION_MODEL in ids, f"missing {GROQ_VISION_MODEL}; have: {sorted(ids)}"
        assert GROQ_TEXT_FALLBACK_MODEL in ids, f"missing {GROQ_TEXT_FALLBACK_MODEL}"
        assert GROQ_CODE_MODEL in ids, f"missing {GROQ_CODE_MODEL}"
        print(f"E2E models OK: {GROQ_VISION_MODEL}, {GROQ_TEXT_FALLBACK_MODEL}, {GROQ_CODE_MODEL}")


class TestFusionE2E:
    def test_solve_screen_real_vision_chain(self):
        """Real Groq vision: screenshot of a coding problem + transcript -> code answer."""
        import io

        from PIL import Image, ImageDraw

        from interview_pipeline import solve_with_screen

        img = Image.new("RGB", (900, 500), (255, 255, 255))
        d = ImageDraw.Draw(img)
        lines = [
            "Given an array of integers, return indices of the two numbers",
            "that add up to a specific target.",
            "",
            "Example: nums = [2, 7, 11, 15], target = 9 -> [0, 1]",
            "",
            "def two_sum(nums, target):",
            "    # TODO: finish this",
            "    pass",
        ]
        y = 40
        for line in lines:
            d.text((40, y), line, fill=(0, 0, 0))
            y += 36
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)

        result = solve_with_screen(
            [buf.getvalue()],
            "Can you walk me through your approach and finish the function?",
            history=[],
            variant="solve",
        )
        assert result["detected"], result
        assert result["answer_text"], "no spoken answer produced"
        assert result["code_solution"], "no code produced"
        assert result["next_steps"], "no next steps produced"
        assert result["provider"] in ("groq_vision", "gemini_vision", "groq_text", "gemini_text")
        print(f"\nE2E fusion provider={result['provider']} lang={result['language']}")
        print(f"E2E fusion answer: {result['answer_text'][:140]}")
        print(f"E2E fusion code: {result['code_solution'][:140]}")


class TestContextE2E:
    def test_clipboard_stash_to_solve(self, live_session):
        """Real OS clipboard -> stash_capture -> text-only solve (no screenshot)."""
        from context_capture import read_text, write_text

        problem = (
            "Given an array of integers nums and an integer target, return indices "
            "of the two numbers such that they add up to target.\n"
            "You may assume that each input would have exactly one solution, and "
            "you may not use the same element twice.\n"
            "Example:\nInput: nums = [2,7,11,15], target = 9\nOutput: [0,1]"
        )
        original = read_text()
        try:
            assert write_text(problem), "clipboard write failed"
            assert problem.splitlines()[0] in read_text()

            live_session.clipboard_reader = None  # force the real OS clipboard path
            res = live_session.stash_capture("problem")
            assert res["ok"], res

            live_session.start_live()
            result = live_session.solve_screen()  # no frame provider -> stash only
            assert result["ok"], result
            solve = _wait_for(live_session.out_queue, "screen_solve", timeout=60)
            assert solve is not None, "no screen_solve produced"
            assert solve["detected"], solve
            assert solve["code"], "no code produced from the stashed problem"
            print(f"\nE2E stash-only solve provider={solve['provider']}")
            print(f"E2E stash code: {solve['code'][:140]}")
            live_session.stop_live()
        finally:
            write_text(original or "")


class TestProviderE2E:
    def test_gemini_serves_live_answer_with_history(self):
        """Manual switch to Gemini: identical history, different provider, real API."""
        from interview_pipeline import detect_and_answer
        from provider_router import ProviderRouter

        router = ProviderRouter()
        router.switch("gemini")
        history = [
            {"question": "Tell me about yourself.", "answer": "I'm Micheal, a backend engineer."},
            {"question": "What is your strongest language?", "answer": "Python."},
        ]
        result = detect_and_answer(
            "What project are you most proud of?",
            history=history, router=router,
        )
        assert not result.get("error"), result
        assert result["is_question"], result
        assert result["provider"] == "gemini"
        assert len(result["answer"]) > 40, result
        print(f"\nE2E gemini answer: {result['answer'][:140]}")

    def test_groq_serves_live_answer_with_history(self):
        """Same call on the default provider — both sides of the router are live."""
        from interview_pipeline import detect_and_answer
        from provider_router import ProviderRouter

        router = ProviderRouter()  # preferred/active: groq
        history = [
            {"question": "Tell me about yourself.", "answer": "I'm Micheal, a backend engineer."},
        ]
        result = detect_and_answer(
            "Why are you interested in this role?",
            history=history, router=router,
        )
        assert not result.get("error"), result
        assert result["is_question"], result
        assert result["provider"] == "groq"
        assert len(result["answer"]) > 40, result
        print(f"\nE2E groq answer: {result['answer'][:140]}")


class TestOverlayWSE2E:
    def test_ws_actions_drive_real_pane(self):
        """Server WS action -> GUI thread -> real Qt pane (no fakes)."""
        from fastapi.testclient import TestClient

        import server
        from overlay_window import get_overlay

        with TestClient(server.app) as client:
            try:
                with client.websocket_connect("/ws/interview") as ws:
                    hello = ws.receive_json()
                    assert hello["type"] == "status"
                    assert "overlay" in hello and "tab" in hello

                    # A RAM-starved first QtWebEngine boot can exceed the
                    # server's action timeout while the pane still finishes
                    # opening — a second tap then succeeds instantly.
                    opened = None
                    for attempt in range(2):
                        ws.send_json({"action": "overlay_open"})
                        result, state_ok = None, False
                        deadline = time.time() + 30
                        while time.time() < deadline and not (result and state_ok):
                            msg = ws.receive_json()
                            if (msg.get("type") == "action_result"
                                    and msg.get("action") == "overlay_open"):
                                result = msg
                            elif (msg.get("type") == "overlay_state"
                                  and msg.get("open") is True):
                                state_ok = True
                        if result and result.get("ok") and state_ok:
                            opened = result
                            break
                        time.sleep(3)  # let the slow open finish, then re-tap
                    assert opened is not None and opened["ok"] and opened["open"], (
                        "overlay_open never succeeded: %r" % (opened,)
                    )
                    time.sleep(1.0)  # pane paints

                    ws.send_json({"action": "overlay_move", "dx": 12, "dy": 6})
                    moved, _ = _recv_until(ws, "action_result", actions=("overlay_move",))
                    assert moved["ok"] is True

                    ws.send_json({"action": "overlay_close"})
                    closed, _ = _recv_until(ws, "action_result", actions=("overlay_close",))
                    assert closed["ok"] is True and closed["open"] is False
                    print("\nE2E WS->real pane OK")
            finally:
                # Never leak an open pane into the next test (a timed-out
                # overlay_open keeps running on the GUI thread).
                try:
                    ov = get_overlay()
                    ov.submit(ov.close).result(timeout=10)
                except Exception:  # noqa: BLE001
                    pass


class TestPaneMirrorE2E:
    """The read-only pane renders exactly what the phone broadcasts:

    live feed frames, quiz answers, counts, and toasts — proven by reading
    the pane's own DOM from the Qt GUI thread.
    """

    @staticmethod
    def _pane_js(ov, expr, timeout=15):
        """Evaluate JS inside the pane's page (GUI thread, nested event loop)."""
        def run():
            from PySide6.QtCore import QEventLoop, QTimer

            if ov._win is None:
                return None
            box = {"v": None}
            loop = QEventLoop()
            ov._win._view.page().runJavaScript(
                expr, lambda v: (box.update({"v": v}), loop.quit())
            )
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            return box["v"]

        return ov.submit(run).result(timeout=timeout)

    def _wait_js(self, ov, expr, probe, timeout=8.0):
        deadline = time.time() + timeout
        value = None
        while time.time() < deadline:
            value = self._pane_js(ov, expr)
            if probe(value):
                return value
            time.sleep(0.3)
        return value

    @staticmethod
    def _start_backend():
        """Serve the real app over HTTPS in-process (no extra process).

        The pane always loads a live https origin — TestClient alone cannot
        satisfy that — so E2E binds its own uvicorn thread on a side port.
        """
        import threading

        import uvicorn

        import server

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cert, key = os.path.join(root, "cert.pem"), os.path.join(root, "key.pem")
        if not (os.path.exists(cert) and os.path.exists(key)):
            from gen_cert import generate_self_signed_cert

            generate_self_signed_cert(root)

        port = 8443
        config = uvicorn.Config(
            server.app, host="127.0.0.1", port=port,
            ssl_certfile=cert, ssl_keyfile=key, log_level="warning",
        )
        srv = uvicorn.Server(config)
        thread = threading.Thread(target=srv.run, name="e2e-uvicorn", daemon=True)
        thread.start()
        deadline = time.time() + 15
        while time.time() < deadline and not srv.started:
            time.sleep(0.1)
        if not srv.started:
            srv.should_exit = True
            pytest.skip(f"could not bind 127.0.0.1:{port} for pane E2E")
        return srv, thread, port

    def test_pane_mirrors_feed_answers_and_is_read_only(self):
        import server
        from overlay_window import OVERLAY_URL, get_overlay

        ov = get_overlay()
        # Drop any pane leaked by a prior test — _ensure_window would return
        # it unchanged, still bound to the old (dead) URL.
        try:
            ov.submit(ov.close).result(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        srv, thread, port = self._start_backend()
        orig_url = ov.url
        ov.url = f"https://127.0.0.1:{port}/?display=overlay"
        try:
            st = ov.submit(ov.open).result(timeout=30)
            assert st["open"], st
            assert st["capture_excluded"] is True, st
            time.sleep(2.5)  # QtWebEngine loads the https mirror page + WS

            # --- read-only: overlay variant kills every pointer event ---
            pe = self._wait_js(
                ov, "getComputedStyle(document.body).pointerEvents",
                lambda v: v == "none", timeout=8,
            )
            assert pe == "none", pe
            cls = self._pane_js(ov, "document.body.className")
            assert "display-overlay" in cls, cls

            # --- quiz answer broadcast renders inside the pane ---
            server.session.emit({"type": "quiz_result", "result": {
                "detected": True, "type": "MCQ",
                "question": "Which port does HTTPS use?",
                "answer": "B) 443", "provider": "groq_vision",
            }})
            txt = self._wait_js(
                ov, "document.getElementById('answerContent').textContent",
                lambda v: v and "443" in v,
            ) or ""
            assert "443" in txt, f"pane did not render the answer: {txt!r}"
            count = self._pane_js(ov, "document.getElementById('countText').textContent")
            assert count == "1 answered", count

            # --- live feed frames arrive in the pane ---
            # Synthetic camera (same pattern as test_live_stream): a real
            # dxcam grab can block on a static desktop, so never depend on it.
            import threading

            import numpy as np

            class _FakeCamera:
                def grab(self, region=None):
                    left, top, right, bottom = region
                    return np.full((bottom - top, right - left, 3), 42, dtype=np.uint8)

                def stop(self):
                    pass

            screen = server.screen
            try:
                screen.set_region(0, 0, 640, 400)
                screen._frame.clear()
                with screen._jpeg_lock:
                    screen._jpeg = None
                    screen._jpeg_sequence = 0
                    screen._jpeg_captured_at = 0.0
                screen._camera = _FakeCamera()
                screen._running = True
                screen._thread = threading.Thread(
                    target=screen._capture_loop, daemon=True
                )
                screen._thread.start()
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"screen capture unavailable: {exc}")
            try:
                region = {"left": 0, "top": 0, "right": 640, "bottom": 400}
                server.session.emit(
                    {"type": "stream_state", "running": True, "region": region}
                )
                width = self._wait_js(
                    ov, "document.getElementById('streamPreview').naturalWidth",
                    lambda v: bool(v) and v > 0, timeout=10,
                )
                assert width and width > 0, "pane never received a live frame"
                print(f"\nE2E pane live feed OK ({width}px wide frame)")

                # --- toast relay mirrors into the pane ---
                server.session.emit(
                    {"type": "toast", "text": "E2E mirrored toast", "nonce": "e2e"}
                )
                shown = self._wait_js(
                    ov, "document.getElementById('toast').textContent",
                    lambda v: v == "E2E mirrored toast",
                )
                assert shown == "E2E mirrored toast", shown
            finally:
                server.screen.stop()
                server.session.emit(
                    {"type": "stream_state", "running": False, "region": None}
                )

            # stream stop tears the pane's frame loop down
            stopped = self._wait_js(
                ov, "document.getElementById('streamPreview').style.display",
                lambda v: v == "none", timeout=5,
            )
            assert stopped == "none", stopped
            print("\nE2E pane mirror (read-only, answer, feed, toast) OK")
        finally:
            try:
                ov.submit(ov.close).result(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            ov.url = orig_url
            srv.should_exit = True
            thread.join(timeout=5)
        # Deliberately does NOT stop the controller: TestOverlayE2E below
        # owns the singleton shutdown for this process.


class TestOverlayE2E:
    def test_real_pane_lifecycle(self):
        """Real Qt pane on the desktop: open -> move -> resize -> zoom -> hide -> close."""
        from overlay_window import get_overlay

        ov = get_overlay()
        try:
            st = ov.submit(ov.open).result(timeout=30)
            assert st["open"], st
            x0, y0, w0 = st["x"], st["y"], st["w"]
            time.sleep(1.5)  # let QtWebEngine paint the mirror page

            st = ov.submit(ov.move_by, 25, 30).result(timeout=10)
            assert st["x"] == x0 + 25 and st["y"] == y0 + 30, st

            st = ov.submit(ov.step, "w", 40).result(timeout=10)
            assert st["w"] == w0 + 40, st

            st = ov.submit(ov.zoom_step, 0.1).result(timeout=10)
            assert st["zoom"] == 0.85, st  # 0.75 default + 0.1 step

            st = ov.submit(ov.opacity_step, -0.1).result(timeout=10)
            assert st["opacity"] == 0.9, st

            # Scroll mode: unlock must clear WS_EX_TRANSPARENT on the real
            # HWND (native wheel/scrollbar input works), lock must restore
            # it — and WS_EX_NOACTIVATE must survive throughout so the pane
            # never steals focus from the app beneath.
            import ctypes
            import ctypes.wintypes

            u32 = ctypes.windll.user32
            u32.GetWindowLongPtrW.argtypes = [
                ctypes.wintypes.HWND,
                ctypes.c_int,
            ]
            u32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
            GWL_EXSTYLE, WS_EX_TRANSPARENT, WS_EX_NOACTIVATE = -20, 0x20, 0x08000000
            pane_hwnd = ctypes.wintypes.HWND(int(ov._win.winId()))

            def _exstyle():
                return u32.GetWindowLongPtrW(pane_hwnd, GWL_EXSTYLE)

            st = ov.submit(ov.set_interact, True).result(timeout=10)
            assert st["interact"] is True, st
            assert not (_exstyle() & WS_EX_TRANSPARENT), hex(_exstyle())
            assert _exstyle() & WS_EX_NOACTIVATE, hex(_exstyle())

            st = ov.submit(ov.set_interact, False).result(timeout=10)
            assert st["interact"] is False, st
            assert _exstyle() & WS_EX_TRANSPARENT, hex(_exstyle())

            # unlock + Hide -> fail-closed re-lock while the pane hides
            ov.submit(ov.set_interact, True).result(timeout=10)
            st = ov.submit(ov.set_hidden, True).result(timeout=10)
            assert st["interact"] is False, st
            assert _exstyle() & WS_EX_TRANSPARENT, hex(_exstyle())

            # pane is permanently read-only: no click-through toggle exists
            assert "click_through" not in st, st
            assert not hasattr(ov, "set_click_through")

            # the pane HWND must be excluded from every screen capture —
            # otherwise the feed reflects the pane and loops to infinity
            import ctypes
            import ctypes.wintypes

            hwnd = ctypes.wintypes.HWND(int(ov._win.winId()))
            aff = ctypes.wintypes.DWORD()
            get_aff = ctypes.windll.user32.GetWindowDisplayAffinity
            get_aff.argtypes = [
                ctypes.wintypes.HWND,
                ctypes.POINTER(ctypes.wintypes.DWORD),
            ]
            get_aff.restype = ctypes.wintypes.BOOL
            assert get_aff(hwnd, ctypes.byref(aff)), "GetWindowDisplayAffinity failed"
            assert aff.value == 0x11, hex(aff.value)
            assert st["capture_excluded"] is True, st

            st = ov.submit(ov.set_hidden, True).result(timeout=10)
            assert st["hidden"] is True, st
            time.sleep(0.5)
            st = ov.submit(ov.set_hidden, False).result(timeout=10)
            assert st["hidden"] is False, st

            # global hotkey Ctrl+Shift+` must toggle hide (synthetic press)
            import ctypes

            u32 = ctypes.windll.user32
            VK_CONTROL, VK_SHIFT, VK_OEM_3, KEYUP = 0x11, 0x10, 0xC0, 0x0002

            def _press_hotkey():
                for k in (VK_CONTROL, VK_SHIFT, VK_OEM_3):
                    u32.keybd_event(k, 0, 0, 0)
                for k in (VK_OEM_3, VK_SHIFT, VK_CONTROL):
                    u32.keybd_event(k, 0, KEYUP, 0)

            _press_hotkey()
            deadline = time.time() + 3
            while time.time() < deadline and not ov.status()["hidden"]:
                time.sleep(0.1)
            assert ov.status()["hidden"] is True, "hotkey did not hide the pane"
            _press_hotkey()
            deadline = time.time() + 3
            while time.time() < deadline and ov.status()["hidden"]:
                time.sleep(0.1)
            assert ov.status()["hidden"] is False, "hotkey did not show the pane"

            st = ov.submit(ov.preset, "tiny").result(timeout=10)
            assert (st["w"], st["h"]) == (240, 150), st

            st = ov.submit(ov.close).result(timeout=10)
            assert st["open"] is False, st
            print(f"\nE2E overlay pane lifecycle OK at ({x0},{y0}) {w0}px wide")
        finally:
            try:
                ov.submit(ov.close).result(timeout=10)
            except Exception:
                pass
            ov.stop()

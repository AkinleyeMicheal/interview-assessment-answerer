"""Unit tests for interview mode — no network, no audio devices."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import queue
import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# transcript response parsing
# ---------------------------------------------------------------------------

class TestParseInterviewResponse:
    def test_yes_full(self):
        from interview_pipeline import _parse_interview_response
        text = (
            "IS_QUESTION: YES\n"
            "QUESTION: Tell me about a conflict.\n"
            "ANSWER: Last year two teammates disagreed and I mediated."
        )
        r = _parse_interview_response(text)
        assert r["is_question"] is True
        assert r["question"] == "Tell me about a conflict."
        assert r["answer"] == "Last year two teammates disagreed and I mediated."

    def test_no_returns_empty_answer(self):
        from interview_pipeline import _parse_interview_response
        text = "IS_QUESTION: NO\nQUESTION: N/A\nANSWER: N/A"
        r = _parse_interview_response(text)
        assert r["is_question"] is False
        assert r["question"] == ""
        assert r["answer"] == ""

    def test_is_question_collision(self):
        """QUESTION: must not match inside IS_QUESTION:"""
        from interview_pipeline import _parse_interview_response
        text = "IS_QUESTION: YES\nQUESTION: What is your greatest strength?\nANSWER: My strength is X."
        r = _parse_interview_response(text)
        assert r["question"] == "What is your greatest strength?"
        assert "YES" not in r["question"]

    def test_multiline_answer(self):
        from interview_pipeline import _parse_interview_response
        text = (
            "IS_QUESTION: YES\n"
            "QUESTION: Explain TCP handshake.\n"
            "ANSWER: First the client sends SYN.\n"
            "Then the server replies SYN-ACK.\n"
            "Finally the client sends ACK."
        )
        r = _parse_interview_response(text)
        assert "SYN-ACK" in r["answer"]
        assert r["answer"].count("\n") == 2

    def test_missing_markers_falls_back_to_answer(self):
        from interview_pipeline import _parse_interview_response
        r = _parse_interview_response("Just a plain answer without markers.")
        assert r["is_question"] is True
        assert r["answer"] == "Just a plain answer without markers."

    def test_empty_input(self):
        from interview_pipeline import _parse_interview_response
        r = _parse_interview_response("")
        assert r == {"is_question": False, "incomplete": False, "needs_screen": False,
                     "question": "", "answer": ""}

    def test_incomplete_and_needs_screen_flags(self):
        from interview_pipeline import _parse_interview_response
        text = (
            "IS_QUESTION: YES\n"
            "INCOMPLETE: YES\n"
            "NEEDS_SCREEN: YES\n"
            "QUESTION: How do you reverse a linked list?\n"
            "ANSWER: You swap pointers."
        )
        r = _parse_interview_response(text)
        assert r["incomplete"] is True
        assert r["needs_screen"] is True
        assert r["is_question"] is True

    def test_flags_default_false(self):
        from interview_pipeline import _parse_interview_response
        text = "IS_QUESTION: NO\nQUESTION: N/A\nANSWER: N/A"
        r = _parse_interview_response(text)
        assert r["incomplete"] is False
        assert r["needs_screen"] is False


# ---------------------------------------------------------------------------
# audio resampling
# ---------------------------------------------------------------------------

class TestResample:
    def test_48k_stereo_to_16k_mono_length(self):
        from audio_capture import resample_to_16k_mono
        src = np.zeros(48000 * 2 * 2, dtype=np.int16)  # 2 sec, stereo interleaved
        out = resample_to_16k_mono(src.tobytes(), 48000, 2)
        n = len(out) // 2
        assert abs(n - 32000) <= 2  # ~2 sec at 16 kHz

    def test_16k_mono_passthrough(self):
        from audio_capture import resample_to_16k_mono
        src = (np.arange(1600, dtype=np.int16) * 10)
        out = resample_to_16k_mono(src.tobytes(), 16000, 1)
        assert out == src.tobytes()

    def test_empty_input(self):
        from audio_capture import resample_to_16k_mono
        assert resample_to_16k_mono(b"", 48000, 2) == b""

    def test_stereo_channels_averaged(self):
        from audio_capture import resample_to_16k_mono
        left = np.full(3, 1000, dtype=np.int16)
        right = np.full(3, 3000, dtype=np.int16)
        stereo = np.column_stack([left, right]).ravel()
        out = resample_to_16k_mono(stereo.tobytes(), 16000, 2)
        samples = np.frombuffer(out, dtype=np.int16)
        assert samples[0] == 2000


# ---------------------------------------------------------------------------
# interview session logic (external services mocked)
# ---------------------------------------------------------------------------

@pytest.fixture()
def session(monkeypatch):
    import interview_session as mod

    monkeypatch.setattr(mod, "get_elevenlabs_key", lambda: "test-key")

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
            self.on_partial = on_partial
            self.on_committed = on_committed
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
        def start(self, callback):
            self.running = True
            self.callback = callback
            return {"ok": True, "device": "fake"}
        def stop(self):
            self.running = False
        def is_running(self):
            return self.running

    class FakeTTS:
        def __init__(self):
            self.spoken = []
            self.stops = 0
        def speak(self, text, block=True):
            self.spoken.append(text)
        def stop_playback(self):
            self.stops += 1

    monkeypatch.setattr(mod, "ElevenLabsSTT", FakeSTT)
    monkeypatch.setattr(mod, "SystemAudioCapture", FakeCapture)
    monkeypatch.setattr(mod, "MicCapture", FakeCapture)
    monkeypatch.setattr(mod, "ElevenLabsTTS", FakeTTS)

    s = mod.InterviewSession()
    s.start()
    yield s
    s.shutdown()
    # drain for next test
    for q in (s.out_queue, s.live_work, s.mock_work):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass


def _drain(q, timeout=8.0):
    """Collect messages from out_queue until timeout, return list."""
    items = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            items.append(q.get(timeout=0.2))
        except queue.Empty:
            pass
    return items


def _wait_for(q, msg_type, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = q.get(timeout=0.2)
        except queue.Empty:
            continue
        if msg.get("type") == msg_type:
            return msg
    return None


def _wait_for_status(q, state, timeout=8.0):
    """Wait for a status message with a specific state (pops other messages)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = q.get(timeout=0.2)
        except queue.Empty:
            continue
        if msg.get("type") == "status" and msg.get("state") == state:
            return msg
    return None


def _wait_for_answer(session, timeout=10.0):
    """Mock protocol helper: wait until the mic is open (listening state)."""
    msg = _wait_for_status(session.out_queue, "listening_answer", timeout=timeout)
    assert msg is not None, f"never reached listening; got {_drain(session.out_queue)}"
    return msg


class TestLiveMode:
    def test_start_live_success(self, session, monkeypatch):
        result = session.start_live()
        assert result["ok"] is True
        assert session.mode == "live"
        assert session.loopback.is_running()
        assert session.live_stt.is_running()

    def test_start_live_requires_key(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "get_elevenlabs_key", lambda: "")
        result = session.start_live()
        assert result["ok"] is False
        assert "ELEVENLABS_API_KEY" in result["error"]
        assert session.mode is None

    def test_committed_transcript_emits_answer(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "detect_and_answer", lambda t, **kw: {
            "is_question": True,
            "question": t,
            "answer": "Here is my full spoken answer.",
        })
        session.start_live()
        session._on_live_committed({"text": "Tell me about yourself"})
        msg = _wait_for(session.out_queue, "answer", timeout=10)
        assert msg is not None, f"no answer; got {_drain(session.out_queue)}"
        assert msg["answer"] == "Here is my full spoken answer."
        assert msg["question"] == "Tell me about yourself"
        assert len(session.history) == 1

    def test_non_question_no_answer(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "detect_and_answer", lambda t, **kw: {
            "is_question": False, "question": "", "answer": "",
        })
        session.start_live()
        session._on_live_committed({"text": "Go on."})
        time.sleep(0.5)
        msgs = _drain(session.out_queue, timeout=1.0)
        assert not any(m.get("type") == "answer" for m in msgs)

    def test_dedupe_identical_committed(self, session, monkeypatch):
        calls = []
        import interview_session as mod
        monkeypatch.setattr(mod, "detect_and_answer", lambda t, **kw: calls.append(t) or {
            "is_question": False, "question": "", "answer": "",
        })
        session.start_live()
        session._on_live_committed({"text": "Interesting."})
        session._on_live_committed({"text": "Interesting."})  # within 5s -> deduped
        time.sleep(0.6)
        assert len(calls) == 1

    def test_partial_and_committed_emit_transcript(self, session):
        session.start_live()
        session._on_live_partial({"text": "Tell me about"})
        session._on_live_committed({"text": "Tell me about your project"})
        partials = []
        finals = []
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                m = session.out_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if m.get("type") == "transcript":
                partials.append(m) if not m["final"] else finals.append(m)
        assert any(p["text"] == "Tell me about" for p in partials)
        assert any(f["text"] == "Tell me about your project" for f in finals)

    def test_stop_live_cleans_up(self, session):
        session.start_live()
        session.stop_live()
        assert session.mode is None
        assert not session.loopback.is_running()
        assert not session.live_stt.is_running()


def _health_msgs(session, timeout=0.5):
    msgs = _drain(session.out_queue, timeout=timeout)
    return (
        [m for m in msgs if m.get("type") == "audio_health"],
        [m for m in msgs if m.get("type") == "error"],
    )


class TestLiveHealth:
    """audio_health broadcast: the phone must SEE capture/STT truth."""

    def test_no_frames_reports_dead_and_warns_once(self, session):
        session.start_live()
        # lambda first, then throttle reset: the background endpoint thread
        # must never consume the tick window while activity is still absent.
        session.loopback.activity = lambda: {
            "running": True, "frames": 0, "heard": False,
            "last_frame_age": None, "last_loud_age": None,
        }
        session._health_at = 0.0
        session._live_started_at = time.monotonic() - 30.0  # past grace
        session._live_health_tick(time.monotonic())
        health, errors = _health_msgs(session)
        assert health and health[-1]["level"] == "dead"
        assert health[-1]["audio"] is False
        assert errors and "No PC audio" in errors[0]["message"]

        # unchanged state -> no duplicate broadcast, no duplicate toast
        session._health_at = 0.0
        session._live_health_tick(time.monotonic())
        health2, errors2 = _health_msgs(session)
        assert health2 == []
        assert errors2 == []
        session.stop_live()

    def test_healthy_audio_reports_ok(self, session):
        session.start_live()
        session.loopback.activity = lambda: {
            "running": True, "frames": 100, "heard": True,
            "last_frame_age": 0.1, "last_loud_age": 0.2,
        }
        session._health_at = 0.0
        session._live_started_at = time.monotonic() - 5.0
        session._live_health_tick(time.monotonic())
        health, errors = _health_msgs(session)
        assert health, _drain(session.out_queue, timeout=0.1)
        assert health[-1]["level"] == "ok"
        assert health[-1]["audio"] is True
        assert health[-1]["heard"] is True
        assert errors == []
        session.stop_live()

    def test_never_heard_reports_quiet(self, session):
        session.start_live()
        session.loopback.activity = lambda: {
            "running": True, "frames": 500, "heard": False,
            "last_frame_age": 0.1, "last_loud_age": None,
        }
        session._health_at = 0.0
        session._live_started_at = time.monotonic() - 30.0
        session._live_health_tick(time.monotonic())
        health, errors = _health_msgs(session)
        assert health and health[-1]["level"] == "quiet"
        assert errors and "silence" in errors[0]["message"]
        session.stop_live()

    def test_voices_without_transcript_reports_stt(self, session):
        session.start_live()
        session.loopback.activity = lambda: {
            "running": True, "frames": 500, "heard": True,
            "last_frame_age": 0.1, "last_loud_age": 1.0,
        }
        session._health_at = 0.0
        session._live_started_at = time.monotonic() - 30.0
        session._last_stt_at = 0.0  # STT never produced anything
        session._live_health_tick(time.monotonic())
        health, _ = _health_msgs(session)
        assert health and health[-1]["level"] == "stt"
        assert health[-1]["stt"] is False
        session.stop_live()

    def test_silence_does_not_warn_transcription(self, session):
        """A quiet interviewer must NOT trip the STT warning — only voices
        without transcription do."""
        session.start_live()
        session.loopback.activity = lambda: {
            "running": True, "frames": 500, "heard": True,
            "last_frame_age": 0.1, "last_loud_age": 55.0,
        }
        session._health_at = 0.0
        session._live_started_at = time.monotonic() - 60.0
        session._last_stt_at = time.monotonic() - 60.0  # spoke once, long ago
        session._live_health_tick(time.monotonic())
        health, errors = _health_msgs(session)
        assert health and health[-1]["level"] == "ok"
        assert health[-1]["stt"] is True
        assert errors == []
        session.stop_live()

    def test_missing_activity_method_is_harmless(self, session):
        """FakeCapture (tests) and any future loopback without activity()."""
        session.start_live()
        session._health_at = 0.0
        session._live_health_tick(time.monotonic())  # must not raise
        health, _ = _health_msgs(session)
        assert health == []  # skipped, nothing emitted
        session.stop_live()


class TestMockMode:
    def test_start_mock_generates_question(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: "What is your biggest strength?")
        result = session.start_mock("desktop")
        assert result["ok"] is True
        assert session.mode == "mock"
        msg = _wait_for(session.out_queue, "mock_question", timeout=8)
        assert msg is not None, _drain(session.out_queue)
        assert msg["question"] == "What is your biggest strength?"
        assert msg["question_number"] == 1
        _wait_for_answer(session)
        assert session.mock_state == "listening"

    def test_answer_evaluation_flow(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: "Q?")
        monkeypatch.setattr(mod, "evaluate_mock_answer", lambda q, a, **kw: {
            "score": 8, "feedback": "Good structure.", "improved_answer": "model answer"
        })
        session.start_mock("desktop")
        _wait_for(session.out_queue, "mock_question", timeout=8)
        _wait_for_answer(session)

        # candidate speaks -> partial, then Done button -> commit -> evaluate
        session._on_mock_partial({"text": "I led a migration that cut costs 30%"})
        session.mic_stt.committed_text = "I led a migration that cut costs 30%"
        result = session.finish_mock_answer()
        assert result["ok"] is True

        fb = _wait_for(session.out_queue, "mock_feedback", timeout=10)
        assert fb is not None, _drain(session.out_queue)
        assert fb["score"] == 8
        assert fb["feedback"] == "Good structure."
        assert fb["user_answer"] == "I led a migration that cut costs 30%"
        assert session.mic_stt.commit_calls == 1

        nxt = _wait_for(session.out_queue, "mock_question", timeout=10)
        assert nxt is not None, "no follow-up question"
        assert nxt["question_number"] == 2

    def test_phone_mic_audio_feeds_stt(self, session, monkeypatch):
        import base64
        import interview_session as mod
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: "Q?")
        session.start_mock("phone")
        _wait_for_answer(session)
        session.handle_phone_mic_audio(base64.b64encode(b"\x01\x02" * 100).decode())
        assert session.mic_stt.sent, "phone audio not forwarded to STT"

    def test_stop_mock_cleans_up(self, session):
        session.start_mock("desktop")
        session.stop_mock()
        assert session.mode is None
        assert session.mock_state is None
        assert not session.mic_stt.is_running()
        assert not session.mic_capture.is_running()

    def test_modes_are_exclusive(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: "Q?")
        session.start_live()
        session.start_mock("desktop")
        assert session.mode == "mock"
        assert not session.loopback.is_running()
        session.stop_mock()
        session.start_live()
        assert session.mode == "live"
        assert not session.mic_stt.is_running()


class TestMockTurnTaking:
    """The v2 turn-taking fixes: mic gating, echo drop, endpointing, skip."""

    def test_mic_deaf_unless_listening(self, session, monkeypatch):
        import base64
        import interview_session as mod
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: "Q?")
        session.start_mock("desktop")
        _wait_for_answer(session)

        session.mock_state = "speaking"
        session._mock_desktop_audio(b"\x01\x02" * 10)
        session.handle_phone_mic_audio(base64.b64encode(b"\x01\x02" * 10).decode())
        assert not session.mic_stt.sent, "mic must be deaf while the question is spoken"

        session.mock_state = "evaluating"
        session._mock_desktop_audio(b"\x01\x02" * 10)
        assert not session.mic_stt.sent, "mic must be deaf while evaluating"

        session.mock_state = "listening"
        session._mock_desktop_audio(b"\x01\x02" * 10)
        assert session.mic_stt.sent, "mic must be open while listening"

    def test_tts_echo_committed_is_dropped(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: "What is your greatest strength?")
        monkeypatch.setattr(mod, "evaluate_mock_answer", lambda q, a, **kw: {
            "score": 1, "feedback": "bad", "improved_answer": "",
        })
        session.start_mock("desktop")
        _wait_for_answer(session)

        # the interviewer's own TTS audio reaches STT and "commits" during evaluation
        session.mock_state = "evaluating"
        session._on_mock_answer({"text": "What is your greatest strength?"})
        time.sleep(0.4)
        msgs = _drain(session.out_queue, timeout=0.6)
        assert not any(m.get("type") == "mock_user_answer" for m in msgs), "echo became an answer"
        assert not any(m.get("type") == "mock_feedback" for m in msgs), "echo triggered evaluation"

    def test_done_without_speech_rejected(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: "Q?")
        session.start_mock("desktop")
        _wait_for_answer(session)
        result = session.finish_mock_answer()
        assert result["ok"] is False
        assert "No answer captured" in result["error"]

    def test_endpoint_silence_auto_commits(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: "Q?")
        monkeypatch.setattr(mod, "evaluate_mock_answer", lambda q, a, **kw: {
            "score": 7, "feedback": "ok", "improved_answer": "",
        })
        session.endpoint_silence_secs = 0.3
        session.start_mock("desktop")
        _wait_for_answer(session)

        session._on_mock_partial({"text": "I always write tests first"})
        session.mic_stt.committed_text = "I always write tests first"

        fb = _wait_for(session.out_queue, "mock_feedback", timeout=6)
        assert fb is not None, "silence endpoint never fired: " + str(_drain(session.out_queue))
        assert fb["user_answer"] == "I always write tests first"
        assert session.mic_stt.commit_calls == 1

    def test_rolling_segments_concatenated_on_endpoint(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: "Q?")
        monkeypatch.setattr(mod, "evaluate_mock_answer", lambda q, a, **kw: {
            "score": 6, "feedback": "ok", "improved_answer": "",
        })
        session.start_mock("desktop")
        _wait_for_answer(session)

        # Scribe ~36s rolling auto-commit mid-answer keeps the segment open
        session._on_mock_answer({"text": "First half of my long answer."})
        assert session.mock_state == "listening"
        session._on_mock_partial({"text": "Second half."})
        session.mic_stt.committed_text = "Second half."
        session._finish_mock_answer("silence")

        fb = _wait_for(session.out_queue, "mock_feedback", timeout=8)
        assert fb is not None, _drain(session.out_queue)
        assert fb["user_answer"] == "First half of my long answer. Second half."

    def test_commit_rescue_delivers_when_stt_silent(self, session, monkeypatch):
        import interview_session as mod
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: "Q?")
        monkeypatch.setattr(mod, "evaluate_mock_answer", lambda q, a, **kw: {
            "score": 5, "feedback": "ok", "improved_answer": "",
        })
        session.commit_rescue_secs = 0.3
        session.start_mock("desktop")
        _wait_for_answer(session)

        session._on_mock_partial({"text": "my tracked answer text"})
        # FakeSTT.commit fires with empty committed_text -> silent commit
        session.mic_stt.committed_text = ""
        session._finish_mock_answer("done")
        # state=committing; rescue path should deliver the tracked partial
        fb = _wait_for(session.out_queue, "mock_feedback", timeout=6)
        assert fb is not None, "rescue never fired: " + str(_drain(session.out_queue))
        assert fb["user_answer"] == "my tracked answer text"

    def test_skip_stops_playback_and_advances(self, session, monkeypatch):
        import interview_session as mod
        questions = iter(["Q1", "Q2", "Q3"])
        monkeypatch.setattr(mod, "generate_mock_question", lambda **kw: next(questions))
        session.start_mock("desktop")
        _wait_for(session.out_queue, "mock_question", timeout=8)
        _wait_for_answer(session)

        stops_before = session.tts.stops
        assert session.skip_mock_question()["ok"] is True
        q2 = _wait_for(session.out_queue, "mock_question", timeout=8)
        assert q2 is not None, "skip did not advance: " + str(_drain(session.out_queue))
        assert q2["question_number"] == 2
        assert session.tts.stops > stops_before, "skip must stop current playback"
        session.stop_mock()


class TestLiveFragmentMerge:
    def test_incomplete_fragments_merged_before_answer(self, session, monkeypatch):
        import interview_session as mod
        calls = []

        def fake_detect(t, **kw):
            calls.append(t)
            if len(calls) == 1:
                return {"is_question": False, "incomplete": True, "needs_screen": False,
                        "question": "", "answer": ""}
            return {"is_question": True, "incomplete": False, "needs_screen": False,
                    "question": t, "answer": "MERGED ANSWER"}

        monkeypatch.setattr(mod, "detect_and_answer", fake_detect)
        session.start_live()
        session._on_live_committed({"text": "How do you reverse"})

        # wait specifically for the fragment-buffering status (not the start status)
        msg = None
        deadline = time.time() + 6
        while time.time() < deadline and msg is None:
            try:
                candidate = session.out_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if (candidate.get("type") == "status"
                    and "Waiting for the rest" in (candidate.get("message") or "")):
                msg = candidate
        assert msg is not None, "fragment was not buffered: " + str(_drain(session.out_queue))

        session._on_live_committed({"text": "a linked list in Python?"})
        answer = _wait_for(session.out_queue, "answer", timeout=8)
        assert answer is not None, _drain(session.out_queue)
        assert answer["heard"] == "How do you reverse a linked list in Python?"
        assert answer["answer"] == "MERGED ANSWER"
        assert len(calls) == 2, f"expected detect on fragment then merged, got {calls}"

    def test_fragment_window_flushes_buffer(self, session):
        session.start_live()
        session._live_fragments = ["partial interviewer thought"]
        session._live_fragments_at = time.monotonic() - 60
        session._live_fragment_tick(time.monotonic())
        assert session._live_fragments == []
        assert session._live_force_text == "partial interviewer thought"
        try:
            merged = session.live_work.get_nowait()
        except Exception:
            pytest.fail("flush did not enqueue merged text")
        assert merged == "partial interviewer thought"
        session.stop_live()


class TestScreenFusion:
    def test_parse_solve_json_clean(self):
        from interview_pipeline import _parse_solve_response
        raw = ('{"detected": true, "answer_text": "I would iterate with two pointers.",'
               '"language": "python", "code_solution": "def solve(a):\\n    return a",'
               '"next_steps": ["handle empty input", "add tests"]}')
        r = _parse_solve_response(raw)
        assert r["detected"] is True
        assert "two pointers" in r["answer_text"]
        assert r["language"] == "python"
        assert "def solve" in r["code_solution"]
        assert r["next_steps"] == ["handle empty input", "add tests"]

    def test_parse_solve_fenced_json(self):
        from interview_pipeline import _parse_solve_response
        raw = 'Here you go:\n```json\n{"detected": true, "answer_text": "ok", "language": "js", "code_solution": "x=1", "next_steps": ["step one"]}\n```'
        r = _parse_solve_response(raw)
        assert r["detected"] is True
        assert r["code_solution"] == "x=1"

    def test_parse_solve_garbage(self):
        from interview_pipeline import _parse_solve_response
        r = _parse_solve_response("not json at all")
        assert r["detected"] is False
        assert r["answer_text"] == ""

    def test_parse_solve_steps_string_fallback(self):
        from interview_pipeline import _parse_solve_response
        raw = '{"detected": true, "answer_text": "a", "language": "go", "code_solution": "", "next_steps": "- first\\n- second"}'
        r = _parse_solve_response(raw)
        assert r["next_steps"] == ["first", "second"]

    def test_solve_with_screen_groq_vision_success(self, monkeypatch):
        import interview_pipeline as ip

        def fake_groq(model, messages, *, max_tokens, timeout, compound=False):
            assert model == ip.GROQ_VISION_MODEL
            content = messages[0]["content"]
            assert any(part.get("type") == "image_url" for part in content)
            return ('{"detected": true, "answer_text": "spoken answer",'
                    '"language": "python", "code_solution": "print(1)",'
                    '"next_steps": ["edge cases"]}', {})

        monkeypatch.setattr(ip, "_groq_chat", fake_groq)
        result = ip.solve_with_screen([b"fake-jpeg"], "Reverse a linked list", history=[])
        assert result["provider"] == "groq_vision"
        assert result["code_solution"] == "print(1)"

    def test_solve_with_screen_falls_back_to_text_model(self, monkeypatch):
        import interview_pipeline as ip
        calls = []

        def fake_groq(model, messages, *, max_tokens, timeout, compound=False):
            calls.append(model)
            if model == ip.GROQ_VISION_MODEL:
                raise RuntimeError("404 model not found")
            return ('{"detected": true, "answer_text": "text only answer",'
                    '"language": "n/a", "code_solution": "", "next_steps": []}', {})

        monkeypatch.setattr(ip, "_groq_chat", fake_groq)
        monkeypatch.setattr(ip, "_solve_with_gemini", lambda images, prompt: (_ for _ in ()).throw(RuntimeError("no gemini")))
        result = ip.solve_with_screen([b"fake-jpeg"], "some transcript", history=[])
        assert result["provider"] == "groq_text"
        assert ip.GROQ_CODE_MODEL in calls

    def test_session_solve_screen_emits_message(self, session, monkeypatch):
        import time as _time
        import interview_session as mod
        monkeypatch.setattr(mod, "solve_with_screen", lambda *a, **k: {
            "detected": True, "answer_text": "Use a hash map.",
            "language": "python", "code_solution": "def f(): pass",
            "next_steps": ["check collisions"],
        })
        session.frame_provider = lambda: (b"jpeg-bytes", 1, _time.time())
        session.start_live()
        result = session.solve_screen(variant="solve")
        assert result["ok"] is True
        msg = _wait_for(session.out_queue, "screen_solve", timeout=4)
        assert msg is not None, _drain(session.out_queue)
        assert msg["code"] == "def f(): pass"
        assert msg["steps"] == ["check collisions"]
        assert msg["variant"] == "solve"
        session.stop_live()

    def test_auto_solve_throttled_and_deduped(self, session, monkeypatch):
        import time as _time
        import interview_session as mod
        monkeypatch.setattr(mod, "solve_with_screen", lambda *a, **k: {
            "detected": True, "answer_text": "x", "language": "n/a",
            "code_solution": "", "next_steps": [],
        })
        session.frame_provider = lambda: (b"same-frame", 1, _time.time())
        session.start_live()
        first = session.solve_screen(variant="solve", auto=True)
        assert first["ok"] is True
        second = session.solve_screen(variant="solve", auto=True)
        assert second["ok"] is False
        assert second["error"] in ("throttled", "duplicate frame")
        session.stop_live()

    def test_safe_auto_solve_surfaces_no_frame(self, session):
        """A missing region used to fail SILENTLY while Auto showed On."""
        session.start_live()
        session._safe_auto_solve()  # no frame, nothing stashed
        msg = _wait_for(session.out_queue, "error", timeout=4)
        assert msg is not None, _drain(session.out_queue, timeout=0.5)
        assert "Auto assist" in msg["message"]
        assert "No screen frame" in msg["message"]
        session.stop_live()

    def test_safe_auto_solve_quiet_on_expected_skips(self, session, monkeypatch):
        import time as _time
        import interview_session as mod
        monkeypatch.setattr(mod, "solve_with_screen", lambda *a, **k: {
            "detected": True, "answer_text": "x", "language": "n/a",
            "code_solution": "", "next_steps": [],
        })
        session.frame_provider = lambda: (b"same-frame", 1, _time.time())
        session.start_live()
        assert session.solve_screen(variant="solve", auto=True)["ok"] is True
        session._safe_auto_solve()  # throttled -> stays silent
        errors = [m for m in _drain(session.out_queue, timeout=1.0)
                  if m.get("type") == "error"]
        assert errors == []
        session.stop_live()

    def test_safe_auto_solve_crash_toasts(self, session, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("kaput")
        session.start_live()
        monkeypatch.setattr(session, "solve_screen", boom)
        session._safe_auto_solve()
        msg = _wait_for(session.out_queue, "error", timeout=4)
        assert msg is not None, _drain(session.out_queue, timeout=0.5)
        assert "Auto assist crashed" in msg["message"]
        assert "kaput" in msg["message"]
        session.stop_live()

    def test_solve_screen_exception_returns_error_without_broadcast(
            self, session, monkeypatch):
        """Manual Solve toasts via action_result — no duplicate broadcast."""
        import interview_session as mod
        session.start_live()
        session.frame_provider = lambda: (b"x", 1, 0.0)
        monkeypatch.setattr(
            mod, "solve_with_screen",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vision down")),
        )
        result = session.solve_screen(variant="solve")
        assert result["ok"] is False
        assert "vision down" in result["error"]
        errors = [m for m in _drain(session.out_queue, timeout=1.0)
                  if m.get("type") == "error"]
        assert errors == []
        session.stop_live()

    def test_set_auto_assist_toggles(self, session):
        assert session.auto_screen_assist is True
        result = session.set_auto_assist(False)
        assert result["ok"] is True
        assert session.auto_screen_assist is False
        status = session.status()
        assert status["auto_screen_assist"] is False


# ---------------------------------------------------------------------------
# pipeline format-string sanity (no network)
# ---------------------------------------------------------------------------

class TestPromptFormatting:
    def test_mock_question_prompt_formats(self):
        from interview_pipeline import MOCK_QUESTION_PROMPT
        out = MOCK_QUESTION_PROMPT.format(
            role="data engineer", level="senior", question_number=3,
            previous="Already asked:\n- Q: hi",
        )
        assert "data engineer" in out
        assert "#3" in out

    def test_eval_prompt_formats(self):
        from interview_pipeline import MOCK_EVALUATION_PROMPT
        out = MOCK_EVALUATION_PROMPT.format(question="Q1", answer="A1")
        assert "Q1" in out and "A1" in out


# ---------------------------------------------------------------------------
# context capture: clipboard stash + screenshot segments
# ---------------------------------------------------------------------------

PROBLEM_CHUNK_1 = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target."
PROBLEM_CHUNK_2 = "Output: [0,1]"


class TestStashMerge:
    def test_empty_existing_takes_new(self):
        from context_capture import merge_stash
        assert merge_stash("", PROBLEM_CHUNK_1) == PROBLEM_CHUNK_1

    def test_duplicate_copy_ignored(self):
        from context_capture import merge_stash
        assert merge_stash(PROBLEM_CHUNK_1, PROBLEM_CHUNK_1) == PROBLEM_CHUNK_1

    def test_superset_replaces(self):
        from context_capture import merge_stash
        merged = merge_stash("part one", "part one part two")
        assert merged == "part one part two"

    def test_unrelated_chunks_append(self):
        from context_capture import merge_stash
        merged = merge_stash(PROBLEM_CHUNK_1, PROBLEM_CHUNK_2)
        assert merged == PROBLEM_CHUNK_1 + "\n" + PROBLEM_CHUNK_2

    def test_overlapping_seam_glued(self):
        from context_capture import merge_stash
        merged = merge_stash("returns the sum of two numbers", "two numbers in the array")
        assert merged == "returns the sum of two numbers in the array"

    def test_short_tail_does_not_glue_unrelated(self):
        from context_capture import merge_stash
        merged = merge_stash("count is 42", "2 is prime")
        assert merged == "count is 42\n2 is prime"


class TestContextStash:
    def test_stash_capture_problem_and_code(self, session):
        session.clipboard_reader = lambda: PROBLEM_CHUNK_1
        res = session.stash_capture("problem")
        assert res["ok"] is True
        assert session.context["problem"] == PROBLEM_CHUNK_1

        session.clipboard_reader = lambda: "def two_sum(nums, target): ..."
        res = session.stash_capture("code")
        assert res["ok"] is True
        assert session.context["code"].startswith("def two_sum")

        status = session.context_status()
        assert status["problem"]["chars"] == len(PROBLEM_CHUNK_1)
        assert status["code"]["chars"] > 0
        assert status["segments"] == 0

    def test_stash_rejects_short_clipboard(self, session):
        session.clipboard_reader = lambda: "oops"
        res = session.stash_capture("problem")
        assert res["ok"] is False
        assert "too short" in res["error"]
        assert session.context["problem"] == ""

    def test_stash_appends_chunked_copies(self, session):
        session.clipboard_reader = lambda: PROBLEM_CHUNK_1
        session.stash_capture("problem")
        session.clipboard_reader = lambda: PROBLEM_CHUNK_2
        res = session.stash_capture("problem")
        assert res["ok"] is True
        assert session.context["problem"] == PROBLEM_CHUNK_1 + "\n" + PROBLEM_CHUNK_2

    def test_stash_capture_emits_context_update(self, session):
        session.clipboard_reader = lambda: PROBLEM_CHUNK_1
        session.stash_capture("problem")
        msg = _wait_for(session.out_queue, "context_update", timeout=4)
        assert msg is not None, _drain(session.out_queue)
        assert msg["context"]["problem"]["chars"] == len(PROBLEM_CHUNK_1)

    def test_stash_clear_single_and_all(self, session):
        session.clipboard_reader = lambda: PROBLEM_CHUNK_1
        session.stash_capture("problem")
        session.stash_clear("problem")
        assert session.context["problem"] == ""
        session.clipboard_reader = lambda: PROBLEM_CHUNK_1
        session.stash_capture("problem")
        session.stash_clear("all")
        assert session.context == {"problem": "", "code": "", "segments": []}
        res = session.stash_clear("bogus")
        assert res["ok"] is False

    def test_stash_rejects_unknown_slot(self, session):
        session.clipboard_reader = lambda: PROBLEM_CHUNK_1
        assert session.stash_capture("notes")["ok"] is False

    def test_add_segment_uses_frame_provider(self, session):
        import time as _time
        assert session.add_segment()["ok"] is False  # no frame provider yet
        session.frame_provider = lambda: (b"jpeg-A", 1, _time.time())
        res = session.add_segment()
        assert res["ok"] is True
        assert res["segments"] == 1
        session.frame_provider = lambda: (b"jpeg-B", 2, _time.time())
        session.add_segment()
        assert session.context["segments"] == [b"jpeg-A", b"jpeg-B"]

    def test_add_segment_accepts_supplied_image(self, session):
        import base64
        res = session.add_segment(base64.b64encode(b"jpeg-raw").decode())
        assert res["ok"] is True
        assert session.context["segments"] == [b"jpeg-raw"]

    def test_solve_without_frame_uses_stashed_problem(self, session, monkeypatch):
        import interview_session as mod
        captured = {}

        def fake_solve(images, transcript, **kw):
            captured.update(kw)
            captured["images"] = images
            return {"detected": True, "answer_text": "spoken", "language": "python",
                    "code_solution": "code", "next_steps": []}

        monkeypatch.setattr(mod, "solve_with_screen", fake_solve)
        session.clipboard_reader = lambda: PROBLEM_CHUNK_1
        session.stash_capture("problem")
        session.start_live()
        result = session.solve_screen()  # no frame provider, no image
        assert result["ok"] is True
        assert captured["images"] == []
        assert captured["problem_text"] == PROBLEM_CHUNK_1
        assert captured["router"] is session.router
        session.stop_live()

    def test_solve_uses_segments_when_no_frame(self, session, monkeypatch):
        import time as _time
        import interview_session as mod
        captured = {}
        monkeypatch.setattr(mod, "solve_with_screen",
                            lambda images, transcript, **kw: captured.update(images=images) or {
                                "detected": True, "answer_text": "a", "language": "n/a",
                                "code_solution": "", "next_steps": []})
        session.frame_provider = lambda: (b"seg", 1, _time.time())
        session.add_segment()
        session.frame_provider = None
        session.start_live()
        result = session.solve_screen()
        assert result["ok"] is True
        assert captured["images"] == [b"seg"]
        session.stop_live()

    def test_solve_errors_when_no_context_at_all(self, session):
        session.start_live()
        result = session.solve_screen()
        assert result["ok"] is False
        assert "stash the problem" in result["error"]
        session.stop_live()


# ---------------------------------------------------------------------------
# provider router: manual switch + context-carrying failover
# ---------------------------------------------------------------------------

class TestProviderRouter:
    def test_defaults_and_status(self):
        from provider_router import ProviderRouter
        r = ProviderRouter()
        st = r.status()
        assert st["preferred"] == "groq" and st["active"] == "groq"
        assert r.stage_order() == ["groq", "gemini"]

    def test_manual_switch_moves_both_and_notifies(self):
        from provider_router import ProviderRouter
        events = []
        r = ProviderRouter(on_change=lambda *a: events.append(a))
        st = r.switch("gemini")
        assert st == {"preferred": "gemini", "active": "gemini", "last_error": ""}
        assert events and events[0][0] == "manual"
        assert events[0][2] == "gemini"
        assert r.stage_order() == ["gemini", "groq"]

    def test_switch_rejects_unknown(self):
        from provider_router import ProviderRouter
        with pytest.raises(ValueError):
            ProviderRouter().switch("openai")

    def test_run_primary_first(self):
        from provider_router import ProviderRouter
        events = []
        r = ProviderRouter(on_change=lambda *a: events.append(a))
        result, served = r.run({"groq": lambda: "ok-g", "gemini": lambda: "ok-m"})
        assert (result, served) == ("ok-g", "groq")
        assert events == []
        assert r.active == "groq"

    def test_failover_flips_active_and_notifies_fallback(self):
        from provider_router import ProviderRouter
        events = []
        r = ProviderRouter(on_change=lambda *a: events.append(a))

        def boom():
            raise ConnectionError("groq down")

        result, served = r.run({"groq": boom, "gemini": lambda: "ok-m"})
        assert (result, served) == ("ok-m", "gemini")
        assert r.active == "gemini"
        assert r.status()["last_error"]
        assert events[0][0] == "fallback" and events[0][1] == "groq" and events[0][2] == "gemini"
        # sticky: next call goes straight to the survivor
        result, served = r.run({"groq": boom, "gemini": lambda: "ok-m-2"})
        assert served == "gemini"

    def test_total_failure_raises_and_keeps_active(self):
        from provider_router import ProviderRouter
        r = ProviderRouter()

        def boom_a():
            raise RuntimeError("A")

        def boom_b():
            raise ValueError("B")

        with pytest.raises(RuntimeError, match="A"):
            r.run({"groq": boom_a, "gemini": boom_b})
        assert r.active == "groq"  # nothing proven -> no flip

    def test_note_served_flips_when_other_wins(self):
        from provider_router import ProviderRouter
        events = []
        r = ProviderRouter(on_change=lambda *a: events.append(a))
        r.note_served("gemini", failed="groq", error="vision 404")
        assert r.active == "gemini"
        assert events[0][0] == "fallback"
        r.note_served("gemini")
        assert len(events) == 1  # no event when the active provider served


class TestProviderContextCarry:
    """History/stash is rendered BEFORE the provider is chosen."""

    def test_detect_and_answer_includes_history_block(self, monkeypatch):
        import interview_pipeline as ip
        from provider_router import ProviderRouter
        captured = {}

        def fake_chat(model, messages, *, max_tokens, timeout, compound=False):
            captured["messages"] = messages
            return "IS_QUESTION: YES\nQUESTION: And you?\nANSWER: I agree with everything so far."

        def fail_gemini(*a, **k):
            raise AssertionError("gemini must not be called when groq succeeds")

        monkeypatch.setattr(ip, "_chat", fake_chat)
        monkeypatch.setattr(ip, "_gemini_chat", fail_gemini)
        result = ip.detect_and_answer(
            "So what do you think?",
            history=[{"question": "Tell me about yourself", "answer": "I am Micheal."}],
            router=ProviderRouter(),
        )
        assert result["is_question"] is True
        assert result["provider"] == "groq"
        joined = "\n".join(str(m.get("content")) for m in captured["messages"])
        assert "I am Micheal." in joined
        assert "So what do you think?" in joined

    def test_detect_fails_over_with_same_context(self, monkeypatch):
        import interview_pipeline as ip
        from provider_router import ProviderRouter
        events = []
        router = ProviderRouter(on_change=lambda *a: events.append(a))
        captured = {}

        def fail_groq(*a, **k):
            raise ConnectionError("groq down")

        def fake_gemini(messages, max_tokens, timeout=30):
            captured["messages"] = messages
            return "IS_QUESTION: YES\nQUESTION: Q?\nANSWER: A."

        monkeypatch.setattr(ip, "_chat", fail_groq)
        monkeypatch.setattr(ip, "_gemini_chat", fake_gemini)
        result = ip.detect_and_answer(
            "Next question",
            history=[{"question": "Q0", "answer": "A0"}],
            router=router,
        )
        assert result["provider"] == "gemini"
        assert router.active == "gemini"
        assert events and events[0][0] == "fallback"
        joined = "\n".join(str(m.get("content")) for m in captured["messages"])
        assert "Q0" in joined and "A0" in joined  # context reached the fallback

    def test_manual_switch_changes_which_provider_answers(self, monkeypatch):
        import interview_pipeline as ip
        from provider_router import ProviderRouter
        router = ProviderRouter()
        called = []

        def fail_groq(*a, **k):
            called.append("groq")
            raise AssertionError("groq must not run after switch to gemini")

        def fake_gemini(messages, max_tokens, timeout=30):
            called.append("gemini")
            return "IS_QUESTION: YES\nQUESTION: Q?\nANSWER: A."

        monkeypatch.setattr(ip, "_chat", fail_groq)
        monkeypatch.setattr(ip, "_gemini_chat", fake_gemini)
        router.switch("gemini")
        result = ip.detect_and_answer("Q", router=router)
        assert result["provider"] == "gemini"
        assert called == ["gemini"]


class TestSolveContextSlots:
    def test_solve_prompt_includes_problem_and_code(self):
        from interview_pipeline import _solve_prompt
        prompt = _solve_prompt(
            "Do it with O(n)", [{"question": "Q", "answer": "A"}], "solve",
            problem_text=PROBLEM_CHUNK_1, starter_code="def f(): pass",
            has_images=False,
        )
        assert PROBLEM_CHUNK_1 in prompt
        assert "def f(): pass" in prompt
        assert "No screenshot is attached" in prompt
        assert "O(n)" in prompt

    def test_text_only_solve_uses_problem_stash(self, monkeypatch):
        import interview_pipeline as ip
        captured = {}

        def fake_chat(model, messages, *, max_tokens, timeout, compound=False):
            captured["messages"] = messages
            captured["model"] = model
            return ('{"detected": true, "answer_text": "spoken", "language": "python",'
                    '"code_solution": "print(1)", "next_steps": []}')

        def fail_gemini(*a, **k):
            raise AssertionError("gemini must not run when groq text succeeds")

        monkeypatch.setattr(ip, "_chat", fake_chat)
        monkeypatch.setattr(ip, "_gemini_chat", fail_gemini)
        result = ip.solve_with_screen([], "transcript", history=[],
                                      problem_text=PROBLEM_CHUNK_1)
        assert result["provider"] == "groq_text"
        assert captured["model"] == ip.GROQ_CODE_MODEL
        prompt = captured["messages"][0]["content"]
        assert PROBLEM_CHUNK_1 in prompt
        assert "No screenshot is attached" in prompt

    def test_gemini_active_runs_gemini_text_first(self, monkeypatch):
        import interview_pipeline as ip
        from provider_router import ProviderRouter
        called = []
        router = ProviderRouter()
        router.switch("gemini")

        def fail_groq(*a, **k):
            called.append("groq")
            raise AssertionError("groq must not run before active gemini")

        def fake_gemini(messages, max_tokens, timeout=30):
            called.append("gemini")
            return ('{"detected": true, "answer_text": "g", "language": "go",'
                    '"code_solution": "fmt.Println(1)", "next_steps": []}')

        monkeypatch.setattr(ip, "_chat", fail_groq)
        monkeypatch.setattr(ip, "_gemini_chat", fake_gemini)
        result = ip.solve_with_screen([], "t", history=[], problem_text=PROBLEM_CHUNK_1,
                                      router=router)
        assert result["provider"] == "gemini_text"
        assert called == ["gemini"]

    def test_solve_fallback_flips_router(self, monkeypatch):
        import interview_pipeline as ip
        from provider_router import ProviderRouter
        events = []
        router = ProviderRouter(on_change=lambda *a: events.append(a))

        def fail_groq(*a, **k):
            raise ConnectionError("groq down")

        def fake_gemini(messages, max_tokens, timeout=30):
            return ('{"detected": true, "answer_text": "g", "language": "go",'
                    '"code_solution": "", "next_steps": []}')

        monkeypatch.setattr(ip, "_chat", fail_groq)
        monkeypatch.setattr(ip, "_gemini_chat", fake_gemini)
        result = ip.solve_with_screen([], "t", history=[], problem_text=PROBLEM_CHUNK_1,
                                      router=router)
        assert result["provider"] == "gemini_text"
        assert router.active == "gemini"
        assert events and events[0][0] == "fallback"


class TestSessionProviderSwitch:
    def test_set_provider_emits_provider_changed(self, session):
        res = session.set_provider("gemini")
        assert res["ok"] is True
        assert res["active"] == "gemini"
        msg = _wait_for(session.out_queue, "provider_changed", timeout=4)
        assert msg is not None, _drain(session.out_queue)
        assert msg["reason"] == "manual"
        assert msg["active"] == "gemini"
        assert session.status()["provider"]["active"] == "gemini"

    def test_set_provider_rejects_unknown(self, session):
        res = session.set_provider("openai")
        assert res["ok"] is False
        assert session.router.active == "groq"

    def test_failover_emits_provider_changed(self, session):
        session.router.note_served("gemini", failed="groq", error="boom")
        msg = _wait_for(session.out_queue, "provider_changed", timeout=4)
        assert msg is not None, _drain(session.out_queue)
        assert msg["reason"] == "fallback"
        assert msg["previous"] == "groq"
        assert msg["active"] == "gemini"


# ---------------------------------------------------------------------------
# rate gate (spacing + backoff instead of failure toasts)
# ---------------------------------------------------------------------------

class FakeTime:
    """Deterministic clock/sleep for RateGate."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def clock(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 3))
        self.t += seconds


class TestRateGate:
    def _gate(self, ft, **kw):
        from rate_gate import RateGate
        kw.setdefault("min_interval", 2.0)
        kw.setdefault("base_delay", 2.0)
        return RateGate(sleep=ft.sleep, clock=ft.clock, **kw)

    def test_min_interval_spaces_call_starts(self):
        ft = FakeTime()
        gate = self._gate(ft)
        assert gate.call(lambda: "a") == "a"
        assert gate.call(lambda: "b") == "b"
        assert ft.sleeps == [2.0]

    def test_rate_limit_retries_then_succeeds(self):
        ft = FakeTime()
        gate = self._gate(ft)
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("429 Too Many Requests")
            return "ok"

        waits = []
        result = gate.call(flaky, on_wait=lambda secs, what: waits.append(secs), what="answer")
        assert result == "ok"
        assert calls["n"] == 3
        assert len(ft.sleeps) >= 2  # at least the two backoffs (jitter + spacing may add more)
        assert any(w > 0 for w in waits)
        assert waits[-1] == 0  # recovery clears the banner

    def test_non_rate_limit_error_raises_immediately(self):
        ft = FakeTime()
        gate = self._gate(ft)

        def boom():
            raise RuntimeError("kaput")

        with pytest.raises(RuntimeError, match="kaput"):
            gate.call(boom)
        assert ft.sleeps == []  # no backoff for unrelated failures

    def test_gives_up_after_waiting_budget(self):
        ft = FakeTime()
        gate = self._gate(ft, timeout=5.0)

        def always_limited():
            raise RuntimeError("quota exceeded (429)")

        with pytest.raises(RuntimeError, match="gave up"):
            gate.call(always_limited, on_wait=lambda s, w: None)
        assert ft.t >= 5.0  # the budget was actually spent waiting

    def test_run_passes_router_through(self):
        from provider_router import ProviderRouter
        ft = FakeTime()
        gate = self._gate(ft, min_interval=0.0)
        router = ProviderRouter()
        result, served = gate.run(
            router, {"groq": lambda: "g", "gemini": lambda: "m"})
        assert (result, served) == ("g", "groq")

    def test_is_rate_limited_markers(self):
        from rate_gate import is_rate_limited
        assert is_rate_limited("HTTP 429")
        assert is_rate_limited("RESOURCE_EXHAUSTED: quota")
        assert is_rate_limited("Service Unavailable, retry later")
        # substring traps: bare "rate" must NOT match inside ordinary words
        assert not is_rate_limited("iterate over the keys")
        assert not is_rate_limited("connection reset by peer")


# ---------------------------------------------------------------------------
# design interview mode: parsers
# ---------------------------------------------------------------------------

class TestDesignParsers:
    def test_parse_design_response_full_blocks(self):
        from interview_pipeline import _parse_design_response
        text = (
            "STAGE: high_level\n"
            "SPOKEN: So I would put a load balancer in front of two app "
            "servers backed by a cache.\n"
            "MERMAID: flowchart TD\n  LB[Load Balancer] --> A[App]\n"
            "FACTS: 100M reads/day; p99 under 200ms\n"
            "DECISIONS: Use Redis for the hot cache\n"
            "NEXT: Deep dive on how reads hit the cache."
        )
        result = _parse_design_response(text)
        assert result["stage"] == "high_level"
        assert result["spoken"].startswith("So I would put")
        assert result["mermaid"].startswith("flowchart TD")
        assert result["facts"] == ["100M reads/day", "p99 under 200ms"]
        assert result["decisions"] == ["Use Redis for the hot cache"]
        assert result["next"].startswith("Deep dive")

    def test_parse_design_response_na_means_keep(self):
        from interview_pipeline import _parse_design_response
        text = (
            "STAGE: clarify\n"
            "SPOKEN: Quick question — is this read-heavy?\n"
            "MERMAID: N/A\nFACTS: N/A\nDECISIONS: N/A\nNEXT: N/A"
        )
        result = _parse_design_response(text)
        assert result["mermaid"] == ""      # keep previous diagram
        assert result["facts"] == []        # no new facts
        assert result["decisions"] == []
        assert result["next"] == ""

    def test_parse_design_response_format_drift_falls_back(self):
        from interview_pipeline import _parse_design_response
        result = _parse_design_response("I would start by asking about scale.")
        assert result["spoken"] == "I would start by asking about scale."

    def test_parse_design_response_unknown_stage_dropped(self):
        from interview_pipeline import _parse_design_response
        result = _parse_design_response(
            "STAGE: dancing\nSPOKEN: Hello there.\nMERMAID: N/A\n"
            "FACTS: N/A\nDECISIONS: N/A\nNEXT: N/A")
        assert result["stage"] == ""
        assert result["spoken"] == "Hello there."

    def test_parse_design_extract(self):
        from interview_pipeline import _parse_design_extract
        text = (
            "QUESTION: Design a rate limiter for our API gateway.\n"
            "INSTRUCTIONS:\n- Start with clarifying requirements\n"
            "- Give a high-level design\n- Deep dive on bottlenecks"
        )
        result = _parse_design_extract(text)
        assert result["question"].startswith("Design a rate limiter")
        assert len(result["instructions"]) == 3
        assert result["instructions"][0] == "Start with clarifying requirements"

    def test_ladder_defaults_when_empty(self):
        from interview_pipeline import _design_ladder_text, DESIGN_DEFAULT_LADDER
        assert "1." in _design_ladder_text([])
        assert len(_design_ladder_text([]).splitlines()) == len(DESIGN_DEFAULT_LADDER)


# ---------------------------------------------------------------------------
# design interview mode: pipeline calls (models stubbed)
# ---------------------------------------------------------------------------

class TestDesignPipeline:
    def test_design_turn_parses_canned_response(self, monkeypatch):
        import interview_pipeline as ip
        captured = {}

        def fake_chat(model, messages, *, max_tokens, timeout, compound=False):
            captured["messages"] = messages
            return ("STAGE: requirements\n"
                    "SPOKEN: We need about 10k writes per second.\n"
                    "MERMAID: flowchart TD\n  C[Client] --> G[Gateway]\n"
                    "FACTS: 10k writes/sec; multi-region\n"
                    "DECISIONS: Append-only event log\n"
                    "NEXT: Sketch the high-level design.")

        monkeypatch.setattr(ip, "_chat", fake_chat)
        state = {
            "question": "Design a notification system",
            "instructions": ["Clarify requirements", "High-level design"],
            "stage": "clarify", "facts": [], "decisions": [], "diagram": "",
        }
        result = ip.design_turn("How about throughput?", state)
        assert "error" not in result
        assert result["stage"] == "requirements"
        assert "10k writes" in result["spoken"]
        assert result["mermaid"].startswith("flowchart TD")
        assert result["facts"] == ["10k writes/sec", "multi-region"]
        assert result["decisions"] == ["Append-only event log"]
        assert result["provider"] == "groq"
        # prompt carries question + ladder + state
        joined = "\n".join(str(m.get("content")) for m in captured["messages"])
        assert "notification system" in joined
        assert "Clarify requirements" in joined

    def test_design_turn_returns_error_when_all_providers_fail(self, monkeypatch):
        import interview_pipeline as ip

        def fail(*a, **k):
            raise RuntimeError("kaput")

        monkeypatch.setattr(ip, "_chat", fail)
        monkeypatch.setattr(ip, "_gemini_chat", fail)
        result = ip.design_turn("hello", {"question": "Q", "instructions": []})
        assert "error" in result
        assert "kaput" in result["error"]

    def test_design_extract_uses_problem_text_alone(self, monkeypatch):
        import interview_pipeline as ip

        def fake_chat(model, messages, *, max_tokens, timeout, compound=False):
            return ("QUESTION: Design a chat system\n"
                    "INSTRUCTIONS:\n- Clarify\n- High-level design")

        monkeypatch.setattr(ip, "_chat", fake_chat)
        result = ip.design_extract([], "Design a chat system with 1M DAU")
        assert result["question"] == "Design a chat system"
        assert result["instructions"] == ["Clarify", "High-level design"]
        assert result["provider"] == "groq_text"


# ---------------------------------------------------------------------------
# design interview mode: session state machine
# ---------------------------------------------------------------------------

class TestDesignSession:
    def _activate(self, session):
        session.design.update({
            "active": True,
            "question": "Design a URL shortener",
            "instructions": ["Clarify requirements", "High-level design"],
        })

    def test_start_requires_live_mode(self, session):
        res = session.start_design()
        assert res["ok"] is False
        assert "live" in res["error"].lower()

    def test_start_without_frame_errors(self, session):
        session.start_live()
        session.frame_provider = None
        session.context = {"problem": "", "code": "", "segments": []}
        res = session.start_design()
        assert res["ok"] is False
        assert "region" in res["error"].lower()
        session.stop_live()

    def test_start_extracts_and_emits_state(self, session, monkeypatch):
        import interview_session as mod
        session.start_live()
        _drain(session.out_queue)  # drop start_live's cleared design_state
        session.frame_provider = lambda: (b"frame", 1, time.time())
        monkeypatch.setattr(mod, "design_extract", lambda *a, **k: {
            "question": "Design a URL shortener",
            "instructions": ["Clarify requirements", "Bottlenecks"],
            "provider": "groq_vision",
        })
        res = session.start_design()
        assert res["ok"] is True
        assert res["active"] is True
        assert res["question"] == "Design a URL shortener"
        assert res["instructions"] == ["Clarify requirements", "Bottlenecks"]
        msg = _wait_for(session.out_queue, "design_state", timeout=4)
        assert msg is not None, _drain(session.out_queue)
        assert msg["active"] is True
        session.stop_live()

    def test_turn_merges_state_and_emits_answer(self, session, monkeypatch):
        import interview_session as mod
        self._activate(session)
        monkeypatch.setattr(mod, "design_turn", lambda utterance, state, **k: {
            "stage": "requirements",
            "spoken": "We handle about a billion redirects a month.",
            "mermaid": "flowchart TD\n  A[API] --> DB[(Store)]",
            "facts": ["1B redirects/month"],
            "decisions": ["302 redirect with cache"],
            "next": "Sketch the high-level design.",
            "provider": "groq",
        })
        outcome = session._run_design_turn("what scale?")
        assert outcome == {"answered": True}
        msg = _wait_for(session.out_queue, "design_answer", timeout=4)
        assert msg is not None, _drain(session.out_queue)
        assert msg["stage"] == "requirements"
        assert msg["answer"]["spoken"].startswith("We handle")
        assert msg["diagram"].startswith("flowchart TD")
        assert msg["facts"] == ["1B redirects/month"]
        assert "Sketch" in msg["next"]
        # session state merged
        assert session.design["stage"] == "requirements"
        assert session.design["facts"] == ["1B redirects/month"]
        assert session.design["decisions"] == ["302 redirect with cache"]
        # history keeps cross-mode continuity
        assert session.history and session.history[-1]["answer"].startswith("We handle")

    def test_compaction_caps_turns_but_keeps_facts(self, session, monkeypatch):
        import interview_session as mod
        self._activate(session)
        turn = {"n": 0}

        def fake_turn(utterance, state, **k):
            turn["n"] += 1
            n = turn["n"]
            return {
                "stage": "deep_dive",
                "spoken": f"Turn {n} answer.",
                "mermaid": f"flowchart TD\n  N{n}[Node{n}]",
                "facts": [f"fact-{n}"],
                "decisions": [],
                "next": "",
                "provider": "groq",
            }

        monkeypatch.setattr(mod, "design_turn", fake_turn)
        for i in range(9):
            session._run_design_turn(f"utterance {i}")
        assert len(session.design["turns"]) <= 6          # bounded window
        assert len(session.design["facts"]) == 9          # state keeps everything
        assert session.design["facts"][0] == "fact-1"
        assert session.design["diagram"] == "flowchart TD\n  N9[Node9]"

    def test_live_worker_routes_to_design_when_active(self, session, monkeypatch):
        import interview_session as mod
        session.start_live()
        self._activate(session)  # AFTER start_live: it resets design state
        seen = {}

        def fake_design_turn(utterance, state, **k):
            seen["utterance"] = utterance
            seen["state"] = state
            return {"stage": "clarify", "spoken": "Answer.", "mermaid": "",
                    "facts": [], "decisions": [], "next": "",
                    "provider": "groq"}

        def detect_should_not_run(*a, **k):
            raise AssertionError("detect_and_answer must not run in design mode")

        monkeypatch.setattr(mod, "design_turn", fake_design_turn)
        monkeypatch.setattr(mod, "detect_and_answer", detect_should_not_run)
        session._on_live_committed({"text": "so what about the write path?"})
        msg = _wait_for(session.out_queue, "design_answer", timeout=6)
        assert msg is not None, _drain(session.out_queue)
        assert seen["utterance"] == "so what about the write path?"
        assert seen["state"]["question"] == "Design a URL shortener"
        session.stop_live()

    def test_live_worker_uses_detect_when_design_inactive(self, session, monkeypatch):
        import interview_session as mod
        assert session.design["active"] is False
        monkeypatch.setattr(mod, "detect_and_answer", lambda t, **k: {
            "is_question": True, "question": t,
            "answer": "Normal live answer.", "provider": "groq",
        })
        session.start_live()
        session._on_live_committed({"text": "Tell me about yourself"})
        msg = _wait_for(session.out_queue, "answer", timeout=6)
        assert msg is not None, _drain(session.out_queue)
        assert msg["answer"] == "Normal live answer."
        session.stop_live()

    def test_design_next_throttles_auto_poll(self, session, monkeypatch):
        import interview_session as mod
        session.start_live()
        self._activate(session)
        calls = []
        monkeypatch.setattr(mod, "design_turn", lambda u, s, **k: calls.append(u) or {
            "stage": "clarify", "spoken": "ok", "mermaid": "",
            "facts": [], "decisions": [], "next": "", "provider": "groq"})
        res = session.design_next(force=False)
        assert res["ok"] is True
        res = session.design_next(force=False)  # inside the floor window
        assert res["ok"] is True and res.get("skipped") is True
        assert res["error"] == "throttled"
        assert len(calls) == 1
        res = session.design_next(force=True)  # manual tap always runs
        assert res["ok"] is True
        assert len(calls) == 2
        session.stop_live()

    def test_design_next_requires_active_session(self, session):
        res = session.design_next()
        assert res["ok"] is False
        assert "active" in res["error"].lower()

    def test_design_end_clears_and_emits(self, session):
        self._activate(session)
        res = session.design_end()
        assert res["ok"] is True
        assert session.design["active"] is False
        msg = _wait_for(session.out_queue, "design_state", timeout=4)
        assert msg is not None, _drain(session.out_queue)
        assert msg["active"] is False

    def test_stop_live_clears_design(self, session):
        session.start_live()
        _drain(session.out_queue)  # drop start_live's cleared design_state
        self._activate(session)
        session.stop_live()
        assert session.design["active"] is False
        msg = _wait_for(session.out_queue, "design_state", timeout=4)
        assert msg is not None, _drain(session.out_queue)
        assert msg["active"] is False

    def test_rate_wait_callback_emits(self, session):
        session._on_rate_wait(4, "answer")
        msg = _wait_for(session.out_queue, "rate_wait", timeout=4)
        assert msg is not None
        assert msg["seconds"] == 4
        assert msg["what"] == "answer"

    def test_status_includes_design_payload(self, session):
        status = session.status()
        assert status["design"]["active"] is False
        assert "stage" in status["design"]

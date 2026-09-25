"""Interview session manager.

Owns all interview state for both Live and Mock modes:
- Live: loopback capture -> Scribe STT -> question detection -> answers
- Mock: Groq questions -> ElevenLabs TTS -> mic capture -> evaluation

Mock turn-taking state machine (echo-proof):
    ASKING (mic deaf) -> SPEAKING (mic deaf) -> LISTENING (mic open)
        -> COMMITTING (mic deaf) -> EVALUATING (mic deaf) -> next ASKING
Only LISTENING accepts audio, so TTS playback can never be transcribed as a
"candidate answer", and a single-flight worker means questions never overlap.

Answer endpointing (hybrid): Scribe runs in manual-commit mode; the session
commits when the partial stream goes quiet for ``endpoint_silence_secs``, when
the client sends ``answer_done`` (Done button), or when Scribe's own ~36s
rolling cap fires (segments are concatenated).

Live mode fragments: an utterance flagged INCOMPLETE is buffered and merged
with what follows, so a cut-off question is answered whole (max
LIVE_FRAGMENT_MAX fragments, flushed after LIVE_FRAGMENT_WINDOW seconds).

Screen fusion: ``solve_screen`` fuses the latest region frame with the live
transcript (auto-triggered when a question is flagged NEEDS_SCREEN, or
manually via Solve / Next step). Stashed context (problem statement / starter
code copied from the challenge panel, or stacked screenshot segments when the
platform blocks copy) is merged into the same prompt.

Provider routing: every model call goes through the session's ProviderRouter
(Groq <-> Gemini). The user can switch providers mid-session; history and
stashed context are rendered into the prompt before the provider is chosen,
so a switch — manual or automatic failover — never loses the conversation.

Thread model:
- Native audio callbacks and ElevenLabs STT events arrive on their own
  threads; they only ever call ``out_queue.put`` / job slots.
- A dedicated worker thread consumes Groq work (sync calls).
- A dedicated endpoint thread drives hybrid answer endpointing.
- The FastAPI event loop runs a pump task that broadcasts ``out_queue``
  messages to every connected WebSocket.
"""

from __future__ import annotations

import base64
import hashlib
import queue
import threading
import time
import uuid

from audio_capture import MicCapture, SystemAudioCapture
from elevenlabs_client import ElevenLabsSTT, ElevenLabsTTS, get_elevenlabs_key
from interview_pipeline import (
    detect_and_answer,
    evaluate_mock_answer,
    generate_mock_question,
    solve_with_screen,
)
from context_capture import merge_stash, read_text as read_clipboard_text
from provider_router import ProviderRouter

import logging

log = logging.getLogger("quizbox.interview")

# Hybrid endpointing (mock)
ENDPOINT_SILENCE_SECS = 6.0      # partial-stream quiet time before auto-commit
COMMIT_RESCUE_SECS = 4.0         # commit() produced no event -> deliver local text
NUDGE_AFTER_SECS = 15.0          # listening with zero speech -> gentle nudge

# Live fragment merging
LIVE_FRAGMENT_MAX = 2
LIVE_FRAGMENT_WINDOW = 8.0

# Auto screen fusion throttle / dedupe
AUTO_SOLVE_MIN_INTERVAL = 5.0
SOLVE_FRAME_TTL = 60.0

# Context capture (clipboard stash + screenshot segments)
CTX_MIN_CHARS = 20        # ignore accidental short copies (links, single words)
CTX_MAX_SEGMENTS = 6      # stacked problem screenshots kept for stitching
CTX_PREVIEW = 120         # preview characters echoed to the phone

MOCK_STATE_LABELS = {
    "asking": ("thinking", "Generating question..."),
    "speaking": ("speaking", "Interviewer is asking..."),
    "listening": ("listening_answer", "Your turn \u2014 speak your answer"),
    "committing": ("committing", "Finishing your answer..."),
    "evaluating": ("evaluating", "Evaluating your answer..."),
}


def _extract_text(data) -> str:
    if isinstance(data, dict):
        return (data.get("text") or data.get("transcript") or "").strip()
    return str(data or "").strip()


class InterviewSession:
    def __init__(self):
        self.mode: str | None = None  # None | "live" | "mock"
        self.clients: set = set()
        self.out_queue: "queue.Queue[dict]" = queue.Queue()
        self.history: list[dict] = []

        # Live components
        self.live_stt = ElevenLabsSTT()
        self.loopback = SystemAudioCapture()

        # Mock components (separate STT so modes never interfere)
        self.mic_stt = ElevenLabsSTT()
        self.mic_capture = MicCapture()
        self.tts = ElevenLabsTTS()

        # Mock state machine
        self.mic_source: str = "desktop"  # "desktop" | "phone"
        self.mock_role = "software engineer"
        self.mock_level = "mid-level"
        self.mock_state: str | None = None  # asking|speaking|listening|committing|evaluating
        self.mock_question: str = ""
        self.mock_question_number: int = 0
        self.mock_history: list[dict] = []
        self._question_requested = False
        self._mock_skip_token = 0

        # Hybrid endpointing tracking
        self.endpoint_silence_secs = ENDPOINT_SILENCE_SECS
        self.commit_rescue_secs = COMMIT_RESCUE_SECS
        self.nudge_after_secs = NUDGE_AFTER_SECS
        self._mock_partial_text = ""
        self._mock_partial_at = 0.0
        self._mock_answer_segments: list[str] = []
        self._mock_listening_at = 0.0
        self._mock_commit_at = 0.0
        self._mock_nudged = False

        # Live fragment merging
        self._live_fragments: list[str] = []
        self._live_fragments_at = 0.0
        self._live_force_text: str | None = None
        self._last_live_heard = ""

        # Screen fusion
        self.frame_provider = None  # set by server: screen.get_latest_jpeg
        self.auto_screen_assist = True
        self.last_solve: dict | None = None
        self._last_auto_solve_at = 0.0
        self._last_solve_hash = ""
        self._last_solve_at = 0.0

        # Context capture: problem/code stashed from the clipboard, plus
        # stacked screenshot segments for copy-disabled platforms.
        self.context: dict = {"problem": "", "code": "", "segments": []}
        self.clipboard_reader = None  # test seam; None -> context_capture.read_text

        # Provider routing (Groq <-> Gemini) with context-carrying failover
        self.router = ProviderRouter(preferred="groq", on_change=self._on_provider_change)

        # Work queues
        self.live_work: "queue.Queue[str]" = queue.Queue()
        self.mock_work: "queue.Queue[dict]" = queue.Queue()  # evaluations only
        self._worker: threading.Thread | None = None
        self._mock_worker: threading.Thread | None = None
        self._endpoint_thread: threading.Thread | None = None
        self._running = False
        self._last_committed = ""
        self._last_committed_at = 0.0

    # ------------------------------------------------------------------
    # outgoing messages
    # ------------------------------------------------------------------

    def emit(self, message: dict) -> None:
        message.setdefault("ts", time.time())
        self.out_queue.put(message)

    def set_state(self, state: str, extra: dict | None = None) -> None:
        payload = {"type": "status", "state": state}
        if extra:
            payload.update(extra)
        self.emit(payload)

    def _set_mock_state(self, state: str | None, message: str | None = None) -> None:
        self.mock_state = state
        if state is None:
            return
        ui_state, label = MOCK_STATE_LABELS.get(state, (state, state))
        self.set_state(ui_state, {"message": message or label, "mock_state": state})

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._live_worker, daemon=True, name="live-worker")
        self._worker.start()
        self._mock_worker = threading.Thread(target=self._mock_worker_loop, daemon=True, name="mock-worker")
        self._mock_worker.start()
        self._endpoint_thread = threading.Thread(target=self._endpoint_loop, daemon=True, name="mock-endpoint")
        self._endpoint_thread.start()
        log.info("Interview session manager started")

    def shutdown(self) -> None:
        self._running = False
        self.stop_live()
        self.stop_mock()
        for thread in (self._worker, self._mock_worker, self._endpoint_thread):
            if thread and thread.is_alive():
                thread.join(timeout=3)
        log.info("Interview session manager stopped")

    # ------------------------------------------------------------------
    # endpoint / flush timer (hybrid answer endpointing + live fragments)
    # ------------------------------------------------------------------

    def _endpoint_loop(self) -> None:
        while self._running:
            time.sleep(0.25)
            if not self._running:
                return
            now = time.monotonic()
            try:
                if self.mode == "mock":
                    self._endpoint_tick(now)
                elif self.mode == "live":
                    self._live_fragment_tick(now)
            except Exception as exc:
                log.error("Endpoint tick failed: %s", exc)

    def _endpoint_tick(self, now: float) -> None:
        state = self.mock_state
        if state == "listening":
            has_content = bool(self._mock_partial_text or self._mock_answer_segments)
            if has_content and now - self._mock_partial_at > self.endpoint_silence_secs:
                log.info("Endpoint: partial silence %.1fs -> committing answer", now - self._mock_partial_at)
                self._finish_mock_answer("silence")
            elif (not has_content and not self._mock_nudged
                  and now - self._mock_listening_at > self.nudge_after_secs):
                self._mock_nudged = True
                self.set_state("listening_answer", {
                    "message": "Still listening \u2014 take your time, or press Skip.",
                    "mock_state": "listening",
                })
        elif state == "committing":
            if now - self._mock_commit_at > self.commit_rescue_secs:
                text = self._collect_mock_answer()
                if text:
                    log.warning("Endpoint: STT commit silent; delivering local text")
                    self._deliver_mock_answer(text)
                else:
                    # Nothing captured at all — keep waiting for speech.
                    self._begin_listening()

    def _live_fragment_tick(self, now: float) -> None:
        if self._live_fragments and now - self._live_fragments_at > LIVE_FRAGMENT_WINDOW:
            merged = " ".join(self._live_fragments)
            self._live_fragments = []
            self._live_force_text = merged
            log.info("Live fragment window expired; processing buffered text")
            self.live_work.put(merged)

    # ------------------------------------------------------------------
    # LIVE MODE
    # ------------------------------------------------------------------

    def start_live(self) -> dict:
        if self.mode == "live":
            return {"ok": True, "message": "Already running"}
        if self.mode == "mock":
            self.stop_mock()
        if not get_elevenlabs_key():
            return {"ok": False, "error": "ELEVENLABS_API_KEY is not configured"}

        try:
            self.live_stt.start(
                on_partial=self._on_live_partial,
                on_committed=self._on_live_committed,
                on_error=self._on_live_error,
                commit_strategy="vad",
                vad_silence_threshold_secs=3.0,
                watchdog=True,
            )
        except Exception as exc:
            log.error("Live STT start failed: %s", exc)
            return {"ok": False, "error": f"Speech recognition failed to start: {exc}"}

        result = self.loopback.start(self.live_stt.send_audio)
        if not result.get("ok"):
            self.live_stt.stop()
            return result

        self.mode = "live"
        self._last_committed = ""
        self._live_fragments = []
        self._live_force_text = None
        self._last_live_heard = ""
        self.history = []
        self.set_state("listening", {"message": "Live interview started"})
        self.emit({"type": "history", "items": self.history})
        log.info("Live interview mode started")
        return {"ok": True, "device": result.get("device")}

    def stop_live(self) -> dict:
        was_live = self.mode == "live"
        self.loopback.stop()
        self.live_stt.stop()
        if was_live:
            self.mode = None
            self._live_fragments = []
            self.set_state("idle", {"message": "Live interview stopped"})
        # drain pending live work
        try:
            while True:
                self.live_work.get_nowait()
        except queue.Empty:
            pass
        return {"ok": True}

    def _on_live_partial(self, data) -> None:
        text = _extract_text(data)
        if text:
            self.emit({"type": "transcript", "text": text, "final": False})

    def _on_live_committed(self, data) -> None:
        text = _extract_text(data)
        if not text:
            return
        now = time.monotonic()
        if text == self._last_committed and now - self._last_committed_at < 5.0:
            return
        self._last_committed = text
        self._last_committed_at = now
        self.emit({"type": "transcript", "text": text, "final": True})
        if self.mode == "live":
            self.live_work.put(text)

    def _on_live_error(self, data) -> None:
        text = str(data)
        if text == "connection_closed":
            if self.mode == "live":
                self.emit({"type": "error", "message": "Speech recognition disconnected"})
            return
        log.error("Live STT error: %s", data)
        self.emit({"type": "error", "message": f"STT error: {text[:200]}"})

    def _live_worker(self) -> None:
        while self._running:
            try:
                text = self.live_work.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.mode != "live":
                continue
            self.set_state("processing")

            # A window flush already merged the buffer into this item.
            force = self._live_force_text is not None and self._live_force_text == text
            if force:
                self._live_force_text = None

            # If a fragment is buffered, detect on the MERGED text directly.
            if self._live_fragments and not force:
                text = " ".join([*self._live_fragments, text])
                self._live_fragments = []

            try:
                result = detect_and_answer(text, history=list(self.history), router=self.router)
            except Exception as exc:
                log.error("detect_and_answer crashed: %s", exc)
                self.emit({"type": "error", "message": f"Answer generation failed: {exc}"})
                self.set_state("listening")
                continue
            if result.get("error"):
                self.emit({"type": "error", "message": f"Answer generation failed: {result['error']}"})
                if self.mode == "live":
                    self.set_state("listening")
                continue

            # --- fragment merge: buffer cut-off utterances ---------------
            if (result.get("incomplete") and not force
                    and len(self._live_fragments) < LIVE_FRAGMENT_MAX):
                if not self._live_fragments:
                    self._live_fragments_at = time.monotonic()
                self._live_fragments.append(text)
                self.set_state("listening", {"message": "Waiting for the rest of the question\u2026"})
                continue

            self._last_live_heard = text
            if result.get("is_question") and result.get("answer"):
                entry = {
                    "id": uuid.uuid4().hex[:12],
                    "question": result.get("question") or text,
                    "heard": text,
                    "answer": result["answer"],
                    "ts": time.time(),
                }
                self.history.append(entry)
                self.emit({"type": "answer", "provider": result.get("provider", ""), **entry})

            # --- auto screen fusion for technical questions --------------
            if (result.get("needs_screen") and self.auto_screen_assist
                    and (result.get("is_question") or result.get("incomplete"))):
                threading.Thread(
                    target=self._safe_auto_solve, daemon=True, name="auto-solve",
                ).start()

            if self.mode == "live":
                self.set_state("listening")

    # ------------------------------------------------------------------
    # screen + transcript fusion
    # ------------------------------------------------------------------

    def _frame_bytes(self) -> bytes | None:
        if not self.frame_provider:
            return None
        try:
            snapshot = self.frame_provider()  # (jpeg, sequence, captured_at)
            return snapshot[0] if snapshot else None
        except Exception as exc:
            log.warning("frame_provider failed: %s", exc)
            return None

    def _solve_transcript(self) -> str:
        if self.mode == "live":
            return self._last_live_heard or (self.history[-1]["question"] if self.history else "")
        if self.mode == "mock":
            return self.mock_question
        return ""

    def solve_screen(self, image_b64: str | None = None, variant: str = "solve",
                     auto: bool = False) -> dict:
        """Fuse the screen (or stashed segments/problem text) with the
        transcript. Emits a ``screen_solve`` message."""
        if self.mode not in ("live", "mock"):
            return {"ok": False, "error": "Start an interview session first"}

        frame: bytes | None = None
        if image_b64:
            try:
                frame = base64.b64decode(image_b64)
            except Exception:
                return {"ok": False, "error": "Invalid image data"}
        else:
            frame = self._frame_bytes()

        problem = self.context["problem"] or None
        starter = self.context["code"] or None
        # Frame wins; otherwise fall back to stitched segments (max 3 per
        # request — qwen3.8-27b accepts up to three images).
        images = [frame] if frame else list(self.context["segments"])[-3:]
        if not images and not problem:
            return {"ok": False,
                    "error": "No screen frame. Select a region, add a segment, or stash the problem first."}

        transcript = self._solve_transcript()
        history = list(self.history[-4:])

        now = time.monotonic()
        if auto:
            if now - self._last_auto_solve_at < AUTO_SOLVE_MIN_INTERVAL:
                return {"ok": False, "error": "throttled"}
            self._last_auto_solve_at = now
            digest = hashlib.sha256(
                (frame or b"") + (problem or "").encode("utf-8", "ignore")
                + transcript.encode("utf-8", "ignore")
            ).hexdigest()
            if digest == self._last_solve_hash and now - self._last_solve_at < SOLVE_FRAME_TTL:
                return {"ok": False, "error": "duplicate frame"}

        try:
            result = solve_with_screen(
                images, transcript, history=history, variant=variant,
                problem_text=problem, starter_code=starter, router=self.router,
            )
        except Exception as exc:
            log.error("solve_screen failed: %s", exc)
            self.emit({"type": "error", "message": f"Screen solve failed: {exc}"})
            return {"ok": False, "error": str(exc)[:300]}

        self._last_solve_hash = hashlib.sha256(
            (frame or b"") + (problem or "").encode("utf-8", "ignore")
            + transcript.encode("utf-8", "ignore")
        ).hexdigest()
        self._last_solve_at = time.monotonic()
        self.last_solve = {**result, "variant": variant}

        if result.get("detected") and result.get("answer_text"):
            # record so subsequent Next step calls see the context
            self.history.append({
                "id": uuid.uuid4().hex[:12],
                "question": transcript or "(screen solve)",
                "heard": transcript,
                "answer": result["answer_text"],
                "ts": time.time(),
            })

        self.emit({
            "type": "screen_solve",
            "variant": variant,
            "auto": auto,
            "detected": result.get("detected", False),
            "answer": result.get("answer_text", ""),
            "code": result.get("code_solution", ""),
            "language": result.get("language", "n/a"),
            "steps": result.get("next_steps", []),
            "provider": result.get("provider", ""),
        })
        log.info("screen_solve variant=%s provider=%s", variant, result.get("provider"))
        return {"ok": True, "variant": variant}

    def _safe_auto_solve(self) -> None:
        try:
            self.solve_screen(variant="solve", auto=True)
        except Exception as exc:
            log.error("auto solve crashed: %s", exc)

    def set_auto_assist(self, enabled: bool) -> dict:
        self.auto_screen_assist = bool(enabled)
        return {"ok": True, "auto_screen_assist": self.auto_screen_assist}

    # ------------------------------------------------------------------
    # provider switching (Groq <-> Gemini, context carried automatically)
    # ------------------------------------------------------------------

    def _on_provider_change(self, reason: str, previous: str, current: str,
                            status: dict) -> None:
        """Single emit point for manual switches and automatic failovers."""
        log.info("Provider changed (%s): %s -> %s", reason, previous, current)
        self.emit({"type": "provider_changed", "reason": reason,
                   "previous": previous, "active": current, **status})

    def set_provider(self, provider: str) -> dict:
        """Manual mid-session switch. History/context flows to the new model."""
        try:
            status = self.router.switch(provider)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **status}

    def provider_status(self) -> dict:
        return self.router.status()

    # ------------------------------------------------------------------
    # context capture (clipboard stash + screenshot segments)
    # ------------------------------------------------------------------

    def _read_clipboard(self) -> str:
        if callable(self.clipboard_reader):
            return (self.clipboard_reader() or "").strip()
        try:
            return (read_clipboard_text() or "").strip()
        except Exception as exc:
            log.warning("clipboard read failed: %s", exc)
            return ""

    @staticmethod
    def _brief(text: str) -> dict:
        return {"chars": len(text), "preview": text[:CTX_PREVIEW]}

    def context_status(self) -> dict:
        return {
            "problem": self._brief(self.context["problem"]),
            "code": self._brief(self.context["code"]),
            "segments": len(self.context["segments"]),
        }

    def _emit_context(self) -> None:
        self.emit({"type": "context_update", "context": self.context_status()})

    def stash_capture(self, slot: str) -> dict:
        """Stash the current clipboard into ``problem`` or ``code``.

        The candidate copies with their own keyboard; this only READS the
        clipboard, which no interview platform can observe.
        """
        if slot not in ("problem", "code"):
            return {"ok": False, "error": f"Unknown stash slot: {slot!r}"}
        text = self._read_clipboard()
        existing = self.context[slot]
        # The length gate protects the FIRST capture; once a stash exists,
        # short chunks are legitimate (scrollable statements copied piecewise).
        if len(text) < CTX_MIN_CHARS and not existing:
            return {"ok": False, "error": "Clipboard is empty or too short — copy the text first"}
        if not text:
            return {"ok": False, "error": "Clipboard is empty"}
        if existing and len(text) < 4:
            return {"ok": False, "error": "Clipboard is too short to append"}
        merged = merge_stash(existing, text)
        changed = merged != existing
        self.context[slot] = merged
        if changed:
            self._emit_context()
        return {"ok": True, "slot": slot, "changed": changed,
                "context": self.context_status()}

    def stash_clear(self, slot: str | None = None) -> dict:
        """Clear one slot (problem|code|segments) or everything when empty."""
        if not slot or slot == "all":
            self.context = {"problem": "", "code": "", "segments": []}
        elif slot in ("problem", "code"):
            self.context[slot] = ""
        elif slot == "segments":
            self.context["segments"] = []
        else:
            return {"ok": False, "error": f"Unknown stash slot: {slot!r}"}
        self._emit_context()
        return {"ok": True, "context": self.context_status()}

    def add_segment(self, image_b64: str | None = None) -> dict:
        """Append the current region (or a supplied frame) as a stitched
        segment — the copy-disabled fallback for scrollable problems."""
        frame: bytes | None = None
        if image_b64:
            try:
                frame = base64.b64decode(image_b64)
            except Exception:
                return {"ok": False, "error": "Invalid image data"}
        else:
            frame = self._frame_bytes()
        if not frame:
            return {"ok": False, "error": "No screen frame available. Select a region first."}
        segments = self.context["segments"]
        segments.append(frame)
        self.context["segments"] = segments[-CTX_MAX_SEGMENTS:]
        self._emit_context()
        return {"ok": True, "segments": len(self.context["segments"]),
                "context": self.context_status()}

    # ------------------------------------------------------------------
    # MOCK MODE
    # ------------------------------------------------------------------

    def start_mock(self, mic_source: str = "desktop", role: str = "software engineer",
                   level: str = "mid-level") -> dict:
        if self.mode == "mock":
            return {"ok": True, "message": "Already running"}
        if self.mode == "live":
            self.stop_live()
        if not get_elevenlabs_key():
            return {"ok": False, "error": "ELEVENLABS_API_KEY is not configured"}

        self.mic_source = mic_source if mic_source in ("desktop", "phone") else "desktop"
        self.mock_role = role
        self.mock_level = level
        self.mock_question = ""
        self.mock_question_number = 0
        self.mock_history = []
        self.mock_state = None
        self._question_requested = False
        self._mock_skip_token += 1
        self._clear_mock_answer_tracking()
        self.mode = "mock"
        self.history = []

        try:
            self.mic_stt.start(
                on_committed=self._on_mock_answer,
                on_partial=self._on_mock_partial,
                on_error=lambda d: log.error("Mock STT error: %s", d),
                commit_strategy="manual",
                watchdog=False,
            )
        except Exception as exc:
            self.mode = None
            return {"ok": False, "error": f"Speech recognition failed to start: {exc}"}

        if self.mic_source == "desktop":
            result = self.mic_capture.start(self._mock_desktop_audio)
            if not result.get("ok"):
                self.mic_stt.stop()
                self.mode = None
                return result

        self.emit({"type": "history", "items": []})
        self._question_requested = True
        log.info("Mock interview started (mic=%s)", self.mic_source)
        return {"ok": True, "mic_source": self.mic_source}

    def stop_mock(self) -> dict:
        was_mock = self.mode == "mock"
        self.tts.stop_playback()
        self.mic_capture.stop()
        self.mic_stt.stop()
        if was_mock:
            self.mode = None
            self.mock_state = None
            self._question_requested = False
            self._clear_mock_answer_tracking()
            self.set_state("idle", {"message": "Mock interview stopped"})
        try:
            while True:
                self.mock_work.get_nowait()
        except queue.Empty:
            pass
        return {"ok": True}

    def _mock_desktop_audio(self, pcm: bytes) -> None:
        """Desktop mic feed — only while the candidate is supposed to speak."""
        if self.mode == "mock" and self.mock_state == "listening":
            self.mic_stt.send_audio(pcm)

    def handle_phone_mic_audio(self, b64_data: str) -> None:
        if self.mode != "mock" or self.mic_source != "phone":
            return
        if self.mock_state != "listening":
            return  # mic deaf while asking/evaluating — blocks TTS echo
        try:
            pcm = base64.b64decode(b64_data)
        except Exception:
            return
        if pcm:
            self.mic_stt.send_audio(pcm)

    def skip_mock_question(self) -> dict:
        if self.mode != "mock":
            return {"ok": False, "error": "Mock interview is not running"}
        self.tts.stop_playback()
        self._mock_skip_token += 1
        self._question_requested = True
        # abandon any in-progress answer for the skipped question
        self._clear_mock_answer_tracking()
        if self.mock_state in ("listening", "committing", "speaking"):
            self._set_mock_state("asking", "Skipping to the next question...")
        return {"ok": True}

    def finish_mock_answer(self) -> dict:
        """Done button: end the current answer now (hybrid endpointing trigger)."""
        if self.mode != "mock":
            return {"ok": False, "error": "Mock interview is not running"}
        if self.mock_state != "listening":
            return {"ok": False, "error": "Not accepting an answer right now"}
        if not (self._mock_partial_text or self._mock_answer_segments):
            return {"ok": False, "error": "No answer captured yet \u2014 keep talking first"}
        self._finish_mock_answer("done")
        return {"ok": True}

    # ---- mock answer endpointing -------------------------------------

    def _clear_mock_answer_tracking(self) -> None:
        self._mock_partial_text = ""
        self._mock_partial_at = 0.0
        self._mock_answer_segments = []
        self._mock_nudged = False
        self._mock_commit_at = 0.0

    def _collect_mock_answer(self) -> str:
        parts = [*self._mock_answer_segments]
        if self._mock_partial_text:
            parts.append(self._mock_partial_text)
        return " ".join(p for p in parts if p).strip()

    def _begin_listening(self) -> None:
        self._mock_partial_text = ""
        self._mock_answer_segments = []
        self._mock_nudged = False
        now = time.monotonic()
        self._mock_listening_at = now
        self._mock_partial_at = now
        self._set_mock_state("listening")

    def _finish_mock_answer(self, trigger: str) -> None:
        if self.mode != "mock" or self.mock_state != "listening":
            return
        if self._mock_partial_text:
            # Ask Scribe for the authoritative final text; the committed
            # callback delivers it (rescue path covers a silent commit).
            self._set_mock_state("committing")
            self._mock_commit_at = time.monotonic()
            self.mic_stt.commit()
        else:
            buffered = self._collect_mock_answer()
            if buffered:
                self._deliver_mock_answer(buffered)
            elif trigger == "done":
                self.set_state("listening_answer", {
                    "message": "No answer captured yet \u2014 keep talking, then press Done.",
                    "mock_state": "listening",
                })

    def _deliver_mock_answer(self, text: str) -> None:
        text = (text or "").strip()
        if not text or self.mode != "mock":
            return
        self._clear_mock_answer_tracking()
        self.emit({"type": "mock_user_answer", "text": text, "final": True})
        self._set_mock_state("evaluating")
        self.mock_work.put({"type": "evaluate", "answer": text})

    def _on_mock_partial(self, data) -> None:
        if self.mode != "mock" or self.mock_state != "listening":
            return
        text = _extract_text(data)
        if text:
            self._mock_partial_text = text
            self._mock_partial_at = time.monotonic()
            self._mock_nudged = False
            self.emit({"type": "mock_partial", "text": text, "final": False})

    def _on_mock_answer(self, data) -> None:
        """Scribe committed transcript (from endpoint commit or ~36s cap)."""
        text = _extract_text(data)
        if not text or self.mode != "mock":
            return
        if self.mock_state == "committing":
            full = " ".join([*self._mock_answer_segments, text]).strip() or self._collect_mock_answer()
            self._deliver_mock_answer(full)
        elif self.mock_state == "listening":
            # Rolling ~36s auto-commit mid-answer: keep the segment, stay open.
            self._mock_answer_segments.append(text)
            self._mock_partial_text = ""
            self._mock_partial_at = time.monotonic()
            self.emit({
                "type": "mock_partial",
                "text": " ".join(self._mock_answer_segments),
                "final": False,
            })
        # else: audio arrived while asking/evaluating (e.g. TTS echo) — drop.

    # ---- mock worker -------------------------------------------------

    def _mock_worker_loop(self) -> None:
        while self._running:
            if self._question_requested:
                self._question_requested = False
                if self.mode == "mock":
                    try:
                        self._handle_mock_question_job()
                    except Exception as exc:
                        log.error("Mock question job failed: %s", exc)
                        self.emit({"type": "error", "message": f"Mock interview error: {exc}"})
                continue
            try:
                job = self.mock_work.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.mode != "mock":
                continue
            try:
                if job.get("type") == "question":  # legacy path
                    self._handle_mock_question_job()
                elif job.get("type") == "evaluate":
                    self._handle_mock_evaluate_job(job)
            except Exception as exc:
                log.error("Mock worker job failed: %s", exc)
                self.emit({"type": "error", "message": f"Mock interview error: {exc}"})
                self._set_mock_state("listening")

    def _handle_mock_question_job(self) -> None:
        token = self._mock_skip_token
        self._set_mock_state("asking")
        self.tts.stop_playback()  # never overlap question audio
        question = generate_mock_question(
            role=getattr(self, "mock_role", "software engineer"),
            level=getattr(self, "mock_level", "mid-level"),
            question_number=self.mock_question_number + 1,
            previous_qa=self.mock_history,
            router=self.router,
        )
        if token != self._mock_skip_token or self.mode != "mock":
            return  # superseded by skip/stop while generating
        self.mock_question_number += 1
        self.mock_question = question
        self.mock_history.append({"question": question, "answer": ""})
        entry = {
            "id": uuid.uuid4().hex[:12],
            "question_number": self.mock_question_number,
            "question": question,
            "ts": time.time(),
        }
        self.emit({"type": "mock_question", **entry})
        self._set_mock_state("speaking")
        try:
            self.tts.speak(question, block=True)  # single-flight: blocks worker until done
        except Exception as exc:
            log.error("TTS failed: %s", exc)
            self.emit({"type": "error", "message": f"Voice output failed: {exc}"})
        if self.mode == "mock" and token == self._mock_skip_token:
            self._begin_listening()  # mic opens only once the question is finished

    def _handle_mock_evaluate_job(self, job: dict) -> None:
        answer = job.get("answer", "")
        if self.mock_state != "evaluating":
            self._set_mock_state("evaluating")
        evaluation = evaluate_mock_answer(
            self.mock_question, answer,
            history=self.mock_history, router=self.router,
        )
        if self.mock_history:
            self.mock_history[-1]["answer"] = answer
        feedback = {
            "type": "mock_feedback",
            "question": self.mock_question,
            "question_number": self.mock_question_number,
            "user_answer": answer,
            **evaluation,
            "ts": time.time(),
        }
        self.emit(feedback)
        if self.mode == "mock":
            self._question_requested = True  # depth-1: at most one pending question

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "live_running": self.mode == "live" and self.loopback.is_running() and self.live_stt.is_running(),
            "mock_running": self.mode == "mock",
            "stt_connected": self.live_stt.is_running() if self.mode == "live" else (self.mic_stt.is_running() if self.mode == "mock" else False),
            "mock_state": self.mock_state,
            "mock_question_number": self.mock_question_number,
            "mic_source": self.mic_source,
            "auto_screen_assist": self.auto_screen_assist,
            "history_len": len(self.history),
            "elevenlabs_key_set": bool(get_elevenlabs_key()),
            "clients": len(self.clients),
            "provider": self.router.status(),
            "context": self.context_status(),
        }


session = InterviewSession()

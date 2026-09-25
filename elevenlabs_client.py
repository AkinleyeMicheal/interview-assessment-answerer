"""ElevenLabs integration: real-time STT for live interview, TTS for mock interview."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import time

from config import get_settings

log = logging.getLogger("quizbox.interview")

STT_MODEL = "scribe_v2_realtime"
TTS_MODEL = "eleven_flash_v2_5"
TTS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"


def get_elevenlabs_key() -> str:
    return get_settings().get("ELEVENLABS_API_KEY", "")


class ElevenLabsSTT:
    """Real-time speech-to-text over the Scribe Realtime WebSocket.

    Runs its own asyncio loop on a daemon thread so it never blocks the
    FastAPI event loop.  Callbacks are invoked from the async thread; the
    caller is responsible for thread-safe handoff (e.g. via a queue).
    """

    def __init__(self):
        self._connection = None
        self._client = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._on_partial = None
        self._on_committed = None
        self._on_error = None
        self._last_audio_time = 0.0
        self._has_pending_speech = False
        self._watchdog: threading.Thread | None = None
        self._last_committed_text = ""
        self._connect_error: Exception | None = None
        self._commit_strategy = "vad"
        self._vad_threshold = 3.0
        self._watchdog_enabled = True

    # -- lifecycle ---------------------------------------------------------

    def start(self, on_partial=None, on_committed=None, on_error=None,
              commit_strategy: str = "vad", vad_silence_threshold_secs: float = 3.0,
              watchdog: bool = True):
        """Start the STT session.  Returns after the WebSocket is open.

        commit_strategy:
          - "vad": Scribe commits automatically after ``vad_silence_threshold_secs``
            of silence (used by live mode).
          - "manual": Scribe never commits on silence; the caller decides when an
            utterance is finished via ``commit()`` (used by mock mode so thinking
            pauses never cut an answer in half).
        watchdog: force-commit on audio stall (live loopback needs it; mock does not).
        """
        if self._running:
            return
        self._on_partial = on_partial
        self._on_committed = on_committed
        self._on_error = on_error
        self._commit_strategy = commit_strategy
        self._vad_threshold = vad_silence_threshold_secs
        self._watchdog_enabled = watchdog
        self._running = True
        self._last_audio_time = time.monotonic()
        self._has_pending_speech = False
        self._connect_error = None
        started = threading.Event()

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._connect(started))
            except Exception as exc:
                self._connect_error = exc
                started.set()
                log.error("STT connect failed: %s", exc)
                return
            if self._running:
                self._loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True, name="elevenlabs-stt")
        self._thread.start()
        if not started.wait(timeout=15):
            self.stop()
            raise TimeoutError("ElevenLabs STT did not connect within 15s")
        if self._connect_error is not None:
            error = self._connect_error
            self.stop()
            raise error
        if self._watchdog_enabled:
            self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True, name="stt-watchdog")
            self._watchdog.start()

    async def _connect(self, started: threading.Event):
        from elevenlabs.client import ElevenLabs

        key = get_elevenlabs_key()
        if not key:
            raise ValueError("ELEVENLABS_API_KEY is not configured")

        self._client = ElevenLabs(api_key=key)

        from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy

        strategy = CommitStrategy.MANUAL if self._commit_strategy == "manual" else CommitStrategy.VAD
        options = {
            "model_id": STT_MODEL,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": 16000,
            "commit_strategy": strategy,
            "min_speech_duration_ms": 200,
            "min_silence_duration_ms": 400,
            "language_code": "eng",
        }
        if strategy == CommitStrategy.VAD:
            options["vad_silence_threshold_secs"] = self._vad_threshold
        self._connection = await self._client.speech_to_text.realtime.connect(options)

        from elevenlabs.realtime.connection import RealtimeEvents

        def extract_text(data) -> str:
            if isinstance(data, dict):
                return data.get("text") or data.get("transcript") or ""
            return str(data or "")

        def handle_partial(data):
            if extract_text(data).strip():
                self._has_pending_speech = True
            self._safe(self._on_partial, data)

        def handle_committed(data):
            text = extract_text(data)
            self._has_pending_speech = False
            self._last_committed_text = text
            self._safe(self._on_committed, data)

        self._connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, handle_partial)
        self._connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, handle_committed)
        self._connection.on(
            RealtimeEvents.ERROR,
            lambda *args: self._safe(self._on_error, args[0] if args else "error"),
        )
        self._connection.on(
            RealtimeEvents.CLOSE,
            lambda *args: self._safe(self._on_error, "connection_closed"),
        )

        started.set()
        log.info("ElevenLabs STT connected (%s)", STT_MODEL)

    def _watchdog_loop(self):
        """Force a manual commit when audio stalls after speech.

        WASAPI loopback delivers frames only while something is rendering.
        If the audio stream goes quiet right after the interviewer finishes,
        VAD never sees trailing silence, so we commit manually.
        """
        while self._running and self._watchdog_enabled:
            time.sleep(0.5)
            if not self._running or not self._has_pending_speech:
                continue
            if time.monotonic() - self._last_audio_time > 1.5 and self._connection and self._loop:
                log.debug("Audio stalled with pending speech; forcing commit")
                self._has_pending_speech = False
                try:
                    asyncio.run_coroutine_threadsafe(self._connection.commit(), self._loop)
                except Exception as exc:
                    log.error("Manual commit failed: %s", exc)

    def commit(self):
        """Flush buffered audio now (manual commit strategy)."""
        if not self._running or self._connection is None or self._loop is None:
            return
        self._has_pending_speech = False
        try:
            asyncio.run_coroutine_threadsafe(self._connection.commit(), self._loop)
        except Exception as exc:
            log.error("Manual commit failed: %s", exc)

    def send_audio(self, pcm_chunk: bytes):
        """Feed a raw 16-bit mono 16 kHz PCM chunk into the transcriber."""
        if not self._running or self._connection is None or self._loop is None:
            return
        self._last_audio_time = time.monotonic()
        b64 = base64.b64encode(pcm_chunk).decode("ascii")
        asyncio.run_coroutine_threadsafe(
            self._connection.send({"audio_base_64": b64}), self._loop
        )

    def stop(self):
        self._running = False
        if self._watchdog and self._watchdog.is_alive():
            self._watchdog.join(timeout=3)
        self._watchdog = None
        if self._loop and self._connection and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._connection.close(), self._loop)
                future.result(timeout=3)
            except Exception:
                pass
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._connection = None
        self._loop = None
        self._thread = None
        log.info("ElevenLabs STT stopped")

    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def _safe(callback, data):
        if callback is None:
            return
        try:
            callback(data)
        except Exception as exc:
            log.error("STT callback error: %s", exc)


class ElevenLabsTTS:
    """Text-to-speech for mock interview questions."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from elevenlabs.client import ElevenLabs

            key = get_elevenlabs_key()
            if not key:
                raise ValueError("ELEVENLABS_API_KEY is not configured")
            self._client = ElevenLabs(api_key=key)
        return self._client

    def synthesize(self, text: str) -> bytes:
        """Return MP3 bytes for the given text."""
        return self._synthesize(text, "mp3_22050_32")

    def synthesize_pcm(self, text: str) -> bytes:
        """Return raw 16 kHz mono 16-bit PCM bytes for the given text."""
        return self._synthesize(text, "pcm_16000")

    def _synthesize(self, text: str, output_format: str) -> bytes:
        client = self._get_client()
        audio_iter = client.text_to_speech.convert(
            voice_id=TTS_VOICE_ID,
            text=text,
            model_id=TTS_MODEL,
            output_format=output_format,
        )
        chunks = []
        for chunk in audio_iter:
            if isinstance(chunk, bytes):
                chunks.append(chunk)
        return b"".join(chunks)

    def speak(self, text: str, block: bool = True):
        """Synthesize and play through the default audio device.

        Uses raw PCM + sounddevice so no ffplay/ffmpeg dependency exists.
        Safe to call from a worker thread.  Call stop_playback() to cut off.
        """
        pcm = self._synthesize(text, "pcm_24000")
        if not pcm:
            return
        import numpy as np
        import sounddevice as sd

        samples = np.frombuffer(pcm, dtype=np.int16)
        sd.play(samples, samplerate=24000, blocking=False)
        if block:
            sd.wait()

    @staticmethod
    def stop_playback():
        try:
            import sounddevice as sd
            sd.stop()
        except Exception as exc:
            log.warning("stop_playback failed: %s", exc)

    def is_configured(self) -> bool:
        return bool(get_elevenlabs_key())


def check_elevenlabs_status() -> dict:
    """Cheap status check without spending credits on inference."""
    key = get_elevenlabs_key()
    if not key:
        return {"configured": False, "valid": False, "error": "No API key set"}
    import urllib.request
    import urllib.error

    request = urllib.request.Request(
        "https://api.elevenlabs.io/v1/models",
        headers={"xi-api-key": key, "User-Agent": "Quizbox/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
        return {"configured": True, "valid": True}
    except urllib.error.HTTPError as error:
        return {"configured": True, "valid": False, "error": f"HTTP {error.code}"}
    except Exception as error:
        return {"configured": True, "valid": False, "error": str(error)}
